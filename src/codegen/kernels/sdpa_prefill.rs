//! MSL emit for `FusionKind::SdpaPrefill`.
//!
//! Prefill SDPA: Q has L>1 rows (the prompt tokens being processed in
//! one pass). Each threadgroup owns one (head_id, l_q) pair and runs
//! the same three-stage softmax-attention as the decode kernel, with
//! a causal mask gating each query's visible kv positions.
//!
//! Grid layout: gid = l_q * n_heads + head_id, so 14×L threadgroups
//! for Qwen-0.5B at prefill L=N.
//!
//! Buffer layout — runtime must pass:
//!   * Q : [L, n_heads, head_dim]            contiguous
//!   * K : [n_kv_heads, capacity, head_dim]  contiguous (cache layout)
//!   * V : [n_kv_heads, capacity, head_dim]  contiguous (cache layout)
//!   * kv_len  : single u32 — total visible kv positions (prior + L)
//!   * L       : single u32 — current Q row count
//!   * capacity: single u32 — cache stride per kv_head
//!   * out: [L, n_heads, head_dim]           contiguous
//!
//! Causal mask: position q_kv = (kv_len - L) + l_q. Scores at kv > q_kv
//! get -INFINITY so the softmax assigns zero probability.
//!
//! Naive O(L · kv_len · head_dim) — for very long prompts a tiled
//! FlashAttention-style rewrite would win, but the naive variant is
//! fine for short-to-medium prefill that fits MAX_KV_LEN in SMEM.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

const MAX_KV_LEN: u32 = 4096;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::SdpaPrefill) {
        bail!("sdpa_prefill emitter: wrong kind {:?}", fusion.kind);
    }
    if !matches!(fusion.attrs, FusionAttrs::SdpaPrefill { .. }) {
        bail!("sdpa_prefill emitter: attrs mismatch");
    }

    let n_heads = meta.num_attention_heads;
    let n_kv_heads = meta.num_key_value_heads;
    let head_dim = meta.head_dim;
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let t = crate::codegen::kernels::act_ty(meta);

    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: SdpaPrefill (n_heads={}, n_kv={}, head_dim={}, scale={})",
        fusion.id, n_heads, n_kv_heads, head_dim, scale
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *Q         [[buffer(0)]],"));
    w.line(format!("device const {t} *K         [[buffer(1)]],"));
    w.line(format!("device const {t} *V         [[buffer(2)]],"));
    w.line("constant uint       &kv_len    [[buffer(3)]],");
    w.line("constant uint       &L         [[buffer(4)]],");
    w.line("constant uint       &capacity  [[buffer(5)]],");
    w.line(format!("device {t}       *out       [[buffer(6)]],"));
    w.line("uint tid       [[thread_position_in_threadgroup]],");
    w.line("uint tg_size   [[threads_per_threadgroup]],");
    w.line("uint gid       [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint HEAD_DIM    = {head_dim};"));
    w.writeln(format_args!("constexpr uint N_HEADS     = {n_heads};"));
    w.writeln(format_args!("constexpr uint N_KV_HEADS  = {n_kv_heads};"));
    w.writeln(format_args!(
        "constexpr uint HEADS_PER_KV = {heads_per_kv};"
    ));
    w.writeln(format_args!("constexpr float SCALE      = {scale};"));
    w.writeln(format_args!("constexpr uint MAX_KV_LEN  = {MAX_KV_LEN};"));
    w.blank();

    // Unpack gid into (l_q, head_id). N_HEADS is a constexpr so the
    // compiler folds the divmod.
    w.line("uint l_q     = gid / N_HEADS;");
    w.line("uint head_id = gid - l_q * N_HEADS;");
    w.line("uint kv_head = head_id / HEADS_PER_KV;");
    // Causal mask threshold: this query's logical position in the KV
    // sequence. For a fresh prefill prior_kv_len==0 so q_kv == l_q.
    // For continued generation after a previous prefill, the new
    // tokens start at offset (kv_len - L).
    w.line("uint q_kv = (kv_len - L) + l_q;");
    w.blank();

    w.line("threadgroup float scores[MAX_KV_LEN];");
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.blank();

    // ── Stage 1: scores[kv] = Q[l_q, head, :] · K[kv_head, kv, :] · SCALE
    // Mask kv > q_kv to -INFINITY so the subsequent softmax excludes
    // future positions.
    w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
    w.push_indent();
    w.line("if (kv > q_kv) { scores[kv] = -INFINITY; continue; }");
    w.line(format!(
        "device const {t}* q_row = Q + (l_q * N_HEADS + head_id) * HEAD_DIM;"
    ));
    w.line(format!(
        "device const {t}* k_row = K + (kv_head * capacity + kv) * HEAD_DIM;"
    ));
    w.line("float dot = 0.0f;");
    w.line("for (uint d = 0u; d < HEAD_DIM; ++d) {");
    w.push_indent();
    w.line("dot += float(q_row[d]) * float(k_row[d]);");
    w.pop_indent();
    w.line("}");
    w.line("scores[kv] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // ── Stage 2: numerically stable softmax (max + sum + normalise)
    // Identical to sdpa_decode's Stage 2, with the additional kv > q_kv
    // entries already being -INFINITY (don't contribute to max or sum).
    w.line("float t_max = -INFINITY;");
    w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
    w.push_indent();
    w.line("t_max = max(t_max, scores[kv]);");
    w.pop_indent();
    w.line("}");
    w.line("t_max = simd_max(t_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_max = reduce_buf[0];");
    w.blank();

    w.line("float t_sum = 0.0f;");
    w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
    w.push_indent();
    w.line("scores[kv] = exp(scores[kv] - global_max);");
    w.line("t_sum += scores[kv];");
    w.pop_indent();
    w.line("}");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_sum = reduce_buf[0];");
    w.blank();

    // Hoist the divide out of the Stage 3 inner kv loop — same trick
    // as the decode kernel.
    w.line("float inv_sum = 1.0f / global_sum;");
    w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
    w.push_indent();
    w.line("scores[kv] *= inv_sum;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // ── Stage 3: out[l_q, head, d] = Σ_kv scores[kv] · V[kv_head, kv, d]
    w.line("for (uint d = tid; d < HEAD_DIM; d += tg_size) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    w.line("for (uint kv = 0u; kv < kv_len; ++kv) {");
    w.push_indent();
    w.line(format!(
        "device const {t}* v_row = V + (kv_head * capacity + kv) * HEAD_DIM;"
    ));
    w.line("acc += scores[kv] * float(v_row[d]);");
    w.pop_indent();
    w.line("}");
    w.line(format!(
        "out[(l_q * N_HEADS + head_id) * HEAD_DIM + d] = {t}(acc);"
    ));
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_sdpa_prefill_f{}", fusion.id)
}
