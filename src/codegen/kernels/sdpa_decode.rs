//! MSL emit for `FusionKind::SdpaDecode`.
//!
//! Decode-only (Q has L=1). One threadgroup per query head; threads
//! inside the group cooperate on the softmax reduction across the
//! key/value sequence dimension. GQA is handled by mapping
//! `head_id → kv_head_id = head_id / heads_per_kv`.
//!
//! Buffer layout — at runtime the Python wrapper must pass:
//!   * Q : [n_heads, head_dim]                contiguous
//!   * K : [n_kv_heads, kv_len, head_dim]     contiguous
//!   * V : [n_kv_heads, kv_len, head_dim]     contiguous
//!   * kv_len : single u32 in a buffer
//!   * out : [n_heads, head_dim]              contiguous
//!
//! `mx.contiguous(...)` on the K/V cache tensors guarantees that
//! layout regardless of how mlx-lm composed the transposes upstream.
//!
//! ```text
//!   per head:
//!     1. scores[kv] = (Q · K[kv]) * scale         kv ∈ [0, kv_len)
//!     2. softmax(scores) — two-pass max+sum
//!     3. out[d] = Σ_kv softmax(scores)[kv] · V[kv, d]
//! ```

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

/// Largest kv_len the legacy two-pass path statically allocates scratch
/// for. The default emit is now flash-style (block-streamed online
/// softmax) which doesn't need this — `FUSEKIT_SDPA_LEGACY=1` opts back
/// into the old code for A/B comparison.
const MAX_KV_LEN: u32 = 4096;

/// KV positions processed per flash-attention block. Sized to match a
/// simdgroup width so Stage 1's `simd_max` collapses cleanly. SMEM
/// footprint per TG: `BLOCK * 4` bytes (256 B at BLOCK=64) plus the
/// tiny reduce_buf — vs the legacy `MAX_KV_LEN * 4` (16 KB). The
/// occupancy gain is the entire win at long context.
pub const FLASH_BLOCK: u32 = 64;

/// Flash-Decoding KV chunk size — each (head, chunk) TG processes this
/// many KV positions. At KV_CHUNK=128 with kv=800 we launch 7 chunks
/// per head × 14 heads = 98 TGs, vs n_heads=14 TGs without splitting.
/// Closes the parallelism starvation at decode (only 14 heads → only
/// 14 TGs, GPU sits half-empty at <2 TG/SM on M5).
pub const FLASH_DEC_CHUNK: u32 = 128;

/// Maximum chunks per head Flash-Decoding allocates partials for.
/// Caps supported kv_len at MAX_CHUNKS × KV_CHUNK = 4096. TGs with
/// chunk_id × KV_CHUNK ≥ kv_len early-exit without touching the
/// atomic counter; only "active" chunks count toward the combine.
pub const FLASH_DEC_MAX_CHUNKS: u32 = 32;

pub fn flash_decoding_enabled() -> bool {
    std::env::var_os("FUSEKIT_SDPA_FLASH_DECODING").is_some()
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::SdpaDecode) {
        bail!("sdpa_decode emitter: wrong kind {:?}", fusion.kind);
    }
    let FusionAttrs::SdpaDecode { scale, .. } = fusion.attrs else {
        bail!("sdpa_decode emitter: attrs mismatch");
    };

    let q_shape = fusion
        .inputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let out_shape = fusion
        .outputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let v_shape = fusion
        .inputs
        .get(2)
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let inputless_semantic_decode = fusion.inputs.is_empty();
    let n_heads = meta.num_attention_heads;
    let n_kv_heads = meta.num_key_value_heads;
    let qk_dim = if inputless_semantic_decode {
        meta.head_dim
    } else {
        q_shape
            .last()
            .copied()
            .unwrap_or(meta.head_dim as i32)
            .max(1) as u32
    };
    let v_dim = if inputless_semantic_decode {
        meta.head_dim
    } else {
        v_shape
            .last()
            .copied()
            .or_else(|| out_shape.last().copied())
            .unwrap_or(meta.head_dim as i32)
            .max(1) as u32
    };
    let active_m = out_shape
        .get(0..)
        .filter(|s| !s.is_empty())
        .map(|s| crate::target::current_chip().active_tokens_for_shape(s))
        .unwrap_or_else(|| crate::target::current_chip().active_m(8))
        .max(1);
    let direct_scaled_q = (scale - 1.0).abs() < f32::EPSILON && !inputless_semantic_decode;
    let q_head_major = inputless_semantic_decode
        || direct_scaled_q
        || infer_q_head_major(q_shape, n_heads, active_m, qk_dim);
    let q_seq = if q_head_major {
        trace_q_head_stride(q_shape).max(1)
    } else {
        1
    };
    let out_seq = active_m;
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = if scale.abs() < f32::EPSILON {
        1.0f32 / (qk_dim as f32).sqrt()
    } else {
        scale
    };
    let t = crate::codegen::kernels::act_ty(meta);

    // Flash-Decoding splits each query head's KV across MAX_CHUNKS TGs;
    // the grid is `n_heads × MAX_CHUNKS` and `gid` decomposes into
    // `head_id × MAX_CHUNKS + chunk_id`. The single-TG-per-head emits
    // keep their existing `head_id = gid` convention.
    let flash_decoding = flash_decoding_enabled();
    // Bake layer_idx so each layer's SDPA kernel indexes into the
    // shared global partials/counter buffers at its own offset. Allows
    // ONE global buffer instead of n_layers per-layer scratches.
    let layer_idx = fusion
        .layer_hint
        .and_then(|t| match t {
            crate::graph::LayerTag::Layer(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0);

    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: SdpaDecode (n_heads={}, n_kv={}, qk_dim={}, v_dim={}, scale={}, layer={}, flash_decoding={})",
        fusion.id, n_heads, n_kv_heads, qk_dim, v_dim, scale, layer_idx, flash_decoding
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *Q         [[buffer(0)]],"));
    w.line(format!("device const {t} *K         [[buffer(1)]],"));
    w.line(format!("device const {t} *V         [[buffer(2)]],"));
    w.line("constant uint       &kv_len    [[buffer(3)]],");
    // capacity = stride between kv_heads in the cache layout
    // (1, n_kv, capacity, head_dim). For the legacy contiguous layout
    // pass capacity == kv_len; for cache-resident dispatch pass the
    // cache's allocated capacity.
    w.line("constant uint       &capacity  [[buffer(4)]],");
    w.line(format!("device {t}       *out       [[buffer(5)]],"));
    if flash_decoding {
        w.line("device float       *partials  [[buffer(6)]],");
        w.line("device atomic_uint *counter   [[buffer(7)]],");
    }
    w.line("uint tid       [[thread_position_in_threadgroup]],");
    w.line("uint tg_size   [[threads_per_threadgroup]],");
    w.line("uint head_id   [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint QK_DIM      = {qk_dim};"));
    w.writeln(format_args!("constexpr uint V_DIM       = {v_dim};"));
    w.writeln(format_args!("constexpr uint Q_SEQ       = {q_seq};"));
    w.writeln(format_args!("constexpr uint OUT_SEQ     = {out_seq};"));
    w.writeln(format_args!("constexpr uint N_HEADS     = {n_heads};"));
    w.writeln(format_args!("constexpr uint N_KV_HEADS  = {n_kv_heads};"));
    w.writeln(format_args!(
        "constexpr bool Q_HEAD_MAJOR = {};",
        if q_head_major { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr uint HEADS_PER_KV = {heads_per_kv};"
    ));
    w.writeln(format_args!("constexpr float SCALE      = {scale};"));
    w.writeln(format_args!("constexpr uint MAX_KV_LEN  = {MAX_KV_LEN};"));
    w.writeln(format_args!("constexpr uint BLOCK       = {FLASH_BLOCK};"));
    if flash_decoding {
        w.writeln(format_args!(
            "constexpr uint KV_CHUNK    = {FLASH_DEC_CHUNK};"
        ));
        w.writeln(format_args!(
            "constexpr uint MAX_CHUNKS  = {FLASH_DEC_MAX_CHUNKS};"
        ));
        w.writeln(format_args!("constexpr uint LAYER_IDX   = {layer_idx};"));
        w.writeln(format_args!(
            "constexpr uint PARTIALS_STRIDE_PER_LAYER = OUT_SEQ * N_HEADS * MAX_CHUNKS * (V_DIM + 2u);"
        ));
        w.writeln(format_args!(
            "constexpr uint PARTIALS_STRIDE_PER_HEAD  = MAX_CHUNKS * (V_DIM + 2u);"
        ));
        w.writeln(format_args!(
            "constexpr uint COUNTER_STRIDE_PER_LAYER  = OUT_SEQ * N_HEADS;"
        ));
    }
    w.blank();

    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.blank();

    // Default: legacy two-pass (full kv_len in SMEM, 16 KB cap @ 2 TGs/SM).
    // Opt-in: FUSEKIT_SDPA_FLASH=1 → block-streamed online softmax.
    // Both correctly match mlx-lm bit-for-bit; the flash path's SMEM
    // win (16 KB → 0.4 KB) doesn't translate to speed by itself because
    // single-head-per-TG only launches `n_heads × 1` TGs total (14 on
    // 0.5B Qwen). FUSEKIT_SDPA_FLASH_DECODING=1 takes the additional
    // step of splitting the KV axis across TGs, raising launch count
    // to `n_heads × ceil(kv_len/KV_CHUNK)` and closing the parallelism
    // starvation at long context.
    if flash_decoding {
        emit_flash_decoding(w, t);
    } else {
        w.line("uint query_row = head_id / N_HEADS;");
        w.line("uint head_id_real = head_id - query_row * N_HEADS;");
        w.line("if (query_row >= OUT_SEQ) return;");
        w.line("uint kv_len_eff = kv_len + query_row;");
        w.line("uint kv_head = head_id_real / HEADS_PER_KV;");
        w.blank();
        let use_flash = std::env::var_os("FUSEKIT_SDPA_FLASH").is_some();
        if use_flash {
            emit_flash_online(w, t);
        } else {
            emit_legacy_two_pass(w, t);
        }
    }

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_sdpa_decode_f{}", fusion.id)
}

fn trace_q_head_stride(q_shape: &[i32]) -> u32 {
    if q_shape.len() >= 3 {
        q_shape[q_shape.len() - 2].max(1) as u32
    } else {
        1
    }
}

fn infer_q_head_major(q_shape: &[i32], n_heads: u32, active_m: u32, qk_dim: u32) -> bool {
    if q_shape.len() < 3 {
        return false;
    }
    let Some(&last) = q_shape.last() else {
        return false;
    };
    if last.max(1) as u32 != qk_dim {
        return false;
    }
    if trace_q_head_stride(q_shape).max(1) != active_m.max(1) {
        return false;
    }
    let head_prefix = q_shape[..q_shape.len() - 2]
        .iter()
        .map(|&d| d.max(1) as u64)
        .product::<u64>();
    head_prefix == n_heads.max(1) as u64
}

/// Flash-Decoding: each (head, kv_chunk) TG runs the flash 2-pass over
/// its KV slice, writes partial (m, l, o[V_DIM]) to the global
/// `partials` buffer, and atomic_fetch_adds the per-(layer, head)
/// counter. The TG whose increment returns `(active_chunks - 1)`
/// performs the cross-chunk combine and resets the counter.
///
/// Memory ordering: `threadgroup_barrier(mem_device)` flushes this TG's
/// partials writes; the atomic_fetch_add (default seq_cst on Metal)
/// release-publishes them, so the combiner — which observes a counter
/// value of `active_chunks - 1` — necessarily acquire-sees all earlier
/// chunks' partials. No spin-wait — the LAST incrementor is the
/// combiner; everyone else exits.
///
/// `gid` decomposes as `head_id × MAX_CHUNKS + chunk_id`. The actual
/// kernel argument `head_id` (kept for signature compatibility with
/// the non-flash-decoding emit) is the COMPOUND gid; we re-decompose
/// it here.
fn emit_flash_decoding(w: &mut MslWriter, t: &str) {
    w.line("// Flash-Decoding: per-(head, chunk) TG → global partials + atomic combine.");
    // Re-decompose head_id (which is gid here, since Metal pinned the
    // dispatch axis to threadgroup_position_in_grid).
    w.line("uint gid_compound = head_id;");
    w.line("uint query_head   = gid_compound / MAX_CHUNKS;");
    w.line("uint chunk_id     = gid_compound % MAX_CHUNKS;");
    w.line("uint query_row    = query_head / N_HEADS;");
    w.line("uint head_id_real = query_head - query_row * N_HEADS;");
    w.line("if (query_row >= OUT_SEQ) return;");
    w.line("uint kv_head      = head_id_real / HEADS_PER_KV;");
    w.line("uint kv_len_eff   = kv_len + query_row;");
    w.line("uint kv_start     = chunk_id * KV_CHUNK;");
    w.line("uint active_chunks = (kv_len_eff + KV_CHUNK - 1u) / KV_CHUNK;");
    w.blank();

    // Empty-chunk early exit. These TGs don't touch the counter so
    // `active_chunks - 1` always identifies the last ACTIVE chunk.
    w.line("if (kv_start >= kv_len_eff) return;");
    w.line("uint kv_end = min(kv_start + KV_CHUNK, kv_len_eff);");
    w.blank();

    // SMEM for inner-block streaming.
    w.line("threadgroup float s_block[BLOCK];");
    w.line("threadgroup float reduce_buf[32];");
    w.blank();

    // Per-thread state mirrors the single-TG flash emit: each thread
    // tid < V_DIM owns one output dim; threads ≥ V_DIM help in
    // Stage 1 reductions only.
    w.line("bool owns_d = tid < V_DIM;");
    w.line("float o_d = 0.0f;");
    w.line("float m = -INFINITY;");
    w.line("float l = 0.0f;");
    w.blank();

    // Inner BLOCK-streamed loop over the chunk's KV range.
    w.line("for (uint block = kv_start; block < kv_end; block += BLOCK) {");
    w.push_indent();
    w.line("uint block_end_inner = min(block + BLOCK, kv_end);");
    w.line("uint block_size = block_end_inner - block;");
    w.blank();

    w.line("// Stage 1: per-kv dot products (thread-per-kv within the block).");
    w.line("if (tid < block_size) {");
    w.push_indent();
    w.line("uint kv = block + tid;");
    w.line(format!(
        "device const {t}* q_row = Q + ((Q_HEAD_MAJOR ? (head_id_real * OUT_SEQ + query_row) : ((query_row * N_HEADS + head_id_real) * Q_SEQ)) * QK_DIM);"
    ));
    w.line(format!(
        "device const {t}* k_row = K + (kv_head * capacity + kv) * QK_DIM;"
    ));
    w.line("float dot = 0.0f;");
    w.line("for (uint d = 0u; d < QK_DIM; ++d) {");
    w.push_indent();
    w.line("dot += float(q_row[d]) * float(k_row[d]);");
    w.pop_indent();
    w.line("}");
    w.line("s_block[tid] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // Stage 2a: block max via simd reduce.
    w.line("float t_max = (tid < block_size) ? s_block[tid] : -INFINITY;");
    w.line("t_max = simd_max(t_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float m_block = reduce_buf[0];");
    w.line("float m_new = max(m, m_block);");
    w.line("float alpha = metal::precise::exp(m - m_new);");
    w.blank();

    // Stage 2b: exp(s - m_new), sum.
    w.line("if (tid < block_size) s_block[tid] = metal::precise::exp(s_block[tid] - m_new);");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float t_sum = (tid < block_size) ? s_block[tid] : 0.0f;");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float l_block = reduce_buf[0];");
    w.line("l = l * alpha + l_block;");
    w.blank();

    // Stage 3: o_d update, Σ_i beta_i * V[block+i, d].
    w.line("if (owns_d) {");
    w.push_indent();
    w.line("float contrib = 0.0f;");
    w.line("for (uint i = 0u; i < block_size; ++i) {");
    w.push_indent();
    w.line("uint kv = block + i;");
    w.line(format!(
        "device const {t}* v_row = V + (kv_head * capacity + kv) * V_DIM;"
    ));
    w.line("contrib += s_block[i] * float(v_row[tid]);");
    w.pop_indent();
    w.line("}");
    w.line("o_d = o_d * alpha + contrib;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("m = m_new;");
    w.pop_indent();
    w.line("}");
    w.blank();

    // Write partial state to global partials buffer.
    w.line("uint query_head_flat = query_row * N_HEADS + head_id_real;");
    w.line("uint partial_base = LAYER_IDX * PARTIALS_STRIDE_PER_LAYER");
    w.line("                  + query_head_flat * PARTIALS_STRIDE_PER_HEAD");
    w.line("                  + chunk_id * (V_DIM + 2u);");
    w.line("if (tid == 0u) {");
    w.push_indent();
    w.line("partials[partial_base + 0u] = m;");
    w.line("partials[partial_base + 1u] = l;");
    w.pop_indent();
    w.line("}");
    w.line("if (owns_d) {");
    w.push_indent();
    w.line("partials[partial_base + 2u + tid] = o_d;");
    w.pop_indent();
    w.line("}");
    w.line("// Make partials visible to other TGs that observe our counter.");
    w.line("threadgroup_barrier(mem_flags::mem_device);");
    w.blank();

    // Atomic increment, broadcast to TG.
    w.line("threadgroup uint prev_count_smem[1];");
    w.line("if (tid == 0u) {");
    w.push_indent();
    w.line("uint counter_idx = LAYER_IDX * COUNTER_STRIDE_PER_LAYER + query_head_flat;");
    w.line(
        "uint prev = atomic_fetch_add_explicit(&counter[counter_idx], 1u, memory_order_relaxed);",
    );
    w.line("prev_count_smem[0] = prev;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("uint prev_count = prev_count_smem[0];");
    w.blank();

    w.line("// Non-last chunks exit. Last chunk to increment is the combiner.");
    w.line("if (prev_count + 1u != active_chunks) return;");
    w.blank();

    // We are the LAST. Reset counter for next decode step before
    // combine (so even if the GPU re-orders the writes, future TGs
    // observe the reset only after combine completes).
    w.line("if (tid == 0u) {");
    w.push_indent();
    w.line("uint counter_idx = LAYER_IDX * COUNTER_STRIDE_PER_LAYER + query_head_flat;");
    w.line("atomic_store_explicit(&counter[counter_idx], 0u, memory_order_relaxed);");
    w.pop_indent();
    w.line("}");
    w.blank();

    // Acquire — observe all chunks' partials.
    w.line("threadgroup_barrier(mem_flags::mem_device);");
    w.blank();

    w.line("// Combine: online-softmax-merge across `active_chunks` partials.");
    w.line("if (owns_d) {");
    w.push_indent();
    w.line("float final_m = -INFINITY;");
    w.line("float final_l = 0.0f;");
    w.line("float final_o = 0.0f;");
    w.line("for (uint c = 0u; c < active_chunks; ++c) {");
    w.push_indent();
    w.line("uint c_base = LAYER_IDX * PARTIALS_STRIDE_PER_LAYER");
    w.line("            + query_head_flat * PARTIALS_STRIDE_PER_HEAD");
    w.line("            + c * (V_DIM + 2u);");
    w.line("float m_c = partials[c_base + 0u];");
    w.line("float l_c = partials[c_base + 1u];");
    w.line("float o_c = partials[c_base + 2u + tid];");
    w.line("float m_new = max(final_m, m_c);");
    w.line("float alpha_c = metal::precise::exp(final_m - m_new);");
    w.line("float beta_c  = metal::precise::exp(m_c       - m_new);");
    w.line("final_l = final_l * alpha_c + l_c * beta_c;");
    w.line("final_o = final_o * alpha_c + o_c * beta_c;");
    w.line("final_m = m_new;");
    w.pop_indent();
    w.line("}");
    w.line(format!(
        "out[(query_row * N_HEADS + head_id_real) * V_DIM + tid] = {t}(final_o / final_l);"
    ));
    w.pop_indent();
    w.line("}");
}

/// Flash-attention 2-pass online-softmax SDPA decode.
///
/// One TG per query head. KV is streamed in blocks of `BLOCK`
/// positions; per block we compute the partial Q·K dots, update the
/// running (max, sum, output) state via the online-softmax recurrence,
/// then move on. SMEM footprint is BLOCK floats (256 B at BLOCK=64)
/// regardless of `kv_len` — the entire long-context win over the
/// legacy emit, which sized scratch to `MAX_KV_LEN=4096` (16 KB) and
/// capped occupancy at 2 TG/SM on M5.
///
/// Online softmax recurrence (Milakov & Gimelshein, 2018):
///   m_new = max(m, max(s_block))
///   alpha = exp(m - m_new)
///   l_new = l * alpha + sum_i exp(s_block[i] - m_new)
///   o_new[d] = o[d] * alpha + sum_i exp(s_block[i] - m_new) * V[i, d]
/// After the final block, `out = o / l`.
fn emit_flash_online(w: &mut MslWriter, t: &str) {
    w.line("// Flash-attention 2-pass: block-streamed online softmax.");
    w.line("threadgroup float s_block[BLOCK];");
    w.line("threadgroup float reduce_buf[32];");
    w.blank();

    w.line("// Per-thread running state. Threads tid < V_DIM own one");
    w.line("// output dim each; the rest help with Stage 1/2 only.");
    w.line("bool owns_d = tid < V_DIM;");
    w.line("float o_d = 0.0f;");
    w.line("float m = -INFINITY;");
    w.line("float l = 0.0f;");
    w.blank();

    w.line("for (uint block = 0u; block < kv_len_eff; block += BLOCK) {");
    w.push_indent();
    w.line("uint block_end = min(block + BLOCK, kv_len_eff);");
    w.line("uint block_size = block_end - block;");
    w.blank();

    // ── Stage 1: per-block Q·K dot products ────────────────
    // Thread tid < block_size computes one full dot Q · K[block + tid].
    // For BLOCK=64 and TG_SIZE=128, threads 64..127 are idle here but
    // they cooperate in the reductions below.
    w.line("// Stage 1: s_block[i] = (Q · K[block + i]) * SCALE  for i ∈ [0, block_size).");
    w.line("if (tid < block_size) {");
    w.push_indent();
    w.line("uint kv = block + tid;");
    w.line(format!(
        "device const {t}* q_row = Q + ((Q_HEAD_MAJOR ? (head_id_real * OUT_SEQ + query_row) : ((query_row * N_HEADS + head_id_real) * Q_SEQ)) * QK_DIM);"
    ));
    w.line(format!(
        "device const {t}* k_row = K + (kv_head * capacity + kv) * QK_DIM;"
    ));
    w.line("float dot = 0.0f;");
    w.line("for (uint d = 0u; d < QK_DIM; ++d) {");
    w.push_indent();
    w.line("dot += float(q_row[d]) * float(k_row[d]);");
    w.pop_indent();
    w.line("}");
    w.line("s_block[tid] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // ── Stage 2: find max(s_block) via simd reduce ───────────
    w.line("// Stage 2a: m_block = max(s_block[0..block_size]).");
    w.line("float t_max = (tid < block_size) ? s_block[tid] : -INFINITY;");
    w.line("t_max = simd_max(t_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float m_block = reduce_buf[0];");
    w.blank();

    w.line("float m_new = max(m, m_block);");
    w.line("float alpha = metal::precise::exp(m - m_new);");
    w.blank();

    // ── Stage 2b: compute beta[i] = exp(s_block[i] - m_new), reuse SMEM
    w.line("// Stage 2b: s_block[i] ← beta_i = exp(s_block[i] - m_new). l_block = Σ beta_i.");
    w.line("if (tid < block_size) {");
    w.push_indent();
    w.line("s_block[tid] = metal::precise::exp(s_block[tid] - m_new);");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    w.line("float t_sum = (tid < block_size) ? s_block[tid] : 0.0f;");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float l_block = reduce_buf[0];");
    w.blank();

    w.line("l = l * alpha + l_block;");
    w.blank();

    // ── Stage 3: per-block output update ──────────────────
    w.line("// Stage 3: o[d] = o[d]*alpha + Σ_i beta_i * V[block + i, d].");
    w.line("if (owns_d) {");
    w.push_indent();
    w.line("float contrib = 0.0f;");
    w.line("for (uint i = 0u; i < block_size; ++i) {");
    w.push_indent();
    w.line("uint kv = block + i;");
    w.line(format!(
        "device const {t}* v_row = V + (kv_head * capacity + kv) * V_DIM;"
    ));
    w.line("contrib += s_block[i] * float(v_row[tid]);");
    w.pop_indent();
    w.line("}");
    w.line("o_d = o_d * alpha + contrib;");
    w.pop_indent();
    w.line("}");
    w.line("// Required before next block overwrites s_block.");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();
    w.line("m = m_new;");
    w.pop_indent();
    w.line("}");
    w.blank();

    // ── Final: normalize and write output ──────────────────
    w.line("// Final: out[d] = o_d / l. Combining the divide into the");
    w.line("// last store hoists kv_len*(V_DIM-1) divides out of the loop.");
    w.line("if (owns_d) {");
    w.push_indent();
    w.line("float inv_l = 1.0f / l;");
    w.line(format!(
        "out[(query_row * N_HEADS + head_id_real) * V_DIM + tid] = {t}(o_d * inv_l);"
    ));
    w.pop_indent();
    w.line("}");
}

/// Legacy two-pass SDPA decode: materializes the full scores vector
/// in SMEM (size `MAX_KV_LEN * 4` bytes), runs a TG-wide max+sum
/// softmax, then does the `scores @ V` reduction.
///
/// Retained behind `FUSEKIT_SDPA_LEGACY=1` for A/B perf comparison
/// against the flash emit. At long context the SMEM cap (16 KB) holds
/// occupancy to 2 TGs/SM on M5 — the flash path's main edge.
fn emit_legacy_two_pass(w: &mut MslWriter, t: &str) {
    w.line("threadgroup float scores[MAX_KV_LEN];");
    w.line("threadgroup float reduce_buf[32];");
    w.blank();

    // Stage 1: thread-per-kv (one thread owns one kv position).
    // Each thread independently iterates QK_DIM serially. Tested
    // both thread-per-kv and simdgroup-coop variants and the
    // thread-per-kv path wins consistently — 128 threads parallel-
    // dispatching different kvs (vs 4 simdgroups doing 1 kv at a time)
    // beats the simd_sum cost on Apple GPU. Stage 3 below is the
    // opposite: the per-thread serial loop over kv_len there was the
    // long-context bottleneck, so cooperation helps.
    // Stage 1: thread-per-kv with vec4 inner loop.
    // Each thread owns one kv position and reads its full K row.
    // Vec4 loads (`<T>4` per access) cut memory transactions by 4×
    // vs scalar loads — Apple GPU coalesces these into a single
    // ~16-byte transaction per lane. QK_DIM must be a multiple of 4
    // (true for all Llama/Qwen variants: 64/128/256).
    w.line("// Stage 1: dot product Q · K[kv] per kv position (thread-per-kv, vec4).");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line(format!(
        "device const {t}4* q_row4 = (device const {t}4*)(Q + ((Q_HEAD_MAJOR ? (head_id_real * OUT_SEQ + query_row) : ((query_row * N_HEADS + head_id_real) * Q_SEQ)) * QK_DIM));"
    ));
    w.line(format!(
        "device const {t}4* k_row4 = (device const {t}4*)(K + (kv_head * capacity + kv) * QK_DIM);"
    ));
    w.line("float dot = 0.0f;");
    w.line("for (uint d4 = 0u; d4 < QK_DIM / 4u; ++d4) {");
    w.push_indent();
    w.line(format!("{t}4 q4 = q_row4[d4];"));
    w.line(format!("{t}4 k4 = k_row4[d4];"));
    w.line("dot += float(q4.x) * float(k4.x) + float(q4.y) * float(k4.y)");
    w.line("     + float(q4.z) * float(k4.z) + float(q4.w) * float(k4.w);");
    w.pop_indent();
    w.line("}");
    w.line("scores[kv] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    w.line("// Stage 2a: threadgroup max(scores).");
    w.line("float t_max = -INFINITY;");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line("t_max = max(t_max, scores[kv]);");
    w.pop_indent();
    w.line("}");
    w.line("t_max = simd_max(t_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_max = reduce_buf[0];");
    w.blank();

    w.line("// Stage 2b: exp(scores - max) and threadgroup sum.");
    w.line("float t_sum = 0.0f;");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line("scores[kv] = metal::precise::exp(scores[kv] - global_max);");
    w.line("t_sum += scores[kv];");
    w.pop_indent();
    w.line("}");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_sum = reduce_buf[0];");
    w.blank();

    w.line("float inv_sum = 1.0f / global_sum;");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line("scores[kv] *= inv_sum;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // Stage 3: simdgroup-cooperative weighted sum with vec4 V loads.
    // Each SG owns V_DIM/(sgs_per_tg×4) groups of 4 dims. Within a
    // group, all 4 dims are loaded together via one vec4 read per kv;
    // lanes stripe the kv axis, simd_sum collapses each lane's partial.
    // 4× fewer memory transactions vs scalar V loads + 4 accumulators
    // amortise the simd_sum overhead across 4 outputs.
    //
    // Per-lane work at kv=800: 800/32 = 25 vec4 reads, 25 fma×4 per
    // accumulator, then 4 simd_sums. Critical path roughly matches
    // the scalar Stage 3 coop but with 4× the V bandwidth utilisation.
    w.line("// Stage 3: weighted sum of V along kv axis (simdgroup-coop + vec4).");
    w.line("for (uint d_base = simd_id * 4u; d_base < V_DIM; d_base += sgs_per_tg * 4u) {");
    w.push_indent();
    w.line("float4 partial = float4(0.0f);");
    w.line("for (uint kv = simd_lane; kv < kv_len_eff; kv += 32u) {");
    w.push_indent();
    w.line(format!(
        "device const {t}4* v_row4 = (device const {t}4*)(V + (kv_head * capacity + kv) * V_DIM);"
    ));
    w.line(format!("{t}4 v4 = v_row4[d_base / 4u];"));
    w.line("float s = scores[kv];");
    w.line("partial.x += s * float(v4.x);");
    w.line("partial.y += s * float(v4.y);");
    w.line("partial.z += s * float(v4.z);");
    w.line("partial.w += s * float(v4.w);");
    w.pop_indent();
    w.line("}");
    w.line("float4 acc;");
    w.line("acc.x = simd_sum(partial.x);");
    w.line("acc.y = simd_sum(partial.y);");
    w.line("acc.z = simd_sum(partial.z);");
    w.line("acc.w = simd_sum(partial.w);");
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    w.line(format!(
        "device {t}4* out4 = (device {t}4*)(out + (query_row * N_HEADS + head_id_real) * V_DIM);"
    ));
    w.line(format!(
        "out4[d_base / 4u] = {t}4({t}(acc.x), {t}(acc.y), {t}(acc.z), {t}(acc.w));"
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
}
