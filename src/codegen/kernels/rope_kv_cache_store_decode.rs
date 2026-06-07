//! MSL emit for `FusionKind::RopeKvCacheStoreDecode`.
//!
//! Combines RoPE on the K vector with the KV-cache store step that
//! mlx-lm normally splits into `cache.update_and_fetch(...)`. After
//! this kernel runs, `cache_keys[0, h, offset, :]` and
//! `cache_values[0, h, offset, :]` are populated for every kv_head;
//! SDPA can then read the cumulative cache for the current decode
//! step.
//!
//! ```text
//!     K (n_kv, head_dim)         ─► RoPE ─► cache_keys[0, h, offset, :]
//!     V (n_kv, head_dim)         ─► copy ─► cache_values[0, h, offset, :]
//! ```

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::RopeKvCacheStoreDecode) {
        bail!("rope_kv_cache_store emitter: wrong kind {:?}", fusion.kind);
    }
    let FusionAttrs::RopeKvCacheStoreDecode { theta } = fusion.attrs else {
        bail!("rope_kv_cache_store emitter: attrs mismatch");
    };

    let n_kv = meta.num_key_value_heads;
    let head_dim = meta.head_dim;
    let half = head_dim / 2;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_tokens_for_shape(&b.shape))
        .unwrap_or_else(|| crate::target::current_chip().active_m(8))
        .max(1);

    w.line(format!(
        "// Fusion #{}: RopeKvCacheStoreDecode (active_m={}, n_kv={}, head_dim={}, theta={})",
        fusion.id, active_m, n_kv, head_dim, theta
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *K            [[buffer(0)]],"));
    w.line(format!("device const {t} *V            [[buffer(1)]],"));
    w.line(format!("device {t}       *cache_keys   [[buffer(2)]],"));
    w.line(format!("device {t}       *cache_values [[buffer(3)]],"));
    w.line("constant uint       &offset       [[buffer(4)]],");
    w.line("constant uint       &capacity     [[buffer(5)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint N_KV     = {n_kv};"));
    w.writeln(format_args!("constexpr uint ACTIVE_M = {active_m};"));
    w.writeln(format_args!("constexpr uint HEAD_DIM = {head_dim};"));
    w.writeln(format_args!("constexpr uint HALF     = {half};"));
    w.line("constexpr uint K_RANGE = N_KV * HEAD_DIM;");
    w.line("constexpr uint TOKEN_RANGE = 2u * K_RANGE;");
    // BASE_LOG2 matches mlx's `mx.fast.rope` — see rope_q_decode for
    // the full ULP-equivalence rationale (exp2 / fast::cos / fast::sin).
    let log2_theta = (theta as f64).log2() as f32;
    w.writeln(format_args!("constexpr float BASE_LOG2 = {log2_theta:e}f;"));
    w.blank();

    w.line("uint g = gid * tg_size + tid;");
    w.line("uint row = g / TOKEN_RANGE;");
    w.line("uint local = g - row * TOKEN_RANGE;");
    w.line("if (row >= ACTIVE_M) return;");
    // K work: g ∈ [0, K_RANGE) — apply RoPE pair-rotation, write to cache slot.
    // V work: g ∈ [K_RANGE, V_RANGE) — straight copy to cache slot.
    w.line("bool is_k = (local < K_RANGE);");
    w.line("bool is_v = (local >= K_RANGE && local < TOKEN_RANGE);");
    w.line("uint k_idx = is_k ? local : 0u;");
    w.line("uint v_idx = is_v ? (local - K_RANGE) : 0u;");
    w.blank();
    // ── K-side: pair-read + RoPE ──────────────────────────────
    w.line("uint kh = k_idx / HEAD_DIM;");
    w.line("uint kd = k_idx % HEAD_DIM;");
    w.line("uint pair_idx = (kd < HALF) ? kd : (kd - HALF);");
    w.line("float d_norm = float(pair_idx) / float(HALF);");
    w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
    w.line("float angle = float(offset + row) * inv_freq;");
    w.line("float c = metal::fast::cos(angle);");
    w.line("float s = metal::fast::sin(angle);");
    w.line("uint kbase = (row * N_KV + kh) * HEAD_DIM;");
    w.line("float lo = float(K[kbase + pair_idx]);");
    w.line("float hi = float(K[kbase + pair_idx + HALF]);");
    w.line("float k_result = (kd < HALF) ? (lo * c - hi * s) : (hi * c + lo * s);");
    // All threads sync — pair-wise reads done before writes.
    w.line("threadgroup_barrier(mem_flags::mem_device);");
    w.blank();
    // ── Write K (rotated) into cache_keys[0, kh, offset, kd] ──
    w.line("if (is_k) {");
    w.push_indent();
    w.line("uint cache_off = kh * capacity * HEAD_DIM + (offset + row) * HEAD_DIM + kd;");
    w.line(format!("cache_keys[cache_off] = {t}(k_result);"));
    w.pop_indent();
    w.line("}");
    w.blank();
    // ── Write V into cache_values[0, vh, offset, vd] ──
    w.line("if (is_v) {");
    w.push_indent();
    w.line("uint vh = v_idx / HEAD_DIM;");
    w.line("uint vd = v_idx % HEAD_DIM;");
    w.line("uint cache_off = vh * capacity * HEAD_DIM + (offset + row) * HEAD_DIM + vd;");
    w.line(format!(
        "cache_values[cache_off] = {t}(V[(row * N_KV * HEAD_DIM) + v_idx]);"
    ));
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rope_kv_cache_store_decode_f{}", fusion.id)
}
