//! MSL emit for `FusionKind::RopeQDecode`.
//!
//! Decode-time Q-rotation in place. Q has shape `(n_heads, head_dim)`.
//! Each pair `(d, d + half)` rotates by angle `θ_d · offset` where
//! `θ_d = THETA^(-2d/head_dim)` for `d ∈ [0, half)`. mlx-lm uses the
//! "non-traditional" rotation: the two halves of head_dim are rotated
//! together rather than adjacent pairs.
//!
//! ```text
//!   Q[h, d]        = Q[h, d]        · cos(θ_d · pos) - Q[h, d+half] · sin(θ_d · pos)
//!   Q[h, d+half]   = Q[h, d+half]   · cos(θ_d · pos) + Q[h, d]      · sin(θ_d · pos)
//! ```
//!
//! Multi-TG: each thread owns one (head, d) slot. The thread reads
//! BOTH its element and the partner element from the SAME pair so it
//! can compute its own new value with the right sign — no race.

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
    if !matches!(fusion.kind, FusionKind::RopeQDecode) {
        bail!("rope_q_decode emitter: wrong kind {:?}", fusion.kind);
    }
    let FusionAttrs::RopeQDecode { theta } = fusion.attrs else {
        bail!("rope_q_decode emitter: attrs mismatch");
    };

    let n_heads = meta.num_attention_heads;
    let head_dim = meta.head_dim;
    let half = head_dim / 2;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .or_else(|| fusion.outputs.first())
        .map(|b| crate::target::current_chip().active_tokens_for_shape(&b.shape))
        .unwrap_or_else(|| crate::target::current_chip().active_m_for_unshaped(8))
        .max(1);

    w.line(format!(
        "// Fusion #{}: RopeQDecode (active_m={}, n_heads={}, head_dim={}, theta={})",
        fusion.id, active_m, n_heads, head_dim, theta
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device {t}       *Q       [[buffer(0)]],"));
    w.line("constant uint       &offset  [[buffer(1)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint N_HEADS  = {n_heads};"));
    w.writeln(format_args!("constexpr uint ACTIVE_M = {active_m};"));
    w.writeln(format_args!("constexpr uint HEAD_DIM = {head_dim};"));
    w.writeln(format_args!("constexpr uint HALF     = {half};"));
    // mlx's `mx.fast.rope` Metal kernel (rope.metal::rope_impl) computes
    //   d        = pair_idx / HALF          (normalized to [0, 1))
    //   inv_freq = metal::exp2(-d * BASE)   (BASE = log2(theta))
    //   theta    = float(offset) * inv_freq
    //   c, s     = metal::fast::cos(theta), metal::fast::sin(theta)
    // We mirror that ULP-for-ULP. Bake `BASE` host-side in f64 →
    // truncated to f32; the resulting `inv_freq` matches mlx's
    // kernel byte-for-byte (same `exp2` argument, same precision),
    // and `metal::fast::cos`/`sin` produce identical bit patterns
    // because they're hardware intrinsics — not the libm-grade
    // versions Metal's plain `cos`/`sin` invoke.
    let log2_theta = (theta as f64).log2() as f32;
    w.writeln(format_args!("constexpr float BASE_LOG2 = {log2_theta:e}f;"));
    w.blank();

    w.line("uint g = gid * tg_size + tid;");
    w.line("bool active = (g < ACTIVE_M * N_HEADS * HEAD_DIM);");
    // All threads must reach the barrier, so we clamp inactive threads
    // to safe-but-unused indices and only do the write under `active`.
    w.line("uint row = active ? (g / (N_HEADS * HEAD_DIM)) : 0u;");
    w.line("uint in_row = active ? (g - row * N_HEADS * HEAD_DIM) : 0u;");
    w.line("uint h = active ? (in_row / HEAD_DIM) : 0u;");
    w.line("uint d = active ? (g % HEAD_DIM) : 0u;");
    w.line("uint pair_idx = (d < HALF) ? d : (d - HALF);");
    w.line("float d_norm = float(pair_idx) / float(HALF);");
    w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
    w.line("float angle = float(offset + row) * inv_freq;");
    w.line("float c = metal::fast::cos(angle);");
    w.line("float s = metal::fast::sin(angle);");
    w.line("uint base = (row * N_HEADS + h) * HEAD_DIM;");
    w.line("float lo = float(Q[base + pair_idx]);");
    w.line("float hi = float(Q[base + pair_idx + HALF]);");
    w.line("float result = (d < HALF) ? (lo * c - hi * s) : (hi * c + lo * s);");
    // Sync all reads of this TG's pair before any writes go in. The
    // pair's two halves live within the same TG provided HEAD_DIM
    // divides TG_SIZE (true for 0.5B, 1.5B — head_dim 64/128 vs 256).
    w.line("threadgroup_barrier(mem_flags::mem_device);");
    w.line(format!("if (active) Q[g] = {t}(result);"));

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rope_q_decode_f{}", fusion.id)
}

/// Emit `constexpr float FREQS[HALF] = { … };` with the RoPE frequencies
/// precomputed in host f64. mlx's `mx.fast.rope` C++ path computes the
/// per-pair freq with f64 intermediate precision before downcasting; we
/// match that here so the bf16 round-to-nearest of the rotated K/V
/// matches mlx bit-for-bit. Without this, Metal's inline `pow(THETA, …)`
/// rounds differently from mlx and the K-cache drifts by ~1% per
/// decode position on head_dim>=128.
pub fn emit_rope_freq_table(w: &mut MslWriter, theta: f64, half: usize) {
    let head_dim = (half * 2) as f64;
    w.line(format!("constexpr float FREQS[{half}] = {{", half = half));
    w.push_indent();
    // 8 entries per line for readability; values are f32 hex literals
    // would be more precise but Metal accepts decimal floats which
    // round-trip through Rust's default formatter at f32 ULP.
    let mut line = String::new();
    for i in 0..half {
        let exponent = -2.0_f64 * (i as f64) / head_dim;
        let freq = theta.powf(exponent) as f32;
        line.push_str(&format!("{:e}f", freq));
        if i + 1 < half {
            line.push_str(", ");
        }
        if (i + 1) % 8 == 0 || i + 1 == half {
            w.line(line.clone());
            line.clear();
        }
    }
    w.pop_indent();
    w.line("};");
}
