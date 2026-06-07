//! MSL emit for `FusionKind::RmsNormGateUpSwiglu`.
//!
//! ```text
//!   ┌─ kernel(x, W_norm, W_gate, W_up, out) ────────────────┐
//!   │   threadgroup float norm_y[HIDDEN]                     │
//!   │   rmsnorm(x, W_norm, norm_y, EPS)                      │
//!   │   barrier                                              │
//!   │                                                        │
//!   │   for i in [tid, N_INT, tg_size):                      │
//!   │     gate_i = Σ norm_y[k] · W_gate[i, k]                │
//!   │     up_i   = Σ norm_y[k] · W_up[i, k]                  │
//!   │     out[i] = sigmoid(gate_i) · gate_i · up_i           │
//!   └────────────────────────────────────────────────────────┘
//! ```
//!
//! Gate and Up never touch global memory — both live in per-thread
//! registers between the projection and the SwiGLU multiply. Saves
//! 2·N_INT·bf16 of round-trip traffic per layer vs the separate
//! RmsNormGateUp + SwigluActivation pair.

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
    if !matches!(fusion.kind, FusionKind::RmsNormGateUpSwiglu) {
        bail!(
            "rmsnorm_gateup_swiglu emitter: wrong kind {:?}",
            fusion.kind
        );
    }
    let (eps, quant) = match fusion.attrs {
        FusionAttrs::RmsNormGateUpSwiglu { eps, quant } => (eps, quant),
        _ => bail!("rmsnorm_gateup_swiglu emitter: attrs mismatch"),
    };

    let hidden = fusion
        .inputs
        .iter()
        .find(|b| b.role == "x")
        .and_then(|b| b.shape.last().copied())
        .unwrap_or(meta.hidden_size as i32)
        .max(1) as u32;
    let n_int = fusion
        .outputs
        .first()
        .and_then(|b| b.shape.last().copied())
        .unwrap_or(meta.intermediate_size as i32)
        .max(1) as u32;
    let is_quant = quant.is_some();
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: RmsNormGateUpSwiglu (eps = {}, hidden = {}, n_int = {}, quant = {:?}, dt = {})",
        fusion.id, eps, hidden, n_int, quant, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    if is_quant {
        // Buffer order matches Python install: x, norm_w, W_gate_packed,
        // W_up_packed, W_gate_scales, W_up_scales, W_gate_biases,
        // W_up_biases, out.
        w.line(format!("device const {t} *x          [[buffer(0)]],"));
        w.line(format!("device const {t} *norm_w     [[buffer(1)]],"));
        w.line("device const uint   *W_gate     [[buffer(2)]],");
        w.line("device const uint   *W_up       [[buffer(3)]],");
        w.line(format!("device const {t}   *W_gate_s   [[buffer(4)]],"));
        w.line(format!("device const {t}   *W_up_s     [[buffer(5)]],"));
        w.line(format!("device const {t}   *W_gate_b   [[buffer(6)]],"));
        w.line(format!("device const {t}   *W_up_b     [[buffer(7)]],"));
        w.line(format!("device {t}       *out        [[buffer(8)]],"));
    } else {
        w.line(format!("device const {t} *x        [[buffer(0)]],"));
        w.line(format!("device const {t} *norm_w   [[buffer(1)]],"));
        w.line(format!("device const {t} *W_gate   [[buffer(2)]],"));
        w.line(format!("device const {t} *W_up     [[buffer(3)]],"));
        w.line(format!("device {t}       *out      [[buffer(4)]],"));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint HIDDEN = {hidden};"));
    w.writeln(format_args!("constexpr uint N_INT  = {n_int};"));
    w.writeln(format_args!("constexpr float EPS   = {eps};"));
    w.blank();

    // ── RMSNorm into threadgroup memory ───────────────────────────
    w.line("// Stage 1: RMSNorm. norm_y = x / sqrt(mean(x²) + EPS) * norm_w.");
    w.line("threadgroup float norm_y[HIDDEN];");
    w.line("threadgroup float sum_sq_partial[32];");
    w.blank();
    crate::codegen::kernels::emit_rmsnorm_reduction(w, hidden, "x", "sum_sq_partial");
    w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS);");
    crate::codegen::kernels::emit_norm_writeback_with_act_ty(w, hidden, "x", "norm_w", t);
    w.blank();

    // ── Fused proj + swiglu (simdgroup-cooperative + bfloat4) ──
    // Each simdgroup of 32 threads handles ONE output row. Threads
    // in a simdgroup read adjacent positions (coalesced) along the
    // HIDDEN reduction axis. simd_sum collapses the partial dots.
    w.line("// Stage 2: 32-thread cooperative dot product per output.");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint i = gid * sgs_per_tg + simd_id;");
    w.line("if (i >= N_INT) return;");
    w.line("float acc_g = 0.0f;");
    w.line("float acc_u = 0.0f;");
    if let Some(q) = quant {
        let bits = q.bits;
        let group_size = q.group_size;
        let packed_words = crate::codegen::kernels::quant_packed_words(hidden as u32, bits);
        let pack_factor = match bits {
            3 | 5 => 8,
            6 => 4,
            _ => 32 / bits.max(1),
        };
        let packs_per_thread = if bits == 2 { 1 } else { 2 };
        let block_size = pack_factor * packs_per_thread * 32;
        let use_mlx_tg_qmv = group_size == 64
            && matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
            && (hidden as u32).is_multiple_of(block_size);
        let emit_fallback = |w: &mut MslWriter| {
            // Two parallel dequant GEMVs sharing the simdgroup. Each
            // simdgroup row `i` computes acc_g and acc_u from the same
            // norm_y SMEM stream.
            let packed_g = format!("W_gate + i * {packed_words}u");
            let packed_u = format!("W_up   + i * {packed_words}u");
            let scales_g = format!("W_gate_s + i * (HIDDEN / {group_size}u)");
            let scales_u = format!("W_up_s   + i * (HIDDEN / {group_size}u)");
            let biases_g = format!("W_gate_b + i * (HIDDEN / {group_size}u)");
            let biases_u = format!("W_up_b   + i * (HIDDEN / {group_size}u)");
            crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
                w,
                "acc_g",
                &packed_g,
                &scales_g,
                &biases_g,
                "HIDDEN",
                bits,
                group_size,
                crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                    bits,
                    hidden as u32,
                    n_int as u32,
                ),
                |k| format!("norm_y[{k}]"),
            );
            crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
                w,
                "acc_u",
                &packed_u,
                &scales_u,
                &biases_u,
                "HIDDEN",
                bits,
                group_size,
                crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                    bits,
                    hidden as u32,
                    n_int as u32,
                ),
                |k| format!("norm_y[{k}]"),
            );
        };
        if use_mlx_tg_qmv {
            w.line("#if FUSEKIT_HAVE_MLX_NAX");
            w.push_indent();
            w.writeln(format_args!(
                "fusekit_qmv_fast_tg2_partial_impl<{t}, {group_size}, {bits}>(W_gate, W_gate_s, W_gate_b, W_up, W_up_s, W_up_b, norm_y, HIDDEN, int(i), simd_lane, &acc_g, &acc_u);"
            ));
            w.pop_indent();
            w.line("#else");
            w.push_indent();
            emit_fallback(w);
            w.pop_indent();
            w.line("#endif");
        } else {
            emit_fallback(w);
        }
    } else if hidden % 8 == 0 {
        // 16-byte vector loads = full Apple GPU memory transaction width.
        w.line(format!(
            "device const {t}4* Wg4 = (device const {t}4*)(W_gate + i * HIDDEN);"
        ));
        w.line(format!(
            "device const {t}4* Wu4 = (device const {t}4*)(W_up   + i * HIDDEN);"
        ));
        w.line("threadgroup float4* ny4 = (threadgroup float4*)norm_y;");
        w.line("for (uint k4 = simd_lane * 2u; k4 < HIDDEN / 4u; k4 += 64u) {");
        w.push_indent();
        w.line("float4 ny_a = ny4[k4];");
        w.line("float4 ny_b = ny4[k4 + 1u];");
        w.line("acc_g += dot(ny_a, float4(Wg4[k4]))     + dot(ny_b, float4(Wg4[k4+1u]));");
        w.line("acc_u += dot(ny_a, float4(Wu4[k4]))     + dot(ny_b, float4(Wu4[k4+1u]));");
        w.pop_indent();
        w.line("}");
    } else if hidden % 4 == 0 {
        w.line(format!(
            "device const {t}4* Wg4 = (device const {t}4*)(W_gate + i * HIDDEN);"
        ));
        w.line(format!(
            "device const {t}4* Wu4 = (device const {t}4*)(W_up   + i * HIDDEN);"
        ));
        w.line("threadgroup float4* ny4 = (threadgroup float4*)norm_y;");
        w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
        w.push_indent();
        w.line("float4 ny = ny4[k4];");
        w.line("acc_g += dot(ny, float4(Wg4[k4]));");
        w.line("acc_u += dot(ny, float4(Wu4[k4]));");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint kx = simd_lane; kx < HIDDEN; kx += 32u) {");
        w.push_indent();
        w.line("float ny = norm_y[kx];");
        w.line("acc_g += ny * float(W_gate[i * HIDDEN + kx]);");
        w.line("acc_u += ny * float(W_up[i * HIDDEN + kx]);");
        w.pop_indent();
        w.line("}");
    }
    w.line("acc_g = simd_sum(acc_g);");
    w.line("acc_u = simd_sum(acc_u);");
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    w.line(format!("{t} gate_v = {t}(acc_g);"));
    w.line(format!("{t} up_v = {t}(acc_u);"));
    match std::env::var("FUSEKIT_DEBUG_RMSGU_SWIGLU_OUT")
        .ok()
        .as_deref()
    {
        Some("gate") => {
            w.line("out[i] = gate_v;");
        }
        Some("up") => {
            w.line("out[i] = up_v;");
        }
        _ => {
            w.line(format!("{t} one_v = {t}(1.0f);"));
            w.line(format!(
                "{t} sig_y = one_v / (one_v + metal::exp(metal::abs(gate_v)));"
            ));
            w.line(format!(
                "{t} sig_v = (gate_v < {t}(0.0f)) ? sig_y : {t}(one_v - sig_y);"
            ));
            w.line(format!("{t} silu_v = {t}(gate_v * sig_v);"));
            w.line(format!("out[i] = {t}(silu_v * up_v);"));
        }
    }
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rmsnorm_gateup_swiglu_f{}", fusion.id)
}
