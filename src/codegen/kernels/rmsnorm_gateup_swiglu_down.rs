//! MSL emit for `FusionKind::RmsNormGateUpSwigluDown`.
//!
//! ```text
//!   ┌─ kernel(x, W_norm, W_gate, W_up, W_down, out) ─────────────┐
//!   │   threadgroup float norm_y[HIDDEN]                          │
//!   │   threadgroup float silu_up[N_INT]                          │
//!   │   rmsnorm(x, W_norm, norm_y, EPS)                           │
//!   │   barrier                                                   │
//!   │   for i in [tid, N_INT, tg_size):                           │
//!   │     gate_i = Σ norm_y[k] · W_gate[i, k]                     │
//!   │     up_i   = Σ norm_y[k] · W_up[i, k]                       │
//!   │     silu_up[i] = sigmoid(gate_i) · gate_i · up_i            │
//!   │   barrier                                                   │
//!   │   for d in [tid, HIDDEN, tg_size):                          │
//!   │     out[d] = Σ silu_up[k] · W_down[d, k] + x[d]             │
//!   └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! SwiGLU output (`silu_up`) lives in threadgroup memory, so the
//! down-projection's inner reduction reads from SLM instead of DRAM.
//! Residual `x` is read once from DRAM at the final add.
//!
//! SMEM budget (Qwen-0.5B): norm_y 3584 B + silu_up 19456 B + reduce
//! buf 128 B ≈ 23 KB. Safety budget is 28 KB.

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
    if !matches!(fusion.kind, FusionKind::RmsNormGateUpSwigluDown) {
        bail!(
            "rmsnorm_gateup_swiglu_down emitter: wrong kind {:?}",
            fusion.kind
        );
    }
    let FusionAttrs::RmsNormGateUpSwigluDown { eps, quant: _ } = fusion.attrs else {
        bail!("rmsnorm_gateup_swiglu_down emitter: attrs mismatch");
    };

    let hidden = meta.hidden_size;
    let n_int = meta.intermediate_size;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: RmsNormGateUpSwigluDown (eps = {}, hidden = {}, n_int = {}, dt = {})",
        fusion.id, eps, hidden, n_int, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x        [[buffer(0)]],"));
    w.line(format!("device const {t} *norm_w   [[buffer(1)]],"));
    w.line(format!("device const {t} *W_gate   [[buffer(2)]],"));
    w.line(format!("device const {t} *W_up     [[buffer(3)]],"));
    w.line(format!("device const {t} *W_down   [[buffer(4)]],"));
    w.line(format!("device {t}       *out      [[buffer(5)]],"));
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

    // ── Stage 1: RMSNorm into SMEM ─────────────────────────────
    w.line("// Stage 1: RMSNorm into norm_y[HIDDEN] (SMEM).");
    w.line("threadgroup float norm_y[HIDDEN];");
    w.line("threadgroup float silu_up[N_INT];");
    w.line("threadgroup float sum_sq_partial[32];");
    w.blank();
    crate::codegen::kernels::emit_rmsnorm_reduction(w, hidden, "x", "sum_sq_partial");
    w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS);");
    crate::codegen::kernels::emit_norm_writeback_with_act_ty(w, hidden, "x", "norm_w", t);
    w.blank();

    // ── Stage 2: gate · up · swiglu → silu_up in SMEM ─────────
    w.line("// Stage 2: gate/up projections + SwiGLU; result stays in SMEM.");
    w.line("for (uint i = tid; i < N_INT; i += tg_size) {");
    w.push_indent();
    w.line("float acc_g = 0.0f;");
    w.line("float acc_u = 0.0f;");
    w.line("for (uint kx = 0u; kx < HIDDEN; ++kx) {");
    w.push_indent();
    w.line("float ny = norm_y[kx];");
    w.line("acc_g += ny * float(W_gate[i * HIDDEN + kx]);");
    w.line("acc_u += ny * float(W_up[i * HIDDEN + kx]);");
    w.pop_indent();
    w.line("}");
    w.line("silu_up[i] = acc_g / (1.0f + exp(-acc_g)) * acc_u;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // ── Stage 3: down_proj + residual ─────────────────────────
    w.line("// Stage 3: down_proj reads silu_up from SMEM, adds residual x.");
    w.line("for (uint d = tid; d < HIDDEN; d += tg_size) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    w.line("for (uint k = 0u; k < N_INT; ++k) {");
    w.push_indent();
    w.line("acc += silu_up[k] * float(W_down[d * N_INT + k]);");
    w.pop_indent();
    w.line("}");
    w.line(format!("out[d] = {t}(acc + float(x[d]));"));
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rmsnorm_gateup_swiglu_down_f{}", fusion.id)
}
