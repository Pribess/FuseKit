//! MSL emit for `FusionKind::RmsNormGateUp`.
//!
//! Pre-MLP norm + two projections. Same structure as `RmsNormQkv`
//! but with two outputs (gate, up) instead of three, and no biases
//! (Qwen-style MLPs use bias-free Linear).
//!
//! ```text
//!   ┌─ kernel(x, W_norm, W_gate, W_up) ─────────────────────┐
//!   │   threadgroup float norm_y[HIDDEN]                     │
//!   │   rmsnorm(x, W_norm, norm_y, EPS)                      │
//!   │   barrier                                              │
//!   │   gate[i] = Σ norm_y[k] · W_gate[i, k]                 │
//!   │   up[i]   = Σ norm_y[k] · W_up[i, k]                   │
//!   └────────────────────────────────────────────────────────┘
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
    if !matches!(fusion.kind, FusionKind::RmsNormGateUp) {
        bail!(
            "rmsnorm_gateup emitter called with wrong kind: {:?}",
            fusion.kind
        );
    }
    let FusionAttrs::RmsNormGateUp { eps, quant: _ } = fusion.attrs else {
        bail!("rmsnorm_gateup emitter: attrs mismatch");
    };

    let hidden = meta.hidden_size;
    let n_int = meta.intermediate_size;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: RmsNormGateUp (eps = {}, hidden = {}, intermediate = {}, dt = {})",
        fusion.id, eps, hidden, n_int, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x        [[buffer(0)]],"));
    w.line(format!("device const {t} *norm_w   [[buffer(1)]],"));
    w.line(format!("device const {t} *W_gate   [[buffer(2)]],"));
    w.line(format!("device const {t} *W_up     [[buffer(3)]],"));
    w.line(format!("device {t}       *gate_out [[buffer(4)]],"));
    w.line(format!("device {t}       *up_out   [[buffer(5)]],"));
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

    // RMSNorm into threadgroup memory.
    w.line("// Stage 1: RMSNorm. norm_y = x / sqrt(mean(x²) + EPS) * norm_w.");
    w.line("threadgroup float norm_y[HIDDEN];");
    w.line("threadgroup float sum_sq_partial[32];");
    w.blank();
    crate::codegen::kernels::emit_rmsnorm_reduction(w, hidden, "x", "sum_sq_partial");
    w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS);");
    crate::codegen::kernels::emit_norm_writeback_with_act_ty(w, hidden, "x", "norm_w", t);
    w.blank();

    emit_proj(w, "gate_out", "W_gate", t);
    emit_proj(w, "up_out", "W_up", t);

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

fn emit_proj(w: &mut MslWriter, out: &str, weight: &str, t: &str) {
    w.line(format!(
        "// {out}[i] = Σ norm_y[k] · {weight}[i, k]   (i ∈ [0, N_INT))"
    ));
    w.line("for (uint i = tid; i < N_INT; i += tg_size) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    w.line("for (uint kx = 0u; kx < HIDDEN; ++kx) {");
    w.push_indent();
    w.line(format!(
        "acc += norm_y[kx] * float({weight}[i * HIDDEN + kx]);"
    ));
    w.pop_indent();
    w.line("}");
    w.line(format!("{out}[i] = {t}(acc);"));
    w.pop_indent();
    w.line("}");
    w.blank();
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rmsnorm_gateup_f{}", fusion.id)
}
