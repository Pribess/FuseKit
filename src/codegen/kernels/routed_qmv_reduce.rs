use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_routed_qmv_reduce_f{}", fusion.id)
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::RoutedQmvReduce) {
        bail!("RoutedQmvReduce emitter called for non-RoutedQmvReduce fusion");
    }
    let FusionAttrs::RoutedQmvReduce { target } = fusion.attrs else {
        bail!("RoutedQmvReduce attrs missing");
    };
    if fusion.inputs.len() != 7 || fusion.outputs.len() != 1 {
        bail!("RoutedQmvReduce malformed inputs/outputs");
    }

    let x_shape = &fusion.inputs[0].shape;
    let w_shape = &fusion.inputs[1].shape;
    let s_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if x_shape.len() < 5 || w_shape.len() != 3 || s_shape.len() != 3 || out_shape.len() < 3 {
        bail!("RoutedQmvReduce unsupported shape");
    }

    let t = crate::codegen::kernels::act_ty(meta);
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, t);
    let fn_name = function_name(fusion);
    let k_dim = target.k_dim;
    let out_dim = target.out_dim;
    let n_experts = w_shape[0].max(1) as u32;
    let packed = w_shape[2].max(1) as u32;
    let n_groups = s_shape[2].max(1) as u32;
    let tokens = out_shape.get(1).copied().unwrap_or(1).max(1) as u32;
    let active_tokens = crate::target::current_chip().active_tokens_for_shape(out_shape);
    let top_k = x_shape[2..x_shape.len().saturating_sub(1)]
        .iter()
        .copied()
        .filter(|&d| d > 0)
        .product::<i32>()
        .max(1) as u32;

    w.line(format!(
        "// Fusion #{}: RoutedQmvReduce (K={}, routes={}, O={}, experts={}, bits={}, gs={})",
        fusion.id, k_dim, top_k, out_dim, n_experts, target.bits, target.group_size
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x [[buffer(0)]],"));
    w.line("device const uint *wq [[buffer(1)]],");
    w.line(format!("device const {t} *scales [[buffer(2)]],"));
    w.line(format!("device const {t} *biases [[buffer(3)]],"));
    w.line("device const uint *row_idx [[buffer(4)]],");
    w.line("device const uint *expert_idx [[buffer(5)]],");
    w.line(format!("device const {t} *route_weight [[buffer(6)]],"));
    w.line(format!("device {out_t} *out0 [[buffer(7)]],"));
    w.line("uint tid [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint K = {k_dim};"));
    w.line(format!("constexpr uint N_EXPERTS = {n_experts};"));
    w.line(format!("constexpr uint O = {out_dim};"));
    w.line(format!("constexpr uint PACKED = {packed};"));
    w.line(format!("constexpr uint N_GROUPS = {n_groups};"));
    w.line(format!("constexpr uint TOKENS = {tokens};"));
    w.line(format!("constexpr uint ACTIVE_TOKENS = {active_tokens};"));
    w.line(format!("constexpr uint TOP_K = {top_k};"));
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint sg = gid * sgs_per_tg + simd_id;");
    w.line("uint n_work = ACTIVE_TOKENS * O;");
    w.line("if (sg >= n_work) return;");
    w.line("uint tok = sg / O;");
    w.line("uint o = sg % O;");
    w.line("float acc = 0.0f;");
    w.line("for (uint slot = 0u; slot < TOP_K; ++slot) {");
    w.push_indent();
    w.line("uint route_idx = tok * TOP_K + slot;");
    w.line("uint x_row = uint(row_idx[route_idx]);");
    w.line("uint expert = min(uint(expert_idx[route_idx]), N_EXPERTS - 1u);");
    w.line("float dot = 0.0f;");
    crate::codegen::kernels::emit_int4_gemv_inner(
        w,
        "dot",
        "wq + (expert * O + o) * PACKED",
        "scales + (expert * O + o) * N_GROUPS",
        "biases + (expert * O + o) * N_GROUPS",
        "K",
        target.bits,
        target.group_size,
        |off| format!("float(x[x_row * K + {off}])"),
    );
    w.line("dot = simd_sum(dot);");
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    w.line(format!(
        "acc += float({t}(dot)) * float(route_weight[route_idx]);"
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("if (simd_lane != 0u) return;");
    w.line(format!("out0[tok * O + o] = {out_t}(acc);"));
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
