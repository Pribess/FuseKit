use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta, QmvFanoutTarget};
use crate::task::ir::PartitionPlan;

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_routed_qmv_swiglu_f{}", fusion.id)
}

fn compatible(gate: QmvFanoutTarget, up: QmvFanoutTarget) -> bool {
    gate.k_dim == up.k_dim
        && gate.out_dim == up.out_dim
        && gate.active_rows == up.active_rows
        && gate.bits == up.bits
        && gate.group_size == up.group_size
        && matches!(gate.bits, 4 | 8)
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::RoutedQmvSwiglu) {
        bail!("RoutedQmvSwiglu emitter called for non-RoutedQmvSwiglu fusion");
    }
    let FusionAttrs::RoutedQmvSwiglu { gate, up } = fusion.attrs else {
        bail!("RoutedQmvSwiglu attrs missing");
    };
    if !compatible(gate, up) {
        bail!("RoutedQmvSwiglu gate/up target mismatch");
    }
    if fusion.inputs.len() != 9 || fusion.outputs.len() != 1 {
        bail!("RoutedQmvSwiglu malformed inputs/outputs");
    }

    let x_shape = &fusion.inputs[0].shape;
    let w_shape = &fusion.inputs[1].shape;
    let s_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if x_shape.is_empty() || w_shape.len() != 3 || s_shape.len() != 3 || out_shape.len() < 5 {
        bail!("RoutedQmvSwiglu unsupported shape");
    }

    let t = crate::codegen::kernels::act_ty(meta);
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, t);
    let fn_name = function_name(fusion);
    let k_dim = gate.k_dim;
    let out_dim = gate.out_dim;
    let n_experts = w_shape[0].max(1) as u32;
    let packed = w_shape[2].max(1) as u32;
    let n_groups = s_shape[2].max(1) as u32;
    let tokens = out_shape.get(1).copied().unwrap_or(1).max(1) as u32;
    let active_tokens = crate::target::current_chip().active_tokens_for_shape(out_shape);
    let top_k = out_shape.get(2).copied().unwrap_or(1).max(1) as u32;
    let routes_per_token = out_shape[2..out_shape.len().saturating_sub(1)]
        .iter()
        .copied()
        .filter(|&d| d > 0)
        .product::<i32>()
        .max(1) as u32;
    let active_routes = active_tokens.saturating_mul(routes_per_token).max(1);
    let chip = crate::target::current_chip();
    let chip_rows = chip.gatherqmm_rows_per_simdgroup();
    let rows_per_sg = if chip_rows > 1 && out_dim >= chip_rows && out_dim % chip_rows == 0 {
        chip_rows
    } else {
        1
    };
    let accum_rows = rows_per_sg.saturating_mul(2);

    w.line(format!(
        "// Fusion #{}: RoutedQmvSwiglu (K={}, routes={}, O={}, experts={}, bits={}, gs={}, rows/sg={})",
        fusion.id,
        k_dim,
        active_routes,
        out_dim,
        n_experts,
        gate.bits,
        gate.group_size,
        rows_per_sg
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x [[buffer(0)]],"));
    w.line("device const uint *gate_w [[buffer(1)]],");
    w.line(format!("device const {t} *gate_s [[buffer(2)]],"));
    w.line(format!("device const {t} *gate_b [[buffer(3)]],"));
    w.line("device const uint *up_w [[buffer(4)]],");
    w.line(format!("device const {t} *up_s [[buffer(5)]],"));
    w.line(format!("device const {t} *up_b [[buffer(6)]],"));
    w.line("device const uint *row_idx [[buffer(7)]],");
    w.line("device const uint *expert_idx [[buffer(8)]],");
    w.line(format!("device {out_t} *out0 [[buffer(9)]],"));
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
    w.line(format!("constexpr uint ACTIVE_ROUTES = {active_routes};"));
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));
    w.line("constexpr uint ROW_BLOCKS = (O + ROWS_PER_SG - 1u) / ROWS_PER_SG;");
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint sg = gid * sgs_per_tg + simd_id;");
    w.line("if (sg >= ACTIVE_ROUTES * ROW_BLOCKS) return;");
    w.line("uint route = sg / ROW_BLOCKS;");
    w.line("uint o_base = (sg % ROW_BLOCKS) * ROWS_PER_SG;");
    w.line("uint slot = route % TOP_K;");
    w.line("uint tok = (route / TOP_K) % TOKENS;");
    w.line("if (tok >= ACTIVE_TOKENS) return;");
    w.line("uint route_idx = tok * TOP_K + slot;");
    w.line("uint x_row = uint(row_idx[route_idx]);");
    w.line("uint expert = min(uint(expert_idx[route_idx]), N_EXPERTS - 1u);");
    for r in 0..rows_per_sg {
        w.line(format!("float acc_gate{r} = 0.0f;"));
    }
    for r in 0..rows_per_sg {
        w.line(format!("float acc_up{r} = 0.0f;"));
    }
    if rows_per_sg == 1 {
        crate::codegen::kernels::emit_int4_gemv_inner(
            w,
            "acc_gate0",
            "gate_w + (expert * O + o_base) * PACKED",
            "gate_s + (expert * O + o_base) * N_GROUPS",
            "gate_b + (expert * O + o_base) * N_GROUPS",
            "K",
            gate.bits,
            gate.group_size,
            |off| format!("float(x[x_row * K + {off}])"),
        );
        crate::codegen::kernels::emit_int4_gemv_inner(
            w,
            "acc_up0",
            "up_w + (expert * O + o_base) * PACKED",
            "up_s + (expert * O + o_base) * N_GROUPS",
            "up_b + (expert * O + o_base) * N_GROUPS",
            "K",
            gate.bits,
            gate.group_size,
            |off| format!("float(x[x_row * K + {off}])"),
        );
    } else {
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| {
                if r < rows_per_sg {
                    format!("acc_gate{r}")
                } else {
                    format!("acc_up{}", r - rows_per_sg)
                }
            },
            |r| {
                if r < rows_per_sg {
                    format!("gate_w + (expert * O + (o_base + {r}u)) * PACKED")
                } else {
                    format!(
                        "up_w + (expert * O + (o_base + {}u)) * PACKED",
                        r - rows_per_sg
                    )
                }
            },
            |r| {
                if r < rows_per_sg {
                    format!("gate_s + (expert * O + (o_base + {r}u)) * N_GROUPS")
                } else {
                    format!(
                        "up_s + (expert * O + (o_base + {}u)) * N_GROUPS",
                        r - rows_per_sg
                    )
                }
            },
            |r| {
                if r < rows_per_sg {
                    format!("gate_b + (expert * O + (o_base + {r}u)) * N_GROUPS")
                } else {
                    format!(
                        "up_b + (expert * O + (o_base + {}u)) * N_GROUPS",
                        r - rows_per_sg
                    )
                }
            },
            "K",
            gate.bits,
            gate.group_size,
            accum_rows,
            |off| format!("float(x[x_row * K + {off}])"),
        );
    }
    for r in 0..rows_per_sg {
        w.line(format!("acc_gate{r} = simd_sum(acc_gate{r});"));
        w.line(format!("acc_up{r} = simd_sum(acc_up{r});"));
    }
    w.line("if (simd_lane != 0u) return;");
    for r in 0..rows_per_sg {
        w.line(format!("if (o_base + {r}u < O) {{"));
        w.push_indent();
        w.line(format!("float g{r} = float({t}(acc_gate{r}));"));
        w.line(format!("float u{r} = float({t}(acc_up{r}));"));
        w.line(format!("float sig{r} = 1.0f / (1.0f + exp(-g{r}));"));
        w.line(format!(
            "out0[route * O + o_base + {r}u] = {out_t}((sig{r} * g{r}) * u{r});"
        ));
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
