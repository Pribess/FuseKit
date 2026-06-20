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
    let FusionAttrs::RoutedQmvReduce {
        target,
        input_preweighted,
        residual_linear,
        has_extra_residual,
    } = fusion.attrs
    else {
        bail!("RoutedQmvReduce attrs missing");
    };
    let mut expected_inputs = if input_preweighted { 6 } else { 7 };
    if residual_linear.is_some() {
        expected_inputs += 4 + usize::from(has_extra_residual);
    }
    if fusion.inputs.len() != expected_inputs || fusion.outputs.len() != 1 {
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
    let out_cols = crate::target::current_chip().routed_reduce_cols_per_simdgroup(
        target.bits,
        k_dim,
        out_dim,
        top_k,
    );
    let fold_routes = crate::codegen::kernels::routed_qmv_reduce_fold_routes(fusion);

    if let Some(shared) = residual_linear {
        let shared_base = if input_preweighted { 6 } else { 7 };
        let shared_x_shape = &fusion.inputs[shared_base].shape;
        let shared_w_shape = &fusion.inputs[shared_base + 1].shape;
        let shared_s_shape = &fusion.inputs[shared_base + 2].shape;
        if shared_x_shape.is_empty() || shared_w_shape.len() != 2 || shared_s_shape.len() != 2 {
            bail!("RoutedQmvReduce residual linear unsupported shape");
        }
        let shared_k = shared.k_dim;
        let shared_packed = shared_w_shape[1].max(1) as u32;
        let shared_groups = shared_s_shape[1].max(1) as u32;
        if shared.out_dim != out_dim || shared.active_rows != 1 || top_k + 1 > 8 {
            bail!("RoutedQmvReduce residual linear unsupported target");
        }
        let mut combined_cols = std::env::var("FUSEKIT_ROUTED_REDUCE_LINEAR_COLS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| (1..=4).contains(&n))
            .unwrap_or(1)
            .min(out_dim.max(1));
        while combined_cols > 1 && !out_dim.is_multiple_of(combined_cols) {
            combined_cols -= 1;
        }
        w.line(format!(
            "// Fusion #{}: RoutedQmvReduce+LinearResidual (route K={}, routes={}, shared K={}, O={}, experts={}, bits={}/{}, gs={}/{}, input_preweighted={}, extra_residual={}, cols={})",
            fusion.id,
            k_dim,
            top_k,
            shared_k,
            out_dim,
            n_experts,
            target.bits,
            shared.bits,
            target.group_size,
            shared.group_size,
            input_preweighted,
            has_extra_residual,
            combined_cols
        ));
        w.line(format!("kernel void {fn_name}("));
        w.push_indent();
        w.line(format!("device const {t} *x [[buffer(0)]],"));
        w.line("device const uint *wq [[buffer(1)]],");
        w.line(format!("device const {t} *scales [[buffer(2)]],"));
        w.line(format!("device const {t} *biases [[buffer(3)]],"));
        w.line("device const uint *row_idx [[buffer(4)]],");
        w.line("device const uint *expert_idx [[buffer(5)]],");
        let mut next_slot = 6u32;
        if !input_preweighted {
            w.line(format!(
                "device const {t} *route_weight [[buffer({next_slot})]],"
            ));
            next_slot += 1;
        }
        w.line(format!(
            "device const {t} *shared_x [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "device const uint *shared_w [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "device const {t} *shared_s [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "device const {t} *shared_b [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        if has_extra_residual {
            w.line(format!(
                "device const {t} *residual_extra [[buffer({next_slot})]],"
            ));
            next_slot += 1;
        }
        w.line(format!("device {out_t} *out0 [[buffer({next_slot})]],"));
        w.line("uint tid [[thread_position_in_threadgroup]],");
        w.line("uint gid [[threadgroup_position_in_grid]]");
        w.pop_indent();
        w.line(") {");
        w.push_indent();
        w.line(format!("constexpr uint ROUTE_K = {k_dim};"));
        w.line(format!("constexpr uint SHARED_K = {shared_k};"));
        w.line(format!("constexpr uint N_EXPERTS = {n_experts};"));
        w.line(format!("constexpr uint O = {out_dim};"));
        w.line(format!("constexpr uint ROUTE_PACKED = {packed};"));
        w.line(format!("constexpr uint ROUTE_GROUPS = {n_groups};"));
        w.line(format!("constexpr uint SHARED_PACKED = {shared_packed};"));
        w.line(format!("constexpr uint SHARED_GROUPS = {shared_groups};"));
        w.line(format!("constexpr uint TOKENS = {tokens};"));
        w.line(format!("constexpr uint ACTIVE_TOKENS = {active_tokens};"));
        w.line(format!("constexpr uint TOP_K = {top_k};"));
        w.line(format!("constexpr uint COLS = {combined_cols};"));
        w.line("constexpr uint O_BLOCKS = (O + COLS - 1u) / COLS;");
        w.line("constexpr uint PARTIALS = TOP_K + 1u;");
        w.line("threadgroup float partial[8 * COLS];");
        w.line("uint simd_lane = tid % 32u;");
        w.line("uint simd_id = tid / 32u;");
        w.line("uint n_work = ACTIVE_TOKENS * O_BLOCKS;");
        w.line("if (gid >= n_work) return;");
        w.line("uint tok = gid / O_BLOCKS;");
        w.line("uint o_base = (gid - tok * O_BLOCKS) * COLS;");
        w.line("if (simd_id < PARTIALS && simd_lane == 0u) {");
        w.push_indent();
        for r in 0..combined_cols {
            w.line(format!("partial[simd_id * COLS + {r}u] = 0.0f;"));
        }
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id < TOP_K) {");
        w.push_indent();
        w.line("uint slot = simd_id;");
        w.line("uint route_idx = tok * TOP_K + slot;");
        w.line("uint x_row = uint(row_idx[route_idx]);");
        w.line("uint expert = min(uint(expert_idx[route_idx]), N_EXPERTS - 1u);");
        for r in 0..combined_cols {
            w.line(format!("float dot{r} = 0.0f;"));
        }
        if combined_cols == 1 && target.bits == 4 && target.group_size == 64 && k_dim >= 1024 {
            crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
                w,
                "dot0",
                "wq + (expert * O + o_base) * ROUTE_PACKED",
                "scales + (expert * O + o_base) * ROUTE_GROUPS",
                "biases + (expert * O + o_base) * ROUTE_GROUPS",
                "ROUTE_K",
                target.bits,
                target.group_size,
                false,
                false,
                crate::codegen::kernels::QdotPolicy::Force,
                |off| format!("float(x[x_row * ROUTE_K + {off}])"),
            );
        } else if combined_cols == 1 {
            crate::codegen::kernels::emit_int4_gemv_inner(
                w,
                "dot0",
                "wq + (expert * O + o_base) * ROUTE_PACKED",
                "scales + (expert * O + o_base) * ROUTE_GROUPS",
                "biases + (expert * O + o_base) * ROUTE_GROUPS",
                "ROUTE_K",
                target.bits,
                target.group_size,
                |off| format!("float(x[x_row * ROUTE_K + {off}])"),
            );
        } else {
            crate::codegen::kernels::emit_int4_gemv_inner_multirow(
                w,
                |r| format!("dot{r}"),
                |r| format!("wq + (expert * O + (o_base + {r}u)) * ROUTE_PACKED"),
                |r| format!("scales + (expert * O + (o_base + {r}u)) * ROUTE_GROUPS"),
                |r| format!("biases + (expert * O + (o_base + {r}u)) * ROUTE_GROUPS"),
                "ROUTE_K",
                target.bits,
                target.group_size,
                combined_cols,
                |off| format!("float(x[x_row * ROUTE_K + {off}])"),
            );
        }
        for r in 0..combined_cols {
            w.line(format!("dot{r} = simd_sum(dot{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..combined_cols {
            if input_preweighted {
                w.line(format!("partial[slot * COLS + {r}u] = float({t}(dot{r}));"));
            } else {
                w.line(format!(
                    "partial[slot * COLS + {r}u] = float(route_weight[route_idx]) * float({t}(dot{r}));"
                ));
            }
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else if (simd_id == TOP_K) {");
        w.push_indent();
        for r in 0..combined_cols {
            w.line(format!("float dot{r} = 0.0f;"));
        }
        if combined_cols == 1 && shared.bits == 4 && shared.group_size == 64 && shared_k >= 1024 {
            crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
                w,
                "dot0",
                "shared_w + o_base * SHARED_PACKED",
                "shared_s + o_base * SHARED_GROUPS",
                "shared_b + o_base * SHARED_GROUPS",
                "SHARED_K",
                shared.bits,
                shared.group_size,
                false,
                false,
                crate::codegen::kernels::QdotPolicy::Force,
                |off| format!("float(shared_x[tok * SHARED_K + {off}])"),
            );
        } else if combined_cols == 1 {
            crate::codegen::kernels::emit_int4_gemv_inner(
                w,
                "dot0",
                "shared_w + o_base * SHARED_PACKED",
                "shared_s + o_base * SHARED_GROUPS",
                "shared_b + o_base * SHARED_GROUPS",
                "SHARED_K",
                shared.bits,
                shared.group_size,
                |off| format!("float(shared_x[tok * SHARED_K + {off}])"),
            );
        } else {
            crate::codegen::kernels::emit_int4_gemv_inner_multirow(
                w,
                |r| format!("dot{r}"),
                |r| format!("shared_w + (o_base + {r}u) * SHARED_PACKED"),
                |r| format!("shared_s + (o_base + {r}u) * SHARED_GROUPS"),
                |r| format!("shared_b + (o_base + {r}u) * SHARED_GROUPS"),
                "SHARED_K",
                shared.bits,
                shared.group_size,
                combined_cols,
                |off| format!("float(shared_x[tok * SHARED_K + {off}])"),
            );
        }
        for r in 0..combined_cols {
            w.line(format!("dot{r} = simd_sum(dot{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..combined_cols {
            w.line(format!(
                "partial[TOP_K * COLS + {r}u] = float({t}(dot{r}));"
            ));
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (tid == 0u) {");
        w.push_indent();
        for r in 0..combined_cols {
            w.line(format!("if (o_base + {r}u < O) {{"));
            w.push_indent();
            w.line("float acc = 0.0f;");
            w.line(format!(
                "for (uint p = 0u; p < PARTIALS; ++p) acc += partial[p * COLS + {r}u];"
            ));
            if has_extra_residual {
                w.line(format!(
                    "acc += float(residual_extra[tok * O + o_base + {r}u]);"
                ));
            }
            w.line(format!("out0[tok * O + o_base + {r}u] = {out_t}(acc);"));
            w.pop_indent();
            w.line("}");
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }

    w.line(format!(
        "// Fusion #{}: RoutedQmvReduce (K={}, routes={}, O={}, experts={}, bits={}, gs={}, cols/sg={}, fold_routes={}, input_preweighted={})",
        fusion.id,
        k_dim,
        top_k,
        out_dim,
        n_experts,
        target.bits,
        target.group_size,
        out_cols,
        fold_routes,
        input_preweighted
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x [[buffer(0)]],"));
    w.line("device const uint *wq [[buffer(1)]],");
    w.line(format!("device const {t} *scales [[buffer(2)]],"));
    w.line(format!("device const {t} *biases [[buffer(3)]],"));
    w.line("device const uint *row_idx [[buffer(4)]],");
    w.line("device const uint *expert_idx [[buffer(5)]],");
    if input_preweighted {
        w.line(format!("device {out_t} *out0 [[buffer(6)]],"));
    } else {
        w.line(format!("device const {t} *route_weight [[buffer(6)]],"));
        w.line(format!("device {out_t} *out0 [[buffer(7)]],"));
    }
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
    w.line(format!("constexpr uint OUT_COLS = {out_cols};"));
    w.line("constexpr uint O_BLOCKS = (O + OUT_COLS - 1u) / OUT_COLS;");
    if fold_routes {
        w.line("uint simd_lane = tid % 32u;");
        w.line("uint n_work = ACTIVE_TOKENS * O;");
        w.line("if (gid >= n_work) return;");
        w.line("uint tok = gid / O;");
        w.line("uint o = gid % O;");
        w.line("float acc = 0.0f;");
        w.line("for (uint slot = 0u; slot < TOP_K; ++slot) {");
        w.push_indent();
        w.line("uint route_idx = tok * TOP_K + slot;");
        w.line("uint x_row = uint(row_idx[route_idx]);");
        w.line("uint expert = min(uint(expert_idx[route_idx]), N_EXPERTS - 1u);");
        w.line("float dot = 0.0f;");
        if target.bits == 4 && target.group_size == 64 && k_dim >= 1024 {
            crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
                w,
                "dot",
                "wq + (expert * O + o) * PACKED",
                "scales + (expert * O + o) * N_GROUPS",
                "biases + (expert * O + o) * N_GROUPS",
                "K",
                target.bits,
                target.group_size,
                false,
                false,
                crate::codegen::kernels::QdotPolicy::Force,
                |off| format!("float(x[x_row * K + {off}])"),
            );
        } else {
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
        }
        w.line("dot = simd_sum(dot);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if input_preweighted {
            w.line(format!("acc += float({t}(dot));"));
        } else {
            w.line(format!(
                "acc += float(route_weight[route_idx]) * float({t}(dot));"
            ));
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line(format!("out0[tok * O + o] = {out_t}(acc);"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }
    w.line("threadgroup float route_partial[TOP_K * OUT_COLS];");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id = tid / 32u;");
    w.line("uint n_work = ACTIVE_TOKENS * O_BLOCKS;");
    w.line("if (gid >= n_work) return;");
    w.line("uint tok = gid / O_BLOCKS;");
    w.line("uint o_base = (gid % O_BLOCKS) * OUT_COLS;");
    w.line("uint slot = simd_id;");
    w.line("if (slot < TOP_K) {");
    w.push_indent();
    w.line("uint route_idx = tok * TOP_K + slot;");
    w.line("uint x_row = uint(row_idx[route_idx]);");
    w.line("uint expert = min(uint(expert_idx[route_idx]), N_EXPERTS - 1u);");
    for r in 0..out_cols {
        w.line(format!("float dot{r} = 0.0f;"));
    }
    if out_cols == 1 && target.bits == 4 && target.group_size == 64 && k_dim >= 1024 {
        crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
            w,
            "dot0",
            "wq + (expert * O + o_base) * PACKED",
            "scales + (expert * O + o_base) * N_GROUPS",
            "biases + (expert * O + o_base) * N_GROUPS",
            "K",
            target.bits,
            target.group_size,
            false,
            false,
            crate::codegen::kernels::QdotPolicy::Force,
            |off| format!("float(x[x_row * K + {off}])"),
        );
    } else if out_cols == 2 && matches!(target.bits, 4 | 8) && target.group_size == 64 {
        crate::codegen::kernels::emit_int4_gemv_inner_pair(
            w,
            "dot0",
            "dot1",
            "wq + (expert * O + o_base) * PACKED",
            "scales + (expert * O + o_base) * N_GROUPS",
            "biases + (expert * O + o_base) * N_GROUPS",
            "wq + (expert * O + (o_base + 1u)) * PACKED",
            "scales + (expert * O + (o_base + 1u)) * N_GROUPS",
            "biases + (expert * O + (o_base + 1u)) * N_GROUPS",
            "K",
            target.bits,
            target.group_size,
            |off| format!("float(x[x_row * K + {off}])"),
        );
    } else {
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("dot{r}"),
            |r| format!("wq + (expert * O + (o_base + {r}u)) * PACKED"),
            |r| format!("scales + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            |r| format!("biases + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            "K",
            target.bits,
            target.group_size,
            out_cols,
            |off| format!("float(x[x_row * K + {off}])"),
        );
    }
    for r in 0..out_cols {
        w.line(format!("dot{r} = simd_sum(dot{r});"));
    }
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    if input_preweighted {
        w.line("float rw = 1.0f;");
    } else {
        w.line("float rw = float(route_weight[route_idx]);");
    }
    for r in 0..out_cols {
        w.line(format!(
            "route_partial[slot * OUT_COLS + {r}u] = float({t}(dot{r})) * rw;"
        ));
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (tid < OUT_COLS) {");
    w.push_indent();
    w.line("uint o = o_base + tid;");
    w.line("if (o < O) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    w.line("for (uint slot_i = 0u; slot_i < TOP_K; ++slot_i) {");
    w.push_indent();
    w.line("acc += route_partial[slot_i * OUT_COLS + tid];");
    w.pop_indent();
    w.line("}");
    w.line(format!("out0[tok * O + o] = {out_t}(acc);"));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
