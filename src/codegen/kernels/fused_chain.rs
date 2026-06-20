//! Emit MSL for `FusionKind::FusedChain` — the general-fusion product.
//!
//! The chain is produced by `fusion::general` after pattern matching.
//! Today it handles the simplest case: every step is a Pointwise op
//! with the same output shape. Each thread processes one element;
//! intermediate values stay in registers (Metal compiler will keep
//! them in registers as long as they're scalars and not address-
//! taken). No threadgroup memory needed.
//!
//! Reduction-tail / heavy-with-epilogue fusion will extend this same
//! emitter — same chain walker, just a richer ChainStep emit and
//! optional SMEM scratch declared up front.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::category::{categorize_name, is_chain_absorbable_view, OpCategory};
use crate::fusion::ir::{ChainOperand, ChainStep, Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

fn round_scalar_to_act_expr(t: &str, expr: &str) -> String {
    match t {
        "bfloat" => format!("float(bfloat({expr}))"),
        "half" => format!("float(half({expr}))"),
        _ => expr.to_string(),
    }
}

fn round_scalar_to_dtype_expr(
    dtype: crate::graph::Dtype,
    fallback_t: &'static str,
    expr: &str,
) -> String {
    match crate::codegen::kernels::scalar_ty_for_dtype(dtype, fallback_t) {
        "bfloat" => format!("float(bfloat({expr}))"),
        "half" => format!("float(half({expr}))"),
        _ => format!("float({expr})"),
    }
}

fn round_vec4_to_act_expr(t: &str, expr: &str) -> String {
    match t {
        "bfloat" => format!("float4(bfloat4({expr}))"),
        "half" => format!("float4(half4({expr}))"),
        _ => expr.to_string(),
    }
}

fn chain_norm_smem_ty(t: &str) -> &str {
    let _ = t;
    "float"
}

fn chain_norm_smem_ty_for_quant_bits(t: &str, bits: u32) -> &str {
    if bits != 8 {
        return chain_norm_smem_ty(t);
    }
    match t {
        "bfloat" | "half" => t,
        _ => "float",
    }
}

fn chain_norm_vec_store_expr_for(norm_ty: &str, expr: &str) -> String {
    match norm_ty {
        "bfloat" => format!("bfloat4({expr})"),
        "half" => format!("half4({expr})"),
        _ => expr.to_string(),
    }
}

fn chain_norm_vec_store_expr(t: &str, expr: &str) -> String {
    chain_norm_vec_store_expr_for(chain_norm_smem_ty(t), expr)
}

fn chain_norm_scalar_store_expr_for(norm_ty: &str, expr: &str) -> String {
    match norm_ty {
        "bfloat" => format!("bfloat({expr})"),
        "half" => format!("half({expr})"),
        _ => expr.to_string(),
    }
}

fn chain_norm_scalar_store_expr(t: &str, expr: &str) -> String {
    chain_norm_scalar_store_expr_for(chain_norm_smem_ty(t), expr)
}

fn chain_norm_vec_store_rounded_expr_for(norm_ty: &str, t: &str, expr: &str) -> String {
    match norm_ty {
        "bfloat" => format!("bfloat4({expr})"),
        "half" => format!("half4({expr})"),
        _ => round_vec4_to_act_expr(t, expr),
    }
}

fn chain_norm_vec_store_rounded_expr(t: &str, expr: &str) -> String {
    chain_norm_vec_store_rounded_expr_for(chain_norm_smem_ty(t), t, expr)
}

fn chain_norm_scalar_store_rounded_expr_for(norm_ty: &str, t: &str, expr: &str) -> String {
    match norm_ty {
        "bfloat" => format!("bfloat({expr})"),
        "half" => format!("half({expr})"),
        _ => round_scalar_to_act_expr(t, expr),
    }
}

fn chain_norm_scalar_store_rounded_expr(t: &str, expr: &str) -> String {
    chain_norm_scalar_store_rounded_expr_for(chain_norm_smem_ty(t), t, expr)
}

fn env_on(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_bool_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(default)
}

fn emit_dual_qgemv_inner_with_unroll2_default<F: Fn(&str) -> String>(
    w: &mut MslWriter,
    acc_g: &str,
    acc_u: &str,
    gate_w_packed_row: &str,
    gate_scales_row: &str,
    gate_biases_row: &str,
    up_w_packed_row: &str,
    up_scales_row: &str,
    up_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    qgemv_unroll2_default: bool,
    x_expr: F,
) {
    let lanes_per_word = 32 / bits;
    let qgemv_unroll2 = env_bool_default("FUSEKIT_QGEMV_UNROLL2", qgemv_unroll2_default);
    // The dual gate/up qMV body has enough independent work to keep the
    // adjacent two-word schedule profitable for q8 large projections: both
    // words stay in one quant group, so one scale/bias load feeds two packed
    // words for gate and up. q4 measured worse with the extra live unpack
    // state, so it stays on the strided schedule.
    let adjacent_default = bits == 8;
    let qgemv_adjacent_unroll2 =
        env_bool_default("FUSEKIT_QGEMV_ADJACENT_UNROLL2", adjacent_default)
            && qgemv_unroll2
            && matches!(bits, 4 | 8)
            && group_size.is_multiple_of(lanes_per_word * 2);
    let mask = if bits == 4 { "0xFu" } else { "0xFFu" };

    w.line("{");
    w.push_indent();
    w.writeln(format_args!("constexpr uint LANES_ = {lanes_per_word}u;"));
    w.writeln(format_args!("constexpr uint GROUP_ = {group_size}u;"));
    w.writeln(format_args!("uint in_words_ = {m_const} / LANES_;"));
    w.writeln(format_args!(
        "device const uint *gate_w_row_ = {gate_w_packed_row};"
    ));
    w.writeln(format_args!(
        "device const uint *up_w_row_ = {up_w_packed_row};"
    ));

    let emit_word_accum = |w: &mut MslWriter,
                           suffix: &str,
                           word_g: &str,
                           word_u: &str,
                           inb: &str| {
        for off in 0..lanes_per_word {
            w.writeln(format_args!(
                "float x{suffix}_{off}_ = {};",
                x_expr(&format!("{inb} + {off}u"))
            ));
        }
        if bits == 8 {
            w.writeln(format_args!("float xsum{suffix}_ = 0.0f;"));
            w.writeln(format_args!("float qacc_g{suffix}_ = 0.0f;"));
            w.writeln(format_args!("float qacc_u{suffix}_ = 0.0f;"));
            for off in 0..lanes_per_word {
                let shift = off * bits;
                w.writeln(format_args!("xsum{suffix}_ += x{suffix}_{off}_;"));
                w.writeln(format_args!(
                    "qacc_g{suffix}_ += x{suffix}_{off}_ * float(({word_g} >> {shift}u) & {mask});"
                ));
                w.writeln(format_args!(
                    "qacc_u{suffix}_ += x{suffix}_{off}_ * float(({word_u} >> {shift}u) & {mask});"
                ));
            }
            w.writeln(format_args!(
                "{acc_g} += gate_scale{suffix}_ * qacc_g{suffix}_ + gate_bias{suffix}_ * xsum{suffix}_;"
            ));
            w.writeln(format_args!(
                "{acc_u} += up_scale{suffix}_ * qacc_u{suffix}_ + up_bias{suffix}_ * xsum{suffix}_;"
            ));
        } else {
            for off in 0..lanes_per_word {
                let shift = off * bits;
                w.writeln(format_args!(
                    "{acc_g} += x{suffix}_{off}_ * (float(({word_g} >> {shift}u) & {mask}) * gate_scale{suffix}_ + gate_bias{suffix}_);"
                ));
            }
            for off in 0..lanes_per_word {
                let shift = off * bits;
                w.writeln(format_args!(
                    "{acc_u} += x{suffix}_{off}_ * (float(({word_u} >> {shift}u) & {mask}) * up_scale{suffix}_ + up_bias{suffix}_);"
                ));
            }
        }
    };

    if qgemv_unroll2 {
        if qgemv_adjacent_unroll2 {
            w.line("uint widx_ = simd_lane * 2u;");
            w.line("for (; widx_ + 1u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.line("uint gate_word0_ = gate_w_row_[widx_];");
            w.line("uint gate_word1_ = gate_w_row_[widx_ + 1u];");
            w.line("uint up_word0_ = up_w_row_[widx_];");
            w.line("uint up_word1_ = up_w_row_[widx_ + 1u];");
            w.writeln(format_args!("uint inb0_ = widx_ * {lanes_per_word}u;"));
            w.writeln(format_args!("uint inb1_ = inb0_ + {lanes_per_word}u;"));
            w.line("uint group0_ = inb0_ / GROUP_;");
            w.writeln(format_args!(
                "float gate_scale0_ = float(({gate_scales_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float gate_bias0_ = float(({gate_biases_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float up_scale0_ = float(({up_scales_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float up_bias0_ = float(({up_biases_row})[group0_]);"
            ));
            w.line("float gate_scale1_ = gate_scale0_;");
            w.line("float gate_bias1_ = gate_bias0_;");
            w.line("float up_scale1_ = up_scale0_;");
            w.line("float up_bias1_ = up_bias0_;");
            emit_word_accum(w, "0", "gate_word0_", "up_word0_", "inb0_");
            emit_word_accum(w, "1", "gate_word1_", "up_word1_", "inb1_");
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
            w.line("uint gate_wordt_ = gate_w_row_[widx_];");
            w.line("uint up_wordt_ = up_w_row_[widx_];");
            w.writeln(format_args!("uint inbt_ = widx_ * {lanes_per_word}u;"));
            w.line("uint groupt_ = inbt_ / GROUP_;");
            w.writeln(format_args!(
                "float gate_scalet_ = float(({gate_scales_row})[groupt_]);"
            ));
            w.writeln(format_args!(
                "float gate_biast_ = float(({gate_biases_row})[groupt_]);"
            ));
            w.writeln(format_args!(
                "float up_scalet_ = float(({up_scales_row})[groupt_]);"
            ));
            w.writeln(format_args!(
                "float up_biast_ = float(({up_biases_row})[groupt_]);"
            ));
            emit_word_accum(w, "t", "gate_wordt_", "up_wordt_", "inbt_");
            w.pop_indent();
            w.line("}");
        } else {
            w.line("uint widx_ = simd_lane;");
            w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.line("uint gate_word0_ = gate_w_row_[widx_];");
            w.line("uint gate_word1_ = gate_w_row_[widx_ + 32u];");
            w.line("uint up_word0_ = up_w_row_[widx_];");
            w.line("uint up_word1_ = up_w_row_[widx_ + 32u];");
            w.writeln(format_args!("uint inb0_ = widx_ * {lanes_per_word}u;"));
            w.writeln(format_args!(
                "uint inb1_ = (widx_ + 32u) * {lanes_per_word}u;"
            ));
            w.line("uint group0_ = inb0_ / GROUP_;");
            w.line("uint group1_ = inb1_ / GROUP_;");
            w.writeln(format_args!(
                "float gate_scale0_ = float(({gate_scales_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float gate_bias0_ = float(({gate_biases_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float up_scale0_ = float(({up_scales_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float up_bias0_ = float(({up_biases_row})[group0_]);"
            ));
            w.writeln(format_args!(
                "float gate_scale1_ = float(({gate_scales_row})[group1_]);"
            ));
            w.writeln(format_args!(
                "float gate_bias1_ = float(({gate_biases_row})[group1_]);"
            ));
            w.writeln(format_args!(
                "float up_scale1_ = float(({up_scales_row})[group1_]);"
            ));
            w.writeln(format_args!(
                "float up_bias1_ = float(({up_biases_row})[group1_]);"
            ));
            emit_word_accum(w, "0", "gate_word0_", "up_word0_", "inb0_");
            emit_word_accum(w, "1", "gate_word1_", "up_word1_", "inb1_");
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
            w.line("uint gate_wordt_ = gate_w_row_[widx_];");
            w.line("uint up_wordt_ = up_w_row_[widx_];");
            w.writeln(format_args!("uint inbt_ = widx_ * {lanes_per_word}u;"));
            w.line("uint groupt_ = inbt_ / GROUP_;");
            w.writeln(format_args!(
                "float gate_scalet_ = float(({gate_scales_row})[groupt_]);"
            ));
            w.writeln(format_args!(
                "float gate_biast_ = float(({gate_biases_row})[groupt_]);"
            ));
            w.writeln(format_args!(
                "float up_scalet_ = float(({up_scales_row})[groupt_]);"
            ));
            w.writeln(format_args!(
                "float up_biast_ = float(({up_biases_row})[groupt_]);"
            ));
            emit_word_accum(w, "t", "gate_wordt_", "up_wordt_", "inbt_");
            w.pop_indent();
            w.line("}");
        }
    } else {
        w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
        w.push_indent();
        w.line("uint gate_word_ = gate_w_row_[widx_];");
        w.line("uint up_word_ = up_w_row_[widx_];");
        w.writeln(format_args!("uint inb_ = widx_ * {lanes_per_word}u;"));
        w.line("uint group_ = inb_ / GROUP_;");
        w.writeln(format_args!(
            "float gate_scale_ = float(({gate_scales_row})[group_]);"
        ));
        w.writeln(format_args!(
            "float gate_bias_ = float(({gate_biases_row})[group_]);"
        ));
        w.writeln(format_args!(
            "float up_scale_ = float(({up_scales_row})[group_]);"
        ));
        w.writeln(format_args!(
            "float up_bias_ = float(({up_biases_row})[group_]);"
        ));
        emit_word_accum(w, "", "gate_word_", "up_word_", "inb_");
        w.pop_indent();
        w.line("}");
    }

    w.pop_indent();
    w.line("}");
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::FusedChain) {
        bail!("fused_chain emitter: wrong kind {:?}", fusion.kind);
    }
    let steps = match &fusion.attrs {
        FusionAttrs::FusedChain { steps, .. } => steps.clone(),
        _ => bail!("fused_chain emitter: attrs mismatch"),
    };
    if fusion.outputs.is_empty() {
        bail!("fused_chain emitter: at least one output required");
    }
    if steps.is_empty() {
        bail!("fused_chain emitter: empty chain");
    }
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    let fn_name = function_name(fusion);
    let op_summary: String = steps
        .iter()
        .map(|s| s.op_name.as_str())
        .collect::<Vec<_>>()
        .join(" → ");

    if let Some(reduction_idx) = pointwise_prefix_reduction_index(&steps) {
        return emit_pointwise_prefix_reduction_chain(
            fusion,
            &steps,
            reduction_idx,
            meta,
            w,
            &fn_name,
            &op_summary,
        );
    }

    // Choose body emit by the first step's category. Heavy-first
    // chains (today only Matmul) own their thread shape and pass the
    // accumulator through to the pointwise tail in a register.
    let first_cat = categorize_name(&steps[0].op_name);
    if first_cat == OpCategory::Heavy {
        return emit_heavy_first_chain(fusion, &steps, meta, w, &fn_name, &op_summary);
    }
    if first_cat == OpCategory::Reduction {
        // Count the interior Heavy steps. One → single-output
        // Reduction → Heavy chain. Multiple → fan-out (Q/K/V style):
        // the normed row stays in SMEM and each Heavy step writes to
        // its own output buffer.
        let heavy_count = steps
            .iter()
            .skip(1)
            .filter(|s| categorize_name(&s.op_name) == OpCategory::Heavy)
            .count();
        match heavy_count {
            0 => return emit_reduction_first_chain(fusion, &steps, meta, w, &fn_name, &op_summary),
            1 => {
                return emit_reduction_then_heavy_chain(
                    fusion,
                    &steps,
                    meta,
                    w,
                    &fn_name,
                    &op_summary,
                )
            }
            _ => {
                return emit_reduction_fanout_chain(fusion, &steps, meta, w, &fn_name, &op_summary)
            }
        }
    }
    let n: u32 = crate::target::current_chip()
        .active_elems(&fusion.outputs[0].shape)
        .min(u32::MAX as u64) as u32;
    let _ = t;
    w.line(format!(
        "// Fusion #{}: FusedChain (n={}, chain: {})",
        fusion.id, n, op_summary
    ));

    // Route pointwise chains through the piece-IR composer. Same MSL
    // output as the legacy inline emit, but now `fused_chain` doesn't
    // own a separate kernel signature path — the composer's signature
    // emit + auto-barriers + SMEM dedup are reused. Validates the
    // composer end-to-end on a real (general-fusion-produced) kernel
    // rather than just the smoke kernel.
    let n_inputs_u32 = n_inputs as u32;
    let fallback_t = crate::codegen::kernels::act_ty(meta);
    let input_tys = fusion
        .inputs
        .iter()
        .map(|b| crate::codegen::kernels::scalar_ty_for_dtype(b.dtype, fallback_t).to_string())
        .collect::<Vec<_>>();
    let output_ty =
        crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, fallback_t)
            .to_string();
    let pieces: Vec<Box<dyn crate::codegen::pieces::KernelPiece>> = vec![Box::new(
        crate::codegen::pieces::pointwise_chain::PointwiseChain {
            n,
            n_inputs: n_inputs_u32,
            input_tys,
            output_ty,
            steps: steps.clone(),
        },
    )];
    let ctx = crate::codegen::pieces::EmitCtx {
        meta,
        act_ty: crate::codegen::kernels::act_ty(meta),
    };
    crate::codegen::pieces::compose(&fn_name, &pieces, &ctx, w);
    Ok(())
}

fn chain_shape_elems(shape: &[i32]) -> i32 {
    shape.iter().product::<i32>().max(1)
}

fn pointwise_prefix_reduction_index(steps: &[ChainStep]) -> Option<usize> {
    let mut reduction_idx = None;
    for (idx, step) in steps.iter().enumerate() {
        match categorize_name(&step.op_name) {
            OpCategory::Reduction => {
                if reduction_idx.replace(idx).is_some() {
                    return None;
                }
            }
            OpCategory::Pointwise => {}
            OpCategory::View
                if matches!(
                    step.op_name.as_str(),
                    "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous"
                ) => {}
            _ => return None,
        }
    }
    let idx = reduction_idx?;
    if idx == 0 || steps[idx].op_name != "RMSNorm" {
        return None;
    }
    Some(idx)
}

fn emit_pointwise_prefix_reduction_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    reduction_idx: usize,
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let fallback_t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    let rms = &steps[reduction_idx];
    if rms.op_name != "RMSNorm" || rms.inputs.len() != 2 {
        bail!("pointwise-prefix reduction chain expects RMSNorm(x, weight)");
    }
    let w_buf = match rms.inputs[1] {
        ChainOperand::External { buffer_index } => buffer_index,
        _ => bail!("RMSNorm weight must be an external buffer"),
    };
    let eps = match rms.op_attrs {
        crate::graph::Attrs::RmsNorm { eps } => eps,
        _ => 1e-6_f32,
    };
    let norm_shape = rms.output_shape.clone();
    if norm_shape.is_empty() {
        bail!("pointwise-prefix reduction chain: RMSNorm output shape empty");
    }
    if fusion.outputs.is_empty() || fusion.outputs.len() > 2 {
        bail!("pointwise-prefix reduction chain expects one output, or prefix+reduction outputs");
    }
    let reduction_out_index = if fusion.outputs.len() == 2 {
        1usize
    } else {
        0usize
    };
    let prefix_out_index = (fusion.outputs.len() == 2).then_some(0usize);
    if let Some(prefix_idx) = prefix_out_index {
        if chain_shape_elems(&fusion.outputs[prefix_idx].shape) != chain_shape_elems(&norm_shape) {
            bail!("pointwise-prefix reduction chain: prefix output shape mismatch");
        }
    }
    let reduction_out_t = crate::codegen::kernels::scalar_ty_for_dtype(
        fusion.outputs[reduction_out_index].dtype,
        fallback_t,
    );
    let hidden = *norm_shape.last().unwrap();
    let rows: i32 = crate::target::current_chip()
        .active_rows_before_last(&norm_shape)
        .min(i32::MAX as u64) as i32;
    let out_shape = fusion.outputs[reduction_out_index].shape.clone();
    let weight_elems: i32 = fusion
        .inputs
        .get(w_buf as usize)
        .map(|b| b.shape.iter().product::<i32>().max(1))
        .unwrap_or(1);
    let weight_is_vector = weight_elems >= hidden;

    let emit_prefix_registers = |w: &mut MslWriter| -> Result<()> {
        for (idx, step) in steps.iter().enumerate().take(reduction_idx) {
            let expr = render_pointwise_expr_reduction(step, fusion, hidden, rows, &norm_shape)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "pointwise-prefix reduction chain: unsupported prefix op `{}`",
                        step.op_name
                    )
                })?;
            let rounded = round_scalar_to_dtype_expr(step.output_dtype, fallback_t, &expr);
            w.line(format!("float r{idx} = {rounded};"));
        }
        Ok(())
    };

    w.line(format!(
        "// Fusion #{}: FusedChain (pointwise-prefix RMSNorm, ROWS={}, HIDDEN={}, chain: {})",
        fusion.id, rows, hidden, op_summary
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i].dtype, fallback_t);
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    for (out_idx, output) in fusion.outputs.iter().enumerate() {
        let ty = crate::codegen::kernels::scalar_ty_for_dtype(output.dtype, fallback_t);
        w.line(format!(
            "device {ty}       *out{out_idx} [[buffer({})]],",
            n_inputs + out_idx
        ));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint HIDDEN = {hidden};"));
    w.line(format!("constexpr uint ROWS   = {rows};"));
    w.line(format!("constexpr float EPS   = {eps};"));
    w.line(format!(
        "constexpr bool WEIGHT_IS_VECTOR = {};",
        if weight_is_vector { "true" } else { "false" }
    ));
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint row = gid;");
    w.line("if (row >= ROWS) return;");
    w.line("float sum_sq = 0.0f;");
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    emit_prefix_registers(w)?;
    let x_expr = operand_expr_reduction(&rms.inputs[0], fusion, hidden, rows, &norm_shape);
    w.line(format!("float xi = {x_expr};"));
    w.line("sum_sq += xi * xi;");
    w.pop_indent();
    w.line("}");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    w.line(format!(
        "float scalar_weight = WEIGHT_IS_VECTOR ? 1.0f : float(in{w_buf}[0]);"
    ));
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    emit_prefix_registers(w)?;
    if let Some(prefix_idx) = prefix_out_index {
        let prefix_step_idx = reduction_idx - 1;
        let prefix_t = crate::codegen::kernels::scalar_ty_for_dtype(
            fusion.outputs[prefix_idx].dtype,
            fallback_t,
        );
        w.line(format!(
            "out{prefix_idx}[row * HIDDEN + i] = {prefix_t}(r{prefix_step_idx});"
        ));
    }
    let x_expr = operand_expr_reduction(&rms.inputs[0], fusion, hidden, rows, &norm_shape);
    w.line(format!("float xi = {x_expr};"));
    let rms_t = crate::codegen::kernels::scalar_ty_for_dtype(rms.output_dtype, fallback_t);
    w.line(format!("{rms_t} normed = {rms_t}(xi * rms_scale);"));
    w.line(format!(
        "{rms_t} wi = WEIGHT_IS_VECTOR ? in{w_buf}[i] : {rms_t}(scalar_weight);"
    ));
    w.line(format!(
        "float r{reduction_idx} = float({rms_t}(wi * normed));"
    ));
    let n_steps = steps.len();
    for (idx, step) in steps.iter().enumerate().skip(reduction_idx + 1) {
        let is_last = idx + 1 == n_steps;
        let expr = render_pointwise_expr_reduction(step, fusion, hidden, rows, &out_shape)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "pointwise-prefix reduction chain: unsupported tail op `{}`",
                    step.op_name
                )
            })?;
        if is_last {
            w.line(format!(
                "out{reduction_out_index}[row * HIDDEN + i] = {reduction_out_t}({expr});"
            ));
        } else {
            w.line(format!("float r{idx} = float({expr});"));
        }
    }
    if reduction_idx + 1 == n_steps {
        w.line(format!(
            "out{reduction_out_index}[row * HIDDEN + i] = {reduction_out_t}(r{reduction_idx});"
        ));
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

fn transposed_weight_slots(fusion: &Fusion) -> &[u32] {
    match &fusion.attrs {
        FusionAttrs::FusedChain {
            transposed_weight_slots,
            ..
        } => transposed_weight_slots.as_slice(),
        _ => &[],
    }
}

fn is_transposed_weight_slot(fusion: &Fusion, slot: u32) -> bool {
    transposed_weight_slots(fusion).binary_search(&slot).is_ok()
}

/// Emit a chain whose first step is Heavy (Matmul today). Each thread
/// computes one output element through the heavy op's inner loop,
/// keeps the accumulator in a register, and applies the pointwise
/// tail in place. The shape of the heavy op drives the kernel's
/// thread layout, so this branch can't share the pointwise emitter's
/// flat-N indexing.
fn emit_heavy_first_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let first = &steps[0];
    match first.op_name.as_str() {
        "Matmul" => emit_matmul_chain(fusion, steps, meta, w, fn_name, op_summary),
        "QuantizedMatmul" => emit_qmm_chain(fusion, steps, meta, w, fn_name, op_summary),
        other => bail!(
            "fused_chain: heavy first-step `{}` not supported yet",
            other
        ),
    }
}

fn emit_matmul_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    let first = &steps[0];
    if first.inputs.len() != 2 {
        bail!("Matmul step expects exactly 2 inputs");
    }
    // Resolve A/B as external buffers (Heavy at step 0 can't reference
    // a register — there isn't one yet).
    let (a_buf, b_buf) = match (first.inputs[0], first.inputs[1]) {
        (
            ChainOperand::External { buffer_index: a },
            ChainOperand::External { buffer_index: b },
        ) => (a, b),
        _ => bail!("Matmul step at chain head must read external buffers"),
    };
    let a_shape = &fusion.inputs[a_buf as usize].shape;
    let b_shape = &fusion.inputs[b_buf as usize].shape;
    if a_shape.len() < 2 || b_shape.len() < 2 {
        bail!("Matmul shape rank < 2 (a={:?}, b={:?})", a_shape, b_shape);
    }
    let m = a_shape[a_shape.len() - 2] as u32;
    let k = a_shape[a_shape.len() - 1] as u32;
    let n_dim = b_shape[b_shape.len() - 1] as u32;
    let a_batch: i32 = a_shape[..a_shape.len() - 2].iter().product::<i32>().max(1);
    let b_batch: i32 = b_shape[..b_shape.len() - 2].iter().product::<i32>().max(1);
    if a_batch != b_batch {
        bail!(
            "Matmul in chain: a/b batch mismatch (a={:?}, b={:?})",
            a_shape,
            b_shape
        );
    }
    let batch = a_batch as u32;
    if k != b_shape[b_shape.len() - 2] as u32 {
        bail!("Matmul K mismatch: a={:?} vs b={:?}", a_shape, b_shape);
    }
    let is_lm_head_projection = n_dim == meta.vocab_size
        && b_shape.len() == 2
        && fusion
            .outputs
            .first()
            .and_then(|o| o.shape.last())
            .copied()
            .map(|d| d as u32)
            == Some(meta.vocab_size);
    let b_is_transposed_param = is_transposed_weight_slot(fusion, b_buf) || is_lm_head_projection;

    w.line(format!(
        "// Fusion #{}: FusedChain (heavy-first, BATCH={}, M={}, K={}, N={}, chain: {})",
        fusion.id, batch, m, k, n_dim, op_summary
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        w.line(format!("device const {t} *in{i} [[buffer({i})]],"));
    }
    w.line(format!("device {t}       *out0 [[buffer({})]],", n_inputs));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint tpg     [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint BATCH = {batch};"));
    w.line(format!("constexpr uint M = {m};"));
    w.line(format!("constexpr uint K = {k};"));
    w.line(format!("constexpr uint N_DIM = {n_dim};"));
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(&fusion.outputs[0].shape)
        .max(1) as u32;
    w.line(format!("constexpr uint ACTIVE_ROWS = {active_rows};"));
    w.line("constexpr uint ROWS_PER_BATCH = M * N_DIM;");
    // Chip-aware multi-col coalesce. Each simdgroup owns ROWS_PER_SG
    // consecutive output columns; the K-loop's `xa` load is shared
    // across cols so weight reads dominate. N_DIM must be R-divisible.
    let chip = crate::target::current_chip();
    let dense_mode = chip.dense_matmul_mode();
    let gemm_tile_requested = chip.use_simdgroup_gemm(active_rows);
    let chip_rows = chip.dense_gemv_rows_per_simdgroup(k, n_dim);
    let cols_per_sg = if chip_rows > 1 && n_dim >= chip_rows && n_dim % chip_rows == 0 {
        chip_rows
    } else {
        1
    };
    w.line(format!(
        "// dense matmul knob: {} (active_rows={}, simdgroup_gemm={})",
        dense_mode.as_str(),
        active_rows,
        gemm_tile_requested,
    ));
    w.line(format!("constexpr uint ROWS_PER_SG = {cols_per_sg}u;"));
    w.line("constexpr uint OUT_TOTAL = ACTIVE_ROWS * N_DIM;");
    // Simdgroup-cooperative GEMV: 32 threads share R output cols.
    // K-axis reduction striped across the simdgroup, collapsed by
    // simd_sum. Tail runs on lane 0 only.
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint i_base = (gid * sgs_per_tg + simd_id) * ROWS_PER_SG;");
    w.line("if (i_base >= OUT_TOTAL) return;");
    w.line("uint logical_row = i_base / N_DIM;");
    w.line("uint b_idx = logical_row / M;");
    w.line("uint row   = logical_row - b_idx * M;");
    w.line("uint col_base = i_base - logical_row * N_DIM;");
    for r in 0..cols_per_sg {
        w.line(format!("float acc{r} = 0.0f;"));
    }
    // Vectorise the per-k B[k, col_base..col_base+R] load when R is 2
    // or 4. col_base is always a multiple of ROWS_PER_SG (i_base
    // computation guarantees alignment), so the float2/float4 read
    // is properly aligned. Cuts B-side loads from R scalar to 1 wide.
    let use_k2_unroll = dense_k2_unroll_enabled(k, n_dim)
        && b_is_transposed_param
        && !is_lm_head_projection
        && k % 8 == 0;
    if use_k2_unroll {
        w.line(format!(
            "device const {t}4* A4 = (device const {t}4*)(in{a} + (b_idx * M + row) * K);",
            a = a_buf
        ));
        for r in 0..cols_per_sg {
            w.line(format!(
                "device const {t}4* W4_{r} = (device const {t}4*)(in{b} + (col_base + {r}u) * K);",
                b = b_buf
            ));
        }
        w.line("constexpr uint K4 = K / 4u;");
        w.line("for (uint k4 = simd_lane * 2u; k4 < K4; k4 += 64u) {");
        w.push_indent();
        w.line("float4 av0 = float4(A4[k4]);");
        w.line("float4 av1 = float4(A4[k4 + 1u]);");
        for r in 0..cols_per_sg {
            w.line(format!(
                "acc{r} += dot(av0, float4(W4_{r}[k4])) + dot(av1, float4(W4_{r}[k4 + 1u]));"
            ));
        }
        w.pop_indent();
        w.line("}");
    } else if b_is_transposed_param && k % 4 == 0 {
        w.line(format!(
            "device const {t}4* A4 = (device const {t}4*)(in{a} + (b_idx * M + row) * K);",
            a = a_buf
        ));
        for r in 0..cols_per_sg {
            w.line(format!(
                "device const {t}4* W4_{r} = (device const {t}4*)(in{b} + (col_base + {r}u) * K);",
                b = b_buf
            ));
        }
        w.line("constexpr uint K4 = K / 4u;");
        w.line("for (uint k4 = simd_lane; k4 < K4; k4 += 32u) {");
        w.push_indent();
        w.line("float4 av = float4(A4[k4]);");
        for r in 0..cols_per_sg {
            w.line(format!("acc{r} += dot(av, float4(W4_{r}[k4]));"));
        }
        w.pop_indent();
        w.line("}");
    } else if cols_per_sg == 4 && !b_is_transposed_param {
        w.line("for (uint k = simd_lane; k < K; k += 32u) {");
        w.push_indent();
        w.line(format!(
            "float xa = float(in{a}[(b_idx * M + row) * K + k]);",
            a = a_buf
        ));
        w.line(format!(
            "device const {t}4* B4 = (device const {t}4*)(in{b} + (b_idx * K + k) * N_DIM + col_base);",
            b = b_buf
        ));
        w.line("float4 bv = float4(B4[0]);");
        w.line("acc0 += xa * bv.x;");
        w.line("acc1 += xa * bv.y;");
        w.line("acc2 += xa * bv.z;");
        w.line("acc3 += xa * bv.w;");
        w.pop_indent();
        w.line("}");
    } else if cols_per_sg == 2 && !b_is_transposed_param {
        w.line("for (uint k = simd_lane; k < K; k += 32u) {");
        w.push_indent();
        w.line(format!(
            "float xa = float(in{a}[(b_idx * M + row) * K + k]);",
            a = a_buf
        ));
        w.line(format!(
            "device const {t}2* B2 = (device const {t}2*)(in{b} + (b_idx * K + k) * N_DIM + col_base);",
            b = b_buf
        ));
        w.line("float2 bv = float2(B2[0]);");
        w.line("acc0 += xa * bv.x;");
        w.line("acc1 += xa * bv.y;");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint k = simd_lane; k < K; k += 32u) {");
        w.push_indent();
        w.line(format!(
            "float xa = float(in{a}[(b_idx * M + row) * K + k]);",
            a = a_buf
        ));
        for r in 0..cols_per_sg {
            if b_is_transposed_param {
                // Runtime binds the original linear weight [out, in],
                // not the trace-only transpose scratch [in, out].
                w.line(format!(
                    "acc{r} += xa * float(in{b}[(col_base + {r}u) * K + k]);",
                    b = b_buf
                ));
            } else {
                w.line(format!(
                    "acc{r} += xa * float(in{b}[(b_idx * K + k) * N_DIM + col_base + {r}u]);",
                    b = b_buf
                ));
            }
        }
        w.pop_indent();
        w.line("}");
    }
    for r in 0..cols_per_sg {
        w.line(format!("acc{r} = simd_sum(acc{r});"));
    }
    w.line("if (simd_lane != 0u) return;");

    let n_steps = steps.len();
    for r in 0..cols_per_sg {
        w.line("{");
        w.push_indent();
        w.line(format!("uint i = i_base + {r}u;"));
        w.line(format!(
            "float r0 = {};",
            round_scalar_to_dtype_expr(first.output_dtype, t, &format!("acc{r}"))
        ));
        for (idx, step) in steps.iter().enumerate().skip(1) {
            let is_last = idx + 1 == n_steps;
            let expr = render_pointwise_expr(step).ok_or_else(|| {
                anyhow::anyhow!("fused_chain heavy tail: unsupported op `{}`", step.op_name)
            })?;
            if is_last {
                w.line(format!("out0[i] = {t}({expr});"));
            } else {
                w.line(format!(
                    "float r{idx} = {};",
                    round_scalar_to_dtype_expr(step.output_dtype, t, &expr)
                ));
            }
        }
        if n_steps == 1 {
            w.line(format!("out0[i] = {t}(r0);"));
        }
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_chain_f{}", fusion.id)
}

fn dense_k2_unroll_enabled(k_dim: u32, out_dim: u32) -> bool {
    if let Ok(raw) = std::env::var("FUSEKIT_DENSE_K2_UNROLL") {
        let raw = raw.trim().to_ascii_lowercase();
        return matches!(raw.as_str(), "1" | "true" | "yes" | "on");
    }

    // Large dense decode GEMV is weight-streaming dominated. On M5,
    // two K4 vectors per lane improves the 4B-class bf16 FusedChain
    // path without exposing the smaller hidden/output shapes to the
    // live-range pressure that made the global M5 K2 default lose.
    if matches!(crate::target::current_chip(), crate::target::ChipModel::M5) {
        return k_dim >= 2048 && out_dim >= 2048;
    }
    true
}

fn dense_k4_unroll_enabled(k_dim: u32, out_dim: u32) -> bool {
    matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && k_dim >= 2048
        && out_dim >= 2048
        && k_dim.is_multiple_of(16)
}

fn dense_k8_unroll_enabled(k_dim: u32, out_dim: u32) -> bool {
    matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && k_dim >= 2048
        && out_dim >= 2048
        && k_dim.is_multiple_of(32)
}

/// Emit a 4-way vectorized GEMV inner loop for a dense (non-quantized)
/// Heavy step in a reduction-fanout chain. Matches the `unroll4`
/// branch of `rmsnorm_qkv.rs` — loads `bfloat4`/`half4` weight slabs,
/// runs `dot(ny4[k4], float4(W4[k4]))` so the Metal compiler can pipe
/// 4 FMAs through the FPU in one fetch.
///
/// Caller is responsible for:
///   - declaring the accumulator passed in `acc_var`
///   - calling `simd_sum(acc_var)` afterwards
///   - the lane-0 gated write to the output buffer
///
/// `out_idx_var` is the local variable name that holds the row index
/// (typically `"out_idx"` per fanout chain emit convention).
fn emit_chain_dense_matmul(
    w: &mut MslWriter,
    t: &str,
    w_buf: u32,
    out_idx_var: &str,
    acc_var: &str,
    hidden: u32,
    virtual_transposed: bool,
    out_dim: u32,
    out_dim_expr: &str,
) {
    if virtual_transposed {
        w.line("for (uint k = simd_lane; k < HIDDEN; k += 32u) {");
        w.push_indent();
        w.line(format!("uint logical = ({out_idx_var}) * HIDDEN + k;"));
        w.line(format!("uint tr_row = logical / ({out_dim_expr});"));
        w.line(format!(
            "uint tr_col = logical - tr_row * ({out_dim_expr});"
        ));
        w.line(format!(
            "{acc_var} += float(norm_y[k]) * float(in{w_buf}[tr_col * HIDDEN + tr_row]);"
        ));
        w.pop_indent();
        w.line("}");
        return;
    }
    w.line(format!(
        "device const {t}4* W4 = (device const {t}4*)(in{w_buf} + ({out_idx_var}) * HIDDEN);"
    ));
    let norm_ty = chain_norm_smem_ty(t);
    w.line(format!(
        "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
    ));
    if dense_k8_unroll_enabled(hidden, out_dim) {
        w.line("for (uint k4 = simd_lane * 8u; k4 < HIDDEN / 4u; k4 += 256u) {");
        w.push_indent();
        w.line(format!(
            "{acc_var} += dot(float4(ny4[k4]),     float4(W4[k4]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+2u]), float4(W4[k4+2u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+3u]), float4(W4[k4+3u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+4u]), float4(W4[k4+4u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+5u]), float4(W4[k4+5u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+6u]), float4(W4[k4+6u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+7u]), float4(W4[k4+7u]));"
        ));
        w.pop_indent();
        w.line("}");
    } else if dense_k4_unroll_enabled(hidden, out_dim) {
        w.line("for (uint k4 = simd_lane * 4u; k4 < HIDDEN / 4u; k4 += 128u) {");
        w.push_indent();
        w.line(format!(
            "{acc_var} += dot(float4(ny4[k4]),     float4(W4[k4]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+2u]), float4(W4[k4+2u]))"
        ));
        w.line(format!(
            "          + dot(float4(ny4[k4+3u]), float4(W4[k4+3u]));"
        ));
        w.pop_indent();
        w.line("}");
    } else if dense_k2_unroll_enabled(hidden, out_dim) && hidden % 8 == 0 {
        w.line("for (uint k4 = simd_lane * 2u; k4 < HIDDEN / 4u; k4 += 64u) {");
        w.push_indent();
        w.line(format!(
            "{acc_var} += dot(float4(ny4[k4]), float4(W4[k4])) + dot(float4(ny4[k4 + 1u]), float4(W4[k4 + 1u]));"
        ));
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
        w.push_indent();
        w.line(format!(
            "{acc_var} += dot(float4(ny4[k4]), float4(W4[k4]));"
        ));
        w.pop_indent();
        w.line("}");
    }
}

/// Multi-row dense GEMV: each simdgroup computes `rows` output rows
/// at once, reusing the activation load (`ny4` is in SMEM, but the
/// register-resident `xv` block re-uses 4-wide loads across rows).
/// Caller declares `acc0..accN`, calls `simd_sum` on each, writes
/// each through lane-0. Used when the chip has spare registers
/// (`ChipModel::gemv_rows_per_simdgroup() > 1`).
///
/// `out_idx_base_var` is the variable holding the first output row's
/// index; subsequent rows are at `out_idx_base + 1u`, etc. Caller is
/// responsible for ensuring `out_idx_base + rows - 1 < OUT_DIM`.
fn emit_chain_dense_matmul_multirow_named(
    w: &mut MslWriter,
    t: &str,
    w_buf: u32,
    out_idx_base_var: &str,
    acc_prefix: &str,
    rows: u32,
    hidden: u32,
    virtual_transposed: bool,
    out_dim: u32,
    out_dim_expr: &str,
) {
    if rows <= 1 {
        emit_chain_dense_matmul(
            w,
            t,
            w_buf,
            out_idx_base_var,
            &format!("{acc_prefix}0"),
            hidden,
            virtual_transposed,
            out_dim,
            out_dim_expr,
        );
        return;
    }
    if virtual_transposed {
        w.line("for (uint k = simd_lane; k < HIDDEN; k += 32u) {");
        w.push_indent();
        w.line("float xv = float(norm_y[k]);");
        for r in 0..rows {
            w.line(format!(
                "uint logical_{r} = (({out_idx_base_var}) + {r}u) * HIDDEN + k;"
            ));
            w.line(format!("uint tr_row_{r} = logical_{r} / ({out_dim_expr});"));
            w.line(format!(
                "uint tr_col_{r} = logical_{r} - tr_row_{r} * ({out_dim_expr});"
            ));
            w.line(format!(
                "{acc_prefix}{r} += xv * float(in{w_buf}[tr_col_{r} * HIDDEN + tr_row_{r}]);"
            ));
        }
        w.pop_indent();
        w.line("}");
        return;
    }
    for r in 0..rows {
        w.line(format!(
            "device const {t}4* W4_{r} = (device const {t}4*)(in{w_buf} + (({out_idx_base_var}) + {r}u) * HIDDEN);"
        ));
    }
    let norm_ty = chain_norm_smem_ty(t);
    w.line(format!(
        "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
    ));
    if dense_k4_unroll_enabled(hidden, out_dim) {
        w.line("for (uint k4 = simd_lane * 4u; k4 < HIDDEN / 4u; k4 += 128u) {");
        w.push_indent();
        w.line("float4 xv0 = float4(ny4[k4]);");
        w.line("float4 xv1 = float4(ny4[k4 + 1u]);");
        w.line("float4 xv2 = float4(ny4[k4 + 2u]);");
        w.line("float4 xv3 = float4(ny4[k4 + 3u]);");
        for r in 0..rows {
            w.line(format!(
                "{acc_prefix}{r} += dot(xv0, float4(W4_{r}[k4])) + dot(xv1, float4(W4_{r}[k4 + 1u])) + dot(xv2, float4(W4_{r}[k4 + 2u])) + dot(xv3, float4(W4_{r}[k4 + 3u]));"
            ));
        }
        w.pop_indent();
        w.line("}");
    } else if dense_k2_unroll_enabled(hidden, out_dim) && hidden % 8 == 0 {
        w.line("for (uint k4 = simd_lane * 2u; k4 < HIDDEN / 4u; k4 += 64u) {");
        w.push_indent();
        w.line("float4 xv0 = float4(ny4[k4]);");
        w.line("float4 xv1 = float4(ny4[k4 + 1u]);");
        for r in 0..rows {
            w.line(format!(
                "{acc_prefix}{r} += dot(xv0, float4(W4_{r}[k4])) + dot(xv1, float4(W4_{r}[k4 + 1u]));"
            ));
        }
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
        w.push_indent();
        w.line("float4 xv = float4(ny4[k4]);");
        for r in 0..rows {
            w.line(format!("{acc_prefix}{r} += dot(xv, float4(W4_{r}[k4]));"));
        }
        w.pop_indent();
        w.line("}");
    }
}

fn emit_chain_dense_matmul_multirow(
    w: &mut MslWriter,
    t: &str,
    w_buf: u32,
    out_idx_base_var: &str,
    rows: u32,
    hidden: u32,
    virtual_transposed: bool,
    out_dim: u32,
    out_dim_expr: &str,
) {
    emit_chain_dense_matmul_multirow_named(
        w,
        t,
        w_buf,
        out_idx_base_var,
        "acc",
        rows,
        hidden,
        virtual_transposed,
        out_dim,
        out_dim_expr,
    );
}

/// Emit a chain whose first step is `QuantizedMatmul` with a pointwise
/// tail. Mirrors the int4 affine dequant inner loop from
/// `linear_residual::emit` but lands the accumulator in register `r0`
/// so the tail can keep going.
fn emit_qmm_chain_mlx_qmv_fast(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    x_buf: u32,
    w_buf: u32,
    s_buf: u32,
    b_buf: u32,
    k_dim: u32,
    out_dim: u32,
    active_rows: u64,
    bits: u32,
    group_size: u32,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    w.line(format!(
        "// Fusion #{}: FusedChain (heavy-first QuantizedMatmul MLX qmv_fast, M={}, N={}, active_m={}, bits={}, gs={})",
        fusion.id, k_dim, out_dim, active_rows, bits, group_size,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = if i as u32 == w_buf { "uint" } else { t };
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    w.line(format!("device {t}       *out0 [[buffer({n_inputs})]],"));
    w.line("uint3 tgid [[threadgroup_position_in_grid]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("#if FUSEKIT_HAVE_MLX_NAX");
    w.push_indent();
    w.writeln(format_args!("constexpr int K = {k_dim};"));
    w.writeln(format_args!("constexpr int N = {out_dim};"));
    let token_rows =
        crate::codegen::kernels::qmv_fast_token_rows(active_rows, bits, k_dim, out_dim);
    let qmv_sgs = crate::codegen::kernels::qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim);
    let qmv_cols_per_sg =
        crate::codegen::kernels::qmv_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim);
    if token_rows > 1 {
        w.writeln(format_args!("constexpr int ACTIVE_ROWS = {active_rows};"));
        w.writeln(format_args!(
            "fusekit_qmv_fast_m_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}, {token_rows}>(in{w_buf}, in{s_buf}, in{b_buf}, in{x_buf}, out0, K, N, ACTIVE_ROWS, tgid, simd_gid, simd_lid);"
        ));
    } else {
        w.writeln(format_args!(
            "fusekit_qmv_fast_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}>(in{w_buf}, in{s_buf}, in{b_buf}, in{x_buf}, out0, K, N, tgid, simd_gid, simd_lid);"
        ));
    }
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit MLX qmv requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

fn emit_qmm_chain_nax(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    x_buf: u32,
    w_buf: u32,
    s_buf: u32,
    b_buf: u32,
    k_dim: u32,
    out_dim: u32,
    active_rows: u64,
    bits: u32,
    group_size: u32,
) -> Result<()> {
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| crate::codegen::kernels::scalar_ty_for_dtype(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        bail!("FusedChain qmm NAX: unsupported output dtype");
    }
    let n_inputs = fusion.inputs.len();
    let aligned_n = if out_dim.is_multiple_of(64) {
        "true"
    } else {
        "false"
    };
    let (wm, wn) = if active_rows <= 16 { (1, 4) } else { (2, 2) };
    w.line(format!(
        "// Fusion #{}: FusedChain (heavy-first QuantizedMatmul NAX qmm_t, M={}, N={}, active_m={}, bits={}, gs={})",
        fusion.id, k_dim, out_dim, active_rows, bits, group_size,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = if i as u32 == w_buf {
            "uint"
        } else {
            let dtype = fusion.inputs[i].dtype;
            crate::codegen::kernels::scalar_ty_for_dtype(dtype, fallback)
        };
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    w.line(format!("device {t}       *out0 [[buffer({n_inputs})]],"));
    w.line("uint3 tgid [[threadgroup_position_in_grid]],");
    w.line("uint tid [[thread_index_in_threadgroup]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("#if FUSEKIT_HAVE_MLX_NAX");
    w.push_indent();
    w.writeln(format_args!("constexpr int K = {k_dim};"));
    w.writeln(format_args!("constexpr int N = {out_dim};"));
    w.writeln(format_args!("constexpr int M = {active_rows};"));
    w.line(format!("threadgroup {t} Ws[64 * (32 + 16 / sizeof({t}))];"));
    w.writeln(format_args!(
        "fusekit_qmm_t_nax_tgp_impl<{t}, {group_size}, {bits}, {aligned_n}, 64, 32, 64, {wm}, {wn}>("
    ));
    w.push_indent();
    w.line(format!(
        "in{w_buf}, in{s_buf}, in{b_buf}, in{x_buf}, out0, Ws, K, N, M, tgid, tid, simd_gid, simd_lid);"
    ));
    w.pop_indent();
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit FusedChain qmm NAX requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub(crate) fn emit_qmm_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    let first = &steps[0];
    if first.inputs.len() < 4 {
        bail!("QuantizedMatmul chain head expects 4 external inputs");
    }
    // 4 operands = standard QM; 5 = QM + linear bias (absorbed by
    // try_absorb_qm_bias for chains ending in `+ bias_param`).
    let mut bufs = [0u32; 4];
    for (i, op) in first.inputs.iter().take(4).enumerate() {
        match op {
            ChainOperand::External { buffer_index } => bufs[i] = *buffer_index,
            ChainOperand::Register { .. } => {
                bail!("QuantizedMatmul step 0 must read external buffers")
            }
        }
    }
    let (x_buf, w_buf, s_buf, b_buf) = (bufs[0], bufs[1], bufs[2], bufs[3]);
    let lin_bias_buf: Option<u32> = if first.inputs.len() >= 5 {
        match first.inputs[4] {
            ChainOperand::External { buffer_index } => Some(buffer_index),
            _ => bail!("QM linear-bias operand must be External"),
        }
    } else {
        None
    };

    // mlx's affine layout (verified against `mx.quantized_matmul`):
    //   x        shape (..., in_dim)
    //   W_packed shape (out_dim, in_dim / lanes)            uint32
    //   scales   shape (out_dim, in_dim / group_size)
    //   biases   shape (out_dim, in_dim / group_size)
    //   out      shape (..., out_dim)
    let x_shape = &fusion.inputs[x_buf as usize].shape;
    let w_shape = &fusion.inputs[w_buf as usize].shape;
    if x_shape.is_empty() || w_shape.len() != 2 {
        bail!("QuantizedMatmul: unexpected shape ranks");
    }
    let in_dim = *x_shape.last().unwrap() as u32;
    let out_dim = w_shape[0] as u32;

    // Quant params live on the captured Attrs::Quantized.
    let (bits, group_size) = match &first.op_attrs {
        crate::graph::Attrs::Quantized { bits, group_size } => (*bits, *group_size),
        _ => (4u32, 64u32),
    };
    let lanes_per_word = 32 / bits;
    let packed_words = crate::codegen::kernels::quant_packed_words(in_dim, bits);
    let active_rows_for_nax = crate::target::current_chip()
        .active_rows_before_last(&fusion.outputs[0].shape)
        .max(1);
    let use_chain_nax = env_on("FUSEKIT_QMM_CHAIN_NAX")
        && steps.len() == 1
        && first.inputs.len() == 4
        && matches!((bits, group_size), (4, 64) | (8, 64))
        && (active_rows_for_nax == 1 || env_on("FUSEKIT_QMM_CHAIN_NAX_M"))
        && (active_rows_for_nax == 1 || out_dim >= 64);
    if use_chain_nax {
        return emit_qmm_chain_nax(
            fusion,
            meta,
            w,
            fn_name,
            x_buf,
            w_buf,
            s_buf,
            b_buf,
            in_dim,
            out_dim,
            active_rows_for_nax,
            bits,
            group_size,
        );
    }
    if let Some((k_dim, out_dim, active_rows, qmv_bits, qmv_group_size)) =
        crate::codegen::kernels::qmm_chain_mlx_qmv_fast_shape(fusion)
    {
        return emit_qmm_chain_mlx_qmv_fast(
            fusion,
            meta,
            w,
            fn_name,
            x_buf,
            w_buf,
            s_buf,
            b_buf,
            k_dim,
            out_dim,
            active_rows,
            qmv_bits,
            qmv_group_size,
        );
    }

    w.line(format!(
        "// Fusion #{}: FusedChain (heavy-first QuantizedMatmul, M={}, N={}, bits={}, gs={}, chain: {})",
        fusion.id, in_dim, out_dim, bits, group_size, op_summary
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = if i as u32 == w_buf {
            "uint"
        } else {
            crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i].dtype, t)
        };
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, t);
    w.line(format!(
        "device {out_t}       *out0 [[buffer({})]],",
        n_inputs
    ));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint tpg     [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint M     = {in_dim};"));
    w.line(format!("constexpr uint N     = {out_dim};"));
    w.line(format!("constexpr uint BITS  = {bits}u;"));
    w.line(format!("constexpr uint GROUP = {group_size}u;"));
    w.line(format!("constexpr uint LANES = {lanes_per_word}u;"));
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(&fusion.outputs[0].shape)
        .max(1) as u32;
    w.line(format!("constexpr uint ACTIVE_ROWS = {active_rows};"));
    w.line("constexpr uint MASK = (1u << BITS) - 1u;");
    w.line(format!("constexpr uint IN_WORDS  = {packed_words}u;"));
    w.line("constexpr uint N_GROUPS  = M / GROUP;");
    // Chip-aware multi-row coalesce. With R rows per simdgroup the
    // grid shrinks by R× and weight reads share the same dequant
    // scale/bias broadcast across R output rows (q4: 8 nibbles per
    // word amortized over R rows = 8R FMAs per word). out_dim must
    // be R-divisible — fall back to R=1 otherwise so the chain stays
    // correct on heads / mlp-dims that aren't aligned.
    let chip = crate::target::current_chip();
    let chip_rows = chip.gemv_rows_per_simdgroup();
    let allow_output_row_coalesce = matches!(bits, 4 | 8);
    let rows_per_sg = if allow_output_row_coalesce
        && chip_rows > 1
        && out_dim >= chip_rows
        && out_dim % chip_rows == 0
    {
        chip_rows
    } else {
        1
    };
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));
    // Simdgroup-cooperative QM GEMV: 32 threads share R consecutive
    // output rows. Each lane strides the packed-word axis by 32;
    // partial dots collapsed by simd_sum.
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    let token_rows_per_sg =
        if rows_per_sg == 1 && crate::codegen::kernels::mtp_token_m_qgemv_enabled() {
            chip.qgemm_m_rows_per_simdgroup(active_rows, bits, in_dim, out_dim)
        } else {
            1
        };
    w.line(format!(
        "constexpr uint TOKEN_ROWS_PER_SG = {token_rows_per_sg}u;"
    ));
    if token_rows_per_sg > 1 {
        let token_cols_per_sg =
            chip.qgemm_output_cols_per_simdgroup(active_rows, bits, in_dim, out_dim);
        let full_token_blocks = active_rows % token_rows_per_sg == 0;
        w.line(
            "// Small-M qGEMM: one simdgroup shares quantized weight rows across active M rows and adjacent output columns.",
        );
        w.line(format!(
            "constexpr uint TOKEN_COLS_PER_SG = {token_cols_per_sg}u;"
        ));
        w.line("constexpr uint ROW_BLOCKS = (ACTIVE_ROWS + TOKEN_ROWS_PER_SG - 1u) / TOKEN_ROWS_PER_SG;");
        w.line("constexpr uint COL_BLOCKS = (N + TOKEN_COLS_PER_SG - 1u) / TOKEN_COLS_PER_SG;");
        w.line("uint sg_flat = gid * sgs_per_tg + simd_id;");
        w.line("uint row_block = sg_flat / COL_BLOCKS;");
        w.line("uint col_block = sg_flat - row_block * COL_BLOCKS;");
        w.line("uint i_base = col_block * TOKEN_COLS_PER_SG;");
        w.line("uint row_base = row_block * TOKEN_ROWS_PER_SG;");
        w.line("if (row_block >= ROW_BLOCKS || i_base >= N) return;");
        for r in 0..token_rows_per_sg {
            for c in 0..token_cols_per_sg {
                w.line(format!("float acc{r}_{c} = 0.0f;"));
            }
        }
        crate::codegen::kernels::emit_int4_gemv_inner_token_cols_with_options(
            w,
            |r, c| format!("acc{r}_{c}"),
            |c| format!("in{w_buf} + (i_base + {c}u) * IN_WORDS"),
            |c| format!("in{s_buf} + (i_base + {c}u) * N_GROUPS"),
            |c| format!("in{b_buf} + (i_base + {c}u) * N_GROUPS"),
            "M",
            bits,
            group_size,
            token_rows_per_sg,
            token_cols_per_sg,
            crate::codegen::kernels::qgemv_unroll2_default_for_shape(bits, in_dim, out_dim),
            |r, off| {
                if full_token_blocks {
                    format!("float(in{x_buf}[(row_base + {r}u) * M + {off}])")
                } else {
                    format!(
                        "((row_base + {r}u < ACTIVE_ROWS) ? float(in{x_buf}[(row_base + {r}u) * M + {off}]) : 0.0f)"
                    )
                }
            },
        );
        for r in 0..token_rows_per_sg {
            for c in 0..token_cols_per_sg {
                w.line(format!("acc{r}_{c} = simd_sum(acc{r}_{c});"));
            }
        }
        w.line("if (simd_lane != 0u) return;");
        if let Some(lin_bias_buf) = lin_bias_buf {
            for c in 0..token_cols_per_sg {
                w.line(format!(
                    "float lin_bias{c} = float(in{lin_bias_buf}[i_base + {c}u]);"
                ));
                for r in 0..token_rows_per_sg {
                    w.line(format!("acc{r}_{c} += lin_bias{c};"));
                }
            }
        }
        let n_steps = steps.len();
        for r in 0..token_rows_per_sg {
            if !full_token_blocks {
                w.line(format!("if (row_base + {r}u < ACTIVE_ROWS) {{"));
                w.push_indent();
            }
            for c in 0..token_cols_per_sg {
                w.line("{");
                w.push_indent();
                w.line(format!("uint i = (row_base + {r}u) * N + i_base + {c}u;"));
                w.line(format!(
                    "float r0 = {};",
                    round_scalar_to_dtype_expr(first.output_dtype, t, &format!("acc{r}_{c}"))
                ));
                for (idx, step) in steps.iter().enumerate().skip(1) {
                    let is_last = idx + 1 == n_steps;
                    let expr = render_pointwise_expr(step).ok_or_else(|| {
                        anyhow::anyhow!("fused_chain qmm tail: unsupported op `{}`", step.op_name)
                    })?;
                    if is_last {
                        w.line(format!("out0[i] = {t}({expr});"));
                    } else {
                        w.line(format!(
                            "float r{idx} = {};",
                            round_scalar_to_dtype_expr(step.output_dtype, t, &expr)
                        ));
                    }
                }
                if n_steps == 1 {
                    w.line(format!("out0[i] = {t}(r0);"));
                }
                w.pop_indent();
                w.line("}");
            }
            if !full_token_blocks {
                w.pop_indent();
                w.line("}");
            }
        }
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }
    w.line("uint flat_base = (gid * sgs_per_tg + simd_id) * ROWS_PER_SG;");
    w.line("if (flat_base >= ACTIVE_ROWS * N) return;");
    w.line("uint row = flat_base / N;");
    w.line("uint i_base = flat_base - row * N;");
    for r in 0..rows_per_sg {
        w.line(format!("float acc{r} = 0.0f;"));
    }
    if rows_per_sg <= 1 {
        let w_row_expr = format!("in{w_buf} + i_base * IN_WORDS");
        let s_row_expr = format!("in{s_buf} + i_base * N_GROUPS");
        let b_row_expr = format!("in{b_buf} + i_base * N_GROUPS");
        crate::codegen::kernels::emit_int4_gemv_inner_with_options(
            w,
            "acc0",
            &w_row_expr,
            &s_row_expr,
            &b_row_expr,
            "M",
            bits,
            group_size,
            crate::codegen::kernels::qgemv_unroll2_default_for_shape(bits, in_dim, out_dim),
            crate::codegen::kernels::qgemv_qparam_broadcast_default_for_shape(
                bits, in_dim, out_dim,
            ),
            |off| format!("float(in{x_buf}[row * M + {off}])"),
        );
    } else {
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("acc{r}"),
            |r| format!("in{w_buf} + (i_base + {r}u) * IN_WORDS"),
            |r| format!("in{s_buf} + (i_base + {r}u) * N_GROUPS"),
            |r| format!("in{b_buf} + (i_base + {r}u) * N_GROUPS"),
            "M",
            bits,
            group_size,
            rows_per_sg,
            |off| format!("float(in{x_buf}[row * M + {off}])"),
        );
    }
    for r in 0..rows_per_sg {
        w.line(format!("acc{r} = simd_sum(acc{r});"));
    }
    w.line("if (simd_lane != 0u) return;");
    // QM with absorbed linear bias — apply per-output bias to each
    // acc before the pointwise tail. The mlx trace's
    // `nn.QuantizedLinear(..., bias=True)` lowers to `QM` followed by
    // a separate `Add(qm_out, bias_param)`; `try_absorb_qm_bias` in
    // general fusion attaches the bias param as the 5th operand here.
    if let Some(lin_bias_buf) = lin_bias_buf {
        for r in 0..rows_per_sg {
            w.line(format!("acc{r} += float(in{lin_bias_buf}[i_base + {r}u]);"));
        }
    }

    // Pointwise tail emitted once per row in its own scope — `i` and
    // `r0` shadow per-row so each row independently evaluates the
    // chain's downstream steps.
    let n_steps = steps.len();
    for r in 0..rows_per_sg {
        w.line("{");
        w.push_indent();
        w.line(format!("uint i = row * N + i_base + {r}u;"));
        w.line(format!(
            "float r0 = {};",
            round_scalar_to_dtype_expr(first.output_dtype, t, &format!("acc{r}"))
        ));
        for (idx, step) in steps.iter().enumerate().skip(1) {
            let is_last = idx + 1 == n_steps;
            let expr = render_pointwise_expr(step).ok_or_else(|| {
                anyhow::anyhow!("fused_chain qmm tail: unsupported op `{}`", step.op_name)
            })?;
            if is_last {
                w.line(format!("out0[i] = {t}({expr});"));
            } else {
                w.line(format!(
                    "float r{idx} = {};",
                    round_scalar_to_dtype_expr(step.output_dtype, t, &expr)
                ));
            }
        }
        if n_steps == 1 {
            w.line(format!("out0[i] = {t}(r0);"));
        }
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

/// Build an MSL expression for one chain step's output value. Returns
/// `None` when the op isn't supported by the pointwise-only fast path
/// (the caller propagates an error so the chain falls back to the
/// per-op emit path until reduction/heavy support lands here).
fn render_pointwise_expr(step: &ChainStep) -> Option<String> {
    let a = step.inputs.first().map(operand_expr);
    let b = step.inputs.get(1).map(operand_expr);
    let c = step.inputs.get(2).map(operand_expr);
    match step.op_name.as_str() {
        "Add" => Some(format!("{} + {}", a?, b?)),
        "Subtract" => Some(format!("{} - {}", a?, b?)),
        "Multiply" => Some(format!("{} * {}", a?, b?)),
        "Divide" => Some(format!("{} / {}", a?, b?)),
        "Power" => Some(format!("pow({}, {})", a?, b?)),
        "Maximum" => Some(format!("max({}, {})", a?, b?)),
        "Minimum" => Some(format!("min({}, {})", a?, b?)),
        "Sigmoid" => Some(format!("(1.0f / (1.0f + exp(-{})))", a?)),
        "Exp" => Some(format!("exp({})", a?)),
        "Negative" => Some(format!("(-{})", a?)),
        "LogAddExp" => Some(format!(
            "(max({a},{b}) + log(1.0f + exp(-fabs({a} - {b}))))",
            a = a?,
            b = b?
        )),
        "Select" => Some(format!(
            "(({a} != 0.0f) ? {b} : {c})",
            a = a?,
            b = b?,
            c = c?
        )),
        "GreaterEqual" => Some(format!("(({a} >= {b}) ? 1.0f : 0.0f)", a = a?, b = b?)),
        // AsType inside a chain is a no-op: every register carries a
        // `float` already; the only dtype cast is at the final store
        // to `out0[i]`.
        "AsType" => Some(a?),
        // Element-contiguous view ops pass through as identity reads —
        // element order is unchanged so the same register holds the
        // value at the same flat index after the rebrand.
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => Some(a?),
        _ => None,
    }
}

fn operand_expr(op: &ChainOperand) -> String {
    match op {
        ChainOperand::External { buffer_index } => {
            format!("float(in{}[i])", buffer_index)
        }
        ChainOperand::Register { step_index } => {
            format!("r{}", step_index)
        }
    }
}

/// Same as `render_pointwise_expr` but uses `operand_expr_reduction`
/// so external operands index per-row + per-column inside the
/// reduction body (`[row * HIDDEN + i]` for same-shape inputs,
/// `[i]` for hidden-broadcast, `[0]` for scalar).
fn render_pointwise_expr_reduction(
    step: &ChainStep,
    fusion: &Fusion,
    hidden: i32,
    rows: i32,
    out_shape: &[i32],
) -> Option<String> {
    let a = step
        .inputs
        .first()
        .map(|op| operand_expr_reduction(op, fusion, hidden, rows, out_shape));
    let b = step
        .inputs
        .get(1)
        .map(|op| operand_expr_reduction(op, fusion, hidden, rows, out_shape));
    let c = step
        .inputs
        .get(2)
        .map(|op| operand_expr_reduction(op, fusion, hidden, rows, out_shape));
    match step.op_name.as_str() {
        "Add" => Some(format!("{} + {}", a?, b?)),
        "Subtract" => Some(format!("{} - {}", a?, b?)),
        "Multiply" => Some(format!("{} * {}", a?, b?)),
        "Divide" => Some(format!("{} / {}", a?, b?)),
        "Power" => Some(format!("pow({}, {})", a?, b?)),
        "Maximum" => Some(format!("max({}, {})", a?, b?)),
        "Minimum" => Some(format!("min({}, {})", a?, b?)),
        "Sigmoid" => Some(format!("(1.0f / (1.0f + exp(-{})))", a?)),
        "Exp" => Some(format!("exp({})", a?)),
        "Negative" => Some(format!("(-{})", a?)),
        "LogAddExp" => Some(format!(
            "(max({a},{b}) + log(1.0f + exp(-fabs({a} - {b}))))",
            a = a?,
            b = b?
        )),
        "Select" => Some(format!(
            "(({a} != 0.0f) ? {b} : {c})",
            a = a?,
            b = b?,
            c = c?
        )),
        "GreaterEqual" => Some(format!("(({a} >= {b}) ? 1.0f : 0.0f)", a = a?, b = b?)),
        "AsType" => Some(a?),
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => Some(a?),
        _ => None,
    }
}

fn operand_expr_reduction(
    op: &ChainOperand,
    fusion: &Fusion,
    hidden: i32,
    rows: i32,
    out_shape: &[i32],
) -> String {
    match op {
        ChainOperand::External { buffer_index } => {
            let s = fusion
                .inputs
                .get(*buffer_index as usize)
                .map(|b| b.shape.as_slice())
                .unwrap_or(&[]);
            let elems: i32 = s.iter().product::<i32>().max(1);
            let active_elems = hidden.saturating_mul(rows).max(1);
            let out_elems: i32 = out_shape.iter().product::<i32>().max(1);
            let same = s == out_shape;
            let broadcast = elems == hidden && s.iter().any(|&d| d == hidden);
            let flat_alias = elems == out_elems;
            let contiguous_active = elems >= active_elems
                && (same
                    || flat_alias
                    || s.last().copied() == Some(hidden)
                    || elems == active_elems);
            if contiguous_active {
                format!("float(in{}[row * HIDDEN + i])", buffer_index)
            } else if broadcast {
                format!("float(in{}[i])", buffer_index)
            } else {
                // scalar — covered by `reduction_merge_tail_ok` gate.
                format!("float(in{}[0])", buffer_index)
            }
        }
        ChainOperand::Register { step_index } => {
            format!("r{}", step_index)
        }
    }
}

#[derive(Clone, Copy)]
struct GatedMultiplyTail {
    gate_step_idx: u32,
    sigmoid_step_idx: u32,
    other: ChainOperand,
}

fn reduction_fanout_gated_multiply_tail(steps: &[ChainStep]) -> Option<GatedMultiplyTail> {
    let tail_idx =
        (1..steps.len()).find(|&idx| categorize_name(&steps[idx].op_name) != OpCategory::Heavy)?;
    if tail_idx + 2 > steps.len() {
        return None;
    }
    let sig = &steps[tail_idx];
    let mul = &steps[tail_idx + 1];
    if sig.op_name != "Sigmoid" || mul.op_name != "Multiply" {
        return None;
    }
    if steps
        .iter()
        .skip(tail_idx + 2)
        .any(|s| !is_chain_absorbable_view(&s.op_name))
    {
        return None;
    }
    let gate_step_idx = match sig.inputs.first().copied()? {
        ChainOperand::Register { step_index }
            if step_index > 0 && (step_index as usize) < tail_idx =>
        {
            step_index
        }
        _ => return None,
    };
    let sigmoid_step_idx = tail_idx as u32;
    let a = mul.inputs.first().copied()?;
    let b = mul.inputs.get(1).copied()?;
    let other = match (a, b) {
        (ChainOperand::Register { step_index }, x) if step_index == sigmoid_step_idx => x,
        (x, ChainOperand::Register { step_index }) if step_index == sigmoid_step_idx => x,
        _ => return None,
    };
    match other {
        ChainOperand::External { .. } => Some(GatedMultiplyTail {
            gate_step_idx,
            sigmoid_step_idx,
            other,
        }),
        ChainOperand::Register { step_index } if step_index == gate_step_idx => {
            Some(GatedMultiplyTail {
                gate_step_idx,
                sigmoid_step_idx,
                other,
            })
        }
        _ => None,
    }
}

fn gated_tail_other_expr(
    fusion: &Fusion,
    tail: GatedMultiplyTail,
    out_dim: i32,
    out_dim_expr: &str,
    out_idx_expr: &str,
    gate_expr: &str,
) -> Option<String> {
    let ChainOperand::External { buffer_index } = tail.other else {
        return match tail.other {
            ChainOperand::Register { step_index } if step_index == tail.gate_step_idx => {
                Some(gate_expr.to_string())
            }
            _ => None,
        };
    };
    let boundary = fusion.inputs.get(buffer_index as usize)?;
    let elems = boundary
        .shape
        .iter()
        .map(|&d| d.max(1) as u64)
        .product::<u64>()
        .max(1);
    let last = boundary.shape.last().copied().unwrap_or(1).max(1) as u64;
    let buf = buffer_index;
    if elems == 1 {
        Some(format!("float(in{buf}[0])"))
    } else if elems == last && last == out_dim.max(1) as u64 {
        Some(format!("float(in{buf}[{out_idx_expr}])"))
    } else {
        Some(format!(
            "float(in{buf}[row * {out_dim_expr} + {out_idx_expr}])"
        ))
    }
}

/// Emit a chain whose head is RMSNorm followed by **multiple** Heavy
/// steps that all read the same normed row (Q/K/V style fan-out).
/// One TG handles one input row × a strip of output elements; the
/// normed row is computed once into threadgroup memory and shared
/// across every Heavy step. Each simdgroup looks at its `out_idx` and
/// dispatches to whichever step's output range it falls into.
/// Emit the SwiGLU-absorbed variant of the reduction-fanout chain.
/// Shape: RMSNorm → QM_gate → QM_up → Sigmoid(gate) → Multiply(_, gate)
/// → Multiply(_, up). Each simdgroup owns ROWS_PER_SG consecutive
/// output columns, computes BOTH gate and up at those columns into
/// register accumulators, then lane-0 applies the SwiGLU activation in
/// register and writes ONE output (silu_up). vs the standard fanout
/// emit (each SG picks gate-or-up via flat_out_base range, two output
/// buffers): halves the SG count and eliminates the gate/up DRAM
/// round-trip that the standalone pointwise SwiGLU chain otherwise
/// performs.
fn emit_reduction_swiglu_absorb_chain(
    fusion: &Fusion,
    _steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
    x_buf: u32,
    w_rms_buf: u32,
    eps: f32,
    hidden: i32,
    rows: i32,
    qm_g: (u32, u32, u32, u32, u32), // (w_buf, s_buf, b_buf, bits, group_size)
    qm_u: (u32, u32, u32, u32, u32),
    out_dim: i32,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    let (gw, gs, gb, gbits, ggroup) = qm_g;
    let (uw, us, ub, ubits, ugroup) = qm_u;
    if gbits != ubits || ggroup != ugroup {
        bail!("swiglu absorb: gate/up quant params must match");
    }
    let bits = gbits;
    let group_size = ggroup;
    let lanes_per_word = 32 / bits;
    let packed_words = crate::codegen::kernels::quant_packed_words(hidden as u32, bits);

    w.line(format!(
        "// Fusion #{}: FusedChain (swiglu-absorbed, ROWS={}, HIDDEN={}, OUT_DIM={}, chain: {})",
        fusion.id, rows, hidden, out_dim, op_summary,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    let is_w_buf = |i: u32| i == gw || i == uw;
    for i in 0..n_inputs as u32 {
        let ty = if is_w_buf(i) {
            "uint"
        } else {
            crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i as usize].dtype, t)
        };
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, t);
    w.line(format!(
        "device {out_t}       *out0 [[buffer({})]],",
        n_inputs
    ));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint tpg     [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint HIDDEN  = {hidden};"));
    w.line(format!("constexpr uint OUT_DIM = {out_dim};"));
    w.line(format!("constexpr uint ROWS    = {rows};"));
    w.line(format!("constexpr float EPS    = {eps};"));

    let chip_rows = crate::target::current_chip().gemv_rows_per_simdgroup();
    let rows_per_sg =
        if chip_rows > 1 && (out_dim as u32) >= chip_rows && (out_dim as u32) % chip_rows == 0 {
            chip_rows
        } else {
            1
        };
    let token_rows_per_sg = if rows > 1
        && rows_per_sg == 1
        && crate::codegen::kernels::mtp_token_m_reduction_fanout_enabled()
        && matches!(bits, 4 | 8)
    {
        let base = crate::target::current_chip().qgemm_m_rows_per_simdgroup(
            rows as u32,
            bits,
            hidden as u32,
            out_dim as u32,
        );
        if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
            && group_size == 64
            && out_dim >= 2048
            && rows >= 4
        {
            base.max(4).min(rows as u32)
        } else {
            base
        }
    } else {
        1
    };
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");

    if token_rows_per_sg > 1 {
        let full_token_blocks = (rows as u32).is_multiple_of(token_rows_per_sg);
        w.line("// Small-M swiglu-absorbed qGEMM: share gate/up weight rows across active M rows.");
        w.line(format!(
            "constexpr uint TOKEN_ROWS_PER_SG = {token_rows_per_sg}u;"
        ));
        w.line("constexpr uint ROW_BLOCKS = (ROWS + TOKEN_ROWS_PER_SG - 1u) / TOKEN_ROWS_PER_SG;");
        w.line("uint tgs_per_row_block = max(tpg / ROW_BLOCKS, 1u);");
        w.line("uint row_block = gid / tgs_per_row_block;");
        w.line("if (row_block >= ROW_BLOCKS) return;");
        w.line("uint row_base = row_block * TOKEN_ROWS_PER_SG;");
        w.line("uint tg_in_row = gid - row_block * tgs_per_row_block;");
        w.line("uint out_idx_base = (tg_in_row * sgs_per_tg + simd_id) * ROWS_PER_SG;");
        w.line("if (out_idx_base >= OUT_DIM) return;");
        w.line(format!(
            "threadgroup float reduce_m_buf[{}];",
            token_rows_per_sg * 32
        ));
        for r in 0..token_rows_per_sg {
            w.line(format!("float sum{r} = 0.0f;"));
        }
        w.line("if constexpr (HIDDEN % 4u == 0u) {");
        w.push_indent();
        w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
        for r in 0..token_rows_per_sg {
            if !full_token_blocks {
                w.line(format!("if (row_base + {r}u < ROWS) {{"));
                w.push_indent();
            }
            w.line(format!(
                "device const {t}4* x{r}_4 = (device const {t}4*)(in{x_buf} + (row_base + {r}u) * HIDDEN);"
            ));
            w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
            w.push_indent();
            w.line(format!("float4 x{r}v = float4(x{r}_4[i]);"));
            w.line(format!("sum{r} += dot(x{r}v, x{r}v);"));
            w.pop_indent();
            w.line("}");
            if !full_token_blocks {
                w.pop_indent();
                w.line("}");
            }
        }
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        for r in 0..token_rows_per_sg {
            if !full_token_blocks {
                w.line(format!("if (row_base + {r}u < ROWS) {{"));
                w.push_indent();
            }
            w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
            w.push_indent();
            w.line(format!(
                "float x{r}v = float(in{x_buf}[(row_base + {r}u) * HIDDEN + i]);"
            ));
            w.line(format!("sum{r} += x{r}v * x{r}v;"));
            w.pop_indent();
            w.line("}");
            if !full_token_blocks {
                w.pop_indent();
                w.line("}");
            }
        }
        w.pop_indent();
        w.line("}");
        for r in 0..token_rows_per_sg {
            w.line(format!("sum{r} = simd_sum(sum{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..token_rows_per_sg {
            w.line(format!("reduce_m_buf[{r}u * 32u + simd_id] = sum{r};"));
        }
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        for r in 0..token_rows_per_sg {
            w.line(format!(
                "float v{r} = (tid < tg_size / 32u) ? reduce_m_buf[{r}u * 32u + tid] : 0.0f;"
            ));
            w.line(format!("v{r} = simd_sum(v{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..token_rows_per_sg {
            w.line(format!("reduce_m_buf[{r}] = v{r};"));
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        for r in 0..token_rows_per_sg {
            w.line(format!(
                "float rms_scale{r} = metal::precise::rsqrt(reduce_m_buf[{r}] / float(HIDDEN) + EPS);"
            ));
            w.line(format!("float acc_g{r} = 0.0f;"));
            w.line(format!("float acc_u{r} = 0.0f;"));
        }
        w.line(format!("constexpr uint BITS  = {bits}u;"));
        w.line(format!("constexpr uint GROUP = {group_size}u;"));
        w.line(format!("constexpr uint LANES = {lanes_per_word}u;"));
        w.line("constexpr uint MASK = (1u << BITS) - 1u;");
        w.line(format!("constexpr uint IN_WORDS = {packed_words}u;"));
        w.line("constexpr uint N_GROUPS = HIDDEN / GROUP;");
        crate::codegen::kernels::emit_int4_gemv_inner_token_cols_with_options(
            w,
            |r, c| {
                if c == 0 {
                    format!("acc_g{r}")
                } else {
                    format!("acc_u{r}")
                }
            },
            |c| {
                let buf = if c == 0 { gw } else { uw };
                format!("in{buf} + out_idx_base * IN_WORDS")
            },
            |c| {
                let buf = if c == 0 { gs } else { us };
                format!("in{buf} + out_idx_base * N_GROUPS")
            },
            |c| {
                let buf = if c == 0 { gb } else { ub };
                format!("in{buf} + out_idx_base * N_GROUPS")
            },
            "HIDDEN",
            bits,
            group_size,
            token_rows_per_sg,
            2,
            crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                bits,
                hidden as u32,
                out_dim as u32,
            ),
            |r, kk| {
                let expr = round_scalar_to_act_expr(
                    t,
                    &format!(
                        "float(in{x_buf}[(row_base + {r}u) * HIDDEN + {kk}]) * rms_scale{r} * float(in{w_rms_buf}[{kk}])"
                    ),
                );
                if full_token_blocks {
                    expr
                } else {
                    format!("((row_base + {r}u < ROWS) ? ({expr}) : 0.0f)")
                }
            },
        );
        for r in 0..token_rows_per_sg {
            w.line(format!("acc_g{r} = simd_sum(acc_g{r});"));
            w.line(format!("acc_u{r} = simd_sum(acc_u{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..token_rows_per_sg {
            if !full_token_blocks {
                w.line(format!("if (row_base + {r}u < ROWS) {{"));
                w.push_indent();
            }
            w.line("{");
            w.push_indent();
            w.line(format!("{t} gate_v = {t}(acc_g{r});"));
            w.line(format!("{t} up_v = {t}(acc_u{r});"));
            w.line(format!("{t} one_v = {t}(1.0f);"));
            w.line(format!(
                "{t} sig_y = one_v / (one_v + metal::exp(metal::abs(gate_v)));"
            ));
            w.line(format!(
                "{t} sig_v = (gate_v < {t}(0.0f)) ? sig_y : {t}(one_v - sig_y);"
            ));
            w.line(format!("{t} silu_v = {t}(gate_v * sig_v);"));
            w.line(format!(
                "out0[(row_base + {r}u) * OUT_DIM + out_idx_base] = {t}(silu_v * up_v);"
            ));
            w.pop_indent();
            w.line("}");
            if !full_token_blocks {
                w.pop_indent();
                w.line("}");
            }
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }

    let norm_ty = chain_norm_smem_ty_for_quant_bits(t, bits);
    w.line(format!("threadgroup {norm_ty} norm_y[HIDDEN];"));
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint tgs_per_row = max(tpg / ROWS, 1u);");
    w.line("uint row       = gid / tgs_per_row;");
    w.line("uint tg_in_row = gid - row * tgs_per_row;");
    w.line("if (row >= ROWS) return;");
    w.blank();

    // Stage 1: sum_sq + stash x into norm_y SMEM (vec4-fast path).
    w.line("float sum_sq = 0.0f;");
    let rf_norm_scalar = std::env::var("FUSEKIT_RF_NORM_SCALAR").as_deref() == Ok("1");
    if rf_norm_scalar {
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
        w.line(format!(
            "norm_y[i] = {};",
            chain_norm_scalar_store_expr_for(norm_ty, "xi")
        ));
        w.line("sum_sq += xi * xi;");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("if constexpr (HIDDEN % 4u == 0u) {");
        w.push_indent();
        w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(in{x_buf} + row * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4_init = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
        w.push_indent();
        w.line("float4 xv = float4(x4[i]);");
        w.line(format!(
            "ny4_init[i] = {};",
            chain_norm_vec_store_expr_for(norm_ty, "xv")
        ));
        w.line("sum_sq += dot(xv, xv);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
        w.line(format!(
            "norm_y[i] = {};",
            chain_norm_scalar_store_expr_for(norm_ty, "xi")
        ));
        w.line("sum_sq += xi * xi;");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    }
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    w.line("if constexpr (HIDDEN % 4u == 0u) {");
    w.push_indent();
    w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
    w.line(format!(
        "device const {t}4* w4 = (device const {t}4*)(in{w_rms_buf});"
    ));
    w.line(format!(
        "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
    ));
    w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "ny4[i] = {};",
        chain_norm_vec_store_rounded_expr_for(
            norm_ty,
            t,
            "float4(ny4[i]) * rms_scale * float4(w4[i])"
        )
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "norm_y[i] = {};",
        chain_norm_scalar_store_rounded_expr_for(
            norm_ty,
            t,
            &format!("float(norm_y[i]) * rms_scale * float(in{w_rms_buf}[i])")
        )
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // Stage 3: per-SG dual GEMV → SwiGLU in register → 1 write.
    // The stride loop lets a capped dispatch reuse the normalized row
    // across multiple output strips. With the default exact dispatch the
    // loop has one iteration, preserving the old schedule.
    w.line(format!("constexpr uint BITS  = {bits}u;"));
    w.line(format!("constexpr uint GROUP = {group_size}u;"));
    w.line(format!("constexpr uint LANES = {lanes_per_word}u;"));
    w.line("constexpr uint MASK = (1u << BITS) - 1u;");
    w.line(format!("constexpr uint IN_WORDS = {packed_words}u;"));
    w.line("constexpr uint N_GROUPS = HIDDEN / GROUP;");

    let gw_cap = gw;
    let gs_cap = gs;
    let gb_cap = gb;
    let uw_cap = uw;
    let us_cap = us;
    let ub_cap = ub;
    let emit_dual_row_fallback = |w: &mut MslWriter| {
        emit_dual_qgemv_inner_with_unroll2_default(
            w,
            "acc_g0",
            "acc_u0",
            &format!("in{gw_cap} + out_idx_base * IN_WORDS"),
            &format!("in{gs_cap} + out_idx_base * N_GROUPS"),
            &format!("in{gb_cap} + out_idx_base * N_GROUPS"),
            &format!("in{uw_cap} + out_idx_base * IN_WORDS"),
            &format!("in{us_cap} + out_idx_base * N_GROUPS"),
            &format!("in{ub_cap} + out_idx_base * N_GROUPS"),
            "HIDDEN",
            bits,
            group_size,
            crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                bits,
                hidden as u32,
                out_dim as u32,
            ),
            |off| format!("norm_y[{off}]"),
        );
    };
    let emit_single_row_fallback = |w: &mut MslWriter| {
        crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
            w,
            "acc_g0",
            &format!("in{gw_cap} + out_idx_base * IN_WORDS"),
            &format!("in{gs_cap} + out_idx_base * N_GROUPS"),
            &format!("in{gb_cap} + out_idx_base * N_GROUPS"),
            "HIDDEN",
            bits,
            group_size,
            crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                bits,
                hidden as u32,
                out_dim as u32,
            ),
            |off| format!("norm_y[{off}]"),
        );
        crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
            w,
            "acc_u0",
            &format!("in{uw_cap} + out_idx_base * IN_WORDS"),
            &format!("in{us_cap} + out_idx_base * N_GROUPS"),
            &format!("in{ub_cap} + out_idx_base * N_GROUPS"),
            "HIDDEN",
            bits,
            group_size,
            crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                bits,
                hidden as u32,
                out_dim as u32,
            ),
            |off| format!("norm_y[{off}]"),
        );
    };
    let pack_factor = match bits {
        3 | 5 => 8,
        6 => 4,
        _ => 32 / bits.max(1),
    };
    let packs_per_thread = crate::codegen::kernels::qmv_fast_tg2_packs_per_thread(bits)
        .unwrap_or_else(|| if bits == 2 { 1 } else { 2 });
    let block_size = pack_factor * packs_per_thread * 32;
    let use_mlx_tg_qmv = group_size == 64
        && matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
        && (hidden as u32) % block_size == 0;
    let use_mlx_tg_qmv_tail = std::env::var("FUSEKIT_FUSEDCHAIN_TG_QMV_TAIL").as_deref() == Ok("1")
        && group_size == 64
        && matches!(bits, 4 | 8)
        && (hidden as u32) % block_size != 0;
    w.line("uint sg_global = tg_in_row * sgs_per_tg + simd_id;");
    w.line("uint sg_stride = tgs_per_row * sgs_per_tg;");
    w.line("for (uint out_idx_base = sg_global * ROWS_PER_SG;");
    w.line("     out_idx_base < OUT_DIM;");
    w.line("     out_idx_base += sg_stride * ROWS_PER_SG) {");
    w.push_indent();
    for r in 0..rows_per_sg {
        w.line(format!("float acc_g{r} = 0.0f;"));
        w.line(format!("float acc_u{r} = 0.0f;"));
    }

    if rows_per_sg <= 1 {
        if use_mlx_tg_qmv || use_mlx_tg_qmv_tail {
            let helper = if use_mlx_tg_qmv_tail {
                "fusekit_qmv_fast_tg2_partial_tail_impl"
            } else {
                "fusekit_qmv_fast_tg2_partial_impl"
            };
            w.line("#if FUSEKIT_HAVE_MLX_NAX");
            w.push_indent();
            w.writeln(format_args!(
                "{helper}<{t}, {group_size}, {bits}>(in{gw_cap}, in{gs_cap}, in{gb_cap}, in{uw_cap}, in{us_cap}, in{ub_cap}, norm_y, HIDDEN, int(out_idx_base), simd_lane, &acc_g0, &acc_u0);"
            ));
            w.pop_indent();
            w.line("#else");
            w.push_indent();
            if matches!(bits, 4 | 8) {
                emit_dual_row_fallback(w);
            } else {
                emit_single_row_fallback(w);
            }
            w.pop_indent();
            w.line("#endif");
        } else if matches!(bits, 4 | 8) {
            emit_dual_row_fallback(w);
        } else {
            emit_single_row_fallback(w);
        }
    } else {
        // Use the existing multirow GEMV inner helper for gate, then up,
        // sharing norm_y SMEM. Each accumulates to its own acc_g/acc_u.
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("acc_g{r}"),
            |r| format!("in{gw_cap} + (out_idx_base + {r}u) * IN_WORDS"),
            |r| format!("in{gs_cap} + (out_idx_base + {r}u) * N_GROUPS"),
            |r| format!("in{gb_cap} + (out_idx_base + {r}u) * N_GROUPS"),
            "HIDDEN",
            bits,
            group_size,
            rows_per_sg,
            |off| format!("norm_y[{off}]"),
        );
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("acc_u{r}"),
            |r| format!("in{uw_cap} + (out_idx_base + {r}u) * IN_WORDS"),
            |r| format!("in{us_cap} + (out_idx_base + {r}u) * N_GROUPS"),
            |r| format!("in{ub_cap} + (out_idx_base + {r}u) * N_GROUPS"),
            "HIDDEN",
            bits,
            group_size,
            rows_per_sg,
            |off| format!("norm_y[{off}]"),
        );
    }
    for r in 0..rows_per_sg {
        w.line(format!("acc_g{r} = simd_sum(acc_g{r});"));
        w.line(format!("acc_u{r} = simd_sum(acc_u{r});"));
    }
    w.line("if (simd_lane == 0u) {");
    w.push_indent();

    // SwiGLU: silu(gate) * up = sigmoid(g) * g * u, written into out0.
    for r in 0..rows_per_sg {
        w.line("{");
        w.push_indent();
        w.line(format!("uint out_idx = out_idx_base + {r}u;"));
        w.line(format!("{t} gate_v = {t}(acc_g{r});"));
        w.line(format!("{t} up_v = {t}(acc_u{r});"));
        w.line(format!("{t} one_v = {t}(1.0f);"));
        w.line(format!(
            "{t} sig_y = one_v / (one_v + metal::exp(metal::abs(gate_v)));"
        ));
        w.line(format!(
            "{t} sig_v = (gate_v < {t}(0.0f)) ? sig_y : {t}(one_v - sig_y);"
        ));
        w.line(format!("{t} silu_v = {t}(gate_v * sig_v);"));
        w.line(format!(
            "out0[row * OUT_DIM + out_idx] = {t}(silu_v * up_v);"
        ));
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

/// Dense (`Matmul`) sibling of `emit_reduction_swiglu_absorb_chain`.
/// It preserves the same semantic lowering, but replaces quantized qMV
/// with dense bf16/fp16 GEMV. The important part is structural: gate
/// and up projections share the RMSNorm SMEM row, keep both accumulators
/// in registers, apply SwiGLU immediately, and write only the activated
/// MLP intermediate. That removes the separate gate/up materialization
/// and pointwise activation pass for bf16 models.
fn emit_reduction_dense_swiglu_absorb_chain(
    fusion: &Fusion,
    _steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
    x_buf: u32,
    w_rms_buf: u32,
    eps: f32,
    hidden: i32,
    rows: i32,
    gate_w_buf: u32,
    up_w_buf: u32,
    out_dim: i32,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();

    w.line(format!(
        "// Fusion #{}: FusedChain (swiglu-absorbed-dense, ROWS={}, HIDDEN={}, OUT_DIM={}, chain: {})",
        fusion.id, rows, hidden, out_dim, op_summary,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs as u32 {
        let ty = crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i as usize].dtype, t);
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, t);
    w.line(format!(
        "device {out_t}       *out0 [[buffer({})]],",
        n_inputs
    ));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint tpg     [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint HIDDEN  = {hidden};"));
    w.line(format!("constexpr uint OUT_DIM = {out_dim};"));
    w.line(format!("constexpr uint ROWS    = {rows};"));
    w.line(format!("constexpr float EPS    = {eps};"));

    let chip_rows =
        crate::target::current_chip().dense_gemv_rows_per_simdgroup(hidden as u32, out_dim as u32);
    let rows_per_sg =
        if chip_rows > 1 && (out_dim as u32) >= chip_rows && (out_dim as u32) % chip_rows == 0 {
            chip_rows
        } else {
            1
        };
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    let norm_ty = chain_norm_smem_ty(t);
    w.line(format!("threadgroup {norm_ty} norm_y[HIDDEN];"));
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint tgs_per_row = max(tpg / ROWS, 1u);");
    w.line("uint row       = gid / tgs_per_row;");
    w.line("uint tg_in_row = gid - row * tgs_per_row;");
    w.line("if (row >= ROWS) return;");
    w.blank();

    w.line("float sum_sq = 0.0f;");
    let rf_norm_scalar = std::env::var("FUSEKIT_RF_NORM_SCALAR").as_deref() == Ok("1");
    if rf_norm_scalar {
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
        w.line(format!(
            "norm_y[i] = {};",
            chain_norm_scalar_store_expr(t, "xi")
        ));
        w.line("sum_sq += xi * xi;");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("if constexpr (HIDDEN % 4u == 0u) {");
        w.push_indent();
        w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(in{x_buf} + row * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4_init = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
        w.push_indent();
        w.line("float4 xv = float4(x4[i]);");
        w.line(format!(
            "ny4_init[i] = {};",
            chain_norm_vec_store_expr(t, "xv")
        ));
        w.line("sum_sq += dot(xv, xv);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
        w.line(format!(
            "norm_y[i] = {};",
            chain_norm_scalar_store_expr(t, "xi")
        ));
        w.line("sum_sq += xi * xi;");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    }
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    w.line("if constexpr (HIDDEN % 4u == 0u) {");
    w.push_indent();
    w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
    w.line(format!(
        "device const {t}4* w4 = (device const {t}4*)(in{w_rms_buf});"
    ));
    w.line(format!(
        "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
    ));
    w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "ny4[i] = {};",
        chain_norm_vec_store_rounded_expr(t, "float4(ny4[i]) * rms_scale * float4(w4[i])")
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "norm_y[i] = {};",
        chain_norm_scalar_store_rounded_expr(
            t,
            &format!("float(norm_y[i]) * rms_scale * float(in{w_rms_buf}[i])")
        )
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    w.line("uint sg_global = tg_in_row * sgs_per_tg + simd_id;");
    w.line("uint sg_stride = tgs_per_row * sgs_per_tg;");
    w.line("for (uint out_idx_base = sg_global * ROWS_PER_SG;");
    w.line("     out_idx_base < OUT_DIM;");
    w.line("     out_idx_base += sg_stride * ROWS_PER_SG) {");
    w.push_indent();
    for r in 0..rows_per_sg {
        w.line(format!("float acc_g{r} = 0.0f;"));
        w.line(format!("float acc_u{r} = 0.0f;"));
    }
    if rows_per_sg <= 1 {
        w.line("{");
        w.push_indent();
        emit_chain_dense_matmul(
            w,
            t,
            gate_w_buf,
            "out_idx_base",
            "acc_g0",
            hidden as u32,
            false,
            out_dim as u32,
            "OUT_DIM",
        );
        w.pop_indent();
        w.line("}");
        w.line("{");
        w.push_indent();
        emit_chain_dense_matmul(
            w,
            t,
            up_w_buf,
            "out_idx_base",
            "acc_u0",
            hidden as u32,
            false,
            out_dim as u32,
            "OUT_DIM",
        );
        w.pop_indent();
        w.line("}");
    } else {
        w.line("{");
        w.push_indent();
        emit_chain_dense_matmul_multirow_named(
            w,
            t,
            gate_w_buf,
            "out_idx_base",
            "acc_g",
            rows_per_sg,
            hidden as u32,
            false,
            out_dim as u32,
            "OUT_DIM",
        );
        w.pop_indent();
        w.line("}");
        w.line("{");
        w.push_indent();
        emit_chain_dense_matmul_multirow_named(
            w,
            t,
            up_w_buf,
            "out_idx_base",
            "acc_u",
            rows_per_sg,
            hidden as u32,
            false,
            out_dim as u32,
            "OUT_DIM",
        );
        w.pop_indent();
        w.line("}");
    }
    for r in 0..rows_per_sg {
        w.line(format!("acc_g{r} = simd_sum(acc_g{r});"));
        w.line(format!("acc_u{r} = simd_sum(acc_u{r});"));
    }
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    for r in 0..rows_per_sg {
        w.line("{");
        w.push_indent();
        w.line(format!("uint out_idx = out_idx_base + {r}u;"));
        w.line("if (out_idx < OUT_DIM) {");
        w.push_indent();
        w.line(format!("{t} gate_v = {t}(acc_g{r});"));
        w.line(format!("{t} up_v = {t}(acc_u{r});"));
        w.line(format!("{t} one_v = {t}(1.0f);"));
        w.line(format!(
            "{t} sig_y = one_v / (one_v + metal::exp(metal::abs(gate_v)));"
        ));
        w.line(format!(
            "{t} sig_v = (gate_v < {t}(0.0f)) ? sig_y : {t}(one_v - sig_y);"
        ));
        w.line(format!("{t} silu_v = {t}(gate_v * sig_v);"));
        w.line(format!(
            "out0[row * OUT_DIM + out_idx] = {t}(silu_v * up_v);"
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

fn emit_reduction_fanout_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    if steps.len() < 3 {
        bail!("reduction-fanout chain expects at least RMSNorm + 2 Heavy steps");
    }
    let head = &steps[0];
    if head.op_name != "RMSNorm" {
        bail!("reduction-fanout chain: head must be RMSNorm");
    }
    if head.inputs.len() != 2 {
        bail!("RMSNorm head expects 2 external inputs (x, weight)");
    }
    let gated_tail = reduction_fanout_gated_multiply_tail(steps);
    // SwiGLU absorb routing: chain is [RMSNorm, Heavy_g, Heavy_u, Sigmoid,
    // Multiply, Multiply] with single output (silu_up). Each SG computes
    // both heavies + tail in registers — different SG-to-output mapping
    // than the standard fanout emit. Detect and dispatch.
    if steps.len() == 6
        && fusion.outputs.len() == 1
        && steps[3].op_name == "Sigmoid"
        && steps[4].op_name == "Multiply"
        && steps[5].op_name == "Multiply"
        && ((steps[1].op_name == "QuantizedMatmul" && steps[2].op_name == "QuantizedMatmul")
            || (steps[1].op_name == "Matmul" && steps[2].op_name == "Matmul"))
    {
        let (x_buf_a, w_rms_buf_a) = match (head.inputs[0], head.inputs[1]) {
            (
                ChainOperand::External { buffer_index: a },
                ChainOperand::External { buffer_index: b },
            ) => (a, b),
            _ => bail!("swiglu absorb: RMSNorm head must read external buffers"),
        };
        let eps_a = match head.op_attrs {
            crate::graph::Attrs::RmsNorm { eps } => eps,
            _ => 1e-6_f32,
        };
        let head_out_a = head.output_shape.clone();
        let hidden_a = *head_out_a.last().unwrap();
        let rows_a: i32 = crate::target::current_chip()
            .active_rows_before_last(&head_out_a)
            .min(i32::MAX as u64) as i32;
        let out_dim_a = *steps[1].output_shape.last().unwrap();
        if steps[1].op_name == "QuantizedMatmul" {
            let extract_qm = |s: &ChainStep| -> Result<(u32, u32, u32, u32, u32)> {
                if s.inputs.len() != 4 {
                    bail!("swiglu absorb: QM step expects 4 operands");
                }
                let resolve = |idx: usize| -> Result<u32> {
                    match s.inputs[idx] {
                        ChainOperand::External { buffer_index } => Ok(buffer_index),
                        _ => bail!("swiglu absorb: QM weights must be External"),
                    }
                };
                let w_buf = resolve(1)?;
                let s_buf = resolve(2)?;
                let b_buf = resolve(3)?;
                let (bits, gs) = match s.op_attrs {
                    crate::graph::Attrs::Quantized { bits, group_size } => (bits, group_size),
                    _ => (4u32, 64u32),
                };
                Ok((w_buf, s_buf, b_buf, bits, gs))
            };
            let qm_g = extract_qm(&steps[1])?;
            let qm_u = extract_qm(&steps[2])?;
            return emit_reduction_swiglu_absorb_chain(
                fusion,
                steps,
                meta,
                w,
                fn_name,
                op_summary,
                x_buf_a,
                w_rms_buf_a,
                eps_a,
                hidden_a,
                rows_a,
                qm_g,
                qm_u,
                out_dim_a,
            );
        }
        if steps[1].op_name == "Matmul" && steps[2].op_name == "Matmul" {
            let extract_dense_w = |s: &ChainStep| -> Result<u32> {
                if s.inputs.len() < 2 {
                    bail!("swiglu dense absorb: Matmul step expects 2 operands");
                }
                match s.inputs[1] {
                    ChainOperand::External { buffer_index } => Ok(buffer_index),
                    _ => bail!("swiglu dense absorb: Matmul weight must be External"),
                }
            };
            let gate_w = extract_dense_w(&steps[1])?;
            let up_w = extract_dense_w(&steps[2])?;
            return emit_reduction_dense_swiglu_absorb_chain(
                fusion,
                steps,
                meta,
                w,
                fn_name,
                op_summary,
                x_buf_a,
                w_rms_buf_a,
                eps_a,
                hidden_a,
                rows_a,
                gate_w,
                up_w,
                out_dim_a,
            );
        }
    }
    let (x_buf, w_rms_buf) = match (head.inputs[0], head.inputs[1]) {
        (
            ChainOperand::External { buffer_index: a },
            ChainOperand::External { buffer_index: b },
        ) => (a, b),
        _ => bail!("RMSNorm head must read external buffers"),
    };
    let eps = match head.op_attrs {
        crate::graph::Attrs::RmsNorm { eps } => eps,
        _ => 1e-6_f32,
    };
    let head_out = head.output_shape.clone();
    if head_out.is_empty() {
        bail!("RMSNorm output shape empty");
    }
    let hidden = *head_out.last().unwrap();
    let rows: i32 = head_out[..head_out.len() - 1]
        .iter()
        .product::<i32>()
        .max(1);
    let active_rows: i32 = crate::target::current_chip()
        .active_rows_before_last(&head_out)
        .min(i32::MAX as u64) as i32;

    // Pre-compute per-Heavy-step metadata: which buffers it reads,
    // its output dim, and which output index in `fusion.outputs` it
    // writes to. We assume one fusion output per heavy step, in the
    // same order as the steps themselves — that's what
    // `do_fanout_merge` produces.
    struct Heavy {
        idx: u32,     // step index
        name: String, // "Matmul" | "AddMM" | "QuantizedMatmul"
        out_dim: i32,
        out_buf: u32, // index into fusion.outputs
        // Matmul/AddMM: w_buf (+ bias_buf for AddMM).
        // QM: (w_buf, s_buf, b_buf, bits, group_size).
        w_buf: u32,
        bias_buf: Option<u32>,            // AddMM bias slot
        qm: Option<(u32, u32, u32, u32)>, // (s_buf, b_buf, bits, group_size)
    }
    let mut heavies: Vec<Heavy> = Vec::new();
    let heavy_end = gated_tail
        .map(|tail| tail.sigmoid_step_idx as usize)
        .unwrap_or(steps.len());
    for (i, step) in steps.iter().enumerate().skip(1) {
        if i >= heavy_end {
            break;
        }
        match categorize_name(&step.op_name) {
            OpCategory::Heavy => {}
            _ => bail!(
                "reduction-fanout chain: non-Heavy step `{}` at position {}",
                step.op_name,
                i
            ),
        }
        match step.inputs.first() {
            Some(ChainOperand::Register { step_index: 0 }) => {}
            _ => bail!(
                "reduction-fanout chain: heavy step {} must read Register(0) first",
                i
            ),
        }
        let out_dim = *step
            .output_shape
            .last()
            .ok_or_else(|| anyhow::anyhow!("heavy step {} output shape empty", i))?;
        let out_buf = heavies.len() as u32;
        match step.op_name.as_str() {
            "Matmul" => {
                if step.inputs.len() != 2 {
                    bail!("Matmul heavy step expects 2 operands");
                }
                let w_buf = match step.inputs[1] {
                    ChainOperand::External { buffer_index } => buffer_index,
                    _ => bail!("Matmul weight must be External"),
                };
                heavies.push(Heavy {
                    idx: i as u32,
                    name: "Matmul".into(),
                    out_dim,
                    out_buf,
                    w_buf,
                    bias_buf: None,
                    qm: None,
                });
            }
            "AddMM" => {
                if step.inputs.len() != 3 {
                    bail!("AddMM heavy step expects 3 operands (x, W, bias)");
                }
                let w_buf = match step.inputs[1] {
                    ChainOperand::External { buffer_index } => buffer_index,
                    _ => bail!("AddMM weight must be External"),
                };
                let bias_buf = match step.inputs[2] {
                    ChainOperand::External { buffer_index } => buffer_index,
                    _ => bail!("AddMM bias must be External"),
                };
                heavies.push(Heavy {
                    idx: i as u32,
                    name: "AddMM".into(),
                    out_dim,
                    out_buf,
                    w_buf,
                    bias_buf: Some(bias_buf),
                    qm: None,
                });
            }
            "QuantizedMatmul" => {
                // 4 operands = standard QM (x, packed_w, scales, biases)
                // 5 operands = QM + linear bias absorbed (see general
                // fusion's `try_absorb_qm_bias`); slot 4 = bias param.
                if !(step.inputs.len() == 4 || step.inputs.len() == 5) {
                    bail!(
                        "QuantizedMatmul heavy step expects 4 or 5 operands (got {})",
                        step.inputs.len()
                    );
                }
                let extract = |k: usize| -> Result<u32> {
                    match step.inputs[k] {
                        ChainOperand::External { buffer_index } => Ok(buffer_index),
                        _ => bail!("QM operand {} must be External", k),
                    }
                };
                let w_buf = extract(1)?;
                let s_buf = extract(2)?;
                let b_buf = extract(3)?;
                let lin_bias_buf = if step.inputs.len() == 5 {
                    Some(extract(4)?)
                } else {
                    None
                };
                let (bits, group_size) = match step.op_attrs {
                    crate::graph::Attrs::Quantized { bits, group_size } => (bits, group_size),
                    _ => (4u32, 64u32),
                };
                heavies.push(Heavy {
                    idx: i as u32,
                    name: "QuantizedMatmul".into(),
                    out_dim,
                    out_buf,
                    w_buf,
                    bias_buf: lin_bias_buf,
                    qm: Some((s_buf, b_buf, bits, group_size)),
                });
            }
            other => bail!("reduction-fanout chain: unsupported heavy op `{}`", other),
        }
    }
    if heavies.len() != fusion.outputs.len() {
        bail!(
            "reduction-fanout chain: {} heavy steps but {} fusion outputs",
            heavies.len(),
            fusion.outputs.len()
        );
    }

    let total_out_dim: i32 = heavies.iter().map(|h| h.out_dim).sum();
    let token_fanout_m_rows = if active_rows > 1
        && gated_tail.is_none()
        && crate::codegen::kernels::mtp_token_m_reduction_fanout_enabled()
        && heavies.len() >= 2
        && heavies
            .iter()
            .all(|h| h.name == "QuantizedMatmul" && h.qm.is_some())
    {
        let chip = crate::target::current_chip();
        heavies
            .iter()
            .filter_map(|h| h.qm.map(|(_, _, bits, _)| (bits, h.out_dim as u32)))
            .map(|(bits, out_dim)| {
                chip.qgemm_m_rows_per_simdgroup(active_rows as u32, bits, hidden as u32, out_dim)
            })
            .min()
            .unwrap_or(1)
    } else {
        1
    };
    let token_m_qmm_fanout = token_fanout_m_rows > 1;

    // Signature: every external input, then every fusion output.
    w.line(format!(
        "// Fusion #{}: FusedChain (reduction-fanout, ROWS={}, HIDDEN={}, OUT_TOTAL={}, chain: {})",
        fusion.id, rows, hidden, total_out_dim, op_summary,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    // Per-input type rule: packed QM weights are `uint`; every other
    // input follows the captured tensor dtype.
    let mut is_w_buf = vec![false; n_inputs];
    for h in &heavies {
        if let Some((s, b, _, _)) = h.qm {
            is_w_buf[h.w_buf as usize] = true;
            let _ = (s, b);
        }
    }
    for i in 0..n_inputs {
        let ty = if is_w_buf[i] {
            "uint"
        } else {
            crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i].dtype, t)
        };
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    for (i, _) in fusion.outputs.iter().enumerate() {
        let slot = n_inputs + i;
        let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[i].dtype, t);
        w.line(format!("device {out_t}       *out{i} [[buffer({slot})]],"));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint tpg     [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint HIDDEN     = {hidden};"));
    w.line(format!("constexpr uint OUT_TOTAL  = {total_out_dim};"));
    w.line(format!("constexpr uint ROWS       = {rows};"));
    w.line(format!("constexpr uint ACTIVE_ROWS = {active_rows};"));
    w.line(format!("constexpr float EPS       = {eps};"));
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");

    if token_m_qmm_fanout {
        let full_token_blocks = (active_rows as u32) % token_fanout_m_rows == 0;
        w.line(
            "// Small-M reduction-fanout: share each quantized weight row across active M rows.",
        );
        w.line("(void)ROWS;");
        w.line(format!(
            "constexpr uint TOKEN_ROWS_PER_SG = {token_fanout_m_rows}u;"
        ));
        w.line("constexpr uint ROW_BLOCKS = (ACTIVE_ROWS + TOKEN_ROWS_PER_SG - 1u) / TOKEN_ROWS_PER_SG;");
        w.line("uint tgs_per_row_block = max(tpg / ROW_BLOCKS, 1u);");
        w.line("uint row_block = gid / tgs_per_row_block;");
        w.line("if (row_block >= ROW_BLOCKS) return;");
        w.line("uint row_base = row_block * TOKEN_ROWS_PER_SG;");
        w.line("uint tg_in_row = gid - row_block * tgs_per_row_block;");
        w.line(format!(
            "threadgroup float reduce_m_buf[{}];",
            token_fanout_m_rows * 32
        ));
        for r in 0..token_fanout_m_rows {
            w.line(format!("float sum{r} = 0.0f;"));
        }
        w.line("if constexpr (HIDDEN % 4u == 0u) {");
        w.push_indent();
        w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
        for r in 0..token_fanout_m_rows {
            if !full_token_blocks {
                w.line(format!("if (row_base + {r}u < ACTIVE_ROWS) {{"));
                w.push_indent();
            }
            w.line(format!(
                "device const {t}4* x{r}_4 = (device const {t}4*)(in{x_buf} + (row_base + {r}u) * HIDDEN);"
            ));
            w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
            w.push_indent();
            w.line(format!("float4 x{r}v = float4(x{r}_4[i]);"));
            w.line(format!("sum{r} += dot(x{r}v, x{r}v);"));
            w.pop_indent();
            w.line("}");
            if !full_token_blocks {
                w.pop_indent();
                w.line("}");
            }
        }
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        for r in 0..token_fanout_m_rows {
            if !full_token_blocks {
                w.line(format!("if (row_base + {r}u < ACTIVE_ROWS) {{"));
                w.push_indent();
            }
            w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
            w.push_indent();
            w.line(format!(
                "float x{r}v = float(in{x_buf}[(row_base + {r}u) * HIDDEN + i]);"
            ));
            w.line(format!("sum{r} += x{r}v * x{r}v;"));
            w.pop_indent();
            w.line("}");
            if !full_token_blocks {
                w.pop_indent();
                w.line("}");
            }
        }
        w.pop_indent();
        w.line("}");
        for r in 0..token_fanout_m_rows {
            w.line(format!("sum{r} = simd_sum(sum{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..token_fanout_m_rows {
            w.line(format!("reduce_m_buf[{r}u * 32u + simd_id] = sum{r};"));
        }
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        for r in 0..token_fanout_m_rows {
            w.line(format!(
                "float v{r} = (tid < tg_size / 32u) ? reduce_m_buf[{r}u * 32u + tid] : 0.0f;"
            ));
            w.line(format!("v{r} = simd_sum(v{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..token_fanout_m_rows {
            w.line(format!("reduce_m_buf[{r}] = v{r};"));
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        for r in 0..token_fanout_m_rows {
            w.line(format!(
                "float rms_scale{r} = metal::precise::rsqrt(reduce_m_buf[{r}] / float(HIDDEN) + EPS);"
            ));
        }
        w.line("uint sg_global = tg_in_row * sgs_per_tg + simd_id;");
        w.line("uint sg_stride = tgs_per_row_block * sgs_per_tg;");
        w.line("for (uint flat_out_base = sg_global;");
        w.line("     flat_out_base < OUT_TOTAL;");
        w.line("     flat_out_base += sg_stride) {");
        w.push_indent();

        let mut cum: i32 = 0;
        for (k, h) in heavies.iter().enumerate() {
            let lo = cum;
            cum += h.out_dim;
            let hi = cum;
            let keyword = if k == 0 { "if" } else { "else if" };
            w.line(format!("{keyword} (flat_out_base < {hi}u) {{"));
            w.push_indent();
            w.line(format!("uint out_idx_base = flat_out_base - {lo}u;"));
            w.line(format!("constexpr uint OUT_DIM_{k} = {};", h.out_dim));
            for r in 0..token_fanout_m_rows {
                w.line(format!("float acc{r} = 0.0f;"));
            }
            let (s_buf, b_buf, bits, group_size) = h.qm.unwrap();
            let lanes_per_word = 32 / bits;
            let packed_words = crate::codegen::kernels::quant_packed_words(hidden as u32, bits);
            w.line(format!("constexpr uint BITS_{k}  = {bits}u;"));
            w.line(format!("constexpr uint GROUP_{k} = {group_size}u;"));
            w.line(format!("constexpr uint LANES_{k} = {lanes_per_word}u;"));
            w.line(format!("constexpr uint MASK_{k}  = (1u << BITS_{k}) - 1u;"));
            w.line(format!("constexpr uint IN_WORDS_{k} = {packed_words}u;"));
            w.line(format!("constexpr uint N_GROUPS_{k} = HIDDEN / GROUP_{k};"));
            let packed_row = format!("in{} + out_idx_base * IN_WORDS_{k}", h.w_buf);
            let scales_row = format!("in{} + out_idx_base * N_GROUPS_{k}", s_buf);
            let biases_row = format!("in{} + out_idx_base * N_GROUPS_{k}", b_buf);
            let x_expr = |r, kk: &str| {
                if full_token_blocks {
                    format!("(float(in{x_buf}[(row_base + {r}u) * HIDDEN + {kk}]) * rms_scale{r} * float(in{w_rms_buf}[{kk}]))")
                } else {
                    format!(
                        "((row_base + {r}u < ACTIVE_ROWS) ? (float(in{x_buf}[(row_base + {r}u) * HIDDEN + {kk}]) * rms_scale{r} * float(in{w_rms_buf}[{kk}])) : 0.0f)"
                    )
                }
            };
            crate::codegen::kernels::emit_int4_gemv_inner_token_rows_with_options(
                w,
                |r| format!("acc{r}"),
                &packed_row,
                &scales_row,
                &biases_row,
                "HIDDEN",
                bits,
                group_size,
                token_fanout_m_rows,
                crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                    bits,
                    hidden as u32,
                    h.out_dim as u32,
                ),
                x_expr,
            );
            for r in 0..token_fanout_m_rows {
                w.line(format!("acc{r} = simd_sum(acc{r});"));
            }
            w.line("if (simd_lane == 0u) {");
            w.push_indent();
            for r in 0..token_fanout_m_rows {
                if !full_token_blocks {
                    w.line(format!("if (row_base + {r}u < ACTIVE_ROWS) {{"));
                    w.push_indent();
                }
                if let Some(lin_bias_buf) = h.bias_buf {
                    w.line(format!("acc{r} += float(in{lin_bias_buf}[out_idx_base]);"));
                }
                w.line(format!(
                    "out{}[(row_base + {r}u) * OUT_DIM_{k} + out_idx_base] = {t}(acc{r});",
                    h.out_buf
                ));
                if !full_token_blocks {
                    w.pop_indent();
                    w.line("}");
                }
            }
            w.pop_indent();
            w.line("}");
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

    let all_q8_heavies = heavies
        .iter()
        .all(|h| h.qm.map(|(_, _, bits, _)| bits == 8).unwrap_or(false));
    let norm_ty = chain_norm_smem_ty_for_quant_bits(t, if all_q8_heavies { 8 } else { 0 });
    w.line(format!("threadgroup {norm_ty} norm_y[HIDDEN];"));
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint tgs_per_row = max(tpg / ACTIVE_ROWS, 1u);");
    w.line("uint row       = gid / tgs_per_row;");
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("uint tg_in_row = gid - row * tgs_per_row;");
    w.line("(void)ROWS;");
    w.blank();
    // Stage 1+2: RMSNorm → SMEM. Vectorised 4-wide read + dot reduces
    // the per-thread iteration count to HIDDEN/(4·tg_size). Same FP add
    // order as the scalar variant (within each lane the dot is summed
    // in 4-wide chunks; across lanes the simd_sum order is unchanged),
    // so output matches scalar bit-for-bit when HIDDEN is a multiple of
    // 4 — which holds for every transformer hidden_size on the Qwen
    // family. Scalar fallback below for the (currently unreached) odd
    // case.
    // Stage 1 stashes x into norm_y SMEM so the writeback in Stage 2
    // can multiply from SMEM — one DRAM pass over x instead of two.
    w.line("float sum_sq = 0.0f;");
    w.line("if constexpr (HIDDEN % 4u == 0u) {");
    w.push_indent();
    w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
    w.line(format!(
        "device const {t}4* x4 = (device const {t}4*)(in{x_buf} + row * HIDDEN);"
    ));
    w.line(format!(
        "threadgroup {norm_ty}4* ny4_init = (threadgroup {norm_ty}4*)norm_y;"
    ));
    w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
    w.push_indent();
    w.line("float4 xv = float4(x4[i]);");
    w.line(format!(
        "ny4_init[i] = {};",
        chain_norm_vec_store_expr_for(norm_ty, "xv")
    ));
    w.line("sum_sq += dot(xv, xv);");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
    w.line(format!(
        "norm_y[i] = {};",
        chain_norm_scalar_store_expr_for(norm_ty, "xi")
    ));
    w.line("sum_sq += xi * xi;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    // Stage 2: in-place SMEM normalise (no x re-read; weight from DRAM).
    w.line("if constexpr (HIDDEN % 4u == 0u) {");
    w.push_indent();
    w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
    w.line(format!(
        "device const {t}4* w4 = (device const {t}4*)(in{w_rms_buf});"
    ));
    w.line(format!(
        "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
    ));
    w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "ny4[i] = {};",
        chain_norm_vec_store_rounded_expr_for(
            norm_ty,
            t,
            "float4(ny4[i]) * rms_scale * float4(w4[i])"
        )
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "norm_y[i] = {};",
        chain_norm_scalar_store_rounded_expr_for(
            norm_ty,
            t,
            &format!("float(norm_y[i]) * rms_scale * float(in{w_rms_buf}[i])")
        )
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();
    // Stage 3: per-simdgroup heavy GEMV. Each simdgroup owns `ROWS_PER_SG`
    // consecutive output columns of the concatenated [Q | K | V | …]
    // space; the chip target picks `ROWS_PER_SG ≥ 1` from measured
    // decode throughput. Setting `ROWS_PER_SG > 1`
    // amortises the SMEM x-load across multiple W-rows and shrinks the
    // grid by the same factor — fewer TGs, more work per TG, same total
    // FMAs. Disabled (ROWS_PER_SG=1) when any heavy's out_dim isn't a
    // multiple of the chip preference (would split a row group across
    // Heavy boundaries — codegen avoids that since the chain ranges
    // also have to stay aligned).
    let chip = crate::target::current_chip();
    let max_out = heavies.iter().map(|h| h.out_dim as u32).max().unwrap_or(1);
    let all_dense = heavies.iter().all(|h| h.qm.is_none());
    let chip_rows = if all_dense {
        chip.dense_gemv_rows_per_simdgroup(hidden as u32, max_out)
    } else {
        chip.gemv_rows_per_simdgroup()
    };
    let all_aligned = heavies
        .iter()
        .all(|h| (h.out_dim as u32) % chip_rows == 0 && (h.out_dim as u32) >= chip_rows);
    let rows_per_sg = if chip_rows > 1 && all_aligned {
        chip_rows
    } else {
        1
    };
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));

    w.line("uint sg_global = tg_in_row * sgs_per_tg + simd_id;");
    w.line("uint sg_stride = tgs_per_row * sgs_per_tg;");
    w.line("for (uint flat_out_base = sg_global * ROWS_PER_SG;");
    w.line("     flat_out_base < OUT_TOTAL;");
    w.line("     flat_out_base += sg_stride * ROWS_PER_SG) {");
    w.push_indent();

    let mut cum: i32 = 0;
    for (k, h) in heavies.iter().enumerate() {
        let lo = cum;
        cum += h.out_dim;
        let hi = cum;
        let keyword = if k == 0 { "if" } else { "else if" };
        w.line(format!("{keyword} (flat_out_base < {hi}u) {{"));
        w.push_indent();
        w.line(format!("uint out_idx_base = flat_out_base - {lo}u;"));
        w.line(format!("constexpr uint OUT_DIM_{k} = {};", h.out_dim));
        for r in 0..rows_per_sg {
            w.line(format!("float acc{r} = 0.0f;"));
        }
        match h.name.as_str() {
            "Matmul" => {
                let virtual_transposed = false;
                let out_dim_expr = format!("OUT_DIM_{k}");
                if rows_per_sg <= 1 {
                    emit_chain_dense_matmul(
                        w,
                        t,
                        h.w_buf,
                        "out_idx_base",
                        "acc0",
                        hidden as u32,
                        virtual_transposed,
                        h.out_dim as u32,
                        &out_dim_expr,
                    );
                } else {
                    emit_chain_dense_matmul_multirow(
                        w,
                        t,
                        h.w_buf,
                        "out_idx_base",
                        rows_per_sg,
                        hidden as u32,
                        virtual_transposed,
                        h.out_dim as u32,
                        &out_dim_expr,
                    );
                }
            }
            "AddMM" => {
                let virtual_transposed = false;
                let out_dim_expr = format!("OUT_DIM_{k}");
                if rows_per_sg <= 1 {
                    emit_chain_dense_matmul(
                        w,
                        t,
                        h.w_buf,
                        "out_idx_base",
                        "acc0",
                        hidden as u32,
                        virtual_transposed,
                        h.out_dim as u32,
                        &out_dim_expr,
                    );
                } else {
                    emit_chain_dense_matmul_multirow(
                        w,
                        t,
                        h.w_buf,
                        "out_idx_base",
                        rows_per_sg,
                        hidden as u32,
                        virtual_transposed,
                        h.out_dim as u32,
                        &out_dim_expr,
                    );
                }
                for r in 0..rows_per_sg {
                    w.line(format!("acc{r} = simd_sum(acc{r});"));
                }
                w.line("if (simd_lane == 0u) {");
                w.push_indent();
                let bias_buf = h.bias_buf.expect("AddMM bias_buf populated");
                for r in 0..rows_per_sg {
                    w.line(format!(
                        "acc{r} += float(in{bias_buf}[out_idx_base + {r}u]);"
                    ));
                    w.line(format!(
                        "out{out}[row * OUT_DIM_{k} + out_idx_base + {r}u] = {t}(acc{r});",
                        out = h.out_buf
                    ));
                }
                w.pop_indent();
                w.line("}");
                w.pop_indent();
                w.line("}");
                continue;
            }
            "QuantizedMatmul" => {
                let (s_buf, b_buf, bits, group_size) = h.qm.unwrap();
                let lanes_per_word = 32 / bits;
                let packed_words = crate::codegen::kernels::quant_packed_words(hidden as u32, bits);
                w.line(format!("constexpr uint BITS_{k}  = {bits}u;"));
                w.line(format!("constexpr uint GROUP_{k} = {group_size}u;"));
                w.line(format!("constexpr uint LANES_{k} = {lanes_per_word}u;"));
                w.line(format!("constexpr uint MASK_{k}  = (1u << BITS_{k}) - 1u;"));
                w.line(format!("constexpr uint IN_WORDS_{k} = {packed_words}u;"));
                w.line(format!("constexpr uint N_GROUPS_{k} = HIDDEN / GROUP_{k};"));
                if rows_per_sg <= 1 {
                    let packed_row = format!("in{} + out_idx_base * IN_WORDS_{k}", h.w_buf);
                    let scales_row = format!("in{} + out_idx_base * N_GROUPS_{k}", s_buf);
                    let biases_row = format!("in{} + out_idx_base * N_GROUPS_{k}", b_buf);
                    crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
                        w,
                        "acc0",
                        &packed_row,
                        &scales_row,
                        &biases_row,
                        "HIDDEN",
                        bits,
                        group_size,
                        crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                            bits,
                            hidden as u32,
                            h.out_dim as u32,
                        ),
                        |kk| format!("norm_y[{kk}]"),
                    );
                } else {
                    let w_buf_cap = h.w_buf;
                    let s_buf_cap = s_buf;
                    let b_buf_cap = b_buf;
                    let k_cap = k;
                    crate::codegen::kernels::emit_int4_gemv_inner_multirow(
                        w,
                        |r| format!("acc{r}"),
                        |r| format!("in{w_buf_cap} + (out_idx_base + {r}u) * IN_WORDS_{k_cap}"),
                        |r| format!("in{s_buf_cap} + (out_idx_base + {r}u) * N_GROUPS_{k_cap}"),
                        |r| format!("in{b_buf_cap} + (out_idx_base + {r}u) * N_GROUPS_{k_cap}"),
                        "HIDDEN",
                        bits,
                        group_size,
                        rows_per_sg,
                        |kk| format!("norm_y[{kk}]"),
                    );
                }
            }
            other => bail!("unreachable heavy name `{}`", other),
        }
        for r in 0..rows_per_sg {
            w.line(format!("acc{r} = simd_sum(acc{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        // QM with absorbed linear bias: apply per-output bias before
        // writing. AddMM already did this above (different branch);
        // here we cover QM with bias_buf populated via
        // try_absorb_qm_bias.
        // DEBUG: gate bias-add with env so we can test absorption
        // STRUCTURE without the additive bias to isolate magnitude
        // mismatch sources.
        if h.name == "QuantizedMatmul"
            && std::env::var_os("FUSEKIT_QM_BIAS_DEBUG_SKIP_ADD").is_none()
        {
            if let Some(lin_bias_buf) = h.bias_buf {
                for r in 0..rows_per_sg {
                    w.line(format!(
                        "acc{r} += float(in{lin_bias_buf}[out_idx_base + {r}u]);"
                    ));
                }
            }
        }
        let gated_for_this = gated_tail.filter(|tail| tail.gate_step_idx == h.idx);
        let debug_write_norm_to_first_out = std::env::var_os("FUSEKIT_RF_DEBUG_NORM_OUT0")
            .is_some()
            && k == 0
            && h.out_dim == hidden;
        for r in 0..rows_per_sg {
            let out_idx_expr = format!("out_idx_base + {r}u");
            if let Some(tail) = gated_for_this {
                let other_expr = gated_tail_other_expr(
                    fusion,
                    tail,
                    h.out_dim,
                    &format!("OUT_DIM_{k}"),
                    &out_idx_expr,
                    &format!("gate{r}"),
                )
                .ok_or_else(|| anyhow::anyhow!("gated multiply tail: bad other operand"))?;
                w.line(format!("float gate{r} = acc{r};"));
                w.line(format!("float sig{r} = 1.0f / (1.0f + exp(-gate{r}));"));
                w.line(format!(
                    "out{out}[row * OUT_DIM_{k} + {out_idx_expr}] = {t}({other_expr} * sig{r});",
                    out = h.out_buf
                ));
            } else {
                let value_expr = if debug_write_norm_to_first_out {
                    format!("norm_y[{out_idx_expr}]")
                } else {
                    format!("acc{r}")
                };
                w.line(format!(
                    "out{out}[row * OUT_DIM_{k} + {out_idx_expr}] = {t}({value_expr});",
                    out = h.out_buf
                ));
            }
        }
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

/// Emit a chain whose head is RMSNorm and step 1 is a Heavy op
/// (Matmul or QuantizedMatmul), with an optional pointwise tail. Each
/// threadgroup is sized to one input row × a strip of output elements
/// (`out_idx = simd_id`); the row's RMSNorm-normalized values land in
/// threadgroup memory so simdgroups can share them for the heavy GEMV.
fn emit_reduction_then_heavy_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let t = crate::codegen::kernels::act_ty(meta);
    let n_inputs = fusion.inputs.len();
    if steps.len() < 2 {
        bail!("reduction→heavy chain expects at least 2 steps");
    }
    let head = &steps[0];
    if head.op_name != "RMSNorm" {
        bail!("reduction→heavy chain: head must be RMSNorm");
    }
    if head.inputs.len() != 2 {
        bail!("RMSNorm head expects 2 external inputs (x, weight)");
    }
    let (x_buf, w_rms_buf) = match (head.inputs[0], head.inputs[1]) {
        (
            ChainOperand::External { buffer_index: a },
            ChainOperand::External { buffer_index: b },
        ) => (a, b),
        _ => bail!("RMSNorm head must read external buffers"),
    };
    let eps = match head.op_attrs {
        crate::graph::Attrs::RmsNorm { eps } => eps,
        _ => 1e-6_f32,
    };
    let head_out = head.output_shape.clone();
    if head_out.is_empty() {
        bail!("RMSNorm output shape empty");
    }
    let hidden = *head_out.last().unwrap();
    let rows: i32 = head_out[..head_out.len() - 1]
        .iter()
        .product::<i32>()
        .max(1);
    let active_rows: i32 = crate::target::current_chip()
        .active_rows_before_last(&head_out)
        .min(i32::MAX as u64) as i32;

    let heavy = &steps[1];
    let heavy_name = heavy.op_name.as_str();
    let heavy_out = heavy.output_shape.clone();
    if heavy_out.is_empty() {
        bail!("heavy step output shape empty");
    }
    let out_dim = *heavy_out.last().unwrap();
    let final_out_shape = fusion.outputs[0].shape.clone();
    // Heavy step's first operand must be the register holding the
    // RMSNorm-normalized row (Register{step_index: 0}). The merge logic
    // guarantees this for Reduction → Heavy folds.
    match heavy.inputs.first() {
        Some(ChainOperand::Register { step_index: 0 }) => {}
        _ => bail!("reduction→heavy chain: heavy step 1's first input must be Register(0)"),
    }

    // Resolve the heavy op's external operands so we can type their
    // buffer slots correctly (QM needs uint W + half scales/biases).
    // QM with 5 operands = absorbed linear-bias (try_absorb_qm_bias).
    let (qm_w, qm_s, qm_b, bits, group_size, qm_lin_bias) = if heavy_name == "QuantizedMatmul" {
        if !(heavy.inputs.len() == 4 || heavy.inputs.len() == 5) {
            bail!(
                "QuantizedMatmul heavy step expects 4 or 5 operands (got {})",
                heavy.inputs.len()
            );
        }
        let resolve = |idx: usize| -> Result<u32> {
            match heavy.inputs[idx] {
                ChainOperand::External { buffer_index } => Ok(buffer_index),
                ChainOperand::Register { .. } => {
                    bail!("QM weight/scale/bias must be External in reduction→heavy chain")
                }
            }
        };
        let w_buf = resolve(1)?;
        let s_buf = resolve(2)?;
        let b_buf = resolve(3)?;
        let lin_bias = if heavy.inputs.len() == 5 {
            Some(resolve(4)?)
        } else {
            None
        };
        let (bits, gs) = match heavy.op_attrs {
            crate::graph::Attrs::Quantized { bits, group_size } => (bits, group_size),
            _ => (4u32, 64u32),
        };
        (Some(w_buf), Some(s_buf), Some(b_buf), bits, gs, lin_bias)
    } else if heavy_name == "Matmul" {
        if heavy.inputs.len() != 2 {
            bail!("Matmul heavy step expects 2 operands");
        }
        (None, None, None, 0u32, 0u32, None)
    } else {
        bail!(
            "reduction→heavy chain: unsupported heavy op `{}`",
            heavy_name
        );
    };

    w.line(format!(
        "// Fusion #{}: FusedChain (reduction→heavy, ROWS={}, HIDDEN={}, OUT_DIM={}, chain: {})",
        fusion.id, rows, hidden, out_dim, op_summary,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = match (qm_w, qm_s, qm_b) {
            (Some(wi), _, _) if i as u32 == wi => "uint",
            _ => crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i].dtype, t),
        };
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, t);
    w.line(format!(
        "device {out_t}       *out0 [[buffer({})]],",
        n_inputs
    ));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint tpg     [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint HIDDEN  = {hidden};"));
    w.line(format!("constexpr uint OUT_DIM = {out_dim};"));
    w.line(format!("constexpr uint ROWS    = {rows};"));
    w.line(format!("constexpr uint ACTIVE_ROWS = {active_rows};"));
    w.line(format!("constexpr float EPS    = {eps};"));
    let norm_ty = chain_norm_smem_ty(t);
    w.line(format!("threadgroup {norm_ty} norm_y[HIDDEN];"));
    w.line("threadgroup float reduce_buf[32];");
    // Chip-aware multi-row coalesce: each simdgroup owns
    // ROWS_PER_SG consecutive output rows of the Heavy step. Shares
    // the normed-row SMEM read across R weight rows; saves one
    // sum_sq + writeback per SG. OUT_DIM must be R-divisible.
    let chip = crate::target::current_chip();
    let chip_rows = if heavy_name == "Matmul" || heavy_name == "AddMM" {
        chip.dense_gemv_rows_per_simdgroup(hidden as u32, out_dim as u32)
    } else {
        chip.gemv_rows_per_simdgroup()
    };
    let rows_per_sg =
        if chip_rows > 1 && (out_dim as u32) >= chip_rows && (out_dim as u32) % chip_rows == 0 {
            chip_rows
        } else {
            1
        };
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    // tgs_per_row comes from the dispatched grid. Decode compiles with
    // ACTIVE_ROWS=1; prefill/MTP patch modes can raise it and reuse the
    // same row mapping across the whole chain.
    w.line("uint tgs_per_row = max(tpg / ACTIVE_ROWS, 1u);");
    w.line("uint row       = gid / tgs_per_row;");
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("uint tg_in_row = gid - row * tgs_per_row;");
    w.line("(void)ROWS;");
    w.blank();
    // ── Stage 1: sum_sq reduction across hidden axis ─────────────
    // x[i] is also stashed into norm_y SMEM so Stage 2's writeback
    // can multiply from SMEM instead of re-reading the row from DRAM
    // — one DRAM pass over x instead of two. The common transformer
    // hidden sizes are 4-aligned, so use vector reads there and keep
    // a scalar fallback for odd shapes.
    w.line("float sum_sq = 0.0f;");
    w.line("if constexpr (HIDDEN % 4u == 0u) {");
    w.push_indent();
    w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
    w.line(format!(
        "device const {t}4* x4 = (device const {t}4*)(in{x_buf} + row * HIDDEN);"
    ));
    w.line(format!(
        "threadgroup {norm_ty}4* ny4_init = (threadgroup {norm_ty}4*)norm_y;"
    ));
    w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
    w.push_indent();
    w.line("float4 xv = float4(x4[i]);");
    w.line(format!(
        "ny4_init[i] = {};",
        chain_norm_vec_store_expr(t, "xv")
    ));
    w.line("sum_sq += dot(xv, xv);");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
    w.line(format!(
        "norm_y[i] = {};",
        chain_norm_scalar_store_expr(t, "xi")
    ));
    w.line("sum_sq += xi * xi;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    w.blank();
    // ── Stage 2: normalise in SMEM (no x re-read; weight from DRAM).
    w.line("if constexpr (HIDDEN % 4u == 0u) {");
    w.push_indent();
    w.line("constexpr uint HIDDEN4 = HIDDEN / 4u;");
    w.line(format!(
        "device const {t}4* w4 = (device const {t}4*)(in{w_rms_buf});"
    ));
    w.line(format!(
        "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
    ));
    w.line("for (uint i = tid; i < HIDDEN4; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "ny4[i] = {};",
        chain_norm_vec_store_rounded_expr(t, "float4(ny4[i]) * rms_scale * float4(w4[i])")
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "norm_y[i] = {};",
        chain_norm_scalar_store_rounded_expr(
            t,
            &format!("float(norm_y[i]) * rms_scale * float(in{w_rms_buf}[i])")
        )
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();
    // ── Stage 3: per-simdgroup heavy GEMV from SMEM ──────────────
    // Strided coverage: when the grid is sized smaller than
    // `ceil(OUT_DIM, sgs_per_tg * ROWS_PER_SG)` (e.g. LM-head cap),
    // each SG processes multiple R-row blocks in succession, sharing
    // the same `norm_y` SMEM. Equivalent to "for each block this SG
    // owns". When the grid is sized exactly, the loop runs once.
    w.line("uint sg_global = tg_in_row * sgs_per_tg + simd_id;");
    w.line("uint sg_stride = tgs_per_row * sgs_per_tg;");
    w.line("for (uint out_idx_base = sg_global * ROWS_PER_SG;");
    w.line("     out_idx_base < OUT_DIM;");
    w.line("     out_idx_base += sg_stride * ROWS_PER_SG) {");
    w.push_indent();
    for r in 0..rows_per_sg {
        w.line(format!("float acc{r} = 0.0f;"));
    }
    if heavy_name == "Matmul" {
        let w_buf = match heavy.inputs[1] {
            ChainOperand::External { buffer_index } => buffer_index,
            _ => bail!("Matmul weight must be External"),
        };
        // mlx-lm `Linear` stores weight in (out, in) row-major and
        // emits `x @ W.T` so the trace sees W as (in, out). The
        // underlying memory IS still (out, in) — read `W[out_idx, k]`
        // at byte offset `out_idx * HIDDEN + k`. Multi-row coalesce
        // shares the `ny4` load across R weight rows.
        for r in 0..rows_per_sg {
            w.line(format!(
                "device const {t}4* W4_{r} = (device const {t}4*)(in{w_buf} + (out_idx_base + {r}u) * HIDDEN);"
            ));
        }
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
        w.push_indent();
        w.line("float4 xv = float4(ny4[k4]);");
        for r in 0..rows_per_sg {
            w.line(format!("acc{r} += dot(xv, float4(W4_{r}[k4]));"));
        }
        w.pop_indent();
        w.line("}");
    } else {
        let lanes_per_word = 32 / bits;
        let packed_words = crate::codegen::kernels::quant_packed_words(hidden as u32, bits);
        w.line(format!("constexpr uint BITS  = {bits}u;"));
        w.line(format!("constexpr uint GROUP = {group_size}u;"));
        w.line(format!("constexpr uint LANES = {lanes_per_word}u;"));
        w.line("constexpr uint MASK = (1u << BITS) - 1u;");
        w.line(format!("constexpr uint IN_WORDS = {packed_words}u;"));
        w.line("constexpr uint N_GROUPS = HIDDEN / GROUP;");
        let w_buf = qm_w.unwrap();
        let s_buf = qm_s.unwrap();
        let b_buf = qm_b.unwrap();
        if rows_per_sg <= 1 {
            crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
                w,
                "acc0",
                &format!("in{w_buf} + out_idx_base * IN_WORDS"),
                &format!("in{s_buf} + out_idx_base * N_GROUPS"),
                &format!("in{b_buf} + out_idx_base * N_GROUPS"),
                "HIDDEN",
                bits,
                group_size,
                crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                    bits,
                    hidden as u32,
                    out_dim as u32,
                ),
                |off| format!("norm_y[{off}]"),
            );
        } else {
            crate::codegen::kernels::emit_int4_gemv_inner_multirow(
                w,
                |r| format!("acc{r}"),
                |r| format!("in{w_buf} + (out_idx_base + {r}u) * IN_WORDS"),
                |r| format!("in{s_buf} + (out_idx_base + {r}u) * N_GROUPS"),
                |r| format!("in{b_buf} + (out_idx_base + {r}u) * N_GROUPS"),
                "HIDDEN",
                bits,
                group_size,
                rows_per_sg,
                |off| format!("norm_y[{off}]"),
            );
        }
    }
    for r in 0..rows_per_sg {
        w.line(format!("acc{r} = simd_sum(acc{r});"));
    }
    w.blank();
    // ── Pointwise tail (steps 2..N) emitted once per row ──────────
    // Inside stride loop so we can't `return` on non-lane-0 — use a
    // guarded block instead.
    let n_steps = steps.len();
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    for r in 0..rows_per_sg {
        w.line("{");
        w.push_indent();
        w.line(format!("uint out_idx = out_idx_base + {r}u;"));
        // Apply absorbed linear bias (QM + Add(bias)) before any
        // pointwise tail. mlx's QuantizedLinear bias is per-output.
        if let Some(lin_bias) = qm_lin_bias {
            w.line(format!("acc{r} += float(in{lin_bias}[out_idx]);"));
        }
        w.line(format!(
            "float r1 = {};",
            round_scalar_to_dtype_expr(steps[1].output_dtype, t, &format!("acc{r}"))
        ));
        for (idx, step) in steps.iter().enumerate().skip(2) {
            let is_last = idx + 1 == n_steps;
            let expr = render_pointwise_expr_heavy_tail(step, fusion, out_dim, &final_out_shape)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "fused_chain reduction→heavy tail: unsupported op `{}`",
                        step.op_name
                    )
                })?;
            if is_last {
                w.line(format!("out0[row * OUT_DIM + out_idx] = {t}({expr});"));
            } else {
                w.line(format!(
                    "float r{idx} = {};",
                    round_scalar_to_dtype_expr(step.output_dtype, t, &expr)
                ));
            }
        }
        if n_steps == 2 {
            w.line(format!("out0[row * OUT_DIM + out_idx] = {t}(r1);"));
        }
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

/// Pointwise-tail expression renderer for the reduction→heavy emit.
/// External operands are indexed by `[row * OUT_DIM + out_idx]` when
/// they share the full output shape, by `[out_idx]` when they're an
/// OUT_DIM-broadcast (e.g. a per-out bias), and by `[0]` when scalar.
fn render_pointwise_expr_heavy_tail(
    step: &ChainStep,
    fusion: &Fusion,
    out_dim: i32,
    out_shape: &[i32],
) -> Option<String> {
    let a = step
        .inputs
        .first()
        .map(|op| operand_expr_heavy_tail(op, fusion, out_dim, out_shape));
    let b = step
        .inputs
        .get(1)
        .map(|op| operand_expr_heavy_tail(op, fusion, out_dim, out_shape));
    let c = step
        .inputs
        .get(2)
        .map(|op| operand_expr_heavy_tail(op, fusion, out_dim, out_shape));
    match step.op_name.as_str() {
        "Add" => Some(format!("{} + {}", a?, b?)),
        "Subtract" => Some(format!("{} - {}", a?, b?)),
        "Multiply" => Some(format!("{} * {}", a?, b?)),
        "Divide" => Some(format!("{} / {}", a?, b?)),
        "Power" => Some(format!("pow({}, {})", a?, b?)),
        "Maximum" => Some(format!("max({}, {})", a?, b?)),
        "Minimum" => Some(format!("min({}, {})", a?, b?)),
        "Sigmoid" => Some(format!("(1.0f / (1.0f + exp(-{})))", a?)),
        "Exp" => Some(format!("exp({})", a?)),
        "Negative" => Some(format!("(-{})", a?)),
        "LogAddExp" => Some(format!(
            "(max({a},{b}) + log(1.0f + exp(-fabs({a} - {b}))))",
            a = a?,
            b = b?
        )),
        "Select" => Some(format!(
            "(({a} != 0.0f) ? {b} : {c})",
            a = a?,
            b = b?,
            c = c?
        )),
        "GreaterEqual" => Some(format!("(({a} >= {b}) ? 1.0f : 0.0f)", a = a?, b = b?)),
        "AsType" => Some(a?),
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => Some(a?),
        _ => None,
    }
}

fn operand_expr_heavy_tail(
    op: &ChainOperand,
    fusion: &Fusion,
    out_dim: i32,
    out_shape: &[i32],
) -> String {
    match op {
        ChainOperand::External { buffer_index } => {
            let s = fusion
                .inputs
                .get(*buffer_index as usize)
                .map(|b| b.shape.as_slice())
                .unwrap_or(&[]);
            let elems: i32 = s.iter().product::<i32>().max(1);
            let same = s == out_shape;
            let broadcast = elems == out_dim && s.iter().any(|&d| d == out_dim);
            if same {
                format!("float(in{}[row * OUT_DIM + out_idx])", buffer_index)
            } else if broadcast {
                format!("float(in{}[out_idx])", buffer_index)
            } else {
                format!("float(in{}[0])", buffer_index)
            }
        }
        ChainOperand::Register { step_index } => {
            format!("r{}", step_index)
        }
    }
}

/// Emit a chain whose first step is RMSNorm. One threadgroup per row;
/// threads cooperate via simdgroup_sum to compute the row's
/// `rms_scale`, then each thread iterates the hidden axis applying
/// the RMSNorm body (`x[i] * rms_scale * w[i]`) into register `r0`
/// and runs the pointwise tail in place.
fn emit_reduction_first_chain(
    fusion: &Fusion,
    steps: &[ChainStep],
    meta: &ModelMeta,
    w: &mut MslWriter,
    fn_name: &str,
    op_summary: &str,
) -> Result<()> {
    let fallback_t = crate::codegen::kernels::act_ty(meta);
    let out_t = crate::codegen::kernels::scalar_ty_for_dtype(fusion.outputs[0].dtype, fallback_t);
    let n_inputs = fusion.inputs.len();
    let first = &steps[0];
    if first.op_name != "RMSNorm" {
        bail!("fused_chain reduction head: only RMSNorm supported");
    }
    if first.inputs.len() != 2 {
        bail!("RMSNorm chain head expects 2 external inputs (x, weight)");
    }
    let (x_buf, w_buf) = match (first.inputs[0], first.inputs[1]) {
        (
            ChainOperand::External { buffer_index: a },
            ChainOperand::External { buffer_index: b },
        ) => (a, b),
        _ => bail!("RMSNorm head must read external buffers"),
    };
    let eps = match first.op_attrs {
        crate::graph::Attrs::RmsNorm { eps } => eps,
        _ => 1e-6_f32,
    };
    let out_shape = fusion.outputs[0].shape.clone();
    if out_shape.is_empty() {
        bail!("RMSNorm chain output has no shape");
    }
    let norm_shape = first.output_shape.clone();
    if norm_shape.is_empty() {
        bail!("RMSNorm chain head has no output shape");
    }
    let hidden = *norm_shape.last().unwrap();
    // Pass-wide active-M policy: decode uses one token row; prefill/MTP
    // patch modes can retain more sequence rows while preserving
    // post-sequence axes such as Qwen3 GatedDelta `[B,S,H,D]`.
    let rows: i32 = crate::target::current_chip()
        .active_rows_before_last(&norm_shape)
        .min(i32::MAX as u64) as i32;
    let weight_elems: i32 = fusion
        .inputs
        .get(w_buf as usize)
        .map(|b| b.shape.iter().product::<i32>().max(1))
        .unwrap_or(1);
    let weight_is_vector = weight_elems >= hidden;

    w.line(format!(
        "// Fusion #{}: FusedChain (reduction-first RMSNorm, ROWS={}, HIDDEN={}, chain: {})",
        fusion.id, rows, hidden, op_summary
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = crate::codegen::kernels::scalar_ty_for_dtype(fusion.inputs[i].dtype, fallback_t);
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    w.line(format!(
        "device {out_t}       *out0 [[buffer({})]],",
        n_inputs
    ));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line(format!("constexpr uint HIDDEN = {hidden};"));
    w.line(format!("constexpr uint ROWS   = {rows};"));
    w.line(format!("constexpr float EPS   = {eps};"));
    w.line(format!(
        "constexpr bool WEIGHT_IS_VECTOR = {};",
        if weight_is_vector { "true" } else { "false" }
    ));
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint row = gid;");
    w.line("if (row >= ROWS) return;");
    // ── sum_sq reduction across hidden axis ───────────────────
    w.line("float sum_sq = 0.0f;");
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!("float xi = float(in{x_buf}[row * HIDDEN + i]);"));
    w.line("sum_sq += xi * xi;");
    w.pop_indent();
    w.line("}");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    w.line(format!(
        "float scalar_weight = WEIGHT_IS_VECTOR ? 1.0f : float(in{w_buf}[0]);"
    ));
    w.blank();
    // ── per-element pass: emit r0, then pointwise tail, then store ──
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "float wi = WEIGHT_IS_VECTOR ? float(in{w_buf}[i]) : scalar_weight;"
    ));
    w.line(format!(
        "float r0 = float(in{x_buf}[row * HIDDEN + i]) * rms_scale * wi;"
    ));
    let n_steps = steps.len();
    for (idx, step) in steps.iter().enumerate().skip(1) {
        let is_last = idx + 1 == n_steps;
        let expr = render_pointwise_expr_reduction(step, fusion, hidden, rows, &out_shape)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "fused_chain reduction tail: unsupported op `{}`",
                    step.op_name
                )
            })?;
        if is_last {
            w.line(format!("out0[row * HIDDEN + i] = {out_t}({expr});"));
        } else {
            w.line(format!("float r{idx} = float({expr});"));
        }
    }
    if n_steps == 1 {
        w.line(format!("out0[row * HIDDEN + i] = {out_t}(r0);"));
    }
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
