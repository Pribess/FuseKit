//! Universal per-op MSL emitter — the "1:1 emit" path.
//!
//! Every unmatched op in the graph becomes a `FusionKind::Single(op_name)`
//! at Stage 2. The pattern emitters above this file handle hand-tuned
//! fusions; this one is the universal fallback so that no op leaves
//! Stage 4 without a metallib function. That's the prerequisite for
//! "all ops live inside the megakernel" — fusekit's compiler invariant.
//!
//! Op categories drive the emitter shape:
//! - **Pointwise** (Add, Multiply, Sigmoid, Exp, Negative, LogAddExp,
//!   AsType, Select, GreaterEqual): one element per thread, same shape
//!   in/out.
//! - **View** (Reshape, Transpose, Broadcast, Flatten, Unflatten,
//!   Squeeze, ExpandDims): zero-compute. mlx's view ops alias the
//!   underlying storage. We emit either a no-op kernel (when the
//!   downstream consumer needs the buffer materialized) or an explicit
//!   copy.
//! - **Heavy** (Matmul, Softmax, RoPE, RMSNorm, Convolution,
//!   CustomKernel, Gather, Arange, Quantize, Full): each has its own
//!   per-op emitter below.
//!
//! Unknown op names produce an MSL function whose body is a single
//! `// TODO` comment plus a stub write of zeros. This keeps the
//! metallib compiling; the Python install will refuse to dispatch a
//! TODO kernel so the runtime fails loud rather than producing
//! silent garbage.
//!
//! All buffer types are derived from `act_ty(meta)` so the same emitter
//! works for bf16 and fp16 models — see codegen::kernels::act_ty.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{
    AttentionMask, ChainOperand, ChainStep, Fusion, FusionAttrs, FusionBoundary, FusionKind,
    ModelMeta,
};
use crate::graph::Dtype;
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    let (op_name, op_attrs) = match &fusion.attrs {
        FusionAttrs::Single { op_name, op_attrs } => (op_name.clone(), op_attrs.clone()),
        _ => bail!(
            "single emitter: attrs mismatch — expected Single, got {:?}",
            fusion.attrs
        ),
    };
    if !matches!(&fusion.kind, FusionKind::Single(_)) {
        bail!(
            "single emitter called with non-Single fusion: {:?}",
            fusion.kind
        );
    }

    match op_name.as_str() {
        // ── Pointwise (binary) ────────────────────────────────────
        "Add" => emit_pointwise_binary(fusion, meta, w, "+"),
        "Subtract" => emit_pointwise_binary(fusion, meta, w, "-"),
        "Multiply" => emit_pointwise_binary(fusion, meta, w, "*"),
        "Divide" => emit_pointwise_binary(fusion, meta, w, "/"),
        "Power" => emit_power(fusion, meta, w),
        "Maximum" => emit_pointwise_binary_fn(fusion, meta, w, "max"),
        "Minimum" => emit_pointwise_binary_fn(fusion, meta, w, "min"),

        // ── Pointwise (unary) ─────────────────────────────────────
        "Sigmoid" => {
            emit_pointwise_unary(fusion, meta, w, "(1.0f / (1.0f + metal::fast::exp(-_x)))")
        }
        "Tanh" => emit_pointwise_unary(
            fusion,
            meta,
            w,
            "metal::fast::tanh(clamp(_x, -20.0f, 20.0f))",
        ),
        "Exp" => emit_pointwise_unary(fusion, meta, w, "exp(_x)"),
        "Negative" => emit_pointwise_unary(fusion, meta, w, "(-_x)"),

        // ── Pointwise (ternary / mixed) ───────────────────────────
        "LogAddExp" => emit_log_add_exp(fusion, meta, w),
        "Select" => emit_select(fusion, meta, w),
        "GreaterEqual" => emit_greater_equal(fusion, meta, w),

        // ── Type cast ─────────────────────────────────────────────
        "AsType" => emit_astype(fusion, meta, w),

        // ── View-only (element-contiguous passthrough) ────────────
        // Flatten/Unflatten/Squeeze/ExpandDims never change element
        // order in mlx — output is element-identical to input. Same
        // for Reshape, which mlx materializes contiguously when its
        // input isn't.
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => {
            emit_view_passthrough(fusion, meta, w)
        }
        // Stride-aware view ops:
        "Broadcast" => emit_broadcast(fusion, &op_attrs, meta, w),
        "Transpose" => emit_transpose(fusion, &op_attrs, meta, w),
        "TransposeSplit" => emit_transpose_split(fusion, &op_attrs, meta, w),

        // ── Heavy ─────────────────────────────────────────────────
        "Matmul" => emit_matmul(fusion, meta, w),
        "ScaledDotProductAttention" => {
            emit_scaled_dot_product_attention_prefill(fusion, partition, meta, w)
        }
        "Softmax" => emit_softmax(fusion, &op_attrs, meta, w),
        "SoftmaxTopKRoute" => emit_softmax_topk_route(fusion, meta, w),
        "Arange" => emit_arange(fusion, &op_attrs, meta, w),
        "Full" => emit_full(fusion, meta, w),
        "Slice" => emit_slice(fusion, &op_attrs, meta, w),
        "ArgPartition" | "Argpartition" => emit_argpartition(fusion, meta, w),
        "ArgTopKAsc" => emit_argtopk(fusion, meta, w, false),
        "ArgTopKDesc" => emit_argtopk(fusion, meta, w, true),
        "GatherAxis" => emit_gather_axis(fusion, meta, w),
        "GatherAxisScaleExpand" => emit_gather_axis_scale_expand(fusion, meta, w),
        "GatherQMM" => emit_gather_qmm(fusion, meta, w),
        "GatherQMMReduce" => emit_gather_qmm_reduce(fusion, meta, w),
        "Gather" => emit_gather(fusion, &op_attrs, meta, w),
        "Concatenate" => emit_concatenate(fusion, &op_attrs, meta, w),
        "ScaledSumReduce" | "WeightedRouteReduce" => emit_scaled_sum_reduce(fusion, meta, w),
        "Reduce" => emit_reduce_sum(fusion, meta, w),
        "Convolution" => emit_convolution(fusion, meta, w, false),
        "ConvolutionSilu" => emit_convolution(fusion, meta, w, true),
        "Split" => emit_split(fusion, &op_attrs, meta, w),
        "RMSNorm" => emit_rmsnorm(fusion, &op_attrs, meta, w),
        "QuantizedMatmul" => emit_quantized_matmul(fusion, &op_attrs, meta, w),
        // mlx's `mx.fast.metal_kernel` escape hatch — we already have
        // the full Metal source in op_attrs; drop it into the metallib
        // verbatim. dedup happens via function_name returning the
        // mlx-mangled name (same across all callsites).
        "CustomKernel" => emit_custom_kernel(&op_attrs, w),
        // mx.fast.rope — standalone trace op (vs synthetic
        // `RopeQDecode` which is hardcoded per-layer in build.rs).
        // This emit is used when the synthetic injection is conditional
        // (Direction A) and trace's `Single(RoPE)` op survives.
        "RoPE" => emit_rope(fusion, &op_attrs, meta, w),
        // Convolution / Quantize are still TODO.
        _ => emit_todo_stub(fusion, meta, w, op_name.as_str()),
    }
}

pub fn function_name(fusion: &Fusion) -> String {
    // CustomKernel ops share one MSL function across all callsites —
    // mlx mangles `(dtype, head_dim, num_heads, …)` into the name so
    // two CustomKernel fusions with the same mangled name necessarily
    // execute identical code. Return that mangled name so codegen's
    // (kind, shape, partition) dedup collapses all 48 callsites to a
    // single emitted function. Non-CustomKernel Single fusions stay
    // on the per-fusion `fusekit_op_{kind}_fN` scheme.
    if let FusionAttrs::Single {
        op_attrs: crate::graph::Attrs::CustomKernel { name, .. },
        ..
    } = &fusion.attrs
    {
        return name.clone();
    }
    let kind = match &fusion.kind {
        FusionKind::Single(name) => name.to_ascii_lowercase(),
        _ => "unknown".into(),
    };
    format!("fusekit_op_{kind}_f{}", fusion.id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaPrefillPlan {
    BlockNax,
    FlashGqa,
    Online,
    OnlineVector,
    Materialized,
}

pub fn sdpa_prefill_flash_tiling(meta: &ModelMeta) -> (u32, u32, u32, u32) {
    let n_heads = meta.num_attention_heads.max(1);
    let n_kv_heads = meta.num_key_value_heads.max(1);
    let heads_per_kv = (n_heads / n_kv_heads).max(1);
    let max_sgs = crate::target::current_chip()
        .budgets()
        .max_simdgroups_per_tg
        .clamp(1, 8) as u32;
    let q_heads_per_tg = heads_per_kv.min(4).min(max_sgs).max(1);
    // Keep one query row per threadgroup by default. A row-blocked variant
    // amortizes K/V tile loads across multiple rows, but profiling on M5
    // showed the larger 256-thread TG increasing SDPA GPU time for long
    // prefill because barrier/occupancy cost exceeded the saved loads.
    let q_rows_per_tg = 1;
    let q_head_blocks = heads_per_kv.div_ceil(q_heads_per_tg).max(1);
    let flash_sgs = q_heads_per_tg.saturating_mul(q_rows_per_tg).max(1);
    (q_heads_per_tg, q_rows_per_tg, q_head_blocks, flash_sgs)
}

fn sdpa_prefill_dims(fusion: &Fusion, meta: &ModelMeta) -> Option<(u32, u32, u32, u32, u32)> {
    let out_shape = fusion
        .outputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    if out_shape.len() < 4 {
        return None;
    }
    let out_seq = out_shape[out_shape.len() - 2].max(1) as u32;
    let active_m = crate::target::current_chip().active_m(out_seq).max(1);
    let n_heads = meta.num_attention_heads.max(1);
    let n_kv_heads = meta.num_key_value_heads.max(1);
    let head_dim = meta.head_dim.max(1);
    let heads_per_kv = (n_heads / n_kv_heads).max(1);
    Some((active_m, n_heads, n_kv_heads, head_dim, heads_per_kv))
}

fn sdpa_single_decode_dims(fusion: &Fusion, meta: &ModelMeta) -> Option<(u32, u32, u32, u32)> {
    let out_shape = fusion.outputs.first().map(|b| b.shape.as_slice())?;
    if out_shape.len() < 3 {
        return None;
    }
    let n_heads = meta.num_attention_heads.max(1);
    let q_shape = fusion
        .inputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let qk_dim = q_shape
        .last()
        .copied()
        .unwrap_or(meta.head_dim as i32)
        .max(1) as u32;
    let v_dim = fusion
        .inputs
        .get(2)
        .and_then(|b| b.shape.last().copied())
        .or_else(|| out_shape.last().copied())
        .unwrap_or(meta.head_dim as i32)
        .max(1) as u32;
    let active_m =
        crate::codegen::kernels::sdpa_decode::active_rows_for_shape(out_shape, n_heads, v_dim)
            .max(1);
    Some((active_m, n_heads, qk_dim, v_dim))
}

/// Single SDPA ops in a decode pipeline are already rebound to semantic
/// Q/K/V cache buffers by task metadata. When the shape is a single
/// active decode row, lower them through the decode SDPA emitter instead
/// of the prefill materialized-score path. This is an op/shape rule, not
/// a model-name or layer pattern.
pub fn sdpa_single_decode_flash_enabled(fusion: &Fusion, meta: &ModelMeta) -> bool {
    if !sdpa_single_decode_enabled(fusion, meta) {
        return false;
    }
    let Some((active_m, n_heads, qk_dim, v_dim)) = sdpa_single_decode_dims(fusion, meta) else {
        return false;
    };
    crate::codegen::kernels::sdpa_decode::flash_decoding_enabled_for(
        n_heads,
        meta.num_key_value_heads.max(1),
        qk_dim,
        v_dim,
        active_m,
    )
}

pub fn sdpa_single_decode_enabled(fusion: &Fusion, meta: &ModelMeta) -> bool {
    if !matches!(&fusion.kind, FusionKind::Single(name) if name == "ScaledDotProductAttention") {
        return false;
    }
    let Some((active_m, _n_heads, _qk_dim, v_dim)) = sdpa_single_decode_dims(fusion, meta) else {
        return false;
    };
    // Decode lowering is valid for a single active query row. The
    // decode emitter itself decides whether it can use flash chunking
    // or the one-TG-per-head online path. Keep the V_DIM capability
    // aligned with the decode kernel's one-output-dimension-per-thread
    // implementation.
    active_m == 1 && v_dim <= 256
}

pub fn sdpa_prefill_plan(fusion: &Fusion, meta: &ModelMeta) -> SdpaPrefillPlan {
    let Some((active_m, _n_heads, _n_kv_heads, head_dim, heads_per_kv)) =
        sdpa_prefill_dims(fusion, meta)
    else {
        return SdpaPrefillPlan::Materialized;
    };
    let forced = std::env::var("FUSEKIT_SDPA_PREFILL_MODE")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    if forced == "materialized" || forced == "scores" || forced == "gemm" {
        return SdpaPrefillPlan::Materialized;
    }
    if forced == "online" {
        return SdpaPrefillPlan::Online;
    }
    if forced == "vector" || forced == "mlx-vector" {
        return SdpaPrefillPlan::OnlineVector;
    }
    let can_block_nax = active_m >= 512
        && matches!(head_dim, 64 | 128)
        && std::env::var("FUSEKIT_DISABLE_SDPA_PREFILL_BLOCK_NAX").as_deref() != Ok("1");
    if forced == "block" || forced == "blocknax" || forced == "nax" {
        return if can_block_nax {
            SdpaPrefillPlan::BlockNax
        } else {
            SdpaPrefillPlan::Materialized
        };
    }
    let can_flash = active_m > 1
        && head_dim <= 128
        && heads_per_kv >= 2
        && std::env::var("FUSEKIT_DISABLE_SDPA_PREFILL_FLASH").as_deref() != Ok("1");
    if forced == "flash" {
        return if can_flash {
            SdpaPrefillPlan::FlashGqa
        } else {
            SdpaPrefillPlan::Online
        };
    }
    if active_m > 1 && head_dim <= 128 {
        if can_block_nax {
            SdpaPrefillPlan::BlockNax
        } else if can_flash && active_m <= 256 {
            SdpaPrefillPlan::FlashGqa
        } else if std::env::var("FUSEKIT_DISABLE_SDPA_PREFILL_ONLINE")
            .ok()
            .as_deref()
            != Some("1")
            && active_m < 512
        {
            SdpaPrefillPlan::Online
        } else {
            SdpaPrefillPlan::Materialized
        }
    } else {
        SdpaPrefillPlan::Materialized
    }
}

fn emit_custom_kernel(op_attrs: &crate::graph::Attrs, w: &mut MslWriter) -> Result<()> {
    let (name, source) = match op_attrs {
        crate::graph::Attrs::CustomKernel { name, source, .. } => (name.clone(), source.clone()),
        _ => bail!("CustomKernel emitter: attrs mismatch"),
    };
    w.line(format!("// CustomKernel passthrough: {name}"));
    // The source already contains a `[[kernel]] void <name>(...) { ... }`
    // declaration whose function name is `name`. Drop it verbatim;
    // metal compile sees it the same as a hand-written kernel of
    // ours. Dedup is handled at the codegen level by `function_name`
    // returning `name`.
    if name.contains("gated_delta_step")
        && crate::target::current_chip().active_m_for_unshaped(8) > 1
    {
        w.line(add_gated_delta_mid_state_output(&source)?);
    } else {
        w.line(source);
    }
    w.blank();
    Ok(())
}

fn add_gated_delta_mid_state_output(source: &str) -> Result<String> {
    let sig_anchor = "device float* state_out [[buffer(8)]],";
    let body_anchor = "          // Increment data pointers to next time step";
    if !source.contains(sig_anchor) {
        bail!("gated_delta custom kernel signature did not contain state_out buffer");
    }
    if !source.contains(body_anchor) {
        bail!("gated_delta custom kernel body did not contain time-step anchor");
    }
    let with_sig = source.replace(
        sig_anchor,
        "device float* state_out [[buffer(8)]],\n  device float* mid_state_out [[buffer(9)]],",
    );
    Ok(with_sig.replace(
        body_anchor,
        "          if (t == 0 && T > 1) {\n            for (int i = 0; i < n_per_t; ++i) {\n              auto s_idx = n_per_t * dk_idx + i;\n              mid_state_out[(n * Dv + dv_idx) * Dk + s_idx] = static_cast<StT>(state[i]);\n            }\n          }\n          // Increment data pointers to next time step",
    ))
}

// ─── helpers ───────────────────────────────────────────────────────────────

fn total_elements(boundary_shape: &[i32]) -> u32 {
    boundary_shape
        .iter()
        .copied()
        .filter(|&d| d > 0)
        .product::<i32>() as u32
}

fn active_elements(boundary_shape: &[i32]) -> u32 {
    crate::target::current_chip()
        .active_elems(boundary_shape)
        .min(u32::MAX as u64) as u32
}

fn emit_header(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    n_inputs: usize,
    n_outputs: usize,
    descr: &str,
) -> (String, String) {
    let fallback = crate::codegen::kernels::act_ty(meta);
    let out_t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single({}) — {}",
        fusion.id, descr, descr
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    for i in 0..n_inputs {
        let ty = fusion
            .inputs
            .get(i)
            .map(|b| msl_ty(b.dtype, fallback))
            .unwrap_or(fallback);
        w.line(format!("device const {ty} *in{i} [[buffer({i})]],"));
    }
    for i in 0..n_outputs {
        let b = n_inputs + i;
        let ty = fusion
            .outputs
            .get(i)
            .map(|bd| msl_ty(bd.dtype, fallback))
            .unwrap_or(fallback);
        w.line(format!("device {ty}       *out{i} [[buffer({b})]],"));
    }
    let attr_base = n_inputs + n_outputs;
    w.line(format!("uint tid     [[thread_position_in_threadgroup]],"));
    w.line(format!("uint tg_size [[threads_per_threadgroup]],"));
    w.line(format!("uint gid     [[threadgroup_position_in_grid]]"));
    let _ = attr_base;
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    (out_t.to_string(), fn_name)
}

fn msl_ty(dtype: Dtype, fallback: &'static str) -> &'static str {
    match dtype {
        Dtype::Bool => "bool",
        Dtype::U8 => "uchar",
        Dtype::U16 => "ushort",
        Dtype::U32 => "uint",
        Dtype::U64 => "ulong",
        Dtype::I8 => "char",
        Dtype::I16 => "short",
        Dtype::I32 => "int",
        Dtype::I64 => "long",
        Dtype::F16 => "half",
        Dtype::F32 => "float",
        Dtype::Bf16 => "bfloat",
        Dtype::F64 | Dtype::Complex64 | Dtype::Other(_) => fallback,
    }
}

fn emit_close(w: &mut MslWriter) {
    w.pop_indent();
    w.line("}");
    w.blank();
}

// ─── pointwise ─────────────────────────────────────────────────────────────

fn emit_pointwise_binary(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    op: &str,
) -> Result<()> {
    if fusion.inputs.len() < 2 || fusion.outputs.is_empty() {
        bail!("pointwise binary: expected 2+ inputs, 1+ outputs");
    }
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "pointwise binary");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line(format!("out0[i] = {t}(float(in0[i]) {op} float(in1[i]));"));
    emit_close(w);
    Ok(())
}

fn emit_pointwise_binary_fn(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    func: &str,
) -> Result<()> {
    if fusion.inputs.len() < 2 || fusion.outputs.is_empty() {
        bail!("pointwise binary fn: expected 2+ inputs, 1+ outputs");
    }
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "pointwise binary fn");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line(format!(
        "out0[i] = {t}({func}(float(in0[i]), float(in1[i])));"
    ));
    emit_close(w);
    Ok(())
}

fn emit_power(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    if fusion.inputs.len() < 2 || fusion.outputs.is_empty() {
        bail!("Power: expected 2+ inputs, 1+ outputs");
    }
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "Power");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line("float a = float(in0[i]);");
    w.line("float b = float(in1[i]);");
    w.line("float y;");
    w.line("if (fabs(b - 2.0f) < 1.0e-6f) {");
    w.push_indent();
    w.line("y = a * a;");
    w.pop_indent();
    w.line("} else if (fabs(b - 3.0f) < 1.0e-6f) {");
    w.push_indent();
    w.line("y = a * a * a;");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("y = pow(a, b);");
    w.pop_indent();
    w.line("}");
    w.line(format!("out0[i] = {t}(y);"));
    emit_close(w);
    Ok(())
}

fn emit_pointwise_unary(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    expr_using_x: &str, // expression referring to `_x`
) -> Result<()> {
    if fusion.inputs.is_empty() || fusion.outputs.is_empty() {
        bail!("pointwise unary: expected 1+ input, 1+ output");
    }
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 1, 1, "pointwise unary");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line("float _x = float(in0[i]);");
    w.line(format!("out0[i] = {t}({expr_using_x});"));
    emit_close(w);
    Ok(())
}

fn emit_log_add_exp(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "LogAddExp");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    // Numerically stable: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|)).
    // (Metal lacks `log1p`; use the expanded form.)
    w.line("float a = float(in0[i]);");
    w.line("float b = float(in1[i]);");
    w.line("float m = max(a, b);");
    w.line(format!("out0[i] = {t}(m + log(1.0f + exp(-fabs(a - b))));"));
    emit_close(w);
    Ok(())
}

fn emit_select(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    if fusion.inputs.len() < 3 {
        bail!("Select: expected 3 inputs (cond, a, b)");
    }
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 3, 1, "Select");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    // mlx Select: out = cond ? a : b. cond is bool/uint8 in mlx but we
    // still receive it as a typed buffer; treat any non-zero element as
    // true to be robust to representations.
    w.line("bool c = float(in0[i]) != 0.0f;");
    w.line(format!("out0[i] = c ? in1[i] : in2[i];"));
    let _ = t;
    emit_close(w);
    Ok(())
}

fn emit_greater_equal(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "GreaterEqual");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    // mlx returns bool but we don't have a bool buffer dtype in our
    // mapping yet — write 1.0/0.0 in the activation dtype so a
    // downstream Select treats it correctly. This is a placeholder;
    // a typed bool buffer is on the followup list.
    w.line(format!(
        "out0[i] = (float(in0[i]) >= float(in1[i])) ? {t}(1) : {t}(0);"
    ));
    emit_close(w);
    Ok(())
}

fn emit_astype(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    let n = active_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 1, 1, "AsType");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line(format!("out0[i] = {t}(in0[i]);"));
    emit_close(w);
    Ok(())
}

// ─── view-like ─────────────────────────────────────────────────────────────

fn emit_view_passthrough(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // mlx view ops alias storage; a strict element-count check tells us
    // the layout is element-contiguous in our typed scalar units.
    let n_in = total_elements(&fusion.inputs[0].shape);
    let n_out = total_elements(&fusion.outputs[0].shape);
    if n_in != n_out {
        bail!(
            "view passthrough: element count mismatch (in {n_in} vs out {n_out}) — \
             non-contiguous view requires explicit emit"
        );
    }
    let (_t, _) = emit_header(fusion, meta, w, 1, 1, "view passthrough (memcpy)");
    let n = n_out;
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line("out0[i] = in0[i];");
    emit_close(w);
    Ok(())
}

fn emit_broadcast(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // mlx Broadcast: per-dim, a size-1 input dim expands to the target's
    // dim, all other input dims must already match. Indexing rule: for
    // output index out_idx with multi-dim coordinates [c0..cR), the
    // input is read at coordinates where c_d = 0 if input.dim(d) == 1
    // else c_d. Compute row-major strides from in_shape with
    // broadcast-zero stride for size-1 dims.
    let in_shape = &fusion.inputs[0].shape;
    let out_shape: Vec<i32> = match op_attrs {
        crate::graph::Attrs::Broadcast { target_shape } => target_shape.clone(),
        _ => fusion.outputs[0].shape.clone(),
    };
    let rank = out_shape.len();
    // Left-pad input shape to output rank.
    let pad = rank.saturating_sub(in_shape.len());
    let in_shape_padded: Vec<i32> = std::iter::repeat(1)
        .take(pad)
        .chain(in_shape.iter().copied())
        .collect();
    // Row-major strides of the *input* in its broadcast view (size-1 dim
    // has stride 0, otherwise the contiguous stride of in_shape).
    let mut in_strides = vec![0i64; rank];
    let mut s: i64 = 1;
    for d in (0..rank).rev() {
        in_strides[d] = if in_shape_padded[d] == 1 { 0 } else { s };
        s *= in_shape_padded[d] as i64;
    }
    let n_out = total_elements(&out_shape);
    let (_t, _) = emit_header(fusion, meta, w, 1, 1, "Broadcast (per-dim strides)");
    w.line(format!("constexpr uint N_OUT = {n_out};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    // Decompose i into out coords, then compute in offset via strides.
    let mut div_acc: u64 = 1;
    w.line("uint _rem = i;");
    w.line("ulong _in_off = 0;");
    for d in (0..rank).rev() {
        let dim = out_shape[d].max(1);
        // c = (_rem / div_acc) % dim
        let div = div_acc;
        let stride = in_strides[d];
        w.line(format!(
            "{{ uint _c = (_rem / {}u) % {}u; _in_off += (ulong)_c * {}ul; }}",
            div, dim, stride
        ));
        div_acc *= dim as u64;
    }
    w.line("out0[i] = in0[(uint)_in_off];");
    emit_close(w);
    Ok(())
}

fn emit_transpose(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // mlx Transpose permutes axes; storage is unchanged but the logical
    // index order differs. To materialize: for output index oi with
    // coords (c0..cR), input coords are (c_{inv_axes[0]} .. c_{inv_axes[R-1]}).
    // input offset = Σ_d in_coord[d] * in_stride[d].
    let in_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    let rank = in_shape.len();
    let axes: Vec<i32> = match op_attrs {
        crate::graph::Attrs::Transpose { axes } => axes.clone(),
        _ => (0..rank as i32).rev().collect(), // default: reverse
    };
    if axes.len() != rank || out_shape.len() != rank {
        return emit_view_passthrough(fusion, meta, w);
    }
    if rank == 4 && axes.as_slice() == [0, 2, 1, 3] {
        let d0 = out_shape[0].max(1) as u32;
        let d1 = out_shape[1].max(1) as u32;
        let d2 = out_shape[2].max(1) as u32;
        let d3 = out_shape[3].max(1) as u32;
        let n_out = total_elements(out_shape);
        let (_t, _) = emit_header(fusion, meta, w, 1, 1, "Transpose (4D swap 1/2)");
        w.line(format!("constexpr uint D0 = {d0};"));
        w.line(format!("constexpr uint D1 = {d1};"));
        w.line(format!("constexpr uint D2 = {d2};"));
        w.line(format!("constexpr uint D3 = {d3};"));
        w.line(format!("constexpr uint N_OUT = {n_out};"));
        w.line("uint i = gid * tg_size + tid;");
        w.line("if (i >= N_OUT) return;");
        w.line("uint c3 = i % D3;");
        w.line("uint q = i / D3;");
        w.line("uint c2 = q % D2;");
        w.line("q /= D2;");
        w.line("uint c1 = q % D1;");
        w.line("uint c0 = q / D1;");
        w.line("uint in_off = ((c0 * D2 + c2) * D1 + c1) * D3 + c3;");
        w.line("out0[i] = in0[in_off];");
        emit_close(w);
        return Ok(());
    }
    // out_shape[d] == in_shape[axes[d]]
    // out coords (c0..cR) → in coord[axes[d]] = c[d]
    // Compute input contiguous strides.
    let mut in_strides = vec![0i64; rank];
    let mut s: i64 = 1;
    for d in (0..rank).rev() {
        in_strides[d] = s;
        s *= in_shape[d] as i64;
    }
    let n_out = total_elements(out_shape);
    let (_t, _) = emit_header(fusion, meta, w, 1, 1, "Transpose");
    w.line(format!("constexpr uint N_OUT = {n_out};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint _rem = i;");
    w.line("ulong _in_off = 0;");
    let mut div_acc: u64 = 1;
    for d in (0..rank).rev() {
        let dim = out_shape[d].max(1) as u64;
        let in_dim = axes[d] as usize;
        let stride = in_strides[in_dim];
        w.line(format!(
            "{{ uint _c = (_rem / {}u) % {}u; _in_off += (ulong)_c * {}ul; }}",
            div_acc, dim, stride
        ));
        div_acc *= dim;
    }
    w.line("out0[i] = in0[(uint)_in_off];");
    emit_close(w);
    Ok(())
}

fn emit_transpose_split(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // General layout+split sink:
    //   y = transpose(x, axes); outs = split(y, axis)
    // materialized as one pass over the split outputs. This avoids the
    // full transposed temporary when the transpose has exactly one
    // consumer. It is shape/dataflow based; the axes come from the
    // captured Transpose op and the split axis is inferred from output
    // shapes.
    if fusion.inputs.len() != 1 || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "TransposeSplit (arity)");
    }
    let in_shape = &fusion.inputs[0].shape;
    let rank = in_shape.len();
    let axes: Vec<i32> = match op_attrs {
        crate::graph::Attrs::Transpose { axes } => axes.clone(),
        _ => (0..rank as i32).rev().collect(),
    };
    if rank == 0 || axes.len() != rank {
        return emit_todo_stub(fusion, meta, w, "TransposeSplit (rank)");
    }
    let mut seen = vec![false; rank];
    for &axis in &axes {
        if axis < 0 || axis as usize >= rank || seen[axis as usize] {
            return emit_todo_stub(fusion, meta, w, "TransposeSplit (axes)");
        }
        seen[axis as usize] = true;
    }
    let trans_shape = axes
        .iter()
        .map(|&axis| in_shape[axis as usize])
        .collect::<Vec<_>>();
    let Some(axis_idx) = infer_split_axis(&trans_shape, &fusion.outputs) else {
        return emit_todo_stub(fusion, meta, w, "TransposeSplit (split axis)");
    };

    let mut in_strides = vec![0i64; rank];
    let mut stride: i64 = 1;
    for d in (0..rank).rev() {
        in_strides[d] = stride;
        stride *= in_shape[d].max(1) as i64;
    }
    let outer: u32 = trans_shape[..axis_idx]
        .iter()
        .copied()
        .map(|x| x.max(1) as u32)
        .product();
    let inner: u32 = trans_shape[axis_idx + 1..]
        .iter()
        .copied()
        .map(|x| x.max(1) as u32)
        .product();
    let trans_axis = trans_shape[axis_idx].max(1) as u32;
    let out_elems: Vec<u32> = fusion
        .outputs
        .iter()
        .map(|b| total_elements(&b.shape))
        .collect();
    let total: u32 = out_elems.iter().sum();
    let (t, _) = emit_header(
        fusion,
        meta,
        w,
        1,
        fusion.outputs.len(),
        &format!("TransposeSplit(axis={axis_idx})"),
    );
    w.line(format!("constexpr uint OUTER = {outer}u;"));
    w.line(format!("constexpr uint INNER = {inner}u;"));
    w.line(format!("constexpr uint TRANS_AXIS = {trans_axis}u;"));
    w.line(format!("constexpr uint TOTAL = {total}u;"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= TOTAL) return;");
    let mut elem_base = 0u32;
    let mut axis_base = 0u32;
    for (idx, out) in fusion.outputs.iter().enumerate() {
        let axis_len = out.shape.get(axis_idx).copied().unwrap_or(1).max(1) as u32;
        let n = out_elems[idx];
        let cond = if idx == 0 {
            format!("if (i < {n}u) {{")
        } else if idx == fusion.outputs.len() - 1 {
            "else {".to_string()
        } else {
            format!("else if (i < {}u) {{", elem_base + n)
        };
        w.line(cond);
        w.push_indent();
        w.line(format!("uint local = i - {elem_base}u;"));
        w.line(format!("uint along_len = {axis_len}u;"));
        w.line("uint outer = local / (along_len * INNER);");
        w.line("uint rest = local - outer * (along_len * INNER);");
        w.line("uint along = rest / INNER;");
        w.line("uint inner = rest % INNER;");
        w.line(format!(
            "uint trans_i = (outer * TRANS_AXIS + ({axis_base}u + along)) * INNER + inner;"
        ));
        w.line("uint _rem = trans_i;");
        w.line("ulong _in_off = 0;");
        let mut div_acc: u64 = 1;
        for d in (0..rank).rev() {
            let dim = trans_shape[d].max(1) as u64;
            let in_dim = axes[d] as usize;
            let stride = in_strides[in_dim];
            w.line(format!(
                "{{ uint _c = (_rem / {}u) % {}u; _in_off += (ulong)_c * {}ul; }}",
                div_acc, dim, stride
            ));
            div_acc *= dim;
        }
        w.line(format!("out{idx}[local] = {t}(in0[(uint)_in_off]);"));
        w.pop_indent();
        w.line("}");
        elem_base += n;
        axis_base += axis_len;
    }
    emit_close(w);
    Ok(())
}

// ─── heavy ─────────────────────────────────────────────────────────────────

fn emit_matmul(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Generic 2D matmul: out = a @ b. Shapes from boundaries:
    //   a: (..., M, K)
    //   b: (..., K, N)
    //   out: (..., M, N)
    // mlx allows batched matmul; we handle the no-batch / single-batch
    // case (rank 2 or rank 3 with leading dim 1). Batched > 1 falls
    // back to a TODO copy. One thread per output element.
    if fusion.inputs.len() < 2 || fusion.outputs.is_empty() {
        bail!("Matmul: expected 2 inputs, 1 output");
    }
    let a_shape = &fusion.inputs[0].shape;
    let b_shape = &fusion.inputs[1].shape;
    let out_shape = &fusion.outputs[0].shape;
    if a_shape.len() < 2 || b_shape.len() < 2 || out_shape.len() < 2 {
        return emit_todo_stub(fusion, meta, w, "Matmul (rank<2)");
    }
    let m = a_shape[a_shape.len() - 2] as u32;
    let k = a_shape[a_shape.len() - 1] as u32;
    let k2 = b_shape[b_shape.len() - 2] as u32;
    let n = b_shape[b_shape.len() - 1] as u32;
    if k != k2 {
        return emit_todo_stub(fusion, meta, w, "Matmul (K mismatch)");
    }
    // Leading dims must multiply to 1 for the no-batch case.
    let a_batch: i32 = a_shape[..a_shape.len() - 2].iter().product();
    let b_batch: i32 = b_shape[..b_shape.len() - 2].iter().product();
    if a_batch.max(1) != 1 || b_batch.max(1) != 1 {
        return emit_todo_stub(fusion, meta, w, "Matmul (batched)");
    }
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "Matmul (2D)");
    w.line(format!("constexpr uint M = {m};"));
    w.line(format!("constexpr uint K = {k};"));
    w.line(format!("constexpr uint N_DIM = {n};"));
    // One thread per (row, col) of out. Linearize as i = row * N + col.
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= M * N_DIM) return;");
    w.line("uint row = i / N_DIM;");
    w.line("uint col = i % N_DIM;");
    w.line("float acc = 0.0f;");
    w.line("for (uint k = 0u; k < K; ++k) {");
    w.push_indent();
    w.line("acc += float(in0[row * K + k]) * float(in1[k * N_DIM + col]);");
    w.pop_indent();
    w.line("}");
    w.line(format!("out0[i] = {t}(acc);"));
    emit_close(w);
    Ok(())
}

fn emit_quantized_matmul(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if crate::codegen::kernels::qmm_prefill_nax_preferred(fusion)
        && emit_quantized_matmul_nax(fusion, op_attrs, meta, w)?
    {
        return Ok(());
    }
    if emit_quantized_matmul_mlx_qmmt(fusion, meta, w)? {
        return Ok(());
    }
    if emit_quantized_matmul_mlx_qmv_impl(fusion, meta, w)? {
        return Ok(());
    }
    if emit_quantized_matmul_mlx_qmv(fusion, meta, w)? {
        return Ok(());
    }
    if emit_quantized_matmul_nax(fusion, op_attrs, meta, w)? {
        return Ok(());
    }
    let output_shape = fusion
        .outputs
        .first()
        .map(|b| b.shape.clone())
        .unwrap_or_default();
    let step = ChainStep {
        op_name: "QuantizedMatmul".into(),
        op_attrs: op_attrs.clone(),
        inputs: (0..fusion.inputs.len() as u32)
            .map(|buffer_index| ChainOperand::External { buffer_index })
            .collect(),
        output_shape,
        output_dtype: fusion.outputs[0].dtype,
    };
    let fn_name = function_name(fusion);
    crate::codegen::kernels::fused_chain::emit_qmm_chain(
        fusion,
        &[step],
        meta,
        w,
        &fn_name,
        "QuantizedMatmul",
    )
}

fn emit_quantized_matmul_mlx_qmmt(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<bool> {
    let Some((k_dim, out_dim, active_rows, bits, group_size)) =
        crate::codegen::kernels::qmm_mlx_qmmt_shape(fusion)
    else {
        return Ok(false);
    };
    if fusion.inputs.len() < 4 || fusion.outputs.is_empty() {
        return Ok(false);
    }
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        return Ok(false);
    }
    let fn_name = function_name(fusion);
    let aligned_n = if out_dim.is_multiple_of(32) {
        "true"
    } else {
        "false"
    };
    let (bm, bk, bn) =
        crate::codegen::kernels::qmm_mlx_qmmt_tile(k_dim, out_dim, active_rows, bits);
    w.line(format!(
        "// Fusion #{}: Single(QuantizedMatmul MLX qmm_t M={} K={} N={} bits={} gs={} tile={}x{}x{}) — MLX qmm_t-compatible",
        fusion.id, active_rows, k_dim, out_dim, bits, group_size, bm, bk, bn
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line(format!("device {t}       *out0 [[buffer(4)]],"));
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
    w.line(format!(
        "threadgroup {t} Xs[{bm} * ({bk} + 16 / sizeof({t}))];"
    ));
    w.line(format!(
        "threadgroup {t} Ws[{bk} * ({bn} + 16 / sizeof({t}))];"
    ));
    w.writeln(format_args!(
        "fusekit_qmm_t_impl<{t}, {group_size}, {bits}, {aligned_n}, {bm}, {bk}, {bn}>("
    ));
    w.push_indent();
    w.line("in1, in2, in3, in0, out0, Xs, Ws, K, N, M, K, tgid, tid, simd_gid, simd_lid);");
    w.pop_indent();
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line("if (tid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }");
    w.line(
        "static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit MLX qmm_t requires MLX Metal headers\");",
    );
    w.pop_indent();
    w.line("#endif");
    emit_close(w);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn emit_quantized_matmul_mlx_qmv_argmax_pair(
    fusion: &Fusion,
    w: &mut MslWriter,
    t: &str,
    k_dim: u32,
    out_dim: u32,
    active_rows: u64,
    bits: u32,
    group_size: u32,
    qmv_sgs: u32,
    qmv_cols_per_sg: u32,
    token_rows: u32,
) {
    let cols_per_tg = qmv_sgs.saturating_mul(qmv_cols_per_sg).max(1);
    let n_out_groups = out_dim.div_ceil(cols_per_tg);
    let scan_name = qmv_argmax_function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(QuantizedMatmul MLX qmv_fast → argmax scratch) — direct greedy head",
        fusion.id
    ));
    w.line(format!("kernel void {scan_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line("device uint       *out0 [[buffer(4)]],");
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
    w.writeln(format_args!("constexpr int ACTIVE_ROWS = {active_rows};"));
    w.writeln(format_args!("constexpr int N_OUT_GROUPS = {n_out_groups};"));
    w.line("threadgroup float max_buf[4 * 32];");
    w.line("threadgroup uint idx_buf[4 * 32];");
    w.writeln(format_args!(
        "fusekit_qmv_fast_m_argmax_scan_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}, {token_rows}>(in1, in2, in3, in0, out0, K, N, ACTIVE_ROWS, N_OUT_GROUPS, max_buf, idx_buf, tgid, simd_gid, simd_lid);"
    ));
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line(
        "if (simd_lid == 0u && simd_gid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }",
    );
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit MLX qmv requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    emit_close(w);

    let finalize_name = qmv_argmax_finalize_function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(QuantizedMatmul MLX qmv_fast argmax finalize)",
        fusion.id
    ));
    w.line(format!("kernel void {finalize_name}("));
    w.push_indent();
    w.line("device const uint *in0 [[buffer(0)]],");
    w.line("device uint       *out0 [[buffer(1)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint row     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("threadgroup uint key_buf[32];");
    w.line("threadgroup uint idx_buf[32];");
    w.writeln(format_args!(
        "fusekit_qmv_argmax_finalize_impl<{n_out_groups}, {active_rows}>(in0, out0, key_buf, idx_buf, tid, tg_size, row);"
    ));
    w.pop_indent();
    w.line("}");
    w.blank();
}

pub fn qmv_argmax_function_name(fusion: &Fusion) -> String {
    format!("{}_argmax", function_name(fusion))
}

pub fn qmv_argmax_finalize_function_name(fusion: &Fusion) -> String {
    format!("{}_argmax_finalize", function_name(fusion))
}

fn emit_quantized_matmul_mlx_qmv_impl(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<bool> {
    let Some((k_dim, out_dim, _active_rows, bits, group_size)) =
        crate::codegen::kernels::qmm_mlx_qmv_shape(fusion)
    else {
        return Ok(false);
    };
    if fusion.inputs.len() < 4 || fusion.outputs.is_empty() {
        return Ok(false);
    }
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        return Ok(false);
    }
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(QuantizedMatmul MLX qmv) — MLX qmv-compatible",
        fusion.id
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line(format!("device {t}       *out0 [[buffer(4)]],"));
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
    let qmv_sgs = crate::codegen::kernels::qmv_exact_simdgroups();
    let qmv_cols_per_sg = crate::codegen::kernels::qmv_exact_cols_per_simdgroup();
    w.writeln(format_args!(
        "fusekit_qmv_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}>(in1, in2, in3, in0, out0, K, N, tgid, simd_gid, simd_lid);"
    ));
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line(
        "if (simd_lid == 0u && simd_gid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }",
    );
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit MLX qmv requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    emit_close(w);
    Ok(true)
}

fn emit_quantized_matmul_mlx_qmv(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<bool> {
    let Some((k_dim, out_dim, active_rows, bits, group_size)) =
        crate::codegen::kernels::qmm_mlx_qmv_fast_shape(fusion)
    else {
        return Ok(false);
    };
    if fusion.inputs.len() < 4 || fusion.outputs.is_empty() {
        return Ok(false);
    }
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        return Ok(false);
    }
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(QuantizedMatmul MLX qmv_fast) — MLX qmv-compatible",
        fusion.id
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line(format!("device {t}       *out0 [[buffer(4)]],"));
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
    if let Some((vecs_per_tg, k_lanes, _rows_per_tg)) =
        crate::codegen::kernels::qmv_wide_params(active_rows, bits, group_size, k_dim, out_dim)
    {
        w.writeln(format_args!("constexpr int ACTIVE_ROWS = {active_rows};"));
        w.writeln(format_args!(
            "fusekit_qmv_wide_impl<{t}, {group_size}, {bits}, {vecs_per_tg}, {k_lanes}>(in1, in2, in3, in0, out0, K, N, ACTIVE_ROWS, tgid, simd_gid, simd_lid);"
        ));
    } else if token_rows > 1 {
        w.writeln(format_args!("constexpr int ACTIVE_ROWS = {active_rows};"));
        w.writeln(format_args!(
            "fusekit_qmv_fast_m_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}, {token_rows}>(in1, in2, in3, in0, out0, K, N, ACTIVE_ROWS, tgid, simd_gid, simd_lid);"
        ));
    } else {
        w.writeln(format_args!(
            "fusekit_qmv_fast_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}>(in1, in2, in3, in0, out0, K, N, tgid, simd_gid, simd_lid);"
        ));
    }
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line(
        "if (simd_lid == 0u && simd_gid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }",
    );
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit MLX qmv requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    emit_close(w);
    emit_quantized_matmul_mlx_qmv_argmax_pair(
        fusion,
        w,
        t,
        k_dim,
        out_dim,
        active_rows,
        bits,
        group_size,
        qmv_sgs,
        qmv_cols_per_sg,
        token_rows,
    );
    Ok(true)
}

fn emit_quantized_matmul_nax(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<bool> {
    let Some((k_dim, out_dim, active_rows, bits, group_size)) =
        crate::codegen::kernels::qmm_nax_shape(fusion)
    else {
        return Ok(false);
    };
    let crate::graph::Attrs::Quantized { .. } = op_attrs else {
        return Ok(false);
    };
    if fusion.inputs.len() < 4 || fusion.outputs.is_empty() {
        return Ok(false);
    }
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        return Ok(false);
    }
    let fn_name = function_name(fusion);
    let aligned_n = if out_dim.is_multiple_of(64) {
        "true"
    } else {
        "false"
    };
    let (wm, wn) = if active_rows <= 16 { (1, 4) } else { (2, 2) };
    w.line(format!(
        "// Fusion #{}: Single(QuantizedMatmul NAX) — MLX qmm_t_nax-compatible",
        fusion.id
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line(format!("device {t}       *out0 [[buffer(4)]],"));
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
    w.line(format!("threadgroup {t} Ws[64 * (64 + 16 / sizeof({t}))];"));
    w.writeln(format_args!(
        "fusekit_qmm_t_nax_tgp_impl<{t}, {group_size}, {bits}, {aligned_n}, 64, 64, 64, {wm}, {wn}>("
    ));
    w.push_indent();
    w.line("in1, in2, in3, in0, out0, Ws, K, N, M, tgid, tid, simd_gid, simd_lid);");
    w.pop_indent();
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line("if (tid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }");
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit NAX qmm requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    emit_close(w);
    Ok(true)
}

fn emit_scaled_dot_product_attention_prefill(
    fusion: &Fusion,
    partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    let scale = match &fusion.attrs {
        FusionAttrs::Single {
            op_attrs: crate::graph::Attrs::ScaledDotProductAttention { scale },
            ..
        } => *scale,
        _ => (meta.head_dim as f32).powf(-0.5),
    };
    let out_shape = fusion
        .outputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    if out_shape.len() < 4 {
        return emit_todo_stub(fusion, meta, w, "ScaledDotProductAttention (rank)");
    }
    let out_seq = out_shape[out_shape.len() - 2].max(1) as u32;
    let active_m = crate::target::current_chip().active_m(out_seq).max(1);
    let n_heads = meta.num_attention_heads.max(1);
    let n_kv_heads = meta.num_key_value_heads.max(1);
    let head_dim = meta.head_dim.max(1);
    let heads_per_kv = (n_heads / n_kv_heads).max(1);
    let scale = if scale.abs() < f32::EPSILON {
        1.0f32 / (head_dim as f32).sqrt()
    } else {
        scale
    };
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if sdpa_single_decode_enabled(fusion, meta) {
        let decode = Fusion {
            id: fusion.id,
            kind: FusionKind::SdpaDecode,
            op_ids: fusion.op_ids.clone(),
            inputs: vec![],
            outputs: fusion.outputs.clone(),
            attrs: FusionAttrs::SdpaDecode {
                scale,
                trace_q: false,
                q_split: None,
                q_transpose_split: false,
                q1_rope: None,
                mask: AttentionMask::Causal,
            },
            parallel_axes: fusion.parallel_axes.clone(),
            layer_hint: fusion.layer_hint,
        };
        return crate::codegen::kernels::sdpa_decode::emit_with_function_name(
            &decode,
            partition,
            meta,
            w,
            function_name(fusion),
        );
    }
    let fn_name = function_name(fusion);
    let plan = sdpa_prefill_plan(fusion, meta);
    if matches!(plan, SdpaPrefillPlan::BlockNax) {
        w.line(format!(
            "// Fusion #{}: Single(ScaledDotProductAttention block-NAX prefill, active_m={}, n_heads={}, n_kv={}, heads_per_kv={}, head_dim={}, scale={})",
            fusion.id, active_m, n_heads, n_kv_heads, heads_per_kv, head_dim, scale
        ));
        w.line(format!("kernel void {fn_name}("));
        w.push_indent();
        w.line(format!("device const {t} *Q        [[buffer(0)]],"));
        w.line(format!("device const {t} *K        [[buffer(1)]],"));
        w.line(format!("device const {t} *V        [[buffer(2)]],"));
        w.line("constant uint       &kv_len   [[buffer(3)]],");
        w.line("constant uint       &capacity [[buffer(4)]],");
        w.line(format!("device {t}       *out      [[buffer(5)]],"));
        w.line("uint tid       [[thread_position_in_threadgroup]],");
        w.line("uint tg_size   [[threads_per_threadgroup]],");
        w.line("uint gid       [[threadgroup_position_in_grid]]");
        w.pop_indent();
        w.line(") {");
        w.push_indent();
        w.line("(void)tg_size;");
        w.line("uint simd_lane = tid & 31u;");
        w.line("uint simd_id = tid >> 5u;");
        w.line("#if FUSEKIT_HAVE_MLX_NAX");
        w.push_indent();
        w.writeln(format_args!(
            "fusekit_sdpa_prefill_nax_impl<{t}, {active_m}, {n_heads}, {n_kv_heads}, {head_dim}, {heads_per_kv}, 32, 32, 1, 2>(Q, K, V, kv_len, capacity, out, gid, simd_id, simd_lane, {scale:e}f);"
        ));
        w.pop_indent();
        w.line("#else");
        w.push_indent();
        w.line("(void)Q; (void)K; (void)V; (void)kv_len; (void)capacity; (void)out; (void)gid; (void)simd_id; (void)simd_lane;");
        w.pop_indent();
        w.line("#endif");
        emit_close(w);
        return Ok(());
    }
    if matches!(plan, SdpaPrefillPlan::FlashGqa) {
        let d_per_lane = ((head_dim as u32).saturating_add(31) / 32).max(1);
        let kv_tile = if head_dim <= 64 { 32u32 } else { 16u32 };
        let (q_heads_per_tg, q_rows_per_tg, q_head_blocks, flash_sgs) =
            sdpa_prefill_flash_tiling(meta);
        w.line(format!(
            "// Fusion #{}: Single(ScaledDotProductAttention GQA tiled flash prefill, active_m={}, n_heads={}, n_kv={}, heads_per_kv={}, q_rows_per_tg={}, q_heads_per_tg={}, head_dim={}, kv_tile={}, scale={})",
            fusion.id, active_m, n_heads, n_kv_heads, heads_per_kv, q_rows_per_tg, q_heads_per_tg, head_dim, kv_tile, scale
        ));
        w.line(format!("kernel void {fn_name}("));
        w.push_indent();
        w.line(format!("device const {t} *Q        [[buffer(0)]],"));
        w.line(format!("device const {t} *K        [[buffer(1)]],"));
        w.line(format!("device const {t} *V        [[buffer(2)]],"));
        w.line("constant uint       &kv_len   [[buffer(3)]],");
        w.line("constant uint       &capacity [[buffer(4)]],");
        w.line(format!("device {t}       *out      [[buffer(5)]],"));
        w.line("uint tid       [[thread_position_in_threadgroup]],");
        w.line("uint tg_size   [[threads_per_threadgroup]],");
        w.line("uint gid       [[threadgroup_position_in_grid]]");
        w.pop_indent();
        w.line(") {");
        w.push_indent();
        w.writeln(format_args!("constexpr uint OUT_SEQ        = {active_m}u;"));
        w.writeln(format_args!("constexpr uint OUT_SEQ_STRIDE = {out_seq}u;"));
        w.writeln(format_args!("constexpr uint N_HEADS        = {n_heads}u;"));
        w.writeln(format_args!(
            "constexpr uint N_KV_HEADS     = {n_kv_heads}u;"
        ));
        w.writeln(format_args!("constexpr uint HEAD_DIM       = {head_dim}u;"));
        w.writeln(format_args!(
            "constexpr uint HEADS_PER_KV   = {heads_per_kv}u;"
        ));
        w.writeln(format_args!(
            "constexpr uint Q_HEADS_PER_TG = {q_heads_per_tg}u;"
        ));
        w.writeln(format_args!(
            "constexpr uint Q_HEAD_BLOCKS  = {q_head_blocks}u;"
        ));
        w.writeln(format_args!(
            "constexpr uint Q_ROWS_PER_TG = {q_rows_per_tg}u;"
        ));
        w.writeln(format_args!("constexpr uint FLASH_SGS     = {flash_sgs}u;"));
        w.writeln(format_args!(
            "constexpr uint D_PER_LANE     = {d_per_lane}u;"
        ));
        w.writeln(format_args!("constexpr uint KV_TILE        = {kv_tile}u;"));
        w.writeln(format_args!("constexpr float SCALE         = {scale:e}f;"));
        w.line("constexpr uint MAX_KV_LEN     = 4096u;");
        w.line("threadgroup float k_tile[KV_TILE * HEAD_DIM];");
        w.line("threadgroup float v_tile[KV_TILE * HEAD_DIM];");
        w.line("uint simd_lane = tid & 31u;");
        w.line("uint simd_id = tid >> 5u;");
        w.line("uint row_block = gid / (N_KV_HEADS * Q_HEAD_BLOCKS);");
        w.line("uint rem = gid - row_block * (N_KV_HEADS * Q_HEAD_BLOCKS);");
        w.line("uint kv_head = rem / Q_HEAD_BLOCKS;");
        w.line("uint q_head_block = rem - kv_head * Q_HEAD_BLOCKS;");
        w.line("uint local_q_row = simd_id / Q_HEADS_PER_TG;");
        w.line("uint local_q_slot = simd_id - local_q_row * Q_HEADS_PER_TG;");
        w.line("uint row = row_block * Q_ROWS_PER_TG + local_q_row;");
        w.line("uint local_q_head = q_head_block * Q_HEADS_PER_TG + local_q_slot;");
        w.line("uint head_id = kv_head * HEADS_PER_KV + local_q_head;");
        w.line("bool active = simd_id < FLASH_SGS && row < OUT_SEQ && local_q_row < Q_ROWS_PER_TG && local_q_head < HEADS_PER_KV && head_id < N_HEADS;");
        w.line("uint kv_len_eff = min(kv_len + row, MAX_KV_LEN);");
        w.line("uint block_last_row = min(row_block * Q_ROWS_PER_TG + Q_ROWS_PER_TG - 1u, OUT_SEQ - 1u);");
        w.line("uint kv_len_block = min(kv_len + block_last_row, MAX_KV_LEN);");
        w.line("float q_vals[4];");
        w.line("float acc[4];");
        w.line("uint dims[4];");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("dims[i] = simd_lane + i * 32u;");
        w.line("float qv = 0.0f;");
        w.line("if (active && i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line(format!(
            "qv = float(Q[(row * N_HEADS + head_id) * HEAD_DIM + dims[i]]);"
        ));
        w.pop_indent();
        w.line("}");
        w.line("q_vals[i] = qv;");
        w.line("acc[i] = 0.0f;");
        w.pop_indent();
        w.line("}");
        w.line("float m = -INFINITY;");
        w.line("float l = 0.0f;");
        w.line("for (uint tile_start = 0u; tile_start < kv_len_block; tile_start += KV_TILE) {");
        w.push_indent();
        w.line("uint tile_count = min(KV_TILE, kv_len_block - tile_start);");
        w.line("uint tile_elems = tile_count * HEAD_DIM;");
        w.line("for (uint idx = tid; idx < tile_elems; idx += tg_size) {");
        w.push_indent();
        w.line("uint kv_local = idx / HEAD_DIM;");
        w.line("uint d = idx - kv_local * HEAD_DIM;");
        w.line("uint kv = tile_start + kv_local;");
        w.line(format!(
            "device const {t}* k_row = K + (kv_head * capacity + kv) * HEAD_DIM;"
        ));
        w.line(format!(
            "device const {t}* v_row = V + (kv_head * capacity + kv) * HEAD_DIM;"
        ));
        w.line("k_tile[idx] = float(k_row[d]);");
        w.line("v_tile[idx] = float(v_row[d]);");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (active) {");
        w.push_indent();
        w.line("for (uint kk = 0u; kk < tile_count; ++kk) {");
        w.push_indent();
        w.line("uint kv_cur = tile_start + kk;");
        w.line("if (kv_cur >= kv_len_eff) continue;");
        w.line("float dot_part = 0.0f;");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line("dot_part += q_vals[i] * k_tile[kk * HEAD_DIM + dims[i]];");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("float score = simd_sum(dot_part) * SCALE;");
        w.line("float alpha_src = 0.0f;");
        w.line("float beta_src = 0.0f;");
        w.line("float new_m_src = m;");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("new_m_src = max(m, score);");
        w.line("alpha_src = exp(m - new_m_src);");
        w.line("beta_src = exp(score - new_m_src);");
        w.pop_indent();
        w.line("}");
        w.line("float alpha = simd_broadcast(alpha_src, 0u);");
        w.line("float beta = simd_broadcast(beta_src, 0u);");
        w.line("m = simd_broadcast(new_m_src, 0u);");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line("acc[i] = acc[i] * alpha + beta * v_tile[kk * HEAD_DIM + dims[i]];");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("l = l * alpha + beta;");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.pop_indent();
        w.line("}");
        w.line("if (active) {");
        w.push_indent();
        w.line("float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line(format!(
            "out[(head_id * OUT_SEQ_STRIDE + row) * HEAD_DIM + dims[i]] = {t}(acc[i] * inv_l);"
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        emit_close(w);
        return Ok(());
    }
    if matches!(plan, SdpaPrefillPlan::OnlineVector) && head_dim <= 128 {
        let d_per_lane = ((head_dim as u32).saturating_add(31) / 32).max(1);
        w.line(format!(
            "// Fusion #{}: Single(ScaledDotProductAttention online prefill, active_m={}, n_heads={}, n_kv={}, head_dim={}, scale={})",
            fusion.id, active_m, n_heads, n_kv_heads, head_dim, scale
        ));
        w.line(format!("kernel void {fn_name}("));
        w.push_indent();
        w.line(format!("device const {t} *Q        [[buffer(0)]],"));
        w.line(format!("device const {t} *K        [[buffer(1)]],"));
        w.line(format!("device const {t} *V        [[buffer(2)]],"));
        w.line("constant uint       &kv_len   [[buffer(3)]],");
        w.line("constant uint       &capacity [[buffer(4)]],");
        w.line(format!("device {t}       *out      [[buffer(5)]],"));
        w.line("uint tid       [[thread_position_in_threadgroup]],");
        w.line("uint tg_size   [[threads_per_threadgroup]],");
        w.line("uint gid       [[threadgroup_position_in_grid]]");
        w.pop_indent();
        w.line(") {");
        w.push_indent();
        w.writeln(format_args!("constexpr uint OUT_SEQ      = {active_m}u;"));
        w.writeln(format_args!("constexpr uint OUT_SEQ_STRIDE = {out_seq}u;"));
        w.writeln(format_args!("constexpr uint N_HEADS      = {n_heads}u;"));
        w.writeln(format_args!("constexpr uint N_KV_HEADS   = {n_kv_heads}u;"));
        w.writeln(format_args!("constexpr uint HEAD_DIM     = {head_dim}u;"));
        w.writeln(format_args!(
            "constexpr uint HEADS_PER_KV = {heads_per_kv}u;"
        ));
        w.writeln(format_args!("constexpr uint D_PER_LANE   = {d_per_lane}u;"));
        w.writeln(format_args!("constexpr float SCALE       = {scale:e}f;"));
        w.line("constexpr uint MAX_KV_LEN   = 4096u;");
        w.line("constexpr uint SDPA_SGS     = 32u;");
        w.line("constexpr uint SDPA_BD      = 32u;");
        w.line("threadgroup float partial_out[SDPA_SGS * SDPA_BD];");
        w.line("threadgroup float partial_max[SDPA_SGS];");
        w.line("threadgroup float partial_sum[SDPA_SGS];");
        w.line("uint simd_lane = tid & 31u;");
        w.line("uint simd_id = tid >> 5u;");
        w.line("if (simd_id >= SDPA_SGS) return;");
        w.line("uint query_head = gid;");
        w.line("uint query_row = query_head / N_HEADS;");
        w.line("uint head_id = query_head - query_row * N_HEADS;");
        w.line("if (query_row >= OUT_SEQ) return;");
        w.line("uint kv_head = head_id / HEADS_PER_KV;");
        w.line("uint kv_len_eff = min(kv_len + query_row, MAX_KV_LEN);");
        w.line(format!(
            "device const {t}* q_row = Q + (query_row * N_HEADS + head_id) * HEAD_DIM;"
        ));
        w.line("float q_vals[4];");
        w.line("float acc[4];");
        w.line("uint dims[4];");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("dims[i] = simd_lane * D_PER_LANE + i;");
        w.line(
            "q_vals[i] = (i < D_PER_LANE && dims[i] < HEAD_DIM) ? float(q_row[dims[i]]) : 0.0f;",
        );
        w.line("acc[i] = 0.0f;");
        w.pop_indent();
        w.line("}");
        w.line("float m = -INFINITY;");
        w.line("float l = 0.0f;");
        w.line("for (uint kv = simd_id; kv < kv_len_eff; kv += SDPA_SGS) {");
        w.push_indent();
        w.line(format!(
            "device const {t}* k_row = K + (kv_head * capacity + kv) * HEAD_DIM;"
        ));
        w.line(format!(
            "device const {t}* v_row = V + (kv_head * capacity + kv) * HEAD_DIM;"
        ));
        w.line("float dot_part = 0.0f;");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line("dot_part += q_vals[i] * float(k_row[dims[i]]);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("float score = simd_sum(dot_part) * SCALE;");
        w.line("float new_m = max(m, score);");
        w.line("float alpha = fast::exp(m - new_m);");
        w.line("float beta = fast::exp(score - new_m);");
        w.line("m = new_m;");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line("acc[i] = acc[i] * alpha + beta * float(v_row[dims[i]]);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("l = l * alpha + beta;");
        w.pop_indent();
        w.line("}");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("partial_max[simd_id] = m;");
        w.line("partial_sum[simd_id] = l;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("float lane_m = partial_max[simd_lane];");
        w.line("float global_m = simd_max(lane_m);");
        w.line("float merge_factor = (lane_m == -INFINITY || global_m == -INFINITY) ? 0.0f : fast::exp(lane_m - global_m);");
        w.line("float total_l = simd_sum(partial_sum[simd_lane] * merge_factor);");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE) {");
        w.push_indent();
        w.line(
            "partial_out[simd_lane * SDPA_BD + simd_id] = (dims[i] < HEAD_DIM) ? acc[i] : 0.0f;",
        );
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line(
            "float merged = simd_sum(partial_out[simd_id * SDPA_BD + simd_lane] * merge_factor);",
        );
        w.line("float normed = (total_l > 0.0f) ? (merged / total_l) : 0.0f;");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("uint out_d = simd_id * D_PER_LANE + i;");
        w.line("if (out_d < HEAD_DIM) {");
        w.push_indent();
        w.line(format!(
            "out[(head_id * OUT_SEQ_STRIDE + query_row) * HEAD_DIM + out_d] = {t}(normed);"
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        emit_close(w);
        return Ok(());
    }
    if matches!(plan, SdpaPrefillPlan::Online) && head_dim <= 128 {
        let d_per_lane = ((head_dim as u32).saturating_add(31) / 32).max(1);
        w.line(format!(
            "// Fusion #{}: Single(ScaledDotProductAttention online prefill, active_m={}, n_heads={}, n_kv={}, head_dim={}, scale={})",
            fusion.id, active_m, n_heads, n_kv_heads, head_dim, scale
        ));
        w.line(format!("kernel void {fn_name}("));
        w.push_indent();
        w.line(format!("device const {t} *Q        [[buffer(0)]],"));
        w.line(format!("device const {t} *K        [[buffer(1)]],"));
        w.line(format!("device const {t} *V        [[buffer(2)]],"));
        w.line("constant uint       &kv_len   [[buffer(3)]],");
        w.line("constant uint       &capacity [[buffer(4)]],");
        w.line(format!("device {t}       *out      [[buffer(5)]],"));
        w.line("uint tid       [[thread_position_in_threadgroup]],");
        w.line("uint tg_size   [[threads_per_threadgroup]],");
        w.line("uint gid       [[threadgroup_position_in_grid]]");
        w.pop_indent();
        w.line(") {");
        w.push_indent();
        w.writeln(format_args!("constexpr uint OUT_SEQ      = {active_m}u;"));
        w.writeln(format_args!("constexpr uint OUT_SEQ_STRIDE = {out_seq}u;"));
        w.writeln(format_args!("constexpr uint N_HEADS      = {n_heads}u;"));
        w.writeln(format_args!("constexpr uint N_KV_HEADS   = {n_kv_heads}u;"));
        w.writeln(format_args!("constexpr uint HEAD_DIM     = {head_dim}u;"));
        w.writeln(format_args!(
            "constexpr uint HEADS_PER_KV = {heads_per_kv}u;"
        ));
        w.writeln(format_args!("constexpr uint D_PER_LANE   = {d_per_lane}u;"));
        w.writeln(format_args!("constexpr float SCALE       = {scale:e}f;"));
        w.line("constexpr uint MAX_KV_LEN   = 4096u;");
        w.line("uint simd_lane = tid & 31u;");
        w.line("uint simd_id = tid >> 5u;");
        w.line("if (simd_id != 0u) return;");
        w.line("uint query_head = gid;");
        w.line("uint query_row = query_head / N_HEADS;");
        w.line("uint head_id = query_head - query_row * N_HEADS;");
        w.line("if (query_row >= OUT_SEQ) return;");
        w.line("uint kv_head = head_id / HEADS_PER_KV;");
        w.line("uint kv_len_eff = min(kv_len + query_row, MAX_KV_LEN);");
        w.line(format!(
            "device const {t}* q_row = Q + (query_row * N_HEADS + head_id) * HEAD_DIM;"
        ));
        w.line("float q_vals[4];");
        w.line("float acc[4];");
        w.line("uint dims[4];");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("dims[i] = simd_lane + i * 32u;");
        w.line(
            "q_vals[i] = (i < D_PER_LANE && dims[i] < HEAD_DIM) ? float(q_row[dims[i]]) : 0.0f;",
        );
        w.line("acc[i] = 0.0f;");
        w.pop_indent();
        w.line("}");
        w.line("float m = -INFINITY;");
        w.line("float l = 0.0f;");
        w.line("for (uint kv = 0u; kv < kv_len_eff; ++kv) {");
        w.push_indent();
        w.line(format!(
            "device const {t}* k_row = K + (kv_head * capacity + kv) * HEAD_DIM;"
        ));
        w.line(format!(
            "device const {t}* v_row = V + (kv_head * capacity + kv) * HEAD_DIM;"
        ));
        w.line("float dot_part = 0.0f;");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line("dot_part += q_vals[i] * float(k_row[dims[i]]);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("float score = simd_sum(dot_part) * SCALE;");
        w.line("float alpha_src = 0.0f;");
        w.line("float beta_src = 0.0f;");
        w.line("float new_m_src = m;");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("new_m_src = max(m, score);");
        w.line("alpha_src = exp(m - new_m_src);");
        w.line("beta_src = exp(score - new_m_src);");
        w.pop_indent();
        w.line("}");
        w.line("float alpha = simd_broadcast(alpha_src, 0u);");
        w.line("float beta = simd_broadcast(beta_src, 0u);");
        w.line("m = simd_broadcast(new_m_src, 0u);");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line("acc[i] = acc[i] * alpha + beta * float(v_row[dims[i]]);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("l = l * alpha + beta;");
        w.pop_indent();
        w.line("}");
        w.line("float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;");
        w.line("for (uint i = 0u; i < 4u; ++i) {");
        w.push_indent();
        w.line("if (i < D_PER_LANE && dims[i] < HEAD_DIM) {");
        w.push_indent();
        w.line(format!(
            "out[(head_id * OUT_SEQ_STRIDE + query_row) * HEAD_DIM + dims[i]] = {t}(acc[i] * inv_l);"
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        emit_close(w);
        return Ok(());
    }
    w.line(format!(
        "// Fusion #{}: Single(ScaledDotProductAttention materialized-score prefill, active_m={}, n_heads={}, n_kv={}, head_dim={}, scale={})",
        fusion.id, active_m, n_heads, n_kv_heads, head_dim, scale
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *Q        [[buffer(0)]],"));
    w.line(format!("device const {t} *K        [[buffer(1)]],"));
    w.line(format!("device const {t} *V        [[buffer(2)]],"));
    w.line("constant uint       &kv_len   [[buffer(3)]],");
    w.line("constant uint       &capacity [[buffer(4)]],");
    w.line(format!("device {t}       *out      [[buffer(5)]],"));
    w.line("uint tid       [[thread_position_in_threadgroup]],");
    w.line("uint tg_size   [[threads_per_threadgroup]],");
    w.line("uint gid       [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.writeln(format_args!("constexpr uint OUT_SEQ      = {active_m}u;"));
    w.writeln(format_args!("constexpr uint OUT_SEQ_STRIDE = {out_seq}u;"));
    w.writeln(format_args!("constexpr uint N_HEADS      = {n_heads}u;"));
    w.writeln(format_args!("constexpr uint N_KV_HEADS   = {n_kv_heads}u;"));
    w.writeln(format_args!("constexpr uint HEAD_DIM     = {head_dim}u;"));
    w.writeln(format_args!(
        "constexpr uint HEADS_PER_KV = {heads_per_kv}u;"
    ));
    w.writeln(format_args!("constexpr float SCALE       = {scale:e}f;"));
    w.line("constexpr uint MAX_KV_LEN   = 4096u;");
    w.line("uint query_head = gid;");
    w.line("uint query_row = query_head / N_HEADS;");
    w.line("uint head_id = query_head - query_row * N_HEADS;");
    w.line("if (query_row >= OUT_SEQ) return;");
    w.line("uint kv_head = head_id / HEADS_PER_KV;");
    w.line("uint kv_len_eff = kv_len + query_row;");
    w.line("kv_len_eff = min(kv_len_eff, MAX_KV_LEN);");
    w.line("threadgroup float scores[4096];");
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id = tid / 32u;");
    w.blank();
    w.line("// Stage 1: scores[kv] = Q[row, head] dot K[kv_head, kv].");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line(format!(
        "device const {t}* q_row = Q + (query_row * N_HEADS + head_id) * HEAD_DIM;"
    ));
    w.line(format!(
        "device const {t}* k_row = K + (kv_head * capacity + kv) * HEAD_DIM;"
    ));
    w.line("float dot = 0.0f;");
    w.line("for (uint d = 0u; d < HEAD_DIM; ++d) {");
    w.push_indent();
    w.line("dot += float(q_row[d]) * float(k_row[d]);");
    w.pop_indent();
    w.line("}");
    w.line("scores[kv] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();
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
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_max = reduce_buf[0];");
    w.line("float t_sum = 0.0f;");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line("scores[kv] = exp(scores[kv] - global_max);");
    w.line("t_sum += scores[kv];");
    w.pop_indent();
    w.line("}");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float inv_sum = 1.0f / reduce_buf[0];");
    w.blank();
    w.line("// Output layout matches mlx SDPA: [B, H, M, D].");
    w.line("for (uint d = tid; d < HEAD_DIM; d += tg_size) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    w.line("for (uint kv = 0u; kv < kv_len_eff; ++kv) {");
    w.push_indent();
    w.line(format!(
        "device const {t}* v_row = V + (kv_head * capacity + kv) * HEAD_DIM;"
    ));
    w.line("acc += (scores[kv] * inv_sum) * float(v_row[d]);");
    w.pop_indent();
    w.line("}");
    w.line(format!(
        "out[(head_id * OUT_SEQ_STRIDE + query_row) * HEAD_DIM + d] = {t}(acc);"
    ));
    w.pop_indent();
    w.line("}");
    emit_close(w);
    Ok(())
}

fn emit_softmax(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // Numerically stable softmax along the LAST axis. Implementation
    // uses one threadgroup per row; within the TG, simdgroup-cooperative
    // max + sum reductions over the last dim.
    let axis = match op_attrs {
        crate::graph::Attrs::Softmax { axis } => *axis,
        _ => -1,
    };
    let shape = &fusion.outputs[0].shape;
    let last = shape.last().copied().unwrap_or(1) as u32;
    let rows = crate::target::current_chip()
        .active_rows_before_last(shape)
        .min(u32::MAX as u64) as u32;
    let rows = rows.max(1);
    // Only last-axis softmax is supported for now.
    if !(axis == -1 || axis as usize == shape.len() - 1) {
        return emit_todo_stub(fusion, meta, w, "Softmax (non-last axis)");
    }
    let (t, _) = emit_header(fusion, meta, w, 1, 1, "Softmax (last axis)");
    w.line(format!("constexpr uint ROW_N = {last};"));
    w.line(format!("constexpr uint ROWS  = {rows};"));
    w.line("uint row = gid;");
    w.line("if (row >= ROWS) return;");
    w.line("threadgroup float reduce_buf[32];");
    // max
    w.line("float local_max = -INFINITY;");
    w.line("for (uint i = tid; i < ROW_N; i += tg_size) {");
    w.push_indent();
    w.line("local_max = max(local_max, float(in0[row * ROW_N + i]));");
    w.pop_indent();
    w.line("}");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("local_max = simd_max(local_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = local_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float row_max = reduce_buf[0];");
    // sum of exp(x - max)
    w.line("float local_sum = 0.0f;");
    w.line("for (uint i = tid; i < ROW_N; i += tg_size) {");
    w.push_indent();
    w.line("local_sum += exp(float(in0[row * ROW_N + i]) - row_max);");
    w.pop_indent();
    w.line("}");
    w.line("local_sum = simd_sum(local_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = local_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float inv_sum = 1.0f / reduce_buf[0];");
    // out = exp(x - max) / sum
    w.line("for (uint i = tid; i < ROW_N; i += tg_size) {");
    w.push_indent();
    w.line(format!(
        "out0[row * ROW_N + i] = {t}(exp(float(in0[row * ROW_N + i]) - row_max) * inv_sum);"
    ));
    w.pop_indent();
    w.line("}");
    emit_close(w);
    Ok(())
}

fn emit_arange(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    let (start, _stop, step) = match op_attrs {
        crate::graph::Attrs::Arange { start, stop, step } => (*start, *stop, *step),
        _ => (0.0f64, 0.0f64, 1.0f64),
    };
    let n = total_elements(&fusion.outputs[0].shape);
    let (t, _) = emit_header(fusion, meta, w, 0, 1, "Arange");
    w.line(format!("constexpr uint N = {n};"));
    // f32 literals need a decimal — `0f` is parsed as an octal prefix.
    w.line(format!("constexpr float START = {start:?}f;"));
    w.line(format!("constexpr float STEP  = {step:?}f;"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line(format!("out0[i] = {t}(START + STEP * float(i));"));
    emit_close(w);
    Ok(())
}

fn emit_full(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // mlx Full takes a scalar value (passed as a 0-d input tensor in
    // the trace) and writes it to every element of the output. We
    // assume one input (the scalar) plus one output. If no input,
    // emit zeros — Full's `value` argument lives in op.arguments
    // which we don't currently extract into Attrs (followup).
    let n = total_elements(&fusion.outputs[0].shape);
    let n_in = if fusion.inputs.is_empty() { 0 } else { 1 };
    let (t, _) = emit_header(fusion, meta, w, n_in, 1, "Full");
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    if n_in == 1 {
        w.line("out0[i] = in0[0];");
    } else {
        w.line(format!("out0[i] = {t}(0);"));
    }
    emit_close(w);
    Ok(())
}

fn emit_slice(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // General contiguous slice. mlx records start/stop/stride as op
    // arguments; map each output linear index back to a row-major input
    // offset. This covers last-axis top-k slices and middle-axis decode
    // slices such as Gemma4 shared-KV selection without model-specific
    // shape checks.
    if fusion.inputs.is_empty() || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "Slice (missing input/output)");
    }
    let in_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    if in_shape.len() != out_shape.len() || in_shape.is_empty() {
        return emit_todo_stub(fusion, meta, w, "Slice (rank change)");
    }
    if let crate::graph::Attrs::Slice { start, stride, .. } = op_attrs {
        if start.len() == in_shape.len() && stride.len() == in_shape.len() {
            return emit_slice_strided(fusion, meta, w, start, stride);
        }
    }
    if in_shape[..in_shape.len() - 1] != out_shape[..out_shape.len() - 1] {
        return emit_todo_stub(fusion, meta, w, "Slice (non-last-axis)");
    }
    let in_last = in_shape.last().copied().unwrap_or(1).max(1) as u32;
    let out_last = out_shape.last().copied().unwrap_or(1).max(1) as u32;
    if out_last > in_last {
        return emit_todo_stub(fusion, meta, w, "Slice (expanded)");
    }
    let n = total_elements(out_shape);
    let (_t, _) = emit_header(fusion, meta, w, 1, 1, "Slice (last-axis prefix)");
    w.line(format!("constexpr uint IN_LAST  = {in_last};"));
    w.line(format!("constexpr uint OUT_LAST = {out_last};"));
    w.line(format!("constexpr uint N_OUT    = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint outer = i / OUT_LAST;");
    w.line("uint col = i % OUT_LAST;");
    w.line("out0[i] = in0[outer * IN_LAST + col];");
    emit_close(w);
    Ok(())
}

fn emit_slice_strided(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    start: &[i32],
    stride: &[i32],
) -> Result<()> {
    let in_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    let rank = out_shape.len();
    let mut in_strides = vec![1i64; rank];
    for d in (0..rank).rev() {
        if d + 1 < rank {
            in_strides[d] = in_strides[d + 1] * i64::from(in_shape[d + 1].max(1));
        }
    }

    let mut starts = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    for d in 0..rank {
        let dim = in_shape[d].max(1);
        let mut s = start[d];
        if s < 0 {
            s += dim;
        }
        starts.push(s.max(0) as u32);
        strides.push(stride[d].max(1) as u32);
    }

    // A stride-1 full-shape slice is a straight copy; keep the generic
    // path for clarity but reject obvious out-of-range expansions.
    for d in 0..rank {
        let last =
            starts[d] as u64 + (out_shape[d].max(1) as u64).saturating_sub(1) * strides[d] as u64;
        if last >= in_shape[d].max(1) as u64 {
            return emit_todo_stub(fusion, meta, w, "Slice (out of range)");
        }
    }
    if strides.iter().all(|&s| s == 1) {
        let changed_axes = (0..rank)
            .filter(|&d| starts[d] != 0 || out_shape[d].max(1) != in_shape[d].max(1))
            .collect::<Vec<_>>();
        if changed_axes.len() == 1 {
            let axis = changed_axes[0];
            let inner: u64 = out_shape[axis + 1..]
                .iter()
                .map(|&d| d.max(1) as u64)
                .product();
            let outer: u64 = out_shape[..axis].iter().map(|&d| d.max(1) as u64).product();
            let out_axis = out_shape[axis].max(1) as u64;
            let in_axis = in_shape[axis].max(1) as u64;
            let start_axis = starts[axis] as u64;
            let block = out_axis.saturating_mul(inner).max(1);
            let n = total_elements(out_shape);
            let (_t, _) = emit_header(fusion, meta, w, 1, 1, "Slice (single-axis block)");
            w.line(format!("constexpr uint N_OUT = {n};"));
            w.line(format!("constexpr uint OUTER = {outer};"));
            w.line(format!("constexpr uint BLOCK = {block};"));
            w.line(format!("constexpr uint INNER = {inner};"));
            w.line(format!("constexpr uint IN_AXIS = {in_axis};"));
            w.line(format!("constexpr uint START_AXIS = {start_axis};"));
            w.line("uint i = gid * tg_size + tid;");
            w.line("if (i >= N_OUT) return;");
            w.line("uint outer = i / BLOCK;");
            w.line("uint inner = i - outer * BLOCK;");
            w.line("uint in_off = outer * IN_AXIS * INNER + START_AXIS * INNER + inner;");
            w.line("out0[i] = in0[in_off];");
            w.line("(void)OUTER;");
            emit_close(w);
            return Ok(());
        }
    }

    let n = total_elements(out_shape);
    let (_t, _) = emit_header(fusion, meta, w, 1, 1, "Slice (rank-N strided)");
    w.line(format!("constexpr uint N_OUT = {n};"));
    for d in 0..rank {
        w.line(format!(
            "constexpr uint OUT_D{d} = {};",
            out_shape[d].max(1)
        ));
        w.line(format!("constexpr uint START_D{d} = {};", starts[d]));
        w.line(format!("constexpr uint STEP_D{d} = {};", strides[d]));
        w.line(format!(
            "constexpr uint IN_STRIDE_D{d} = {};",
            in_strides[d].max(1)
        ));
    }
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint rem = i;");
    w.line("uint in_off = 0u;");
    for d in (0..rank).rev() {
        w.line(format!("uint c{d} = rem % OUT_D{d};"));
        if d > 0 {
            w.line(format!("rem /= OUT_D{d};"));
        }
        w.line(format!(
            "in_off += (START_D{d} + c{d} * STEP_D{d}) * IN_STRIDE_D{d};"
        ));
    }
    w.line("out0[i] = in0[in_off];");
    emit_close(w);
    Ok(())
}

fn emit_argpartition(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Deterministic full argsort along the last axis. mlx ArgPartition
    // only requires the first K indices downstream here; producing a
    // sorted partition is semantically stronger for the MoE routing use.
    if fusion.inputs.is_empty() || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "ArgPartition (missing input/output)");
    }
    let shape = &fusion.inputs[0].shape;
    if shape.is_empty() || fusion.outputs[0].shape != *shape {
        return emit_todo_stub(fusion, meta, w, "ArgPartition (shape mismatch)");
    }
    let row_n = shape.last().copied().unwrap_or(1).max(1) as u32;
    let n = total_elements(shape);
    let active_rows = if shape.len() >= 3 {
        crate::target::current_chip()
            .active_rows_before_last(shape)
            .min(u32::MAX as u64) as u32
    } else {
        (n / row_n).max(1)
    };
    let active_n = active_rows.saturating_mul(row_n).max(1);
    let bitonic_enabled = std::env::var("FUSEKIT_ARGPARTITION_BITONIC").as_deref() != Ok("0");
    if bitonic_enabled && row_n.is_power_of_two() && row_n <= 256 {
        let (_t, _) = emit_header(
            fusion,
            meta,
            w,
            1,
            1,
            "ArgPartition (last-axis bitonic argsort)",
        );
        w.line(format!("constexpr uint ROW_N = {row_n};"));
        w.line(format!("constexpr uint ROWS  = {active_rows};"));
        w.line("uint row = gid;");
        w.line("if (row >= ROWS) return;");
        w.line("threadgroup float vals[256];");
        w.line("threadgroup uint idxs[256];");
        w.line("if (tid < ROW_N) {");
        w.push_indent();
        w.line("uint col = tid;");
        w.line("vals[col] = float(in0[row * ROW_N + col]);");
        w.line("idxs[col] = col;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("for (uint k = 2u; k <= ROW_N; k <<= 1u) {");
        w.push_indent();
        w.line("for (uint j = k >> 1u; j > 0u; j >>= 1u) {");
        w.push_indent();
        w.line("uint other = tid ^ j;");
        w.line("if (tid < ROW_N && other > tid && other < ROW_N) {");
        w.push_indent();
        w.line("float a = vals[tid];");
        w.line("float b = vals[other];");
        w.line("uint ai = idxs[tid];");
        w.line("uint bi = idxs[other];");
        w.line("bool ascending = ((tid & k) == 0u);");
        w.line("bool a_after_b = (a > b) || (a == b && ai > bi);");
        w.line("bool do_swap = ascending ? a_after_b : !a_after_b;");
        w.line("if (do_swap) {");
        w.push_indent();
        w.line("vals[tid] = b;");
        w.line("vals[other] = a;");
        w.line("idxs[tid] = bi;");
        w.line("idxs[other] = ai;");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("if (tid < ROW_N) {");
        w.push_indent();
        w.line("out0[row * ROW_N + tid] = idxs[tid];");
        w.pop_indent();
        w.line("}");
        emit_close(w);
        return Ok(());
    }

    let (_t, _) = emit_header(fusion, meta, w, 1, 1, "ArgPartition (last-axis argsort)");
    w.line(format!("constexpr uint ROW_N = {row_n};"));
    w.line(format!("constexpr uint N = {active_n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line("uint row = i / ROW_N;");
    w.line("uint rank = i % ROW_N;");
    w.line("uint best_idx = 0u;");
    w.line("for (uint cand = 0u; cand < ROW_N; ++cand) {");
    w.push_indent();
    w.line("float v = float(in0[row * ROW_N + cand]);");
    w.line("uint r = 0u;");
    w.line("for (uint j = 0u; j < ROW_N; ++j) {");
    w.push_indent();
    w.line("float u = float(in0[row * ROW_N + j]);");
    w.line("if (u < v || (u == v && j < cand)) { r += 1u; }");
    w.pop_indent();
    w.line("}");
    w.line("if (r == rank) { best_idx = cand; break; }");
    w.pop_indent();
    w.line("}");
    w.line("out0[i] = best_idx;");
    emit_close(w);
    Ok(())
}

fn emit_argtopk(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    descending: bool,
) -> Result<()> {
    if fusion.inputs.is_empty() || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "ArgTopK (missing input/output)");
    }
    let in_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    if in_shape.is_empty()
        || out_shape.is_empty()
        || in_shape.len() != out_shape.len()
        || in_shape[..in_shape.len() - 1] != out_shape[..out_shape.len() - 1]
    {
        return emit_todo_stub(fusion, meta, w, "ArgTopK (shape)");
    }
    let row_n = in_shape.last().copied().unwrap_or(1).max(1) as u32;
    let top_k = out_shape.last().copied().unwrap_or(1).max(1) as u32;
    if top_k >= row_n || row_n > 256 || top_k > 32 {
        return emit_todo_stub(fusion, meta, w, "ArgTopK (bounds)");
    }
    let rows = crate::target::current_chip()
        .active_rows_before_last(in_shape)
        .min(u32::MAX as u64) as u32;
    let (_t, _) = emit_header(
        fusion,
        meta,
        w,
        1,
        1,
        if descending {
            "ArgTopK (last-axis descending)"
        } else {
            "ArgTopK (last-axis ascending)"
        },
    );
    w.line(format!("constexpr uint ROW_N = {row_n};"));
    w.line(format!("constexpr uint TOP_K = {top_k};"));
    w.line(format!("constexpr uint ROWS  = {};", rows.max(1)));
    w.line("uint row = gid;");
    w.line("if (row >= ROWS) return;");
    w.line("threadgroup float vals[256];");

    w.line("threadgroup uint idxs[256];");
    w.line("if (tid < ROW_N) {");
    w.push_indent();
    w.line("uint col = tid;");
    if descending {
        // Sort `-score` ascending, which is equivalent to sorting the
        // original score descending. This preserves the same tie rule
        // as Negative -> ArgPartition: lower index wins equal scores.
        w.line("vals[col] = -float(in0[row * ROW_N + col]);");
    } else {
        w.line("vals[col] = float(in0[row * ROW_N + col]);");
    }
    w.line("idxs[col] = col;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("for (uint k = 2u; k <= ROW_N; k <<= 1u) {");
    w.push_indent();
    w.line("for (uint j = k >> 1u; j > 0u; j >>= 1u) {");
    w.push_indent();
    w.line("uint other = tid ^ j;");
    w.line("if (tid < ROW_N && other > tid && other < ROW_N) {");
    w.push_indent();
    w.line("float a = vals[tid];");
    w.line("float b = vals[other];");
    w.line("uint ai = idxs[tid];");
    w.line("uint bi = idxs[other];");
    w.line("bool ascending = ((tid & k) == 0u);");
    w.line("bool a_after_b = (a > b) || (a == b && ai > bi);");
    w.line("bool do_swap = ascending ? a_after_b : !a_after_b;");
    w.line("if (do_swap) {");
    w.push_indent();
    w.line("vals[tid] = b;");
    w.line("vals[other] = a;");
    w.line("idxs[tid] = bi;");
    w.line("idxs[other] = ai;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("if (tid < TOP_K) {");
    w.push_indent();
    w.line("out0[row * TOP_K + tid] = idxs[tid];");
    w.pop_indent();
    w.line("}");
    emit_close(w);
    Ok(())
}

fn emit_gather_axis(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Gather along the last axis: data [outer,N], idx [outer,K] -> [outer,K].
    if fusion.inputs.len() < 2 || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "GatherAxis (missing input/output)");
    }
    let data_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    if data_shape.is_empty() || out_shape.is_empty() {
        return emit_todo_stub(fusion, meta, w, "GatherAxis (bad rank)");
    }
    if data_shape[..data_shape.len() - 1] != out_shape[..out_shape.len() - 1] {
        return emit_todo_stub(fusion, meta, w, "GatherAxis (non-last-axis)");
    }
    let in_last = data_shape.last().copied().unwrap_or(1).max(1) as u32;
    let out_last = out_shape.last().copied().unwrap_or(1).max(1) as u32;
    let n = total_elements(out_shape);
    let (_t, _) = emit_header(fusion, meta, w, 2, 1, "GatherAxis (last axis)");
    w.line(format!("constexpr uint IN_LAST  = {in_last};"));
    w.line(format!("constexpr uint OUT_LAST = {out_last};"));
    w.line(format!("constexpr uint N_OUT    = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint outer = i / OUT_LAST;");
    w.line("uint idx = uint(in1[i]);");
    w.line("idx = min(idx, IN_LAST - 1u);");
    w.line("out0[i] = in0[outer * IN_LAST + idx];");
    emit_close(w);
    Ok(())
}

fn emit_gather_axis_scale_expand(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if fusion.inputs.len() < 3 || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "GatherAxisScaleExpand (arity)");
    }
    let data_shape = &fusion.inputs[0].shape;
    let idx_shape = &fusion.inputs[1].shape;
    let scale_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if data_shape.is_empty()
        || idx_shape.is_empty()
        || out_shape.len() != idx_shape.len() + 1
        || out_shape.last().copied().unwrap_or(0) != 1
        || out_shape[..out_shape.len() - 1] != idx_shape[..]
        || data_shape[..data_shape.len() - 1] != idx_shape[..idx_shape.len() - 1]
    {
        return emit_todo_stub(fusion, meta, w, "GatherAxisScaleExpand (shape)");
    }
    if !(scale_shape.is_empty() || scale_shape == idx_shape) {
        return emit_todo_stub(fusion, meta, w, "GatherAxisScaleExpand (scale)");
    }
    let in_last = data_shape.last().copied().unwrap_or(1).max(1) as u32;
    let top_k = idx_shape.last().copied().unwrap_or(1).max(1) as u32;
    let n = total_elements(out_shape);
    let (t, _) = emit_header(fusion, meta, w, 3, 1, "GatherAxisScaleExpand");
    w.line(format!("constexpr uint IN_LAST = {in_last};"));
    w.line(format!("constexpr uint TOP_K = {top_k};"));
    w.line(format!("constexpr uint N_OUT = {n};"));
    w.line(format!(
        "constexpr bool SCALE_SCALAR = {};",
        if scale_shape.is_empty() {
            "true"
        } else {
            "false"
        }
    ));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint gather_i = i; // last output dim is 1");
    w.line("uint outer = gather_i / TOP_K;");
    w.line("uint idx = uint(in1[gather_i]);");
    w.line("idx = min(idx, IN_LAST - 1u);");
    w.line("float scale = SCALE_SCALAR ? float(in2[0]) : float(in2[gather_i]);");
    w.line(format!(
        "out0[i] = {t}(float(in0[outer * IN_LAST + idx]) * scale);"
    ));
    emit_close(w);
    Ok(())
}

fn emit_softmax_topk_route(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    if fusion.inputs.len() < 2 || fusion.outputs.len() != 2 {
        return emit_todo_stub(fusion, meta, w, "SoftmaxTopKRoute (arity)");
    }
    let logits_shape = &fusion.inputs[0].shape;
    let scale_shape = &fusion.inputs[1].shape;
    let idx_shape = &fusion.outputs[0].shape;
    let weight_shape = &fusion.outputs[1].shape;
    if logits_shape.is_empty()
        || idx_shape.is_empty()
        || logits_shape.len() != idx_shape.len()
        || weight_shape.len() != idx_shape.len() + 1
        || weight_shape.last().copied().unwrap_or(0) != 1
        || logits_shape[..logits_shape.len() - 1] != idx_shape[..idx_shape.len() - 1]
        || weight_shape[..weight_shape.len() - 1] != idx_shape[..]
        || fusion.outputs[0].dtype != Dtype::U32
    {
        return emit_todo_stub(fusion, meta, w, "SoftmaxTopKRoute (shape)");
    }
    if !(scale_shape.is_empty() || scale_shape == idx_shape) {
        return emit_todo_stub(fusion, meta, w, "SoftmaxTopKRoute (scale)");
    }
    let row_n = logits_shape.last().copied().unwrap_or(1).max(1) as u32;
    let top_k = idx_shape.last().copied().unwrap_or(1).max(1) as u32;
    if top_k >= row_n || row_n > 256 || top_k > 32 || !row_n.is_power_of_two() {
        return emit_todo_stub(fusion, meta, w, "SoftmaxTopKRoute (bounds)");
    }
    let rows = crate::target::current_chip()
        .active_rows_before_last(logits_shape)
        .min(u32::MAX as u64) as u32;
    let fallback = crate::codegen::kernels::act_ty(meta);
    let weight_t = fusion
        .outputs
        .get(1)
        .map(|bd| msl_ty(bd.dtype, fallback))
        .unwrap_or(fallback);
    let (_out_t, _) = emit_header(fusion, meta, w, 2, 2, "SoftmaxTopKRoute");
    w.line(format!("constexpr uint ROW_N = {row_n};"));
    w.line(format!("constexpr uint TOP_K = {top_k};"));
    w.line(format!("constexpr uint ROWS = {};", rows.max(1)));
    w.line(format!(
        "constexpr bool SCALE_SCALAR = {};",
        if scale_shape.is_empty() {
            "true"
        } else {
            "false"
        }
    ));
    w.line("uint row = gid;");
    w.line("if (row >= ROWS) return;");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id = tid / 32u;");
    w.line("threadgroup float vals[256];");
    w.line("threadgroup uint idxs[256];");
    w.line("threadgroup float reduce_buf[32];");

    w.line("float local_max = -INFINITY;");
    w.line("float local_sum = 0.0f;");
    w.line("if (tid < ROW_N) {");
    w.push_indent();
    w.line("float score = float(in0[row * ROW_N + tid]);");
    w.line("local_max = score;");
    w.pop_indent();
    w.line("}");
    w.line("local_max = simd_max(local_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = local_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float row_max = reduce_buf[0];");
    w.line("for (uint i = tid; i < ROW_N; i += tg_size) {");
    w.push_indent();
    w.line("local_sum += exp(float(in0[row * ROW_N + i]) - row_max);");
    w.pop_indent();
    w.line("}");
    w.line("local_sum = simd_sum(local_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = local_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float inv_sum = 1.0f / reduce_buf[0];");
    w.line("if (tid < ROW_N) {");
    w.push_indent();
    w.line("float score = float(in0[row * ROW_N + tid]);");
    w.line(format!(
        "float prob = float({weight_t}(exp(score - row_max) * inv_sum));"
    ));
    w.line("vals[tid] = -prob;");
    w.line("idxs[tid] = tid;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");

    w.line("for (uint k = 2u; k <= ROW_N; k <<= 1u) {");
    w.push_indent();
    w.line("for (uint j = k >> 1u; j > 0u; j >>= 1u) {");
    w.push_indent();
    w.line("uint other = tid ^ j;");
    w.line("if (tid < ROW_N && other > tid && other < ROW_N) {");
    w.push_indent();
    w.line("float a = vals[tid];");
    w.line("float b = vals[other];");
    w.line("uint ai = idxs[tid];");
    w.line("uint bi = idxs[other];");
    w.line("bool ascending = ((tid & k) == 0u);");
    w.line("bool a_after_b = (a > b) || (a == b && ai > bi);");
    w.line("bool do_swap = ascending ? a_after_b : !a_after_b;");
    w.line("if (do_swap) {");
    w.push_indent();
    w.line("vals[tid] = b;");
    w.line("vals[other] = a;");
    w.line("idxs[tid] = bi;");
    w.line("idxs[other] = ai;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");

    w.line("if (tid < TOP_K) {");
    w.push_indent();
    w.line("uint out_i = row * TOP_K + tid;");
    w.line("uint idx = idxs[tid];");
    w.line("out0[out_i] = idx;");
    w.line("float prob = -vals[tid];");
    w.line("float scale = SCALE_SCALAR ? float(in1[0]) : float(in1[out_i]);");
    w.line(format!("out1[out_i] = {weight_t}(prob * scale);"));
    w.pop_indent();
    w.line("}");
    emit_close(w);
    Ok(())
}

fn emit_gather_qmm_mlx_qmv(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<bool> {
    let Some((k_dim, out_dim, active_routes, bits, group_size)) =
        crate::codegen::kernels::gather_qmm_mlx_qmv_shape(fusion)
    else {
        return Ok(false);
    };
    if fusion.inputs.len() < 6 || fusion.outputs.is_empty() {
        return Ok(false);
    }
    let x_shape = &fusion.inputs[0].shape;
    let w_shape = &fusion.inputs[1].shape;
    let s_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if x_shape.is_empty() || w_shape.len() != 3 || s_shape.len() != 3 || out_shape.len() < 5 {
        return Ok(false);
    }
    let n_experts = w_shape[0].max(1) as u32;
    let packed = w_shape[2].max(1) as u32;
    let n_groups = s_shape[2].max(1) as u32;
    let tokens = out_shape.get(1).copied().unwrap_or(1).max(1) as u32;
    let active_tokens = crate::target::current_chip().active_tokens_for_shape(out_shape);
    let top_k = out_shape.get(2).copied().unwrap_or(1).max(1) as u32;
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        return Ok(false);
    }
    let qmv_sgs = crate::codegen::kernels::qmv_fast_simdgroups(1, bits, k_dim, out_dim);
    let qmv_cols_per_sg =
        crate::codegen::kernels::qmv_fast_cols_per_simdgroup(1, bits, k_dim, out_dim);
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(GatherQMM MLX qmv_fast) — GatherQMM",
        fusion.id
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line("device const uint *in4 [[buffer(4)]],");
    w.line("device const uint *in5 [[buffer(5)]],");
    w.line(format!("device {t}       *out0 [[buffer(6)]],"));
    w.line("uint3 tgid [[threadgroup_position_in_grid]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("#if FUSEKIT_HAVE_MLX_NAX");
    w.push_indent();
    w.line(format!("constexpr int K = {k_dim};"));
    w.line(format!("constexpr int O = {out_dim};"));
    w.line(format!("constexpr uint N_EXPERTS = {n_experts}u;"));
    w.line(format!("constexpr uint PACKED = {packed}u;"));
    w.line(format!("constexpr uint N_GROUPS = {n_groups}u;"));
    w.line(format!("constexpr uint TOKENS = {tokens}u;"));
    w.line(format!("constexpr uint ACTIVE_TOKENS = {active_tokens}u;"));
    w.line(format!("constexpr uint TOP_K = {top_k}u;"));
    w.line(format!("constexpr uint ACTIVE_ROUTES = {active_routes}u;"));
    w.line("uint route = tgid.y;");
    w.line("if (route >= ACTIVE_ROUTES) return;");
    w.line("uint slot = route % TOP_K;");
    w.line("uint tok = (route / TOP_K) % TOKENS;");
    w.line("if (tok >= ACTIVE_TOKENS) return;");
    w.line("uint route_idx = tok * TOP_K + slot;");
    w.line("uint x_row = uint(in4[route_idx]);");
    w.line("uint expert = min(uint(in5[route_idx]), N_EXPERTS - 1u);");
    w.line("uint3 local_tid = uint3(0u, tgid.x, 0u);");
    w.writeln(format_args!(
        "fusekit_qmv_fast_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}>(in1 + expert * O * PACKED, in2 + expert * O * N_GROUPS, in3 + expert * O * N_GROUPS, in0 + x_row * K, out0 + route * O, K, O, local_tid, simd_gid, simd_lid);"
    ));
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line(
        "if (simd_lid == 0u && simd_gid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }",
    );
    w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit GatherQMM qmv requires MLX Metal headers\");");
    w.pop_indent();
    w.line("#endif");
    emit_close(w);
    Ok(true)
}

fn emit_gather_qmm(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Quantized expert matvec for MoE routing.
    // Inputs: x, packed_w[E,O,K/lanes], scales[E,O,K/group],
    // biases[E,O,K/group], row_idx[B,T,top_k], expert_idx[B,T,top_k].
    if emit_gather_qmm_mlx_qmv(fusion, meta, w)? {
        return Ok(());
    }
    if emit_gather_qmm_nax(fusion, meta, w)? {
        return Ok(());
    }
    if fusion.inputs.len() < 6 || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "GatherQMM (missing inputs)");
    }
    let has_output_scale = fusion
        .inputs
        .get(6)
        .is_some_and(|b| b.role == "output_scale");
    let x_shape = &fusion.inputs[0].shape;
    let w_shape = &fusion.inputs[1].shape;
    let s_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if x_shape.is_empty() || w_shape.len() != 3 || s_shape.len() != 3 || out_shape.len() < 4 {
        return emit_todo_stub(fusion, meta, w, "GatherQMM (unsupported shape)");
    }
    let k_dim = x_shape.last().copied().unwrap_or(1).max(1) as u32;
    let n_experts = w_shape[0].max(1) as u32;
    let out_dim = w_shape[1].max(1) as u32;
    let packed = w_shape[2].max(1) as u32;
    let n_groups = s_shape[2].max(1) as u32;
    let bits = ((packed * 32) / k_dim).max(1);
    let group_size = (k_dim / n_groups).max(1);
    if bits != 4 && bits != 8 {
        return emit_todo_stub(fusion, meta, w, "GatherQMM (bits)");
    }
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
    let (t, _) = emit_header(
        fusion,
        meta,
        w,
        if has_output_scale { 7 } else { 6 },
        1,
        if has_output_scale {
            "GatherQMM output-scale"
        } else {
            "GatherQMM"
        },
    );
    w.line(format!("constexpr uint K = {k_dim};"));
    w.line(format!("constexpr uint N_EXPERTS = {n_experts};"));
    w.line(format!("constexpr uint O = {out_dim};"));
    w.line(format!("constexpr uint PACKED = {packed};"));
    w.line(format!("constexpr uint N_GROUPS = {n_groups};"));
    w.line(format!("constexpr uint TOKENS = {tokens};"));
    w.line(format!("constexpr uint ACTIVE_TOKENS = {active_tokens};"));
    w.line(format!("constexpr uint TOP_K = {top_k};"));
    w.line(format!("constexpr uint ACTIVE_ROUTES = {active_routes};"));
    let chip = crate::target::current_chip();
    let chip_rows = chip.gatherqmm_rows_per_simdgroup();
    let rows_per_sg = if chip_rows > 1 && out_dim >= chip_rows && out_dim % chip_rows == 0 {
        chip_rows
    } else {
        1
    };
    let use_x_smem = chip.use_gatherqmm_x_smem(k_dim, out_dim, rows_per_sg, active_routes);
    w.line(format!("constexpr uint ROWS_PER_SG = {rows_per_sg}u;"));
    w.line("constexpr uint ROW_BLOCKS = (O + ROWS_PER_SG - 1u) / ROWS_PER_SG;");
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    let planned_sgs = chip.gatherqmm_simdgroups_per_tg().max(1);
    let row_blocks = out_dim.saturating_add(rows_per_sg).saturating_sub(1) / rows_per_sg.max(1);
    let x_smem_always = use_x_smem && row_blocks.is_multiple_of(planned_sgs);
    if x_smem_always {
        w.line("threadgroup float x_smem[K];");
        w.line(format!(
            "constexpr uint STAGED_SGS_PER_TG = {planned_sgs}u;"
        ));
        w.line("constexpr uint STAGED_TG_SIZE = STAGED_SGS_PER_TG * 32u;");
        w.line("uint sg = gid * STAGED_SGS_PER_TG + simd_id;");
        w.line("if (sg >= ACTIVE_ROUTES * ROW_BLOCKS) return;");
        w.line("uint route = sg / ROW_BLOCKS;");
        w.line("uint o_base = (sg % ROW_BLOCKS) * ROWS_PER_SG;");
        w.line("uint slot = route % TOP_K;");
        w.line("uint tok = (route / TOP_K) % TOKENS;");
        w.line("uint route_idx = tok * TOP_K + slot;");
        w.line("uint x_row_src = (simd_lane == 0u) ? uint(in4[route_idx]) : 0u;");
        w.line(
            "uint expert_src = (simd_lane == 0u) ? min(uint(in5[route_idx]), N_EXPERTS - 1u) : 0u;",
        );
        w.line("uint x_row = simd_broadcast(x_row_src, 0u);");
        w.line("uint expert = simd_broadcast(expert_src, 0u);");
        if has_output_scale {
            w.line("float output_scale_src = (simd_lane == 0u) ? float(in6[route_idx]) : 0.0f;");
            w.line("float output_scale = simd_broadcast(output_scale_src, 0u);");
        }
        w.line("for (uint k = tid; k < K; k += STAGED_TG_SIZE) {");
        w.push_indent();
        w.line("x_smem[k] = float(in0[x_row * K + k]);");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        for r in 0..rows_per_sg {
            w.line(format!("float acc{r} = 0.0f;"));
        }
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("acc{r}"),
            |r| format!("in1 + (expert * O + (o_base + {r}u)) * PACKED"),
            |r| format!("in2 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            |r| format!("in3 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            "K",
            bits,
            group_size,
            rows_per_sg,
            |off| format!("x_smem[{off}]"),
        );
        for r in 0..rows_per_sg {
            w.line(format!("acc{r} = simd_sum(acc{r});"));
        }
        w.line("if (simd_lane != 0u) return;");
        for r in 0..rows_per_sg {
            if has_output_scale {
                w.line(format!(
                    "out0[route * O + o_base + {r}u] = {t}(float({t}(acc{r})) * output_scale);"
                ));
            } else {
                w.line(format!("out0[route * O + o_base + {r}u] = {t}(acc{r});"));
            }
        }
        emit_close(w);
        return Ok(());
    }
    w.line("uint sgs_per_tg = tg_size / 32u;");
    if use_x_smem {
        w.line("threadgroup float x_smem[K];");
        w.line("uint sg0 = gid * sgs_per_tg;");
        w.line("uint sg_last = sg0 + sgs_per_tg - 1u;");
        w.line("bool x_smem_ok = sg_last < ACTIVE_ROUTES * ROW_BLOCKS && (sg0 / ROW_BLOCKS) == (sg_last / ROW_BLOCKS);");
        w.line("if (x_smem_ok) {");
        w.push_indent();
        w.line("uint sg = sg0 + simd_id;");
        w.line("uint route = sg0 / ROW_BLOCKS;");
        w.line("uint o_base = (sg % ROW_BLOCKS) * ROWS_PER_SG;");
        w.line("uint slot = route % TOP_K;");
        w.line("uint tok = (route / TOP_K) % TOKENS;");
        w.line("uint route_idx = tok * TOP_K + slot;");
        w.line("uint x_row = uint(in4[route_idx]);");
        w.line("uint expert = min(uint(in5[route_idx]), N_EXPERTS - 1u);");
        if has_output_scale {
            w.line("float output_scale = float(in6[route_idx]);");
        }
        w.line("for (uint k = tid; k < K; k += tg_size) {");
        w.push_indent();
        w.line("x_smem[k] = float(in0[x_row * K + k]);");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (tok >= ACTIVE_TOKENS) return;");
        for r in 0..rows_per_sg {
            w.line(format!("float acc{r} = 0.0f;"));
        }
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("acc{r}"),
            |r| format!("in1 + (expert * O + (o_base + {r}u)) * PACKED"),
            |r| format!("in2 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            |r| format!("in3 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            "K",
            bits,
            group_size,
            rows_per_sg,
            |off| format!("x_smem[{off}]"),
        );
        for r in 0..rows_per_sg {
            w.line(format!("acc{r} = simd_sum(acc{r});"));
        }
        w.line("if (simd_lane != 0u) return;");
        for r in 0..rows_per_sg {
            if has_output_scale {
                w.line(format!(
                    "out0[route * O + o_base + {r}u] = {t}(float({t}(acc{r})) * output_scale);"
                ));
            } else {
                w.line(format!("out0[route * O + o_base + {r}u] = {t}(acc{r});"));
            }
        }
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
    }
    w.line("uint sg = gid * sgs_per_tg + simd_id;");
    w.line("if (sg >= ACTIVE_ROUTES * ROW_BLOCKS) return;");
    w.line("uint route = sg / ROW_BLOCKS;");
    w.line("uint o_base = (sg % ROW_BLOCKS) * ROWS_PER_SG;");
    w.line("uint slot = route % TOP_K;");
    w.line("uint tok = (route / TOP_K) % TOKENS;");
    w.line("if (tok >= ACTIVE_TOKENS) return;");
    w.line("uint route_idx = tok * TOP_K + slot;");
    w.line("uint x_row = uint(in4[route_idx]);");
    w.line("uint expert = min(uint(in5[route_idx]), N_EXPERTS - 1u);");
    if has_output_scale {
        w.line("float output_scale = float(in6[route_idx]);");
    }
    for r in 0..rows_per_sg {
        w.line(format!("float acc{r} = 0.0f;"));
    }
    crate::codegen::kernels::emit_int4_gemv_inner_multirow(
        w,
        |r| format!("acc{r}"),
        |r| format!("in1 + (expert * O + (o_base + {r}u)) * PACKED"),
        |r| format!("in2 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
        |r| format!("in3 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
        "K",
        bits,
        group_size,
        rows_per_sg,
        |off| format!("float(in0[x_row * K + {off}])"),
    );
    for r in 0..rows_per_sg {
        w.line(format!("acc{r} = simd_sum(acc{r});"));
    }
    w.line("if (simd_lane != 0u) return;");
    for r in 0..rows_per_sg {
        if has_output_scale {
            w.line(format!(
                "out0[route * O + o_base + {r}u] = {t}(float({t}(acc{r})) * output_scale);"
            ));
        } else {
            w.line(format!("out0[route * O + o_base + {r}u] = {t}(acc{r});"));
        }
    }
    if use_x_smem {
        w.pop_indent();
        w.line("}");
    }
    emit_close(w);
    Ok(())
}

fn emit_gather_qmm_reduce(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Fuses the generic op chain:
    //   GatherQMM(route rows, expert rows) -> optional output scale -> Reduce(sum over routes)
    // into one route-parallel threadgroup. This keeps one simdgroup per
    // route so TOP_K parallelism is preserved, and only the route partials
    // round-trip through threadgroup memory.
    let has_route_weight = fusion
        .inputs
        .get(6)
        .is_some_and(|b| b.role == "route_weight");
    let base_inputs = if has_route_weight { 7 } else { 6 };
    let has_residual_linear = fusion
        .inputs
        .get(base_inputs)
        .is_some_and(|b| b.role == "shared_x");
    let has_extra_residual = has_residual_linear
        && fusion
            .inputs
            .get(base_inputs + 4)
            .is_some_and(|b| b.role == "residual_extra");
    let expected_inputs = base_inputs
        + if has_residual_linear {
            4 + usize::from(has_extra_residual)
        } else {
            0
        };
    if fusion.inputs.len() != expected_inputs || fusion.outputs.len() != 1 {
        return emit_todo_stub(fusion, meta, w, "GatherQMMReduce (inputs)");
    }
    let x_shape = &fusion.inputs[0].shape;
    let w_shape = &fusion.inputs[1].shape;
    let s_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if x_shape.len() < 3 || w_shape.len() != 3 || s_shape.len() != 3 || out_shape.len() < 3 {
        return emit_todo_stub(fusion, meta, w, "GatherQMMReduce (shape)");
    }
    let k_dim = x_shape.last().copied().unwrap_or(1).max(1) as u32;
    let n_experts = w_shape[0].max(1) as u32;
    let out_dim = w_shape[1].max(1) as u32;
    let packed = w_shape[2].max(1) as u32;
    let n_groups = s_shape[2].max(1) as u32;
    let bits = ((packed * 32) / k_dim).max(1);
    let group_size = (k_dim / n_groups).max(1);
    if !matches!(bits, 4 | 8) || group_size == 0 || out_dim == 0 {
        return emit_todo_stub(fusion, meta, w, "GatherQMMReduce (quant)");
    }
    let top_k = x_shape[2..x_shape.len().saturating_sub(1)]
        .iter()
        .copied()
        .filter(|&d| d > 0)
        .product::<i32>()
        .max(1) as u32;
    if top_k == 0 || top_k > 8 {
        return emit_todo_stub(fusion, meta, w, "GatherQMMReduce (top_k)");
    }
    let tokens = out_shape.get(1).copied().unwrap_or(1).max(1) as u32;
    let active_tokens = crate::target::current_chip().active_tokens_for_shape(out_shape);
    let out_cols =
        crate::target::current_chip().routed_reduce_cols_per_simdgroup(bits, k_dim, out_dim, top_k);
    let (out_t, _) = emit_header(
        fusion,
        meta,
        w,
        expected_inputs,
        1,
        if has_route_weight {
            "GatherQMMReduce output-scale"
        } else {
            "GatherQMMReduce"
        },
    );
    let t = out_t.as_str();
    if has_residual_linear {
        let shared_base = base_inputs;
        let shared_x_shape = fusion.inputs[shared_base].shape.as_slice();
        let shared_w_shape = fusion.inputs[shared_base + 1].shape.as_slice();
        let shared_s_shape = fusion.inputs[shared_base + 2].shape.as_slice();
        let shared_b_shape = fusion.inputs[shared_base + 3].shape.as_slice();
        if shared_x_shape.is_empty()
            || shared_w_shape.len() != 2
            || shared_s_shape.len() != 2
            || shared_b_shape.len() != 2
        {
            return emit_todo_stub(fusion, meta, w, "GatherQMMReduce+LinearResidual (shape)");
        }
        let shared_k = shared_x_shape.last().copied().unwrap_or(1).max(1) as u32;
        let shared_out = shared_w_shape[0].max(1) as u32;
        let shared_packed = shared_w_shape[1].max(1) as u32;
        let shared_groups = shared_s_shape[1].max(1) as u32;
        let shared_bits = ((shared_packed * 32) / shared_k).max(1);
        let shared_group_size = (shared_k / shared_groups.max(1)).max(1);
        if shared_out != out_dim
            || !matches!(shared_bits, 4 | 8)
            || shared_group_size != 64
            || shared_packed.saturating_mul(32) != shared_k.saturating_mul(shared_bits)
            || shared_s_shape[0].max(1) as u32 != out_dim
            || shared_b_shape[0].max(1) as u32 != out_dim
            || shared_b_shape[1].max(1) as u32 != shared_groups
            || top_k + 1 > 8
        {
            return emit_todo_stub(fusion, meta, w, "GatherQMMReduce+LinearResidual (target)");
        }
        let mut combined_cols = std::env::var("FUSEKIT_GATHER_QMM_REDUCE_LINEAR_COLS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| (1..=4).contains(&n))
            .unwrap_or(1)
            .min(out_dim.max(1));
        while combined_cols > 1 && !out_dim.is_multiple_of(combined_cols) {
            combined_cols -= 1;
        }
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
        w.line("if (simd_id < TOP_K) {");
        w.push_indent();
        w.line("uint slot = simd_id;");
        w.line("uint route_idx = tok * TOP_K + slot;");
        w.line("uint x_row_src = (simd_lane == 0u) ? uint(in4[route_idx]) : 0u;");
        w.line(
            "uint expert_src = (simd_lane == 0u) ? min(uint(in5[route_idx]), N_EXPERTS - 1u) : 0u;",
        );
        w.line("uint x_row = simd_broadcast(x_row_src, 0u);");
        w.line("uint expert = simd_broadcast(expert_src, 0u);");
        if has_route_weight {
            w.line("float rw_src = (simd_lane == 0u) ? float(in6[route_idx]) : 0.0f;");
            w.line("float rw = simd_broadcast(rw_src, 0u);");
        }
        for r in 0..combined_cols {
            w.line(format!("float dot{r} = 0.0f;"));
        }
        if combined_cols == 1 && bits == 4 && group_size == 64 && k_dim >= 1024 {
            crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
                w,
                "dot0",
                "in1 + (expert * O + o_base) * ROUTE_PACKED",
                "in2 + (expert * O + o_base) * ROUTE_GROUPS",
                "in3 + (expert * O + o_base) * ROUTE_GROUPS",
                "ROUTE_K",
                bits,
                group_size,
                false,
                false,
                crate::codegen::kernels::QdotPolicy::Force,
                |off| format!("float(in0[x_row * ROUTE_K + {off}])"),
            );
        } else if combined_cols == 1 {
            crate::codegen::kernels::emit_int4_gemv_inner(
                w,
                "dot0",
                "in1 + (expert * O + o_base) * ROUTE_PACKED",
                "in2 + (expert * O + o_base) * ROUTE_GROUPS",
                "in3 + (expert * O + o_base) * ROUTE_GROUPS",
                "ROUTE_K",
                bits,
                group_size,
                |off| format!("float(in0[x_row * ROUTE_K + {off}])"),
            );
        } else {
            crate::codegen::kernels::emit_int4_gemv_inner_multirow(
                w,
                |r| format!("dot{r}"),
                |r| format!("in1 + (expert * O + (o_base + {r}u)) * ROUTE_PACKED"),
                |r| format!("in2 + (expert * O + (o_base + {r}u)) * ROUTE_GROUPS"),
                |r| format!("in3 + (expert * O + (o_base + {r}u)) * ROUTE_GROUPS"),
                "ROUTE_K",
                bits,
                group_size,
                combined_cols,
                |off| format!("float(in0[x_row * ROUTE_K + {off}])"),
            );
        }
        for r in 0..combined_cols {
            w.line(format!("dot{r} = simd_sum(dot{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..combined_cols {
            if has_route_weight {
                w.line(format!(
                    "partial[slot * COLS + {r}u] = float({t}(float({t}(dot{r})) * rw));"
                ));
            } else {
                w.line(format!("partial[slot * COLS + {r}u] = float({t}(dot{r}));"));
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
        if combined_cols == 1 && shared_bits == 4 && shared_group_size == 64 && shared_k >= 1024 {
            crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
                w,
                "dot0",
                &format!("in{} + o_base * SHARED_PACKED", shared_base + 1),
                &format!("in{} + o_base * SHARED_GROUPS", shared_base + 2),
                &format!("in{} + o_base * SHARED_GROUPS", shared_base + 3),
                "SHARED_K",
                shared_bits,
                shared_group_size,
                false,
                false,
                crate::codegen::kernels::QdotPolicy::Force,
                |off| format!("float(in{shared_base}[tok * SHARED_K + {off}])"),
            );
        } else if combined_cols == 1 {
            crate::codegen::kernels::emit_int4_gemv_inner(
                w,
                "dot0",
                &format!("in{} + o_base * SHARED_PACKED", shared_base + 1),
                &format!("in{} + o_base * SHARED_GROUPS", shared_base + 2),
                &format!("in{} + o_base * SHARED_GROUPS", shared_base + 3),
                "SHARED_K",
                shared_bits,
                shared_group_size,
                |off| format!("float(in{shared_base}[tok * SHARED_K + {off}])"),
            );
        } else {
            crate::codegen::kernels::emit_int4_gemv_inner_multirow(
                w,
                |r| format!("dot{r}"),
                |r| format!("in{} + (o_base + {r}u) * SHARED_PACKED", shared_base + 1),
                |r| format!("in{} + (o_base + {r}u) * SHARED_GROUPS", shared_base + 2),
                |r| format!("in{} + (o_base + {r}u) * SHARED_GROUPS", shared_base + 3),
                "SHARED_K",
                shared_bits,
                shared_group_size,
                combined_cols,
                |off| format!("float(in{shared_base}[tok * SHARED_K + {off}])"),
            );
        }
        for r in 0..combined_cols {
            w.line(format!("dot{r} = simd_sum(dot{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        for r in 0..combined_cols {
            w.line(format!("partial[TOP_K * COLS + {r}u] = dot{r};"));
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
            w.line("float route_acc = 0.0f;");
            w.line(format!(
                "for (uint p = 0u; p < TOP_K; ++p) route_acc += partial[p * COLS + {r}u];"
            ));
            w.line(format!("float route_y = float({out_t}(route_acc));"));
            w.line(format!(
                "{out_t} y0 = {out_t}(partial[TOP_K * COLS + {r}u] + route_y);"
            ));
            if has_extra_residual {
                w.line(format!(
                    "out0[tok * O + o_base + {r}u] = {out_t}(float(y0) + float(in{}[tok * O + o_base + {r}u]));",
                    shared_base + 4
                ));
            } else {
                w.line(format!("out0[tok * O + o_base + {r}u] = y0;"));
            }
            w.pop_indent();
            w.line("}");
        }
        w.pop_indent();
        w.line("}");
        emit_close(w);
        return Ok(());
    }
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
    w.line("(void)tg_size;");
    w.line("threadgroup float route_partial[8 * OUT_COLS];");
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
    w.line("uint x_row_src = (simd_lane == 0u) ? uint(in4[route_idx]) : 0u;");
    w.line("uint expert_src = (simd_lane == 0u) ? min(uint(in5[route_idx]), N_EXPERTS - 1u) : 0u;");
    w.line("uint x_row = simd_broadcast(x_row_src, 0u);");
    w.line("uint expert = simd_broadcast(expert_src, 0u);");
    if has_route_weight {
        w.line("float rw_src = (simd_lane == 0u) ? float(in6[route_idx]) : 0.0f;");
        w.line("float rw = simd_broadcast(rw_src, 0u);");
    }
    for r in 0..out_cols {
        w.line(format!("float dot{r} = 0.0f;"));
    }
    if out_cols == 1 && bits == 4 && group_size == 64 && k_dim >= 1024 {
        crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
            w,
            "dot0",
            "in1 + (expert * O + o_base) * PACKED",
            "in2 + (expert * O + o_base) * N_GROUPS",
            "in3 + (expert * O + o_base) * N_GROUPS",
            "K",
            bits,
            group_size,
            false,
            false,
            crate::codegen::kernels::QdotPolicy::Force,
            |off| format!("float(in0[x_row * K + {off}])"),
        );
    } else if out_cols == 1 {
        crate::codegen::kernels::emit_int4_gemv_inner(
            w,
            "dot0",
            "in1 + (expert * O + o_base) * PACKED",
            "in2 + (expert * O + o_base) * N_GROUPS",
            "in3 + (expert * O + o_base) * N_GROUPS",
            "K",
            bits,
            group_size,
            |off| format!("float(in0[x_row * K + {off}])"),
        );
    } else if out_cols == 2 && matches!(bits, 4 | 8) && group_size == 64 {
        crate::codegen::kernels::emit_int4_gemv_inner_pair(
            w,
            "dot0",
            "dot1",
            "in1 + (expert * O + o_base) * PACKED",
            "in2 + (expert * O + o_base) * N_GROUPS",
            "in3 + (expert * O + o_base) * N_GROUPS",
            "in1 + (expert * O + (o_base + 1u)) * PACKED",
            "in2 + (expert * O + (o_base + 1u)) * N_GROUPS",
            "in3 + (expert * O + (o_base + 1u)) * N_GROUPS",
            "K",
            bits,
            group_size,
            |off| format!("float(in0[x_row * K + {off}])"),
        );
    } else {
        crate::codegen::kernels::emit_int4_gemv_inner_multirow(
            w,
            |r| format!("dot{r}"),
            |r| format!("in1 + (expert * O + (o_base + {r}u)) * PACKED"),
            |r| format!("in2 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            |r| format!("in3 + (expert * O + (o_base + {r}u)) * N_GROUPS"),
            "K",
            bits,
            group_size,
            out_cols,
            |off| format!("float(in0[x_row * K + {off}])"),
        );
    }
    for r in 0..out_cols {
        w.line(format!("dot{r} = simd_sum(dot{r});"));
    }
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    for r in 0..out_cols {
        if has_route_weight {
            w.line(format!(
                "route_partial[slot * OUT_COLS + {r}u] = float({t}(float({t}(dot{r})) * rw));"
            ));
        } else {
            w.line(format!(
                "route_partial[slot * OUT_COLS + {r}u] = float({t}(dot{r}));"
            ));
        }
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
    w.line(format!("out0[tok * O + o] = {t}(acc);"));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    emit_close(w);
    Ok(())
}

fn emit_gather_qmm_nax(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<bool> {
    let Some((k_dim, out_dim, active_routes, bits, group_size)) =
        crate::codegen::kernels::gather_qmm_nax_shape(fusion)
    else {
        return Ok(false);
    };
    if fusion.inputs.len() < 6 || fusion.outputs.is_empty() {
        return Ok(false);
    }
    let x_shape = &fusion.inputs[0].shape;
    let w_shape = &fusion.inputs[1].shape;
    let s_shape = &fusion.inputs[2].shape;
    let out_shape = &fusion.outputs[0].shape;
    if x_shape.is_empty() || w_shape.len() != 3 || s_shape.len() != 3 || out_shape.len() < 5 {
        return Ok(false);
    }
    let fallback = crate::codegen::kernels::act_ty(meta);
    let t = fusion
        .outputs
        .first()
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or(fallback);
    if !matches!(t, "bfloat" | "half" | "float") {
        return Ok(false);
    }
    let idx_t = fusion
        .inputs
        .get(4)
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or("uint");
    let expert_t = fusion
        .inputs
        .get(5)
        .map(|b| msl_ty(b.dtype, fallback))
        .unwrap_or("uint");
    let n_experts = w_shape[0].max(1) as u32;
    let packed = w_shape[2].max(1) as u32;
    let n_groups = s_shape[2].max(1) as u32;
    let tokens = out_shape.get(1).copied().unwrap_or(1).max(1) as u32;
    let active_tokens = crate::target::current_chip().active_tokens_for_shape(out_shape);
    let top_k = out_shape.get(2).copied().unwrap_or(1).max(1) as u32;
    let (_bm_auto, _bk_auto, _bn_auto) =
        crate::codegen::kernels::qmm_mlx_qmmt_tile(k_dim, out_dim, active_routes, bits);
    // Match MLX's gather_qmm_t_nax default tiling. This helper is distinct
    // from the generic qmm_t BlockMMA path: it keeps only the quantized
    // weight tile in threadgroup memory and loads the one routed activation
    // row directly through NAXTile, which is the small-M gather case.
    let bm = 64u32;
    let bk = 64u32;
    let bn = 64u32;
    let aligned_n = if out_dim.is_multiple_of(32) {
        "true"
    } else {
        "false"
    };
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(GatherQMM NAX) — M=1 routed qMM K={} N={} bits={} gs={} tile={}x{}x{} routes={}",
        fusion.id, k_dim, out_dim, bits, group_size, bm, bk, bn, active_routes
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    w.line("device const uint *in1 [[buffer(1)]],");
    w.line(format!("device const {t} *in2 [[buffer(2)]],"));
    w.line(format!("device const {t} *in3 [[buffer(3)]],"));
    w.line(format!("device const {idx_t} *in4 [[buffer(4)]],"));
    w.line(format!("device const {expert_t} *in5 [[buffer(5)]],"));
    w.line(format!("device {t}       *out0 [[buffer(6)]],"));
    w.line("uint3 tgid [[threadgroup_position_in_grid]],");
    w.line("uint tid [[thread_index_in_threadgroup]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("#if FUSEKIT_HAVE_MLX_NAX");
    w.push_indent();
    w.line(format!("constexpr int K = {k_dim};"));
    w.line(format!("constexpr int O = {out_dim};"));
    w.line("constexpr int M = 1;");
    w.line(format!("constexpr uint N_EXPERTS = {n_experts};"));
    w.line(format!("constexpr uint PACKED = {packed};"));
    w.line(format!("constexpr uint N_GROUPS = {n_groups};"));
    w.line(format!("constexpr uint TOKENS = {tokens};"));
    w.line(format!("constexpr uint ACTIVE_TOKENS = {active_tokens};"));
    w.line(format!("constexpr uint TOP_K = {top_k};"));
    w.line(format!("constexpr uint ACTIVE_ROUTES = {active_routes};"));
    w.line("uint route = tgid.y;");
    w.line("if (route >= ACTIVE_ROUTES) return;");
    w.line("uint slot = route % TOP_K;");
    w.line("uint tok = (route / TOP_K) % TOKENS;");
    w.line("if (tok >= ACTIVE_TOKENS) return;");
    w.line("uint route_idx = tok * TOP_K + slot;");
    w.line("uint x_row = uint(in4[route_idx]);");
    w.line("uint expert = min(uint(in5[route_idx]), N_EXPERTS - 1u);");
    w.line(format!(
        "threadgroup {t} Ws[{bk} * ({bn} + 16 / sizeof({t}))];"
    ));
    w.line("device const uint *w_expert = in1 + expert * O * PACKED;");
    w.line(format!(
        "device const {t} *s_expert = in2 + expert * O * N_GROUPS;"
    ));
    w.line(format!(
        "device const {t} *b_expert = in3 + expert * O * N_GROUPS;"
    ));
    w.line(format!(
        "device const {t} *x_row_ptr = in0 + uint64_t(x_row) * uint64_t(K);"
    ));
    w.line(format!(
        "device {t} *y_route = out0 + uint64_t(route) * uint64_t(O);"
    ));
    w.line("uint3 local_tgid = uint3(tgid.x, 0u, 0u);");
    w.line(format!(
        "fusekit_qmm_t_nax_tgp_impl<{t}, {group_size}, {bits}, {aligned_n}, {bm}, {bk}, {bn}, 2, 2>("
    ));
    w.push_indent();
    w.line("w_expert, s_expert, b_expert, x_row_ptr, y_route, Ws, K, O, M,");
    w.line("local_tgid, tid, simd_gid, simd_lid);");
    w.pop_indent();
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line("if (tid == 0u && tgid.x == 0u && tgid.y == 0u) { out0[0] = 0; }");
    w.line(
        "static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit GatherQMM NAX requires MLX Metal headers\");",
    );
    w.pop_indent();
    w.line("#endif");
    emit_close(w);
    Ok(true)
}

fn emit_reduce_sum(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Sum-reduce one axis where the output keeps that axis as size 1.
    if fusion.inputs.is_empty() || fusion.outputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "Reduce (missing input/output)");
    }
    let in_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    if in_shape.len() != out_shape.len() || in_shape.is_empty() {
        return emit_todo_stub(fusion, meta, w, "Reduce (rank mismatch)");
    }
    let axis = in_shape
        .iter()
        .zip(out_shape.iter())
        .position(|(&a, &b)| a > 1 && b == 1);
    let Some(axis) = axis else {
        return emit_todo_stub(fusion, meta, w, "Reduce (axis)");
    };
    let pre: i32 = in_shape[..axis].iter().product();
    let red = in_shape[axis].max(1) as u32;
    let post: i32 = in_shape[axis + 1..].iter().product();
    let pre = pre.max(1) as u32;
    let post = post.max(1) as u32;
    let n = total_elements(out_shape);
    let (t, _) = emit_header(fusion, meta, w, 1, 1, "Reduce (sum)");
    w.line(format!("constexpr uint PRE = {pre};"));
    w.line(format!("constexpr uint RED = {red};"));
    w.line(format!("constexpr uint POST = {post};"));
    w.line(format!("constexpr uint N_OUT = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint pre = i / POST;");
    w.line("uint post = i % POST;");
    w.line("float acc = 0.0f;");
    w.line("for (uint r = 0u; r < RED; ++r) {");
    w.push_indent();
    w.line("acc += float(in0[(pre * RED + r) * POST + post]);");
    w.pop_indent();
    w.line("}");
    w.line(format!("out0[i] = {t}(acc);"));
    emit_close(w);
    Ok(())
}

fn emit_scaled_sum_reduce(fusion: &Fusion, meta: &ModelMeta, w: &mut MslWriter) -> Result<()> {
    // Fuses the generic algebra:
    //   Broadcast(scale) -> Multiply(data, scale) -> Reduce(sum, axis)
    // into one pass over the final output. This is semantic shape
    // lowering, not model-name or model-architecture matching.
    let display_name = match &fusion.kind {
        FusionKind::Single(name) => name.as_str(),
        _ => "ScaledSumReduce",
    };
    if fusion.inputs.len() < 2 || fusion.outputs.is_empty() {
        return emit_todo_stub(
            fusion,
            meta,
            w,
            &format!("{display_name} (missing input/output)"),
        );
    }
    let data_shape = &fusion.inputs[0].shape;
    let weight_shape = &fusion.inputs[1].shape;
    let out_shape = &fusion.outputs[0].shape;
    if data_shape.len() != out_shape.len()
        || weight_shape.len() != data_shape.len()
        || data_shape.is_empty()
    {
        return emit_todo_stub(fusion, meta, w, &format!("{display_name} (rank)"));
    }
    let axis = data_shape
        .iter()
        .zip(out_shape.iter())
        .enumerate()
        .filter(|(_, (&d, &o))| d > 1 && o == 1)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    if axis.len() != 1 {
        return emit_todo_stub(fusion, meta, w, &format!("{display_name} (axis)"));
    }
    let axis = axis[0];
    for i in 0..data_shape.len() {
        if i == axis {
            if weight_shape[i] != data_shape[i] {
                return emit_todo_stub(fusion, meta, w, &format!("{display_name} (weight axis)"));
            }
            continue;
        }
        if out_shape[i] != data_shape[i] {
            return emit_todo_stub(fusion, meta, w, &format!("{display_name} (output shape)"));
        }
        if weight_shape[i] != data_shape[i] && weight_shape[i] != 1 {
            return emit_todo_stub(
                fusion,
                meta,
                w,
                &format!("{display_name} (weight broadcast)"),
            );
        }
    }
    let pre = data_shape[..axis].iter().product::<i32>().max(1) as u32;
    let red = data_shape[axis].max(1) as u32;
    let post = data_shape[axis + 1..].iter().product::<i32>().max(1) as u32;
    let weight_post = weight_shape[axis + 1..].iter().product::<i32>().max(1) as u32;
    if weight_post != 1 && weight_post != post {
        return emit_todo_stub(fusion, meta, w, &format!("{display_name} (weight post)"));
    }
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .min(u32::MAX as u64) as u32;
    let n = active_rows.saturating_mul(post).max(1);
    let (t, _) = emit_header(fusion, meta, w, 2, 1, display_name);
    w.line(format!("constexpr uint PRE = {pre};"));
    w.line(format!("constexpr uint RED = {red};"));
    w.line(format!("constexpr uint POST = {post};"));
    w.line(format!("constexpr uint WEIGHT_POST = {weight_post};"));
    w.line(format!("constexpr uint ACTIVE_ROWS = {active_rows};"));
    w.line(format!("constexpr uint N_OUT = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    w.line("uint pre = i / POST;");
    w.line("uint post = i % POST;");
    w.line("float acc = 0.0f;");
    w.line("for (uint r = 0u; r < RED; ++r) {");
    w.push_indent();
    w.line("uint data_idx = (pre * RED + r) * POST + post;");
    w.line("uint weight_idx = (pre * RED + r) * WEIGHT_POST + (WEIGHT_POST == 1u ? 0u : post);");
    w.line(format!(
        "acc += float({t}(float(in0[data_idx]) * float(in1[weight_idx])));"
    ));
    w.pop_indent();
    w.line("}");
    w.line(format!("out0[i] = {t}(acc);"));
    emit_close(w);
    Ok(())
}

fn emit_gather(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // mlx Gather is the indexed-take primitive (embedding lookup is
    // typically Gather(W, [token_id])). Inputs: (data, idx0, ..., idxK-1).
    // op.attrs gives axes + slice_sizes. The output rank = len(data) -
    // axes + slice_sizes (mlx convention).
    //
    // Production-grade Gather needs full per-axis slicing. The
    // overwhelming Qwen case is "1 axis, slice_sizes = [1, hidden]"
    // for embedding lookup. Implement that case; everything else
    // falls back to passthrough until needed.
    let (axes, slice_sizes) = match op_attrs {
        crate::graph::Attrs::Gather { axes, slice_sizes } => (axes.clone(), slice_sizes.clone()),
        _ => (vec![], vec![]),
    };
    let data_shape = &fusion.inputs[0].shape;
    let out_shape = &fusion.outputs[0].shape;
    if axes.len() != 1 || data_shape.len() != 2 || slice_sizes.len() != 2 {
        return emit_todo_stub(fusion, meta, w, "Gather (non-embed shape)");
    }
    let axis = axes[0] as usize;
    if axis != 0 || slice_sizes[0] != 1 {
        return emit_todo_stub(fusion, meta, w, "Gather (non-embed axis)");
    }
    let hidden = data_shape[1] as u32;
    let n_idx: i32 = fusion.inputs[1].shape.iter().product();
    let n_out = total_elements(out_shape);
    let (_t, _) = emit_header(fusion, meta, w, 2, 1, "Gather (embedding lookup)");
    w.line(format!("constexpr uint HIDDEN = {hidden};"));
    w.line(format!("constexpr uint N_IDX  = {};", n_idx.max(1)));
    w.line(format!("constexpr uint N_OUT  = {n_out};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N_OUT) return;");
    // out is (N_IDX, 1, HIDDEN). idx is (N_IDX, 1). data is (vocab, hidden).
    w.line("uint idx_no = i / HIDDEN;");
    w.line("uint h = i % HIDDEN;");
    w.line("uint row = (uint)(*((device const uint*)(in1 + idx_no)));");
    w.line("out0[i] = in0[row * HIDDEN + h];");
    emit_close(w);
    Ok(())
}

fn emit_concatenate(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // mlx Concatenate along a specific axis. N inputs, 1 output.
    // For each output element, decide which input it came from by
    // accumulating dim[axis] across inputs.
    let axis = match op_attrs {
        crate::graph::Attrs::Concatenate { axis } => *axis,
        _ => -1,
    };
    let out_shape = &fusion.outputs[0].shape;
    let rank = out_shape.len();
    let normalized_axis = if axis < 0 { rank as i32 + axis } else { axis };
    let mut axis_idx = if normalized_axis >= 0 {
        normalized_axis as usize
    } else {
        rank
    };
    if axis_idx >= rank
        || fusion.inputs.is_empty()
        || !concatenate_axis_matches(axis_idx, out_shape, &fusion.inputs)
    {
        axis_idx = match infer_concatenate_axis(out_shape, &fusion.inputs) {
            Some(axis) => axis,
            None => return emit_todo_stub(fusion, meta, w, "Concatenate (bad axis)"),
        };
    }
    if axis_idx >= rank || fusion.inputs.is_empty() {
        return emit_todo_stub(fusion, meta, w, "Concatenate (bad axis)");
    }
    // Outer = product of dims before axis. Inner = product of dims after.
    let outer: u32 = out_shape[..axis_idx]
        .iter()
        .copied()
        .map(|x| x.max(1) as u32)
        .product();
    let inner: u32 = out_shape[axis_idx + 1..]
        .iter()
        .copied()
        .map(|x| x.max(1) as u32)
        .product();
    let along_out = out_shape[axis_idx] as u32;
    let n_in = fusion.inputs.len();
    let (_t, _) = emit_header(
        fusion,
        meta,
        w,
        n_in,
        1,
        &format!("Concatenate(axis={axis_idx}, n_in={n_in})"),
    );
    w.line(format!("constexpr uint OUTER  = {outer};"));
    w.line(format!("constexpr uint INNER  = {inner};"));
    w.line(format!("constexpr uint ALONG  = {along_out};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= OUTER * ALONG * INNER) return;");
    w.line("uint outer = i / (ALONG * INNER);");
    w.line("uint rest  = i - outer * (ALONG * INNER);");
    w.line("uint along = rest / INNER;");
    w.line("uint inner = rest % INNER;");
    // Walk inputs to find which one covers `along`.
    let mut accum: u32 = 0;
    for (idx, b) in fusion.inputs.iter().enumerate() {
        let dim = b.shape.get(axis_idx).copied().unwrap_or(1).max(1) as u32;
        let lo = accum;
        let hi = accum + dim;
        if idx == 0 {
            w.line(format!("if (along < {hi}u) {{"));
        } else if idx == n_in - 1 {
            w.line(format!("else {{"));
        } else {
            w.line(format!("else if (along < {hi}u) {{"));
        }
        w.push_indent();
        w.line(format!("uint local_along = along - {lo}u;"));
        w.line(format!(
            "out0[i] = in{idx}[(outer * {dim}u + local_along) * INNER + inner];"
        ));
        w.pop_indent();
        w.line("}");
        accum = hi;
    }
    emit_close(w);
    Ok(())
}

fn emit_split(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if fusion.inputs.len() != 1 || fusion.outputs.is_empty() {
        bail!("Split: expected 1 input and at least 1 output");
    }
    let in_shape = &fusion.inputs[0].shape;
    let rank = in_shape.len();
    if rank == 0 {
        return emit_todo_stub(fusion, meta, w, "Split (scalar input)");
    }
    let mut axis = match op_attrs {
        crate::graph::Attrs::Split { axis } => *axis,
        _ => -1,
    };
    if axis < 0 {
        axis += rank as i32;
    }
    let mut axis_idx = axis as usize;
    if axis_idx >= rank {
        axis_idx = infer_split_axis(in_shape, &fusion.outputs).unwrap_or(rank - 1);
    }
    let outer: u32 = in_shape[..axis_idx]
        .iter()
        .copied()
        .map(|x| x.max(1) as u32)
        .product();
    let inner: u32 = in_shape[axis_idx + 1..]
        .iter()
        .copied()
        .map(|x| x.max(1) as u32)
        .product();
    let active_outer: u32 = if rank >= 3 && axis_idx >= 2 {
        let active_seq = crate::target::current_chip().active_tokens_for_shape(in_shape);
        let post_seq_prefix: u32 = in_shape[2..axis_idx]
            .iter()
            .copied()
            .map(|x| x.max(1) as u32)
            .product::<u32>()
            .max(1);
        active_seq.saturating_mul(post_seq_prefix).max(1).min(outer)
    } else {
        outer
    };
    let in_axis = in_shape[axis_idx].max(1) as u32;
    let out_elems: Vec<u32> = fusion
        .outputs
        .iter()
        .map(|b| {
            if rank >= 3 && axis_idx >= 2 {
                let axis_len = b.shape.get(axis_idx).copied().unwrap_or(1).max(1) as u32;
                active_outer.saturating_mul(axis_len).saturating_mul(inner)
            } else {
                total_elements(&b.shape)
            }
        })
        .collect();
    let total: u32 = out_elems.iter().sum();
    let (_t, _) = emit_header(
        fusion,
        meta,
        w,
        1,
        fusion.outputs.len(),
        &format!("Split(axis={axis_idx})"),
    );
    w.line(format!("constexpr uint OUTER = {outer}u;"));
    w.line(format!("constexpr uint ACTIVE_OUTER = {active_outer}u;"));
    w.line(format!("constexpr uint INNER = {inner}u;"));
    w.line(format!("constexpr uint IN_AXIS = {in_axis}u;"));
    w.line(format!("constexpr uint TOTAL = {total}u;"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= TOTAL) return;");
    let mut elem_base = 0u32;
    let mut axis_base = 0u32;
    for (idx, out) in fusion.outputs.iter().enumerate() {
        let axis_len = out.shape.get(axis_idx).copied().unwrap_or(1).max(1) as u32;
        let n = out_elems[idx];
        let cond = if idx == 0 {
            format!("if (i < {n}u) {{")
        } else if idx == fusion.outputs.len() - 1 {
            "else {".to_string()
        } else {
            format!("else if (i < {}u) {{", elem_base + n)
        };
        w.line(cond);
        w.push_indent();
        w.line(format!("uint local = i - {elem_base}u;"));
        w.line(format!("uint along_len = {axis_len}u;"));
        w.line("uint outer = local / (along_len * INNER);");
        w.line("uint rest = local - outer * (along_len * INNER);");
        w.line("uint along = rest / INNER;");
        w.line("uint inner = rest % INNER;");
        w.line(format!(
            "uint src = (outer * IN_AXIS + ({axis_base}u + along)) * INNER + inner;"
        ));
        w.line(format!("out{idx}[local] = in0[src];"));
        w.pop_indent();
        w.line("}");
        elem_base += n;
        axis_base += axis_len;
    }
    emit_close(w);
    Ok(())
}

fn infer_split_axis(
    in_shape: &[i32],
    outputs: &[crate::fusion::ir::FusionBoundary],
) -> Option<usize> {
    for axis in 0..in_shape.len() {
        let mut ok = true;
        let mut sum = 0i32;
        for out in outputs {
            if out.shape.len() != in_shape.len() {
                ok = false;
                break;
            }
            for (i, (&a, &b)) in in_shape.iter().zip(out.shape.iter()).enumerate() {
                if i == axis {
                    continue;
                }
                if a != b {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
            sum += out.shape[axis];
        }
        if ok && sum == in_shape[axis] {
            return Some(axis);
        }
    }
    None
}

fn concatenate_axis_matches(axis: usize, out_shape: &[i32], inputs: &[FusionBoundary]) -> bool {
    if inputs.is_empty() || axis >= out_shape.len() {
        return false;
    }
    let mut sum = 0i32;
    for input in inputs {
        if input.shape.len() != out_shape.len() {
            return false;
        }
        for (i, (&a, &b)) in input.shape.iter().zip(out_shape.iter()).enumerate() {
            if i == axis {
                continue;
            }
            if a != b {
                return false;
            }
        }
        sum += input.shape[axis];
    }
    sum == out_shape[axis]
}

fn infer_concatenate_axis(out_shape: &[i32], inputs: &[FusionBoundary]) -> Option<usize> {
    (0..out_shape.len()).find(|&axis| concatenate_axis_matches(axis, out_shape, inputs))
}

fn emit_convolution(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    post_silu: bool,
) -> Result<()> {
    if fusion.inputs.len() < 2 || fusion.outputs.len() != 1 {
        bail!("Convolution: expected input(s), weight, output");
    }
    let weight_idx = fusion.inputs.len() - 1;
    let segments = &fusion.inputs[..weight_idx];
    let w_shape = &fusion.inputs[weight_idx].shape;
    let y_shape = &fusion.outputs[0].shape;
    if segments.is_empty()
        || segments.iter().any(|b| b.shape.len() != 3)
        || y_shape.len() != 3
        || w_shape.len() < 2
    {
        return emit_todo_stub(fusion, meta, w, "Convolution (unsupported rank)");
    }
    let b_i = segments[0].shape[0];
    let c_i = segments[0].shape[2];
    let mut l_in_i = 0i32;
    for segment in segments {
        if segment.shape[0] != b_i || segment.shape[2] != c_i || segment.shape[1] <= 0 {
            return emit_todo_stub(fusion, meta, w, "Convolution (bad virtual concat)");
        }
        l_in_i += segment.shape[1];
    }
    let b = b_i.max(1) as u32;
    let l_in = l_in_i.max(1) as u32;
    let c = c_i.max(1) as u32;
    let l_out = y_shape[1].max(1) as u32;
    let k = w_shape[1].max(1) as u32;
    let groups_tail = w_shape.get(2).copied().unwrap_or(1).max(1) as u32;
    if y_shape[0].max(1) as u32 != b
        || y_shape[2].max(1) as u32 != c
        || w_shape[0].max(1) as u32 != c
        || groups_tail != 1
    {
        return emit_todo_stub(
            fusion,
            meta,
            w,
            "Convolution (only depthwise Conv1d supported)",
        );
    }
    if l_out + k - 1 != l_in {
        return emit_todo_stub(fusion, meta, w, "Convolution (only valid Conv1d supported)");
    }
    let n = total_elements(y_shape);
    let virtual_concat = segments.len() > 1;
    let summary = if virtual_concat && post_silu {
        "Concat+depthwise Conv1d(valid)+SiLU"
    } else if virtual_concat {
        "Concat+depthwise Conv1d(valid)"
    } else if post_silu {
        "depthwise Conv1d(valid)+SiLU"
    } else {
        "depthwise Conv1d(valid)"
    };
    let (t, _) = emit_header(fusion, meta, w, fusion.inputs.len(), 1, summary);
    w.line(format!("constexpr uint B = {b}u;"));
    w.line(format!("constexpr uint L_IN = {l_in}u;"));
    w.line(format!("constexpr uint L_OUT = {l_out}u;"));
    w.line(format!("constexpr uint C = {c}u;"));
    w.line(format!("constexpr uint K = {k}u;"));
    w.line(format!("constexpr uint N = {n}u;"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line("uint ch = i % C;");
    w.line("uint pos = (i / C) % L_OUT;");
    w.line("uint bidx = i / (L_OUT * C);");
    w.line("float acc = 0.0f;");
    w.line("for (uint kk = 0u; kk < K; ++kk) {");
    w.push_indent();
    if virtual_concat {
        w.line("uint src_pos = pos + kk;");
        w.line("float x_val = 0.0f;");
        let mut base = 0u32;
        for (idx, segment) in segments.iter().enumerate() {
            let len = segment.shape[1].max(1) as u32;
            let hi = base + len;
            if idx == 0 {
                w.line(format!("if (src_pos < {hi}u) {{"));
            } else if idx == segments.len() - 1 {
                w.line("else {");
            } else {
                w.line(format!("else if (src_pos < {hi}u) {{"));
            }
            w.push_indent();
            w.line(format!("uint local_pos = src_pos - {base}u;"));
            w.line(format!(
                "uint x_idx = (bidx * {len}u + local_pos) * C + ch;"
            ));
            w.line(format!("x_val = float(in{idx}[x_idx]);"));
            w.pop_indent();
            w.line("}");
            base = hi;
        }
    } else {
        w.line("uint x_idx = (bidx * L_IN + (pos + kk)) * C + ch;");
        w.line("float x_val = float(in0[x_idx]);");
    }
    w.line("uint w_idx = ch * K + kk;");
    w.line(format!("acc += x_val * float(in{weight_idx}[w_idx]);"));
    w.pop_indent();
    w.line("}");
    if post_silu {
        // MLX Conv1d materializes the convolution output in the activation
        // dtype before the following SiLU. Matching that rounding is important
        // for recurrent decode paths where one-ulp conv drift compounds.
        w.line(format!("{t} conv = {t}(acc);"));
        w.line("float conv_f = float(conv);");
        w.line("float sig = 1.0f / (1.0f + metal::fast::exp(-conv_f));");
        w.line(format!("out0[i] = {t}(conv_f * sig);"));
    } else {
        w.line(format!("out0[i] = {t}(acc);"));
    }
    emit_close(w);
    Ok(())
}

fn emit_rmsnorm(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    // Standalone RMSNorm — same kernel shape as the one inlined into
    // fused emitters, but as its own dispatch.
    let eps = match op_attrs {
        crate::graph::Attrs::RmsNorm { eps } => *eps,
        _ => 1e-6f32,
    };
    let shape = &fusion.outputs[0].shape;
    let last = shape.last().copied().unwrap_or(1) as u32;
    let head_major = shape.len() == 4
        && last == meta.head_dim
        && (shape[1].max(1) as u32 == meta.num_attention_heads
            || shape[1].max(1) as u32 == meta.num_key_value_heads);
    let (rows, heads, trace_seq, active_seq) = if head_major {
        let heads = shape[1].max(1) as u32;
        let trace_seq = shape[2].max(1) as u32;
        let active_seq = crate::target::current_chip().active_m(trace_seq).max(1);
        (
            heads.saturating_mul(active_seq).max(1),
            heads,
            trace_seq,
            active_seq,
        )
    } else {
        (
            crate::target::current_chip()
                .active_rows_before_last(shape)
                .min(u32::MAX as u64) as u32,
            1,
            1,
            1,
        )
    };
    let rows = rows.max(1);
    let weight_elems = fusion
        .inputs
        .get(1)
        .map(|b| total_elements(&b.shape))
        .unwrap_or(0);
    let weight_is_vector = weight_elems >= last;
    let (t, _) = emit_header(fusion, meta, w, 2, 1, "RMSNorm");
    w.line(format!("constexpr uint HIDDEN = {last};"));
    w.line(format!("constexpr uint ROWS   = {rows};"));
    w.line(format!("constexpr uint HEADS  = {heads};"));
    w.line(format!("constexpr uint TRACE_SEQ = {trace_seq};"));
    w.line(format!("constexpr uint ACTIVE_SEQ = {active_seq};"));
    w.line(format!("constexpr float EPS   = {eps};"));
    w.line(format!(
        "constexpr bool HEAD_MAJOR = {};",
        if head_major { "true" } else { "false" }
    ));
    w.line(format!(
        "constexpr bool WEIGHT_IS_VECTOR = {};",
        if weight_is_vector { "true" } else { "false" }
    ));
    w.line("uint logical_row = gid;");
    w.line("if (logical_row >= ROWS) return;");
    w.line("uint row = logical_row;");
    w.line("if (HEAD_MAJOR) {");
    w.push_indent();
    w.line("uint seq = logical_row / HEADS;");
    w.line("uint head = logical_row - seq * HEADS;");
    w.line("row = head * TRACE_SEQ + seq;");
    w.pop_indent();
    w.line("}");
    w.line(format!("device const {t} *x_row = in0 + row * HIDDEN;"));
    w.line(format!(
        "device const {t} *w_row = WEIGHT_IS_VECTOR ? in1 : in1;"
    ));
    w.line("threadgroup float reduce_buf[32];");
    crate::codegen::kernels::emit_rmsnorm_reduction(w, last, "x_row", "reduce_buf");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HIDDEN) + EPS);");
    w.line("float scalar_weight = WEIGHT_IS_VECTOR ? 1.0f : float(in1[0]);");
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.line(format!("{t} normed = {t}(float(x_row[i]) * rms_scale);"));
    w.line(format!(
        "{t} wi = WEIGHT_IS_VECTOR ? w_row[i] : {t}(scalar_weight);"
    ));
    w.line(format!("out0[row * HIDDEN + i] = {t}(wi * normed);"));
    w.pop_indent();
    w.line("}");
    emit_close(w);
    Ok(())
}

// ─── RoPE (standalone, from mx.fast.rope trace) ─────────────────────────────

/// Emit a standalone `Single(RoPE)` kernel from a trace `mx.fast.rope`
/// op. Mirrors the synthetic `RopeQDecode` kernel's math but reads
/// attrs (base, scale, dims, traditional) from the captured op so it
/// handles all mx.fast.rope variants — Qwen's 1M theta, Llama's 10K,
/// Llama-3 NTK scaling (via `scale`), DeepSeek MLA partial rope (via
/// `dims < head_dim`), Gemma, and any future model that uses the
/// standard mlx primitive.
///
/// Buffer layout:
///   buffer(0) device T*       — x tensor (same shape as fusion.outputs[0])
///   buffer(1) device int32_t* — offset tensor (1 element; current
///                                KV cache position). Runtime is
///                                expected to bind this to the live
///                                per-layer offset uniform.
///   buffer(2) device float*   — optional `freqs` tensor from mx.fast.rope
///                                when base=None/YARN is used.
///
/// Conventions match mlx.fast.rope: `traditional=true` rotates
/// consecutive pairs, otherwise pair `(d, d+HALF)` rotation is used.
/// Positions `[dims, head_dim)` pass through unchanged (partial-RoPE).
fn emit_rope(
    fusion: &Fusion,
    op_attrs: &crate::graph::Attrs,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    let (dims_attr, traditional, base, scale) = match op_attrs {
        crate::graph::Attrs::Rope {
            dims,
            traditional,
            base,
            scale,
            ..
        } => (*dims as u32, *traditional, *base, *scale),
        _ => bail!("rope emitter: expected Attrs::Rope"),
    };
    let inline_freqs = match op_attrs {
        crate::graph::Attrs::Rope { inline_freqs, .. } => inline_freqs.as_ref(),
        _ => None,
    };
    let shape = &fusion.outputs[0].shape;
    let head_dim = shape.last().copied().unwrap_or(0) as u32;
    // Total leading rows = product of all dims except last (head_dim).
    let rows: i32 = shape[..shape.len().saturating_sub(1)].iter().product();
    let rows = rows.max(1) as u32;
    let has_freqs = fusion.inputs.len() > 2;
    let has_inline_freqs = inline_freqs.is_some();
    let seq_len = shape
        .get(shape.len().saturating_sub(2))
        .copied()
        .unwrap_or(1)
        .max(1) as u32;
    // mlx caps `dims` to head_dim; partial-rope means dims < head_dim.
    let dims = dims_attr.min(head_dim);
    let half = dims / 2;
    if half == 0 {
        // Pure passthrough — emit a memcpy stub.
        return emit_view_passthrough(fusion, meta, w);
    }
    let log2_base = (base as f64).log2() as f32;

    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: Single(RoPE) — dims={} head_dim={} seq_len={} base={} scale={} traditional={} freqs={} inline_freqs={}",
        fusion.id, dims, head_dim, seq_len, base, scale, traditional, has_freqs, has_inline_freqs
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0       [[buffer(0)]],"));
    // in1 is the offset — mlx-captured as int32 tensor with 1 element.
    // The trace's static offset (often 0) gets bound here unless the
    // runtime reroutes input 1 to the live per-layer offset uniform.
    w.line("device const int  *in1       [[buffer(1)]],");
    if has_freqs {
        w.line("device const float *freqs    [[buffer(2)]],");
    }
    let out_slot = if has_freqs { 3 } else { 2 };
    w.line(format!(
        "device {t}       *out0      [[buffer({out_slot})]],"
    ));
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint HEAD_DIM = {head_dim};"));
    w.writeln(format_args!("constexpr uint DIMS     = {dims};"));
    w.writeln(format_args!("constexpr uint HALF     = {half};"));
    w.writeln(format_args!("constexpr uint SEQ_LEN  = {seq_len};"));
    w.writeln(format_args!("constexpr uint ROWS     = {rows};"));
    w.writeln(format_args!("constexpr float BASE_LOG2 = {log2_base:e}f;"));
    w.writeln(format_args!("constexpr float SCALE   = {scale};"));
    w.writeln(format_args!(
        "constexpr bool TRADITIONAL = {};",
        if traditional { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr bool HAS_FREQS = {};",
        if has_freqs { "true" } else { "false" }
    ));
    if let Some(freqs) = inline_freqs {
        let vals = freqs
            .iter()
            .map(|v| format!("{v:.9e}f"))
            .collect::<Vec<_>>()
            .join(", ");
        w.writeln(format_args!(
            "const float inline_freqs[{}] = {{ {vals} }};",
            freqs.len()
        ));
    }
    w.blank();

    w.line("uint g = gid * tg_size + tid;");
    w.line("uint TOTAL = ROWS * HEAD_DIM;");
    w.line("bool active = g < TOTAL;");
    w.line("uint row = active ? (g / HEAD_DIM) : 0u;");
    w.line("uint d   = active ? (g % HEAD_DIM) : 0u;");
    w.blank();

    // Partial-RoPE passthrough: positions [dims, head_dim) keep value.
    w.line("if (d >= DIMS) {");
    w.push_indent();
    w.line("if (active) out0[g] = in0[g];");
    w.line("return;");
    w.pop_indent();
    w.line("}");
    w.blank();

    // Read offset once and add the row's sequence coordinate. Decode
    // executes only slot 0, while prefill-shaped traces still carry
    // the correct position for every sequence row.
    w.line("uint offset = uint(in1[0]);");
    w.line("uint seq_idx = row % SEQ_LEN;");
    w.line("float pos = float(offset + seq_idx);");
    w.blank();

    w.line("uint pair_idx = TRADITIONAL ? (d / 2u) : ((d < HALF) ? d : (d - HALF));");
    w.line("float d_norm = float(pair_idx) / float(HALF);");
    if let Some(freqs) = inline_freqs {
        w.writeln(format_args!(
            "uint freq_idx = (pair_idx < {}u) ? pair_idx : {}u;",
            freqs.len(),
            freqs.len().saturating_sub(1)
        ));
        w.line("float inv_freq = SCALE / inline_freqs[freq_idx];");
    } else if has_freqs {
        w.line("float inv_freq = SCALE / float(freqs[pair_idx]);");
    } else {
        w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2) * SCALE;");
    }
    w.line("float angle = pos * inv_freq;");
    w.line("float c = metal::fast::cos(angle);");
    w.line("float s = metal::fast::sin(angle);");
    w.line("uint base_addr = row * HEAD_DIM;");
    w.line("uint lo_idx = TRADITIONAL ? (pair_idx * 2u) : pair_idx;");
    w.line("uint hi_idx = TRADITIONAL ? (lo_idx + 1u) : (pair_idx + HALF);");
    w.line("float lo = float(in0[base_addr + lo_idx]);");
    w.line("float hi = float(in0[base_addr + hi_idx]);");
    w.line("bool lo_lane = TRADITIONAL ? ((d & 1u) == 0u) : (d < HALF);");
    w.line("float result = lo_lane ? (lo * c - hi * s) : (hi * c + lo * s);");
    w.writeln(format_args!("if (active) out0[g] = {t}(result);"));

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

// ─── unknown / TODO ────────────────────────────────────────────────────────

fn emit_todo_stub(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    op_name: &str,
) -> Result<()> {
    let n = total_elements(
        &fusion
            .outputs
            .first()
            .map(|b| b.shape.clone())
            .unwrap_or(vec![0]),
    );
    let (_t, _) = emit_header(
        fusion,
        meta,
        w,
        fusion.inputs.len().max(1),
        1,
        &format!("TODO {op_name}"),
    );
    w.line(format!(
        "// TODO: implement Single emitter for op '{op_name}'"
    ));
    w.line(format!("constexpr uint N = {n};"));
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= N) return;");
    w.line("out0[i] = in0[i];");
    Ok(emit_close(w))
}
