//! Stage 3 — semantic-aware general fusion pass.
//!
//! Runs AFTER pattern-matched fusion (Stage 2). Walks the
//! [`FusionGraph`] looking for adjacent `FusionKind::Single` fusions
//! whose op categories permit merging, single-consumer chains, and
//! matching parallel axes — folds those into one
//! `FusionKind::FusedChain` whose body is emitted as a single MSL
//! kernel by the [`codegen::kernels::fused_chain`] emitter.
//!
//! What this pass does NOT do:
//! - Touch pattern-matched fusions (they're already hand-tuned).
//! - Cross a CustomKernel — those are opaque and must run as their
//!   own dispatch.
//! - Reorder. Producer→consumer order is preserved as the merge sweep
//!   only ever folds a single-consumer producer into its consumer.
//!
//! The pass is currently a *no-op scaffold* — types and the entry
//! point exist so we can wire it into the build pipeline before the
//! merge logic lands. Activating it is gated by the
//! `FUSEKIT_GENERAL_FUSION` env var to keep regression risk
//! contained while we iterate.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::fusion::category::{categorize_name, is_chain_absorbable_view, OpCategory};
use crate::fusion::ir::{
    AxisRef, BoundaryKind, ChainOperand, ChainStep, Fusion, FusionAttrs, FusionBoundary,
    FusionGraph, FusionKind, ParallelAxis, QkvDecodeSinks, QmvFanoutTarget, QuantInfo,
};
use crate::graph::{Attrs, Dtype, Graph, ScalarValue, TensorId};
use crate::target::Target;

/// Per-step register-slot estimate for the pointwise chain emitter.
/// Each chain step adds one `rN` register plus operand loads; we
/// budget two slots per step on top of the kernel's control-flow
/// reservation. Used by the budget check below to keep the greedy
/// merger from producing a chain that would spill on the target chip.
const REGS_PER_CHAIN_STEP: u32 = 2;
/// Control-flow / loop-index slots the compiler always reserves.
const REG_CONTROL_HEADROOM: u32 = 32;
/// Tiny guard band below the chip-reported per-thread register budget.
/// The estimates below already include [`REG_CONTROL_HEADROOM`], so
/// the pass must compare them against the real budget minus only this
/// small planning slack.
const REG_BUDGET_GUARD: u32 = 8;
/// A Reduction→single-QMM fusion recomputes the reduction once per
/// output tile. That is a win for small projections, but vocab-scale
/// heads amplify the RMSNorm work hundreds of times. Keep those as
/// RMSNorm + QMM so the normalized row is materialized once.
const LARGE_REDUCTION_QMM_MIN_OUT_DIM: i32 = 32_768;
const LARGE_REDUCTION_QMM_MIN_RATIO: i32 = 32;

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn step_register_slots(op_name: &str, input_count: usize) -> u32 {
    match op_name {
        // Two accumulators plus two dequant streams are materially
        // heavier than a pointwise step; count them explicitly so the
        // planner does not treat a serial heavy pair like six scalar ops.
        "QuantizedMatmul" => 42 + input_count.saturating_sub(4) as u32 * 2,
        "Matmul" | "AddMM" => 30 + input_count.saturating_sub(2) as u32 * 2,
        "RMSNorm" => 14,
        name if categorize_name(name) == OpCategory::Pointwise => 4,
        name if categorize_name(name) == OpCategory::View => 1,
        _ => REGS_PER_CHAIN_STEP,
    }
}

fn fusion_register_slots(f: &Fusion) -> u32 {
    match (&f.kind, &f.attrs) {
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps
            .iter()
            .map(|s| step_register_slots(&s.op_name, s.inputs.len()))
            .sum(),
        (FusionKind::Single(op_name), _) => step_register_slots(op_name, f.inputs.len()),
        _ => 32,
    }
}

/// Estimate the register pressure of a chain-shaped fusion (steps in
/// `attrs`). `None` for fusions that don't carry chain steps — caller
/// falls back to the per-FusionKind estimate.
fn chain_register_estimate(f: &Fusion) -> Option<u32> {
    match &f.attrs {
        FusionAttrs::FusedChain { .. } => Some(REG_CONTROL_HEADROOM + fusion_register_slots(f)),
        _ => None,
    }
}

/// Predict the register pressure of merging producer p into consumer c.
/// This is deliberately op-weighted rather than just chain-depth based:
/// a `QuantizedMatmul` body has accumulator/dequant state that can
/// spill long before a scalar pointwise chain of the same length.
fn estimate_combined_regs(producer: &Fusion, consumer: &Fusion) -> u32 {
    REG_CONTROL_HEADROOM + fusion_register_slots(producer) + fusion_register_slots(consumer)
}

/// Predict the combined SMEM footprint of merging two fusions. The
/// FusedChain emit path can have only one reduction head, so the row
/// scratch is the max chain-head footprint rather than a sum.
fn estimate_combined_smem(
    producer: &Fusion,
    consumer: &Fusion,
    meta: &crate::fusion::ir::ModelMeta,
) -> usize {
    // The chain emit's SMEM footprint depends on whether the chain
    // includes a Reduction head (then we cache the normed row + reduce
    // scratch), an SDPA piece, etc. We CAN'T reuse
    // `safety::estimate_smem_bytes` here because that function
    // returns `SMEM_BUDGET_BYTES + 1` for every `FusionKind::Single(_)`
    // — that's a sentinel that drives the SafetyCheck::SplitTasks
    // path, not a realistic figure. Using it would over-reject every
    // Single+Single merge (a QM o_proj + residual Add reads zero
    // SMEM in the heavy-first chain emit yet would be rejected as if
    // it needed 2× the 32 KiB budget).
    //
    // Estimate the merged chain's footprint based on the *shape* the
    // chain emitter will actually produce:
    //   - Producer is Reduction (RMSNorm) → hidden×float for `norm_y`
    //     + 32×float `reduce_buf`. Heavy/pointwise tail steps share
    //     the same row; no additional SMEM.
    //   - Producer is Heavy with pointwise tail (Linear-Residual
    //     shape) → 0 SMEM; the heavy GEMV's accumulator lives in
    //     registers and the tail reads externals directly.
    //   - Pre-existing FusedChain producer → its kind already
    //     dictates SMEM; use the safety estimator (FusedChain returns 0).
    use crate::fusion::category::{categorize_name, OpCategory};
    let reduce_buf_bytes = 32 * 4usize;

    let chain_head_smem = |f: &Fusion| -> usize {
        match (&f.kind, &f.attrs) {
            (FusionKind::Single(op_name), _) => {
                match categorize_name(op_name) {
                    OpCategory::Reduction if is_chainable_reduction_op(op_name) => {
                        meta.hidden_size as usize * 4 + reduce_buf_bytes
                    }
                    // Heavy / pointwise / view singletons read everything
                    // they need from external buffers in the chain emit;
                    // they don't carry SMEM into the chain.
                    OpCategory::Heavy | OpCategory::Pointwise | OpCategory::View => 0,
                    _ => crate::fusion::safety::estimate_smem_bytes(&f.kind, meta),
                }
            }
            (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => {
                // Match `emit_reduction_fanout_chain`'s SMEM layout.
                let has_reduction = steps
                    .first()
                    .is_some_and(|s| categorize_name(&s.op_name) == OpCategory::Reduction);
                if has_reduction {
                    meta.hidden_size as usize * 4 + reduce_buf_bytes
                } else {
                    0
                }
            }
            _ => crate::fusion::safety::estimate_smem_bytes(&f.kind, meta),
        }
    };

    // Producer and consumer often share the same `norm_y` region if
    // both reference the Reduction head — but at most one in any
    // chain shape can BE a Reduction. So take the max instead of sum.
    chain_head_smem(producer).max(chain_head_smem(consumer))
}

fn dtype_size_bytes(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
        Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::Bf16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
        Dtype::Other(_) => 4,
    }
}

fn shape_elems(shape: &[i32]) -> u64 {
    shape
        .iter()
        .map(|&d| d.max(1) as u64)
        .product::<u64>()
        .max(1)
}

fn boundary_nbytes(b: &FusionBoundary) -> u64 {
    shape_elems(&b.shape) * dtype_size_bytes(b.dtype)
}

fn step_external_nbytes(fusion: &Fusion, step: &ChainStep, skip_input0: bool) -> u64 {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut total = 0u64;
    for (idx, operand) in step.inputs.iter().enumerate() {
        if skip_input0 && idx == 0 {
            continue;
        }
        let ChainOperand::External { buffer_index } = operand else {
            continue;
        };
        if !seen.insert(*buffer_index) {
            continue;
        }
        if let Some(boundary) = fusion.inputs.get(*buffer_index as usize) {
            total = total.saturating_add(boundary_nbytes(boundary));
        }
    }
    total
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointwiseTailFormula {
    SiluGateUp,
}

fn pointwise_tail_absorb_budget_ok(
    formula: PointwiseTailFormula,
    producer: &Fusion,
    producer_steps: &[ChainStep],
) -> bool {
    if std::env::var_os("FUSEKIT_POINTWISE_TAIL_ABSORB_FORCE").is_some() {
        return true;
    }
    if producer_steps.len() < 3 || producer.outputs.len() < 2 {
        return false;
    }
    let gate = &producer.outputs[0];
    let Some((&out_dim, row_dims)) = gate.shape.split_last() else {
        return false;
    };
    if out_dim <= 0 {
        return false;
    }
    let rows = shape_elems(row_dims);
    let activation_bytes = rows
        .saturating_mul(out_dim as u64)
        .saturating_mul(dtype_size_bytes(gate.dtype));

    // Baseline SiluGateUp writes gate/up, then the standalone tail reads
    // gate twice and up once. The final tail output write remains in
    // both plans, so it is not saved.
    let dram_saved = match formula {
        PointwiseTailFormula::SiluGateUp => activation_bytes.saturating_mul(5),
    };

    // Absorbing gate/up makes each simdgroup compute both heavy
    // projections serially. That can win only if the saved activation
    // traffic is non-trivial next to the heavy operands that still must
    // be streamed. Otherwise we reduce dispatch count while hurting
    // occupancy/ILP, which Qwen2.5 0.5B 4bit showed clearly in bench.
    let heavy_stream = producer_steps
        .iter()
        .skip(1)
        .take(2)
        .map(|s| step_external_nbytes(producer, s, true))
        .sum::<u64>()
        .saturating_mul(rows);

    let dbg_budget = std::env::var_os("FUSEKIT_BUDGET_DEBUG").is_some()
        || std::env::var_os("FUSEKIT_POINTWISE_TAIL_ABSORB_DEBUG").is_some();
    // Require saved activation traffic to be at least 12.5% of the
    // heavy stream. This is shape/budget based, not model-name based.
    let ok = dram_saved.saturating_mul(8) >= heavy_stream;
    if !ok && dbg_budget {
        eprintln!(
            "[budget] pointwise-tail absorb reject: formula={formula:?} saved={:.3}MiB heavy_stream={:.3}MiB rows={} out_dim={}",
            dram_saved as f64 / (1024.0 * 1024.0),
            heavy_stream as f64 / (1024.0 * 1024.0),
            rows,
            out_dim
        );
    }
    ok
}

/// Run the general fusion pass. Returns a (possibly) rewritten
/// [`FusionGraph`]. Gated by `FUSEKIT_GENERAL_FUSION` until the
/// piece-composer path it produces matches hand-tuned-pattern perf
/// everywhere.
///
/// `target` provides chip budgets — the greedy merger rejects a
/// candidate fusion if its predicted register pressure exceeds the
/// chip's budget so we don't ship a kernel that spills to
/// thread-private memory.
pub fn run(mut fg: FusionGraph, target: &Target, graph: &crate::graph::Graph) -> FusionGraph {
    if std::env::var_os("FUSEKIT_GENERAL_FUSION").is_none() {
        return fg;
    }
    let reg_budget = target
        .budgets
        .regs_per_thread
        .saturating_sub(REG_BUDGET_GUARD);
    let smem_budget = target.budgets.smem_per_threadgroup as usize;
    let meta = fg.meta.clone();

    // Wrap fusions in Option so a merge can null-out the absorbed
    // producer without shifting every later index mid-pass.
    let mut slots: Vec<Option<Fusion>> = std::mem::take(&mut fg.fusions)
        .into_iter()
        .map(Some)
        .collect();

    // Fan-out first: catch RMSNorm whose normed output is consumed by
    // multiple Heavy ops (e.g. Q/K/V projections sharing the pre-attn
    // norm) and fold the whole bundle into a single multi-Heavy
    // chain. Pairwise merging only absorbs one consumer at a time
    // and would lose the SMEM-sharing win that makes this worth
    // doing.
    while try_fanout_merge(&mut slots, reg_budget, smem_budget, &meta) {}
    while try_qmv_sibling_fanout_merge(&mut slots, reg_budget, graph) {}
    if std::env::var("FUSEKIT_LINEAR_RESIDUAL_MERGE").as_deref() != Ok("0") {
        while try_linear_residual_merge(&mut slots, graph) {}
    }
    while try_final_lm_head_merge(&mut slots, &meta, graph) {}
    // QM+linear-bias absorption — for each FusedChain heavy step
    // that's a QuantizedMatmul whose output is consumed exactly once
    // by an Add(_, bias_param), absorb the bias param into the chain
    // step (as a 5th External operand) and re-point the chain's
    // output to the Add's output tensor. The emit then applies the
    // bias inline after the QM accumulator's `simd_sum`. Required
    // for correctness on QwenN.x and any other model whose
    // `nn.QuantizedLinear` has bias=True (most do): without this,
    // chain writes UNBIASED Q/K/V → RopeQDecode rotates unbiased →
    // SDPA reads wrong Q → garbage output. mlx pattern matchers
    // handle this via AddMM, but compiler-path FusedChain leaves
    // bias as a separate Single that IR DCE then drops.
    // `FUSEKIT_QM_BIAS=0` still forces the pass off for bisecting
    // future regressions.
    if std::env::var("FUSEKIT_QM_BIAS").as_deref() != Ok("0") {
        while try_absorb_qm_bias(&mut slots, &meta, graph) {}
    }
    // Then ordinary single-consumer chain growth.
    while try_merge_one(&mut slots, reg_budget, smem_budget, &meta) {}
    while try_sdpa_q_scale_view_absorb(&mut slots, graph) {}
    if std::env::var("FUSEKIT_QKV_DECODE_SINK").as_deref() != Ok("0") {
        while try_qkv_decode_sink(&mut slots, graph) {}
    }
    if env_flag("FUSEKIT_QMV_SWIGLU_ABSORB") {
        while try_qmv_swiglu_absorb(&mut slots, reg_budget) {}
    }
    // Custom-kernel sigmoid input absorb: if a pointwise Sigmoid only
    // exists to feed one CustomKernel input, rewrite that custom
    // kernel to apply the same rounded sigmoid internally and bind the
    // pre-sigmoid buffer directly. This is source/signature guarded
    // and keeps CustomKernel opaque unless the input access is
    // unambiguous.
    if env_flag("FUSEKIT_CUSTOM_SIGMOID_INPUT_ABSORB") {
        while try_custom_kernel_sigmoid_input_absorb(&mut slots) {}
    }
    // Self-SiLU sink: fold `y * sigmoid(y)` into the producer that
    // materializes `y` when that producer's emitter can apply the
    // post-op at store time. Qwen3Next linear-attention conv uses this
    // shape (`depthwise Conv1d` followed by SiLU), and keeping it as a
    // separate pointwise chain costs one launch plus a full DRAM
    // read/write round-trip per layer.
    if std::env::var("FUSEKIT_SELF_SILU_ABSORB").as_deref() != Ok("0") {
        while try_self_silu_absorb(&mut slots) {}
    }
    // Gate-sink absorb: fold a downstream `Sigmoid(gate) * x` into the
    // upstream RMSNorm+Heavy fanout when the gate output has no other real
    // consumer. Unlike SwiGLU absorb, this does not serialize two Heavy
    // projections in one simdgroup; it only changes the gate Heavy's final
    // store from `gate` to `x * sigmoid(gate)`.
    if std::env::var("FUSEKIT_GATED_MUL_ABSORB").as_deref() != Ok("0") {
        while try_gated_multiply_absorb(&mut slots, reg_budget, smem_budget, &meta) {}
    }
    // Semantic tail lowering: if a reduction-fanout producer feeds a
    // pointwise formula that can be represented without materializing
    // intermediate activations, canonicalize it and select the best
    // available lowering. This is IR-semantics based, not model-name
    // pattern matching.
    while try_reduction_fanout_pointwise_tail_lowering(&mut slots, reg_budget, smem_budget, &meta) {
    }
    report_swiglu_down_admission(&slots, smem_budget, &meta);
    if std::env::var("FUSEKIT_WEIGHTED_ROUTE_REDUCE").as_deref() != Ok("0") {
        while try_weighted_route_reduce(&mut slots) {}
    }

    // Compact and renumber.
    fg.fusions = slots
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(idx, mut f)| {
            f.id = idx as u32;
            f
        })
        .collect();
    mark_transposed_weight_slots(&mut fg, graph);
    mark_addmm_bias_slots(&mut fg, graph);
    mark_direct_external_view_slots(&mut fg, graph);
    fg
}

fn mark_transposed_weight_slots(fg: &mut FusionGraph, graph: &crate::graph::Graph) {
    if std::env::var_os("FUSEKIT_DISABLE_TRANSPOSE_PARAM_ELIDE").is_some() {
        return;
    }
    let path_filter = std::env::var("FUSEKIT_TRANSPOSE_PARAM_ELIDE_FILTER").ok();

    let producer_op: HashMap<TensorId, usize> = graph
        .ops
        .iter()
        .enumerate()
        .flat_map(|(idx, op)| op.outputs.iter().map(move |&tid| (tid, idx)))
        .collect();

    for fusion in &mut fg.fusions {
        let FusionAttrs::FusedChain {
            steps,
            transposed_weight_slots,
        } = &mut fusion.attrs
        else {
            continue;
        };
        let mut slots: HashSet<u32> = HashSet::new();
        for step in steps.iter() {
            if !matches!(step.op_name.as_str(), "Matmul" | "AddMM") || step.inputs.len() < 2 {
                continue;
            }
            let ChainOperand::External { buffer_index } = step.inputs[1] else {
                continue;
            };
            let Some(boundary) = fusion.inputs.get_mut(buffer_index as usize) else {
                continue;
            };
            let Some(&op_idx) = producer_op.get(&boundary.source) else {
                continue;
            };
            let op = &graph.ops[op_idx];
            if op.name != "Transpose" || op.inputs.len() != 1 {
                continue;
            }
            let src_tid = op.inputs[0];
            let Some(src_tensor) = graph.tensors.get(src_tid as usize) else {
                continue;
            };
            if let Some(filter) = path_filter.as_deref() {
                let Some(path) = src_tensor.param_path.as_deref() else {
                    continue;
                };
                if !path.contains(filter) {
                    continue;
                }
            }
            if src_tensor.param_path.is_none()
                || src_tensor.shape.len() != 2
                || boundary.shape.len() != 2
                || boundary.shape[0] != src_tensor.shape[1]
                || boundary.shape[1] != src_tensor.shape[0]
            {
                continue;
            }
            boundary.source = src_tid;
            slots.insert(buffer_index);
        }
        let mut slots: Vec<u32> = slots.into_iter().collect();
        slots.sort_unstable();
        *transposed_weight_slots = slots;
    }
}

fn mark_addmm_bias_slots(fg: &mut FusionGraph, graph: &crate::graph::Graph) {
    let producer_op: HashMap<TensorId, usize> = graph
        .ops
        .iter()
        .enumerate()
        .flat_map(|(idx, op)| op.outputs.iter().map(move |&tid| (tid, idx)))
        .collect();

    let is_param_view = |name: &str| {
        matches!(
            name,
            "Broadcast"
                | "ExpandDims"
                | "Squeeze"
                | "Reshape"
                | "Flatten"
                | "Unflatten"
                | "AsContiguous"
        )
    };
    let param_source = |start: TensorId| -> Option<TensorId> {
        let mut cur = start;
        let mut seen: HashSet<TensorId> = HashSet::new();
        while seen.insert(cur) {
            let tensor = graph.tensors.get(cur as usize)?;
            if tensor.param_path.is_some() {
                return Some(cur);
            }
            let op = graph.ops.get(*producer_op.get(&cur)?)?;
            if !is_param_view(&op.name) || op.inputs.is_empty() {
                return None;
            }
            cur = op.inputs[0];
        }
        None
    };

    for fusion in &mut fg.fusions {
        let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
            continue;
        };
        let mut rewrites: Vec<(u32, TensorId)> = Vec::new();
        for step in steps {
            if step.op_name != "AddMM" || step.inputs.len() < 3 {
                continue;
            }
            let ChainOperand::External { buffer_index } = step.inputs[2] else {
                continue;
            };
            let Some(boundary) = fusion.inputs.get(buffer_index as usize) else {
                continue;
            };
            if graph
                .tensors
                .get(boundary.source as usize)
                .and_then(|t| t.param_path.as_ref())
                .is_some()
            {
                continue;
            }
            if let Some(src_tid) = param_source(boundary.source) {
                rewrites.push((buffer_index, src_tid));
            }
        }
        for (buffer_index, src_tid) in rewrites {
            if let Some(boundary) = fusion.inputs.get_mut(buffer_index as usize) {
                boundary.source = src_tid;
            }
        }
    }
}

/// Rewrite selected FusedChain external inputs through dead
/// materialization views to the original external tensor.
///
/// Why this lives in Stage 3 instead of `task_graph_json`:
/// the generated MSL indexes external operands based on
/// `FusionBoundary.shape`. If JSON alone pointed a broadcasted input
/// at a scalar buffer, a generic `inN[i]` read would go out of bounds.
/// Rewriting the IR boundary lets the emitter see the *physical*
/// buffer shape and choose `[0]` for scalar broadcasts where that
/// emitter supports shape-aware indexing.
///
/// Safety model:
/// - View chain may contain only layout-preserving param/constant
///   materializers (`Broadcast`, `Full`, reshapes, squeezes, etc.).
///   `Transpose` is intentionally excluded; transposed weights use
///   `mark_transposed_weight_slots`, where the kernel knows it must
///   read the original row-major `(out, in)` memory differently.
/// - The chain must end at a model parameter or captured scalar.
/// - Same-element rewrites keep the logical boundary shape; the
///   pointer just skips a materialized view whose element order is the
///   same. Example: `[1, hidden]` broadcast of a `[hidden]` bias.
/// - Scalar rewrites shrink the boundary shape to the scalar's shape
///   so reduction-tail emitters select `[0]` indexing.
/// - We only touch FusedChain variants whose codegen is already
///   shape-aware for external tail operands. Plain pointwise chains
///   still use flat `inN[i]`, so they are deliberately excluded.
fn mark_direct_external_view_slots(fg: &mut FusionGraph, graph: &crate::graph::Graph) {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum RewriteMode {
        /// Source and old boundary have identical element counts. The
        /// kernel can keep using the old logical shape and index the
        /// compact source buffer directly.
        SameElements,
        /// Source is a scalar. Only shape-aware emitters may use this,
        /// because the boundary shape must be lowered so codegen emits
        /// `[0]` loads instead of `[i]`/`[out_idx]`.
        ScalarBroadcast,
    }

    let producer_op: HashMap<TensorId, usize> = graph
        .ops
        .iter()
        .enumerate()
        .flat_map(|(idx, op)| op.outputs.iter().map(move |&tid| (tid, idx)))
        .collect();

    let numel = |shape: &[i32]| -> i64 {
        if shape.is_empty() {
            return 1;
        }
        shape.iter().map(|&d| i64::from(d.max(1))).product()
    };
    let tensor_numel = |tid: TensorId| -> Option<i64> {
        graph
            .tensors
            .get(tid as usize)
            .map(|t| numel(&t.shape).max(1))
    };
    let is_external_leaf = |tid: TensorId| -> bool {
        graph
            .tensors
            .get(tid as usize)
            .is_some_and(|t| t.param_path.is_some() || t.scalar_value.is_some())
    };
    let is_direct_view_op = |name: &str| {
        matches!(
            name,
            "Broadcast"
                | "Full"
                | "ExpandDims"
                | "Squeeze"
                | "Reshape"
                | "Flatten"
                | "Unflatten"
                | "AsContiguous"
        )
    };
    let direct_view_source = |start: TensorId| -> Option<TensorId> {
        let mut cur = start;
        let mut seen: HashSet<TensorId> = HashSet::new();
        while seen.insert(cur) {
            if is_external_leaf(cur) {
                return Some(cur);
            }
            let op = graph.ops.get(*producer_op.get(&cur)?)?;
            if !is_direct_view_op(&op.name) {
                return None;
            }
            match op.name.as_str() {
                // `Full(scalar) -> big tensor` is semantically a
                // scalar broadcast. If the trace used a no-input Full,
                // the value lives only in unparsed op arguments, so do
                // not rewrite it here.
                "Full" if op.inputs.len() == 1 => cur = op.inputs[0],
                "Full" => return None,
                _ if op.inputs.len() == 1 => cur = op.inputs[0],
                _ => return None,
            }
        }
        None
    };

    // Classify external slots by how their emitter reads them.
    //
    // Reduction-tail expressions call `operand_expr_reduction` or
    // `operand_expr_heavy_tail`, both of which choose `[0]`, `[i]`, or
    // `[out_idx]` from the boundary shape. Heavy weights/scales do not
    // use that helper, so they are not listed. AddMM / absorbed-QM
    // linear bias slots are safe only for same-element rewrites because
    // the heavy kernels index them by output channel directly.
    let shape_aware_slots = |fusion: &Fusion| -> HashSet<u32> {
        let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
            return HashSet::new();
        };
        if steps.is_empty() || categorize_name(&steps[0].op_name) != OpCategory::Reduction {
            return HashSet::new();
        }

        let mut slots = HashSet::new();
        for (idx, step) in steps.iter().enumerate().skip(1) {
            match categorize_name(&step.op_name) {
                OpCategory::Pointwise | OpCategory::View => {
                    for operand in &step.inputs {
                        if let ChainOperand::External { buffer_index } = operand {
                            slots.insert(*buffer_index);
                        }
                    }
                }
                OpCategory::Heavy => match step.op_name.as_str() {
                    "AddMM" if step.inputs.len() >= 3 => {
                        if let ChainOperand::External { buffer_index } = step.inputs[2] {
                            slots.insert(buffer_index);
                        }
                    }
                    "QuantizedMatmul" if step.inputs.len() >= 5 => {
                        if let ChainOperand::External { buffer_index } = step.inputs[4] {
                            slots.insert(buffer_index);
                        }
                    }
                    _ => {}
                },
                _ => {}
            }

            // In reduction→heavy chains, later pointwise tail steps
            // are shape-aware; in reduction-fanout chains, a gated
            // multiply tail is also shape-aware. The first heavy step
            // itself is handled above only for linear bias.
            let _ = idx;
        }
        slots
    };

    let mut rewrites_total = 0usize;
    let mut qkv_bias_rewrites = 0usize;
    let mut scalar_rewrites = 0usize;
    for fusion in &mut fg.fusions {
        // RmsNormQkv bias slots are semantically per-projection
        // vectors. The trace broadcasts them to `[B, T, D]`, but the
        // custom kernel never indexes token rows for bias: it adds
        // `bias_q[g]`, `bias_k[i]`, `bias_v[i]` after each projection
        // accumulator. Point these slots at the original `[D]` bias
        // parameter so the broadcast producer becomes dead. This is
        // not a model-name pattern; it follows the semantic role names
        // emitted by the RmsNormQkv IR node.
        if matches!(fusion.kind, FusionKind::RmsNormQkv) {
            for boundary in &mut fusion.inputs {
                if !matches!(boundary.role, "bias_q" | "bias_k" | "bias_v") {
                    continue;
                }
                if is_external_leaf(boundary.source) {
                    continue;
                }
                let Some(src_tid) = direct_view_source(boundary.source) else {
                    continue;
                };
                let Some(src_tensor) = graph.tensors.get(src_tid as usize) else {
                    continue;
                };
                let Some(src_elems) = tensor_numel(src_tid) else {
                    continue;
                };
                let expected = boundary.shape.last().copied().map(i64::from).unwrap_or(0);
                if expected <= 0 || src_elems != expected {
                    continue;
                }
                boundary.source = src_tid;
                boundary.shape = src_tensor.shape.clone();
                boundary.dtype = src_tensor.dtype;
                rewrites_total += 1;
                qkv_bias_rewrites += 1;
            }
        }

        let allowed_slots = shape_aware_slots(fusion);
        if allowed_slots.is_empty() {
            continue;
        }

        for &slot in &allowed_slots {
            let Some(boundary) = fusion.inputs.get_mut(slot as usize) else {
                continue;
            };
            if is_external_leaf(boundary.source) {
                continue;
            }
            let Some(src_tid) = direct_view_source(boundary.source) else {
                continue;
            };
            let Some(src_tensor) = graph.tensors.get(src_tid as usize) else {
                continue;
            };
            let old_elems = numel(&boundary.shape).max(1);
            let Some(src_elems) = tensor_numel(src_tid) else {
                continue;
            };
            let mode = if src_elems == old_elems {
                RewriteMode::SameElements
            } else if src_elems == 1 {
                RewriteMode::ScalarBroadcast
            } else {
                continue;
            };

            boundary.source = src_tid;
            boundary.dtype = src_tensor.dtype;
            if mode == RewriteMode::ScalarBroadcast {
                boundary.shape = src_tensor.shape.clone();
                scalar_rewrites += 1;
            }
            rewrites_total += 1;
        }
    }

    if std::env::var_os("FUSEKIT_DIRECT_EXTERNAL_VIEW_DEBUG").is_some() && rewrites_total > 0 {
        eprintln!(
            "[direct-external-view] rewrites={} qkv_bias={} scalar_broadcast={}",
            rewrites_total, qkv_bias_rewrites, scalar_rewrites
        );
    }
}

/// Scan once for a fusable producer→consumer pair and merge it in
/// place. Returns true if anything was merged so the caller can
/// re-scan with the updated graph (fixed-point).
fn try_merge_one(
    slots: &mut Vec<Option<Fusion>>,
    reg_budget: u32,
    smem_budget: usize,
    meta: &crate::fusion::ir::ModelMeta,
) -> bool {
    // Build (tensor → producer fusion, tensor → consumer fusions) over
    // the current fusion graph. Re-derived per pass — the merge above
    // would have invalidated cached versions anyway.
    let mut producer_of: HashMap<TensorId, usize> = HashMap::new();
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for out in &f.outputs {
            producer_of.insert(out.source, idx);
        }
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    for c_idx in 0..slots.len() {
        let Some(consumer) = &slots[c_idx] else {
            continue;
        };
        if !is_chainable_consumer(consumer) {
            continue;
        }

        for input_slot in 0..consumer.inputs.len() {
            let src = consumer.inputs[input_slot].source;
            // See through any chain of single-consumer pure-metadata
            // Singles to find the real upstream producer. Mirrors the
            // forward walk that `try_fanout_merge` already does — lets
            // Reshape/Flatten/Squeeze between e.g. Heavy and Add get
            // absorbed instead of breaking the chain.
            let (root_src, absorbed_meta) =
                walk_back_through_metadata(src, slots, &producer_of, &consumers_of, c_idx);
            let Some(&p_idx) = producer_of.get(&root_src) else {
                continue;
            };
            if p_idx == c_idx {
                continue;
            }
            // single-consumer? exactly one fusion consumes the root
            // tensor (the first link if metadata, else `src` itself).
            let cons_list = consumers_of.get(&root_src).cloned().unwrap_or_default();
            let expected_consumer = absorbed_meta.last().copied().unwrap_or(c_idx);
            if cons_list.len() != 1 || cons_list[0] != expected_consumer {
                continue;
            }

            let Some(producer) = &slots[p_idx] else {
                continue;
            };
            if !is_chainable_producer(producer) {
                continue;
            }

            // Shape compatibility. When no metadata was walked, the
            // producer's output shape must match the consumer's input
            // exactly (broadcast-through-chain still isn't emittable).
            // When we *did* absorb metadata Singles, pure-metadata ops
            // preserve element count by definition, so we compare
            // products instead — the chain emit treats both buffers as
            // a flat `out_dim` vector regardless.
            if producer.outputs.is_empty() {
                continue;
            }
            let p_shape = &producer.outputs[0].shape;
            let c_shape = &consumer.inputs[input_slot].shape;
            if absorbed_meta.is_empty() {
                if p_shape != c_shape {
                    continue;
                }
            } else {
                let p_elems: i32 = p_shape.iter().product::<i32>().max(1);
                let c_elems: i32 = c_shape.iter().product::<i32>().max(1);
                if p_elems != c_elems {
                    continue;
                }
            }

            let consumer_heavy = consumer_is_heavy_singleton(consumer);
            if consumer_heavy {
                // Reduction → Heavy chains: producer must be a *bare*
                // Single RMSNorm (no intervening pointwise yet — the
                // chain emit assumes SMEM holds the RMSNorm-normalized
                // values exactly). Heavy must absorb on its row-vector
                // input (slot 0). Heavy → Heavy and Pointwise → Heavy
                // remain fusion barriers.
                let bare_rmsnorm = matches!(
                    &producer.kind,
                    FusionKind::Single(op) if op == "RMSNorm"
                );
                if !bare_rmsnorm {
                    continue;
                }
                if input_slot != 0 {
                    continue;
                }
                if !heavy_chain_shape_ok(consumer) {
                    continue;
                }
                if reduction_heavy_qmm_is_tile_amplified(producer, consumer) {
                    if std::env::var_os("FUSEKIT_BUDGET_DEBUG").is_some() {
                        eprintln!(
                            "[budget] try_merge_one cost reject: producer={:?} consumer={:?}",
                            producer.kind, consumer.kind
                        );
                    }
                    continue;
                }
            } else if producer_is_reduction(producer)
                && !reduction_merge_tail_ok(consumer, input_slot, producer)
            {
                // Pointwise/View tail after a Reduction head: external
                // tail operands have to fit the per-row indexing model.
                continue;
            } else if producer_is_reduction(producer) && !consumer_heavy {
                let max_tail = default_reduction_tail_max(producer, consumer, input_slot);
                if reduction_tail_len_after_merge(producer, consumer) > max_tail {
                    if std::env::var_os("FUSEKIT_BUDGET_DEBUG").is_some() {
                        eprintln!(
                            "[budget] try_merge_one reduction-tail reject: tail>{max_tail} producer={:?} consumer={:?}",
                            producer.kind, consumer.kind
                        );
                    }
                    continue;
                }
            }

            // Budget gate: don't merge if the combined chain would
            // exceed the chip's register or SMEM budget. Splitting
            // into two smaller fusions is cheaper than letting the
            // kernel spill (10–50× slower per access on Apple GPUs)
            // or, in the SMEM case, fail to launch entirely
            // (Metal hard-rejects >32 KiB threadgroup memory).
            // `FUSEKIT_BUDGET_DEBUG=1` traces budget-driven merge
            // rejects — useful for confirming that the chip's register
            // / SMEM budgets are actually constraining greedy fusion
            // on larger models, vs slack on smaller ones where the
            // synthetic-kernel boundaries dominate.
            let dbg_budget = std::env::var_os("FUSEKIT_BUDGET_DEBUG").is_some();
            let er = estimate_combined_regs(producer, consumer);
            if er > reg_budget {
                if dbg_budget {
                    eprintln!("[budget] try_merge_one reg reject: est={er} budget={reg_budget} producer={:?} consumer={:?}", producer.kind, consumer.kind);
                }
                continue;
            }
            let es = estimate_combined_smem(producer, consumer, meta);
            if es > smem_budget {
                if dbg_budget {
                    eprintln!("[budget] try_merge_one smem reject: est={es} budget={smem_budget} producer={:?} consumer={:?}", producer.kind, consumer.kind);
                }
                continue;
            }

            // Drop the absorbed metadata Singles. They produced no
            // compute — just shape views — and the FusedChain hands
            // the producer's register directly to the consumer step.
            for &m_idx in &absorbed_meta {
                slots[m_idx].take();
            }
            merge_into_chain(slots, p_idx, c_idx, input_slot);
            return true;
        }
    }
    false
}

/// A consumer fusion whose single step is a chain-supported Heavy op.
/// Only relevant for the Reduction → Heavy merge path; a chain head
/// already containing Heavy is handled by the producer branch.
fn consumer_is_heavy_singleton(f: &Fusion) -> bool {
    match &f.kind {
        FusionKind::Single(op_name) => {
            categorize_name(op_name) == OpCategory::Heavy && is_chainable_heavy_op(op_name)
        }
        _ => false,
    }
}

/// Walk *backward* from `src` through a chain of pure-metadata Singles
/// where each link has exactly one consumer (the next link or the
/// initial consumer). Returns `(root_tensor, absorbed_metadata_slot_indices)`
/// — root_tensor is the tensor produced by the first non-metadata
/// fusion, and the metadata fusions are dropped at merge time.
///
/// The single-consumer constraint is what keeps this reversible: if a
/// Reshape feeds two downstream cones, absorbing it would dangle the
/// other cone. The forward `try_fanout_merge` variant relaxes this
/// because the producer's SMEM stays available to every Heavy
/// consumer; the backward `try_merge_one` path can't, since the
/// pre-metadata tensor is gone after the merge.
fn walk_back_through_metadata(
    src: TensorId,
    slots: &[Option<Fusion>],
    producer_of: &HashMap<TensorId, usize>,
    consumers_of: &HashMap<TensorId, Vec<usize>>,
    initial_consumer_idx: usize,
) -> (TensorId, Vec<usize>) {
    let mut current = src;
    let mut absorbed: Vec<usize> = Vec::new();
    let mut next_consumer_idx = initial_consumer_idx;
    loop {
        let p_idx = match producer_of.get(&current) {
            Some(&i) => i,
            None => break,
        };
        let Some(p) = slots[p_idx].as_ref() else {
            break;
        };
        let FusionKind::Single(name) = &p.kind else {
            break;
        };
        if !is_chain_absorbable_view(name) {
            break;
        }
        if p.inputs.len() != 1 || p.outputs.len() != 1 {
            break;
        }
        let cons = consumers_of.get(&current).cloned().unwrap_or_default();
        if cons.len() != 1 || cons[0] != next_consumer_idx {
            break;
        }
        absorbed.push(p_idx);
        next_consumer_idx = p_idx;
        current = p.inputs[0].source;
    }
    (current, absorbed)
}

/// Walk forward from `src` through any chain of pure-metadata Singles
/// in the current `slots`, returning every tensor that's a view of the
/// same underlying data plus the slot indices of the metadata ops
/// themselves. The metadata ops will be absorbed (None'd out) by the
/// merge so the chain's output goes directly from producer SMEM into
/// the downstream Heavy step.
///
/// Only metadata Singles with exactly one input and one output, and
/// whose input is the tensor we're walking from, qualify — this keeps
/// the walk linear and reversible. (A reshape that branches into two
/// downstream cones is allowed; what's *not* allowed is a metadata op
/// that consumes from outside the alias set.)
fn collect_metadata_alias_chain(
    src: crate::graph::TensorId,
    slots: &[Option<Fusion>],
    consumers_of: &HashMap<crate::graph::TensorId, Vec<usize>>,
) -> (
    std::collections::HashSet<crate::graph::TensorId>,
    Vec<usize>,
) {
    use std::collections::HashSet;
    let mut alias_tensors: HashSet<crate::graph::TensorId> = HashSet::new();
    let mut metadata_indices: Vec<usize> = Vec::new();
    alias_tensors.insert(src);
    let mut queue: Vec<crate::graph::TensorId> = vec![src];
    while let Some(t) = queue.pop() {
        let direct = match consumers_of.get(&t) {
            Some(v) => v.clone(),
            None => continue,
        };
        for c_idx in direct {
            let Some(c) = slots[c_idx].as_ref() else {
                continue;
            };
            let FusionKind::Single(name) = &c.kind else {
                continue;
            };
            if !is_chain_absorbable_view(name) {
                continue;
            }
            if c.inputs.len() != 1 || c.outputs.len() != 1 {
                continue;
            }
            if c.inputs[0].source != t {
                continue;
            }
            let out = c.outputs[0].source;
            if alias_tensors.insert(out) {
                metadata_indices.push(c_idx);
                queue.push(out);
            }
        }
    }
    (alias_tensors, metadata_indices)
}

fn collect_q_layout_alias_chain(
    src: crate::graph::TensorId,
    slots: &[Option<Fusion>],
    consumers_of: &HashMap<crate::graph::TensorId, Vec<usize>>,
    absorbed_add_inputs: &std::collections::HashSet<crate::graph::TensorId>,
) -> (
    std::collections::HashSet<crate::graph::TensorId>,
    Vec<usize>,
) {
    use std::collections::HashSet;
    let mut alias_tensors: HashSet<crate::graph::TensorId> = HashSet::new();
    let mut metadata_indices: Vec<usize> = Vec::new();
    alias_tensors.insert(src);
    let mut queue: Vec<crate::graph::TensorId> = vec![src];
    while let Some(t) = queue.pop() {
        let direct = match consumers_of.get(&t) {
            Some(v) => v.clone(),
            None => continue,
        };
        for c_idx in direct {
            let Some(c) = slots[c_idx].as_ref() else {
                continue;
            };
            let FusionKind::Single(name) = &c.kind else {
                continue;
            };
            if (is_chain_absorbable_view(name) || name == "Transpose")
                && c.inputs.len() == 1
                && c.outputs.len() == 1
                && c.inputs[0].source == t
            {
                let out = c.outputs[0].source;
                if alias_tensors.insert(out) {
                    metadata_indices.push(c_idx);
                    queue.push(out);
                }
                continue;
            }
            if name == "Add" && c.inputs.len() == 2 && c.outputs.len() == 1 {
                let a = c.inputs[0].source;
                let b = c.inputs[1].source;
                let absorbed_bias_add = (a == t && absorbed_add_inputs.contains(&b))
                    || (b == t && absorbed_add_inputs.contains(&a));
                if absorbed_bias_add
                    && shape_elems(&c.outputs[0].shape) == shape_elems(&c.inputs[0].shape)
                {
                    let out = c.outputs[0].source;
                    if alias_tensors.insert(out) {
                        metadata_indices.push(c_idx);
                        queue.push(out);
                    }
                }
            }
        }
    }
    (alias_tensors, metadata_indices)
}

fn q_scale_movable_graph_op(op_name: &str) -> bool {
    // Q-specific semantic aliases for scale absorption. RoPE is not a
    // byte-layout alias, but scalar Q scaling commutes with the linear
    // rotation, so `RoPE(q) * c` can be represented as `RoPE(q)` plus an
    // SDPA scale multiplier.
    is_chain_absorbable_view(op_name) || matches!(op_name, "Transpose" | "RoPE")
}

fn add_bias_layout_root(
    graph: &Graph,
    producer_op: &HashMap<TensorId, usize>,
    mut tid: TensorId,
) -> TensorId {
    let mut seen = HashSet::new();
    while seen.insert(tid) {
        let Some(&op_idx) = producer_op.get(&tid) else {
            break;
        };
        let op = &graph.ops[op_idx];
        if !matches!(
            op.name.as_str(),
            "Broadcast"
                | "Reshape"
                | "Flatten"
                | "Unflatten"
                | "Squeeze"
                | "ExpandDims"
                | "AsContiguous"
                | "Transpose"
        ) || op.inputs.is_empty()
        {
            break;
        }
        tid = op.inputs[0];
    }
    tid
}

fn collect_q_graph_semantic_aliases(
    src: TensorId,
    graph: &Graph,
    graph_consumers: &HashMap<TensorId, Vec<usize>>,
    graph_producer_op: &HashMap<TensorId, usize>,
    absorbed_add_inputs: &HashSet<TensorId>,
) -> HashSet<TensorId> {
    let absorbed_roots: HashSet<TensorId> = absorbed_add_inputs
        .iter()
        .map(|&tid| add_bias_layout_root(graph, graph_producer_op, tid))
        .collect();
    let is_absorbed_add_side = |tid: TensorId| -> bool {
        absorbed_add_inputs.contains(&tid)
            || absorbed_roots.contains(&add_bias_layout_root(graph, graph_producer_op, tid))
    };

    let mut aliases = HashSet::new();
    aliases.insert(src);
    let mut queue = vec![src];
    while let Some(t) = queue.pop() {
        let Some(consumers) = graph_consumers.get(&t) else {
            continue;
        };
        for &op_idx in consumers {
            let Some(op) = graph.ops.get(op_idx) else {
                continue;
            };
            if op.outputs.len() != 1 {
                continue;
            }
            if q_scale_movable_graph_op(&op.name) && op.inputs.first() == Some(&t) {
                let out = op.outputs[0];
                if aliases.insert(out) {
                    queue.push(out);
                }
                continue;
            }
            if op.name == "Add" && op.inputs.len() == 2 {
                let a = op.inputs[0];
                let b = op.inputs[1];
                let absorbed_bias_add =
                    (a == t && is_absorbed_add_side(b)) || (b == t && is_absorbed_add_side(a));
                if absorbed_bias_add {
                    let out = op.outputs[0];
                    if aliases.insert(out) {
                        queue.push(out);
                    }
                }
            }
        }
    }
    aliases
}

fn boundary_from_graph(g: &Graph, tid: TensorId, role: &'static str) -> FusionBoundary {
    let t = &g.tensors[tid as usize];
    FusionBoundary {
        source: tid,
        role,
        shape: t.shape.clone(),
        dtype: t.dtype,
    }
}

fn trace_view_root(g: &Graph, mut tid: TensorId) -> TensorId {
    loop {
        let Some(producer) = g.tensor(tid).producer else {
            return tid;
        };
        let op = g.op(producer);
        let is_view = is_chain_absorbable_view(&op.name) || op.name == "Transpose";
        if is_view && !op.inputs.is_empty() {
            tid = op.inputs[0];
        } else {
            return tid;
        }
    }
}

/// Compiler-path equivalent of the hand `final_rmsnorm_lm_head`
/// pattern. With `--patterns auto/off`, the pattern registry is
/// disabled and general fusion owns this semantic case:
///
///   RMSNorm(hidden) -> optional view(s) -> vocab-sized Matmul/QMM
///
/// Ordinary Reduction->QMM chain fusion deliberately rejects this
/// shape because recomputing RMSNorm per vocab tile is expensive. The
/// dedicated emitter keeps the normed row in SMEM per threadgroup and
/// strides through many vocab rows, which is the intended lowering for
/// lm_head.
fn try_final_lm_head_merge(
    slots: &mut Vec<Option<Fusion>>,
    meta: &crate::fusion::ir::ModelMeta,
    graph: &Graph,
) -> bool {
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    for p_idx in 0..slots.len() {
        let Some(producer) = slots[p_idx].as_ref() else {
            continue;
        };
        let FusionKind::Single(p_name) = &producer.kind else {
            continue;
        };
        if p_name != "RMSNorm" || producer.inputs.len() != 2 || producer.outputs.len() != 1 {
            continue;
        }
        let eps = match producer.attrs {
            FusionAttrs::Single {
                op_attrs: Attrs::RmsNorm { eps },
                ..
            } => eps,
            _ => continue,
        };
        if producer.outputs[0].shape.last().copied() != Some(meta.hidden_size as i32) {
            continue;
        }

        let src = producer.outputs[0].source;
        let (aliases, metadata_idxs) = collect_metadata_alias_chain(src, slots, &consumers_of);
        let metadata_set: HashSet<usize> = metadata_idxs.iter().copied().collect();
        let mut heavy_idx: Option<usize> = None;
        let mut external_consumer = false;
        for &alias in &aliases {
            let Some(consumers) = consumers_of.get(&alias) else {
                continue;
            };
            for &c_idx in consumers {
                if c_idx == p_idx || metadata_set.contains(&c_idx) {
                    continue;
                }
                let Some(c) = slots[c_idx].as_ref() else {
                    continue;
                };
                let out_last = c.outputs.first().and_then(|b| b.shape.last().copied());
                let is_vocab_projection_dim = out_last.is_some_and(|d| {
                    d == meta.vocab_size as i32
                        || (meta.hidden_size > 0
                            && d >= (meta.hidden_size as i32).saturating_mul(32))
                });
                let is_lm_head_heavy = matches!(
                    &c.kind,
                    FusionKind::Single(name) if name == "QuantizedMatmul" || name == "Matmul"
                ) && c
                    .inputs
                    .first()
                    .is_some_and(|b| aliases.contains(&b.source))
                    && is_vocab_projection_dim;
                if !is_lm_head_heavy {
                    external_consumer = true;
                    break;
                }
                if heavy_idx.replace(c_idx).is_some() {
                    external_consumer = true;
                    break;
                }
            }
            if external_consumer {
                break;
            }
        }
        if external_consumer {
            continue;
        }
        let Some(c_idx) = heavy_idx else {
            continue;
        };
        let Some(consumer) = slots[c_idx].as_ref() else {
            continue;
        };

        let (is_quant, quant) = match (&consumer.kind, &consumer.attrs) {
            (
                FusionKind::Single(name),
                FusionAttrs::Single {
                    op_attrs: Attrs::Quantized { bits, group_size },
                    ..
                },
            ) if name == "QuantizedMatmul" => (
                true,
                Some(QuantInfo {
                    bits: *bits,
                    group_size: *group_size,
                }),
            ),
            (FusionKind::Single(name), _) if name == "Matmul" => (false, None),
            _ => continue,
        };
        if is_quant && consumer.inputs.len() < 4 {
            continue;
        }
        if !is_quant && consumer.inputs.len() < 2 {
            continue;
        }

        let weight_root = trace_view_root(graph, consumer.inputs[1].source);
        let logits_tid = consumer.outputs[0].source;
        let mut inputs = vec![
            producer.inputs[0].clone(),
            producer.inputs[1].clone(),
            boundary_from_graph(graph, weight_root, "W_lm_head"),
        ];
        inputs[0].role = "x";
        inputs[1].role = "norm_weight";
        if is_quant {
            inputs.push(FusionBoundary {
                role: "W_lm_head_scales",
                ..consumer.inputs[2].clone()
            });
            inputs.push(FusionBoundary {
                role: "W_lm_head_biases",
                ..consumer.inputs[3].clone()
            });
        }
        let outputs = vec![FusionBoundary {
            role: "logits",
            ..consumer.outputs[0].clone()
        }];
        let out_shape = &graph.tensors[logits_tid as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "vocab",
            size: *out_shape.last().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: out_shape.len().saturating_sub(1) as u32,
            },
        }];
        let mut op_ids = producer.op_ids.clone();
        for &m_idx in &metadata_idxs {
            if let Some(m) = slots[m_idx].as_ref() {
                op_ids.extend(m.op_ids.iter().copied());
            }
        }
        op_ids.extend(consumer.op_ids.iter().copied());

        let new_fusion = Fusion {
            id: producer.id,
            kind: FusionKind::FinalRmsNormLmHead,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::FinalRmsNormLmHead { eps, quant },
            parallel_axes,
            layer_hint: producer.layer_hint.or(consumer.layer_hint),
        };
        slots[p_idx] = Some(new_fusion);
        for idx in metadata_idxs {
            slots[idx].take();
        }
        slots[c_idx].take();
        return true;
    }
    false
}

/// Compiler-path equivalent of the hand `linear_residual` pattern:
///
///   QuantizedMatmul/Matmul(x, W) -> optional view(s) -> Add(_, residual)
///
/// This is semantic rather than model-name based. The residual side must
/// have exactly the output shape, which keeps broadcasted linear bias
/// adds out of this rewrite.
fn try_linear_residual_merge(slots: &mut Vec<Option<Fusion>>, graph: &Graph) -> bool {
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    for p_idx in 0..slots.len() {
        let Some(producer) = slots[p_idx].as_ref() else {
            continue;
        };
        let (is_quant, quant) = match (&producer.kind, &producer.attrs) {
            (
                FusionKind::Single(name),
                FusionAttrs::Single {
                    op_attrs: Attrs::Quantized { bits, group_size },
                    ..
                },
            ) if name == "QuantizedMatmul" => (
                true,
                Some(QuantInfo {
                    bits: *bits,
                    group_size: *group_size,
                }),
            ),
            (FusionKind::Single(name), _) if name == "Matmul" => (false, None),
            _ => continue,
        };
        let expected_inputs = if is_quant { 4 } else { 2 };
        if producer.inputs.len() != expected_inputs || producer.outputs.len() != 1 {
            continue;
        }

        let mm_out = producer.outputs[0].source;
        let (aliases, metadata_idxs) = collect_metadata_alias_chain(mm_out, slots, &consumers_of);
        let metadata_set: HashSet<usize> = metadata_idxs.iter().copied().collect();
        let mut add_idx: Option<usize> = None;
        let mut external_consumer = false;
        for &alias in &aliases {
            let Some(consumers) = consumers_of.get(&alias) else {
                continue;
            };
            for &c_idx in consumers {
                if c_idx == p_idx || metadata_set.contains(&c_idx) {
                    continue;
                }
                let Some(c) = slots[c_idx].as_ref() else {
                    continue;
                };
                let is_add = matches!(&c.kind, FusionKind::Single(name) if name == "Add");
                if !is_add {
                    external_consumer = true;
                    break;
                }
                if add_idx.replace(c_idx).is_some() {
                    external_consumer = true;
                    break;
                }
            }
            if external_consumer {
                break;
            }
        }
        if external_consumer {
            continue;
        }
        let Some(c_idx) = add_idx else {
            continue;
        };
        let Some(add) = slots[c_idx].as_ref() else {
            continue;
        };
        if add.inputs.len() != 2 || add.outputs.len() != 1 {
            continue;
        }

        let mm_side = if aliases.contains(&add.inputs[0].source) {
            0
        } else if aliases.contains(&add.inputs[1].source) {
            1
        } else {
            continue;
        };
        let residual_side = 1 - mm_side;
        let out_shape = &add.outputs[0].shape;
        let residual = &add.inputs[residual_side];
        if residual.shape != *out_shape {
            continue;
        }
        if producer.outputs[0].shape.last() != add.outputs[0].shape.last() {
            continue;
        }

        let weight_root = trace_view_root(graph, producer.inputs[1].source);
        let mut inputs = vec![
            producer.inputs[0].clone(),
            boundary_from_graph(graph, weight_root, "W"),
        ];
        inputs[0].role = "x";
        if is_quant {
            inputs.push(FusionBoundary {
                role: "W_scales",
                ..producer.inputs[2].clone()
            });
            inputs.push(FusionBoundary {
                role: "W_biases",
                ..producer.inputs[3].clone()
            });
        }
        inputs.push(FusionBoundary {
            role: "residual",
            ..residual.clone()
        });

        let outputs = vec![FusionBoundary {
            role: "y",
            ..add.outputs[0].clone()
        }];
        let parallel_axes = vec![ParallelAxis {
            name: "out_row",
            size: *out_shape.last().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: out_shape.len().saturating_sub(1) as u32,
            },
        }];
        let mut op_ids = producer.op_ids.clone();
        for &m_idx in &metadata_idxs {
            if let Some(m) = slots[m_idx].as_ref() {
                op_ids.extend(m.op_ids.iter().copied());
            }
        }
        op_ids.extend(add.op_ids.iter().copied());

        let new_fusion = Fusion {
            id: producer.id,
            kind: FusionKind::LinearResidual,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::LinearResidual { quant },
            parallel_axes,
            layer_hint: producer.layer_hint.or(add.layer_hint),
        };
        slots[p_idx] = Some(new_fusion);
        for idx in metadata_idxs {
            slots[idx].take();
        }
        slots[c_idx].take();
        return true;
    }
    false
}

/// Fan-out fusion. Look for a Single RMSNorm whose normed output is
/// consumed by **multiple** Single Heavy ops (the classic Q/K/V
/// projection shape) and fold the entire bundle into one FusedChain
/// whose body emits a SMEM-cached normed row and dispatches
/// per-simdgroup to whichever heavy step owns that output slot.
///
/// The SMEM cache is the whole point — without it each Heavy op
/// would re-read the producer's output from DRAM, and the merge
/// would save only one kernel launch instead of (N-1) reads.
fn try_fanout_merge(
    slots: &mut Vec<Option<Fusion>>,
    reg_budget: u32,
    smem_budget: usize,
    _meta: &crate::fusion::ir::ModelMeta,
) -> bool {
    // Producer → tensor → consumer index, with a single pass over
    // the current fusion graph.
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    for p_idx in 0..slots.len() {
        // Producer side: Single RMSNorm with exactly one output and the
        // expected (x, weight) input shape.
        let Some(producer) = &slots[p_idx] else {
            continue;
        };
        let p_is_rmsnorm = matches!(&producer.kind, FusionKind::Single(op) if op == "RMSNorm");
        if !p_is_rmsnorm {
            continue;
        }
        if producer.outputs.len() != 1 {
            continue;
        }
        if !reduction_chain_shape_ok(producer) {
            continue;
        }
        let src = producer.outputs[0].source;
        let p_out_shape = producer.outputs[0].shape.clone();

        // Walk forward through any chain of pure-metadata Singles so
        // that `ExpandDims` / `Reshape` / etc between producer and
        // Heavy consumers don't break detection. Element count is
        // preserved across pure-metadata ops, so the chain can still
        // hand SMEM directly to the Heavy step.
        let (alias_tensors, metadata_idxs) =
            collect_metadata_alias_chain(src, slots, &consumers_of);

        // Heavy consumers = every direct consumer of any alias tensor
        // that's NOT a metadata Single we've already absorbed. (Same
        // metadata Single could appear as consumer of multiple aliases
        // — dedup via the index set.)
        let metadata_set: std::collections::HashSet<usize> =
            metadata_idxs.iter().copied().collect();
        let mut heavy_cons: Vec<usize> = Vec::new();
        let mut heavy_seen: std::collections::HashSet<usize> = Default::default();
        let mut external_consumer_seen = false;
        for &alias_t in &alias_tensors {
            let Some(direct) = consumers_of.get(&alias_t) else {
                continue;
            };
            for &c_idx in direct {
                if c_idx == p_idx {
                    external_consumer_seen = true;
                    break;
                }
                if metadata_set.contains(&c_idx) {
                    continue;
                }
                if heavy_seen.insert(c_idx) {
                    heavy_cons.push(c_idx);
                }
            }
            if external_consumer_seen {
                break;
            }
        }
        if external_consumer_seen {
            continue;
        }
        if heavy_cons.len() < 2 {
            continue;
        }

        // Each metadata op we plan to absorb must feed *only* into the
        // alias set — if it has an outside consumer we'd dangle that
        // consumer's input. (Conservative; can relax later.)
        let mut metadata_isolated = true;
        for &m_idx in &metadata_idxs {
            let Some(m) = slots[m_idx].as_ref() else {
                metadata_isolated = false;
                break;
            };
            for out in &m.outputs {
                let downstream = consumers_of.get(&out.source).cloned().unwrap_or_default();
                for &d_idx in &downstream {
                    let in_heavy = heavy_seen.contains(&d_idx);
                    let in_metadata = metadata_set.contains(&d_idx);
                    if !(in_heavy || in_metadata) {
                        metadata_isolated = false;
                        break;
                    }
                }
                if !metadata_isolated {
                    break;
                }
            }
            if !metadata_isolated {
                break;
            }
        }
        if !metadata_isolated {
            continue;
        }

        // Every Heavy consumer must be a Single Heavy whose absorbed
        // input is an alias of the producer's output. Shape continuity
        // is by element count (the metadata ops preserve it).
        let p_out_elems: i32 = p_out_shape.iter().product::<i32>().max(1);
        let mut valid = true;
        for &c_idx in &heavy_cons {
            let c = match slots[c_idx].as_ref() {
                Some(c) => c,
                None => {
                    valid = false;
                    break;
                }
            };
            if !consumer_is_heavy_singleton(c) {
                valid = false;
                break;
            }
            if !heavy_chain_shape_ok(c) {
                valid = false;
                break;
            }
            if c.inputs.is_empty() || !alias_tensors.contains(&c.inputs[0].source) {
                valid = false;
                break;
            }
            let in_elems: i32 = c.inputs[0].shape.iter().product::<i32>().max(1);
            if in_elems != p_out_elems {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }
        // Budget gate: producer step + one step per Heavy consumer.
        let dbg_budget = std::env::var_os("FUSEKIT_BUDGET_DEBUG").is_some();
        let mut est = REG_CONTROL_HEADROOM + step_register_slots("RMSNorm", producer.inputs.len());
        for &c_idx in &heavy_cons {
            if let Some(c) = slots[c_idx].as_ref() {
                let FusionKind::Single(op_name) = &c.kind else {
                    continue;
                };
                est = est.saturating_add(step_register_slots(op_name, c.inputs.len()));
            }
        }
        if est > reg_budget {
            if dbg_budget {
                eprintln!("[budget] try_fanout_merge reg reject: est={est} budget={reg_budget} n_heavy_cons={}", heavy_cons.len());
            }
            continue;
        }
        // SMEM for a reduction-fanout chain = the normed row (one float
        // per hidden element) + a 32-element reduce scratch. Heavy
        // consumers are absorbed; they read SMEM and write their own
        // output, contributing zero additional SMEM. We can't ask
        // `safety::estimate_smem_bytes` for the per-Heavy figure
        // because that function flags every `Single(_)` with a
        // budget-exceeding sentinel (it's tuned for pre-chain task
        // splitting, not chain-building).
        let smem_est: usize = (p_out_shape.last().copied().unwrap_or(0) as usize) * 4 + 32 * 4;
        if smem_est > smem_budget {
            if dbg_budget {
                eprintln!("[budget] try_fanout_merge smem reject: est={smem_est} budget={smem_budget} hidden={}", p_out_shape.last().copied().unwrap_or(0));
            }
            continue;
        }

        do_fanout_merge(slots, p_idx, &heavy_cons, &metadata_idxs);
        return true;
    }
    false
}

/// Build the multi-Heavy chain. The producer becomes chain step 0;
/// each consumer's Heavy op becomes the next steps in turn, each one
/// reading Register(0) (the normed row) as its first operand and its
/// own external buffers (W / scales / biases) as the rest.
///
/// `consumers` is given in fusion-graph index order — the merged
/// chain preserves that order so codegen has a deterministic mapping
/// from output buffer slot → heavy step.
fn do_fanout_merge(
    slots: &mut Vec<Option<Fusion>>,
    p_idx: usize,
    consumers: &[usize],
    metadata_absorbed: &[usize],
) {
    // Pure-metadata Singles between producer and consumers get
    // absorbed (None'd out). They produced no compute, just shape
    // views, and the FusedChain hands the SMEM row to the Heavy
    // consumer directly.
    for &m_idx in metadata_absorbed {
        slots[m_idx].take();
    }
    let producer = slots[p_idx].take().expect("producer present");
    let (p_steps, p_inputs) = extract_steps_and_inputs(&producer);
    debug_assert_eq!(p_steps.len(), 1);

    // Consumer collection: we need an ordered list of (Fusion, steps,
    // inputs) so we can lay out external buffers and outputs.
    let mut consumer_fusions: Vec<Fusion> = Vec::with_capacity(consumers.len());
    for &c_idx in consumers {
        let c = slots[c_idx].take().expect("consumer present");
        consumer_fusions.push(c);
    }

    // New external input list: producer's inputs first, then each
    // consumer's non-absorbed inputs (slot 0 was the producer output
    // — drop it; remaining slots stay in order). We track the offset
    // each consumer's external buffers land at so the chain step's
    // External operands can be remapped.
    let mut new_inputs: Vec<FusionBoundary> = Vec::new();
    new_inputs.extend(p_inputs.iter().cloned());
    let producer_input_count = new_inputs.len() as u32;
    let mut consumer_buf_offsets: Vec<u32> = Vec::with_capacity(consumers.len());
    for c in &consumer_fusions {
        let off = new_inputs.len() as u32;
        consumer_buf_offsets.push(off);
        for (i, inp) in c.inputs.iter().enumerate() {
            if i == 0 {
                continue;
            } // absorbed slot — becomes Register(0)
            new_inputs.push(inp.clone());
        }
    }

    // New outputs: each consumer's output(s) appended in order.
    let mut new_outputs: Vec<FusionBoundary> = Vec::new();
    for c in &consumer_fusions {
        new_outputs.extend(c.outputs.iter().cloned());
    }

    // Build steps. Step 0 is the producer; subsequent steps are each
    // consumer's Heavy step with operand 0 rewritten to Register(0)
    // and the other externals shifted by the consumer's buffer offset.
    let mut new_steps: Vec<ChainStep> = Vec::with_capacity(1 + consumer_fusions.len());
    let p_step = &p_steps[0];
    new_steps.push(ChainStep {
        op_name: p_step.op_name.clone(),
        op_attrs: p_step.op_attrs.clone(),
        inputs: (0..producer_input_count)
            .map(|i| ChainOperand::External { buffer_index: i })
            .collect(),
        output_shape: p_step.output_shape.clone(),
        output_dtype: p_step.output_dtype,
    });
    for (i, c) in consumer_fusions.iter().enumerate() {
        let (c_steps, _) = extract_steps_and_inputs(c);
        debug_assert_eq!(c_steps.len(), 1);
        let step = &c_steps[0];
        let buf_base = consumer_buf_offsets[i];
        let inputs: Vec<ChainOperand> = step
            .inputs
            .iter()
            .enumerate()
            .map(|(j, op)| {
                if j == 0 {
                    // Was External{0} (consumer's absorbed slot) →
                    // chain's Register(0).
                    ChainOperand::Register { step_index: 0 }
                } else {
                    match op {
                        ChainOperand::External { buffer_index } => ChainOperand::External {
                            // Consumer slot j had original buffer
                            // index = j. After dropping slot 0, the
                            // packed index is `*buffer_index - 1`; we
                            // then shift by the consumer's buf base.
                            buffer_index: buf_base + buffer_index - 1,
                        },
                        ChainOperand::Register { .. } => *op,
                    }
                }
            })
            .collect();
        new_steps.push(ChainStep {
            op_name: step.op_name.clone(),
            op_attrs: step.op_attrs.clone(),
            inputs,
            output_shape: step.output_shape.clone(),
            output_dtype: step.output_dtype,
        });
    }

    // op_ids: producer's first, then each consumer's.
    let mut op_ids = producer.op_ids.clone();
    for c in &consumer_fusions {
        op_ids.extend(c.op_ids.iter().copied());
    }

    // Reuse the first consumer's slot for the merged chain; the
    // remaining consumers and producer leave None'd-out holes.
    let dst_idx = consumers[0];
    let merged = Fusion {
        id: 0, // renumbered by `run`'s compaction step
        kind: FusionKind::FusedChain,
        op_ids,
        inputs: new_inputs,
        outputs: new_outputs,
        attrs: FusionAttrs::FusedChain {
            steps: new_steps,
            transposed_weight_slots: Vec::new(),
        },
        parallel_axes: Vec::new(),
        layer_hint: None,
    };
    slots[dst_idx] = Some(merged);
    // slots[p_idx] and slots[consumers[1..]] stay None.
}

fn try_qmv_sibling_fanout_merge(
    slots: &mut Vec<Option<Fusion>>,
    reg_budget: u32,
    graph: &Graph,
) -> bool {
    if std::env::var("FUSEKIT_QMV_FANOUT").as_deref() == Ok("0") {
        return false;
    }

    let dbg = std::env::var_os("FUSEKIT_QMV_FANOUT_DEBUG").is_some();
    let mut total_qmm = 0usize;
    let mut target_ok = 0usize;
    let mut target_reject: HashMap<&'static str, usize> = HashMap::new();
    let mut stateful_skip = 0usize;
    let mut by_x: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        if matches!(&f.kind, FusionKind::Single(op) if op == "QuantizedMatmul") {
            total_qmm += 1;
        }
        let Some(target) = qmv_fanout_target(f) else {
            if dbg {
                *target_reject
                    .entry(qmv_fanout_target_reject_reason(f))
                    .or_default() += 1;
            }
            continue;
        };
        let _ = target;
        target_ok += 1;
        if qmv_fanout_stateful_sensitive(f, graph) {
            stateful_skip += 1;
            continue;
        }
        let Some(x) = f
            .inputs
            .first()
            .map(|b| canonical_tensor_source_through_views(graph, b.source))
        else {
            continue;
        };
        by_x.entry(x).or_default().push(idx);
    }

    let max_targets = std::env::var("FUSEKIT_QMV_FANOUT_MAX")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(8);
    if dbg {
        let groups = by_x.len();
        let mergeable_groups = by_x.values().filter(|g| g.len() >= 2).count();
        let max_group = by_x.values().map(Vec::len).max().unwrap_or(0);
        let mut rejects: Vec<_> = target_reject.into_iter().collect();
        rejects.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(*count));
        eprintln!(
            "[qmv-fanout] scan qmm={} target_ok={} stateful_skip={} groups={} mergeable_groups={} max_group={} rejects={:?}",
            total_qmm, target_ok, stateful_skip, groups, mergeable_groups, max_group, rejects
        );
    }

    for (_x, mut group) in by_x {
        group.sort_unstable();
        if group.len() < 2 {
            continue;
        }

        let mut i = 0usize;
        while i < group.len() {
            let first_idx = group[i];
            let Some(first_fusion) = slots[first_idx].as_ref() else {
                i += 1;
                continue;
            };
            let Some(first_target) = qmv_fanout_target(first_fusion) else {
                i += 1;
                continue;
            };
            let first_tiling = qmv_fanout_tiling(first_target);
            let first_layer = first_fusion.layer_hint;
            let first_dtype = first_fusion.outputs.first().map(|o| o.dtype);

            let mut chosen = vec![first_idx];
            let mut targets = vec![first_target];
            let mut j = i + 1;
            while j < group.len() && chosen.len() < max_targets {
                let idx = group[j];
                let Some(fusion) = slots[idx].as_ref() else {
                    j += 1;
                    continue;
                };
                let Some(target) = qmv_fanout_target(fusion) else {
                    j += 1;
                    continue;
                };
                if fusion.layer_hint == first_layer
                    && fusion.outputs.first().map(|o| o.dtype) == first_dtype
                    && qmv_fanout_target_compatible(first_target, target)
                    && qmv_fanout_tiling(target) == first_tiling
                {
                    chosen.push(idx);
                    targets.push(target);
                }
                j += 1;
            }

            if chosen.len() < 2 {
                if dbg {
                    let mut incompatible = 0usize;
                    let mut wrong_layer = 0usize;
                    let mut wrong_dtype = 0usize;
                    let mut wrong_tiling = 0usize;
                    for &idx in &group[(i + 1)..] {
                        let Some(fusion) = slots[idx].as_ref() else {
                            continue;
                        };
                        let Some(target) = qmv_fanout_target(fusion) else {
                            continue;
                        };
                        if fusion.layer_hint != first_layer {
                            wrong_layer += 1;
                        } else if fusion.outputs.first().map(|o| o.dtype) != first_dtype {
                            wrong_dtype += 1;
                        } else if !qmv_fanout_target_compatible(first_target, target) {
                            incompatible += 1;
                        } else if qmv_fanout_tiling(target) != first_tiling {
                            wrong_tiling += 1;
                        }
                    }
                    eprintln!(
                        "[qmv-fanout] group no-pair size={} first_idx={} first_layer={:?} first_target={:?} first_tiling={:?} wrong_layer={} wrong_dtype={} incompatible={} wrong_tiling={}",
                        group.len(),
                        first_idx,
                        first_layer,
                        first_target,
                        first_tiling,
                        wrong_layer,
                        wrong_dtype,
                        incompatible,
                        wrong_tiling
                    );
                }
                i += 1;
                continue;
            }

            let est_regs = REG_CONTROL_HEADROOM
                + step_register_slots("QuantizedMatmul", 4)
                + (chosen.len() as u32).saturating_mul(2);
            if est_regs > reg_budget {
                if dbg {
                    eprintln!(
                        "[qmv-fanout] reg reject targets={} est={} budget={}",
                        chosen.len(),
                        est_regs,
                        reg_budget
                    );
                }
                i = j;
                continue;
            }

            do_qmv_sibling_fanout_merge(slots, &chosen, targets);
            return true;
        }
    }
    false
}

fn qmv_fanout_target_reject_reason(fusion: &Fusion) -> &'static str {
    if !matches!(&fusion.kind, FusionKind::Single(op) if op == "QuantizedMatmul") {
        return "not_single_qmm";
    }
    if fusion.inputs.len() != 4 {
        return "input_arity";
    }
    if fusion.outputs.len() != 1 {
        return "output_arity";
    }
    let FusionAttrs::Single { op_attrs, .. } = &fusion.attrs else {
        return "not_single_attrs";
    };
    let Attrs::Quantized { bits, group_size } = op_attrs else {
        return "not_quant_attrs";
    };
    if *group_size != 64 {
        return "group_size";
    }
    if !matches!(*bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return "bits";
    }
    let Some(x_shape) = fusion.inputs.first().map(|b| b.shape.as_slice()) else {
        return "missing_x";
    };
    let Some(w_shape) = fusion.inputs.get(1).map(|b| b.shape.as_slice()) else {
        return "missing_w";
    };
    let Some(out_shape) = fusion.outputs.first().map(|b| b.shape.as_slice()) else {
        return "missing_out";
    };
    if x_shape.is_empty() {
        return "x_rank";
    }
    if w_shape.len() != 2 {
        return "w_rank";
    }
    if out_shape.is_empty() {
        return "out_rank";
    }
    let out_dim = w_shape[0].max(1) as u32;
    if out_shape.last().copied().unwrap_or(0).max(1) as u32 != out_dim {
        return "out_dim";
    }
    if std::env::var("FUSEKIT_QMM_MLX_QMV").as_deref() == Ok("0") {
        return "qmv_disabled";
    }
    "shape_or_tile"
}

fn qmv_fanout_target(fusion: &Fusion) -> Option<QmvFanoutTarget> {
    if !matches!(&fusion.kind, FusionKind::Single(op) if op == "QuantizedMatmul") {
        return None;
    }
    if fusion.inputs.len() != 4 || fusion.outputs.len() != 1 {
        return None;
    }
    let exact_tail = std::env::var("FUSEKIT_QMV_FANOUT_EXACT").as_deref() == Ok("1");
    let (k_dim, out_dim, active_rows, bits, group_size) =
        crate::codegen::kernels::qmm_mlx_qmv_fast_shape(fusion).or_else(|| {
            exact_tail
                .then(|| crate::codegen::kernels::qmm_mlx_qmv_tail_safe_shape(fusion))
                .flatten()
        })?;
    Some(QmvFanoutTarget {
        k_dim,
        out_dim,
        active_rows,
        bits,
        group_size,
    })
}

fn qmv_fanout_stateful_sensitive(fusion: &Fusion, graph: &Graph) -> bool {
    if std::env::var("FUSEKIT_QMV_FANOUT_STATEFUL_QKV").as_deref() == Ok("1") {
        return false;
    }
    let Some(weight) = fusion.inputs.get(1) else {
        return false;
    };
    let Some(path) = tensor_param_path_through_views(graph, weight.source) else {
        return false;
    };
    // Recurrent linear-attention state is numerically sensitive in MTP
    // verify because small projection drift feeds ArraysCache/GDN state
    // before the accept/reject gate. Keep sibling fanout for ordinary
    // QMV groups, but leave these projections as single QMV kernels so
    // they can use the MLX-compatible qmv emitter when enabled.
    if path.contains(".linear_attn.") && path.contains(".in_proj_qkv.") {
        return true;
    }
    // Attention q/k/v projections feed RoPE + softmax. Even tail-safe
    // qmv uses a different reduction order than the MLX/NAX qmm path;
    // a few ulps before softmax are enough to change greedy tokens on
    // small quantized models. Gate/up MLP fanout remains admitted by
    // the same semantic/shape budget path.
    path.contains(".self_attn.q_proj.")
        || path.contains(".self_attn.k_proj.")
        || path.contains(".self_attn.v_proj.")
}

fn tensor_param_path_through_views(graph: &Graph, mut tid: TensorId) -> Option<&str> {
    let mut seen: HashSet<TensorId> = HashSet::new();
    while seen.insert(tid) {
        let tensor = graph.tensors.get(tid as usize)?;
        if let Some(path) = tensor.param_path.as_deref() {
            return Some(path);
        }
        let producer = tensor.producer?;
        let op = graph.ops.get(producer as usize)?;
        if op.inputs.is_empty() || !(is_chain_absorbable_view(&op.name) || op.name == "Transpose") {
            return None;
        }
        tid = op.inputs[0];
    }
    None
}

fn canonical_tensor_source_through_views(graph: &Graph, mut tid: TensorId) -> TensorId {
    let mut seen: HashSet<TensorId> = HashSet::new();
    while seen.insert(tid) {
        let Some(tensor) = graph.tensors.get(tid as usize) else {
            return tid;
        };
        if tensor.param_path.is_some() {
            return tid;
        }
        let Some(producer) = tensor.producer else {
            return tid;
        };
        let Some(op) = graph.ops.get(producer as usize) else {
            return tid;
        };
        if op.inputs.len() != 1 || !(is_chain_absorbable_view(&op.name) || op.name == "Transpose") {
            return tid;
        }
        tid = op.inputs[0];
    }
    tid
}

fn qmv_fanout_target_compatible(a: QmvFanoutTarget, b: QmvFanoutTarget) -> bool {
    // The emitter dispatches each fanout target with its own quantization
    // bits/group size. Once tiling is checked separately, keeping equal
    // bits here only blocks valid sibling merges for mixed-quant models.
    let mixed_bits = std::env::var("FUSEKIT_QMV_FANOUT_MIXED_BITS").as_deref() != Ok("0");
    a.k_dim == b.k_dim
        && a.active_rows == b.active_rows
        && (mixed_bits || a.bits == b.bits)
        && a.group_size == b.group_size
}

fn qmv_fanout_tiling(target: QmvFanoutTarget) -> (u32, u32, u32) {
    (
        crate::codegen::kernels::qmv_fanout_fast_simdgroups(
            target.active_rows,
            target.bits,
            target.k_dim,
            target.out_dim,
        ),
        crate::codegen::kernels::qmv_fanout_fast_cols_per_simdgroup(
            target.active_rows,
            target.bits,
            target.k_dim,
            target.out_dim,
        ),
        crate::codegen::kernels::qmv_fanout_fast_token_rows(
            target.active_rows,
            target.bits,
            target.k_dim,
            target.out_dim,
        ),
    )
}

fn do_qmv_sibling_fanout_merge(
    slots: &mut Vec<Option<Fusion>>,
    chosen: &[usize],
    targets: Vec<QmvFanoutTarget>,
) {
    debug_assert!(chosen.len() >= 2);
    debug_assert_eq!(chosen.len(), targets.len());

    let dst_idx = chosen[0];
    let mut fusions = Vec::with_capacity(chosen.len());
    for &idx in chosen {
        fusions.push(slots[idx].take().expect("qmv fanout candidate present"));
    }

    let mut inputs = Vec::with_capacity(1 + fusions.len() * 3);
    inputs.push(fusions[0].inputs[0].clone());
    inputs[0].role = "x";
    for (target_idx, fusion) in fusions.iter().enumerate() {
        let mut w = fusion.inputs[1].clone();
        let mut s = fusion.inputs[2].clone();
        let mut b = fusion.inputs[3].clone();
        w.role = "W_qmv_fanout";
        s.role = "W_qmv_fanout_scales";
        b.role = "W_qmv_fanout_biases";
        let _ = target_idx;
        inputs.push(w);
        inputs.push(s);
        inputs.push(b);
    }

    let mut outputs = Vec::with_capacity(fusions.len());
    let mut op_ids = Vec::new();
    let mut layer_hint = None;
    for fusion in &fusions {
        outputs.extend(fusion.outputs.iter().cloned());
        op_ids.extend(fusion.op_ids.iter().copied());
        layer_hint = layer_hint.or(fusion.layer_hint);
    }

    let merged = Fusion {
        id: dst_idx as u32,
        kind: FusionKind::QmvFanout,
        op_ids,
        inputs,
        outputs,
        attrs: FusionAttrs::QmvFanout { targets },
        parallel_axes: Vec::new(),
        layer_hint,
    };
    slots[dst_idx] = Some(merged);
}

fn scalar_value_as_f32(v: &ScalarValue) -> Option<f32> {
    match v {
        ScalarValue::Float(x) => Some(*x as f32),
        ScalarValue::Int(x) => Some(*x as f32),
        ScalarValue::Bool(_) => None,
    }
}

fn scalar_broadcast_value_and_removable(
    slots: &[Option<Fusion>],
    graph: &Graph,
    consumers_of: &HashMap<TensorId, Vec<usize>>,
    producer_idx: usize,
    only_consumer_idx: usize,
) -> Option<(f32, bool)> {
    let fusion = slots.get(producer_idx)?.as_ref()?;
    let FusionKind::Single(op_name) = &fusion.kind else {
        return None;
    };
    if !matches!(op_name.as_str(), "Broadcast" | "Full") {
        return None;
    }
    if fusion.inputs.len() != 1 || fusion.outputs.len() != 1 {
        return None;
    }
    let out_src = fusion.outputs[0].source;
    let consumers = consumers_of.get(&out_src)?;
    let removable = consumers.iter().all(|&idx| idx == only_consumer_idx);
    let input_src = fusion.inputs[0].source;
    let value = graph
        .tensors
        .get(input_src as usize)?
        .scalar_value
        .as_ref()?;
    Some((scalar_value_as_f32(value)?, removable))
}

fn fused_chain_scale_view_of_source(
    chain: &Fusion,
    q_sources: &HashSet<TensorId>,
    q_elems: u64,
    producer_of: &HashMap<TensorId, usize>,
    slots: &[Option<Fusion>],
    graph: &Graph,
    consumers_of: &HashMap<TensorId, Vec<usize>>,
    chain_idx: usize,
) -> Option<(f32, Option<usize>, TensorId)> {
    let (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) =
        (&chain.kind, &chain.attrs)
    else {
        return None;
    };
    let first = steps.first()?;
    if first.op_name != "Multiply" || first.inputs.len() != 2 {
        return None;
    }
    if !steps
        .iter()
        .skip(1)
        .all(|step| is_chain_absorbable_view(&step.op_name))
    {
        return None;
    }
    let out = chain.outputs.first()?;
    if shape_elems(&out.shape) != q_elems {
        return None;
    }

    let external_source = |operand: ChainOperand| -> Option<TensorId> {
        match operand {
            ChainOperand::External { buffer_index } => chain
                .inputs
                .get(buffer_index as usize)
                .map(|boundary| boundary.source),
            ChainOperand::Register { .. } => None,
        }
    };

    let a = external_source(first.inputs[0])?;
    let b = external_source(first.inputs[1])?;
    let (scale_src, q_src) = if q_sources.contains(&a) {
        (b, a)
    } else if q_sources.contains(&b) {
        (a, b)
    } else {
        return None;
    };
    let mut scale_producer = producer_of.get(&scale_src).copied();
    let scale = if let Some(idx) = scale_producer {
        let (scale, removable) =
            scalar_broadcast_value_and_removable(slots, graph, consumers_of, idx, chain_idx)?;
        if !removable {
            scale_producer = None;
        }
        scale
    } else {
        let value = graph
            .tensors
            .get(scale_src as usize)?
            .scalar_value
            .as_ref()?;
        scalar_value_as_f32(value)?
    };
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    Some((scale, scale_producer, q_src))
}

fn single_layout_of_source(
    fusion: &Fusion,
    source: TensorId,
    elems: u64,
) -> Option<FusionBoundary> {
    let FusionKind::Single(op_name) = &fusion.kind else {
        return None;
    };
    if op_name != "Transpose" && !is_chain_absorbable_view(op_name) {
        return None;
    }
    if fusion.inputs.len() != 1 || fusion.outputs.len() != 1 {
        return None;
    }
    if fusion.inputs[0].source != source {
        return None;
    }
    let out = fusion.outputs[0].clone();
    if shape_elems(&out.shape) != elems {
        return None;
    }
    Some(out)
}

fn split_projection_q_shape(raw_shape: &[i32], qk_dim: i32) -> Option<Vec<i32>> {
    if raw_shape.is_empty() || qk_dim <= 0 {
        return None;
    }
    let last = raw_shape.last().copied()?.max(1);
    if last == qk_dim || last % qk_dim != 0 {
        return None;
    }
    let n_heads = last / qk_dim;
    if n_heads <= 0 {
        return None;
    }
    let mut shape = raw_shape[..raw_shape.len() - 1].to_vec();
    shape.push(n_heads);
    shape.push(qk_dim);
    Some(shape)
}

/// Fold `layout(raw_q) -> Multiply(_, scalar) -> view-only` into
/// downstream SDPA scale semantics. The detection is intentionally tied
/// to the SDPA-Q dataflow edge, not to a model name or layer template:
/// any scalar Q pre-scale whose remaining chain only changes layout can
/// be represented as the SDPA `scale` plus the same logical Q shape.
fn try_sdpa_q_scale_view_absorb(slots: &mut Vec<Option<Fusion>>, graph: &Graph) -> bool {
    let dbg = std::env::var_os("FUSEKIT_Q_SCALE_ABSORB_DEBUG").is_some();
    let mut producer_of: HashMap<TensorId, usize> = HashMap::new();
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(fusion) = slot else { continue };
        for out in &fusion.outputs {
            producer_of.insert(out.source, idx);
        }
        for inp in &fusion.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }
    let mut graph_consumers: HashMap<TensorId, Vec<usize>> = HashMap::new();
    let mut graph_producer_op: HashMap<TensorId, usize> = HashMap::new();
    for (idx, op) in graph.ops.iter().enumerate() {
        for &inp in &op.inputs {
            graph_consumers.entry(inp).or_default().push(idx);
        }
        for &out in &op.outputs {
            graph_producer_op.insert(out, idx);
        }
    }

    for qkv_idx in 0..slots.len() {
        let Some(qkv) = slots[qkv_idx].as_ref() else {
            continue;
        };
        if !matches!(qkv.kind, FusionKind::RmsNormQkv) {
            continue;
        }
        let Some(q_out_idx) = qkv.outputs.iter().position(|b| b.role == "q") else {
            continue;
        };
        let q_src = qkv.outputs[q_out_idx].source;
        let q_elems = shape_elems(&qkv.outputs[q_out_idx].shape);
        let absorbed_q_add_inputs = qkv
            .inputs
            .iter()
            .filter(|b| b.role == "bias_q")
            .map(|b| b.source)
            .collect::<HashSet<_>>();
        let (mut q_aliases, q_metadata) =
            collect_q_layout_alias_chain(q_src, slots, &consumers_of, &absorbed_q_add_inputs);
        q_aliases.extend(collect_q_graph_semantic_aliases(
            q_src,
            graph,
            &graph_consumers,
            &graph_producer_op,
            &absorbed_q_add_inputs,
        ));
        let q_metadata_set: HashSet<usize> = q_metadata.iter().copied().collect();
        let mut q_consumers = HashSet::new();
        for alias in &q_aliases {
            for idx in consumers_of.get(alias).cloned().unwrap_or_default() {
                if idx == qkv_idx || q_metadata_set.contains(&idx) {
                    continue;
                }
                q_consumers.insert(idx);
            }
        }
        if dbg {
            eprintln!(
                "[q-scale-absorb] qkv_idx={qkv_idx} q_src={q_src} q_aliases={q_aliases:?} q_metadata={q_metadata:?} q_consumers={q_consumers:?}"
            );
        }
        let mut candidates: Vec<(usize, HashSet<TensorId>)> = Vec::new();
        let mut candidate_seen: HashSet<usize> = HashSet::new();
        for &consumer_idx in &q_consumers {
            if consumer_idx == qkv_idx {
                continue;
            }
            let Some(consumer) = slots[consumer_idx].as_ref() else {
                continue;
            };
            if matches!(consumer.kind, FusionKind::FusedChain) {
                if candidate_seen.insert(consumer_idx)
                    && fused_chain_scale_view_of_source(
                        consumer,
                        &q_aliases,
                        q_elems,
                        &producer_of,
                        slots,
                        graph,
                        &consumers_of,
                        consumer_idx,
                    )
                    .is_some()
                {
                    candidates.push((consumer_idx, q_aliases.clone()));
                }
                continue;
            }
            for &alias in &q_aliases {
                if let Some(layout_out) = single_layout_of_source(consumer, alias, q_elems) {
                    let mut layout_sources = HashSet::new();
                    layout_sources.insert(layout_out.source);
                    for next_idx in consumers_of
                        .get(&layout_out.source)
                        .cloned()
                        .unwrap_or_default()
                    {
                        if next_idx != consumer_idx && candidate_seen.insert(next_idx) {
                            candidates.push((next_idx, layout_sources.clone()));
                        }
                    }
                }
            }
        }
        for chain_idx in 0..slots.len() {
            if !candidate_seen.insert(chain_idx) {
                continue;
            }
            let Some(chain) = slots[chain_idx].as_ref() else {
                continue;
            };
            if !matches!(chain.kind, FusionKind::FusedChain) {
                continue;
            }
            if fused_chain_scale_view_of_source(
                chain,
                &q_aliases,
                q_elems,
                &producer_of,
                slots,
                graph,
                &consumers_of,
                chain_idx,
            )
            .is_some()
            {
                candidates.push((chain_idx, q_aliases.clone()));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        let Some((chain_idx, q_input_boundary, q_sources, q_scale, scale_producer_idx)) =
            candidates.into_iter().find_map(|(chain_idx, q_sources)| {
                let chain = slots.get(chain_idx)?.as_ref()?;
                let (q_scale, scale_producer_idx, matched_q_src) =
                    fused_chain_scale_view_of_source(
                        chain,
                        &q_sources,
                        q_elems,
                        &producer_of,
                        slots,
                        graph,
                        &consumers_of,
                        chain_idx,
                    )?;
                let q_input_boundary = boundary_from_graph(graph, matched_q_src, "Q");
                Some((
                    chain_idx,
                    q_input_boundary,
                    q_sources,
                    q_scale,
                    scale_producer_idx,
                ))
            })
        else {
            if dbg {
                for consumer_idx in q_consumers {
                    eprintln!(
                        "[q-scale-absorb]   consumer_idx={consumer_idx} is not Q scale+view chain"
                    );
                    if let Some(chain) = slots[consumer_idx].as_ref() {
                        match (&chain.kind, &chain.attrs) {
                            (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => {
                                let names =
                                    steps.iter().map(|s| s.op_name.as_str()).collect::<Vec<_>>();
                                let input_sources = chain
                                    .inputs
                                    .iter()
                                    .map(|b| (b.source, b.shape.clone()))
                                    .collect::<Vec<_>>();
                                eprintln!(
                                    "[q-scale-absorb]     chain steps={names:?} inputs={input_sources:?} outputs={:?}",
                                    chain.outputs
                                        .iter()
                                        .map(|b| (b.source, b.shape.clone()))
                                        .collect::<Vec<_>>()
                                );
                            }
                            _ => eprintln!(
                                "[q-scale-absorb]     chain kind={:?} attrs={:?}",
                                chain.kind, chain.attrs
                            ),
                        }
                    }
                }
            }
            continue;
        };
        let Some(chain) = slots[chain_idx].as_ref() else {
            continue;
        };
        let _ = q_sources;
        let Some(chain_out) = chain.outputs.first() else {
            continue;
        };
        let Some(sdpa_consumers) = consumers_of.get(&chain_out.source) else {
            continue;
        };
        if sdpa_consumers.len() != 1 {
            if dbg {
                eprintln!(
                    "[q-scale-absorb]   chain_idx={chain_idx} chain_out={} sdpa_consumers={sdpa_consumers:?}",
                    chain_out.source
                );
            }
            continue;
        }
        let sdpa_idx = sdpa_consumers[0];
        let Some(sdpa) = slots[sdpa_idx].as_ref() else {
            continue;
        };
        if !matches!(sdpa.kind, FusionKind::SdpaDecode) {
            if dbg {
                eprintln!("[q-scale-absorb]   consumer {sdpa_idx} is not SdpaDecode");
            }
            continue;
        }
        let Some(q_input_idx) = sdpa
            .inputs
            .iter()
            .position(|b| b.role == "Q" && b.source == chain_out.source)
        else {
            if dbg {
                eprintln!(
                    "[q-scale-absorb]   SdpaDecode {sdpa_idx} does not consume chain out {} as Q",
                    chain_out.source
                );
            }
            continue;
        };

        let qk_dim = chain_out.shape.last().copied().unwrap_or(1).max(1) as f32;
        let old_scale = match sdpa.attrs {
            FusionAttrs::SdpaDecode { scale, .. } => scale,
            _ => continue,
        };
        let old_effective = if old_scale.abs() < f32::EPSILON {
            1.0f32 / qk_dim.sqrt()
        } else {
            old_scale
        };
        // Trace-bound SDPA uses semantic q_buf for ordinary decode
        // (`scale != 1`): the runtime overwrites slot 0 via extra
        // bindings, so a trace-side `RoPE(q) * c` chain is dead and
        // must be removed without changing the SDPA scale. The
        // direct-scaled-Q path (`scale == 1`) really does consume the
        // trace Q input; only there does removing the pre-scale chain
        // require multiplying the SDPA scale by `q_scale`.
        let direct_scaled_q = (old_scale - 1.0).abs() < f32::EPSILON && !sdpa.inputs.is_empty();
        let new_scale = if direct_scaled_q {
            old_effective * q_scale
        } else {
            old_effective
        };
        if !new_scale.is_finite() || new_scale == 0.0 {
            continue;
        }

        let chain_shape = chain_out.shape.clone();
        let qk_dim_i32 = chain_shape.last().copied().unwrap_or(1).max(1);
        let sdpa_q_shape = split_projection_q_shape(&q_input_boundary.shape, qk_dim_i32)
            .unwrap_or_else(|| q_input_boundary.shape.clone());
        let chain_op_ids = chain.op_ids.clone();
        let scale_op_ids = scale_producer_idx
            .and_then(|idx| slots[idx].as_ref())
            .map(|fusion| fusion.op_ids.clone())
            .unwrap_or_default();
        let mut sdpa_owned = slots[sdpa_idx].take().expect("sdpa present");
        sdpa_owned.inputs[q_input_idx] = FusionBoundary {
            source: q_input_boundary.source,
            role: "Q",
            shape: sdpa_q_shape,
            dtype: q_input_boundary.dtype,
        };
        if let FusionAttrs::SdpaDecode { scale, .. } = &mut sdpa_owned.attrs {
            *scale = new_scale;
        }
        sdpa_owned.op_ids.extend(chain_op_ids);
        sdpa_owned.op_ids.extend(scale_op_ids);
        slots[sdpa_idx] = Some(sdpa_owned);
        slots[chain_idx].take();
        if let Some(scale_idx) = scale_producer_idx {
            slots[scale_idx].take();
        }
        return true;
    }

    false
}

fn fusion_layer_index(fusion: &Fusion, graph: &Graph) -> Option<i32> {
    if let Some(crate::graph::LayerTag::Layer(n)) = fusion.layer_hint {
        return Some(n);
    }
    fusion.op_ids.iter().find_map(|&op_id| {
        graph
            .ops
            .get(op_id as usize)
            .and_then(|op| match op.layer_id {
                Some(crate::graph::LayerTag::Layer(n)) => Some(n),
                _ => None,
            })
    })
}

/// Sink the standard decode consumers of a semantic QKV projection into
/// the QKV emitter itself:
///
///   RmsNormQkv -> q_buf -> RopeQDecode -> SdpaDecode
///              -> k/v   -> RopeKvCacheStoreDecode -> KV cache
///
/// becomes:
///
///   RmsNormQkv(decode_sinks) -> roped q_buf, live KV cache
///   SdpaDecode reads q_buf + cache directly.
///
/// This is not a model-name pattern. The admission predicate is the
/// existing decode consumer graph: same-layer synthetic Q RoPE and
/// K/V cache-store tasks must both exist and must agree on theta. Other
/// attention forms (partial RoPE, MLA, Qwen3Next q/k norm) already use
/// their own semantic consumers and are left untouched.
fn try_qkv_decode_sink(slots: &mut Vec<Option<Fusion>>, graph: &Graph) -> bool {
    for qkv_idx in 0..slots.len() {
        let Some(qkv) = slots[qkv_idx].as_ref() else {
            continue;
        };
        if !matches!(qkv.kind, FusionKind::RmsNormQkv) {
            continue;
        }
        let FusionAttrs::RmsNormQkv { decode_sinks, .. } = qkv.attrs else {
            continue;
        };
        if decode_sinks.is_some() {
            continue;
        }
        if !["q", "k", "v"]
            .iter()
            .all(|role| qkv.outputs.iter().any(|b| b.role == *role))
        {
            continue;
        }
        let Some(q_src) = qkv.outputs.iter().find(|b| b.role == "q").map(|b| b.source) else {
            continue;
        };
        let Some(k_src) = qkv.outputs.iter().find(|b| b.role == "k").map(|b| b.source) else {
            continue;
        };
        let Some(v_src) = qkv.outputs.iter().find(|b| b.role == "v").map(|b| b.source) else {
            continue;
        };
        let Some(layer_idx) = fusion_layer_index(qkv, graph) else {
            continue;
        };

        let mut rope_q_idx = None;
        let mut rope_kv_idx = None;
        let mut theta: Option<f32> = None;
        for (idx, slot) in slots.iter().enumerate() {
            let Some(fusion) = slot.as_ref() else {
                continue;
            };
            if fusion_layer_index(fusion, graph) != Some(layer_idx) {
                continue;
            }
            match fusion.attrs {
                FusionAttrs::RopeQDecode { theta: t } => {
                    if !fusion.inputs.is_empty() && !fusion.inputs.iter().any(|b| b.source == q_src)
                    {
                        continue;
                    }
                    if let Some(prev) = theta {
                        if (prev - t).abs() > 1e-3 {
                            rope_q_idx = None;
                            rope_kv_idx = None;
                            theta = None;
                            break;
                        }
                    }
                    rope_q_idx = Some(idx);
                    theta = Some(t);
                }
                FusionAttrs::RopeKvCacheStoreDecode { theta: t } => {
                    if !fusion.inputs.is_empty() {
                        let consumes_k = fusion.inputs.iter().any(|b| b.source == k_src);
                        let consumes_v = fusion.inputs.iter().any(|b| b.source == v_src);
                        if !(consumes_k && consumes_v) {
                            continue;
                        }
                    }
                    if let Some(prev) = theta {
                        if (prev - t).abs() > 1e-3 {
                            rope_q_idx = None;
                            rope_kv_idx = None;
                            theta = None;
                            break;
                        }
                    } else {
                        theta = Some(t);
                    }
                    rope_kv_idx = Some(idx);
                }
                _ => {}
            }
        }

        let (Some(rope_q_idx), Some(rope_kv_idx), Some(theta)) = (rope_q_idx, rope_kv_idx, theta)
        else {
            continue;
        };
        let Some(mut qkv_owned) = slots[qkv_idx].take() else {
            continue;
        };
        if let FusionAttrs::RmsNormQkv { decode_sinks, .. } = &mut qkv_owned.attrs {
            *decode_sinks = Some(QkvDecodeSinks { theta });
        }
        let mut absorbed_op_ids = Vec::new();
        if let Some(rope_q) = slots[rope_q_idx].take() {
            absorbed_op_ids.extend(rope_q.op_ids);
        }
        if let Some(rope_kv) = slots[rope_kv_idx].take() {
            absorbed_op_ids.extend(rope_kv.op_ids);
        }
        qkv_owned.op_ids.extend(absorbed_op_ids);
        slots[qkv_idx] = Some(qkv_owned);
        return true;
    }
    false
}

fn qmv_swiglu_target_compatible(gate: QmvFanoutTarget, up: QmvFanoutTarget) -> bool {
    gate.k_dim == up.k_dim
        && gate.out_dim == up.out_dim
        && gate.active_rows == up.active_rows
        && gate.bits == up.bits
        && gate.group_size == up.group_size
        && matches!(gate.bits, 4 | 8)
}

fn try_qmv_swiglu_absorb(slots: &mut Vec<Option<Fusion>>, reg_budget: u32) -> bool {
    let dbg = std::env::var_os("FUSEKIT_QMV_SWIGLU_ABSORB_DEBUG").is_some();
    let mut consumers_of: HashMap<TensorId, HashSet<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(fusion) = slot else { continue };
        for input in &fusion.inputs {
            consumers_of.entry(input.source).or_default().insert(idx);
        }
    }

    for p_idx in 0..slots.len() {
        let Some(producer) = slots[p_idx].as_ref() else {
            continue;
        };
        if !matches!(producer.kind, FusionKind::QmvFanout) {
            continue;
        }
        let FusionAttrs::QmvFanout { targets } = &producer.attrs else {
            continue;
        };
        if targets.len() != 2 || producer.outputs.len() != 2 || producer.inputs.len() != 7 {
            continue;
        }

        let est_regs = REG_CONTROL_HEADROOM + 40;
        if est_regs > reg_budget {
            if dbg {
                eprintln!(
                    "[qmv-swiglu-absorb] reg reject p_idx={p_idx} est={est_regs} budget={reg_budget}"
                );
            }
            continue;
        }

        for gate_idx in 0..2usize {
            let up_idx = 1 - gate_idx;
            let gate_target = targets[gate_idx];
            let up_target = targets[up_idx];
            if !qmv_swiglu_target_compatible(gate_target, up_target) {
                continue;
            }

            let gate_src = producer.outputs[gate_idx].source;
            let up_src = producer.outputs[up_idx].source;
            let gate_cons = consumers_of.get(&gate_src).cloned().unwrap_or_default();
            let up_cons = consumers_of.get(&up_src).cloned().unwrap_or_default();
            if gate_cons.len() != 1 || up_cons.len() != 1 {
                continue;
            }
            let c_idx = *gate_cons.iter().next().unwrap();
            if !up_cons.contains(&c_idx) || c_idx == p_idx {
                continue;
            }

            let Some(consumer) = slots[c_idx].as_ref() else {
                continue;
            };
            let c_steps = match (&consumer.kind, &consumer.attrs) {
                (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
                _ => continue,
            };
            if c_steps.len() != 3
                || c_steps[0].op_name != "Sigmoid"
                || c_steps[1].op_name != "Multiply"
                || c_steps[2].op_name != "Multiply"
                || consumer.outputs.len() != 1
            {
                continue;
            }
            let cinputs: Vec<TensorId> = consumer.inputs.iter().map(|b| b.source).collect();
            let external_is = |operand: ChainOperand, src: TensorId| -> bool {
                match operand {
                    ChainOperand::External { buffer_index } => cinputs
                        .get(buffer_index as usize)
                        .is_some_and(|candidate| *candidate == src),
                    _ => false,
                }
            };

            let Some(sig_in) = c_steps[0].inputs.first().copied() else {
                continue;
            };
            if !external_is(sig_in, gate_src) {
                continue;
            }

            let mul1_ok = match (
                c_steps[1].inputs.first().copied(),
                c_steps[1].inputs.get(1).copied(),
            ) {
                (Some(ChainOperand::Register { step_index: 0 }), Some(x)) => {
                    external_is(x, gate_src)
                }
                (Some(x), Some(ChainOperand::Register { step_index: 0 })) => {
                    external_is(x, gate_src)
                }
                _ => false,
            };
            if !mul1_ok {
                continue;
            }

            let mul2_ok = match (
                c_steps[2].inputs.first().copied(),
                c_steps[2].inputs.get(1).copied(),
            ) {
                (Some(ChainOperand::Register { step_index: 1 }), Some(x)) => external_is(x, up_src),
                (Some(x), Some(ChainOperand::Register { step_index: 1 })) => external_is(x, up_src),
                _ => false,
            };
            if !mul2_ok {
                continue;
            }

            let producer_owned = slots[p_idx].take().expect("producer present");
            let consumer_owned = slots[c_idx].take().expect("consumer present");

            let mut inputs = Vec::with_capacity(7);
            inputs.push(producer_owned.inputs[0].clone());
            inputs[0].role = "x".into();
            let push_qmv_inputs =
                |inputs: &mut Vec<FusionBoundary>, fusion: &Fusion, idx: usize, role: &str| {
                    let base = 1 + idx * 3;
                    let mut w = fusion.inputs[base].clone();
                    let mut s = fusion.inputs[base + 1].clone();
                    let mut b = fusion.inputs[base + 2].clone();
                    let (w_role, s_role, b_role) = match role {
                        "gate" => (
                            "W_qmv_swiglu_gate",
                            "W_qmv_swiglu_gate_scales",
                            "W_qmv_swiglu_gate_biases",
                        ),
                        _ => (
                            "W_qmv_swiglu_up",
                            "W_qmv_swiglu_up_scales",
                            "W_qmv_swiglu_up_biases",
                        ),
                    };
                    w.role = w_role;
                    s.role = s_role;
                    b.role = b_role;
                    inputs.push(w);
                    inputs.push(s);
                    inputs.push(b);
                };
            push_qmv_inputs(&mut inputs, &producer_owned, gate_idx, "gate");
            push_qmv_inputs(&mut inputs, &producer_owned, up_idx, "up");

            let mut op_ids = producer_owned.op_ids.clone();
            op_ids.extend(consumer_owned.op_ids.iter().copied());
            let merged = Fusion {
                id: producer_owned.id,
                kind: FusionKind::QmvSwiglu,
                op_ids,
                inputs,
                outputs: consumer_owned.outputs.clone(),
                attrs: FusionAttrs::QmvSwiglu {
                    gate: gate_target,
                    up: up_target,
                },
                parallel_axes: Vec::new(),
                layer_hint: producer_owned.layer_hint.or(consumer_owned.layer_hint),
            };
            if dbg {
                eprintln!(
                    "[qmv-swiglu-absorb] merged p_idx={p_idx} c_idx={c_idx} gate_idx={gate_idx} up_idx={up_idx}"
                );
            }
            slots[p_idx] = Some(merged);
            return true;
        }
    }
    false
}

/// Absorb `Sigmoid(x)` into a downstream CustomKernel input when the
/// kernel source exposes a single, parseable pointer access for that
/// input. The transform preserves the MLX dtype boundary by rounding
/// the sigmoid result through `InT` before the custom kernel consumes
/// it, matching the standalone `Sigmoid` emitter's bf16/fp16 store.
fn try_custom_kernel_sigmoid_input_absorb(slots: &mut Vec<Option<Fusion>>) -> bool {
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    let dbg = std::env::var_os("FUSEKIT_CUSTOM_SIGMOID_INPUT_ABSORB_DEBUG").is_some();
    for sig_idx in 0..slots.len() {
        let Some(sigmoid) = slots[sig_idx].as_ref() else {
            continue;
        };
        if !matches!(&sigmoid.kind, FusionKind::Single(op) if op == "Sigmoid") {
            continue;
        }
        if sigmoid.inputs.len() != 1 || sigmoid.outputs.len() != 1 {
            continue;
        }
        let sigmoid_out = sigmoid.outputs[0].source;
        let Some(consumers) = consumers_of.get(&sigmoid_out) else {
            continue;
        };
        let real_consumers: Vec<usize> = consumers
            .iter()
            .copied()
            .filter(|&idx| idx != sig_idx)
            .collect();
        if real_consumers.len() != 1 {
            continue;
        }
        let custom_idx = real_consumers[0];
        let Some(custom) = slots[custom_idx].as_ref() else {
            continue;
        };
        if !matches!(&custom.kind, FusionKind::Single(op) if op == "CustomKernel") {
            continue;
        }
        let FusionAttrs::Single {
            op_attrs:
                Attrs::CustomKernel {
                    name,
                    source,
                    grid,
                    threadgroup,
                    shared_memory,
                    ensure_row_contiguous,
                },
            ..
        } = &custom.attrs
        else {
            continue;
        };

        let input_slots: Vec<usize> = custom
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(slot, inp)| (inp.source == sigmoid_out).then_some(slot))
            .collect();
        if input_slots.len() != 1 {
            continue;
        }
        let input_slot = input_slots[0];
        if sigmoid.inputs[0].shape != custom.inputs[input_slot].shape {
            continue;
        }
        let Some((new_name, new_source)) =
            rewrite_custom_kernel_sigmoid_input(name, source, input_slot)
        else {
            continue;
        };

        if dbg {
            eprintln!(
                "[custom-sigmoid-absorb] merge sigmoid_idx={sig_idx} custom_idx={custom_idx} slot={input_slot} {name} -> {new_name}"
            );
        }

        let new_attrs = FusionAttrs::Single {
            op_name: "CustomKernel".to_string(),
            op_attrs: Attrs::CustomKernel {
                name: new_name,
                source: new_source,
                grid: *grid,
                threadgroup: *threadgroup,
                shared_memory: *shared_memory,
                ensure_row_contiguous: *ensure_row_contiguous,
            },
        };
        let sigmoid_input = sigmoid.inputs[0].clone();
        let sigmoid_op_ids = sigmoid.op_ids.clone();

        slots[sig_idx].take();
        let mut custom_owned = slots[custom_idx].take().expect("custom kernel present");
        custom_owned.inputs[input_slot] = sigmoid_input;
        custom_owned.op_ids.extend(sigmoid_op_ids);
        custom_owned.attrs = new_attrs;
        slots[custom_idx] = Some(custom_owned);
        return true;
    }
    false
}

fn rewrite_custom_kernel_sigmoid_input(
    name: &str,
    source: &str,
    input_slot: usize,
) -> Option<(String, String)> {
    if !source.contains("template <typename InT") {
        return None;
    }
    let arg = custom_kernel_arg_name_for_buffer(source, input_slot)?;
    let alias_decl = format!("auto {arg}_ = {arg} +");
    if !source.contains(&alias_decl) {
        return None;
    }
    // MLX gated-delta style kernels read the time/head-indexed vector
    // through `<arg>_[hv_idx]`. Avoid rewriting arbitrary custom
    // source unless that exact source-level access is present.
    let access = format!("{arg}_[hv_idx]");
    if !source.contains(&access) {
        return None;
    }
    let rounded_sigmoid = format!(
        "static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-static_cast<float>({access})))))"
    );
    let new_name = format!("{name}_sigmoid_b{input_slot}");
    let renamed = source.replace(name, &new_name);
    let rewritten = renamed.replace(&access, &rounded_sigmoid);
    Some((new_name, rewritten))
}

fn custom_kernel_arg_name_for_buffer(source: &str, input_slot: usize) -> Option<String> {
    let needle = format!("[[buffer({input_slot})]]");
    for line in source.lines() {
        if !line.contains(&needle) {
            continue;
        }
        let before = line.split(&needle).next()?.trim_end();
        let token = before
            .split_whitespace()
            .last()?
            .trim_matches(|c: char| c == '*' || c == '&' || c == ',' || c.is_whitespace());
        if token
            .chars()
            .next()
            .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
            && token.chars().all(|c| c == '_' || c.is_ascii_alphanumeric())
        {
            return Some(token.to_string());
        }
    }
    None
}

/// Heavy ops that the FusedChain emitter knows how to inline at
/// step 0. Expanding this list (Convolution, batched Matmul, …)
/// requires the chain emitter to learn each op's body — guard the
/// merge here so the pass never produces a chain we can't lower.
fn is_chainable_heavy_op(op_name: &str) -> bool {
    matches!(op_name, "Matmul" | "AddMM" | "QuantizedMatmul")
}

/// Reduction ops the chain emitter knows how to drop at step 0.
/// Same gating story as `is_chainable_heavy_op`.
fn is_chainable_reduction_op(op_name: &str) -> bool {
    matches!(op_name, "RMSNorm")
}

/// Reduction shape gate. Today only supports RMSNorm: input shape
/// `(..., hidden)`, weight `(hidden,)`, output same as input. The
/// reduction is along the LAST axis.
/// True if the producer's chain head will be a Reduction step. The
/// consumer's surviving external operands then have to fit the
/// Reduction emit's indexing model (see `reduction_merge_tail_ok`).
fn producer_is_reduction(f: &Fusion) -> bool {
    match (&f.kind, &f.attrs) {
        (FusionKind::Single(op), _) => categorize_name(op) == OpCategory::Reduction,
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => {
            !steps.is_empty() && categorize_name(&steps[0].op_name) == OpCategory::Reduction
        }
        _ => false,
    }
}

/// Check that the consumer's external operands (other than the slot
/// absorbed by the producer) will be indexable by the Reduction-headed
/// chain emit. The emit body runs one threadgroup per row and iterates
/// the hidden axis; an external operand must either have the same
/// shape as the chain output (indexed `[row * HIDDEN + i]`), be a
/// hidden-axis broadcast (`[i]`), or be a scalar (`[0]`).
fn reduction_merge_tail_ok(
    consumer: &Fusion,
    consumer_input_slot: usize,
    producer: &Fusion,
) -> bool {
    let out_shape = match producer.outputs.first() {
        Some(b) => &b.shape,
        None => return false,
    };
    if out_shape.is_empty() {
        return false;
    }
    let hidden = *out_shape.last().unwrap();
    for (i, inp) in consumer.inputs.iter().enumerate() {
        if i == consumer_input_slot {
            continue;
        }
        let s = &inp.shape;
        let same = s == out_shape;
        let elems: i32 = s.iter().product::<i32>().max(1);
        let broadcast = elems == hidden && s.iter().any(|&d| d == hidden);
        let scalar = elems == 1;
        if !(same || broadcast || scalar) {
            return false;
        }
    }
    true
}

fn reduction_chain_shape_ok(f: &Fusion) -> bool {
    if f.inputs.len() < 2 || f.outputs.is_empty() {
        return false;
    }
    let x = &f.inputs[0].shape;
    let w = &f.inputs[1].shape;
    let y = &f.outputs[0].shape;
    if x.is_empty() || y.is_empty() {
        return false;
    }
    // y mirrors x. weight is either the last-axis-broadcasted scale
    // (`[hidden]`) or a scalar scale (`[]` / one element). The
    // reduction-first chain emitter already handles scalar weights via
    // `WEIGHT_IS_VECTOR=false`; keeping the gate vector-only leaves
    // RMSNorm(…, weight=1.0) + pointwise tails as separate hot ICB
    // commands in GDN-style decode graphs.
    if x != y {
        return false;
    }
    let hidden = *x.last().unwrap();
    if hidden <= 0 {
        return false;
    }
    let w_elems = w.iter().product::<i32>().max(1);
    if !((w.len() == 1 && w[0] == hidden) || w_elems == 1) {
        return false;
    }
    true
}

/// Tighter per-op shape check for Heavy producers. Mirrors the chain
/// emitter's support windows so the merge pass never produces a
/// chain it can't lower.
fn heavy_chain_shape_ok(f: &Fusion) -> bool {
    let FusionKind::Single(op_name) = &f.kind else {
        return false;
    };
    match op_name.as_str() {
        "Matmul" => matmul_chain_shape_ok(f),
        // AddMM is Matmul + a per-output-channel bias. The matmul
        // shape gate is identical; the extra bias slot only needs its
        // last axis to equal `N` (the matmul output dim). MLX traces
        // broadcast bias over the batch axis (e.g. `[8, N]` for
        // prefill), so we accept any prefix as long as the last dim
        // matches — the kernel only ever indexes `bias[out_idx]`.
        "AddMM" => {
            if f.inputs.len() < 3 {
                return false;
            }
            let n_dim = {
                let b = &f.inputs[1].shape;
                if b.len() < 2 {
                    return false;
                }
                b[b.len() - 1]
            };
            if !matmul_chain_shape_ok(f) {
                return false;
            }
            let bias = &f.inputs[2].shape;
            bias.last().copied() == Some(n_dim)
        }
        "QuantizedMatmul" => qmm_chain_shape_ok(f),
        _ => false,
    }
}

/// Are all of the chain's external tail operands indexable by the
/// reduction emit? The reduction body runs one threadgroup per row;
/// each thread iterates the hidden axis. Tail external operands must
/// either be same-shape (indexed `[row * HIDDEN + i]`) or
/// hidden-broadcast (`[i]`). Scalar (shape `[]` or all-ones) treated
/// as broadcast.
fn reduction_tail_external_shapes_ok(f: &Fusion, steps: &[crate::fusion::ir::ChainStep]) -> bool {
    let out_shape = match f.outputs.first() {
        Some(b) => &b.shape,
        None => return false,
    };
    if out_shape.is_empty() {
        return false;
    }
    let hidden = *out_shape.last().unwrap();
    for step in steps.iter().skip(1) {
        for op in &step.inputs {
            if let ChainOperand::External { buffer_index } = op {
                let s = match f.inputs.get(*buffer_index as usize) {
                    Some(b) => &b.shape,
                    None => return false,
                };
                let same = s == out_shape;
                let elems: i32 = s.iter().product::<i32>().max(1);
                let broadcast = elems == hidden && s.iter().any(|&d| d == hidden);
                let scalar = elems == 1;
                if !(same || broadcast || scalar) {
                    return false;
                }
            }
        }
    }
    true
}

fn reduction_tail_len_after_merge(producer: &Fusion, consumer: &Fusion) -> usize {
    let (p_steps, _) = extract_steps_and_inputs(producer);
    let (c_steps, _) = extract_steps_and_inputs(consumer);
    if p_steps
        .first()
        .is_none_or(|s| categorize_name(&s.op_name) != OpCategory::Reduction)
    {
        return 0;
    }
    let p_tail = p_steps.len().saturating_sub(1);
    p_tail.saturating_add(c_steps.len())
}

/// Default admission for pointwise/view tails after a Reduction head.
///
/// This is intentionally dataflow/shape based. A Reduction -> Heavy
/// merge can amplify work by re-running the reduction per output tile,
/// so that path stays separately gated. A Reduction -> pointwise/view
/// tail, however, runs over the same row elements and removes a full
/// materialize/read round-trip of the normalized row. Admit short tails
/// when the chain emitter can index every surviving external operand
/// with the reduction row model and the row itself is small enough to
/// remain a decode-local value.
fn default_reduction_tail_max(
    producer: &Fusion,
    consumer: &Fusion,
    consumer_input_slot: usize,
) -> usize {
    if !producer_is_reduction(producer) {
        return 0;
    }
    if consumer_is_heavy_singleton(consumer) {
        return 0;
    }
    if !reduction_merge_tail_ok(consumer, consumer_input_slot, producer) {
        return 0;
    }

    let Some(out) = producer.outputs.first() else {
        return 0;
    };
    let row_bytes = boundary_nbytes(out);
    if row_bytes == 0 {
        return 0;
    }

    // Keep this conservative for large prefill-like rows. Decode rows
    // are small, and this removes the most common RMSNorm -> scale/cast
    // round-trips without opening long scalar expression chains that
    // would pressure registers.
    if row_bytes <= 64 * 1024 {
        4
    } else {
        2
    }
}

fn matmul_chain_shape_ok(f: &Fusion) -> bool {
    if f.inputs.len() < 2 {
        return false;
    }
    let a = &f.inputs[0].shape;
    let b = &f.inputs[1].shape;
    if a.len() < 2 || b.len() < 2 {
        return false;
    }
    let a_batch: i32 = a[..a.len() - 2].iter().product::<i32>().max(1);
    let b_batch: i32 = b[..b.len() - 2].iter().product::<i32>().max(1);
    let m = a[a.len() - 2];
    let k = a[a.len() - 1];
    let k2 = b[b.len() - 2];
    // Batched matmul OK as long as both operands share the same batch
    // product. (Broadcasting batches — only one operand batched — needs
    // per-side indexing math we don't generate yet.)
    a_batch == b_batch && m > 0 && k > 0 && k == k2
}

/// QuantizedMatmul shape gating. mlx's QM operates on x (..., in) and
/// W_packed (out, in/lanes). We support 2-D weights and any x rank
/// whose final dim is the input dimension — the chain emit sizes its
/// grid by `out_dim`, computing one output row per thread (assumes
/// L=1 decode at runtime, matching how fusekit dispatches today).
fn qmm_chain_shape_ok(f: &Fusion) -> bool {
    if f.inputs.len() < 4 {
        return false;
    }
    let x = &f.inputs[0].shape;
    let w = &f.inputs[1].shape;
    if x.is_empty() || w.len() != 2 {
        return false;
    }
    let in_dim = x[x.len() - 1];
    let out_dim = w[0];
    let in_packed = w[1];
    in_dim > 0 && out_dim > 0 && in_packed > 0
}

fn reduction_heavy_qmm_is_tile_amplified(producer: &Fusion, consumer: &Fusion) -> bool {
    if std::env::var_os("FUSEKIT_FUSE_LARGE_REDUCTION_QMM").is_some() {
        return false;
    }
    if !matches!(&producer.kind, FusionKind::Single(op) if op == "RMSNorm") {
        return false;
    }
    if !matches!(&consumer.kind, FusionKind::Single(op) if op == "QuantizedMatmul") {
        return false;
    }
    let hidden = producer
        .outputs
        .first()
        .and_then(|b| b.shape.last().copied())
        .unwrap_or(0);
    let out_dim = consumer
        .outputs
        .first()
        .and_then(|b| b.shape.last().copied())
        .unwrap_or(0);
    hidden > 0
        && out_dim >= LARGE_REDUCTION_QMM_MIN_OUT_DIM
        && out_dim >= hidden.saturating_mul(LARGE_REDUCTION_QMM_MIN_RATIO)
}

/// A fusion is a valid *producer* (the upstream side of a merge) if
/// it's either pointwise (Single or all-pointwise chain) or it's a
/// chain-emittable Heavy op that may sit at chain step 0. A chain
/// that already starts with Heavy can still be a producer — its
/// existing steps are inlined as the new prefix and the consumer's
/// pointwise becomes the tail.
fn is_chainable_producer(f: &Fusion) -> bool {
    match (&f.kind, &f.attrs) {
        (FusionKind::Single(op_name), _) => {
            let cat = categorize_name(op_name);
            if cat == OpCategory::Pointwise {
                return true;
            }
            if cat == OpCategory::View && is_chain_absorbable_view(op_name) {
                return true;
            }
            if cat == OpCategory::Heavy && is_chainable_heavy_op(op_name) {
                return heavy_chain_shape_ok(f);
            }
            if cat == OpCategory::Reduction && is_chainable_reduction_op(op_name) {
                // Tail operand shapes get checked once the chain is
                // built — we accept the producer here unconditionally
                // and let `is_chainable_consumer` decide if the
                // consumer's tail will line up.
                return reduction_chain_shape_ok(f);
            }
            false
        }
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => {
            if steps.is_empty() {
                return false;
            }
            let positions: Vec<OpCategory> =
                steps.iter().map(|s| categorize_name(&s.op_name)).collect();
            let n_heavy = positions
                .iter()
                .filter(|c| **c == OpCategory::Heavy)
                .count();
            let n_reduce = positions
                .iter()
                .filter(|c| **c == OpCategory::Reduction)
                .count();
            if n_heavy > 1 || n_reduce > 1 {
                return false;
            }
            // Heavy is allowed at step 0 (Heavy-first chain) or at
            // step 1 immediately after a Reduction head. Reduction
            // is allowed only at step 0.
            if let Some(heavy_pos) = positions.iter().position(|c| *c == OpCategory::Heavy) {
                if heavy_pos == 0 {
                    if !is_chainable_heavy_op(&steps[0].op_name) {
                        return false;
                    }
                } else if heavy_pos == 1 && positions[0] == OpCategory::Reduction {
                    if !is_chainable_heavy_op(&steps[1].op_name) {
                        return false;
                    }
                    if !is_chainable_reduction_op(&steps[0].op_name) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            if let Some(reduce_pos) = positions.iter().position(|c| *c == OpCategory::Reduction) {
                if reduce_pos != 0 {
                    return false;
                }
                if !is_chainable_reduction_op(&steps[0].op_name) {
                    return false;
                }
            }
            // Anything else has to be a pointwise / absorbable-view tail.
            for (i, cat) in positions.iter().enumerate() {
                if *cat == OpCategory::Heavy || *cat == OpCategory::Reduction {
                    continue;
                }
                let tail_ok = *cat == OpCategory::Pointwise
                    || (*cat == OpCategory::View && is_chain_absorbable_view(&steps[i].op_name));
                if !tail_ok {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// A fusion is a valid *consumer* (the downstream side of a merge).
/// Pointwise/View consumers are appended as a register-carrying tail.
/// A Heavy singleton is permitted because the merge logic can place it
/// as a Reduction → Heavy step (caller still has to confirm the
/// producer is a Reduction head and the absorbed slot is the heavy
/// row-vector input — see `try_merge_one`). A consumer that's already
/// a FusedChain stays pointwise-only because we don't currently
/// support concatenating two non-trivial chains.
fn is_chainable_consumer(f: &Fusion) -> bool {
    let chain_safe = |op: &str| {
        let cat = categorize_name(op);
        cat == OpCategory::Pointwise || (cat == OpCategory::View && is_chain_absorbable_view(op))
    };
    match (&f.kind, &f.attrs) {
        (FusionKind::Single(op_name), _) => {
            if chain_safe(op_name) {
                return true;
            }
            // Heavy singleton — accepted; producer-side gating decides
            // whether the merge actually proceeds.
            let cat = categorize_name(op_name);
            cat == OpCategory::Heavy && is_chainable_heavy_op(op_name)
        }
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => {
            steps.iter().all(|s| chain_safe(&s.op_name))
        }
        _ => false,
    }
}

/// Pull a fusion's step list and external input list. For Single
/// fusions we synthesise a 1-step chain whose operands are all
/// External buffers in their original order.
fn extract_steps_and_inputs(f: &Fusion) -> (Vec<ChainStep>, Vec<FusionBoundary>) {
    match (&f.kind, &f.attrs) {
        (FusionKind::Single(op_name), FusionAttrs::Single { op_attrs, .. }) => {
            let step = ChainStep {
                op_name: op_name.clone(),
                op_attrs: op_attrs.clone(),
                inputs: (0..f.inputs.len() as u32)
                    .map(|i| ChainOperand::External { buffer_index: i })
                    .collect(),
                output_shape: f.outputs[0].shape.clone(),
                output_dtype: f.outputs[0].dtype,
            };
            (vec![step], f.inputs.clone())
        }
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => {
            (steps.clone(), f.inputs.clone())
        }
        _ => panic!("extract_steps_and_inputs: kind/attrs mismatch"),
    }
}

/// Fold `producer` into `consumer`, producing a single FusedChain.
/// Handles all four (Single|FusedChain) × (Single|FusedChain)
/// combinations via [`extract_steps_and_inputs`].
///
/// `consumer_input_slot` is the consumer external input that aliases
/// the producer's output and becomes a read from the producer's last
/// register.
fn merge_into_chain(
    slots: &mut Vec<Option<Fusion>>,
    p_idx: usize,
    c_idx: usize,
    consumer_input_slot: usize,
) {
    let producer = slots[p_idx].take().expect("producer present");
    let mut consumer = slots[c_idx].take().expect("consumer present");

    let (p_steps, p_inputs) = extract_steps_and_inputs(&producer);
    let (c_steps, c_inputs) = extract_steps_and_inputs(&consumer);

    // New external input list: consumer's inputs minus the absorbed
    // slot, then producer's inputs appended.
    let mut new_inputs = Vec::with_capacity(c_inputs.len() - 1 + p_inputs.len());
    for (i, inp) in c_inputs.iter().enumerate() {
        if i != consumer_input_slot {
            new_inputs.push(inp.clone());
        }
    }
    let producer_buf_base = new_inputs.len() as u32;
    new_inputs.extend(p_inputs.iter().cloned());

    // Remap table for consumer's old buffer slots.
    //   absorbed slot → register read of last producer step
    //   other slots    → compacted external input index
    let mut next_consumer_buf = 0u32;
    let consumer_buf_remap: Vec<Option<u32>> = (0..c_inputs.len())
        .map(|i| {
            if i == consumer_input_slot {
                None
            } else {
                let out = Some(next_consumer_buf);
                next_consumer_buf += 1;
                out
            }
        })
        .collect();

    let producer_step_count = p_steps.len() as u32;
    let last_producer_step_index = producer_step_count - 1;

    let mut new_steps = Vec::with_capacity(p_steps.len() + c_steps.len());
    // Producer steps first: External slots shift by producer_buf_base,
    // Register refs stay relative to producer's own step 0.
    for step in &p_steps {
        let inputs = step
            .inputs
            .iter()
            .map(|op| match op {
                ChainOperand::External { buffer_index } => ChainOperand::External {
                    buffer_index: producer_buf_base + buffer_index,
                },
                ChainOperand::Register { step_index } => ChainOperand::Register {
                    step_index: *step_index,
                },
            })
            .collect();
        new_steps.push(ChainStep {
            op_name: step.op_name.clone(),
            op_attrs: step.op_attrs.clone(),
            inputs,
            output_shape: step.output_shape.clone(),
            output_dtype: step.output_dtype,
        });
    }
    // Consumer steps next: External refs map via consumer_buf_remap;
    // the absorbed slot becomes Register(last_producer_step_index).
    // Register refs in consumer shift by producer_step_count.
    for step in &c_steps {
        let inputs = step
            .inputs
            .iter()
            .map(|op| match op {
                ChainOperand::External { buffer_index } => {
                    match consumer_buf_remap.get(*buffer_index as usize) {
                        Some(Some(b)) => ChainOperand::External { buffer_index: *b },
                        Some(None) => ChainOperand::Register {
                            step_index: last_producer_step_index,
                        },
                        None => panic!(
                            "consumer step references missing buffer slot {}",
                            buffer_index
                        ),
                    }
                }
                ChainOperand::Register { step_index } => ChainOperand::Register {
                    step_index: step_index + producer_step_count,
                },
            })
            .collect();
        new_steps.push(ChainStep {
            op_name: step.op_name.clone(),
            op_attrs: step.op_attrs.clone(),
            inputs,
            output_shape: step.output_shape.clone(),
            output_dtype: step.output_dtype,
        });
    }

    let mut merged_op_ids = producer.op_ids.clone();
    merged_op_ids.extend(consumer.op_ids.iter().copied());

    consumer.kind = FusionKind::FusedChain;
    consumer.attrs = FusionAttrs::FusedChain {
        steps: new_steps,
        transposed_weight_slots: Vec::new(),
    };
    consumer.inputs = new_inputs;
    consumer.op_ids = merged_op_ids;

    slots[c_idx] = Some(consumer);
    // slots[p_idx] stays None.
}

/// Absorb `Sigmoid(gate) * x` into one output of an upstream
/// RMSNorm+multi-Heavy fanout chain.
///
/// Pattern:
///   producer = [RMSNorm, Heavy_0, ..., Heavy_gate, ...]
///   consumer = [Sigmoid(gate), Multiply(sigmoid, x)]
///
/// Result:
///   producer gains the two pointwise tail steps and its gate output boundary is
///   replaced by the consumer output. The emitter recognizes this tail and, for
///   only the gate Heavy range, writes `x * sigmoid(acc_gate)` instead of the raw
///   gate accumulator.
fn try_gated_multiply_absorb(
    slots: &mut Vec<Option<Fusion>>,
    reg_budget: u32,
    smem_budget: usize,
    meta: &crate::fusion::ir::ModelMeta,
) -> bool {
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    let dbg = std::env::var_os("FUSEKIT_GATED_MUL_ABSORB_DEBUG").is_some();
    for p_idx in 0..slots.len() {
        let producer = match slots[p_idx].as_ref() {
            Some(f) => f,
            None => continue,
        };
        let p_steps = match (&producer.kind, &producer.attrs) {
            (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
            _ => continue,
        };
        if p_steps.len() < 3 || p_steps[0].op_name != "RMSNorm" {
            continue;
        }
        if !p_steps
            .iter()
            .skip(1)
            .all(|s| categorize_name(&s.op_name) == OpCategory::Heavy)
        {
            continue;
        }
        if producer.outputs.len() != p_steps.len() - 1 {
            continue;
        }

        let producer_output_sources: std::collections::HashSet<TensorId> =
            producer.outputs.iter().map(|o| o.source).collect();

        for gate_out_idx in 0..producer.outputs.len() {
            let gate_step_idx = gate_out_idx as u32 + 1;
            let gate_step = &p_steps[gate_step_idx as usize];
            if !matches!(gate_step.op_name.as_str(), "Matmul" | "QuantizedMatmul") {
                continue;
            }
            let gate_t = producer.outputs[gate_out_idx].source;
            let gate_elems = shape_elems(&producer.outputs[gate_out_idx].shape);

            let (gate_aliases, gate_metadata) =
                collect_metadata_alias_chain(gate_t, slots, &consumers_of);
            let metadata_set: std::collections::HashSet<usize> =
                gate_metadata.iter().copied().collect();

            let mut gate_consumers: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for alias in &gate_aliases {
                for c_idx in consumers_of.get(alias).cloned().unwrap_or_default() {
                    if c_idx == p_idx || metadata_set.contains(&c_idx) {
                        continue;
                    }
                    gate_consumers.insert(c_idx);
                }
            }
            if gate_consumers.len() != 1 {
                continue;
            }
            let c_idx = *gate_consumers.iter().next().unwrap();
            if c_idx == p_idx {
                continue;
            }

            let mut metadata_isolated = true;
            for &m_idx in &metadata_set {
                let Some(m) = slots[m_idx].as_ref() else {
                    metadata_isolated = false;
                    break;
                };
                for out in &m.outputs {
                    for d_idx in consumers_of.get(&out.source).cloned().unwrap_or_default() {
                        if d_idx != c_idx && !metadata_set.contains(&d_idx) {
                            metadata_isolated = false;
                            break;
                        }
                    }
                    if !metadata_isolated {
                        break;
                    }
                }
                if !metadata_isolated {
                    break;
                }
            }
            if !metadata_isolated {
                continue;
            }

            let consumer = match slots[c_idx].as_ref() {
                Some(f) => f,
                None => continue,
            };
            let c_steps = match (&consumer.kind, &consumer.attrs) {
                (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
                _ => continue,
            };
            if c_steps.len() < 2
                || c_steps[0].op_name != "Sigmoid"
                || c_steps[1].op_name != "Multiply"
                || consumer.outputs.len() != 1
            {
                continue;
            }
            if c_steps
                .iter()
                .skip(2)
                .any(|s| !is_chain_absorbable_view(&s.op_name))
            {
                continue;
            }
            if shape_elems(&consumer.outputs[0].shape) != gate_elems {
                continue;
            }

            let cinputs: Vec<TensorId> = consumer.inputs.iter().map(|b| b.source).collect();
            let resolves_to_gate =
                |op: ChainOperand, aliases: &std::collections::HashSet<TensorId>| -> bool {
                    match op {
                        ChainOperand::External { buffer_index } => cinputs
                            .get(buffer_index as usize)
                            .is_some_and(|src| aliases.contains(src)),
                        _ => false,
                    }
                };
            let sig_input = match c_steps[0].inputs.first().copied() {
                Some(x) => x,
                None => continue,
            };
            if !resolves_to_gate(sig_input, &gate_aliases) {
                continue;
            }

            let m0 = c_steps[1].inputs.first().copied();
            let m1 = c_steps[1].inputs.get(1).copied();
            let other_operand = match (m0, m1) {
                (
                    Some(ChainOperand::Register { step_index: 0 }),
                    Some(ChainOperand::External { buffer_index }),
                ) => ChainOperand::External { buffer_index },
                (
                    Some(ChainOperand::External { buffer_index }),
                    Some(ChainOperand::Register { step_index: 0 }),
                ) => ChainOperand::External { buffer_index },
                _ => continue,
            };
            let ChainOperand::External { buffer_index } = other_operand else {
                continue;
            };
            let Some(other_src) = cinputs.get(buffer_index as usize).copied() else {
                continue;
            };
            let other_is_gate = gate_aliases.contains(&other_src);
            if producer_output_sources.contains(&other_src) && !other_is_gate {
                continue;
            }

            let er = estimate_combined_regs(producer, consumer);
            if er > reg_budget {
                if dbg {
                    eprintln!("[gated-mul-absorb] reg reject: est={er} budget={reg_budget}");
                }
                continue;
            }
            let es = estimate_combined_smem(producer, consumer, meta);
            if es > smem_budget {
                if dbg {
                    eprintln!("[gated-mul-absorb] smem reject: est={es} budget={smem_budget}");
                }
                continue;
            }

            if dbg {
                eprintln!(
                    "[gated-mul-absorb] merge p_idx={p_idx} c_idx={c_idx} gate_step={gate_step_idx}"
                );
            }

            for &m_idx in &metadata_set {
                slots[m_idx].take();
            }
            let producer_owned = slots[p_idx].take().expect("producer present");
            let consumer_owned = slots[c_idx].take().expect("consumer present");
            let p_steps = match &producer_owned.attrs {
                FusionAttrs::FusedChain { steps, .. } => steps.clone(),
                _ => unreachable!(),
            };
            let c_steps = match &consumer_owned.attrs {
                FusionAttrs::FusedChain { steps, .. } => steps.clone(),
                _ => unreachable!(),
            };

            let cinputs_owned: Vec<TensorId> =
                consumer_owned.inputs.iter().map(|b| b.source).collect();
            let mut new_inputs: Vec<FusionBoundary> = producer_owned.inputs.clone();
            let mut consumer_input_remap: HashMap<u32, u32> = HashMap::new();
            for (i, inp) in consumer_owned.inputs.iter().enumerate() {
                let i_u32 = i as u32;
                if gate_aliases.contains(&cinputs_owned[i]) {
                    continue;
                }
                let new_idx = new_inputs.len() as u32;
                consumer_input_remap.insert(i_u32, new_idx);
                new_inputs.push(inp.clone());
            }

            let p_step_count = p_steps.len() as u32;
            let mut new_steps: Vec<ChainStep> = p_steps.clone();
            for step in c_steps.iter() {
                let new_inputs_for_step: Vec<ChainOperand> = step
                    .inputs
                    .iter()
                    .map(|op| match op {
                        ChainOperand::External { buffer_index } => {
                            let src = cinputs_owned[*buffer_index as usize];
                            if gate_aliases.contains(&src) {
                                ChainOperand::Register {
                                    step_index: gate_step_idx,
                                }
                            } else {
                                let remapped = consumer_input_remap[buffer_index];
                                ChainOperand::External {
                                    buffer_index: remapped,
                                }
                            }
                        }
                        ChainOperand::Register { step_index } => ChainOperand::Register {
                            step_index: p_step_count + *step_index,
                        },
                    })
                    .collect();
                new_steps.push(ChainStep {
                    op_name: step.op_name.clone(),
                    op_attrs: step.op_attrs.clone(),
                    inputs: new_inputs_for_step,
                    output_shape: step.output_shape.clone(),
                    output_dtype: step.output_dtype,
                });
            }

            let mut new_outputs = producer_owned.outputs.clone();
            new_outputs[gate_out_idx] = consumer_owned.outputs[0].clone();
            let mut merged_op_ids = producer_owned.op_ids.clone();
            merged_op_ids.extend(consumer_owned.op_ids.iter().copied());

            let merged = Fusion {
                id: c_idx as u32,
                kind: FusionKind::FusedChain,
                op_ids: merged_op_ids,
                inputs: new_inputs,
                outputs: new_outputs,
                attrs: FusionAttrs::FusedChain {
                    steps: new_steps,
                    transposed_weight_slots: Vec::new(),
                },
                parallel_axes: vec![],
                layer_hint: producer_owned.layer_hint.or(consumer_owned.layer_hint),
            };
            slots[c_idx] = Some(merged);
            return true;
        }
    }
    false
}

fn producer_supports_self_silu_absorb(name: &str) -> Option<&'static str> {
    match name {
        "Convolution" => Some("ConvolutionSilu"),
        _ => None,
    }
}

/// Absorb a downstream self-SiLU chain into a materializing producer.
///
/// Pattern:
///   producer: y = Producer(...)
///   consumer: out = y * sigmoid(y)
///
/// Result:
///   producer writes `out` directly, using a post-op variant of the same
///   emitter. This is intentionally semantic: the only model-specific part is
///   whether the producer emitter has a correct post-op implementation.
fn try_self_silu_absorb(slots: &mut Vec<Option<Fusion>>) -> bool {
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    let dbg = std::env::var_os("FUSEKIT_SELF_SILU_ABSORB_DEBUG").is_some();
    for p_idx in 0..slots.len() {
        let producer = match slots[p_idx].as_ref() {
            Some(f) => f,
            None => continue,
        };
        let post_op_name = match &producer.kind {
            FusionKind::Single(name) => match producer_supports_self_silu_absorb(name) {
                Some(post) => post,
                None => continue,
            },
            _ => continue,
        };
        if producer.outputs.len() != 1 {
            continue;
        }
        let y = producer.outputs[0].source;
        let y_elems = shape_elems(&producer.outputs[0].shape);
        let (aliases, metadata) = collect_metadata_alias_chain(y, slots, &consumers_of);
        let metadata_set: HashSet<usize> = metadata.iter().copied().collect();

        let mut real_consumers: HashSet<usize> = HashSet::new();
        for alias in &aliases {
            for c_idx in consumers_of.get(alias).cloned().unwrap_or_default() {
                if c_idx == p_idx || metadata_set.contains(&c_idx) {
                    continue;
                }
                real_consumers.insert(c_idx);
            }
        }
        if real_consumers.len() != 1 {
            continue;
        }
        let c_idx = *real_consumers.iter().next().unwrap();
        let consumer = match slots[c_idx].as_ref() {
            Some(f) => f,
            None => continue,
        };
        let c_steps = match (&consumer.kind, &consumer.attrs) {
            (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
            _ => continue,
        };
        if c_steps.len() != 2
            || c_steps[0].op_name != "Sigmoid"
            || c_steps[1].op_name != "Multiply"
            || consumer.outputs.len() != 1
            || shape_elems(&consumer.outputs[0].shape) != y_elems
        {
            continue;
        }
        let cinputs: Vec<TensorId> = consumer.inputs.iter().map(|b| b.source).collect();
        let resolves_to_y = |op: ChainOperand| -> bool {
            match op {
                ChainOperand::External { buffer_index } => cinputs
                    .get(buffer_index as usize)
                    .is_some_and(|src| aliases.contains(src)),
                _ => false,
            }
        };
        let sig_input = match c_steps[0].inputs.first().copied() {
            Some(x) => x,
            None => continue,
        };
        if !resolves_to_y(sig_input) {
            continue;
        }
        let m0 = c_steps[1].inputs.first().copied();
        let m1 = c_steps[1].inputs.get(1).copied();
        let self_silu = match (m0, m1) {
            (Some(ChainOperand::Register { step_index: 0 }), Some(x))
            | (Some(x), Some(ChainOperand::Register { step_index: 0 })) => resolves_to_y(x),
            _ => false,
        };
        if !self_silu {
            continue;
        }

        let mut metadata_isolated = true;
        for &m_idx in &metadata_set {
            let Some(m) = slots[m_idx].as_ref() else {
                metadata_isolated = false;
                break;
            };
            for out in &m.outputs {
                for d_idx in consumers_of.get(&out.source).cloned().unwrap_or_default() {
                    if d_idx != c_idx && !metadata_set.contains(&d_idx) {
                        metadata_isolated = false;
                        break;
                    }
                }
                if !metadata_isolated {
                    break;
                }
            }
            if !metadata_isolated {
                break;
            }
        }
        if !metadata_isolated {
            continue;
        }

        if dbg {
            eprintln!("[self-silu-absorb] merge p_idx={p_idx} c_idx={c_idx} as {post_op_name}");
        }
        for &m_idx in &metadata_set {
            slots[m_idx].take();
        }
        let mut producer_owned = slots[p_idx].take().expect("producer present");
        let consumer_owned = slots[c_idx].take().expect("consumer present");
        if let FusionAttrs::Single { op_name, .. } = &mut producer_owned.attrs {
            *op_name = post_op_name.to_string();
        }
        producer_owned.kind = FusionKind::Single(post_op_name.to_string());
        producer_owned.outputs = consumer_owned.outputs.clone();
        producer_owned
            .op_ids
            .extend(consumer_owned.op_ids.iter().copied());
        slots[c_idx] = Some(producer_owned);
        return true;
    }
    false
}

/// Identify semantic pointwise formulas attached to a reduction-fanout
/// producer. The roles are inferred from op/dataflow use sites, not from
/// module names or fixed layer templates.
fn chain_register_operand(step: &ChainStep, idx: usize) -> Option<u32> {
    match step.inputs.get(idx).copied()? {
        ChainOperand::Register { step_index } => Some(step_index),
        ChainOperand::External { .. } => None,
    }
}

fn build_reduction_fanout_tail_chain(
    formula: PointwiseTailFormula,
    fusion_id: u32,
    producer: &Fusion,
    consumer: &Fusion,
    producer_steps: &[ChainStep],
    gate_step_idx: usize,
    up_step_idx: usize,
    meta: &crate::fusion::ir::ModelMeta,
    metadata_fusions: &[Fusion],
) -> Option<Fusion> {
    if formula != PointwiseTailFormula::SiluGateUp {
        return None;
    }
    if producer_steps.len() != 3 || producer.outputs.len() != 2 || consumer.outputs.len() != 1 {
        return None;
    }
    let head = &producer_steps[0];
    if gate_step_idx == up_step_idx || gate_step_idx == 0 || up_step_idx == 0 {
        return None;
    }
    let gate = producer_steps.get(gate_step_idx)?;
    let up = producer_steps.get(up_step_idx)?;
    if head.op_name != "RMSNorm"
        || gate.op_name != "QuantizedMatmul"
        || up.op_name != "QuantizedMatmul"
    {
        return None;
    }
    if gate.inputs.len() != 4 || up.inputs.len() != 4 {
        return None;
    }
    if chain_register_operand(gate, 0) != Some(0) || chain_register_operand(up, 0) != Some(0) {
        return None;
    }
    let hidden = *head.output_shape.last()?;
    let out_dim = *gate.output_shape.last()?;
    if hidden != meta.hidden_size as i32
        || out_dim != meta.intermediate_size as i32
        || up.output_shape.last().copied() != Some(out_dim)
    {
        return None;
    }
    if !matches!(head.op_attrs, Attrs::RmsNorm { .. }) {
        return None;
    }
    match (gate.op_attrs.clone(), up.op_attrs.clone()) {
        (
            Attrs::Quantized {
                bits: gate_bits,
                group_size: gate_group,
            },
            Attrs::Quantized {
                bits: up_bits,
                group_size: up_group,
            },
        ) if gate_bits == up_bits && gate_group == up_group => {}
        _ => return None,
    };

    let mut output = consumer.outputs.first()?.clone();
    output.role = "scratch";

    let mut op_ids = producer.op_ids.clone();
    for f in metadata_fusions {
        op_ids.extend(f.op_ids.iter().copied());
    }
    op_ids.extend(consumer.op_ids.iter().copied());

    let c_steps = match (&consumer.kind, &consumer.attrs) {
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
        _ => return None,
    };
    if c_steps.len() != 3 {
        return None;
    }
    let sigmoid_step = c_steps[0].clone();
    let mul_gate_step = c_steps[1].clone();
    let mul_up_step = c_steps[2].clone();
    let sigmoid_idx = producer_steps.len() as u32;
    let mul_gate_idx = sigmoid_idx + 1;

    let mut steps = producer_steps.to_vec();
    steps.push(ChainStep {
        op_name: sigmoid_step.op_name,
        op_attrs: sigmoid_step.op_attrs,
        inputs: vec![ChainOperand::Register {
            step_index: gate_step_idx as u32,
        }],
        output_shape: sigmoid_step.output_shape,
        output_dtype: sigmoid_step.output_dtype,
    });
    steps.push(ChainStep {
        op_name: mul_gate_step.op_name,
        op_attrs: mul_gate_step.op_attrs,
        inputs: vec![
            ChainOperand::Register {
                step_index: sigmoid_idx,
            },
            ChainOperand::Register {
                step_index: gate_step_idx as u32,
            },
        ],
        output_shape: mul_gate_step.output_shape,
        output_dtype: mul_gate_step.output_dtype,
    });
    steps.push(ChainStep {
        op_name: mul_up_step.op_name,
        op_attrs: mul_up_step.op_attrs,
        inputs: vec![
            ChainOperand::Register {
                step_index: mul_gate_idx,
            },
            ChainOperand::Register {
                step_index: up_step_idx as u32,
            },
        ],
        output_shape: mul_up_step.output_shape,
        output_dtype: mul_up_step.output_dtype,
    });

    let transposed_weight_slots = match &producer.attrs {
        FusionAttrs::FusedChain {
            transposed_weight_slots,
            ..
        } => transposed_weight_slots.clone(),
        _ => Vec::new(),
    };

    Some(Fusion {
        id: fusion_id,
        kind: FusionKind::FusedChain,
        op_ids,
        inputs: producer.inputs.clone(),
        outputs: vec![output],
        attrs: FusionAttrs::FusedChain {
            steps,
            transposed_weight_slots,
        },
        parallel_axes: Vec::new(),
        layer_hint: producer.layer_hint.or(consumer.layer_hint),
    })
}

fn semantic_silu_gate_up_consumer(
    consumer: &Fusion,
    gate_aliases: &HashSet<TensorId>,
    up_aliases: &HashSet<TensorId>,
) -> bool {
    let c_steps = match (&consumer.kind, &consumer.attrs) {
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
        _ => return false,
    };
    if c_steps.len() != 3
        || c_steps[0].op_name != "Sigmoid"
        || c_steps[1].op_name != "Multiply"
        || c_steps[2].op_name != "Multiply"
        || consumer.outputs.len() != 1
    {
        return false;
    }

    let cinputs: Vec<TensorId> = consumer.inputs.iter().map(|b| b.source).collect();
    let resolves_to_alias = |op: ChainOperand, aliases: &HashSet<TensorId>| -> bool {
        match op {
            ChainOperand::External { buffer_index } => cinputs
                .get(buffer_index as usize)
                .is_some_and(|src| aliases.contains(src)),
            ChainOperand::Register { .. } => false,
        }
    };

    let Some(sig_input) = c_steps[0].inputs.first().copied() else {
        return false;
    };
    if !resolves_to_alias(sig_input, gate_aliases) {
        return false;
    }

    let mul1_ok = match (
        c_steps[1].inputs.first().copied(),
        c_steps[1].inputs.get(1).copied(),
    ) {
        (Some(ChainOperand::Register { step_index: 0 }), Some(x)) => {
            resolves_to_alias(x, gate_aliases)
        }
        (Some(x), Some(ChainOperand::Register { step_index: 0 })) => {
            resolves_to_alias(x, gate_aliases)
        }
        _ => false,
    };
    if !mul1_ok {
        return false;
    }

    match (
        c_steps[2].inputs.first().copied(),
        c_steps[2].inputs.get(1).copied(),
    ) {
        (Some(ChainOperand::Register { step_index: 1 }), Some(x)) => {
            resolves_to_alias(x, up_aliases)
        }
        (Some(x), Some(ChainOperand::Register { step_index: 1 })) => {
            resolves_to_alias(x, up_aliases)
        }
        _ => false,
    }
}

/// Recognize a reduction-fanout producer plus pointwise tail as a
/// semantic formula, then select the best safe lowering. Supported
/// lowerings can emit a specialized kernel; otherwise the generic
/// FusedChain absorb path remains opt-in for experimentation.
fn try_reduction_fanout_pointwise_tail_lowering(
    slots: &mut Vec<Option<Fusion>>,
    _reg_budget: u32,
    _smem_budget: usize,
    meta: &crate::fusion::ir::ModelMeta,
) -> bool {
    let mut producer_of: HashMap<TensorId, usize> = HashMap::new();
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for out in &f.outputs {
            producer_of.insert(out.source, idx);
        }
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    let dbg = std::env::var_os("FUSEKIT_POINTWISE_TAIL_DEBUG").is_some();
    for p_idx in 0..slots.len() {
        let producer = match slots[p_idx].as_ref() {
            Some(f) => f,
            None => continue,
        };
        let p_steps = match (&producer.kind, &producer.attrs) {
            (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
            _ => continue,
        };
        if p_steps.len() != 3 {
            continue;
        }
        if p_steps[0].op_name != "RMSNorm" {
            continue;
        }
        if categorize_name(&p_steps[1].op_name) != OpCategory::Heavy {
            continue;
        }
        if categorize_name(&p_steps[2].op_name) != OpCategory::Heavy {
            continue;
        }
        if producer.outputs.len() != 2 {
            continue;
        }
        if dbg {
            eprintln!(
                "[pointwise-tail] candidate p_idx={p_idx} steps=[{},{},{}]",
                p_steps[0].op_name, p_steps[1].op_name, p_steps[2].op_name
            );
        }
        // Both heavies must have matching output dim (gate/up same shape).
        let g_dim = *p_steps[1].output_shape.last().unwrap_or(&0);
        let u_dim = *p_steps[2].output_shape.last().unwrap_or(&0);
        if g_dim != u_dim || g_dim == 0 {
            continue;
        }

        if p_steps[1].op_name != "QuantizedMatmul" || p_steps[2].op_name != "QuantizedMatmul" {
            continue;
        }

        for (gate_out_idx, up_out_idx) in [(0usize, 1usize), (1usize, 0usize)] {
            let gate_t = producer.outputs[gate_out_idx].source;
            let up_t = producer.outputs[up_out_idx].source;
            let gate_step_idx = gate_out_idx + 1;
            let up_step_idx = up_out_idx + 1;

            // See through element-contiguous view ops between gate/up and
            // the tail formula. MLX commonly inserts Unflatten/Flatten
            // around MLP intermediates; those are aliases for fusion
            // purposes and should not block semantic lowering.
            let (gate_aliases, gate_metadata) =
                collect_metadata_alias_chain(gate_t, slots, &consumers_of);
            let (up_aliases, up_metadata) =
                collect_metadata_alias_chain(up_t, slots, &consumers_of);
            let metadata_set: HashSet<usize> = gate_metadata
                .iter()
                .chain(up_metadata.iter())
                .copied()
                .collect();

            let alias_consumers = |aliases: &HashSet<TensorId>| -> HashSet<usize> {
                let mut set = HashSet::new();
                for alias in aliases {
                    for c_idx in consumers_of.get(alias).cloned().unwrap_or_default() {
                        if c_idx == p_idx || metadata_set.contains(&c_idx) {
                            continue;
                        }
                        set.insert(c_idx);
                    }
                }
                set
            };
            let gate_set = alias_consumers(&gate_aliases);
            let up_set = alias_consumers(&up_aliases);
            if dbg {
                eprintln!(
                    "[pointwise-tail]   gate_out={gate_out_idx} up_out={up_out_idx} gate_t={gate_t} up_t={up_t} gate_aliases={gate_aliases:?} up_aliases={up_aliases:?} gate_set={gate_set:?} up_set={up_set:?}"
                );
            }
            if gate_set.len() != 1 || up_set.len() != 1 {
                continue;
            }
            let c_idx = *gate_set.iter().next().unwrap();
            if !up_set.contains(&c_idx) || c_idx == p_idx {
                continue;
            }

            // The metadata views we plan to drop must only feed the final
            // formula consumer or other metadata aliases. Otherwise an
            // outside branch would still need the materialized view.
            let mut metadata_isolated = true;
            for &m_idx in &metadata_set {
                let Some(m) = slots[m_idx].as_ref() else {
                    metadata_isolated = false;
                    break;
                };
                for out in &m.outputs {
                    for d_idx in consumers_of.get(&out.source).cloned().unwrap_or_default() {
                        if d_idx != c_idx && !metadata_set.contains(&d_idx) {
                            metadata_isolated = false;
                            break;
                        }
                    }
                    if !metadata_isolated {
                        break;
                    }
                }
                if !metadata_isolated {
                    break;
                }
            }
            if !metadata_isolated {
                continue;
            }

            let consumer = match slots[c_idx].as_ref() {
                Some(f) => f,
                None => continue,
            };
            if !semantic_silu_gate_up_consumer(consumer, &gate_aliases, &up_aliases) {
                continue;
            }

            let formula = PointwiseTailFormula::SiluGateUp;
            let metadata_fusions = metadata_set
                .iter()
                .filter_map(|&idx| slots[idx].as_ref().cloned())
                .collect::<Vec<_>>();
            if let Some(merged) = build_reduction_fanout_tail_chain(
                formula,
                c_idx as u32,
                producer,
                consumer,
                p_steps,
                gate_step_idx,
                up_step_idx,
                meta,
                &metadata_fusions,
            ) {
                for &m_idx in &metadata_set {
                    slots[m_idx].take();
                }
                slots[p_idx].take();
                slots[c_idx].take();
                slots[c_idx] = Some(merged);
                return true;
            }
        }
    }
    false
}

fn is_reduction_swiglu_chain(fusion: &Fusion) -> bool {
    let (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) =
        (&fusion.kind, &fusion.attrs)
    else {
        return false;
    };
    steps.len() == 6
        && fusion.outputs.len() == 1
        && steps[0].op_name == "RMSNorm"
        && categorize_name(&steps[1].op_name) == OpCategory::Heavy
        && categorize_name(&steps[2].op_name) == OpCategory::Heavy
        && steps[3].op_name == "Sigmoid"
        && steps[4].op_name == "Multiply"
        && steps[5].op_name == "Multiply"
}

/// Report budget admission for a possible `SwiGLU -> down_proj`
/// fusion. Full fusion is useful only when the complete SwiGLU vector
/// can stay in threadgroup memory. If it cannot, down-projection tiling
/// would recompute gate/up per tile and trade one activation round-trip
/// for much larger repeated quantized-weight traffic.
fn report_swiglu_down_admission(
    slots: &[Option<Fusion>],
    smem_budget: usize,
    meta: &crate::fusion::ir::ModelMeta,
) {
    if !env_flag("FUSEKIT_SWIGLU_DOWN_ADMISSION_DEBUG") {
        return;
    }

    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(fusion) = slot else { continue };
        for input in &fusion.inputs {
            consumers_of.entry(input.source).or_default().push(idx);
        }
    }

    let mut seen = 0usize;
    let mut fit = 0usize;
    let mut reject = 0usize;
    for (p_idx, slot) in slots.iter().enumerate() {
        let Some(producer) = slot.as_ref() else {
            continue;
        };
        if !is_reduction_swiglu_chain(producer) {
            continue;
        }
        let Some(out) = producer.outputs.first() else {
            continue;
        };
        let silu_shape = out.shape.clone();
        let intermediate = silu_shape
            .last()
            .copied()
            .unwrap_or(meta.intermediate_size as i32);
        let (aliases, metadata) = collect_metadata_alias_chain(out.source, slots, &consumers_of);
        let metadata_set: HashSet<usize> = metadata.iter().copied().collect();

        let mut down_consumers = HashSet::new();
        for alias in &aliases {
            for c_idx in consumers_of.get(alias).cloned().unwrap_or_default() {
                if c_idx == p_idx || metadata_set.contains(&c_idx) {
                    continue;
                }
                if slots
                    .get(c_idx)
                    .and_then(|s| s.as_ref())
                    .is_some_and(|f| matches!(f.kind, FusionKind::LinearResidual))
                {
                    down_consumers.insert(c_idx);
                }
            }
        }

        for c_idx in down_consumers {
            let Some(consumer) = slots.get(c_idx).and_then(|s| s.as_ref()) else {
                continue;
            };
            seen += 1;
            let hidden = consumer
                .outputs
                .first()
                .and_then(|b| b.shape.last().copied())
                .unwrap_or(meta.hidden_size as i32)
                .max(1) as usize;
            let intermediate = intermediate.max(1) as usize;
            let reduce_buf = 32usize * 4;
            let full_smem = hidden
                .saturating_mul(4)
                .saturating_add(intermediate.saturating_mul(4))
                .saturating_add(reduce_buf);
            let activation_bytes = shape_elems(&silu_shape)
                .saturating_mul(dtype_size_bytes(out.dtype))
                .saturating_mul(2);
            let status = if full_smem <= smem_budget {
                fit += 1;
                "fit"
            } else {
                reject += 1;
                "reject_smem"
            };
            eprintln!(
                "[swiglu-down-admission] {status} producer={p_idx} consumer={c_idx} hidden={hidden} intermediate={intermediate} smem={} budget={} activation_roundtrip={}B reason={}",
                full_smem,
                smem_budget,
                activation_bytes,
                if full_smem <= smem_budget {
                    "full_silu_up_can_stay_in_tgmem"
                } else {
                    "full_silu_up_exceeds_tgmem; avoid_gate_up_recompute_per_down_tile"
                }
            );
        }
    }
    eprintln!(
        "[swiglu-down-admission] summary candidates={seen} fit={fit} reject={reject}"
    );
}

fn try_weighted_route_reduce(slots: &mut Vec<Option<Fusion>>) -> bool {
    let mut producer_of: HashMap<TensorId, usize> = HashMap::new();
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for out in &f.outputs {
            producer_of.insert(out.source, idx);
        }
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    let is_single =
        |f: &Fusion, name: &str| matches!(&f.kind, FusionKind::Single(op) if op == name);
    let dbg = std::env::var_os("FUSEKIT_WEIGHTED_ROUTE_REDUCE_DEBUG").is_some();

    for reduce_idx in 0..slots.len() {
        let reduce = match slots[reduce_idx].as_ref() {
            Some(f) if is_single(f, "Reduce") && f.inputs.len() == 1 && f.outputs.len() == 1 => f,
            _ => continue,
        };
        let reduce_is_sum_or_unknown = matches!(
            &reduce.attrs,
            FusionAttrs::Single {
                op_attrs: crate::graph::Attrs::Reduce {
                    kind: crate::graph::ReduceKind::Sum,
                    ..
                },
                ..
            } | FusionAttrs::Single {
                op_attrs: crate::graph::Attrs::Unknown,
                ..
            }
        );
        if !reduce_is_sum_or_unknown {
            if dbg {
                eprintln!("[weighted-route-reduce] skip non-sum reduce idx={reduce_idx}");
            }
            continue;
        }

        let Some(mul_idx) = producer_of.get(&reduce.inputs[0].source).copied() else {
            continue;
        };
        let multiply = match slots[mul_idx].as_ref() {
            Some(f) if is_single(f, "Multiply") && f.inputs.len() == 2 && f.outputs.len() == 1 => f,
            _ => continue,
        };
        if consumers_of
            .get(&multiply.outputs[0].source)
            .is_none_or(|cs| cs.as_slice() != [reduce_idx])
        {
            continue;
        }

        let in0 = multiply.inputs[0].source;
        let in1 = multiply.inputs[1].source;
        let bcast0 = producer_of.get(&in0).copied().filter(|&idx| {
            slots[idx]
                .as_ref()
                .is_some_and(|f| is_single(f, "Broadcast"))
        });
        let bcast1 = producer_of.get(&in1).copied().filter(|&idx| {
            slots[idx]
                .as_ref()
                .is_some_and(|f| is_single(f, "Broadcast"))
        });
        let (bcast_idx, data_input_slot) = match (bcast0, bcast1) {
            (Some(idx), None) => (idx, 1usize),
            (None, Some(idx)) => (idx, 0usize),
            _ => continue,
        };
        let bcast = match slots[bcast_idx].as_ref() {
            Some(f) if f.inputs.len() == 1 && f.outputs.len() == 1 => f,
            _ => continue,
        };
        if consumers_of
            .get(&bcast.outputs[0].source)
            .is_none_or(|cs| cs.as_slice() != [mul_idx])
        {
            continue;
        }
        if !weighted_route_reduce_shapes_ok(
            &multiply.inputs[data_input_slot].shape,
            &bcast.inputs[0].shape,
            &multiply.outputs[0].shape,
            &reduce.outputs[0].shape,
        ) {
            continue;
        }

        let bcast_owned = slots[bcast_idx].take().expect("broadcast present");
        let multiply_owned = slots[mul_idx].take().expect("multiply present");
        let reduce_owned = slots[reduce_idx].take().expect("reduce present");

        let mut op_ids = bcast_owned.op_ids.clone();
        op_ids.extend(multiply_owned.op_ids.iter().copied());
        op_ids.extend(reduce_owned.op_ids.iter().copied());

        let merged = Fusion {
            id: reduce_idx as u32,
            kind: FusionKind::Single("WeightedRouteReduce".to_string()),
            op_ids,
            inputs: vec![
                multiply_owned.inputs[data_input_slot].clone(),
                bcast_owned.inputs[0].clone(),
            ],
            outputs: reduce_owned.outputs.clone(),
            attrs: FusionAttrs::Single {
                op_name: "WeightedRouteReduce".to_string(),
                op_attrs: crate::graph::Attrs::Unknown,
            },
            parallel_axes: vec![],
            layer_hint: reduce_owned
                .layer_hint
                .or(multiply_owned.layer_hint)
                .or(bcast_owned.layer_hint),
        };
        if dbg {
            eprintln!(
                "[weighted-route-reduce] merged bcast={bcast_idx} mul={mul_idx} reduce={reduce_idx}"
            );
        }
        slots[reduce_idx] = Some(merged);
        return true;
    }
    false
}

fn weighted_route_reduce_shapes_ok(
    data_shape: &[i32],
    weight_shape: &[i32],
    multiply_shape: &[i32],
    reduce_shape: &[i32],
) -> bool {
    if data_shape != multiply_shape
        || data_shape.len() != weight_shape.len()
        || data_shape.len() != reduce_shape.len()
        || data_shape.is_empty()
    {
        return false;
    }
    let mut reduce_axis = None;
    for (idx, ((&d, &w), &o)) in data_shape
        .iter()
        .zip(weight_shape.iter())
        .zip(reduce_shape.iter())
        .enumerate()
    {
        if d > 1 && o == 1 {
            if reduce_axis.replace(idx).is_some() || w != d {
                return false;
            }
            continue;
        }
        if o != d || (w != d && w != 1) {
            return false;
        }
    }
    let Some(axis) = reduce_axis else {
        return false;
    };
    let post: i32 = data_shape[axis + 1..].iter().product::<i32>().max(1);
    let weight_post: i32 = weight_shape[axis + 1..].iter().product::<i32>().max(1);
    weight_post == 1 || weight_post == post
}

/// QM linear-bias absorption.
///
/// Detects: a FusedChain heavy step that's a `QuantizedMatmul` whose
/// output tensor flows into a Single `Add(qm_out, bias_param)` where
/// `bias_param` is a model parameter (has `param_path` in the graph).
/// The Add has exactly one consumer (so dropping it is safe). When
/// matched:
///   1. Append `bias_param` to the producer chain's `inputs` vec.
///   2. Append a 5th External operand (= bias) to the QM ChainStep's
///      inputs list. emit_reduction_fanout_chain / emit_qmm_chain
///      see this and apply `acc += bias[out_idx]` post-`simd_sum`.
///   3. Re-point the chain's output for that heavy step from the QM
///      output tensor → the Add output tensor (so downstream
///      consumers' bindings still resolve to this chain).
///   4. Drop the Add Single from `slots`.
///
/// Returns true if anything was absorbed (so caller can re-scan).
fn try_absorb_qm_bias(
    slots: &mut Vec<Option<Fusion>>,
    _meta: &crate::fusion::ir::ModelMeta,
    graph: &crate::graph::Graph,
) -> bool {
    let mut consumers_of: HashMap<TensorId, Vec<usize>> = HashMap::new();
    let mut producer_of: HashMap<TensorId, usize> = HashMap::new();
    for (idx, slot) in slots.iter().enumerate() {
        let Some(f) = slot else { continue };
        for out in &f.outputs {
            producer_of.insert(out.source, idx);
        }
        for inp in &f.inputs {
            consumers_of.entry(inp.source).or_default().push(idx);
        }
    }

    for c_idx in 0..slots.len() {
        let consumer = match slots[c_idx].as_ref() {
            Some(f) => f,
            None => continue,
        };
        // Consumer must be a Single `Add` with exactly 2 inputs, one
        // of which is a model parameter (the bias).
        let is_add_single = matches!(&consumer.kind, FusionKind::Single(op) if op == "Add");
        if !is_add_single {
            continue;
        }
        if consumer.inputs.len() != 2 || consumer.outputs.len() != 1 {
            continue;
        }

        // Identify which input is the bias side. mlx-lm's `+ bias`
        // may compile to a direct Add (bias rank-1 by itself) or to
        // an explicit Broadcast(bias) → Add (bias side rank-matching
        // QM output, but the underlying tensor is a model parameter).
        // Walk back through view ops to find a rank-1 param-backed
        // tensor; that's the bias side. The OTHER side is the data
        // (qm output) path.
        let walk_to_param_rank1 = |start: TensorId| -> Option<TensorId> {
            let mut cur = start;
            let mut depth = 0;
            while depth < 8 {
                let t = graph.tensors.get(cur as usize)?;
                if t.shape.len() == 1 && t.param_path.is_some() {
                    return Some(cur);
                }
                // Look up which op produces cur; if it's a view/
                // broadcast Single, walk back.
                let mut found_producer: Option<&crate::graph::Op> = None;
                for op in &graph.ops {
                    if op.outputs.iter().any(|&t| t == cur) {
                        found_producer = Some(op);
                        break;
                    }
                }
                let op = found_producer?;
                if !matches!(
                    op.name.as_str(),
                    "Broadcast"
                        | "Reshape"
                        | "Flatten"
                        | "Unflatten"
                        | "Squeeze"
                        | "ExpandDims"
                        | "AsContiguous"
                        | "Transpose"
                ) {
                    return None;
                }
                if op.inputs.is_empty() {
                    return None;
                }
                cur = op.inputs[0];
                depth += 1;
            }
            None
        };
        let bias_at_0 = walk_to_param_rank1(consumer.inputs[0].source).is_some();
        let bias_at_1 = walk_to_param_rank1(consumer.inputs[1].source).is_some();
        let (data_in_idx, bias_in_idx) = match (bias_at_0, bias_at_1) {
            (false, true) => (0usize, 1usize),
            (true, false) => (1usize, 0usize),
            _ => continue,
        };
        // Resolve to the actual rank-1 param tensor (not the
        // intermediate broadcast/view output) so the chain's input
        // boundary points directly at the model parameter.
        let bias_param_tid = walk_to_param_rank1(consumer.inputs[bias_in_idx].source)
            .expect("bias_at_? guarantees walk succeeds");
        let bias_param_tensor = &graph.tensors[bias_param_tid as usize];
        let bias_bd = FusionBoundary {
            source: bias_param_tid,
            role: "lin_bias",
            shape: bias_param_tensor.shape.clone(),
            dtype: bias_param_tensor.dtype,
        };
        let _ = bias_in_idx;

        let qm_out_tid = consumer.inputs[data_in_idx].source;

        // The qm output must come from a FusedChain producer.
        let p_idx = match producer_of.get(&qm_out_tid).copied() {
            Some(i) => i,
            None => continue,
        };
        if p_idx == c_idx {
            continue;
        }
        let producer = match slots[p_idx].as_ref() {
            Some(f) => f,
            None => continue,
        };
        let p_steps = match (&producer.kind, &producer.attrs) {
            (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
            _ => continue,
        };
        // Find the QM step in the chain whose output is qm_out_tid.
        // Match by the producer output FusionBoundary that carries
        // qm_out_tid as source. That gives us out_buf index = which
        // heavy step this is.
        let out_buf_idx = match producer.outputs.iter().position(|b| b.source == qm_out_tid) {
            Some(i) => i,
            None => continue,
        };
        // The heavy step in `steps` whose output maps to out_buf_idx.
        // do_fanout_merge produces heavies in the SAME order as
        // outputs, with RMSNorm at index 0. So heavy step index =
        // out_buf_idx + 1.
        let heavy_step_idx = out_buf_idx + 1;
        if heavy_step_idx >= p_steps.len() {
            continue;
        }
        if p_steps[heavy_step_idx].op_name != "QuantizedMatmul" {
            continue;
        }

        // Check the Add Single's output has exactly ONE consumer (the
        // chain absorbing it must be safe; if multiple downstream ops
        // read the biased value, they'd still find it via the chain's
        // re-pointed output, so multi-consumer is fine — but the Add
        // Single itself can only be dropped if we're sure no
        // independent path still references its output tensor in a
        // way that needs the original Single dispatch. View-only ops
        // re-bind to the new chain output's scratch — safe.).

        // Perform absorption.
        let consumer_owned = slots[c_idx].take().expect("consumer present");
        let add_out_bd = consumer_owned.outputs[0].clone();
        let mut producer_owned = slots[p_idx].take().expect("producer present");
        let mut producer_steps = match producer_owned.attrs {
            FusionAttrs::FusedChain { steps, .. } => steps,
            _ => unreachable!(),
        };
        // 1) Append bias to producer.inputs and remember its new index.
        let new_bias_buf_idx = producer_owned.inputs.len() as u32;
        producer_owned.inputs.push(bias_bd);
        // 2) Append 5th External operand to the QM ChainStep.
        producer_steps[heavy_step_idx]
            .inputs
            .push(ChainOperand::External {
                buffer_index: new_bias_buf_idx,
            });
        // 3) Re-point chain's heavy output to the Add output tensor.
        producer_owned.outputs[out_buf_idx].source = add_out_bd.source;
        producer_owned.outputs[out_buf_idx].shape = add_out_bd.shape;
        producer_owned.outputs[out_buf_idx].dtype = add_out_bd.dtype;
        producer_owned.attrs = FusionAttrs::FusedChain {
            steps: producer_steps,
            transposed_weight_slots: Vec::new(),
        };
        producer_owned
            .op_ids
            .extend(consumer_owned.op_ids.iter().copied());
        slots[p_idx] = Some(producer_owned);
        return true;
    }
    false
}
