#![allow(dead_code)]

//! Static safety checks for fusion candidates.
//!
//! A pattern matcher returns "this subgraph looks like X". Safety asks:
//! is X actually emittable given hardware limits?
//!
//! ```text
//!   Pattern match → Safety::check ──► Accept       (one task, fits SMEM)
//!                                ──► SplitTasks   (multi-task in megakernel)
//!                                ──► Reject       (true impossibility)
//! ```
//!
//! `SplitTasks` is NOT the same as falling back to a smaller fusion.
//! The megakernel architecture keeps all sub-tasks inside one kernel
//! launch, with global memory carrying intermediates between them.
//! Splitting only adds scheduler events, not kernel launches.

use crate::fusion::ir::{FusionKind, ModelMeta};
use crate::fusion::patterns::Match;

/// Apple Silicon threadgroup memory budget. Conservative — actual M5
/// limit is 32KB per threadgroup; staying below ~28KB leaves room for
/// stack frames and synchronization barriers.
pub const SMEM_BUDGET_BYTES: usize = 28 * 1024;

/// Largest `kv_len` the SdpaDecode kernel statically allocates for in
/// threadgroup memory. Must stay in sync with the constant of the same
/// name in `codegen::kernels::sdpa_decode`.
pub const SDPA_DECODE_MAX_KV_LEN: usize = 4096;

/// Outcome of safety analysis.
#[allow(dead_code)]
pub enum SafetyCheck {
    /// Fusion fits a single task; codegen emits one kernel.
    Accept,
    /// Fusion exceeds SMEM. Codegen emits N tasks that materialize
    /// intermediates to global memory at the split points.
    SplitTasks(SplitPlan),
    /// Truly cannot emit (register pressure, dependency cycle, ...).
    Reject(RejectReason),
}

#[allow(dead_code)]
pub struct SplitPlan {
    /// Indices into the original match's `op_ids` where to cut. Each
    /// entry creates a new sub-task starting at that op.
    pub cut_after: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum RejectReason {
    /// Threadgroup memory required exceeds `SMEM_BUDGET_BYTES`.
    /// Carries the estimated bytes so the caller can log it.
    SmemOverflow { needed: usize, budget: usize },
    /// Estimated register pressure exceeds Metal's per-thread budget.
    /// No spill option (Metal register spills go to threadgroup memory,
    /// which is what we'd be over-budget on anyway).
    RegisterPressure,
    /// Multiple downstream consumers read the same intermediate.
    /// Fusing would force recomputation or a global memory write
    /// anyway, defeating the purpose.
    SharedIntermediate,
    /// Pattern matched but the resulting fusion's parallel_axes don't
    /// admit any valid partition (e.g. requires partitioning along an
    /// axis that's reduced inside).
    NoValidPartition,
}

/// Estimate the threadgroup memory bytes a single emitted kernel for
/// `kind` would allocate. Numbers must track `codegen::kernels::*` —
/// when that emitter changes its `threadgroup` arrays, update here.
///
/// The estimate is per-threadgroup. Across the grid Metal allocates
/// `grid_size * smem_bytes` of physical SLM but each threadgroup only
/// sees its own copy, so the budget check is local.
/// Per-thread live intermediate count estimate, in 32-bit register
/// slots. Used by the codegen heuristic to decide whether a kernel
/// will spill into thread-private memory (which is ~10–50× slower
/// than register access on Apple GPUs).
///
/// We count the worst-case simultaneously live intermediates inside
/// the inner loop, plus the per-thread constants the compiler can't
/// constant-fold (group scale/bias, accumulators, etc.). The numbers
/// are conservative — actual compiler-assigned register usage tends
/// to be lower thanks to schedule fusion, but planning around the
/// upper bound keeps us out of spill territory.
pub fn estimate_register_pressure(kind: &FusionKind, _meta: &ModelMeta) -> u32 {
    match kind {
        // RMSNorm reduction: sum_sq, xi, rms_scale, partial = ~6 slots.
        // QKV inner: acc, word, group_idx, scale, bias, q_nibble × 8 (int4
        // explicit unroll) = ~14. Total worst case ≈ 20.
        FusionKind::RmsNormQkv | FusionKind::RmsNormGateUp | FusionKind::RmsNormGateUpSwiglu => 24,
        // gateup_swiglu_down keeps the silu_up vector in SMEM, so the
        // per-thread live set is similar but the dot product against
        // SMEM bumps live count slightly.
        FusionKind::RmsNormGateUpSwigluDown => 32,
        // SDPA: q_row, k_row, dot, scale, max, sum, w, v_row, acc.
        // 2-stage reduction needs partial_max / partial_sum live too.
        FusionKind::SdpaDecode => 20,
        // Pure GEMV with simdgroup-coop: ~12 (uint4 vectorized weight
        // load, scale, bias, 4 partial products, accumulator).
        FusionKind::LinearResidual | FusionKind::AttnOutResidual => 16,
        // LM head GEMV with quant: similar to qkv inner.
        FusionKind::FinalRmsNormLmHead => 24,
        // Elementwise / RoPE / embed / argmax — small.
        FusionKind::SwigluActivation
        | FusionKind::RopeQDecode
        | FusionKind::RopeKvCacheStoreDecode
        | FusionKind::KvCacheStoreDecode
        | FusionKind::RmsNormRopeQGateDecode
        | FusionKind::RmsNormRopeKvCacheStoreDecode
        | FusionKind::ArraysCacheShiftAppendDecode
        | FusionKind::EmbedLookupDecode
        | FusionKind::ArgmaxLogitsDecode
        | FusionKind::ArgmaxLogitsScan
        | FusionKind::ArgmaxLogitsFinalize => 8,
        // FusedChain: depends on chain depth; over-estimate to keep
        // the planner safe.
        FusionKind::FusedChain | FusionKind::QmvFanout => 32,
        FusionKind::QmvSwiglu => 40,
        // LayerBlock: dominated by SDPA piece + the QKV/down quant
        // GEMV inner loops. Use the SDPA bound (composer barriers
        // enforce serialization between pieces, so worst-case live set
        // is one piece at a time, not all combined).
        FusionKind::LayerBlock => 32,
        // Patterns we haven't emitted — treat as unknown.
        FusionKind::RmsNormProj
        | FusionKind::SdpaPrefill
        | FusionKind::RmsNormSwigluMlp
        | FusionKind::Single(_) => 64,
    }
}

pub fn estimate_smem_bytes(kind: &FusionKind, meta: &ModelMeta) -> usize {
    // Common: a single warp-wide partial-reduction scratch (32 floats).
    // Some kernels reuse one buffer for both stages, others double up.
    let reduce_buf_bytes = 32 * 4; // 32 × float
    match kind {
        // norm_y: float per hidden element + reduce buf.
        FusionKind::RmsNormQkv | FusionKind::RmsNormGateUp | FusionKind::RmsNormGateUpSwiglu => {
            meta.hidden_size as usize * 4 + reduce_buf_bytes
        }
        // RmsNormGateUpSwigluDown additionally holds the SwiGLU
        // activation in SMEM (N_INT floats) so the down-projection
        // reads from threadgroup memory instead of DRAM.
        FusionKind::RmsNormGateUpSwigluDown => {
            meta.hidden_size as usize * 4 + meta.intermediate_size as usize * 4 + reduce_buf_bytes
        }
        // scores: float per kv position + reduce buf.
        FusionKind::SdpaDecode => SDPA_DECODE_MAX_KV_LEN * 4 + reduce_buf_bytes,
        // Element-wise / pure GEMV: no shared workspace.
        FusionKind::SwigluActivation
        | FusionKind::LinearResidual
        | FusionKind::AttnOutResidual
        | FusionKind::QmvFanout
        | FusionKind::QmvSwiglu
        | FusionKind::RopeQDecode
        | FusionKind::RopeKvCacheStoreDecode
        | FusionKind::KvCacheStoreDecode
        | FusionKind::RmsNormRopeQGateDecode
        | FusionKind::RmsNormRopeKvCacheStoreDecode
        | FusionKind::ArraysCacheShiftAppendDecode
        | FusionKind::EmbedLookupDecode
        | FusionKind::ArgmaxLogitsDecode
        | FusionKind::ArgmaxLogitsScan
        | FusionKind::ArgmaxLogitsFinalize => 0,
        // FinalRmsNormLmHead keeps norm_y in SMEM and reads it across
        // many output rows. Each TG recomputes the norm locally (the
        // redundancy is negligible vs the vocab projection compute).
        FusionKind::FinalRmsNormLmHead => meta.hidden_size as usize * 4 + reduce_buf_bytes,
        // Patterns we haven't emitted yet — be conservative; treat as
        // unknown so the planner never silently fuses them into the
        // megakernel until an emitter lands.
        FusionKind::RmsNormProj
        | FusionKind::SdpaPrefill
        | FusionKind::RmsNormSwigluMlp
        | FusionKind::Single(_) => SMEM_BUDGET_BYTES + 1, // forces SplitTasks
        // FusedChain initially produced for pointwise-only paths where
        // each thread carries its values in registers — no SMEM.
        // Future expansion (reduction tail, heavy fusion) will need
        // a per-chain estimator; for now zero is the right answer.
        FusionKind::FusedChain => 0,
        // LayerBlock holds, concurrently in SMEM, the norm scratch row
        // (`norm_y` ≈ hidden×float) and the SDPA scores buffer
        // (≈ SDPA_DECODE_MAX_KV_LEN × float) plus per-stage reduction
        // scratch. The composer dedupes SMEM regions across pieces, so
        // we count once each. Estimate: norm_y + sdpa_scores + sdpa_reduce.
        FusionKind::LayerBlock => {
            meta.hidden_size as usize * 4 + SDPA_DECODE_MAX_KV_LEN * 4 + reduce_buf_bytes
        }
    }
}

/// Run safety analysis on a freshly matched fusion. Caller passes
/// `meta` so the SMEM estimator can use real architecture constants
/// (e.g. `hidden_size` for the rmsnorm scratch row).
pub fn check(m: &Match, meta: &ModelMeta) -> SafetyCheck {
    let needed = estimate_smem_bytes(&m.kind, meta);
    if needed > SMEM_BUDGET_BYTES {
        return SafetyCheck::Reject(RejectReason::SmemOverflow {
            needed,
            budget: SMEM_BUDGET_BYTES,
        });
    }
    SafetyCheck::Accept
}

/// Estimate the combined threadgroup memory if two fusions ran in the
/// same kernel body. Used by the megakernel planner to decide whether
/// a cross-pattern fuse fits the budget.
///
/// Conservative: sum the two estimates. Real codegen can sometimes
/// alias scratch buffers between stages, but that's an optimization to
/// land alongside the megakernel emitter, not here.
pub fn combined_smem_bytes(a: &FusionKind, b: &FusionKind, meta: &ModelMeta) -> usize {
    estimate_smem_bytes(a, meta) + estimate_smem_bytes(b, meta)
}

/// Does fusing `a` then `b` into one kernel body still fit SMEM?
pub fn can_cofuse(a: &FusionKind, b: &FusionKind, meta: &ModelMeta) -> bool {
    combined_smem_bytes(a, b, meta) <= SMEM_BUDGET_BYTES
}
