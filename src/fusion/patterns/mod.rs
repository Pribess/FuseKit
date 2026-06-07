#![allow(dead_code)]

//! Fusion pattern matchers.
//!
//! Each `Pattern` implementation is a function that probes the Stage 1
//! `Graph` at an anchor op and decides whether the surrounding subgraph
//! matches a known fused shape (RMSNorm + Matmul, SDPA, SwiGLU, ...).
//! Match outcomes are first-class so the `build` pass can reject them
//! cleanly when safety checks fail.
//!
//! Patterns are tried in `priority()` order, largest first, so a
//! matcher that covers many ops gets a chance before a narrower one.

use crate::fusion::ir::{FusionAttrs, FusionBoundary, FusionKind, ParallelAxis};
use crate::graph::{Graph, OpId};

/// Index built once per `build_fusions` invocation.
///
/// ```text
///   tensor_id ──► [op_id, op_id, ...]   the ops that READ this tensor
/// ```
///
/// Pattern matchers use this to verify the "single-consumer" condition
/// — fusing through a tensor that has multiple downstream readers
/// would require either re-materialization or memoization, and we
/// don't support either yet.
pub struct UseDef {
    pub consumers: Vec<Vec<OpId>>,
}

impl UseDef {
    pub fn build(g: &Graph) -> Self {
        let mut consumers = vec![Vec::<OpId>::new(); g.tensors.len()];
        for op in &g.ops {
            for &t in &op.inputs {
                consumers[t as usize].push(op.id);
            }
        }
        Self { consumers }
    }
}

/// Op names that only relabel shape/strides without touching values.
/// Patterns walk through these to find the real compute op they feed.
/// Transpose is intentionally excluded — even when MLX implements it
/// as a no-op view at the API level, downstream kernels often want
/// contiguous data, so we treat it as a compute boundary.
pub fn is_view_op(name: &str) -> bool {
    matches!(
        name,
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims"
    )
}

/// Result of walking view-only ops from a source tensor.
pub struct ViewWalk {
    /// The first non-view consumer reached.
    pub target: OpId,
    /// Tensor that actually appears in `target`'s input list (i.e. the
    /// output of the last view op, or the source itself if no views).
    pub feeding_tensor: crate::graph::TensorId,
    /// View ops traversed, in order. Empty when `target` directly
    /// consumes the source. Pattern matchers fold these into their
    /// `op_ids` so the consumed-set covers them.
    pub views: Vec<OpId>,
}

/// Walk single-consumer view ops starting from `tensor`. Returns
/// `None` if the chain forks or dead-ends without hitting a compute
/// op.
pub fn follow_views(
    g: &crate::graph::Graph,
    ud: &UseDef,
    mut tensor: crate::graph::TensorId,
) -> Option<ViewWalk> {
    let mut views = Vec::new();
    loop {
        let consumers = &ud.consumers[tensor as usize];
        if consumers.len() != 1 {
            return None;
        }
        let next = g.op(consumers[0]);
        if !is_view_op(&next.name) {
            return Some(ViewWalk {
                target: next.id,
                feeding_tensor: tensor,
                views,
            });
        }
        if next.outputs.len() != 1 {
            return None;
        }
        views.push(next.id);
        tensor = next.outputs[0];
    }
}

/// Walk back through a single Transpose op on the weight side of a
/// matmul. mlx-lm's `Linear` lowers `x @ W.T` to `Matmul(x, Transpose(W))`
/// for bf16 / fp16 weights — the trace materializes the Transpose
/// even though it's just a stride flip. For QuantizedMatmul the trace
/// keeps the original `[out, in]` weight (the QM primitive handles the
/// transposition internally), so there's no Transpose to walk through.
///
/// Returns `(original_weight_tensor, transpose_op_id_or_none)`. When
/// the input isn't a Transpose output, returns the input unchanged.
/// Pattern matchers fold the transpose op id into their consumed set
/// so it doesn't get re-emitted as its own kernel.
pub fn unwrap_weight_transpose(
    g: &crate::graph::Graph,
    ud: &UseDef,
    w_t: crate::graph::TensorId,
) -> (crate::graph::TensorId, Option<OpId>) {
    let Some(producer_id) = g.tensors[w_t as usize].producer else {
        return (w_t, None);
    };
    let producer = g.op(producer_id);
    if producer.name != "Transpose" {
        return (w_t, None);
    }
    if producer.inputs.len() != 1 || producer.outputs.len() != 1 {
        return (w_t, None);
    }
    // The Transpose's output must have a single consumer (this matmul).
    // Otherwise absorbing the transpose would dangle other readers.
    let consumers = &ud.consumers[w_t as usize];
    if consumers.len() != 1 {
        return (w_t, None);
    }
    // Single-axis-swap permutation check would belong here once the
    // emit grows layouts other than `[out, in]`. For the current
    // patterns (which all assume row-major `[out, in]` weight) any
    // 2-D transpose maps the original `[out, in]` to `[in, out]` —
    // and walking through restores `[out, in]`, which is what the
    // kernels emit.
    (producer.inputs[0], Some(producer.id))
}

/// Result of a successful pattern match.
pub struct Match {
    pub kind: FusionKind,
    pub op_ids: Vec<OpId>,
    pub inputs: Vec<FusionBoundary>,
    pub outputs: Vec<FusionBoundary>,
    pub attrs: FusionAttrs,
    pub parallel_axes: Vec<ParallelAxis>,
}

/// Trait shared by every fusion pattern. Phase 1 has a handful of
/// hand-written impls; the registry is just a `Vec<Box<dyn Pattern>>`
/// sorted by priority.
#[allow(dead_code)]
pub trait Pattern: Sync + Send {
    /// Unique identifier — used in `DecisionLog` entries.
    fn name(&self) -> &'static str;

    /// Larger values are tried first. Patterns covering more ops get
    /// higher priority so they win against narrower fallbacks (e.g.
    /// the full attention block beats just RMSNorm + projection).
    fn priority(&self) -> u32;

    /// Try to match starting at `anchor`. Returns `Some` if every
    /// required op and edge is present. Safety checks happen
    /// separately — a match here is purely structural.
    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match>;

    /// True for patterns that recognise structural subgraphs the rest
    /// of the compiler relies on — e.g. SDPA decode and the
    /// RoPE + KV-cache-store block. Structural patterns stay on even
    /// when `FUSEKIT_NO_PATTERNS=1` disables the hand-tuned epilogues,
    /// because the general-fusion path can't lower those subgraphs on
    /// its own yet (no multi-Heavy chain emit, no KV-cache stride
    /// awareness). Override in the impl when you add a new structural
    /// recogniser.
    fn is_structural(&self) -> bool {
        false
    }
}

pub mod final_rmsnorm_lm_head;
pub mod linear_residual;
pub mod rmsnorm_gateup;
pub mod rmsnorm_gateup_swiglu;
pub mod rmsnorm_gateup_swiglu_down;
pub mod rmsnorm_proj;
pub mod rmsnorm_qkv;
pub mod sdpa_decode;
pub mod swiglu_activation;

pub(crate) fn no_patterns_enabled() -> bool {
    std::env::var("FUSEKIT_NO_PATTERNS")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Build the default pattern registry. Patterns are sorted by
/// `priority()` so wider matches win when their narrower counterparts
/// would have matched the same anchor.
pub fn default_registry() -> Vec<Box<dyn Pattern>> {
    // FUSEKIT_DISABLE_PATTERNS lets us bypass selected matchers so the
    // Stage 3 general-fusion pass gets a chance to claim their ops —
    // mainly useful when validating that a generated chain matches the
    // hand-tuned pattern's output.
    //
    // FUSEKIT_NO_PATTERNS=1 (or `--patterns off`) disables ALL hand-
    // tuned patterns, leaving every op as `FusionKind::Single` for the
    // general fusion + piece composer to claim. This is the
    // "fuse generically" path — only useful once general fusion can
    // match the perf of the patterns it's replacing.
    // When NO_PATTERNS is on we still keep the *structural* recognisers
    // (SDPA, RoPE+KV store) — those aren't epilogue tuning, they
    // recognise subgraph shapes that the general fusion pass + chain
    // emitter can't lower on its own yet. Removing them would force a
    // per-op fallback for attention, which has nothing to do with the
    // "compiler path replaces patterns" goal.
    let no_patterns = no_patterns_enabled();
    let disabled: std::collections::HashSet<String> = std::env::var("FUSEKIT_DISABLE_PATTERNS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    // FUSEKIT_GATEUP_DOWN_MEGA=1 re-enables the all-in-one
    // rmsnorm_gateup_swiglu_down pattern. Default-off because it
    // forces single-TG dispatch (silu_up in SMEM = no parallelism
    // across SMs). Worth enabling when:
    //   - intermediate_size is small (silu_up fits SMEM without tiling)
    //   - per-layer dispatch overhead dominates (e.g. small models)
    //   - speculative decoding where mega-kernels reduce ICB rebuild
    // For Qwen-0.5B intermediate=4864 → 19KB SMEM ✓ fits; for
    // Qwen-27B intermediate=25600 → 100KB ✗ needs mlp_dim tiling
    // (see `rmsnorm_gateup_swiglu_down.rs` Phase 2 — silu_up tile
    // across SMEM chunks).
    let mega_gateup_down = std::env::var_os("FUSEKIT_GATEUP_DOWN_MEGA").is_some();
    let mut v: Vec<Box<dyn Pattern>> = vec![];
    if mega_gateup_down {
        v.push(Box::new(
            rmsnorm_gateup_swiglu_down::RmsNormGateUpSwigluDown,
        ));
    }
    v.extend::<Vec<Box<dyn Pattern>>>(vec![
        Box::new(rmsnorm_gateup_swiglu::RmsNormGateUpSwiglu),
        Box::new(rmsnorm_qkv::RmsNormQkv),
        Box::new(rmsnorm_gateup::RmsNormGateUp),
        Box::new(sdpa_decode::SdpaDecode),
        Box::new(swiglu_activation::SwigluActivation),
        Box::new(linear_residual::LinearResidual),
        Box::new(rmsnorm_proj::RmsNormProj),
        Box::new(final_rmsnorm_lm_head::FinalRmsNormLmHead),
    ]);
    if !disabled.is_empty() {
        v.retain(|p| !disabled.contains(p.name()));
    }
    if no_patterns {
        v.retain(|p| p.is_structural());
    }
    v.sort_by_key(|p| std::cmp::Reverse(p.priority()));
    v
}
