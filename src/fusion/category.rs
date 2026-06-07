//! Op semantic categorisation.
//!
//! The general-fusion pass (`fusion::general`) decides which Single
//! fusions can merge by asking *what kind of op is this*. Pattern
//! matchers above this layer already encode their own structural
//! knowledge; this enum exists for everything the matchers didn't
//! claim — i.e. ops that survive Stage 2 as `FusionKind::Single`.
//!
//! Categories drive the fusion rules:
//! - **Pointwise** ops chain freely. Output of one is the input of the
//!   next, with no inter-thread communication. Intermediate values
//!   stay in registers when fused.
//! - **View** ops compute nothing — they shift indexing. A view in the
//!   middle of a chain can be absorbed by re-deriving downstream
//!   indices.
//! - **Reduction** ops collapse a dimension (Softmax / RMSNorm along
//!   last axis). They need threadgroup memory for the reduce; a
//!   pointwise tail can be epilogue-fused into the same kernel.
//! - **Heavy** ops (Matmul, Convolution) define their own threadgroup
//!   structure. Downstream pointwise can be absorbed as an epilogue;
//!   upstream cannot be merged (different parallel axis).
//! - **Indexing / Materialized** ops re-layout memory. They're fusion
//!   barriers in both directions for now — fusing through them needs
//!   per-output indexing logic we don't generalise yet.
//! - **Generator** ops produce a tensor without consuming one (Arange,
//!   Full). They can chain into a pointwise consumer that broadcasts
//!   the constant.
//! - **CustomKernel** is an opaque user-provided Metal kernel. The
//!   semantics live in arbitrary Metal source — we can't inline it.
//!   It's a hard barrier in both directions.

use crate::graph::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    Pointwise,
    View,
    Reduction,
    Heavy,
    Indexing,
    Materialized,
    Generator,
    CustomKernel,
    Unknown,
}

impl OpCategory {
    /// Can a chain of (producer → consumer) fuse into one kernel given
    /// only their categories? Caller still has to verify the
    /// intermediate tensor has a single consumer and that shapes /
    /// parallel axes match.
    pub fn can_chain_into(self, consumer: OpCategory) -> bool {
        use OpCategory::*;
        match (self, consumer) {
            // CustomKernel is a fusion barrier in every direction.
            (CustomKernel, _) | (_, CustomKernel) => false,
            // Pointwise-pointwise: free.
            (Pointwise, Pointwise) => true,
            // View as a leading or interior step folds into the next
            // op's indexing.
            (View, _) | (_, View) => true,
            // Generator → Pointwise (constants get broadcast / consumed).
            (Generator, Pointwise) => true,
            // Pointwise epilogue on a Reduction or Heavy op.
            (Reduction, Pointwise) => true,
            (Heavy, Pointwise) => true,
            // Anything-into-Reduction or anything-into-Heavy needs
            // bespoke kernel reshaping. Out of scope for the
            // general-fusion pass; let pattern matchers handle those.
            _ => false,
        }
    }
}

/// Whether a `View` op is "element-contiguous" — Reshape, Flatten,
/// Unflatten, Squeeze, ExpandDims, AsContiguous all just rebrand the
/// shape without changing element order, so a chain can absorb them
/// as an identity step. Broadcast/Transpose change element layout
/// and need stride math — not chain-absorbable yet.
pub fn is_chain_absorbable_view(op_name: &str) -> bool {
    matches!(
        op_name,
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous"
    )
}

/// Map an mlx primitive name (as captured into `Op::name`) to its
/// category. Unknown names default to `OpCategory::Unknown` and the
/// general-fusion pass treats them as fusion barriers — safer than
/// guessing.
pub fn categorize(op: &Op) -> OpCategory {
    categorize_name(op.name.as_str())
}

/// Same as [`categorize`] but takes just the op name. Used by the
/// general-fusion pass which carries `FusionKind::Single(String)` and
/// doesn't always have an [`Op`] handy.
pub fn categorize_name(name: &str) -> OpCategory {
    use OpCategory::*;
    match name {
        // ── Pointwise (binary / unary / ternary) ──
        "Add" | "Multiply" | "Sigmoid" | "Exp" | "Negative" | "AsType" | "Select"
        | "GreaterEqual" | "LogAddExp" => Pointwise,

        // ── View-only (storage unchanged or stride-only) ──
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous"
        | "Broadcast" | "Transpose" => View,

        // ── Reduction along an axis ──
        "Softmax" | "RMSNorm" | "LayerNorm" => Reduction,

        // ── Heavy compute kernels ──
        "Matmul"
        | "AddMM"
        | "QuantizedMatmul"
        | "Convolution"
        | "ConvolutionSilu"
        | "RoPE"
        | "ScaledDotProductAttention" => Heavy,

        // ── Indexing / re-layout ──
        "Gather" | "Slice" | "Concatenate" | "Split" => Materialized,

        // ── Constant generators ──
        "Arange" | "Full" => Generator,

        // ── User-provided Metal kernel — opaque, fusion barrier ──
        "CustomKernel" => CustomKernel,

        // ── Quantize lands once per model load, not a hot-path fuse target.
        "Quantize" => Unknown,

        _ => Unknown,
    }
}
