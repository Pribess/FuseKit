//! Fusekit compiler IR centred on a generic `Op` type.
//!
//! ```text
//!   mlx trace (graph::Graph) ─┐
//!                             │  lower_from_fusion
//!                             ▼
//!                       op::Module  ─►  passes (canonicalize / dce / fuse / tile)
//!                             │
//!                             │  lowering → MSL
//!                             ▼
//!                          metallib
//! ```
//!
//! Direction set 2026-05-28. Replaces fusekit's hardcoded `FusionKind`
//! enum with a generic `Op { kind, attrs }` representation so any
//! architecture (Qwen, DeepSeek MLA, Mamba, MoE) flows through the
//! same compiler pipeline. Hand-tuned synthetic kernels
//! (`RopeQDecode`, `EmbedLookupDecode`, etc.) become opaque
//! `OpKind::Synthetic { name }` nodes for now; later passes decompose
//! them into primitives that fuse with the surrounding compute.
//!
//! Naming: `op::Op` (this module's IR-level fused op) vs
//! `graph::Op` (the raw mlx trace op) — distinguished by module path.
//! Same convention for `op::Tensor` / `graph::Tensor`.
//!
//! Five sequential build-out steps tracked in TaskList (#20-24):
//! 1. Generic Op vocab (this file + `op.rs`)
//! 2. Tensor strides/layout
//! 3. Pass manager
//! 4. Algebraic rewrite rules (declarative)
//! 5. Cost model + scheduler
//!
//! Until each step lands, this module sits alongside
//! `fusion::FusionGraph` — `lower_from_fusion` translates back and
//! forth so neither pipeline breaks.

#![allow(dead_code)]

pub mod lower_from_fusion;
pub mod module;
pub mod op;
pub mod passes;

pub use module::{Module, ModuleMetaValue, Partition, Tensor};
pub use op::{Attr, AttrValue, ChainStep, Op, OpKind};
pub use passes::{
    estimate_op_cost, AlgebraicRewrites, Cost, DeadCodeElimination, DecodeL1Specialize,
    InPlaceAnalysis, KvSliceUpdateContiguous, LayoutInference, NoOpPass, Pass, PassError,
    PassManager, Rewrite, RewriteRule, RmsNormFastRsqrt, SoftmaxMaxStable,
};
