//! Stage 2 — fusion analysis.
//!
//! Inputs:  `crate::graph::Graph` (Stage 1, captured op DAG)
//!          `ir::ModelMeta`       (architecture constants from config.json)
//!
//! Outputs: `ir::FusionGraph`     (ops collapsed into fused regions)
//!
//! See `build.rs` for the high-level flow and `patterns/` for the
//! individual pattern matchers.

// Scaffolding for the fusion pass — types and stubs are intentionally
// unreferenced until the first pattern matcher and the `patch` command
// are wired up. Drop this allow once we plug everything in.
#![allow(dead_code)]

pub mod build;
pub mod category;
pub mod general;
pub mod ir;
pub mod layer_block;
pub mod patterns;
pub mod safety;
