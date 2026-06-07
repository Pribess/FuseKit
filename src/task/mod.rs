#![allow(dead_code)]

//! Stage 3 — task scheduling.
//!
//! Inputs:  `crate::fusion::ir::FusionGraph`
//! Outputs: `ir::TaskGraph`  (tasks + events + initial-ready set)
//!
//! Phase 1 emits a trivial 1-fusion = 1-task mapping with no
//! partitioning. Partition decisions and multi-task explosion of a
//! single fusion (head split, batch split, ...) land in Phase 2.

pub mod build;
pub mod ir;
