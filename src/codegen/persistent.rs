#![allow(dead_code)]

//! Persistent megakernel scheduler emit.
//!
//! ```text
//!   ┌────────────────── single Metal kernel launch ──────────────────┐
//!   │                                                                 │
//!   │   ┌──────────────┐         ┌──────────────────────────────┐     │
//!   │   │ scheduler tg │ enqueue │  worker tg 0                  │     │
//!   │   │              │────────►│   loop:                       │     │
//!   │   │ - poll events│         │     pop task                  │     │
//!   │   │ - enqueue    │         │     wait on deps              │     │
//!   │   │   ready tasks│         │     dispatch by (kind,variant)│     │
//!   │   │   to workers │         │     atomic ++ trigger event   │     │
//!   │   └──────────────┘         └──────────────────────────────┘     │
//!   │                            ┌──────────────────────────────┐     │
//!   │                            │  worker tg 1, 2, ...          │     │
//!   │                            └──────────────────────────────┘     │
//!   │                                                                 │
//!   └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Apple-Silicon-specific design notes
//!
//! The mirage MPK model assumes CUDA semantics (cooperative groups,
//! `cuda::pipeline`, `cudaStreamWaitValue`). Metal has none of these
//! natively, so cross-threadgroup sync inside one kernel must ride on
//! device-memory atomic counters plus spin-wait. That works on Apple
//! Silicon because all threadgroups in a single `dispatchThreads`
//! launch are co-resident in SLM — IF the grid is small enough to fit
//! the GPU's concurrent-threadgroup capacity (≈64–256 per chip). Bigger
//! grids deadlock: queued threadgroups never run while the first wave
//! spin-waits for them.
//!
//! Practical implication: a Metal megakernel needs **persistent
//! threadgroups** — launch N TGs sized to the GPU's capacity, each TG
//! loops over multiple output rows per phase. This is the same shape
//! as CUDA persistent-thread kernels (e.g., MPK's worker loop).
//!
//! ## Status
//!
//! - Cross-kernel cofusion landed (`FusionKind::RmsNormGateUpSwiglu`):
//!   one combined MSL kernel keeps gate/up in registers between the
//!   gate/up projection and the SwiGLU multiply, eliminating
//!   2·N_INT·bf16 of round-trip traffic per layer. Demonstrates the
//!   class of optimization a full megakernel would do at scale.
//! - Per-pattern dispatch + command-buffer batching is the current
//!   production path. Cross-TG atomic-barrier mode (this file's
//!   eventual job) still pending.

use anyhow::Result;

use crate::codegen::msl::MslWriter;
use crate::task::ir::TaskGraph;

#[allow(unused_variables)]
pub fn emit_megakernel(graph: &TaskGraph, w: &mut MslWriter) -> Result<()> {
    // Real scheduler emission lands here. Structure (when filled in):
    //
    //   kernel void fusekit_megakernel(...) {
    //       if (tg_id == 0) { run_scheduler(...); return; }
    //       worker_loop(tg_id - 1);
    //   }
    //
    //   void run_scheduler(...) { ... event processing ... }
    //   void worker_loop(uint wid) {
    //       while (true) {
    //           TaskId t = pop(wid);
    //           if (t == TASK_TERMINATE) return;
    //           wait_for_deps(t);
    //           switch ((kind(t), variant(t))) { ... per-kernel calls ... }
    //           atomic_fetch_add_explicit(&counter[trigger(t)], 1, ...);
    //       }
    //   }
    Ok(())
}
