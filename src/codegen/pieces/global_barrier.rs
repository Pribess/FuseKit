//! Cross-threadgroup synchronization piece.
//!
//! Metal's `threadgroup_barrier` only syncs threads within one TG.
//! To stitch piece outputs whose readers live in a different TG than
//! the writer (SDPA writes `sdpa_out[head_id]` in TG `head_id`,
//! LinearResidual reads every head's `sdpa_out` from every TG),
//! we need a grid-level barrier. Apple GPUs don't expose
//! cooperative groups, so we build one via device atomic + spin.
//!
//! Cost: ~few hundred cycles per TG (all TGs land at the barrier
//! together because they run the same body). Per-token cost for
//! 24 layers × 1 barrier = 24 × ~few hundred cycles. Comparable
//! to ICB's per-dispatch boundary cost; the perf comparison vs
//! kernel-split is empirical — that's the experiment.
//!
//! Layout: `barrier_counters` is a small `device atomic_uint` array
//! with one slot per (kernel, stage). The decode `step_init` utility
//! resets it on-GPU before each ICB execute so chunked decode does not
//! need a host round-trip per token.

use crate::codegen::msl::MslWriter;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece};

pub struct GlobalBarrier {
    /// Index of this barrier within the kernel's barrier array.
    /// Each LayerBlock kernel may have multiple stages, each with
    /// its own counter slot.
    pub stage_idx: u32,
    /// Total TGs dispatched for this kernel — every TG must reach
    /// the barrier exactly once per kernel call. The host pins
    /// the grid to this many TGs.
    pub tg_count: u32,
}

impl KernelPiece for GlobalBarrier {
    fn name(&self) -> &str {
        "global_barrier"
    }

    fn buffer_slots(&self, _ctx: &EmitCtx) -> Vec<BufferSlot> {
        vec![BufferSlot {
            var_name: "barrier_counters".into(),
            elem_ty: "atomic_uint".into(),
            access: BufferAccess::Write,
            role: "barrier_counters".into(),
        }]
    }

    fn emit_body(&self, w: &mut MslWriter, _ctx: &EmitCtx) {
        let stage = self.stage_idx;
        let tg_count = self.tg_count;
        w.writeln(format_args!(
            "// global barrier stage {stage} (tg_count={tg_count})"
        ));
        // Drain pending writes from the preceding stage so they're
        // visible globally before the counter increment.
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
        w.line("if (tid == 0u) {");
        w.push_indent();
        // Apple MSL only exposes `memory_order_relaxed`. The
        // surrounding `threadgroup_barrier(... | mem_device)` calls
        // handle the visibility ordering — the atomic just provides
        // the count semantics.
        w.writeln(format_args!(
            "atomic_fetch_add_explicit(&barrier_counters[{stage}u], 1u, memory_order_relaxed);"
        ));
        w.writeln(format_args!(
            "while (atomic_load_explicit(&barrier_counters[{stage}u], memory_order_relaxed) < {tg_count}u) {{}}"
        ));
        w.pop_indent();
        w.line("}");
        // Re-broadcast to all threads so they unblock together.
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
    }
}
