//! Metal dispatch primitives used by the smoke test and (later) the
//! Python-side runtime wrapper.
//!
//! ```text
//!   MetalContext ─► pipeline(name)
//!                       │
//!                       ▼
//!                  ComputePipelineState
//!                       │
//!     buffers           │
//!     ┌────────────┐    │  encoder.set_buffer(slot, ...)
//!     │ inputs     │────┤  encoder.set_compute_pipeline_state(...)
//!     │ outputs    │────┤  encoder.dispatch_threads(grid, tg)
//!     │ uniforms   │────┘
//!     └────────────┘
//!                       │
//!                       ▼
//!                  command buffer commit + wait
//! ```

use anyhow::{anyhow, Context, Result};
use metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use crate::runtime::metal::MetalContext;

/// Run `elementwise_add_bf16(a, b, out, n)` for our smoke test.
///
/// `a` and `b` must hold `n` bf16 values; `out` receives `n` bf16
/// values. Buffers are read+written via shared (CPU+GPU) storage so we
/// can validate from the host afterwards.
pub fn elementwise_add_bf16(
    ctx: &MetalContext,
    a: &[half::bf16],
    b: &[half::bf16],
) -> Result<Vec<half::bf16>> {
    if a.len() != b.len() {
        return Err(anyhow!("elementwise_add_bf16: length mismatch"));
    }
    let n = a.len();
    let nbytes = n * std::mem::size_of::<half::bf16>();

    let buf_a = ctx.buffer_from_slice(a);
    let buf_b = ctx.buffer_from_slice(b);
    let buf_out = ctx.new_buffer(nbytes);
    let n_u32 = n as u32;
    let buf_n = ctx.buffer_from_slice(std::slice::from_ref(&n_u32));

    let pso = ctx.pipeline("elementwise_add_bf16")?;
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_buffer(3, Some(&buf_n), 0);

    // 1-D grid sized to `n`, threadgroup of 256 (works on every M-series).
    let threadgroup = MTLSize::new(256, 1, 1);
    let grid = MTLSize::new(((n + 255) / 256 * 256) as u64, 1, 1);
    enc.dispatch_threads(grid, threadgroup);
    let _: &ComputeCommandEncoderRef = enc; // keep the type explicit for readers
    enc.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // Shared-storage buffer is host-visible; read straight from the
    // pointer.
    let ptr = buf_out.contents() as *const half::bf16;
    let result = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
    Ok(result)
}

/// Convenience for callers that want a raw pipeline by name. Kept in
/// `dispatch.rs` so all kernel-state cache logic lives in one place if
/// we ever add caching.
pub fn pipeline_for(ctx: &MetalContext, name: &str) -> Result<ComputePipelineState> {
    ctx.pipeline(name)
        .with_context(|| format!("pipeline {name}"))
}
