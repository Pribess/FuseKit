//! Metal device / queue / library plumbing.
//!
//! The compiled kernel library lives in `OUT_DIR/fusekit_kernels.metallib`
//! and is embedded via `include_bytes!` so the binary stays self-contained.

use anyhow::{anyhow, Context, Result};
use metal::{CommandQueue, Device, Library, MTLResourceOptions};

/// Embedded at compile time; path comes from `build.rs`.
const KERNELS_METALLIB: &[u8] = include_bytes!(env!("FUSEKIT_KERNELS_METALLIB"));

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_data(KERNELS_METALLIB)
            .map_err(|e| anyhow!("loading kernel metallib: {e}"))?;
        Ok(Self {
            device,
            queue,
            library,
        })
    }

    /// Load a model-specific metallib (the one `fusekit compile` writes
    /// to `<model>/fusekit/kernel.metallib`) instead of the embedded
    /// smoke-test library. Used by the runtime auto-dispatcher.
    pub fn with_metallib(path: &std::path::Path) -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let queue = device.new_command_queue();
        let bytes =
            std::fs::read(path).map_err(|e| anyhow!("reading {}: {}", path.display(), e))?;
        let library = device
            .new_library_with_data(&bytes)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        Ok(Self {
            device,
            queue,
            library,
        })
    }

    /// Allocate a device-private buffer of `nbytes` and zero-initialize it.
    pub fn new_buffer(&self, nbytes: usize) -> metal::Buffer {
        self.device
            .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate a buffer populated from a host slice.
    pub fn buffer_from_slice<T: Copy>(&self, data: &[T]) -> metal::Buffer {
        let nbytes = std::mem::size_of_val(data);
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            nbytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        buf
    }

    /// Look up a kernel by name and build a compute pipeline state.
    /// `supportIndirectCommandBuffers` is enabled so the PSO can be
    /// bound inside a Metal ICB — otherwise encoding into the ICB
    /// silently hangs the first time the PSO is set.
    pub fn pipeline(&self, fn_name: &str) -> Result<metal::ComputePipelineState> {
        let function = self
            .library
            .get_function(fn_name, None)
            .map_err(|e| anyhow!("kernel `{fn_name}` not found: {e}"))?;
        let desc = metal::ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&function));
        desc.set_support_indirect_command_buffers(true);
        self.device
            .new_compute_pipeline_state(&desc)
            .map_err(|e| anyhow!("pipeline for `{fn_name}`: {e}"))
            .context("compute pipeline")
    }
}
