//! Python-callable Rust dispatcher.
//!
//! ```text
//!   Python                              Rust (this crate)
//!   ──────                              ─────────────────
//!   fusekit_runtime/                    _fusekit_runtime_native.so
//!     __init__.py                         load_dispatcher(metallib, tg)
//!         │                                 │   ├─ MTLDevice + queue
//!         │                                 │   ├─ load metallib
//!         │                                 │   └─ build ComputePipelineState
//!         │  pyo3                           │       per unique kernel
//!         ▼                                 │
//!     FusekitWrapper.__call__ ────────────► dispatch_forward(handle, ...)
//!                                              [phase 2 — Metal encoding]
//! ```
//!
//! Phase 1.5 (this iteration): the dispatcher actually opens a Metal
//! device, loads the per-model metallib, and builds a pipeline state
//! for every kernel function in it. `dispatch_forward` doesn't encode
//! commands yet — that needs the mx.array → MTLBuffer bridge, which
//! lands once we wire up the MLX C++ headers via cxx.

use metal::{
    ComputePipelineDescriptor, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, Library, MTLIndirectCommandType, MTLResourceOptions,
    MTLResourceUsage, MTLSize, NSRange,
};
#[macro_use]
extern crate objc;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;

/// Per-model state. Pipelines are pre-built so dispatch hot-path is
/// just buffer binding + encode + commit.
struct Dispatcher {
    metallib_path: PathBuf,
    task_graph: serde_json::Value,
    device: Device,
    queue: metal::CommandQueue,
    library: Library,
    kernel_names: Vec<String>,
    /// Pre-built ComputePipelineState per kernel name. `new_compute_*`
    /// is shader compilation — running it on every dispatch can cost
    /// hundreds of microseconds and quickly dominates the runtime.
    pipelines: HashMap<String, ComputePipelineState>,
    /// Pre-encoded indirect command buffers — index is the handle
    /// returned by `build_icb`. Each entry stores the ICB itself and
    /// the resident-buffer set the encoder must declare via
    /// `useResource:` so the GPU is allowed to touch them.
    icbs: Vec<IcbEntry>,
    /// Most-recently submitted (but not yet waited-on) command
    /// buffer from `submit_icb_async`. `wait_pending_icb` consumes
    /// this for the user-visible per-token barrier.
    last_pending_cmd: Option<metal::CommandBuffer>,
    /// `MTLBuffer` objects allocated via `alloc_uniform`. Kept so we
    /// can recover the `&BufferRef` from a raw data pointer for blit
    /// encoder operations during pipelined decode.
    uniform_buffers: Vec<metal::Buffer>,
    /// Map from `alloc_uniform`-returned data pointer → index into
    /// `uniform_buffers`. Used by the pipelined chunk path to look
    /// up buffer handles for blit `copy_from_buffer` operations.
    uniform_ptr_to_idx: HashMap<usize, usize>,
    /// Host-backed buffers registered by Python as arenas. Slice
    /// pointers inside these ranges are bound as `(base_buffer, offset)`
    /// instead of being wrapped with `newBufferWithBytesNoCopy` again,
    /// which avoids Metal alignment requirements for subviews.
    registered_host_buffers: Vec<RegisteredHostBuffer>,
}

struct RegisteredHostBuffer {
    base: usize,
    nbytes: usize,
    buffer: metal::Buffer,
}

struct IcbEntry {
    icb: IndirectCommandBuffer,
    n_commands: u64,
    resident_bufs: Vec<metal::Buffer>,
    slot_resource_indices: HashMap<(u64, u64), usize>,
    // Pre-flattened `&[&ResourceRef]` view over `resident_bufs` for
    // `use_resources`. The owning buffers live as long as the entry
    // (entries are append-only inside the static `DISPATCHERS`
    // mutex), so the references stay valid; we lengthen the lifetime
    // to `'static` so we can hand the slice out per-submit without
    // re-collecting a Vec each time — that ~50-entry allocation was
    // ~5–10us per token.
    resident_refs: Vec<&'static metal::ResourceRef>,
}

fn refresh_resident_refs(entry: &mut IcbEntry) {
    let mut active: Vec<usize> = entry.slot_resource_indices.values().copied().collect();
    active.sort_unstable();
    active.dedup();
    entry.resident_refs = active
        .into_iter()
        .filter_map(|idx| entry.resident_bufs.get(idx))
        .map(|b| unsafe {
            std::mem::transmute::<&metal::ResourceRef, &'static metal::ResourceRef>(&**b)
        })
        .collect();
}

// Send + Sync is fine: Metal objects are reference-counted via objc
// and the metal crate's wrappers are documented as thread-safe.
unsafe impl Send for Dispatcher {}
unsafe impl Sync for Dispatcher {}

static DISPATCHERS: Mutex<Vec<Dispatcher>> = Mutex::new(Vec::new());

/// Load a per-model dispatcher. Returns an integer handle.
#[pyfunction]
fn load_dispatcher(metallib_path: String, task_graph_path: String) -> PyResult<usize> {
    let tg_bytes = std::fs::read(&task_graph_path).map_err(io_err)?;
    let task_graph: serde_json::Value =
        serde_json::from_slice(&tg_bytes).map_err(|e| value_err(format!("parse tg: {e}")))?;

    if !std::path::Path::new(&metallib_path).is_file() {
        return Err(file_not_found(format!("metallib missing: {metallib_path}")));
    }

    let device = Device::system_default()
        .ok_or_else(|| value_err("no Metal device available".to_string()))?;
    let queue = device.new_command_queue();

    // `new_library_with_file` accepts a metallib produced by xcrun.
    let library = device
        .new_library_with_file(&metallib_path)
        .map_err(|e| value_err(format!("load metallib: {e}")))?;

    let kernel_names = library.function_names();

    // Pre-build every kernel's pipeline state with ICB support so the
    // same PSOs work for both direct dispatch and indirect (ICB) paths.
    let mut pipelines: HashMap<String, ComputePipelineState> = HashMap::new();
    for name in &kernel_names {
        let function = library
            .get_function(name, None)
            .map_err(|e| value_err(format!("get {name}: {e}")))?;
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&function));
        desc.set_support_indirect_command_buffers(true);
        let pso = device
            .new_compute_pipeline_state(&desc)
            .map_err(|e| value_err(format!("build pipeline for {name}: {e}")))?;
        pipelines.insert(name.clone(), pso);
    }

    let mut v = DISPATCHERS.lock().unwrap();
    let handle = v.len();
    v.push(Dispatcher {
        metallib_path: PathBuf::from(metallib_path),
        task_graph,
        device,
        queue,
        library,
        kernel_names,
        pipelines,
        icbs: Vec::new(),
        last_pending_cmd: None,
        uniform_buffers: Vec::new(),
        uniform_ptr_to_idx: HashMap::new(),
        registered_host_buffers: Vec::new(),
    });
    Ok(handle)
}

fn registered_host_buffer_for(
    d: &Dispatcher,
    ptr: usize,
    nbytes: usize,
) -> Option<(metal::Buffer, u64, usize)> {
    let end = ptr.checked_add(nbytes)?;
    for registered in &d.registered_host_buffers {
        let range_end = registered.base.checked_add(registered.nbytes)?;
        if ptr >= registered.base && end <= range_end {
            return Some((
                registered.buffer.clone(),
                (ptr - registered.base) as u64,
                registered.base,
            ));
        }
    }
    None
}

/// Return diagnostic info about a loaded dispatcher. Used by the
/// Python wrapper for the attach banner and by tests to verify the
/// metallib loaded correctly.
#[pyfunction]
fn dispatcher_info(handle: usize) -> PyResult<String> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let task_count = d
        .task_graph
        .get("task_count")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            d.task_graph
                .get("tasks")
                .and_then(|v| v.as_array())
                .map(|tasks| tasks.len() as u64)
        })
        .unwrap_or(0);
    Ok(format!(
        "device={} metallib={} tasks={} kernels={}",
        d.device.name(),
        d.metallib_path.display(),
        task_count,
        d.kernel_names.len(),
    ))
}

/// List kernel functions exported by the loaded metallib. Returned to
/// Python as a list of strings so the wrapper can sanity-check that
/// the kernels referenced by task_graph.json are actually present.
#[pyfunction]
fn kernel_names(handle: usize) -> PyResult<Vec<String>> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    Ok(d.kernel_names.clone())
}

/// Phase-2 dry run: encode and dispatch the first kernel in the
/// loaded metallib with dummy zero-filled buffers. Validates that the
/// full Python → Rust → Metal command-buffer path works before we
/// connect real mx.array data.
///
/// Returns a status string with the dispatch timing so the Python
/// side can sanity-check the wiring.
#[pyfunction]
fn smoke_dispatch(handle: usize) -> PyResult<String> {
    use metal::MTLResourceOptions;
    use std::time::Instant;

    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;

    let Some(kernel_name) = d.kernel_names.first() else {
        return Ok("no kernels in metallib".to_string());
    };

    let function = d
        .library
        .get_function(kernel_name, None)
        .map_err(|e| value_err(format!("get_function({kernel_name}): {e}")))?;
    let pso = d
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| value_err(format!("build pipeline: {e}")))?;

    // Allocate a small zero buffer for every parameter slot the
    // function declares. We don't know the real shapes here, so we
    // just give each buffer enough room for the kernel to read/write
    // without crashing on an obvious OOB. 4 KB per buffer is plenty
    // for the tiny RMSNorm + projection state in our smallest model.
    const SLOT_BYTES: usize = 4096;
    const NUM_SLOTS: usize = 16;
    let buffers: Vec<metal::Buffer> = (0..NUM_SLOTS)
        .map(|_| {
            d.device
                .new_buffer(SLOT_BYTES as u64, MTLResourceOptions::StorageModeShared)
        })
        .collect();

    let started = Instant::now();
    let cmd = d.queue.new_command_buffer_with_unretained_references();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    for (i, b) in buffers.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    // 1-D minimal grid; we're not validating output, just exercising
    // the dispatch + completion path.
    enc.dispatch_threads(metal::MTLSize::new(64, 1, 1), metal::MTLSize::new(64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let elapsed = started.elapsed();

    Ok(format!(
        "smoke ok: kernel={} dispatch_time={}µs",
        kernel_name,
        elapsed.as_micros()
    ))
}

/// Placeholder for the production forward path. Real implementation
/// lands once the mx.array → MTLBuffer bridge does.
#[pyfunction]
fn dispatch_forward(handle: usize, _input_shape: Vec<i64>) -> PyResult<String> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    Ok(format!(
        "fusekit dispatch (passthrough, {} kernels ready)",
        d.kernel_names.len()
    ))
}

/// Test kernels metallib, baked in at compile time. Used by the
/// buffer-bridge smoke path; not shipped to user models.
const TEST_METALLIB: &[u8] = include_bytes!(env!("FUSEKIT_TEST_METALLIB"));

/// Wrap an externally-allocated, host-visible buffer (e.g. the
/// CPU-mapped pointer behind an mx.array on Apple Silicon) as an
/// MTLBuffer and run the `scale_f32` test kernel in place.
///
/// ```text
///   mx.array (Metal, unified mem)
///     │
///     │  __dlpack__()  →  raw pointer
///     │
///     ▼
///   ptr ──► newBufferWithBytesNoCopy ──► MTLBuffer (zero-copy wrap)
///                                            │
///                                            ▼
///                                       dispatch scale_f32 in-place
///                                            │
///   mx.array sees the result ◄──────────────┘
///       (same physical memory, unified arch)
/// ```
///
/// Caller is responsible for keeping the underlying mx.array alive
/// while this runs — we don't take ownership of the pointer.
#[pyfunction]
fn scale_inplace_f32(ptr: usize, n: usize, scale: f32) -> PyResult<()> {
    use metal::{MTLResourceOptions, MTLSize};

    let device = metal::Device::system_default()
        .ok_or_else(|| value_err("no Metal device available".to_string()))?;
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_data(TEST_METALLIB)
        .map_err(|e| value_err(format!("load test metallib: {e}")))?;
    let function = lib
        .get_function("scale_f32", None)
        .map_err(|e| value_err(format!("get scale_f32: {e}")))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| value_err(format!("build pipeline: {e}")))?;

    // Wrap the host-visible pointer as an MTLBuffer without copying.
    // Apple Silicon unified memory makes this a real zero-copy share
    // with the originating mx.array.
    let nbytes = n * std::mem::size_of::<f32>();
    let buf = device.new_buffer_with_bytes_no_copy(
        ptr as *const std::ffi::c_void,
        nbytes as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let scale_val: f32 = scale;
    let buf_scale = device.new_buffer_with_data(
        &scale_val as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let n_val = n as u32;
    let buf_n = device.new_buffer_with_data(
        &n_val as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_buffer(1, Some(&buf), 0); // in-place: out aliases a
    enc.set_buffer(2, Some(&buf_scale), 0);
    enc.set_buffer(3, Some(&buf_n), 0);
    let tg = MTLSize::new(256, 1, 1);
    let grid_x = ((n + 255) / 256 * 256) as u64;
    enc.dispatch_threads(MTLSize::new(grid_x, 1, 1), tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

/// Dispatch a kernel from a *loaded model's* metallib (selected by
/// `handle`) against externally-allocated buffers. This is the
/// production-side cousin of `dispatch_test_kernel` and the entry
/// point the Python wrapper uses to hand off real model state to
/// fusekit.
#[pyfunction]
fn dispatch_kernel(
    handle: usize,
    kernel_name: String,
    buffers: Vec<(usize, usize)>,
    grid: (u64, u64, u64),
    threadgroup: (u64, u64, u64),
) -> PyResult<()> {
    use metal::{MTLResourceOptions, MTLSize};

    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;

    let pso = d
        .pipelines
        .get(&kernel_name)
        .ok_or_else(|| value_err(format!("kernel not in metallib: {kernel_name}")))?;

    let mtl_bufs: Vec<metal::Buffer> = buffers
        .iter()
        .map(|&(ptr, nbytes)| {
            d.device.new_buffer_with_bytes_no_copy(
                ptr as *const std::ffi::c_void,
                nbytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        })
        .collect();

    let cmd = d.queue.new_command_buffer_with_unretained_references();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    for (i, b) in mtl_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.dispatch_threads(
        MTLSize::new(grid.0, grid.1, grid.2),
        MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

/// Pre-encode a batch of dispatches into an `MTLIndirectCommandBuffer`
/// (ICB). The ICB is GPU-resident and can be re-executed every token
/// with zero encoding overhead — same dispatch structure, but the
/// buffer pointers / pipelines are baked in once.
///
/// Returns an opaque `usize` handle; pass to `execute_icb` to run.
///
/// Caller must ensure that:
/// 1. The dispatched buffer pointers don't move between calls (we use
///    pre-allocated scratch + the model's frozen weights, so this
///    holds for us).
/// 2. Uniforms whose values change per-call are written into
///    pre-allocated buffers via `alloc_uniform` — the kernel still
///    reads from the same address.
#[pyfunction]
#[pyo3(signature = (handle, dispatches, barriers=None))]
fn build_icb(
    handle: usize,
    dispatches: Vec<(
        String,
        Vec<(usize, usize)>,
        (u64, u64, u64),
        (u64, u64, u64),
    )>,
    barriers: Option<Vec<bool>>,
) -> PyResult<usize> {
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;

    let desc = IndirectCommandBufferDescriptor::new();
    desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatchThreads);
    desc.set_inherit_buffers(false);
    desc.set_inherit_pipeline_state(false);
    // Size the ICB descriptor from the actual dispatch ABI. Fused
    // kernels can carry semantic side-effect slots beyond IR
    // inputs/outputs (for example QKV decode sinks bind cache K/V plus
    // offset/capacity), so a fixed "typical" slot count silently
    // under-binds valid kernels in ICB mode even though direct dispatch
    // works.
    let max_buffer_slots = dispatches
        .iter()
        .map(|(_, buffers, _, _)| buffers.len())
        .max()
        .unwrap_or(0)
        .max(32);
    desc.set_max_kernel_buffer_bind_count(max_buffer_slots as u64);

    let n = dispatches.len() as u64;
    let icb = d.device.new_indirect_command_buffer_with_descriptor(
        &desc,
        n,
        MTLResourceOptions::StorageModePrivate,
    );

    // Wrap each unique external pointer as a zero-copy MTLBuffer ONCE
    // — the same MTLBuffer object must be referenced everywhere so
    // the encoder's `useResource:` covers the current working set.
    let mut buf_cache: HashMap<usize, usize> = HashMap::new();
    let mut resident_bufs: Vec<metal::Buffer> = Vec::new();
    let mut slot_resource_indices: HashMap<(u64, u64), usize> = HashMap::new();
    let mut get_buf = |ptr: usize, nbytes: usize| -> (metal::Buffer, usize, u64) {
        let registered = registered_host_buffer_for(d, ptr, nbytes);
        let (offset, cache_key) = registered
            .as_ref()
            .map(|(_, offset, base)| (*offset, *base))
            .unwrap_or((0, ptr));
        if let Some(&idx) = buf_cache.get(&cache_key) {
            return (resident_bufs[idx].clone(), idx, offset);
        }
        let buf = registered.map(|(buf, _, _)| buf).unwrap_or_else(|| {
            d.device.new_buffer_with_bytes_no_copy(
                ptr as *const c_void,
                nbytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        });
        let idx = resident_bufs.len();
        resident_bufs.push(buf.clone());
        buf_cache.insert(cache_key, idx);
        (buf, idx, offset)
    };

    for (i, (kernel_name, buffers, grid, threadgroup)) in dispatches.iter().enumerate() {
        let pso = d
            .pipelines
            .get(kernel_name)
            .ok_or_else(|| value_err(format!("kernel not in metallib: {kernel_name}")))?;
        let cmd = icb.indirect_compute_command_at_index(i as u64);
        cmd.set_compute_pipeline_state(pso);
        for (b_idx, &(ptr, nbytes)) in buffers.iter().enumerate() {
            let (mtl_buf, buf_idx, offset) = get_buf(ptr, nbytes);
            cmd.set_kernel_buffer(b_idx as u64, Some(&mtl_buf), offset);
            slot_resource_indices.insert((i as u64, b_idx as u64), buf_idx);
        }
        // Explicit barrier only when a real data dependency exists.
        // Caller does the analysis (it has the IR with read/write
        // sets); we default to barrier-all to stay safe for
        // legacy paths that don't pass a `barriers` list.
        let needs_barrier = barriers
            .as_ref()
            .and_then(|b| b.get(i))
            .copied()
            .unwrap_or(true);
        if needs_barrier {
            cmd.set_barrier();
        }
        cmd.concurrent_dispatch_threads(
            MTLSize::new(grid.0, grid.1, grid.2),
            MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
        );
    }
    let idx = d.icbs.len();
    let mut entry = IcbEntry {
        icb,
        n_commands: n,
        resident_bufs,
        slot_resource_indices,
        resident_refs: Vec::new(),
    };
    refresh_resident_refs(&mut entry);
    d.icbs.push(entry);
    Ok(idx)
}

/// Patch buffer bindings in an existing ICB without rebuilding every command.
/// Each update is `(command_index, buffer_slot, data_ptr, nbytes)`.
#[pyfunction]
fn update_icb_bindings(
    handle: usize,
    icb_id: usize,
    updates: Vec<(usize, u64, usize, usize)>,
) -> PyResult<()> {
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let registered_buffers: Vec<(usize, usize, metal::Buffer)> = d
        .registered_host_buffers
        .iter()
        .map(|b| (b.base, b.nbytes, b.buffer.clone()))
        .collect();
    let entry = d
        .icbs
        .get_mut(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;

    let mut new_buf_cache: HashMap<usize, usize> = HashMap::new();
    for (command_index, slot, ptr, nbytes) in updates {
        if command_index as u64 >= entry.n_commands {
            return Err(value_err(format!(
                "invalid ICB command index {command_index} for {count} commands",
                count = entry.n_commands
            )));
        }
        if ptr == 0 || nbytes == 0 {
            return Err(value_err(
                "cannot bind null/empty buffer into ICB".to_string(),
            ));
        }
        let mut registered_match: Option<(metal::Buffer, u64, usize)> = None;
        let end = ptr
            .checked_add(nbytes)
            .ok_or_else(|| value_err("buffer pointer range overflow".to_string()))?;
        for (base, size, buffer) in &registered_buffers {
            let range_end = base
                .checked_add(*size)
                .ok_or_else(|| value_err("registered buffer range overflow".to_string()))?;
            if ptr >= *base && end <= range_end {
                registered_match = Some((buffer.clone(), (ptr - *base) as u64, *base));
                break;
            }
        }
        let (offset, cache_key) = registered_match
            .as_ref()
            .map(|(_, offset, base)| (*offset, *base))
            .unwrap_or((0, ptr));
        let buf_idx = if let Some(&idx) = new_buf_cache.get(&cache_key) {
            idx
        } else {
            let mtl_buf = registered_match
                .map(|(buffer, _, _)| buffer)
                .unwrap_or_else(|| {
                    d.device.new_buffer_with_bytes_no_copy(
                        ptr as *const c_void,
                        nbytes as u64,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                });
            let idx = entry.resident_bufs.len();
            entry.resident_bufs.push(mtl_buf);
            new_buf_cache.insert(cache_key, idx);
            idx
        };
        let mtl_buf = entry.resident_bufs[buf_idx].clone();
        let cmd = entry
            .icb
            .indirect_compute_command_at_index(command_index as u64);
        cmd.set_kernel_buffer(slot, Some(&mtl_buf), offset);
        entry
            .slot_resource_indices
            .insert((command_index as u64, slot), buf_idx);
    }
    refresh_resident_refs(entry);
    Ok(())
}

/// Async submit: commit the ICB without waiting. Caller must invoke
/// `wait_pending_icb` (or another submit, which transitively serializes)
/// before reading any buffer the ICB writes. Used by the fast generate
/// loop to pipeline GPU work with CPU-side yield/sample.
#[pyfunction]
fn submit_icb_async(handle: usize, icb_id: usize) -> PyResult<()> {
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let entry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;

    let cmd = d.queue.new_command_buffer_with_unretained_references();
    let enc = cmd.new_compute_command_encoder();
    // Batched useResources:count:usage: — single ObjC msg_send
    // instead of one per resource. The pre-flattened
    // `resident_refs` slice avoids re-collecting a ~50-entry Vec on
    // each per-token submit.
    enc.use_resources(
        &entry.resident_refs,
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
    let range = NSRange {
        location: 0,
        length: entry.n_commands,
    };
    let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
    unsafe {
        let _: () = objc::msg_send![
            enc,
            executeCommandsInBuffer: icb_ref
            withRange: range
        ];
    }
    enc.end_encoding();
    cmd.commit();
    // Stash the most-recent command buffer so callers can wait on it.
    d.last_pending_cmd = Some(cmd.to_owned());
    Ok(())
}

/// Block until the most-recently submitted ICB completes. Returns
/// GPU-only elapsed time (seconds) or 0.0 if nothing was pending.
#[pyfunction]
fn wait_pending_icb(handle: usize) -> PyResult<f64> {
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let cmd = match d.last_pending_cmd.take() {
        Some(c) => c,
        None => return Ok(0.0),
    };
    cmd.wait_until_completed();
    let gpu_start: f64 = unsafe { objc::msg_send![cmd, GPUStartTime] };
    let gpu_end: f64 = unsafe { objc::msg_send![cmd, GPUEndTime] };
    Ok(gpu_end - gpu_start)
}

/// Execute a pre-built ICB. Returns GPU-only elapsed time (seconds).
#[pyfunction]
fn execute_icb(handle: usize, icb_id: usize) -> PyResult<f64> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let entry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;

    let cmd = d.queue.new_command_buffer_with_unretained_references();
    let enc = cmd.new_compute_command_encoder();
    // Declare every buffer the ICB will touch as resident. Without
    // this, GPU page-faults on the first access from inside the ICB.
    enc.use_resources(
        &entry.resident_refs,
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
    // metal-rs 0.29 doesn't expose executeCommandsInBuffer on
    // ComputeCommandEncoderRef (only the render variant). Use a
    // direct ObjC message — the selector exists on the underlying
    // MTLComputeCommandEncoder protocol.
    let range = NSRange {
        location: 0,
        length: entry.n_commands,
    };
    let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
    unsafe {
        let _: () = objc::msg_send![
            enc,
            executeCommandsInBuffer: icb_ref
            withRange: range
        ];
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_start: f64 = unsafe { objc::msg_send![cmd, GPUStartTime] };
    let gpu_end: f64 = unsafe { objc::msg_send![cmd, GPUEndTime] };
    Ok(gpu_end - gpu_start)
}

/// Execute a sub-range of a pre-built ICB. This is intentionally a
/// debugging/validation primitive: it uses the exact same encoded
/// indirect commands and resource residency path as `execute_icb`, but
/// limits execution to `[start, start + count)`. That lets Python
/// compare side-effecting decode state (KV rows, recurrent cache
/// state, scratch) immediately after a semantic boundary without
/// switching to the direct `dispatch_batch` path.
#[pyfunction]
fn execute_icb_range(handle: usize, icb_id: usize, start: usize, count: usize) -> PyResult<f64> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let entry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;
    let start_u64 = start as u64;
    let count_u64 = count as u64;
    if count_u64 == 0 || start_u64 >= entry.n_commands || start_u64 + count_u64 > entry.n_commands {
        return Err(value_err(format!(
            "invalid ICB range start={start} count={count} commands={commands}",
            commands = entry.n_commands
        )));
    }

    let cmd = d.queue.new_command_buffer_with_unretained_references();
    let enc = cmd.new_compute_command_encoder();
    enc.use_resources(
        &entry.resident_refs,
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
    let range = NSRange {
        location: start_u64,
        length: count_u64,
    };
    let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
    unsafe {
        let _: () = objc::msg_send![
            enc,
            executeCommandsInBuffer: icb_ref
            withRange: range
        ];
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_start: f64 = unsafe { objc::msg_send![cmd, GPUStartTime] };
    let gpu_end: f64 = unsafe { objc::msg_send![cmd, GPUEndTime] };
    Ok(gpu_end - gpu_start)
}

/// Execute an ICB one indirect command at a time and return GPU time
/// per command. This is a profiling primitive, not the normal fast
/// path: each command is submitted as its own command buffer so we can
/// read `GPUStartTime/GPUEndTime` per generated kernel while still
/// using the exact encoded ICB command and residency set that the
/// runtime normally uses.
#[pyfunction]
#[pyo3(signature = (handle, icb_id, count=None))]
fn profile_icb_commands(handle: usize, icb_id: usize, count: Option<usize>) -> PyResult<Vec<f64>> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let entry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;
    let n = count.unwrap_or(entry.n_commands as usize);
    if n == 0 || n as u64 > entry.n_commands {
        return Err(value_err(format!(
            "invalid ICB profile count={n} commands={commands}",
            commands = entry.n_commands
        )));
    }

    let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let cmd = d.queue.new_command_buffer_with_unretained_references();
        let enc = cmd.new_compute_command_encoder();
        enc.use_resources(
            &entry.resident_refs,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
        let range = NSRange {
            location: i as u64,
            length: 1,
        };
        unsafe {
            let _: () = objc::msg_send![
                enc,
                executeCommandsInBuffer: icb_ref
                withRange: range
            ];
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let gpu_start: f64 = unsafe { objc::msg_send![cmd, GPUStartTime] };
        let gpu_end: f64 = unsafe { objc::msg_send![cmd, GPUEndTime] };
        out.push(gpu_end - gpu_start);
    }
    Ok(out)
}

/// Per-step uniform fanout. Writes `token_id` into the global token
/// uniform and each `(offset, capacity, kv_len)` triple into the
/// per-layer offset/capacity/kv_len uniforms in a single C call,
/// then kicks off the ICB asynchronously. Cuts the per-step Python
/// loop (`for li, c in enumerate(cache): write three uint32s`) down
/// to one FFI hop, which materially shrinks per-token latency on
/// small models where Python overhead is a measurable fraction of
/// the per-step budget.
///
/// `layer_uniforms` is `[(offset_ptr, capacity_ptr, kv_len_ptr), …]`
/// in layer order. `layer_values` is `[(offset, capacity, kv_len), …]`
/// — the host supplies the values mlx-lm's `KVCache` exposes after
/// the prior step.
#[pyfunction]
fn step_decode(
    handle: usize,
    icb_id: usize,
    token_id_ptr: usize,
    token: u32,
    layer_uniforms: Vec<(usize, usize, usize)>,
    layer_values: Vec<(u32, u32, u32)>,
) -> PyResult<()> {
    if layer_uniforms.len() != layer_values.len() {
        return Err(value_err(format!(
            "layer_uniforms/values length mismatch: {} vs {}",
            layer_uniforms.len(),
            layer_values.len(),
        )));
    }
    unsafe {
        std::ptr::write(token_id_ptr as *mut u32, token);
        for ((op, cp, kp), (ov, cv, kv)) in layer_uniforms.iter().zip(layer_values.iter()) {
            std::ptr::write(*op as *mut u32, *ov);
            std::ptr::write(*cp as *mut u32, *cv);
            std::ptr::write(*kp as *mut u32, *kv);
        }
    }
    submit_icb_async(handle, icb_id)
}

/// Shared-uniform variant: when every layer references the same
/// `(offset, capacity, kv_len)` slot (true for mlx-lm's vanilla
/// `KVCache` where layers advance in lockstep), we collapse the
/// per-layer fanout to a single write per role. This is the
/// per-token-cost minimum we know how to dispatch from Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_decode_shared(
    handle: usize,
    icb_id: usize,
    token_id_ptr: usize,
    token: u32,
    offset_ptr: usize,
    offset: u32,
    capacity_ptr: usize,
    capacity: u32,
    kv_len_ptr: usize,
    kv_len: u32,
) -> PyResult<()> {
    unsafe {
        std::ptr::write(token_id_ptr as *mut u32, token);
        std::ptr::write(offset_ptr as *mut u32, offset);
        std::ptr::write(capacity_ptr as *mut u32, capacity);
        std::ptr::write(kv_len_ptr as *mut u32, kv_len);
    }
    submit_icb_async(handle, icb_id)
}

/// Shared-uniform decode with the ICB pipeline counters reset for one
/// engine step, then submitted asynchronously.
///
/// This is the oMLX `GenerationBatch` hot path: it needs the same compact
/// host fanout as `step_decode_shared`, but the generated ICB also contains
/// `fusekit_step_init`/`fusekit_history_capture`. Resetting their counters
/// per token avoids history-buffer overflow while still letting the caller
/// wait/read the argmax on the next scheduler tick.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_decode_shared_state(
    handle: usize,
    icb_id: usize,
    token_id_ptr: usize,
    token: u32,
    offset_ptr: usize,
    offset: u32,
    capacity_ptr: usize,
    capacity: u32,
    kv_len_ptr: usize,
    kv_len: u32,
    step_counter_ptr: usize,
    hist_counter_ptr: usize,
    barrier_counter_ptrs: Vec<usize>,
) -> PyResult<()> {
    unsafe {
        std::ptr::write(token_id_ptr as *mut u32, token);
        std::ptr::write(offset_ptr as *mut u32, offset);
        std::ptr::write(capacity_ptr as *mut u32, capacity);
        std::ptr::write(kv_len_ptr as *mut u32, kv_len);
        std::ptr::write(step_counter_ptr as *mut u32, offset);
        std::ptr::write(hist_counter_ptr as *mut u32, 0u32);
        for ptr in &barrier_counter_ptrs {
            std::ptr::write(*ptr as *mut u32, 0u32);
        }
    }
    submit_icb_async(handle, icb_id)
}

/// Synchronous form of `step_decode_shared_state` for direct `model(...)`
/// calls. The oMLX engine fast path can overlap async submit with scheduler
/// work, but the generic dispatcher immediately exposes logits/argmax views,
/// so it needs the old execute-and-wait semantics while still avoiding the
/// Python-side uniform/barrier fanout.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_decode_shared_state_sync(
    handle: usize,
    icb_id: usize,
    token_id_ptr: usize,
    token: u32,
    offset_ptr: usize,
    offset: u32,
    capacity_ptr: usize,
    capacity: u32,
    kv_len_ptr: usize,
    kv_len: u32,
    step_counter_ptr: usize,
    hist_counter_ptr: usize,
    barrier_counter_ptrs: Vec<usize>,
) -> PyResult<f64> {
    unsafe {
        std::ptr::write(token_id_ptr as *mut u32, token);
        std::ptr::write(offset_ptr as *mut u32, offset);
        std::ptr::write(capacity_ptr as *mut u32, capacity);
        std::ptr::write(kv_len_ptr as *mut u32, kv_len);
        std::ptr::write(step_counter_ptr as *mut u32, offset);
        std::ptr::write(hist_counter_ptr as *mut u32, 0u32);
        for ptr in &barrier_counter_ptrs {
            std::ptr::write(*ptr as *mut u32, 0u32);
        }
    }
    execute_icb(handle, icb_id)
}

/// Decode a contiguous chunk of `count` tokens entirely native-side.
///
/// Each iteration writes the input token + offset/cap/kv_len uniforms,
/// submits the ICB, waits for completion, and reads the in-ICB argmax
/// (`next_tok_ptr`) into `out_tokens[i]`. The first token written into
/// the model is the caller-supplied `start_token`; the i-th GPU step
/// uses `start_offset + i` as its layer_offset.
///
/// Removes the Python yield/FFI hops from the inner loop — measured
/// per-token CPU drops from ~1ms to <0.2ms on M5 / 0.5B Qwen.
///
/// The caller is responsible for advancing `cache.offset` and yielding
/// the resulting tokens to the streaming consumer (mlx-lm) once this
/// returns.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_chunk(
    handle: usize,
    icb_id: usize,
    token_id_ptr: usize,
    next_tok_ptr: usize,
    offset_ptr: usize,
    capacity_ptr: usize,
    kv_len_ptr: usize,
    capacity: u32,
    start_offset: u32,
    start_token: u32,
    out_tokens_ptr: usize,
    count: u32,
) -> PyResult<()> {
    // Pull the entry out once. Use a raw pointer-style stash so we
    // don't re-lock the mutex per iteration.
    let entry_ptr: *const IcbEntry;
    let queue_ptr: *const metal::CommandQueue;
    let mut last_cmd: Option<metal::CommandBuffer>;
    {
        let mut v = DISPATCHERS.lock().unwrap();
        let d = v
            .get_mut(handle)
            .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
        entry_ptr = d
            .icbs
            .get(icb_id)
            .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?
            as *const _;
        queue_ptr = &d.queue as *const _;
        last_cmd = d.last_pending_cmd.take();
    }
    // SAFETY: dispatchers + their icbs/queues live for the process'
    // lifetime (static `Mutex<Vec<Dispatcher>>` that only grows). No
    // concurrent mutation while we hold these raw pointers because
    // Python is single-threaded under the GIL during this call.
    let entry: &IcbEntry = unsafe { &*entry_ptr };
    let queue: &metal::CommandQueue = unsafe { &*queue_ptr };

    let mut token = start_token;
    let out_tokens = out_tokens_ptr as *mut u32;
    for i in 0..count {
        let o = start_offset + i;
        unsafe {
            std::ptr::write(token_id_ptr as *mut u32, token);
            std::ptr::write(offset_ptr as *mut u32, o);
            std::ptr::write(capacity_ptr as *mut u32, capacity);
            std::ptr::write(kv_len_ptr as *mut u32, o + 1);
        }

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.use_resources(
            &entry.resident_refs,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
        let range = NSRange {
            location: 0,
            length: entry.n_commands,
        };
        let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
        unsafe {
            let _: () = objc::msg_send![
                enc,
                executeCommandsInBuffer: icb_ref
                withRange: range
            ];
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // GPU finished step `i`; read in-ICB argmax → tokens[i] and
        // feed it back as `token` for step `i+1`.
        let next: u32 = unsafe { std::ptr::read(next_tok_ptr as *const u32) };
        unsafe {
            std::ptr::write(out_tokens.add(i as usize), next);
        }
        token = next;
        last_cmd = None;
    }
    // Re-stash any leftover command (we never produce one here, but
    // keep the field tidy for callers that mix sync + async submit).
    if last_cmd.is_some() {
        let mut v = DISPATCHERS.lock().unwrap();
        if let Some(d) = v.get_mut(handle) {
            d.last_pending_cmd = last_cmd;
        }
    }
    Ok(())
}

/// Correctness-first native chunk loop used by the oMLX engine fast path.
///
/// Unlike `decode_chunk_gpu_chain`, this does not rely on cross-command
/// GPU-side token feedback. Each step waits for the ICB, reads the argmax
/// token on the CPU, writes that token as the next step's seed, and stages
/// the shared offset/kv_len uniforms exactly like the Python per-step path.
/// It still removes the Python scheduler/FFI loop from the hot path.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_host_loop(
    handle: usize,
    icb_id: usize,
    step_counter_ptr: usize,
    hist_counter_ptr: usize,
    token_id_ptr: usize,
    next_tok_ptr: usize,
    offset_ptr: usize,
    capacity_ptr: usize,
    kv_len_ptr: usize,
    history_ptr: usize,
    capacity: u32,
    start_offset: u32,
    start_token: u32,
    count: u32,
    barrier_counter_ptrs: Vec<usize>,
) -> PyResult<()> {
    let entry_ptr: *const IcbEntry;
    let queue_ptr: *const metal::CommandQueue;
    {
        let mut v = DISPATCHERS.lock().unwrap();
        let d = v
            .get_mut(handle)
            .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
        entry_ptr = d
            .icbs
            .get(icb_id)
            .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?
            as *const _;
        queue_ptr = &d.queue as *const _;
        d.last_pending_cmd = None;
    }
    let entry: &IcbEntry = unsafe { &*entry_ptr };
    let queue: &metal::CommandQueue = unsafe { &*queue_ptr };

    let dbg =
        std::env::var("FUSEKIT_PROFILE").is_ok() || std::env::var("FUSEKIT_DBG_CHAIN").is_ok();
    let t_start = std::time::Instant::now();
    let mut t_uniform = 0u128;
    let mut t_encode = 0u128;
    let mut t_wait = 0u128;
    let mut t_readback = 0u128;
    let mut gpu_active = 0.0;
    let mut gpu_gap = 0.0;
    let mut prev_end = 0.0;

    let mut token = start_token;
    for i in 0..count {
        let o = start_offset + i;
        let s0 = std::time::Instant::now();
        unsafe {
            std::ptr::write(token_id_ptr as *mut u32, token);
            std::ptr::write(step_counter_ptr as *mut u32, o);
            std::ptr::write(hist_counter_ptr as *mut u32, i);
            std::ptr::write(offset_ptr as *mut u32, o);
            std::ptr::write(capacity_ptr as *mut u32, capacity);
            std::ptr::write(kv_len_ptr as *mut u32, o + 1);
            for ptr in &barrier_counter_ptrs {
                std::ptr::write(*ptr as *mut u32, 0u32);
            }
        }
        t_uniform += s0.elapsed().as_nanos();

        let s1 = std::time::Instant::now();
        let cmd = queue.new_command_buffer_with_unretained_references();
        let enc = cmd.new_compute_command_encoder();
        enc.use_resources(
            &entry.resident_refs,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
        let range = NSRange {
            location: 0,
            length: entry.n_commands,
        };
        let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
        unsafe {
            let _: () = objc::msg_send![
                enc,
                executeCommandsInBuffer: icb_ref
                withRange: range
            ];
        }
        enc.end_encoding();
        cmd.commit();
        t_encode += s1.elapsed().as_nanos();

        let s2 = std::time::Instant::now();
        cmd.wait_until_completed();
        t_wait += s2.elapsed().as_nanos();

        let s3 = std::time::Instant::now();
        let next = unsafe { std::ptr::read(next_tok_ptr as *const u32) };
        unsafe {
            std::ptr::write((history_ptr as *mut u32).add(i as usize), next);
        }
        token = next;
        t_readback += s3.elapsed().as_nanos();

        if dbg {
            let gs: f64 = unsafe { objc::msg_send![&*cmd, GPUStartTime] };
            let ge: f64 = unsafe { objc::msg_send![&*cmd, GPUEndTime] };
            gpu_active += ge - gs;
            if i > 0 {
                gpu_gap += gs - prev_end;
            }
            prev_end = ge;
        }
    }

    if dbg {
        let total_us = t_start.elapsed().as_nanos() as f64 / 1e3;
        let n = count as f64;
        eprintln!(
            "[host-loop] N={} total={:.1}us/tok | uniform={:.1} encode={:.1} wait={:.1} readback={:.1} | gpu_active={:.1} gpu_gap={:.1} barriers={}",
            count,
            total_us / n,
            t_uniform as f64 / 1e3 / n,
            t_encode as f64 / 1e3 / n,
            t_wait as f64 / 1e3 / n,
            t_readback as f64 / 1e3 / n,
            gpu_active * 1e6 / n,
            gpu_gap * 1e6 / (count.saturating_sub(1).max(1)) as f64,
            barrier_counter_ptrs.len(),
        );
    }
    Ok(())
}

/// Pre-submit `count` decode cmd buffers in a row with the per-step
/// token feedback closed on the GPU. Relies on the ICB having been
/// built with `fusekit_step_init` at its head (advances the step
/// counter, broadcasts the new offset/kv_len to the per-layer
/// uniforms) and `fusekit_history_capture` at its tail (writes the
/// fresh argmax to `history[i]`). The chain reads its input token
/// from the same buffer the argmax writes to, so cmd N+1 picks up
/// cmd N's output automatically through Metal queue serialisation.
///
/// Host preconditions per chunk:
///   step_counter = base_offset   (cache offset where step 0 writes K/V)
///   hist_counter = 0
///   next_tok     = seed token    (first step's `embed_lookup` reads it)
///
/// After this returns, `history[0..count]` holds the chunk's argmaxes
/// in order; the last element doubles as the seed for the next chunk.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_gpu_chain(
    handle: usize,
    icb_id: usize,
    step_counter_ptr: usize,
    hist_counter_ptr: usize,
    next_tok_ptr: usize,
    base_offset: u32,
    seed_token: u32,
    count: u32,
) -> PyResult<()> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let entry: &IcbEntry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;

    // Init the on-GPU counters + seed the input token buffer. Shared
    // memory is coherent, so these writes are visible to the first
    // submitted cmd buffer's GPU work.
    unsafe {
        std::ptr::write(step_counter_ptr as *mut u32, base_offset);
        std::ptr::write(hist_counter_ptr as *mut u32, 0u32);
        std::ptr::write(next_tok_ptr as *mut u32, seed_token);
    }

    // Pack N ICB executions into a SINGLE command buffer when env var
    // is set. Metal compute encoders run commands in submission order,
    // and the ICB's step_init kernel advances the offset GPU-side, so
    // step (N+1) sees step N's writes without per-cmd-buffer commits.
    // Saves `cmd.commit()` overhead (≈10-20us each on M-series).
    let bundle = std::env::var("FK_BUNDLE_CMDS")
        .ok()
        .as_deref()
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut cmds: Vec<metal::CommandBuffer> = Vec::with_capacity(count as usize);
    let t_start = std::time::Instant::now();
    if bundle {
        let cmd = d.queue.new_command_buffer_with_unretained_references();
        let range = NSRange {
            location: 0,
            length: entry.n_commands,
        };
        let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
        let split_encoders = std::env::var("FK_CHAIN_ENCODERS")
            .ok()
            .as_deref()
            .map(|v| v != "0")
            .unwrap_or(false);
        if split_encoders {
            for _ in 0..count {
                let enc = cmd.new_compute_command_encoder();
                enc.use_resources(
                    &entry.resident_refs,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                unsafe {
                    let _: () = objc::msg_send![
                        enc,
                        executeCommandsInBuffer: icb_ref
                        withRange: range
                    ];
                }
                enc.end_encoding();
            }
        } else {
            let enc = cmd.new_compute_command_encoder();
            enc.use_resources(
                &entry.resident_refs,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            for _ in 0..count {
                unsafe {
                    let _: () = objc::msg_send![
                        enc,
                        executeCommandsInBuffer: icb_ref
                        withRange: range
                    ];
                }
            }
            enc.end_encoding();
        }
        cmd.commit();
        cmds.push(cmd.to_owned());
    } else {
        for _ in 0..count {
            let cmd = d.queue.new_command_buffer_with_unretained_references();
            let enc = cmd.new_compute_command_encoder();
            enc.use_resources(
                &entry.resident_refs,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            let range = NSRange {
                location: 0,
                length: entry.n_commands,
            };
            let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
            unsafe {
                let _: () = objc::msg_send![
                    enc,
                    executeCommandsInBuffer: icb_ref
                    withRange: range
                ];
            }
            enc.end_encoding();
            cmd.commit();
            cmds.push(cmd.to_owned());
        }
    }
    let t_submit_done = t_start.elapsed();
    if let Some(cmd) = cmds.last() {
        cmd.wait_until_completed();
    }
    let t_total = t_start.elapsed();
    if std::env::var("FUSEKIT_PROFILE").is_ok() || std::env::var("FUSEKIT_DBG_CHAIN").is_ok() {
        // Per-cmd GPU active + gap to next cmd's GPU start.
        let mut gpu_active = 0.0;
        let mut gpu_gap = 0.0;
        let mut prev_end = 0.0;
        for (i, c) in cmds.iter().enumerate() {
            let gs: f64 = unsafe { objc::msg_send![&**c, GPUStartTime] };
            let ge: f64 = unsafe { objc::msg_send![&**c, GPUEndTime] };
            gpu_active += ge - gs;
            if i > 0 {
                gpu_gap += gs - prev_end;
            }
            prev_end = ge;
        }
        eprintln!(
            "[chain] count={} submit_total={:.1}ms wait_total={:.1}ms gpu_active_total={:.1}ms gpu_gap_total={:.1}ms per-tok gpu_active={:.0}us gpu_gap={:.0}us",
            count,
            t_submit_done.as_secs_f64() * 1000.0,
            (t_total - t_submit_done).as_secs_f64() * 1000.0,
            gpu_active * 1000.0,
            gpu_gap * 1000.0,
            gpu_active * 1e6 / count as f64,
            gpu_gap * 1e6 / (count.saturating_sub(1).max(1)) as f64,
        );
    }
    Ok(())
}

/// Barrier-safe variant of `decode_chunk_gpu_chain`.
///
/// Some generated kernels contain `GlobalBarrier` pieces backed by
/// host-visible atomic counters. The normal per-token path resets those
/// counters before every ICB execute. A chunked decode loop must preserve
/// that contract; otherwise the next ICB sees stale barrier counts and can
/// race through the fused layer body with incorrect intermediate data.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_gpu_chain_reset_barriers(
    handle: usize,
    icb_id: usize,
    step_counter_ptr: usize,
    hist_counter_ptr: usize,
    next_tok_ptr: usize,
    base_offset: u32,
    seed_token: u32,
    count: u32,
    barrier_counter_ptrs: Vec<usize>,
) -> PyResult<()> {
    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let entry: &IcbEntry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;

    unsafe {
        std::ptr::write(step_counter_ptr as *mut u32, base_offset);
        std::ptr::write(hist_counter_ptr as *mut u32, 0u32);
        std::ptr::write(next_tok_ptr as *mut u32, seed_token);
    }

    let dbg =
        std::env::var("FUSEKIT_PROFILE").is_ok() || std::env::var("FUSEKIT_DBG_CHAIN").is_ok();
    let t_start = std::time::Instant::now();
    let mut t_submit = std::time::Duration::ZERO;
    let mut gpu_active = 0.0;
    let mut gpu_gap = 0.0;
    let mut prev_end = 0.0;

    for i in 0..count {
        unsafe {
            for ptr in &barrier_counter_ptrs {
                std::ptr::write(*ptr as *mut u32, 0u32);
            }
        }

        let s_submit = std::time::Instant::now();
        let cmd = d.queue.new_command_buffer_with_unretained_references();
        let enc = cmd.new_compute_command_encoder();
        enc.use_resources(
            &entry.resident_refs,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
        let range = NSRange {
            location: 0,
            length: entry.n_commands,
        };
        let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
        unsafe {
            let _: () = objc::msg_send![
                enc,
                executeCommandsInBuffer: icb_ref
                withRange: range
            ];
        }
        enc.end_encoding();
        cmd.commit();
        t_submit += s_submit.elapsed();
        cmd.wait_until_completed();

        if dbg {
            let gs: f64 = unsafe { objc::msg_send![&*cmd, GPUStartTime] };
            let ge: f64 = unsafe { objc::msg_send![&*cmd, GPUEndTime] };
            gpu_active += ge - gs;
            if i > 0 {
                gpu_gap += gs - prev_end;
            }
            prev_end = ge;
        }
    }

    if dbg {
        let t_total = t_start.elapsed();
        eprintln!(
            "[chain-reset] count={} barriers={} submit_total={:.1}ms wait_total={:.1}ms gpu_active_total={:.1}ms gpu_gap_total={:.1}ms per-tok gpu_active={:.0}us gpu_gap={:.0}us",
            count,
            barrier_counter_ptrs.len(),
            t_submit.as_secs_f64() * 1000.0,
            (t_total - t_submit).as_secs_f64() * 1000.0,
            gpu_active * 1000.0,
            gpu_gap * 1000.0,
            gpu_active * 1e6 / count as f64,
            gpu_gap * 1e6 / (count.saturating_sub(1).max(1)) as f64,
        );
    }
    Ok(())
}

/// Run `count` consecutive decode steps fully native-side. Each step
/// stages the offset/kv_len/token_id uniforms (shared across layers
/// under the assumption mlx-lm's vanilla `KVCache` advances every
/// layer in lockstep), submits the ICB, waits for completion, and
/// reads the in-ICB argmax into `out_history[i]` for the streaming
/// consumer. Token feedback within the chunk happens in this loop:
/// step i+1's input token is the argmax we just read out of step i.
///
/// We tried a cmd-buffer-spanning pipelined variant that pre-submits
/// all `count` cmd buffers and lets Metal's queue keep the GPU fed
/// without the inter-cmd ~360us launch-latency gap. That requires
/// closing the token feedback loop on the GPU (the next step's
/// embed_lookup reading from the argmax-writes buffer), but the
/// `executeCommandsInBuffer` → blit encoder hazard didn't behave
/// reliably for our ICB (untracked compared to the shared-mode
/// uniforms) — the blit read stale `next_tok` values on the M5. So
/// we keep the simple submit→wait→read loop and accept the gap.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_shared_native(
    handle: usize,
    icb_id: usize,
    token_id_uniform_ptr: usize,
    next_tok_uniform_ptr: usize,
    offset_uniform_ptr: usize,
    kv_len_uniform_ptr: usize,
    offset_array_ptr: usize,
    kv_len_array_ptr: usize,
    out_history_ptr: usize,
    seed_token_ptr: usize,
    count: u32,
) -> PyResult<()> {
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;

    let entry: &IcbEntry = d
        .icbs
        .get(icb_id)
        .ok_or_else(|| value_err(format!("invalid icb id: {icb_id}")))?;

    let dbg = std::env::var("FUSEKIT_PROFILE").is_ok();
    let t_start = std::time::Instant::now();
    let mut t_uniform = 0u128;
    let mut t_encode = 0u128;
    let mut t_wait = 0u128;
    let mut t_readback = 0u128;
    let mut gpu_active = 0f64;
    let mut gpu_gap = 0f64;
    let mut prev_end = 0f64;
    for i in 0..count {
        let s0 = std::time::Instant::now();
        unsafe {
            let offset_val: u32 = std::ptr::read((offset_array_ptr as *const u32).add(i as usize));
            let kv_len_val: u32 = std::ptr::read((kv_len_array_ptr as *const u32).add(i as usize));
            std::ptr::write(offset_uniform_ptr as *mut u32, offset_val);
            std::ptr::write(kv_len_uniform_ptr as *mut u32, kv_len_val);
            let token_val: u32 = if i == 0 {
                std::ptr::read(seed_token_ptr as *const u32)
            } else {
                std::ptr::read(next_tok_uniform_ptr as *const u32)
            };
            std::ptr::write(token_id_uniform_ptr as *mut u32, token_val);
        }
        t_uniform += s0.elapsed().as_nanos();

        let s1 = std::time::Instant::now();
        let cmd = d.queue.new_command_buffer_with_unretained_references();
        let enc = cmd.new_compute_command_encoder();
        enc.use_resources(
            &entry.resident_refs,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
        let range = NSRange {
            location: 0,
            length: entry.n_commands,
        };
        let icb_ref: &metal::IndirectCommandBufferRef = &entry.icb;
        unsafe {
            let _: () = objc::msg_send![
                enc,
                executeCommandsInBuffer: icb_ref
                withRange: range
            ];
        }
        enc.end_encoding();
        cmd.commit();
        t_encode += s1.elapsed().as_nanos();

        let s2 = std::time::Instant::now();
        cmd.wait_until_completed();
        t_wait += s2.elapsed().as_nanos();

        let s3 = std::time::Instant::now();
        unsafe {
            let next: u32 = std::ptr::read(next_tok_uniform_ptr as *const u32);
            std::ptr::write((out_history_ptr as *mut u32).add(i as usize), next);
        }
        t_readback += s3.elapsed().as_nanos();

        if dbg {
            let gs: f64 = unsafe { objc::msg_send![&*cmd, GPUStartTime] };
            let ge: f64 = unsafe { objc::msg_send![&*cmd, GPUEndTime] };
            gpu_active += ge - gs;
            if i > 0 {
                gpu_gap += gs - prev_end;
            }
            prev_end = ge;
        }
    }
    if dbg {
        let total_us = t_start.elapsed().as_nanos() as f64 / 1e3;
        let n = count as f64;
        eprintln!(
            "[prof] N={} total={:.1}us/tok | uniform={:.1} encode={:.1} wait={:.1} readback={:.1} | gpu_active={:.1} gpu_gap={:.1}",
            count,
            total_us / n,
            t_uniform as f64 / 1e3 / n,
            t_encode as f64 / 1e3 / n,
            t_wait as f64 / 1e3 / n,
            t_readback as f64 / 1e3 / n,
            gpu_active * 1e6 / n,
            gpu_gap * 1e6 / (count.saturating_sub(1).max(1)) as f64,
        );
    }
    Ok(())
}

/// Allocate a host-visible Metal buffer and return its data pointer.
/// Used for small, stable-address uniforms (offset, kv_len, etc.) so
/// the Python side doesn't have to round-trip through `mx.array +
/// mx.eval + array_data_ptr` on every layer.
#[pyfunction]
fn alloc_uniform(handle: usize, nbytes: usize) -> PyResult<usize> {
    use metal::MTLResourceOptions;
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    let buf = d
        .device
        .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as usize;
    // Keep the MTLBuffer in the dispatcher's registry so the
    // pipelined-decode path can recover the buffer ref (needed for
    // blit `copy_from_buffer`) from the raw data pointer Python holds.
    let idx = d.uniform_buffers.len();
    d.uniform_buffers.push(buf);
    d.uniform_ptr_to_idx.insert(ptr, idx);
    Ok(ptr)
}

/// Register an existing host-visible memory range as one Metal buffer.
/// Later bindings whose pointers fall inside this range use the same
/// buffer plus an offset, so Python can build scratch arenas and pass
/// subview addresses without creating unaligned no-copy buffers.
#[pyfunction]
fn register_host_buffer(handle: usize, ptr: usize, nbytes: usize) -> PyResult<()> {
    if ptr == 0 || nbytes == 0 {
        return Err(value_err(
            "cannot register null/empty host buffer".to_string(),
        ));
    }
    let mut v = DISPATCHERS.lock().unwrap();
    let d = v
        .get_mut(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;
    if d.registered_host_buffers
        .iter()
        .any(|b| b.base == ptr && b.nbytes >= nbytes)
    {
        return Ok(());
    }
    let buffer = d.device.new_buffer_with_bytes_no_copy(
        ptr as *const c_void,
        nbytes as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    d.registered_host_buffers.push(RegisteredHostBuffer {
        base: ptr,
        nbytes,
        buffer,
    });
    Ok(())
}

/// Batch-dispatch: encode N kernels into ONE command buffer, commit
/// once, wait once. This is the cheapest path to most of the
/// megakernel's "no CPU↔GPU round-trip per task" benefit — same
/// number of GPU shaders, but only one driver handoff per batch.
///
/// `dispatches` is a list of `(kernel_name, buffers, grid, threadgroup)`
/// tuples in execution order. Within one command buffer Metal
/// guarantees in-order execution between dispatches on the same
/// encoder, so write-then-read across dispatches sees up-to-date data
/// without an explicit fence.
#[pyfunction]
fn dispatch_batch(
    handle: usize,
    dispatches: Vec<(
        String,
        Vec<(usize, usize)>,
        (u64, u64, u64),
        (u64, u64, u64),
    )>,
) -> PyResult<f64> {
    use metal::{MTLResourceOptions, MTLSize};

    let v = DISPATCHERS.lock().unwrap();
    let d = v
        .get(handle)
        .ok_or_else(|| value_err(format!("invalid handle: {handle}")))?;

    // Build a stable ptr → MTLBuffer cache for THIS batch. Reusing
    // the same MTLBuffer object across dispatches that touch the
    // same underlying pointer is what lets Metal's encoder
    // auto-fence dependencies — different MTLBuffer wrappers around
    // the same address are treated as independent resources and
    // would skip the barrier, letting a later dispatch read stale
    // data from a not-yet-flushed prior write.
    let mut buf_cache: HashMap<usize, metal::Buffer> = HashMap::new();
    let mut get_buf = |ptr: usize, nbytes: usize, dev: &Device| -> (metal::Buffer, u64) {
        let registered = registered_host_buffer_for(d, ptr, nbytes);
        let (offset, cache_key) = registered
            .as_ref()
            .map(|(_, offset, base)| (*offset, *base))
            .unwrap_or((0, ptr));
        let buf = if let Some(buf) = buf_cache.get(&cache_key) {
            buf.clone()
        } else {
            let buf = registered.map(|(buf, _, _)| buf).unwrap_or_else(|| {
                dev.new_buffer_with_bytes_no_copy(
                    ptr as *const c_void,
                    nbytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            });
            buf_cache.insert(cache_key, buf.clone());
            buf
        };
        (buf, offset)
    };

    // One command buffer, one compute encoder for the whole batch.
    // Between dispatches, insert `memoryBarrierWithResources:` so
    // writes from dispatch N are visible to dispatch N+1. The barrier
    // scope is just the next dispatch's buffer set (bounded ~10
    // resources) — passing the cumulative prior-buf list grew O(N²)
    // ObjC-message traffic and dominated encoding overhead.
    let cmd = d.queue.new_command_buffer_with_unretained_references();
    let enc = cmd.new_compute_command_encoder();
    let mut prev_bufs: Vec<metal::Buffer> = Vec::new();
    for (idx, (kernel_name, buffers, grid, threadgroup)) in dispatches.iter().enumerate() {
        let pso = d
            .pipelines
            .get(kernel_name)
            .ok_or_else(|| value_err(format!("kernel not in metallib: {kernel_name}")))?;

        enc.set_compute_pipeline_state(pso);
        let mtl_bufs: Vec<(metal::Buffer, u64)> = buffers
            .iter()
            .map(|&(ptr, nbytes)| get_buf(ptr, nbytes, &d.device))
            .collect();
        for (i, (b, offset)) in mtl_bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), *offset);
        }
        if idx > 0 && !prev_bufs.is_empty() {
            let refs: Vec<&metal::ResourceRef> = mtl_bufs
                .iter()
                .map(|(b, _)| &**b as &metal::ResourceRef)
                .collect();
            enc.memory_barrier_with_resources(&refs);
        }
        enc.dispatch_threads(
            MTLSize::new(grid.0, grid.1, grid.2),
            MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
        );
        prev_bufs = mtl_bufs.into_iter().map(|(b, _)| b).collect();
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    // Return GPU-only elapsed time (seconds). metal-rs 0.29 doesn't
    // wrap MTLCommandBuffer's gpuStartTime/gpuEndTime, so we go
    // through ObjC messages directly.
    let gpu_start: f64 = unsafe { objc::msg_send![cmd, GPUStartTime] };
    let gpu_end: f64 = unsafe { objc::msg_send![cmd, GPUEndTime] };
    Ok(gpu_end - gpu_start)
}

/// Dispatch an arbitrary test-metallib kernel against a set of
/// externally-allocated buffers. Used to validate the multi-buffer
/// bridge — same machinery the real per-model dispatch will use,
/// minus the metallib selection (test vs `<model>/fusekit/`).
///
/// `buffers` is a list of `(ptr, nbytes)` pairs that match the
/// kernel's argument order. Each ptr is wrapped via
/// `newBufferWithBytesNoCopy:` so unified-memory mx.arrays remain
/// zero-copy.
#[pyfunction]
fn dispatch_test_kernel(
    kernel_name: String,
    buffers: Vec<(usize, usize)>,
    grid: (u64, u64, u64),
    threadgroup: (u64, u64, u64),
) -> PyResult<()> {
    use metal::{MTLResourceOptions, MTLSize};

    let device = metal::Device::system_default()
        .ok_or_else(|| value_err("no Metal device available".to_string()))?;
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_data(TEST_METALLIB)
        .map_err(|e| value_err(format!("load test metallib: {e}")))?;
    let function = lib
        .get_function(&kernel_name, None)
        .map_err(|e| value_err(format!("get {kernel_name}: {e}")))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| value_err(format!("build pipeline: {e}")))?;

    // Wrap each external pointer as a zero-copy MTLBuffer. Apple
    // Silicon's unified memory makes this a true alias of the
    // mx.array's underlying storage.
    let mtl_bufs: Vec<metal::Buffer> = buffers
        .iter()
        .map(|&(ptr, nbytes)| {
            device.new_buffer_with_bytes_no_copy(
                ptr as *const std::ffi::c_void,
                nbytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        })
        .collect();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    for (i, b) in mtl_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.dispatch_threads(
        MTLSize::new(grid.0, grid.1, grid.2),
        MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[repr(C)]
struct DlDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DlDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DlTensor {
    data: *mut c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DlManagedTensor {
    dl_tensor: DlTensor,
    manager_ctx: *mut c_void,
    deleter: *mut c_void,
}

/// Return raw backing-store addresses for a batch of MLX arrays.
///
/// The Python fallback does this via ctypes one array at a time. During cold
/// ICB construction that is thousands of Python/C boundary crossings, so keep
/// the DLPack capsule unwrap on the native side and let Python batch `mx.eval`.
#[pyfunction]
fn array_data_ptrs(py: Python<'_>, arrays: Vec<Py<PyAny>>) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(arrays.len());
    for arr in arrays {
        let cap = arr.bind(py).call_method0("__dlpack__")?;
        let name = unsafe { pyo3::ffi::PyCapsule_GetName(cap.as_ptr()) };
        if name.is_null() {
            return Err(PyErr::fetch(py));
        }
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(cap.as_ptr(), name) };
        if raw.is_null() {
            return Err(PyErr::fetch(py));
        }
        let mt = unsafe { &*(raw as *const DlManagedTensor) };
        out.push(mt.dl_tensor.data as usize);
    }
    Ok(out)
}

#[pymodule]
fn _fusekit_runtime_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_dispatcher, m)?)?;
    m.add_function(wrap_pyfunction!(dispatcher_info, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_names, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_forward, m)?)?;
    m.add_function(wrap_pyfunction!(smoke_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(scale_inplace_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_test_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_batch, m)?)?;
    m.add_function(wrap_pyfunction!(alloc_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(register_host_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(build_icb, m)?)?;
    m.add_function(wrap_pyfunction!(update_icb_bindings, m)?)?;
    m.add_function(wrap_pyfunction!(step_decode, m)?)?;
    m.add_function(wrap_pyfunction!(step_decode_shared, m)?)?;
    m.add_function(wrap_pyfunction!(decode_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(decode_chunk_host_loop, m)?)?;
    m.add_function(wrap_pyfunction!(decode_chunk_shared_native, m)?)?;
    m.add_function(wrap_pyfunction!(decode_chunk_gpu_chain, m)?)?;
    m.add_function(wrap_pyfunction!(decode_chunk_gpu_chain_reset_barriers, m)?)?;
    m.add_function(wrap_pyfunction!(execute_icb, m)?)?;
    m.add_function(wrap_pyfunction!(execute_icb_range, m)?)?;
    m.add_function(wrap_pyfunction!(profile_icb_commands, m)?)?;
    m.add_function(wrap_pyfunction!(submit_icb_async, m)?)?;
    m.add_function(wrap_pyfunction!(step_decode_shared_state, m)?)?;
    m.add_function(wrap_pyfunction!(step_decode_shared_state_sync, m)?)?;
    m.add_function(wrap_pyfunction!(wait_pending_icb, m)?)?;
    m.add_function(wrap_pyfunction!(array_data_ptrs, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------

fn io_err(e: std::io::Error) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
}
fn value_err(msg: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
}
fn file_not_found(msg: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(msg)
}
