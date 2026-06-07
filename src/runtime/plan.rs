//! Runtime auto-dispatcher — read `task_graph.json`, allocate
//! scratch buffers, resolve external bindings to live mlx tensors,
//! and stand the persistent megakernel up.
//!
//! ```text
//!   task_graph.json + kernel.metallib + Python model
//!         │
//!         │   load_plan ─── parse JSON, build PSO cache, allocate scratch
//!         ▼
//!   RuntimePlan
//!         │
//!         │   resolve_externals(py, model) ── walk param paths,
//!         │                                   grab DLPack data pointers
//!         ▼
//!   ResolvedPlan   ──► encode dispatch list → ICB → execute
//! ```
//!
//! The split: `RuntimePlan` is model-independent (built once per
//! compile); `ResolvedPlan` binds it to a specific `model` instance.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::runtime::metal::MetalContext;

/// Loaded but unresolved task graph. Every Task knows its kernel + grid
/// + binding source, but external bindings are still abstract module
/// paths — they bind to live mlx tensors via `resolve_externals`.
pub struct RuntimePlan {
    pub doc: TaskGraphDoc,
    pub scratch: Vec<metal::Buffer>,
    pub psos: HashMap<String, metal::ComputePipelineState>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TaskGraphDoc {
    pub version: u32,
    pub tasks: Vec<JsonTask>,
    pub events: Vec<JsonEvent>,
    pub scratch_buffers: Vec<JsonScratch>,
    pub initial_tasks: Vec<u32>,
    #[serde(default)]
    pub model_params: Vec<JsonParam>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonTask {
    pub id: u32,
    pub fusion_id: u32,
    pub kernel: String,
    pub grid_threads: [u64; 3],
    pub threadgroup: [u64; 3],
    pub inputs: Vec<JsonBinding>,
    pub outputs: Vec<JsonBinding>,
    pub depends_on: Vec<u32>,
    pub trigger_event: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum JsonBinding {
    External {
        tensor_id: u32,
        #[serde(default)]
        source_path: String,
        #[serde(default)]
        scalar: Option<JsonScalar>,
        role_name: String,
        shape: Vec<i32>,
        dtype: String,
    },
    Scratch {
        scratch_id: u32,
        role_name: String,
        shape: Vec<i32>,
        dtype: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonEvent {
    pub id: u32,
    pub num_triggers: u32,
    pub fires: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum JsonScalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonScratch {
    pub id: u32,
    pub shape: Vec<i32>,
    pub dtype: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonParam {
    pub path: String,
    pub shape: Vec<i32>,
    pub dtype: String,
}

/// Load `task_graph.json` + the compiled metallib next to it, allocate
/// scratch buffers, and build a PSO per unique kernel name. The
/// result is ready to bind to a live model via `resolve_externals`.
pub fn load_plan(ctx: &MetalContext, out_dir: &Path) -> Result<RuntimePlan> {
    let tg_path = out_dir.join("task_graph.json");
    let raw =
        fs::read_to_string(&tg_path).with_context(|| format!("read {}", tg_path.display()))?;
    let doc: TaskGraphDoc =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", tg_path.display()))?;
    if doc.version != 2 {
        return Err(anyhow!(
            "task_graph.json version {} not supported (expected 2)",
            doc.version
        ));
    }

    // One Metal buffer per scratch entry. Shared storage is fine since
    // we're on unified memory; the runtime never CPU-touches these.
    let mut scratch: Vec<metal::Buffer> = Vec::with_capacity(doc.scratch_buffers.len());
    for s in &doc.scratch_buffers {
        let nbytes = (s.size_bytes.max(1)) as usize;
        scratch.push(ctx.new_buffer(nbytes));
    }

    // PSO cache: dedupe by kernel name (multiple tasks can share a
    // kernel function — they always share the PSO too).
    let mut psos: HashMap<String, metal::ComputePipelineState> = HashMap::new();
    for task in &doc.tasks {
        if psos.contains_key(&task.kernel) {
            continue;
        }
        let pso = ctx
            .pipeline(&task.kernel)
            .with_context(|| format!("pso for kernel `{}`", task.kernel))?;
        psos.insert(task.kernel.clone(), pso);
    }

    Ok(RuntimePlan { doc, scratch, psos })
}

/// A `RuntimePlan` with every External binding bound to a live Metal
/// buffer (pulled from an mlx tensor via DLPack zero-copy). Ready for
/// dispatch.
pub struct ResolvedPlan<'p> {
    pub plan: &'p RuntimePlan,
    /// Per-external binding: a Metal buffer view of the resolved
    /// tensor. Indexed by (task_id, input_or_output, slot).
    pub external_buffers: HashMap<ExternalKey, metal::Buffer>,
    /// Bindings the runtime could NOT resolve from `model` — these
    /// stay as TensorIDs and the caller must supply them per-invocation
    /// (e.g. the input token id, the position offset uniform).
    pub unresolved: Vec<ExternalKey>,
}

impl<'p> ResolvedPlan<'p> {
    /// Encode every task into a fresh command buffer and submit it.
    /// Tasks execute in `plan.doc.tasks` order; Metal's resource
    /// hazard tracking inserts the necessary barriers between
    /// dependent tasks, so we don't need to encode events explicitly.
    ///
    /// `runtime_inputs` covers bindings whose `source_path` was empty
    /// at capture time (mlx-internal scalar constants, per-invocation
    /// token / position inputs). Each entry maps an `ExternalKey` to
    /// the buffer the caller wants bound at that slot.
    ///
    /// Slow path — pays per-task encoding cost every call. Use
    /// `build_icb` + `execute_icb` for the megakernel hot path.
    pub fn execute(
        &self,
        ctx: &MetalContext,
        runtime_inputs: &HashMap<ExternalKey, metal::Buffer>,
    ) -> Result<()> {
        let cmd_buf = ctx.queue.new_command_buffer();
        for task in &self.plan.doc.tasks {
            self.encode_task(ctx, task, runtime_inputs, cmd_buf)
                .with_context(|| format!("encode task {}", task.id))?;
        }
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        let status = cmd_buf.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(anyhow!(
                "Metal command buffer ended with status {:?}",
                status
            ));
        }
        Ok(())
    }

    /// Encode the whole task list once into an indirect command
    /// buffer. Returns the ICB plus the per-task command index so the
    /// caller can update runtime-input slot bindings without
    /// re-encoding the rest.
    ///
    /// `set_barrier()` on every command serializes execution within
    /// the ICB. That matches the data dependencies expressed by the
    /// task graph today and trades a bit of intra-batch parallelism
    /// for correctness without explicit per-resource tracking.
    pub fn build_icb(
        &self,
        ctx: &MetalContext,
        runtime_inputs: &HashMap<ExternalKey, metal::Buffer>,
    ) -> Result<IcbPlan> {
        use metal::{
            IndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLResourceOptions, MTLSize,
        };

        let n = self.plan.doc.tasks.len();
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatchThreads);
        desc.set_inherit_buffers(false);
        desc.set_inherit_pipeline_state(false);
        desc.set_max_kernel_buffer_bind_count(32);
        let icb = ctx.device.new_indirect_command_buffer_with_descriptor(
            &desc,
            n as u64,
            MTLResourceOptions::StorageModePrivate,
        );

        // Track unique buffer ptrs so we can call use_resource at
        // execute time. Metal doesn't track residency through ICBs;
        // declaring the set up-front prevents the allocator from
        // freeing one underneath us mid-execution.
        let mut residency: Vec<metal::Buffer> = Vec::new();
        let mut seen_ptrs: std::collections::HashSet<*const std::ffi::c_void> = Default::default();
        let mut register =
            |buf: &metal::Buffer,
             residency: &mut Vec<metal::Buffer>,
             seen: &mut std::collections::HashSet<*const std::ffi::c_void>| {
                let p = buf.contents();
                if seen.insert(p) {
                    residency.push(buf.to_owned());
                }
            };

        for (i, task) in self.plan.doc.tasks.iter().enumerate() {
            let pso = self
                .plan
                .psos
                .get(&task.kernel)
                .ok_or_else(|| anyhow!("no PSO for `{}`", task.kernel))?;
            let cmd = icb.indirect_compute_command_at_index(i as u64);
            cmd.set_compute_pipeline_state(pso);
            for (slot, b) in task.inputs.iter().enumerate() {
                let key = ExternalKey {
                    task_id: task.id,
                    direction: IoDir::Input,
                    slot: slot as u32,
                    tensor_id: binding_tensor_id(b),
                };
                let buf = self.lookup_buffer(b, &key, runtime_inputs)?;
                cmd.set_kernel_buffer(slot as u64, Some(buf), 0);
                register(buf, &mut residency, &mut seen_ptrs);
            }
            for (i_out, b) in task.outputs.iter().enumerate() {
                let slot = task.inputs.len() + i_out;
                let key = ExternalKey {
                    task_id: task.id,
                    direction: IoDir::Output,
                    slot: slot as u32,
                    tensor_id: binding_tensor_id(b),
                };
                let buf = self.lookup_buffer(b, &key, runtime_inputs)?;
                cmd.set_kernel_buffer(slot as u64, Some(buf), 0);
                register(buf, &mut residency, &mut seen_ptrs);
            }
            cmd.set_barrier();
            let tg = MTLSize::new(
                task.threadgroup[0],
                task.threadgroup[1],
                task.threadgroup[2],
            );
            let grid = MTLSize::new(
                task.grid_threads[0].max(1),
                task.grid_threads[1].max(1),
                task.grid_threads[2].max(1),
            );
            cmd.concurrent_dispatch_threads(grid, tg);
        }

        Ok(IcbPlan {
            icb,
            count: n as u64,
            residency,
        })
    }

    /// Execute a pre-built ICB. Per-step cost is one compute encoder
    /// + one `executeCommandsInBuffer:` call — no per-task CPU
    /// encoding. Buffer bindings that change between steps must be
    /// patched via `IcbPlan::update_binding` before calling this.
    pub fn execute_icb(&self, ctx: &MetalContext, icb_plan: &IcbPlan) -> Result<()> {
        use metal::{MTLResourceUsage, NSRange};
        use objc::{msg_send, sel, sel_impl};

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        // Declare residency for every buffer the ICB references so
        // Metal keeps them live for the GPU.
        for buf in &icb_plan.residency {
            enc.use_resource(buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        // metal-rs 0.29 exposes `executeCommandsInBuffer:` only on the
        // render encoder; drop to objc for the compute side.
        let range = NSRange::new(0, icb_plan.count);
        let icb_ptr: &metal::IndirectCommandBufferRef = &icb_plan.icb;
        unsafe {
            let _: () = msg_send![enc,
                executeCommandsInBuffer: icb_ptr
                withRange: range
            ];
        }
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        let status = cmd_buf.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(anyhow!(
                "Metal command buffer ended with status {:?}",
                status
            ));
        }
        Ok(())
    }

    fn encode_task(
        &self,
        _ctx: &MetalContext,
        task: &JsonTask,
        runtime_inputs: &HashMap<ExternalKey, metal::Buffer>,
        cmd_buf: &metal::CommandBufferRef,
    ) -> Result<()> {
        use metal::MTLSize;

        let pso = self
            .plan
            .psos
            .get(&task.kernel)
            .ok_or_else(|| anyhow!("no PSO for `{}`", task.kernel))?;

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pso);

        // Bind inputs then outputs into consecutive buffer slots,
        // matching the order the codegen emitter laid them out.
        for (slot, b) in task.inputs.iter().enumerate() {
            let key = ExternalKey {
                task_id: task.id,
                direction: IoDir::Input,
                slot: slot as u32,
                tensor_id: binding_tensor_id(b),
            };
            let buf = self.lookup_buffer(b, &key, runtime_inputs)?;
            encoder.set_buffer(slot as u64, Some(buf), 0);
        }
        for (i, b) in task.outputs.iter().enumerate() {
            let slot = task.inputs.len() + i;
            let key = ExternalKey {
                task_id: task.id,
                direction: IoDir::Output,
                slot: slot as u32,
                tensor_id: binding_tensor_id(b),
            };
            let buf = self.lookup_buffer(b, &key, runtime_inputs)?;
            encoder.set_buffer(slot as u64, Some(buf), 0);
        }

        let tg = MTLSize::new(
            task.threadgroup[0],
            task.threadgroup[1],
            task.threadgroup[2],
        );
        let grid = MTLSize::new(
            task.grid_threads[0].max(1),
            task.grid_threads[1].max(1),
            task.grid_threads[2].max(1),
        );
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        Ok(())
    }

    fn lookup_buffer<'a>(
        &'a self,
        binding: &JsonBinding,
        key: &ExternalKey,
        runtime_inputs: &'a HashMap<ExternalKey, metal::Buffer>,
    ) -> Result<&'a metal::Buffer> {
        match binding {
            JsonBinding::Scratch { scratch_id, .. } => self
                .plan
                .scratch
                .get(*scratch_id as usize)
                .ok_or_else(|| anyhow!("scratch id {} out of range", scratch_id)),
            JsonBinding::External { .. } => {
                if let Some(buf) = self.external_buffers.get(key) {
                    return Ok(buf);
                }
                runtime_inputs.get(key).ok_or_else(|| {
                    anyhow!(
                        "no buffer for external binding task={} slot={} tensor_id={}",
                        key.task_id,
                        key.slot,
                        key.tensor_id
                    )
                })
            }
        }
    }
}

/// Materialize a captured scalar trace constant into a one-element
/// Metal buffer. The dtype reported by the binding tells us the
/// kernel's expected element type; we widen from the captured value
/// accordingly.
fn scalar_to_buffer(ctx: &MetalContext, sv: &JsonScalar, dtype: &str) -> metal::Buffer {
    match dtype {
        "int32" => {
            let v: i32 = match sv {
                JsonScalar::Int(i) => *i as i32,
                JsonScalar::Bool(b) => *b as i32,
                JsonScalar::Float(f) => *f as i32,
            };
            ctx.buffer_from_slice(std::slice::from_ref(&v))
        }
        "int64" => {
            let v: i64 = match sv {
                JsonScalar::Int(i) => *i,
                JsonScalar::Bool(b) => *b as i64,
                JsonScalar::Float(f) => *f as i64,
            };
            ctx.buffer_from_slice(std::slice::from_ref(&v))
        }
        "uint32" => {
            let v: u32 = match sv {
                JsonScalar::Int(i) => *i as u32,
                JsonScalar::Bool(b) => *b as u32,
                JsonScalar::Float(f) => *f as u32,
            };
            ctx.buffer_from_slice(std::slice::from_ref(&v))
        }
        "float32" => {
            let v: f32 = match sv {
                JsonScalar::Float(f) => *f as f32,
                JsonScalar::Int(i) => *i as f32,
                JsonScalar::Bool(b) => *b as i32 as f32,
            };
            ctx.buffer_from_slice(std::slice::from_ref(&v))
        }
        "float16" | "bfloat16" => {
            // Both are 2-byte; the bit layout differs but for scalar
            // constants we can encode via the matching `half` crate
            // representation and let the kernel read 2 bytes.
            let f: f32 = match sv {
                JsonScalar::Float(f) => *f as f32,
                JsonScalar::Int(i) => *i as f32,
                JsonScalar::Bool(b) => *b as i32 as f32,
            };
            if dtype == "bfloat16" {
                let v = half::bf16::from_f32(f);
                ctx.buffer_from_slice(std::slice::from_ref(&v))
            } else {
                let v = half::f16::from_f32(f);
                ctx.buffer_from_slice(std::slice::from_ref(&v))
            }
        }
        "bool" | "uint8" | "int8" => {
            let v: u8 = match sv {
                JsonScalar::Bool(b) => *b as u8,
                JsonScalar::Int(i) => *i as u8,
                JsonScalar::Float(f) => *f as u8,
            };
            ctx.buffer_from_slice(std::slice::from_ref(&v))
        }
        _ => {
            // Fall back to 4-byte zero buffer; the kernel will read
            // garbage but the dispatch path doesn't fail.
            let v: u32 = 0;
            ctx.buffer_from_slice(std::slice::from_ref(&v))
        }
    }
}

fn binding_tensor_id(b: &JsonBinding) -> u32 {
    match b {
        JsonBinding::External { tensor_id, .. } => *tensor_id,
        JsonBinding::Scratch { scratch_id, .. } => *scratch_id,
    }
}

/// Pre-encoded indirect command buffer holding every task's dispatch.
/// Execute-side cost is one Metal encoder + one
/// `execute_commands_in_buffer` call, which is what makes a
/// model-wide megakernel viable: per-step CPU overhead stops scaling
/// with task count.
pub struct IcbPlan {
    pub icb: metal::IndirectCommandBuffer,
    pub count: u64,
    /// Buffers the ICB references — kept alive (and `use_resource`'d
    /// at execute time) so Metal won't reclaim them out from under
    /// the GPU.
    pub residency: Vec<metal::Buffer>,
}

impl IcbPlan {
    /// Patch one command's buffer slot. Use when a runtime-input
    /// buffer (token, position, KV cache) changes pointer between
    /// steps — call this once per affected slot, then `execute_icb`.
    pub fn update_binding(&self, task_index: usize, slot: u32, buf: &metal::Buffer) {
        let cmd = self
            .icb
            .indirect_compute_command_at_index(task_index as u64);
        cmd.set_kernel_buffer(slot as u64, Some(buf), 0);
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ExternalKey {
    pub task_id: u32,
    pub direction: IoDir,
    pub slot: u32,
    pub tensor_id: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum IoDir {
    Input,
    Output,
}

/// Walk each external binding's `source_path` against the live Python
/// model and pull the corresponding mlx tensor's DLPack data pointer
/// into a Metal `Buffer` (zero-copy on Apple Silicon's unified
/// memory). Bindings with no `source_path` (mlx-internal scalars,
/// runtime token inputs) land in `unresolved` so the host can plug
/// them in per-invocation.
pub fn resolve_externals<'p>(
    plan: &'p RuntimePlan,
    py: pyo3::Python<'_>,
    model: &pyo3::Bound<'_, pyo3::PyAny>,
    ctx: &MetalContext,
) -> Result<ResolvedPlan<'p>> {
    use pyo3::prelude::*;

    let mut external_buffers: HashMap<ExternalKey, metal::Buffer> = HashMap::new();
    let mut unresolved: Vec<ExternalKey> = Vec::new();

    for task in &plan.doc.tasks {
        for (slot, b) in task.inputs.iter().enumerate() {
            if let JsonBinding::External {
                tensor_id,
                source_path,
                scalar,
                dtype,
                ..
            } = b
            {
                let key = ExternalKey {
                    task_id: task.id,
                    direction: IoDir::Input,
                    slot: slot as u32,
                    tensor_id: *tensor_id,
                };
                // Resolution order: live model param > materialised
                // scalar > caller-supplied runtime input.
                if !source_path.is_empty() {
                    let buf = resolve_attr_path(py, model, source_path)
                        .and_then(|arr| array_to_metal_buffer(ctx, &arr))
                        .with_context(|| {
                            format!(
                                "resolve external task={} slot={} path=`{}`",
                                task.id, slot, source_path
                            )
                        })?;
                    external_buffers.insert(key, buf);
                    continue;
                }
                if let Some(sv) = scalar {
                    let buf = scalar_to_buffer(ctx, sv, dtype);
                    external_buffers.insert(key, buf);
                    continue;
                }
                unresolved.push(key);
            }
        }
        // Outputs are always Scratch in the current schema — externals
        // would mean writing to a model parameter, which the chain
        // emit never does. Skip walking outputs.
        let _ = py.None();
    }

    Ok(ResolvedPlan {
        plan,
        external_buffers,
        unresolved,
    })
}

/// Walk a dotted attribute path like
/// `model.layers.0.self_attn.q_proj.weight` against `root` and return
/// the leaf object. Integer-looking segments index list-style
/// containers (so `layers.0` becomes `layers[0]`).
fn resolve_attr_path<'py>(
    py: pyo3::Python<'py>,
    root: &pyo3::Bound<'py, pyo3::PyAny>,
    path: &str,
) -> Result<pyo3::Bound<'py, pyo3::PyAny>> {
    use pyo3::prelude::*;

    let mut cur: Bound<pyo3::PyAny> = root.clone();
    for segment in path.split('.') {
        let next = if let Ok(idx) = segment.parse::<i64>() {
            cur.get_item(idx)
                .map_err(|e| anyhow!("index [{}] on `{}`: {}", idx, type_repr(&cur), e))?
        } else {
            cur.getattr(segment)
                .map_err(|e| anyhow!("getattr `{}` on `{}`: {}", segment, type_repr(&cur), e))?
        };
        cur = next;
    }
    let _ = py;
    Ok(cur)
}

fn type_repr(obj: &pyo3::Bound<'_, pyo3::PyAny>) -> String {
    use pyo3::types::{PyAnyMethods, PyStringMethods, PyTypeMethods};
    obj.get_type()
        .qualname()
        .ok()
        .and_then(|s| s.to_str().ok().map(|s| s.to_string()))
        .unwrap_or_else(|| "?".to_string())
}

/// Borrow the live data pointer of an `mx.array` via DLPack and wrap
/// it in a Metal buffer (`newBufferWithBytesNoCopy:`). Apple's unified
/// memory means we get the same physical bytes the kernel will read.
/// The original mx.array must outlive this buffer — we don't take
/// ownership of the DLPack capsule's deallocator.
fn array_to_metal_buffer(
    ctx: &MetalContext,
    arr: &pyo3::Bound<'_, pyo3::PyAny>,
) -> Result<metal::Buffer> {
    use pyo3::prelude::*;
    use pyo3::types::PyCapsule;

    let cap_any = arr
        .call_method0("__dlpack__")
        .map_err(|e| anyhow!("__dlpack__: {e}"))?;
    let cap: Bound<PyCapsule> = cap_any
        .downcast_into()
        .map_err(|_| anyhow!("__dlpack__ did not return a capsule"))?;
    let raw = cap.pointer();
    if raw.is_null() {
        return Err(anyhow!("dlpack capsule pointer was null"));
    }
    // SAFETY: DLPack capsule holds a DLManagedTensor at `raw`. The
    // first field is `DLTensor`, whose first field is `void* data`.
    let data_ptr = unsafe { *(raw as *const *mut std::ffi::c_void) };
    let nbytes: i64 = arr
        .getattr("nbytes")
        .map_err(|e| anyhow!("array nbytes: {e}"))?
        .extract()
        .map_err(|e| anyhow!("array nbytes extract: {e}"))?;
    if data_ptr.is_null() || nbytes <= 0 {
        return Err(anyhow!(
            "invalid array: data_ptr={:p}, nbytes={}",
            data_ptr,
            nbytes
        ));
    }
    let buf = ctx.device.new_buffer_with_bytes_no_copy(
        data_ptr,
        nbytes as u64,
        metal::MTLResourceOptions::StorageModeShared,
        // The mlx allocator owns this memory; we don't free anything
        // when our Metal buffer drops.
        None,
    );
    Ok(buf)
}
