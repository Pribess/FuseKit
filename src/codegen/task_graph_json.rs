//! `task_graph.json` emission.
//!
//! Contract between the Stage 4 compiler and the runtime
//! auto-dispatcher: every Task carries the kernel function name,
//! its dispatch grid, and per-buffer binding metadata so the runtime
//! can build a Metal indirect command buffer without reading the
//! original graph IR. Two binding flavours:
//!
//! - `external` — the input is a model parameter / KV cache slot /
//!   input token. The binding carries the Stage 1 `tensor_id` and,
//!   when the trace's DLPack pointer matched a tree-flattened
//!   parameter, a `source_path` like
//!   `model.layers.3.self_attn.q_proj.weight` so the runtime can
//!   look the live mlx tensor up directly.
//! - `scratch` — the input is the output of an upstream task. Each
//!   cross-task tensor gets a unique `scratch_id`; the runtime
//!   allocates one Metal buffer per scratch id at init time and
//!   reuses it across decode iterations.

use serde::Serialize;

use crate::codegen::kernels::{
    extra_bindings_for, kernel_function_name, kernel_grid_threads_for, kernel_threadgroup3_for,
    slot_role_key,
};
use crate::fusion::ir::{FusionBoundary, FusionGraph, ModelMeta};
use crate::graph::{Dtype, Graph, ScalarValue, TensorId};
use crate::loader::RawParam;
use crate::task::ir::TaskGraph;

const SCHEMA_VERSION: u32 = 2;

#[derive(Serialize)]
pub struct TaskGraphJson {
    pub version: u32,
    /// Number of active token rows this artifact was emitted for.
    /// Root decode artifacts use M=1; MTP/prefill variants may carry
    /// a larger M while sharing the same semantic graph.
    pub active_m: u32,
    pub tasks: Vec<JsonTask>,
    pub events: Vec<JsonEvent>,
    pub scratch_buffers: Vec<JsonScratch>,
    pub initial_tasks: Vec<u32>,
    /// Model parameter tree captured by Stage 1 (`mlx.utils.tree_flatten`
    /// order). The runtime consumes this alongside each task's
    /// `source_path` so external buffer bindings resolve to live mlx
    /// tensors without re-running the trace.
    pub model_params: Vec<JsonParam>,
    /// Forward-pass regions, in execution order. Each entry lists the
    /// task ids the host should pack into one Metal ICB. The runtime
    /// builds N ICBs (one per group), patches each transformer
    /// block's `__call__` to execute its group, and lets mlx-lm's
    /// outer loop keep managing the KV cache between groups.
    pub layer_groups: Vec<JsonLayerGroup>,
    /// Decode-time dispatch list. A strict subset of `tasks` covering
    /// only the kernels the runtime actually fires per generate
    /// step: synthetic decode primitives (embed lookup, RoPE+cache
    /// store, argmax) plus the cache-touching trace patterns
    /// (rmsnorm_qkv, sdpa_decode, linear_residual, rmsnorm_gateup_swiglu,
    /// final_rmsnorm_lm_head). General-fusion / view ops that came
    /// from prefill shape manipulation stay in `tasks` for the
    /// compiler's record but don't appear here. Order matches
    /// execution: embed → 24×(layer pipeline) → final norm + lm_head
    /// → argmax.
    pub decode_pipeline: Vec<u32>,
    /// Capability verdict for the decode fast path. This is derived
    /// from graph semantics and task metadata, not model names.
    pub decode_capabilities: JsonDecodeCapabilities,
}

#[derive(Serialize)]
pub struct JsonLayerGroup {
    /// `"embed"`, `"layer.<N>"`, `"post"`, or `"misc"` for the few
    /// stray tasks that propagation couldn't tag.
    pub region: String,
    pub task_ids: Vec<u32>,
}

#[derive(Serialize)]
pub struct JsonDecodeCapabilities {
    pub dispatchable: bool,
    pub reasons: Vec<JsonCapabilityReason>,
}

#[derive(Serialize)]
pub struct JsonCapabilityReason {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_idx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<u32>,
    pub message: String,
}

/// One hand-tuned buffer slot beyond the IR-bound inputs/outputs.
/// The host wires these up at runtime — `role` names the kind of
/// resource (uniform, cache pointer, named scratch, model param).
#[derive(Serialize)]
pub struct JsonExtraBinding {
    pub slot: u32,
    pub role: String,
}

#[derive(Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum JsonScalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

#[derive(Serialize)]
pub struct JsonParam {
    pub path: String,
    pub shape: Vec<i32>,
    pub dtype: String,
}

#[derive(Serialize)]
pub struct JsonTask {
    pub id: u32,
    pub fusion_id: u32,
    pub kernel: String,
    /// Total threads to dispatch on each grid axis. The host passes
    /// these straight to `dispatchThreads:` (or ceil-divides for
    /// `dispatchThreadgroups:`). Y/Z are 1 for every emitter today.
    pub grid_threads: [u64; 3],
    /// Threadgroup size every emitter assumes. Encoded once per task
    /// so the host doesn't have to know which `TG_THREADS` constant
    /// the compiler used.
    pub threadgroup: [u64; 3],
    pub inputs: Vec<JsonBinding>,
    pub outputs: Vec<JsonBinding>,
    /// Hand-tuned MSL slots beyond `inputs`/`outputs` — uniforms
    /// (offset, kv_len, capacity, token_id, next_token), cache
    /// pointers, and per-layer scratch buffers the kernel author
    /// added without grounding them in IR boundary tensors. Empty
    /// for trace-bound fusions whose full signature is captured by
    /// `inputs` + `outputs`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_bindings: Vec<JsonExtraBinding>,
    /// Layer index this task belongs to, if the task is part of a
    /// per-layer dispatch. The runtime needs this to disambiguate
    /// `LayerOffset` / `LayerCacheK` / etc. across the 24 copies of
    /// each per-layer kernel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_idx: Option<i32>,
    pub depends_on: Vec<u32>,
    pub trigger_event: u32,
}

#[derive(Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum JsonBinding {
    External {
        tensor_id: u32,
        /// Python attribute path for the live mlx tensor backing this
        /// buffer. Empty for mlx-internal constants and runtime
        /// inputs (token ids, position offsets) — the host supplies
        /// those per invocation.
        #[serde(skip_serializing_if = "String::is_empty")]
        source_path: String,
        /// Materialized value for 0-dim trace constants (axis
        /// indices, broadcast dims, scale factors). The runtime
        /// allocates a one-element buffer and writes this in, so the
        /// kernel sees the same constant the trace did.
        #[serde(skip_serializing_if = "Option::is_none")]
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

#[derive(Serialize)]
pub struct JsonEvent {
    pub id: u32,
    pub num_triggers: u32,
    pub fires: Vec<u32>,
}

#[derive(Serialize)]
pub struct JsonScratch {
    pub id: u32,
    pub shape: Vec<i32>,
    pub dtype: String,
    pub size_bytes: u64,
}

/// Serialise `(graph, fg, tg, params)` into the v2 schema. The graph
/// is needed so external bindings can carry the `param_path` field
/// captured during Stage 1 lowering (DLPack-pointer joined against the
/// model's `tree_flatten` walk).
pub fn build(
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
    meta: &ModelMeta,
    params: &[RawParam],
    target: &crate::target::Target,
) -> TaskGraphJson {
    let active_m = emitted_active_m(fg, target).max(1);
    // Map a producer-tensor's `TensorId` to its scratch id. A scratch
    // id is assigned the first time a task produces the tensor — every
    // subsequent task reading that tensor binds the same id.
    let mut scratch_id_by_tensor: std::collections::HashMap<TensorId, u32> = Default::default();
    let mut scratch_buffers: Vec<JsonScratch> = Vec::new();

    let assign_scratch = |bd: &FusionBoundary,
                          tbl: &mut std::collections::HashMap<TensorId, u32>,
                          buf: &mut Vec<JsonScratch>|
     -> u32 {
        if let Some(&id) = tbl.get(&bd.source) {
            return id;
        }
        let id = buf.len() as u32;
        tbl.insert(bd.source, id);
        buf.push(JsonScratch {
            id,
            shape: bd.shape.clone(),
            dtype: dtype_label(bd.dtype).to_string(),
            size_bytes: scratch_size_bytes(&bd.shape, bd.dtype),
        });
        id
    };

    // First pass: register every fusion-output tensor as a scratch
    // buffer. Subsequent inputs reading that tensor become Scratch
    // bindings; inputs that miss the table are model-external.
    for fusion in &fg.fusions {
        for output in &fusion.outputs {
            assign_scratch(output, &mut scratch_id_by_tensor, &mut scratch_buffers);
        }
    }

    // Second pass: alias provably stride-invariant Single view-ops to
    // their upstream scratch. Compiler-path FusedChain emit absorbs
    // metadata Singles during fusion but downstream Heavy steps still
    // reference the absorbed tensor's scratch_id in their inputs. At
    // decode time those Single tasks aren't in decode_pipeline, so
    // without aliasing the chain would read uninitialised memory.
    //
    // Squeeze / ExpandDims are always safe: they add or drop a size-1
    // dimension and CAN'T change byte layout. Flatten / Unflatten /
    // Reshape are safe when they are just a contiguous row-major view
    // over a tensor already produced by a decode fusion. That covers
    // the common MLP bridge shape:
    //   gate/up [L, intermediate] -> Unflatten [B, L, intermediate]
    //   swiglu  [L, intermediate] -> Flatten   [L, intermediate]
    // without running dead view tasks or reading uninitialised scratch.
    //
    // Transpose / AsContiguous can require a materialising permute, so
    // they are still only aliased for the special SDPA bridge described
    // below.
    // Env-gated view aliasing — `FUSEKIT_VIEW_ALIAS=0` disables the
    // post-pass when bisecting perf regressions.
    if std::env::var("FUSEKIT_VIEW_ALIAS").as_deref() != Ok("0") {
        use crate::fusion::ir::FusionKind;
        // (1) Producers tracker: for each tensor, which fusion is its
        // source? Used to bypass view chains that sit downstream of
        // fusekit's SDPA piece, which writes a non-mlx-shaped buffer.
        let producer_of: std::collections::HashMap<TensorId, usize> = fg
            .fusions
            .iter()
            .enumerate()
            .flat_map(|(i, f)| f.outputs.iter().map(move |b| (b.source, i)))
            .collect();
        // Tensors that came (transitively) from an SDPA fusekit output.
        // Once a tensor is in this set, any single-input/single-output
        // view-op consuming it can be aliased to its scratch — we know
        // the SDPA piece's byte layout matches what the o_proj chain's
        // flat `x[k]` indexing wants, regardless of the trace's strided
        // intermediates.
        let mut sdpa_descendant: std::collections::HashSet<TensorId> = Default::default();
        for f in &fg.fusions {
            if matches!(f.kind, FusionKind::SdpaDecode) {
                for out in &f.outputs {
                    sdpa_descendant.insert(out.source);
                }
            }
        }

        let numel = |shape: &[i32]| -> i64 { shape.iter().map(|&d| i64::from(d.max(1))).product() };
        let is_pure_metadata = |op_name: &str| -> bool {
            // Pure structural views (no possible byte permute).
            matches!(op_name, "Squeeze" | "ExpandDims")
        };
        let is_contiguous_reshape =
            |op_name: &str| -> bool { matches!(op_name, "Reshape" | "Flatten" | "Unflatten") };
        let producer_is_decode_value = |tid: TensorId| -> bool {
            let Some(&idx) = producer_of.get(&tid) else {
                return false;
            };
            let Some(f) = fg.fusions.get(idx) else {
                return false;
            };
            !matches!(
                &f.kind,
                FusionKind::Single(name)
                    if matches!(name.as_str(), "Transpose" | "AsContiguous")
            )
        };
        let is_sdpa_bridge_view = |op_name: &str| -> bool {
            // Views that CAN permute bytes (Reshape, Transpose,
            // Flatten, AsContiguous), but downstream of fusekit's
            // SDPA piece are safe to alias because the SDPA output's
            // actual layout matches what flat downstream readers
            // (Heavy-first chain `x[k]`) expect — the trace's
            // Transpose+Reshape only existed to undo mlx's strided
            // SDPA output, which fusekit doesn't produce.
            matches!(
                op_name,
                "Reshape" | "Flatten" | "Unflatten" | "AsContiguous" | "Transpose"
            )
        };
        loop {
            let mut changed = false;
            for fusion in &fg.fusions {
                let FusionKind::Single(op_name) = &fusion.kind else {
                    continue;
                };
                if fusion.inputs.len() != 1 || fusion.outputs.len() != 1 {
                    continue;
                }
                let in_tid = fusion.inputs[0].source;
                let out_tid = fusion.outputs[0].source;
                let same_elements =
                    numel(&fusion.inputs[0].shape) == numel(&fusion.outputs[0].shape);
                let safe = if is_pure_metadata(op_name) {
                    true
                } else if is_contiguous_reshape(op_name)
                    && same_elements
                    && producer_is_decode_value(in_tid)
                {
                    true
                } else if is_sdpa_bridge_view(op_name) && sdpa_descendant.contains(&in_tid) {
                    true
                } else {
                    false
                };
                if !safe {
                    continue;
                }
                let upstream = match scratch_id_by_tensor.get(&in_tid).copied() {
                    Some(id) => id,
                    None => continue,
                };
                let prev = scratch_id_by_tensor.get(&out_tid).copied();
                if prev != Some(upstream) {
                    scratch_id_by_tensor.insert(out_tid, upstream);
                    changed = true;
                }
                // Propagate SDPA descendant tag through aliased views.
                if sdpa_descendant.contains(&in_tid) {
                    if sdpa_descendant.insert(out_tid) {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    // x_in synthetic scratch — layer-0 chain's x input has no producer
    // (Embed Lookup synthetic has empty outputs).
    {
        use crate::fusion::category::{categorize_name, OpCategory};
        use crate::fusion::ir::{ChainOperand, FusionAttrs, FusionKind};
        let mut layer0_x_tid: Option<TensorId> = None;
        for f in &fg.fusions {
            let steps = match (&f.kind, &f.attrs) {
                (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => steps,
                _ => continue,
            };
            if steps.is_empty() {
                continue;
            }
            if categorize_name(&steps[0].op_name) != OpCategory::Reduction {
                continue;
            }
            for op in &steps[0].inputs {
                if let ChainOperand::External { buffer_index } = op {
                    if let Some(bd) = f.inputs.get(*buffer_index as usize) {
                        if !scratch_id_by_tensor.contains_key(&bd.source) {
                            layer0_x_tid = Some(bd.source);
                        }
                        break;
                    }
                }
            }
            if layer0_x_tid.is_some() {
                break;
            }
        }
        if let Some(tid) = layer0_x_tid {
            let tensor = &graph.tensors[tid as usize];
            let bytes = scratch_size_bytes(&tensor.shape, tensor.dtype);
            let id = scratch_buffers.len() as u32;
            scratch_buffers.push(JsonScratch {
                id,
                shape: tensor.shape.clone(),
                dtype: dtype_label(tensor.dtype).to_string(),
                size_bytes: bytes,
            });
            scratch_id_by_tensor.insert(tid, id);
        }
    }

    // View-chain provenance walk — for chain inputs whose tensor has
    // no producer fusion (view chain DCE'd during IR pass leaves the
    // consumer's `inputs` boundary referencing an orphan tensor),
    // walk back through `graph.ops` to find the true upstream producer
    // — which IS in `scratch_id_by_tensor` because its fusion survived
    // DCE. mlx views (Reshape/Transpose/Flatten/Squeeze/ExpandDims/
    // AsContiguous) preserve byte layout; the chain's flat indexing
    // reads the same bytes regardless of orphan logical shape.
    {
        use crate::fusion::ir::FusionKind;
        let producer_op: std::collections::HashMap<TensorId, usize> = graph
            .ops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| op.outputs.iter().map(move |&t| (t, i)))
            .collect();
        // In-place ops (RoPE, Quantize) are byte-equivalent to view
        // ops for alias purposes: synthetic kernels write back to the
        // same buffer the input lives in (ExtraBinding mechanism
        // routes to layer_scratch.q_buf / x_in / etc.). Treating them
        // as view ops lets the orphan walk-back find the canonical
        // upstream scratch.
        let is_view_op = |name: &str| {
            matches!(
                name,
                "Reshape"
                    | "Flatten"
                    | "Unflatten"
                    | "Squeeze"
                    | "ExpandDims"
                    | "AsContiguous"
                    | "Transpose"
                    | "Broadcast"
                    | "RoPE"
                    | "Quantize"
            )
        };
        // mlx's QuantizedLinear has bias as a separate `Add(qm_out,
        // bias)` op in the trace. Chain absorbs only the QM (no
        // AddMM in compiler-path), so the bias Add becomes a Single
        // that gets DCE'd → downstream RoPE input is "orphan". For
        // alias purposes, treat Add as walk-through on input[0] (the
        // heavy/data side; input[1] is the bias tensor with separate
        // param_path).
        // True iff `tid` is produced (directly or via a view chain) by a
        // model-parameter tensor. mlx-lm's `qm_out + linear_bias` lowers
        // to `Add(qm, Broadcast(bias))` — the bias side is a view chain
        // ending at the param, not the param directly.
        let view_back_to_param = |start: TensorId| -> bool {
            let mut cur = start;
            let mut seen: std::collections::HashSet<TensorId> = Default::default();
            while seen.insert(cur) {
                if let Some(t) = graph.tensors.get(cur as usize) {
                    if t.param_path.is_some() {
                        return true;
                    }
                }
                let op = match producer_op.get(&cur) {
                    Some(&i) => &graph.ops[i],
                    None => return false,
                };
                if !is_view_op(&op.name) || op.inputs.is_empty() {
                    return false;
                }
                cur = op.inputs[0];
            }
            false
        };
        let is_bias_add = |op: &crate::graph::Op| -> bool {
            if op.name != "Add" {
                return false;
            }
            if op.inputs.len() != 2 {
                return false;
            }
            view_back_to_param(op.inputs[0]) || view_back_to_param(op.inputs[1])
        };
        let dbg_walk = std::env::var_os("FUSEKIT_WALK_DEBUG").is_some();
        let walk_back = |start: TensorId,
                         scratch_map: &std::collections::HashMap<TensorId, u32>|
         -> Option<u32> {
            let mut cur = start;
            let mut seen: std::collections::HashSet<TensorId> = Default::default();
            while seen.insert(cur) {
                if let Some(&sid) = scratch_map.get(&cur) {
                    if dbg_walk {
                        eprintln!("[walk] {start} → {cur} = scratch {sid}");
                    }
                    return Some(sid);
                }
                let op_idx = match producer_op.get(&cur) {
                    Some(&i) => i,
                    None => {
                        if dbg_walk {
                            eprintln!("[walk] {start} → {cur} no producer");
                        }
                        return None;
                    }
                };
                let op = &graph.ops[op_idx];
                if dbg_walk {
                    eprintln!(
                        "[walk] {start} → {cur} producer={} ({:?})",
                        op.name, op.inputs
                    );
                }
                if !is_view_op(&op.name) && !is_bias_add(op) {
                    if dbg_walk {
                        eprintln!("[walk] stop at non-view {}", op.name);
                    }
                    return None;
                }
                if op.inputs.is_empty() {
                    return None;
                }
                // For bias-add, follow the data side (the input that
                // does NOT reduce back to a model parameter via the
                // view chain).
                if is_bias_add(op) {
                    cur = if view_back_to_param(op.inputs[0]) {
                        op.inputs[1]
                    } else {
                        op.inputs[0]
                    };
                } else {
                    cur = op.inputs[0];
                }
            }
            None
        };
        let consider = |kind: &FusionKind| {
            matches!(
                kind,
                FusionKind::FusedChain
                    | FusionKind::QmvFanout
                    | FusionKind::QmvSwiglu
                    | FusionKind::RoutedQmvSwiglu
                    | FusionKind::RoutedQmvReduce
                    | FusionKind::SdpaDecode
                    | FusionKind::EmbedLookupDecode
            )
        };
        let dbg = std::env::var_os("FUSEKIT_ALIAS_DEBUG").is_some();
        let mut to_insert: Vec<(TensorId, u32)> = Vec::new();
        for f in &fg.fusions {
            if !consider(&f.kind) {
                continue;
            }
            for bd in &f.inputs {
                if scratch_id_by_tensor.contains_key(&bd.source) {
                    continue;
                }
                if let Some(t) = graph.tensors.get(bd.source as usize) {
                    if t.param_path.is_some() {
                        continue;
                    }
                }
                if dbg {
                    let prod = producer_op.get(&bd.source).copied();
                    let prod_name = prod.and_then(|i| graph.ops.get(i)).map(|o| o.name.as_str());
                    eprintln!(
                        "[alias] orphan tensor {} shape={:?} producer_op={:?} op_name={:?}",
                        bd.source, bd.shape, prod, prod_name
                    );
                }
                if let Some(sid) = walk_back(bd.source, &scratch_id_by_tensor) {
                    to_insert.push((bd.source, sid));
                }
            }
        }
        for (tid, sid) in to_insert {
            scratch_id_by_tensor.insert(tid, sid);
        }
    }

    let mut tasks_json: Vec<JsonTask> = Vec::with_capacity(tg.tasks.len());
    for task in &tg.tasks {
        let fusion = &fg.fusions[task.fusion as usize];
        let kernel = kernel_function_name(fusion);
        if std::env::var("FUSEKIT_DEBUG_QMM_CHAIN_NAX_ID")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .is_some_and(|id| id == fusion.id)
        {
            eprintln!(
                "[task-graph-json] fusion={} kind={:?} kernel={} active_m_env={:?} nax_env={:?} nax_m_env={:?}",
                fusion.id,
                fusion.kind,
                kernel,
                std::env::var("FUSEKIT_ACTIVE_M").ok(),
                std::env::var("FUSEKIT_QMM_CHAIN_NAX").ok(),
                std::env::var("FUSEKIT_QMM_CHAIN_NAX_M").ok()
            );
        }
        let inputs: Vec<JsonBinding> = fusion
            .inputs
            .iter()
            .map(|bd| build_binding(bd, &scratch_id_by_tensor, graph))
            .collect();
        let outputs: Vec<JsonBinding> = fusion
            .outputs
            .iter()
            .map(|bd| build_binding(bd, &scratch_id_by_tensor, graph))
            .collect();
        let extra: Vec<JsonExtraBinding> = extra_bindings_for(fusion, meta)
            .into_iter()
            .map(|eb| JsonExtraBinding {
                slot: eb.slot,
                role: slot_role_key(eb.role),
            })
            .collect();
        // Layer index for the host's per-layer dispatch context.
        // Synthetic fusions (`op_ids` empty) set this explicitly via
        // `layer_hint`; trace-bound fusions get it from the majority
        // `layer_id` of the ops they absorbed — same rule the layer
        // groups use, kept consistent here.
        let layer_idx = match fusion.layer_hint {
            Some(crate::graph::LayerTag::Layer(n)) => Some(n),
            Some(_) => None,
            None => {
                let mut tally: std::collections::BTreeMap<i32, usize> = Default::default();
                for &op_id in &fusion.op_ids {
                    if let Some(crate::graph::LayerTag::Layer(n)) =
                        graph.ops[op_id as usize].layer_id
                    {
                        *tally.entry(n).or_insert(0) += 1;
                    }
                }
                tally
                    .into_iter()
                    .max_by_key(|(_, n)| *n)
                    .map(|(idx, _)| idx)
            }
        };
        let grid_threads = kernel_grid_threads_for(fusion, meta, target);
        let tg = kernel_threadgroup3_for(fusion, meta, target);
        tasks_json.push(JsonTask {
            id: task.id,
            fusion_id: task.fusion,
            kernel,
            grid_threads,
            threadgroup: tg,
            inputs,
            outputs,
            extra_bindings: extra,
            layer_idx,
            depends_on: task.deps.clone(),
            trigger_event: task.trigger,
        });
    }

    let events_json: Vec<JsonEvent> = tg
        .events
        .iter()
        .map(|e| JsonEvent {
            id: e.id,
            num_triggers: e.num_triggers,
            fires: e.fires.clone(),
        })
        .collect();

    let model_params: Vec<JsonParam> = params
        .iter()
        .map(|p| JsonParam {
            path: p.path.clone(),
            shape: p.shape.clone(),
            dtype: p.dtype.clone(),
        })
        .collect();

    let decode_pipeline = build_decode_pipeline(graph, fg, tg, &scratch_id_by_tensor);
    let decode_capabilities = analyze_decode_capabilities(graph, fg, tg, &decode_pipeline);
    let layer_groups = build_layer_groups(graph, fg, tg, &scratch_id_by_tensor);

    TaskGraphJson {
        version: SCHEMA_VERSION,
        active_m,
        tasks: tasks_json,
        events: events_json,
        scratch_buffers,
        initial_tasks: tg.initial.clone(),
        model_params,
        layer_groups,
        decode_pipeline,
        decode_capabilities,
    }
}

fn emitted_active_m(fg: &FusionGraph, target: &crate::target::Target) -> u32 {
    let trace_m = fg
        .fusions
        .iter()
        .flat_map(|f| f.inputs.iter().chain(f.outputs.iter()))
        .filter_map(|b| {
            if b.shape.len() >= 3 {
                Some(b.shape[1].max(1) as u32)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(1);
    target.chip.active_m(trace_m)
}

fn analyze_decode_capabilities(
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
    decode_pipeline: &[u32],
) -> JsonDecodeCapabilities {
    use crate::fusion::ir::{FusionAttrs, FusionKind};
    use crate::graph::LayerTag;
    use std::collections::{BTreeMap, BTreeSet};

    let decode_task_ids: BTreeSet<u32> = decode_pipeline.iter().copied().collect();
    let task_by_fusion: BTreeMap<usize, u32> =
        tg.tasks.iter().map(|t| (t.fusion as usize, t.id)).collect();
    let mut producer_task_by_tensor: BTreeMap<crate::graph::TensorId, u32> = BTreeMap::new();
    for (fusion_idx, fusion) in fg.fusions.iter().enumerate() {
        let Some(&task_id) = task_by_fusion.get(&fusion_idx) else {
            continue;
        };
        for output in &fusion.outputs {
            producer_task_by_tensor.insert(output.source, task_id);
        }
    }
    let semantic_sdpa_trace_deps = semantic_sdpa_trace_dependency_closure(fg, tg);
    let mut reasons: Vec<JsonCapabilityReason> = Vec::new();

    let single_decode_supported = |name: &str| -> bool {
        matches!(
            name,
            "Add"
                | "Subtract"
                | "Multiply"
                | "Divide"
                | "Power"
                | "Maximum"
                | "Minimum"
                | "Sigmoid"
                | "Exp"
                | "Negative"
                | "LogAddExp"
                | "Select"
                | "GreaterEqual"
                | "AsType"
                | "Reshape"
                | "Flatten"
                | "Unflatten"
                | "Squeeze"
                | "ExpandDims"
                | "AsContiguous"
                | "Broadcast"
                | "Transpose"
                | "Slice"
                | "Matmul"
                | "ScaledDotProductAttention"
                | "Softmax"
                | "ArgPartition"
                | "Argpartition"
                | "Arange"
                | "Full"
                | "Gather"
                | "GatherAxis"
                | "GatherQMM"
                | "WeightedRouteReduce"
                | "Concatenate"
                | "Reduce"
                | "Convolution"
                | "ConvolutionSilu"
                | "Split"
                | "RMSNorm"
                | "QuantizedMatmul"
                | "CustomKernel"
                | "RoPE"
        )
    };

    let layer_index = |fusion: &crate::fusion::ir::Fusion| -> Option<i32> {
        if let Some(LayerTag::Layer(i)) = fusion.layer_hint {
            return Some(i);
        }
        let mut tally: BTreeMap<i32, usize> = Default::default();
        for &op_id in &fusion.op_ids {
            if let Some(LayerTag::Layer(i)) = graph.ops[op_id as usize].layer_id {
                *tally.entry(i).or_insert(0) += 1;
            }
        }
        tally.into_iter().max_by_key(|(_, n)| *n).map(|(i, _)| i)
    };

    let mut sdpa_layers: BTreeSet<i32> = BTreeSet::new();
    let mut kv_store_layers: BTreeSet<i32> = BTreeSet::new();
    for task_id in decode_pipeline {
        let task = &tg.tasks[*task_id as usize];
        let fusion = &fg.fusions[task.fusion as usize];
        let Some(layer) = layer_index(fusion) else {
            continue;
        };
        match fusion.kind {
            FusionKind::SdpaDecode => {
                sdpa_layers.insert(layer);
            }
            FusionKind::Single(ref name) if name == "ScaledDotProductAttention" => {
                sdpa_layers.insert(layer);
            }
            FusionKind::RopeKvCacheStoreDecode => {
                kv_store_layers.insert(layer);
            }
            FusionKind::KvCacheStoreDecode => {
                kv_store_layers.insert(layer);
            }
            FusionKind::RmsNormRopeKvCacheStoreDecode => {
                kv_store_layers.insert(layer);
            }
            FusionKind::RmsNormQkv
                if matches!(
                    fusion.attrs,
                    FusionAttrs::RmsNormQkv {
                        decode_sinks: Some(_),
                        ..
                    }
                ) =>
            {
                kv_store_layers.insert(layer);
            }
            _ => {}
        }
    }

    for layer in sdpa_layers.difference(&kv_store_layers) {
        reasons.push(JsonCapabilityReason {
            code: "missing_current_token_kv_store".to_string(),
            layer_idx: Some(*layer),
            task_id: None,
            message: "decode SDPA is present, but no semantic KV-cache store primitive writes the current token before SDPA".to_string(),
        });
    }

    for task_id in decode_pipeline {
        let Some(task) = tg.tasks.get(*task_id as usize) else {
            continue;
        };
        let fusion = &fg.fusions[task.fusion as usize];
        let consumer_layer = layer_index(fusion);
        let overridden_input_slots: BTreeSet<usize> = extra_bindings_for(fusion, &fg.meta)
            .into_iter()
            .filter_map(|eb| {
                let slot = eb.slot as usize;
                (slot < fusion.inputs.len()).then_some(slot)
            })
            .collect();
        for &dep_task_id in &task.deps {
            if decode_task_ids.contains(&dep_task_id) {
                continue;
            }
            let mut only_overridden_inputs = false;
            if !overridden_input_slots.is_empty() {
                let mut seen_dep_input = false;
                only_overridden_inputs = true;
                for (slot, input) in fusion.inputs.iter().enumerate() {
                    if producer_task_by_tensor.get(&input.source) != Some(&dep_task_id) {
                        continue;
                    }
                    seen_dep_input = true;
                    if !overridden_input_slots.contains(&slot) {
                        only_overridden_inputs = false;
                        break;
                    }
                }
                only_overridden_inputs &= seen_dep_input;
            }
            if only_overridden_inputs {
                continue;
            }
            if matches!(fusion.kind, FusionKind::SdpaDecode)
                && semantic_sdpa_trace_deps.contains(&dep_task_id)
            {
                continue;
            }
            if is_decode_safe_omitted_dependency(fg, tg, dep_task_id, &decode_task_ids) {
                continue;
            }
            let Some(dep_task) = tg.tasks.get(dep_task_id as usize) else {
                continue;
            };
            let dep_fusion = &fg.fusions[dep_task.fusion as usize];
            let dep_layer = layer_index(dep_fusion).or(consumer_layer);
            reasons.push(JsonCapabilityReason {
                code: "undispatched_decode_dependency".to_string(),
                layer_idx: dep_layer,
                task_id: Some(dep_task_id),
                message: format!(
                    "decode task {task_id} depends on producer task {dep_task_id} ({:?}), but that producer is not in the decode pipeline and is not a safe omitted view",
                    dep_fusion.kind
                ),
            });
        }
    }

    for (fusion_idx, fusion) in fg.fusions.iter().enumerate() {
        let Some(layer) = layer_index(fusion) else {
            continue;
        };
        let task_id = task_by_fusion.get(&fusion_idx).copied();
        let in_decode = task_id.is_some_and(|id| decode_task_ids.contains(&id));
        match (&fusion.kind, &fusion.attrs) {
            (FusionKind::Single(name), _) if in_decode && !single_decode_supported(name) => {
                reasons.push(JsonCapabilityReason {
                    code: "unsupported_single_decode_op".to_string(),
                    layer_idx: Some(layer),
                    task_id,
                    message: format!(
                        "decode pipeline contains Single({name}) but fusekit has no correct emitter for it"
                    ),
                });
            }
            (FusionKind::Single(name), _)
                if name == "ScaledDotProductAttention"
                    && !in_decode
                    && !sdpa_layers.contains(&layer) =>
            {
                reasons.push(JsonCapabilityReason {
                    code: "attention_not_lowered".to_string(),
                    layer_idx: Some(layer),
                    task_id,
                    message: "scaled-dot-product attention is present in the graph but was not lowered into a decode attention primitive".to_string(),
                });
            }
            (FusionKind::Single(name), _)
                if matches!(name.as_str(), "Convolution" | "ConvolutionSilu") && !in_decode =>
            {
                reasons.push(JsonCapabilityReason {
                    code: "stateful_convolution_not_lowered".to_string(),
                    layer_idx: Some(layer),
                    task_id,
                    message: "stateful decode convolution is present in the graph but has no runtime state/cache binding".to_string(),
                });
            }
            (
                FusionKind::Single(_),
                FusionAttrs::Single {
                    op_attrs: crate::graph::Attrs::CustomKernel { name, .. },
                    ..
                },
            ) if name.contains("gated_delta") && !in_decode => {
                reasons.push(JsonCapabilityReason {
                    code: "stateful_gated_delta_not_lowered".to_string(),
                    layer_idx: Some(layer),
                    task_id,
                    message: "gated-delta custom kernel is present in the graph but its recurrent state is not bound into the decode pipeline".to_string(),
                });
            }
            _ => {}
        }
    }

    reasons
        .sort_by(|a, b| (a.layer_idx, a.task_id, &a.code).cmp(&(b.layer_idx, b.task_id, &b.code)));
    reasons
        .dedup_by(|a, b| a.code == b.code && a.layer_idx == b.layer_idx && a.task_id == b.task_id);

    JsonDecodeCapabilities {
        dispatchable: reasons.is_empty(),
        reasons,
    }
}

fn is_decode_safe_omitted_dependency(
    fg: &FusionGraph,
    tg: &TaskGraph,
    task_id: u32,
    decode_task_ids: &std::collections::BTreeSet<u32>,
) -> bool {
    use crate::fusion::ir::FusionKind;
    use std::collections::{BTreeSet, VecDeque};

    fn view_name(fusion: &crate::fusion::ir::Fusion) -> Option<&str> {
        match &fusion.kind {
            FusionKind::Single(name)
                if matches!(
                    name.as_str(),
                    "Reshape"
                        | "Flatten"
                        | "Unflatten"
                        | "Squeeze"
                        | "ExpandDims"
                        | "Transpose"
                        | "AsContiguous"
                ) =>
            {
                Some(name.as_str())
            }
            _ => None,
        }
    }
    fn is_decode_embed_gather(fusion: &crate::fusion::ir::Fusion) -> bool {
        let FusionKind::Single(name) = &fusion.kind else {
            return false;
        };
        if name != "Gather" || fusion.inputs.len() < 2 || fusion.outputs.len() != 1 {
            return false;
        }
        let table = &fusion.inputs[0];
        let index = &fusion.inputs[1];
        let out = &fusion.outputs[0];
        let index_dtype = matches!(
            index.dtype,
            crate::graph::Dtype::I32
                | crate::graph::Dtype::I64
                | crate::graph::Dtype::U32
                | crate::graph::Dtype::U64
        );
        index_dtype
            && table.shape.len() == 2
            && out.shape.last() == table.shape.last()
            && table.shape[0] > table.shape[1].max(1)
    }

    let mut queue = VecDeque::from([task_id]);
    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut saw_selected_producer = false;
    let mut requires_sdpa_bridge = false;
    let mut saw_selected_sdpa = false;

    while let Some(cur_task_id) = queue.pop_front() {
        if !visited.insert(cur_task_id) {
            continue;
        }
        let Some(task) = tg.tasks.get(cur_task_id as usize) else {
            return false;
        };
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            return false;
        };

        if decode_task_ids.contains(&cur_task_id) {
            saw_selected_producer = true;
            if matches!(fusion.kind, FusionKind::SdpaDecode) {
                saw_selected_sdpa = true;
            }
            continue;
        }

        if is_decode_embed_gather(fusion) {
            saw_selected_producer = true;
            continue;
        }

        let Some(name) = view_name(fusion) else {
            return false;
        };
        if matches!(name, "Transpose" | "AsContiguous") {
            requires_sdpa_bridge = true;
        }
        if task.deps.is_empty() {
            saw_selected_producer = true;
        }
        for &dep in &task.deps {
            queue.push_back(dep);
        }
    }

    saw_selected_producer && (!requires_sdpa_bridge || saw_selected_sdpa)
}

fn semantic_decode_boundary_task(kind: &crate::fusion::ir::FusionKind) -> bool {
    use crate::fusion::ir::FusionKind;
    matches!(
        kind,
        FusionKind::EmbedLookupDecode
            | FusionKind::RmsNormQkv
            | FusionKind::RopeQDecode
            | FusionKind::RopeKvCacheStoreDecode
            | FusionKind::KvCacheStoreDecode
            | FusionKind::RmsNormRopeQGateDecode
            | FusionKind::RmsNormRopeKvCacheStoreDecode
            | FusionKind::SdpaDecode
            | FusionKind::FusedChain
            | FusionKind::QmvFanout
            | FusionKind::QmvSwiglu
            | FusionKind::RoutedQmvSwiglu
            | FusionKind::RoutedQmvReduce
            | FusionKind::LayerBlock
            | FusionKind::LinearResidual
            | FusionKind::RmsNormGateUpSwiglu
            | FusionKind::RmsNormGateUpSwigluDown
            | FusionKind::FinalRmsNormLmHead
            | FusionKind::ArgmaxLogitsDecode
            | FusionKind::ArgmaxLogitsScan
            | FusionKind::ArgmaxLogitsFinalize
    )
}

/// Materialization views whose value has become unreachable after IR
/// input rewrites.
///
/// Example: RmsNormQkv bias inputs are lowered from
/// `Broadcast(param_bias) -> bias_q` to direct `param_bias` binding.
/// The Broadcast fusion remains in the full task list because the task
/// graph is built from the original fusion set, but no fusion input
/// references its output tensor anymore. Such a task has no semantic
/// side effect and would only create an init-time scratch round-trip if
/// kept in layer groups.
fn orphan_materialization_view_tasks(
    fg: &FusionGraph,
    tg: &TaskGraph,
) -> std::collections::BTreeSet<u32> {
    use crate::fusion::ir::FusionKind;
    use std::collections::BTreeSet;

    let consumed: BTreeSet<TensorId> = fg
        .fusions
        .iter()
        .flat_map(|fusion| fusion.inputs.iter().map(|bd| bd.source))
        .collect();

    tg.tasks
        .iter()
        .filter_map(|task| {
            let fusion = fg.fusions.get(task.fusion as usize)?;
            let is_dead_materialization = matches!(
                &fusion.kind,
                FusionKind::Single(name)
                    if matches!(
                        name.as_str(),
                        "Broadcast"
                            | "Full"
                            | "ExpandDims"
                            | "Squeeze"
                            | "Reshape"
                            | "Flatten"
                            | "Unflatten"
                            | "AsContiguous"
                    )
            ) && fusion
                .outputs
                .iter()
                .all(|output| !consumed.contains(&output.source));
            is_dead_materialization.then_some(task.id)
        })
        .collect()
}

fn extra_bound_input_slots(
    fusion: &crate::fusion::ir::Fusion,
    meta: &ModelMeta,
) -> std::collections::BTreeSet<usize> {
    extra_bindings_for(fusion, meta)
        .into_iter()
        .filter_map(|eb| {
            let slot = eb.slot as usize;
            (slot < fusion.inputs.len()).then_some(slot)
        })
        .collect()
}

fn fusion_input_source_is_extra_bound(
    fusion: &crate::fusion::ir::Fusion,
    meta: &ModelMeta,
    source: TensorId,
) -> bool {
    extra_bound_input_slots(fusion, meta)
        .into_iter()
        .any(|slot| {
            fusion
                .inputs
                .get(slot)
                .is_some_and(|bd| bd.source == source)
        })
}

/// Trace subgraphs feeding an extra-bound input are runtime dead: the
/// dispatcher binds that slot to a semantic live resource (KV cache,
/// q_buf, offset, etc.) instead of the trace tensor. Walk backward from
/// those input-specific producers and collect only non-semantic
/// materialization/view tasks. Semantic decode producers are hard
/// boundaries, so QKV/cache-store work that is still needed elsewhere
/// is not pruned.
fn semantic_sdpa_trace_dependency_closure(
    fg: &FusionGraph,
    tg: &TaskGraph,
) -> std::collections::BTreeSet<u32> {
    use crate::fusion::ir::FusionKind;
    use std::collections::{BTreeSet, HashMap, VecDeque};

    if std::env::var("FUSEKIT_ROUNDTRIP_PRUNE").as_deref() == Ok("0") {
        return BTreeSet::new();
    }
    let has_semantic_attention_replacement = fg.fusions.iter().any(|fusion| {
        matches!(fusion.kind, FusionKind::SdpaDecode | FusionKind::LayerBlock)
            || matches!(
                &fusion.kind,
                FusionKind::Single(name) if name == "ScaledDotProductAttention"
            )
    });
    if !has_semantic_attention_replacement {
        return BTreeSet::new();
    }

    let producer_by_tensor: HashMap<TensorId, u32> = tg
        .tasks
        .iter()
        .flat_map(|task| {
            let fusion = &fg.fusions[task.fusion as usize];
            fusion.outputs.iter().map(move |bd| (bd.source, task.id))
        })
        .collect();
    let consumers_by_tensor: HashMap<TensorId, Vec<u32>> = tg
        .tasks
        .iter()
        .flat_map(|task| {
            let fusion = &fg.fusions[task.fusion as usize];
            fusion.inputs.iter().map(move |bd| (bd.source, task.id))
        })
        .fold(HashMap::new(), |mut acc, (source, task_id)| {
            acc.entry(source).or_default().push(task_id);
            acc
        });

    let mut ignored = BTreeSet::new();
    let mut queue = VecDeque::new();
    for task in &tg.tasks {
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            continue;
        };
        for slot in extra_bound_input_slots(fusion, &fg.meta) {
            let Some(input) = fusion.inputs.get(slot) else {
                continue;
            };
            if let Some(&producer_task_id) = producer_by_tensor.get(&input.source) {
                queue.push_back(producer_task_id);
            }
        }
    }

    while let Some(task_id) = queue.pop_front() {
        let Some(task) = tg.tasks.get(task_id as usize) else {
            continue;
        };
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            continue;
        };
        if semantic_decode_boundary_task(&fusion.kind) {
            continue;
        }
        if !ignored.insert(task_id) {
            continue;
        }
        queue.extend(task.deps.iter().copied());
    }

    // If a candidate output is consumed by any non-candidate task on a
    // non-overridden slot, it is still live. Iterate because removing
    // one task can make its producer live too.
    loop {
        let mut removed = Vec::new();
        for &task_id in &ignored {
            let Some(task) = tg.tasks.get(task_id as usize) else {
                removed.push(task_id);
                continue;
            };
            let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
                removed.push(task_id);
                continue;
            };
            let mut live_outside = false;
            for output in &fusion.outputs {
                for &consumer_task_id in consumers_by_tensor
                    .get(&output.source)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                {
                    if consumer_task_id == task_id || ignored.contains(&consumer_task_id) {
                        continue;
                    }
                    let Some(consumer_task) = tg.tasks.get(consumer_task_id as usize) else {
                        live_outside = true;
                        break;
                    };
                    let Some(consumer_fusion) = fg.fusions.get(consumer_task.fusion as usize)
                    else {
                        live_outside = true;
                        break;
                    };
                    if !fusion_input_source_is_extra_bound(consumer_fusion, &fg.meta, output.source)
                    {
                        live_outside = true;
                        break;
                    }
                }
                if live_outside {
                    break;
                }
            }
            if live_outside {
                removed.push(task_id);
            }
        }
        if removed.is_empty() {
            break;
        }
        for task_id in removed {
            ignored.remove(&task_id);
        }
    }

    if std::env::var_os("FUSEKIT_ROUNDTRIP_PRUNE_DEBUG").is_some() && !ignored.is_empty() {
        let saved_bytes: u64 = ignored
            .iter()
            .filter_map(|&task_id| tg.tasks.get(task_id as usize))
            .filter_map(|task| fg.fusions.get(task.fusion as usize))
            .flat_map(|fusion| fusion.outputs.iter())
            .map(|bd| scratch_size_bytes(&bd.shape, bd.dtype).saturating_mul(2))
            .sum();
        eprintln!(
            "[roundtrip-prune] extra-bound trace deps: tasks={} approx_saved={:.3}MiB/step",
            ignored.len(),
            saved_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    ignored
}

/// Filter the full task list down to the kernels the host actually
/// fires per decode step, in execution order. Built by picking out
/// the cache-touching pattern fusions + the synthetic decode
/// primitives, dropping every general-fusion / view op that's only
/// meaningful during prefill.
fn build_decode_pipeline(
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
    scratch_id_by_tensor: &std::collections::HashMap<TensorId, u32>,
) -> Vec<u32> {
    use crate::fusion::ir::FusionKind;
    use crate::graph::LayerTag;

    // Per-layer ordering within each region. Smaller = earlier in
    // the decode pipeline. LinearResidual appears twice per layer
    // (o_proj after sdpa, down_proj after mlp); both share the
    // same kind so we tiebreak by op-id within a layer.
    //
    // FusedChain is overloaded across positions (pre-attention QKV
    // projection, post-SDPA o_proj+residual, MLP gate/up fanout,
    // down_proj+residual) so its order is computed per-fusion at the
    // call site (`fused_chain_order` closure below).
    fn order_key(kind: &FusionKind) -> Option<u32> {
        match kind {
            FusionKind::EmbedLookupDecode => Some(0),
            FusionKind::RmsNormQkv => Some(1),
            FusionKind::RopeQDecode => Some(2),
            FusionKind::RmsNormRopeQGateDecode => Some(2),
            FusionKind::RopeKvCacheStoreDecode => Some(3),
            FusionKind::KvCacheStoreDecode => Some(3),
            FusionKind::RmsNormRopeKvCacheStoreDecode => Some(3),
            FusionKind::SdpaDecode => Some(4),
            FusionKind::LayerBlock => Some(4),
            FusionKind::LinearResidual => Some(5),
            FusionKind::RmsNormGateUpSwiglu => Some(6),
            FusionKind::RmsNormGateUpSwigluDown => Some(6),
            FusionKind::SwigluActivation => Some(6),
            FusionKind::FinalRmsNormLmHead => Some(8),
            FusionKind::ArgmaxLogitsDecode => Some(9),
            // Multi-TG variants — scan runs before finalize.
            FusionKind::ArgmaxLogitsScan => Some(9),
            FusionKind::ArgmaxLogitsFinalize => Some(10),
            // Per-fusion (computed below).
            FusionKind::FusedChain => Some(99),
            FusionKind::QmvFanout => Some(99),
            FusionKind::QmvSwiglu => Some(99),
            FusionKind::RoutedQmvSwiglu => Some(99),
            FusionKind::RoutedQmvReduce => Some(99),
            FusionKind::Single(name)
                if matches!(
                    name.as_str(),
                    "RMSNorm" | "QuantizedMatmul" | "Sigmoid" | "Multiply" | "Add"
                ) =>
            {
                Some(99)
            }
            // Including view-op Singles in the decode pipeline was
            // tried but breaks fast-path correctness — fusekit's
            // SDPA decode piece writes its output in a flat
            // `(head_id * HEAD_DIM + d)` layout (per-decode L=1
            // row), while the trace's Transpose+Reshape were
            // designed for mlx's `(B, n_kv, n_h, L, hd)` strided
            // SDPA output. Running those view kernels against the
            // fusekit-SDPA-shaped buffer produces garbage. The
            // alias-from-SDPA-to-chain logic below handles this
            // case at the scratch-binding level instead, so view
            // Singles stay OUT of the pipeline.
            _ => None,
        }
    }

    // When LayerBlock mode is active, per-layer tasks for `SdpaDecode`
    // and the o_proj variant of `LinearResidual` get filtered out of
    // the decode pipeline — the LayerBlock task subsumes them. The
    // down_proj `LinearResidual` (later op-id in the layer) stays.
    let lb_mode = crate::fusion::layer_block::enabled();
    let lb_layers: std::collections::BTreeSet<i32> = if lb_mode {
        fg.fusions
            .iter()
            .filter_map(|f| match (&f.kind, f.layer_hint) {
                (FusionKind::LayerBlock, Some(LayerTag::Layer(i))) => Some(i),
                _ => None,
            })
            .collect()
    } else {
        std::collections::BTreeSet::new()
    };

    let layer_index = |fusion: &crate::fusion::ir::Fusion| -> Option<LayerTag> {
        if let Some(tag) = fusion.layer_hint {
            return Some(tag);
        }
        let mut tally: std::collections::BTreeMap<LayerTag, usize> = Default::default();
        for &op_id in &fusion.op_ids {
            if let Some(tag) = graph.ops[op_id as usize].layer_id {
                *tally.entry(tag).or_insert(0) += 1;
            }
        }
        tally.into_iter().max_by_key(|(_, n)| *n).map(|(t, _)| t)
    };
    let input_param_ends_with = |fusion: &crate::fusion::ir::Fusion, suffix: &str| -> bool {
        fusion.inputs.iter().any(|bd| {
            graph
                .tensors
                .get(bd.source as usize)
                .and_then(|t| t.param_path.as_deref())
                .is_some_and(|path| path.ends_with(suffix))
        })
    };
    let projection_order = |fusion: &crate::fusion::ir::Fusion| -> Option<u32> {
        if input_param_ends_with(fusion, "self_attn.q_proj.weight")
            || input_param_ends_with(fusion, "self_attn.k_proj.weight")
            || input_param_ends_with(fusion, "self_attn.v_proj.weight")
        {
            return Some(1);
        }
        if input_param_ends_with(fusion, "self_attn.o_proj.weight") {
            return Some(5);
        }
        if input_param_ends_with(fusion, "mlp.gate_proj.weight")
            || input_param_ends_with(fusion, "mlp.up_proj.weight")
        {
            return Some(6);
        }
        if input_param_ends_with(fusion, "mlp.down_proj.weight") {
            return Some(7);
        }
        None
    };

    let gated_delta_layers: std::collections::BTreeSet<i32> = fg
        .fusions
        .iter()
        .filter_map(|fusion| {
            let (
                FusionKind::Single(_),
                crate::fusion::ir::FusionAttrs::Single {
                    op_attrs: crate::graph::Attrs::CustomKernel { name, .. },
                    ..
                },
            ) = (&fusion.kind, &fusion.attrs)
            else {
                return None;
            };
            if !name.contains("gated_delta") {
                return None;
            }
            match layer_index(fusion) {
                Some(LayerTag::Layer(n)) => Some(n),
                _ => None,
            }
        })
        .collect();
    let prepared_kv_layers: std::collections::BTreeSet<i32> = fg
        .fusions
        .iter()
        .filter_map(|fusion| match (fusion.kind.clone(), layer_index(fusion)) {
            (FusionKind::KvCacheStoreDecode, Some(LayerTag::Layer(n))) => Some(n),
            _ => None,
        })
        .collect();
    let qwen3_next_attention_layers: std::collections::BTreeSet<i32> = fg
        .fusions
        .iter()
        .filter_map(|fusion| match (fusion.kind.clone(), layer_index(fusion)) {
            (FusionKind::RmsNormRopeQGateDecode, Some(LayerTag::Layer(n))) => Some(n),
            _ => None,
        })
        .collect();
    let semantic_sdpa_trace_deps = semantic_sdpa_trace_dependency_closure(fg, tg);

    // Region priority used to lay the pipeline out: embed → layers →
    // post. Layer tasks are then sorted by `order_key`, with op-id
    // min as tiebreaker (handles the two LinearResiduals per layer).
    fn region_key(tag: LayerTag) -> (u8, i32) {
        match tag {
            LayerTag::Embed => (0, 0),
            LayerTag::Layer(n) => (1, n),
            LayerTag::Post => (2, 0),
        }
    }

    // Per-fusion order for FusedChain — overloaded across the
    // transformer block. Detection by output count + last-dim,
    // mapping to a distinct slot in the standard order_key scheme so
    // the cross-kind sort places everything in the right sequence:
    //   3 outputs = Q/K/V pre-attention → 1 (before RoPE/SDPA)
    //   1 out, last==hidden, sees SDPA = o_proj+residual → 5
    //   2 outs same shape = gate/up MLP fanout → 6
    //   1 out, rank>=4 && last==head_dim = attention prep view/scale → 3
    //   1 out, last!=hidden && last!=vocab = silu_up activation → 7
    //   1 out, last==hidden, post-MLP = down_proj+residual → 7+pass2 split (8)
    //   1 out, last==vocab = final RMSNorm + LM head → 8
    //   else: post-attention misc (Q-scale view, mask) → 5
    let hidden = fg.meta.hidden_size as i32;
    let head_dim = fg.meta.head_dim as i32;
    let attn_dim = (fg.meta.num_attention_heads * fg.meta.head_dim) as i32;
    let vocab = fg.meta.vocab_size as i32;
    let is_vocab_projection_dim =
        |d: i32| -> bool { d == vocab || (hidden > 0 && d >= hidden.saturating_mul(32)) };
    let is_decode_embed_preamble = |fusion: &crate::fusion::ir::Fusion| -> bool {
        match &fusion.kind {
            FusionKind::Single(name) if name == "Gather" => {
                let Some(table) = fusion.inputs.first() else {
                    return false;
                };
                let Some(index) = fusion.inputs.get(1) else {
                    return false;
                };
                let Some(out) = fusion.outputs.first() else {
                    return false;
                };
                let index_dtype = matches!(
                    index.dtype,
                    Dtype::I32 | Dtype::I64 | Dtype::U32 | Dtype::U64
                );
                index_dtype
                    && table.shape.len() == 2
                    && table.shape.first() == Some(&vocab)
                    && table.shape.last() == Some(&hidden)
                    && out.shape.last() == Some(&hidden)
            }
            FusionKind::Single(name) if name == "Squeeze" => {
                if layer_index(fusion) != Some(LayerTag::Embed) {
                    return false;
                }
                let Some(input) = fusion.inputs.first() else {
                    return false;
                };
                let Some(out) = fusion.outputs.first() else {
                    return false;
                };
                input.shape.len() == 4
                    && out.shape.len() == 3
                    && input.shape.first() == Some(&1)
                    && input.shape.get(2) == Some(&1)
                    && input.shape.last() == Some(&hidden)
                    && out.shape.first() == Some(&1)
                    && out.shape.last() == Some(&hidden)
            }
            _ => false,
        }
    };
    let is_aliased_noop_view = |fusion: &crate::fusion::ir::Fusion| -> bool {
        let FusionKind::Single(name) = &fusion.kind else {
            return false;
        };
        if !matches!(
            name.as_str(),
            "Reshape"
                | "Flatten"
                | "Unflatten"
                | "Squeeze"
                | "ExpandDims"
                | "AsContiguous"
                | "Transpose"
        ) {
            return false;
        }
        if fusion.inputs.len() != 1 || fusion.outputs.len() != 1 {
            return false;
        }
        let in_tid = fusion.inputs[0].source;
        let out_tid = fusion.outputs[0].source;
        scratch_id_by_tensor.get(&in_tid).copied().is_some()
            && scratch_id_by_tensor.get(&in_tid) == scratch_id_by_tensor.get(&out_tid)
    };
    let fused_chain_order = |fusion: &crate::fusion::ir::Fusion| -> u32 {
        let outs = &fusion.outputs;
        let last_dim =
            |i: usize| -> Option<i32> { outs.get(i).and_then(|b| b.shape.last().copied()) };
        match outs.len() {
            3 => 1,
            2 if last_dim(0) == last_dim(1) => 6,
            1 => match last_dim(0) {
                Some(d) if d == head_dim && outs.first().is_some_and(|b| b.shape.len() >= 4) => 3,
                Some(d) if d == hidden => 5,
                Some(d) if d == attn_dim && d != hidden => 5,
                Some(d) if is_vocab_projection_dim(d) => 8,
                Some(_) => 7,
                None => 5,
            },
            _ => 5,
        }
    };
    let orphan_materialization_views = orphan_materialization_view_tasks(fg, tg);

    // Pass 1: collect (region, layer_rank, kind_order, op_id_min, task_id).
    // LinearResidual gets a transient order=5; we split it into o_proj
    // (stays 5) vs down_proj (promoted to 7) in pass 2 below.
    let mut entries: Vec<(u8, i32, u32, u32, u32)> = Vec::new();
    let mut forced_gated_delta_layers: std::collections::BTreeMap<u32, i32> = Default::default();
    for task in &tg.tasks {
        if orphan_materialization_views.contains(&task.id) {
            continue;
        }
        if semantic_sdpa_trace_deps.contains(&task.id) {
            continue;
        }
        let fusion = &fg.fusions[task.fusion as usize];
        if is_aliased_noop_view(fusion) {
            continue;
        }
        if is_decode_embed_preamble(fusion) {
            continue;
        }
        let out_last = fusion.outputs.first().and_then(|b| b.shape.last().copied());
        let force_post_vocab_qmm = matches!(
            &fusion.kind,
            FusionKind::Single(name)
                if name == "QuantizedMatmul" && out_last.is_some_and(is_vocab_projection_dim)
        );
        let tag = if force_post_vocab_qmm {
            Some(LayerTag::Post)
        } else {
            layer_index(fusion)
        };
        let Some(tag) = tag else {
            continue;
        };
        if let LayerTag::Layer(n) = tag {
            if qwen3_next_attention_layers.contains(&n)
                && matches!(fusion.kind, FusionKind::SdpaDecode)
                && !fusion.inputs.is_empty()
            {
                continue;
            }
        }
        if let LayerTag::Layer(n) = tag {
            if gated_delta_layers.contains(&n) {
                let include = !fusion.op_ids.is_empty()
                    || matches!(fusion.kind, FusionKind::ArraysCacheShiftAppendDecode);
                if include {
                    let trace_order =
                        if matches!(fusion.kind, FusionKind::ArraysCacheShiftAppendDecode) {
                            u32::MAX - 1
                        } else {
                            fusion.op_ids.iter().copied().min().unwrap_or(u32::MAX - 2)
                        };
                    let (rk0, rk1) = region_key(tag);
                    entries.push((rk0, rk1, trace_order, trace_order, task.id));
                    forced_gated_delta_layers.insert(task.id, n);
                }
                continue;
            }
        }
        let explicit_projection_order = projection_order(fusion);
        let Some(mut order) = explicit_projection_order.or_else(|| order_key(&fusion.kind)) else {
            continue;
        };
        if matches!(fusion.kind, FusionKind::FusedChain) {
            order = fused_chain_order(fusion);
        } else if let FusionKind::Single(name) = &fusion.kind {
            let is_post = matches!(tag, LayerTag::Post);
            let is_qwen3_next_attention_layer = match tag {
                LayerTag::Layer(n) => qwen3_next_attention_layers.contains(&n),
                _ => false,
            };
            let keep = if explicit_projection_order.is_some() {
                true
            } else {
                match name.as_str() {
                    "RMSNorm" => {
                        (is_post && out_last == Some(hidden))
                            || (is_qwen3_next_attention_layer && out_last == Some(hidden))
                    }
                    "QuantizedMatmul" => {
                        (is_post && out_last.is_some_and(is_vocab_projection_dim))
                            || is_qwen3_next_attention_layer
                    }
                    "Sigmoid" | "Multiply" => {
                        is_qwen3_next_attention_layer && out_last == Some(hidden)
                    }
                    "Add" => matches!(tag, LayerTag::Layer(_)) && out_last == Some(hidden),
                    _ => false,
                }
            };
            if !keep {
                continue;
            }
            order = if is_post {
                8
            } else if let Some(order) = explicit_projection_order {
                order
            } else {
                match name.as_str() {
                    "RMSNorm" | "QuantizedMatmul" => 1,
                    "Sigmoid" | "Multiply" => 4,
                    "Add" => 5,
                    _ => order,
                }
            };
        }
        // In LayerBlock mode, drop the SDPA task for layers where a
        // LayerBlock exists — LayerBlock fully replaces it. The
        // o_proj LinearResidual is filtered separately in pass 2
        // (we need both LinearResiduals visible to identify which
        // one is o_proj vs down_proj).
        if lb_mode {
            if let LayerTag::Layer(n) = tag {
                if matches!(fusion.kind, FusionKind::SdpaDecode) && lb_layers.contains(&n) {
                    continue;
                }
            }
        }
        let (rk0, rk1) = region_key(tag);
        let tiebreak = fusion.op_ids.iter().copied().min().unwrap_or(u32::MAX);
        entries.push((rk0, rk1, order, tiebreak, task.id));
    }

    // Some architectures expose Q/K/V and MLP projections as plain
    // trace-local producers rather than pattern fusions: e.g. a bias
    // add can make q_proj look like a generic LinearResidual, while
    // gate/up can remain separate QuantizedMatmul tasks. Pull the
    // same-layer producer closure for already-selected decode tasks so
    // the runtime executes the semantic inputs it will later bind as
    // q_buf/k_buf/v_buf/gate/up. This stays graph-structural: the only
    // role-specific part is parameter-path classification, not model
    // names.
    {
        let mut present: std::collections::BTreeSet<u32> = entries.iter().map(|e| e.4).collect();
        let producer_by_tensor: std::collections::HashMap<TensorId, u32> = tg
            .tasks
            .iter()
            .flat_map(|task| {
                let fusion = &fg.fusions[task.fusion as usize];
                fusion.outputs.iter().map(move |bd| (bd.source, task.id))
            })
            .collect();
        let producer_by_scratch: std::collections::HashMap<u32, u32> = tg
            .tasks
            .iter()
            .filter(|task| {
                let fusion = &fg.fusions[task.fusion as usize];
                !is_aliased_noop_view(fusion)
            })
            .flat_map(|task| {
                let fusion = &fg.fusions[task.fusion as usize];
                fusion.outputs.iter().filter_map(move |bd| {
                    scratch_id_by_tensor
                        .get(&bd.source)
                        .copied()
                        .map(|sid| (sid, task.id))
                })
            })
            .collect();
        let mut queue: std::collections::VecDeque<(LayerTag, u32, u32)> = entries
            .iter()
            .filter_map(|entry| {
                let task = tg.tasks.get(entry.4 as usize)?;
                let fusion = fg.fusions.get(task.fusion as usize)?;
                let tag = layer_index(fusion)?;
                Some((tag, entry.4, entry.2))
            })
            .collect();

        while let Some((consumer_tag, task_id, consumer_order)) = queue.pop_front() {
            let Some(task) = tg.tasks.get(task_id as usize) else {
                continue;
            };
            let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
                continue;
            };
            let mut deps: std::collections::BTreeSet<u32> = task.deps.iter().copied().collect();
            let overridden_input_slots = extra_bound_input_slots(fusion, &fg.meta);
            for (input_idx, input) in fusion.inputs.iter().enumerate() {
                if overridden_input_slots.contains(&input_idx) {
                    continue;
                }
                if let Some(dep) = producer_by_tensor.get(&input.source).copied() {
                    deps.insert(dep);
                }
                if let Some(dep) = scratch_id_by_tensor
                    .get(&input.source)
                    .and_then(|sid| producer_by_scratch.get(sid))
                    .copied()
                {
                    deps.insert(dep);
                }
            }
            for dep_task_id in deps {
                if dep_task_id == task_id {
                    continue;
                }
                if present.contains(&dep_task_id) {
                    continue;
                }
                if orphan_materialization_views.contains(&dep_task_id) {
                    continue;
                }
                if semantic_sdpa_trace_deps.contains(&dep_task_id) {
                    continue;
                }
                let Some(dep_task) = tg.tasks.get(dep_task_id as usize) else {
                    continue;
                };
                let Some(dep_fusion) = fg.fusions.get(dep_task.fusion as usize) else {
                    continue;
                };
                if is_aliased_noop_view(dep_fusion) {
                    continue;
                }
                if is_decode_embed_preamble(dep_fusion) {
                    continue;
                }
                let dep_tag = layer_index(dep_fusion).unwrap_or(consumer_tag);
                if dep_tag != consumer_tag {
                    continue;
                }
                present.insert(dep_task_id);
                let trace_order = dep_fusion
                    .op_ids
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(dep_task_id);
                let dep_order = projection_order(dep_fusion).unwrap_or(consumer_order);
                let (rk0, rk1) = region_key(dep_tag);
                entries.push((rk0, rk1, dep_order, trace_order, dep_task_id));
                queue.push_back((dep_tag, dep_task_id, dep_order));
            }
        }
    }

    // GatedDelta linear layers intentionally run the trace-local op
    // sequence instead of the standard attention-pattern subset. Pull
    // in any untagged producers those tasks depend on, such as scalar
    // Broadcast/Full constants captured outside the layer tag.
    if !forced_gated_delta_layers.is_empty() {
        let producer_by_tensor: std::collections::HashMap<TensorId, u32> = tg
            .tasks
            .iter()
            .flat_map(|task| {
                let fusion = &fg.fusions[task.fusion as usize];
                fusion.outputs.iter().map(move |bd| (bd.source, task.id))
            })
            .collect();
        let mut present: std::collections::BTreeSet<u32> = entries.iter().map(|e| e.4).collect();
        let mut queue: std::collections::VecDeque<u32> =
            forced_gated_delta_layers.keys().copied().collect();
        while let Some(task_id) = queue.pop_front() {
            let Some(&layer_n) = forced_gated_delta_layers.get(&task_id) else {
                continue;
            };
            let task = &tg.tasks[task_id as usize];
            let fusion = &fg.fusions[task.fusion as usize];
            for input in &fusion.inputs {
                let Some(&prod_task_id) = producer_by_tensor.get(&input.source) else {
                    continue;
                };
                let pf = &fg.fusions[tg.tasks[prod_task_id as usize].fusion as usize];
                if orphan_materialization_views.contains(&prod_task_id) {
                    continue;
                }
                if is_aliased_noop_view(pf) {
                    continue;
                }
                if is_decode_embed_preamble(pf) {
                    continue;
                }
                if !present.insert(prod_task_id) {
                    continue;
                }
                forced_gated_delta_layers.insert(prod_task_id, layer_n);
                queue.push_back(prod_task_id);
                let order = pf.op_ids.iter().copied().min().unwrap_or(prod_task_id);
                let (rk0, rk1) = region_key(LayerTag::Layer(layer_n));
                entries.push((rk0, rk1, order, order, prod_task_id));
            }
        }
    }

    if !prepared_kv_layers.is_empty() {
        let mut present: std::collections::BTreeSet<u32> = entries.iter().map(|e| e.4).collect();
        let mut prepared_seen: std::collections::BTreeSet<u32> = Default::default();
        let mut queue: std::collections::VecDeque<(i32, u32)> = Default::default();
        for task in &tg.tasks {
            let fusion = &fg.fusions[task.fusion as usize];
            let Some(LayerTag::Layer(layer_n)) = layer_index(fusion) else {
                continue;
            };
            if !prepared_kv_layers.contains(&layer_n) {
                continue;
            }
            match fusion.kind {
                FusionKind::SdpaDecode => {
                    // K/V slots are overridden to semantic cache buffers at
                    // runtime, but keeping the direct SDPA dependencies in the
                    // prep band preserves topological ordering for trace tasks
                    // that still reference those deps.
                    for &dep in &task.deps {
                        queue.push_back((layer_n, dep));
                    }
                }
                FusionKind::KvCacheStoreDecode => {
                    for &dep in &task.deps {
                        queue.push_back((layer_n, dep));
                    }
                }
                _ => {}
            }
        }
        while let Some((layer_n, prod_task_id)) = queue.pop_front() {
            let Some(task) = tg.tasks.get(prod_task_id as usize) else {
                continue;
            };
            let pf = &fg.fusions[task.fusion as usize];
            if orphan_materialization_views.contains(&prod_task_id) {
                continue;
            }
            if is_aliased_noop_view(pf) {
                continue;
            }
            if is_decode_embed_preamble(pf) {
                continue;
            }
            if !prepared_seen.insert(prod_task_id) {
                continue;
            }
            // Prepared-KV paths (e.g. MLA) materialize Q/K/V through a
            // trace-local producer subgraph before the semantic cache
            // store and SDPA kernels run. Keep those producers in the
            // attention-prep slot; their op-id remains the tiebreaker so
            // the original local order is preserved without letting raw
            // graph op ids push them behind SDPA/order=4. Some producers
            // are already in the normal decode set as generic FusedChain
            // tasks; rewrite those entries in-place instead of leaving
            // them in post-attention/MLP slots.
            let trace_order = pf.op_ids.iter().copied().min().unwrap_or(prod_task_id);
            if present.insert(prod_task_id) {
                let (rk0, rk1) = region_key(LayerTag::Layer(layer_n));
                entries.push((rk0, rk1, 2, trace_order, prod_task_id));
            } else if let Some(entry) = entries.iter_mut().find(|entry| entry.4 == prod_task_id) {
                if entry.2 > 2 {
                    entry.2 = 2;
                }
                entry.3 = entry.3.min(trace_order);
            }
            for &dep in &task.deps {
                queue.push_back((layer_n, dep));
            }
        }
    }

    // General-fusion chains can legally read trace constants that sit
    // outside every layer tag, e.g. Qwen2 attention scale lowered as
    // `Broadcast(0.125)` and shared by all layers. These are not safe to
    // omit: the runtime would otherwise allocate the referenced scratch as
    // zeros. Pull no-dependency scalar/materialized constants into the
    // decode pipeline next to their consumers and let the dependency sorter
    // place them before first use.
    {
        let mut present: std::collections::BTreeSet<u32> = entries.iter().map(|e| e.4).collect();
        let mut additions: Vec<(u8, i32, u32, u32, u32)> = Vec::new();
        let is_decode_constant_producer =
            |fusion: &crate::fusion::ir::Fusion, task: &crate::task::ir::Task| -> bool {
                if !task.deps.is_empty() {
                    return false;
                }
                matches!(
                    &fusion.kind,
                    FusionKind::Single(name) if matches!(name.as_str(), "Broadcast" | "Full")
                )
            };

        for &(rk0, rk1, order, tiebreak, task_id) in &entries {
            let Some(task) = tg.tasks.get(task_id as usize) else {
                continue;
            };
            for &dep_task_id in &task.deps {
                if present.contains(&dep_task_id) {
                    continue;
                }
                let Some(dep_task) = tg.tasks.get(dep_task_id as usize) else {
                    continue;
                };
                let Some(dep_fusion) = fg.fusions.get(dep_task.fusion as usize) else {
                    continue;
                };
                if is_aliased_noop_view(dep_fusion) {
                    continue;
                }
                if !is_decode_constant_producer(dep_fusion, dep_task) {
                    continue;
                }
                present.insert(dep_task_id);
                let dep_tiebreak = dep_fusion
                    .op_ids
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(dep_task_id)
                    .min(tiebreak);
                additions.push((rk0, rk1, order, dep_tiebreak, dep_task_id));
            }
        }
        entries.extend(additions);
    }

    // MoE decode paths (DeepSeek-style routed experts) are already in
    // the general IR as trace-local Single/FusedChain tasks. Pull the
    // producer closure for the MLP band into the decode pipeline so
    // GatherQMM/GatherAxis/top-k/Reduce materialization is explicit
    // instead of falling back to zero-initialized scratch.
    {
        let moe_layers: std::collections::BTreeSet<i32> = fg
            .fusions
            .iter()
            .filter_map(|fusion| match (&fusion.kind, layer_index(fusion)) {
                (FusionKind::Single(name), Some(LayerTag::Layer(n)))
                    if matches!(name.as_str(), "GatherQMM" | "GatherAxis") =>
                {
                    Some(n)
                }
                _ => None,
            })
            .collect();

        if !moe_layers.is_empty() {
            let mut present: std::collections::BTreeSet<u32> =
                entries.iter().map(|e| e.4).collect();
            let mut queue: std::collections::VecDeque<(i32, u32, u32)> = entries
                .iter()
                .filter_map(|entry| {
                    let task = tg.tasks.get(entry.4 as usize)?;
                    let fusion = fg.fusions.get(task.fusion as usize)?;
                    let Some(LayerTag::Layer(n)) = layer_index(fusion) else {
                        return None;
                    };
                    (moe_layers.contains(&n) && entry.2 >= 5).then_some((n, entry.4, entry.2))
                })
                .collect();

            while let Some((layer_n, task_id, consumer_order)) = queue.pop_front() {
                let Some(task) = tg.tasks.get(task_id as usize) else {
                    continue;
                };
                for &dep_task_id in &task.deps {
                    if present.contains(&dep_task_id) {
                        continue;
                    }
                    if orphan_materialization_views.contains(&dep_task_id) {
                        continue;
                    }
                    let Some(dep_task) = tg.tasks.get(dep_task_id as usize) else {
                        continue;
                    };
                    let Some(dep_fusion) = fg.fusions.get(dep_task.fusion as usize) else {
                        continue;
                    };
                    if is_aliased_noop_view(dep_fusion) {
                        continue;
                    }
                    if is_decode_embed_preamble(dep_fusion) {
                        continue;
                    }
                    let dep_layer = layer_index(dep_fusion);
                    let dep_is_same_layer = dep_layer == Some(LayerTag::Layer(layer_n));
                    let dep_is_moe_trace_op = matches!(
                        &dep_fusion.kind,
                        FusionKind::Single(name)
                            if matches!(
                                name.as_str(),
                                "Arange"
                                    | "Reshape"
                                    | "Flatten"
                                    | "Squeeze"
                                    | "ExpandDims"
                                    | "Broadcast"
                                    | "Full"
                                    | "Slice"
                                    | "ArgPartition"
                                    | "Argpartition"
                                    | "GatherAxis"
                                    | "GatherQMM"
                                    | "WeightedRouteReduce"
                                    | "Reduce"
                                    | "Multiply"
                                    | "Softmax"
                                    | "Negative"
                                    | "QuantizedMatmul"
                            )
                    );
                    if !dep_is_same_layer && !dep_is_moe_trace_op {
                        continue;
                    }
                    present.insert(dep_task_id);
                    let trace_order = dep_fusion
                        .op_ids
                        .iter()
                        .copied()
                        .min()
                        .unwrap_or(dep_task_id);
                    let (rk0, rk1) = region_key(LayerTag::Layer(layer_n));
                    entries.push((rk0, rk1, consumer_order, trace_order, dep_task_id));
                    queue.push_back((layer_n, dep_task_id, consumer_order));
                }
            }
        }
    }

    // SDPA bridge views are represented by scratch aliasing above. If a
    // producer-closure pulls them back into the decode set, drop them here:
    // running a trace Transpose/Reshape over fusekit's flat decode SDPA output
    // can corrupt the aliased buffer, while downstream o_proj already reads the
    // SDPA scratch through the canonical alias.
    {
        let mut bridge_views: std::collections::BTreeSet<u32> = Default::default();
        let sdpa_tasks: std::collections::BTreeSet<u32> = entries
            .iter()
            .filter_map(|entry| {
                let task = tg.tasks.get(entry.4 as usize)?;
                let fusion = fg.fusions.get(task.fusion as usize)?;
                matches!(fusion.kind, FusionKind::SdpaDecode).then_some(entry.4)
            })
            .collect();
        let is_view_single = |fusion: &crate::fusion::ir::Fusion| -> bool {
            matches!(
                &fusion.kind,
                FusionKind::Single(name)
                    if matches!(
                        name.as_str(),
                        "Reshape"
                            | "Flatten"
                            | "Unflatten"
                            | "AsContiguous"
                            | "Transpose"
                            | "Squeeze"
                            | "ExpandDims"
                    )
            )
        };
        loop {
            let mut changed = false;
            for entry in &entries {
                if bridge_views.contains(&entry.4) {
                    continue;
                }
                let Some(task) = tg.tasks.get(entry.4 as usize) else {
                    continue;
                };
                let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
                    continue;
                };
                if !is_view_single(fusion) {
                    continue;
                }
                let is_sdpa_descendant = task
                    .deps
                    .iter()
                    .any(|dep| sdpa_tasks.contains(dep) || bridge_views.contains(dep));
                if is_sdpa_descendant {
                    bridge_views.insert(entry.4);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if !bridge_views.is_empty() {
            for entry in &mut entries {
                let Some(task) = tg.tasks.get(entry.4 as usize) else {
                    continue;
                };
                let consumes_bridge_view = task.deps.iter().any(|dep| bridge_views.contains(dep));
                if consumes_bridge_view && entry.2 < 5 {
                    entry.2 = 5;
                }
            }
            entries.retain(|entry| !bridge_views.contains(&entry.4));
        }
    }

    // Pass 2: per layer, find the two LinearResidual entries. The
    // earlier (smaller op_id_min) is o_proj — keep order=5. The
    // later is down_proj — promote to order=7 so it lands after
    // rmsnorm_gateup_swiglu (order=6).
    //
    // In LayerBlock mode, the o_proj is absorbed → mark the first
    // entry for removal and only the second (down_proj) stays.
    let mut by_layer: std::collections::BTreeMap<(u8, i32), Vec<usize>> = Default::default();
    for (i, e) in entries.iter().enumerate() {
        let task = &tg.tasks[e.4 as usize];
        let fusion = &fg.fusions[task.fusion as usize];
        let is_hidden_residual = fusion
            .outputs
            .first()
            .and_then(|b| b.shape.last().copied())
            .is_some_and(|d| d == hidden);
        if e.2 == 5 && is_hidden_residual {
            by_layer.entry((e.0, e.1)).or_default().push(i);
        }
    }
    let mut drop_idxs: std::collections::BTreeSet<usize> = Default::default();
    for ((_, layer_n), idxs) in by_layer.iter_mut() {
        if idxs.len() != 2 {
            continue;
        }
        idxs.sort_by_key(|&i| entries[i].3);
        let first = idxs[0];
        let second = idxs[1];
        // Second is down_proj — bump to 7.
        entries[second].2 = 7;
        // First is o_proj — drop in LayerBlock mode for layers covered.
        if lb_mode && lb_layers.contains(layer_n) {
            drop_idxs.insert(first);
        }
    }
    if !drop_idxs.is_empty() {
        entries = entries
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !drop_idxs.contains(i))
            .map(|(_, e)| e)
            .collect();
    }
    entries.retain(|entry| {
        if orphan_materialization_views.contains(&entry.4) {
            return false;
        }
        tg.tasks
            .get(entry.4 as usize)
            .and_then(|task| fg.fusions.get(task.fusion as usize))
            .is_none_or(|fusion| !is_aliased_noop_view(fusion))
    });
    entries.sort();
    let ordered = entries.into_iter().map(|(_, _, _, _, id)| id).collect();
    dependency_order_decode_pipeline(ordered, graph, tg, fg, scratch_id_by_tensor)
}

fn dependency_order_decode_pipeline(
    mut ordered: Vec<u32>,
    graph: &Graph,
    tg: &TaskGraph,
    fg: &FusionGraph,
    scratch_id_by_tensor: &std::collections::HashMap<TensorId, u32>,
) -> Vec<u32> {
    use crate::fusion::ir::FusionKind;
    use crate::graph::LayerTag;

    let selected: std::collections::BTreeSet<u32> = ordered.iter().copied().collect();
    let layer_hint = |task_id: u32| -> Option<i32> {
        let task = tg.tasks.get(task_id as usize)?;
        let fusion = fg.fusions.get(task.fusion as usize)?;
        if let Some(layer) = match fusion.layer_hint {
            Some(LayerTag::Layer(n)) => Some(n),
            _ => None,
        } {
            return Some(layer);
        }
        let mut tally: std::collections::BTreeMap<i32, usize> = Default::default();
        for &op_id in &fusion.op_ids {
            if let Some(LayerTag::Layer(n)) = graph.ops[op_id as usize].layer_id {
                *tally.entry(n).or_insert(0) += 1;
            }
        }
        tally.into_iter().max_by_key(|(_, n)| *n).map(|(n, _)| n)
    };
    let mut producer_by_tensor: std::collections::HashMap<TensorId, u32> = Default::default();
    let mut producer_by_scratch: std::collections::HashMap<u32, u32> = Default::default();
    for task_id in &selected {
        let Some(task) = tg.tasks.get(*task_id as usize) else {
            continue;
        };
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            continue;
        };
        for output in &fusion.outputs {
            producer_by_tensor.insert(output.source, *task_id);
            if let Some(&sid) = scratch_id_by_tensor.get(&output.source) {
                producer_by_scratch.insert(sid, *task_id);
            }
        }
    }

    let data_deps = |task_id: u32| -> Vec<u32> {
        let Some(task) = tg.tasks.get(task_id as usize) else {
            return Vec::new();
        };
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            return Vec::new();
        };
        let mut deps = Vec::new();
        for input in &fusion.inputs {
            let mut dep = producer_by_tensor.get(&input.source).copied();
            if dep.is_none() {
                dep = scratch_id_by_tensor
                    .get(&input.source)
                    .and_then(|sid| producer_by_scratch.get(sid))
                    .copied();
            }
            let Some(dep) = dep else {
                continue;
            };
            if dep == task_id || !selected.contains(&dep) || deps.contains(&dep) {
                continue;
            }
            deps.push(dep);
        }
        deps
    };

    let mut kv_store_by_layer: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
    for task_id in &selected {
        let Some(task) = tg.tasks.get(*task_id as usize) else {
            continue;
        };
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            continue;
        };
        if matches!(
            fusion.kind,
            FusionKind::KvCacheStoreDecode
                | FusionKind::RopeKvCacheStoreDecode
                | FusionKind::RmsNormRopeKvCacheStoreDecode
        ) {
            if let Some(layer) = layer_hint(*task_id) {
                kv_store_by_layer.entry(layer).or_default().push(*task_id);
            }
        }
    }

    let mut done: std::collections::BTreeSet<u32> = Default::default();
    let mut out = Vec::with_capacity(ordered.len());

    while !ordered.is_empty() {
        let Some(pos) = ordered.iter().position(|task_id| {
            let Some(task) = tg.tasks.get(*task_id as usize) else {
                return true;
            };
            let event_deps_ready = task
                .deps
                .iter()
                .all(|dep| !selected.contains(dep) || done.contains(dep));
            let data_deps_ready = data_deps(*task_id)
                .into_iter()
                .all(|dep| done.contains(&dep));
            let deps_ready = event_deps_ready && data_deps_ready;
            if !deps_ready {
                return false;
            }
            let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
                return true;
            };
            if !matches!(fusion.kind, FusionKind::SdpaDecode) {
                return true;
            }
            let Some(layer) = layer_hint(*task_id) else {
                return true;
            };
            kv_store_by_layer.get(&layer).is_none_or(|stores| {
                stores
                    .iter()
                    .all(|dep| !selected.contains(dep) || done.contains(dep))
            })
        }) else {
            // Should not happen for a DAG, but keep emission robust:
            // preserve the existing priority order for any cycle or
            // malformed task reference rather than dropping tasks.
            out.extend(ordered);
            return out;
        };
        let task_id = ordered.remove(pos);
        done.insert(task_id);
        out.push(task_id);
    }

    out
}

/// Assign each task to a layer group by majority-vote over the
/// `LayerTag` of every op the underlying fusion absorbed. Tasks whose
/// fusion has no tagged ops at all land in the `"misc"` group — the
/// runtime should bundle these with the global pre/postamble or run
/// them under mlx's fallback path.
fn build_layer_groups(
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
    scratch_id_by_tensor: &std::collections::HashMap<TensorId, u32>,
) -> Vec<JsonLayerGroup> {
    use crate::graph::LayerTag;
    use std::collections::BTreeMap;
    let pruned_trace_deps = semantic_sdpa_trace_dependency_closure(fg, tg);
    let orphan_materialization_views = orphan_materialization_view_tasks(fg, tg);
    let is_aliased_noop_view = |fusion: &crate::fusion::ir::Fusion| -> bool {
        use crate::fusion::ir::FusionKind;
        let FusionKind::Single(name) = &fusion.kind else {
            return false;
        };
        if !matches!(
            name.as_str(),
            "Reshape"
                | "Flatten"
                | "Unflatten"
                | "Squeeze"
                | "ExpandDims"
                | "AsContiguous"
                | "Transpose"
        ) || fusion.inputs.len() != 1
            || fusion.outputs.len() != 1
        {
            return false;
        }
        scratch_id_by_tensor
            .get(&fusion.inputs[0].source)
            .zip(scratch_id_by_tensor.get(&fusion.outputs[0].source))
            .is_some_and(|(a, b)| a == b)
    };
    let region_name = |tag: Option<LayerTag>| -> String {
        match tag {
            Some(LayerTag::Embed) => "embed".to_string(),
            Some(LayerTag::Layer(n)) => format!("layer.{}", n),
            Some(LayerTag::Post) => "post".to_string(),
            None => "misc".to_string(),
        }
    };

    // Preserve execution order: walk tasks in their existing index
    // order, append to the matching group as we see it, and remember
    // the first task id per group as the ordering key.
    let mut group_first_task: BTreeMap<String, u32> = BTreeMap::new();
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for task in &tg.tasks {
        if orphan_materialization_views.contains(&task.id) {
            continue;
        }
        if pruned_trace_deps.contains(&task.id) {
            continue;
        }
        let fusion = &fg.fusions[task.fusion as usize];
        if is_aliased_noop_view(fusion) {
            continue;
        }
        // Synthetic fusions (RoPE-decode / argmax / embed lookup)
        // have empty op_ids; their layer is set explicitly via
        // `layer_hint`. Trace-bound fusions populate op_ids and let
        // propagation decide via majority vote.
        let chosen = if let Some(hint) = fusion.layer_hint {
            Some(hint)
        } else {
            let mut tally: BTreeMap<LayerTag, usize> = BTreeMap::new();
            for &op_id in &fusion.op_ids {
                if let Some(tag) = graph.ops[op_id as usize].layer_id {
                    *tally.entry(tag).or_insert(0) += 1;
                }
            }
            tally
                .into_iter()
                .max_by_key(|(_, n)| *n)
                .map(|(tag, _)| tag)
        };
        let name = region_name(chosen);
        group_first_task.entry(name.clone()).or_insert(task.id);
        groups.entry(name).or_default().push(task.id);
    }

    // Sort groups so embed precedes layers, layers ascend by index,
    // post follows, and stragglers ("misc") trail.
    let region_key = |name: &str| -> (u8, i32) {
        if name == "embed" {
            return (0, 0);
        }
        if let Some(rest) = name.strip_prefix("layer.") {
            if let Ok(n) = rest.parse::<i32>() {
                return (1, n);
            }
        }
        if name == "post" {
            return (2, 0);
        }
        (3, 0)
    };
    let mut out: Vec<JsonLayerGroup> = groups
        .into_iter()
        .map(|(region, task_ids)| JsonLayerGroup { region, task_ids })
        .collect();
    out.sort_by_key(|g| region_key(&g.region));
    out
}

fn build_binding(
    bd: &FusionBoundary,
    scratch_map: &std::collections::HashMap<TensorId, u32>,
    graph: &Graph,
) -> JsonBinding {
    let dtype = dtype_label(bd.dtype).to_string();
    if let Some(&id) = scratch_map.get(&bd.source) {
        JsonBinding::Scratch {
            scratch_id: id,
            role_name: bd.role.to_string(),
            shape: bd.shape.clone(),
            dtype,
        }
    } else {
        let tensor = graph.tensors.get(bd.source as usize);
        let source_path = tensor
            .and_then(|t| t.param_path.clone())
            .unwrap_or_default();
        let scalar = tensor
            .and_then(|t| t.scalar_value.as_ref())
            .map(|sv| match sv {
                ScalarValue::Bool(b) => JsonScalar::Bool(*b),
                ScalarValue::Int(i) => JsonScalar::Int(*i),
                ScalarValue::Float(f) => JsonScalar::Float(*f),
            });
        JsonBinding::External {
            tensor_id: bd.source,
            scalar,
            source_path,
            role_name: bd.role.to_string(),
            shape: bd.shape.clone(),
            dtype,
        }
    }
}

fn fused_chain_quant_weight_source(fusion: &crate::fusion::ir::Fusion) -> Option<TensorId> {
    use crate::fusion::ir::{ChainOperand, FusionAttrs};
    let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
        return None;
    };
    let step = steps.first()?;
    if step.op_name != "QuantizedMatmul" || step.inputs.len() != 4 {
        return None;
    }
    let ChainOperand::External { buffer_index } = step.inputs[1] else {
        return None;
    };
    fusion.inputs.get(buffer_index as usize).map(|b| b.source)
}

fn fused_chain_swiglu_absorb_weight_sources(
    fusion: &crate::fusion::ir::Fusion,
) -> Option<(TensorId, TensorId)> {
    use crate::fusion::ir::{ChainOperand, FusionAttrs};
    let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
        return None;
    };
    if steps.len() != 6
        || fusion.outputs.len() != 1
        || steps[0].op_name != "RMSNorm"
        || steps[1].op_name != "QuantizedMatmul"
        || steps[2].op_name != "QuantizedMatmul"
        || steps[3].op_name != "Sigmoid"
        || steps[4].op_name != "Multiply"
        || steps[5].op_name != "Multiply"
    {
        return None;
    }
    let crate::graph::Attrs::Quantized {
        bits: gate_bits,
        group_size: gate_group,
    } = steps[1].op_attrs
    else {
        return None;
    };
    let crate::graph::Attrs::Quantized {
        bits: up_bits,
        group_size: up_group,
    } = steps[2].op_attrs
    else {
        return None;
    };
    if gate_bits != up_bits
        || gate_group != up_group
        || gate_group != 64
        || !matches!(gate_bits, 4 | 8)
    {
        return None;
    }
    let out_dim = *steps[1].output_shape.last()? as u32;
    let chip_rows = crate::target::current_chip().gemv_rows_per_simdgroup();
    let rows_per_sg = if chip_rows > 1 && out_dim >= chip_rows && out_dim % chip_rows == 0 {
        chip_rows
    } else {
        1
    };
    if rows_per_sg > 1 {
        return None;
    }
    let ChainOperand::External {
        buffer_index: gate_buf,
    } = steps[1].inputs.get(1).copied()?
    else {
        return None;
    };
    let ChainOperand::External {
        buffer_index: up_buf,
    } = steps[2].inputs.get(1).copied()?
    else {
        return None;
    };
    let gate = fusion.inputs.get(gate_buf as usize)?.source;
    let up = fusion.inputs.get(up_buf as usize)?.source;
    Some((gate, up))
}

fn fused_chain_reduction_fanout_weight_sources(
    fusion: &crate::fusion::ir::Fusion,
) -> Option<Vec<TensorId>> {
    use crate::fusion::category::{categorize_name, OpCategory};
    use crate::fusion::ir::{ChainOperand, FusionAttrs};
    let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
        return None;
    };
    let head = steps.first()?;
    if steps.len() < 3 || head.op_name != "RMSNorm" {
        return None;
    }
    let head_shape = &head.output_shape;
    let hidden = *head_shape.last()? as u32;
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(head_shape)
        .max(1) as u32;
    let mut weights = Vec::new();
    let mut best_rows = u32::MAX;
    let mut out_dims = Vec::new();
    for step in steps.iter().skip(1) {
        if categorize_name(&step.op_name) != OpCategory::Heavy {
            return None;
        }
        if step.op_name != "QuantizedMatmul" || !(step.inputs.len() == 4 || step.inputs.len() == 5)
        {
            return None;
        }
        match step.inputs.first() {
            Some(ChainOperand::Register { step_index: 0 }) => {}
            _ => return None,
        }
        let crate::graph::Attrs::Quantized { bits, group_size } = step.op_attrs else {
            return None;
        };
        if group_size != 64 || !matches!(bits, 4 | 8) {
            return None;
        }
        let out_dim = *step.output_shape.last()? as u32;
        out_dims.push(out_dim);
        if active_rows > 1 && crate::codegen::kernels::mtp_token_m_reduction_fanout_enabled() {
            best_rows = best_rows.min(crate::target::current_chip().qgemm_m_rows_per_simdgroup(
                active_rows,
                bits,
                hidden,
                out_dim,
            ));
        }
        let ChainOperand::External { buffer_index } = step.inputs.get(1).copied()? else {
            return None;
        };
        weights.push(fusion.inputs.get(buffer_index as usize)?.source);
    }
    let chip_rows = crate::target::current_chip().gemv_rows_per_simdgroup();
    let rows_per_sg = if chip_rows > 1
        && out_dims
            .iter()
            .all(|&d| d >= chip_rows && d % chip_rows == 0)
    {
        chip_rows
    } else {
        1
    };
    let token_m = active_rows > 1
        && crate::codegen::kernels::mtp_token_m_reduction_fanout_enabled()
        && best_rows > 1;
    if weights.len() < 2 || (!token_m && rows_per_sg > 1) {
        return None;
    }
    Some(weights)
}

fn dtype_label(d: Dtype) -> &'static str {
    match d {
        Dtype::Bool => "bool",
        Dtype::U8 => "uint8",
        Dtype::U16 => "uint16",
        Dtype::U32 => "uint32",
        Dtype::U64 => "uint64",
        Dtype::I8 => "int8",
        Dtype::I16 => "int16",
        Dtype::I32 => "int32",
        Dtype::I64 => "int64",
        Dtype::F16 => "float16",
        Dtype::F32 => "float32",
        Dtype::F64 => "float64",
        Dtype::Bf16 => "bfloat16",
        Dtype::Complex64 => "complex64",
        Dtype::Other(_) => "other",
    }
}

fn scratch_size_bytes(shape: &[i32], dtype: Dtype) -> u64 {
    let elems: i64 = shape.iter().map(|&d| d as i64).product::<i64>().max(0);
    let elem_bytes: u64 = match dtype {
        Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
        Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::Bf16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
        Dtype::Other(_) => 0,
    };
    (elems as u64).saturating_mul(elem_bytes)
}
