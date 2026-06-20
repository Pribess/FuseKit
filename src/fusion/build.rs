#![allow(dead_code)]

//! Build the Stage 2 `FusionGraph` from a Stage 1 `Graph`.
//!
//! ```text
//!   Graph (Vec<Op>)
//!      │
//!      │ build_fusions(graph, meta)
//!      ▼
//!   1. Build UseDef index
//!   2. Walk ops in topo order
//!   3. For each unconsumed op, try patterns by priority
//!   4. Pattern hit → Safety check → Accept/Split/Reject
//!   5. Unmatched op → Fusion::Single(op.name)
//!      │
//!      ▼
//!   FusionGraph (Vec<Fusion>) + DecisionLog
//! ```
//!
//! Phase 1 has no patterns registered, so every op falls through to
//! `Single`. The pipeline plumbing is still useful because downstream
//! stages can already consume `FusionGraph`.

use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::fusion::ir::{
    AttentionMask, Fusion, FusionAttrs, FusionBoundary, FusionGraph, FusionKind, ModelMeta,
};
use crate::fusion::patterns::{default_registry, UseDef};
use crate::fusion::safety::{self, SafetyCheck};
use crate::graph::{Graph, OpId};

pub fn build_fusions(g: &Graph, meta: ModelMeta) -> Result<FusionGraph> {
    let prefill_tiled_qmm = std::env::var("FUSEKIT_PREFILL_TILED_QMM").as_deref() == Ok("1");
    let ud = UseDef::build(g);
    let registry = default_registry();
    let mut fusions = Vec::with_capacity(g.ops.len());
    let mut consumed: HashSet<OpId> = HashSet::new();

    for op in &g.ops {
        if consumed.contains(&op.id) {
            continue;
        }

        // Try registered patterns, highest priority first. The first
        // match whose safety check accepts wins; lower-priority
        // patterns get a chance when a higher-priority match is
        // rejected (e.g. mlp_block over-budget → fall back to gateup_sw).
        let matched = registry.iter().find_map(|p| {
            if prefill_tiled_qmm && p.name() == "rmsnorm_qkv" {
                return None;
            }
            let m = p.try_match(g, &ud, op.id)?;
            if p.name() == "rmsnorm_qkv" && qkv_projection_outputs_feed_post_norm(g, &ud, &m) {
                return None;
            }
            match safety::check(&m, &meta) {
                SafetyCheck::Accept => Some(m),
                SafetyCheck::Reject(_) | SafetyCheck::SplitTasks(_) => None,
            }
        });

        if let Some(m) = matched {
            // Safety check is a no-op stub for now; once kernels exist
            // we'll route through SafetyCheck::SplitTasks here.
            for &id in &m.op_ids {
                consumed.insert(id);
            }
            // Retract any Single fusions emitted earlier for op_ids
            // this pattern now claims. SDPA at the Softmax anchor
            // claims [Matmul(Q·Kᵀ), Select(mask), Softmax, Matmul(·V)] —
            // the two Matmuls and the Select were visited (and
            // Single-fused) before the loop reached Softmax. Without
            // this retract the FusionGraph carries both the SDPA
            // fusion AND duplicate Singles for the same primitive
            // ops; general fusion then happily merges those Singles
            // with neighbouring Broadcast/RoPE Singles into chains
            // (`Matmul → Select`, `Multiply → Unflatten`, …) that read
            // uninitialised trace scratches and dispatch phantom work
            // at decode time. Mask-broadcasts left orphaned this way
            // also stay in the graph until task-graph DCE; the retract
            // keeps the graph clean instead.
            let claimed: std::collections::HashSet<OpId> = m.op_ids.iter().copied().collect();
            fusions.retain(|f: &Fusion| match &f.kind {
                FusionKind::Single(_) => {
                    // Single fusions hold exactly one op_id.
                    !claimed.contains(&f.op_ids[0])
                }
                _ => true,
            });
            let layer_hint = op_ids_layer_hint(g, &m.op_ids);
            fusions.push(Fusion {
                id: fusions.len() as u32,
                kind: m.kind,
                op_ids: m.op_ids,
                inputs: m.inputs,
                outputs: m.outputs,
                attrs: m.attrs,
                parallel_axes: m.parallel_axes,
                layer_hint,
            });
            continue;
        }

        // Fallback: wrap this single op as its own fusion. Codegen
        // will dispatch the op via a per-op MSL template (or, later,
        // by delegating back to MLX's kernel for that primitive).
        // Boundaries mirror the op's real tensor I/O so the Stage 3
        // dataflow analysis sees correct dependencies.
        let inputs = op
            .inputs
            .iter()
            .map(|&tid| boundary(g, tid, "in"))
            .collect();
        let outputs = op
            .outputs
            .iter()
            .map(|&tid| boundary(g, tid, "out"))
            .collect();
        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::Single(op.name.clone()),
            op_ids: vec![op.id],
            inputs,
            outputs,
            attrs: FusionAttrs::Single {
                op_name: op.name.clone(),
                op_attrs: op.attrs.clone(),
            },
            parallel_axes: vec![],
            layer_hint: op.layer_id,
        });
        consumed.insert(op.id);
    }

    inline_rope_constant_freqs(g, &mut fusions);

    // Direction A: synthetic decode-utility kernel injection is now
    // *conditional on the trace's RoPE op matching the synthetic's
    // assumptions* (full-dim rotation, no per-layer freqs override).
    // Per-layer decision:
    //   - if the layer's `Single(RoPE)` ops match standard MHA/GQA
    //     shape AND `dims == head_dim` AND no custom `freqs` → inject
    //     synthetic Q/KV-store decode kernels using the trace's `base`
    //     (no longer hardcoded to Qwen 2.5's 1M).
    //   - else (partial RoPE like DeepSeek MLA, or freqs-driven rope
    //     like Llama 3 / YARN) → leave the `Single(RoPE)` ops in the
    //     fusion graph so codegen emits them via the `single::emit_rope`
    //     path (which honours all `mx.fast.rope` attrs).
    //
    // Same story for `Single(Quantize)` against the embedding — kept
    // unconditional for now since EmbedLookupDecode is shape-agnostic.

    // Decide synthetic RoPE eligibility per layer, not per model. Gemma4
    // mixes sliding attention (`dims == head_dim`) with full attention
    // partial RoPE (`dims < head_dim`). A single global flag would remove
    // the partial RoPE producer and leave downstream Q as an anonymous
    // external trace tensor.
    let mut rope_by_layer: HashMap<i32, (f32, bool)> = HashMap::new();
    let mut trace_rope_layers: HashSet<i32> = HashSet::new();
    let producer_op = |tid: crate::graph::TensorId| -> Option<&crate::graph::Op> {
        g.tensors
            .get(tid as usize)?
            .producer
            .map(|id| &g.ops[id as usize])
    };
    let reaches_rmsnorm_through_views = |tid: crate::graph::TensorId| -> bool {
        let mut stack = vec![(tid, 0u8)];
        let mut seen = HashSet::new();
        while let Some((cur, depth)) = stack.pop() {
            if !seen.insert(cur) || depth > 8 {
                continue;
            }
            let Some(op) = producer_op(cur) else {
                continue;
            };
            if op.name == "RMSNorm" {
                return true;
            }
            if !matches!(
                op.name.as_str(),
                "Broadcast"
                    | "ExpandDims"
                    | "Squeeze"
                    | "Transpose"
                    | "AsContiguous"
                    | "Flatten"
                    | "Unflatten"
                    | "Reshape"
            ) {
                continue;
            }
            for &input in &op.inputs {
                stack.push((input, depth + 1));
            }
        }
        false
    };
    for fusion in &fusions {
        let (FusionKind::Single(name), FusionAttrs::Single { op_attrs, .. }) =
            (&fusion.kind, &fusion.attrs)
        else {
            continue;
        };
        if name != "RoPE" {
            continue;
        }
        let Some(layer) = fusion_layer_index(g, fusion) else {
            continue;
        };
        let Some(input_head_dim) = fusion.inputs.first().and_then(|b| b.shape.last()).copied()
        else {
            continue;
        };
        if fusion
            .inputs
            .first()
            .is_some_and(|input| reaches_rmsnorm_through_views(input.source))
        {
            trace_rope_layers.insert(layer);
        }
        let (base, compatible) = match op_attrs {
            crate::graph::Attrs::Rope {
                base,
                dims,
                traditional,
                scale,
                ..
            } => (
                *base,
                *dims == input_head_dim
                    && fusion.inputs.len() <= 2
                    && !*traditional
                    && (*scale - 1.0_f32).abs() < 1e-6,
            ),
            _ => (1_000_000.0_f32, false),
        };
        rope_by_layer
            .entry(layer)
            .and_modify(|(prev_base, all_compatible)| {
                *all_compatible = *all_compatible && compatible && (*prev_base - base).abs() < 1e-3;
            })
            .or_insert((base, compatible));
    }
    let no_rope_ops = rope_by_layer.is_empty();
    let synthetic_rope_base_for_layer = |layer: i32| -> Option<f32> {
        if no_rope_ops {
            return Some(1_000_000.0_f32);
        }
        if trace_rope_layers.contains(&layer) {
            return None;
        }
        rope_by_layer
            .get(&layer)
            .and_then(|(base, compatible)| compatible.then_some(*base))
    };
    let rope_fusion_is_synthetic_covered = |fusion: &Fusion| -> bool {
        let Some(layer) = fusion_layer_index(g, fusion) else {
            return false;
        };
        synthetic_rope_base_for_layer(layer).is_some()
    };

    // Retract Single(Quantize) for embed unconditionally; Single(RoPE)
    // only when the synthetic will supersede it. Prefill-tiled QMM
    // still uses the same semantic RoPE/cache-store primitives; only
    // Q/K/V projection lowering differs.
    fusions.retain(|f: &Fusion| match &f.kind {
        FusionKind::Single(name) => match name.as_str() {
            "Quantize" => false,
            "RoPE" => !rope_fusion_is_synthetic_covered(f),
            _ => true,
        },
        _ => true,
    });

    // Synthetic decode utilities are runtime-bound by semantic slots
    // (`layer_scratch.q_buf`, live KV cache), but we still attach the
    // Q/K/V producer tensor ids as Fusion inputs when a layer has a
    // standard semantic QKV producer. That keeps the Stage-2 dataflow
    // graph honest: later general passes can prove that a QKV decode
    // sink is absorbing the actual consumers, not merely "something
    // with the same layer index".
    let qkv_inputs_by_layer = qkv_decode_inputs_by_layer(g, &fusions);

    // Append synthetic decode-utility fusions only when the trace's
    // RoPE config matches synthetic assumptions. `theta` now comes from
    // the trace (`base` attr), not hardcoded — Qwen 2.5 has 1M, Llama
    // has 10K, Qwen 3.x has 1M, etc.
    {
        let n_layers = meta.num_hidden_layers as i32;
        for layer in 0..n_layers {
            let Some(theta) = synthetic_rope_base_for_layer(layer) else {
                continue;
            };
            let semantic_inputs = qkv_inputs_by_layer.get(&layer);
            fusions.push(Fusion {
                id: fusions.len() as u32,
                kind: FusionKind::RopeQDecode,
                op_ids: vec![],
                inputs: semantic_inputs
                    .map(|(q, _, _)| vec![q.clone()])
                    .unwrap_or_default(),
                outputs: vec![],
                attrs: FusionAttrs::RopeQDecode { theta },
                parallel_axes: vec![],
                layer_hint: Some(crate::graph::LayerTag::Layer(layer)),
            });
            if let Some(kv_inputs) = semantic_inputs.and_then(|(_, k, v)| {
                let k_layer = tensor_semantic_layer(g, k.source);
                let v_layer = tensor_semantic_layer(g, v.source);
                (k_layer == Some(layer) && v_layer == Some(layer))
                    .then(|| vec![k.clone(), v.clone()])
            }) {
                fusions.push(Fusion {
                    id: fusions.len() as u32,
                    kind: FusionKind::RopeKvCacheStoreDecode,
                    op_ids: vec![],
                    inputs: kv_inputs,
                    outputs: vec![],
                    attrs: FusionAttrs::RopeKvCacheStoreDecode { theta },
                    parallel_axes: vec![],
                    layer_hint: Some(crate::graph::LayerTag::Layer(layer)),
                });
            }
        }
        let embed_output_scale = if trace_rope_layers.is_empty() {
            infer_decode_embed_output_scale(g, &meta)
        } else {
            // When trace-local q/k RMSNorm+RoPE producers must survive,
            // the layer-0 embed-scale/RMSNorm preamble also remains live.
            // In that case the graph applies the embedding scale itself;
            // scaling in EmbedLookupDecode would multiply it twice.
            1.0
        };
        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::EmbedLookupDecode,
            op_ids: vec![],
            inputs: vec![],
            outputs: vec![],
            attrs: FusionAttrs::EmbedLookupDecode {
                output_scale: embed_output_scale,
            },
            parallel_axes: vec![],
            layer_hint: Some(crate::graph::LayerTag::Embed),
        });
        // Split large-vocab argmax into an N-TG scan followed by a
        // single-TG finalize. `0` / `off` keeps the legacy single-TG
        // path for bisecting. The default is intentionally only for
        // vocab-scale logits; small heads don't amortize the second
        // kernel launch.
        let multi_tg = argmax_multi_tg(meta.vocab_size);
        if let Some(n_tgs) = multi_tg {
            fusions.push(Fusion {
                id: fusions.len() as u32,
                kind: FusionKind::ArgmaxLogitsScan,
                op_ids: vec![],
                inputs: vec![],
                outputs: vec![],
                attrs: FusionAttrs::ArgmaxLogitsScan { n_tgs },
                parallel_axes: vec![],
                layer_hint: Some(crate::graph::LayerTag::Post),
            });
            fusions.push(Fusion {
                id: fusions.len() as u32,
                kind: FusionKind::ArgmaxLogitsFinalize,
                op_ids: vec![],
                inputs: vec![],
                outputs: vec![],
                attrs: FusionAttrs::ArgmaxLogitsFinalize { n_tgs },
                parallel_axes: vec![],
                layer_hint: Some(crate::graph::LayerTag::Post),
            });
        } else {
            fusions.push(Fusion {
                id: fusions.len() as u32,
                kind: FusionKind::ArgmaxLogitsDecode,
                op_ids: vec![],
                inputs: vec![],
                outputs: vec![],
                attrs: FusionAttrs::ArgmaxLogitsDecode,
                parallel_axes: vec![],
                layer_hint: Some(crate::graph::LayerTag::Post),
            });
        }
    }
    append_qwen3_next_attention_decode(g, &mut fusions);
    append_prepared_kv_cache_store_decode(g, &mut fusions);
    append_arrays_cache_shift_append(g, &mut fusions);
    // Renumber sequentially. The `fusions.retain` above strips Single(RoPE)
    // / Single(Quantize) but leaves remaining fusions' `id` fields holding
    // their pre-retain values. The subsequent synthetic-fusion pushes use
    // `id = fusions.len()` — which, after the retain, collides with surviving
    // pre-retract IDs. The collision lets two distinct fusions share an `id`,
    // so `task::build_tasks` (which sets `task.fusion = fusion.id`) emits
    // two tasks both pointing at the same fusion, scheduling RoPE/embed
    // dispatches twice per layer. Sequential renumber before downstream
    // stages eliminates the collision.
    for (idx, f) in fusions.iter_mut().enumerate() {
        f.id = idx as u32;
    }
    Ok(FusionGraph { fusions, meta })
}

fn qkv_projection_outputs_feed_post_norm(
    g: &Graph,
    ud: &UseDef,
    m: &crate::fusion::patterns::Match,
) -> bool {
    if !matches!(m.kind, FusionKind::RmsNormQkv) {
        return false;
    }

    fn reaches_rmsnorm_through_views(g: &Graph, ud: &UseDef, tid: crate::graph::TensorId) -> bool {
        let mut stack = vec![(tid, 0u8)];
        let mut seen = std::collections::HashSet::new();
        while let Some((cur, depth)) = stack.pop() {
            if !seen.insert(cur) || depth > 6 {
                continue;
            }
            for &op_id in ud.consumers.get(cur as usize).into_iter().flatten() {
                let op = g.op(op_id);
                if op.name == "RMSNorm" {
                    return true;
                }
                if matches!(
                    op.name.as_str(),
                    "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous"
                ) {
                    for &out in &op.outputs {
                        stack.push((out, depth + 1));
                    }
                }
            }
        }
        false
    }

    m.outputs
        .iter()
        .filter(|b| matches!(b.role, "q" | "k" | "v"))
        .any(|b| reaches_rmsnorm_through_views(g, ud, b.source))
}

fn op_ids_layer_hint(g: &Graph, op_ids: &[OpId]) -> Option<crate::graph::LayerTag> {
    let mut tally: std::collections::BTreeMap<crate::graph::LayerTag, usize> = Default::default();
    for &op_id in op_ids {
        if let Some(tag) = g.ops.get(op_id as usize).and_then(|op| op.layer_id) {
            *tally.entry(tag).or_insert(0) += 1;
        }
    }
    tally
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(tag, _)| tag)
}

fn inline_rope_constant_freqs(g: &Graph, fusions: &mut [Fusion]) {
    for fusion in fusions.iter_mut() {
        let FusionKind::Single(name) = &fusion.kind else {
            continue;
        };
        if name != "RoPE" || fusion.inputs.len() <= 2 {
            continue;
        }
        let Some(freqs) = const_f32_vector(g, fusion.inputs[2].source, 0) else {
            continue;
        };
        if freqs.is_empty() || freqs.len() > 256 {
            continue;
        }
        let FusionAttrs::Single { op_attrs, .. } = &mut fusion.attrs else {
            continue;
        };
        let crate::graph::Attrs::Rope { inline_freqs, .. } = op_attrs else {
            continue;
        };
        *inline_freqs = Some(freqs);
        fusion.inputs.remove(2);
    }
}

fn const_f32_vector(g: &Graph, tid: crate::graph::TensorId, depth: u8) -> Option<Vec<f32>> {
    let values = const_f64_vector(g, tid, depth)?;
    Some(values.into_iter().map(|v| v as f32).collect())
}

fn const_f64_vector(g: &Graph, tid: crate::graph::TensorId, depth: u8) -> Option<Vec<f64>> {
    if depth > 16 {
        return None;
    }
    let tensor = g.tensors.get(tid as usize)?;
    let elem_count = tensor
        .shape
        .iter()
        .copied()
        .map(|x| x.max(1) as usize)
        .product::<usize>()
        .max(1);
    if elem_count > 256 {
        return None;
    }
    if let Some(value) = tensor.scalar_value.as_ref() {
        let v = match value {
            crate::graph::ScalarValue::Bool(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
            crate::graph::ScalarValue::Int(v) => *v as f64,
            crate::graph::ScalarValue::Float(v) => *v,
        };
        return Some(vec![v]);
    }
    let op = tensor.producer.and_then(|id| g.ops.get(id as usize))?;
    match op.name.as_str() {
        "Arange" => {
            let crate::graph::Attrs::Arange { start, stop, step } = &op.attrs else {
                return None;
            };
            if *step == 0.0 {
                return None;
            }
            let mut out = Vec::with_capacity(elem_count);
            let mut cur = *start;
            for _ in 0..elem_count {
                if (*step > 0.0 && cur >= *stop) || (*step < 0.0 && cur <= *stop) {
                    break;
                }
                out.push(cur);
                cur += *step;
            }
            if out.len() == elem_count {
                Some(out)
            } else {
                None
            }
        }
        "Broadcast" => {
            let input = *op.inputs.first()?;
            let v = const_f64_vector(g, input, depth + 1)?;
            if v.len() == elem_count {
                Some(v)
            } else if v.len() == 1 {
                Some(vec![v[0]; elem_count])
            } else {
                None
            }
        }
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => {
            let input = *op.inputs.first()?;
            let v = const_f64_vector(g, input, depth + 1)?;
            if v.len() == elem_count {
                Some(v)
            } else {
                None
            }
        }
        "Add" | "Subtract" | "Multiply" | "Divide" | "Power" | "Maximum" | "Minimum" => {
            if op.inputs.len() < 2 {
                return None;
            }
            let lhs = const_f64_vector(g, op.inputs[0], depth + 1)?;
            let rhs = const_f64_vector(g, op.inputs[1], depth + 1)?;
            const_binary(&lhs, &rhs, elem_count, op.name.as_str())
        }
        "Negative" => {
            let input = *op.inputs.first()?;
            let mut v = const_f64_vector(g, input, depth + 1)?;
            if v.len() != elem_count {
                return None;
            }
            for x in &mut v {
                *x = -*x;
            }
            Some(v)
        }
        _ => None,
    }
}

fn const_binary(lhs: &[f64], rhs: &[f64], n: usize, op: &str) -> Option<Vec<f64>> {
    if !(lhs.len() == n || lhs.len() == 1) || !(rhs.len() == n || rhs.len() == 1) {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = if lhs.len() == 1 { lhs[0] } else { lhs[i] };
        let b = if rhs.len() == 1 { rhs[0] } else { rhs[i] };
        let v = match op {
            "Add" => a + b,
            "Subtract" => a - b,
            "Multiply" => a * b,
            "Divide" => a / b,
            "Power" => a.powf(b),
            "Maximum" => a.max(b),
            "Minimum" => a.min(b),
            _ => return None,
        };
        out.push(v);
    }
    Some(out)
}

fn const_scalar_f64(g: &Graph, tid: crate::graph::TensorId, depth: u8) -> Option<f64> {
    if depth > 16 {
        return None;
    }
    let tensor = g.tensors.get(tid as usize)?;
    if let Some(value) = tensor.scalar_value.as_ref() {
        return match value {
            crate::graph::ScalarValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            crate::graph::ScalarValue::Int(v) => Some(*v as f64),
            crate::graph::ScalarValue::Float(v) => Some(*v),
        };
    }
    let op = tensor.producer.and_then(|id| g.ops.get(id as usize))?;
    match op.name.as_str() {
        "Broadcast" | "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims"
        | "AsContiguous" => {
            let input = *op.inputs.first()?;
            const_scalar_f64(g, input, depth + 1)
        }
        "Add" | "Subtract" | "Multiply" | "Divide" | "Power" | "Maximum" | "Minimum" => {
            if op.inputs.len() != 2 {
                return None;
            }
            let lhs = const_scalar_f64(g, op.inputs[0], depth + 1)?;
            let rhs = const_scalar_f64(g, op.inputs[1], depth + 1)?;
            match op.name.as_str() {
                "Add" => Some(lhs + rhs),
                "Subtract" => Some(lhs - rhs),
                "Multiply" => Some(lhs * rhs),
                "Divide" => Some(lhs / rhs),
                "Power" => Some(lhs.powf(rhs)),
                "Maximum" => Some(lhs.max(rhs)),
                "Minimum" => Some(lhs.min(rhs)),
                _ => None,
            }
        }
        "Negative" => {
            let input = *op.inputs.first()?;
            Some(-const_scalar_f64(g, input, depth + 1)?)
        }
        _ => None,
    }
}

fn infer_decode_embed_output_scale(g: &Graph, meta: &ModelMeta) -> f32 {
    fn depends_on_embed_weight(
        g: &Graph,
        meta: &ModelMeta,
        tid: crate::graph::TensorId,
        depth: u8,
    ) -> bool {
        if depth > 12 {
            return false;
        }
        let Some(tensor) = g.tensors.get(tid as usize) else {
            return false;
        };
        if tensor
            .param_path
            .as_deref()
            .is_some_and(|p| p.ends_with("embed_tokens.weight"))
            && tensor.shape.len() == 2
            && tensor.shape.first().copied().unwrap_or_default() as u32 == meta.vocab_size
        {
            return true;
        }
        let Some(op) = tensor.producer.and_then(|id| g.ops.get(id as usize)) else {
            return false;
        };
        if op.name == "Gather" && op.inputs.len() >= 2 {
            return depends_on_embed_weight(g, meta, op.inputs[0], depth + 1);
        }
        if matches!(
            op.name.as_str(),
            "Squeeze"
                | "ExpandDims"
                | "Reshape"
                | "Flatten"
                | "Unflatten"
                | "AsContiguous"
                | "Quantize"
        ) {
            return op
                .inputs
                .iter()
                .copied()
                .any(|input| depends_on_embed_weight(g, meta, input, depth + 1));
        }
        false
    }

    for op in &g.ops {
        if op.name != "Multiply" || op.inputs.len() != 2 || op.outputs.len() != 1 {
            continue;
        }
        let Some(out) = g.tensors.get(op.outputs[0] as usize) else {
            continue;
        };
        if out.shape.last().copied().unwrap_or_default() as u32 != meta.hidden_size {
            continue;
        }
        for data_idx in 0..2 {
            let data_tid = op.inputs[data_idx];
            if !depends_on_embed_weight(g, meta, data_tid, 0) {
                continue;
            }
            let scale_tid = op.inputs[1 - data_idx];
            if let Some(value) = const_scalar_f64(g, scale_tid, 0) {
                let scale = value as f32;
                if scale.is_finite() && scale > 0.0 {
                    return scale;
                }
            }
        }
    }
    1.0
}

fn argmax_multi_tg(vocab_size: u32) -> Option<u32> {
    if let Ok(s) = std::env::var("FUSEKIT_ARGMAX_MULTI_TG") {
        let s = s.trim();
        if s == "0" || s.eq_ignore_ascii_case("off") {
            return None;
        }
        return s.parse::<u32>().ok().filter(|&n| (2..=256).contains(&n));
    }
    if vocab_size >= 65_536 {
        Some(80)
    } else {
        None
    }
}

fn append_arrays_cache_shift_append(g: &Graph, fusions: &mut Vec<Fusion>) {
    for f in fusions.clone() {
        let FusionKind::Single(name) = &f.kind else {
            continue;
        };
        if name != "Convolution" || f.inputs.len() < 2 || f.op_ids.is_empty() {
            continue;
        }
        let Some(crate::graph::LayerTag::Layer(layer_idx)) = g.ops[f.op_ids[0] as usize].layer_id
        else {
            continue;
        };
        let conv_in = f.inputs[0].source;
        let Some(prod_id) = g.tensors[conv_in as usize].producer else {
            continue;
        };
        let prod = &g.ops[prod_id as usize];
        if prod.name != "Concatenate" || prod.inputs.len() < 2 {
            continue;
        }
        let state = &g.tensors[prod.inputs[0] as usize];
        let current_tid = prod.inputs[1];
        let current = &g.tensors[current_tid as usize];
        if state.shape.len() != 3 || current.shape.is_empty() {
            continue;
        }
        let state_len = state.shape[1].max(1) as u32;
        let width = *current.shape.last().unwrap_or(&0);
        if width <= 0 || state.shape.last().copied() != Some(width) {
            continue;
        }
        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::ArraysCacheShiftAppendDecode,
            op_ids: vec![],
            inputs: vec![boundary(g, current_tid, "current")],
            outputs: vec![],
            attrs: FusionAttrs::ArraysCacheShiftAppendDecode {
                cache_index: 0,
                state_len,
                width: width as u32,
            },
            parallel_axes: vec![],
            layer_hint: Some(crate::graph::LayerTag::Layer(layer_idx)),
        });
    }
}

fn append_prepared_kv_cache_store_decode(g: &Graph, fusions: &mut Vec<Fusion>) {
    let producer = |tid: crate::graph::TensorId| -> Option<&crate::graph::Op> {
        g.tensors
            .get(tid as usize)?
            .producer
            .map(|id| &g.ops[id as usize])
    };
    let strip_prepared_kv_views =
        |tid: crate::graph::TensorId, last_dim: i32| -> crate::graph::TensorId {
            let mut cur = tid;
            let mut seen = std::collections::HashSet::new();
            while seen.insert(cur) {
                let Some(op) = producer(cur) else {
                    break;
                };
                if !matches!(
                    op.name.as_str(),
                    "Broadcast"
                        | "ExpandDims"
                        | "Squeeze"
                        | "Transpose"
                        | "AsContiguous"
                        | "Flatten"
                        | "Unflatten"
                        | "Reshape"
                ) || op.inputs.is_empty()
                {
                    break;
                }
                let next = op
                    .inputs
                    .iter()
                    .copied()
                    .find(|&input| {
                        g.tensors
                            .get(input as usize)
                            .and_then(|t| t.shape.last().copied())
                            == Some(last_dim)
                    })
                    .unwrap_or(op.inputs[0]);
                cur = next;
            }
            cur
        };
    let has_rope_ancestor = |tid: crate::graph::TensorId| -> bool {
        let mut cur = tid;
        let mut seen = std::collections::HashSet::new();
        while seen.insert(cur) {
            let Some(op) = producer(cur) else {
                return false;
            };
            if op.name == "RoPE" {
                return true;
            }
            if !matches!(
                op.name.as_str(),
                "Broadcast"
                    | "ExpandDims"
                    | "Squeeze"
                    | "Transpose"
                    | "AsContiguous"
                    | "Flatten"
                    | "Unflatten"
                    | "Reshape"
            ) || op.inputs.is_empty()
            {
                return false;
            }
            cur = op.inputs[0];
        }
        false
    };
    let has_rmsnorm_ancestor = |tid: crate::graph::TensorId| -> bool {
        let mut stack = vec![(tid, 0u8)];
        let mut seen = std::collections::HashSet::new();
        while let Some((cur, depth)) = stack.pop() {
            if !seen.insert(cur) || depth > 10 {
                continue;
            }
            let Some(op) = producer(cur) else {
                continue;
            };
            if op.name == "RMSNorm" {
                return true;
            }
            if !matches!(
                op.name.as_str(),
                "Broadcast"
                    | "ExpandDims"
                    | "Squeeze"
                    | "Transpose"
                    | "AsContiguous"
                    | "Flatten"
                    | "Unflatten"
                    | "Reshape"
                    | "RoPE"
            ) {
                continue;
            }
            for &input in &op.inputs {
                stack.push((input, depth + 1));
            }
        }
        false
    };
    let existing_kv_store_layers: std::collections::HashSet<i32> = fusions
        .iter()
        .filter(|fusion| {
            matches!(
                fusion.kind,
                FusionKind::RopeKvCacheStoreDecode
                    | FusionKind::RmsNormRopeKvCacheStoreDecode
                    | FusionKind::KvCacheStoreDecode
            )
        })
        .filter_map(|fusion| fusion_layer_index(g, fusion))
        .collect();

    for sdpa in fusions.clone() {
        let is_decode_sdpa = matches!(sdpa.kind, FusionKind::SdpaDecode);
        let is_direct_sdpa =
            matches!(&sdpa.kind, FusionKind::Single(name) if name == "ScaledDotProductAttention");
        if (!is_decode_sdpa && !is_direct_sdpa) || sdpa.inputs.len() < 3 {
            continue;
        }
        let layer_idx = sdpa
            .layer_hint
            .and_then(|tag| match tag {
                crate::graph::LayerTag::Layer(n) => Some(n),
                _ => None,
            })
            .or_else(|| {
                sdpa.op_ids
                    .iter()
                    .find_map(|&op_id| match g.ops[op_id as usize].layer_id {
                        Some(crate::graph::LayerTag::Layer(n)) => Some(n),
                        _ => None,
                    })
            });
        let Some(layer_idx) = layer_idx else {
            continue;
        };
        if is_direct_sdpa && existing_kv_store_layers.contains(&layer_idx) {
            continue;
        }
        let direct_normed_kv = is_direct_sdpa && has_rmsnorm_ancestor(sdpa.inputs[1].source);
        if is_direct_sdpa && !direct_normed_kv {
            continue;
        }

        let Some(qk_dim) = sdpa.inputs[0].shape.last().copied() else {
            continue;
        };
        let Some(v_dim) = sdpa.inputs[2].shape.last().copied() else {
            continue;
        };
        let scale = match sdpa.attrs {
            FusionAttrs::SdpaDecode { scale, .. } => scale,
            FusionAttrs::Single {
                op_attrs: crate::graph::Attrs::ScaledDotProductAttention { scale },
                ..
            } => scale,
            _ => (qk_dim as f32).powf(-0.5),
        };
        let k_current = strip_prepared_kv_views(sdpa.inputs[1].source, qk_dim);
        let v_current = if direct_normed_kv {
            sdpa.inputs[2].source
        } else {
            strip_prepared_kv_views(sdpa.inputs[2].source, v_dim)
        };
        let k_shape = &g.tensors[k_current as usize].shape;
        let Some(k_dim) = k_shape.last().copied() else {
            continue;
        };
        if k_dim != qk_dim {
            continue;
        }
        // Standard full-dim RoPE MHA/GQA is covered by the synthetic
        // RopeKvCacheStoreDecode path. Prepared-store is for trace-prepared
        // K/V: asymmetric MLA, or partial/freqs RoPE where Single(RoPE)
        // intentionally survives and has already produced the exact K row.
        if qk_dim == v_dim && !has_rope_ancestor(k_current) {
            continue;
        }

        if direct_normed_kv {
            if let Some(existing) = fusions.iter_mut().find(|fusion| fusion.id == sdpa.id) {
                existing.kind = FusionKind::SdpaDecode;
                existing.attrs = FusionAttrs::SdpaDecode {
                    scale,
                    trace_q: true,
                    q_split: None,
                    q_transpose_split: false,
                    q1_rope: None,
                    mask: AttentionMask::Causal,
                };
            }
        }

        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::KvCacheStoreDecode,
            op_ids: vec![],
            inputs: vec![
                boundary(g, k_current, "k_current"),
                boundary(g, v_current, "v_current"),
            ],
            outputs: vec![],
            attrs: FusionAttrs::KvCacheStoreDecode {
                k1_rope: None,
                kv_transpose_split: None,
            },
            parallel_axes: vec![],
            layer_hint: Some(crate::graph::LayerTag::Layer(layer_idx)),
        });
    }
}

fn append_qwen3_next_attention_decode(g: &Graph, fusions: &mut Vec<Fusion>) {
    use crate::graph::Attrs;
    use std::collections::HashMap;

    let mut consumers: HashMap<crate::graph::TensorId, Vec<crate::graph::OpId>> = HashMap::new();
    for op in &g.ops {
        for &tid in &op.inputs {
            consumers.entry(tid).or_default().push(op.id);
        }
    }

    let producer = |tid: crate::graph::TensorId| -> Option<&crate::graph::Op> {
        g.tensors
            .get(tid as usize)?
            .producer
            .map(|id| &g.ops[id as usize])
    };
    let semantic_parent_op = |name: &str| -> bool {
        matches!(
            name,
            "Multiply"
                | "Broadcast"
                | "Reshape"
                | "Flatten"
                | "Unflatten"
                | "Transpose"
                | "ExpandDims"
                | "Squeeze"
                | "AsContiguous"
        )
    };
    let ancestor_op_named =
        |tid: crate::graph::TensorId, wanted: &str| -> Option<&crate::graph::Op> {
            let mut stack = vec![(tid, 0u8)];
            let mut seen = std::collections::HashSet::new();
            while let Some((cur, depth)) = stack.pop() {
                if !seen.insert(cur) || depth > 10 {
                    continue;
                }
                let Some(op) = producer(cur) else {
                    continue;
                };
                if op.name == wanted {
                    return Some(op);
                }
                if !semantic_parent_op(&op.name) {
                    continue;
                }
                for &input in &op.inputs {
                    stack.push((input, depth + 1));
                }
            }
            None
        };
    let strip_head_dim_views =
        |tid: crate::graph::TensorId, head_dim: i32| -> crate::graph::TensorId {
            let mut cur = tid;
            let mut seen = std::collections::HashSet::new();
            while seen.insert(cur) {
                let Some(op) = producer(cur) else {
                    break;
                };
                if !matches!(
                    op.name.as_str(),
                    "Broadcast"
                        | "ExpandDims"
                        | "Squeeze"
                        | "Transpose"
                        | "AsContiguous"
                        | "Flatten"
                        | "Unflatten"
                        | "Reshape"
                ) {
                    break;
                }
                let Some(&next) = op.inputs.iter().find(|&&input| {
                    g.tensors
                        .get(input as usize)
                        .and_then(|t| t.shape.last().copied())
                        == Some(head_dim)
                }) else {
                    break;
                };
                cur = next;
            }
            cur
        };
    let reshape_to_last =
        |tid: crate::graph::TensorId, last_dim: i32| -> Option<crate::graph::TensorId> {
            let mut stack = vec![(tid, 0u8)];
            let mut seen = std::collections::HashSet::new();
            while let Some((cur, depth)) = stack.pop() {
                if !seen.insert(cur) || depth > 4 {
                    continue;
                }
                if g.tensors[cur as usize].shape.last().copied() == Some(last_dim)
                    && g.tensors[cur as usize].shape.len() >= 2
                    && cur != tid
                {
                    return Some(cur);
                }
                for &oid in consumers.get(&cur).into_iter().flatten() {
                    let op = &g.ops[oid as usize];
                    if matches!(
                        op.name.as_str(),
                        "Reshape" | "Flatten" | "Unflatten" | "Transpose"
                    ) {
                        for &out in &op.outputs {
                            stack.push((out, depth + 1));
                        }
                    }
                }
            }
            None
        };

    for sdpa in fusions.clone() {
        let is_decode_sdpa = matches!(sdpa.kind, FusionKind::SdpaDecode);
        let is_direct_sdpa =
            matches!(&sdpa.kind, FusionKind::Single(name) if name == "ScaledDotProductAttention");
        if (!is_decode_sdpa && !is_direct_sdpa) || sdpa.inputs.len() < 3 {
            continue;
        }
        let layer_idx = sdpa
            .layer_hint
            .and_then(|tag| match tag {
                crate::graph::LayerTag::Layer(n) => Some(n),
                _ => None,
            })
            .or_else(|| {
                sdpa.op_ids
                    .iter()
                    .find_map(|&op_id| match g.ops[op_id as usize].layer_id {
                        Some(crate::graph::LayerTag::Layer(n)) => Some(n),
                        _ => None,
                    })
            });
        let Some(layer_idx) = layer_idx else {
            continue;
        };
        let Some(q_rope) = ancestor_op_named(sdpa.inputs[0].source, "RoPE") else {
            continue;
        };
        let Some(k_rope) = ancestor_op_named(sdpa.inputs[1].source, "RoPE") else {
            continue;
        };
        if q_rope.inputs.is_empty() || k_rope.inputs.is_empty() || sdpa.outputs.is_empty() {
            continue;
        }
        let (theta, rope_dims) = match (&q_rope.attrs, &k_rope.attrs) {
            (
                Attrs::Rope {
                    base: q_base,
                    dims: q_dims,
                    traditional: false,
                    ..
                },
                Attrs::Rope {
                    base: k_base,
                    dims: k_dims,
                    traditional: false,
                    ..
                },
            ) if (*q_base - *k_base).abs() < 1e-3 && q_dims == k_dims => {
                (*q_base, (*q_dims).max(0) as u32)
            }
            _ => continue,
        };

        let Some(q_transpose) = producer(q_rope.inputs[0]) else {
            continue;
        };
        let Some(q_norm) = producer(q_transpose.inputs[0]) else {
            continue;
        };
        if q_transpose.name != "Transpose" || q_norm.name != "RMSNorm" || q_norm.inputs.len() < 2 {
            continue;
        }
        let q_split_out = q_norm.inputs[0];
        let Some(q_split) = producer(q_split_out) else {
            continue;
        };
        if q_split.name != "Split" || q_split.inputs.is_empty() || q_split.outputs.len() < 2 {
            continue;
        }
        let Some(&gate_split_out) = q_split.outputs.iter().find(|&&tid| tid != q_split_out) else {
            continue;
        };
        let sdpa_out = sdpa.outputs[0].source;
        let sdpa_shape = &g.tensors[sdpa_out as usize].shape;
        if sdpa_shape.len() < 4 {
            continue;
        }
        let attn_dim = sdpa_shape[1..sdpa_shape.len() - 2]
            .iter()
            .product::<i32>()
            .max(1)
            * sdpa_shape.last().copied().unwrap_or(1).max(1);
        let Some(gate_flat) = reshape_to_last(gate_split_out, attn_dim) else {
            continue;
        };
        let Some(sdpa_flat) = reshape_to_last(sdpa_out, attn_dim) else {
            continue;
        };

        let Some(k_transpose) = producer(k_rope.inputs[0]) else {
            continue;
        };
        let Some(k_norm) = producer(k_transpose.inputs[0]) else {
            continue;
        };
        if k_transpose.name != "Transpose" || k_norm.name != "RMSNorm" || k_norm.inputs.len() < 2 {
            continue;
        }

        let head_dim = g.tensors[k_norm.inputs[0] as usize]
            .shape
            .last()
            .copied()
            .unwrap_or(1);
        let v_raw = strip_head_dim_views(sdpa.inputs[2].source, head_dim);
        let scale = match sdpa.attrs {
            FusionAttrs::SdpaDecode { scale, .. } => scale,
            FusionAttrs::Single {
                op_attrs: Attrs::ScaledDotProductAttention { scale },
                ..
            } => scale,
            _ => (g.tensors[sdpa.inputs[0].source as usize]
                .shape
                .last()
                .copied()
                .unwrap_or(1) as f32)
                .powf(-0.5),
        };
        let eps = match q_norm.attrs {
            Attrs::RmsNorm { eps } => eps,
            _ => 1e-6,
        };

        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::RmsNormRopeQGateDecode,
            op_ids: vec![],
            inputs: vec![
                boundary(g, q_split.inputs[0], "q_proj"),
                boundary(g, q_norm.inputs[1], "q_norm_weight"),
            ],
            outputs: vec![
                boundary(g, sdpa.inputs[0].source, "q_out"),
                boundary(g, gate_flat, "gate_out"),
            ],
            attrs: FusionAttrs::RmsNormRopeQGateDecode {
                eps,
                theta,
                rope_dims,
            },
            parallel_axes: vec![],
            layer_hint: Some(crate::graph::LayerTag::Layer(layer_idx)),
        });

        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::RmsNormRopeKvCacheStoreDecode,
            op_ids: vec![],
            inputs: vec![
                boundary(g, k_norm.inputs[0], "k_proj"),
                boundary(g, v_raw, "v_proj"),
                boundary(g, k_norm.inputs[1], "k_norm_weight"),
            ],
            outputs: vec![],
            attrs: FusionAttrs::RmsNormRopeKvCacheStoreDecode {
                eps,
                theta,
                rope_dims,
            },
            parallel_axes: vec![],
            layer_hint: Some(crate::graph::LayerTag::Layer(layer_idx)),
        });

        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::SdpaDecode,
            op_ids: vec![],
            inputs: vec![],
            outputs: vec![boundary(g, sdpa_flat, "out")],
            attrs: FusionAttrs::SdpaDecode {
                scale,
                trace_q: false,
                q_split: None,
                q_transpose_split: false,
                q1_rope: None,
                mask: AttentionMask::Causal,
            },
            parallel_axes: vec![],
            layer_hint: Some(crate::graph::LayerTag::Layer(layer_idx)),
        });
    }
}

fn boundary(g: &Graph, tid: crate::graph::TensorId, role: &'static str) -> FusionBoundary {
    let t = &g.tensors[tid as usize];
    FusionBoundary {
        source: tid,
        role,
        shape: t.shape.clone(),
        dtype: t.dtype,
        scalar_value: t.scalar_value.clone(),
    }
}

fn fusion_layer_index(g: &Graph, fusion: &Fusion) -> Option<i32> {
    if let Some(crate::graph::LayerTag::Layer(layer)) = fusion.layer_hint {
        return Some(layer);
    }
    fusion.op_ids.iter().find_map(|&op_id| {
        g.ops.get(op_id as usize).and_then(|op| match op.layer_id {
            Some(crate::graph::LayerTag::Layer(layer)) => Some(layer),
            _ => None,
        })
    })
}

fn tensor_semantic_layer(g: &Graph, tid: crate::graph::TensorId) -> Option<i32> {
    let mut cur = tid;
    let mut seen = std::collections::HashSet::new();
    while seen.insert(cur) {
        let tensor = g.tensors.get(cur as usize)?;
        let op_id = tensor.producer?;
        let op = g.ops.get(op_id as usize)?;
        if let Some(crate::graph::LayerTag::Layer(layer)) = op.layer_id {
            return Some(layer);
        }
        if !matches!(
            op.name.as_str(),
            "Reshape"
                | "Flatten"
                | "Unflatten"
                | "Squeeze"
                | "ExpandDims"
                | "Transpose"
                | "AsContiguous"
                | "RoPE"
                | "Slice"
        ) || op.inputs.is_empty()
        {
            return None;
        }
        cur = op.inputs[0];
    }
    None
}

fn qkv_decode_inputs_by_layer(
    g: &Graph,
    fusions: &[Fusion],
) -> HashMap<i32, (FusionBoundary, FusionBoundary, FusionBoundary)> {
    let mut out: HashMap<i32, (i32, (FusionBoundary, FusionBoundary, FusionBoundary))> =
        HashMap::new();

    let producer = |tid: crate::graph::TensorId| -> Option<&crate::graph::Op> {
        g.tensors
            .get(tid as usize)?
            .producer
            .map(|id| &g.ops[id as usize])
    };
    let strip_rope_transpose = |tid: crate::graph::TensorId| -> crate::graph::TensorId {
        let mut cur = tid;
        let mut seen = std::collections::HashSet::new();
        while seen.insert(cur) {
            let Some(op) = producer(cur) else {
                break;
            };
            if !matches!(
                op.name.as_str(),
                "RoPE"
                    | "Transpose"
                    | "AsContiguous"
                    | "Squeeze"
                    | "ExpandDims"
                    | "Broadcast"
                    | "Reshape"
                    | "Flatten"
                    | "Unflatten"
            ) || op.inputs.is_empty()
            {
                break;
            }
            cur = op.inputs[0];
        }
        cur
    };
    let tensor_score = |layer: i32, tid: crate::graph::TensorId| -> i32 {
        let mut score = 0;
        if let Some(op) = producer(tid) {
            score += 4;
            if matches!(
                op.name.as_str(),
                "RMSNorm" | "RoPE" | "Transpose" | "Reshape" | "QuantizedMatmul" | "Matmul"
            ) {
                score += 1;
            }
            if op.layer_id == Some(crate::graph::LayerTag::Layer(layer)) {
                score += 4;
            }
        }
        if tensor_semantic_layer(g, tid) == Some(layer) {
            score += 4;
        }
        score
    };
    let insert_candidate =
        |out: &mut HashMap<i32, (i32, (FusionBoundary, FusionBoundary, FusionBoundary))>,
         layer: i32,
         score: i32,
         q: FusionBoundary,
         k: FusionBoundary,
         v: FusionBoundary| {
            let replace = out
                .get(&layer)
                .is_none_or(|(existing_score, _)| score > *existing_score);
            if replace {
                out.insert(layer, (score, (q, k, v)));
            }
        };
    let debug_qkv = std::env::var("FUSEKIT_QKV_DEBUG").as_deref() == Ok("1");

    for op in &g.ops {
        if op.name != "ScaledDotProductAttention" || op.inputs.len() < 3 {
            continue;
        }
        let Some(crate::graph::LayerTag::Layer(layer)) = op.layer_id else {
            continue;
        };
        let q = strip_rope_transpose(op.inputs[0]);
        let k = strip_rope_transpose(op.inputs[1]);
        let v = strip_rope_transpose(op.inputs[2]);
        let q_shape = &g.tensors[q as usize].shape;
        let k_shape = &g.tensors[k as usize].shape;
        let v_shape = &g.tensors[v as usize].shape;
        if debug_qkv {
            eprintln!(
                "[qkv-debug] raw-sdpa op={} layer={:?} q={} {:?} q_layer={:?} k={} {:?} k_layer={:?} v={} {:?} v_layer={:?}",
                op.id,
                op.layer_id,
                q,
                q_shape,
                tensor_semantic_layer(g, q),
                k,
                k_shape,
                tensor_semantic_layer(g, k),
                v,
                v_shape,
                tensor_semantic_layer(g, v)
            );
        }
        if q_shape.len() < 3 || k_shape.len() < 3 || v_shape.len() < 3 {
            continue;
        }
        let Some(q_last) = q_shape.last().copied() else {
            continue;
        };
        let Some(k_last) = k_shape.last().copied() else {
            continue;
        };
        let Some(v_last) = v_shape.last().copied() else {
            continue;
        };
        if q_last <= 0 || k_last <= 0 || v_last <= 0 || k_last != v_last {
            continue;
        }
        let score = 32 + tensor_score(layer, q) + tensor_score(layer, k) + tensor_score(layer, v);
        insert_candidate(
            &mut out,
            layer,
            score,
            boundary(g, q, "q"),
            boundary(g, k, "k"),
            boundary(g, v, "v"),
        );
    }

    for fusion in fusions {
        let is_sdpa = matches!(fusion.kind, FusionKind::SdpaDecode)
            || matches!(&fusion.kind, FusionKind::Single(name) if name == "ScaledDotProductAttention");
        if !is_sdpa || fusion.inputs.len() < 3 {
            continue;
        }
        let Some(layer) = fusion_layer_index(g, fusion) else {
            continue;
        };
        let q = strip_rope_transpose(fusion.inputs[0].source);
        let k = strip_rope_transpose(fusion.inputs[1].source);
        let v = strip_rope_transpose(fusion.inputs[2].source);
        let q_shape = &g.tensors[q as usize].shape;
        let k_shape = &g.tensors[k as usize].shape;
        let v_shape = &g.tensors[v as usize].shape;
        if debug_qkv {
            eprintln!(
                "[qkv-debug] fusion-sdpa kind={:?} layer={:?} q={} {:?} q_layer={:?} k={} {:?} k_layer={:?} v={} {:?} v_layer={:?}",
                fusion.kind,
                fusion_layer_index(g, fusion),
                q,
                q_shape,
                tensor_semantic_layer(g, q),
                k,
                k_shape,
                tensor_semantic_layer(g, k),
                v,
                v_shape,
                tensor_semantic_layer(g, v)
            );
        }
        if q_shape.len() < 3 || k_shape.len() < 3 || v_shape.len() < 3 {
            continue;
        }
        let Some(q_last) = q_shape.last().copied() else {
            continue;
        };
        let Some(k_last) = k_shape.last().copied() else {
            continue;
        };
        let Some(v_last) = v_shape.last().copied() else {
            continue;
        };
        if q_last <= 0 || k_last <= 0 || v_last <= 0 || k_last != v_last {
            continue;
        }
        let mut score = 16;
        score += tensor_score(layer, q);
        score += tensor_score(layer, k);
        score += tensor_score(layer, v);
        if matches!(&fusion.kind, FusionKind::Single(name) if name == "ScaledDotProductAttention") {
            score += 4;
        }
        insert_candidate(
            &mut out,
            layer,
            score,
            boundary(g, q, "q"),
            boundary(g, k, "k"),
            boundary(g, v, "v"),
        );
    }

    for fusion in fusions {
        if !matches!(fusion.kind, FusionKind::RmsNormQkv) {
            continue;
        }
        let Some(layer) = fusion_layer_index(g, fusion) else {
            continue;
        };
        let Some(q) = fusion.outputs.iter().find(|b| b.role == "q").cloned() else {
            continue;
        };
        let Some(k) = fusion.outputs.iter().find(|b| b.role == "k").cloned() else {
            continue;
        };
        let Some(v) = fusion.outputs.iter().find(|b| b.role == "v").cloned() else {
            continue;
        };
        // Fallback for synthetic/test graphs without a surviving SDPA edge.
        // Real decode traces prefer the SDPA-derived semantic sources above:
        // if Q/K/V pass through post-projection norms before attention, the
        // strip walk stops at those norm outputs instead of the raw projection.
        let score = 4
            + tensor_score(layer, q.source)
            + tensor_score(layer, k.source)
            + tensor_score(layer, v.source);
        insert_candidate(&mut out, layer, score, q, k, v);
    }

    out.into_iter()
        .map(|(layer, (_, qkv))| (layer, qkv))
        .collect()
}
