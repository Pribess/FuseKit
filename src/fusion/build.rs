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
            fusions.push(Fusion {
                id: fusions.len() as u32,
                kind: m.kind,
                op_ids: m.op_ids,
                inputs: m.inputs,
                outputs: m.outputs,
                attrs: m.attrs,
                parallel_axes: m.parallel_axes,
                layer_hint: None,
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
            layer_hint: None,
        });
        consumed.insert(op.id);
    }

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

    // Detect a representative `Single(RoPE)`'s attrs to derive theta
    // and check the "standard pattern" predicate. mlx-lm models use
    // the same RoPE config across all layers so reading the first
    // surviving op is sufficient.
    let head_dim = meta.head_dim as i32;
    let (rope_base, rope_matches_synthetic) = fusions
        .iter()
        .find_map(|f| match (&f.kind, &f.attrs) {
            (FusionKind::Single(name), FusionAttrs::Single { op_attrs, .. }) if name == "RoPE" => {
                match op_attrs {
                    crate::graph::Attrs::Rope {
                        base,
                        dims,
                        traditional,
                        scale,
                    } => {
                        // Synthetic kernel assumes full-dim rotation,
                        // half-split (non-traditional) pairing, no
                        // freqs/scale override. Anything else falls back
                        // to the Single emit.
                        let matches =
                            *dims == head_dim && !*traditional && (*scale - 1.0_f32).abs() < 1e-6;
                        Some((*base, matches))
                    }
                    _ => Some((1_000_000.0_f32, false)),
                }
            }
            _ => None,
        })
        .unwrap_or((1_000_000.0_f32, true)); // No RoPE op in trace → default
                                             // (e.g. tests with stub graphs).

    // Retract Single(Quantize) for embed unconditionally; Single(RoPE)
    // only when the synthetic will supersede it. Prefill-tiled QMM
    // still uses the same semantic RoPE/cache-store primitives; only
    // Q/K/V projection lowering differs.
    fusions.retain(|f: &Fusion| match &f.kind {
        FusionKind::Single(name) => match name.as_str() {
            "Quantize" => false,
            "RoPE" => !rope_matches_synthetic,
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
        let theta = rope_base;
        let n_layers = meta.num_hidden_layers as i32;
        if rope_matches_synthetic {
            for layer in 0..n_layers {
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
                fusions.push(Fusion {
                    id: fusions.len() as u32,
                    kind: FusionKind::RopeKvCacheStoreDecode,
                    op_ids: vec![],
                    inputs: semantic_inputs
                        .map(|(_, k, v)| vec![k.clone(), v.clone()])
                        .unwrap_or_default(),
                    outputs: vec![],
                    attrs: FusionAttrs::RopeKvCacheStoreDecode { theta },
                    parallel_axes: vec![],
                    layer_hint: Some(crate::graph::LayerTag::Layer(layer)),
                });
            }
        }
        fusions.push(Fusion {
            id: fusions.len() as u32,
            kind: FusionKind::EmbedLookupDecode,
            op_ids: vec![],
            inputs: vec![],
            outputs: vec![],
            attrs: FusionAttrs::EmbedLookupDecode,
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
    let pre_transpose_input = |tid: crate::graph::TensorId| -> crate::graph::TensorId {
        match producer(tid) {
            Some(op) if op.name == "Transpose" && !op.inputs.is_empty() => op.inputs[0],
            _ => tid,
        }
    };

    for sdpa in fusions.clone() {
        if !matches!(sdpa.kind, FusionKind::SdpaDecode) || sdpa.inputs.len() < 3 {
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

        let Some(qk_dim) = sdpa.inputs[0].shape.last().copied() else {
            continue;
        };
        let Some(v_dim) = sdpa.inputs[2].shape.last().copied() else {
            continue;
        };
        // Standard MHA/GQA has qk_dim == v_dim and is covered by the
        // RoPE+KV synthetic path. This primitive is for trace-prepared
        // K/V with asymmetric SDPA dimensions, e.g. MLA qk=192, v=128.
        if qk_dim == v_dim {
            continue;
        }

        let k_current = pre_transpose_input(sdpa.inputs[1].source);
        let v_current = pre_transpose_input(sdpa.inputs[2].source);
        let k_shape = &g.tensors[k_current as usize].shape;
        let Some(k_dim) = k_shape.last().copied() else {
            continue;
        };
        if k_dim != qk_dim {
            continue;
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
            attrs: FusionAttrs::KvCacheStoreDecode,
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

fn qkv_decode_inputs_by_layer(
    g: &Graph,
    fusions: &[Fusion],
) -> HashMap<i32, (FusionBoundary, FusionBoundary, FusionBoundary)> {
    let mut out = HashMap::new();
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
        // If a layer somehow has multiple semantic QKV producers,
        // keeping the first avoids guessing. The later sink pass will
        // only absorb consumers whose input source matches.
        out.entry(layer).or_insert((q, k, v));
    }

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
                "RoPE" | "Transpose" | "AsContiguous" | "Squeeze" | "ExpandDims"
            ) || op.inputs.is_empty()
            {
                break;
            }
            cur = op.inputs[0];
        }
        cur
    };

    for fusion in fusions {
        let FusionKind::Single(name) = &fusion.kind else {
            continue;
        };
        if name != "ScaledDotProductAttention" || fusion.inputs.len() < 3 {
            continue;
        }
        let Some(layer) = fusion_layer_index(g, fusion) else {
            continue;
        };
        if out.contains_key(&layer) {
            continue;
        }
        let q = strip_rope_transpose(fusion.inputs[0].source);
        let k = strip_rope_transpose(fusion.inputs[1].source);
        let v = strip_rope_transpose(fusion.inputs[2].source);
        let q_shape = &g.tensors[q as usize].shape;
        let k_shape = &g.tensors[k as usize].shape;
        let v_shape = &g.tensors[v as usize].shape;
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
        out.insert(
            layer,
            (
                boundary(g, q, "q"),
                boundary(g, k, "k"),
                boundary(g, v, "v"),
            ),
        );
    }
    out
}
