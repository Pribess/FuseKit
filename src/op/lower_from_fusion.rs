//! Lower an existing `fusion::FusionGraph` into the new `Module`.
//!
//! Translates each `Fusion` into an `Op` whose `OpKind` reflects the
//! semantic shape (`Primitive` for `FusionKind::Single`, `Chain` for
//! `FusionKind::FusedChain`, `Synthetic` for hand-tuned decode kernels
//! like RoPE / argmax / embed lookup that haven't been decomposed yet).
//!
//! This is the "fusion graph → IR" bridge. Once the IR refactor is far
//! enough along, downstream codegen consumes the IR directly and this
//! converter retires.

use std::collections::BTreeMap;

use crate::fusion::ir::{Fusion, FusionAttrs, FusionGraph, FusionKind};
use crate::graph::Graph;
use crate::op::module::{Module, ModuleMetaValue, Tensor};
use crate::op::op::{Attr, AttrValue, ChainStep, Op, OpKind};

pub fn lower(fg: &FusionGraph) -> Module {
    lower_with_graph(fg, None)
}

/// Variant that also takes the raw trace `Graph`, so we can use
/// `graph.root` as the canonical module output. Without it, we'd have
/// to guess outputs from "tensors with zero consumers" which
/// accidentally treats dead chain outputs as live roots — breaking
/// the DCE pass.
pub fn lower_with_graph(fg: &FusionGraph, graph: Option<&Graph>) -> Module {
    let mut module = Module::new();

    // Per-tensor metadata gets populated by scanning every fusion's
    // boundaries — the FusionGraph doesn't carry a flat Vec<Tensor>
    // like the raw `Graph` does, so we synthesize one here. (`shape`/
    // `dtype` come from FusionBoundary; `producer` reflects the IR-op
    // index assigned in the loop below.)
    let mut tensor_meta: BTreeMap<_, Tensor> = BTreeMap::new();
    for (idx, fusion) in fg.fusions.iter().enumerate() {
        for out in &fusion.outputs {
            tensor_meta.entry(out.source).or_insert(Tensor {
                id: out.source,
                shape: out.shape.clone(),
                dtype: out.dtype,
                producer: Some(idx as u32),
                strides: None,
                partitioning: None,
            });
        }
        for inp in &fusion.inputs {
            // Inputs that aren't another fusion's output — treat as
            // leaves (params / model inputs). No producer.
            tensor_meta.entry(inp.source).or_insert(Tensor {
                id: inp.source,
                shape: inp.shape.clone(),
                dtype: inp.dtype,
                producer: None,
                strides: None,
                partitioning: None,
            });
        }
    }

    module.ops = fg
        .fusions
        .iter()
        .enumerate()
        .map(|(idx, f)| lower_fusion(idx as u32, f))
        .collect();

    module.tensors = tensor_meta.into_values().collect();

    // Module outputs anchor the DCE pass's backward reachability. The
    // canonical model output is `Graph::root` (= the final logits
    // tensor in an mlx trace). When the caller passes a `Graph` we
    // use that; otherwise fall back to the synthetic-only
    // "0-consumer tensors" heuristic, which is wrong if the fusion
    // graph contains dead chains (their outputs have 0 consumers too).
    if let Some(g) = graph {
        module.outputs.push(g.root);
    } else {
        let mut consumer_count: BTreeMap<crate::graph::TensorId, usize> = BTreeMap::new();
        for f in &fg.fusions {
            for inp in &f.inputs {
                *consumer_count.entry(inp.source).or_insert(0) += 1;
            }
        }
        for f in &fg.fusions {
            for out in &f.outputs {
                if consumer_count.get(&out.source).copied().unwrap_or(0) == 0 {
                    module.outputs.push(out.source);
                }
            }
        }
    }

    // Module-level meta sourced from FusionGraph.meta (hidden_size,
    // num_attention_heads, head_dim, etc.). Passes read this without
    // routing through ModelMeta directly.
    let meta = &fg.meta;
    module
        .meta
        .insert("hidden_size".into(), ModuleMetaValue::U32(meta.hidden_size));
    module.meta.insert(
        "num_attention_heads".into(),
        ModuleMetaValue::U32(meta.num_attention_heads),
    );
    module.meta.insert(
        "num_key_value_heads".into(),
        ModuleMetaValue::U32(meta.num_key_value_heads),
    );
    module
        .meta
        .insert("head_dim".into(), ModuleMetaValue::U32(meta.head_dim));
    module.meta.insert(
        "intermediate_size".into(),
        ModuleMetaValue::U32(meta.intermediate_size),
    );
    module.meta.insert(
        "num_hidden_layers".into(),
        ModuleMetaValue::U32(meta.num_hidden_layers),
    );
    module
        .meta
        .insert("vocab_size".into(), ModuleMetaValue::U32(meta.vocab_size));

    module
}

fn lower_fusion(id: u32, f: &Fusion) -> Op {
    let kind = lower_kind(f);
    let attrs = lower_attrs(f);
    Op {
        id,
        kind,
        op_ids: f.op_ids.clone(),
        inputs: f.inputs.iter().map(|b| b.source).collect(),
        outputs: f.outputs.iter().map(|b| b.source).collect(),
        attrs,
        layer_hint: f.layer_hint,
    }
}

fn lower_kind(f: &Fusion) -> OpKind {
    match (&f.kind, &f.attrs) {
        (FusionKind::Single(op_name), FusionAttrs::Single { op_attrs, .. }) => OpKind::Primitive {
            name: op_name.clone(),
            raw_attrs: op_attrs.clone(),
        },
        (FusionKind::FusedChain, FusionAttrs::FusedChain { steps, .. }) => OpKind::Chain {
            steps: steps
                .iter()
                .map(|s| ChainStep {
                    op_name: s.op_name.clone(),
                    raw_attrs: s.op_attrs.clone(),
                    inputs: s
                        .inputs
                        .iter()
                        .map(|op| match op {
                            crate::fusion::ir::ChainOperand::External { buffer_index } => {
                                Attr::External {
                                    buffer_index: *buffer_index,
                                }
                            }
                            crate::fusion::ir::ChainOperand::Register { step_index } => {
                                Attr::Register {
                                    step_index: *step_index,
                                }
                            }
                        })
                        .collect(),
                    output_shape: s.output_shape.clone(),
                    output_dtype: s.output_dtype,
                })
                .collect(),
        },
        // Synthetic / pattern fusions — opaque blocks until passes
        // decompose them into primitives.
        (kind, _) => OpKind::Synthetic {
            name: format!("{kind:?}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::ir::{
        AttentionMask, ChainOperand as FgChainOperand, ChainStep as FgChainStep, Fusion,
        FusionAttrs, FusionBoundary, FusionGraph, FusionKind, ModelMeta,
    };
    use crate::graph::{Attrs, Dtype};

    fn mk_meta() -> ModelMeta {
        ModelMeta {
            model_type: "qwen2".into(),
            hidden_size: 896,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_kv_shared_layers: 0,
            vocab_size: 151936,
            quant_group_size: None,
            quant_bits: None,
            activation_dtype: crate::graph::Dtype::F16,
            mtp_num_hidden_layers: 0,
            num_nextn_predict_layers: 0,
            mtp: crate::fusion::ir::MtpInfo::none(),
        }
    }

    fn mk_boundary(source: u32, role: &'static str, shape: Vec<i32>) -> FusionBoundary {
        FusionBoundary {
            source,
            role,
            shape,
            dtype: Dtype::F16,
            scalar_value: None,
        }
    }

    #[test]
    fn lower_single_to_primitive() {
        let fg = FusionGraph {
            fusions: vec![Fusion {
                id: 0,
                kind: FusionKind::Single("Add".into()),
                op_ids: vec![0],
                inputs: vec![
                    mk_boundary(1, "x", vec![1, 8, 896]),
                    mk_boundary(2, "y", vec![1, 8, 896]),
                ],
                outputs: vec![mk_boundary(3, "out", vec![1, 8, 896])],
                attrs: FusionAttrs::Single {
                    op_name: "Add".into(),
                    op_attrs: Attrs::Unknown,
                },
                parallel_axes: vec![],
                layer_hint: None,
            }],
            meta: mk_meta(),
        };
        let m = lower(&fg);
        assert_eq!(m.ops.len(), 1);
        let op = &m.ops[0];
        match &op.kind {
            OpKind::Primitive { name, .. } => assert_eq!(name, "Add"),
            other => panic!("expected Primitive, got {other:?}"),
        }
        assert_eq!(op.inputs, vec![1, 2]);
        assert_eq!(op.outputs, vec![3]);
        // Tensor 3 has no consumer → module output.
        assert!(m.outputs.contains(&3));
        // Meta carries hidden_size etc.
        assert!(matches!(
            m.meta.get("hidden_size"),
            Some(ModuleMetaValue::U32(896))
        ));
    }

    #[test]
    fn lower_fused_chain_preserves_steps() {
        let fg = FusionGraph {
            fusions: vec![Fusion {
                id: 0,
                kind: FusionKind::FusedChain,
                op_ids: vec![0, 1],
                inputs: vec![
                    mk_boundary(1, "x", vec![896]),
                    mk_boundary(2, "w", vec![896, 112]),
                ],
                outputs: vec![mk_boundary(3, "out", vec![896])],
                attrs: FusionAttrs::FusedChain {
                    steps: vec![
                        FgChainStep {
                            op_name: "QuantizedMatmul".into(),
                            op_attrs: Attrs::Unknown,
                            inputs: vec![
                                FgChainOperand::External { buffer_index: 0 },
                                FgChainOperand::External { buffer_index: 1 },
                            ],
                            output_shape: vec![896],
                            output_dtype: Dtype::F16,
                        },
                        FgChainStep {
                            op_name: "Add".into(),
                            op_attrs: Attrs::Unknown,
                            inputs: vec![
                                FgChainOperand::Register { step_index: 0 },
                                FgChainOperand::External { buffer_index: 0 },
                            ],
                            output_shape: vec![896],
                            output_dtype: Dtype::F16,
                        },
                    ],
                    transposed_weight_slots: Vec::new(),
                },
                parallel_axes: vec![],
                layer_hint: None,
            }],
            meta: mk_meta(),
        };
        let m = lower(&fg);
        let chain = match &m.ops[0].kind {
            OpKind::Chain { steps } => steps,
            other => panic!("expected Chain, got {other:?}"),
        };
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].op_name, "QuantizedMatmul");
        assert_eq!(chain[1].op_name, "Add");
        assert!(matches!(
            chain[1].inputs[0],
            Attr::Register { step_index: 0 }
        ));
    }

    #[test]
    fn lower_sdpa_carries_scale_attr() {
        let fg = FusionGraph {
            fusions: vec![Fusion {
                id: 0,
                kind: FusionKind::SdpaDecode,
                op_ids: vec![0, 1, 2, 3],
                inputs: vec![
                    mk_boundary(1, "Q", vec![1, 14, 1, 64]),
                    mk_boundary(2, "Kt", vec![1, 14, 64, 8]),
                    mk_boundary(3, "V", vec![1, 14, 8, 64]),
                ],
                outputs: vec![mk_boundary(4, "out", vec![1, 14, 1, 64])],
                attrs: FusionAttrs::SdpaDecode {
                    scale: 0.125,
                    trace_q: false,
                    q_split: None,
                    q_transpose_split: false,
                    q1_rope: None,
                    mask: AttentionMask::Causal,
                },
                parallel_axes: vec![],
                layer_hint: None,
            }],
            meta: mk_meta(),
        };
        let m = lower(&fg);
        let op = &m.ops[0];
        match &op.kind {
            OpKind::Synthetic { name } => assert!(name.contains("SdpaDecode")),
            other => panic!("expected Synthetic for SDPA, got {other:?}"),
        }
        assert!(matches!(
            op.attrs.get("scale"),
            Some(AttrValue::F32(s)) if (*s - 0.125).abs() < 1e-9
        ));
    }
}

fn lower_attrs(f: &Fusion) -> BTreeMap<String, AttrValue> {
    let mut attrs = BTreeMap::new();
    match &f.attrs {
        FusionAttrs::SdpaDecode {
            scale,
            trace_q,
            q_split,
            q_transpose_split,
            q1_rope,
            mask,
        } => {
            attrs.insert("scale".into(), AttrValue::F32(*scale));
            attrs.insert("trace_q".into(), AttrValue::Bool(*trace_q));
            if let Some(q_split) = q_split {
                attrs.insert("q_split".into(), AttrValue::U32(*q_split));
            }
            attrs.insert(
                "q_transpose_split".into(),
                AttrValue::Bool(*q_transpose_split),
            );
            if let Some(info) = q1_rope {
                attrs.insert("q1_rope_dims".into(), AttrValue::U32(info.dims));
            }
            attrs.insert("mask".into(), AttrValue::String(format!("{mask:?}")));
        }
        FusionAttrs::RopeQDecode { theta } | FusionAttrs::RopeKvCacheStoreDecode { theta } => {
            attrs.insert("theta".into(), AttrValue::F32(*theta));
        }
        FusionAttrs::RmsNormRopeQGateDecode {
            eps,
            theta,
            rope_dims,
        }
        | FusionAttrs::RmsNormRopeKvCacheStoreDecode {
            eps,
            theta,
            rope_dims,
        } => {
            attrs.insert("eps".into(), AttrValue::F32(*eps));
            attrs.insert("theta".into(), AttrValue::F32(*theta));
            attrs.insert("rope_dims".into(), AttrValue::U32(*rope_dims));
        }
        FusionAttrs::LinearResidual { quant } => {
            if let Some(q) = quant {
                attrs.insert("quant_bits".into(), AttrValue::U32(q.bits));
                attrs.insert("quant_group_size".into(), AttrValue::U32(q.group_size));
            }
        }
        FusionAttrs::LayerBlock {
            layer_idx,
            eps,
            theta,
            quant,
        } => {
            attrs.insert("layer_idx".into(), AttrValue::I32(*layer_idx));
            attrs.insert("eps".into(), AttrValue::F32(*eps));
            attrs.insert("theta".into(), AttrValue::F32(*theta));
            if let Some(q) = quant {
                attrs.insert("quant_bits".into(), AttrValue::U32(q.bits));
                attrs.insert("quant_group_size".into(), AttrValue::U32(q.group_size));
            }
        }
        _ => {}
    }
    attrs
}
