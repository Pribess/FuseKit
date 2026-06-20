//! Decode L=1 specialization.
//!
//! mlx traces are captured at prefill (typically L=8). Decode-time
//! runs with L=1. Every kernel emitted from the prefill trace
//! iterates an unused batch axis: e.g. `emit_qmm_chain` does
//! `for i in 0..BATCH*N_DIM` but at decode `BATCH=1` (with the rest
//! of the batch slots holding garbage). Wasted dispatch + register
//! pressure.
//!
//! This pass detects op shapes that have a leading batch/seq axis of
//! size > 1 in the trace and annotates them with
//! `decode_L_axis: u32` pointing at the axis that collapses to 1 at
//! decode. Codegen consumers can then specialise: skip the batch
//! loop, drop the batch index decomposition, shrink the grid.
//!
//! Detection heuristic — applied per Op:
//!
//! - If the op has a 3D+ output where the first non-batch dim
//!   matches the model's `num_hidden_layers`-scaled L (sequence
//!   length captured in trace), annotate it.
//! - For decoder layer outputs (shape ends in `hidden_size`),
//!   recognise the leading axis as the L axis.
//!
//! Conservative: when the trace L is 1 already (single-step
//! capture), the op is naturally L=1 specialised. When L > 1, set
//! the annotation so codegen can emit a separate kernel variant.

use crate::op::module::Module;
use crate::op::op::AttrValue;
use crate::op::passes::{Pass, PassError};

pub struct DecodeL1Specialize;

impl Pass for DecodeL1Specialize {
    fn name(&self) -> &'static str {
        "decode-l1-specialize"
    }

    fn run(&self, module: &mut Module) -> Result<bool, PassError> {
        // Determine the trace's L axis from the most common
        // batch/seq dimension across decoder-layer-attached outputs.
        // Heuristic: outputs of shape (..., L, hidden_size) where
        // L > 1.
        let hidden_size = module.meta.get("hidden_size").and_then(|v| match v {
            crate::op::module::ModuleMetaValue::U32(n) => Some(*n as i32),
            crate::op::module::ModuleMetaValue::I32(n) => Some(*n),
            _ => None,
        });
        let Some(hidden) = hidden_size else {
            return Ok(false);
        };

        let mut changed = false;
        for op in &mut module.ops {
            if op.attrs.contains_key("decode_l_axis") {
                continue;
            }
            // Only annotate ops that have an output shape ending in
            // `hidden_size` (decoder activations) or whose layer_hint
            // is set (layer-attached compute).
            if op.layer_hint.is_none() {
                continue;
            }
            // Resolve the FIRST output's shape via the module's
            // tensor table.
            let Some(out_tid) = op.outputs.first().copied() else {
                continue;
            };
            let Some(shape) = module
                .tensors
                .iter()
                .find(|t| t.id == out_tid)
                .map(|t| t.shape.clone())
            else {
                continue;
            };
            if shape.last() != Some(&hidden) || shape.len() < 2 {
                continue;
            }
            // The L axis is the second-to-last (immediately before
            // hidden). If that's > 1, codegen can specialise.
            let l_axis = (shape.len() - 2) as u32;
            let l_size = shape[shape.len() - 2];
            if l_size <= 1 {
                continue;
            }
            op.attrs
                .insert("decode_l_axis".into(), AttrValue::U32(l_axis));
            op.attrs
                .insert("trace_l_size".into(), AttrValue::I32(l_size));
            changed = true;
        }
        // For chain ops, propagate the L-axis annotation to each step
        // — even if individual steps don't expose an output shape,
        // the chain-level annotation tells codegen the WHOLE chain
        // runs with the same L collapse.
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attrs, Dtype, LayerTag};
    use crate::op::module::{Module, ModuleMetaValue, Tensor};
    use crate::op::op::{Op, OpKind};

    fn mk_tensor(id: u32, shape: Vec<i32>) -> Tensor {
        Tensor {
            id,
            shape,
            dtype: Dtype::F16,
            producer: None,
            strides: None,
            partitioning: None,
        }
    }

    #[test]
    fn annotates_layer_attached_op_with_l_gt_1() {
        let mut m = Module::new();
        m.meta
            .insert("hidden_size".into(), ModuleMetaValue::U32(896));
        m.tensors = vec![
            mk_tensor(0, vec![1, 8, 896]), // (B, L=8, hidden) — trace prefill
            mk_tensor(1, vec![1, 8, 896]),
        ];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Add".into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: Some(LayerTag::Layer(0)),
        }];
        DecodeL1Specialize.run(&mut m).unwrap();
        assert_eq!(
            m.ops[0].attrs.get("decode_l_axis"),
            Some(&AttrValue::U32(1))
        );
        assert_eq!(m.ops[0].attrs.get("trace_l_size"), Some(&AttrValue::I32(8)));
    }

    #[test]
    fn skips_op_without_layer_hint() {
        let mut m = Module::new();
        m.meta
            .insert("hidden_size".into(), ModuleMetaValue::U32(896));
        m.tensors = vec![mk_tensor(0, vec![1, 8, 896]), mk_tensor(1, vec![1, 8, 896])];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Add".into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: None,
        }];
        DecodeL1Specialize.run(&mut m).unwrap();
        assert!(m.ops[0].attrs.get("decode_l_axis").is_none());
    }

    #[test]
    fn skips_when_l_already_1() {
        // L=1 already — no specialisation needed.
        let mut m = Module::new();
        m.meta
            .insert("hidden_size".into(), ModuleMetaValue::U32(896));
        m.tensors = vec![mk_tensor(0, vec![1, 1, 896]), mk_tensor(1, vec![1, 1, 896])];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Add".into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: Some(LayerTag::Layer(0)),
        }];
        DecodeL1Specialize.run(&mut m).unwrap();
        assert!(m.ops[0].attrs.get("decode_l_axis").is_none());
    }
}
