//! In-place + alias analysis pass.
//!
//! Detects when an op's output can safely share storage with one of
//! its inputs. Conditions for in-place:
//!
//! 1. The input has exactly ONE consumer (this op). If it had more,
//!    overwriting the buffer would clobber data the other consumers
//!    need.
//! 2. Input and output have the same shape and dtype (otherwise the
//!    buffer can't be reused — different layout/size).
//! 3. The input is NOT a module input or a leaf weight (those have
//!    longer lifetimes than one op).
//!
//! When the conditions hold, the pass annotates the op's `attrs` with
//! `in_place_input_slot: u32` pointing at which input the output
//! aliases. Codegen consumers (when they eventually run on IR) bind
//! the input and output to the same buffer, saving one scratch
//! allocation per in-place op.
//!
//! On the current fusekit synthetic-kernel layout, `RopeQDecode`
//! conceptually does `q_buf = rotate(q_buf)` (in-place mutation of
//! q_buf). The hand-written kernel knows this; the pass makes that
//! knowledge explicit in the IR so when codegen migrates, in-place
//! semantics fall out of the analysis instead of being hardcoded.

use std::collections::HashMap;

use crate::graph::TensorId;
use crate::op::module::Module;
use crate::op::op::AttrValue;
use crate::op::passes::{Pass, PassError};

pub struct InPlaceAnalysis;

impl Pass for InPlaceAnalysis {
    fn name(&self) -> &'static str {
        "in-place-analysis"
    }

    fn run(&self, module: &mut Module) -> Result<bool, PassError> {
        // For every tensor, count how many ops consume it.
        let mut consumer_count: HashMap<TensorId, usize> = HashMap::new();
        for op in &module.ops {
            for &in_tid in &op.inputs {
                *consumer_count.entry(in_tid).or_insert(0) += 1;
            }
        }
        // Module outputs are "consumed" externally (by the runtime
        // reading them) — pin them as ineligible for in-place.
        let pinned: std::collections::HashSet<TensorId> = module.outputs.iter().copied().collect();

        // Snapshot tensor metadata into a side table so we can mutate
        // module.ops without aliasing the read path.
        let shape_dtype: HashMap<TensorId, (Vec<i32>, crate::graph::Dtype, bool)> = module
            .tensors
            .iter()
            .map(|t| (t.id, (t.shape.clone(), t.dtype, t.producer.is_some())))
            .collect();

        let mut changed = false;
        for op in &mut module.ops {
            if op.attrs.contains_key("in_place_input_slot") {
                continue;
            }
            let mut chosen_slot: Option<u32> = None;
            for (slot_idx, &in_tid) in op.inputs.iter().enumerate() {
                if pinned.contains(&in_tid) {
                    continue;
                }
                if consumer_count.get(&in_tid).copied().unwrap_or(0) != 1 {
                    continue;
                }
                let Some((in_shape, in_dtype, has_producer)) = shape_dtype.get(&in_tid).cloned()
                else {
                    continue;
                };
                if !has_producer {
                    continue;
                }
                let out_match = op.outputs.iter().any(|out_tid| {
                    shape_dtype
                        .get(out_tid)
                        .map(|(s, d, _)| s == &in_shape && *d == in_dtype)
                        .unwrap_or(false)
                });
                if !out_match {
                    continue;
                }
                chosen_slot = Some(slot_idx as u32);
                break;
            }
            if let Some(slot) = chosen_slot {
                op.attrs
                    .insert("in_place_input_slot".into(), AttrValue::U32(slot));
                changed = true;
            }
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attrs, Dtype};
    use crate::op::module::{Module, Tensor};
    use crate::op::op::{Op, OpKind};

    fn mk_tensor(id: u32, shape: Vec<i32>, producer: Option<u32>) -> Tensor {
        Tensor {
            id,
            shape,
            dtype: Dtype::F16,
            producer,
            strides: None,
            partitioning: None,
        }
    }

    fn primitive(id: u32, name: &str, inputs: Vec<u32>, outputs: Vec<u32>) -> Op {
        Op {
            id,
            kind: OpKind::Primitive {
                name: name.into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs,
            outputs,
            attrs: Default::default(),
            layer_hint: None,
        }
    }

    #[test]
    fn marks_in_place_when_input_single_consumer_and_same_shape() {
        // op0 produces tensor 1.
        // op1 consumes tensor 1 → produces tensor 2 (same shape).
        // Tensor 1 has exactly one consumer (op1) and matches shape.
        // → op1 in-place on slot 0.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 8, 896], None),
            mk_tensor(1, vec![1, 8, 896], Some(0)),
            mk_tensor(2, vec![1, 8, 896], Some(1)),
        ];
        m.ops = vec![
            primitive(0, "Add", vec![0], vec![1]),
            primitive(1, "Sigmoid", vec![1], vec![2]),
        ];
        m.outputs = vec![2];
        InPlaceAnalysis.run(&mut m).unwrap();
        // op0 (Add) input is a leaf (tensor 0, producer=None) → not eligible.
        // op1 (Sigmoid) input is tensor 1 (single consumer + same shape) → eligible.
        assert_eq!(
            m.ops[1].attrs.get("in_place_input_slot"),
            Some(&AttrValue::U32(0))
        );
        assert!(m.ops[0].attrs.get("in_place_input_slot").is_none());
    }

    #[test]
    fn rejects_when_input_has_multiple_consumers() {
        // op1 consumes tensor 1; op2 also consumes tensor 1.
        // → op1 NOT eligible (overwriting would clobber op2).
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 8, 896], None),
            mk_tensor(1, vec![1, 8, 896], Some(0)),
            mk_tensor(2, vec![1, 8, 896], Some(1)),
            mk_tensor(3, vec![1, 8, 896], Some(2)),
        ];
        m.ops = vec![
            primitive(0, "Add", vec![0], vec![1]),
            primitive(1, "Sigmoid", vec![1], vec![2]),
            primitive(2, "Negative", vec![1], vec![3]),
        ];
        InPlaceAnalysis.run(&mut m).unwrap();
        assert!(m.ops[1].attrs.get("in_place_input_slot").is_none());
        assert!(m.ops[2].attrs.get("in_place_input_slot").is_none());
    }

    #[test]
    fn rejects_when_shape_differs() {
        // op1 consumes tensor 1 (shape [8,896]) → produces tensor 2 (shape [896]).
        // Different shape → not in-place safe.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![8, 896], None),
            mk_tensor(1, vec![8, 896], Some(0)),
            mk_tensor(2, vec![896], Some(1)),
        ];
        m.ops = vec![
            primitive(0, "Add", vec![0], vec![1]),
            primitive(1, "Reduce", vec![1], vec![2]),
        ];
        InPlaceAnalysis.run(&mut m).unwrap();
        assert!(m.ops[1].attrs.get("in_place_input_slot").is_none());
    }

    #[test]
    fn rejects_module_output() {
        // tensor 1 is a module output → can't alias.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 896], None),
            mk_tensor(1, vec![1, 896], Some(0)),
            mk_tensor(2, vec![1, 896], Some(1)),
        ];
        m.ops = vec![
            primitive(0, "Add", vec![0], vec![1]),
            primitive(1, "Sigmoid", vec![1], vec![2]),
        ];
        m.outputs = vec![1, 2]; // pin tensor 1
        InPlaceAnalysis.run(&mut m).unwrap();
        assert!(m.ops[1].attrs.get("in_place_input_slot").is_none());
    }
}
