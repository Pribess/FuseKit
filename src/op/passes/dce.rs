//! Dead-code elimination — drop ops whose outputs don't reach any
//! live sink.
//!
//! Standard compiler DCE: build the reverse-reachability set from
//! "live roots" and retain only ops in that set. Live roots are:
//!
//! - **Module outputs** — tensors in `Module::outputs` (the final
//!   logits / KV cache writes / next-token output that the runtime
//!   reads).
//!
//! - **Synthetic ops** — opaque hand-tuned decode kernels
//!   (`OpKind::Synthetic { name: "RopeQDecode" / ... }`). They have
//!   external effects we don't model in IR yet (their virtual reads
//!   resolve through dispatch.py's `extra_bindings` at runtime), so
//!   we conservatively keep them all. Step 23/24 will wire their
//!   actual tensor dependencies and let backward propagation reach
//!   them via inputs instead.
//!
//! Real wins this pass unlocks on the current Qwen2.5-Coder-0.5B-4bit
//! compiler-path output: 73 dead chains (`Multiply → Unflatten` for
//! Q*scale, `Add → Reshape` for mask broadcasts, etc.) that the SDPA
//! piece supersedes but `general::run` happily merges anyway. The
//! chains write to scratches no downstream live op consumes — strictly
//! wasted dispatch + memory.

use std::collections::{HashMap, HashSet};

use crate::graph::TensorId;
use crate::op::module::Module;
use crate::op::op::OpKind;
use crate::op::passes::{Pass, PassError};

pub struct DeadCodeElimination;

impl Pass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, module: &mut Module) -> Result<bool, PassError> {
        // tensor_id → op.id that produces it
        let producer_of: HashMap<TensorId, u32> = module
            .ops
            .iter()
            .flat_map(|op| op.outputs.iter().map(|&t| (t, op.id)))
            .collect();
        // op.id → op index (for stable lookup by id while we walk)
        let id_to_idx: HashMap<u32, usize> = module
            .ops
            .iter()
            .enumerate()
            .map(|(i, op)| (op.id, i))
            .collect();
        let outputs_set: HashSet<TensorId> = module.outputs.iter().copied().collect();

        // Seed live set with: (a) ops producing module outputs,
        // (b) synthetic ops (conservatively always live).
        let mut live: HashSet<u32> = HashSet::new();
        for op in &module.ops {
            if matches!(op.kind, OpKind::Synthetic { .. }) {
                live.insert(op.id);
                continue;
            }
            if op.outputs.iter().any(|t| outputs_set.contains(t)) {
                live.insert(op.id);
            }
        }

        // Backward reachability: a producer is live if any of its
        // outputs is consumed by a live op. Iterate to fixed point.
        loop {
            let mut grew = false;
            let snapshot: Vec<u32> = live.iter().copied().collect();
            for op_id in snapshot {
                let Some(&idx) = id_to_idx.get(&op_id) else {
                    continue;
                };
                let op = &module.ops[idx];
                for &in_tid in &op.inputs {
                    if let Some(&producer_id) = producer_of.get(&in_tid) {
                        if live.insert(producer_id) {
                            grew = true;
                        }
                    }
                }
            }
            if !grew {
                break;
            }
        }

        let before = module.ops.len();
        module.ops.retain(|op| live.contains(&op.id));
        let after = module.ops.len();
        Ok(before != after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attrs, Dtype};
    use crate::op::module::{Module, Tensor};
    use crate::op::op::{Op, OpKind};

    fn mk_tensor(id: u32, producer: Option<u32>) -> Tensor {
        Tensor {
            id,
            shape: vec![1, 8, 896],
            dtype: Dtype::F16,
            producer,
            strides: None,
            partitioning: None,
        }
    }

    fn mk_primitive_op(id: u32, name: &str, inputs: Vec<u32>, outputs: Vec<u32>) -> Op {
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
    fn drops_op_with_no_consumer() {
        // Op produces tensor 100, no consumer, not module output → dead.
        let mut m = Module::new();
        m.tensors = vec![mk_tensor(100, Some(0))];
        m.ops = vec![mk_primitive_op(0, "Multiply", vec![], vec![100])];
        m.outputs = vec![]; // no module outputs
        DeadCodeElimination.run(&mut m).unwrap();
        assert_eq!(m.ops.len(), 0);
    }

    #[test]
    fn keeps_chain_to_module_output() {
        // op0 produces tensor 1, op1 reads 1 → 2, tensor 2 is module
        // output. Both ops live.
        let mut m = Module::new();
        m.tensors = vec![mk_tensor(1, Some(0)), mk_tensor(2, Some(1))];
        m.ops = vec![
            mk_primitive_op(0, "Add", vec![], vec![1]),
            mk_primitive_op(1, "Multiply", vec![1], vec![2]),
        ];
        m.outputs = vec![2];
        DeadCodeElimination.run(&mut m).unwrap();
        assert_eq!(m.ops.len(), 2);
    }

    #[test]
    fn drops_orphan_branch_only() {
        // op0 → tensor 1 → op2 → module output
        //                 (also)
        // op1 → tensor 10 → op3 → tensor 11 (no consumer, no output)
        // Op0, op2 live. Op1, op3 dead.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(1, Some(0)),
            mk_tensor(10, Some(1)),
            mk_tensor(2, Some(2)),
            mk_tensor(11, Some(3)),
        ];
        m.ops = vec![
            mk_primitive_op(0, "Add", vec![], vec![1]),
            mk_primitive_op(1, "Multiply", vec![], vec![10]),
            mk_primitive_op(2, "Sigmoid", vec![1], vec![2]),
            mk_primitive_op(3, "Reshape", vec![10], vec![11]),
        ];
        m.outputs = vec![2];
        DeadCodeElimination.run(&mut m).unwrap();
        // op0 and op2 remain.
        let ids: Vec<u32> = m.ops.iter().map(|op| op.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn synthetic_always_live_propagates_backward() {
        // op0 produces tensor 1, op1 (Synthetic) "reads" it via
        // inputs[] (model dependency wired explicitly). Synthetic is
        // always live, so op0 is reachable via backward propagation.
        // Tests the seed + propagate.
        let mut m = Module::new();
        m.tensors = vec![mk_tensor(1, Some(0))];
        m.ops = vec![
            mk_primitive_op(0, "Add", vec![], vec![1]),
            Op {
                id: 1,
                kind: OpKind::Synthetic {
                    name: "RopeQDecode".into(),
                },
                op_ids: vec![],
                inputs: vec![1],
                outputs: vec![],
                attrs: Default::default(),
                layer_hint: None,
            },
        ];
        m.outputs = vec![];
        DeadCodeElimination.run(&mut m).unwrap();
        assert_eq!(m.ops.len(), 2);
    }
}
