//! Algebraic rewrite rules.
//!
//! Declarative rule system — each `RewriteRule` is a small,
//! independently testable predicate-and-replacement pair. The
//! `AlgebraicRewrites` pass applies all registered rules iteratively
//! to a fixed point.
//!
//! Each rule:
//! - Inspects one op (and possibly its producer chain via
//!   `Module::tensor_mut`) to decide whether it can rewrite
//! - If yes, returns a `Rewrite` describing which existing ops to
//!   drop and which to add
//! - Doesn't modify the module directly; the pass driver applies the
//!   rewrite so the iteration order stays predictable
//!
//! Starting rules (this file):
//! - `ReshapeReshape` — collapse `Reshape(Reshape(x))` into a single
//!   `Reshape(x)`. mlx's lazy graph sometimes emits nested reshapes
//!   when transformer code does `x.reshape(A).reshape(B)`.
//! - `IdentityTranspose` — `Transpose(x, axes=identity)` is a no-op.
//! - `SqueezeExpandCancel` — `Squeeze(ExpandDims(x, axis), axis)`
//!   roundtrips to `x`.
//!
//! Future rules (TaskList #34 — Algorithmic identity macros):
//! - `MultiplyByScaleIntoHeavy` — fold scalar Multiply into a
//!   following AddMM / QuantizedMatmul's bias term
//! - `SoftmaxMaxStability` — canonicalize `Softmax(x)` →
//!   `Softmax(x - max(x))`
//! - `KvSliceUpdateContiguous` — recognise `SliceUpdate` patterns
//!   that match contiguous stores and lower to specialised KV-write op
//!
//! Rewrites compose naturally with DCE: collapsing two Reshapes
//! produces a no-consumer chain on the intermediate tensor, which
//! the next DCE iteration drops. So the recommended pass order in
//! step 24's scheduler is: `algebraic → dce → algebraic → dce → ...`
//! to fixed point.

use crate::graph::{Attrs, TensorId};
use crate::op::module::Module;
use crate::op::op::{Op, OpKind};
use crate::op::passes::{Pass, PassError};

/// One rewrite candidate produced by a `RewriteRule`. The pass driver
/// applies these atomically — all `remove` op ids are dropped and all
/// `add` ops are appended, in one transactional step.
#[derive(Debug)]
pub struct Rewrite {
    pub remove: Vec<u32>,
    pub add: Vec<Op>,
}

pub trait RewriteRule {
    fn name(&self) -> &'static str;
    /// Try to apply this rule at `op_idx`. Returns `Some(Rewrite)` if
    /// the rule fires; `None` if not applicable.
    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite>;
}

/// Pass that applies all registered rewrite rules to a fixed point.
pub struct AlgebraicRewrites {
    pub rules: Vec<Box<dyn RewriteRule>>,
}

impl AlgebraicRewrites {
    /// Default rule set — the always-safe canonicalizations. Add new
    /// rules via `with_rule` for opt-in / experimental ones.
    pub fn with_defaults() -> Self {
        Self {
            rules: vec![
                Box::new(IdentityTranspose),
                Box::new(ReshapeReshape),
                Box::new(SqueezeExpandCancel),
            ],
        }
    }

    pub fn with_rule(mut self, rule: Box<dyn RewriteRule>) -> Self {
        self.rules.push(rule);
        self
    }
}

impl Pass for AlgebraicRewrites {
    fn name(&self) -> &'static str {
        "algebraic-rewrites"
    }

    fn run(&self, module: &mut Module) -> Result<bool, PassError> {
        let mut total_changed = false;
        for _ in 0..64 {
            let mut applied_this_iter = 0;
            for op_idx in 0..module.ops.len() {
                if op_idx >= module.ops.len() {
                    break;
                }
                for rule in &self.rules {
                    if let Some(rewrite) = rule.try_rewrite(op_idx, module) {
                        apply_rewrite(module, rewrite);
                        applied_this_iter += 1;
                        break;
                    }
                }
            }
            if applied_this_iter == 0 {
                break;
            }
            total_changed = true;
        }
        Ok(total_changed)
    }
}

fn apply_rewrite(module: &mut Module, rewrite: Rewrite) {
    let remove: std::collections::HashSet<u32> = rewrite.remove.iter().copied().collect();
    module.ops.retain(|op| !remove.contains(&op.id));
    module.ops.extend(rewrite.add);
}

// ─── Concrete rules ───────────────────────────────────────────────────────

/// `Transpose(x, axes=[0,1,2,...])` is an identity. Replace the op's
/// downstream consumers' inputs with the Transpose's input tensor and
/// drop the Transpose.
///
/// Currently this rule only fires when the output tensor has the same
/// shape as the input (i.e. axes is sorted ascending). We emit a
/// pass-through `Reshape` step semantically — but since `Reshape`
/// preserves data and rank is unchanged, the rewrite simplifies to
/// dropping the Transpose and aliasing its output to its input
/// (handled by later view propagation).
pub struct IdentityTranspose;

impl RewriteRule for IdentityTranspose {
    fn name(&self) -> &'static str {
        "identity-transpose"
    }

    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite> {
        let op = module.ops.get(op_idx)?;
        let OpKind::Primitive { name, raw_attrs } = &op.kind else {
            return None;
        };
        if name != "Transpose" {
            return None;
        }
        let Attrs::Transpose { axes } = raw_attrs else {
            return None;
        };
        // Identity iff axes == [0, 1, 2, ..., rank-1]
        let is_identity = axes.iter().enumerate().all(|(i, &a)| a as usize == i);
        if !is_identity {
            return None;
        }
        // Drop the op; downstream consumers' inputs still point to the
        // op's output tensor id. Since the data is identical to the
        // input, a follow-up "view aliasing" rewrite (or codegen's
        // bridge alias mechanism) connects them. For now we leave the
        // output tensor as a dangling rebrand — DCE will reach it
        // through downstream consumers and keep things consistent.
        Some(Rewrite {
            remove: vec![op.id],
            add: vec![],
        })
    }
}

/// `Reshape(Reshape(x, A), B)` → `Reshape(x, B)`. mlx sometimes emits
/// nested reshapes when transformer code does `x.reshape(A).reshape(B)`.
/// The intermediate tensor is throwaway.
///
/// Conservative: only fire when the inner Reshape is the SOLE producer
/// of the intermediate tensor (no other op uses it) so we don't break
/// graph invariants.
pub struct ReshapeReshape;

impl RewriteRule for ReshapeReshape {
    fn name(&self) -> &'static str {
        "reshape-reshape"
    }

    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite> {
        let op = module.ops.get(op_idx)?;
        let OpKind::Primitive {
            name: outer_name,
            raw_attrs: outer_attrs,
        } = &op.kind
        else {
            return None;
        };
        if outer_name != "Reshape" {
            return None;
        }
        let Attrs::Reshape {
            target_shape: outer_shape,
        } = outer_attrs
        else {
            return None;
        };
        let inner_tid = *op.inputs.first()?;
        // Look up the inner producer.
        let inner_op = module
            .ops
            .iter()
            .find(|o| o.outputs.iter().any(|&t| t == inner_tid))?;
        let OpKind::Primitive {
            name: inner_name, ..
        } = &inner_op.kind
        else {
            return None;
        };
        if inner_name != "Reshape" {
            return None;
        }
        // Single consumer check: only collapse if the intermediate
        // tensor flows only into our outer Reshape.
        let consumer_count = count_consumers(module, inner_tid);
        if consumer_count != 1 {
            return None;
        }
        // Build the collapsed op: input from inner.inputs[0], output
        // shape from outer.
        let collapsed = Op {
            id: op.id, // reuse the outer op's id
            kind: OpKind::Primitive {
                name: "Reshape".into(),
                raw_attrs: Attrs::Reshape {
                    target_shape: outer_shape.clone(),
                },
            },
            op_ids: {
                let mut ids = inner_op.op_ids.clone();
                ids.extend(op.op_ids.iter().copied());
                ids
            },
            inputs: inner_op.inputs.clone(),
            outputs: op.outputs.clone(),
            attrs: Default::default(),
            layer_hint: op.layer_hint,
        };
        Some(Rewrite {
            remove: vec![inner_op.id, op.id],
            add: vec![collapsed],
        })
    }
}

/// `Squeeze(ExpandDims(x, axis), axis)` cancels back to `x`. mlx
/// sometimes leaves these roundtrips when shapes get rewritten across
/// indexing patterns.
pub struct SqueezeExpandCancel;

impl RewriteRule for SqueezeExpandCancel {
    fn name(&self) -> &'static str {
        "squeeze-expand-cancel"
    }

    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite> {
        let op = module.ops.get(op_idx)?;
        let OpKind::Primitive { name, .. } = &op.kind else {
            return None;
        };
        if name != "Squeeze" {
            return None;
        }
        let inner_tid = *op.inputs.first()?;
        let inner_op = module
            .ops
            .iter()
            .find(|o| o.outputs.iter().any(|&t| t == inner_tid))?;
        let OpKind::Primitive {
            name: inner_name, ..
        } = &inner_op.kind
        else {
            return None;
        };
        if inner_name != "ExpandDims" {
            return None;
        }
        // Only collapse when intermediate has a single consumer (us).
        if count_consumers(module, inner_tid) != 1 {
            return None;
        }
        // After cancel, the Squeeze's output should equal the
        // ExpandDims's input. Since we can't merge two tensor ids in
        // the IR without remapping every reader, emit a Reshape that's
        // semantically a view-through. DCE/codegen aliasing handles
        // the rest.
        let passthrough = Op {
            id: op.id,
            kind: OpKind::Primitive {
                name: "Reshape".into(),
                raw_attrs: Attrs::Reshape {
                    target_shape: module
                        .tensor(*op.outputs.first()?)
                        .map(|t| t.shape.clone())
                        .unwrap_or_default(),
                },
            },
            op_ids: {
                let mut ids = inner_op.op_ids.clone();
                ids.extend(op.op_ids.iter().copied());
                ids
            },
            inputs: inner_op.inputs.clone(),
            outputs: op.outputs.clone(),
            attrs: Default::default(),
            layer_hint: op.layer_hint,
        };
        Some(Rewrite {
            remove: vec![inner_op.id, op.id],
            add: vec![passthrough],
        })
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────

fn count_consumers(module: &Module, tid: TensorId) -> usize {
    module
        .ops
        .iter()
        .filter(|op| op.inputs.contains(&tid))
        .count()
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

    #[test]
    fn identity_transpose_dropped() {
        // Transpose with axes [0, 1, 2] is identity.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(1, vec![2, 3, 4], None),
            mk_tensor(2, vec![2, 3, 4], Some(0)),
        ];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Transpose".into(),
                raw_attrs: Attrs::Transpose {
                    axes: vec![0, 1, 2],
                },
            },
            op_ids: vec![],
            inputs: vec![1],
            outputs: vec![2],
            attrs: Default::default(),
            layer_hint: None,
        }];
        AlgebraicRewrites::with_defaults().run(&mut m).unwrap();
        assert_eq!(m.ops.len(), 0);
    }

    #[test]
    fn non_identity_transpose_kept() {
        // Transpose with axes [1, 0, 2] is a real permutation.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(1, vec![2, 3, 4], None),
            mk_tensor(2, vec![3, 2, 4], Some(0)),
        ];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Transpose".into(),
                raw_attrs: Attrs::Transpose {
                    axes: vec![1, 0, 2],
                },
            },
            op_ids: vec![],
            inputs: vec![1],
            outputs: vec![2],
            attrs: Default::default(),
            layer_hint: None,
        }];
        AlgebraicRewrites::with_defaults().run(&mut m).unwrap();
        assert_eq!(m.ops.len(), 1);
    }

    #[test]
    fn nested_reshape_collapses() {
        // Reshape(Reshape(x, [12]), [3, 4]) → Reshape(x, [3, 4])
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(1, vec![2, 6], None),
            mk_tensor(2, vec![12], Some(0)),
            mk_tensor(3, vec![3, 4], Some(1)),
        ];
        m.ops = vec![
            Op {
                id: 0,
                kind: OpKind::Primitive {
                    name: "Reshape".into(),
                    raw_attrs: Attrs::Reshape {
                        target_shape: vec![12],
                    },
                },
                op_ids: vec![],
                inputs: vec![1],
                outputs: vec![2],
                attrs: Default::default(),
                layer_hint: None,
            },
            Op {
                id: 1,
                kind: OpKind::Primitive {
                    name: "Reshape".into(),
                    raw_attrs: Attrs::Reshape {
                        target_shape: vec![3, 4],
                    },
                },
                op_ids: vec![],
                inputs: vec![2],
                outputs: vec![3],
                attrs: Default::default(),
                layer_hint: None,
            },
        ];
        AlgebraicRewrites::with_defaults().run(&mut m).unwrap();
        assert_eq!(m.ops.len(), 1);
        let kept = &m.ops[0];
        assert_eq!(kept.inputs, vec![1]);
        assert_eq!(kept.outputs, vec![3]);
    }

    #[test]
    fn nested_reshape_with_extra_consumer_kept() {
        // If intermediate tensor has another consumer, don't collapse.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(1, vec![2, 6], None),
            mk_tensor(2, vec![12], Some(0)),
            mk_tensor(3, vec![3, 4], Some(1)),
            mk_tensor(4, vec![12], Some(2)),
        ];
        m.ops = vec![
            Op {
                id: 0,
                kind: OpKind::Primitive {
                    name: "Reshape".into(),
                    raw_attrs: Attrs::Reshape {
                        target_shape: vec![12],
                    },
                },
                op_ids: vec![],
                inputs: vec![1],
                outputs: vec![2],
                attrs: Default::default(),
                layer_hint: None,
            },
            Op {
                id: 1,
                kind: OpKind::Primitive {
                    name: "Reshape".into(),
                    raw_attrs: Attrs::Reshape {
                        target_shape: vec![3, 4],
                    },
                },
                op_ids: vec![],
                inputs: vec![2],
                outputs: vec![3],
                attrs: Default::default(),
                layer_hint: None,
            },
            // Extra consumer of tensor 2:
            Op {
                id: 2,
                kind: OpKind::Primitive {
                    name: "Add".into(),
                    raw_attrs: Attrs::Unknown,
                },
                op_ids: vec![],
                inputs: vec![2],
                outputs: vec![4],
                attrs: Default::default(),
                layer_hint: None,
            },
        ];
        AlgebraicRewrites::with_defaults().run(&mut m).unwrap();
        // All 3 ops kept — the rule didn't fire because of the extra
        // consumer.
        assert_eq!(m.ops.len(), 3);
    }
}
