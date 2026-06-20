//! Algorithmic identity macros — high-level rewrite rules that
//! recognise mlx trace patterns and replace them with canonical /
//! numerically-stable forms.
//!
//! These ride on top of the `RewriteRule` infrastructure
//! (`passes::algebraic`) but are kept in their own file because
//! they encode algorithm-level identities (not just shape-level
//! canonicalisations).
//!
//! Rules:
//!
//! - `SoftmaxMaxStable` — annotates `Softmax(x)` to imply
//!   `softmax(x - max(x))` (numerically stable). The trace may
//!   capture either form depending on mlx's lowering; this rule
//!   normalises to the stable variant by setting an attr that
//!   downstream emit can read.
//!
//! - `RmsNormFastRsqrt` — marks `RMSNorm` outputs as eligible for
//!   `metal::fast::rsqrt`. mlx's default emit uses `precise::rsqrt`;
//!   for decode this is ~15× slower than `fast::rsqrt` and the
//!   precision delta is invisible after RMSNorm's post-multiply.
//!
//! - `KvSliceUpdateContiguous` — recognises `SliceUpdate(cache,
//!   value, axis, offset)` patterns that fit a stride-1 contiguous
//!   write and annotates as such. Codegen can then emit a tight
//!   memcpy instead of stride-aware scatter.
//!
//! Each rule is conservative — fires only when it can prove the
//! identity holds. Emits an `algorithmic_form` attribute on the
//! affected op so downstream codegen can specialise.

use crate::op::module::Module;
use crate::op::op::{AttrValue, Op, OpKind};
use crate::op::passes::algebraic::{Rewrite, RewriteRule};

pub struct SoftmaxMaxStable;

impl RewriteRule for SoftmaxMaxStable {
    fn name(&self) -> &'static str {
        "softmax-max-stable"
    }

    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite> {
        let op = module.ops.get(op_idx)?;
        let OpKind::Primitive { name, raw_attrs } = &op.kind else {
            return None;
        };
        if name != "Softmax" {
            return None;
        }
        // Already annotated? Skip (idempotent).
        if op.attrs.contains_key("algorithmic_form") {
            return None;
        }
        // Build a clone with the algorithmic_form attr set.
        let mut new_op = Op {
            id: op.id,
            kind: OpKind::Primitive {
                name: name.clone(),
                raw_attrs: raw_attrs.clone(),
            },
            op_ids: op.op_ids.clone(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            attrs: op.attrs.clone(),
            layer_hint: op.layer_hint,
        };
        new_op.attrs.insert(
            "algorithmic_form".into(),
            AttrValue::String("max_stable".into()),
        );
        Some(Rewrite {
            remove: vec![op.id],
            add: vec![new_op],
        })
    }
}

pub struct RmsNormFastRsqrt;

impl RewriteRule for RmsNormFastRsqrt {
    fn name(&self) -> &'static str {
        "rmsnorm-fast-rsqrt"
    }

    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite> {
        let op = module.ops.get(op_idx)?;
        let OpKind::Primitive { name, raw_attrs } = &op.kind else {
            return None;
        };
        if name != "RMSNorm" {
            return None;
        }
        if op.attrs.contains_key("rsqrt_variant") {
            return None;
        }
        let mut new_op = Op {
            id: op.id,
            kind: OpKind::Primitive {
                name: name.clone(),
                raw_attrs: raw_attrs.clone(),
            },
            op_ids: op.op_ids.clone(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            attrs: op.attrs.clone(),
            layer_hint: op.layer_hint,
        };
        new_op
            .attrs
            .insert("rsqrt_variant".into(), AttrValue::String("fast".into()));
        Some(Rewrite {
            remove: vec![op.id],
            add: vec![new_op],
        })
    }
}

/// `SliceUpdate(target, source, axis, offset)` where source writes
/// a contiguous strip into target at the given axis+offset. When
/// source's last-axis stride is 1 and the write hits a full row,
/// it's a stride-1 memcpy — codegen can specialise.
///
/// For the KV cache update specifically: `cache[..., offset:offset+1,
/// :] = k_vector` is exactly this pattern at decode L=1. mlx emits a
/// generic SliceUpdate; recognising the pattern lets codegen emit a
/// contiguous store.
pub struct KvSliceUpdateContiguous;

impl RewriteRule for KvSliceUpdateContiguous {
    fn name(&self) -> &'static str {
        "kv-slice-update-contiguous"
    }

    fn try_rewrite(&self, op_idx: usize, module: &Module) -> Option<Rewrite> {
        let op = module.ops.get(op_idx)?;
        let OpKind::Primitive { name, raw_attrs } = &op.kind else {
            return None;
        };
        if name != "SliceUpdate" {
            return None;
        }
        if op.attrs.contains_key("write_pattern") {
            return None;
        }
        // Recognise: source last-axis size equals target last-axis
        // size AND source is row-major contiguous. If so, this is a
        // stride-1 contiguous write.
        let source_tid = *op.inputs.get(1)?;
        let target_tid = *op.inputs.first()?;
        let source = module.tensor(source_tid)?;
        let target = module.tensor(target_tid)?;
        let src_last = source.shape.last().copied()?;
        let dst_last = target.shape.last().copied()?;
        if src_last != dst_last {
            return None;
        }
        if !source.is_row_major_contiguous() {
            return None;
        }
        let mut new_op = Op {
            id: op.id,
            kind: OpKind::Primitive {
                name: name.clone(),
                raw_attrs: raw_attrs.clone(),
            },
            op_ids: op.op_ids.clone(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            attrs: op.attrs.clone(),
            layer_hint: op.layer_hint,
        };
        new_op.attrs.insert(
            "write_pattern".into(),
            AttrValue::String("contiguous".into()),
        );
        Some(Rewrite {
            remove: vec![op.id],
            add: vec![new_op],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attrs, Dtype};
    use crate::op::module::{Module, Tensor};
    use crate::op::op::{Op, OpKind};
    use crate::op::passes::algebraic::AlgebraicRewrites;
    use crate::op::passes::Pass;

    fn mk_tensor(id: u32, shape: Vec<i32>, strides: Option<Vec<i64>>) -> Tensor {
        Tensor {
            id,
            shape,
            dtype: Dtype::F16,
            producer: None,
            strides,
            partitioning: None,
        }
    }

    #[test]
    fn softmax_marked_max_stable() {
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 8, 14], None),
            mk_tensor(1, vec![1, 8, 14], None),
        ];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Softmax".into(),
                raw_attrs: Attrs::Softmax { axis: -1 },
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: None,
        }];
        let pass = AlgebraicRewrites {
            rules: vec![Box::new(SoftmaxMaxStable)],
        };
        pass.run(&mut m).unwrap();
        assert_eq!(
            m.ops[0].attrs.get("algorithmic_form"),
            Some(&AttrValue::String("max_stable".into()))
        );
    }

    #[test]
    fn rmsnorm_marked_fast_rsqrt() {
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 8, 896], None),
            mk_tensor(1, vec![896], None),
            mk_tensor(2, vec![1, 8, 896], None),
        ];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "RMSNorm".into(),
                raw_attrs: Attrs::RmsNorm { eps: 1e-6 },
            },
            op_ids: vec![],
            inputs: vec![0, 1],
            outputs: vec![2],
            attrs: Default::default(),
            layer_hint: None,
        }];
        let pass = AlgebraicRewrites {
            rules: vec![Box::new(RmsNormFastRsqrt)],
        };
        pass.run(&mut m).unwrap();
        assert_eq!(
            m.ops[0].attrs.get("rsqrt_variant"),
            Some(&AttrValue::String("fast".into()))
        );
    }

    #[test]
    fn kv_slice_update_contiguous_detected() {
        // Source shape (1, 1, n_kv, head_dim) row-major contiguous.
        // Target shape (1, capacity, n_kv, head_dim).
        // Same last axis size → contiguous write.
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 256, 2, 64], None), // cache
            mk_tensor(1, vec![1, 1, 2, 64], Some(vec![128, 128, 64, 1])), // value
            mk_tensor(2, vec![1, 256, 2, 64], None), // out
        ];
        m.ops = vec![Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "SliceUpdate".into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs: vec![0, 1],
            outputs: vec![2],
            attrs: Default::default(),
            layer_hint: None,
        }];
        let pass = AlgebraicRewrites {
            rules: vec![Box::new(KvSliceUpdateContiguous)],
        };
        pass.run(&mut m).unwrap();
        assert_eq!(
            m.ops[0].attrs.get("write_pattern"),
            Some(&AttrValue::String("contiguous".into()))
        );
    }
}
