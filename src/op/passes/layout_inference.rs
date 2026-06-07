//! Layout inference — populate `Tensor::strides` for every tensor
//! in the module.
//!
//! First productive pass of the IR refactor (step 2). Walks ops in
//! their declared order and propagates strides based on each op's
//! semantic effect:
//!
//! - **Module inputs / leaves** (`Tensor::producer == None`): default
//!   to row-major contiguous strides. Matches mlx's default layout
//!   for fresh allocations + the loader's weight storage convention.
//!
//! - **Primitive ops that preserve layout** (Add, Multiply, Sigmoid,
//!   RMSNorm output, Matmul output, QM output, Softmax along
//!   contiguous axis, AsType): output gets row-major contiguous in
//!   its declared shape. Even if inputs are strided, these ops
//!   write to a fresh buffer so output is contiguous.
//!
//! - **View ops with stride changes** (Transpose, Broadcast,
//!   Squeeze, ExpandDims, Reshape): output strides computed from
//!   input strides + op-specific permutation/rule.
//!
//! - **Chain outputs**: each chain step writes one register output;
//!   the chain's final output is the last step's write, treated like
//!   a fresh materialisation → row-major contiguous.
//!
//! - **Synthetic ops** (RopeQDecode, etc.): opaque, declared layout
//!   matches what the synthetic kernel produces (treat as row-major
//!   contiguous for the chain output shape).
//!
//! Layout info enables: (a) detecting when a Reshape is a pure
//! "view" vs requires materialisation (step 2 deliverable used by
//! the DCE/canonicalize passes later), (b) view propagation pass
//! that retires the SDPA-bridge alias hack, (c) the scheduler
//! choosing stride-friendly access patterns when assigning
//! partitioning (step 5).

use crate::graph::Attrs;
use crate::op::module::Module;
use crate::op::op::{Op, OpKind};
use crate::op::passes::{Pass, PassError};

pub struct LayoutInference;

impl Pass for LayoutInference {
    fn name(&self) -> &'static str {
        "layout-inference"
    }

    fn run(&self, module: &mut Module) -> Result<bool, PassError> {
        // First: seed every leaf tensor (no producer) with row-major
        // contiguous strides. mlx's loaders give us contiguous tensors
        // for weights/inputs.
        let mut changed = seed_leaf_strides(module);

        // Second pass: walk ops in declared order and propagate.
        // Ops are already in topological order from the FusionGraph
        // (which itself preserves trace order via build_fusions).
        for op_idx in 0..module.ops.len() {
            // Clone the op data we need so we can mutate `module.tensors`
            // without aliasing.
            let op = module.ops[op_idx].clone();
            if propagate_op(&op, module) {
                changed = true;
            }
        }

        Ok(changed)
    }
}

fn seed_leaf_strides(module: &mut Module) -> bool {
    let mut changed = false;
    for tensor in &mut module.tensors {
        if tensor.producer.is_some() {
            continue;
        }
        let strides = crate::op::module::Tensor::row_major_strides(&tensor.shape);
        if tensor.strides.as_ref() != Some(&strides) {
            tensor.strides = Some(strides);
            changed = true;
        }
    }
    changed
}

fn propagate_op(op: &Op, module: &mut Module) -> bool {
    let mut changed = false;
    match &op.kind {
        OpKind::Primitive { name, raw_attrs } => {
            for &out_tid in &op.outputs {
                let strides = primitive_output_strides(name, raw_attrs, op, module, out_tid);
                if set_strides(module, out_tid, strides) {
                    changed = true;
                }
            }
        }
        OpKind::Chain { .. } | OpKind::Synthetic { .. } => {
            // Chain output / synthetic kernel output: written by the
            // fusekit-emitted kernel which always writes row-major
            // contiguous in the declared output shape. (No chain emit
            // today writes non-contiguous.)
            for &out_tid in &op.outputs {
                let shape = match module.tensor(out_tid) {
                    Some(t) => t.shape.clone(),
                    None => continue,
                };
                let strides = crate::op::module::Tensor::row_major_strides(&shape);
                if set_strides(module, out_tid, strides) {
                    changed = true;
                }
            }
        }
    }
    changed
}

fn set_strides(module: &mut Module, tid: crate::graph::TensorId, strides: Vec<i64>) -> bool {
    let Some(tensor) = module.tensor_mut(tid) else {
        return false;
    };
    if tensor.strides.as_ref() == Some(&strides) {
        return false;
    }
    tensor.strides = Some(strides);
    true
}

fn primitive_output_strides(
    name: &str,
    attrs: &Attrs,
    op: &Op,
    module: &Module,
    out_tid: crate::graph::TensorId,
) -> Vec<i64> {
    let out_shape = match module.tensor(out_tid) {
        Some(t) => t.shape.clone(),
        None => return Vec::new(),
    };
    let row_major = || crate::op::module::Tensor::row_major_strides(&out_shape);

    match name {
        // View ops — propagate input strides with op-specific rule.
        "Transpose" => match attrs {
            Attrs::Transpose { axes } => {
                if let Some(in_tid) = op.inputs.first().copied() {
                    if let Some(in_strides) = module.tensor(in_tid).and_then(|t| t.strides.clone())
                    {
                        if axes.len() == in_strides.len() {
                            // Output stride at position d = input stride
                            // at axes[d]. This expresses the view
                            // without materialisation.
                            return axes
                                .iter()
                                .map(|&a| in_strides.get(a as usize).copied().unwrap_or(0))
                                .collect();
                        }
                    }
                }
                row_major()
            }
            _ => row_major(),
        },

        "Broadcast" => match attrs {
            Attrs::Broadcast { target_shape } => {
                // Broadcast: a size-1 input dim expands to the target's
                // dim with stride 0. Other dims keep input stride.
                if let Some(in_tid) = op.inputs.first().copied() {
                    if let (Some(in_tensor), Some(in_strides)) = (
                        module.tensor(in_tid),
                        module.tensor(in_tid).and_then(|t| t.strides.clone()),
                    ) {
                        let rank = target_shape.len();
                        let pad = rank.saturating_sub(in_tensor.shape.len());
                        let in_shape_padded: Vec<i32> = std::iter::repeat(1)
                            .take(pad)
                            .chain(in_tensor.shape.iter().copied())
                            .collect();
                        let in_strides_padded: Vec<i64> = std::iter::repeat(0i64)
                            .take(pad)
                            .chain(in_strides.iter().copied())
                            .collect();
                        return (0..rank)
                            .map(|d| {
                                if in_shape_padded[d] == 1 {
                                    0
                                } else {
                                    in_strides_padded[d]
                                }
                            })
                            .collect();
                    }
                }
                row_major()
            }
            _ => row_major(),
        },

        "Reshape" => {
            // Reshape: if input is row-major contiguous, the output is
            // ALSO row-major contiguous in the new shape (mlx
            // guarantees this for true views). If input has non-trivial
            // strides, mlx materialises — output is still row-major
            // contiguous in target. Either way, output is contiguous.
            row_major()
        }

        "Squeeze" | "Unflatten" | "Flatten" | "ExpandDims" | "AsContiguous" => row_major(),

        // Element-wise ops always materialise contiguous output.
        "Add" | "Multiply" | "Sigmoid" | "Exp" | "Negative" | "AsType" | "Select"
        | "GreaterEqual" | "LogAddExp" => row_major(),

        // Heavy ops materialise contiguous output.
        "Matmul" | "AddMM" | "QuantizedMatmul" => row_major(),

        // Reductions produce contiguous output along non-reduced axes.
        "RMSNorm" | "LayerNorm" | "Softmax" => row_major(),

        // Unknown ops — assume contiguous output (safe default; if it
        // turns out to be wrong for some primitive, add an explicit
        // arm above).
        _ => row_major(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Dtype;
    use crate::op::module::{Module, Tensor};
    use crate::op::op::{Op, OpKind};

    fn module_with(tensors: Vec<Tensor>, ops: Vec<Op>) -> Module {
        let mut m = Module::new();
        m.tensors = tensors;
        m.ops = ops;
        m
    }

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
    fn seeds_leaf_strides_to_row_major() {
        let mut m = module_with(vec![mk_tensor(0, vec![2, 7, 64], None)], vec![]);
        let pass = LayoutInference;
        pass.run(&mut m).unwrap();
        let t = m.tensor(0).unwrap();
        // (2, 7, 64) row-major: strides = (7*64, 64, 1) = (448, 64, 1)
        assert_eq!(t.strides.as_deref(), Some(&[448i64, 64, 1][..]));
    }

    #[test]
    fn transpose_permutes_strides() {
        let input = mk_tensor(0, vec![1, 2, 7, 8, 64], None);
        let output = mk_tensor(1, vec![1, 2, 8, 7, 64], Some(0));
        let op = Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Transpose".into(),
                raw_attrs: Attrs::Transpose {
                    axes: vec![0, 1, 3, 2, 4],
                },
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: None,
        };
        let mut m = module_with(vec![input, output], vec![op]);
        LayoutInference.run(&mut m).unwrap();
        // Input strides row-major (1,2,7,8,64) = (7168, 3584, 512, 64, 1)
        // After Transpose with axes (0,1,3,2,4):
        // output.strides[d] = input.strides[axes[d]]
        //   d=0: input.strides[0] = 7168
        //   d=1: input.strides[1] = 3584
        //   d=2: input.strides[3] = 64   ← swap reflected here
        //   d=3: input.strides[2] = 512
        //   d=4: input.strides[4] = 1
        let out = m.tensor(1).unwrap();
        assert_eq!(
            out.strides.as_deref(),
            Some(&[7168i64, 3584, 64, 512, 1][..])
        );
        // Crucially, this is NOT row-major contiguous in (1,2,8,7,64).
        assert!(!out.is_row_major_contiguous());
    }

    #[test]
    fn broadcast_stride_zero_on_expanded_axes() {
        // Input shape (1, 64) → broadcast to (8, 64). Axis 0 is the
        // size-1 dim that gets expanded; its output stride is 0.
        let input = mk_tensor(0, vec![1, 64], None);
        let output = mk_tensor(1, vec![8, 64], Some(0));
        let op = Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Broadcast".into(),
                raw_attrs: Attrs::Broadcast {
                    target_shape: vec![8, 64],
                },
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: None,
        };
        let mut m = module_with(vec![input, output], vec![op]);
        LayoutInference.run(&mut m).unwrap();
        let out = m.tensor(1).unwrap();
        assert_eq!(out.strides.as_deref(), Some(&[0i64, 1][..]));
    }

    #[test]
    fn primitive_output_is_contiguous() {
        // Matmul output must be row-major contiguous regardless of
        // input strides — the kernel materialises into a fresh buffer.
        let a = mk_tensor(0, vec![1, 896], None);
        let b = mk_tensor(1, vec![896, 896], None);
        let out = mk_tensor(2, vec![1, 896], Some(0));
        let op = Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Matmul".into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs: vec![0, 1],
            outputs: vec![2],
            attrs: Default::default(),
            layer_hint: None,
        };
        let mut m = module_with(vec![a, b, out], vec![op]);
        LayoutInference.run(&mut m).unwrap();
        assert!(m.tensor(2).unwrap().is_row_major_contiguous());
    }
}
