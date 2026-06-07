//! Per-op cost model.
//!
//! Replaces hardcoded chip→TG mapping and one-off heuristics
//! (`gemv_rows_per_simdgroup`, `kernel_threadgroup_for`'s special
//! cases) with a single cost function per op. Downstream scheduling
//! decisions (multi-row coalesce factor, TG size, simdgroup_matrix
//! eligibility) consume the cost model instead of branching on chip
//! family directly.
//!
//! ```text
//!   Op  ─►  Cost { regs, smem_bytes, dispatch_us, mem_bytes, flops }
//!                  └── consumed by scheduler / tiling pass
//! ```
//!
//! Costs are *estimates* — exact perf depends on chip + thermal + KV
//! length. The model produces a relative ordering so the scheduler
//! picks one variant over another.

use crate::graph::Attrs;
use crate::op::module::Module;
use crate::op::op::{Op, OpKind};

/// Cost estimate for one Op.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// Live registers per thread at peak.
    pub regs: u32,
    /// Threadgroup-shared memory bytes per dispatch.
    pub smem_bytes: u32,
    /// Approximate device memory read + write bytes per dispatch.
    pub mem_bytes: u64,
    /// Approximate floating-point ops per dispatch.
    pub flops: u64,
    /// Apple-Metal dispatch overhead — per-kernel launch cost roughly
    /// constant ~10µs on M5. Used by DCE-related heuristics to
    /// estimate the win of dropping an op.
    pub dispatch_us: u32,
}

impl Cost {
    pub const ZERO: Cost = Cost {
        regs: 0,
        smem_bytes: 0,
        mem_bytes: 0,
        flops: 0,
        dispatch_us: 0,
    };

    /// Sum costs (used when modeling fused chains as the sum of their
    /// step costs).
    pub fn sum(&self, other: &Cost) -> Cost {
        Cost {
            regs: self.regs.max(other.regs),
            smem_bytes: self.smem_bytes + other.smem_bytes,
            mem_bytes: self.mem_bytes + other.mem_bytes,
            flops: self.flops + other.flops,
            dispatch_us: self.dispatch_us.max(other.dispatch_us),
        }
    }
}

/// Estimate the cost of one Op given the model's metadata.
pub fn estimate_op_cost(op: &Op, module: &Module) -> Cost {
    match &op.kind {
        OpKind::Primitive { name, raw_attrs } => primitive_cost(name, raw_attrs, op, module),
        OpKind::Chain { steps } => {
            // Chain cost = max regs + sum SMEM/mem/flops + single
            // dispatch (the chain is ONE kernel launch).
            let mut total = Cost::ZERO;
            for step in steps {
                let step_op = Op {
                    id: 0,
                    kind: OpKind::Primitive {
                        name: step.op_name.clone(),
                        raw_attrs: step.raw_attrs.clone(),
                    },
                    op_ids: vec![],
                    inputs: op.inputs.clone(),
                    outputs: op.outputs.clone(),
                    attrs: Default::default(),
                    layer_hint: None,
                };
                let c = primitive_cost(&step.op_name, &step.raw_attrs, &step_op, module);
                total = total.sum(&c);
            }
            total.dispatch_us = 10;
            total
        }
        OpKind::Synthetic { name } => synthetic_cost(name),
    }
}

fn module_last_dim(module: &Module, tid: crate::graph::TensorId) -> Option<u64> {
    module
        .tensor(tid)
        .and_then(|t| t.shape.last().map(|&d| d.max(1) as u64))
}

fn primitive_cost(name: &str, attrs: &Attrs, op: &Op, module: &Module) -> Cost {
    let elems = |tid: crate::graph::TensorId| -> u64 {
        module
            .tensor(tid)
            .map(|t| t.shape.iter().map(|&d| d.max(1) as u64).product::<u64>())
            .unwrap_or(0)
    };
    let out_elems = op.outputs.iter().map(|&t| elems(t)).sum::<u64>();
    let in_elems = op.inputs.iter().map(|&t| elems(t)).sum::<u64>();
    // mem_bytes = (read + write) × bytes-per-element. Assume f16/bf16
    // (2 bytes) — refined later via dtype lookup.
    let mem = (in_elems + out_elems) * 2;
    match name {
        // ── Pointwise ──
        "Add" | "Multiply" | "Sigmoid" | "Exp" | "Negative" | "AsType" | "Select"
        | "GreaterEqual" | "LogAddExp" => Cost {
            regs: 16,
            smem_bytes: 0,
            mem_bytes: mem,
            flops: out_elems,
            dispatch_us: 10,
        },

        // ── Reductions ──
        "RMSNorm" | "LayerNorm" => Cost {
            regs: 32,
            smem_bytes: 4 * 1024, // norm_y + reduce_buf
            mem_bytes: mem,
            flops: out_elems * 4, // mul + sum + rsqrt-like
            dispatch_us: 10,
        },
        "Softmax" => Cost {
            regs: 32,
            smem_bytes: 4 * 256, // partial scratch
            mem_bytes: mem,
            flops: out_elems * 4, // exp + sum + div
            dispatch_us: 10,
        },

        // ── Heavy: Matmul / QM ──
        // K dim (reduction axis) = input[0]'s last dim. For A@B with
        // A of shape (..., M, K), B of shape (K, N), output (..., M, N).
        "Matmul" | "AddMM" => {
            let k_dim = op
                .inputs
                .first()
                .and_then(|&t| module_last_dim(module, t))
                .unwrap_or(1)
                .max(1);
            Cost {
                regs: 64,
                smem_bytes: 0,
                mem_bytes: mem,
                flops: 2 * out_elems * k_dim,
                dispatch_us: 10,
            }
        }
        "QuantizedMatmul" => {
            let k_dim = op
                .inputs
                .first()
                .and_then(|&t| module_last_dim(module, t))
                .unwrap_or(1)
                .max(1);
            // Half the device-mem footprint of bf16 Matmul: q4 weight
            // is 4× compressed (per element) but x + biases stay f16.
            Cost {
                regs: 64,
                smem_bytes: 0,
                mem_bytes: mem / 2,
                flops: 2 * out_elems * k_dim,
                dispatch_us: 10,
            }
        }

        // ── Views: free at the kernel level (codegen aliasing) ──
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => Cost {
            regs: 8,
            smem_bytes: 0,
            mem_bytes: mem,
            flops: 0,
            dispatch_us: 10,
        },

        // ── Strided views: actually move bytes ──
        "Transpose" => match attrs {
            Attrs::Transpose { axes } => {
                let is_identity = axes.iter().enumerate().all(|(i, &a)| a as usize == i);
                Cost {
                    regs: 8,
                    smem_bytes: 0,
                    mem_bytes: if is_identity { 0 } else { mem },
                    flops: 0,
                    dispatch_us: if is_identity { 0 } else { 10 },
                }
            }
            _ => Cost {
                regs: 8,
                smem_bytes: 0,
                mem_bytes: mem,
                flops: 0,
                dispatch_us: 10,
            },
        },

        "Broadcast" => Cost {
            regs: 8,
            smem_bytes: 0,
            mem_bytes: out_elems * 2, // write only — read is stride-0 cached
            flops: 0,
            dispatch_us: 10,
        },

        // Unknown primitive — conservative default.
        _ => Cost {
            regs: 32,
            smem_bytes: 0,
            mem_bytes: mem,
            flops: out_elems,
            dispatch_us: 10,
        },
    }
}

fn synthetic_cost(name: &str) -> Cost {
    // Synthetic kernels' cost roughly tracks their workload. Use
    // string match — these are the small fixed set of fusekit
    // synthetic kernels.
    if name.contains("Sdpa") {
        Cost {
            regs: 48,
            smem_bytes: 16 * 1024, // scores + reduce
            mem_bytes: 8 * 1024,
            flops: 16_384,
            dispatch_us: 15,
        }
    } else if name.contains("Rope") {
        Cost {
            regs: 24,
            smem_bytes: 0,
            mem_bytes: 1024,
            flops: 512,
            dispatch_us: 10,
        }
    } else if name.contains("EmbedLookup") {
        Cost {
            regs: 8,
            smem_bytes: 0,
            mem_bytes: 2 * 1024,
            flops: 0,
            dispatch_us: 10,
        }
    } else if name.contains("Argmax") {
        Cost {
            regs: 16,
            smem_bytes: 1024,
            mem_bytes: 300 * 1024,
            flops: 150_000,
            dispatch_us: 10,
        }
    } else {
        Cost {
            regs: 32,
            smem_bytes: 0,
            mem_bytes: 0,
            flops: 0,
            dispatch_us: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attrs, Dtype};
    use crate::op::module::{Module, Tensor};
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
    fn matmul_flops_track_shape() {
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 896]),   // x: (1, 896)
            mk_tensor(1, vec![896, 896]), // w: (896, 896)
            mk_tensor(2, vec![1, 896]),   // out: (1, 896)
        ];
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
        let c = estimate_op_cost(&op, &m);
        // out_elems = 896, in_elems = 1*896 + 896*896 = 803712
        // k_dim ≈ in/out = 803712/896 ≈ 896
        // flops = 2 * 896 * 896 = 1,605,632
        // Within a factor of 2 of the analytical (2*M*N*K) value.
        assert!(c.flops > 1_000_000);
        assert!(c.flops < 4_000_000);
    }

    #[test]
    fn quantized_matmul_lower_memory_than_dense() {
        let mut m = Module::new();
        m.tensors = vec![
            mk_tensor(0, vec![1, 896]),
            mk_tensor(1, vec![896, 896]),
            mk_tensor(2, vec![1, 896]),
        ];
        let matmul = Op {
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
        let qmm = Op {
            id: 1,
            kind: OpKind::Primitive {
                name: "QuantizedMatmul".into(),
                raw_attrs: Attrs::Unknown,
            },
            op_ids: vec![],
            inputs: vec![0, 1],
            outputs: vec![2],
            attrs: Default::default(),
            layer_hint: None,
        };
        let dense = estimate_op_cost(&matmul, &m);
        let quant = estimate_op_cost(&qmm, &m);
        assert!(quant.mem_bytes < dense.mem_bytes);
    }

    #[test]
    fn identity_transpose_costs_zero() {
        let mut m = Module::new();
        m.tensors = vec![mk_tensor(0, vec![2, 3, 4]), mk_tensor(1, vec![2, 3, 4])];
        let op = Op {
            id: 0,
            kind: OpKind::Primitive {
                name: "Transpose".into(),
                raw_attrs: Attrs::Transpose {
                    axes: vec![0, 1, 2],
                },
            },
            op_ids: vec![],
            inputs: vec![0],
            outputs: vec![1],
            attrs: Default::default(),
            layer_hint: None,
        };
        let c = estimate_op_cost(&op, &m);
        assert_eq!(c.mem_bytes, 0);
        assert_eq!(c.dispatch_us, 0);
    }
}
