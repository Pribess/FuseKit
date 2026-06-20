//! Pattern: `Matmul → Add(residual)`.
//!
//! ```text
//!   x ─► Matmul ─► t ─► Add(t, residual) ─► y
//!         W            ▲
//!                      │
//!         (external residual tensor, e.g. attention/MLP input h)
//! ```
//!
//! Covers `o_proj + attn_residual` and `down_proj + mlp_residual` in
//! Qwen-style transformers — both are `y = x @ W + h`. The matmul
//! must have a single downstream consumer that is an `Add` with
//! exactly two operands (matmul output + external residual).

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis, QuantInfo,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId};

pub struct LinearResidual;

impl Pattern for LinearResidual {
    fn name(&self) -> &'static str {
        "linear_residual"
    }

    fn priority(&self) -> u32 {
        60
    }

    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match> {
        let mm = g.op(anchor);
        // Accept both regular Matmul (bf16 path) and QuantizedMatmul
        // (mlx affine int4/int8 path). QuantizedMatmul carries scales
        // and biases as additional inputs.
        let is_q = mm.name == "QuantizedMatmul";
        let is_p = mm.name == "Matmul";
        if !is_q && !is_p {
            return None;
        }
        let expected_inputs = if is_q { 4 } else { 2 };
        if mm.inputs.len() != expected_inputs || mm.outputs.len() != 1 {
            return None;
        }
        let mm_out = mm.outputs[0];

        let walk = crate::fusion::patterns::follow_views(g, ud, mm_out)?;
        let add = g.op(walk.target);
        if add.name != "Add" {
            return None;
        }
        if add.inputs.len() != 2 || add.outputs.len() != 1 {
            return None;
        }

        let mm_feed = walk.feeding_tensor;
        let residual_side_idx = if add.inputs[0] == mm_feed {
            1
        } else if add.inputs[1] == mm_feed {
            0
        } else {
            return None;
        };
        let residual_tid = add.inputs[residual_side_idx];

        let bnd = |tid, role: &'static str| -> FusionBoundary {
            let t = &g.tensors[tid as usize];
            FusionBoundary {
                source: tid,
                role,
                shape: t.shape.clone(),
                dtype: t.dtype,
                scalar_value: t.scalar_value.clone(),
            }
        };

        let x_t = mm.inputs[0];
        // mlx-lm's `Linear` lowers `x @ W.T` to `Matmul(x, Transpose(W))`
        // when W is dense (bf16/fp16). The trace materializes the
        // Transpose; absorbing it here lets us bind the original
        // `[out, in]` weight that the kernel's emit indexing assumes.
        // QuantizedMatmul keeps W in `[out, in]` natively, so this is
        // a no-op for the quantized path.
        let (w_t, transpose_id) =
            crate::fusion::patterns::unwrap_weight_transpose(g, ud, mm.inputs[1]);
        let out_t = add.outputs[0];

        let out_shape = &g.tensors[out_t as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "out_row",
            size: *out_shape.last().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: out_shape.len().saturating_sub(1) as u32,
            },
        }];

        let mut op_ids = vec![anchor];
        if let Some(tid) = transpose_id {
            op_ids.push(tid);
        }
        op_ids.extend(walk.views.iter().copied());
        op_ids.push(add.id);

        // Build inputs: (x, W, [scales, biases for quantized,] residual).
        // Boundary `role` tells codegen / Python which is which.
        let mut inputs = vec![bnd(x_t, "x"), bnd(w_t, "W")];
        let quant = if is_q {
            inputs.push(bnd(mm.inputs[2], "W_scales"));
            inputs.push(bnd(mm.inputs[3], "W_biases"));
            // Quantization params land on the op via Attrs::Quantized.
            // Default to (4, 64) — mlx's standard — if attrs unavailable.
            let (bits, group_size) = match mm.attrs {
                Attrs::Quantized { bits, group_size } => (bits, group_size),
                _ => (4, 64),
            };
            Some(QuantInfo { bits, group_size })
        } else {
            None
        };
        inputs.push(bnd(residual_tid, "residual"));

        Some(Match {
            kind: FusionKind::LinearResidual,
            op_ids,
            inputs,
            outputs: vec![bnd(out_t, "y")],
            attrs: FusionAttrs::LinearResidual { quant },
            parallel_axes,
        })
    }
}
