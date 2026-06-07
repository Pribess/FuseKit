//! Pattern: `RMSNorm → Matmul` (or `RMSNorm → AddMM`).
//!
//! ```text
//!   x ─►┌─────────┐         ┌────────┐
//!       │ RMSNorm │─► t ───►│ Matmul │─► y
//!       └─────────┘         └────────┘
//!         W_norm              W_proj
//! ```
//!
//! Fuses a pre-projection norm into the projection that consumes it.
//! Saves the norm output round-trip through global memory: codegen
//! keeps `t` in threadgroup memory and feeds the matmul directly.
//!
//! Match rules:
//! - Anchor op is `RMSNorm`.
//! - Its output tensor has exactly one consumer (otherwise we'd be
//!   re-materializing or memoizing, which we don't support).
//! - That consumer is `Matmul` or `AddMM`, and the norm output is the
//!   matmul's first operand (the activation side, not the weight).

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis,
};
use crate::fusion::patterns::{follow_views, Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId};

pub struct RmsNormProj;

impl Pattern for RmsNormProj {
    fn name(&self) -> &'static str {
        "rmsnorm_proj"
    }

    fn priority(&self) -> u32 {
        // Smaller than rmsnorm_qkv (3-way) so the wider pattern wins
        // when it applies.
        50
    }

    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match> {
        let rms = g.op(anchor);
        if rms.name != "RMSNorm" {
            return None;
        }
        let Attrs::RmsNorm { eps } = rms.attrs else {
            return None;
        };

        // RMSNorm always has (x, norm_weight) as inputs.
        if rms.inputs.len() != 2 || rms.outputs.len() != 1 {
            return None;
        }
        let norm_out = rms.outputs[0];

        // Walk through view-only ops (Reshape/Flatten/...) to find the
        // first real compute consumer. The view chain must be linear
        // (each step single-consumer) — fork means we can't fuse
        // without re-materializing intermediates.
        let walk = follow_views(g, ud, norm_out)?;
        let mm_id = walk.target;
        let view_out = walk.feeding_tensor;
        let mm = g.op(mm_id);

        // Match Matmul (y = x · W) or AddMM (y = α · x · W + β · b).
        // Both put the activation at input[0] in MLX's primitive
        // ordering — AddMM is (x, W, bias).
        let is_addmm = mm.name == "AddMM";
        let is_matmul = mm.name == "Matmul";
        if !is_matmul && !is_addmm {
            return None;
        }
        if mm.inputs.first() != Some(&view_out) {
            return None;
        }

        let x_t = rms.inputs[0];
        let norm_weight_t = rms.inputs[1];
        let proj_weight_t = *mm.inputs.get(1)?;
        let bias_t = if is_addmm {
            mm.inputs.get(2).copied()
        } else {
            None
        };
        let out_t = mm.outputs[0];

        let bnd = |tid, role| -> FusionBoundary {
            let t = &g.tensors[tid as usize];
            FusionBoundary {
                source: tid,
                role,
                shape: t.shape.clone(),
                dtype: t.dtype,
            }
        };

        let mut inputs = vec![
            bnd(x_t, "x"),
            bnd(norm_weight_t, "norm_weight"),
            bnd(proj_weight_t, "proj_weight"),
        ];
        if let Some(b) = bias_t {
            inputs.push(bnd(b, "bias"));
        }

        // The projection's output rows can be split across threadgroups
        // without inter-task communication — natural parallel axis.
        let out_shape = &g.tensors[out_t as usize].shape;
        let out_rows_dim = out_shape.len().saturating_sub(1) as u32;
        let out_rows = *out_shape.last().unwrap_or(&1) as u32;
        let parallel_axes = vec![ParallelAxis {
            name: "out_row",
            size: out_rows,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: out_rows_dim,
            },
        }];

        // Consume the anchor, any view ops we walked through, and the
        // projection itself. Without listing the views here, the build
        // pass would re-emit them as Single fusions.
        let mut op_ids = vec![anchor];
        op_ids.extend(walk.views.iter().copied());
        op_ids.push(mm_id);

        Some(Match {
            kind: FusionKind::RmsNormProj,
            op_ids,
            inputs,
            outputs: vec![bnd(out_t, "y")],
            attrs: FusionAttrs::RmsNormProj { eps, quant: None },
            parallel_axes,
        })
    }
}
