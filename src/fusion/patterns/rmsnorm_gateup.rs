//! Pattern: `RMSNorm → view → (gate, up) projections`.
//!
//! ```text
//!   x ─►┌─────────┐  view  ─►┬─► Matmul ─► gate
//!       │ RMSNorm │          │   (W_gate)
//!       └─────────┘          └─► Matmul ─► up
//!         W_norm                  (W_up)
//! ```
//!
//! MLP preludes in Llama/Qwen-style transformers split the normalized
//! activation into two projections that then feed SwiGLU. This is the
//! same shape as `RmsNormQkv` but with fan-out = 2 instead of 3, so
//! it lives in its own pattern to keep the downstream `FusionKind`
//! distinct (different codegen — feeds into SwiGLU MLP, not attention).

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId, TensorId};

pub struct RmsNormGateUp;

impl Pattern for RmsNormGateUp {
    fn name(&self) -> &'static str {
        "rmsnorm_gateup"
    }

    fn priority(&self) -> u32 {
        90 // Between rmsnorm_qkv (100) and rmsnorm_proj (50).
    }

    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match> {
        let rms = g.op(anchor);
        if rms.name != "RMSNorm" {
            return None;
        }
        let Attrs::RmsNorm { eps } = rms.attrs else {
            return None;
        };
        if rms.inputs.len() != 2 || rms.outputs.len() != 1 {
            return None;
        }
        let norm_out = rms.outputs[0];

        let (feeding_tensor, view_ops) = walk_until_fanout(g, ud, norm_out)?;
        let cs = &ud.consumers[feeding_tensor as usize];
        if cs.len() != 2 {
            return None;
        }

        let mut projs: Vec<(OpId, TensorId, TensorId)> = Vec::with_capacity(2);
        for &cid in cs {
            let op = g.op(cid);
            // MLP preludes use plain Matmul (no bias) in Qwen-style
            // models; reject AddMM here to keep the kind semantics
            // tight. If a future model uses biased projections we'll
            // relax this in a separate pattern.
            if op.name != "Matmul" {
                return None;
            }
            if op.inputs.first() != Some(&feeding_tensor) {
                return None;
            }
            let weight = *op.inputs.get(1)?;
            let output = *op.outputs.first()?;
            projs.push((cid, weight, output));
        }

        let bnd = |tid: TensorId, role: &'static str| -> FusionBoundary {
            let t = &g.tensors[tid as usize];
            FusionBoundary {
                source: tid,
                role,
                shape: t.shape.clone(),
                dtype: t.dtype,
                scalar_value: t.scalar_value.clone(),
            }
        };

        let inputs = vec![
            bnd(rms.inputs[0], "x"),
            bnd(rms.inputs[1], "norm_weight"),
            bnd(projs[0].1, "W_gate"),
            bnd(projs[1].1, "W_up"),
        ];
        let outputs = vec![bnd(projs[0].2, "gate"), bnd(projs[1].2, "up")];

        let gate_shape = &g.tensors[projs[0].2 as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "out_row",
            size: *gate_shape.last().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: gate_shape.len().saturating_sub(1) as u32,
            },
        }];

        let mut op_ids = vec![anchor];
        op_ids.extend(view_ops);
        op_ids.extend(projs.iter().map(|(id, _, _)| *id));

        Some(Match {
            kind: FusionKind::RmsNormGateUp,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::RmsNormGateUp { eps, quant: None },
            parallel_axes,
        })
    }
}

fn walk_until_fanout(g: &Graph, ud: &UseDef, start: TensorId) -> Option<(TensorId, Vec<OpId>)> {
    let mut tensor = start;
    let mut views = Vec::new();
    loop {
        let cs = &ud.consumers[tensor as usize];
        if cs.len() != 1 {
            return Some((tensor, views));
        }
        let next = g.op(cs[0]);
        if !crate::fusion::patterns::is_view_op(&next.name) {
            return Some((tensor, views));
        }
        if next.outputs.len() != 1 {
            return None;
        }
        views.push(next.id);
        tensor = next.outputs[0];
    }
}
