//! Pattern: `Sigmoid → Multiply → Multiply` ≡ SwiGLU activation.
//!
//! ```text
//!   gate ─► Sigmoid ─► s
//!                       \
//!   gate ────────────► Multiply ─► silu = sigmoid(gate) · gate
//!                                       │
//!   up   ──────────► Multiply ◄─────────┘
//!                       │
//!                       ▼
//!                  swiglu_out = silu · up
//! ```
//!
//! Anchored on `Sigmoid` to keep the matcher simple. Walks two
//! `Multiply` ops downstream and requires each step's intermediate to
//! have a single consumer so we can keep them in registers/SMEM.

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Graph, OpId, TensorId};

pub struct SwigluActivation;

impl Pattern for SwigluActivation {
    fn name(&self) -> &'static str {
        "swiglu_activation"
    }

    fn priority(&self) -> u32 {
        70 // Above linear_residual so its Multiply consumers don't get
           // misinterpreted as something else first.
    }

    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match> {
        let sig = g.op(anchor);
        if sig.name != "Sigmoid" {
            return None;
        }
        if sig.inputs.len() != 1 || sig.outputs.len() != 1 {
            return None;
        }
        let gate_tid = sig.inputs[0];
        let sig_out = sig.outputs[0];

        // Sigmoid output feeds the first Multiply directly.
        let mul1_id = single_consumer(ud, sig_out)?;
        let mul1 = g.op(mul1_id);
        if mul1.name != "Multiply" || mul1.inputs.len() != 2 {
            return None;
        }
        // The other operand of Multiply 1 must be the original gate.
        let other_mul1 = if mul1.inputs[0] == sig_out {
            mul1.inputs[1]
        } else if mul1.inputs[1] == sig_out {
            mul1.inputs[0]
        } else {
            return None;
        };
        if other_mul1 != gate_tid {
            return None;
        }
        let silu_out = mul1.outputs[0];

        // Multiply 1 output feeds Multiply 2. Other operand of mul2 is
        // the up_proj output — that's the external `up` input to the
        // fusion.
        let mul2_id = single_consumer(ud, silu_out)?;
        let mul2 = g.op(mul2_id);
        if mul2.name != "Multiply" || mul2.inputs.len() != 2 {
            return None;
        }
        let up_tid = if mul2.inputs[0] == silu_out {
            mul2.inputs[1]
        } else if mul2.inputs[1] == silu_out {
            mul2.inputs[0]
        } else {
            return None;
        };
        let out_tid = mul2.outputs[0];

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

        let out_shape = &g.tensors[out_tid as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "elementwise",
            size: out_shape.iter().product::<i32>() as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: 0,
            },
        }];

        Some(Match {
            kind: FusionKind::SwigluActivation,
            op_ids: vec![anchor, mul1.id, mul2.id],
            inputs: vec![bnd(gate_tid, "gate"), bnd(up_tid, "up")],
            outputs: vec![bnd(out_tid, "out")],
            attrs: FusionAttrs::SwigluActivation,
            parallel_axes,
        })
    }
}

fn single_consumer(ud: &UseDef, tid: TensorId) -> Option<OpId> {
    let cs = &ud.consumers[tid as usize];
    if cs.len() == 1 {
        Some(cs[0])
    } else {
        None
    }
}
