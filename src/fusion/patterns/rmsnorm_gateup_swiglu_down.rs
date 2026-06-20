//! Pattern: full post-attn MLP block in one kernel.
//!
//! Extends `rmsnorm_gateup_swiglu` by also absorbing the downstream
//! `down_proj` Matmul and its residual Add. The SwiGLU output stays
//! in threadgroup memory and feeds the down-projection's reduction
//! axis, eliminating an N_INT·bf16 DRAM round-trip per layer.
//!
//! ```text
//!   x ─►┬──────────────────────────────────────────────┐    (residual)
//!       │                                              │
//!       └► RMSNorm ─► view ─► fan ─► gate · up ─► swiglu ─► Matmul ──► Add ─► out
//!                                       │              ▲       (W_down)   ▲
//!                                       └─SMEM resident┘                  │
//!                                                                          │
//!                                                                     x ───┘
//! ```
//!
//! Priority above `rmsnorm_gateup_swiglu` (110) so the wider match wins
//! when the down_proj + residual chain is present.

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId, TensorId};

pub struct RmsNormGateUpSwigluDown;

impl Pattern for RmsNormGateUpSwigluDown {
    fn name(&self) -> &'static str {
        "rmsnorm_gateup_swiglu_down"
    }

    fn priority(&self) -> u32 {
        130
    }

    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match> {
        // Reuse the gateup_swiglu walk verbatim, then continue past the
        // mul2 (swiglu) output into Matmul → Add.
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
        let x_tid = rms.inputs[0];
        let norm_w_tid = rms.inputs[1];
        let norm_out = rms.outputs[0];

        let (feeding_tensor, view_ops) = walk_until_fanout(g, ud, norm_out)?;
        let cs = &ud.consumers[feeding_tensor as usize];
        if cs.len() != 2 {
            return None;
        }
        let mut projs: Vec<(OpId, TensorId, TensorId)> = Vec::with_capacity(2);
        for &cid in cs {
            let op = g.op(cid);
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
        let (a_visible, a_views) = walk_until_fanout(g, ud, projs[0].2)?;
        let (b_visible, b_views) = walk_until_fanout(g, ud, projs[1].2)?;
        let projs_visible = [
            (projs[0].0, projs[0].1, projs[0].2, a_visible, a_views),
            (projs[1].0, projs[1].1, projs[1].2, b_visible, b_views),
        ];
        let (gate_idx, up_idx) = identify_gate_up_visible(ud, &projs_visible)?;
        let (gate_op_id, w_gate, _, gate_visible, gate_view_chain) =
            projs_visible[gate_idx].clone();
        let (up_op_id, w_up, _, up_visible, up_view_chain) = projs_visible[up_idx].clone();

        let (sig_id, mul1_id) = find_sigmoid_and_mul1(g, ud, gate_visible)?;
        let sig = g.op(sig_id);
        let mul1 = g.op(mul1_id);
        if sig.inputs.len() != 1 || sig.outputs.len() != 1 || sig.inputs[0] != gate_visible {
            return None;
        }
        let sig_out = sig.outputs[0];
        if mul1.name != "Multiply" || mul1.inputs.len() != 2 {
            return None;
        }
        let mul1_inputs_match = (mul1.inputs[0] == gate_visible && mul1.inputs[1] == sig_out)
            || (mul1.inputs[0] == sig_out && mul1.inputs[1] == gate_visible);
        if !mul1_inputs_match {
            return None;
        }
        let silu_out = mul1.outputs[0];

        let mul2_id = single_consumer(ud, silu_out)?;
        let mul2 = g.op(mul2_id);
        if mul2.name != "Multiply" || mul2.inputs.len() != 2 || mul2.outputs.len() != 1 {
            return None;
        }
        let other = if mul2.inputs[0] == silu_out {
            mul2.inputs[1]
        } else if mul2.inputs[1] == silu_out {
            mul2.inputs[0]
        } else {
            return None;
        };
        if other != up_visible {
            return None;
        }
        let swiglu_out = mul2.outputs[0];

        // ── Continue: Matmul (down_proj) ─────────────────────────
        // SwiGLU output (or a single view of it) feeds the down_proj
        // Matmul. Walk one optional view step.
        let (down_input, swiglu_view_chain) = walk_until_fanout(g, ud, swiglu_out)?;
        let down_matmul_id = single_consumer(ud, down_input)?;
        let down_mm = g.op(down_matmul_id);
        if down_mm.name != "Matmul" || down_mm.inputs.len() != 2 || down_mm.outputs.len() != 1 {
            return None;
        }
        if down_mm.inputs.first() != Some(&down_input) {
            return None;
        }
        let w_down = down_mm.inputs[1];
        let down_out = down_mm.outputs[0];

        // ── Continue: Add (residual) ────────────────────────────
        // Down-projection result may also pass through view ops before
        // the residual add (MLX often reshapes back to (1, L, D)).
        let (add_input, down_view_chain) = walk_until_fanout(g, ud, down_out)?;
        let add_id = single_consumer(ud, add_input)?;
        let add = g.op(add_id);
        if add.name != "Add" || add.inputs.len() != 2 || add.outputs.len() != 1 {
            return None;
        }
        // The other Add input must be `x` (or its view) — the same
        // tensor that fed the RMSNorm. We accept either a direct ref
        // or a view-chained ref; this keeps the matcher robust to
        // mlx-lm's reshape habits.
        let other_add = if add.inputs[0] == add_input {
            add.inputs[1]
        } else if add.inputs[1] == add_input {
            add.inputs[0]
        } else {
            return None;
        };
        if !tensors_match_through_views(g, ud, other_add, x_tid) {
            return None;
        }
        let out_tid = add.outputs[0];

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
            bnd(x_tid, "x"),
            bnd(norm_w_tid, "norm_weight"),
            bnd(w_gate, "W_gate"),
            bnd(w_up, "W_up"),
            bnd(w_down, "W_down"),
        ];
        let outputs = vec![bnd(out_tid, "out")];

        let out_shape = &g.tensors[out_tid as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "d",
            size: *out_shape.last().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: out_shape.len().saturating_sub(1) as u32,
            },
        }];

        let mut op_ids = vec![anchor];
        op_ids.extend(view_ops);
        op_ids.push(gate_op_id);
        op_ids.push(up_op_id);
        op_ids.extend(gate_view_chain);
        op_ids.extend(up_view_chain);
        op_ids.push(sig_id);
        op_ids.push(mul1_id);
        op_ids.push(mul2_id);
        op_ids.extend(swiglu_view_chain);
        op_ids.push(down_matmul_id);
        op_ids.extend(down_view_chain);
        op_ids.push(add_id);

        Some(Match {
            kind: FusionKind::RmsNormGateUpSwigluDown,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::RmsNormGateUpSwigluDown { eps, quant: None },
            parallel_axes,
        })
    }
}

/// Check that two tensors refer to the same underlying source, allowing
/// either to be wrapped in a chain of view ops. Used for the residual-
/// branch input where MLX may reshape `x` independently.
fn tensors_match_through_views(g: &Graph, _ud: &UseDef, a: TensorId, b: TensorId) -> bool {
    fn root(g: &Graph, mut tid: TensorId) -> TensorId {
        loop {
            let producer = match g.tensor(tid).producer {
                Some(p) => p,
                None => return tid,
            };
            let op = g.op(producer);
            if crate::fusion::patterns::is_view_op(&op.name) && !op.inputs.is_empty() {
                tid = op.inputs[0];
            } else {
                return tid;
            }
        }
    }
    root(g, a) == root(g, b)
}

fn identify_gate_up_visible(
    ud: &UseDef,
    projs: &[(OpId, TensorId, TensorId, TensorId, Vec<OpId>)],
) -> Option<(usize, usize)> {
    let n_consumers = |tid: TensorId| ud.consumers[tid as usize].len();
    let (a_visible, b_visible) = (projs[0].3, projs[1].3);
    match (n_consumers(a_visible), n_consumers(b_visible)) {
        (2, 1) => Some((0, 1)),
        (1, 2) => Some((1, 0)),
        _ => None,
    }
}

fn find_sigmoid_and_mul1(g: &Graph, ud: &UseDef, gate_out: TensorId) -> Option<(OpId, OpId)> {
    let cs = &ud.consumers[gate_out as usize];
    if cs.len() != 2 {
        return None;
    }
    let (a, b) = (cs[0], cs[1]);
    let (na, nb) = (&g.op(a).name, &g.op(b).name);
    match (na.as_str(), nb.as_str()) {
        ("Sigmoid", "Multiply") => Some((a, b)),
        ("Multiply", "Sigmoid") => Some((b, a)),
        _ => None,
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
