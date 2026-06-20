//! Pattern: `RMSNorm → (gate, up) projections → SwiGLU` in one kernel.
//!
//! Extends `rmsnorm_gateup` by also absorbing the downstream
//! `Sigmoid → Multiply → Multiply` chain. Avoids writing `gate` and
//! `up` to global memory — both stay in per-thread registers, and the
//! kernel emits only the SwiGLU output.
//!
//! ```text
//!                                ┌─► Matmul ─► gate ─┬─► Sigmoid ─► s ─┐
//!   x ─► RMSNorm ─► view ─► fan ─┤                   │                  │
//!                                │                   └────► Mult ◄──────┘  ← s · gate
//!                                │                            │
//!                                │                       silu = sigmoid(gate)·gate
//!                                │                            │
//!                                └─► Matmul ─► up ────► Mult ◄┘  ← silu · up
//!                                                       │
//!                                                       ▼
//!                                                 swiglu_out
//! ```
//!
//! Priority is higher than `rmsnorm_gateup` and `swiglu_activation`
//! standalone so the cofuse wins when both are present.

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis, QuantInfo,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId, TensorId};

pub struct RmsNormGateUpSwiglu;

impl Pattern for RmsNormGateUpSwiglu {
    fn name(&self) -> &'static str {
        "rmsnorm_gateup_swiglu"
    }

    fn priority(&self) -> u32 {
        // Above rmsnorm_gateup (90) so the cofuse claims the subgraph
        // first when the downstream SwiGLU is present.
        110
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

        // RMSNorm → view chain → fan-out into 2 Matmuls.
        let (feeding_tensor, view_ops) = walk_until_fanout(g, ud, norm_out)?;
        let cs = &ud.consumers[feeding_tensor as usize];
        if cs.len() != 2 {
            return None;
        }

        // Each of the two consumers must be a Matmul or QuantizedMatmul
        // reading the normed activation as its first input.
        // (gate vs up distinguished below by downstream consumer pattern.)
        // Quant info captured per-projection; both must agree.
        // (op_id, weight, output, scales, biases) — keep .2 = output to
        // minimize churn in the SwiGLU chain that follows.
        let mut projs: Vec<(
            OpId,
            TensorId,
            TensorId,
            Option<TensorId>,
            Option<TensorId>,
            Option<OpId>,
        )> = Vec::with_capacity(2);
        let mut quant: Option<QuantInfo> = None;
        for &cid in cs {
            let op = g.op(cid);
            let is_q = op.name == "QuantizedMatmul";
            let is_p = op.name == "Matmul";
            if !is_q && !is_p {
                return None;
            }
            if op.inputs.first() != Some(&feeding_tensor) {
                return None;
            }
            let weight_raw = *op.inputs.get(1)?;
            // Bf16/fp16 path: trace contains `Transpose(W)` for the
            // `x @ W.T` lowering. Absorb the Transpose so the kernel's
            // `[out, in]` indexing is correct.
            let (weight, transpose_op) = if is_q {
                (weight_raw, None)
            } else {
                crate::fusion::patterns::unwrap_weight_transpose(g, ud, weight_raw)
            };
            let (scales, biases) = if is_q {
                let s = op.inputs.get(2).copied()?;
                let b = op.inputs.get(3).copied()?;
                let q = match op.attrs {
                    Attrs::Quantized { bits, group_size } => QuantInfo { bits, group_size },
                    _ => QuantInfo {
                        bits: 4,
                        group_size: 64,
                    },
                };
                if let Some(prev) = quant {
                    if prev.bits != q.bits || prev.group_size != q.group_size {
                        return None;
                    }
                }
                quant = Some(q);
                (Some(s), Some(b))
            } else {
                (None, None)
            };
            let output = *op.outputs.first()?;
            projs.push((cid, weight, output, scales, biases, transpose_op));
        }
        // All-or-nothing: don't mix quant + dense.
        let any_q = projs.iter().any(|p| p.3.is_some());
        let all_q = projs.iter().all(|p| p.3.is_some());
        if any_q && !all_q {
            return None;
        }
        if crate::fusion::patterns::no_patterns_enabled() && !all_q {
            return None;
        }

        // MLX inserts `Unflatten` view ops between each projection's
        // Matmul and the SwiGLU consumers. Walk through those so we
        // see the real fan-out points. `walk_until_fanout` returns the
        // first tensor with multi-consumer or non-view next op.
        let (a_visible, a_views) = walk_until_fanout(g, ud, projs[0].2)?;
        let (b_visible, b_views) = walk_until_fanout(g, ud, projs[1].2)?;
        let projs_visible = [
            (projs[0].0, projs[0].1, projs[0].2, a_visible, a_views),
            (projs[1].0, projs[1].1, projs[1].2, b_visible, b_views),
        ];

        // `gate` is the visible-tensor with 2 consumers (Sigmoid +
        // first Multiply). `up` is the one with 1 consumer (the second
        // Multiply). If neither side matches that shape, this is not a
        // SwiGLU — bail out and let the plain RmsNormGateUp matcher
        // claim it.
        let (gate_idx, up_idx) = identify_gate_up_visible(ud, &projs_visible)?;
        let (gate_op_id, w_gate, _gate_proj_out, gate_visible, gate_view_chain) =
            projs_visible[gate_idx].clone();
        let (up_op_id, w_up, _up_proj_out, up_visible, up_view_chain) =
            projs_visible[up_idx].clone();

        // Walk the SwiGLU chain anchored at `gate_visible`. Either of
        // its two consumers can be Sigmoid; the other must be the
        // first Multiply.
        let (sig_id, mul1_id) = find_sigmoid_and_mul1(g, ud, gate_visible)?;
        let sig = g.op(sig_id);
        let mul1 = g.op(mul1_id);
        if sig.inputs.len() != 1 || sig.outputs.len() != 1 {
            return None;
        }
        if sig.inputs[0] != gate_visible {
            return None;
        }
        let sig_out = sig.outputs[0];
        // Multiply 1: inputs are {gate_visible, sig_out}, in either order.
        if mul1.name != "Multiply" || mul1.inputs.len() != 2 {
            return None;
        }
        let mul1_inputs_match = (mul1.inputs[0] == gate_visible && mul1.inputs[1] == sig_out)
            || (mul1.inputs[0] == sig_out && mul1.inputs[1] == gate_visible);
        if !mul1_inputs_match {
            return None;
        }
        let silu_out = mul1.outputs[0];

        // Multiply 2: silu · up.
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
        let out_tid = mul2.outputs[0];

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

        let mut inputs = vec![
            bnd(rms.inputs[0], "x"),
            bnd(rms.inputs[1], "norm_weight"),
            bnd(w_gate, "W_gate"),
            bnd(w_up, "W_up"),
        ];
        if all_q {
            // Push gate/up scales then gate/up biases — matches the
            // kernel buffer layout. We have to re-fetch the scales/biases
            // tensors from projs by matching against w_gate/w_up.
            let (gate_s, gate_b) = projs
                .iter()
                .find(|p| p.1 == w_gate)
                .map(|p| (p.3.unwrap(), p.4.unwrap()))
                .unwrap();
            let (up_s, up_b) = projs
                .iter()
                .find(|p| p.1 == w_up)
                .map(|p| (p.3.unwrap(), p.4.unwrap()))
                .unwrap();
            inputs.push(bnd(gate_s, "W_gate_scales"));
            inputs.push(bnd(up_s, "W_up_scales"));
            inputs.push(bnd(gate_b, "W_gate_biases"));
            inputs.push(bnd(up_b, "W_up_biases"));
        }
        let outputs = vec![bnd(out_tid, "out")];

        // The intermediate-feature dimension (`N_INT`) is the natural
        // parallel axis — each output element is independent.
        let out_shape = &g.tensors[out_tid as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "n_int",
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
        // Absorb the Transpose(W) ops the bf16/fp16 lowering inserts.
        for p in &projs {
            if let Some(tid) = p.5 {
                op_ids.push(tid);
            }
        }

        Some(Match {
            kind: FusionKind::RmsNormGateUpSwiglu,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::RmsNormGateUpSwiglu { eps, quant },
            parallel_axes,
        })
    }
}

/// Decide which of the two parallel projections is `gate` (the one
/// whose post-view tensor feeds 2 consumers — Sigmoid + first
/// Multiply) vs `up` (post-view tensor feeds 1 — second Multiply).
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

/// Among the two downstream consumers of `gate_out`, find which is
/// Sigmoid and which is the first Multiply.
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
