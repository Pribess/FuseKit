//! Pattern: `Matmul (Q·Kᵀ) → Select (causal mask) → Softmax → Matmul (·V)`.
//!
//! ```text
//!                                          (mask cond)    (-inf bcast)
//!                                              │              │
//!   Q' ─►┐                                     ▼              │
//!        ├─► Matmul ─► raw_scores ─► Select ◄──┘              │
//!   Kᵀ ─►┘                              ▲   ◄─────────────────┘
//!                                       │
//!                                       ▼
//!                                    masked ─► Softmax ─► attn ─► Matmul ─► out
//!                                                                    ▲
//!                                                                    V
//! ```
//!
//! Anchored on `Softmax` since there's exactly one per attention block.
//! Scale multiplication on Q happens upstream as a separate Multiply
//! op which we deliberately leave outside the fusion — the kernel
//! re-applies `scale = 1/√head_dim` internally so the wrapper can
//! pass *unscaled* Q.

use crate::fusion::ir::{
    AttentionMask, AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Graph, OpId, TensorId};

pub struct SdpaDecode;

impl Pattern for SdpaDecode {
    fn name(&self) -> &'static str {
        "sdpa_decode"
    }

    fn priority(&self) -> u32 {
        // Above linear_residual (60) so its Matmul children don't get
        // mistaken for LinearResidual matmuls first.
        80
    }

    fn is_structural(&self) -> bool {
        // SDPA decode is structural: the general-fusion path has no
        // multi-Heavy chain emitter for the Q·Kᵀ → softmax → ·V shape
        // yet, so this recogniser stays on under FUSEKIT_NO_PATTERNS=1
        // to keep attention lowering correct on the compiler path.
        true
    }

    fn try_match(&self, g: &Graph, ud: &UseDef, anchor: OpId) -> Option<Match> {
        let sm = g.op(anchor);
        if sm.name != "Softmax" {
            return None;
        }
        if sm.inputs.len() != 1 || sm.outputs.len() != 1 {
            return None;
        }
        let sm_in = sm.inputs[0];
        let sm_out = sm.outputs[0];

        // ── Walk back: Select (causal mask) ────────────────────────
        // mlx-lm builds causal attention via
        //   masked = where(causal_mask, scores, -inf)
        // which lowers to a Select primitive with three inputs:
        // condition, then-value (Matmul output), else-value (broadcast
        // of -inf). The Softmax consumes the Select output.
        let sel_id = g.tensor(sm_in).producer?;
        let sel = g.op(sel_id);
        if sel.name != "Select" || sel.inputs.len() != 3 || sel.outputs.len() != 1 {
            return None;
        }
        if ud.consumers[sel.outputs[0] as usize].as_slice() != [anchor] {
            return None;
        }
        // Identify which Select input is the Matmul-side branch (the
        // "then" value). The other two are broadcast tensors.
        let matmul_side = sel.inputs.iter().position(|&t| {
            g.tensor(t)
                .producer
                .map(|p| g.op(p).name == "Matmul")
                .unwrap_or(false)
        })?;

        // ── Walk back further: Matmul (Q · Kᵀ) ─────────────────────
        let qk_id = g.tensor(sel.inputs[matmul_side]).producer?;
        let qk = g.op(qk_id);
        if qk.name != "Matmul" || qk.inputs.len() != 2 || qk.outputs.len() != 1 {
            return None;
        }
        if ud.consumers[qk.outputs[0] as usize].as_slice() != [sel_id] {
            return None;
        }
        let q_tid = qk.inputs[0];
        let kt_tid = qk.inputs[1];
        let q_already_scaled = g
            .tensor(q_tid)
            .producer
            .map(|p| g.op(p).name == "Multiply")
            .unwrap_or(false);

        // ── Walk forward: Matmul (attn · V) ───────────────────────
        let av_consumers = &ud.consumers[sm_out as usize];
        if av_consumers.len() != 1 {
            return None;
        }
        let av_id = av_consumers[0];
        let av = g.op(av_id);
        if av.name != "Matmul" || av.inputs.len() != 2 || av.outputs.len() != 1 {
            return None;
        }
        let attn_side = av.inputs.iter().position(|&t| t == sm_out)?;
        let v_side = 1 - attn_side;
        let v_tid = av.inputs[v_side];
        let out_tid = av.outputs[0];

        let bnd = |tid: TensorId, role: &'static str| -> FusionBoundary {
            let t = &g.tensors[tid as usize];
            FusionBoundary {
                source: tid,
                role,
                shape: t.shape.clone(),
                dtype: t.dtype,
            }
        };

        // Heads are the natural parallel axis: each query head is
        // independent across kv_len.
        let out_shape = &g.tensors[out_tid as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "head",
            size: *out_shape.first().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: 0,
            },
        }];

        Some(Match {
            kind: FusionKind::SdpaDecode,
            op_ids: vec![qk_id, sel_id, anchor, av_id],
            inputs: vec![bnd(q_tid, "Q"), bnd(kt_tid, "Kt"), bnd(v_tid, "V")],
            outputs: vec![bnd(out_tid, "out")],
            attrs: FusionAttrs::SdpaDecode {
                // Some MLX traces lower `scale=` as a pre-multiply on Q
                // before the QK matmul. In that case the fused SDPA must
                // not apply another 1/sqrt(dim). A zero sentinel keeps the
                // historical emitter fallback for unscaled Q traces.
                scale: if q_already_scaled { 1.0 } else { 0.0 },
                mask: AttentionMask::Causal,
            },
            parallel_axes,
        })
    }
}
