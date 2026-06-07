//! Pattern: final `RMSNorm → Matmul` (LM-head projection).
//!
//! ```text
//!   hidden ─► RMSNorm ─► view ─► Matmul ─► logits
//!              W_norm              W_lm_head
//! ```
//!
//! Anchored on `RMSNorm`. Distinguishes the final norm from per-layer
//! norms by checking that the post-view tensor has exactly ONE
//! consumer (the LM-head Matmul). Per-layer pre-attn / pre-MLP norms
//! fan out into 2–3 projection Matmuls, so they fail the single-
//! consumer check and fall through to the dedicated patterns.

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis, QuantInfo,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId, TensorId};

pub struct FinalRmsNormLmHead;

impl Pattern for FinalRmsNormLmHead {
    fn name(&self) -> &'static str {
        "final_rmsnorm_lm_head"
    }

    fn priority(&self) -> u32 {
        // Below per-layer norm patterns (rmsnorm_qkv = 100, mlp blocks
        // 110–130) so those win when they would match. The final-norm
        // anchor only matches when none of those structural shapes
        // are present downstream.
        50
    }

    fn is_structural(&self) -> bool {
        false
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
        let x_tid = rms.inputs[0];
        let norm_w_tid = rms.inputs[1];
        let norm_out = rms.outputs[0];

        // Walk through view ops. Final-norm's output goes
        // RMSNorm → Flatten → Matmul (one consumer).
        let (feed, view_ops) = walk_until_fanout(g, ud, norm_out)?;
        let cs = &ud.consumers[feed as usize];
        if cs.len() != 1 {
            return None;
        }
        let mm_id = cs[0];
        let mm = g.op(mm_id);
        let is_q = mm.name == "QuantizedMatmul";
        let is_p = mm.name == "Matmul";
        if !is_q && !is_p {
            return None;
        }
        let expected_inputs = if is_q { 4 } else { 2 };
        if mm.inputs.len() != expected_inputs || mm.outputs.len() != 1 {
            return None;
        }
        if mm.inputs.first() != Some(&feed) {
            return None;
        }
        // The Matmul's weight operand may be either the raw lm_head
        // tensor or a Transpose view of the embedding table (mlx-lm
        // models with `tie_word_embeddings=True`). Either way, follow
        // the producer chain back to the underlying constant.
        let mm_weight_tid = mm.inputs[1];
        let w_root = trace_to_root(g, mm_weight_tid);
        let logits_tid = mm.outputs[0];
        let (quant, w_scales, w_biases) = if is_q {
            let s = mm.inputs[2];
            let b = mm.inputs[3];
            let q = match mm.attrs {
                Attrs::Quantized { bits, group_size } => QuantInfo { bits, group_size },
                _ => QuantInfo {
                    bits: 4,
                    group_size: 64,
                },
            };
            (Some(q), Some(s), Some(b))
        } else {
            (None, None, None)
        };
        if crate::fusion::patterns::no_patterns_enabled() && quant.is_none() {
            return None;
        }

        let bnd = |tid: TensorId, role: &'static str| -> FusionBoundary {
            let t = &g.tensors[tid as usize];
            FusionBoundary {
                source: tid,
                role,
                shape: t.shape.clone(),
                dtype: t.dtype,
            }
        };

        let mut inputs = vec![
            bnd(x_tid, "x"),
            bnd(norm_w_tid, "norm_weight"),
            bnd(w_root, "W_lm_head"),
        ];
        if let (Some(s), Some(b)) = (w_scales, w_biases) {
            inputs.push(bnd(s, "W_lm_head_scales"));
            inputs.push(bnd(b, "W_lm_head_biases"));
        }
        let outputs = vec![bnd(logits_tid, "logits")];

        let out_shape = &g.tensors[logits_tid as usize].shape;
        let parallel_axes = vec![ParallelAxis {
            name: "vocab",
            size: *out_shape.last().unwrap_or(&1) as u32,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: out_shape.len().saturating_sub(1) as u32,
            },
        }];

        let mut op_ids = vec![anchor];
        op_ids.extend(view_ops);
        op_ids.push(mm_id);

        Some(Match {
            kind: FusionKind::FinalRmsNormLmHead,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::FinalRmsNormLmHead { eps, quant },
            parallel_axes,
        })
    }
}

/// Walk view-only producers back to the root storage tensor. Used to
/// resolve `W_lm_head = embed_tokens.T` to the raw embedding table.
fn trace_to_root(g: &Graph, mut tid: TensorId) -> TensorId {
    loop {
        let producer = match g.tensor(tid).producer {
            Some(p) => p,
            None => return tid,
        };
        let op = g.op(producer);
        if crate::fusion::patterns::is_view_op(&op.name) && !op.inputs.is_empty() {
            tid = op.inputs[0];
        } else if op.name == "Transpose" && !op.inputs.is_empty() {
            // Transpose is intentionally NOT in is_view_op (downstream
            // kernels often want contiguous data), but for resolving
            // the lm_head's raw storage we deliberately walk through
            // it — the underlying memory layout is what the kernel
            // indexes anyway.
            tid = op.inputs[0];
        } else {
            return tid;
        }
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
