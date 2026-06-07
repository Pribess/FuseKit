//! Pattern: `RMSNorm → view → (Q, K, V) parallel projections`.
//!
//! ```text
//!                              ┌─► Matmul/AddMM ─► q
//!   x ─►┌─────────┐  view  ─►──┤   (W_q)
//!       │ RMSNorm │            ├─► Matmul/AddMM ─► k
//!       └─────────┘            │   (W_k)
//!         W_norm               └─► Matmul/AddMM ─► v
//!                                  (W_v)
//! ```
//!
//! In transformer attention preludes, the post-norm activation feeds
//! three separate projections (Q, K, V). MLX emits this as a single
//! RMSNorm followed by three Matmul/AddMM ops reading the same
//! (possibly view-relabelled) tensor. Fusing them keeps the norm
//! output resident in threadgroup memory for all three projections.

use crate::fusion::ir::{
    AxisRef, BoundaryKind, FusionAttrs, FusionBoundary, FusionKind, ParallelAxis, QuantInfo,
};
use crate::fusion::patterns::{Match, Pattern, UseDef};
use crate::graph::{Attrs, Graph, OpId, TensorId};

pub struct RmsNormQkv;

impl Pattern for RmsNormQkv {
    fn name(&self) -> &'static str {
        "rmsnorm_qkv"
    }

    fn priority(&self) -> u32 {
        // Wider than rmsnorm_proj (single projection), so try first.
        100
    }

    fn is_structural(&self) -> bool {
        true
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

        // Walk through chained view ops. The result tensor is what
        // the three projections actually read.
        //
        // `follow_views` requires single-consumer links along the
        // view chain, so we manually inspect the final tensor's
        // consumers separately to allow the fan-out we need here.
        let (feeding_tensor, view_ops) = walk_until_fanout(g, ud, norm_out)?;
        let cs = &ud.consumers[feeding_tensor as usize];
        if cs.len() != 3 {
            return None;
        }

        // All three consumers must be Matmul/AddMM/QuantizedMatmul, and
        // each must read `feeding_tensor` on the activation side.
        let mut projs: Vec<(OpId, MatmulInfo)> = Vec::with_capacity(3);
        let mut quant: Option<QuantInfo> = None;
        for &cid in cs {
            let op = g.op(cid);
            let info = MatmulInfo::extract_with_bias(g, ud, op, feeding_tensor)?;
            if let Some(q) = info.quant {
                // All three must share the same quant config — otherwise
                // we'd be emitting two different inner loops in one kernel.
                if let Some(prev) = quant {
                    if prev.bits != q.bits || prev.group_size != q.group_size {
                        return None;
                    }
                }
                quant = Some(q);
            }
            projs.push((cid, info));
        }
        // All-or-nothing: don't mix quant + dense projections.
        let any_quant = projs.iter().any(|(_, p)| p.quant.is_some());
        let all_quant = projs.iter().all(|(_, p)| p.quant.is_some());
        if any_quant && !all_quant {
            return None;
        }
        if crate::fusion::patterns::no_patterns_enabled() && !all_quant {
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

        // Layout: (x, norm_weight, W_q, W_k, W_v,
        //          [W_q_scales, W_k_scales, W_v_scales,
        //           W_q_biases, W_k_biases, W_v_biases]  for QuantizedMatmul,
        //          [bias_q, bias_k, bias_v]              for AddMM).
        // The three projections each output one tensor (q, k, v).
        let mut inputs = vec![bnd(rms.inputs[0], "x"), bnd(rms.inputs[1], "norm_weight")];
        for (i, (_, info)) in projs.iter().enumerate() {
            let role = match i {
                0 => "W_q",
                1 => "W_k",
                _ => "W_v",
            };
            inputs.push(bnd(info.weight, role));
        }
        if all_quant {
            for (i, (_, info)) in projs.iter().enumerate() {
                let role = match i {
                    0 => "W_q_scales",
                    1 => "W_k_scales",
                    _ => "W_v_scales",
                };
                inputs.push(bnd(info.scales.unwrap(), role));
            }
            for (i, (_, info)) in projs.iter().enumerate() {
                let role = match i {
                    0 => "W_q_biases",
                    1 => "W_k_biases",
                    _ => "W_v_biases",
                };
                inputs.push(bnd(info.biases.unwrap(), role));
            }
        }
        // Bias is independent of quant: QuantizedLinear with bias=True
        // still emits a separate Add(qm_out, bias) which extract_with_bias
        // folds into info.bias.
        for (i, (_, info)) in projs.iter().enumerate() {
            if let Some(b) = info.bias {
                let role = match i {
                    0 => "bias_q",
                    1 => "bias_k",
                    _ => "bias_v",
                };
                inputs.push(bnd(b, role));
            }
        }

        let outputs = projs
            .iter()
            .enumerate()
            .map(|(i, (_, info))| {
                let role = match i {
                    0 => "q",
                    1 => "k",
                    _ => "v",
                };
                bnd(info.output, role)
            })
            .collect::<Vec<_>>();

        // Q's output_dim is the natural axis to split across heads.
        let q_shape = &g.tensors[projs[0].1.output as usize].shape;
        let q_rows = *q_shape.last().unwrap_or(&1) as u32;
        let parallel_axes = vec![ParallelAxis {
            name: "out_row",
            size: q_rows,
            on: AxisRef {
                boundary_kind: BoundaryKind::Output,
                boundary_index: 0,
                dim: q_shape.len().saturating_sub(1) as u32,
            },
        }];

        let mut op_ids = vec![anchor];
        op_ids.extend(view_ops);
        op_ids.extend(projs.iter().map(|(id, _)| *id));
        // Absorb the QM-bias Add ops that extract_with_bias folded in.
        for (_, info) in &projs {
            if let Some(add_id) = info.bias_add_op {
                op_ids.push(add_id);
            }
            if let Some(tid) = info.weight_transpose_op {
                op_ids.push(tid);
            }
        }

        Some(Match {
            kind: FusionKind::RmsNormQkv,
            op_ids,
            inputs,
            outputs,
            attrs: FusionAttrs::RmsNormQkv {
                eps,
                quant,
                decode_sinks: None,
            },
            parallel_axes,
        })
    }
}

struct MatmulInfo {
    weight: TensorId,
    bias: Option<TensorId>,
    scales: Option<TensorId>,
    biases: Option<TensorId>,
    quant: Option<QuantInfo>,
    output: TensorId,
    /// OpId of the downstream Add(qm_out, bias) for QuantizedMatmul, if folded.
    /// Used to extend the fusion's `op_ids` so the Add isn't left as a Single.
    bias_add_op: Option<OpId>,
    /// OpId of the upstream `Transpose(W)` for bf16/fp16 Matmul/AddMM
    /// paths — mlx-lm's `Linear` lowers `x @ W.T` so the trace has an
    /// explicit Transpose. Pattern emit indexes W as `[out, in]`, so
    /// we walk back through the Transpose and bind the pre-transpose
    /// weight; the Transpose op gets consumed by this fusion.
    weight_transpose_op: Option<OpId>,
}

impl MatmulInfo {
    fn extract_with_bias(
        g: &Graph,
        ud: &UseDef,
        op: &crate::graph::Op,
        x_tensor: TensorId,
    ) -> Option<Self> {
        // MLX primitive ordering:
        //   Matmul          (x, W)
        //   AddMM           (x, W, bias)
        //   QuantizedMatmul (x, W_packed, scales, biases)
        // For QuantizedLinear with bias=True, mlx emits the bias add
        // as a SEPARATE Add op downstream of the QuantizedMatmul. So
        // we walk one step past the QM looking for that Add.
        let is_addmm = op.name == "AddMM";
        let is_matmul = op.name == "Matmul";
        let is_quant = op.name == "QuantizedMatmul";
        if !is_matmul && !is_addmm && !is_quant {
            return None;
        }
        if op.inputs.first() != Some(&x_tensor) {
            return None;
        }
        let weight_raw = *op.inputs.get(1)?;
        // For dense Matmul/AddMM (bf16/fp16), the trace materializes
        // the `W.T` of mlx-lm's Linear as an explicit Transpose op.
        // Walk back through it so we bind the original `[out, in]`
        // weight that the kernel emit assumes. QuantizedMatmul keeps
        // W as `[out, in]` natively, so the walk is a no-op there.
        let (weight, weight_transpose_op) = if is_quant {
            (weight_raw, None)
        } else {
            crate::fusion::patterns::unwrap_weight_transpose(g, ud, weight_raw)
        };
        let mut bias = if is_addmm {
            op.inputs.get(2).copied()
        } else {
            None
        };
        let (scales, biases, quant) = if is_quant {
            let s = op.inputs.get(2).copied()?;
            let b = op.inputs.get(3).copied()?;
            let q = match op.attrs {
                Attrs::Quantized { bits, group_size } => QuantInfo { bits, group_size },
                _ => QuantInfo {
                    bits: 4,
                    group_size: 64,
                },
            };
            (Some(s), Some(b), Some(q))
        } else {
            (None, None, None)
        };
        let mut output = *op.outputs.first()?;
        let mut bias_add_op: Option<OpId> = None;
        // For QM, check for a downstream `Add(qm_out, bias_const)`. Only
        // claim it if Add has exactly 2 inputs and qm_out has a single
        // consumer (so absorbing the Add doesn't strand other uses).
        if is_quant {
            let consumers = &ud.consumers[output as usize];
            if consumers.len() == 1 {
                let next = g.op(consumers[0]);
                if next.name == "Add" && next.inputs.len() == 2 && next.outputs.len() == 1 {
                    let bias_tid = if next.inputs[0] == output {
                        next.inputs[1]
                    } else {
                        next.inputs[0]
                    };
                    bias = Some(bias_tid);
                    output = next.outputs[0];
                    bias_add_op = Some(next.id);
                }
            }
        }
        Some(Self {
            weight,
            bias,
            scales,
            biases,
            quant,
            output,
            bias_add_op,
            weight_transpose_op,
        })
    }
}

/// Walk view ops greedily — each step must be a single-consumer view —
/// but stop before the final tensor whose consumer count we'll check
/// externally. Returns the final tensor + view op chain.
fn walk_until_fanout(g: &Graph, ud: &UseDef, start: TensorId) -> Option<(TensorId, Vec<OpId>)> {
    let mut tensor = start;
    let mut views = Vec::new();
    loop {
        let cs = &ud.consumers[tensor as usize];
        if cs.len() != 1 {
            // Multi-consumer: stop and let the caller check fan-out.
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
