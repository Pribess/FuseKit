#![allow(dead_code)]

//! Layer-block matcher — groups per-layer fusions into one
//! [`FusionKind::LayerBlock`] per layer.
//!
//! Inputs: a fully built [`FusionGraph`] (post-pattern, post-general).
//! Output: an additional vector of [`Fusion`] records, one per
//!   `LayerTag::Layer(N)` found in the graph.
//!
//! The matcher is currently **read-only** — it returns synthetic
//! LayerBlock fusions for the codegen emitter to compile into the
//! metallib (validating that the composer output passes Metal's
//! shader compiler for every layer). The task graph + dispatcher
//! still use the per-pattern fusions; a follow-up step will rewrite
//! the graph to dispatch one LayerBlock task per layer instead.
//!
//! Gating: enabled by `FUSEKIT_LAYER_MEGAKERNEL=1`. Off by default so
//! existing decode paths are unchanged.

use std::collections::BTreeSet;

use crate::fusion::ir::{Fusion, FusionAttrs, FusionGraph, FusionKind, QuantInfo};
use crate::graph::{Graph, LayerTag};

/// Did the user enable per-layer megakernel synthesis?
pub fn enabled() -> bool {
    std::env::var("FUSEKIT_LAYER_MEGAKERNEL").map_or(false, |v| v == "1")
}

/// Append synthetic [`FusionKind::LayerBlock`] fusions to `fg` — one per
/// `LayerTag::Layer(N)` found in the graph. Idempotent: if a LayerBlock
/// fusion for layer N already exists, it is left alone.
///
/// The absorbed pattern fusions (`SdpaDecode`, `LinearResidual(o_proj)`)
/// are *not* removed. Task generation still runs for them; the
/// dispatcher's decode pipeline (built in `task_graph_json`) replaces
/// the two with the LayerBlock task when `enabled()` is true.
pub fn integrate(fg: &mut FusionGraph, graph: &Graph) {
    let layer_block_already: BTreeSet<i32> = fg
        .fusions
        .iter()
        .filter_map(|f| match (&f.kind, f.layer_hint) {
            (FusionKind::LayerBlock, Some(LayerTag::Layer(i))) => Some(i),
            _ => None,
        })
        .collect();
    let new_fusions = synthesize(fg, graph);
    for mut f in new_fusions {
        let layer_idx = match f.attrs {
            FusionAttrs::LayerBlock { layer_idx, .. } => layer_idx,
            _ => continue,
        };
        if layer_block_already.contains(&layer_idx) {
            continue;
        }
        f.id = fg.fusions.len() as u32;
        fg.fusions.push(f);
    }
}

/// Walk `fg` and emit one synthetic `LayerBlock` fusion per
/// `LayerTag::Layer(N)` present in the graph. The synthesized
/// fusions get fresh ids starting at `fg.fusions.len()` (callers
/// that want to merge them into the graph should re-id; for now
/// the result is only used as input to the codegen emitter).
///
/// Per-layer params (eps, theta, quant) come from the existing
/// fusions in that layer:
///   - eps from the layer's `RmsNormQkv` / `RmsNormGateUp*` attrs,
///   - quant from the same (uniform across the layer's int4 weights),
///   - theta from the layer's `RopeQDecode` / `RopeKvCacheStoreDecode`.
/// If a layer is missing one of these, the matcher skips it.
/// For each layer, find the `LinearResidual(o_proj)` fusion — the
/// earlier (smaller op-id-min) of the two `LinearResidual`s in the
/// layer. Used to copy its boundary tensors into the LayerBlock so
/// the trace-bound scratch routing handles residual stream binding.
fn o_proj_fusion_for_layer(fg: &FusionGraph, graph: &Graph, layer_idx: i32) -> Option<usize> {
    let mut candidates: Vec<(u32, usize)> = Vec::new();
    for (i, f) in fg.fusions.iter().enumerate() {
        if !matches!(f.kind, FusionKind::LinearResidual) {
            continue;
        }
        let in_layer_by_op = f.op_ids.iter().any(|&op_id| {
            matches!(graph.ops[op_id as usize].layer_id, Some(LayerTag::Layer(n)) if n == layer_idx)
        });
        let in_layer_by_weight = f.inputs.iter().any(|bd| {
            graph
                .tensors
                .get(bd.source as usize)
                .and_then(|t| t.param_path.as_deref())
                .is_some_and(|path| {
                    path.ends_with("self_attn.o_proj.weight")
                        && param_path_layer_idx(path) == Some(layer_idx)
                })
        });
        if !in_layer_by_op && !in_layer_by_weight {
            continue;
        }
        let opid_min = f.op_ids.iter().copied().min().unwrap_or(u32::MAX);
        candidates.push((opid_min, i));
    }
    candidates.sort();
    candidates.first().map(|&(_, i)| i)
}

fn fusion_touches_layer(f: &Fusion, graph: &Graph, layer_idx: i32) -> bool {
    matches!(f.layer_hint, Some(LayerTag::Layer(n)) if n == layer_idx)
        || f.op_ids.iter().any(|&op_id| {
            matches!(graph.ops[op_id as usize].layer_id, Some(LayerTag::Layer(n)) if n == layer_idx)
        })
}

fn has_full_attention_for_layer(fg: &FusionGraph, graph: &Graph, layer_idx: i32) -> bool {
    fg.fusions.iter().any(|f| {
        matches!(f.kind, FusionKind::SdpaDecode) && fusion_touches_layer(f, graph, layer_idx)
    })
}

fn param_path_layer_idx(path: &str) -> Option<i32> {
    let marker = ".layers.";
    let start = path.find(marker)? + marker.len();
    let digits: String = path[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

pub fn synthesize(fg: &FusionGraph, graph: &Graph) -> Vec<Fusion> {
    // Collect layer indices in stable order. Some traces leave
    // `layer_hint` unset on pattern fusions, so fall back to the raw
    // graph op layer tags carried by `op_ids`.
    let mut layers: BTreeSet<i32> = BTreeSet::new();
    for f in &fg.fusions {
        if let Some(LayerTag::Layer(i)) = f.layer_hint {
            layers.insert(i);
        }
        for &op_id in &f.op_ids {
            if let Some(LayerTag::Layer(i)) = graph.ops[op_id as usize].layer_id {
                layers.insert(i);
            }
        }
    }

    let mut out = Vec::with_capacity(layers.len());
    let mut next_id = fg.fusions.len() as u32;
    // eps + quant + theta are uniform across the model's layers, so
    // we scan globally once instead of per-layer. (Per-layer scan
    // would miss pattern fusions whose `layer_hint` is None — the
    // build pass leaves it unset on trace-bound matches.)
    let global = scan_layer_params(fg);
    for layer_idx in layers {
        let (Some(eps), Some(theta)) = (global.eps, global.theta) else {
            // No eps + theta recovered anywhere in the graph (e.g.
            // synthetic test capture). Skip entirely — emitting a
            // LayerBlock with zeroed params would silently produce
            // wrong numerics.
            continue;
        };
        if !has_full_attention_for_layer(fg, graph, layer_idx) {
            continue;
        }
        // Borrow boundary tensors from the layer's o_proj
        // LinearResidual fusion so the trace-bound scratch router can
        // hook up the residual stream automatically:
        //   - LinearResidual.inputs[-1] = "residual" — layer N's
        //     input x tensor (same as RmsNormQkv.inputs[0]).
        //   - LinearResidual.outputs[0] = "y" — layer N's post-attn x.
        // LayerBlock consumes residual and writes y, exactly the
        // same trace tensors. Composer dedups: LinearResidual
        // piece's `residual` slot (role "x_input") merges with
        // RmsNormReduce piece's `x` slot (also role "x_input") into
        // one input buffer; the `y_o` slot stays as a separate
        // output buffer. So in fusion IR we set inputs=[residual]
        // and outputs=[y].
        let Some(o_proj_idx) = o_proj_fusion_for_layer(fg, graph, layer_idx) else {
            continue;
        };
        let lr = &fg.fusions[o_proj_idx];
        let residual = lr
            .inputs
            .iter()
            .rev()
            .find(|b| b.role == "residual")
            .cloned();
        let y = lr.outputs.first().cloned();
        let (Some(residual), Some(y)) = (residual, y) else {
            continue;
        };
        let lb_inputs = vec![residual];
        let lb_outputs = vec![y];
        out.push(Fusion {
            id: next_id,
            kind: FusionKind::LayerBlock,
            op_ids: vec![],
            inputs: lb_inputs,
            outputs: lb_outputs,
            attrs: FusionAttrs::LayerBlock {
                layer_idx,
                eps,
                theta,
                quant: global.quant,
            },
            parallel_axes: vec![],
            layer_hint: Some(LayerTag::Layer(layer_idx)),
        });
        next_id += 1;
    }
    out
}

#[derive(Default, Debug, Clone, Copy)]
struct LayerParams {
    eps: Option<f32>,
    theta: Option<f32>,
    quant: Option<QuantInfo>,
}

fn scan_layer_params(fg: &FusionGraph) -> LayerParams {
    let mut p = LayerParams::default();
    for f in &fg.fusions {
        match &f.attrs {
            FusionAttrs::RmsNormQkv { eps, quant, .. }
            | FusionAttrs::RmsNormGateUp { eps, quant }
            | FusionAttrs::RmsNormGateUpSwiglu { eps, quant }
            | FusionAttrs::RmsNormGateUpSwigluDown { eps, quant } => {
                p.eps.get_or_insert(*eps);
                if p.quant.is_none() {
                    p.quant = *quant;
                }
            }
            FusionAttrs::RopeQDecode { theta }
            | FusionAttrs::RopeKvCacheStoreDecode { theta }
            | FusionAttrs::RmsNormRopeQGateDecode { theta, .. }
            | FusionAttrs::RmsNormRopeKvCacheStoreDecode { theta, .. } => {
                p.theta.get_or_insert(*theta);
            }
            _ => {}
        }
    }
    p
}
