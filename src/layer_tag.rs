//! Tag every captured op with the forward-pass region it belongs to.
//!
//! Layer-granular dispatch builds one Metal ICB per `LayerTag`. The
//! captured trace itself doesn't carry layer information — mlx's
//! per-primitive callback fires inside the model's forward without
//! any module-scope context — so we recover the partition from the
//! Python attribute paths that `loader::lower` already stamped onto
//! each external Tensor.
//!
//! The pass runs in two stages:
//!
//! 1. Direct hit. For every op, look at the inputs whose
//!    `param_path` is non-empty. Parse `model.layers.<N>.…` →
//!    `Layer(N)`, `model.embed_tokens.…` → `Embed`,
//!    `model.norm.…` / `lm_head.…` → `Post`. If multiple inputs
//!    point at different regions (rare — happens on residual ops
//!    that combine the pre-layer x with a per-layer norm weight),
//!    take the highest tag because that's the region this op is
//!    actively producing for.
//!
//! 2. Forward propagation. For every still-unlabelled op, copy the
//!    highest tag from its producer ops (resolved through each
//!    input tensor's `producer`). Topological order means by the
//!    time we look at an op all its producers have already been
//!    classified, so a single linear sweep suffices.
//!
//! Anything that's still `None` after both stages reads only
//! non-parameter inputs (KV cache slots on the cold path, the input
//! token tensor) and is left untagged for the runtime to handle
//! separately.

use crate::graph::{Graph, LayerTag};

pub fn infer_layer_tags(g: &mut Graph) {
    // Pass 1: direct hits via param_path.
    for op_idx in 0..g.ops.len() {
        let mut tags: Vec<LayerTag> = Vec::new();
        for &tid in &g.ops[op_idx].inputs {
            if let Some(path) = g.tensors[tid as usize].param_path.as_deref() {
                if let Some(tag) = parse_path(path) {
                    tags.push(tag);
                }
            }
        }
        if let Some(&max_tag) = tags.iter().max() {
            g.ops[op_idx].layer_id = Some(max_tag);
        }
    }

    // Pass 2: inherit from producers. Ops were appended to `g.ops`
    // in topological order during `lower`, so a single forward sweep
    // sees every producer tag before its consumer.
    for op_idx in 0..g.ops.len() {
        if g.ops[op_idx].layer_id.is_some() {
            continue;
        }
        let mut tags: Vec<LayerTag> = Vec::new();
        let inputs = g.ops[op_idx].inputs.clone();
        for tid in inputs {
            if let Some(producer_id) = g.tensors[tid as usize].producer {
                if let Some(tag) = g.ops[producer_id as usize].layer_id {
                    tags.push(tag);
                }
            }
        }
        if let Some(&max_tag) = tags.iter().max() {
            g.ops[op_idx].layer_id = Some(max_tag);
        }
    }
}

/// Map a Python attribute path to its forward-pass region. Returns
/// `None` for paths that don't match any known position — those
/// happen for shared weights or non-standard model layouts.
///
/// Multimodal mlx-vlm models nest the decoder under one or more
/// outer scopes (`language_model.model.layers.N…`,
/// `text_model.model.layers.N…`). Rather than hard-code each
/// wrapper, locate the `.layers.<N>.` (or `.embed_tokens.`,
/// `.norm.`, `lm_head.`) segment by substring and grab the layer
/// index that follows.
fn parse_path(path: &str) -> Option<LayerTag> {
    if let Some(pos) = path.find(".layers.") {
        let after = &path[pos + ".layers.".len()..];
        let idx_part = after.split('.').next()?;
        let n: i32 = idx_part.parse().ok()?;
        return Some(LayerTag::Layer(n));
    }
    // Bare `layers.` at the start (e.g. trace flattens the model root).
    if let Some(rest) = path.strip_prefix("layers.") {
        let idx_part = rest.split('.').next()?;
        let n: i32 = idx_part.parse().ok()?;
        return Some(LayerTag::Layer(n));
    }
    if path.contains(".embed_tokens.") || path.starts_with("embed_tokens.") {
        return Some(LayerTag::Embed);
    }
    if path.ends_with(".norm.weight")
        || path == "norm.weight"
        || path.contains(".norm.")
        || path.starts_with("lm_head.")
        || path.contains(".lm_head.")
    {
        return Some(LayerTag::Post);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_layer_idx() {
        assert_eq!(
            parse_path("model.layers.3.self_attn.q_proj.weight"),
            Some(LayerTag::Layer(3))
        );
        assert_eq!(
            parse_path("model.layers.17.mlp.down_proj.scales"),
            Some(LayerTag::Layer(17))
        );
    }
    #[test]
    fn parses_embed_and_post() {
        assert_eq!(
            parse_path("model.embed_tokens.weight"),
            Some(LayerTag::Embed)
        );
        assert_eq!(parse_path("model.norm.weight"), Some(LayerTag::Post));
        assert_eq!(parse_path("lm_head.weight"), Some(LayerTag::Post));
    }
    #[test]
    fn ordering_max_picks_active_region() {
        let mut v = vec![LayerTag::Layer(2), LayerTag::Embed, LayerTag::Layer(3)];
        v.sort();
        assert_eq!(v.last(), Some(&LayerTag::Layer(3)));
        let mut v2 = vec![LayerTag::Layer(23), LayerTag::Post];
        v2.sort();
        assert_eq!(v2.last(), Some(&LayerTag::Post));
    }
}
