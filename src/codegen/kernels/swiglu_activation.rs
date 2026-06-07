//! MSL emit for `FusionKind::SwigluActivation`.
//!
//! ```text
//!   out[i] = sigmoid(gate[i]) · gate[i] · up[i]   (elementwise)
//! ```
//!
//! Pure elementwise — no reductions, no shared memory needed. One
//! thread per element.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::codegen::pieces::{compose, swiglu_activation::SwigluActivation, EmitCtx, KernelPiece};
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

/// First emitter routed through the piece composer instead of
/// emitting its own kernel signature + body inline. Validates the
/// composer plumbing end-to-end: SMEM dedup is trivial here (none),
/// buffer slot assignment must yield (gate=0, up=1, out=2) which
/// matches the legacy emit signature, and the composer's standard
/// locals (`simd_lane`, `simd_id`, `sgs_per_tg`) are unused but
/// harmless for pure elementwise.
pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::SwigluActivation) {
        bail!("swiglu_activation emitter: wrong kind {:?}", fusion.kind);
    }
    if !matches!(fusion.attrs, FusionAttrs::SwigluActivation) {
        bail!("swiglu_activation emitter: attrs mismatch");
    }
    let out = &fusion.outputs[0];
    let n: u32 = out.shape.iter().map(|d| *d as u32).product();
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: SwigluActivation (N = {})",
        fusion.id, n
    ));
    let pieces: Vec<Box<dyn KernelPiece>> = vec![Box::new(SwigluActivation { n })];
    let ctx = EmitCtx {
        meta,
        act_ty: crate::codegen::kernels::act_ty(meta),
    };
    compose(&fn_name, &pieces, &ctx, w);
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_swiglu_act_f{}", fusion.id)
}
