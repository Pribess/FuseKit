//! Per-layer megakernel emitter.
//!
//! Composes the attention block of one transformer layer
//!   `RMSNorm → SDPA → o_proj+residual`
//! into one MSL function via the piece composer. This is the
//! production hookup of the per-layer megakernel skeleton — the
//! attention-block portion. Full-layer (MLP included) needs the
//! mlp_dim-tiling work to land before silu_up SMEM fits the budget.
//!
//! The piece sequence here is intentionally shorter than what a
//! full Qwen-style decode layer does (no in-kernel QKV/RoPE yet)
//! because:
//!   - the QKV piece is simd-cooperative (1 SG per output row),
//!   - the RoPE piece is thread-per-element,
//!   - the SDPA piece is TG-per-head with 128 threads pinned,
//!   - the o_proj piece is simd-cooperative again.
//! Reconciling these grid topologies into one dispatch shape is
//! follow-up work. For now we pre-stage Q (via the upstream
//! `rmsnorm_qkv` kernel) and K/V cache (via `rope_kv_cache_store`)
//! and run only the SDPA→o_proj portion fused with the input norm.

use anyhow::Result;

use crate::codegen::msl::MslWriter;
use crate::codegen::pieces::{
    self, global_barrier::GlobalBarrier, linear_residual::LinearResidual,
    rmsnorm_reduce::RmsNormReduce, sdpa_decode::SdpaDecode, EmitCtx, KernelPiece,
};
use crate::fusion::ir::{Fusion, FusionAttrs, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_layer_block_f{}", fusion.id)
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    let (eps, quant, _theta, layer_idx) = match &fusion.attrs {
        FusionAttrs::LayerBlock {
            eps,
            quant,
            theta,
            layer_idx,
            ..
        } => (*eps, *quant, *theta, *layer_idx),
        _ => anyhow::bail!("layer_block::emit called with non-LayerBlock attrs"),
    };

    let ctx = EmitCtx {
        meta,
        act_ty: super::act_ty(meta),
    };
    let fn_name = function_name(fusion);

    // The piece composer wires:
    //   - RmsNormReduce writes SMEM `norm_y` (read by GEMV pieces if
    //     later added, ignored here since SDPA reads Q from DRAM).
    //   - SdpaDecode declares its own SMEM regions and reads
    //     `q_buf`/`cache_k`/`cache_v` — staged by upstream kernels.
    //   - LinearResidual(o_proj) reads sdpa_out and writes into the
    //     `x_input` role which RmsNormReduce also touches — composer
    //     unifies into one read+write `x` buffer slot.
    let n_heads = meta.num_attention_heads;
    let pieces: Vec<Box<dyn KernelPiece>> = vec![
        Box::new(RmsNormReduce::new(eps)),
        Box::new(SdpaDecode::bf16()),
        // Cross-TG sync: SDPA writes sdpa_out[head h] in TG h;
        // LinearResidual reads ALL heads' sdpa_out from EVERY TG.
        // Without this barrier the o_proj reads stale/uninitialised
        // values from other TGs' sdpa writes (Apple Metal has no
        // grid-level sync API). Atomic-counter spin barrier with
        // tg_count=n_heads pinned at dispatch time.
        Box::new(GlobalBarrier {
            stage_idx: layer_idx as u32,
            tg_count: n_heads,
        }),
        Box::new(LinearResidual {
            m: (meta.num_attention_heads * meta.head_dim),
            n: meta.hidden_size,
            act_var: "sdpa_out".into(),
            act_role: "sdpa_out".into(),
            tag: "o".into(),
            quant,
            has_bias: false,
        }),
    ];

    pieces::compose(&fn_name, &pieces, &ctx, w);
    Ok(())
}
