#![allow(dead_code)]

//! Stage 4+5 — MSL codegen and Metal compilation.
//!
//! Inputs:  `crate::task::ir::TaskGraph` + `crate::fusion::ir::ModelMeta`
//! Outputs: `<model_dir>/fusekit/kernel.metallib`
//!          `<model_dir>/fusekit/task_graph.json`
//!
//! ```text
//!   TaskGraph
//!      │
//!      │ 1. group tasks by (FusionKind, shape, partition) → variants
//!      │ 2. emit MSL function per unique variant (kernels::emit_for)
//!      │ 3. emit persistent megakernel scheduler (persistent::emit_megakernel)
//!      │ 4. write kernel.metal source
//!      │ 5. metal_compile::compile_to_metallib
//!      ▼
//!   kernel.metallib + task_graph.json
//! ```
//!
//! Phase 1 ships the plumbing but emitters are stubs; first real
//! emission lands with the first pattern.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use crate::codegen::kernels::variant_name;
use crate::codegen::msl::{prelude, MslWriter};
use crate::fusion::ir::{FusionGraph, ModelMeta};
use crate::graph::Graph;
use crate::loader::RawParam;
use crate::target::Target;
use crate::task::ir::TaskGraph;

pub mod kernels;
pub mod metal_compile;
pub mod msl;
pub mod persistent;
pub mod pieces;
pub mod task_graph_json;

/// Top-level entry. Writes both `kernel.metallib` and `task_graph.json`
/// under `out_dir` and returns the metallib path.
pub fn emit_metallib(
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
    meta: &ModelMeta,
    params: &[RawParam],
    out_dir: &Path,
    target: Target,
) -> Result<PathBuf> {
    // Surface chip-aware decisions before codegen runs: budget-driven
    // TG size and register-pressure spill warnings. The piece-composer
    // path will read the same budgets when it goes live; today the
    // existing monolithic emitters still use their hardcoded MSL but
    // the task graph's `threadgroup` / `grid_threads` reflect the
    // budget pick.
    {
        use crate::fusion::safety::estimate_register_pressure;
        let b = &target.budgets;
        // Reserve ~32 slots for control flow / loop indices the
        // compiler always allocates.
        let reg_headroom = b.regs_per_thread.saturating_sub(32);
        let mut spilled: Vec<String> = Vec::new();
        for f in &fg.fusions {
            let need = estimate_register_pressure(&f.kind, meta);
            if need > reg_headroom {
                spilled.push(format!(
                    "{:?}: need ~{} regs, budget {}",
                    f.kind, need, reg_headroom
                ));
            }
        }
        if !spilled.is_empty() {
            eprintln!(
                "[fusekit] warning: {} fusion(s) may spill on {}; consider --chip with larger reg budget:",
                spilled.len(),
                target.chip.display()
            );
            for s in spilled.iter().take(5) {
                eprintln!("  - {s}");
            }
        }
    }
    // Publish the chip to the codegen-wide thread-local so kernel
    // emitters can read it without each signature gaining a Target
    // parameter. Safe because emit_metallib is a single-threaded entry
    // point — chained calls all see the same value.
    crate::target::set_current_chip(target.chip);
    fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let mut w = MslWriter::new();
    prelude(&mut w);

    let mut emit_fusions = fg.fusions.clone();
    for fusion in &mut emit_fusions {
        for boundary in fusion.inputs.iter_mut().chain(fusion.outputs.iter_mut()) {
            if boundary.scalar_value.is_none() {
                boundary.scalar_value = graph
                    .tensors
                    .get(boundary.source as usize)
                    .and_then(|t| t.scalar_value.clone());
            }
        }
    }

    // Per-variant kernel functions. Tasks sharing (kind, shape,
    // partition) emit one shared MSL function.
    let mut seen: std::collections::HashSet<String> = Default::default();
    for task in &tg.tasks {
        let fusion = &emit_fusions[task.fusion as usize];
        let name = variant_name(fusion, &task.partition);
        if seen.insert(name) {
            kernels::emit_for(fusion, &task.partition, meta, &mut w)?;
        }
    }

    // KV-quant variants. Emit q4 + q8 for SDPA, cache-store, and
    // QKV decode-sink fusions alongside the bf16 defaults so the
    // dispatcher can pick whichever matches the live `KVCache` /
    // `QuantizedKVCache` at runtime without rebuilding the metallib.
    // QKV sink variants keep the same graph semantics but bind the live
    // cache as packed/scales/biases triplets instead of dense K/V rows.
    // Metal compiler dedupes pipeline state, so binary bloat is moderate.
    {
        use crate::fusion::ir::FusionKind;
        let mut seen_quant: std::collections::HashSet<String> = Default::default();
        for task in &tg.tasks {
            let fusion = &emit_fusions[task.fusion as usize];
            match fusion.kind {
                FusionKind::SdpaDecode => {
                    for bits in [4u32, 8u32] {
                        let key = kernels::sdpa_decode_quant::function_name_for(fusion, bits);
                        if seen_quant.insert(key) {
                            kernels::sdpa_decode_quant::emit(
                                fusion,
                                &task.partition,
                                meta,
                                bits,
                                64,
                                &mut w,
                            )?;
                        }
                    }
                }
                FusionKind::RopeKvCacheStoreDecode => {
                    for bits in [4u32, 8u32] {
                        let key = kernels::rope_kv_cache_store_decode_quant::function_name_for(
                            fusion, bits,
                        );
                        if seen_quant.insert(key) {
                            kernels::rope_kv_cache_store_decode_quant::emit(
                                fusion,
                                &task.partition,
                                meta,
                                bits,
                                64,
                                &mut w,
                            )?;
                        }
                    }
                }
                FusionKind::RmsNormRopeKvCacheStoreDecode => {
                    for bits in [4u32, 8u32] {
                        let key =
                            kernels::rmsnorm_rope_kv_cache_store_decode_quant::function_name_for(
                                fusion, bits,
                            );
                        if seen_quant.insert(key) {
                            kernels::rmsnorm_rope_kv_cache_store_decode_quant::emit(
                                fusion,
                                &task.partition,
                                meta,
                                bits,
                                64,
                                &mut w,
                            )?;
                        }
                    }
                }
                FusionKind::RmsNormQkv => {
                    if let crate::fusion::ir::FusionAttrs::RmsNormQkv {
                        decode_sinks: Some(_),
                        ..
                    } = fusion.attrs
                    {
                        for bits in [4u32, 8u32] {
                            let key = kernels::rmsnorm_qkv::function_name_for(fusion, bits);
                            if seen_quant.insert(key) {
                                kernels::rmsnorm_qkv::emit_quant_cache_sink(
                                    fusion,
                                    &task.partition,
                                    meta,
                                    bits,
                                    64,
                                    &mut w,
                                )?;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Pipelined-decode utility kernels. `step_init` atomically
    // advances the offset counter at the head of each ICB execution
    // (broadcasts new offset/kv_len to the shared uniforms every
    // layer reads). `history_capture` appends the freshly-written
    // argmax to the chunk's history buffer at the tail. Together
    // they let the runtime pre-submit a chain of decode cmd buffers
    // with the per-token feedback loop closed entirely on the GPU.
    kernels::emit_pipeline_utilities(&mut w);

    // Smoke kernels composed via the piece-IR. Not dispatched at
    // runtime — emit unconditionally so Metal's compiler validates
    // the composer's MSL output every build. Caught regressions
    // here as early as possible; remove once the composer drives
    // real fusion patterns and that's the implicit smoke.
    {
        use crate::codegen::pieces::{
            compose, linear_residual::LinearResidual, residual_add::ResidualAdd,
            rmsnorm_reduce::RmsNormReduce, rope_kv_cache_store::RopeKvCacheStore,
            sdpa_decode::SdpaDecode, swiglu_activation::SwigluActivation, EmitCtx, KernelPiece,
        };
        let _ = ResidualAdd { n: 0 }; // silence unused-import lint when omitted
        let ctx = EmitCtx {
            meta,
            act_ty: kernels::act_ty(meta),
        };
        w.line("// ── piece-IR smoke kernels ─────────────────────────");
        let pieces: Vec<Box<dyn KernelPiece>> = vec![Box::new(SwigluActivation { n: 64 })];
        compose("fusekit_piece_smoke", &pieces, &ctx, &mut w);
        // Compose with SDPA — biggest piece, exercises SMEM regions +
        // multi-stage barriers + composer's per-stage flow.
        let sdpa_pieces: Vec<Box<dyn KernelPiece>> = vec![Box::new(SdpaDecode::bf16())];
        compose("fusekit_piece_smoke_sdpa", &sdpa_pieces, &ctx, &mut w);
        // SDPA piece with quantized KV cache — both 4-bit and 8-bit
        // baked. Validates the dequant inner-loop emit. Group size
        // 64 is mlx-lm's `QuantizedKVCache` default.
        let sdpa_q4_pieces: Vec<Box<dyn KernelPiece>> = vec![Box::new(SdpaDecode::quant(4, 64))];
        compose("fusekit_piece_smoke_sdpa_q4", &sdpa_q4_pieces, &ctx, &mut w);
        let sdpa_q8_pieces: Vec<Box<dyn KernelPiece>> = vec![Box::new(SdpaDecode::quant(8, 64))];
        compose("fusekit_piece_smoke_sdpa_q8", &sdpa_q8_pieces, &ctx, &mut w);
        // RoPE+KVCacheStore piece — bf16 default + quant variants
        // (matches mlx-lm's `QuantizedKVCache` group-wise affine layout).
        // theta=1e6 is the Qwen2.5 default; runtime gets the right
        // value from the per-layer RopeKvCacheStoreDecode fusion attrs.
        let theta = 1e6_f32;
        let kvstore_pieces: Vec<Box<dyn KernelPiece>> =
            vec![Box::new(RopeKvCacheStore::bf16(theta))];
        compose("fusekit_piece_smoke_kvstore", &kvstore_pieces, &ctx, &mut w);
        let kvstore_q4_pieces: Vec<Box<dyn KernelPiece>> =
            vec![Box::new(RopeKvCacheStore::quant(theta, 4, 64))];
        compose(
            "fusekit_piece_smoke_kvstore_q4",
            &kvstore_q4_pieces,
            &ctx,
            &mut w,
        );
        let kvstore_q8_pieces: Vec<Box<dyn KernelPiece>> =
            vec![Box::new(RopeKvCacheStore::quant(theta, 8, 64))];
        compose(
            "fusekit_piece_smoke_kvstore_q8",
            &kvstore_q8_pieces,
            &ctx,
            &mut w,
        );
        // Per-layer megakernel skeleton: RMSNorm → SDPA → Residual.
        // Composer now unifies the `x_input` role across
        // RmsNormReduce (Read) and ResidualAdd (Write) into a
        // single read+write `x` buffer slot — the layer residual
        // stream shared between the input norm and the trailing
        // residual add. SMEM regions (`norm_y`, `scores`,
        // `sum_sq_partial`, `sdpa_reduce_buf`) all coexist in 19.5
        // KiB, fitting Apple's 32 KiB-per-threadgroup budget.
        // Fuller per-layer skeleton: RmsNorm → SDPA → LinearResidual
        // (o_proj + residual into the layer x stream). The
        // `x_input` role is shared by RmsNormReduce (Read),
        // LinearResidual (Read+Write — residual add destination),
        // composer collapses to one `device {t} *x` slot.
        let quant = meta.quant_bits.and_then(|bits| {
            meta.quant_group_size
                .map(|gs| crate::fusion::ir::QuantInfo {
                    bits,
                    group_size: gs,
                })
        });
        let n_q_heads = meta.num_attention_heads * meta.head_dim;
        let layer_pieces: Vec<Box<dyn KernelPiece>> = vec![
            Box::new(RmsNormReduce::new(1e-6)),
            Box::new(SdpaDecode::bf16()),
            Box::new(LinearResidual {
                m: n_q_heads,
                n: meta.hidden_size,
                act_var: "sdpa_out".into(),
                act_role: "sdpa_out".into(),
                tag: "o".into(),
                quant,
                has_bias: false,
            }),
        ];
        compose("fusekit_piece_smoke_layer", &layer_pieces, &ctx, &mut w);
    }

    // Layer-block matcher emit. When FUSEKIT_LAYER_MEGAKERNEL=1, the
    // matcher walks `fg` for per-layer fusions, synthesizes one
    // LayerBlock per layer index it finds, and emits each one
    // through `kernels::emit_for`. Production dispatch still uses the
    // per-pattern fusions — this path validates that the matcher +
    // composer + Metal shader compiler agree on the per-layer kernel
    // for the model's actual params (eps, theta, quant). Once the
    // task-graph rewrite lands the dispatcher will route through
    // these kernels instead of the 7 per-layer pattern kernels.
    if crate::fusion::layer_block::enabled() {
        let blocks = crate::fusion::layer_block::synthesize(fg, graph);
        if !blocks.is_empty() {
            w.line("// ── layer-block matcher emits ──────────────────────");
            let placeholder_partition = crate::task::ir::PartitionPlan {
                grid: (1, 1, 1),
                bindings: vec![],
            };
            for fusion in &blocks {
                kernels::emit_for(fusion, &placeholder_partition, meta, &mut w)?;
            }
        }
    }

    persistent::emit_megakernel(tg, &mut w)?;

    let msl_path = out_dir.join("kernel.metal");
    fs::write(&msl_path, w.finish()).with_context(|| format!("write {}", msl_path.display()))?;

    let metallib_path = out_dir.join("kernel.metallib");
    metal_compile::compile_to_metallib(&msl_path, &metallib_path)?;

    // task_graph.json is emitted last so a half-finished metallib can't
    // be mistaken for a complete artifact by the runtime. v2 schema —
    // see `task_graph_json.rs` for shape; the runtime auto-dispatcher
    // (D-3) consumes it directly.
    let tg_path = out_dir.join("task_graph.json");
    let tg_doc = task_graph_json::build(graph, fg, tg, meta, params, &target);
    let tg_json = serde_json::to_string_pretty(&tg_doc)?;
    fs::write(&tg_path, tg_json)?;

    Ok(metallib_path)
}
