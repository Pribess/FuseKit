//! `fusekit patch <model>` — compile + install hook.
//!
//! ```text
//!   patch flow
//!   ──────────
//!   1. resolve model dir
//!   2. check existing manifest:
//!        - matches running fusekit → no-op (unless --force)
//!        - older / corrupt        → wipe <model>/fusekit and rebuild
//!        - missing                → fresh build
//!   3. compile pipeline → kernel.metallib + task_graph.json
//!   4. extract runtime payload → <model>/fusekit/runtime/
//!   5. write manifest.json
//!   6. ensure user-site .pth is installed
//! ```

use anyhow::{Context, Result};

use crate::fusion::ir::ModelMeta;
use crate::manifest::{self, CompileStats, CompileVariant, Manifest, ModelInfo};
use crate::target::Target;
use crate::{codegen, fusion, hook, loader, omlx, runtime_payload, task, ui};

struct CompileGraph {
    fg: crate::fusion::ir::FusionGraph,
    tg: crate::task::ir::TaskGraph,
}

pub fn run(model: &str, chip: Option<&str>, force: bool) -> Result<()> {
    let target = Target::resolve(chip).map_err(|e| anyhow::anyhow!("--chip: {e}"))?;
    ui::dim_kv(
        "target",
        format!(
            "{} (regs={} smem={}KB)",
            target.chip.display(),
            target.budgets.regs_per_thread,
            target.budgets.smem_per_threadgroup / 1024,
        ),
    );
    let m = omlx::find_model(model)?;
    let fusekit_dir = m.path.join("fusekit");

    // Skip-or-replace decision based on existing manifest.
    if let Some(existing) = Manifest::read_if_exists(&fusekit_dir)? {
        if existing.matches_running_binary() && !force {
            ui::ok(format!(
                "{model} already patched with fusekit {}",
                existing.fusekit_version
            ));
            ui::dim_kv("compiled_at", &existing.compiled_at);
            ui::skip("nothing to do (use --force to recompile)");
            return Ok(());
        }
        if force {
            ui::step("rebuilding patch (--force)");
        } else {
            ui::step(format!(
                "upgrading patch ({} → {})",
                existing.fusekit_version,
                manifest::fusekit_version()
            ));
        }
        std::fs::remove_dir_all(&fusekit_dir)
            .with_context(|| format!("remove old {}", fusekit_dir.display()))?;
    } else if fusekit_dir.exists() {
        // Stale directory without a manifest (e.g. interrupted patch).
        // Clean it out so we never mix old artifacts with new ones.
        std::fs::remove_dir_all(&fusekit_dir)
            .with_context(|| format!("remove stale {}", fusekit_dir.display()))?;
    }

    ui::step(format!("compiling megakernel for {model}"));
    let meta = ModelMeta::from_config_file(&m.path.join("config.json"))?;
    let raw = loader::capture(&m.path)?;
    let mut graph = loader::lower(&raw)?;
    crate::layer_tag::infer_layer_tags(&mut graph);
    let requested_active_m = env_u32("FUSEKIT_ACTIVE_M");
    let requested_dense = std::env::var("FUSEKIT_DENSE_MATMUL_MODE")
        .unwrap_or_else(|_| target.chip.dense_matmul_mode().as_str().to_string());
    let base = build_compile_graph(&graph, &meta, target, 1)?;
    let base_summary = super::coverage::summarize(&graph, &base.fg, &base.tg);
    super::coverage::print_summary("M=1", &base_summary);

    emit_with_active_m(
        &graph,
        &base.fg,
        &base.tg,
        &meta,
        &raw.params,
        &fusekit_dir,
        target,
        1,
    )
    .context("emit metallib")?;
    ui::ok(format!(
        "wrote {}",
        fusekit_dir.join("kernel.metallib").display()
    ));
    if std::env::var_os("FUSEKIT_WRITE_GRAPHS").is_some() {
        let viz = super::coverage::write_visualizations(
            &m.name,
            &graph,
            &base.fg,
            &base.tg,
            &fusekit_dir,
        )
        .context("write graph visualizations")?;
        super::coverage::print_visualization_paths(&viz);
    }

    let mut variants = Vec::new();
    if let Some(max_mtp_m) = mtp_variant_active_m(&meta, &base.fg, requested_active_m) {
        for mtp_m in 2..=max_mtp_m {
            let variant_name = format!("mtp_m{mtp_m}");
            let variant_dir = fusekit_dir.join("variants").join(&variant_name);
            ui::step(format!("compiling {variant_name} variant"));
            let variant = build_compile_graph(&graph, &meta, target, mtp_m)?;
            let variant_summary = super::coverage::summarize(&graph, &variant.fg, &variant.tg);
            super::coverage::print_summary(&variant_name, &variant_summary);
            emit_with_active_m(
                &graph,
                &variant.fg,
                &variant.tg,
                &meta,
                &raw.params,
                &variant_dir,
                target,
                mtp_m,
            )
            .with_context(|| format!("emit {variant_name} metallib"))?;
            ui::ok(format!(
                "wrote {}",
                variant_dir.join("kernel.metallib").display()
            ));
            if std::env::var_os("FUSEKIT_WRITE_GRAPHS").is_some() {
                let variant_viz = super::coverage::write_visualizations(
                    &format!("{} {variant_name}", m.name),
                    &graph,
                    &variant.fg,
                    &variant.tg,
                    &variant_dir,
                )
                .with_context(|| format!("write {variant_name} graph visualizations"))?;
                ui::dim_kv(
                    &format!("{variant_name} graph"),
                    variant_viz.emitted_html.display(),
                );
            }
            variants.push(CompileVariant {
                name: variant_name,
                path: "variants".to_string() + &format!("/mtp_m{mtp_m}"),
                active_m: mtp_m,
                dense_matmul: requested_dense.clone(),
                tasks: variant.tg.tasks.len(),
                unique_kernels: variant.tg.tasks.len(),
            });
        }
    }

    runtime_payload::materialize_into(&fusekit_dir).context("extract fusekit_runtime payload")?;
    ui::ok("extracted runtime into <model>/fusekit/runtime/");

    let manifest_obj = Manifest {
        schema_version: manifest::CURRENT_SCHEMA_VERSION,
        fusekit_version: manifest::fusekit_version().to_string(),
        compiled_at: manifest::now_iso8601(),
        model: ModelInfo {
            name: m.name.clone(),
            model_type: meta.model_type.clone(),
            hidden_size: meta.hidden_size,
            num_hidden_layers: meta.num_hidden_layers,
            mtp: meta.mtp.clone(),
        },
        stats: CompileStats {
            ops: graph.ops.len(),
            fusions: base.fg.fusions.len(),
            tasks: base.tg.tasks.len(),
            unique_kernels: base.tg.tasks.len(), // dedupe lands in a later step
        },
        variants,
    };
    manifest_obj.write(&fusekit_dir).context("write manifest")?;
    ui::ok("manifest written");

    // The .pth scans every model's fusekit/runtime/ at interpreter
    // start. We pass the glob pattern (with `**` for recursive) so
    // any future patched model is auto-discovered without re-installing.
    let models_glob = omlx::models_dir()?.join("**/fusekit/runtime");
    if let Err(e) = hook::install(&models_glob) {
        let _ = std::fs::remove_dir_all(&fusekit_dir);
        return Err(e.context("install fusekit hook"));
    }
    ui::ok("hook installed (single .pth in user site-packages)");
    ui::dim_kv("fusekit_version", manifest::fusekit_version());
    if meta.mtp.is_declared() {
        ui::dim_kv(
            "mtp",
            format!(
                "{:?} layers={} max_depth={} weights={}",
                meta.mtp.format,
                meta.mtp.declared_layers,
                meta.mtp
                    .max_depth
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                meta.mtp.has_weights
            ),
        );
    }
    ui::dim_kv("fusekit dir", fusekit_dir.display());
    Ok(())
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| n >= 1)
}

fn build_compile_graph(
    graph: &crate::graph::Graph,
    meta: &ModelMeta,
    target: Target,
    active_m: u32,
) -> Result<CompileGraph> {
    with_env_override(
        "FUSEKIT_ACTIVE_M",
        Some(active_m.max(1).to_string()),
        || {
            let fg = fusion::build::build_fusions(graph, meta.clone())?;
            // Stage 3: semantic-aware general fusion. Build it under the
            // active-M for this artifact, so MTP variants admit M>1 merges
            // instead of inheriting the base decode graph.
            let mut fg = fusion::general::run(fg, &target, graph);
            if crate::fusion::layer_block::enabled() {
                crate::fusion::layer_block::integrate(&mut fg, graph);
            }
            apply_ir_dce(graph, &mut fg, active_m);
            let tg = task::build::build_tasks(&fg)?;
            Ok(CompileGraph { fg, tg })
        },
    )
}

fn apply_ir_dce(
    graph: &crate::graph::Graph,
    fg: &mut crate::fusion::ir::FusionGraph,
    active_m: u32,
) {
    // IR-driven analysis (first chunk of codegen→IR migration).
    // Lower FusionGraph → op::Module, run passes (layout inference,
    // DCE), then apply the DCE verdict back onto FusionGraph. The
    // emit side still consumes FusionGraph for now — this is the
    // minimum-viable bridge that makes IR work affect actual output.
    // `FUSEKIT_IR_DISABLE` opt-out for bisecting perf regressions.
    if std::env::var_os("FUSEKIT_IR_DISABLE").is_some() {
        return;
    }

    let mut ir_module = crate::op::lower_from_fusion::lower_with_graph(fg, Some(graph));
    let ops_before_dce = ir_module.ops.len();
    let mut pm = crate::op::PassManager::new();
    pm.add(Box::new(crate::op::LayoutInference));
    // Algebraic rewrites + algorithmic macros all run under one
    // AlgebraicRewrites instance to share the fixed-point loop.
    let mut rewriter = crate::op::AlgebraicRewrites::with_defaults();
    rewriter.rules.push(Box::new(crate::op::SoftmaxMaxStable));
    rewriter.rules.push(Box::new(crate::op::RmsNormFastRsqrt));
    rewriter
        .rules
        .push(Box::new(crate::op::KvSliceUpdateContiguous));
    pm.add(Box::new(rewriter));
    pm.add(Box::new(crate::op::InPlaceAnalysis));
    pm.add(Box::new(crate::op::DecodeL1Specialize));
    pm.add(Box::new(crate::op::DeadCodeElimination));
    if let Err(e) = pm.run(&mut ir_module) {
        ui::ok(format!("[ir M={active_m}] pass error: {e}"));
        return;
    }

    // Live set = IR ops that survived DCE. Op.id matches fusion.id
    // because `fusion::general::run` compacts + renumbers fusions
    // sequentially, and `lower_from_fusion` mirrors that. Retain only
    // the live fusions and recompact so downstream `task::build` sees a
    // clean graph with contiguous IDs.
    let live: std::collections::HashSet<u32> = ir_module.ops.iter().map(|o| o.id).collect();
    let fusions_before = fg.fusions.len();
    fg.fusions.retain(|f| {
        live.contains(&f.id)
            || !matches!(
                f.kind,
                crate::fusion::ir::FusionKind::Single(_)
                    | crate::fusion::ir::FusionKind::FusedChain
            )
    });
    for (idx, f) in fg.fusions.iter_mut().enumerate() {
        f.id = idx as u32;
    }
    let dropped = fusions_before - fg.fusions.len();
    ui::ok(format!(
        "[ir M={active_m}] DCE: {ops_before_dce} → {} ops, {} fusions dropped from emit",
        ir_module.ops.len(),
        dropped,
    ));
}

fn trace_m_from_fusions(fg: &crate::fusion::ir::FusionGraph) -> u32 {
    fg.fusions
        .iter()
        .flat_map(|f| f.inputs.iter().chain(f.outputs.iter()))
        .filter_map(|b| {
            if b.shape.len() >= 3 {
                Some(b.shape[1].max(1) as u32)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(1)
}

fn mtp_variant_active_m(
    meta: &ModelMeta,
    fg: &crate::fusion::ir::FusionGraph,
    requested_active_m: Option<u32>,
) -> Option<u32> {
    if !meta.mtp.is_declared() || !meta.mtp.has_weights {
        return None;
    }
    let default_m = meta
        .mtp
        .max_depth
        .unwrap_or_else(|| meta.mtp.declared_layers.saturating_add(1).max(2));
    let requested = requested_active_m.unwrap_or(default_m).max(2);
    let trace_m = trace_m_from_fusions(fg).max(1);
    Some(requested.min(trace_m).max(2))
}

fn emit_with_active_m(
    graph: &crate::graph::Graph,
    fg: &crate::fusion::ir::FusionGraph,
    tg: &crate::task::ir::TaskGraph,
    meta: &ModelMeta,
    params: &[crate::loader::RawParam],
    out_dir: &std::path::Path,
    target: Target,
    active_m: u32,
) -> Result<std::path::PathBuf> {
    with_env_override("FUSEKIT_ACTIVE_M", Some(active_m.to_string()), || {
        codegen::emit_metallib(graph, fg, tg, meta, params, out_dir, target)
    })
}

fn with_env_override<T>(
    key: &str,
    value: Option<String>,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let old = std::env::var_os(key);
    match value {
        Some(value) => std::env::set_var(key, value),
        None => std::env::remove_var(key),
    }
    let result = f();
    match old {
        Some(value) => std::env::set_var(key, value),
        None => std::env::remove_var(key),
    }
    result
}
