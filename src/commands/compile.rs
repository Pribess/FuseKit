//! `fusekit compile <model>` — drive the compile pipeline end-to-end.
//!
//! ```text
//!   model_dir
//!     │  loader::capture
//!     ▼
//!   Graph (Stage 1)
//!     │  fusion::build_fusions
//!     ▼
//!   FusionGraph (Stage 2)
//!     │  task::build_tasks
//!     ▼
//!   TaskGraph (Stage 3)
//!     │  codegen::emit_metallib
//!     ▼
//!   <out>/kernel.metallib + task_graph.json
//! ```
//!
//! No mlx-lm hook is installed here — that's `fusekit patch`. This
//! command exists so we can iterate on the pipeline without touching
//! the user's mlx-lm installation.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::fusion::ir::ModelMeta;
use crate::target::Target;
use crate::{codegen, fusion, loader, omlx, task, ui};

pub fn run(model: &str, out: Option<PathBuf>, chip: Option<&str>) -> Result<()> {
    let target = Target::resolve(chip).map_err(|e| anyhow::anyhow!("--chip: {e}"))?;
    let m = omlx::find_model(model)?;
    ui::ok(format!("model: {}", m.name));
    ui::dim_kv(
        "target",
        format!(
            "{} (regs={} smem={}KB)",
            target.chip.display(),
            target.budgets.regs_per_thread,
            target.budgets.smem_per_threadgroup / 1024,
        ),
    );

    let meta =
        ModelMeta::from_config_file(&m.path.join("config.json")).context("read model config")?;
    ui::dim_kv("layers", meta.num_hidden_layers);
    ui::dim_kv("hidden", meta.hidden_size);
    ui::dim_kv(
        "heads/kv",
        format!("{}/{}", meta.num_attention_heads, meta.num_key_value_heads),
    );
    match (meta.quant_bits, meta.quant_group_size) {
        (Some(b), Some(g)) => ui::dim_kv("quant", format!("{}-bit g{}", b, g)),
        _ => ui::dim_kv("quant", "bf16"),
    }

    ui::step("capture (Stage 1)");
    let raw = loader::capture(&m.path).context("capture graph")?;
    let mut graph = loader::lower(&raw).context("lower IR")?;
    crate::layer_tag::infer_layer_tags(&mut graph);
    ui::ok(format!(
        "  graph: {} tensors, {} ops",
        graph.tensors.len(),
        graph.ops.len()
    ));

    ui::step("fuse (Stage 2)");
    let fusion_graph =
        fusion::build::build_fusions(&graph, meta.clone()).context("build fusions")?;
    // Stage 3: semantic-aware general fusion (no-op unless
    // `FUSEKIT_GENERAL_FUSION` env var is set).
    let mut fusion_graph = fusion::general::run(fusion_graph, &target, &graph);
    if crate::fusion::layer_block::enabled() {
        crate::fusion::layer_block::integrate(&mut fusion_graph, &graph);
    }
    if std::env::var_os("FUSEKIT_IR_DISABLE").is_none() {
        let mut ir_module =
            crate::op::lower_from_fusion::lower_with_graph(&fusion_graph, Some(&graph));
        let ops_before_dce = ir_module.ops.len();
        let mut pm = crate::op::PassManager::new();
        pm.add(Box::new(crate::op::LayoutInference));
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
            ui::ok(format!("[ir] pass error: {e}"));
        } else {
            let live: std::collections::HashSet<u32> = ir_module.ops.iter().map(|o| o.id).collect();
            let fusions_before = fusion_graph.fusions.len();
            fusion_graph.fusions.retain(|f| {
                live.contains(&f.id)
                    || !matches!(
                        f.kind,
                        crate::fusion::ir::FusionKind::Single(_)
                            | crate::fusion::ir::FusionKind::FusedChain
                    )
            });
            for (idx, f) in fusion_graph.fusions.iter_mut().enumerate() {
                f.id = idx as u32;
            }
            let dropped = fusions_before - fusion_graph.fusions.len();
            ui::ok(format!(
                "[ir] DCE: {ops_before_dce} -> {} ops, {} fusions dropped from emit",
                ir_module.ops.len(),
                dropped,
            ));
        }
    }
    ui::ok(format!("  fusions: {}", fusion_graph.fusions.len()));

    ui::step("schedule (Stage 3)");
    let task_graph = task::build::build_tasks(&fusion_graph).context("build tasks")?;
    ui::ok(format!(
        "  tasks: {}, events: {}, initial: {}",
        task_graph.tasks.len(),
        task_graph.events.len(),
        task_graph.initial.len()
    ));
    let out_dir = out.unwrap_or_else(|| m.path.join("fusekit"));
    let summary = super::coverage::summarize(&graph, &fusion_graph, &task_graph);
    super::coverage::print_summary("compile", &summary);

    ui::step(format!("codegen + compile → {}", out_dir.display()));
    let lib = codegen::emit_metallib(
        &graph,
        &fusion_graph,
        &task_graph,
        &meta,
        &raw.params,
        &out_dir,
        target,
    )
    .context("emit metallib")?;
    ui::ok(format!("  wrote {}", lib.display()));
    if std::env::var_os("FUSEKIT_WRITE_GRAPHS").is_some() {
        let viz = super::coverage::write_visualizations(
            &m.name,
            &graph,
            &fusion_graph,
            &task_graph,
            &out_dir,
        )
        .context("write graph visualizations")?;
        super::coverage::print_visualization_paths(&viz);
    }

    Ok(())
}
