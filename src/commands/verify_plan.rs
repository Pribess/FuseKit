//! `fusekit verify-plan` — smoke-test the runtime auto dispatcher.
//!
//! Loads the precompiled `task_graph.json` + metallib for a model,
//! resolves every external binding against the live mlx model loaded
//! through PyO3, and reports how many bindings landed. Useful as a
//! self-test before wiring the dispatcher into the patch hook.

use anyhow::{anyhow, Context, Result};
use owo_colors::OwoColorize;

use std::collections::HashMap;

use crate::runtime::metal::MetalContext;
use crate::runtime::plan::{load_plan, resolve_externals, ExternalKey, IoDir, JsonBinding};
use crate::{omlx, paths, ui};

pub fn run(model: &str, execute: bool) -> Result<()> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyList, PyListMethods};

    let m = omlx::find_model(model)?;
    ui::step(format!("verify-plan: {}", m.name));

    let out_dir = m.path.join("fusekit");
    if !out_dir.join("task_graph.json").exists() {
        return Err(anyhow!(
            "no task_graph.json under {}; run `fusekit compile` first",
            out_dir.display()
        ));
    }

    let metallib = out_dir.join("kernel.metallib");
    let ctx = MetalContext::with_metallib(&metallib).context("metal context")?;
    let plan = load_plan(&ctx, &out_dir).context("load_plan")?;
    ui::ok(format!(
        "  tasks={}, events={}, scratch={}, kernels={}",
        plan.doc.tasks.len(),
        plan.doc.events.len(),
        plan.scratch.len(),
        plan.psos.len()
    ));

    // Load the live model through the same PyO3 path the trace uses.
    if std::env::var_os("PYTHONHOME").is_none() {
        let app = paths::mlx_app_root()?;
        std::env::set_var("PYTHONHOME", app.join("Contents/Python/cpython-3.11"));
    }
    pyo3::prepare_freethreaded_python();

    let site_packages = paths::mlx_site_packages()?;
    let model_dir_str = m
        .path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 model path"))?
        .to_string();

    let (resolved_count, unresolved_count) = Python::with_gil(|py| -> Result<(usize, usize)> {
        let sys = py
            .import_bound("sys")
            .map_err(|e| anyhow!("import sys: {e}"))?;
        let path: Bound<PyList> = sys
            .getattr("path")
            .map_err(|e| anyhow!("sys.path: {e}"))?
            .downcast_into()
            .map_err(|_| anyhow!("sys.path is not a list"))?;
        path.insert(0, site_packages.to_str().unwrap())
            .map_err(|e| anyhow!("sys.path.insert: {e}"))?;

        let mlx_lm = py
            .import_bound("mlx_lm")
            .map_err(|e| anyhow!("import mlx_lm: {e}"))?;
        let load_tuple = mlx_lm
            .getattr("load")
            .map_err(|e| anyhow!("mlx_lm.load: {e}"))?
            .call1((model_dir_str.as_str(),))
            .map_err(|e| anyhow!("mlx_lm.load(model): {e}"))?;
        let model = load_tuple
            .get_item(0)
            .map_err(|e| anyhow!("mlx_lm.load tuple [0]: {e}"))?;

        let resolved = resolve_externals(&plan, py, &model, &ctx)?;
        let ext_count = resolved.external_buffers.len();
        let unres_count = resolved.unresolved.len();

        if execute {
            // Allocate zero-filled scratch buffers for every
            // unresolved external (mlx-internal scalars, runtime
            // token inputs) so the dispatch path can run end-to-end
            // and surface kernel-level issues. Output bytes won't be
            // meaningful — this is a wiring smoke test, not a
            // correctness check.
            let mut runtime_inputs: HashMap<ExternalKey, metal::Buffer> = HashMap::new();
            for key in &resolved.unresolved {
                let (shape, dtype) = lookup_binding_shape(&plan.doc.tasks, key)
                    .ok_or_else(|| anyhow!("unresolved key not found in tasks"))?;
                let nbytes = bytes_for(&shape, &dtype).max(1);
                runtime_inputs.insert(*key, ctx.new_buffer(nbytes));
            }

            ui::step("execute (per-task encoder)");
            let t0 = std::time::Instant::now();
            resolved.execute(&ctx, &runtime_inputs).context("execute")?;
            ui::ok(format!("  done in {:?}", t0.elapsed()));

            ui::step("build ICB");
            let t0 = std::time::Instant::now();
            let icb = resolved
                .build_icb(&ctx, &runtime_inputs)
                .context("build_icb")?;
            ui::ok(format!(
                "  encoded {} commands in {:?}",
                icb.count,
                t0.elapsed()
            ));

            ui::step("execute (ICB)");
            // Warm-up + 3 timed runs so the first commit's residency
            // setup doesn't dominate.
            resolved
                .execute_icb(&ctx, &icb)
                .context("execute_icb warmup")?;
            let mut total = std::time::Duration::ZERO;
            let trials = 3;
            for _ in 0..trials {
                let t0 = std::time::Instant::now();
                resolved.execute_icb(&ctx, &icb).context("execute_icb")?;
                total += t0.elapsed();
            }
            ui::ok(format!("  mean {:?} ({} trials)", total / trials, trials));
        }

        Ok((ext_count, unres_count))
    })?;

    let total = resolved_count + unresolved_count;
    let pct = if total > 0 {
        100.0 * resolved_count as f64 / total as f64
    } else {
        0.0
    };
    ui::ok(format!(
        "  externals: {}/{} resolved ({:.1}%), {} unresolved",
        resolved_count.to_string().cyan(),
        total,
        pct,
        unresolved_count
    ));

    // Breakdown of what couldn't bind to a model parameter, separated
    // into "materialised scalar" (covered by the captured trace
    // constant value) and "truly runtime" (the caller must supply
    // per-invocation).
    let mut scalar_breakdown: std::collections::BTreeMap<(Vec<i32>, String), usize> =
        Default::default();
    let mut runtime_breakdown: std::collections::BTreeMap<(Vec<i32>, String), usize> =
        Default::default();
    for task in &plan.doc.tasks {
        for b in task.inputs.iter() {
            if let JsonBinding::External {
                source_path,
                scalar,
                shape,
                dtype,
                ..
            } = b
            {
                if !source_path.is_empty() {
                    continue;
                }
                let key = (shape.clone(), dtype.clone());
                if scalar.is_some() {
                    *scalar_breakdown.entry(key).or_insert(0) += 1;
                } else {
                    *runtime_breakdown.entry(key).or_insert(0) += 1;
                }
            }
        }
    }
    if !scalar_breakdown.is_empty() {
        println!("    {} resolved via captured scalar:", "·".dimmed());
        for ((shape, dtype), n) in &scalar_breakdown {
            println!("      {:>4}×  shape={:?} dtype={}", n, shape, dtype);
        }
    }
    if !runtime_breakdown.is_empty() {
        println!("    {} runtime-supplied (host must bind):", "·".dimmed());
        for ((shape, dtype), n) in &runtime_breakdown {
            println!("      {:>4}×  shape={:?} dtype={}", n, shape, dtype);
        }
    }

    Ok(())
}

fn lookup_binding_shape(
    tasks: &[crate::runtime::plan::JsonTask],
    key: &ExternalKey,
) -> Option<(Vec<i32>, String)> {
    let task = tasks.iter().find(|t| t.id == key.task_id)?;
    let bindings = match key.direction {
        IoDir::Input => &task.inputs,
        IoDir::Output => &task.outputs,
    };
    let binding = bindings.get(key.slot as usize)?;
    match binding {
        JsonBinding::External { shape, dtype, .. } => Some((shape.clone(), dtype.clone())),
        _ => None,
    }
}

fn bytes_for(shape: &[i32], dtype: &str) -> usize {
    let elems: i64 = shape.iter().map(|&d| d as i64).product::<i64>().max(1);
    let elem_bytes: usize = match dtype {
        "bool" | "uint8" | "int8" => 1,
        "uint16" | "int16" | "float16" | "bfloat16" => 2,
        "uint32" | "int32" | "float32" => 4,
        "uint64" | "int64" | "float64" | "complex64" => 8,
        _ => 4,
    };
    (elems as usize).saturating_mul(elem_bytes)
}
