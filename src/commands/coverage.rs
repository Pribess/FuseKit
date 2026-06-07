use anyhow::{Context, Result};
use comfy_table::{
    modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Attribute, Cell, CellAlignment, Color,
    ContentArrangement, Table,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;
use std::path::{Path, PathBuf};

use crate::fusion::ir::{Fusion, FusionGraph, FusionKind};
use crate::graph::{Graph, LayerTag};
use crate::task::ir::TaskGraph;
use crate::ui;

#[derive(Debug, Clone)]
pub struct EmissionSummary {
    pub source_ops_total: usize,
    pub source_ops_emitted: usize,
    pub source_ops_single: usize,
    pub source_ops_specialized: usize,
    pub tasks_total: usize,
    pub tasks_fusekit_kernel: usize,
    pub tasks_unimplemented: usize,
    pub tasks_single: usize,
    pub tasks_specialized: usize,
    pub unique_kernels: usize,
    pub kind_hist: Vec<(String, usize)>,
}

pub struct VizPaths {
    pub captured_dot: PathBuf,
    pub captured_html: PathBuf,
    pub emitted_dot: PathBuf,
    pub emitted_html: PathBuf,
}

pub fn summarize(graph: &Graph, fg: &FusionGraph, tg: &TaskGraph) -> EmissionSummary {
    let mut op_class: HashMap<u32, &'static str> = HashMap::new();
    let mut kind_hist: BTreeMap<String, usize> = BTreeMap::new();

    for fusion in &fg.fusions {
        let class = if matches!(fusion.kind, FusionKind::Single(_)) {
            "single"
        } else {
            "specialized"
        };
        for &op_id in &fusion.op_ids {
            op_class
                .entry(op_id)
                .and_modify(|old| {
                    if class == "specialized" {
                        *old = "specialized";
                    }
                })
                .or_insert(class);
        }
        *kind_hist.entry(kind_label(&fusion.kind)).or_insert(0) += 1;
    }

    let mut unique_kernels: BTreeSet<String> = BTreeSet::new();
    let mut tasks_unimplemented = 0usize;
    let mut tasks_single = 0usize;
    let mut tasks_specialized = 0usize;
    for task in &tg.tasks {
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            tasks_unimplemented += 1;
            continue;
        };
        let kernel = crate::codegen::kernels::kernel_function_name(fusion);
        if kernel.starts_with("fusekit_unimpl_") {
            tasks_unimplemented += 1;
        } else {
            unique_kernels.insert(kernel);
        }
        if matches!(fusion.kind, FusionKind::Single(_)) {
            tasks_single += 1;
        } else {
            tasks_specialized += 1;
        }
    }

    let source_ops_single = op_class.values().filter(|&&v| v == "single").count();
    let source_ops_specialized = op_class.values().filter(|&&v| v == "specialized").count();
    let mut kind_hist: Vec<_> = kind_hist.into_iter().collect();
    kind_hist.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    EmissionSummary {
        source_ops_total: graph.ops.len(),
        source_ops_emitted: op_class.len(),
        source_ops_single,
        source_ops_specialized,
        tasks_total: tg.tasks.len(),
        tasks_fusekit_kernel: tg.tasks.len().saturating_sub(tasks_unimplemented),
        tasks_unimplemented,
        tasks_single,
        tasks_specialized,
        unique_kernels: unique_kernels.len(),
        kind_hist,
    }
}

pub fn print_summary(label: &str, summary: &EmissionSummary) {
    ui::ok(format!("[{label}] patch coverage"));
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Metric").add_attribute(Attribute::Bold),
            Cell::new("Count")
                .add_attribute(Attribute::Bold)
                .set_alignment(CellAlignment::Right),
            Cell::new("Share")
                .add_attribute(Attribute::Bold)
                .set_alignment(CellAlignment::Right),
        ]);

    table.add_row(metric_row(
        "custom task kernels",
        summary.tasks_fusekit_kernel,
        summary.tasks_total,
        Color::Green,
    ));
    table.add_row(metric_row(
        "native fallback tasks",
        summary.tasks_unimplemented,
        summary.tasks_total,
        if summary.tasks_unimplemented == 0 {
            Color::DarkGrey
        } else {
            Color::Red
        },
    ));
    table.add_row(metric_row(
        "fused/specialized tasks",
        summary.tasks_specialized,
        summary.tasks_total,
        Color::Cyan,
    ));
    table.add_row(metric_row(
        "single-op custom tasks",
        summary.tasks_single,
        summary.tasks_total,
        Color::Yellow,
    ));
    table.add_row(metric_row(
        "source ops covered",
        summary.source_ops_emitted,
        summary.source_ops_total,
        Color::Green,
    ));
    table.add_row(metric_row(
        "source ops fused",
        summary.source_ops_specialized,
        summary.source_ops_emitted,
        Color::Cyan,
    ));
    table.add_row(metric_row(
        "source ops single",
        summary.source_ops_single,
        summary.source_ops_emitted,
        Color::Yellow,
    ));
    table.add_row(metric_row(
        "DCE/synthetic/runtime ops",
        summary
            .source_ops_total
            .saturating_sub(summary.source_ops_emitted),
        summary.source_ops_total,
        Color::DarkGrey,
    ));
    println!("{table}");
    ui::dim_kv("unique custom kernels", summary.unique_kernels);
    let top = summary
        .kind_hist
        .iter()
        .take(10)
        .map(|(name, n)| format!("{name}={n}"))
        .collect::<Vec<_>>()
        .join(", ");
    if !top.is_empty() {
        ui::dim_kv("top fusion kinds", top);
    }
}

pub fn write_visualizations(
    model_name: &str,
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
    out_dir: &Path,
) -> Result<VizPaths> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let captured_dot = out_dir.join("captured_graph.dot");
    let captured_html = out_dir.join("captured_graph.html");
    let emitted_dot = out_dir.join("emitted_task_graph.dot");
    let emitted_html = out_dir.join("emitted_task_graph.html");

    let raw_dot = graph.to_dot();
    std::fs::write(&captured_dot, &raw_dot)
        .with_context(|| format!("write {}", captured_dot.display()))?;
    std::fs::write(
        &captured_html,
        render_html(&format!("{model_name} captured graph"), &raw_dot),
    )
    .with_context(|| format!("write {}", captured_html.display()))?;

    let emitted = emitted_task_graph_dot(model_name, graph, fg, tg);
    std::fs::write(&emitted_dot, &emitted)
        .with_context(|| format!("write {}", emitted_dot.display()))?;
    std::fs::write(
        &emitted_html,
        render_html(&format!("{model_name} emitted task graph"), &emitted),
    )
    .with_context(|| format!("write {}", emitted_html.display()))?;

    Ok(VizPaths {
        captured_dot,
        captured_html,
        emitted_dot,
        emitted_html,
    })
}

pub fn print_visualization_paths(paths: &VizPaths) {
    ui::ok("graph visualizations written");
    ui::dim_kv("captured html", paths.captured_html.display());
    ui::dim_kv("emitted html", paths.emitted_html.display());
    ui::dim_kv("captured dot", paths.captured_dot.display());
    ui::dim_kv("emitted dot", paths.emitted_dot.display());
}

fn emitted_task_graph_dot(
    model_name: &str,
    graph: &Graph,
    fg: &FusionGraph,
    tg: &TaskGraph,
) -> String {
    let summary = summarize(graph, fg, tg);
    let mut s = String::new();
    s.push_str("digraph fusekit_emitted {\n");
    s.push_str("  rankdir=TB;\n");
    let _ = writeln!(
        s,
        "  graph [splines=ortho, nodesep=0.22, ranksep=0.34, labelloc=t, label=\"{}\\nTasks: {}/{} fusekit, specialized task share {}, source-op specialized share {}\"];\n",
        dot_quote(model_name),
        summary.tasks_fusekit_kernel,
        summary.tasks_total,
        pct(summary.tasks_specialized, summary.tasks_total),
        pct(summary.source_ops_specialized, summary.source_ops_emitted),
    );
    s.push_str("  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=9, color=\"#64748b\"];\n");
    s.push_str("  edge [fontname=\"Helvetica\", fontsize=8, color=\"#94a3b8\"];\n");

    let mut event_producer: HashMap<u32, u32> = HashMap::new();
    for task in &tg.tasks {
        event_producer.insert(task.trigger, task.id);
    }

    for task in &tg.tasks {
        let Some(fusion) = fg.fusions.get(task.fusion as usize) else {
            continue;
        };
        let kernel = crate::codegen::kernels::kernel_function_name(fusion);
        let kind = kind_label(&fusion.kind);
        let fill = node_fill(fusion, &kernel);
        let layer = layer_label(fusion);
        let _ = writeln!(
            s,
            "  t{} [fillcolor=\"{}\", label=\"#{} f{} {}\\n{}\\n{}\\nops={} grid={:?} tg={:?}\"];",
            task.id,
            fill,
            task.id,
            task.fusion,
            dot_quote(&layer),
            dot_quote(&kind),
            dot_quote(&kernel),
            fusion.op_ids.len(),
            task.partition.grid,
            crate::codegen::kernels::kernel_threadgroup(fusion),
        );
    }

    for task in &tg.tasks {
        for dep in &task.deps {
            if let Some(src) = event_producer.get(dep) {
                let _ = writeln!(s, "  t{} -> t{};", src, task.id);
            }
        }
    }

    s.push_str("}\n");
    s
}

fn kind_label(kind: &FusionKind) -> String {
    match kind {
        FusionKind::Single(name) => format!("Single({name})"),
        other => format!("{other:?}"),
    }
}

fn layer_label(fusion: &Fusion) -> String {
    match fusion.layer_hint {
        Some(LayerTag::Embed) => "embed".to_string(),
        Some(LayerTag::Layer(n)) => format!("L{n}"),
        Some(LayerTag::Post) => "post".to_string(),
        None => "misc".to_string(),
    }
}

fn node_fill(fusion: &Fusion, kernel: &str) -> &'static str {
    if kernel.starts_with("fusekit_unimpl_") {
        "#fecaca"
    } else {
        match fusion.kind {
            FusionKind::Single(_) => "#fef3c7",
            FusionKind::FusedChain | FusionKind::QmvFanout | FusionKind::QmvSwiglu => "#dcfce7",
            _ => "#dbeafe",
        }
    }
}

fn pct(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", (n as f64) * 100.0 / (d as f64))
    }
}

fn metric_row(label: &str, n: usize, d: usize, color: Color) -> Vec<Cell> {
    vec![
        Cell::new(label).fg(color),
        Cell::new(format!("{n}/{d}"))
            .fg(color)
            .set_alignment(CellAlignment::Right),
        Cell::new(pct(n, d))
            .fg(color)
            .set_alignment(CellAlignment::Right),
    ]
}

fn render_html(title: &str, dot: &str) -> String {
    let escaped = dot
        .replace('\\', "\\\\")
        .replace('`', "\\`")
        .replace("${", "$\\{");
    format!(
        r#"<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<title>fusekit graph - {title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; }}
  #toolbar {{ position: fixed; top: 0; left: 0; right: 0; height: 36px; background: #111827; color: #e5e7eb; display: flex; align-items: center; gap: 12px; padding: 0 12px; font-size: 12px; z-index: 2; }}
  #graph {{ width: 100vw; height: 100vh; padding-top: 36px; box-sizing: border-box; overflow: hidden; }}
  #graph svg {{ width: 100%; height: calc(100vh - 36px); }}
  button {{ background: #1f2937; color: #f9fafb; border: 1px solid #374151; border-radius: 4px; padding: 4px 8px; }}
</style>
<div id="toolbar">
  <strong>{title}</strong>
  <button id="fit">Fit</button>
  <span>Blue=specialized semantic kernel, green=general fused kernel, yellow=single-op custom kernel, red=unimplemented</span>
</div>
<div id="graph"></div>
<script src="https://cdn.jsdelivr.net/npm/@viz-js/viz@3.8.0/lib/viz-standalone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.2/dist/svg-pan-zoom.min.js"></script>
<script>
const dot = `{escaped}`;
const container = document.getElementById('graph');
let panZoom = null;
Viz.instance().then(viz => {{
  const svg = viz.renderSVGElement(dot);
  container.appendChild(svg);
  panZoom = svgPanZoom(svg, {{ controlIconsEnabled: true, fit: true, center: true, minZoom: 0.02, maxZoom: 64 }});
}});
document.getElementById('fit').onclick = () => {{
  if (panZoom) {{ panZoom.fit(); panZoom.center(); }}
}};
</script>
</html>
"#
    )
}

fn dot_quote(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
