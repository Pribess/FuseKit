use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use std::collections::BTreeMap;

use crate::{loader, omlx, ui};

pub fn run(model: &str) -> Result<()> {
    let m = omlx::find_model(model)?;
    ui::step(format!("capturing graph for {}", m.name));

    let raw = loader::capture(&m.path).context("capture graph via Python driver")?;
    ui::ok(format!("captured {} records", raw.records.len()));

    let mut graph = loader::lower(&raw).context("lower captured graph into IR")?;
    crate::layer_tag::infer_layer_tags(&mut graph);
    ui::ok(format!(
        "lowered to {} tensors, {} ops",
        graph.tensors.len(),
        graph.ops.len()
    ));

    if !raw.model_dir.is_empty() {
        ui::dim_kv("source", &raw.model_dir);
    }
    if let Some(obj) = raw.config_summary.as_object() {
        for key in [
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
        ] {
            if let Some(v) = obj.get(key) {
                ui::dim_kv(key, &v.to_string());
            }
        }
    }
    if let Ok(meta) = crate::fusion::ir::ModelMeta::from_config_file(&m.path.join("config.json")) {
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
    }

    // Op kind histogram + a sample of scalar arguments per op kind. The
    // first non-empty `arguments` payload we see for each primitive name
    // doubles as a hint of what `Attrs` should one day carry.
    let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
    let mut sample_args: BTreeMap<&str, &Vec<serde_json::Value>> = BTreeMap::new();
    for record in &raw.records {
        if record.kind != "primitive" {
            continue;
        }
        let Some(name) = record.name.as_deref() else {
            continue;
        };
        *hist.entry(name).or_insert(0) += 1;
        if !record.arguments.is_empty() {
            sample_args.entry(name).or_insert(&record.arguments);
        }
    }

    println!();
    println!("{}", "op histogram:".bold());
    let mut sorted: Vec<_> = hist.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, n) in &sorted {
        let args = sample_args
            .get(name)
            .map(|v| {
                let s = serde_json::to_string(v).unwrap_or_default();
                if s.len() > 60 {
                    format!("{}…", &s[..60])
                } else {
                    s
                }
            })
            .unwrap_or_default();
        println!(
            "  {:>5}  {:<32} {}",
            n.to_string().cyan(),
            name,
            args.dimmed()
        );
    }

    println!();
    println!("{}", "root tensor:".bold());
    let root = graph.tensor(graph.root);
    println!(
        "  t{}  {:<6}  {:?}  (produced by op #{})",
        root.id,
        root.dtype.short(),
        root.shape,
        root.producer
            .map(|o| o.to_string())
            .unwrap_or_else(|| "—".into()),
    );

    // Attrs coverage: how many ops have a typed Attrs vs how many fell
    // back to Unknown. This is the headline signal for the loader phase.
    let typed = graph
        .ops
        .iter()
        .filter(|o| !matches!(o.attrs, crate::graph::Attrs::Unknown))
        .count();
    println!();
    println!(
        "{} {} / {} ops have typed Attrs",
        "attrs:".bold(),
        typed.to_string().cyan(),
        graph.ops.len()
    );

    // Layer-tag coverage. For layer-granular dispatch every op needs a
    // tag — the per-bucket count is the size of each layer's ICB.
    let mut tag_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut untagged = 0usize;
    for op in &graph.ops {
        match op.layer_id {
            Some(crate::graph::LayerTag::Embed) => {
                *tag_counts.entry("embed".to_string()).or_insert(0) += 1
            }
            Some(crate::graph::LayerTag::Layer(n)) => {
                *tag_counts.entry(format!("layer.{:>2}", n)).or_insert(0) += 1
            }
            Some(crate::graph::LayerTag::Post) => {
                *tag_counts.entry("post".to_string()).or_insert(0) += 1
            }
            None => untagged += 1,
        }
    }
    println!();
    println!("{}", "layer tags:".bold());
    for (tag, n) in &tag_counts {
        println!("  {:>5}  {}", n.to_string().cyan(), tag);
    }
    if untagged > 0 {
        println!("  {:>5}  {}", untagged.to_string().yellow(), "untagged");
    }

    println!();
    println!("{}", "last 5 ops:".bold());
    let tail = graph.ops.len().saturating_sub(5);
    for op in &graph.ops[tail..] {
        let outs: Vec<String> = op.outputs.iter().map(|t| format!("t{t}")).collect();
        let (kind, payload) = match &op.attrs {
            crate::graph::Attrs::Unknown => ("·", String::new()),
            other => ("✓", format!("{:?}", other)),
        };
        println!(
            "  #{:<5} {} {:<14} -> [{}]  {}",
            op.id,
            kind,
            op.name,
            outs.join(", "),
            payload.dimmed()
        );
    }

    // Dump a self-contained HTML viewer (no graphviz install required —
    // viz.js renders the DOT in-browser). Also write the raw .dot
    // alongside it for users who prefer offline rendering.
    let base = m.name.replace('/', "__");
    let cwd = std::env::current_dir()?;
    let dot_path = cwd.join(format!("{}.dot", base));
    let html_path = cwd.join(format!("{}.html", base));
    let dot = graph.to_dot();
    std::fs::write(&dot_path, &dot).with_context(|| format!("writing {}", dot_path.display()))?;
    std::fs::write(&html_path, render_html(&m.name, &dot))
        .with_context(|| format!("writing {}", html_path.display()))?;
    println!();
    ui::ok(format!("wrote {}", html_path.display()));
    ui::dim_kv("open", format!("open {}", html_path.display()));
    ui::dim_kv("dot", dot_path.display());

    Ok(())
}

fn render_html(title: &str, dot: &str) -> String {
    // Single file: viz.js (graphviz WASM) + svg-pan-zoom from a CDN.
    // The DOT string is embedded as a JS template literal — we escape
    // backticks and backslashes so arbitrary op names cannot break out.
    let escaped = dot
        .replace('\\', "\\\\")
        .replace('`', "\\`")
        .replace("${", "\\${");
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fusekit graph — {title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, sans-serif; }}
  #status {{ position: fixed; top: 8px; left: 12px; padding: 4px 10px; background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.15); font-size: 12px; z-index: 10; }}
  #graph {{ width: 100vw; height: 100vh; }}
  #graph svg {{ width: 100%; height: 100%; }}
</style>
</head>
<body>
<div id="status">rendering…</div>
<div id="graph"></div>
<script src="https://cdn.jsdelivr.net/npm/@viz-js/viz@3.7.0/lib/viz-standalone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.2/dist/svg-pan-zoom.min.js"></script>
<script>
const DOT = `{escaped}`;
const status = document.getElementById('status');
const container = document.getElementById('graph');
const t0 = performance.now();
Viz.instance().then(viz => {{
  const svg = viz.renderSVGElement(DOT);
  container.appendChild(svg);
  const ms = (performance.now() - t0).toFixed(0);
  status.textContent = `rendered in ${{ms}} ms — scroll to zoom, drag to pan`;
  svgPanZoom(svg, {{ controlIconsEnabled: true, minZoom: 0.1, maxZoom: 50, zoomScaleSensitivity: 0.4 }});
}}).catch(err => {{
  status.textContent = 'render failed: ' + err.message;
  status.style.background = '#fdd';
}});
</script>
</body>
</html>
"#
    )
}
