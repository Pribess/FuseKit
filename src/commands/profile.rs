//! `fusekit profile` — record a unified CPU+GPU trace via xctrace while
//! a baseline MLX-LM inference run executes against an oMLX model.
//!
//! Synchronization is handled by `xctrace record --launch -- <binary>`:
//! the recorder is set up before the child is spawned, so there is no
//! race window for early events.
//!
//! Today the inner run does a pure MLX baseline (no fusekit kernels) so
//! we can measure what we're competing against. Once the megakernel
//! pipeline lands, a `--fusekit` mode will dispatch through fusekit's
//! compiled metallib instead.

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use std::process::Command;

use crate::{omlx, paths, ui};

/// Outer entry: spawn xctrace, which re-execs this binary in
/// `profile-inner` mode under recording.
pub fn run(
    model: &str,
    output: Option<PathBuf>,
    open: bool,
    prompt: &str,
    max_tokens: usize,
) -> Result<()> {
    // Resolve the model up front so the user gets a clear error before
    // we hand off to xctrace (which would otherwise fail mid-trace).
    let m = omlx::find_model(model)?;
    ui::ok(format!("model: {} ({})", m.name, m.path.display()));

    let exe = std::env::current_exe().context("locating current exe")?;
    let output = output.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("fusekit-{}.trace", timestamp_secs()))
    });

    ui::step(format!("xctrace record → {}", output.display()));

    // `--target-stdout -` forwards the child's stdout to our terminal
    // so the user sees generated tokens stream as they're produced.
    // (xctrace does not expose a `--target-stderr` flag; stderr from
    // the child still surfaces normally because xctrace doesn't grab it.)
    let status = Command::new("xctrace")
        .args(["record", "--template", "Metal System Trace", "--output"])
        .arg(&output)
        .args(["--target-stdout", "-"])
        .args(["--launch", "--"])
        .arg(&exe)
        .args(["profile-inner", &m.name])
        .args(["--prompt", prompt])
        .args(["--max-tokens", &max_tokens.to_string()])
        .status()
        .context("invoking xctrace; is it on $PATH (try `xcode-select -p`)?")?;

    if !status.success() {
        return Err(anyhow!("xctrace exited with {status}"));
    }

    ui::ok(format!("trace saved: {}", output.display()));
    ui::dim_kv("open", format!("open {}", output.display()));

    if open {
        Command::new("open").arg(&output).status().ok();
    }
    Ok(())
}

/// Inner entry: actually exercise the workload while xctrace records us.
/// Today: baseline MLX-LM inference via the bundled oMLX Python.
pub fn run_inner(model: &str, prompt: &str, max_tokens: usize) -> Result<()> {
    let _all = crate::profile::interval("baseline", "inference");

    let m = omlx::find_model(model)?;
    let python = paths::mlx_python_bin()?;
    let site_packages = paths::mlx_site_packages()?;
    let script = locate_baseline_script()?;

    ui::step(format!(
        "baseline inference: {} tokens on {}",
        max_tokens, m.name
    ));

    let status = Command::new(&python)
        .arg(&script)
        .arg(&m.path)
        .arg(prompt)
        .arg(max_tokens.to_string())
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONPATH", &site_packages)
        .status()
        .context("running baseline_infer.py")?;

    if !status.success() {
        return Err(anyhow!("baseline_infer.py exited with {status}"));
    }
    Ok(())
}

fn locate_baseline_script() -> Result<PathBuf> {
    // Look next to the executable first (release install), then fall
    // back to the source-tree layout used during development. Matches
    // `loader::locate_driver_script`'s contract.
    let exe = std::env::current_exe()?;
    let candidates = [
        exe.parent().map(|p| p.join("scripts/baseline_infer.py")),
        exe.parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("scripts/baseline_infer.py")),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/baseline_infer.py")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(anyhow!("baseline_infer.py not found"))
}

fn timestamp_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
