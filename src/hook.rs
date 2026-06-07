//! Auto-load hook for the bundled oMLX Python.
//!
//! ```text
//!   user site-packages          model directory
//!   ──────────────────          ───────────────
//!   fusekit.pth (one file)      <model>/fusekit/
//!       │                          manifest.json
//!       │  glob ~/.omlx/models/**  kernel.metallib
//!       ▼                          task_graph.json
//!       discovers runtime/         runtime/
//!       inserts into sys.path        fusekit_runtime/
//!       imports fusekit_runtime        __init__.py
//! ```
//!
//! The `.pth` file is the only thing fusekit adds outside the model
//! directory. The actual Python package lives per-model (option C),
//! so `rm -rf <model>/fusekit` removes everything fusekit-related
//! for that model.
//!
//! When multiple models are patched all their `runtime/` copies are
//! identical (they came from the same fusekit binary's payload). The
//! `.pth` picks the freshest by `manifest.json` mtime — that's the
//! one most likely to match the running binary.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

const PTH_FILENAME: &str = "fusekit.pth";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookStatus {
    Installed,
    NotInstalled,
}

/// User site-packages, resolved by the bundled Python directly. We
/// don't hard-code the minor version because it changes over time.
pub fn user_site_packages() -> Result<PathBuf> {
    let python = crate::paths::mlx_python_bin()?;
    let out = std::process::Command::new(&python)
        .args([
            "-c",
            "import site, sys; sys.stdout.write(site.getusersitepackages())",
        ])
        .output()
        .context("invoke bundled python to resolve user site-packages")?;
    if !out.status.success() {
        return Err(anyhow!("python failed to report user site path"));
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

pub fn status() -> Result<(PathBuf, HookStatus)> {
    let usp = user_site_packages()?;
    let pth = usp.join(PTH_FILENAME);
    let st = if pth.is_file() {
        HookStatus::Installed
    } else {
        HookStatus::NotInstalled
    };
    Ok((pth, st))
}

/// Install the `.pth`. `models_glob` is the glob pattern the .pth
/// file scans at every interpreter startup — typically
/// `~/.omlx/models/**/fusekit/runtime`.
pub fn install(models_glob: &Path) -> Result<()> {
    let usp = user_site_packages()?;
    std::fs::create_dir_all(&usp).with_context(|| format!("create {}", usp.display()))?;

    let escaped = models_glob.display().to_string().replace('\\', "\\\\");
    // One-line .pth (must start with `import`). Resolves the freshest
    // runtime by mtime so the most recently patched model's copy —
    // assumed to match the running fusekit binary — wins.
    let pth_line = format!(
        "import sys, os, glob; \
         _cs = sorted(glob.glob(r\"{escaped}\", recursive=True), key=os.path.getmtime, reverse=True); \
         _cs and (sys.path.insert(0, _cs[0]) or __import__(\"fusekit_runtime\"))\n"
    );

    let pth = usp.join(PTH_FILENAME);
    std::fs::write(&pth, pth_line).with_context(|| format!("write {}", pth.display()))?;
    Ok(())
}

/// Remove the `.pth`. Idempotent.
pub fn uninstall() -> Result<()> {
    let usp = user_site_packages()?;
    let pth = usp.join(PTH_FILENAME);
    if pth.exists() {
        std::fs::remove_file(&pth).with_context(|| format!("remove {}", pth.display()))?;
    }
    // Sweep up stragglers from older layouts. Safe — these names are
    // ours.
    let stale_pkg = usp.join("fusekit_runtime");
    if stale_pkg.is_dir() {
        let _ = std::fs::remove_dir_all(&stale_pkg);
    }
    Ok(())
}
