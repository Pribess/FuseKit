//! Central path discovery for fusekit.
//!
//! All knowledge of where oMLX is installed lives here. Every other module
//! asks this one. Resolution order for each path is:
//!
//!   1. Explicit env var (FUSEKIT_*)
//!   2. Auto-detection in a small list of well-known locations
//!   3. Error with a clear message
//!
//! No string literals like "/Applications/oMLX.app/..." should appear
//! outside this file.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

// --- oMLX runtime data (model registry, logs, cache) ----------------------

/// Returns the oMLX data root (default `~/.omlx`).
pub fn omlx_data_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FUSEKIT_OMLX_DATA") {
        return ok_if_dir(PathBuf::from(p));
    }
    let home = home_dir()?;
    ok_if_dir(home.join(".omlx"))
}

/// Returns the directory that holds the user's model subdirectories.
pub fn omlx_models_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FUSEKIT_OMLX_DIR") {
        return ok_if_dir(PathBuf::from(p));
    }
    // Prefer omlx_data_root()/models if oMLX has been initialised at least
    // once; otherwise fall back to a plain ~/models for users who keep
    // their models elsewhere.
    let home = home_dir()?;
    for candidate in [home.join(".omlx").join("models"), home.join("models")] {
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Err(anyhow!(
        "no oMLX models directory found (set FUSEKIT_OMLX_DIR or use ~/.omlx/models)"
    ))
}

// --- oMLX.app bundle ------------------------------------------------------

/// Returns the root of the oMLX application bundle (the `.app` directory).
pub fn mlx_app_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FUSEKIT_MLX_APP") {
        return ok_if_dir(PathBuf::from(p));
    }
    let home = home_dir()?;
    let candidates = [
        PathBuf::from("/Applications/oMLX.app"),
        home.join("Applications/oMLX.app"),
    ];
    for c in candidates {
        if c.is_dir() {
            return Ok(c);
        }
    }
    Err(anyhow!(
        "could not locate oMLX.app (set FUSEKIT_MLX_APP=<path-to-oMLX.app>)"
    ))
}

/// Path to the Python 3 binary bundled inside oMLX.
pub fn mlx_python_bin() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FUSEKIT_PYTHON") {
        return ok_if_file(PathBuf::from(p));
    }
    let app = mlx_app_root()?;
    let candidates = [
        app.join("Contents/Python/cpython-3.11/bin/python3.11"),
        app.join("Contents/MacOS/python3"),
    ];
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(anyhow!(
        "could not find a Python binary inside oMLX.app (set FUSEKIT_PYTHON)"
    ))
}

/// Returns the `lib/python3.X/site-packages` directory used by oMLX.
/// This is where `mlx`, `mlx_lm`, etc. live.
pub fn mlx_site_packages() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FUSEKIT_MLX_SITE_PACKAGES") {
        return ok_if_dir(PathBuf::from(p));
    }
    let app = mlx_app_root()?;
    let lib_root = app.join("Contents/Python/framework-mlx-framework/lib");
    if let Ok(entries) = std::fs::read_dir(&lib_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if name.starts_with("python") {
                let sp = path.join("site-packages");
                if sp.is_dir() {
                    return Ok(sp);
                }
            }
        }
    }
    Err(anyhow!(
        "could not find mlx site-packages under {}",
        lib_root.display()
    ))
}

/// Returns `mlx_lm/utils.py` inside the oMLX-bundled site-packages.
/// Kept for status reporting and ad-hoc debugging — the production
/// hook path no longer writes into this file (the bundle is signed
/// read-only); see `hook.rs` for the user-site-packages flow.
#[allow(dead_code)]
pub fn mlx_lm_utils_py() -> Result<PathBuf> {
    let sp = mlx_site_packages()?;
    let candidate = sp.join("mlx_lm").join("utils.py");
    if candidate.is_file() {
        return Ok(candidate);
    }
    Err(anyhow!(
        "mlx_lm utils.py not found at {}",
        candidate.display()
    ))
}

// --- Internal helpers -----------------------------------------------------

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().context("could not resolve home directory")
}

fn ok_if_dir(p: PathBuf) -> Result<PathBuf> {
    if p.is_dir() {
        Ok(p)
    } else {
        Err(anyhow!("not a directory: {}", p.display()))
    }
}

fn ok_if_file(p: PathBuf) -> Result<PathBuf> {
    if p.is_file() {
        Ok(p)
    } else {
        Err(anyhow!("not a file: {}", p.display()))
    }
}

/// Joins a known-safe relative subpath onto the data root. Useful when a
/// caller wants e.g. `~/.omlx/logs/server.log` without retyping `~/.omlx`.
#[allow(dead_code)]
pub fn omlx_data_subpath(rel: &str) -> Result<PathBuf> {
    let root = omlx_data_root()?;
    Ok(root.join(rel.trim_start_matches(std::path::MAIN_SEPARATOR)))
}

/// Whether `path` lives under `root`. Symlink-aware.
#[allow(dead_code)]
pub fn is_within(path: &Path, root: &Path) -> bool {
    match (path.canonicalize(), root.canonicalize()) {
        (Ok(p), Ok(r)) => p.starts_with(r),
        _ => path.starts_with(root),
    }
}
