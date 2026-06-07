#![allow(dead_code)]

//! Shell out to Apple's metal toolchain to turn generated MSL into a
//! linkable `.metallib`.
//!
//! ```text
//!   kernel.metal ──(xcrun metal -c)──► kernel.air
//!   kernel.air   ──(xcrun metallib)──► kernel.metallib
//! ```
//!
//! Both tools live behind `xcrun --sdk macosx --find <name>` so we
//! pick up whatever Xcode Command Line Tools the user has installed.
//! Build script `build.rs` already locates them at compile-time for
//! fusekit's own dev shaders; this module does the same dance at
//! runtime so `fusekit patch` can compile a per-model metallib.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Compile `msl_path` into a `.metallib` at `out_path`. Returns the
/// path on success.
pub fn compile_to_metallib(msl_path: &Path, out_path: &Path) -> Result<PathBuf> {
    let metal_bin = find_tool("metal")?;
    let metallib_bin = find_tool("metallib").ok();

    let air_path = out_path.with_extension("air");
    let mut cmd = Command::new(&metal_bin);
    cmd.args(["-c", "-O3", "-ffast-math", "-Wno-unused-variable"]);
    if let Some(include_dir) = mlx_include_dir() {
        cmd.arg("-I").arg(include_dir);
    }
    let status = cmd
        .arg(msl_path)
        .arg("-o")
        .arg(&air_path)
        .status()
        .with_context(|| format!("invoking {}", metal_bin.display()))?;
    if !status.success() {
        return Err(anyhow!("metal compile failed for {}", msl_path.display()));
    }

    let link = metallib_bin.as_deref().unwrap_or(metal_bin.as_path());
    let status = Command::new(link)
        .arg(&air_path)
        .arg("-o")
        .arg(out_path)
        .status()
        .with_context(|| format!("invoking {}", link.display()))?;
    if !status.success() {
        return Err(anyhow!("metallib link failed"));
    }

    Ok(out_path.to_path_buf())
}

fn mlx_include_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FUSEKIT_MLX_INCLUDE_DIR") {
        let path = PathBuf::from(path);
        if path.is_dir() {
            return Some(path);
        }
    }
    let omlx = PathBuf::from(
        "/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages/mlx/include",
    );
    if omlx.is_dir() {
        return Some(omlx);
    }
    None
}

fn find_tool(name: &str) -> Result<PathBuf> {
    let out = Command::new("xcrun")
        .args(["--sdk", "macosx", "--find", name])
        .output()
        .with_context(|| format!("xcrun --find {name}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "{name} not found (install Xcode Command Line Tools)"
        ));
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}
