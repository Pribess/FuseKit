//! Embedded `fusekit_runtime` Python package, materialized **into the
//! model's `fusekit/` directory** so each patched model is fully
//! self-contained.
//!
//! ```text
//!   compile-time payload          install-time layout
//!   ────────────────────          ───────────────────
//!   python/fusekit_runtime/       <model>/fusekit/runtime/
//!     __init__.py                   fusekit_runtime/
//!         │                            __init__.py
//!         │  include_str!              │
//!         ▼                            ▲
//!     baked into binary  ─────► extracted per patch
//! ```
//!
//! No global cache directory under the user's home. Trade-off: every
//! patched model carries an identical copy of the runtime (~few KB).
//! Worth it for the cleaner unpatch story — `rm -rf <model>/fusekit`
//! removes everything fusekit-related for that model.

use anyhow::{Context, Result};
use std::path::Path;

/// Text files in `python/fusekit_runtime/`.
const TEXT_FILES: &[(&str, &str)] = &[
    (
        "__init__.py",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/python/fusekit_runtime/__init__.py"
        )),
    ),
    (
        "dispatch.py",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/python/fusekit_runtime/dispatch.py"
        )),
    ),
];

/// Compiled Rust extension module (cdylib built from
/// `python_runtime/`). `build.rs` produces this and emits the env
/// var consumed by the `include_bytes!` below.
const NATIVE_SO: &[u8] = include_bytes!(env!("FUSEKIT_RUNTIME_NATIVE_SO"));

/// Materialize the runtime package into `<fusekit_dir>/runtime/
/// fusekit_runtime/`. Idempotent; overwrites any existing files so
/// running `fusekit patch` with a newer binary brings the runtime
/// forward.
pub fn materialize_into(fusekit_dir: &Path) -> Result<()> {
    let pkg = fusekit_dir.join("runtime").join("fusekit_runtime");
    std::fs::create_dir_all(&pkg).with_context(|| format!("create {}", pkg.display()))?;

    for (relpath, contents) in TEXT_FILES {
        let dst = pkg.join(relpath);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        std::fs::write(&dst, contents.as_bytes())
            .with_context(|| format!("write {}", dst.display()))?;
    }

    // The Python extension is loaded by CPython via dlopen, so it
    // needs to sit alongside __init__.py with the standard "_" prefix.
    let so_dst = pkg.join("_fusekit_runtime_native.so");
    std::fs::write(&so_dst, NATIVE_SO).with_context(|| format!("write {}", so_dst.display()))?;

    Ok(())
}
