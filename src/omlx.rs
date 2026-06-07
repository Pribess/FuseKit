//! oMLX model directory listing.
//!
//! Resolution of the models directory itself is delegated to
//! [`crate::paths::omlx_models_dir`]. This module just walks that
//! directory and presents the result.
//!
//! A model is any subdirectory (one or two levels deep, to support the
//! `namespace/model-name/` layout used by mlx-community) that contains a
//! `config.json`.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use crate::paths;

/// One discovered model entry.
pub struct Model {
    /// Display name (relative path from the omlx root, e.g. "Qwen3-Coder-8bit"
    /// or "mlx-community/SmolLM2-1.7B-4bit").
    pub name: String,
    /// Absolute path to the model directory.
    pub path: PathBuf,
}

impl Model {
    /// Whether a `fusekit/` subdirectory exists (i.e. patched).
    pub fn is_patched(&self) -> bool {
        self.path.join("fusekit").is_dir()
    }

    /// Sum of safetensors file sizes (excluding `.cache`).
    pub fn weight_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(&self.path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                if name.ends_with(".safetensors") || name.ends_with(".npz") {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        total += meta.len();
                    }
                }
            }
        }
        total
    }

    /// `model_type` field from `config.json`, if available.
    pub fn model_type(&self) -> Option<String> {
        let path = self.path.join("config.json");
        let bytes = std::fs::read(&path).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        v.get("model_type")?.as_str().map(|s| s.to_string())
    }

    /// Native-MTP metadata resolved from config plus local weight files.
    pub fn mtp_info(&self) -> Option<crate::fusion::ir::MtpInfo> {
        crate::fusion::ir::ModelMeta::from_config_file(&self.path.join("config.json"))
            .ok()
            .map(|meta| meta.mtp)
    }
}

/// Format bytes as a human-readable string.
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "—".to_string();
    }
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit + 1 < UNITS.len() {
        size /= 1024.0;
        unit += 1;
    }
    if size >= 100.0 {
        format!("{:.0} {}", size, UNITS[unit])
    } else if size >= 10.0 {
        format!("{:.1} {}", size, UNITS[unit])
    } else {
        format!("{:.2} {}", size, UNITS[unit])
    }
}

/// Resolve the active oMLX models directory (delegates to [`paths`]).
pub fn models_dir() -> Result<PathBuf> {
    paths::omlx_models_dir()
}

/// List all discoverable models under the active models directory.
pub fn list_models() -> Result<Vec<Model>> {
    let root = models_dir()?;
    let mut out = Vec::new();
    walk(&root, &root, 0, 2, &mut out)?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Look up a single model by name (matches against `name` from `list_models`).
pub fn find_model(name: &str) -> Result<Model> {
    list_models()?
        .into_iter()
        .find(|m| m.name == name)
        .ok_or_else(|| anyhow!("model not found: {name}"))
}

fn walk(
    root: &Path,
    cur: &Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<Model>,
) -> Result<()> {
    for entry in std::fs::read_dir(cur)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.starts_with('.'))
        {
            continue;
        }

        if path.join("config.json").is_file() {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            out.push(Model {
                name: rel.to_string_lossy().into_owned(),
                path,
            });
            continue;
        }

        if depth < max_depth {
            walk(root, &path, depth + 1, max_depth, out)?;
        }
    }
    Ok(())
}
