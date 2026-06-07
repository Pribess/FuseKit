//! Per-model `fusekit/manifest.json` — versioning + provenance.
//!
//! Why this exists:
//!  1. Cheap "is this model patched with the version I'm running?" check.
//!  2. Drives the upgrade path in `fusekit patch` — old manifest →
//!     wipe + repatch, matching manifest → no-op (or `--force`).
//!  3. Makes diffing two patched models obvious during debugging.
//!
//! Schema is versioned (`schema_version`) so we can change fields
//! later without breaking older patches: bump it and add a migration
//! arm in `read`.

use crate::fusion::ir::MtpInfo;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Current on-disk schema. Bump on breaking changes.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Fusekit binary version. Tied to `Cargo.toml` so a `cargo bump`
/// propagates automatically.
pub fn fusekit_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema of this JSON document.
    pub schema_version: u32,
    /// fusekit binary version that produced these artifacts.
    pub fusekit_version: String,
    /// RFC 3339 UTC timestamp of when this patch was produced.
    pub compiled_at: String,
    /// Helpful free-form info for debugging — populated by patch.rs.
    pub model: ModelInfo,
    /// Headline numbers from the compile pipeline.
    pub stats: CompileStats,
    /// Additional compiled dispatch variants, such as MTP multi-row
    /// kernels alongside the default single-token decode path.
    #[serde(default)]
    pub variants: Vec<CompileVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    #[serde(default)]
    pub mtp: MtpInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileStats {
    pub ops: usize,
    pub fusions: usize,
    pub tasks: usize,
    pub unique_kernels: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileVariant {
    pub name: String,
    pub path: String,
    pub active_m: u32,
    pub dense_matmul: String,
    pub tasks: usize,
    pub unique_kernels: usize,
}

impl Manifest {
    pub fn write(&self, fusekit_dir: &Path) -> Result<()> {
        let path = fusekit_dir.join("manifest.json");
        let bytes = serde_json::to_vec_pretty(self).context("encode manifest")?;
        std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn read(fusekit_dir: &Path) -> Result<Self> {
        let path = fusekit_dir.join("manifest.json");
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let m: Manifest = serde_json::from_slice(&bytes).context("parse manifest")?;
        Ok(m)
    }

    /// Read if present; returns Ok(None) when the file does not exist.
    pub fn read_if_exists(fusekit_dir: &Path) -> Result<Option<Self>> {
        let path = fusekit_dir.join("manifest.json");
        if !path.is_file() {
            return Ok(None);
        }
        Ok(Some(Self::read(fusekit_dir)?))
    }

    /// True when this manifest was produced by the running fusekit
    /// binary. The schema version check protects us from interpreting
    /// a future schema we don't understand.
    pub fn matches_running_binary(&self) -> bool {
        self.schema_version == CURRENT_SCHEMA_VERSION && self.fusekit_version == fusekit_version()
    }
}

/// Produce an ISO-8601 UTC timestamp without pulling in chrono.
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Rough but adequate; precise to the second is enough for "when
    // did I last patch this model?".
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn epoch_to_ymdhms(mut t: u64) -> (i32, u32, u32, u32, u32, u32) {
    let s = (t % 60) as u32;
    t /= 60;
    let mi = (t % 60) as u32;
    t /= 60;
    let h = (t % 24) as u32;
    t /= 24;
    let mut days = t as i64;
    let mut year: i32 = 1970;
    loop {
        let yd = if is_leap(year) { 366 } else { 365 };
        if days < yd {
            break;
        }
        days -= yd;
        year += 1;
    }
    let months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0u32;
    while mo < 12 {
        let dim = if mo == 1 && is_leap(year) {
            29
        } else {
            months[mo as usize]
        };
        if days < dim {
            break;
        }
        days -= dim;
        mo += 1;
    }
    (year, mo + 1, (days + 1) as u32, h, mi, s)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
