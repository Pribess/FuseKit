//! Apple Silicon target detection + per-chip budgets.
//!
//! Codegen heuristics (TG size, register-pressure spill threshold,
//! SMEM allocation) need numeric budgets that vary by chip family.
//! This module owns:
//!   - host detection via `sysctl machdep.cpu.brand_string`,
//!   - a `ChipModel` enum covering shipping Apple Silicon families,
//!   - a `Budgets` struct that the codegen consults.
//!
//! Default: `Target::detect_host()` picks the chip the compile is
//! running on. CLI flag `--chip <name>` overrides for cross-machine
//! builds (e.g. compile a metallib on an M5 dev box that will run on
//! an M1 deployment target).

use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChipModel {
    /// M1 / M1 Pro / M1 Max / M1 Ultra (all share the same SIMD
    /// generation + register file for our purposes).
    M1,
    /// M2 family.
    M2,
    /// M3 family — bigger register file, supports dynamic caching.
    M3,
    /// M4 family.
    M4,
    /// M5 family.
    M5,
    /// Unknown / forward-compatible. Uses conservative budgets.
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseMatmulMode {
    /// Single-token decode path. Treat the active sequence length as 1
    /// and use simdgroup-cooperative GEMV kernels.
    DecodeGemv,
    /// Force the tiled 8x8 simdgroup-matrix path. Intended for MTP /
    /// prefill experiments where the runtime really supplies M>=8
    /// active rows.
    SimdgroupGemm,
    /// Pick GEMM only when the codegen/runtime path reports enough
    /// active rows to fill the 8x8 tile.
    Auto,
}

impl DenseMatmulMode {
    pub fn as_str(self) -> &'static str {
        match self {
            DenseMatmulMode::DecodeGemv => "decode-gemv",
            DenseMatmulMode::SimdgroupGemm => "simdgroup-gemm",
            DenseMatmulMode::Auto => "auto",
        }
    }
}

fn clamp_active_m(requested: u32, trace_m: u32) -> u32 {
    requested.max(1).min(trace_m.max(1))
}

fn env_active_m(trace_m: u32) -> Option<u32> {
    std::env::var("FUSEKIT_ACTIVE_M")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .map(|n| clamp_active_m(n, trace_m))
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

impl ChipModel {
    /// Per-chip codegen budgets. Values are conservative — they err
    /// on the side of "fewer registers, smaller TG" so codegen
    /// decisions don't trigger compiler spills.
    pub fn budgets(self) -> Budgets {
        match self {
            // M1/M2: smaller register file relative to M3+. Tuned for
            // the GPU we see in those machines.
            ChipModel::M1 | ChipModel::M2 => Budgets {
                regs_per_thread: 128,
                smem_per_threadgroup: 32 * 1024,
                simd_width: 32,
                preferred_simdgroups_per_tg: 8,
                max_simdgroups_per_tg: 8,
            },
            // M3 introduced dynamic caching + larger register pool.
            ChipModel::M3 | ChipModel::M4 => Budgets {
                regs_per_thread: 192,
                smem_per_threadgroup: 32 * 1024,
                simd_width: 32,
                preferred_simdgroups_per_tg: 8,
                max_simdgroups_per_tg: 8,
            },
            // M5: same SMEM, but expanded register pool again.
            // (Tried 16 SGs/TG — broke numerical correctness in some
            // pattern kernels whose 32-slot `sum_sq_partial` reduction
            // assumes ≤8 simdgroups. Pinned back to 8; future work
            // could rewrite those reductions to handle larger TGs.)
            ChipModel::M5 => Budgets {
                regs_per_thread: 256,
                smem_per_threadgroup: 32 * 1024,
                simd_width: 32,
                preferred_simdgroups_per_tg: 8,
                max_simdgroups_per_tg: 8,
            },
            // Be conservative on anything we don't recognise so emit
            // stays safe across future hardware until we measure it.
            ChipModel::Unknown => Budgets {
                regs_per_thread: 128,
                smem_per_threadgroup: 32 * 1024,
                simd_width: 32,
                preferred_simdgroups_per_tg: 8,
                max_simdgroups_per_tg: 8,
            },
        }
    }

    /// True for chips with `simdgroup_matrix` intrinsics (8×8 bf16/fp16
    /// MMA in one instruction). M3 onwards. Used by chain emit to
    /// pick the high-throughput dense GEMV/GEMM path on capable chips.
    pub fn supports_simdgroup_matrix(self) -> bool {
        matches!(self, ChipModel::M3 | ChipModel::M4 | ChipModel::M5)
    }

    /// Dense matmul lowering mode. Keep decode on the measured GEMV
    /// path by default; future MTP/prefill work can opt into GEMM
    /// without deleting or weakening the decode-specialized path.
    ///
    /// Accepted values:
    /// - `decode-gemv` / `gemv` / `decode`: force single-token GEMV.
    /// - `simdgroup-gemm` / `gemm` / `simdgroup`: force tiled GEMM.
    /// - `auto`: use GEMM when active rows fill an 8x8 tile.
    ///
    /// `FUSEKIT_SIMDGROUP_MATRIX=1` remains as a compatibility alias
    /// for the forced GEMM experiment knob.
    pub fn dense_matmul_mode(self) -> DenseMatmulMode {
        if let Ok(raw) = std::env::var("FUSEKIT_DENSE_MATMUL_MODE") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "decode-gemv" | "gemv" | "decode" => return DenseMatmulMode::DecodeGemv,
                "simdgroup-gemm" | "gemm" | "simdgroup" => {
                    return DenseMatmulMode::SimdgroupGemm;
                }
                "auto" => return DenseMatmulMode::Auto,
                _ => {}
            }
        }
        if std::env::var_os("FUSEKIT_SIMDGROUP_MATRIX").is_some() {
            return DenseMatmulMode::SimdgroupGemm;
        }
        DenseMatmulMode::DecodeGemv
    }

    pub fn use_simdgroup_gemm(self, active_rows: u32) -> bool {
        if !self.supports_simdgroup_matrix() {
            return false;
        }
        match self.dense_matmul_mode() {
            DenseMatmulMode::DecodeGemv => false,
            DenseMatmulMode::SimdgroupGemm => true,
            DenseMatmulMode::Auto => active_rows >= 8,
        }
    }

    /// Active token rows for codegen. Most traces are captured with
    /// shape `[B, S, ...]`; decode consumes one row, while prefill/MTP
    /// experiments may consume several. `FUSEKIT_ACTIVE_M=N` overrides
    /// this globally at patch time, clamped to the captured trace S.
    pub fn active_m(self, trace_m: u32) -> u32 {
        if let Some(n) = env_active_m(trace_m) {
            return n;
        }
        match self.dense_matmul_mode() {
            DenseMatmulMode::DecodeGemv => 1,
            DenseMatmulMode::SimdgroupGemm | DenseMatmulMode::Auto => trace_m.max(1),
        }
    }

    /// Active token rows for semantic kernels whose trace inputs/outputs
    /// are replaced by runtime bindings and therefore have no local shape.
    /// In prefill/MTP variant codegen the active M is the variant contract,
    /// not an arbitrary fallback trace length, so do not clamp the env value
    /// to `fallback_trace_m`.
    pub fn active_m_for_unshaped(self, fallback_trace_m: u32) -> u32 {
        if let Some(n) = std::env::var("FUSEKIT_ACTIVE_M")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1)
        {
            return n;
        }
        self.active_m(fallback_trace_m)
    }

    /// For tensor shapes captured as `[B, S, suffix...]`, return the
    /// active S rows. Rank-2 matmul operands use their explicit M
    /// dimension instead of a batch/sequence prefix.
    pub fn active_tokens_for_shape(self, shape: &[i32]) -> u32 {
        let trace_m = if shape.len() >= 3 {
            shape[1].max(1) as u32
        } else if shape.len() >= 2 {
            shape[shape.len() - 2].max(1) as u32
        } else {
            1
        };
        self.active_m(trace_m)
    }

    /// Number of logical rows before the last dimension after applying
    /// the active-M policy. Preserves post-sequence axes such as heads
    /// or MoE routes: `[B,S,H,D] -> active_S*H`.
    pub fn active_rows_before_last(self, shape: &[i32]) -> u64 {
        if shape.is_empty() {
            return 1;
        }
        if shape.len() >= 3 {
            let active = self.active_tokens_for_shape(shape) as u64;
            let suffix = shape[2..shape.len() - 1]
                .iter()
                .map(|&d| d.max(1) as u64)
                .product::<u64>()
                .max(1);
            active.saturating_mul(suffix).max(1)
        } else {
            shape[..shape.len().saturating_sub(1)]
                .iter()
                .map(|&d| d.max(1) as u64)
                .product::<u64>()
                .max(1)
        }
    }

    /// Active element count for pointwise/view kernels under the same
    /// active-M policy.
    pub fn active_elems(self, shape: &[i32]) -> u64 {
        if shape.is_empty() {
            return 1;
        }
        let last = shape.last().copied().unwrap_or(1).max(1) as u64;
        self.active_rows_before_last(shape)
            .saturating_mul(last)
            .max(1)
    }

    /// Output rows each simdgroup computes per pass in a multi-row
    /// coalesced GEMV. Higher = better activation-load reuse but more
    /// per-thread register pressure. Apple's over-spawn scheduler
    /// likes lots of small TGs. Measured decode throughput on M5 wins
    /// at 1 row despite the larger register file: M3/M4 → 2 rows,
    /// M1/M2/M5 → 1 row. Override via `FUSEKIT_GEMV_ROWS=N`.
    pub fn gemv_rows_per_simdgroup(self) -> u32 {
        let env = std::env::var("FUSEKIT_GEMV_ROWS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1 && n <= 8);
        if let Some(n) = env {
            return n;
        }
        match self {
            ChipModel::M3 | ChipModel::M4 => 2,
            ChipModel::M1 | ChipModel::M2 | ChipModel::M5 | ChipModel::Unknown => 1,
        }
    }

    /// Dense bf16/fp16 GEMV coalescing policy. Ordinary decode emits
    /// one active token row; computing multiple output columns per
    /// simdgroup can reuse activation loads, but also increases register
    /// pressure and can reduce occupancy.
    /// Keep the legacy `FUSEKIT_GEMV_ROWS` override for sweeps, while
    /// making M5 shape-aware. Current M5 late-token profiling shows
    /// large bf16 decode wins at one output column per simdgroup: the
    /// extra register pressure from multi-column coalescing costs more
    /// than the activation-load reuse once the chain is already fused.
    pub fn dense_gemv_rows_per_simdgroup(self, _k_dim: u32, _out_dim: u32) -> u32 {
        let env = std::env::var("FUSEKIT_DENSE_GEMV_ROWS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1 && n <= 8)
            .or_else(|| {
                std::env::var("FUSEKIT_GEMV_ROWS")
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .filter(|&n| n >= 1 && n <= 8)
            });
        if let Some(n) = env {
            return n;
        }
        match self {
            ChipModel::M3 | ChipModel::M4 => 2,
            ChipModel::M1 | ChipModel::M2 | ChipModel::M5 | ChipModel::Unknown => 1,
        }
    }

    /// GatherQMM is used by routed expert decode. It has many more
    /// independent route rows than the ordinary decode GEMV path, so a
    /// small multi-row simdgroup improves x-row reuse without changing
    /// the global GEMV policy used by Qwen dense/fanout chains.
    pub fn gatherqmm_rows_per_simdgroup(self) -> u32 {
        let env = std::env::var("FUSEKIT_GATHERQMM_ROWS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1 && n <= 8);
        if let Some(n) = env {
            return n;
        }
        match self {
            ChipModel::M3 | ChipModel::M4 => 2,
            ChipModel::M1 | ChipModel::M2 | ChipModel::M5 | ChipModel::Unknown => 1,
        }
    }

    /// Active token rows a quantized small-M GEMM simdgroup computes
    /// for a single output column. This is the M dimension counterpart
    /// of `gemv_rows_per_simdgroup()`: it shares one quantized weight
    /// stream across several active token rows instead of hard-coding
    /// the MTP M=2 case.
    ///
    /// `FUSEKIT_QGEMM_M_ROWS=N` is the sweep/override knob. The default
    /// stays register-budgeted because every extra M row costs another
    /// accumulator plus activation live ranges in the inner loop.
    pub fn qgemm_m_rows_per_simdgroup(
        self,
        active_rows: u32,
        bits: u32,
        k_dim: u32,
        out_dim: u32,
    ) -> u32 {
        let active_rows = active_rows.max(1);
        if active_rows <= 1 {
            return 1;
        }
        if let Some(n) = std::env::var("FUSEKIT_QGEMM_M_ROWS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1 && n <= 8)
        {
            return n.min(active_rows).max(1);
        }

        let regs = self.budgets().regs_per_thread;
        let mut cap = if regs >= 192 { 4 } else { 2 };
        if !matches!(bits, 4 | 8) {
            // q5/q6 continuous-bit unpack carries more loop temporaries
            // than q4/q8, so keep the token fanout conservative while
            // the output-column tile still amortizes the weight stream.
            cap = cap.min(2);
        }
        if k_dim < 1024 || out_dim < 1024 {
            cap = cap.min(2);
        }
        active_rows.min(cap).max(1)
    }

    /// Output columns a quantized small-M GEMM simdgroup computes for a
    /// single active-token block. This restores the output-column reuse
    /// that ordinary qmv gets from computing several adjacent rows while
    /// still allowing the MTP/M dimension to be fused in the same custom
    /// kernel.
    pub fn qgemm_output_cols_per_simdgroup(
        self,
        active_rows: u32,
        bits: u32,
        k_dim: u32,
        out_dim: u32,
    ) -> u32 {
        if active_rows <= 1 {
            return 1;
        }
        if let Some(n) = std::env::var("FUSEKIT_QGEMM_OUT_COLS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1 && n <= 8)
        {
            let mut n = n.min(out_dim.max(1)).max(1);
            while n > 1 && !out_dim.is_multiple_of(n) {
                n -= 1;
            }
            return n.max(1);
        }

        let _ = (bits, k_dim, out_dim);
        1
    }

    /// Simdgroups per threadgroup for Single(GatherQMM). This mirrors
    /// `kernel_threadgroup_for`'s generic Single path so emit-time
    /// shape heuristics stay aligned with dispatch sizing.
    pub fn gatherqmm_simdgroups_per_tg(self) -> u32 {
        let b = self.budgets();
        let default = b
            .preferred_simdgroups_per_tg
            .min(b.max_simdgroups_per_tg)
            .max(2);
        std::env::var("FUSEKIT_TG_SGS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| (1..=16).contains(&n))
            .unwrap_or(default)
    }

    /// Whether GatherQMM should stage the routed x row in threadgroup
    /// memory. The path pays one barrier and consumes `K*sizeof(float)`
    /// SMEM, so it is only a default win when a full threadgroup's
    /// simdgroups are likely to reuse that row for adjacent output
    /// rows. `FUSEKIT_GATHERQMM_X_SMEM=0/1` remains an explicit sweep
    /// knob, but true still respects the chip SMEM budget.
    pub fn use_gatherqmm_x_smem(
        self,
        k_dim: u32,
        out_dim: u32,
        rows_per_sg: u32,
        active_routes: u32,
    ) -> bool {
        let smem_bytes = (k_dim as u64).saturating_mul(4);
        let fits = smem_bytes <= self.budgets().smem_per_threadgroup as u64;
        if let Some(force) = env_bool("FUSEKIT_GATHERQMM_X_SMEM") {
            return force && fits;
        }
        if !fits || k_dim < 1024 || active_routes == 0 {
            return false;
        }
        let rows_per_sg = rows_per_sg.max(1);
        let row_blocks = out_dim.saturating_add(rows_per_sg).saturating_sub(1) / rows_per_sg;
        row_blocks >= self.gatherqmm_simdgroups_per_tg().max(1)
    }

    pub fn display(self) -> &'static str {
        match self {
            ChipModel::M1 => "Apple M1",
            ChipModel::M2 => "Apple M2",
            ChipModel::M3 => "Apple M3",
            ChipModel::M4 => "Apple M4",
            ChipModel::M5 => "Apple M5",
            ChipModel::Unknown => "Unknown",
        }
    }
}

// ─── Codegen-time current target ───────────────────────────────────────
//
// Per-emit functions (`fused_chain::emit`, `emit_int4_gemv_inner`, …)
// take an MSL writer + fusion + meta but no Target — adding it to
// every signature would touch every kernel module. Instead the
// top-level `emit_metallib` driver sets this thread-local at the
// start of codegen; helpers that want chip-aware decisions
// (`gemv_rows_per_simdgroup`, `supports_simdgroup_matrix`) read it
// via `current_chip()`. Always returns `Unknown` when codegen isn't
// running, which makes helpers fall back to safe defaults.

thread_local! {
    static CURRENT_CHIP: std::cell::Cell<ChipModel> =
        std::cell::Cell::new(ChipModel::Unknown);
}

pub fn set_current_chip(chip: ChipModel) {
    CURRENT_CHIP.with(|c| c.set(chip));
}

pub fn current_chip() -> ChipModel {
    CURRENT_CHIP.with(|c| c.get())
}

impl FromStr for ChipModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        // Strip common prefixes/suffixes so `--chip "Apple M5 Max"`
        // and `--chip m5` and `--chip M5_Pro` all resolve.
        let core = s
            .trim_start_matches("apple")
            .trim_start_matches(' ')
            .trim_start_matches('-')
            .trim_start_matches('_');
        let core = core.split([' ', '_', '-']).next().unwrap_or(core);
        match core {
            "m1" => Ok(ChipModel::M1),
            "m2" => Ok(ChipModel::M2),
            "m3" => Ok(ChipModel::M3),
            "m4" => Ok(ChipModel::M4),
            "m5" => Ok(ChipModel::M5),
            "unknown" | "" => Ok(ChipModel::Unknown),
            other => Err(format!(
                "unknown chip `{other}` (expected one of: m1, m2, m3, m4, m5)"
            )),
        }
    }
}

/// Numeric budgets the codegen heuristics read. All fields are
/// per-threadgroup unless noted.
#[derive(Clone, Copy, Debug)]
pub struct Budgets {
    /// 32-bit register slots per thread we'll plan around. Crossing
    /// this triggers compiler spills to thread-private memory, which
    /// is 10-50× slower than register access.
    pub regs_per_thread: u32,
    /// Threadgroup-shared memory bytes available. All Apple GPUs to
    /// date expose 32 KiB; pinned here so consumers don't reach into
    /// Metal at codegen time.
    pub smem_per_threadgroup: u32,
    /// Lanes per simdgroup (== threadgroup width unit). 32 on all
    /// shipping Apple chips; kept as a field so future variants
    /// can override without leaking the constant.
    pub simd_width: u32,
    /// Preferred number of simdgroups in a threadgroup when SMEM and
    /// register pressure allow it. The codegen uses this as the
    /// default upper bound for `sgs_per_tg`.
    pub preferred_simdgroups_per_tg: u32,
    /// Hard cap on simdgroups per threadgroup. Going above this hurts
    /// occupancy more than it gains from data sharing.
    pub max_simdgroups_per_tg: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub chip: ChipModel,
    pub budgets: Budgets,
}

impl Target {
    /// Auto-detect from the running host. Falls back to `Unknown`
    /// (with conservative budgets) if `sysctl` is unavailable or the
    /// CPU string doesn't match.
    #[cfg(target_os = "macos")]
    pub fn detect_host() -> Self {
        let chip = detect_chip_macos().unwrap_or(ChipModel::Unknown);
        Self {
            chip,
            budgets: chip.budgets(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn detect_host() -> Self {
        Self {
            chip: ChipModel::Unknown,
            budgets: ChipModel::Unknown.budgets(),
        }
    }

    /// Build a target explicitly (typically from a CLI flag).
    pub fn from_chip(chip: ChipModel) -> Self {
        Self {
            chip,
            budgets: chip.budgets(),
        }
    }

    /// Convenience: resolve `--chip <opt>` against the host default.
    /// `None` → detect host, `Some("auto")` → detect host, otherwise
    /// parse and use the explicit chip.
    pub fn resolve(opt: Option<&str>) -> Result<Self, String> {
        match opt {
            None => Ok(Self::detect_host()),
            Some(s) if s.eq_ignore_ascii_case("auto") || s.eq_ignore_ascii_case("host") => {
                Ok(Self::detect_host())
            }
            Some(s) => {
                let chip = s.parse::<ChipModel>()?;
                Ok(Self::from_chip(chip))
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn detect_chip_macos() -> Option<ChipModel> {
    // `sysctl -n machdep.cpu.brand_string` returns something like
    // "Apple M5" / "Apple M3 Max" / "Apple M1 Ultra". Parse the
    // generation digit out of the first token after "Apple M".
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    parse_brand_string(s.trim())
}

fn parse_brand_string(s: &str) -> Option<ChipModel> {
    // Tokens: "Apple", "M5", "Max" etc. The second token usually
    // starts with `M` followed by the generation digit.
    let tok = s.split_whitespace().nth(1)?;
    let rest = tok.strip_prefix('M')?;
    let gen: u32 = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    match gen {
        1 => Some(ChipModel::M1),
        2 => Some(ChipModel::M2),
        3 => Some(ChipModel::M3),
        4 => Some(ChipModel::M4),
        5 => Some(ChipModel::M5),
        _ => Some(ChipModel::Unknown),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_brand() {
        assert_eq!(parse_brand_string("Apple M5"), Some(ChipModel::M5));
        assert_eq!(parse_brand_string("Apple M5 Max"), Some(ChipModel::M5));
        assert_eq!(parse_brand_string("Apple M1 Ultra"), Some(ChipModel::M1));
        assert_eq!(parse_brand_string("Apple M3 Pro"), Some(ChipModel::M3));
        assert_eq!(parse_brand_string("Some Other CPU"), None);
    }

    #[test]
    fn parse_cli() {
        assert_eq!("m5".parse::<ChipModel>().unwrap(), ChipModel::M5);
        assert_eq!("M5".parse::<ChipModel>().unwrap(), ChipModel::M5);
        assert_eq!("Apple M5 Max".parse::<ChipModel>().unwrap(), ChipModel::M5);
        assert_eq!("apple-m1".parse::<ChipModel>().unwrap(), ChipModel::M1);
        assert!("m9".parse::<ChipModel>().is_err());
    }
}
