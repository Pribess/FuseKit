#![allow(dead_code)]

//! Per-`FusionKind` MSL emitters.
//!
//! Each fusion kind has its own emitter that produces an MSL kernel
//! function for one (Fusion, PartitionPlan) pair. Codegen dispatches
//! to the right emitter by matching on `FusionKind`.
//!
//! ```text
//!   Task ─┬─► Fusion (FusionKind X)
//!         │      │
//!         │      │ kernels::emit(fusion, partition, meta, writer)
//!         │      ▼
//!         │   appends MSL function to writer
//!         │
//!         └─► variant_id: index by (kind, shape, partition) to dedupe
//! ```
//!
//! Phase 1 doesn't ship any real emitters — `emit_for` returns `Ok(())`
//! and the persistent scheduler just dispatches on op name as a
//! fallback. Each pattern lands here as its matcher is wired up.

use anyhow::Result;

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{ChainOperand, Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::graph::Dtype;
use crate::task::ir::PartitionPlan;

/// MSL scalar type name matching the model's activation dtype. Use this
/// everywhere the kernel reads/writes a tensor that's allocated by
/// mlx-lm at the model's compute precision — activations, norm
/// weights, biases, KV cache, output buffers. Mismatch here = silent
/// garbage (the buffer is binary-compatible but the bit layout is
/// wrong, since bf16 and fp16 have different exponent widths).
pub fn act_ty(meta: &ModelMeta) -> &'static str {
    match meta.activation_dtype {
        Dtype::Bf16 => "bfloat",
        Dtype::F16 => "half",
        Dtype::F32 => "float",
        _ => "bfloat",
    }
}

pub fn scalar_ty_for_dtype(dtype: Dtype, fallback: &'static str) -> &'static str {
    match dtype {
        Dtype::Bool => "bool",
        Dtype::U8 => "uchar",
        Dtype::U16 => "ushort",
        Dtype::U32 => "uint",
        Dtype::U64 => "ulong",
        Dtype::I8 => "char",
        Dtype::I16 => "short",
        Dtype::I32 => "int",
        Dtype::I64 => "long",
        Dtype::F16 => "half",
        Dtype::F32 => "float",
        Dtype::Bf16 => "bfloat",
        Dtype::F64 | Dtype::Complex64 | Dtype::Other(_) => fallback,
    }
}

fn env_bool_on(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_bool_off(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn ceil_div_u64(n: u64, d: u64) -> u64 {
    n.saturating_add(d.saturating_sub(1)) / d.max(1)
}

pub fn mtp_token_m_qgemv_enabled() -> bool {
    !env_bool_off("FUSEKIT_MTP_TOKEN2_QGEMV")
}

pub fn mtp_token_m_reduction_fanout_enabled() -> bool {
    env_bool_on("FUSEKIT_MTP_TOKEN2_REDUCTION_FANOUT")
}

pub fn qmv_fanout_pair2_compatible(
    a: crate::fusion::ir::QmvFanoutTarget,
    b: crate::fusion::ir::QmvFanoutTarget,
) -> bool {
    let pair2_env = std::env::var("FUSEKIT_QMV_FANOUT_PAIR2").ok();
    if env_bool_off("FUSEKIT_QMV_FANOUT_PAIR2") {
        return false;
    }
    if a.k_dim != b.k_dim
        || a.out_dim != b.out_dim
        || a.active_rows != b.active_rows
        || a.bits != b.bits
        || a.group_size != b.group_size
    {
        return false;
    }
    if pair2_env.is_none() && qmv_fanout_pair2_large_decode_pressure(a) {
        return false;
    }
    qmv_fanout_fast_token_rows(a.active_rows, a.bits, a.k_dim, a.out_dim) == 1
        && qmv_fanout_fast_token_rows(b.active_rows, b.bits, b.k_dim, b.out_dim) == 1
        && qmv_fanout_fast_simdgroups(a.active_rows, a.bits, a.k_dim, a.out_dim)
            == qmv_fanout_fast_simdgroups(b.active_rows, b.bits, b.k_dim, b.out_dim)
        && qmv_fanout_fast_cols_per_simdgroup(a.active_rows, a.bits, a.k_dim, a.out_dim)
            == qmv_fanout_fast_cols_per_simdgroup(b.active_rows, b.bits, b.k_dim, b.out_dim)
}

fn qmv_fanout_pair2_large_decode_pressure(t: crate::fusion::ir::QmvFanoutTarget) -> bool {
    t.active_rows <= 1 && t.group_size == 64 && matches!(t.bits, 4 | 5 | 6 | 8) && t.k_dim >= 1024
}

pub fn qmv_fanout_pair2_m_compatible(
    a: crate::fusion::ir::QmvFanoutTarget,
    b: crate::fusion::ir::QmvFanoutTarget,
) -> bool {
    if !env_bool_on("FUSEKIT_QMV_FANOUT_PAIR2_M") {
        return false;
    }
    if a.k_dim != b.k_dim
        || a.out_dim != b.out_dim
        || a.active_rows != b.active_rows
        || a.bits != b.bits
        || a.group_size != b.group_size
        || !matches!(a.bits, 4 | 8)
    {
        return false;
    }
    let a_rows = qmv_fanout_fast_token_rows(a.active_rows, a.bits, a.k_dim, a.out_dim);
    let b_rows = qmv_fanout_fast_token_rows(b.active_rows, b.bits, b.k_dim, b.out_dim);
    a_rows > 1
        && a_rows == b_rows
        && qmv_fanout_fast_simdgroups(a.active_rows, a.bits, a.k_dim, a.out_dim)
            == qmv_fanout_fast_simdgroups(b.active_rows, b.bits, b.k_dim, b.out_dim)
        && qmv_fanout_fast_cols_per_simdgroup(a.active_rows, a.bits, a.k_dim, a.out_dim)
            == qmv_fanout_fast_cols_per_simdgroup(b.active_rows, b.bits, b.k_dim, b.out_dim)
}

pub fn qmv_fanout_pair2_any_compatible(
    a: crate::fusion::ir::QmvFanoutTarget,
    b: crate::fusion::ir::QmvFanoutTarget,
) -> bool {
    qmv_fanout_pair2_compatible(a, b) || qmv_fanout_pair2_m_compatible(a, b)
}

pub fn qmv_fanout_pair2_same_tile_enabled(targets: &[crate::fusion::ir::QmvFanoutTarget]) -> bool {
    targets.len() == 2 && qmv_fanout_pair2_any_compatible(targets[0], targets[1])
}

pub fn qmv_fanout_exact_enabled() -> bool {
    std::env::var("FUSEKIT_QMV_FANOUT_EXACT").as_deref() == Ok("1")
}

pub fn qmv_fanout_exact_pair2_enabled() -> bool {
    qmv_fanout_exact_enabled() && !env_bool_off("FUSEKIT_QMV_FANOUT_EXACT_PAIR2")
}

pub fn qmv_fanout_exact_pair2_compatible(
    a: crate::fusion::ir::QmvFanoutTarget,
    b: crate::fusion::ir::QmvFanoutTarget,
) -> bool {
    qmv_fanout_exact_pair2_enabled()
        && a.k_dim == b.k_dim
        && a.out_dim == b.out_dim
        && a.active_rows == b.active_rows
        && a.bits == b.bits
        && a.group_size == b.group_size
}

pub fn qmv_exact_simdgroups() -> u32 {
    std::env::var("FUSEKIT_QMV_EXACT_SGS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| (1..=8).contains(&n))
        .unwrap_or(2)
}

pub fn qmv_exact_cols_per_simdgroup() -> u32 {
    std::env::var("FUSEKIT_QMV_EXACT_COLS_PER_SG")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8))
        .unwrap_or(4)
}

pub fn qmv_exact_cols_per_threadgroup() -> u32 {
    qmv_exact_simdgroups() * qmv_exact_cols_per_simdgroup()
}

pub fn qmv_fanout_exact_simdgroups() -> u32 {
    qmv_exact_simdgroups()
}

pub fn qmv_fanout_exact_cols_per_simdgroup() -> u32 {
    qmv_exact_cols_per_simdgroup()
}

pub fn qmv_fanout_exact_cols_per_threadgroup() -> u32 {
    qmv_exact_cols_per_threadgroup()
}

pub fn qmv_fanout_target_y_tiles(target: crate::fusion::ir::QmvFanoutTarget) -> u64 {
    if qmv_fanout_exact_enabled() {
        return ceil_div_u64(
            target.out_dim as u64,
            qmv_fanout_exact_cols_per_threadgroup() as u64,
        );
    }
    let cols_per_tg = qmv_fanout_fast_cols_per_threadgroup(
        target.active_rows,
        target.bits,
        target.k_dim,
        target.out_dim,
    ) as u64;
    ceil_div_u64(target.out_dim as u64, cols_per_tg)
}

fn qmv_fanout_homogeneous_q4_cols_per_simdgroup(
    targets: &[crate::fusion::ir::QmvFanoutTarget],
) -> Option<u32> {
    let first = *targets.first()?;
    if targets.len() < 2
        || first.bits != 4
        || first.group_size != 64
        || first.active_rows > 2
        || first.k_dim < 1024
        || first.out_dim < 4096
    {
        return None;
    }
    if !targets.iter().all(|target| {
        target.k_dim == first.k_dim
            && target.out_dim == first.out_dim
            && target.active_rows == first.active_rows
            && target.bits == first.bits
            && target.group_size == first.group_size
    }) {
        return None;
    }
    if std::env::var("FUSEKIT_QMV_FANOUT_HOMOG_Q4_COLS_PER_SG").as_deref() == Ok("0") {
        return None;
    }
    let cols = std::env::var("FUSEKIT_QMV_FANOUT_HOMOG_Q4_COLS_PER_SG")
        .ok()
        .as_deref()
        .map(str::trim)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8))
        .unwrap_or_else(|| {
            if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
                && first.active_rows <= 1
                && first.out_dim <= 16_384
            {
                // M5 q4 decode fanouts are still register-pressure
                // sensitive, but cols/sg=2 launches enough extra tiles to
                // lose at the full decode-chain level even when an isolated
                // kernel profile looks better. cols/sg=4 is the fastest
                // verified point for this small-M homogeneous fanout class.
                4
            } else {
                8
            }
        });
    if first.out_dim.is_multiple_of(cols) {
        Some(cols)
    } else {
        None
    }
}

fn qmv_fanout_small_q45_cols_per_simdgroup(
    targets: &[crate::fusion::ir::QmvFanoutTarget],
) -> Option<u32> {
    let first = *targets.first()?;
    if targets.len() < 2
        || !matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        || first.group_size != 64
        || first.active_rows > 1
        || first.k_dim < 1024
    {
        return None;
    }
    if !targets.iter().all(|target| {
        target.k_dim == first.k_dim
            && target.active_rows == first.active_rows
            && target.group_size == first.group_size
            && matches!(target.bits, 4 | 5)
            && target.out_dim <= 4096
    }) {
        return None;
    }
    if std::env::var("FUSEKIT_QMV_FANOUT_SMALL_Q45_COLS_PER_SG").as_deref() == Ok("0") {
        return None;
    }
    let cols = std::env::var("FUSEKIT_QMV_FANOUT_SMALL_Q45_COLS_PER_SG")
        .ok()
        .as_deref()
        .map(str::trim)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8))
        .unwrap_or(4);
    targets
        .iter()
        .all(|target| target.out_dim.is_multiple_of(cols))
        .then_some(cols)
}

pub fn qmv_fanout_targets_use_homogeneous_q4_tiling(
    targets: &[crate::fusion::ir::QmvFanoutTarget],
) -> bool {
    qmv_fanout_homogeneous_q4_cols_per_simdgroup(targets).is_some()
}

pub fn qmv_fanout_targets_use_set_tiling(targets: &[crate::fusion::ir::QmvFanoutTarget]) -> bool {
    qmv_fanout_homogeneous_q4_cols_per_simdgroup(targets).is_some()
        || qmv_fanout_small_q45_cols_per_simdgroup(targets).is_some()
}

pub fn qmv_fanout_targets_cols_per_simdgroup(
    targets: &[crate::fusion::ir::QmvFanoutTarget],
) -> u32 {
    if let Some(cols) = qmv_fanout_homogeneous_q4_cols_per_simdgroup(targets) {
        return cols;
    }
    if let Some(cols) = qmv_fanout_small_q45_cols_per_simdgroup(targets) {
        return cols;
    }
    targets
        .first()
        .map(|target| {
            qmv_fanout_fast_cols_per_simdgroup(
                target.active_rows,
                target.bits,
                target.k_dim,
                target.out_dim,
            )
        })
        .unwrap_or(4)
}

pub fn qmv_fanout_targets_simdgroups(targets: &[crate::fusion::ir::QmvFanoutTarget]) -> u32 {
    let Some(first) = targets.first().copied() else {
        return 2;
    };
    let cols_per_sg = qmv_fanout_targets_cols_per_simdgroup(targets);
    qmv_fanout_fast_simdgroups_with_cols(
        first.active_rows,
        first.bits,
        first.k_dim,
        first.out_dim,
        cols_per_sg,
    )
}

pub fn qmv_fanout_targets_cols_per_threadgroup(
    targets: &[crate::fusion::ir::QmvFanoutTarget],
) -> u32 {
    qmv_fanout_targets_simdgroups(targets)
        .saturating_mul(qmv_fanout_targets_cols_per_simdgroup(targets))
        .max(1)
}

pub fn qmv_fanout_y_tiles(targets: &[crate::fusion::ir::QmvFanoutTarget]) -> u64 {
    let target_set_cols_per_tg = (!qmv_fanout_exact_enabled()
        && qmv_fanout_targets_use_set_tiling(targets))
    .then(|| qmv_fanout_targets_cols_per_threadgroup(targets) as u64);
    let mut tiles = 0u64;
    let mut idx = 0usize;
    while idx < targets.len() {
        let target = targets[idx];
        let target_tiles = target_set_cols_per_tg
            .map(|cols_per_tg| ceil_div_u64(target.out_dim as u64, cols_per_tg))
            .unwrap_or_else(|| qmv_fanout_target_y_tiles(target));
        tiles = tiles.saturating_add(target_tiles);
        let pair2 = idx + 1 < targets.len()
            && if qmv_fanout_exact_enabled() {
                qmv_fanout_exact_pair2_compatible(target, targets[idx + 1])
            } else {
                qmv_fanout_pair2_any_compatible(target, targets[idx + 1])
            };
        if pair2 {
            idx += 2;
        } else {
            idx += 1;
        }
    }
    tiles
}

/// Emit two tiny utility kernels used to close the per-step token
/// feedback loop on the GPU so the runtime can pre-submit a chain
/// of decode command buffers without per-step CPU sync.
///
/// `fusekit_step_init` atomically increments a step counter and
/// broadcasts the new `(offset, kv_len)` to the shared uniforms
/// every layer reads — replaces the host-side `for i: write
/// offset_uniform = base + i` we used to do between cmd buffers.
///
/// `fusekit_history_capture` appends the just-written argmax to
/// the chunk's `history` buffer at the next free slot (atomic
/// counter). Lets the streaming consumer pull all chunk tokens
/// out of one host-readable buffer after the cmd buffers commit.
///
/// Both kernels are intentionally trivial (1 thread, 1 threadgroup);
/// the cost we care about is the order they impose, not their
/// own runtime.
pub fn emit_pipeline_utilities(w: &mut MslWriter) {
    w.line("// ── Pipelined-decode utility kernels ────────────────────────");
    w.line("kernel void fusekit_step_init(");
    w.push_indent();
    w.line("device atomic_uint *step_counter [[buffer(0)]],");
    w.line("device uint        *offset_out   [[buffer(1)]],");
    w.line("device uint        *kv_len_out   [[buffer(2)]],");
    w.line("device atomic_uint *barrier_counters [[buffer(3)]],");
    w.line("device const uint  *barrier_count    [[buffer(4)]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("uint o = atomic_fetch_add_explicit(step_counter, 1u, memory_order_relaxed);");
    w.line("*offset_out = o;");
    w.line("*kv_len_out = o + 1u;");
    w.line("for (uint i = 0u; i < *barrier_count; ++i) {");
    w.push_indent();
    w.line("atomic_store_explicit(&barrier_counters[i], 0u, memory_order_relaxed);");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();

    w.line("kernel void fusekit_history_capture(");
    w.push_indent();
    w.line("device atomic_uint *hist_counter [[buffer(0)]],");
    w.line("device const uint  *next_tok     [[buffer(1)]],");
    w.line("device uint        *history      [[buffer(2)]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("uint i = atomic_fetch_add_explicit(hist_counter, 1u, memory_order_relaxed);");
    w.line("history[i] = *next_tok;");
    w.pop_indent();
    w.line("}");
    w.blank();
}

pub mod argmax_logits_decode;
pub mod argmax_logits_finalize;
pub mod argmax_logits_scan;
pub mod arrays_cache_shift_append_decode;
pub mod embed_lookup_decode;
pub mod final_rmsnorm_lm_head;
pub mod fused_chain;
pub mod kv_cache_store_decode;
pub mod layer_block;
pub mod linear_residual;
pub mod qmv_fanout;
pub mod qmv_swiglu;
pub mod rmsnorm_gateup;
pub mod rmsnorm_gateup_swiglu;
pub mod rmsnorm_gateup_swiglu_down;
pub mod rmsnorm_qkv;
pub mod rmsnorm_rope_kv_cache_store_decode;
pub mod rmsnorm_rope_kv_cache_store_decode_quant;
pub mod rmsnorm_rope_qgate_decode;
pub mod rope_kv_cache_store_decode;
pub mod rope_kv_cache_store_decode_quant;
pub mod rope_q_decode;
pub mod sdpa_decode;
pub mod sdpa_decode_quant;
pub mod sdpa_prefill;
pub mod simdgroup_matrix;
pub mod single;
pub mod swiglu_activation;

/// Emit a simdgroup-cooperative GEMV inner loop for affine int4 weights.
/// Each simdgroup handles ONE output row `i`; lanes stride by 32 over
/// the packed-word axis.
///
/// - `x_expr_fn(lane)` returns an MSL expression for `x[in_base + lane]`
///   as a `float`. Lets callers feed either DRAM (`float(x[...])`) or
///   threadgroup SMEM (`norm_y[...]`).
/// - `acc` is the float accumulator name (caller must declare it).
/// - `w_packed` / `w_scales` / `w_biases`: device-pointer expressions
///   pointing at row `i` (e.g. `W_q_packed + i * (HIDDEN / 8)`).
/// - `m_const`: a constexpr name for the input dim (e.g. "HIDDEN").
pub fn qgemv_unroll2_default(bits: u32) -> bool {
    bits == 4 && matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
}

pub fn quant_packed_words(input_dim: u32, bits: u32) -> u32 {
    let bits = bits.max(1) as u64;
    ((input_dim as u64 * bits + 31) / 32) as u32
}

pub fn qmm_mlx_qmv_fast_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    let qmv_env = std::env::var("FUSEKIT_QMM_MLX_QMV").ok();
    if qmv_env.as_deref() == Some("0") {
        return None;
    }
    let (k_dim, out_dim, active_rows, bits, group_size) = qmm_mlx_qmv_tail_safe_shape(fusion)?;
    // MLX qmv_fast has no tail path. Decode and MTP verify use very small
    // active row counts, where qmv is the intended quantized matvec path.
    let auto_small_m = matches!(bits, 4 | 5 | 6 | 8);
    let allow_qmv = auto_small_m
        || qmv_env.as_deref() == Some("1")
        || std::env::var("FUSEKIT_QMM_MLX_QMV_M").as_deref() == Ok("1");
    let tail_free = qmv_fast_shape_is_tail_free(k_dim, out_dim, active_rows, bits);
    if !allow_qmv || !tail_free {
        if std::env::var_os("FUSEKIT_QMM_MLX_QMV_DEBUG").is_some() {
            let block = qmv_fast_block_size(bits).unwrap_or(0);
            let out_cols = qmv_fast_cols_per_threadgroup(active_rows, bits, k_dim, out_dim);
            eprintln!(
                "[fusekit-qmv-shape] reject fusion={} k={} out={} active_rows={} bits={} gs={} allow={} tail_free={} block={} out_cols={}",
                fusion.id,
                k_dim,
                out_dim,
                active_rows,
                bits,
                group_size,
                allow_qmv,
                tail_free,
                block,
                out_cols
            );
        }
        return None;
    }
    Some((k_dim, out_dim, active_rows, bits, group_size))
}

pub fn qmv_fast_token_rows(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    let active_rows = active_rows.max(1).min(u32::MAX as u64) as u32;
    if active_rows <= 1 {
        return 1;
    }
    if let Some(n) = std::env::var("FUSEKIT_QMM_MLX_QMV_TOKEN_ROWS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
    {
        if n == 0 {
            return 1;
        }
        return n.min(active_rows).clamp(1, 4);
    }
    if !mtp_token_m_qgemv_enabled() || !matches!(bits, 4 | 5 | 6 | 8) {
        return 1;
    }
    crate::target::current_chip()
        .qgemm_m_rows_per_simdgroup(active_rows, bits, k_dim, out_dim)
        .min(active_rows)
        .clamp(1, 4)
}

pub fn qmv_fanout_token_rows(targets: &[crate::fusion::ir::QmvFanoutTarget]) -> u32 {
    if qmv_fanout_exact_enabled() {
        return 1;
    }
    targets
        .iter()
        .map(|target| {
            qmv_fanout_fast_token_rows(
                target.active_rows,
                target.bits,
                target.k_dim,
                target.out_dim,
            )
        })
        .min()
        .unwrap_or(1)
        .max(1)
}

pub fn qmv_fanout_fast_token_rows(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    let active_rows_u32 = active_rows.max(1).min(u32::MAX as u64) as u32;
    if active_rows_u32 <= 1 {
        return 1;
    }
    if let Some(n) = std::env::var("FUSEKIT_QMV_FANOUT_TOKEN_ROWS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
    {
        if n == 0 {
            return 1;
        }
        return n.min(active_rows_u32).clamp(1, 4);
    }
    // QmvFanout already spends registers on multiple output targets.
    // Sharing one weight stream across token rows adds another
    // accumulator dimension and measured worse on small-M decode, so
    // keep M parallelism in the grid instead of inside each simdgroup.
    let _ = (bits, k_dim, out_dim);
    1
}

pub fn linear_residual_qmv_fast_token_rows(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
) -> u32 {
    let active_rows = active_rows.max(1).min(u32::MAX as u64) as u32;
    if active_rows <= 1 {
        return 1;
    }
    if let Some(n) = std::env::var("FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_TOKEN_ROWS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
    {
        if n == 0 {
            return 1;
        }
        return n.min(active_rows).clamp(1, 4);
    }
    if !mtp_token_m_qgemv_enabled() || !matches!(bits, 4 | 5 | 6 | 8) {
        return 1;
    }
    if active_rows == 2 && out_dim <= 3072 {
        // Smaller hidden-width MTP verifiers are more sensitive to the
        // row-shared qMV accumulation drift in residual projections. Keep
        // the M dimension parallel in the grid for those cases; wider
        // models (for example 4K hidden) measured faster with row sharing.
        let _ = k_dim;
        return 1;
    }
    crate::target::current_chip()
        .qgemm_m_rows_per_simdgroup(active_rows, bits, k_dim, out_dim)
        .min(active_rows)
        .clamp(1, 4)
}

pub fn linear_residual_qmv_fast_simdgroups(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
) -> u32 {
    let parsed = std::env::var("FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_FAST_SGS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| (1..=8).contains(&n));
    let sgs = parsed.unwrap_or_else(|| qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim));
    let cols = sgs
        .saturating_mul(linear_residual_qmv_fast_cols_per_simdgroup(
            active_rows,
            bits,
            k_dim,
            out_dim,
        ))
        .max(1);
    if out_dim.is_multiple_of(cols) {
        sgs
    } else {
        qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim)
    }
}

pub fn linear_residual_qmv_fast_cols_per_simdgroup(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
) -> u32 {
    let bit_specific = match bits {
        4 => "FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q4_COLS_PER_SG",
        5 => "FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q5_COLS_PER_SG",
        6 => "FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q6_COLS_PER_SG",
        8 => "FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q8_COLS_PER_SG",
        _ => "",
    };
    let parsed = (!bit_specific.is_empty())
        .then(|| std::env::var(bit_specific).ok())
        .flatten()
        .or_else(|| std::env::var("FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_COLS_PER_SG").ok())
        .as_deref()
        .map(str::trim)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8));
    let cols = parsed.unwrap_or_else(|| {
        if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
            && bits == 4
            && active_rows <= 1
            && k_dim >= 8192
            && out_dim >= 512
        {
            // Large-reduction residual qMV (MLP down-proj) repeatedly
            // streams enough q4 weight that column fanout becomes register
            // pressure before it saves meaningful activation bandwidth. M5
            // profiles on 4B-class q4 down-proj favored one output column
            // per simdgroup; smaller residual projections keep the generic
            // policy below.
            1
        } else if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
            && bits == 8
            && active_rows <= 1
            && k_dim >= 4096
            && out_dim >= 512
        {
            // q8 keeps enough byte-lane memory work in flight to benefit
            // from the wider tile; leave the previous large-reduction
            // policy in place until q8 profiles say otherwise.
            8
        } else if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
            && matches!(bits, 4 | 5)
            && active_rows <= 1
            && k_dim <= 4096
            && out_dim >= 1024
        {
            // q4/q5 residual decode projections on M5 are register-pressure
            // sensitive. Smaller output tiles measured lower GPU time for
            // single-token residual GEMV with a <=4K reduction axis.
            2
        } else {
            qmv_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim)
        }
    });
    if out_dim.is_multiple_of(cols) {
        cols
    } else {
        qmv_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim)
    }
}

pub fn linear_residual_qmv_fast_cols_per_threadgroup(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
) -> u32 {
    linear_residual_qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim)
        * linear_residual_qmv_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim)
}

pub fn qmv_fast_simdgroups(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    let parsed = std::env::var("FUSEKIT_QMM_MLX_QMV_FAST_SGS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| (1..=8).contains(&n));
    let sgs = parsed.unwrap_or(2).max(1);
    let cols = sgs.saturating_mul(4).max(1);
    let _ = (active_rows, bits, k_dim);
    if out_dim.is_multiple_of(cols) {
        sgs
    } else {
        2
    }
}

fn qmv_base_fast_simdgroups(out_dim: u32, cols_per_sg: u32) -> u32 {
    let sgs = 2u32;
    let cols = sgs.saturating_mul(cols_per_sg).max(1);
    if out_dim.is_multiple_of(cols) {
        sgs
    } else {
        2
    }
}

fn qmv_fanout_base_fast_simdgroups(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
    cols_per_sg: u32,
) -> u32 {
    let sgs = if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && bits == 4
        && active_rows <= 1
        && k_dim >= 1024
        && out_dim <= 16_384
    {
        // Small-M fanout runs inside a long decode chain. One simdgroup per
        // threadgroup gives the best verified chain-active time for this
        // class by avoiding the extra per-TG pressure from the generic qMV
        // default.
        1
    } else {
        qmv_base_fast_simdgroups(out_dim, cols_per_sg)
    };
    let cols = sgs.saturating_mul(cols_per_sg).max(1);
    if out_dim.is_multiple_of(cols) {
        sgs
    } else {
        qmv_base_fast_simdgroups(out_dim, cols_per_sg)
    }
}

pub fn qmv_fanout_fast_simdgroups(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    let cols_per_sg = qmv_fanout_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim);
    qmv_fanout_fast_simdgroups_with_cols(active_rows, bits, k_dim, out_dim, cols_per_sg)
}

fn qmv_fanout_fast_simdgroups_with_cols(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
    cols_per_sg: u32,
) -> u32 {
    let bit_specific = match bits {
        4 => "FUSEKIT_QMV_FANOUT_Q4_FAST_SGS",
        5 => "FUSEKIT_QMV_FANOUT_Q5_FAST_SGS",
        6 => "FUSEKIT_QMV_FANOUT_Q6_FAST_SGS",
        8 => "FUSEKIT_QMV_FANOUT_Q8_FAST_SGS",
        _ => "",
    };
    let parsed = (!bit_specific.is_empty())
        .then(|| std::env::var(bit_specific).ok())
        .flatten()
        .or_else(|| std::env::var("FUSEKIT_QMV_FANOUT_FAST_SGS").ok())
        .as_deref()
        .map(str::trim)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| (1..=8).contains(&n));
    let sgs = parsed.unwrap_or_else(|| {
        qmv_fanout_base_fast_simdgroups(active_rows, bits, k_dim, out_dim, cols_per_sg)
    });
    let cols = sgs.saturating_mul(cols_per_sg).max(1);
    if out_dim.is_multiple_of(cols) {
        sgs
    } else {
        qmv_fanout_base_fast_simdgroups(active_rows, bits, k_dim, out_dim, cols_per_sg)
    }
}

fn qmv_base_fast_cols_per_simdgroup(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    let cols = if bits == 4 && active_rows <= 2 && out_dim >= 131_072 && k_dim >= 1024 {
        8
    } else {
        4
    };
    if out_dim.is_multiple_of(cols) {
        cols
    } else {
        4
    }
}

pub fn qmv_fast_cols_per_simdgroup(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    let bit_specific = match bits {
        4 => "FUSEKIT_QMM_MLX_QMV_Q4_COLS_PER_SG",
        8 => "FUSEKIT_QMM_MLX_QMV_Q8_COLS_PER_SG",
        _ => "",
    };
    let parsed = (!bit_specific.is_empty())
        .then(|| std::env::var(bit_specific).ok())
        .flatten()
        .or_else(|| std::env::var("FUSEKIT_QMM_MLX_QMV_COLS_PER_SG").ok())
        .as_deref()
        .map(str::trim)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8));
    let cols = parsed
        .unwrap_or_else(|| qmv_base_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim));
    if out_dim.is_multiple_of(cols) {
        cols
    } else {
        4
    }
}

pub fn qmv_fanout_fast_cols_per_simdgroup(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
) -> u32 {
    let bit_specific = match bits {
        4 => "FUSEKIT_QMV_FANOUT_Q4_COLS_PER_SG",
        5 => "FUSEKIT_QMV_FANOUT_Q5_COLS_PER_SG",
        6 => "FUSEKIT_QMV_FANOUT_Q6_COLS_PER_SG",
        8 => "FUSEKIT_QMV_FANOUT_Q8_COLS_PER_SG",
        _ => "",
    };
    let parsed = (!bit_specific.is_empty())
        .then(|| std::env::var(bit_specific).ok())
        .flatten()
        .or_else(|| std::env::var("FUSEKIT_QMV_FANOUT_COLS_PER_SG").ok())
        .as_deref()
        .map(str::trim)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8));
    let cols = parsed
        .unwrap_or_else(|| qmv_base_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim));
    if out_dim.is_multiple_of(cols) {
        cols
    } else {
        qmv_base_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim)
    }
}

pub fn qmv_fast_cols_per_threadgroup(active_rows: u64, bits: u32, k_dim: u32, out_dim: u32) -> u32 {
    qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim)
        * qmv_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim)
}

pub fn qmv_fanout_fast_cols_per_threadgroup(
    active_rows: u64,
    bits: u32,
    k_dim: u32,
    out_dim: u32,
) -> u32 {
    qmv_fanout_fast_simdgroups(active_rows, bits, k_dim, out_dim)
        * qmv_fanout_fast_cols_per_simdgroup(active_rows, bits, k_dim, out_dim)
}

pub fn qmm_mlx_qmv_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    let shape = qmm_mlx_qmv_tail_safe_shape(fusion)?;
    if std::env::var("FUSEKIT_QMM_MLX_QMV_IMPL").as_deref() == Ok("1") {
        return Some(shape);
    }
    if std::env::var("FUSEKIT_QMM_MLX_QMV_IMPL").as_deref() == Ok("0") {
        return None;
    }

    let (k_dim, out_dim, active_rows, bits, _group_size) = shape;
    // qmv_fast intentionally rejects tail shapes because it reads a full
    // MLX qMV block. For single-token decode, the exact qMV variant is still
    // cheaper than falling back to qmm_t_nax on small quantized projections.
    if active_rows == 1
        && matches!(bits, 4 | 8)
        && !qmv_fast_shape_is_tail_free(k_dim, out_dim, active_rows, bits)
    {
        return Some(shape);
    }
    None
}

pub fn qmm_mlx_qmv_tail_safe_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    let FusionAttrs::Single { op_attrs, .. } = &fusion.attrs else {
        return None;
    };
    let crate::graph::Attrs::Quantized { bits, group_size } = op_attrs else {
        return None;
    };
    if *group_size != 64 || !matches!(*bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return None;
    }
    let x_shape = fusion.inputs.first().map(|b| b.shape.as_slice())?;
    let w_shape = fusion.inputs.get(1).map(|b| b.shape.as_slice())?;
    let out_shape = fusion.outputs.first().map(|b| b.shape.as_slice())?;
    if x_shape.is_empty() || w_shape.len() != 2 || out_shape.is_empty() {
        return None;
    }
    let k_dim = x_shape.last().copied().unwrap_or(0).max(1) as u32;
    let out_dim = w_shape[0].max(1) as u32;
    if out_shape.last().copied().unwrap_or(0).max(1) as u32 != out_dim {
        return None;
    }
    let active_rows = qmm_active_rows(out_shape).max(1);
    Some((k_dim, out_dim, active_rows, *bits, *group_size))
}

fn qmm_active_rows(out_shape: &[i32]) -> u64 {
    if out_shape.len() == 2 {
        let trace_m = out_shape[0].max(1) as u32;
        return crate::target::current_chip().active_m(trace_m).max(1) as u64;
    }
    crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1)
}

pub fn qmv_fast_block_size(bits: u32) -> Option<u32> {
    let pack_factor = match bits {
        3 | 5 => 8,
        6 => 4,
        2 | 4 | 8 => 32 / bits,
        _ => return None,
    };
    let packs_per_thread = if bits == 2 { 1 } else { 2 };
    Some(pack_factor * packs_per_thread * 32)
}

pub fn qmv_fast_shape_is_tail_free(k_dim: u32, out_dim: u32, active_rows: u64, bits: u32) -> bool {
    let Some(block_size) = qmv_fast_block_size(bits) else {
        return false;
    };
    let out_cols = qmv_fast_cols_per_threadgroup(active_rows, bits, k_dim, out_dim);
    k_dim.is_multiple_of(block_size) && out_dim.is_multiple_of(out_cols)
}

pub fn qmm_mlx_qmmt_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    if std::env::var("FUSEKIT_QMM_MLX_QMMT").as_deref() != Ok("1") {
        return None;
    }
    let FusionAttrs::Single { op_attrs, .. } = &fusion.attrs else {
        return None;
    };
    let crate::graph::Attrs::Quantized { bits, group_size } = op_attrs else {
        return None;
    };
    if *group_size != 64 || !matches!(*bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return None;
    }
    let x_shape = fusion.inputs.first().map(|b| b.shape.as_slice())?;
    let w_shape = fusion.inputs.get(1).map(|b| b.shape.as_slice())?;
    let out_shape = fusion.outputs.first().map(|b| b.shape.as_slice())?;
    if x_shape.is_empty() || w_shape.len() != 2 || out_shape.is_empty() {
        return None;
    }
    let k_dim = x_shape.last().copied().unwrap_or(0).max(1) as u32;
    let out_dim = w_shape[0].max(1) as u32;
    if out_shape.last().copied().unwrap_or(0).max(1) as u32 != out_dim {
        return None;
    }
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1);
    if !k_dim.is_multiple_of(32) {
        return None;
    }
    Some((k_dim, out_dim, active_rows, *bits, *group_size))
}

pub fn qmm_chain_mlx_qmv_fast_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    let debug = std::env::var("FUSEKIT_DEBUG_QMV_CHAIN_ID")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .is_some_and(|id| id == fusion.id);
    macro_rules! reject {
        ($reason:expr) => {{
            if debug {
                eprintln!(
                    "[fusekit-qmv-chain] reject fusion={} reason={}",
                    fusion.id, $reason
                );
            }
            return None;
        }};
    }
    if std::env::var("FUSEKIT_QMM_MLX_QMV_CHAIN_M").as_deref() == Ok("0") {
        reject!("disabled");
    }
    let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
        reject!("not_fused_chain");
    };
    if steps.len() != 1 {
        reject!(format!("steps={}", steps.len()));
    }
    let step = &steps[0];
    if step.op_name != "QuantizedMatmul" || step.inputs.len() != 4 {
        reject!(format!("op={} inputs={}", step.op_name, step.inputs.len()));
    }
    let crate::graph::Attrs::Quantized { bits, group_size } = step.op_attrs else {
        reject!("attrs_not_quantized");
    };
    if group_size != 64 || !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        reject!(format!("bits={bits} group_size={group_size}"));
    }
    let ChainOperand::External {
        buffer_index: x_buf,
    } = step.inputs[0]
    else {
        reject!("x_not_external");
    };
    let ChainOperand::External {
        buffer_index: w_buf,
    } = step.inputs[1]
    else {
        reject!("w_not_external");
    };
    let x_shape = fusion
        .inputs
        .get(x_buf as usize)
        .map(|b| b.shape.as_slice())?;
    let w_shape = fusion
        .inputs
        .get(w_buf as usize)
        .map(|b| b.shape.as_slice())?;
    let out_shape = fusion.outputs.first().map(|b| b.shape.as_slice())?;
    if x_shape.is_empty() || w_shape.len() != 2 || out_shape.is_empty() {
        reject!(format!(
            "ranks x={} w={} out={}",
            x_shape.len(),
            w_shape.len(),
            out_shape.len()
        ));
    }
    let k_dim = x_shape.last().copied().unwrap_or(0).max(1) as u32;
    let out_dim = w_shape[0].max(1) as u32;
    if out_shape.last().copied().unwrap_or(0).max(1) as u32 != out_dim {
        return None;
    }
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1);
    if active_rows <= 1 {
        reject!(format!("active_rows={active_rows}"));
    }
    let pack_factor = match bits {
        3 | 5 => 8,
        6 => 4,
        _ => 32 / bits,
    };
    let packs_per_thread = if bits == 2 { 1 } else { 2 };
    let block_size = pack_factor * packs_per_thread * 32;
    if debug {
        eprintln!(
            "[fusekit-qmv-chain] candidate fusion={} k={} out={} active_rows={} bits={} gs={} block={} k_ok={} n_ok={}",
            fusion.id,
            k_dim,
            out_dim,
            active_rows,
            bits,
            group_size,
            block_size,
            k_dim.is_multiple_of(block_size),
            out_dim.is_multiple_of(8),
        );
    }
    if !k_dim.is_multiple_of(block_size) || !out_dim.is_multiple_of(8) {
        reject!("alignment");
    }
    Some((k_dim, out_dim, active_rows, bits, group_size))
}

pub fn qmm_chain_nax_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    let debug = std::env::var("FUSEKIT_DEBUG_QMM_CHAIN_NAX_ID")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .is_some_and(|id| id == fusion.id);
    macro_rules! reject {
        ($reason:expr) => {{
            if debug {
                eprintln!(
                    "[fusekit-qmm-chain-nax] reject fusion={} reason={}",
                    fusion.id, $reason
                );
            }
            return None;
        }};
    }
    if !env_bool_on("FUSEKIT_QMM_CHAIN_NAX") {
        reject!("disabled");
    }
    let (bits, group_size, x_buf, w_buf) = match &fusion.attrs {
        FusionAttrs::FusedChain { steps, .. } => {
            if steps.len() != 1 {
                reject!(format!("steps={}", steps.len()));
            }
            let step = &steps[0];
            if step.op_name != "QuantizedMatmul" || step.inputs.len() != 4 {
                reject!(format!("op={} inputs={}", step.op_name, step.inputs.len()));
            }
            let crate::graph::Attrs::Quantized { bits, group_size } = step.op_attrs else {
                reject!("attrs_not_quantized");
            };
            let ChainOperand::External {
                buffer_index: x_buf,
            } = step.inputs[0]
            else {
                reject!("x_not_external");
            };
            let ChainOperand::External {
                buffer_index: w_buf,
            } = step.inputs[1]
            else {
                reject!("w_not_external");
            };
            (bits, group_size, x_buf, w_buf)
        }
        FusionAttrs::Single { op_attrs, .. } => {
            let crate::graph::Attrs::Quantized { bits, group_size } = op_attrs else {
                reject!("single_attrs_not_quantized");
            };
            (*bits, *group_size, 0, 1)
        }
        _ => reject!("attrs_not_qmm_chain"),
    };
    if !matches!((bits, group_size), (4, 64) | (8, 64)) {
        reject!(format!("bits={bits} group_size={group_size}"));
    }
    let Some(x_shape) = fusion
        .inputs
        .get(x_buf as usize)
        .map(|b| b.shape.as_slice())
    else {
        reject!("missing_x_shape");
    };
    let Some(w_shape) = fusion
        .inputs
        .get(w_buf as usize)
        .map(|b| b.shape.as_slice())
    else {
        reject!("missing_w_shape");
    };
    let Some(out_shape) = fusion.outputs.first().map(|b| b.shape.as_slice()) else {
        reject!("missing_out_shape");
    };
    if x_shape.is_empty() || w_shape.len() != 2 || out_shape.is_empty() {
        reject!(format!(
            "ranks x={} w={} out={}",
            x_shape.len(),
            w_shape.len(),
            out_shape.len()
        ));
    }
    let k_dim = x_shape.last().copied().unwrap_or(0).max(1) as u32;
    let out_dim = w_shape[0].max(1) as u32;
    if out_shape.last().copied().unwrap_or(0).max(1) as u32 != out_dim {
        reject!("out_dim_mismatch");
    }
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1);
    if active_rows != 1 && !env_bool_on("FUSEKIT_QMM_CHAIN_NAX_M") {
        reject!(format!("active_rows={active_rows} m_disabled"));
    }
    if active_rows != 1 && out_dim < 64 {
        reject!(format!("out_dim={out_dim} active_rows={active_rows}"));
    }
    if debug {
        eprintln!(
            "[fusekit-qmm-chain-nax] accept fusion={} k={} out={} active_rows={} bits={} gs={}",
            fusion.id, k_dim, out_dim, active_rows, bits, group_size
        );
    }
    Some((k_dim, out_dim, active_rows, bits, group_size))
}

pub fn qmm_nax_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    if std::env::var("FUSEKIT_QMM_NAX").as_deref() == Ok("0") {
        return None;
    }
    let FusionAttrs::Single { op_attrs, .. } = &fusion.attrs else {
        return None;
    };
    let crate::graph::Attrs::Quantized { bits, group_size } = op_attrs else {
        return None;
    };
    if !matches!((*bits, *group_size), (4, 64) | (8, 64)) {
        return None;
    }
    let x_shape = fusion.inputs.first().map(|b| b.shape.as_slice())?;
    let w_shape = fusion.inputs.get(1).map(|b| b.shape.as_slice())?;
    let out_shape = fusion.outputs.first().map(|b| b.shape.as_slice())?;
    if x_shape.is_empty() || w_shape.len() != 2 || out_shape.is_empty() {
        return None;
    }
    let k_dim = x_shape.last().copied().unwrap_or(0).max(1) as u32;
    let out_dim = w_shape[0].max(1) as u32;
    if out_shape.last().copied().unwrap_or(0).max(1) as u32 != out_dim {
        return None;
    }
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1);
    let allow_m = std::env::var("FUSEKIT_QMM_NAX_M").as_deref() == Ok("1")
        || std::env::var("FUSEKIT_QMM_CHAIN_NAX_M").as_deref() == Ok("1");
    if active_rows != 1 && !allow_m {
        return None;
    }
    if active_rows != 1 && out_dim < 64 {
        return None;
    }
    Some((k_dim, out_dim, active_rows, *bits, *group_size))
}

pub fn linear_residual_mlx_qmv_fast_shape(fusion: &Fusion) -> Option<(u32, u32, u64, u32, u32)> {
    if std::env::var("FUSEKIT_LINEAR_RESIDUAL_MLX_QMV").as_deref() == Ok("0") {
        return None;
    }
    let FusionAttrs::LinearResidual { quant: Some(q) } = &fusion.attrs else {
        return None;
    };
    if q.group_size != 64 || !matches!(q.bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return None;
    }
    let x_shape = fusion.inputs.first().map(|b| b.shape.as_slice())?;
    let out_shape = fusion.outputs.first().map(|b| b.shape.as_slice())?;
    if x_shape.is_empty() || out_shape.is_empty() {
        return None;
    }
    let k_dim = x_shape.last().copied().unwrap_or(0).max(1) as u32;
    let out_dim = out_shape.last().copied().unwrap_or(0).max(1) as u32;
    let max_out_dim = std::env::var("FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_MAX_N")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(4096);
    if out_dim > max_out_dim {
        return None;
    }
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1);
    let pack_factor = match q.bits {
        3 | 5 => 8,
        6 => 4,
        _ => 32 / q.bits,
    };
    let packs_per_thread = if q.bits == 2 { 1 } else { 2 };
    let block_size = pack_factor * packs_per_thread * 32;
    let auto_small_m = active_rows > 1 && matches!(q.bits, 4 | 5 | 6 | 8);
    let allow_m = active_rows == 1
        || auto_small_m
        || std::env::var("FUSEKIT_QMM_MLX_QMV_M").as_deref() == Ok("1")
        || std::env::var("FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_M").as_deref() == Ok("1");
    if !allow_m || !k_dim.is_multiple_of(block_size) || !out_dim.is_multiple_of(8) {
        return None;
    }
    Some((k_dim, out_dim, active_rows, q.bits, q.group_size))
}

pub fn qgemv_unroll2_default_for_dim(bits: u32, input_dim: u32) -> bool {
    qgemv_unroll2_default_for_shape(bits, input_dim, u32::MAX)
}

pub fn qgemv_unroll2_default_for_shape(bits: u32, input_dim: u32, output_dim: u32) -> bool {
    if !matches!(crate::target::current_chip(), crate::target::ChipModel::M5) {
        return false;
    }
    match bits {
        // Small q4 GEMV loses more to live-value pressure than it gains from
        // the 2-word unroll. Check both axes: decode down/lm_head can have a
        // large reduction axis but still only a small number of active output
        // rows per layer/model. Larger q4 and q8 keep enough memory work in
        // flight to benefit from the wider loop body.
        4 => input_dim > 1024 && output_dim > 1024,
        // q8 explicit unpack wins on the 4B-class decode projections
        // where both axes are large enough to hide the extra live ranges.
        // Smaller q8 models keep the MLX qdot helper path; forcing the
        // generated q8 body there can stall the oMLX engine path.
        8 => input_dim >= 2048 && output_dim >= 2048,
        _ => false,
    }
}

pub fn qgemv_qparam_broadcast_default_for_shape(
    bits: u32,
    input_dim: u32,
    output_dim: u32,
) -> bool {
    matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && bits == 4
        && input_dim <= 1536
        && output_dim >= 32_768
}

pub fn emit_int4_gemv_inner<F: Fn(&str) -> String>(
    w: &mut MslWriter,
    acc: &str,
    w_packed_row: &str,
    w_scales_row: &str,
    w_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    x_expr: F,
) {
    emit_int4_gemv_inner_with_unroll2_default(
        w,
        acc,
        w_packed_row,
        w_scales_row,
        w_biases_row,
        m_const,
        bits,
        group_size,
        qgemv_unroll2_default(bits),
        x_expr,
    );
}

pub fn emit_int4_gemv_inner_with_unroll2_default<F: Fn(&str) -> String>(
    w: &mut MslWriter,
    acc: &str,
    w_packed_row: &str,
    w_scales_row: &str,
    w_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    qgemv_unroll2_default: bool,
    x_expr: F,
) {
    emit_int4_gemv_inner_with_options(
        w,
        acc,
        w_packed_row,
        w_scales_row,
        w_biases_row,
        m_const,
        bits,
        group_size,
        qgemv_unroll2_default,
        false,
        x_expr,
    );
}

pub fn emit_int4_gemv_inner_with_options<F: Fn(&str) -> String>(
    w: &mut MslWriter,
    acc: &str,
    w_packed_row: &str,
    w_scales_row: &str,
    w_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    qgemv_unroll2_default: bool,
    qparam_broadcast_default: bool,
    x_expr: F,
) {
    let lanes_per_word = 32 / bits;
    let qparam_broadcast = std::env::var("FUSEKIT_QPARAM_BROADCAST")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(qparam_broadcast_default);
    let qgroup_words = if group_size > 0 {
        group_size / lanes_per_word
    } else {
        0
    };
    let qparam_broadcast = qparam_broadcast
        && group_size.is_multiple_of(lanes_per_word)
        && qgroup_words >= 2
        && qgroup_words <= 32
        && 32u32.is_multiple_of(qgroup_words);
    let qgemv_unroll2 = std::env::var("FUSEKIT_QGEMV_UNROLL2")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(qgemv_unroll2_default)
        && !qparam_broadcast;
    let qgemv_prefetch = std::env::var("FUSEKIT_QGEMV_PREFETCH")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
        && !qparam_broadcast
        && !qgemv_unroll2;
    w.line("{");
    w.push_indent();
    w.writeln(format_args!("constexpr uint LANES_ = {lanes_per_word}u;"));
    w.writeln(format_args!("constexpr uint BITS_  = {bits}u;"));
    w.writeln(format_args!("constexpr uint GROUP_ = {group_size}u;"));
    w.line("constexpr uint MASK_ = (1u << BITS_) - 1u;");
    w.writeln(format_args!("uint in_words_ = {m_const} / LANES_;"));
    w.writeln(format_args!("device const uint *w_row_ = {w_packed_row};"));
    let scale_at_group = format!("({w_scales_row})[group_]");
    let bias_at_group = format!("({w_biases_row})[group_]");
    let qgemv_mlx_qdot = std::env::var("FUSEKIT_QGEMV_MLX_QDOT")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        // q8 uses the MLX qdot helper for small shapes, but large M5
        // decode projections pass qgemv_unroll2_default=true from the
        // shape heuristic above and use the generated explicit-unpack
        // path. This keeps the 4B q8 win without forcing the small-q8
        // engine path into the body that can stall.
        .unwrap_or(bits == 8 && !qgemv_unroll2_default)
        && (bits == 4 || bits == 8)
        && group_size.is_multiple_of(lanes_per_word * 2);
    if qgemv_mlx_qdot {
        let emit_word = |w: &mut MslWriter, p: u32| {
            w.writeln(format_args!("uint word_{p}_ = w_row_[widx_ + {p}u];"));
            w.writeln(format_args!("uint inb_{p}_ = (widx_ + {p}u) * LANES_;"));
            for off in 0..lanes_per_word {
                let x = x_expr(&format!("inb_{p}_ + {off}u"));
                w.writeln(format_args!("float x_{p}_{off}_ = {x};"));
            }
            if bits == 4 {
                w.writeln(format_args!(
                    "xsum_ += x_{p}_0_ + x_{p}_1_ + x_{p}_2_ + x_{p}_3_;"
                ));
                w.writeln(format_args!(
                    "xsum_ += x_{p}_4_ + x_{p}_5_ + x_{p}_6_ + x_{p}_7_;"
                ));
                w.writeln(format_args!("uint word_{p}_lo_ = word_{p}_ & 0xFFFFu;"));
                w.writeln(format_args!("uint word_{p}_hi_ = word_{p}_ >> 16u;"));
                w.writeln(format_args!(
                    "qacc_ += (x_{p}_0_ * float(word_{p}_lo_ & 0x000Fu) +"
                ));
                w.writeln(format_args!(
                    "          (x_{p}_1_ / 16.0f) * float(word_{p}_lo_ & 0x00F0u) +"
                ));
                w.writeln(format_args!(
                    "          (x_{p}_2_ / 256.0f) * float(word_{p}_lo_ & 0x0F00u) +"
                ));
                w.writeln(format_args!(
                    "          (x_{p}_3_ / 4096.0f) * float(word_{p}_lo_ & 0xF000u));"
                ));
                w.writeln(format_args!(
                    "qacc_ += (x_{p}_4_ * float(word_{p}_hi_ & 0x000Fu) +"
                ));
                w.writeln(format_args!(
                    "          (x_{p}_5_ / 16.0f) * float(word_{p}_hi_ & 0x00F0u) +"
                ));
                w.writeln(format_args!(
                    "          (x_{p}_6_ / 256.0f) * float(word_{p}_hi_ & 0x0F00u) +"
                ));
                w.writeln(format_args!(
                    "          (x_{p}_7_ / 4096.0f) * float(word_{p}_hi_ & 0xF000u));"
                ));
            } else {
                for chunk in 0..((lanes_per_word + 3) / 4) {
                    let start = chunk * 4;
                    let end = (start + 4).min(lanes_per_word);
                    let terms = (start..end)
                        .map(|off| {
                            let shift = off * bits;
                            format!("x_{p}_{off}_ * float((word_{p}_ >> {shift}u) & MASK_)")
                        })
                        .collect::<Vec<_>>()
                        .join(" + ");
                    for off in start..end {
                        w.writeln(format_args!("xsum_ += x_{p}_{off}_;"));
                    }
                    w.writeln(format_args!("qacc_ += ({terms});"));
                }
            }
        };
        w.line("constexpr uint PACKS_PER_THREAD_ = 2u;");
        w.line("for (uint widx_ = simd_lane * PACKS_PER_THREAD_; widx_ < in_words_; widx_ += 32u * PACKS_PER_THREAD_) {");
        w.push_indent();
        w.line("uint inb_ = widx_ * LANES_;");
        w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
        w.line(format!("float scale_ = float({scale_at_group});"));
        w.line(format!("float bias_  = float({bias_at_group});"));
        w.line("float qacc_ = 0.0f;");
        w.line("float xsum_ = 0.0f;");
        emit_word(w, 0);
        w.line("if (widx_ + 1u < in_words_) {");
        w.push_indent();
        emit_word(w, 1);
        w.pop_indent();
        w.line("}");
        w.writeln(format_args!("{acc} += scale_ * qacc_ + bias_ * xsum_;"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        return;
    }
    // Inner loop body specialized for common bits values. Explicit
    // unroll lets the Metal compiler issue all N fma's into the FPU
    // pipeline rather than the loop version where it can only see a
    // 1-iter sliding window. Generic else-branch handles any other
    // BITS value (e.g. future 6-bit).
    if bits == 4 {
        // int4 → 8 nibbles per uint32 word. Tried float4 dot
        // vectorisation (2 dot4 vs 8 dependent FMAs): regressed
        // -1% on Apple M5. The Metal compiler already overlaps
        // the dependent FMA chain with independent loads so the
        // serial appearance doesn't actually stall the FPU.
        if qgemv_unroll2 {
            let x0 = |off: u32| x_expr(&format!("inb0_ + {off}u"));
            let x1 = |off: u32| x_expr(&format!("inb1_ + {off}u"));
            let xt = |off: u32| x_expr(&format!("inb_ + {off}u"));
            w.line("uint widx_ = simd_lane;");
            w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.line("uint word0_ = w_row_[widx_];");
            w.line("uint word1_ = w_row_[widx_ + 32u];");
            w.line("uint inb0_ = widx_ * 8u;");
            w.line("uint inb1_ = (widx_ + 32u) * 8u;");
            w.writeln(format_args!("uint group0_ = inb0_ / {group_size}u;"));
            w.writeln(format_args!("uint group1_ = inb1_ / {group_size}u;"));
            w.line(format!("float scale0_ = float(({w_scales_row})[group0_]);"));
            w.line(format!("float bias0_  = float(({w_biases_row})[group0_]);"));
            w.line(format!("float scale1_ = float(({w_scales_row})[group1_]);"));
            w.line(format!("float bias1_  = float(({w_biases_row})[group1_]);"));
            for off in 0..8u32 {
                let shift = off * 4;
                w.writeln(format_args!(
                    "{acc} += {x} * (float((word0_ >> {shift}u) & 0xFu) * scale0_ + bias0_);",
                    x = x0(off)
                ));
            }
            for off in 0..8u32 {
                let shift = off * 4;
                w.writeln(format_args!(
                    "{acc} += {x} * (float((word1_ >> {shift}u) & 0xFu) * scale1_ + bias1_);",
                    x = x1(off)
                ));
            }
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
            w.line("uint word_   = w_row_[widx_];");
            w.line("uint inb_    = widx_ * 8u;");
            w.writeln(format_args!("uint group_  = inb_ / {group_size}u;"));
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            for off in 0..8u32 {
                let shift = off * 4;
                w.writeln(format_args!(
                    "{acc} += {x} * (float((word_ >> {shift}u) & 0xFu) * scale_ + bias_);",
                    x = xt(off)
                ));
            }
            w.pop_indent();
            w.line("}");
        } else if qgemv_prefetch {
            let next_scale_at_group = format!("({w_scales_row})[next_group_]");
            let next_bias_at_group = format!("({w_biases_row})[next_group_]");
            w.line("if (simd_lane < in_words_) {");
            w.push_indent();
            w.line("uint widx_   = simd_lane;");
            w.line("uint word_   = w_row_[widx_];");
            w.line("uint inb_    = widx_ * 8u;");
            w.writeln(format_args!("uint group_  = inb_ / {group_size}u;"));
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            w.line("while (true) {");
            w.push_indent();
            w.line("uint next_widx_ = widx_ + 32u;");
            w.line("bool has_next_ = next_widx_ < in_words_;");
            w.line("uint next_word_ = has_next_ ? w_row_[next_widx_] : 0u;");
            w.line("uint next_inb_ = next_widx_ * 8u;");
            w.writeln(format_args!(
                "uint next_group_ = next_inb_ / {group_size}u;"
            ));
            w.line(format!(
                "float next_scale_ = has_next_ ? float({next_scale_at_group}) : 0.0f;"
            ));
            w.line(format!(
                "float next_bias_  = has_next_ ? float({next_bias_at_group}) : 0.0f;"
            ));
        } else {
            w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
            w.push_indent();
            w.line("uint word_   = w_row_[widx_];");
            w.line("uint inb_    = widx_ * 8u;");
            w.writeln(format_args!("uint group_  = inb_ / {group_size}u;"));
            if qparam_broadcast {
                w.writeln(format_args!(
                    "constexpr uint QGROUP_WORDS_ = {qgroup_words}u;"
                ));
                w.line("uint qgroup_src_lane_ = (simd_lane / QGROUP_WORDS_) * QGROUP_WORDS_;");
                w.line(format!(
                    "float scale_src_ = (simd_lane == qgroup_src_lane_) ? float({scale_at_group}) : 0.0f;"
                ));
                w.line(format!(
                    "float bias_src_  = (simd_lane == qgroup_src_lane_) ? float({bias_at_group}) : 0.0f;"
                ));
                w.line("float scale_ = simd_broadcast(scale_src_, qgroup_src_lane_);");
                w.line("float bias_  = simd_broadcast(bias_src_, qgroup_src_lane_);");
            } else {
                w.line(format!("float scale_ = float({scale_at_group});"));
                w.line(format!("float bias_  = float({bias_at_group});"));
            }
        }
        if !qgemv_unroll2 {
            // Each fma uses the same scale/bias, so the compiler can hoist
            // the per-nibble (scale·q + bias) into one FMA after dequant.
            let x = |off: u32| x_expr(&format!("inb_ + {off}u"));
            w.writeln(format_args!(
                "{acc} += {x0} * (float((word_      ) & 0xFu) * scale_ + bias_);",
                x0 = x(0)
            ));
            w.writeln(format_args!(
                "{acc} += {x1} * (float((word_ >>  4u) & 0xFu) * scale_ + bias_);",
                x1 = x(1)
            ));
            w.writeln(format_args!(
                "{acc} += {x2} * (float((word_ >>  8u) & 0xFu) * scale_ + bias_);",
                x2 = x(2)
            ));
            w.writeln(format_args!(
                "{acc} += {x3} * (float((word_ >> 12u) & 0xFu) * scale_ + bias_);",
                x3 = x(3)
            ));
            w.writeln(format_args!(
                "{acc} += {x4} * (float((word_ >> 16u) & 0xFu) * scale_ + bias_);",
                x4 = x(4)
            ));
            w.writeln(format_args!(
                "{acc} += {x5} * (float((word_ >> 20u) & 0xFu) * scale_ + bias_);",
                x5 = x(5)
            ));
            w.writeln(format_args!(
                "{acc} += {x6} * (float((word_ >> 24u) & 0xFu) * scale_ + bias_);",
                x6 = x(6)
            ));
            w.writeln(format_args!(
                "{acc} += {x7} * (float((word_ >> 28u) & 0xFu) * scale_ + bias_);",
                x7 = x(7)
            ));
            if qgemv_prefetch {
                w.line("if (!has_next_) break;");
                w.line("widx_ = next_widx_;");
                w.line("word_ = next_word_;");
                w.line("inb_ = next_inb_;");
                w.line("group_ = next_group_;");
                w.line("scale_ = next_scale_;");
                w.line("bias_ = next_bias_;");
            }
            w.pop_indent();
            w.line("}");
            if qgemv_prefetch {
                w.pop_indent();
                w.line("}");
            }
        }
    } else if bits == 8 {
        // int8 → 4 bytes per uint32 word. Same hoisting principle as
        // the 4-bit path; q8 has 2× the precision so the same MSL
        // pattern matters even more for compile-time fma scheduling.
        if qgemv_unroll2 {
            let x0 = |off: u32| x_expr(&format!("inb0_ + {off}u"));
            let x1 = |off: u32| x_expr(&format!("inb1_ + {off}u"));
            let xt = |off: u32| x_expr(&format!("inb_ + {off}u"));
            w.line("uint widx_ = simd_lane;");
            w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.line("uint word0_ = w_row_[widx_];");
            w.line("uint word1_ = w_row_[widx_ + 32u];");
            w.line("uint inb0_ = widx_ * 4u;");
            w.line("uint inb1_ = (widx_ + 32u) * 4u;");
            w.writeln(format_args!("uint group0_ = inb0_ / {group_size}u;"));
            w.writeln(format_args!("uint group1_ = inb1_ / {group_size}u;"));
            w.line(format!("float scale0_ = float(({w_scales_row})[group0_]);"));
            w.line(format!("float bias0_  = float(({w_biases_row})[group0_]);"));
            w.line(format!("float scale1_ = float(({w_scales_row})[group1_]);"));
            w.line(format!("float bias1_  = float(({w_biases_row})[group1_]);"));
            for off in 0..4u32 {
                let shift = off * 8;
                w.writeln(format_args!(
                    "{acc} += {x} * (float((word0_ >> {shift}u) & 0xFFu) * scale0_ + bias0_);",
                    x = x0(off)
                ));
            }
            for off in 0..4u32 {
                let shift = off * 8;
                w.writeln(format_args!(
                    "{acc} += {x} * (float((word1_ >> {shift}u) & 0xFFu) * scale1_ + bias1_);",
                    x = x1(off)
                ));
            }
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
            w.line("uint word_   = w_row_[widx_];");
            w.line("uint inb_    = widx_ * 4u;");
            w.writeln(format_args!("uint group_  = inb_ / {group_size}u;"));
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            for off in 0..4u32 {
                let shift = off * 8;
                w.writeln(format_args!(
                    "{acc} += {x} * (float((word_ >> {shift}u) & 0xFFu) * scale_ + bias_);",
                    x = xt(off)
                ));
            }
            w.pop_indent();
            w.line("}");
        } else if qgemv_prefetch {
            let next_scale_at_group = format!("({w_scales_row})[next_group_]");
            let next_bias_at_group = format!("({w_biases_row})[next_group_]");
            w.line("if (simd_lane < in_words_) {");
            w.push_indent();
            w.line("uint widx_   = simd_lane;");
            w.line("uint word_   = w_row_[widx_];");
            w.line("uint inb_    = widx_ * 4u;");
            w.writeln(format_args!("uint group_  = inb_ / {group_size}u;"));
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            w.line("while (true) {");
            w.push_indent();
            w.line("uint next_widx_ = widx_ + 32u;");
            w.line("bool has_next_ = next_widx_ < in_words_;");
            w.line("uint next_word_ = has_next_ ? w_row_[next_widx_] : 0u;");
            w.line("uint next_inb_ = next_widx_ * 4u;");
            w.writeln(format_args!(
                "uint next_group_ = next_inb_ / {group_size}u;"
            ));
            w.line(format!(
                "float next_scale_ = has_next_ ? float({next_scale_at_group}) : 0.0f;"
            ));
            w.line(format!(
                "float next_bias_  = has_next_ ? float({next_bias_at_group}) : 0.0f;"
            ));
        } else {
            w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
            w.push_indent();
            w.line("uint word_   = w_row_[widx_];");
            w.line("uint inb_    = widx_ * 4u;");
            w.writeln(format_args!("uint group_  = inb_ / {group_size}u;"));
            if qparam_broadcast {
                w.writeln(format_args!(
                    "constexpr uint QGROUP_WORDS_ = {qgroup_words}u;"
                ));
                w.line("uint qgroup_src_lane_ = (simd_lane / QGROUP_WORDS_) * QGROUP_WORDS_;");
                w.line(format!(
                    "float scale_src_ = (simd_lane == qgroup_src_lane_) ? float({scale_at_group}) : 0.0f;"
                ));
                w.line(format!(
                    "float bias_src_  = (simd_lane == qgroup_src_lane_) ? float({bias_at_group}) : 0.0f;"
                ));
                w.line("float scale_ = simd_broadcast(scale_src_, qgroup_src_lane_);");
                w.line("float bias_  = simd_broadcast(bias_src_, qgroup_src_lane_);");
            } else {
                w.line(format!("float scale_ = float({scale_at_group});"));
                w.line(format!("float bias_  = float({bias_at_group});"));
            }
        }
        if !qgemv_unroll2 {
            let x = |off: u32| x_expr(&format!("inb_ + {off}u"));
            w.writeln(format_args!(
                "{acc} += {x0} * (float((word_      ) & 0xFFu) * scale_ + bias_);",
                x0 = x(0)
            ));
            w.writeln(format_args!(
                "{acc} += {x1} * (float((word_ >>  8u) & 0xFFu) * scale_ + bias_);",
                x1 = x(1)
            ));
            w.writeln(format_args!(
                "{acc} += {x2} * (float((word_ >> 16u) & 0xFFu) * scale_ + bias_);",
                x2 = x(2)
            ));
            w.writeln(format_args!(
                "{acc} += {x3} * (float((word_ >> 24u) & 0xFFu) * scale_ + bias_);",
                x3 = x(3)
            ));
            if qgemv_prefetch {
                w.line("if (!has_next_) break;");
                w.line("widx_ = next_widx_;");
                w.line("word_ = next_word_;");
                w.line("inb_ = next_inb_;");
                w.line("group_ = next_group_;");
                w.line("scale_ = next_scale_;");
                w.line("bias_ = next_bias_;");
            }
            w.pop_indent();
            w.line("}");
            if qgemv_prefetch {
                w.pop_indent();
                w.line("}");
            }
        }
    } else {
        // Non power-of-two-friendly widths such as 5/6-bit are packed
        // as a continuous bitstream by MLX, not as floor(32 / bits)
        // independent lanes with padding per uint32 word. For 64-aligned
        // quant groups, each lane's k and k+32 entries share scale/bias;
        // pair them to halve loop/qparam overhead while preserving the
        // per-lane accumulation order used by the scalar fallback.
        if group_size >= 64 && group_size.is_multiple_of(64) {
            w.line("uint k0_ = simd_lane;");
            w.writeln(format_args!("for (; k0_ + 32u < {m_const}; k0_ += 64u) {{"));
            w.push_indent();
            w.line("uint k1_ = k0_ + 32u;");
            w.line("uint group_ = k0_ / GROUP_;");
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            w.line("uint bit0_   = k0_ * BITS_;");
            w.line("uint widx0_  = bit0_ >> 5u;");
            w.line("uint shift0_ = bit0_ & 31u;");
            w.line("uint word0_  = w_row_[widx0_] >> shift0_;");
            w.line("if (shift0_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word0_ |= w_row_[widx0_ + 1u] << (32u - shift0_);");
            w.pop_indent();
            w.line("}");
            w.line("uint bit1_   = k1_ * BITS_;");
            w.line("uint widx1_  = bit1_ >> 5u;");
            w.line("uint shift1_ = bit1_ & 31u;");
            w.line("uint word1_  = w_row_[widx1_] >> shift1_;");
            w.line("if (shift1_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word1_ |= w_row_[widx1_ + 1u] << (32u - shift1_);");
            w.pop_indent();
            w.line("}");
            w.writeln(format_args!("float x0_ = {};", x_expr("k0_")));
            w.line("uint q0_ = word0_ & MASK_;");
            w.writeln(format_args!(
                "{acc} += x0_ * (float(q0_) * scale_ + bias_);"
            ));
            w.writeln(format_args!("float x1_ = {};", x_expr("k1_")));
            w.line("uint q1_ = word1_ & MASK_;");
            w.writeln(format_args!(
                "{acc} += x1_ * (float(q1_) * scale_ + bias_);"
            ));
            w.pop_indent();
            w.line("}");
            w.writeln(format_args!("if (k0_ < {m_const}) {{"));
            w.push_indent();
            w.line("uint bit_    = k0_ * BITS_;");
            w.line("uint widx_   = bit_ >> 5u;");
            w.line("uint shift_  = bit_ & 31u;");
            w.line("uint word_   = w_row_[widx_] >> shift_;");
            w.line("if (shift_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word_ |= w_row_[widx_ + 1u] << (32u - shift_);");
            w.pop_indent();
            w.line("}");
            w.line("uint group_  = k0_ / GROUP_;");
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            w.writeln(format_args!("float xj_ = {};", x_expr("k0_")));
            w.line("uint q_ = word_ & MASK_;");
            w.writeln(format_args!("{acc} += xj_ * (float(q_) * scale_ + bias_);"));
            w.pop_indent();
            w.line("}");
        } else {
            w.writeln(format_args!(
                "for (uint k_ = simd_lane; k_ < {m_const}; k_ += 32u) {{"
            ));
            w.push_indent();
            w.line("uint bit_    = k_ * BITS_;");
            w.line("uint widx_   = bit_ >> 5u;");
            w.line("uint shift_  = bit_ & 31u;");
            w.line("uint word_   = w_row_[widx_] >> shift_;");
            w.line("if (shift_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word_ |= w_row_[widx_ + 1u] << (32u - shift_);");
            w.pop_indent();
            w.line("}");
            w.line("uint group_  = k_ / GROUP_;");
            w.line(format!("float scale_ = float({scale_at_group});"));
            w.line(format!("float bias_  = float({bias_at_group});"));
            w.writeln(format_args!("float xj_ = {};", x_expr("k_")));
            w.line("uint  q_  = word_ & MASK_;");
            w.line("float wj_ = float(q_) * scale_ + bias_;");
            w.writeln(format_args!("{acc} += xj_ * wj_;"));
            w.pop_indent();
            w.line("}");
        }
    }
    w.pop_indent();
    w.line("}");
}

/// Multi-row variant of `emit_int4_gemv_inner`. Each simdgroup
/// computes `rows` output rows simultaneously, sharing the dequant
/// math for `x` across rows (only the weight word changes per row).
/// On chips with the register headroom (M3+), rows=2 reduces grid TG
/// count by 2× without doubling memory traffic — weight reads stay
/// the same since each row has its own packed nibbles, but the
/// dequant+fma sequence amortizes one scale/bias broadcast per row.
///
/// Callers:
///   - declare `float acc0=0, acc1=0, ...` for each row
///   - call `simd_sum` on each, lane-0-gate each store
///   - pass `w_packed_row_fn(r)` → pointer expression for row r's packed weights
///   - pass `w_scales_row_fn(r)`, `w_biases_row_fn(r)` similarly
///
/// Rows=1 collapses to the single-row helper with identical output.
pub fn emit_int4_gemv_inner_multirow<F, FW, FS, FB>(
    w: &mut MslWriter,
    acc_fn: F,
    w_packed_row_fn: FW,
    w_scales_row_fn: FS,
    w_biases_row_fn: FB,
    m_const: &str,
    bits: u32,
    group_size: u32,
    rows: u32,
    x_expr_in_smem: impl Fn(&str) -> String,
) where
    F: Fn(u32) -> String,
    FW: Fn(u32) -> String,
    FS: Fn(u32) -> String,
    FB: Fn(u32) -> String,
{
    if rows <= 1 {
        emit_int4_gemv_inner(
            w,
            &acc_fn(0),
            &w_packed_row_fn(0),
            &w_scales_row_fn(0),
            &w_biases_row_fn(0),
            m_const,
            bits,
            group_size,
            x_expr_in_smem,
        );
        return;
    }
    let lanes_per_word = 32 / bits;
    let qparam_broadcast = std::env::var("FUSEKIT_QPARAM_BROADCAST")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    let qgroup_words = if group_size > 0 {
        group_size / lanes_per_word
    } else {
        0
    };
    let qparam_broadcast = qparam_broadcast
        && group_size.is_multiple_of(lanes_per_word)
        && qgroup_words >= 2
        && qgroup_words <= 32
        && 32u32.is_multiple_of(qgroup_words);
    let qgemv_multirow_unroll2 = std::env::var("FUSEKIT_QGEMV_MULTIROW_UNROLL2")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
        && (bits == 4 || bits == 8)
        && !qparam_broadcast;
    w.line("{");
    w.push_indent();
    w.writeln(format_args!("constexpr uint LANES_ = {lanes_per_word}u;"));
    w.writeln(format_args!("constexpr uint BITS_  = {bits}u;"));
    w.writeln(format_args!("constexpr uint GROUP_ = {group_size}u;"));
    w.line("constexpr uint MASK_ = (1u << BITS_) - 1u;");
    w.writeln(format_args!("uint in_words_ = {m_const} / LANES_;"));
    for r in 0..rows {
        w.writeln(format_args!(
            "device const uint *w_row_{r} = {p};",
            p = w_packed_row_fn(r)
        ));
    }
    if bits == 4 {
        if qgemv_multirow_unroll2 {
            w.line("uint widx_ = simd_lane;");
            w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.line("uint inb0_ = widx_ * 8u;");
            w.line("uint inb1_ = (widx_ + 32u) * 8u;");
            w.writeln(format_args!("uint group0_ = inb0_ / {group_size}u;"));
            w.writeln(format_args!("uint group1_ = inb1_ / {group_size}u;"));
            for off in 0..8u32 {
                w.writeln(format_args!(
                    "float x0_{off}_ = {};",
                    x_expr_in_smem(&format!("inb0_ + {off}u"))
                ));
            }
            for off in 0..8u32 {
                w.writeln(format_args!(
                    "float x1_{off}_ = {};",
                    x_expr_in_smem(&format!("inb1_ + {off}u"))
                ));
            }
            for r in 0..rows {
                w.writeln(format_args!("uint w0_{r}_ = w_row_{r}[widx_];"));
                w.writeln(format_args!("uint w1_{r}_ = w_row_{r}[widx_ + 32u];"));
                let s_expr0 = format!("({})[group0_]", w_scales_row_fn(r));
                let b_expr0 = format!("({})[group0_]", w_biases_row_fn(r));
                let s_expr1 = format!("({})[group1_]", w_scales_row_fn(r));
                let b_expr1 = format!("({})[group1_]", w_biases_row_fn(r));
                w.writeln(format_args!("float s0_{r}_ = float({s_expr0});"));
                w.writeln(format_args!("float b0_{r}_ = float({b_expr0});"));
                w.writeln(format_args!("float s1_{r}_ = float({s_expr1});"));
                w.writeln(format_args!("float b1_{r}_ = float({b_expr1});"));
                for off in 0..8u32 {
                    let shift = off * 4;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                        "{acc} += x0_{off}_ * (float((w0_{r}_ >> {shift}u) & 0xFu) * s0_{r}_ + b0_{r}_);"
                    ));
                }
                for off in 0..8u32 {
                    let shift = off * 4;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                        "{acc} += x1_{off}_ * (float((w1_{r}_ >> {shift}u) & 0xFu) * s1_{r}_ + b1_{r}_);"
                    ));
                }
            }
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
            w.line("uint inb_   = widx_ * 8u;");
            w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
            for off in 0..8u32 {
                w.writeln(format_args!(
                    "float x_{off}_ = {};",
                    x_expr_in_smem(&format!("inb_ + {off}u"))
                ));
            }
            for r in 0..rows {
                w.writeln(format_args!("uint w_{r}_ = w_row_{r}[widx_];"));
                let s_expr = format!("({})[group_]", w_scales_row_fn(r));
                let b_expr = format!("({})[group_]", w_biases_row_fn(r));
                w.writeln(format_args!("float s_{r}_ = float({s_expr});"));
                w.writeln(format_args!("float b_{r}_ = float({b_expr});"));
                for off in 0..8u32 {
                    let shift = off * 4;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                        "{acc} += x_{off}_ * (float((w_{r}_ >> {shift}u) & 0xFu) * s_{r}_ + b_{r}_);"
                    ));
                }
            }
            w.pop_indent();
            w.line("}");
        } else {
            w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
            w.push_indent();
            w.line("uint inb_   = widx_ * 8u;");
            w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
            if qparam_broadcast {
                w.writeln(format_args!(
                    "constexpr uint QGROUP_WORDS_ = {qgroup_words}u;"
                ));
                w.line("uint qgroup_src_lane_ = (simd_lane / QGROUP_WORDS_) * QGROUP_WORDS_;");
            }
            // Load shared x values once per word.
            for off in 0..8u32 {
                w.writeln(format_args!(
                    "float x_{off}_ = {};",
                    x_expr_in_smem(&format!("inb_ + {off}u"))
                ));
            }
            for r in 0..rows {
                w.writeln(format_args!("uint w_{r}_ = w_row_{r}[widx_];"));
                let s_expr = format!("({})[group_]", w_scales_row_fn(r));
                let b_expr = format!("({})[group_]", w_biases_row_fn(r));
                if qparam_broadcast {
                    w.writeln(format_args!(
                    "float s_src_{r}_ = (simd_lane == qgroup_src_lane_) ? float({s_expr}) : 0.0f;"
                ));
                    w.writeln(format_args!(
                    "float b_src_{r}_ = (simd_lane == qgroup_src_lane_) ? float({b_expr}) : 0.0f;"
                ));
                    w.writeln(format_args!(
                        "float s_{r}_ = simd_broadcast(s_src_{r}_, qgroup_src_lane_);"
                    ));
                    w.writeln(format_args!(
                        "float b_{r}_ = simd_broadcast(b_src_{r}_, qgroup_src_lane_);"
                    ));
                } else {
                    w.writeln(format_args!("float s_{r}_ = float({s_expr});"));
                    w.writeln(format_args!("float b_{r}_ = float({b_expr});"));
                }
                for off in 0..8u32 {
                    let shift = off * 4;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                    "{acc} += x_{off}_ * (float((w_{r}_ >> {shift}u) & 0xFu) * s_{r}_ + b_{r}_);"
                ));
                }
            }
            w.pop_indent();
            w.line("}");
        }
    } else if bits == 8 {
        if qgemv_multirow_unroll2 {
            w.line("uint widx_ = simd_lane;");
            w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.line("uint inb0_ = widx_ * 4u;");
            w.line("uint inb1_ = (widx_ + 32u) * 4u;");
            w.writeln(format_args!("uint group0_ = inb0_ / {group_size}u;"));
            w.writeln(format_args!("uint group1_ = inb1_ / {group_size}u;"));
            for off in 0..4u32 {
                w.writeln(format_args!(
                    "float x0_{off}_ = {};",
                    x_expr_in_smem(&format!("inb0_ + {off}u"))
                ));
            }
            for off in 0..4u32 {
                w.writeln(format_args!(
                    "float x1_{off}_ = {};",
                    x_expr_in_smem(&format!("inb1_ + {off}u"))
                ));
            }
            for r in 0..rows {
                w.writeln(format_args!("uint w0_{r}_ = w_row_{r}[widx_];"));
                w.writeln(format_args!("uint w1_{r}_ = w_row_{r}[widx_ + 32u];"));
                let s_expr0 = format!("({})[group0_]", w_scales_row_fn(r));
                let b_expr0 = format!("({})[group0_]", w_biases_row_fn(r));
                let s_expr1 = format!("({})[group1_]", w_scales_row_fn(r));
                let b_expr1 = format!("({})[group1_]", w_biases_row_fn(r));
                w.writeln(format_args!("float s0_{r}_ = float({s_expr0});"));
                w.writeln(format_args!("float b0_{r}_ = float({b_expr0});"));
                w.writeln(format_args!("float s1_{r}_ = float({s_expr1});"));
                w.writeln(format_args!("float b1_{r}_ = float({b_expr1});"));
                for off in 0..4u32 {
                    let shift = off * 8;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                        "{acc} += x0_{off}_ * (float((w0_{r}_ >> {shift}u) & 0xFFu) * s0_{r}_ + b0_{r}_);"
                    ));
                }
                for off in 0..4u32 {
                    let shift = off * 8;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                        "{acc} += x1_{off}_ * (float((w1_{r}_ >> {shift}u) & 0xFFu) * s1_{r}_ + b1_{r}_);"
                    ));
                }
            }
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
            w.line("uint inb_   = widx_ * 4u;");
            w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
            for off in 0..4u32 {
                w.writeln(format_args!(
                    "float x_{off}_ = {};",
                    x_expr_in_smem(&format!("inb_ + {off}u"))
                ));
            }
            for r in 0..rows {
                w.writeln(format_args!("uint w_{r}_ = w_row_{r}[widx_];"));
                let s_expr = format!("({})[group_]", w_scales_row_fn(r));
                let b_expr = format!("({})[group_]", w_biases_row_fn(r));
                w.writeln(format_args!("float s_{r}_ = float({s_expr});"));
                w.writeln(format_args!("float b_{r}_ = float({b_expr});"));
                for off in 0..4u32 {
                    let shift = off * 8;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                        "{acc} += x_{off}_ * (float((w_{r}_ >> {shift}u) & 0xFFu) * s_{r}_ + b_{r}_);"
                    ));
                }
            }
            w.pop_indent();
            w.line("}");
        } else {
            w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
            w.push_indent();
            w.line("uint inb_   = widx_ * 4u;");
            w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
            if qparam_broadcast {
                w.writeln(format_args!(
                    "constexpr uint QGROUP_WORDS_ = {qgroup_words}u;"
                ));
                w.line("uint qgroup_src_lane_ = (simd_lane / QGROUP_WORDS_) * QGROUP_WORDS_;");
            }
            for off in 0..4u32 {
                w.writeln(format_args!(
                    "float x_{off}_ = {};",
                    x_expr_in_smem(&format!("inb_ + {off}u"))
                ));
            }
            for r in 0..rows {
                w.writeln(format_args!("uint w_{r}_ = w_row_{r}[widx_];"));
                let s_expr = format!("({})[group_]", w_scales_row_fn(r));
                let b_expr = format!("({})[group_]", w_biases_row_fn(r));
                if qparam_broadcast {
                    w.writeln(format_args!(
                    "float s_src_{r}_ = (simd_lane == qgroup_src_lane_) ? float({s_expr}) : 0.0f;"
                ));
                    w.writeln(format_args!(
                    "float b_src_{r}_ = (simd_lane == qgroup_src_lane_) ? float({b_expr}) : 0.0f;"
                ));
                    w.writeln(format_args!(
                        "float s_{r}_ = simd_broadcast(s_src_{r}_, qgroup_src_lane_);"
                    ));
                    w.writeln(format_args!(
                        "float b_{r}_ = simd_broadcast(b_src_{r}_, qgroup_src_lane_);"
                    ));
                } else {
                    w.writeln(format_args!("float s_{r}_ = float({s_expr});"));
                    w.writeln(format_args!("float b_{r}_ = float({b_expr});"));
                }
                for off in 0..4u32 {
                    let shift = off * 8;
                    let acc = acc_fn(r);
                    w.writeln(format_args!(
                    "{acc} += x_{off}_ * (float((w_{r}_ >> {shift}u) & 0xFFu) * s_{r}_ + b_{r}_);"
                ));
                }
            }
            w.pop_indent();
            w.line("}");
        }
    } else {
        // Other bit widths fall back to single-row + serialised loop.
        for r in 0..rows {
            let acc = acc_fn(r);
            emit_int4_gemv_inner(
                w,
                &acc,
                &w_packed_row_fn(r),
                &w_scales_row_fn(r),
                &w_biases_row_fn(r),
                m_const,
                bits,
                group_size,
                &x_expr_in_smem,
            );
        }
    }
    w.pop_indent();
    w.line("}");
}

/// Small-M quantized GEMM against one quantized weight row. One
/// simdgroup owns one output column and accumulates `token_rows`
/// active M rows while loading/dequantizing the packed weight stream
/// once. This is the M-dimension counterpart to output-row coalescing:
/// it keeps decode M=1 cheap, MTP M=2 efficient, and larger M variants
/// budgeted instead of hard-coded.
pub fn emit_int4_gemv_inner_token_rows<FA, FX>(
    w: &mut MslWriter,
    acc_fn: FA,
    w_packed_row: &str,
    w_scales_row: &str,
    w_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    token_rows: u32,
    x_expr: FX,
) where
    FA: Fn(u32) -> String,
    FX: Fn(u32, &str) -> String,
{
    emit_int4_gemv_inner_token_rows_with_options(
        w,
        acc_fn,
        w_packed_row,
        w_scales_row,
        w_biases_row,
        m_const,
        bits,
        group_size,
        token_rows,
        qgemv_unroll2_default(bits),
        x_expr,
    );
}

pub fn emit_int4_gemv_inner_token_rows_with_options<FA, FX>(
    w: &mut MslWriter,
    acc_fn: FA,
    w_packed_row: &str,
    w_scales_row: &str,
    w_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    token_rows: u32,
    qgemv_unroll2_default: bool,
    x_expr: FX,
) where
    FA: Fn(u32) -> String,
    FX: Fn(u32, &str) -> String,
{
    if token_rows <= 1 {
        emit_int4_gemv_inner_with_unroll2_default(
            w,
            &acc_fn(0),
            w_packed_row,
            w_scales_row,
            w_biases_row,
            m_const,
            bits,
            group_size,
            qgemv_unroll2_default,
            |kk| x_expr(0, kk),
        );
        return;
    }
    let qgemv_unroll2 = std::env::var("FUSEKIT_QGEMM_UNROLL2")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(qgemv_unroll2_default);
    if bits != 4 && bits != 8 {
        w.line("{");
        w.push_indent();
        w.writeln(format_args!("constexpr uint BITS_  = {bits}u;"));
        w.writeln(format_args!("constexpr uint GROUP_ = {group_size}u;"));
        w.line("constexpr uint MASK_ = (1u << BITS_) - 1u;");
        w.writeln(format_args!("device const uint *w_row_ = {w_packed_row};"));
        if group_size >= 64 && group_size.is_multiple_of(64) {
            w.line("uint k0_ = simd_lane;");
            w.writeln(format_args!("for (; k0_ + 32u < {m_const}; k0_ += 64u) {{"));
            w.push_indent();
            w.line("uint k1_ = k0_ + 32u;");
            w.line("uint group_ = k0_ / GROUP_;");
            w.writeln(format_args!(
                "float scale_ = float(({w_scales_row})[group_]);"
            ));
            w.writeln(format_args!(
                "float bias_  = float(({w_biases_row})[group_]);"
            ));
            w.line("uint bit0_   = k0_ * BITS_;");
            w.line("uint widx0_  = bit0_ >> 5u;");
            w.line("uint shift0_ = bit0_ & 31u;");
            w.line("uint word0_  = w_row_[widx0_] >> shift0_;");
            w.line("if (shift0_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word0_ |= w_row_[widx0_ + 1u] << (32u - shift0_);");
            w.pop_indent();
            w.line("}");
            w.line("uint bit1_   = k1_ * BITS_;");
            w.line("uint widx1_  = bit1_ >> 5u;");
            w.line("uint shift1_ = bit1_ & 31u;");
            w.line("uint word1_  = w_row_[widx1_] >> shift1_;");
            w.line("if (shift1_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word1_ |= w_row_[widx1_ + 1u] << (32u - shift1_);");
            w.pop_indent();
            w.line("}");
            w.line("uint q0_ = word0_ & MASK_;");
            w.line("float wj0_ = float(q0_) * scale_ + bias_;");
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += {x} * wj0_;",
                    acc = acc_fn(r),
                    x = x_expr(r, "k0_")
                ));
            }
            w.line("uint q1_ = word1_ & MASK_;");
            w.line("float wj1_ = float(q1_) * scale_ + bias_;");
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += {x} * wj1_;",
                    acc = acc_fn(r),
                    x = x_expr(r, "k1_")
                ));
            }
            w.pop_indent();
            w.line("}");
            w.writeln(format_args!("if (k0_ < {m_const}) {{"));
            w.push_indent();
            w.line("uint bit_    = k0_ * BITS_;");
            w.line("uint widx_   = bit_ >> 5u;");
            w.line("uint shift_  = bit_ & 31u;");
            w.line("uint word_   = w_row_[widx_] >> shift_;");
            w.line("if (shift_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word_ |= w_row_[widx_ + 1u] << (32u - shift_);");
            w.pop_indent();
            w.line("}");
            w.line("uint group_  = k0_ / GROUP_;");
            w.writeln(format_args!(
                "float scale_ = float(({w_scales_row})[group_]);"
            ));
            w.writeln(format_args!(
                "float bias_  = float(({w_biases_row})[group_]);"
            ));
            w.line("uint q_ = word_ & MASK_;");
            w.line("float wj_ = float(q_) * scale_ + bias_;");
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += {x} * wj_;",
                    acc = acc_fn(r),
                    x = x_expr(r, "k0_")
                ));
            }
            w.pop_indent();
            w.line("}");
        } else {
            w.writeln(format_args!(
                "for (uint k_ = simd_lane; k_ < {m_const}; k_ += 32u) {{"
            ));
            w.push_indent();
            w.line("uint bit_    = k_ * BITS_;");
            w.line("uint widx_   = bit_ >> 5u;");
            w.line("uint shift_  = bit_ & 31u;");
            w.line("uint word_   = w_row_[widx_] >> shift_;");
            w.line("if (shift_ + BITS_ > 32u) {");
            w.push_indent();
            w.line("word_ |= w_row_[widx_ + 1u] << (32u - shift_);");
            w.pop_indent();
            w.line("}");
            w.line("uint group_  = k_ / GROUP_;");
            w.writeln(format_args!(
                "float scale_ = float(({w_scales_row})[group_]);"
            ));
            w.writeln(format_args!(
                "float bias_  = float(({w_biases_row})[group_]);"
            ));
            w.line("uint q_ = word_ & MASK_;");
            w.line("float wj_ = float(q_) * scale_ + bias_;");
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += {x} * wj_;",
                    acc = acc_fn(r),
                    x = x_expr(r, "k_")
                ));
            }
            w.pop_indent();
            w.line("}");
        }
        w.pop_indent();
        w.line("}");
        return;
    }
    let lanes_per_word = 32 / bits;
    w.line("{");
    w.push_indent();
    w.writeln(format_args!("constexpr uint LANES_ = {lanes_per_word}u;"));
    w.writeln(format_args!("constexpr uint BITS_  = {bits}u;"));
    w.writeln(format_args!("constexpr uint GROUP_ = {group_size}u;"));
    w.line("constexpr uint MASK_ = (1u << BITS_) - 1u;");
    w.writeln(format_args!("uint in_words_ = {m_const} / LANES_;"));
    w.writeln(format_args!("device const uint *w_row_ = {w_packed_row};"));
    if qgemv_unroll2 {
        w.line("uint widx_ = simd_lane;");
        w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
        w.push_indent();
        w.line("uint word0_ = w_row_[widx_];");
        w.line("uint word1_ = w_row_[widx_ + 32u];");
        w.writeln(format_args!("uint inb0_ = widx_ * {lanes_per_word}u;"));
        w.writeln(format_args!(
            "uint inb1_ = (widx_ + 32u) * {lanes_per_word}u;"
        ));
        w.writeln(format_args!("uint group0_ = inb0_ / {group_size}u;"));
        w.writeln(format_args!("uint group1_ = inb1_ / {group_size}u;"));
        w.writeln(format_args!(
            "float scale0_ = float(({w_scales_row})[group0_]);"
        ));
        w.writeln(format_args!(
            "float bias0_  = float(({w_biases_row})[group0_]);"
        ));
        w.writeln(format_args!(
            "float scale1_ = float(({w_scales_row})[group1_]);"
        ));
        w.writeln(format_args!(
            "float bias1_  = float(({w_biases_row})[group1_]);"
        ));
        if bits == 4 {
            for off in 0..8u32 {
                let shift = off * 4;
                w.writeln(format_args!(
                    "float wv0_{off}_ = float((word0_ >> {shift}u) & 0xFu) * scale0_ + bias0_;"
                ));
                for r in 0..token_rows {
                    w.writeln(format_args!(
                        "{acc} += {x} * wv0_{off}_;",
                        acc = acc_fn(r),
                        x = x_expr(r, &format!("inb0_ + {off}u"))
                    ));
                }
            }
            for off in 0..8u32 {
                let shift = off * 4;
                w.writeln(format_args!(
                    "float wv1_{off}_ = float((word1_ >> {shift}u) & 0xFu) * scale1_ + bias1_;"
                ));
                for r in 0..token_rows {
                    w.writeln(format_args!(
                        "{acc} += {x} * wv1_{off}_;",
                        acc = acc_fn(r),
                        x = x_expr(r, &format!("inb1_ + {off}u"))
                    ));
                }
            }
        } else {
            for off in 0..4u32 {
                let shift = off * 8;
                w.writeln(format_args!(
                    "float wv0_{off}_ = float((word0_ >> {shift}u) & 0xFFu) * scale0_ + bias0_;"
                ));
                for r in 0..token_rows {
                    w.writeln(format_args!(
                        "{acc} += {x} * wv0_{off}_;",
                        acc = acc_fn(r),
                        x = x_expr(r, &format!("inb0_ + {off}u"))
                    ));
                }
            }
            for off in 0..4u32 {
                let shift = off * 8;
                w.writeln(format_args!(
                    "float wv1_{off}_ = float((word1_ >> {shift}u) & 0xFFu) * scale1_ + bias1_;"
                ));
                for r in 0..token_rows {
                    w.writeln(format_args!(
                        "{acc} += {x} * wv1_{off}_;",
                        acc = acc_fn(r),
                        x = x_expr(r, &format!("inb1_ + {off}u"))
                    ));
                }
            }
        }
        w.pop_indent();
        w.line("}");
        w.line("if (widx_ < in_words_) {");
        w.push_indent();
    } else {
        w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
        w.push_indent();
    }
    w.line("uint word_ = w_row_[widx_];");
    w.writeln(format_args!("uint inb_ = widx_ * {lanes_per_word}u;"));
    w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
    w.writeln(format_args!(
        "float scale_ = float(({w_scales_row})[group_]);"
    ));
    w.writeln(format_args!(
        "float bias_  = float(({w_biases_row})[group_]);"
    ));
    if bits == 4 {
        for off in 0..8u32 {
            let shift = off * 4;
            w.writeln(format_args!(
                "float wv_{off}_ = float((word_ >> {shift}u) & 0xFu) * scale_ + bias_;"
            ));
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += {x} * wv_{off}_;",
                    acc = acc_fn(r),
                    x = x_expr(r, &format!("inb_ + {off}u"))
                ));
            }
        }
    } else if bits == 8 {
        for off in 0..4u32 {
            let shift = off * 8;
            w.writeln(format_args!(
                "float wv_{off}_ = float((word_ >> {shift}u) & 0xFFu) * scale_ + bias_;"
            ));
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += {x} * wv_{off}_;",
                    acc = acc_fn(r),
                    x = x_expr(r, &format!("inb_ + {off}u"))
                ));
            }
        }
    }
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
}

pub fn emit_int4_gemv_inner_token_cols_with_options<FA, FW, FS, FB, FX>(
    w: &mut MslWriter,
    acc_fn: FA,
    w_packed_col_fn: FW,
    w_scales_col_fn: FS,
    w_biases_col_fn: FB,
    m_const: &str,
    bits: u32,
    group_size: u32,
    token_rows: u32,
    col_rows: u32,
    qgemv_unroll2_default: bool,
    x_expr: FX,
) where
    FA: Fn(u32, u32) -> String,
    FW: Fn(u32) -> String,
    FS: Fn(u32) -> String,
    FB: Fn(u32) -> String,
    FX: Fn(u32, &str) -> String,
{
    let token_rows = token_rows.max(1);
    let col_rows = col_rows.max(1);
    if col_rows <= 1 {
        emit_int4_gemv_inner_token_rows_with_options(
            w,
            |r| acc_fn(r, 0),
            &w_packed_col_fn(0),
            &w_scales_col_fn(0),
            &w_biases_col_fn(0),
            m_const,
            bits,
            group_size,
            token_rows,
            qgemv_unroll2_default,
            x_expr,
        );
        return;
    }

    let qgemv_unroll2 = std::env::var("FUSEKIT_QGEMM_UNROLL2")
        .ok()
        .map(|s| {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(qgemv_unroll2_default);

    w.line("{");
    w.push_indent();
    w.writeln(format_args!("constexpr uint BITS_  = {bits}u;"));
    w.writeln(format_args!("constexpr uint GROUP_ = {group_size}u;"));
    w.line("constexpr uint MASK_ = (1u << BITS_) - 1u;");
    for c in 0..col_rows {
        w.writeln(format_args!(
            "device const uint *w_col_{c} = {p};",
            p = w_packed_col_fn(c)
        ));
    }

    if bits == 4 || bits == 8 {
        let lanes_per_word = 32 / bits;
        let mask = if bits == 4 { "0xFu" } else { "0xFFu" };
        w.writeln(format_args!("constexpr uint LANES_ = {lanes_per_word}u;"));
        w.writeln(format_args!("uint in_words_ = {m_const} / LANES_;"));
        if qgemv_unroll2 {
            w.line("uint widx_ = simd_lane;");
            w.line("for (; widx_ + 32u < in_words_; widx_ += 64u) {");
            w.push_indent();
            w.writeln(format_args!("uint inb0_ = widx_ * {lanes_per_word}u;"));
            w.writeln(format_args!(
                "uint inb1_ = (widx_ + 32u) * {lanes_per_word}u;"
            ));
            w.writeln(format_args!("uint group0_ = inb0_ / {group_size}u;"));
            w.writeln(format_args!("uint group1_ = inb1_ / {group_size}u;"));
            for r in 0..token_rows {
                for off in 0..lanes_per_word {
                    w.writeln(format_args!(
                        "float x0_{r}_{off}_ = {};",
                        x_expr(r, &format!("inb0_ + {off}u"))
                    ));
                }
                for off in 0..lanes_per_word {
                    w.writeln(format_args!(
                        "float x1_{r}_{off}_ = {};",
                        x_expr(r, &format!("inb1_ + {off}u"))
                    ));
                }
            }
            for c in 0..col_rows {
                w.writeln(format_args!("uint word0_{c}_ = w_col_{c}[widx_];"));
                w.writeln(format_args!("uint word1_{c}_ = w_col_{c}[widx_ + 32u];"));
                w.writeln(format_args!(
                    "float scale0_{c}_ = float(({})[group0_]);",
                    w_scales_col_fn(c)
                ));
                w.writeln(format_args!(
                    "float bias0_{c}_  = float(({})[group0_]);",
                    w_biases_col_fn(c)
                ));
                w.writeln(format_args!(
                    "float scale1_{c}_ = float(({})[group1_]);",
                    w_scales_col_fn(c)
                ));
                w.writeln(format_args!(
                    "float bias1_{c}_  = float(({})[group1_]);",
                    w_biases_col_fn(c)
                ));
                for off in 0..lanes_per_word {
                    let shift = off * bits;
                    w.writeln(format_args!(
                        "float wv0_{c}_{off}_ = float((word0_{c}_ >> {shift}u) & {mask}) * scale0_{c}_ + bias0_{c}_;"
                    ));
                    for r in 0..token_rows {
                        w.writeln(format_args!(
                            "{acc} += x0_{r}_{off}_ * wv0_{c}_{off}_;",
                            acc = acc_fn(r, c)
                        ));
                    }
                }
                for off in 0..lanes_per_word {
                    let shift = off * bits;
                    w.writeln(format_args!(
                        "float wv1_{c}_{off}_ = float((word1_{c}_ >> {shift}u) & {mask}) * scale1_{c}_ + bias1_{c}_;"
                    ));
                    for r in 0..token_rows {
                        w.writeln(format_args!(
                            "{acc} += x1_{r}_{off}_ * wv1_{c}_{off}_;",
                            acc = acc_fn(r, c)
                        ));
                    }
                }
            }
            w.pop_indent();
            w.line("}");
            w.line("if (widx_ < in_words_) {");
            w.push_indent();
        } else {
            w.line("for (uint widx_ = simd_lane; widx_ < in_words_; widx_ += 32u) {");
            w.push_indent();
        }
        w.writeln(format_args!("uint inb_ = widx_ * {lanes_per_word}u;"));
        w.writeln(format_args!("uint group_ = inb_ / {group_size}u;"));
        for r in 0..token_rows {
            for off in 0..lanes_per_word {
                w.writeln(format_args!(
                    "float x_{r}_{off}_ = {};",
                    x_expr(r, &format!("inb_ + {off}u"))
                ));
            }
        }
        for c in 0..col_rows {
            w.writeln(format_args!("uint word_{c}_ = w_col_{c}[widx_];"));
            w.writeln(format_args!(
                "float scale_{c}_ = float(({})[group_]);",
                w_scales_col_fn(c)
            ));
            w.writeln(format_args!(
                "float bias_{c}_  = float(({})[group_]);",
                w_biases_col_fn(c)
            ));
            for off in 0..lanes_per_word {
                let shift = off * bits;
                w.writeln(format_args!(
                    "float wv_{c}_{off}_ = float((word_{c}_ >> {shift}u) & {mask}) * scale_{c}_ + bias_{c}_;"
                ));
                for r in 0..token_rows {
                    w.writeln(format_args!(
                        "{acc} += x_{r}_{off}_ * wv_{c}_{off}_;",
                        acc = acc_fn(r, c)
                    ));
                }
            }
        }
        w.pop_indent();
        w.line("}");
    } else if group_size >= 64 && group_size.is_multiple_of(64) {
        w.line("uint k0_ = simd_lane;");
        w.writeln(format_args!("for (; k0_ + 32u < {m_const}; k0_ += 64u) {{"));
        w.push_indent();
        w.line("uint k1_ = k0_ + 32u;");
        w.line("uint group_ = k0_ / GROUP_;");
        for r in 0..token_rows {
            w.writeln(format_args!("float x0_{r}_ = {};", x_expr(r, "k0_")));
            w.writeln(format_args!("float x1_{r}_ = {};", x_expr(r, "k1_")));
        }
        w.line("uint bit0_   = k0_ * BITS_;");
        w.line("uint widx0_  = bit0_ >> 5u;");
        w.line("uint shift0_ = bit0_ & 31u;");
        w.line("uint bit1_   = k1_ * BITS_;");
        w.line("uint widx1_  = bit1_ >> 5u;");
        w.line("uint shift1_ = bit1_ & 31u;");
        for c in 0..col_rows {
            w.writeln(format_args!(
                "float scale_{c}_ = float(({})[group_]);",
                w_scales_col_fn(c)
            ));
            w.writeln(format_args!(
                "float bias_{c}_  = float(({})[group_]);",
                w_biases_col_fn(c)
            ));
            w.writeln(format_args!(
                "uint word0_{c}_ = w_col_{c}[widx0_] >> shift0_;"
            ));
            w.line("if (shift0_ + BITS_ > 32u) {");
            w.push_indent();
            w.writeln(format_args!(
                "word0_{c}_ |= w_col_{c}[widx0_ + 1u] << (32u - shift0_);"
            ));
            w.pop_indent();
            w.line("}");
            w.writeln(format_args!(
                "uint word1_{c}_ = w_col_{c}[widx1_] >> shift1_;"
            ));
            w.line("if (shift1_ + BITS_ > 32u) {");
            w.push_indent();
            w.writeln(format_args!(
                "word1_{c}_ |= w_col_{c}[widx1_ + 1u] << (32u - shift1_);"
            ));
            w.pop_indent();
            w.line("}");
            w.writeln(format_args!(
                "float wv0_{c}_ = float(word0_{c}_ & MASK_) * scale_{c}_ + bias_{c}_;"
            ));
            w.writeln(format_args!(
                "float wv1_{c}_ = float(word1_{c}_ & MASK_) * scale_{c}_ + bias_{c}_;"
            ));
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += x0_{r}_ * wv0_{c}_;",
                    acc = acc_fn(r, c)
                ));
                w.writeln(format_args!(
                    "{acc} += x1_{r}_ * wv1_{c}_;",
                    acc = acc_fn(r, c)
                ));
            }
        }
        w.pop_indent();
        w.line("}");
        w.writeln(format_args!("if (k0_ < {m_const}) {{"));
        w.push_indent();
        w.line("uint group_ = k0_ / GROUP_;");
        for r in 0..token_rows {
            w.writeln(format_args!("float x_{r}_ = {};", x_expr(r, "k0_")));
        }
        w.line("uint bit_    = k0_ * BITS_;");
        w.line("uint widx_   = bit_ >> 5u;");
        w.line("uint shift_  = bit_ & 31u;");
        for c in 0..col_rows {
            w.writeln(format_args!(
                "float scale_{c}_ = float(({})[group_]);",
                w_scales_col_fn(c)
            ));
            w.writeln(format_args!(
                "float bias_{c}_  = float(({})[group_]);",
                w_biases_col_fn(c)
            ));
            w.writeln(format_args!("uint word_{c}_ = w_col_{c}[widx_] >> shift_;"));
            w.line("if (shift_ + BITS_ > 32u) {");
            w.push_indent();
            w.writeln(format_args!(
                "word_{c}_ |= w_col_{c}[widx_ + 1u] << (32u - shift_);"
            ));
            w.pop_indent();
            w.line("}");
            w.writeln(format_args!(
                "float wv_{c}_ = float(word_{c}_ & MASK_) * scale_{c}_ + bias_{c}_;"
            ));
            for r in 0..token_rows {
                w.writeln(format_args!(
                    "{acc} += x_{r}_ * wv_{c}_;",
                    acc = acc_fn(r, c)
                ));
            }
        }
        w.pop_indent();
        w.line("}");
    } else {
        for c in 0..col_rows {
            emit_int4_gemv_inner_token_rows_with_options(
                w,
                |r| acc_fn(r, c),
                &w_packed_col_fn(c),
                &w_scales_col_fn(c),
                &w_biases_col_fn(c),
                m_const,
                bits,
                group_size,
                token_rows,
                qgemv_unroll2_default,
                |r, kk| x_expr(r, kk),
            );
        }
    }
    w.pop_indent();
    w.line("}");
}

/// Compatibility wrapper for the old M=2 callsites.
pub fn emit_int4_gemv_inner_token2<FX0, FX1>(
    w: &mut MslWriter,
    acc0: &str,
    acc1: &str,
    w_packed_row: &str,
    w_scales_row: &str,
    w_biases_row: &str,
    m_const: &str,
    bits: u32,
    group_size: u32,
    x0_expr: FX0,
    x1_expr: FX1,
) where
    FX0: Fn(&str) -> String,
    FX1: Fn(&str) -> String,
{
    emit_int4_gemv_inner_token_rows(
        w,
        |r| {
            if r == 0 {
                acc0.to_string()
            } else {
                acc1.to_string()
            }
        },
        w_packed_row,
        w_scales_row,
        w_biases_row,
        m_const,
        bits,
        group_size,
        2,
        |r, kk| {
            if r == 0 {
                x0_expr(kk)
            } else {
                x1_expr(kk)
            }
        },
    );
}

/// Reduction-emit mode for RmsNorm-style sum-of-squares. Naive is
/// fastest but TG-size-dependent (FP non-associativity → output drift
/// when `tg_size` changes). Pairwise is TG-invariant. Kahan reduces
/// per-thread sequential drift. PairwiseKahan combines both.
///
/// Selected at codegen time via `FUSEKIT_NORM_REDUCE` env var:
///   * `naive` (default) — match historical behavior, fastest
///   * `pairwise`        — TG-invariant via fixed thread chunking +
///                         fixed-depth inter-SG tree
///   * `kahan`           — compensated per-thread sum; HW simd_sum
///                         still order-dep so not TG-invariant
///   * `pairwise_kahan`  — both fixes combined
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormReductionMode {
    Naive,
    Pairwise,
    Kahan,
    PairwiseKahan,
}

pub fn norm_reduction_mode() -> NormReductionMode {
    match std::env::var("FUSEKIT_NORM_REDUCE").as_deref() {
        Ok(s) => match s.trim() {
            "pairwise" => NormReductionMode::Pairwise,
            "kahan" => NormReductionMode::Kahan,
            "pairwise_kahan" | "both" => NormReductionMode::PairwiseKahan,
            _ => NormReductionMode::Naive,
        },
        _ => NormReductionMode::Naive,
    }
}

/// Emit a RmsNorm-style `sum_sq = Σ x[i]²` reduction. Caller has
/// already declared `tid`, `tg_size`, `threadgroup float
/// {partial_buf}[32]`, and bound `{input_var}` (an act_ty array of
/// length `hidden`). After this call returns:
///   - `simd_lane`, `simd_id` are declared as `uint` locals
///   - `{partial_buf}[0]` holds the global Σx² (visible to all threads)
///   - SMEM barrier already issued; safe to read immediately
///
/// Pick mode via [`norm_reduction_mode`] (env-controlled). See
/// `memory/project_fusekit_rmsnorm_drift.md` for analysis of why
/// naive mode drifts and how each fix works.
pub fn emit_rmsnorm_reduction(w: &mut MslWriter, hidden: u32, input_var: &str, partial_buf: &str) {
    let mode = norm_reduction_mode();
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    match mode {
        NormReductionMode::Naive => emit_norm_naive(w, hidden, input_var, partial_buf),
        NormReductionMode::Pairwise => emit_norm_pairwise(w, hidden, input_var, partial_buf),
        NormReductionMode::Kahan => emit_norm_kahan(w, hidden, input_var, partial_buf),
        NormReductionMode::PairwiseKahan => {
            emit_norm_pairwise_kahan(w, hidden, input_var, partial_buf)
        }
    }
}

/// Emit the `norm_y[i] = x[i] * rms_scale * norm_w[i]` writeback loop.
/// Caller must have already declared `rms_scale`, `norm_y` (threadgroup
/// array of size HIDDEN), and bound `{input_var}` + `{norm_w_var}`.
///
/// In `naive` mode: standard `for (i = tid; i < HIDDEN; i += tg_size)`
/// stride pattern. Per-i compute is bit-identical but the *write
/// order* depends on tg_size — Apple Metal's SMEM store coalescing
/// is implementation-defined, contributing to TG-size logit drift.
///
/// In `pairwise`/`pairwise_kahan` modes: fixed CHUNK mapping. Only
/// the first `HIDDEN/CHUNK` threads write, each handling a contiguous
/// CHUNK of elements. Write address pattern is TG-invariant: thread
/// `t` always writes `norm_y[t*CHUNK .. t*CHUNK+CHUNK-1]` regardless
/// of `tg_size`.
pub fn emit_norm_writeback(w: &mut MslWriter, hidden: u32, input_var: &str, norm_w_var: &str) {
    emit_norm_writeback_with_act_ty(w, hidden, input_var, norm_w_var, "float");
}

pub fn emit_norm_writeback_with_act_ty(
    w: &mut MslWriter,
    hidden: u32,
    input_var: &str,
    norm_w_var: &str,
    act_ty: &str,
) {
    let mode = norm_reduction_mode();
    let mlx_rmsnorm_value = |x_expr: String, w_expr: String| match act_ty {
        // MLX rms_norm writes: out[i] = w[i] * T(x[i] * inv_rms).
        // Keep the same dtype boundary even when the FuseKit consumer reads
        // from threadgroup float scratch instead of a materialized T tensor.
        "bfloat" | "half" => {
            format!("float({act_ty}(({w_expr}) * {act_ty}(({x_expr}) * rms_scale)))")
        }
        _ => format!("({x_expr}) * rms_scale * ({w_expr})"),
    };
    let fixed = matches!(
        mode,
        NormReductionMode::Pairwise | NormReductionMode::PairwiseKahan
    );
    if fixed {
        let chunk = if hidden.is_multiple_of(8) {
            8
        } else if hidden.is_multiple_of(4) {
            4
        } else if hidden.is_multiple_of(2) {
            2
        } else {
            1
        };
        w.writeln(format_args!("constexpr uint NORM_WB_CHUNK = {chunk}u;"));
        w.line("constexpr uint NORM_WB_ACTIVE = HIDDEN / NORM_WB_CHUNK;");
        w.line("for (uint block = tid; block < NORM_WB_ACTIVE; block += tg_size) {");
        w.push_indent();
        w.line("uint base = block * NORM_WB_CHUNK;");
        w.line("for (uint j = 0u; j < NORM_WB_CHUNK; ++j) {");
        w.push_indent();
        w.writeln(format_args!(
            "norm_y[base + j] = {};",
            mlx_rmsnorm_value(
                format!("float({input_var}[base + j])"),
                format!("{norm_w_var}[base + j]"),
            )
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.writeln(format_args!(
            "norm_y[i] = {};",
            mlx_rmsnorm_value(format!("float({input_var}[i])"), format!("{norm_w_var}[i]"))
        ));
        w.pop_indent();
        w.line("}");
    }
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

/// Historical reduction. Per-thread loop iterates `tid..hidden by tg_size`,
/// inter-SG uses `simd_sum` on a variable-NSG subset. Order depends on
/// tg_size → FP drift when TG sizing changes.
fn emit_norm_naive(w: &mut MslWriter, _hidden: u32, input_var: &str, partial_buf: &str) {
    w.line("float sum_sq = 0.0f;");
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.writeln(format_args!("float xi = float({input_var}[i]);"));
    w.line("sum_sq += xi * xi;");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.writeln(format_args!(
        "if (simd_lane == 0u) {partial_buf}[simd_id] = sum_sq;"
    ));
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.writeln(format_args!(
        "float v = (tid < (tg_size / 32u)) ? {partial_buf}[tid] : 0.0f;"
    ));
    w.line("v = simd_sum(v);");
    w.writeln(format_args!("if (simd_lane == 0u) {partial_buf}[0] = v;"));
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

/// Pairwise tree + fixed thread chunking. TG-invariant.
fn emit_norm_pairwise(w: &mut MslWriter, hidden: u32, input_var: &str, partial_buf: &str) {
    // Pick CHUNK so HIDDEN % CHUNK == 0 and CHUNK divides ≥ 4. For
    // typical Qwen hidden (896, 1024, 1536, 2560, 5120), CHUNK=4
    // always works.
    let chunk = if hidden.is_multiple_of(8) {
        8
    } else if hidden.is_multiple_of(4) {
        4
    } else if hidden.is_multiple_of(2) {
        2
    } else {
        1
    };
    w.writeln(format_args!("constexpr uint NORM_CHUNK = {chunk}u;"));
    w.line("constexpr uint NORM_ACTIVE = HIDDEN / NORM_CHUNK;");
    w.line("float sum_sq = 0.0f;");
    // Fixed thread→element mapping: thread `t` always reads elements
    // [t*CHUNK, t*CHUNK+CHUNK). Independent of tg_size.
    w.line("for (uint block = tid; block < NORM_ACTIVE; block += tg_size) {");
    w.push_indent();
    w.line("uint base = block * NORM_CHUNK;");
    w.line("for (uint j = 0u; j < NORM_CHUNK; ++j) {");
    w.push_indent();
    w.writeln(format_args!("float xi = float({input_var}[base + j]);"));
    w.line("sum_sq += xi * xi;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    // Intra-SG: HW simd_sum is order-deterministic across any TG size.
    w.line("sum_sq = simd_sum(sum_sq);");
    w.writeln(format_args!(
        "if (simd_lane == 0u) {partial_buf}[simd_id] = sum_sq;"
    ));
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    // Inter-SG: lane-0 sequential sum over a FIXED 16 positions.
    // `simd_shuffle_down` was the obvious primitive here, but its
    // out-of-range semantics are **implementation-defined** on Apple
    // Metal (lane id + delta >= SIMD-width). Even when our tree
    // arithmetic is mathematically correct, the OOR reads can affect
    // register state in vendor-dependent ways and produce ulp-level
    // differences between TG sizes. Sequential sum in lane 0 only
    // is slower (16 dependent adds vs 4-depth tree) but bit-identical
    // across any TG size — every TG reads the same 16 positions
    // (`sum_sq_partial[0..15]`) in the same order, with zero-padding
    // for unused slots.
    w.line("if (simd_id == 0u && simd_lane == 0u) {");
    w.push_indent();
    w.line("uint nsg = tg_size / 32u;");
    w.line("float total = 0.0f;");
    w.line("for (uint i = 0u; i < 16u; ++i) {");
    w.push_indent();
    w.writeln(format_args!(
        "float v = (i < nsg) ? {partial_buf}[i] : 0.0f;"
    ));
    w.line("total += v;");
    w.pop_indent();
    w.line("}");
    w.writeln(format_args!("{partial_buf}[0] = total;"));
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

/// Kahan compensated per-thread sum. Drives sequential accumulation
/// error to `~ε²` instead of `N·ε`. Inter-SG combine remains naive
/// (HW simd_sum on variable-NSG subset) → not TG-invariant.
fn emit_norm_kahan(w: &mut MslWriter, _hidden: u32, input_var: &str, partial_buf: &str) {
    w.line("float sum_sq = 0.0f;");
    w.line("float kahan_c = 0.0f;");
    w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
    w.push_indent();
    w.writeln(format_args!("float xi = float({input_var}[i]);"));
    w.line("float term = xi * xi;");
    // Kahan: y = term - c; t = sum + y; c = (t - sum) - y; sum = t.
    w.line("float y = term - kahan_c;");
    w.line("float t = sum_sq + y;");
    w.line("kahan_c = (t - sum_sq) - y;");
    w.line("sum_sq = t;");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.writeln(format_args!(
        "if (simd_lane == 0u) {partial_buf}[simd_id] = sum_sq;"
    ));
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.writeln(format_args!(
        "float v = (tid < (tg_size / 32u)) ? {partial_buf}[tid] : 0.0f;"
    ));
    w.line("v = simd_sum(v);");
    w.writeln(format_args!("if (simd_lane == 0u) {partial_buf}[0] = v;"));
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

/// Pairwise tree + Kahan compensated per-thread sum. Maximum accuracy
/// + TG-invariance. ~4-5% per-norm overhead.
fn emit_norm_pairwise_kahan(w: &mut MslWriter, hidden: u32, input_var: &str, partial_buf: &str) {
    let chunk = if hidden.is_multiple_of(8) {
        8
    } else if hidden.is_multiple_of(4) {
        4
    } else if hidden.is_multiple_of(2) {
        2
    } else {
        1
    };
    w.writeln(format_args!("constexpr uint NORM_CHUNK = {chunk}u;"));
    w.line("constexpr uint NORM_ACTIVE = HIDDEN / NORM_CHUNK;");
    w.line("float sum_sq = 0.0f;");
    w.line("float kahan_c = 0.0f;");
    w.line("for (uint block = tid; block < NORM_ACTIVE; block += tg_size) {");
    w.push_indent();
    w.line("uint base = block * NORM_CHUNK;");
    w.line("for (uint j = 0u; j < NORM_CHUNK; ++j) {");
    w.push_indent();
    w.writeln(format_args!("float xi = float({input_var}[base + j]);"));
    w.line("float term = xi * xi;");
    w.line("float y = term - kahan_c;");
    w.line("float t = sum_sq + y;");
    w.line("kahan_c = (t - sum_sq) - y;");
    w.line("sum_sq = t;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.writeln(format_args!(
        "if (simd_lane == 0u) {partial_buf}[simd_id] = sum_sq;"
    ));
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u && simd_lane == 0u) {");
    w.push_indent();
    w.line("uint nsg = tg_size / 32u;");
    w.line("float total = 0.0f;");
    w.line("for (uint i = 0u; i < 16u; ++i) {");
    w.push_indent();
    w.writeln(format_args!(
        "float v = (i < nsg) ? {partial_buf}[i] : 0.0f;"
    ));
    w.line("total += v;");
    w.pop_indent();
    w.line("}");
    w.writeln(format_args!("{partial_buf}[0] = total;"));
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

/// Emit the MSL kernel function for `fusion` under `partition`. The
/// function name must be stable for the megakernel's dispatch switch
/// to bind to it.
pub fn emit_for(
    fusion: &Fusion,
    partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    match fusion.kind {
        FusionKind::RmsNormQkv => rmsnorm_qkv::emit(fusion, partition, meta, w),
        FusionKind::RmsNormGateUp => rmsnorm_gateup::emit(fusion, partition, meta, w),
        FusionKind::RmsNormGateUpSwiglu => rmsnorm_gateup_swiglu::emit(fusion, partition, meta, w),
        FusionKind::RmsNormGateUpSwigluDown => {
            rmsnorm_gateup_swiglu_down::emit(fusion, partition, meta, w)
        }
        FusionKind::FinalRmsNormLmHead => final_rmsnorm_lm_head::emit(fusion, partition, meta, w),
        FusionKind::RopeQDecode => rope_q_decode::emit(fusion, partition, meta, w),
        FusionKind::RopeKvCacheStoreDecode => {
            rope_kv_cache_store_decode::emit(fusion, partition, meta, w)
        }
        FusionKind::KvCacheStoreDecode => kv_cache_store_decode::emit(fusion, partition, meta, w),
        FusionKind::RmsNormRopeQGateDecode => {
            rmsnorm_rope_qgate_decode::emit(fusion, partition, meta, w)
        }
        FusionKind::RmsNormRopeKvCacheStoreDecode => {
            rmsnorm_rope_kv_cache_store_decode::emit(fusion, partition, meta, w)
        }
        FusionKind::ArraysCacheShiftAppendDecode => {
            arrays_cache_shift_append_decode::emit(fusion, partition, meta, w)
        }
        FusionKind::EmbedLookupDecode => embed_lookup_decode::emit(fusion, partition, meta, w),
        FusionKind::ArgmaxLogitsScan => argmax_logits_scan::emit(fusion, partition, meta, w),
        FusionKind::ArgmaxLogitsFinalize => {
            argmax_logits_finalize::emit(fusion, partition, meta, w)
        }
        FusionKind::SdpaPrefill => sdpa_prefill::emit(fusion, partition, meta, w),
        FusionKind::ArgmaxLogitsDecode => argmax_logits_decode::emit(fusion, partition, meta, w),
        FusionKind::LinearResidual => linear_residual::emit(fusion, partition, meta, w),
        FusionKind::SwigluActivation => swiglu_activation::emit(fusion, partition, meta, w),
        FusionKind::SdpaDecode => sdpa_decode::emit(fusion, partition, meta, w),
        // Universal 1:1 emitter for any unmatched op. The compiler
        // invariant: every fusion lands an MSL function in the
        // metallib, never silently no-ops.
        FusionKind::Single(_) => single::emit(fusion, partition, meta, w),
        FusionKind::QmvFanout => qmv_fanout::emit(fusion, partition, meta, w),
        FusionKind::QmvSwiglu => qmv_swiglu::emit(fusion, partition, meta, w),
        // Stage 3 general-fusion product — one MSL kernel per chain,
        // intermediates in registers.
        FusionKind::FusedChain => fused_chain::emit(fusion, partition, meta, w),
        // Per-layer megakernel — composes norm + SDPA + o-residual
        // (attention block) into one MSL function via the piece composer.
        FusionKind::LayerBlock => layer_block::emit(fusion, partition, meta, w),
        _ => Ok(()),
    }
}

/// Per-buffer-slot role tag emitted alongside each kernel. The
/// fusion-graph binding tells the runtime *where* a buffer comes
/// from (model param vs scratch vs scalar); this enum tells it *how*
/// to use it — which slot is the offset uniform, which is the cache
/// pointer the host has to keep up to date, which is the token-id
/// input it writes per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotRole {
    /// The current input token id. Host writes a `uint32` here each
    /// generate step.
    TokenIdIn,
    /// The picked next-token id (in-ICB argmax output). Host reads
    /// after `execute_icb` to feed the next step.
    NextTokenOut,
    /// `cache.offset` for the owning layer.
    LayerOffset,
    /// `cache.keys.shape[2]` — total capacity.
    LayerCapacity,
    /// `offset + L` — host writes each step.
    LayerKvLen,
    /// Live mlx-lm `cache[N].keys` pointer.
    LayerCacheK,
    /// Live mlx-lm `cache[N].values` pointer.
    LayerCacheV,
    /// Live mlx-lm `ArraysCache` slot pointer, for state-space layers.
    LayerArraysCache { index: u32 },
    /// Runtime-owned `ArraysCache` rollback snapshot slot. MTP verify
    /// with `n_confirmed=1` needs the state after token 0 while the
    /// active-M kernel advances the live cache through all rows.
    LayerArraysCacheRollback { index: u32 },
    /// Per-layer scratch buffer the hand-tuned signature reuses
    /// across calls — `q_buf`, `k_buf`, `v_buf`, `sdpa_buf`, etc.
    /// The runtime allocates one buffer per layer per role.
    LayerScratch(LayerScratchKind),
    /// Global pre/post-amble scratch (the embed output that feeds
    /// layer 0, or the logits buffer that feeds the argmax).
    GlobalScratch(GlobalScratchKind),
    /// External model parameter resolved by Python attribute path.
    Param(ParamKind),
    /// Per-layer model parameter — runtime resolves to
    /// `model.layers.{layer_idx}.{kind.relpath()}`.
    LayerParam {
        kind: LayerParamKind,
        layer_idx: i32,
    },
    /// Per-layer atomic counter buffer used by `GlobalBarrier`
    /// piece. Host allocates one `atomic_uint` slot per stage and
    /// resets to zero before each ICB execute.
    BarrierCounters { layer_idx: i32 },
}

/// Named per-layer scratch buffers the hand-tuned signatures reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerScratchKind {
    QBuf,
    KBuf,
    VBuf,
    SdpaBuf,
    HAttn,
    OutBuf,
    SiluUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalScratchKind {
    /// First-layer input (output of embed lookup).
    XIn,
    /// Logits row produced by `FinalRmsNormLmHead`.
    Logits,
    /// Per-row, per-TG (max_key, idx) scratch produced by
    /// `ArgmaxLogitsScan` and consumed by `ArgmaxLogitsFinalize`.
    /// Shape [ACTIVE_ROWS, N_TGS, 2] uint.
    ArgmaxScratch,
    /// Flash-Decoding partials across ALL layers. Single global
    /// buffer indexed by `(layer_idx, head_id, chunk_id)` with the
    /// kernel emit baking `layer_idx` as a constexpr per fusion.
    /// Sized: n_layers × n_heads × MAX_CHUNKS × (HEAD_DIM + 2) f32.
    SdpaPartials,
    /// Flash-Decoding per-(layer, head) atomic counter. Sized:
    /// n_layers × n_heads uint32. Each active chunk does ONE
    /// atomic_fetch_add — the TG whose increment returns
    /// (active_chunks - 1) performs combine and resets the counter.
    SdpaCounter,
}

/// Model-parameter slot kinds the hand-tuned signatures know about
/// by structural role. The runtime resolves each to a live mx.array
/// by walking the model module tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKind {
    /// `model.embed_tokens.weight`.
    EmbedTokens,
    /// `model.embed_tokens.scales` for quantized embeddings.
    EmbedTokensScales,
    /// `model.embed_tokens.biases` for quantized embeddings.
    EmbedTokensBiases,
    /// `model.norm.weight`.
    FinalNorm,
    /// `lm_head.weight` (+ `.scales` / `.biases`).
    LmHead,
}

/// Per-layer parameter slot. Synthetic fusions like
/// [`FusionKind::LayerBlock`] need to bind ~20 per-layer weights
/// (norms, QKV proj + bias, o_proj, gate/up/down) without enumerating
/// every (layer, weight) pair as its own enum variant. The runtime
/// resolves `(kind, layer_idx)` to a path like
/// `model.layers.{idx}.{relpath}` and looks it up in the model's
/// `tree_flatten` index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerParamKind {
    /// `input_layernorm.weight` — pre-attention RMSNorm.
    InputLayernorm,
    /// `post_attention_layernorm.weight` — pre-MLP RMSNorm.
    PostAttnLayernorm,
    /// `self_attn.q_proj.weight` (packed `uint32` for quantized).
    QProjWeight,
    /// `self_attn.q_proj.scales` — quantized only.
    QProjScales,
    /// `self_attn.q_proj.biases` — quantized only.
    QProjBiases,
    /// `self_attn.q_proj.bias` — Qwen adds bias to QKV (orthogonal to quant).
    QProjBiasVec,
    KProjWeight,
    KProjScales,
    KProjBiases,
    KProjBiasVec,
    VProjWeight,
    VProjScales,
    VProjBiases,
    VProjBiasVec,
    OProjWeight,
    OProjScales,
    OProjBiases,
    GateProjWeight,
    GateProjScales,
    GateProjBiases,
    UpProjWeight,
    UpProjScales,
    UpProjBiases,
    DownProjWeight,
    DownProjScales,
    DownProjBiases,
}

impl LayerParamKind {
    /// Relative path within `model.layers.{idx}.` for runtime resolution.
    pub fn relpath(self) -> &'static str {
        use LayerParamKind::*;
        match self {
            InputLayernorm => "input_layernorm.weight",
            PostAttnLayernorm => "post_attention_layernorm.weight",
            QProjWeight => "self_attn.q_proj.weight",
            QProjScales => "self_attn.q_proj.scales",
            QProjBiases => "self_attn.q_proj.biases",
            QProjBiasVec => "self_attn.q_proj.bias",
            KProjWeight => "self_attn.k_proj.weight",
            KProjScales => "self_attn.k_proj.scales",
            KProjBiases => "self_attn.k_proj.biases",
            KProjBiasVec => "self_attn.k_proj.bias",
            VProjWeight => "self_attn.v_proj.weight",
            VProjScales => "self_attn.v_proj.scales",
            VProjBiases => "self_attn.v_proj.biases",
            VProjBiasVec => "self_attn.v_proj.bias",
            OProjWeight => "self_attn.o_proj.weight",
            OProjScales => "self_attn.o_proj.scales",
            OProjBiases => "self_attn.o_proj.biases",
            GateProjWeight => "mlp.gate_proj.weight",
            GateProjScales => "mlp.gate_proj.scales",
            GateProjBiases => "mlp.gate_proj.biases",
            UpProjWeight => "mlp.up_proj.weight",
            UpProjScales => "mlp.up_proj.scales",
            UpProjBiases => "mlp.up_proj.biases",
            DownProjWeight => "mlp.down_proj.weight",
            DownProjScales => "mlp.down_proj.scales",
            DownProjBiases => "mlp.down_proj.biases",
        }
    }
}

/// Extra slot the runtime needs to bind beyond the fusion's IR
/// inputs/outputs. Trace-bound patterns may add a handful of uniform
/// slots; synthetic patterns (no `op_ids`) carry their whole
/// signature here.
#[derive(Debug, Clone, Copy)]
pub struct ExtraBinding {
    /// Buffer index in the emitted MSL function signature.
    pub slot: u32,
    /// What the host should bind into that slot.
    pub role: SlotRole,
}

/// Per-fusion extra-slot list. Trace-bound fusions return an empty
/// vec unless the kernel author added uniform/cache slots beyond IR
/// bindings. Synthetic fusions return the whole signature.
pub fn extra_bindings_for(fusion: &Fusion, meta: &ModelMeta) -> Vec<ExtraBinding> {
    use GlobalScratchKind::*;
    use LayerScratchKind::*;
    use ParamKind::*;
    match fusion.kind {
        FusionKind::FusedChain => {
            if let Some(slot) = fused_chain_lm_head_weight_slot(fusion) {
                vec![ExtraBinding {
                    slot,
                    role: SlotRole::Param(LmHead),
                }]
            } else {
                Vec::new()
            }
        }
        FusionKind::RmsNormQkv => {
            if !matches!(
                fusion.attrs,
                FusionAttrs::RmsNormQkv {
                    decode_sinks: Some(_),
                    ..
                }
            ) {
                return Vec::new();
            }
            let base = (fusion.inputs.len() + fusion.outputs.len()) as u32;
            vec![
                ExtraBinding {
                    slot: base,
                    role: SlotRole::LayerCacheK,
                },
                ExtraBinding {
                    slot: base + 1,
                    role: SlotRole::LayerCacheV,
                },
                ExtraBinding {
                    slot: base + 2,
                    role: SlotRole::LayerOffset,
                },
                ExtraBinding {
                    slot: base + 3,
                    role: SlotRole::LayerCapacity,
                },
            ]
        }
        // Trace-bound `SdpaDecode`. K/V always read the live cache.
        // Q normally reads the compact semantic q_buf and the SDPA
        // kernel applies its scale. MLA-style traces, however, carry
        // an already-scaled Q tensor in the direct task input
        // (scale == 1); leave slot 0 unbound there so the runtime
        // passes that trace-layout scratch instead.
        FusionKind::SdpaDecode => {
            let direct_scaled_q = matches!(
                fusion.attrs,
                FusionAttrs::SdpaDecode { scale, .. } if (scale - 1.0).abs() < f32::EPSILON
            ) && !fusion.inputs.is_empty();
            let mut v = Vec::new();
            if !direct_scaled_q {
                v.push(ExtraBinding {
                    slot: 0,
                    role: SlotRole::LayerScratch(LayerScratchKind::QBuf),
                });
            }
            v.extend([
                ExtraBinding {
                    slot: 1,
                    role: SlotRole::LayerCacheK,
                },
                ExtraBinding {
                    slot: 2,
                    role: SlotRole::LayerCacheV,
                },
                ExtraBinding {
                    slot: 3,
                    role: SlotRole::LayerKvLen,
                },
                ExtraBinding {
                    slot: 4,
                    role: SlotRole::LayerCapacity,
                },
            ]);
            if sdpa_decode::flash_decoding_enabled() {
                // Slot 5 (out) is normally filled by `t["outputs"]`
                // append in dispatch.py, but adding extras at slots 6+
                // breaks that index assumption. Bind out explicitly so
                // the slot-5 hole gets filled before the appends.
                v.push(ExtraBinding {
                    slot: 5,
                    role: SlotRole::LayerScratch(LayerScratchKind::SdpaBuf),
                });
                v.push(ExtraBinding {
                    slot: 6,
                    role: SlotRole::GlobalScratch(GlobalScratchKind::SdpaPartials),
                });
                v.push(ExtraBinding {
                    slot: 7,
                    role: SlotRole::GlobalScratch(GlobalScratchKind::SdpaCounter),
                });
            }
            v
        }
        // `LinearResidual`'s o_proj variant needs its `x` slot
        // overridden to `LayerScratch(SdpaBuf)` because the trace
        // routes SDPA's output through a transpose+reshape we skip
        // in the decode pipeline. down_proj's chain is correct
        // through scratch ids (it reads the gate-up output
        // directly). We can't disambiguate "o_proj vs down_proj"
        // from `FusionKind` alone — the Python hook tracks call
        // order per layer and patches o_proj's slot 0 there.
        // Synthetic — full signature lives here.
        // Dense: (embed_table, token_id) → (x_in_scratch)
        // Quant: (packed, scales, biases, token_id) → (x_in_scratch)
        FusionKind::EmbedLookupDecode => {
            if meta.quant_bits.is_some() && meta.quant_group_size.is_some() {
                vec![
                    ExtraBinding {
                        slot: 0,
                        role: SlotRole::Param(EmbedTokens),
                    },
                    ExtraBinding {
                        slot: 1,
                        role: SlotRole::Param(EmbedTokensScales),
                    },
                    ExtraBinding {
                        slot: 2,
                        role: SlotRole::Param(EmbedTokensBiases),
                    },
                    ExtraBinding {
                        slot: 3,
                        role: SlotRole::TokenIdIn,
                    },
                    ExtraBinding {
                        slot: 4,
                        role: SlotRole::GlobalScratch(XIn),
                    },
                ]
            } else {
                vec![
                    ExtraBinding {
                        slot: 0,
                        role: SlotRole::Param(EmbedTokens),
                    },
                    ExtraBinding {
                        slot: 1,
                        role: SlotRole::TokenIdIn,
                    },
                    ExtraBinding {
                        slot: 2,
                        role: SlotRole::GlobalScratch(XIn),
                    },
                ]
            }
        }
        // (q_buf, offset) → ()  — q_buf rotated in place
        FusionKind::RopeQDecode => vec![
            ExtraBinding {
                slot: 0,
                role: SlotRole::LayerScratch(QBuf),
            },
            ExtraBinding {
                slot: 1,
                role: SlotRole::LayerOffset,
            },
        ],
        // (k_buf, v_buf, cache_k, cache_v, offset, capacity) → ()
        FusionKind::RopeKvCacheStoreDecode => vec![
            ExtraBinding {
                slot: 0,
                role: SlotRole::LayerScratch(KBuf),
            },
            ExtraBinding {
                slot: 1,
                role: SlotRole::LayerScratch(VBuf),
            },
            ExtraBinding {
                slot: 2,
                role: SlotRole::LayerCacheK,
            },
            ExtraBinding {
                slot: 3,
                role: SlotRole::LayerCacheV,
            },
            ExtraBinding {
                slot: 4,
                role: SlotRole::LayerOffset,
            },
            ExtraBinding {
                slot: 5,
                role: SlotRole::LayerCapacity,
            },
        ],
        // (current_k, current_v, cache_k, cache_v, offset, capacity) → ()
        FusionKind::KvCacheStoreDecode => vec![
            ExtraBinding {
                slot: 2,
                role: SlotRole::LayerCacheK,
            },
            ExtraBinding {
                slot: 3,
                role: SlotRole::LayerCacheV,
            },
            ExtraBinding {
                slot: 4,
                role: SlotRole::LayerOffset,
            },
            ExtraBinding {
                slot: 5,
                role: SlotRole::LayerCapacity,
            },
        ],
        // (q_proj, q_norm_weight, offset) → (q_buf, gate_buf)
        FusionKind::RmsNormRopeQGateDecode => vec![ExtraBinding {
            slot: 2,
            role: SlotRole::LayerOffset,
        }],
        // (k_proj, v_proj, k_norm_weight, cache_k, cache_v, offset, capacity) → ()
        FusionKind::RmsNormRopeKvCacheStoreDecode => vec![
            ExtraBinding {
                slot: 3,
                role: SlotRole::LayerCacheK,
            },
            ExtraBinding {
                slot: 4,
                role: SlotRole::LayerCacheV,
            },
            ExtraBinding {
                slot: 5,
                role: SlotRole::LayerOffset,
            },
            ExtraBinding {
                slot: 6,
                role: SlotRole::LayerCapacity,
            },
        ],
        // (current_token_projection, arrays_cache[cache_index]) → ()
        FusionKind::ArraysCacheShiftAppendDecode => {
            let cache_index = match fusion.attrs {
                crate::fusion::ir::FusionAttrs::ArraysCacheShiftAppendDecode {
                    cache_index,
                    ..
                } => cache_index,
                _ => 0,
            };
            let mut out = vec![ExtraBinding {
                slot: 1,
                role: SlotRole::LayerArraysCache { index: cache_index },
            }];
            if crate::target::current_chip().active_m(8) > 1 {
                out.push(ExtraBinding {
                    slot: 2,
                    role: SlotRole::LayerArraysCacheRollback { index: cache_index },
                });
            }
            out
        }
        // (logits) → (next_token)
        FusionKind::ArgmaxLogitsDecode => vec![
            ExtraBinding {
                slot: 0,
                role: SlotRole::GlobalScratch(Logits),
            },
            ExtraBinding {
                slot: 1,
                role: SlotRole::NextTokenOut,
            },
        ],
        // (logits) → (tg_argmax scratch)
        FusionKind::ArgmaxLogitsScan => vec![
            ExtraBinding {
                slot: 0,
                role: SlotRole::GlobalScratch(Logits),
            },
            ExtraBinding {
                slot: 1,
                role: SlotRole::GlobalScratch(ArgmaxScratch),
            },
        ],
        // (tg_argmax scratch) → (next_token)
        FusionKind::ArgmaxLogitsFinalize => vec![
            ExtraBinding {
                slot: 0,
                role: SlotRole::GlobalScratch(ArgmaxScratch),
            },
            ExtraBinding {
                slot: 1,
                role: SlotRole::NextTokenOut,
            },
        ],
        // LayerBlock — synthetic per-layer megakernel composed from
        // RmsNormReduce + SdpaDecode + LinearResidual(o_proj). Slot
        // order matches the composer's role-deduplicated output (see
        // `codegen/kernels/layer_block.rs::emit`):
        //   0. x          read+write residual stream
        //   1. norm_w     pre-attention RmsNorm weights
        //   2. Q          per-layer Q scratch (upstream rmsnorm_qkv+RoPE)
        //   3. K          KV cache K pointer for this layer
        //   4. V          KV cache V pointer for this layer
        //   5. kv_len_buf single u32 uniform — current step's kv length
        //   6. capacity   single u32 uniform — cache capacity
        //   7. sdpa_out   intermediate SDPA output → o_proj input
        //   8. W_o_packed o_proj weights (packed int4 or bf16)
        //   9. W_o_s      o_proj scales (quant only)
        //  10. W_o_b      o_proj biases (quant only)
        FusionKind::LayerBlock => {
            let layer_idx = match fusion.attrs {
                crate::fusion::ir::FusionAttrs::LayerBlock { layer_idx, .. } => layer_idx,
                _ => return Vec::new(),
            };
            // Slot 0 (residual `x`, role x_input) is IR-bound via
            // fusion.inputs[0] (= layer's input residual tensor,
            // copied from absorbed LinearResidual(o_proj).inputs[-1]).
            // Last slot (y, role linres_y_o) is IR-bound via
            // fusion.outputs[0] (= post-attn residual tensor).
            // Dispatch.py order: inputs first → extras at explicit
            // slots (overwriting if collide) → outputs appended.
            // So extras populate slots 1..N (skipping 0 and trailing y).
            // Slot layout after GlobalBarrier piece inserted between
            // SdpaDecode and LinearResidual:
            //   0: x (residual)        — IR-bound via fusion.inputs[0]
            //   1: norm_w
            //   2: q_buf
            //   3: cache_k
            //   4: cache_v
            //   5: kv_len_buf
            //   6: capacity_buf
            //   7: sdpa_out (scratch)
            //   8: barrier_counters    — atomic_uint device buffer
            //   9: W_o_packed
            //  10: W_o_scales
            //  11: W_o_biases
            //  12: y_o                 — IR-bound via fusion.outputs[0]
            let mut v = vec![
                ExtraBinding {
                    slot: 1,
                    role: SlotRole::LayerParam {
                        kind: LayerParamKind::InputLayernorm,
                        layer_idx,
                    },
                },
                ExtraBinding {
                    slot: 2,
                    role: SlotRole::LayerScratch(LayerScratchKind::QBuf),
                },
                ExtraBinding {
                    slot: 3,
                    role: SlotRole::LayerCacheK,
                },
                ExtraBinding {
                    slot: 4,
                    role: SlotRole::LayerCacheV,
                },
                ExtraBinding {
                    slot: 5,
                    role: SlotRole::LayerKvLen,
                },
                ExtraBinding {
                    slot: 6,
                    role: SlotRole::LayerCapacity,
                },
                ExtraBinding {
                    slot: 7,
                    role: SlotRole::LayerScratch(LayerScratchKind::SdpaBuf),
                },
                ExtraBinding {
                    slot: 8,
                    role: SlotRole::BarrierCounters { layer_idx },
                },
                ExtraBinding {
                    slot: 9,
                    role: SlotRole::LayerParam {
                        kind: LayerParamKind::OProjWeight,
                        layer_idx,
                    },
                },
            ];
            // Quant-only o_proj scales/biases.
            let quant = match fusion.attrs {
                crate::fusion::ir::FusionAttrs::LayerBlock { quant, .. } => quant,
                _ => None,
            };
            if quant.is_some() {
                v.push(ExtraBinding {
                    slot: 10,
                    role: SlotRole::LayerParam {
                        kind: LayerParamKind::OProjScales,
                        layer_idx,
                    },
                });
                v.push(ExtraBinding {
                    slot: 11,
                    role: SlotRole::LayerParam {
                        kind: LayerParamKind::OProjBiases,
                        layer_idx,
                    },
                });
            }
            v
        }
        // Standalone `Single(RoPE)` from a surviving trace op. Slot 1
        // is the offset tensor (mlx captures it as a 1-elem int32 at
        // trace time — value would be the static prompt-cache offset
        // at trace, not the live decode position). Reroute to the
        // per-layer `LayerOffset` uniform so the kernel sees the
        // current cache.offset on every decode step. Used when the
        // synthetic `RopeQDecode` injection (build.rs) is conditional
        // and the trace's `Single(RoPE)` survives — Direction A.
        FusionKind::Single(ref name) if name == "RoPE" => vec![ExtraBinding {
            slot: 1,
            role: SlotRole::LayerOffset,
        }],
        _ => Vec::new(),
    }
}

fn fused_chain_lm_head_weight_slot(fusion: &Fusion) -> Option<u32> {
    let steps = match &fusion.attrs {
        FusionAttrs::FusedChain { steps, .. } => steps,
        _ => return None,
    };
    let first = steps.first()?;
    if first.op_name != "Matmul" || first.inputs.len() != 2 {
        return None;
    }
    let b_slot = match first.inputs[1] {
        ChainOperand::External { buffer_index } => buffer_index,
        ChainOperand::Register { .. } => return None,
    };
    let a_slot = match first.inputs[0] {
        ChainOperand::External { buffer_index } => buffer_index,
        ChainOperand::Register { .. } => return None,
    };
    let a_shape = fusion.inputs.get(a_slot as usize)?.shape.as_slice();
    let b_shape = fusion.inputs.get(b_slot as usize)?.shape.as_slice();
    let out_last = *fusion.outputs.first()?.shape.last()?;
    if a_shape.len() >= 2
        && b_shape.len() == 2
        && a_shape[a_shape.len() - 1] == b_shape[0]
        && b_shape[1] == out_last
        && out_last >= 10_000
        && fusion.outputs.len() == 1
        && fusion.outputs[0].shape.len() >= 2
    {
        Some(b_slot)
    } else {
        None
    }
}

/// Stable string keys for `SlotRole`, used in `task_graph.json` and
/// matched against in the Python hook. Kept centrally so a typo in
/// either side surfaces as a build error rather than silent
/// mis-binding.
pub fn slot_role_key(role: SlotRole) -> String {
    use GlobalScratchKind::*;
    use LayerScratchKind::*;
    use ParamKind::*;
    match role {
        SlotRole::TokenIdIn => "token_id_in".into(),
        SlotRole::NextTokenOut => "next_token_out".into(),
        SlotRole::LayerOffset => "layer_offset".into(),
        SlotRole::LayerCapacity => "layer_capacity".into(),
        SlotRole::LayerKvLen => "layer_kv_len".into(),
        SlotRole::LayerCacheK => "layer_cache_k".into(),
        SlotRole::LayerCacheV => "layer_cache_v".into(),
        SlotRole::LayerArraysCache { index } => format!("layer_arrays_cache.{index}"),
        SlotRole::LayerArraysCacheRollback { index } => {
            format!("layer_arrays_cache_rollback.{index}")
        }
        SlotRole::LayerScratch(QBuf) => "layer_scratch.q_buf".into(),
        SlotRole::LayerScratch(KBuf) => "layer_scratch.k_buf".into(),
        SlotRole::LayerScratch(VBuf) => "layer_scratch.v_buf".into(),
        SlotRole::LayerScratch(SdpaBuf) => "layer_scratch.sdpa_buf".into(),
        SlotRole::LayerScratch(HAttn) => "layer_scratch.h_attn".into(),
        SlotRole::LayerScratch(OutBuf) => "layer_scratch.out_buf".into(),
        SlotRole::LayerScratch(SiluUp) => "layer_scratch.silu_up".into(),
        SlotRole::GlobalScratch(XIn) => "global_scratch.x_in".into(),
        SlotRole::GlobalScratch(Logits) => "global_scratch.logits".into(),
        SlotRole::GlobalScratch(ArgmaxScratch) => "global_scratch.argmax_scratch".into(),
        SlotRole::GlobalScratch(GlobalScratchKind::SdpaPartials) => {
            "global_scratch.sdpa_partials".into()
        }
        SlotRole::GlobalScratch(GlobalScratchKind::SdpaCounter) => {
            "global_scratch.sdpa_counter".into()
        }
        SlotRole::Param(EmbedTokens) => "param.embed_tokens".into(),
        SlotRole::Param(EmbedTokensScales) => "param.embed_tokens.scales".into(),
        SlotRole::Param(EmbedTokensBiases) => "param.embed_tokens.biases".into(),
        SlotRole::Param(FinalNorm) => "param.final_norm".into(),
        SlotRole::Param(LmHead) => "param.lm_head".into(),
        SlotRole::LayerParam { kind, layer_idx } => {
            format!("layer_param.{layer_idx}.{}", kind.relpath())
        }
        SlotRole::BarrierCounters { layer_idx } => {
            format!("barrier_counters.{layer_idx}")
        }
    }
}

/// Full MSL function name as it appears in the emitted metallib.
/// This is what the runtime should ask Metal to bind to a PSO. The
/// names follow `fusekit_<kind>_f<id>` per emitter, with `f<id>` as a
/// stable suffix so two tasks pointing at the same fusion always
/// resolve to the same function symbol.
pub fn kernel_function_name(fusion: &Fusion) -> String {
    match fusion.kind {
        FusionKind::RmsNormQkv => rmsnorm_qkv::function_name(fusion),
        FusionKind::RmsNormGateUp => rmsnorm_gateup::function_name(fusion),
        FusionKind::RmsNormGateUpSwiglu => rmsnorm_gateup_swiglu::function_name(fusion),
        FusionKind::RmsNormGateUpSwigluDown => rmsnorm_gateup_swiglu_down::function_name(fusion),
        FusionKind::FinalRmsNormLmHead => final_rmsnorm_lm_head::function_name(fusion),
        FusionKind::RopeQDecode => rope_q_decode::function_name(fusion),
        FusionKind::RopeKvCacheStoreDecode => rope_kv_cache_store_decode::function_name(fusion),
        FusionKind::KvCacheStoreDecode => kv_cache_store_decode::function_name(fusion),
        FusionKind::RmsNormRopeQGateDecode => rmsnorm_rope_qgate_decode::function_name(fusion),
        FusionKind::RmsNormRopeKvCacheStoreDecode => {
            rmsnorm_rope_kv_cache_store_decode::function_name(fusion)
        }
        FusionKind::ArraysCacheShiftAppendDecode => {
            arrays_cache_shift_append_decode::function_name(fusion)
        }
        FusionKind::EmbedLookupDecode => embed_lookup_decode::function_name(fusion),
        FusionKind::ArgmaxLogitsDecode => argmax_logits_decode::function_name(fusion),
        FusionKind::ArgmaxLogitsScan => argmax_logits_scan::function_name(fusion),
        FusionKind::ArgmaxLogitsFinalize => argmax_logits_finalize::function_name(fusion),
        FusionKind::SdpaPrefill => sdpa_prefill::function_name(fusion),
        FusionKind::LinearResidual => linear_residual::function_name(fusion),
        FusionKind::SwigluActivation => swiglu_activation::function_name(fusion),
        FusionKind::SdpaDecode => sdpa_decode::function_name(fusion),
        FusionKind::Single(_) => single::function_name(fusion),
        FusionKind::QmvFanout => qmv_fanout::function_name(fusion),
        FusionKind::QmvSwiglu => qmv_swiglu::function_name(fusion),
        FusionKind::FusedChain => fused_chain::function_name(fusion),
        FusionKind::LayerBlock => layer_block::function_name(fusion),
        // Kinds without a landed emitter yet — fall back to a stable
        // name so the JSON consumer can still log them; the runtime
        // will currently fail to find them in the metallib.
        _ => format!("fusekit_unimpl_f{}", fusion.id),
    }
}

/// Default threadgroup size when a pattern doesn't pick its own.
/// 128 threads = 4 simdgroups, multiple of 32 (Metal's simdgroup
/// width) — enough warps for the partial-sum reduction every
/// emitter uses while leaving headroom for SMEM scratch.
pub const TG_THREADS: u64 = 128;

/// Preferred threadgroup width for a given fusion. Pattern kernels
/// were hand-tuned at specific widths (RmsNormQkv et al at 256, SDPA
/// at 64 to keep the per-warp K-axis loop short) so the partial-sum
/// reduction order is stable across runs. Using TG=128 for every
/// kernel shifted greedy-decoded tokens by float-precision drift —
/// not incorrect, but not byte-identical with the reference.
pub fn kernel_threadgroup(fusion: &Fusion) -> u64 {
    // Legacy fixed-TG path kept for callers that don't have a Target
    // yet (e.g., synthetic Single fallbacks). Prefer the budget-aware
    // `kernel_threadgroup_for` below when the chip target is known.
    match fusion.kind {
        FusionKind::SdpaDecode | FusionKind::SdpaPrefill => 128,
        // LayerBlock pins to 128 because the embedded SDPA piece needs
        // sgs_per_tg=4 for its hand-tuned partial-sum reduction order.
        FusionKind::LayerBlock => 128,
        FusionKind::RmsNormQkv
        | FusionKind::RmsNormGateUp
        | FusionKind::RmsNormGateUpSwiglu
        | FusionKind::RmsNormGateUpSwigluDown
        | FusionKind::LinearResidual
        | FusionKind::FinalRmsNormLmHead
        | FusionKind::EmbedLookupDecode
        | FusionKind::ArraysCacheShiftAppendDecode
        | FusionKind::ArgmaxLogitsDecode
        | FusionKind::ArgmaxLogitsScan
        | FusionKind::ArgmaxLogitsFinalize
        | FusionKind::RopeQDecode
        | FusionKind::KvCacheStoreDecode
        | FusionKind::RopeKvCacheStoreDecode => 256,
        // Match MLX rms_norm.metal for head_dim <= RMS_LOOPED_LIMIT:
        // RMS_N_READS=4 and 64 threads for head_dim=256.
        FusionKind::RmsNormRopeQGateDecode | FusionKind::RmsNormRopeKvCacheStoreDecode => 64,
        FusionKind::QmvFanout | FusionKind::QmvSwiglu => 64,
        _ => TG_THREADS,
    }
}

/// Budget-aware threadgroup-size pick. Reads the chip's
/// preferred/max simdgroup counts and SMEM budget, plus the fusion's
/// own SMEM estimate, and picks the largest TG that fits — capped at
/// the chip's max simdgroups so we don't sacrifice occupancy.
///
/// Returns thread count (multiple of 32). All Apple Silicon chips use
/// 32-thread simdgroups so `tg = sgs * 32`.
pub fn kernel_threadgroup_for(
    fusion: &Fusion,
    meta: &ModelMeta,
    target: &crate::target::Target,
) -> u64 {
    let b = &target.budgets;
    let smem_needed = crate::fusion::safety::estimate_smem_bytes(&fusion.kind, meta) as u32;

    // SMEM budget gives a hard ceiling on threadgroup count per SM,
    // which indirectly bounds usable TG size (bigger TG = fewer
    // resident, but per-TG SMEM is the same regardless of TG threads).
    // We don't shrink TG to fit SMEM — Apple gives 32 KiB regardless
    // of TG size. The shrink-on-SMEM case is for FUTURE pieces that
    // size SMEM proportional to TG threads.
    let _ = smem_needed;

    // SDPA has a hand-tuned dependency: TG=128 + sgs_per_tg=4 keeps
    // the partial-sum reduction order stable. Pin it regardless of
    // budget; revisiting would need a re-bench. `FUSEKIT_SDPA_TG_SGS=N`
    // lets us A/B smaller TGs against the legacy default without a
    // rebuild — sized for the flash path where Stage 3 only uses
    // HEAD_DIM threads (≤64) and bigger TGs just idle the rest.
    let sgs = match fusion.kind {
        FusionKind::SdpaDecode | FusionKind::LayerBlock => std::env::var("FUSEKIT_SDPA_TG_SGS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&n| (1..=8).contains(&n))
            .unwrap_or(4),
        FusionKind::FusedChain if fused_chain_prefers_dense_tg4(fusion) => {
            std::env::var("FUSEKIT_CHAIN_TG_SGS")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&n| (1..=16).contains(&n))
                .unwrap_or(4)
        }
        FusionKind::FusedChain
            if fused_chain_qmm_nax_dispatch_shape(fusion).is_some()
                || qmm_chain_nax_shape(fusion).is_some() =>
        {
            4
        }
        FusionKind::FusedChain if qmm_chain_mlx_qmv_fast_shape(fusion).is_some() => {
            let (k_dim, out_dim, active_rows, bits, _) =
                qmm_chain_mlx_qmv_fast_shape(fusion).expect("shape checked");
            qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim)
        }
        FusionKind::QmvFanout => {
            if let FusionAttrs::QmvFanout { targets } = &fusion.attrs {
                if qmv_fanout_exact_enabled() {
                    qmv_fanout_exact_simdgroups()
                } else if !targets.is_empty() {
                    qmv_fanout_targets_simdgroups(targets)
                } else {
                    2
                }
            } else {
                2
            }
        }
        FusionKind::QmvSwiglu => {
            if let FusionAttrs::QmvSwiglu { gate, .. } = &fusion.attrs {
                qmv_fast_simdgroups(gate.active_rows, gate.bits, gate.k_dim, gate.out_dim)
            } else {
                2
            }
        }
        FusionKind::Single(ref name)
            if name == "QuantizedMatmul" && qmm_mlx_qmmt_shape(fusion).is_some() =>
        {
            4
        }
        FusionKind::Single(ref name)
            if name == "QuantizedMatmul" && qmm_mlx_qmv_shape(fusion).is_some() =>
        {
            qmv_exact_simdgroups()
        }
        FusionKind::Single(ref name)
            if name == "QuantizedMatmul" && qmm_mlx_qmv_fast_shape(fusion).is_some() =>
        {
            let (k_dim, out_dim, active_rows, bits, _) =
                qmm_mlx_qmv_fast_shape(fusion).expect("shape checked");
            qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim)
        }
        FusionKind::LinearResidual if linear_residual_mlx_qmv_fast_shape(fusion).is_some() => {
            let (k_dim, out_dim, active_rows, bits, _) =
                linear_residual_mlx_qmv_fast_shape(fusion).expect("shape checked");
            linear_residual_qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim)
        }
        FusionKind::Single(ref name)
            if name == "QuantizedMatmul"
                && std::env::var("FUSEKIT_QMM_NAX").as_deref() != Ok("0") =>
        {
            4
        }
        _ => b
            .preferred_simdgroups_per_tg
            .min(b.max_simdgroups_per_tg)
            .max(2),
    };
    // Global override — `FUSEKIT_TG_SGS=N` (N ∈ [1..=16]) pins every
    // non-SDPA/LayerBlock kernel to N simdgroups per TG. Quick way to
    // sweep TG sizes without recompiling per change. Bypasses the
    // chip-budget pick above.
    let sgs = match fusion.kind {
        FusionKind::SdpaDecode | FusionKind::LayerBlock => sgs,
        _ => std::env::var("FUSEKIT_TG_SGS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&n| (1..=16).contains(&n))
            .unwrap_or(sgs),
    };
    // Debug override — `FUSEKIT_PIN_TG8=kind1,kind2,...` forces those
    // FusionKinds back to 8 SGs/TG even when the chip budget says 16+.
    // Used to binary-search which kernel breaks at TG=512.
    if let Ok(pin) = std::env::var("FUSEKIT_PIN_TG8") {
        let kind_name = match &fusion.kind {
            FusionKind::RmsNormQkv => "rmsnorm_qkv",
            FusionKind::RmsNormGateUp => "rmsnorm_gateup",
            FusionKind::RmsNormGateUpSwiglu => "rmsnorm_gateup_swiglu",
            FusionKind::RmsNormGateUpSwigluDown => "rmsnorm_gateup_swiglu_down",
            FusionKind::FinalRmsNormLmHead => "final_rmsnorm_lm_head",
            FusionKind::LinearResidual => "linear_residual",
            FusionKind::QmvSwiglu => "qmv_swiglu",
            FusionKind::RopeQDecode => "rope_q",
            FusionKind::RopeKvCacheStoreDecode => "rope_kv_cache_store",
            FusionKind::KvCacheStoreDecode => "kv_cache_store",
            FusionKind::RmsNormRopeQGateDecode => "rmsnorm_rope_qgate",
            FusionKind::RmsNormRopeKvCacheStoreDecode => "rmsnorm_rope_kv_cache_store",
            FusionKind::EmbedLookupDecode => "embed_lookup",
            FusionKind::ArgmaxLogitsDecode => "argmax",
            _ => "",
        };
        if pin.split(',').any(|s| s.trim() == kind_name) {
            return (8 * b.simd_width) as u64;
        }
    }
    (sgs * b.simd_width) as u64
}

fn fused_chain_prefers_dense_tg4(fusion: &Fusion) -> bool {
    let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
        return false;
    };
    let mut has_dense_heavy = false;
    for step in steps {
        match step.op_name.as_str() {
            "QuantizedMatmul" | "GatherQMM" | "QuantizedGatherQMM" => return false,
            "Matmul" | "AddMM" => has_dense_heavy = true,
            _ => {}
        }
    }
    has_dense_heavy
}

fn fused_chain_qmm_nax_dispatch_shape(fusion: &Fusion) -> Option<(u64, u64)> {
    let debug = std::env::var("FUSEKIT_DEBUG_QMM_CHAIN_NAX_ID")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .is_some_and(|id| id == fusion.id);
    macro_rules! reject {
        ($reason:expr) => {{
            if debug {
                eprintln!(
                    "[fusekit-qmm-chain-nax-dispatch] reject fusion={} reason={}",
                    fusion.id, $reason
                );
            }
            return None;
        }};
    }
    if !env_bool_on("FUSEKIT_QMM_CHAIN_NAX") {
        reject!("disabled");
    }
    let FusionAttrs::FusedChain { steps, .. } = &fusion.attrs else {
        reject!("not_fused_chain");
    };
    if steps.len() != 1 {
        reject!(format!("steps={}", steps.len()));
    }
    let step = &steps[0];
    if step.op_name != "QuantizedMatmul" || step.inputs.len() != 4 {
        reject!(format!("op={} inputs={}", step.op_name, step.inputs.len()));
    }
    let crate::graph::Attrs::Quantized { bits, group_size } = step.op_attrs else {
        reject!("attrs_not_quantized");
    };
    if !matches!((bits, group_size), (4, 64) | (8, 64)) {
        reject!(format!("bits={bits} group_size={group_size}"));
    }
    let Some(out) = fusion.outputs.first() else {
        reject!("missing_out");
    };
    let out_shape = out.shape.as_slice();
    let out_dim = out_shape.last().copied().unwrap_or(0).max(1) as u64;
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(out_shape)
        .max(1);
    if active_rows != 1 && (!env_bool_on("FUSEKIT_QMM_CHAIN_NAX_M") || out_dim < 64) {
        reject!(format!(
            "active_rows={active_rows} out_dim={out_dim} m_enabled={}",
            env_bool_on("FUSEKIT_QMM_CHAIN_NAX_M")
        ));
    }
    if debug {
        eprintln!(
            "[fusekit-qmm-chain-nax-dispatch] accept fusion={} out={} active_rows={} bits={} gs={}",
            fusion.id, out_dim, active_rows, bits, group_size
        );
    }
    Some((out_dim, active_rows))
}

/// Full 3-D threadgroup shape for kernels that use non-flat Metal
/// coordinates. Most fusekit kernels flatten onto X, but mlx
/// `CustomKernel` sources often rely on their captured `(x, y, z)`
/// threadgroup exactly.
pub fn kernel_threadgroup3_for(
    fusion: &Fusion,
    meta: &ModelMeta,
    target: &crate::target::Target,
) -> [u64; 3] {
    if let FusionAttrs::Single {
        op_attrs: crate::graph::Attrs::CustomKernel { threadgroup, .. },
        ..
    } = &fusion.attrs
    {
        return [
            threadgroup[0].max(1) as u64,
            threadgroup[1].max(1) as u64,
            threadgroup[2].max(1) as u64,
        ];
    }
    [kernel_threadgroup_for(fusion, meta, target), 1, 1]
}

/// Total threads to dispatch for `fusion`. Returned as a 3-tuple
/// because Metal's grid is 3-D, but every emitter we have flattens
/// work onto the X axis (Y=Z=1) — the kernels themselves decompose
/// the flat index back into row/output/head via constexpr divisions.
///
/// Each branch here mirrors the dispatch shape the matching emitter
/// assumes. When you teach a new emitter, add its rule here too —
/// otherwise the runtime will dispatch too few threads and produce
/// partially-filled output buffers.
/// Budget-aware grid sizing — same as `kernel_grid_threads` but uses
/// the chip's actual TG pick. Wrapper so we don't duplicate the
/// per-FusionKind dispatch math.
pub fn kernel_grid_threads_for(
    fusion: &Fusion,
    meta: &ModelMeta,
    target: &crate::target::Target,
) -> [u64; 3] {
    // The grid computation needs the TG size — temporarily swap by
    // calling the body with a target-derived tg. Implemented by
    // inlining the dispatch and using `kernel_threadgroup_for`.
    let tg = kernel_threadgroup_for(fusion, meta, target);
    kernel_grid_threads_impl(fusion, meta, tg)
}

pub fn kernel_grid_threads(fusion: &Fusion, meta: &ModelMeta) -> [u64; 3] {
    let tg = kernel_threadgroup(fusion);
    kernel_grid_threads_impl(fusion, meta, tg)
}

fn kernel_grid_threads_impl(fusion: &Fusion, meta: &ModelMeta, tg: u64) -> [u64; 3] {
    use crate::fusion::category::{categorize_name, OpCategory};
    use crate::fusion::ir::FusionAttrs;

    let elems = |shape: &[i32]| -> u64 {
        shape
            .iter()
            .map(|&d| d.max(0) as u64)
            .product::<u64>()
            .max(1)
    };
    // Decode-time L=1: collapse the sequence dimension out of every
    // batch-or-seq-leading shape before sizing the grid. Trace's
    // prefill L≥2 baked the sequence into the captured output shape,
    // and dispatching one threadgroup per "row" of that prefill-sized
    // output was producing 7× more idle TGs per generate step. The
    // per-row compute is L-independent (every emitter writes one row
    // and early-exits the rest), so collapsing here only changes
    // grid size, not values.
    let rows_and_last = |shape: &[i32]| -> (u64, u64) {
        if shape.is_empty() {
            return (1, 1);
        }
        let last = *shape.last().unwrap() as u64;
        let rows = crate::target::current_chip().active_rows_before_last(shape);
        (rows.max(1), last.max(1))
    };
    let ceil_div = |n: u64, d: u64| -> u64 { (n + d - 1) / d };
    let sgs_per_tg = tg / 32;
    let active_m_from_shape = |shape: Option<&[i32]>| -> u64 {
        shape
            .filter(|s| !s.is_empty())
            .map(|s| crate::target::current_chip().active_tokens_for_shape(s) as u64)
            .unwrap_or_else(|| crate::target::current_chip().active_m(8) as u64)
            .max(1)
    };
    let argmax_active_rows = || -> u64 {
        let default_active_rows = crate::target::current_chip().active_m(8).max(1);
        std::env::var("FUSEKIT_ARGMAX_ACTIVE_ROWS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(default_active_rows)
            .min(default_active_rows) as u64
    };

    match &fusion.kind {
        FusionKind::QmvFanout => {
            let FusionAttrs::QmvFanout { targets } = &fusion.attrs else {
                return [TG_THREADS, 1, 1];
            };
            let Some(first) = targets.first() else {
                return [TG_THREADS, 1, 1];
            };
            let token_rows = qmv_fanout_token_rows(targets) as u64;
            let qmv_sgs = if qmv_fanout_exact_enabled() {
                qmv_fanout_exact_simdgroups() as u64
            } else {
                qmv_fanout_targets_simdgroups(targets) as u64
            };
            let y_tiles = qmv_fanout_y_tiles(targets);
            [
                ceil_div(first.active_rows.max(1), token_rows.max(1))
                    .saturating_mul(qmv_sgs.saturating_mul(32)),
                y_tiles.max(1),
                1,
            ]
        }
        FusionKind::QmvSwiglu => {
            let FusionAttrs::QmvSwiglu { gate, .. } = &fusion.attrs else {
                return [TG_THREADS, 1, 1];
            };
            let token_rows =
                qmv_fast_token_rows(gate.active_rows, gate.bits, gate.k_dim, gate.out_dim) as u64;
            let qmv_sgs =
                qmv_fast_simdgroups(gate.active_rows, gate.bits, gate.k_dim, gate.out_dim) as u64;
            let cols_per_tg = qmv_fast_cols_per_threadgroup(
                gate.active_rows,
                gate.bits,
                gate.k_dim,
                gate.out_dim,
            ) as u64;
            let y_tiles = ceil_div(gate.out_dim as u64, cols_per_tg.max(1));
            [
                ceil_div(gate.active_rows.max(1), token_rows.max(1))
                    .saturating_mul(qmv_sgs.saturating_mul(32)),
                y_tiles.max(1),
                1,
            ]
        }
        FusionKind::FusedChain => {
            let steps = match &fusion.attrs {
                FusionAttrs::FusedChain { steps, .. } => steps,
                _ => return [TG_THREADS, 1, 1],
            };
            if steps.is_empty() {
                return [TG_THREADS, 1, 1];
            }
            if env_bool_on("FUSEKIT_QMM_CHAIN_NAX") && steps.len() == 1 {
                let step = &steps[0];
                if step.op_name == "QuantizedMatmul" && step.inputs.len() == 4 {
                    if let crate::graph::Attrs::Quantized { bits, group_size } = step.op_attrs {
                        if matches!((bits, group_size), (4, 64) | (8, 64)) {
                            if let (
                                ChainOperand::External {
                                    buffer_index: x_buf,
                                },
                                ChainOperand::External {
                                    buffer_index: w_buf,
                                },
                            ) = (step.inputs[0], step.inputs[1])
                            {
                                if let (Some(x), Some(w), Some(out)) = (
                                    fusion.inputs.get(x_buf as usize),
                                    fusion.inputs.get(w_buf as usize),
                                    fusion.outputs.first(),
                                ) {
                                    let x_shape = x.shape.as_slice();
                                    let w_shape = w.shape.as_slice();
                                    let out_shape = out.shape.as_slice();
                                    if !x_shape.is_empty()
                                        && w_shape.len() == 2
                                        && !out_shape.is_empty()
                                    {
                                        let out_dim = w_shape[0].max(1) as u64;
                                        if out_shape.last().copied().unwrap_or(0).max(1) as u64
                                            == out_dim
                                        {
                                            let active_rows = crate::target::current_chip()
                                                .active_rows_before_last(out_shape)
                                                .max(1);
                                            if active_rows == 1
                                                || (env_bool_on("FUSEKIT_QMM_CHAIN_NAX_M")
                                                    && out_dim >= 64)
                                            {
                                                return [
                                                    ceil_div(out_dim, 64) * 128,
                                                    ceil_div(active_rows, 64),
                                                    1,
                                                ];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let Some((_k_dim, out_dim, active_rows, _bits, _group_size)) =
                qmm_chain_nax_shape(fusion)
            {
                return [
                    ceil_div(out_dim as u64, 64) * 128,
                    ceil_div(active_rows, 64),
                    1,
                ];
            }
            if let Some((out_dim, active_rows)) = fused_chain_qmm_nax_dispatch_shape(fusion) {
                return [ceil_div(out_dim, 64) * 128, ceil_div(active_rows, 64), 1];
            }
            if let Some((k_dim, out_dim, active_rows, bits, _group_size)) =
                qmm_chain_mlx_qmv_fast_shape(fusion)
            {
                let token_rows = qmv_fast_token_rows(active_rows, bits, k_dim, out_dim) as u64;
                let qmv_sgs = qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim) as u64;
                let cols_per_tg =
                    qmv_fast_cols_per_threadgroup(active_rows, bits, k_dim, out_dim) as u64;
                return [
                    ceil_div(active_rows, token_rows.max(1))
                        .saturating_mul(qmv_sgs.saturating_mul(32)),
                    ceil_div(out_dim as u64, cols_per_tg.max(1)),
                    1,
                ];
            }
            let first_cat = categorize_name(&steps[0].op_name);
            let heavy_count = steps
                .iter()
                .skip(1)
                .filter(|s| categorize_name(&s.op_name) == OpCategory::Heavy)
                .count();
            match first_cat {
                OpCategory::Reduction if heavy_count >= 1 => {
                    // SMEM-cached normed row × per-simdgroup heavy
                    // GEMV. Each simdgroup owns `rows_per_sg`
                    // consecutive output columns when the chip target
                    // allows it (M3+) and every Heavy step's out_dim
                    // is divisible by that factor. This mirrors the
                    // gating in `emit_reduction_fanout_chain` so the
                    // dispatch shape matches the kernel's expectation.
                    let (rows, _) = rows_and_last(&steps[0].output_shape);
                    let heavy_dims: Vec<u64> = steps
                        .iter()
                        .skip(1)
                        .filter(|s| categorize_name(&s.op_name) == OpCategory::Heavy)
                        .map(|s| *s.output_shape.last().unwrap_or(&1) as u64)
                        .collect();
                    let swiglu_absorbed = steps.len() == 6
                        && fusion.outputs.len() == 1
                        && steps[0].op_name == "RMSNorm"
                        && categorize_name(&steps[1].op_name) == OpCategory::Heavy
                        && categorize_name(&steps[2].op_name) == OpCategory::Heavy
                        && steps[3].op_name == "Sigmoid"
                        && steps[4].op_name == "Multiply"
                        && steps[5].op_name == "Multiply";
                    let out_total: u64 = if swiglu_absorbed {
                        heavy_dims.first().copied().unwrap_or(1)
                    } else {
                        heavy_dims.iter().sum::<u64>().max(1)
                    };
                    let hidden_dim =
                        steps[0].output_shape.last().copied().unwrap_or(1).max(1) as u32;
                    let max_out = heavy_dims.iter().copied().max().unwrap_or(1) as u32;
                    let all_dense = steps
                        .iter()
                        .skip(1)
                        .filter(|s| categorize_name(&s.op_name) == OpCategory::Heavy)
                        .all(|s| s.op_name == "Matmul" || s.op_name == "AddMM");
                    let chip_rows = if all_dense {
                        crate::target::current_chip()
                            .dense_gemv_rows_per_simdgroup(hidden_dim, max_out)
                    } else {
                        crate::target::current_chip().gemv_rows_per_simdgroup()
                    } as u64;
                    let all_aligned = heavy_dims
                        .iter()
                        .all(|&d| d >= chip_rows && d % chip_rows == 0);
                    let rows_per_sg = if chip_rows > 1 && all_aligned {
                        chip_rows
                    } else {
                        1
                    };
                    let sg_chunk = sgs_per_tg.saturating_mul(rows_per_sg).max(1);
                    let exact_tgs = ceil_div(out_total, sg_chunk);
                    // LM-head-shape cap: when there's one Heavy and the
                    // output dim is huge (vocab-scale), the per-TG
                    // RMSNorm becomes redundant work amortised across
                    // very few outputs each. Cap the per-row grid at 64
                    // TGs and let the emit's stride loop cover the rest.
                    // Matches the empirical sweet spot from the legacy
                    // FinalRmsNormLmHead pattern. Triggered only when
                    // rows=1 (decode) and out_total ≥ 10000.
                    // Grid cap: amortise the redundant per-TG RMSNorm
                    // across more outputs via the emit's stride loop.
                    // Empirically only LM-head (out_total ≥ 10K, single
                    // heavy) benefits — fan-out chains (Q/K/V, gate/up)
                    // regress under cap because Apple's scheduler fills
                    // the pipeline better with the larger TG count.
                    // Stride loop in the emit STILL runs (works for any
                    // grid size); cap just disabled for fan-out.
                    // `FUSEKIT_LMHEAD_CAP` / `FUSEKIT_FANOUT_CAP`
                    // override for sweeps on other chips/models.
                    let cap_lmhead: u64 = std::env::var("FUSEKIT_LMHEAD_CAP")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or(u64::MAX);
                    let cap_fanout: u64 = std::env::var("FUSEKIT_FANOUT_CAP")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or(u64::MAX); // disabled by default
                    let token_m_rows = if rows > 1
                        && rows_per_sg == 1
                        && steps[0].op_name == "RMSNorm"
                        && heavy_count >= 2
                        && mtp_token_m_reduction_fanout_enabled()
                    {
                        let mut qmm_only = true;
                        let mut best = u32::MAX;
                        for s in steps
                            .iter()
                            .skip(1)
                            .filter(|s| categorize_name(&s.op_name) == OpCategory::Heavy)
                        {
                            if s.op_name != "QuantizedMatmul" {
                                qmm_only = false;
                                break;
                            }
                            let (bits, _) = match s.op_attrs {
                                crate::graph::Attrs::Quantized { bits, group_size } => {
                                    (bits, group_size)
                                }
                                _ => (4u32, 64u32),
                            };
                            let out_dim = s.output_shape.last().copied().unwrap_or(1).max(1) as u32;
                            best =
                                best.min(crate::target::current_chip().qgemm_m_rows_per_simdgroup(
                                    rows as u32,
                                    bits,
                                    hidden_dim,
                                    out_dim,
                                ));
                        }
                        if qmm_only {
                            best.max(1)
                        } else {
                            1
                        }
                    } else {
                        1
                    };
                    let tgs_per_row = if rows == 1 && heavy_count == 1 && out_total >= 10000 {
                        exact_tgs.min(cap_lmhead)
                    } else if rows == 1 && heavy_count >= 2 && exact_tgs > cap_fanout {
                        exact_tgs.min(cap_fanout)
                    } else {
                        exact_tgs
                    };
                    let row_groups = if token_m_rows > 1 {
                        ceil_div(rows.max(1), token_m_rows as u64)
                    } else {
                        rows.max(1)
                    };
                    [row_groups * tgs_per_row * tg, 1, 1]
                }
                OpCategory::Reduction => {
                    // Pure RMSNorm + pointwise tail — 1 TG per active
                    // semantic row. Decode collapses batch/sequence,
                    // but preserves axes after sequence such as
                    // Qwen3Next's `[B,S,H,D]` head rows.
                    let shape = &steps[0].output_shape;
                    let (rows, _) = rows_and_last(shape);
                    [rows * tg, 1, 1]
                }
                OpCategory::Heavy => {
                    // Heavy-first chain: each simdgroup owns
                    // `rows_per_sg` consecutive output columns when
                    // the chip target allows (chip_rows > 1) and
                    // the output's last dim is R-divisible. Mirrors
                    // emit_qmm_chain / emit_matmul_chain's gating.
                    // Size from the same active-M policy as the emitter
                    // so decode (M=1), prefill, and MTP patch modes all
                    // dispatch enough work without over-spawning the
                    // captured trace sequence length.
                    let shape = &fusion.outputs[0].shape;
                    let last_dim = shape.last().copied().unwrap_or(0) as u64;
                    let active_rows = crate::target::current_chip().active_rows_before_last(shape);
                    let k_dim = steps[0]
                        .inputs
                        .first()
                        .and_then(|op| match op {
                            ChainOperand::External { buffer_index } => fusion
                                .inputs
                                .get(*buffer_index as usize)
                                .and_then(|input| input.shape.last().copied()),
                            ChainOperand::Register { .. } => None,
                        })
                        .unwrap_or(1)
                        .max(1) as u32;
                    let chip_rows = if steps[0].op_name == "Matmul" || steps[0].op_name == "AddMM" {
                        crate::target::current_chip()
                            .dense_gemv_rows_per_simdgroup(k_dim, last_dim as u32)
                    } else {
                        crate::target::current_chip().gemv_rows_per_simdgroup()
                    } as u64;
                    let rows_per_sg =
                        if chip_rows > 1 && last_dim >= chip_rows && last_dim % chip_rows == 0 {
                            chip_rows
                        } else {
                            1
                        };
                    let token_m_rows = if steps[0].op_name == "QuantizedMatmul"
                        && active_rows > 1
                        && rows_per_sg == 1
                        && mtp_token_m_qgemv_enabled()
                    {
                        let (bits, _) = match steps[0].op_attrs {
                            crate::graph::Attrs::Quantized { bits, group_size } => {
                                (bits, group_size)
                            }
                            _ => (4u32, 64u32),
                        };
                        crate::target::current_chip().qgemm_m_rows_per_simdgroup(
                            active_rows as u32,
                            bits,
                            k_dim,
                            last_dim as u32,
                        ) as u64
                    } else {
                        1
                    };
                    let active_row_groups = if token_m_rows > 1 {
                        ceil_div(active_rows.max(1), token_m_rows)
                    } else {
                        active_rows.max(1)
                    };
                    let qgemm_cols = if token_m_rows > 1 {
                        let (bits, _) = match steps[0].op_attrs {
                            crate::graph::Attrs::Quantized { bits, group_size } => {
                                (bits, group_size)
                            }
                            _ => (4u32, 64u32),
                        };
                        crate::target::current_chip().qgemm_output_cols_per_simdgroup(
                            active_rows as u32,
                            bits,
                            k_dim,
                            last_dim as u32,
                        ) as u64
                    } else {
                        rows_per_sg
                    };
                    [
                        active_row_groups
                            * ceil_div(
                                last_dim.max(1),
                                sgs_per_tg.saturating_mul(qgemm_cols).max(1),
                            )
                            * tg,
                        1,
                        1,
                    ]
                }
                _ => {
                    // Pointwise/view chains follow the same active-M
                    // policy as their emitter. This preserves
                    // post-sequence axes such as heads or MoE routes.
                    let shape = &fusion.outputs[0].shape;
                    let n = crate::target::current_chip().active_elems(shape);
                    [ceil_div(n, tg) * tg, 1, 1]
                }
            }
        }
        FusionKind::Single(_) => {
            // Most Single kernels use the same 1-thread-per-element
            // pointwise layout; the ones that don't (Quantized,
            // CustomKernel) land below.
            use crate::fusion::ir::FusionAttrs;
            if let FusionAttrs::Single {
                op_attrs:
                    crate::graph::Attrs::CustomKernel {
                        grid, threadgroup, ..
                    },
                ..
            } = &fusion.attrs
            {
                // mlx `metal_kernel(grid=..., threadgroup=...)` uses
                // `grid` as dispatchThreads' total thread grid, not as
                // threadgroup count. Preserve it exactly; multiplying
                // by threadgroup over-dispatches kernels such as
                // gated_delta by 4x on Y and corrupts state.
                let _ = threadgroup;
                return [
                    grid[0].max(1) as u64,
                    grid[1].max(1) as u64,
                    grid[2].max(1) as u64,
                ];
            }
            if let FusionKind::Single(name) = &fusion.kind {
                if name == "Split" {
                    let n: u64 = fusion.outputs.iter().map(|o| elems(&o.shape)).sum();
                    return [ceil_div(n.max(1), tg) * tg, 1, 1];
                }
                if name == "ArgPartition" || name == "Argpartition" {
                    let shape = &fusion.outputs[0].shape;
                    let row_n = shape.last().copied().unwrap_or(1).max(1) as u64;
                    let rows = if shape.len() >= 3 {
                        crate::target::current_chip().active_rows_before_last(shape)
                    } else {
                        (elems(shape).max(1) / row_n).max(1)
                    };
                    let bitonic_enabled =
                        std::env::var("FUSEKIT_ARGPARTITION_BITONIC").as_deref() != Ok("0");
                    if bitonic_enabled && row_n.is_power_of_two() && row_n <= 256 {
                        return [rows * tg, 1, 1];
                    }
                    return [ceil_div(rows.saturating_mul(row_n).max(1), tg) * tg, 1, 1];
                }
                if name == "QuantizedMatmul" {
                    if let Some((_k_dim, out_dim, active_rows, _bits, _group_size)) =
                        qmm_mlx_qmmt_shape(fusion)
                    {
                        return [
                            ceil_div(out_dim as u64, 32) * 128,
                            ceil_div(active_rows, 32),
                            1,
                        ];
                    }
                    if let Some((_k_dim, out_dim, active_rows, _bits, _group_size)) =
                        qmm_mlx_qmv_shape(fusion)
                    {
                        return [
                            active_rows.saturating_mul(qmv_exact_simdgroups() as u64 * 32),
                            ceil_div(out_dim as u64, qmv_exact_cols_per_threadgroup() as u64),
                            1,
                        ];
                    }
                    if let Some((k_dim, out_dim, active_rows, bits, _group_size)) =
                        qmm_mlx_qmv_fast_shape(fusion)
                    {
                        let token_rows =
                            qmv_fast_token_rows(active_rows, bits, k_dim, out_dim) as u64;
                        let qmv_sgs = qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim) as u64;
                        let cols_per_tg =
                            qmv_fast_cols_per_threadgroup(active_rows, bits, k_dim, out_dim) as u64;
                        return [
                            ceil_div(active_rows, token_rows.max(1))
                                .saturating_mul(qmv_sgs.saturating_mul(32)),
                            ceil_div(out_dim as u64, cols_per_tg.max(1)),
                            1,
                        ];
                    }
                    if let Some((_k_dim, out_dim, active_rows, _bits, _group_size)) =
                        qmm_nax_shape(fusion)
                    {
                        return [
                            ceil_div(out_dim as u64, 64) * 128,
                            ceil_div(active_rows, 64),
                            1,
                        ];
                    }
                    // The Single QMM emitter reuses the FusedChain
                    // simdgroup-cooperative GEMV path: one simdgroup
                    // owns one or more output columns, not one thread
                    // per output element. Size from active M rows x
                    // output dimension so decode, prefill, and MTP
                    // patch modes all agree with the emitter.
                    let last_dim = fusion.outputs[0].shape.last().copied().unwrap_or(0) as u64;
                    let active_rows = crate::target::current_chip()
                        .active_rows_before_last(&fusion.outputs[0].shape);
                    let chip_rows = crate::target::current_chip().gemv_rows_per_simdgroup() as u64;
                    let (bits, _) = match &fusion.attrs {
                        FusionAttrs::Single { op_attrs, .. } => match op_attrs {
                            crate::graph::Attrs::Quantized { bits, group_size } => {
                                (*bits, *group_size)
                            }
                            _ => (4u32, 64u32),
                        },
                        _ => (4u32, 64u32),
                    };
                    let allow_output_row_coalesce = matches!(bits, 4 | 8);
                    let rows_per_sg = if allow_output_row_coalesce
                        && chip_rows > 1
                        && last_dim >= chip_rows
                        && last_dim % chip_rows == 0
                    {
                        chip_rows
                    } else {
                        1
                    };
                    let token_m_rows =
                        if active_rows > 1 && rows_per_sg == 1 && mtp_token_m_qgemv_enabled() {
                            let k_dim = fusion
                                .inputs
                                .first()
                                .and_then(|input| input.shape.last().copied())
                                .unwrap_or(1)
                                .max(1) as u32;
                            crate::target::current_chip().qgemm_m_rows_per_simdgroup(
                                active_rows as u32,
                                bits,
                                k_dim,
                                last_dim as u32,
                            ) as u64
                        } else {
                            1
                        };
                    let active_row_groups = if token_m_rows > 1 {
                        ceil_div(active_rows.max(1), token_m_rows)
                    } else {
                        active_rows.max(1)
                    };
                    let qgemm_cols = if token_m_rows > 1 {
                        let k_dim = fusion
                            .inputs
                            .first()
                            .and_then(|input| input.shape.last().copied())
                            .unwrap_or(1)
                            .max(1) as u32;
                        crate::target::current_chip().qgemm_output_cols_per_simdgroup(
                            active_rows as u32,
                            bits,
                            k_dim,
                            last_dim as u32,
                        ) as u64
                    } else {
                        rows_per_sg
                    };
                    return [
                        active_row_groups
                            * ceil_div(
                                last_dim.max(1),
                                sgs_per_tg.saturating_mul(qgemm_cols).max(1),
                            )
                            * tg,
                        1,
                        1,
                    ];
                }
                if name == "GatherQMM" {
                    // Routed expert QMM uses the same simdgroup-coop
                    // dot-product layout as regular QMM, but the
                    // traced output is `[B, trace_T, routes..., O]`
                    // while decode consumers only read the active
                    // token row. Size from the post-sequence route
                    // suffix, not from `elems(shape)`, or DeepSeek
                    // MoE over-computes by trace_T.
                    let shape = &fusion.outputs[0].shape;
                    let last_dim = shape.last().copied().unwrap_or(1).max(1) as u64;
                    let active_tokens =
                        crate::target::current_chip().active_tokens_for_shape(shape) as u64;
                    let routes_per_token = if shape.len() >= 3 {
                        shape[2..shape.len().saturating_sub(1)]
                            .iter()
                            .map(|&d| d.max(0) as u64)
                            .product::<u64>()
                            .max(1)
                    } else {
                        (elems(shape).max(1) / last_dim).max(1)
                    };
                    let routes = active_tokens.saturating_mul(routes_per_token).max(1);
                    let chip_rows =
                        crate::target::current_chip().gatherqmm_rows_per_simdgroup() as u64;
                    let rows_per_sg =
                        if chip_rows > 1 && last_dim >= chip_rows && last_dim % chip_rows == 0 {
                            chip_rows
                        } else {
                            1
                        };
                    let row_blocks = ceil_div(last_dim, rows_per_sg.max(1));
                    let total_sgs = routes.saturating_mul(row_blocks).max(1);
                    return [ceil_div(total_sgs, sgs_per_tg.max(1)) * tg, 1, 1];
                }
                if name == "WeightedRouteReduce" {
                    let n = crate::target::current_chip().active_elems(&fusion.outputs[0].shape);
                    return [ceil_div(n.max(1), tg) * tg, 1, 1];
                }
                if name == "RMSNorm" {
                    // Standalone reductions index `gid` as the row.
                    // Preserve the pass-wide active-M policy while
                    // keeping post-sequence row axes such as heads.
                    let shape = &fusion.outputs[0].shape;
                    let rows = crate::target::current_chip().active_rows_before_last(shape);
                    return [rows * tg, 1, 1];
                }
            }
            let n = elems(&fusion.outputs[0].shape);
            [ceil_div(n, tg) * tg, 1, 1]
        }
        // Pattern-matched kernels. Each pattern emitter encodes its
        // own dispatch shape; the numbers below mirror those.
        FusionKind::RmsNormQkv | FusionKind::RmsNormGateUp | FusionKind::RmsNormGateUpSwiglu => {
            // 1 TG per row × ceil(max_out / sgs_per_tg) TGs per row.
            // The matched output covers (Q,K,V) or (gate,up); we use
            // the total output dim as the cooperative work.
            let total_out: u64 = if matches!(
                fusion.attrs,
                FusionAttrs::RmsNormQkv {
                    decode_sinks: Some(_),
                    ..
                }
            ) && matches!(fusion.kind, FusionKind::RmsNormQkv)
            {
                let q = fusion
                    .outputs
                    .iter()
                    .find(|o| o.role == "q")
                    .and_then(|o| o.shape.last().copied())
                    .unwrap_or(1)
                    .max(1) as u64;
                let k = fusion
                    .outputs
                    .iter()
                    .find(|o| o.role == "k")
                    .and_then(|o| o.shape.last().copied())
                    .unwrap_or(1)
                    .max(1) as u64;
                let v = fusion
                    .outputs
                    .iter()
                    .find(|o| o.role == "v")
                    .and_then(|o| o.shape.last().copied())
                    .unwrap_or(1)
                    .max(1) as u64;
                q.div_ceil(2) + k.div_ceil(2) + v
            } else {
                fusion
                    .outputs
                    .iter()
                    .map(|o| *o.shape.last().unwrap_or(&1) as u64)
                    .sum()
            };
            let total_out = total_out.max(1);
            let (rows, _) = rows_and_last(&fusion.outputs[0].shape);
            [rows * ceil_div(total_out, sgs_per_tg) * tg, 1, 1]
        }
        FusionKind::RmsNormGateUpSwigluDown => {
            // Single-TG-per-row (down-proj reduces along intermediate
            // and writes hidden, all inside one TG).
            let (rows, _) = rows_and_last(&fusion.outputs[0].shape);
            [rows * tg, 1, 1]
        }
        FusionKind::FinalRmsNormLmHead => {
            // Per active token row, each SG strides across multiple
            // vocab rows so the per-TG RMSNorm recomputation amortizes
            // over more outputs. The emitter derives row and local TG
            // index from the flattened grid.
            let rows = fusion
                .outputs
                .first()
                .map(|b| {
                    crate::target::current_chip()
                        .active_rows_before_last(&b.shape)
                        .max(1)
                })
                .unwrap_or(1);
            let vocab = *fusion.outputs[0].shape.last().unwrap_or(&1) as u64;
            let default_tgs = match fusion.attrs {
                FusionAttrs::FinalRmsNormLmHead { quant: Some(q), .. }
                    if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
                        && q.bits == 4
                        && meta.hidden_size <= 1024 =>
                {
                    128
                }
                FusionAttrs::FinalRmsNormLmHead { quant: Some(q), .. }
                    if matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
                        && q.bits == 4
                        && meta.hidden_size <= 4096 =>
                {
                    32
                }
                _ => 64,
            };
            let target_tgs: u64 = std::env::var("FUSEKIT_FINAL_LM_HEAD_TGS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(default_tgs);
            let max_useful_tgs = ceil_div(vocab, sgs_per_tg);
            let n_tgs = target_tgs.min(max_useful_tgs);
            [rows * n_tgs * tg, 1, 1]
        }
        FusionKind::SdpaDecode => {
            let n_heads = meta.num_attention_heads as u64;
            let active_m = active_m_from_shape(fusion.outputs.first().map(|b| b.shape.as_slice()));
            if sdpa_decode::flash_decoding_enabled() {
                // Flash-Decoding: n_heads × MAX_CHUNKS TGs. Empty
                // chunks early-exit before touching the counter, so
                // the over-spawn is essentially free; the win is the
                // active chunks getting more SMs busy at long context.
                let max_chunks = sdpa_decode::FLASH_DEC_MAX_CHUNKS as u64;
                [active_m * n_heads * max_chunks * tg, 1, 1]
            } else {
                // 1 TG per (batch, head).
                [active_m * n_heads * tg, 1, 1]
            }
        }
        FusionKind::LayerBlock => {
            // LayerBlock composes [RmsNormReduce, SdpaDecode,
            // LinearResidual(o_proj)]. SDPA piece keys off `gid ==
            // head_id` so grid = n_heads TGs. norm + o_proj run
            // redundantly in each TG (norm_y SMEM is per-TG; o_proj
            // simdgroup-coop dispatches one output row per SG, with
            // 14*4=56 SGs covering hidden=1024 → multi-row loop in
            // LinearResidual handles the rest).
            let n_heads = meta.num_attention_heads as u64;
            [n_heads * tg, 1, 1]
        }
        FusionKind::LinearResidual => {
            if let Some((k_dim, out_dim, active_rows, bits, _group_size)) =
                linear_residual_mlx_qmv_fast_shape(fusion)
            {
                let token_rows =
                    linear_residual_qmv_fast_token_rows(active_rows, bits, k_dim, out_dim) as u64;
                let qmv_sgs =
                    linear_residual_qmv_fast_simdgroups(active_rows, bits, k_dim, out_dim) as u64;
                let cols_per_tg = linear_residual_qmv_fast_cols_per_threadgroup(
                    active_rows,
                    bits,
                    k_dim,
                    out_dim,
                ) as u64;
                return [
                    ceil_div(active_rows.max(1), token_rows.max(1))
                        .saturating_mul(qmv_sgs.saturating_mul(32)),
                    ceil_div(out_dim as u64, cols_per_tg.max(1)),
                    1,
                ];
            }
            // Simdgroup-coop GEMV — 1 simdgroup per output row.
            let n_out = elems(&fusion.outputs[0].shape);
            [ceil_div(n_out, sgs_per_tg) * tg, 1, 1]
        }
        FusionKind::SwigluActivation => {
            let n = elems(&fusion.outputs[0].shape);
            [ceil_div(n, tg) * tg, 1, 1]
        }
        FusionKind::RopeQDecode => {
            let active_m = active_m_from_shape(
                fusion
                    .outputs
                    .first()
                    .map(|b| b.shape.as_slice())
                    .or_else(|| fusion.inputs.first().map(|b| b.shape.as_slice())),
            );
            let n = active_m * (meta.num_attention_heads as u64) * (meta.head_dim as u64);
            [ceil_div(n, tg) * tg, 1, 1]
        }
        FusionKind::RmsNormRopeQGateDecode => {
            let active_m = active_m_from_shape(
                fusion
                    .outputs
                    .first()
                    .map(|b| b.shape.as_slice())
                    .or_else(|| fusion.inputs.first().map(|b| b.shape.as_slice())),
            );
            let n_heads = meta.num_attention_heads as u64;
            [active_m * n_heads * tg, 1, 1]
        }
        FusionKind::RopeKvCacheStoreDecode => {
            // The BF16 cache-store emitter uses a flat element range:
            // [0, n_kv * head_dim) writes K and
            // [n_kv * head_dim, 2 * n_kv * head_dim) writes V.
            let active_m = active_m_from_shape(
                fusion
                    .outputs
                    .first()
                    .map(|b| b.shape.as_slice())
                    .or_else(|| fusion.inputs.first().map(|b| b.shape.as_slice())),
            );
            let n = active_m * 2 * (meta.num_key_value_heads as u64) * (meta.head_dim as u64);
            [ceil_div(n, tg) * tg, 1, 1]
        }
        FusionKind::KvCacheStoreDecode => {
            let active_m = active_m_from_shape(
                fusion
                    .outputs
                    .first()
                    .map(|b| b.shape.as_slice())
                    .or_else(|| fusion.inputs.first().map(|b| b.shape.as_slice())),
            );
            let k_dim = fusion
                .inputs
                .first()
                .and_then(|b| b.shape.last())
                .copied()
                .unwrap_or(meta.head_dim as i32)
                .max(1) as u64;
            let v_dim = fusion
                .inputs
                .get(1)
                .and_then(|b| b.shape.last())
                .copied()
                .unwrap_or(meta.head_dim as i32)
                .max(1) as u64;
            let n_kv = if let Some(shape) = fusion.inputs.first().map(|b| b.shape.as_slice()) {
                if shape.len() >= 4 {
                    shape[shape.len() - 3].max(1) as u64
                } else if shape.len() >= 2 {
                    shape[0].max(1) as u64
                } else {
                    meta.num_key_value_heads as u64
                }
            } else {
                meta.num_key_value_heads as u64
            };
            [ceil_div(active_m * n_kv * (k_dim + v_dim), tg) * tg, 1, 1]
        }
        FusionKind::RmsNormRopeKvCacheStoreDecode => {
            let active_m = active_m_from_shape(
                fusion
                    .outputs
                    .first()
                    .map(|b| b.shape.as_slice())
                    .or_else(|| fusion.inputs.first().map(|b| b.shape.as_slice())),
            );
            let n_kv = meta.num_key_value_heads as u64;
            [active_m * 2 * n_kv * tg, 1, 1]
        }
        FusionKind::EmbedLookupDecode => {
            let active_m = crate::target::current_chip().active_m(8) as u64;
            let n = active_m.max(1) * meta.hidden_size as u64;
            [ceil_div(n, tg) * tg, 1, 1]
        }
        FusionKind::ArraysCacheShiftAppendDecode => {
            let n = match fusion.attrs {
                crate::fusion::ir::FusionAttrs::ArraysCacheShiftAppendDecode { width, .. } => {
                    width as u64
                }
                _ => meta.hidden_size as u64,
            };
            [ceil_div(n, tg) * tg, 1, 1]
        }
        FusionKind::ArgmaxLogitsDecode => {
            // One TG per active row; each TG reduces across vocab.
            let rows = argmax_active_rows();
            [rows.max(1) * tg, 1, 1]
        }
        FusionKind::ArgmaxLogitsScan => {
            // N_TGS threadgroups across the vocab for each active row.
            let n_tgs = match fusion.attrs {
                crate::fusion::ir::FusionAttrs::ArgmaxLogitsScan { n_tgs } => n_tgs as u64,
                _ => 1,
            };
            let rows = argmax_active_rows();
            [rows.max(1) * n_tgs * tg, 1, 1]
        }
        FusionKind::ArgmaxLogitsFinalize => {
            // One TG per active row reduces that row's per-TG scratch.
            let rows = argmax_active_rows();
            [rows.max(1) * tg, 1, 1]
        }
        FusionKind::SdpaPrefill => {
            // (l_q, head_id) grid — n_heads × L threadgroups. L is a
            // runtime uniform; the dispatcher binds it via output
            // shape (output is [L, n_heads, head_dim]).
            let n_heads = meta.num_attention_heads as u64;
            let l = elems(&fusion.outputs[0].shape)
                .checked_div(n_heads * meta.head_dim as u64)
                .unwrap_or(1)
                .max(1);
            [n_heads * l * tg, 1, 1]
        }
        _ => [tg, 1, 1],
    }
}

/// Stable function-name suffix derived from kind + shape + partition.
/// Codegen dedupes by this key — two tasks with the same suffix share
/// a single MSL function.
pub fn variant_name(fusion: &Fusion, _partition: &PartitionPlan) -> String {
    // CustomKernel: dedup across all callsites by the mlx-mangled
    // kernel name (mlx encodes dtype + dims into it). Otherwise per
    // fusion id, which keeps every other fusion's MSL function
    // distinct so they get their own PSO.
    use crate::fusion::ir::FusionAttrs;
    if let FusionAttrs::Single {
        op_attrs: crate::graph::Attrs::CustomKernel { name, .. },
        ..
    } = &fusion.attrs
    {
        return name.clone();
    }
    format!("f{}", fusion.id)
}
