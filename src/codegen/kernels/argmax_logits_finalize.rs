//! MSL emit for `FusionKind::ArgmaxLogitsFinalize`.
//!
//! Reduces `argmax_logits_scan`'s per-row, per-TG (max_key, idx) scratch
//! into `next_tok[row]`. 1 TG per active row; thread `tid` reads one scratch
//! entry, then the same simd_max + masked simd_min reduce as the scan
//! kernel narrows to the global winner. Scratch holds the float max
//! pre-encoded as a monotonic uint (see `argmax_logits_scan`) so we can
//! compare via plain `uint` ordering — no float comparisons needed.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    _meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::ArgmaxLogitsFinalize) {
        bail!(
            "argmax_logits_finalize emitter: wrong kind {:?}",
            fusion.kind
        );
    }
    let n_tgs = match fusion.attrs {
        FusionAttrs::ArgmaxLogitsFinalize { n_tgs } => n_tgs,
        _ => bail!("argmax_logits_finalize emitter: attrs mismatch"),
    };

    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: ArgmaxLogitsFinalize (n_tgs = {})",
        fusion.id, n_tgs
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line("device const uint  *tg_argmax    [[buffer(0)]],");
    w.line("device uint        *next_tok     [[buffer(1)]],");
    w.line("device atomic_uint *hist_counter [[buffer(2)]],");
    w.line("device uint        *history      [[buffer(3)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint row     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint N_TGS = {n_tgs}u;"));
    let default_active_rows = crate::target::current_chip().active_m(8).max(1);
    let active_rows = std::env::var("FUSEKIT_ARGMAX_ACTIVE_ROWS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default_active_rows)
        .min(default_active_rows);
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows}u;"));
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("uint row_base = row * N_TGS * 2u;");
    w.line("threadgroup uint key_buf[32];");
    w.line("threadgroup uint idx_buf[32];");
    w.blank();

    // Each thread reads one entry (or strides if N_TGS > tg_size).
    w.line("uint local_key = 0u;       // 0 = MIN under our monotonic encoding");
    w.line("uint local_idx = 0xFFFFFFFFu;");
    w.line("for (uint i = tid; i < N_TGS; i += tg_size) {");
    w.push_indent();
    w.line("uint k = tg_argmax[row_base + i * 2u + 0u];");
    w.line("uint v = tg_argmax[row_base + i * 2u + 1u];");
    w.line("if (k > local_key) { local_key = k; local_idx = v; }");
    w.pop_indent();
    w.line("}");
    w.blank();
    // Per-SG: simd_max(key) then masked simd_min(idx) to pick the
    // lane whose local_key matched (lowest-idx tie-break).
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint sg_max = simd_max(local_key);");
    w.line("uint sg_arg = (local_key == sg_max) ? local_idx : 0xFFFFFFFFu;");
    w.line("uint sg_idx = simd_min(sg_arg);");
    w.line("if (simd_lane == 0u) { key_buf[simd_id] = sg_max; idx_buf[simd_id] = sg_idx; }");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("uint v = (simd_lane < sgs_per_tg) ? key_buf[simd_lane] : 0u;");
    w.line("uint i = (simd_lane < sgs_per_tg) ? idx_buf[simd_lane] : 0xFFFFFFFFu;");
    w.line("uint gm = simd_max(v);");
    w.line("uint mi = (v == gm) ? i : 0xFFFFFFFFu;");
    w.line("uint gi = simd_min(mi);");
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    w.line("next_tok[row] = gi;");
    w.line("if (row == 0u) {");
    w.push_indent();
    w.line("uint h = atomic_fetch_add_explicit(hist_counter, 1u, memory_order_relaxed);");
    w.line("history[h] = gi;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_argmax_logits_finalize_f{}", fusion.id)
}
