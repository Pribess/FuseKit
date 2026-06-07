//! MSL emit for `FusionKind::ArgmaxLogitsScan`.
//!
//! Multi-TG argmax over the LM-head logits. Splits the vocab-wide scan
//! across `N_TGS` threadgroups so M5's 10 GPU cores all participate
//! instead of a single core driving the whole scan. Each TG owns
//! `vocab/N_TGS` consecutive vocab elements; runs the same simd_max +
//! masked simd_min reduce as the single-TG argmax kernel; writes its
//! per-TG (max, idx) pair to a tiny scratch buffer of shape
//! `[ACTIVE_ROWS, N_TGS, 2]`.
//! `argmax_logits_finalize` reads the scratch and reduces to next_tok.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::ArgmaxLogitsScan) {
        bail!("argmax_logits_scan emitter: wrong kind {:?}", fusion.kind);
    }
    let n_tgs = match fusion.attrs {
        FusionAttrs::ArgmaxLogitsScan { n_tgs } => n_tgs,
        _ => bail!("argmax_logits_scan emitter: attrs mismatch"),
    };

    let vocab = meta.vocab_size;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: ArgmaxLogitsScan (vocab = {}, n_tgs = {}, dt = {})",
        fusion.id, vocab, n_tgs, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *logits         [[buffer(0)]],"));
    // scratch shape [N_TGS, 2]: pair of (uint, uint) per TG, where the
    // first uint holds the max value reinterpreted via `as_type<uint>`
    // (preserves IEEE 754 ordering bit-pattern for positive floats; we
    // use signed-flip during finalize for full-range correctness), and
    // the second uint holds the winning vocab index for that TG.
    w.line("device uint        *tg_argmax     [[buffer(1)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint VOCAB = {vocab};"));
    w.writeln(format_args!("constexpr uint N_TGS = {n_tgs}u;"));
    let default_active_rows = crate::target::current_chip().active_m(8).max(1);
    let active_rows = std::env::var("FUSEKIT_ARGMAX_ACTIVE_ROWS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default_active_rows)
        .min(default_active_rows);
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows}u;"));
    w.line("uint row = gid / N_TGS;");
    w.line("uint tg  = gid - row * N_TGS;");
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("device const auto* row_logits = logits + row * VOCAB;");
    w.blank();

    // Each TG covers [tg_start, tg_end) of the vocab. Ceil-div so the
    // last TG handles the tail when VOCAB doesn't divide N_TGS.
    w.line("uint slice_size = (VOCAB + N_TGS - 1u) / N_TGS;");
    w.line("uint tg_start = tg * slice_size;");
    w.line("uint tg_end   = min(tg_start + slice_size, VOCAB);");
    w.blank();

    w.line("threadgroup float vals[32];");
    w.line("threadgroup uint  idxs[32];");
    w.blank();
    // Stage 1 — per-thread stripe scan over this TG's vocab slice.
    w.line("float local_max = -INFINITY;");
    w.line("uint  local_idx = 0u;");
    w.line("for (uint i = tg_start + tid; i < tg_end; i += tg_size) {");
    w.push_indent();
    w.line("float v = float(row_logits[i]);");
    w.line("if (v > local_max) { local_max = v; local_idx = i; }");
    w.pop_indent();
    w.line("}");
    w.blank();
    // Stage 2 — per-SG argmax via simd_max + masked simd_min, mirrors
    // the single-TG kernel's reduce pattern.
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("float sg_max = simd_max(local_max);");
    w.line("uint  sg_arg = (local_max == sg_max) ? local_idx : 0xFFFFFFFFu;");
    w.line("uint  sg_idx = simd_min(sg_arg);");
    w.line("if (simd_lane == 0u) { vals[simd_id] = sg_max; idxs[simd_id] = sg_idx; }");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    // Stage 3 — single-SG reduce over per-SG pairs.
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (simd_lane < sgs_per_tg) ? vals[simd_lane] : -INFINITY;");
    w.line("uint  i = (simd_lane < sgs_per_tg) ? idxs[simd_lane] : 0xFFFFFFFFu;");
    w.line("float gm = simd_max(v);");
    w.line("uint  mi = (v == gm) ? i : 0xFFFFFFFFu;");
    w.line("uint  gi = simd_min(mi);");
    // Stage 4 — lane 0 writes (max_bits, idx) for this TG. We encode
    // the float max as bits via `as_type` so finalize can reduce via
    // monotonic-uint compare. Sign-flip for negative floats:
    //   pos:  bits[31]=0  → toggle MSB to make it sortable ascending
    //   neg:  bits[31]=1  → flip all bits to make it sort below positives
    // Result is an unsigned key that strictly preserves IEEE float order.
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    w.line("uint vbits = as_type<uint>(gm);");
    w.line("uint key   = (vbits & 0x80000000u) ? (~vbits) : (vbits ^ 0x80000000u);");
    w.line("uint out_base = (row * N_TGS + tg) * 2u;");
    w.line("tg_argmax[out_base + 0u] = key;");
    w.line("tg_argmax[out_base + 1u] = gi;");
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
    format!("fusekit_argmax_logits_scan_f{}", fusion.id)
}
