//! MSL emit for `FusionKind::ArgmaxLogitsDecode`.
//!
//! Single-TG argmax over each active LM-head logits row. Writes the
//! winning vocab index into `next_tok[row]`, a uint32 uniform array that
//! step reads as `token_id`. With this kernel embedded in the ICB,
//! the Python decode loop never has to round-trip logits to the CPU.

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
    if !matches!(fusion.kind, FusionKind::ArgmaxLogitsDecode) {
        bail!("argmax_logits_decode emitter: wrong kind {:?}", fusion.kind);
    }
    if !matches!(fusion.attrs, FusionAttrs::ArgmaxLogitsDecode) {
        bail!("argmax_logits_decode emitter: attrs mismatch");
    }

    let vocab = meta.vocab_size;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: ArgmaxLogitsDecode (vocab = {}, dt = {})",
        fusion.id, vocab, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *logits   [[buffer(0)]],"));
    w.line("device uint         *next_tok [[buffer(1)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint row     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint VOCAB = {vocab};"));
    let default_active_rows = crate::target::current_chip().active_m(8).max(1);
    let active_rows = std::env::var("FUSEKIT_ARGMAX_ACTIVE_ROWS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default_active_rows)
        .min(default_active_rows);
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows}u;"));
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("device const auto* row_logits = logits + row * VOCAB;");
    // Inter-SG reduce SMEM: 32 entries covers tg_size up to 1024
    // (Metal cap of 32 simdgroups per threadgroup). Argmax tie-breaks
    // toward the smallest vocab index to stay deterministic.
    w.line("threadgroup float vals[32];");
    w.line("threadgroup uint  idxs[32];");
    w.blank();
    w.line("float local_max = -INFINITY;");
    w.line("uint  local_idx = 0u;");
    w.line("for (uint i = tid; i < VOCAB; i += tg_size) {");
    w.push_indent();
    w.line("float v = float(row_logits[i]);");
    w.line("if (v > local_max) { local_max = v; local_idx = i; }");
    w.pop_indent();
    w.line("}");
    w.blank();
    // Per-SG argmax via simd_max + masked simd_min. simd_max picks
    // the SG-wide max value; lanes whose local_max matches keep
    // their idx (others mask to ~0u); simd_min then selects the
    // smallest matching idx.
    w.line("uint simd_lane  = tid % 32u;");
    w.line("uint simd_id    = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("float sg_max = simd_max(local_max);");
    w.line("uint  sg_arg = (local_max == sg_max) ? local_idx : 0xFFFFFFFFu;");
    w.line("uint  sg_idx = simd_min(sg_arg);");
    w.line("if (simd_lane == 0u) { vals[simd_id] = sg_max; idxs[simd_id] = sg_idx; }");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    // Inter-SG reduce via the same simd primitive trick. Replaces
    // a 256-iter single-thread sequential reduce with parallel
    // simd_max/simd_min over the per-SG (max, idx) pairs.
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (simd_lane < sgs_per_tg) ? vals[simd_lane] : -INFINITY;");
    w.line("uint  i = (simd_lane < sgs_per_tg) ? idxs[simd_lane] : 0xFFFFFFFFu;");
    w.line("float gm = simd_max(v);");
    w.line("uint  mi = (v == gm) ? i : 0xFFFFFFFFu;");
    w.line("uint  gi = simd_min(mi);");
    w.line("if (simd_lane == 0u) next_tok[row] = gi;");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_argmax_logits_decode_f{}", fusion.id)
}
