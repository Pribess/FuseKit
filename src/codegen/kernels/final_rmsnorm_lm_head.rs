//! MSL emit for `FusionKind::FinalRmsNormLmHead`.
//!
//! Decode/MTP-only (active token rows in, active logits rows out).
//! Multi-threadgroup dispatch since vocab is huge (~152K for Qwen).
//! Each threadgroup recomputes the RMSNorm into its own SMEM (cheap vs
//! the vocab projection that follows) and then handles its slice of
//! output rows. This avoids a global-memory normed-x staging buffer.
//!
//! ```text
//!   per active row and per threadgroup tg_id:
//!     1. norm_y[k] = x_row[k] / sqrt(mean(x²)+EPS) · W_norm[k] in SMEM
//!     2. for v in tg's slice of [0, VOCAB):
//!          logits_row[v] = Σ_k norm_y[k] · W_lm_head[v, k]
//! ```
//!
//! `W_lm_head` is in `(vocab, hidden)` row-major storage — same as
//! the embedding table when `tie_word_embeddings = true`. The kernel
//! indexes `W_lm_head[v * HIDDEN + k]` directly; any upstream
//! Transpose is a stride-only view that doesn't change memory layout.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

fn norm_smem_ty(t: &str, quant: Option<crate::fusion::ir::QuantInfo>) -> &str {
    let _ = (t, quant);
    "float"
}

fn emit_norm_writeback_to_smem_ty(
    w: &mut MslWriter,
    hidden: u32,
    input_var: &str,
    norm_w_var: &str,
    act_ty: &str,
    norm_ty: &str,
) {
    if norm_ty == "float" {
        crate::codegen::kernels::emit_norm_writeback_with_act_ty(
            w, hidden, input_var, norm_w_var, act_ty,
        );
        return;
    }

    let value = |idx: &str| match act_ty {
        "bfloat" => format!(
            "bfloat(({norm_w_var})[{idx}] * bfloat(float(({input_var})[{idx}]) * rms_scale))"
        ),
        "half" => {
            format!("half(({norm_w_var})[{idx}] * half(float(({input_var})[{idx}]) * rms_scale))")
        }
        _ => format!(
            "{norm_ty}(float(({input_var})[{idx}]) * rms_scale * float(({norm_w_var})[{idx}]))"
        ),
    };

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
    w.line("uint idx = base + j;");
    w.writeln(format_args!("norm_y[idx] = {};", value("idx")));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

fn emit_quant_lm_head_dot(
    w: &mut MslWriter,
    t: &str,
    acc: &str,
    bits: u32,
    group_size: u32,
    hidden: u32,
    vocab: u32,
) {
    let pack_factor = match bits {
        3 | 5 => 8,
        6 => 4,
        _ => 32 / bits.max(1),
    };
    let packs_per_thread = if bits == 2 { 1 } else { 2 };
    let block_size = pack_factor * packs_per_thread * 32;
    let use_explicit_small_q8 =
        matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
            && bits == 8
            && hidden <= 1024;
    let use_mlx_tg_qmv = group_size == 64
        && matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
        && hidden % block_size == 0
        && !use_explicit_small_q8;
    if use_mlx_tg_qmv {
        w.line("#if FUSEKIT_HAVE_MLX_NAX");
        w.push_indent();
        w.line(format!(
            "{acc} += fusekit_qmv_fast_tg_partial_impl<{t}, {group_size}, {bits}>(W_lm_head, W_lm_head_s, W_lm_head_b, norm_y, HIDDEN, int(v), simd_lane);"
        ));
        w.pop_indent();
        w.line("#else");
        w.push_indent();
    }

    let packed_words = crate::codegen::kernels::quant_packed_words(hidden, bits);
    let packed = format!("W_lm_head + v * {packed_words}u");
    let scales = format!("W_lm_head_s + v * (HIDDEN / {group_size}u)");
    let biases = format!("W_lm_head_b + v * (HIDDEN / {group_size}u)");
    crate::codegen::kernels::emit_int4_gemv_inner_with_options_and_qdot(
        w,
        acc,
        &packed,
        &scales,
        &biases,
        "HIDDEN",
        bits,
        group_size,
        crate::codegen::kernels::qgemv_unroll2_default_for_shape(bits, hidden, vocab),
        false,
        crate::codegen::kernels::QdotPolicy::Disable,
        |k| format!("norm_y[{k}]"),
    );

    if use_mlx_tg_qmv {
        w.pop_indent();
        w.line("#endif");
    }
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::FinalRmsNormLmHead) {
        bail!(
            "final_rmsnorm_lm_head emitter: wrong kind {:?}",
            fusion.kind
        );
    }
    let (eps, quant) = match fusion.attrs {
        FusionAttrs::FinalRmsNormLmHead { eps, quant } => (eps, quant),
        _ => bail!("final_rmsnorm_lm_head emitter: attrs mismatch"),
    };

    let hidden = meta.hidden_size;
    let vocab = meta.vocab_size;
    let active_rows = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_rows_before_last(&b.shape))
        .unwrap_or(1)
        .max(1);
    let t = crate::codegen::kernels::act_ty(meta);
    let is_quant = quant.is_some();
    let norm_ty = norm_smem_ty(t, quant);
    let fn_name = function_name(fusion);

    w.line(format!(
        "// Fusion #{}: FinalRmsNormLmHead (eps = {}, hidden = {}, vocab = {}, active_rows = {}, quant = {:?})",
        fusion.id, eps, hidden, vocab, active_rows, quant
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    if is_quant {
        w.line(format!("device const {t} *x             [[buffer(0)]],"));
        w.line(format!("device const {t} *norm_w        [[buffer(1)]],"));
        w.line("device const uint   *W_lm_head     [[buffer(2)]],");
        w.line(format!("device const {t}   *W_lm_head_s   [[buffer(3)]],"));
        w.line(format!("device const {t}   *W_lm_head_b   [[buffer(4)]],"));
        w.line(format!("device {t}       *logits        [[buffer(5)]],"));
    } else {
        w.line(format!("device const {t} *x          [[buffer(0)]],"));
        w.line(format!("device const {t} *norm_w     [[buffer(1)]],"));
        w.line(format!("device const {t} *W_lm_head  [[buffer(2)]],"));
        w.line(format!("device {t}       *logits     [[buffer(3)]],"));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    // `n_tgs` lets each simdgroup stride across `sgs_total` output
    // rows instead of doing one row each. Amortizes the per-TG
    // RMSNorm recomputation (norm_y is in SMEM, read by every row
    // the TG produces) — fewer TGs = fewer redundant norms.
    w.line("uint n_tgs   [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint HIDDEN = {hidden};"));
    w.writeln(format_args!("constexpr uint VOCAB  = {vocab};"));
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows}u;"));
    w.writeln(format_args!("constexpr float EPS   = {eps};"));
    w.blank();

    w.line("uint tgs_per_row = max(1u, n_tgs / ACTIVE_ROWS);");
    w.line("uint row = gid / tgs_per_row;");
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("uint row_gid = gid - row * tgs_per_row;");
    w.line(format!("device const {t} *x_row = x + row * HIDDEN;"));
    w.line(format!("device {t} *logits_row = logits + row * VOCAB;"));
    w.blank();

    // ── Stage 1: RMSNorm into SMEM (per-TG, recomputed each TG) ─
    w.line(format!("threadgroup {norm_ty} norm_y[HIDDEN];"));
    w.line("threadgroup float sum_sq_partial[32];");
    w.blank();
    crate::codegen::kernels::emit_rmsnorm_reduction(w, hidden, "x_row", "sum_sq_partial");
    w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS);");
    emit_norm_writeback_to_smem_ty(w, hidden, "x_row", "norm_w", t, norm_ty);
    w.blank();

    // ── Stage 2: GEMV over the TG's slice of vocab ─────────────
    // Each thread owns one output row identified by global thread
    // index `gid * tg_size + tid`. Threads with idx >= VOCAB idle.
    w.line("// Stage 2: each thread computes one logit (idle if out of range).");
    // Simdgroup-cooperative GEMV: each simdgroup (32 threads) handles
    // ONE output row, with each thread reducing HIDDEN/32 partial
    // dot products. simd_sum collapses them. The TG has tg_size/32
    // simdgroups working on tg_size/32 output rows in parallel.
    // Adjacent threads now read ADJACENT memory (coalesced) instead
    // of each thread scanning its own row.
    // Reuse simd_lane / simd_id declared during the RMSNorm reduction.
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint sgs_total  = tgs_per_row * sgs_per_tg;");
    // Stride loop: each simdgroup walks output rows in steps of
    // `sgs_total` (= total simdgroups across the whole grid). With
    // fewer TGs the same SG covers more rows, sharing the SMEM
    // `norm_y` across them — amortizes the per-TG RMSNorm work.
    // Host should size the grid so `n_tgs ≈ <occupancy ceiling>` —
    // smaller than the old `ceil(VOCAB / sgs_per_tg)`.
    w.line("for (uint v = row_gid * sgs_per_tg + simd_id; v < VOCAB; v += sgs_total) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    if let Some(q) = quant {
        let bits = q.bits;
        let group_size = q.group_size;
        emit_quant_lm_head_dot(w, t, "acc", bits, group_size, hidden, vocab);
    } else if dense_lm_head_k8_unroll_enabled(hidden, vocab) && hidden % 32 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane * 8u; k4 < HIDDEN / 4u; k4 += 256u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]))");
        w.line("     + dot(float4(ny4[k4+2u]), float4(W4[k4+2u]))");
        w.line("     + dot(float4(ny4[k4+3u]), float4(W4[k4+3u]))");
        w.line("     + dot(float4(ny4[k4+4u]), float4(W4[k4+4u]))");
        w.line("     + dot(float4(ny4[k4+5u]), float4(W4[k4+5u]))");
        w.line("     + dot(float4(ny4[k4+6u]), float4(W4[k4+6u]))");
        w.line("     + dot(float4(ny4[k4+7u]), float4(W4[k4+7u]));");
        w.pop_indent();
        w.line("}");
    } else if dense_lm_head_k4_unroll_enabled(hidden, vocab) && hidden % 16 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane * 4u; k4 < HIDDEN / 4u; k4 += 128u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]))");
        w.line("     + dot(float4(ny4[k4+2u]), float4(W4[k4+2u]))");
        w.line("     + dot(float4(ny4[k4+3u]), float4(W4[k4+3u]));");
        w.pop_indent();
        w.line("}");
    } else if hidden % 8 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane * 2u; k4 < HIDDEN / 4u; k4 += 64u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]));");
        w.pop_indent();
        w.line("}");
    } else if hidden % 4 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]), float4(W4[k4]));");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint k = simd_lane; k < HIDDEN; k += 32u) {");
        w.push_indent();
        w.line("acc += float(norm_y[k]) * float(W_lm_head[v * HIDDEN + k]);");
        w.pop_indent();
        w.line("}");
    }
    w.line("acc = simd_sum(acc);");
    w.line(format!("if (simd_lane == 0u) logits_row[v] = {t}(acc);"));
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    emit_argmax_pair(fusion, meta, w, eps, quant)?;
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_final_rmsnorm_lm_head_f{}", fusion.id)
}

pub fn argmax_function_name(fusion: &Fusion) -> String {
    format!("{}_argmax", function_name(fusion))
}

pub fn argmax_finalize_function_name(fusion: &Fusion) -> String {
    format!("{}_argmax_finalize", function_name(fusion))
}

fn dense_lm_head_k4_unroll_enabled(hidden: u32, vocab: u32) -> bool {
    matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && hidden >= 2048
        && vocab >= 131_072
}

fn dense_lm_head_k8_unroll_enabled(hidden: u32, vocab: u32) -> bool {
    matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && hidden >= 2048
        && vocab >= 131_072
        && hidden.is_multiple_of(32)
}

fn final_head_target_tgs(meta: &ModelMeta, quant: Option<crate::fusion::ir::QuantInfo>) -> u64 {
    let vocab = meta.vocab_size as u64;
    let chip = crate::target::current_chip();
    let b = chip.budgets();
    let sgs_per_tg = b
        .preferred_simdgroups_per_tg
        .min(b.max_simdgroups_per_tg)
        .max(2) as u64;
    let default_tgs = match quant {
        None if matches!(chip, crate::target::ChipModel::M5)
            && meta.hidden_size >= 2048
            && vocab >= 131_072 =>
        {
            128
        }
        Some(q)
            if matches!(chip, crate::target::ChipModel::M5)
                && q.bits == 4
                && meta.hidden_size <= 1024 =>
        {
            128
        }
        Some(q)
            if matches!(chip, crate::target::ChipModel::M5)
                && q.bits == 4
                && meta.hidden_size <= 4096 =>
        {
            // Mid-size q4 LM heads are memory-streaming dominated, but too
            // many slices repeat the RMSNorm/vocab reduction overhead. M5
            // L512 profiles favor 32 slices for this hidden-width class.
            32
        }
        Some(q)
            if matches!(chip, crate::target::ChipModel::M5)
                && q.bits == 8
                && (2048..=4096).contains(&meta.hidden_size) =>
        {
            // Mid-size q8 LM heads are dominated by streaming the vocab
            // matrix, but too many scan slices spend more time in the
            // per-slice reduction/finalize path. L512 M5 profiles favored 32
            // slices over 48/64 for this hidden-width class.
            32
        }
        _ => 64,
    };
    let target_tgs: u64 = std::env::var("FUSEKIT_FINAL_LM_HEAD_TGS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default_tgs);
    let max_useful_tgs = vocab.div_ceil(sgs_per_tg);
    target_tgs.min(max_useful_tgs).max(1)
}

fn emit_argmax_pair(
    fusion: &Fusion,
    meta: &ModelMeta,
    w: &mut MslWriter,
    eps: f32,
    quant: Option<crate::fusion::ir::QuantInfo>,
) -> Result<()> {
    let hidden = meta.hidden_size;
    let vocab = meta.vocab_size;
    let active_rows = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_rows_before_last(&b.shape))
        .unwrap_or(1)
        .max(1);
    let n_tgs = final_head_target_tgs(meta, quant);
    let t = crate::codegen::kernels::act_ty(meta);
    let is_quant = quant.is_some();
    let norm_ty = norm_smem_ty(t, quant);
    let scan_name = argmax_function_name(fusion);

    w.line(format!(
        "// Fusion #{}: FinalRmsNormLmHead direct argmax scan (hidden = {}, vocab = {}, active_rows = {}, n_tgs = {}, quant = {:?})",
        fusion.id, hidden, vocab, active_rows, n_tgs, quant
    ));
    w.line(format!("kernel void {scan_name}("));
    w.push_indent();
    if is_quant {
        w.line(format!("device const {t} *x             [[buffer(0)]],"));
        w.line(format!("device const {t} *norm_w        [[buffer(1)]],"));
        w.line("device const uint   *W_lm_head     [[buffer(2)]],");
        w.line(format!("device const {t}   *W_lm_head_s   [[buffer(3)]],"));
        w.line(format!("device const {t}   *W_lm_head_b   [[buffer(4)]],"));
        w.line("device uint        *tg_argmax     [[buffer(5)]],");
    } else {
        w.line(format!("device const {t} *x             [[buffer(0)]],"));
        w.line(format!("device const {t} *norm_w        [[buffer(1)]],"));
        w.line(format!("device const {t} *W_lm_head     [[buffer(2)]],"));
        w.line("device uint        *tg_argmax     [[buffer(3)]],");
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint n_tgs   [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.writeln(format_args!("constexpr uint HIDDEN = {hidden};"));
    w.writeln(format_args!("constexpr uint VOCAB  = {vocab};"));
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows}u;"));
    w.writeln(format_args!("constexpr uint TGS_PER_ROW = {n_tgs}u;"));
    w.writeln(format_args!("constexpr float EPS   = {eps};"));
    w.line("(void)n_tgs;");
    w.line("uint tgs_per_row = TGS_PER_ROW;");
    w.line("uint row = gid / tgs_per_row;");
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("uint row_gid = gid - row * tgs_per_row;");
    w.line(format!("device const {t} *x_row = x + row * HIDDEN;"));
    w.blank();

    w.line(format!("threadgroup {norm_ty} norm_y[HIDDEN];"));
    w.line("threadgroup float sum_sq_partial[32];");
    crate::codegen::kernels::emit_rmsnorm_reduction(w, hidden, "x_row", "sum_sq_partial");
    w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS);");
    emit_norm_writeback_to_smem_ty(w, hidden, "x_row", "norm_w", t, norm_ty);
    w.blank();

    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint sgs_total = tgs_per_row * sgs_per_tg;");
    w.line("threadgroup float vals[32];");
    w.line("threadgroup uint idxs[32];");
    w.line("uint first_v = row_gid * sgs_per_tg + simd_id;");
    w.line("float local_max = -INFINITY;");
    w.line("uint local_idx = (first_v < VOCAB) ? first_v : 0xFFFFFFFFu;");
    w.line("for (uint v = first_v; v < VOCAB; v += sgs_total) {");
    w.push_indent();
    w.line("float acc = 0.0f;");
    if let Some(q) = quant {
        let bits = q.bits;
        let group_size = q.group_size;
        emit_quant_lm_head_dot(w, t, "acc", bits, group_size, hidden, vocab);
    } else if dense_lm_head_k8_unroll_enabled(hidden, vocab) && hidden % 32 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane * 8u; k4 < HIDDEN / 4u; k4 += 256u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]))");
        w.line("     + dot(float4(ny4[k4+2u]), float4(W4[k4+2u]))");
        w.line("     + dot(float4(ny4[k4+3u]), float4(W4[k4+3u]))");
        w.line("     + dot(float4(ny4[k4+4u]), float4(W4[k4+4u]))");
        w.line("     + dot(float4(ny4[k4+5u]), float4(W4[k4+5u]))");
        w.line("     + dot(float4(ny4[k4+6u]), float4(W4[k4+6u]))");
        w.line("     + dot(float4(ny4[k4+7u]), float4(W4[k4+7u]));");
        w.pop_indent();
        w.line("}");
    } else if dense_lm_head_k4_unroll_enabled(hidden, vocab) && hidden % 16 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane * 4u; k4 < HIDDEN / 4u; k4 += 128u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]))");
        w.line("     + dot(float4(ny4[k4+2u]), float4(W4[k4+2u]))");
        w.line("     + dot(float4(ny4[k4+3u]), float4(W4[k4+3u]));");
        w.pop_indent();
        w.line("}");
    } else if hidden % 8 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane * 2u; k4 < HIDDEN / 4u; k4 += 64u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(ny4[k4+1u]), float4(W4[k4+1u]));");
        w.pop_indent();
        w.line("}");
    } else if hidden % 4 == 0 {
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W_lm_head + v * HIDDEN);"
        ));
        w.line(format!(
            "threadgroup {norm_ty}4* ny4 = (threadgroup {norm_ty}4*)norm_y;"
        ));
        w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
        w.push_indent();
        w.line("acc += dot(float4(ny4[k4]), float4(W4[k4]));");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint k = simd_lane; k < HIDDEN; k += 32u) {");
        w.push_indent();
        w.line("acc += float(norm_y[k]) * float(W_lm_head[v * HIDDEN + k]);");
        w.pop_indent();
        w.line("}");
    }
    w.line("acc = simd_sum(acc);");
    // Match the non-direct path: the logits kernel writes `{t}(acc)` and
    // MLX argmax observes that rounded value. Comparing the full-float
    // accumulator here can flip near-ties on quantized heads.
    w.line(format!("float cmp = float({t}(acc));"));
    w.line("if (simd_lane == 0u && (cmp > local_max || (cmp == local_max && v < local_idx))) {");
    w.push_indent();
    w.line("local_max = cmp;");
    w.line("local_idx = v;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("if (simd_lane == 0u) { vals[simd_id] = local_max; idxs[simd_id] = local_idx; }");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (simd_lane < sgs_per_tg) ? vals[simd_lane] : -INFINITY;");
    w.line("uint i = (simd_lane < sgs_per_tg) ? idxs[simd_lane] : 0xFFFFFFFFu;");
    w.line("float gm = simd_max(v);");
    w.line("uint mi = (v == gm && i != 0xFFFFFFFFu) ? i : 0xFFFFFFFFu;");
    w.line("uint gi = simd_min(mi);");
    w.line("if (simd_lane == 0u) {");
    w.push_indent();
    w.line("uint vbits = as_type<uint>(gm);");
    w.line("uint key = (vbits & 0x80000000u) ? (~vbits) : (vbits ^ 0x80000000u);");
    w.line("uint out_base = (row * tgs_per_row + row_gid) * 2u;");
    w.line("tg_argmax[out_base + 0u] = key;");
    w.line("tg_argmax[out_base + 1u] = gi;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();

    let finalize_name = argmax_finalize_function_name(fusion);
    w.line(format!(
        "// Fusion #{}: FinalRmsNormLmHead direct argmax finalize (n_tgs = {}, active_rows = {})",
        fusion.id, n_tgs, active_rows
    ));
    w.line(format!("kernel void {finalize_name}("));
    w.push_indent();
    w.line("device const uint *in0 [[buffer(0)]],");
    w.line("device uint       *out0 [[buffer(1)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint row     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("threadgroup uint key_buf[32];");
    w.line("threadgroup uint idx_buf[32];");
    w.writeln(format_args!(
        "fusekit_qmv_argmax_finalize_impl<{n_tgs}, {active_rows}>(in0, out0, key_buf, idx_buf, tid, tg_size, row);"
    ));
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
