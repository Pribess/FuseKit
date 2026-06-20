//! MSL emit for `FusionKind::RmsNormQkv`.
//!
//! Produces a single kernel that streams `x` once, computes the RMS
//! norm into threadgroup memory, then projects to Q/K/V in turn —
//! reading the cached norm output from SMEM instead of going back to
//! global memory.
//!
//! ```text
//!   ┌─ kernel(x, W_norm, W_q, W_k, W_v, [bias_q,k,v]) ────────────┐
//!   │                                                              │
//!   │   threadgroup bfloat norm_y[HIDDEN]                           │
//!   │   1. rms = sqrt(mean(x²) + eps);  norm_y = x / rms * W_norm  │
//!   │   2. barrier                                                  │
//!   │   3. q[i] = Σ norm_y[k] * W_q[i,k]   (+ bias_q[i])           │
//!   │   4. k[i] = Σ norm_y[k] * W_k[i,k]   (+ bias_k[i])           │
//!   │   5. v[i] = Σ norm_y[k] * W_v[i,k]   (+ bias_v[i])           │
//!   └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Phase 1 emits a straightforward, threadgroup-per-row layout —
//! correctness first, performance later. Tile shapes, simdgroup
//! matmul, and Q4 dequant inlining all land as separate refinements.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    emit_impl(fusion, partition, meta, None, w)
}

pub fn emit_quant_cache_sink(
    fusion: &Fusion,
    partition: &PartitionPlan,
    meta: &ModelMeta,
    bits: u32,
    group_size: u32,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(bits, 4 | 8) {
        bail!("rmsnorm_qkv quant cache sink: only bits in {{4, 8}}, got {bits}");
    }
    emit_impl(fusion, partition, meta, Some((bits, group_size)), w)
}

fn emit_impl(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    kv_cache_quant: Option<(u32, u32)>,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::RmsNormQkv) {
        bail!(
            "rmsnorm_qkv emitter called with wrong kind: {:?}",
            fusion.kind
        );
    }
    let (eps, quant, decode_sinks) = match fusion.attrs {
        FusionAttrs::RmsNormQkv {
            eps,
            quant,
            decode_sinks,
        } => (eps, quant, decode_sinks),
        _ => bail!("rmsnorm_qkv emitter called with non-matching attrs"),
    };

    let boundary_last_dim = |role: &str| -> Option<u32> {
        fusion
            .outputs
            .iter()
            .find(|b| b.role == role)
            .and_then(|b| b.shape.last().copied())
            .and_then(|d| u32::try_from(d).ok())
    };
    let input_last_dim = |role: &str| -> Option<u32> {
        fusion
            .inputs
            .iter()
            .find(|b| b.role == role)
            .and_then(|b| b.shape.last().copied())
            .and_then(|d| u32::try_from(d).ok())
    };

    let hidden = input_last_dim("x").unwrap_or(meta.hidden_size);
    // Qwen3Next/Qwen3.5 full-attention q_proj carries `[query, gate]`.
    // The query width alone is `num_heads * head_dim`, but the kernel
    // must materialize the full q_proj output because the downstream
    // qgate piece consumes the gate half. Derive projection widths from
    // fusion boundary shapes instead of model meta.
    let n_q = boundary_last_dim("q").unwrap_or(meta.num_attention_heads * meta.head_dim);
    let n_kv = boundary_last_dim("k").unwrap_or(meta.num_key_value_heads * meta.head_dim);
    let active_rows = fusion
        .outputs
        .first()
        .map(|b| crate::target::current_chip().active_rows_before_last(&b.shape))
        .or_else(|| {
            fusion
                .inputs
                .first()
                .map(|b| crate::target::current_chip().active_rows_before_last(&b.shape))
        })
        .unwrap_or(1)
        .max(1);
    if decode_sinks.is_some()
        && (meta.head_dim == 0
            || meta.head_dim % 2 != 0
            || n_q % meta.head_dim != 0
            || n_kv % meta.head_dim != 0)
    {
        bail!(
            "rmsnorm_qkv decode sink requires full even-head RoPE shape: head_dim={} q={} kv={}",
            meta.head_dim,
            n_q,
            n_kv
        );
    }
    let has_bias = fusion.inputs.iter().any(|b| b.role.starts_with("bias_"));
    let is_quant = quant.is_some();
    let quant_cache_sink = kv_cache_quant.is_some();
    let t = crate::codegen::kernels::act_ty(meta);
    let bias_t = t;

    if quant_cache_sink && decode_sinks.is_none() {
        bail!("rmsnorm_qkv quant cache sink requires decode_sinks");
    }
    let (cache_bits, cache_group_size) = kv_cache_quant.unwrap_or((0, 0));
    if quant_cache_sink {
        let el_per_int = 32 / cache_bits;
        if meta.head_dim % el_per_int != 0 || meta.head_dim % cache_group_size.max(1) != 0 {
            bail!(
                "rmsnorm_qkv quant cache sink requires packed head_dim: head_dim={} bits={} group_size={}",
                meta.head_dim,
                cache_bits,
                cache_group_size
            );
        }
    }

    let fn_name = kv_cache_quant
        .map(|(bits, _)| function_name_for(fusion, bits))
        .unwrap_or_else(|| function_name(fusion));

    // Function signature. Buffer slot ordering matches `Fusion::inputs`
    // / `Fusion::outputs` ordering — the runtime relies on it.
    w.line(format!(
        "// Fusion #{}: RmsNormQkv (eps = {}, hidden = {}, q = {}, kv = {}, active_rows = {}, quant = {:?}, decode_sinks = {:?})",
        fusion.id, eps, hidden, n_q, n_kv, active_rows, quant, decode_sinks
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x         [[buffer(0)]],"));
    w.line(format!("device const {t} *norm_w    [[buffer(1)]],"));
    let mut next_slot;
    if is_quant {
        w.line("device const uint   *W_q       [[buffer(2)]],");
        w.line("device const uint   *W_k       [[buffer(3)]],");
        w.line("device const uint   *W_v       [[buffer(4)]],");
        w.line(format!("device const {t}   *W_q_s     [[buffer(5)]],"));
        w.line(format!("device const {t}   *W_k_s     [[buffer(6)]],"));
        w.line(format!("device const {t}   *W_v_s     [[buffer(7)]],"));
        w.line(format!("device const {t}   *W_q_b     [[buffer(8)]],"));
        w.line(format!("device const {t}   *W_k_b     [[buffer(9)]],"));
        w.line(format!("device const {t}   *W_v_b     [[buffer(10)]],"));
        next_slot = 11;
    } else {
        w.line(format!("device const {t} *W_q       [[buffer(2)]],"));
        w.line(format!("device const {t} *W_k       [[buffer(3)]],"));
        w.line(format!("device const {t} *W_v       [[buffer(4)]],"));
        next_slot = 5;
    }
    if has_bias {
        w.line(format!(
            "device const {bias_t} *bias_q   [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "device const {bias_t} *bias_k   [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "device const {bias_t} *bias_v   [[buffer({next_slot})]],"
        ));
        next_slot += 1;
    }
    w.line(format!(
        "device {t}       *q         [[buffer({next_slot})]],"
    ));
    next_slot += 1;
    w.line(format!(
        "device {t}       *k         [[buffer({next_slot})]],"
    ));
    next_slot += 1;
    w.line(format!(
        "device {t}       *v         [[buffer({next_slot})]],"
    ));
    next_slot += 1;
    if decode_sinks.is_some() {
        if quant_cache_sink {
            w.line(format!(
                "device uint     *cache_keys_packed   [[buffer({next_slot})]],"
            ));
            next_slot += 1;
            w.line(format!(
                "device {t}       *cache_keys_scales   [[buffer({next_slot})]],"
            ));
            next_slot += 1;
            w.line(format!(
                "device {t}       *cache_keys_biases   [[buffer({next_slot})]],"
            ));
            next_slot += 1;
            w.line(format!(
                "device uint     *cache_values_packed [[buffer({next_slot})]],"
            ));
            next_slot += 1;
            w.line(format!(
                "device {t}       *cache_values_scales [[buffer({next_slot})]],"
            ));
            next_slot += 1;
            w.line(format!(
                "device {t}       *cache_values_biases [[buffer({next_slot})]],"
            ));
            next_slot += 1;
        } else {
            w.line(format!(
                "device {t}       *cache_keys   [[buffer({next_slot})]],"
            ));
            next_slot += 1;
            w.line(format!(
                "device {t}       *cache_values [[buffer({next_slot})]],"
            ));
            next_slot += 1;
        }
        w.line(format!(
            "constant uint       &offset       [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "constant uint       &write_offset [[buffer({next_slot})]],"
        ));
        next_slot += 1;
        w.line(format!(
            "constant uint       &capacity     [[buffer({next_slot})]],"
        ));
    }
    // Metal requires all thread-position attributes share dimensionality.
    // Phase 1 uses a 1-D grid, so everything is a plain `uint`.
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]],");
    w.line("uint n_tgs   [[threadgroups_per_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    // Compile-time constants come from ModelMeta. Burning them as
    // `constexpr` lets the Metal compiler unroll loops and pick
    // optimal tile sizes.
    w.writeln(format_args!("constexpr uint HIDDEN = {hidden};"));
    w.writeln(format_args!("constexpr uint N_Q    = {n_q};"));
    w.writeln(format_args!("constexpr uint N_KV   = {n_kv};"));
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows}u;"));
    w.writeln(format_args!("constexpr float EPS   = {eps};"));
    if let Some(sinks) = decode_sinks {
        let half = meta.head_dim / 2;
        let log2_theta = (sinks.theta as f64).log2() as f32;
        w.writeln(format_args!(
            "constexpr uint HEAD_DIM = {}u;",
            meta.head_dim
        ));
        w.writeln(format_args!("constexpr uint HALF     = {half}u;"));
        w.line("constexpr uint Q_PAIRS = N_Q / 2u;");
        w.line("constexpr uint K_PAIRS = N_KV / 2u;");
        w.writeln(format_args!("constexpr float BASE_LOG2 = {log2_theta:e}f;"));
        if quant_cache_sink {
            let el_per_int = 32 / cache_bits;
            let n_words = meta.head_dim / el_per_int;
            let n_groups = meta.head_dim / cache_group_size.max(1);
            let mask = (1u32 << cache_bits) - 1;
            w.line("constexpr uint N_KV_HEADS = N_KV / HEAD_DIM;");
            w.writeln(format_args!("constexpr uint CACHE_BITS = {cache_bits}u;"));
            w.writeln(format_args!(
                "constexpr uint CACHE_EL_PER_INT = {el_per_int}u;"
            ));
            w.writeln(format_args!("constexpr uint CACHE_N_WORDS = {n_words}u;"));
            w.writeln(format_args!(
                "constexpr uint CACHE_GROUP_SZ = {cache_group_size}u;"
            ));
            w.writeln(format_args!("constexpr uint CACHE_N_GROUPS = {n_groups}u;"));
            w.writeln(format_args!("constexpr uint CACHE_MASK = {mask}u;"));
        }
    }
    w.blank();

    w.line("uint tgs_per_row = max(1u, n_tgs / ACTIVE_ROWS);");
    w.line("uint row = gid / tgs_per_row;");
    w.line("if (row >= ACTIVE_ROWS) return;");
    w.line("uint row_gid = gid - row * tgs_per_row;");
    w.line(format!("device const {t} *x_row = x + row * HIDDEN;"));
    w.line(format!("device {t} *q_row = q + row * N_Q;"));
    w.line(format!("device {t} *k_row = k + row * N_KV;"));
    w.line(format!("device {t} *v_row = v + row * N_KV;"));
    w.blank();

    // ----- (1) RMS norm into threadgroup memory -----------------------
    w.line("// Stage 1: RMSNorm. y = x / sqrt(mean(x²) + EPS) * norm_w.");
    w.line("threadgroup float norm_y[HIDDEN];");
    w.line("threadgroup float sum_sq_partial[32];");
    if quant_cache_sink {
        w.line("threadgroup float kv_qrow[HEAD_DIM];");
        w.line("threadgroup float kv_partial_min[32];");
        w.line("threadgroup float kv_partial_max[32];");
        w.line("threadgroup float kv_gmin[32];");
        w.line("threadgroup float kv_gmax[32];");
    }
    w.blank();
    crate::codegen::kernels::emit_rmsnorm_reduction(w, hidden, "x_row", "sum_sq_partial");
    w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS);");
    crate::codegen::kernels::emit_norm_writeback_with_act_ty(w, hidden, "x_row", "norm_w", t);
    w.blank();

    // ----- (2) Three projections ------------------------------------
    // Simdgroup-cooperative GEMV across Q/K/V projections.
    // Without decode sinks, one simdgroup computes one output element.
    // With decode sinks, Q and K work is mapped by RoPE pair: one
    // simdgroup computes both half-pair projection rows, rounds them
    // through the activation dtype, applies the same RoPE math as the
    // standalone kernels, then writes roped Q or live K cache directly.
    // V is projected elementwise and stored directly into the V cache.
    // DIAGNOSTIC: force non-unroll4 path to isolate bf16 Q bug
    let unroll4 = hidden % 4 == 0 && std::env::var_os("FK_NO_UNROLL4").is_none();
    let bits = quant.map(|q| q.bits).unwrap_or(0);
    let group_size = quant.map(|q| q.group_size).unwrap_or(0);
    let qmv_tg_helper = is_quant
        && decode_sinks.is_none()
        && matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && active_rows <= 1
        && group_size == 64
        && bits == 8
        && crate::codegen::kernels::qmv_fast_block_size(bits)
            .is_some_and(|block| (hidden as u32).is_multiple_of(block))
        && hidden >= 2048;
    let emit_dot = |w: &mut MslWriter,
                    acc: &str,
                    weight: &str,
                    scales: &str,
                    biases: &str,
                    idx: &str,
                    out_dim: u32| {
        if is_quant {
            if qmv_tg_helper {
                w.writeln(format_args!(
                    "{acc} = fusekit_qmv_fast_tg_partial_impl<{t}, {group_size}, {bits}>({weight}, {scales}, {biases}, norm_y, int(HIDDEN), int({idx}), simd_lane);"
                ));
                return;
            }
            // int4 affine dequant inline GEMV using SMEM-resident norm_y
            // as the activation source.
            let packed_words = crate::codegen::kernels::quant_packed_words(hidden as u32, bits);
            let packed_row = format!("{weight} + ({idx}) * {packed_words}u");
            let scales_row = format!("{scales} + ({idx}) * (HIDDEN / {group_size}u)");
            let biases_row = format!("{biases} + ({idx}) * (HIDDEN / {group_size}u)");
            crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
                w,
                acc,
                &packed_row,
                &scales_row,
                &biases_row,
                "HIDDEN",
                bits,
                group_size,
                crate::codegen::kernels::qgemv_unroll2_default_for_shape(
                    bits,
                    hidden as u32,
                    out_dim,
                ),
                |k| format!("norm_y[{k}]"),
            );
        } else if unroll4 {
            w.line(format!(
                "device const {t}4* W4_{acc} = (device const {t}4*)({weight} + ({idx}) * HIDDEN);"
            ));
            w.line("threadgroup float4* ny4 = (threadgroup float4*)norm_y;");
            w.line("for (uint k4 = simd_lane; k4 < HIDDEN / 4u; k4 += 32u) {");
            w.push_indent();
            w.line(format!("{acc} += dot(ny4[k4], float4(W4_{acc}[k4]));"));
            w.pop_indent();
            w.line("}");
        } else {
            w.line("for (uint kx = simd_lane; kx < HIDDEN; kx += 32u) {");
            w.push_indent();
            w.writeln(format_args!(
                "{acc} += norm_y[kx] * float({weight}[({idx}) * HIDDEN + kx]);"
            ));
            w.pop_indent();
            w.line("}");
        }
    };

    let emit_scalar_projection = |w: &mut MslWriter,
                                  weight: &str,
                                  scales: &str,
                                  biases: &str,
                                  bias: &str,
                                  out: &str,
                                  idx: &str,
                                  out_dim: u32| {
        w.line("float acc = 0.0f;");
        emit_dot(w, "acc", weight, scales, biases, idx, out_dim);
        w.line("acc = simd_sum(acc);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.writeln(format_args!("acc += float({bias}[{idx}]);"));
        }
        w.line(format!("{out}_row[{idx}] = {t}(acc);"));
        w.pop_indent();
        w.line("}");
    };

    // simd_lane/simd_id were declared during norm; reuse them.
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("uint g = row_gid * sgs_per_tg + simd_id;");
    if decode_sinks.is_some() {
        w.line("uint pos = offset + row;");
        w.line("uint cache_offset = (write_offset + row) % capacity;");
    }
    if decode_sinks.is_some() && quant_cache_sink {
        w.line("uint q_tgs = (Q_PAIRS + sgs_per_tg - 1u) / sgs_per_tg;");
        w.line("if (row_gid < q_tgs) {");
        w.push_indent();
        w.line("uint qg = row_gid * sgs_per_tg + simd_id;");
        w.line("if (qg < Q_PAIRS) {");
        w.push_indent();
        w.line("uint h = qg / HALF;");
        w.line("uint pair_idx = qg - h * HALF;");
        w.line("uint q_lo = h * HEAD_DIM + pair_idx;");
        w.line("uint q_hi = q_lo + HALF;");
        w.line("float acc_q_lo = 0.0f;");
        w.line("float acc_q_hi = 0.0f;");
        emit_dot(w, "acc_q_lo", "W_q", "W_q_s", "W_q_b", "q_lo", n_q);
        emit_dot(w, "acc_q_hi", "W_q", "W_q_s", "W_q_b", "q_hi", n_q);
        w.line("acc_q_lo = simd_sum(acc_q_lo);");
        w.line("acc_q_hi = simd_sum(acc_q_hi);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.line("acc_q_lo += float(bias_q[q_lo]);");
            w.line("acc_q_hi += float(bias_q[q_hi]);");
        }
        w.line(format!("float lo = float({t}(acc_q_lo));"));
        w.line(format!("float hi = float({t}(acc_q_hi));"));
        w.line("float d_norm = float(pair_idx) / float(HALF);");
        w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
        w.line("float angle = float(pos) * inv_freq;");
        w.line("float c = metal::fast::cos(angle);");
        w.line("float s = metal::fast::sin(angle);");
        w.line(format!("q_row[q_lo] = {t}(lo * c - hi * s);"));
        w.line(format!("q_row[q_hi] = {t}(hi * c + lo * s);"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("uint kv_tg = row_gid - q_tgs;");
        w.line("if (kv_tg < 2u * N_KV_HEADS) {");
        w.push_indent();
        w.line("bool is_v = kv_tg >= N_KV_HEADS;");
        w.line("uint kh = is_v ? (kv_tg - N_KV_HEADS) : kv_tg;");
        w.line("if (is_v) {");
        w.push_indent();
        w.line("for (uint d = simd_id; d < HEAD_DIM; d += sgs_per_tg) {");
        w.push_indent();
        w.line("uint proj_idx = kh * HEAD_DIM + d;");
        w.line("float acc_vq = 0.0f;");
        emit_dot(w, "acc_vq", "W_v", "W_v_s", "W_v_b", "proj_idx", n_kv);
        w.line("acc_vq = simd_sum(acc_vq);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.line("acc_vq += float(bias_v[proj_idx]);");
        }
        w.line(format!("kv_qrow[d] = float({t}(acc_vq));"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("for (uint pair_idx = simd_id; pair_idx < HALF; pair_idx += sgs_per_tg) {");
        w.push_indent();
        w.line("uint k_lo = kh * HEAD_DIM + pair_idx;");
        w.line("uint k_hi = k_lo + HALF;");
        w.line("float acc_kq_lo = 0.0f;");
        w.line("float acc_kq_hi = 0.0f;");
        emit_dot(w, "acc_kq_lo", "W_k", "W_k_s", "W_k_b", "k_lo", n_kv);
        emit_dot(w, "acc_kq_hi", "W_k", "W_k_s", "W_k_b", "k_hi", n_kv);
        w.line("acc_kq_lo = simd_sum(acc_kq_lo);");
        w.line("acc_kq_hi = simd_sum(acc_kq_hi);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.line("acc_kq_lo += float(bias_k[k_lo]);");
            w.line("acc_kq_hi += float(bias_k[k_hi]);");
        }
        w.line(format!("float lo = float({t}(acc_kq_lo));"));
        w.line(format!("float hi = float({t}(acc_kq_hi));"));
        w.line("float d_norm = float(pair_idx) / float(HALF);");
        w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
        w.line("float angle = float(pos) * inv_freq;");
        w.line("float c = metal::fast::cos(angle);");
        w.line("float s = metal::fast::sin(angle);");
        w.line("kv_qrow[pair_idx] = lo * c - hi * s;");
        w.line("kv_qrow[pair_idx + HALF] = hi * c + lo * s;");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("for (uint qg = 0u; qg < CACHE_N_GROUPS; ++qg) {");
        w.push_indent();
        w.line("uint g_start = qg * CACHE_GROUP_SZ;");
        w.line("float my_min = INFINITY; float my_max = -INFINITY;");
        w.line("for (uint d = tid; d < CACHE_GROUP_SZ; d += tg_size) {");
        w.push_indent();
        w.line("float vx = kv_qrow[g_start + d];");
        w.line("my_min = min(my_min, vx); my_max = max(my_max, vx);");
        w.pop_indent();
        w.line("}");
        w.line("my_min = simd_min(my_min); my_max = simd_max(my_max);");
        w.line("if (simd_lane == 0u) { kv_partial_min[simd_id] = my_min; kv_partial_max[simd_id] = my_max; }");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        w.line("uint nsg = tg_size / 32u;");
        w.line("float v_min = (tid < nsg) ? kv_partial_min[tid] : INFINITY;");
        w.line("float v_max = (tid < nsg) ? kv_partial_max[tid] : -INFINITY;");
        w.line("v_min = simd_min(v_min); v_max = simd_max(v_max);");
        w.line("if (simd_lane == 0u) { kv_gmin[qg] = v_min; kv_gmax[qg] = v_max; }");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.pop_indent();
        w.line("}");
        w.line("uint co_packed = kh * capacity * CACHE_N_WORDS + cache_offset * CACHE_N_WORDS;");
        w.line("uint co_meta = kh * capacity * CACHE_N_GROUPS + cache_offset * CACHE_N_GROUPS;");
        w.line("for (uint wi = tid; wi < CACHE_N_WORDS; wi += tg_size) {");
        w.push_indent();
        w.line("uint d_base = wi * CACHE_EL_PER_INT;");
        w.line("uint qg = d_base / CACHE_GROUP_SZ;");
        w.line("float v_min = kv_gmin[qg];");
        w.line("float v_max = kv_gmax[qg];");
        w.line("float scale_abs = max((v_max - v_min) / float(CACHE_MASK), 1.0e-7f);");
        w.line("bool use_lo_anchor = metal::fabs(v_min) > metal::fabs(v_max);");
        w.line("float scale = use_lo_anchor ? scale_abs : -scale_abs;");
        w.line("float edge = use_lo_anchor ? v_min : v_max;");
        w.line("float q0 = round(edge / scale);");
        w.line("bool at_zero = (q0 == 0.0f);");
        w.line("scale = at_zero ? scale : (edge / q0);");
        w.line("float bias = at_zero ? 0.0f : edge;");
        w.line("float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;");
        w.line("uint packed = 0u;");
        w.line("for (uint el = 0u; el < CACHE_EL_PER_INT; ++el) {");
        w.push_indent();
        w.line("float xq = kv_qrow[d_base + el];");
        w.line("int qi = int(round((xq - bias) * inv_scale));");
        w.line("uint q = (uint)clamp(qi, 0, int(CACHE_MASK));");
        w.line("packed |= (q & CACHE_MASK) << (el * CACHE_BITS);");
        w.pop_indent();
        w.line("}");
        w.line("if (is_v) cache_values_packed[co_packed + wi] = packed;");
        w.line("else      cache_keys_packed[co_packed + wi] = packed;");
        w.pop_indent();
        w.line("}");
        w.line("for (uint qg = tid; qg < CACHE_N_GROUPS; qg += tg_size) {");
        w.push_indent();
        w.line("float v_min = kv_gmin[qg];");
        w.line("float v_max = kv_gmax[qg];");
        w.line("float scale_abs = max((v_max - v_min) / float(CACHE_MASK), 1.0e-7f);");
        w.line("bool use_lo_anchor = metal::fabs(v_min) > metal::fabs(v_max);");
        w.line("float scale = use_lo_anchor ? scale_abs : -scale_abs;");
        w.line("float edge = use_lo_anchor ? v_min : v_max;");
        w.line("float q0 = round(edge / scale);");
        w.line("bool at_zero = (q0 == 0.0f);");
        w.line("scale = at_zero ? scale : (edge / q0);");
        w.line("float bias = at_zero ? 0.0f : edge;");
        w.line("if (is_v) {");
        w.push_indent();
        w.line(format!("cache_values_scales[co_meta + qg] = {t}(scale);"));
        w.line(format!("cache_values_biases[co_meta + qg] = {t}(bias);"));
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line(format!("cache_keys_scales[co_meta + qg] = {t}(scale);"));
        w.line(format!("cache_keys_biases[co_meta + qg] = {t}(bias);"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    } else if decode_sinks.is_some() {
        w.line("if (g < Q_PAIRS) {");
        w.push_indent();
        w.line("uint h = g / HALF;");
        w.line("uint pair_idx = g - h * HALF;");
        w.line("uint q_lo = h * HEAD_DIM + pair_idx;");
        w.line("uint q_hi = q_lo + HALF;");
        w.line("float acc_q_lo = 0.0f;");
        w.line("float acc_q_hi = 0.0f;");
        emit_dot(w, "acc_q_lo", "W_q", "W_q_s", "W_q_b", "q_lo", n_q);
        emit_dot(w, "acc_q_hi", "W_q", "W_q_s", "W_q_b", "q_hi", n_q);
        w.line("acc_q_lo = simd_sum(acc_q_lo);");
        w.line("acc_q_hi = simd_sum(acc_q_hi);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.line("acc_q_lo += float(bias_q[q_lo]);");
            w.line("acc_q_hi += float(bias_q[q_hi]);");
        }
        w.line(format!("float lo = float({t}(acc_q_lo));"));
        w.line(format!("float hi = float({t}(acc_q_hi));"));
        w.line("float d_norm = float(pair_idx) / float(HALF);");
        w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
        w.line("float angle = float(pos) * inv_freq;");
        w.line("float c = metal::fast::cos(angle);");
        w.line("float s = metal::fast::sin(angle);");
        w.line(format!("q_row[q_lo] = {t}(lo * c - hi * s);"));
        w.line(format!("q_row[q_hi] = {t}(hi * c + lo * s);"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else if (g < Q_PAIRS + K_PAIRS) {");
        w.push_indent();
        w.line("uint kg = g - Q_PAIRS;");
        w.line("uint kh = kg / HALF;");
        w.line("uint pair_idx = kg - kh * HALF;");
        w.line("uint k_lo = kh * HEAD_DIM + pair_idx;");
        w.line("uint k_hi = k_lo + HALF;");
        w.line("float acc_k_lo = 0.0f;");
        w.line("float acc_k_hi = 0.0f;");
        emit_dot(w, "acc_k_lo", "W_k", "W_k_s", "W_k_b", "k_lo", n_kv);
        emit_dot(w, "acc_k_hi", "W_k", "W_k_s", "W_k_b", "k_hi", n_kv);
        w.line("acc_k_lo = simd_sum(acc_k_lo);");
        w.line("acc_k_hi = simd_sum(acc_k_hi);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.line("acc_k_lo += float(bias_k[k_lo]);");
            w.line("acc_k_hi += float(bias_k[k_hi]);");
        }
        w.line(format!("float lo = float({t}(acc_k_lo));"));
        w.line(format!("float hi = float({t}(acc_k_hi));"));
        w.line("float d_norm = float(pair_idx) / float(HALF);");
        w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
        w.line("float angle = float(pos) * inv_freq;");
        w.line("float c = metal::fast::cos(angle);");
        w.line("float s = metal::fast::sin(angle);");
        w.line("uint cache_base = kh * capacity * HEAD_DIM + cache_offset * HEAD_DIM;");
        w.line(format!(
            "cache_keys[cache_base + pair_idx] = {t}(lo * c - hi * s);"
        ));
        w.line(format!(
            "cache_keys[cache_base + pair_idx + HALF] = {t}(hi * c + lo * s);"
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("} else if (g < Q_PAIRS + K_PAIRS + N_KV) {");
        w.push_indent();
        w.line("uint i = g - Q_PAIRS - K_PAIRS;");
        w.line("float acc_v = 0.0f;");
        emit_dot(w, "acc_v", "W_v", "W_v_s", "W_v_b", "i", n_kv);
        w.line("acc_v = simd_sum(acc_v);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if has_bias {
            w.line("acc_v += float(bias_v[i]);");
        }
        w.line("uint vh = i / HEAD_DIM;");
        w.line("uint vd = i - vh * HEAD_DIM;");
        w.line("uint cache_off = vh * capacity * HEAD_DIM + cache_offset * HEAD_DIM + vd;");
        w.line(format!("cache_values[cache_off] = {t}(acc_v);"));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("if (g < N_Q) {");
        w.push_indent();
        emit_scalar_projection(w, "W_q", "W_q_s", "W_q_b", "bias_q", "q", "g", n_q);
        w.pop_indent();
        w.line("} else if (g < N_Q + N_KV) {");
        w.push_indent();
        w.line("uint i = g - N_Q;");
        emit_scalar_projection(w, "W_k", "W_k_s", "W_k_b", "bias_k", "k", "i", n_kv);
        w.pop_indent();
        w.line("} else if (g < N_Q + 2u * N_KV) {");
        w.push_indent();
        w.line("uint i = g - N_Q - N_KV;");
        emit_scalar_projection(w, "W_v", "W_v_s", "W_v_b", "bias_v", "v", "i", n_kv);
        w.pop_indent();
        w.line("}");
    }
    if decode_sinks.is_some() {
        // The decode-sink variant has semantic side effects: it writes
        // the live K/V cache that the following SDPA command reads.
        // Direct dispatch adds an encoder memory barrier between
        // commands, but ICB execution relies on the kernel making its
        // device-memory writes visible before the next indirect command.
        w.line("threadgroup_barrier(mem_flags::mem_device);");
    }
    // emit_proj retained but unused — kept around in case the
    // single-TG variant is needed for debug. Suppress dead-code warn.
    let _ = (emit_proj, n_q, n_kv);

    w.pop_indent();
    w.line("}");
    w.blank();

    Ok(())
}

fn emit_proj(
    w: &mut MslWriter,
    out: &str,
    weight: &str,
    bias: &str,
    _n: u32,
    n_const: &str,
    has_bias: bool,
) {
    w.writeln(format_args!(
        "// {out} projection: {out}[i] = Σ norm_y[k] * {weight}[i, k]"
    ));
    w.writeln(format_args!(
        "for (uint i = tid; i < {n_const}; i += tg_size) {{"
    ));
    w.push_indent();
    w.line("float acc = 0.0f;");
    w.line("for (uint kx = 0u; kx < HIDDEN; ++kx) {");
    w.push_indent();
    w.writeln(format_args!(
        "acc += norm_y[kx] * float({weight}[i * HIDDEN + kx]);"
    ));
    w.pop_indent();
    w.line("}");
    if has_bias {
        w.writeln(format_args!("acc += float({bias}[i]);"));
    }
    w.writeln(format_args!("{out}[i] = bfloat(acc);"));
    w.pop_indent();
    w.line("}");
    w.blank();
}

/// Stable MSL function name for this fusion. The persistent megakernel
/// scheduler uses this to dispatch by `(kind, variant)`.
pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rmsnorm_qkv_f{}", fusion.id)
}

pub fn function_name_for(fusion: &Fusion, bits: u32) -> String {
    format!("fusekit_rmsnorm_qkv_q{bits}_f{}", fusion.id)
}
