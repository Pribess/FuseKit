//! Quantized-cache variant of `RmsNormRopeKvCacheStoreDecode`.
//!
//! The bf16 fused kernel normalizes K, applies partial RoPE, and writes
//! K/V directly to `KVCache`. This variant keeps the same semantic work
//! but writes `QuantizedKVCache`'s `(packed, scales, biases)` layout so
//! the later SDPA quant kernel can dequantize inside attention without a
//! bf16 cache round-trip.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    bits: u32,
    group_size: u32,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::RmsNormRopeKvCacheStoreDecode) {
        bail!(
            "rmsnorm_rope_kv_cache_store_quant emitter: wrong kind {:?}",
            fusion.kind
        );
    }
    let FusionAttrs::RmsNormRopeKvCacheStoreDecode {
        eps,
        theta,
        rope_dims,
    } = fusion.attrs
    else {
        bail!("rmsnorm_rope_kv_cache_store_quant emitter: attrs mismatch");
    };
    if !matches!(bits, 4 | 8) {
        bail!("rmsnorm_rope_kv_cache_store_quant: only bits in {{4, 8}}, got {bits}");
    }

    let k_shape = fusion
        .inputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let n_kv = k_shape
        .get(k_shape.len().saturating_sub(2))
        .copied()
        .unwrap_or(meta.num_key_value_heads as i32)
        .max(1) as u32;
    let head_dim = k_shape
        .last()
        .copied()
        .unwrap_or(meta.head_dim as i32)
        .max(1) as u32;
    let dims = rope_dims.min(head_dim);
    let half = (dims / 2).max(1);
    let active_m = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_tokens_for_shape(&b.shape))
        .unwrap_or_else(|| crate::target::current_chip().active_m_for_unshaped(8))
        .max(1);
    let t = crate::codegen::kernels::act_ty(meta);
    let el_per_int = 32 / bits;
    let n_words = head_dim / el_per_int;
    let n_groups = head_dim / group_size.max(1);
    let mask = (1u32 << bits) - 1;
    let log2_theta = (theta as f64).log2() as f32;
    let fn_name = function_name_for(fusion, bits);

    w.line(format!(
        "// Fusion #{}: RmsNormRopeKvCacheStoreDecode quant b{} (active_m={}, n_kv={}, head_dim={}, rope_dims={}, theta={}, gs={})",
        fusion.id, bits, active_m, n_kv, head_dim, dims, theta, group_size
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!(
        "device const {t} *k_proj              [[buffer(0)]],"
    ));
    w.line(format!(
        "device const {t} *v_proj              [[buffer(1)]],"
    ));
    w.line(format!(
        "device const {t} *k_weight            [[buffer(2)]],"
    ));
    w.line("device uint       *cache_keys_packed   [[buffer(3)]],");
    w.line(format!(
        "device {t}       *cache_keys_scales   [[buffer(4)]],"
    ));
    w.line(format!(
        "device {t}       *cache_keys_biases   [[buffer(5)]],"
    ));
    w.line("device uint       *cache_values_packed [[buffer(6)]],");
    w.line(format!(
        "device {t}       *cache_values_scales [[buffer(7)]],"
    ));
    w.line(format!(
        "device {t}       *cache_values_biases [[buffer(8)]],"
    ));
    w.line("constant uint      &offset             [[buffer(9)]],");
    w.line("constant uint      &write_offset       [[buffer(10)]],");
    w.line("constant uint      &capacity           [[buffer(11)]],");
    w.line("uint tid       [[thread_position_in_threadgroup]],");
    w.line("uint tg_size   [[threads_per_threadgroup]],");
    w.line("uint group_id  [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint N_KV       = {n_kv};"));
    w.writeln(format_args!("constexpr uint ACTIVE_M   = {active_m};"));
    w.writeln(format_args!("constexpr uint HEAD_DIM   = {head_dim};"));
    w.writeln(format_args!("constexpr uint ROPE_DIMS  = {dims};"));
    w.writeln(format_args!("constexpr uint HALF       = {half};"));
    w.writeln(format_args!("constexpr uint BITS       = {bits};"));
    w.writeln(format_args!("constexpr uint EL_PER_INT = {el_per_int};"));
    w.writeln(format_args!("constexpr uint N_WORDS    = {n_words};"));
    w.writeln(format_args!("constexpr uint GROUP_SZ   = {group_size};"));
    w.writeln(format_args!("constexpr uint N_GROUPS   = {n_groups};"));
    w.writeln(format_args!("constexpr uint MASK       = {mask};"));
    w.writeln(format_args!("constexpr float EPS       = {eps:e}f;"));
    w.writeln(format_args!("constexpr float BASE_LOG2 = {log2_theta:e}f;"));
    w.line("threadgroup float qrow[HEAD_DIM];");
    w.line("threadgroup float reduce_buf[32];");
    w.line("threadgroup float partial_min[32];");
    w.line("threadgroup float partial_max[32];");
    w.line("threadgroup float gmin[32];");
    w.line("threadgroup float gmax[32];");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.blank();

    w.line("uint row = group_id / (2u * N_KV);");
    w.line("uint local_group = group_id - row * (2u * N_KV);");
    w.line("if (row >= ACTIVE_M) return;");
    w.line("bool is_v = (local_group >= N_KV);");
    w.line("uint head = is_v ? (local_group - N_KV) : local_group;");
    w.line("if (head >= N_KV) return;");
    w.line("uint base = (row * N_KV + head) * HEAD_DIM;");
    w.line("uint pos = offset + row;");
    w.line("uint cache_offset = (write_offset + row) % capacity;");
    w.blank();

    w.line("if (!is_v) {");
    w.push_indent();
    w.line("float sum_sq = 0.0f;");
    w.line("for (uint d = tid; d < HEAD_DIM; d += tg_size) {");
    w.push_indent();
    w.line("float x = float(k_proj[base + d]);");
    w.line("sum_sq += x * x;");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = metal::precise::rsqrt(reduce_buf[0] / float(HEAD_DIM) + EPS);");
    w.line("for (uint d = tid; d < HEAD_DIM; d += tg_size) {");
    w.push_indent();
    w.line("float val = float(k_proj[base + d]) * rms_scale * float(k_weight[d]);");
    w.line("if (d < ROPE_DIMS) {");
    w.push_indent();
    w.line("uint pair_idx = (d < HALF) ? d : (d - HALF);");
    w.line("float lo = float(k_proj[base + pair_idx]) * rms_scale * float(k_weight[pair_idx]);");
    w.line("float hi = float(k_proj[base + pair_idx + HALF]) * rms_scale * float(k_weight[pair_idx + HALF]);");
    w.line("float d_norm = float(pair_idx) / float(HALF);");
    w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
    w.line("float angle = float(pos) * inv_freq;");
    w.line("float c = metal::fast::cos(angle);");
    w.line("float s = metal::fast::sin(angle);");
    w.line("val = (d < HALF) ? (lo * c - hi * s) : (hi * c + lo * s);");
    w.pop_indent();
    w.line("}");
    w.line("qrow[d] = val;");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.line("for (uint d = tid; d < HEAD_DIM; d += tg_size) {");
    w.push_indent();
    w.line("qrow[d] = float(v_proj[base + d]);");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    w.line("for (uint g = 0u; g < N_GROUPS; ++g) {");
    w.push_indent();
    w.line("uint g_start = g * GROUP_SZ;");
    w.line("float my_min = INFINITY; float my_max = -INFINITY;");
    w.line("for (uint d = tid; d < GROUP_SZ; d += tg_size) {");
    w.push_indent();
    w.line("float v = qrow[g_start + d];");
    w.line("my_min = min(my_min, v); my_max = max(my_max, v);");
    w.pop_indent();
    w.line("}");
    w.line("my_min = simd_min(my_min); my_max = simd_max(my_max);");
    w.line(
        "if (simd_lane == 0u) { partial_min[simd_id] = my_min; partial_max[simd_id] = my_max; }",
    );
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("uint nsg = tg_size / 32u;");
    w.line("float v_min = (tid < nsg) ? partial_min[tid] : INFINITY;");
    w.line("float v_max = (tid < nsg) ? partial_max[tid] : -INFINITY;");
    w.line("v_min = simd_min(v_min); v_max = simd_max(v_max);");
    w.line("if (simd_lane == 0u) { gmin[g] = v_min; gmax[g] = v_max; }");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.pop_indent();
    w.line("}");
    w.blank();

    w.line("uint co_packed = head * capacity * N_WORDS + cache_offset * N_WORDS;");
    w.line("uint co_meta   = head * capacity * N_GROUPS + cache_offset * N_GROUPS;");
    w.line("for (uint wi = tid; wi < N_WORDS; wi += tg_size) {");
    w.push_indent();
    w.line("uint d_base = wi * EL_PER_INT;");
    w.line("uint g = d_base / GROUP_SZ;");
    w.line("float v_min = gmin[g];");
    w.line("float v_max = gmax[g];");
    w.line("float scale_abs = max((v_max - v_min) / float(MASK), 1.0e-7f);");
    w.line("bool use_lo_anchor = metal::fabs(v_min) > metal::fabs(v_max);");
    w.line("float scale = use_lo_anchor ? scale_abs : -scale_abs;");
    w.line("float edge = use_lo_anchor ? v_min : v_max;");
    w.line("float q0 = round(edge / scale);");
    w.line("bool at_zero = (q0 == 0.0f);");
    w.line("scale = at_zero ? scale : (edge / q0);");
    w.line("float bias = at_zero ? 0.0f : edge;");
    w.line("float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;");
    w.line("uint packed = 0u;");
    w.line("for (uint el = 0u; el < EL_PER_INT; ++el) {");
    w.push_indent();
    w.line("float x = qrow[d_base + el];");
    w.line("int qi = int(round((x - bias) * inv_scale));");
    w.line("uint q = (uint)clamp(qi, 0, int(MASK));");
    w.line("packed |= (q & MASK) << (el * BITS);");
    w.pop_indent();
    w.line("}");
    w.line("if (is_v) cache_values_packed[co_packed + wi] = packed;");
    w.line("else      cache_keys_packed[co_packed + wi]   = packed;");
    w.pop_indent();
    w.line("}");
    w.line("for (uint g = tid; g < N_GROUPS; g += tg_size) {");
    w.push_indent();
    w.line("float v_min = gmin[g];");
    w.line("float v_max = gmax[g];");
    w.line("float scale_abs = max((v_max - v_min) / float(MASK), 1.0e-7f);");
    w.line("bool use_lo_anchor = metal::fabs(v_min) > metal::fabs(v_max);");
    w.line("float scale = use_lo_anchor ? scale_abs : -scale_abs;");
    w.line("float edge = use_lo_anchor ? v_min : v_max;");
    w.line("float q0 = round(edge / scale);");
    w.line("bool at_zero = (q0 == 0.0f);");
    w.line("scale = at_zero ? scale : (edge / q0);");
    w.line("float bias = at_zero ? 0.0f : edge;");
    w.line("if (is_v) {");
    w.push_indent();
    w.writeln(format_args!(
        "cache_values_scales[co_meta + g] = {t}(scale);"
    ));
    w.writeln(format_args!(
        "cache_values_biases[co_meta + g] = {t}(bias);"
    ));
    w.pop_indent();
    w.line("} else {");
    w.push_indent();
    w.writeln(format_args!("cache_keys_scales[co_meta + g] = {t}(scale);"));
    w.writeln(format_args!("cache_keys_biases[co_meta + g] = {t}(bias);"));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name_for(fusion: &Fusion, bits: u32) -> String {
    format!(
        "fusekit_rmsnorm_rope_kv_cache_store_decode_q{bits}_f{}",
        fusion.id
    )
}
