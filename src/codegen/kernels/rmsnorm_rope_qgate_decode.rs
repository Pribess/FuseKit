//! Qwen3Next-style decode query prep:
//! q_proj emits `[query, gate]` per attention head. This kernel applies
//! per-head RMSNorm + partial RoPE to the query half and materializes the
//! gate half as the flat vector consumed by the post-SDPA sigmoid gate.

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
    if !matches!(fusion.kind, FusionKind::RmsNormRopeQGateDecode) {
        bail!(
            "rmsnorm_rope_qgate_decode emitter: wrong kind {:?}",
            fusion.kind
        );
    }
    let FusionAttrs::RmsNormRopeQGateDecode {
        eps,
        theta,
        rope_dims,
    } = fusion.attrs
    else {
        bail!("rmsnorm_rope_qgate_decode emitter: attrs mismatch");
    };

    let n_heads = meta.num_attention_heads;
    let head_dim = meta.head_dim;
    let dims = rope_dims.min(head_dim);
    let half = (dims / 2).max(1);
    let t = crate::codegen::kernels::act_ty(meta);
    let log2_theta = (theta as f64).log2() as f32;
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_tokens_for_shape(&b.shape))
        .unwrap_or_else(|| crate::target::current_chip().active_m(8))
        .max(1);

    w.line(format!(
        "// Fusion #{}: RmsNormRopeQGateDecode (active_m={}, n_heads={}, head_dim={}, rope_dims={}, theta={})",
        fusion.id, active_m, n_heads, head_dim, dims, theta
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *q_proj   [[buffer(0)]],"));
    w.line(format!("device const {t} *q_weight [[buffer(1)]],"));
    w.line("constant uint       &offset   [[buffer(2)]],");
    w.line(format!("device {t}       *q_out    [[buffer(3)]],"));
    w.line(format!("device {t}       *gate_out [[buffer(4)]],"));
    w.line("uint tid       [[thread_position_in_threadgroup]],");
    w.line("uint tg_size   [[threads_per_threadgroup]],");
    w.line("uint head_id   [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint N_HEADS   = {n_heads};"));
    w.writeln(format_args!("constexpr uint ACTIVE_M  = {active_m};"));
    w.writeln(format_args!("constexpr uint HEAD_DIM  = {head_dim};"));
    w.writeln(format_args!("constexpr uint Q_STRIDE  = HEAD_DIM * 2u;"));
    w.writeln(format_args!("constexpr uint ROPE_DIMS = {dims};"));
    w.writeln(format_args!("constexpr uint HALF      = {half};"));
    w.writeln(format_args!("constexpr float EPS      = {eps:e}f;"));
    w.writeln(format_args!("constexpr float BASE_LOG2 = {log2_theta:e}f;"));
    w.line("threadgroup float reduce_buf[32];");
    w.line("threadgroup float local_inv_mean[1];");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.blank();

    w.line("float sum_sq = 0.0f;");
    w.line("uint row = head_id / N_HEADS;");
    w.line("uint h = head_id - row * N_HEADS;");
    w.line("if (row >= ACTIVE_M) return;");
    w.line("uint q_base = (row * N_HEADS + h) * Q_STRIDE;");
    w.line("constexpr uint RMS_N_READS = 4u;");
    w.line("uint read_base = tid * RMS_N_READS;");
    w.line("for (uint j = 0u; j < RMS_N_READS; ++j) {");
    w.push_indent();
    w.line("uint d = read_base + j;");
    w.line("if (d >= HEAD_DIM) continue;");
    w.line("float x = float(q_proj[q_base + d]);");
    w.line("sum_sq += x * x;");
    w.pop_indent();
    w.line("}");
    w.line("sum_sq = simd_sum(sum_sq);");
    w.line("if (simd_id == 0u) reduce_buf[simd_lane] = 0.0f;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = sum_sq;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) local_inv_mean[0] = metal::precise::rsqrt(v / float(HEAD_DIM) + EPS);");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float rms_scale = local_inv_mean[0];");
    w.blank();

    w.line("for (uint j = 0u; j < RMS_N_READS; ++j) {");
    w.push_indent();
    w.line("uint d = read_base + j;");
    w.line("if (d >= HEAD_DIM) continue;");
    w.writeln(format_args!(
        "float val = float({t}(q_weight[d] * {t}(float(q_proj[q_base + d]) * rms_scale)));"
    ));
    w.line("if (d < ROPE_DIMS) {");
    w.push_indent();
    w.line("uint pair_idx = (d < HALF) ? d : (d - HALF);");
    w.writeln(format_args!(
        "float lo = float({t}(q_weight[pair_idx] * {t}(float(q_proj[q_base + pair_idx]) * rms_scale)));"
    ));
    w.writeln(format_args!(
        "float hi = float({t}(q_weight[pair_idx + HALF] * {t}(float(q_proj[q_base + pair_idx + HALF]) * rms_scale)));"
    ));
    w.line("float d_norm = float(pair_idx) / float(HALF);");
    w.line("float inv_freq = metal::exp2(-d_norm * BASE_LOG2);");
    w.line("float angle = float(offset + row) * inv_freq;");
    w.line("float c = metal::fast::cos(angle);");
    w.line("float s = metal::fast::sin(angle);");
    w.line("val = (d < HALF) ? (lo * c - hi * s) : (hi * c + lo * s);");
    w.pop_indent();
    w.line("}");
    w.line("uint q_out_idx = (h * ACTIVE_M + row) * HEAD_DIM + d;");
    w.line("uint gate_idx = (row * N_HEADS + h) * HEAD_DIM + d;");
    w.writeln(format_args!("q_out[q_out_idx] = {t}(val);"));
    w.line("gate_out[gate_idx] = q_proj[q_base + HEAD_DIM + d];");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_rmsnorm_rope_qgate_decode_f{}", fusion.id)
}
