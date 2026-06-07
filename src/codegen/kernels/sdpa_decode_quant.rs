//! Quantized-cache SDPA decode emitter.
//!
//! Pattern-side counterpart to `sdpa_decode.rs` that reads K and V
//! from a 3-tensor packed cache `(packed_uint32, scales, biases)`
//! instead of bf16. mlx-lm's `QuantizedKVCache` produces this layout
//! when `generate(..., kv_bits=4|8)` is set. `bits` is baked into the
//! kernel name (`_q4_` / `_q8_`) so the runtime dispatcher selects
//! the matching kernel for the live cache type.
//!
//! Cache layout (matches `mx.quantize` output):
//! ```text
//!   packed   (B, n_kv_heads, capacity, head_dim / el_per_int)   uint32
//!   scales   (B, n_kv_heads, capacity, head_dim / group_size)   bf16/fp16
//!   biases   (B, n_kv_heads, capacity, head_dim / group_size)   bf16/fp16
//! ```
//! `el_per_int = 32 / bits`. bits=4 → 8 lanes per uint32; bits=8 → 4.
//!
//! Decode-BW saving: for bits=4, K/V cache read shrinks ~3.5× vs bf16
//! (4-bit + small fp16 per-group metadata). At growing kv_len this
//! dominates SDPA stages 1+3.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

const MAX_KV_LEN: u32 = 4096;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    bits: u32,
    group_size: u32,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::SdpaDecode) {
        bail!("sdpa_decode_quant emitter: wrong kind {:?}", fusion.kind);
    }
    let FusionAttrs::SdpaDecode { scale, .. } = fusion.attrs else {
        bail!("sdpa_decode_quant emitter: attrs mismatch");
    };
    if !matches!(bits, 4 | 8) {
        bail!("sdpa_decode_quant: only bits ∈ {{4, 8}} supported, got {bits}");
    }

    let n_heads = meta.num_attention_heads;
    let n_kv_heads = meta.num_key_value_heads;
    let q_shape = fusion
        .inputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let v_shape = fusion
        .inputs
        .get(2)
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let out_shape = fusion
        .outputs
        .first()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let inputless_semantic_decode = fusion.inputs.is_empty();
    let qk_dim = if inputless_semantic_decode {
        meta.head_dim
    } else {
        q_shape
            .last()
            .copied()
            .unwrap_or(meta.head_dim as i32)
            .max(1) as u32
    };
    let v_dim = if inputless_semantic_decode {
        meta.head_dim
    } else {
        v_shape
            .last()
            .copied()
            .or_else(|| out_shape.last().copied())
            .unwrap_or(meta.head_dim as i32)
            .max(1) as u32
    };
    let heads_per_kv = n_heads / n_kv_heads.max(1);
    let scale = if scale.abs() < f32::EPSILON {
        1.0f32 / (qk_dim as f32).sqrt()
    } else {
        scale
    };
    let t = crate::codegen::kernels::act_ty(meta);
    let el_per_int = 32 / bits;
    let n_words_k = qk_dim / el_per_int;
    let n_words_v = v_dim / el_per_int;
    let n_groups_k = qk_dim / group_size.max(1);
    let n_groups_v = v_dim / group_size.max(1);
    let mask = (1u32 << bits) - 1;
    let direct_scaled_q = (scale - 1.0).abs() < f32::EPSILON && !inputless_semantic_decode;
    let active_m = out_shape
        .get(0..)
        .filter(|s| !s.is_empty())
        .map(|s| crate::target::current_chip().active_tokens_for_shape(s))
        .unwrap_or_else(|| crate::target::current_chip().active_m(8))
        .max(1);
    let q_head_major = inputless_semantic_decode
        || direct_scaled_q
        || infer_q_head_major(q_shape, n_heads, active_m, qk_dim);
    let q_seq = if q_head_major {
        trace_q_head_stride(q_shape).max(1)
    } else {
        1
    };
    let out_seq = active_m;
    let flash_decoding = super::sdpa_decode::flash_decoding_enabled();
    let layer_idx = fusion
        .layer_hint
        .and_then(|t| match t {
            crate::graph::LayerTag::Layer(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0);

    let fn_name = function_name_for(fusion, bits);
    w.line(format!(
        "// Fusion #{}: SdpaDecode quant b{} (n_heads={}, n_kv={}, qk_dim={}, v_dim={}, group_size={}, flash_decoding={})",
        fusion.id, bits, n_heads, n_kv_heads, qk_dim, v_dim, group_size, flash_decoding,
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t}   *Q          [[buffer(0)]],"));
    w.line("device const uint  *K_packed   [[buffer(1)]],");
    w.line(format!("device const {t}   *K_scales   [[buffer(2)]],"));
    w.line(format!("device const {t}   *K_biases   [[buffer(3)]],"));
    w.line("device const uint  *V_packed   [[buffer(4)]],");
    w.line(format!("device const {t}   *V_scales   [[buffer(5)]],"));
    w.line(format!("device const {t}   *V_biases   [[buffer(6)]],"));
    w.line("constant uint       &kv_len     [[buffer(7)]],");
    w.line("constant uint       &capacity   [[buffer(8)]],");
    w.line(format!("device {t}         *out        [[buffer(9)]],"));
    if flash_decoding {
        w.line("device float       *partials   [[buffer(10)]],");
        w.line("device atomic_uint *counter    [[buffer(11)]],");
    }
    w.line("uint tid       [[thread_position_in_threadgroup]],");
    w.line("uint tg_size   [[threads_per_threadgroup]],");
    w.line("uint head_id   [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint QK_DIM       = {qk_dim};"));
    w.writeln(format_args!("constexpr uint V_DIM        = {v_dim};"));
    w.writeln(format_args!("constexpr uint Q_SEQ        = {q_seq};"));
    w.writeln(format_args!("constexpr uint OUT_SEQ      = {out_seq};"));
    w.writeln(format_args!("constexpr uint N_HEADS      = {n_heads};"));
    w.writeln(format_args!("constexpr uint N_KV_HEADS   = {n_kv_heads};"));
    w.writeln(format_args!(
        "constexpr bool Q_HEAD_MAJOR = {};",
        if q_head_major { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr uint HEADS_PER_KV = {heads_per_kv};"
    ));
    w.writeln(format_args!("constexpr float SCALE       = {scale};"));
    w.writeln(format_args!("constexpr uint MAX_KV_LEN   = {MAX_KV_LEN};"));
    w.writeln(format_args!(
        "constexpr uint BLOCK        = {};",
        super::sdpa_decode::FLASH_BLOCK
    ));
    w.writeln(format_args!("constexpr uint BITS         = {bits};"));
    w.writeln(format_args!("constexpr uint EL_PER_INT   = {el_per_int};"));
    w.writeln(format_args!("constexpr uint N_WORDS_K    = {n_words_k};"));
    w.writeln(format_args!("constexpr uint N_WORDS_V    = {n_words_v};"));
    w.writeln(format_args!("constexpr uint N_GROUPS_K   = {n_groups_k};"));
    w.writeln(format_args!("constexpr uint N_GROUPS_V   = {n_groups_v};"));
    w.writeln(format_args!("constexpr uint GROUP_SIZE   = {group_size};"));
    w.writeln(format_args!("constexpr uint MASK         = {mask};"));
    if flash_decoding {
        w.writeln(format_args!(
            "constexpr uint KV_CHUNK     = {};",
            super::sdpa_decode::FLASH_DEC_CHUNK
        ));
        w.writeln(format_args!(
            "constexpr uint MAX_CHUNKS   = {};",
            super::sdpa_decode::FLASH_DEC_MAX_CHUNKS
        ));
        w.writeln(format_args!("constexpr uint LAYER_IDX    = {layer_idx};"));
        w.writeln(format_args!(
            "constexpr uint PARTIALS_STRIDE_PER_LAYER = OUT_SEQ * N_HEADS * MAX_CHUNKS * (V_DIM + 2u);"
        ));
        w.writeln(format_args!(
            "constexpr uint PARTIALS_STRIDE_PER_HEAD  = MAX_CHUNKS * (V_DIM + 2u);"
        ));
        w.writeln(format_args!(
            "constexpr uint COUNTER_STRIDE_PER_LAYER  = OUT_SEQ * N_HEADS;"
        ));
    }
    w.blank();

    if flash_decoding {
        emit_flash_decoding(w, t, bits);
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }

    w.line("uint query_row = head_id / N_HEADS;");
    w.line("uint head_id_real = head_id - query_row * N_HEADS;");
    w.line("if (query_row >= OUT_SEQ) return;");
    w.line("uint kv_len_eff = kv_len + query_row;");
    w.line("uint kv_head = head_id_real / HEADS_PER_KV;");
    w.line("threadgroup float scores[MAX_KV_LEN];");
    w.line("threadgroup float reduce_buf[32];");
    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.blank();

    w.line("threadgroup float q_local[QK_DIM];");
    w.line(format!(
        "device const {t}* q_base = Q + ((Q_HEAD_MAJOR ? (head_id_real * OUT_SEQ + query_row) : ((query_row * N_HEADS + head_id_real) * Q_SEQ)) * QK_DIM);"
    ));
    w.line("for (uint d = tid; d < QK_DIM; d += tg_size) {");
    w.push_indent();
    w.line("q_local[d] = float(q_base[d]);");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // Stage 1: scores[kv] = Q · dequant(K[kv]) * scale.
    // Q is staged once into SMEM, so long-context decode no longer
    // re-reads the same query row for every kv position. K remains
    // packed and is dequantized in-register per word.
    w.line("// Stage 1: dot product Q · dequant(K[kv]) per kv position.");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.writeln(format_args!(
        "device const {t}4* q_row4 = (device const {t}4*)(Q + ((Q_HEAD_MAJOR ? (head_id_real * OUT_SEQ + query_row) : ((query_row * N_HEADS + head_id_real) * Q_SEQ)) * QK_DIM));"
    ));
    w.line("uint kv_off = (kv_head * capacity + kv);");
    w.line("device const uint *kp = K_packed + kv_off * N_WORDS_K;");
    w.line(format!(
        "device const {t} *ks = K_scales + kv_off * N_GROUPS_K;"
    ));
    w.line(format!(
        "device const {t} *kb = K_biases + kv_off * N_GROUPS_K;"
    ));
    w.line("float dot = 0.0f;");
    if n_words_k % 4 == 0 {
        w.line("device const uint4 *kp4 = (device const uint4*)kp;");
        w.line("for (uint wi4 = 0u; wi4 < N_WORDS_K / 4u; ++wi4) {");
        w.push_indent();
        w.line("uint4 words = kp4[wi4];");
        for lane in 0..4 {
            let comp = ["x", "y", "z", "w"][lane as usize];
            w.writeln(format_args!("uint word{lane} = words.{comp};"));
            w.writeln(format_args!(
                "uint d_base{lane} = (wi4 * 4u + {lane}u) * EL_PER_INT;"
            ));
            w.writeln(format_args!("uint g{lane} = d_base{lane} / GROUP_SIZE;"));
            w.writeln(format_args!("float s{lane} = float(ks[g{lane}]);"));
            w.writeln(format_args!("float b{lane} = float(kb[g{lane}]);"));
            if bits == 4 {
                for i in 0..8 {
                    let shift = i * 4;
                    w.writeln(format_args!(
                        "dot += q_local[d_base{lane} + {i}u] * (float((word{lane} >> {shift}u) & 0xFu) * s{lane} + b{lane});"
                    ));
                }
            } else {
                for i in 0..4 {
                    let shift = i * 8;
                    w.writeln(format_args!(
                        "dot += q_local[d_base{lane} + {i}u] * (float((word{lane} >> {shift}u) & 0xFFu) * s{lane} + b{lane});"
                    ));
                }
            }
        }
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint wi = 0u; wi < N_WORDS_K; ++wi) {");
        w.push_indent();
        w.line("uint word = kp[wi];");
        w.line("uint d_base = wi * EL_PER_INT;");
        w.line("uint g = d_base / GROUP_SIZE;");
        w.line("float s = float(ks[g]);");
        w.line("float b = float(kb[g]);");
        // Explicit unroll per bits — same hoisting principle as weight GEMV.
        if bits == 4 {
            w.line("dot += q_local[d_base + 0u] * (float((word      ) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 1u] * (float((word >>  4u) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 2u] * (float((word >>  8u) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 3u] * (float((word >> 12u) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 4u] * (float((word >> 16u) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 5u] * (float((word >> 20u) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 6u] * (float((word >> 24u) & 0xFu) * s + b);");
            w.line("dot += q_local[d_base + 7u] * (float((word >> 28u) & 0xFu) * s + b);");
        } else {
            // bits == 8
            w.line("dot += q_local[d_base + 0u] * (float((word      ) & 0xFFu) * s + b);");
            w.line("dot += q_local[d_base + 1u] * (float((word >>  8u) & 0xFFu) * s + b);");
            w.line("dot += q_local[d_base + 2u] * (float((word >> 16u) & 0xFFu) * s + b);");
            w.line("dot += q_local[d_base + 3u] * (float((word >> 24u) & 0xFFu) * s + b);");
        }
        w.pop_indent();
        w.line("}");
    }
    w.line("scores[kv] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // Stage 2a/b — softmax (operates on `scores` SMEM). Same as bf16 path.
    w.line("float t_max = -INFINITY;");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line("t_max = max(t_max, scores[kv]);");
    w.pop_indent();
    w.line("}");
    w.line("t_max = simd_max(t_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_max = reduce_buf[0];");
    w.line("float t_sum = 0.0f;");
    w.line("for (uint kv = tid; kv < kv_len_eff; kv += tg_size) {");
    w.push_indent();
    w.line("scores[kv] = metal::precise::exp(scores[kv] - global_max);");
    w.line("t_sum += scores[kv];");
    w.pop_indent();
    w.line("}");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < tg_size / 32u) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float global_sum = reduce_buf[0];");
    w.blank();

    // Stage 2c: keep `scores` as unnormalised exp values.  Stage 3
    // multiplies by `inv_sum` at the consumer, avoiding an extra
    // scores write/read pass and a threadgroup barrier.
    w.line("float inv_sum = 1.0f / global_sum;");
    w.blank();

    // Stage 3: simdgroup-cooperative weighted sum of dequant(V).
    // One SG now owns one packed V word, not one output dimension.
    // That fans out a single V_packed/scales/biases load to 4 or 8
    // output dims and removes repeated reads of the same cache word.
    w.line("// Stage 3: weighted sum of dequant(V), packed-word fanout.");
    w.line("for (uint wi = simd_id; wi < N_WORDS_V; wi += sgs_per_tg) {");
    w.push_indent();
    w.line("uint d_base = wi * EL_PER_INT;");
    w.line("uint g  = d_base / GROUP_SIZE;");
    if bits == 4 {
        for i in 0..8 {
            w.writeln(format_args!("float partial{i} = 0.0f;"));
        }
    } else {
        for i in 0..4 {
            w.writeln(format_args!("float partial{i} = 0.0f;"));
        }
    }
    w.line("for (uint kv = simd_lane; kv < kv_len_eff; kv += 32u) {");
    w.push_indent();
    w.line("uint kv_off = (kv_head * capacity + kv);");
    w.line("uint word = V_packed[kv_off * N_WORDS_V + wi];");
    w.line("float s = float(V_scales[kv_off * N_GROUPS_V + g]);");
    w.line("float b = float(V_biases[kv_off * N_GROUPS_V + g]);");
    w.line("float score = scores[kv] * inv_sum;");
    if bits == 4 {
        for i in 0..8 {
            let shift = i * 4;
            w.writeln(format_args!(
                "partial{i} += score * (float((word >> {shift}u) & 0xFu) * s + b);"
            ));
        }
    } else {
        for i in 0..4 {
            let shift = i * 8;
            w.writeln(format_args!(
                "partial{i} += score * (float((word >> {shift}u) & 0xFFu) * s + b);"
            ));
        }
    }
    w.pop_indent();
    w.line("}");
    if bits == 4 {
        for i in 0..8 {
            w.writeln(format_args!("float acc{i} = simd_sum(partial{i});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("uint out_base = (query_row * N_HEADS + head_id_real) * V_DIM + d_base;");
        for i in 0..8 {
            w.writeln(format_args!("out[out_base + {i}u] = {t}(acc{i});"));
        }
        w.pop_indent();
        w.line("}");
    } else {
        for i in 0..4 {
            w.writeln(format_args!("float acc{i} = simd_sum(partial{i});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("uint out_base = (query_row * N_HEADS + head_id_real) * V_DIM + d_base;");
        for i in 0..4 {
            w.writeln(format_args!("out[out_base + {i}u] = {t}(acc{i});"));
        }
        w.pop_indent();
        w.line("}");
    }
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

/// Quantized-KV Flash-Decoding. Dense SDPA already has the chunked
/// decomposition; this variant keeps K/V packed and dequantizes inside
/// the chunk kernel. Stage 3 is word-owned rather than dim-owned so one
/// V packed/scales/biases load fans out to all values in that word.
fn emit_flash_decoding(w: &mut MslWriter, t: &str, bits: u32) {
    let lanes = 32 / bits;
    w.line("// Quant Flash-Decoding: per-(head, chunk) TG with packed K/V dequant.");
    w.line("uint gid_compound = head_id;");
    w.line("uint query_head   = gid_compound / MAX_CHUNKS;");
    w.line("uint chunk_id     = gid_compound % MAX_CHUNKS;");
    w.line("uint query_row    = query_head / N_HEADS;");
    w.line("uint head_id_real = query_head - query_row * N_HEADS;");
    w.line("if (query_row >= OUT_SEQ) return;");
    w.line("uint kv_head      = head_id_real / HEADS_PER_KV;");
    w.line("uint kv_len_eff   = kv_len + query_row;");
    w.line("uint kv_start     = chunk_id * KV_CHUNK;");
    w.line("uint active_chunks = (kv_len_eff + KV_CHUNK - 1u) / KV_CHUNK;");
    w.line("if (kv_start >= kv_len_eff) return;");
    w.line("uint kv_end = min(kv_start + KV_CHUNK, kv_len_eff);");
    w.blank();

    w.line("uint simd_lane = tid % 32u;");
    w.line("uint simd_id   = tid / 32u;");
    w.line("uint sgs_per_tg = tg_size / 32u;");
    w.line("threadgroup float q_local[QK_DIM];");
    w.line("threadgroup float s_block[BLOCK];");
    w.line("threadgroup float reduce_buf[32];");
    w.line(format!(
        "device const {t}* q_base = Q + ((Q_HEAD_MAJOR ? (head_id_real * OUT_SEQ + query_row) : ((query_row * N_HEADS + head_id_real) * Q_SEQ)) * QK_DIM);"
    ));
    w.line("for (uint d = tid; d < QK_DIM; d += tg_size) {");
    w.push_indent();
    w.line("q_local[d] = float(q_base[d]);");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    w.line("bool owns_word = tid < N_WORDS_V;");
    for i in 0..lanes {
        w.writeln(format_args!("float o{i} = 0.0f;"));
    }
    w.line("float m = -INFINITY;");
    w.line("float l = 0.0f;");
    w.blank();

    w.line("for (uint block = kv_start; block < kv_end; block += BLOCK) {");
    w.push_indent();
    w.line("uint block_end_inner = min(block + BLOCK, kv_end);");
    w.line("uint block_size = block_end_inner - block;");
    w.blank();

    w.line("// Stage 1: one thread computes one packed-K row dot within the block.");
    w.line("if (tid < block_size) {");
    w.push_indent();
    w.line("uint kv = block + tid;");
    w.line("uint kv_off = (kv_head * capacity + kv);");
    w.line("device const uint *kp = K_packed + kv_off * N_WORDS_K;");
    w.line(format!(
        "device const {t} *ks = K_scales + kv_off * N_GROUPS_K;"
    ));
    w.line(format!(
        "device const {t} *kb = K_biases + kv_off * N_GROUPS_K;"
    ));
    w.line("float dot = 0.0f;");
    w.line("for (uint wi = 0u; wi < N_WORDS_K; ++wi) {");
    w.push_indent();
    w.line("uint word = kp[wi];");
    w.line("uint d_base = wi * EL_PER_INT;");
    w.line("uint g = d_base / GROUP_SIZE;");
    w.line("float s = float(ks[g]);");
    w.line("float b = float(kb[g]);");
    if bits == 4 {
        for i in 0..8 {
            let shift = i * 4;
            w.writeln(format_args!(
                "dot += q_local[d_base + {i}u] * (float((word >> {shift}u) & 0xFu) * s + b);"
            ));
        }
    } else {
        for i in 0..4 {
            let shift = i * 8;
            w.writeln(format_args!(
                "dot += q_local[d_base + {i}u] * (float((word >> {shift}u) & 0xFFu) * s + b);"
            ));
        }
    }
    w.pop_indent();
    w.line("}");
    w.line("s_block[tid] = dot * SCALE;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    w.line("float t_max = (tid < block_size) ? s_block[tid] : -INFINITY;");
    w.line("t_max = simd_max(t_max);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_max;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : -INFINITY;");
    w.line("v = simd_max(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float m_block = reduce_buf[0];");
    w.line("float m_new = max(m, m_block);");
    w.line("float alpha = metal::precise::exp(m - m_new);");
    w.blank();

    w.line("if (tid < block_size) s_block[tid] = metal::precise::exp(s_block[tid] - m_new);");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float t_sum = (tid < block_size) ? s_block[tid] : 0.0f;");
    w.line("t_sum = simd_sum(t_sum);");
    w.line("if (simd_lane == 0u) reduce_buf[simd_id] = t_sum;");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("if (simd_id == 0u) {");
    w.push_indent();
    w.line("float v = (tid < sgs_per_tg) ? reduce_buf[tid] : 0.0f;");
    w.line("v = simd_sum(v);");
    w.line("if (simd_lane == 0u) reduce_buf[0] = v;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("float l_block = reduce_buf[0];");
    w.line("l = l * alpha + l_block;");
    w.blank();

    w.line("// Stage 3: one thread owns one packed V word and fans out its lanes.");
    w.line("if (owns_word) {");
    w.push_indent();
    w.line("uint wi = tid;");
    w.line("uint d_base = wi * EL_PER_INT;");
    w.line("uint g = d_base / GROUP_SIZE;");
    for i in 0..lanes {
        w.writeln(format_args!("float contrib{i} = 0.0f;"));
    }
    w.line("for (uint i = 0u; i < block_size; ++i) {");
    w.push_indent();
    w.line("uint kv = block + i;");
    w.line("uint kv_off = (kv_head * capacity + kv);");
    w.line("uint word = V_packed[kv_off * N_WORDS_V + wi];");
    w.line(format!(
        "float s = float(V_scales[kv_off * N_GROUPS_V + g]);"
    ));
    w.line(format!(
        "float b = float(V_biases[kv_off * N_GROUPS_V + g]);"
    ));
    w.line("float score = s_block[i];");
    if bits == 4 {
        for i in 0..8 {
            let shift = i * 4;
            w.writeln(format_args!(
                "contrib{i} += score * (float((word >> {shift}u) & 0xFu) * s + b);"
            ));
        }
    } else {
        for i in 0..4 {
            let shift = i * 8;
            w.writeln(format_args!(
                "contrib{i} += score * (float((word >> {shift}u) & 0xFFu) * s + b);"
            ));
        }
    }
    w.pop_indent();
    w.line("}");
    for i in 0..lanes {
        w.writeln(format_args!("o{i} = o{i} * alpha + contrib{i};"));
    }
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("m = m_new;");
    w.pop_indent();
    w.line("}");
    w.blank();

    w.line("uint query_head_flat = query_row * N_HEADS + head_id_real;");
    w.line("uint partial_base = LAYER_IDX * PARTIALS_STRIDE_PER_LAYER");
    w.line("                  + query_head_flat * PARTIALS_STRIDE_PER_HEAD");
    w.line("                  + chunk_id * (V_DIM + 2u);");
    w.line("if (tid == 0u) {");
    w.push_indent();
    w.line("partials[partial_base + 0u] = m;");
    w.line("partials[partial_base + 1u] = l;");
    w.pop_indent();
    w.line("}");
    w.line("if (owns_word) {");
    w.push_indent();
    w.line("uint d_base = tid * EL_PER_INT;");
    for i in 0..lanes {
        w.writeln(format_args!("partials[partial_base + 2u + d_base + {i}u] = o{i};"));
    }
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_device);");
    w.blank();

    w.line("threadgroup uint prev_count_smem[1];");
    w.line("if (tid == 0u) {");
    w.push_indent();
    w.line("uint counter_idx = LAYER_IDX * COUNTER_STRIDE_PER_LAYER + query_head_flat;");
    w.line("uint prev = atomic_fetch_add_explicit(&counter[counter_idx], 1u, memory_order_relaxed);");
    w.line("prev_count_smem[0] = prev;");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.line("uint prev_count = prev_count_smem[0];");
    w.line("if (prev_count + 1u != active_chunks) return;");
    w.blank();

    w.line("if (tid == 0u) {");
    w.push_indent();
    w.line("uint counter_idx = LAYER_IDX * COUNTER_STRIDE_PER_LAYER + query_head_flat;");
    w.line("atomic_store_explicit(&counter[counter_idx], 0u, memory_order_relaxed);");
    w.pop_indent();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_device);");
    w.blank();

    w.line("if (owns_word) {");
    w.push_indent();
    w.line("uint d_base = tid * EL_PER_INT;");
    w.line("float final_m = -INFINITY;");
    w.line("float final_l = 0.0f;");
    for i in 0..lanes {
        w.writeln(format_args!("float final_o{i} = 0.0f;"));
    }
    w.line("for (uint c = 0u; c < active_chunks; ++c) {");
    w.push_indent();
    w.line("uint c_base = LAYER_IDX * PARTIALS_STRIDE_PER_LAYER");
    w.line("            + query_head_flat * PARTIALS_STRIDE_PER_HEAD");
    w.line("            + c * (V_DIM + 2u);");
    w.line("float m_c = partials[c_base + 0u];");
    w.line("float l_c = partials[c_base + 1u];");
    w.line("float m_new = max(final_m, m_c);");
    w.line("float alpha_c = metal::precise::exp(final_m - m_new);");
    w.line("float beta_c  = metal::precise::exp(m_c       - m_new);");
    for i in 0..lanes {
        w.writeln(format_args!(
            "float o_c{i} = partials[c_base + 2u + d_base + {i}u];"
        ));
    }
    w.line("final_l = final_l * alpha_c + l_c * beta_c;");
    for i in 0..lanes {
        w.writeln(format_args!("final_o{i} = final_o{i} * alpha_c + o_c{i} * beta_c;"));
    }
    w.line("final_m = m_new;");
    w.pop_indent();
    w.line("}");
    w.line("uint out_base = (query_row * N_HEADS + head_id_real) * V_DIM + d_base;");
    for i in 0..lanes {
        w.writeln(format_args!("out[out_base + {i}u] = {t}(final_o{i} / final_l);"));
    }
    w.pop_indent();
    w.line("}");
}

pub fn function_name_for(fusion: &Fusion, bits: u32) -> String {
    format!("fusekit_sdpa_decode_q{bits}_f{}", fusion.id)
}

fn trace_q_head_stride(q_shape: &[i32]) -> u32 {
    if q_shape.len() >= 3 {
        q_shape[q_shape.len() - 2].max(1) as u32
    } else {
        1
    }
}

fn infer_q_head_major(q_shape: &[i32], n_heads: u32, active_m: u32, qk_dim: u32) -> bool {
    if q_shape.len() < 3 {
        return false;
    }
    let Some(&last) = q_shape.last() else {
        return false;
    };
    if last.max(1) as u32 != qk_dim {
        return false;
    }
    if trace_q_head_stride(q_shape).max(1) != active_m.max(1) {
        return false;
    }
    let head_prefix = q_shape[..q_shape.len() - 2]
        .iter()
        .map(|&d| d.max(1) as u64)
        .product::<u64>();
    head_prefix == n_heads.max(1) as u64
}
