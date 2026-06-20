//! MSL emit for `FusionKind::KvCacheStoreDecode`.
//!
//! Stores already-prepared current-token K/V rows into the decode KV
//! cache. Unlike `RopeKvCacheStoreDecode`, this kernel does not rotate:
//! upstream trace-preserved ops have already applied any partial RoPE
//! and concatenation. K and V may have different last dimensions, which
//! is required for MLA-style attention (`qk_dim != v_dim`).

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta, RopeInfo};
use crate::task::ir::PartitionPlan;

#[derive(Clone, Copy)]
enum CurrentOrder {
    HeadSeqDim,
    SeqHeadDim,
}

#[derive(Clone, Copy)]
struct CurrentLayout {
    n_kv: u32,
    seq: u32,
    dim: u32,
    order: CurrentOrder,
}

fn current_layout(shape: &[i32], meta_n_kv: u32) -> Result<CurrentLayout> {
    if shape.len() >= 4 {
        let n = shape.len();
        let a = shape[n - 3].max(1) as u32;
        let b = shape[n - 2].max(1) as u32;
        let dim = shape[n - 1].max(1) as u32;
        // Prepared attention K/V commonly arrives as either
        // `[B, H, T, D]` (MLX attention layout) or `[B, T, H, D]`.
        // Preserve the semantic head axis instead of assuming token-major
        // storage; MLA traces use `[B, H, T, D]`.
        if meta_n_kv > 0 && b == meta_n_kv && a != meta_n_kv {
            return Ok(CurrentLayout {
                n_kv: b,
                seq: a,
                dim,
                order: CurrentOrder::SeqHeadDim,
            });
        }
        return Ok(CurrentLayout {
            n_kv: a,
            seq: b,
            dim,
            order: CurrentOrder::HeadSeqDim,
        });
    }
    if shape.len() == 3 {
        let a = shape[0].max(1) as u32;
        let b = shape[1].max(1) as u32;
        let dim = shape[2].max(1) as u32;
        if meta_n_kv > 0 && b == meta_n_kv && a != meta_n_kv {
            return Ok(CurrentLayout {
                n_kv: b,
                seq: a,
                dim,
                order: CurrentOrder::SeqHeadDim,
            });
        }
        return Ok(CurrentLayout {
            n_kv: a,
            seq: b,
            dim,
            order: CurrentOrder::HeadSeqDim,
        });
    }
    if shape.len() == 2 {
        return Ok(CurrentLayout {
            n_kv: shape[0].max(1) as u32,
            seq: 1,
            dim: shape[1].max(1) as u32,
            order: CurrentOrder::HeadSeqDim,
        });
    }
    bail!("KvCacheStoreDecode: unsupported current K/V shape {shape:?}");
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::KvCacheStoreDecode) {
        bail!("kv_cache_store emitter: wrong kind {:?}", fusion.kind);
    }
    let FusionAttrs::KvCacheStoreDecode {
        k1_rope,
        kv_transpose_split,
    } = &fusion.attrs
    else {
        bail!("kv_cache_store emitter: attrs mismatch");
    };
    if fusion.inputs.len() < 2 {
        bail!("KvCacheStoreDecode expects K and V inputs");
    }

    let meta_n_kv = meta.num_key_value_heads.max(1);
    let split_k = fusion.inputs.len() >= 3;
    let mut k0_layout = current_layout(&fusion.inputs[0].shape, meta_n_kv)?;
    let kv_packed_k0_dim = if split_k { *kv_transpose_split } else { None };
    let kv_packed_dim = k0_layout.dim;
    if let Some(k0_dim) = kv_packed_k0_dim {
        if k0_dim == 0 || k0_dim >= kv_packed_dim {
            bail!(
                "KvCacheStoreDecode packed K/V split has invalid k0_dim {} for packed dim {}",
                k0_dim,
                kv_packed_dim
            );
        }
        k0_layout.dim = k0_dim;
    }
    let k1_layout = if split_k {
        Some(current_layout(&fusion.inputs[1].shape, meta_n_kv)?)
    } else {
        None
    };
    let v_input_idx = if split_k { 2 } else { 1 };
    let mut v_layout = current_layout(&fusion.inputs[v_input_idx].shape, meta_n_kv)?;
    if let Some(k0_dim) = kv_packed_k0_dim {
        if v_layout.n_kv != k0_layout.n_kv
            || v_layout.seq != k0_layout.seq
            || v_layout.dim != kv_packed_dim
            || !matches!(
                (v_layout.order, k0_layout.order),
                (CurrentOrder::HeadSeqDim, CurrentOrder::HeadSeqDim)
                    | (CurrentOrder::SeqHeadDim, CurrentOrder::SeqHeadDim)
            )
        {
            bail!("KvCacheStoreDecode packed K/V aliases have incompatible layouts");
        }
        v_layout.dim = kv_packed_dim - k0_dim;
    }
    if let Some(k1) = k1_layout {
        if (k1.n_kv != 1 && k0_layout.n_kv != k1.n_kv) || k0_layout.seq != k1.seq {
            bail!(
                "KvCacheStoreDecode split-K layout mismatch: k0 n_kv/seq {}:{}, k1 {}:{}",
                k0_layout.n_kv,
                k0_layout.seq,
                k1.n_kv,
                k1.seq
            );
        }
    }
    let n_kv = k0_layout.n_kv;
    let v_n_kv = v_layout.n_kv;
    let k_dim = k0_layout.dim + k1_layout.map(|k| k.dim).unwrap_or(0);
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .and_then(|_| {
            let trace_m = k0_layout.seq.min(v_layout.seq).max(1);
            Some(crate::target::current_chip().active_m(trace_m))
        })
        .unwrap_or_else(|| crate::target::current_chip().active_m_for_unshaped(8))
        .max(1);
    let k0_head_major = matches!(k0_layout.order, CurrentOrder::HeadSeqDim);
    let k1_head_major = k1_layout
        .map(|k| matches!(k.order, CurrentOrder::HeadSeqDim))
        .unwrap_or(k0_head_major);
    let v_head_major = matches!(v_layout.order, CurrentOrder::HeadSeqDim);

    w.line(format!(
        "// Fusion #{}: KvCacheStoreDecode (active_m={}, n_kv={}, v_n_kv={}, k_dim={}, v_dim={}, k_seq={}, v_seq={}, split_k={}, kv_transpose_split={}, k0_head_major={}, k1_head_major={}, v_head_major={})",
        fusion.id,
        active_m,
        n_kv,
        v_n_kv,
        k_dim,
        v_layout.dim,
        k0_layout.seq,
        v_layout.seq,
        split_k,
        kv_packed_k0_dim.is_some(),
        k0_head_major,
        k1_head_major,
        v_head_major
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    if split_k {
        w.line(format!("device const {t} *K0           [[buffer(0)]],"));
        w.line(format!("device const {t} *K1           [[buffer(1)]],"));
        w.line(format!("device const {t} *V            [[buffer(2)]],"));
        w.line(format!("device {t}       *cache_keys   [[buffer(3)]],"));
        w.line(format!("device {t}       *cache_values [[buffer(4)]],"));
        w.line("constant uint       &offset       [[buffer(5)]],");
        w.line("constant uint       &write_offset [[buffer(6)]],");
        w.line("constant uint       &capacity     [[buffer(7)]],");
    } else {
        w.line(format!("device const {t} *K            [[buffer(0)]],"));
        w.line(format!("device const {t} *V            [[buffer(1)]],"));
        w.line(format!("device {t}       *cache_keys   [[buffer(2)]],"));
        w.line(format!("device {t}       *cache_values [[buffer(3)]],"));
        w.line("constant uint       &offset       [[buffer(4)]],");
        w.line("constant uint       &write_offset [[buffer(5)]],");
        w.line("constant uint       &capacity     [[buffer(6)]],");
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.writeln(format_args!("constexpr uint N_KV  = {n_kv};"));
    w.writeln(format_args!("constexpr uint V_N_KV = {v_n_kv};"));
    w.writeln(format_args!(
        "constexpr uint K1_N_KV = {};",
        k1_layout.map(|k| k.n_kv).unwrap_or(0)
    ));
    w.writeln(format_args!("constexpr uint ACTIVE_M = {active_m};"));
    w.writeln(format_args!("constexpr uint K_SEQ = {};", k0_layout.seq));
    w.writeln(format_args!(
        "constexpr uint K1_SEQ = {};",
        k1_layout.map(|k| k.seq).unwrap_or(0)
    ));
    w.writeln(format_args!("constexpr uint V_SEQ = {};", v_layout.seq));
    w.writeln(format_args!("constexpr uint K0_DIM = {};", k0_layout.dim));
    w.writeln(format_args!(
        "constexpr uint KV_PACKED_DIM = {kv_packed_dim};"
    ));
    w.writeln(format_args!(
        "constexpr uint K1_DIM = {};",
        k1_layout.map(|k| k.dim).unwrap_or(0)
    ));
    w.writeln(format_args!("constexpr uint K_DIM = {k_dim};"));
    w.writeln(format_args!("constexpr uint V_DIM = {};", v_layout.dim));
    w.writeln(format_args!(
        "constexpr bool SPLIT_K = {};",
        if split_k { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr bool KV_TRANSPOSE_SPLIT = {};",
        if kv_packed_k0_dim.is_some() {
            "true"
        } else {
            "false"
        }
    ));
    w.line("constexpr uint K0_SRC_DIM = KV_TRANSPOSE_SPLIT ? KV_PACKED_DIM : K0_DIM;");
    w.line("constexpr uint V_SRC_DIM = KV_TRANSPOSE_SPLIT ? KV_PACKED_DIM : V_DIM;");
    w.line("constexpr uint V_SRC_BASE = KV_TRANSPOSE_SPLIT ? K0_DIM : 0u;");
    w.writeln(format_args!(
        "constexpr bool K0_HEAD_MAJOR = {};",
        if k0_head_major { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr bool K1_HEAD_MAJOR = {};",
        if k1_head_major { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr bool V_HEAD_MAJOR = {};",
        if v_head_major { "true" } else { "false" }
    ));
    emit_k1_rope_constants(w, k1_rope.as_ref(), k1_layout.map(|k| k.dim).unwrap_or(0));
    w.line("constexpr uint K_RANGE = N_KV * K_DIM;");
    w.line("constexpr uint V_RANGE = V_N_KV * V_DIM;");
    w.line("constexpr uint TOKEN_RANGE = K_RANGE + V_RANGE;");
    w.blank();

    w.line("uint g = gid * tg_size + tid;");
    w.line("uint row = g / TOKEN_RANGE;");
    w.line("uint local = g - row * TOKEN_RANGE;");
    w.line("if (row >= ACTIVE_M) return;");
    w.line("uint cache_row = (write_offset + row) % capacity;");
    w.line("if (local < K_RANGE) {");
    w.push_indent();
    w.line("uint h = local / K_DIM;");
    w.line("uint d = local - h * K_DIM;");
    w.line("uint dst = h * capacity * K_DIM + cache_row * K_DIM + d;");
    if split_k {
        w.line("if (d < K0_DIM) {");
        w.push_indent();
        w.line("uint src = K0_HEAD_MAJOR ? ((h * K_SEQ + row) * K0_SRC_DIM + d) : ((row * N_KV + h) * K0_SRC_DIM + d);");
        w.line("cache_keys[dst] = K0[src];");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("uint d1 = d - K0_DIM;");
        w.line("uint h1 = (K1_N_KV == 1u) ? 0u : h;");
        w.line("uint src = K1_HEAD_MAJOR ? ((h1 * K1_SEQ + row) * K1_DIM + d1) : ((row * K1_N_KV + h1) * K1_DIM + d1);");
        if k1_rope_active(k1_rope.as_ref(), k1_layout.map(|k| k.dim).unwrap_or(0)).is_some() {
            w.line("if (d1 >= K1_ROPE_DIMS) {");
            w.push_indent();
            w.line("cache_keys[dst] = K1[src];");
            w.pop_indent();
            w.line("} else {");
            w.push_indent();
            w.line("uint pair_idx = K1_ROPE_TRADITIONAL ? (d1 / 2u) : ((d1 < K1_ROPE_HALF) ? d1 : (d1 - K1_ROPE_HALF));");
            w.line("float d_norm = float(pair_idx) / float(K1_ROPE_HALF);");
            if let Some(freqs) = k1_rope
                .as_ref()
                .and_then(|info| info.inline_freqs.as_ref())
                .filter(|freqs| !freqs.is_empty())
            {
                w.writeln(format_args!(
                    "uint freq_idx = (pair_idx < {}u) ? pair_idx : {}u;",
                    freqs.len(),
                    freqs.len().saturating_sub(1)
                ));
                w.line("float inv_freq = K1_ROPE_SCALE / k1_inline_freqs[freq_idx];");
            } else {
                w.line(
                    "float inv_freq = metal::exp2(-d_norm * K1_ROPE_BASE_LOG2) * K1_ROPE_SCALE;",
                );
            }
            w.line("float angle = float(offset + row) * inv_freq;");
            w.line("float c = metal::fast::cos(angle);");
            w.line("float s = metal::fast::sin(angle);");
            w.line("uint lo_idx = K1_ROPE_TRADITIONAL ? (pair_idx * 2u) : pair_idx;");
            w.line(
                "uint hi_idx = K1_ROPE_TRADITIONAL ? (lo_idx + 1u) : (pair_idx + K1_ROPE_HALF);",
            );
            w.line("uint lo_src = K1_HEAD_MAJOR ? ((h1 * K1_SEQ + row) * K1_DIM + lo_idx) : ((row * K1_N_KV + h1) * K1_DIM + lo_idx);");
            w.line("uint hi_src = K1_HEAD_MAJOR ? ((h1 * K1_SEQ + row) * K1_DIM + hi_idx) : ((row * K1_N_KV + h1) * K1_DIM + hi_idx);");
            w.line("float lo = float(K1[lo_src]);");
            w.line("float hi = float(K1[hi_src]);");
            w.line("bool lo_lane = K1_ROPE_TRADITIONAL ? ((d1 & 1u) == 0u) : (d1 < K1_ROPE_HALF);");
            w.line(format!(
                "cache_keys[dst] = {t}(lo_lane ? (lo * c - hi * s) : (hi * c + lo * s));"
            ));
            w.pop_indent();
            w.line("}");
        } else {
            w.line("cache_keys[dst] = K1[src];");
        }
        w.pop_indent();
        w.line("}");
    } else {
        w.line("uint src = K0_HEAD_MAJOR ? ((h * K_SEQ + row) * K_DIM + d) : ((row * N_KV + h) * K_DIM + d);");
        w.line("cache_keys[dst] = K[src];");
    }
    w.pop_indent();
    w.line("} else if (local < TOKEN_RANGE) {");
    w.push_indent();
    w.line("uint vg = local - K_RANGE;");
    w.line("uint h = vg / V_DIM;");
    w.line("uint d = vg - h * V_DIM;");
    w.line("uint src = V_HEAD_MAJOR ? ((h * V_SEQ + row) * V_SRC_DIM + V_SRC_BASE + d) : ((row * V_N_KV + h) * V_SRC_DIM + V_SRC_BASE + d);");
    w.line("uint dst = h * capacity * V_DIM + cache_row * V_DIM + d;");
    w.line("cache_values[dst] = V[src];");
    w.pop_indent();
    w.line("}");

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_kv_cache_store_decode_f{}", fusion.id)
}

fn k1_rope_active(info: Option<&RopeInfo>, k1_dim: u32) -> Option<&RopeInfo> {
    let info = info?;
    if info.dims.min(k1_dim) >= 2 {
        Some(info)
    } else {
        None
    }
}

fn emit_k1_rope_constants(w: &mut MslWriter, info: Option<&RopeInfo>, k1_dim: u32) {
    let active = k1_rope_active(info, k1_dim);
    if let Some(info) = active {
        let dims = info.dims.min(k1_dim);
        let half = dims / 2;
        let log2_base = (info.base as f64).log2() as f32;
        w.line("constexpr bool K1_ROPE      = true;");
        w.writeln(format_args!("constexpr uint K1_ROPE_DIMS = {dims};"));
        w.writeln(format_args!("constexpr uint K1_ROPE_HALF = {half};"));
        w.writeln(format_args!(
            "constexpr float K1_ROPE_BASE_LOG2 = {log2_base:e}f;"
        ));
        w.writeln(format_args!(
            "constexpr float K1_ROPE_SCALE = {};",
            info.scale
        ));
        w.writeln(format_args!(
            "constexpr bool K1_ROPE_TRADITIONAL = {};",
            if info.traditional { "true" } else { "false" }
        ));
        if let Some(freqs) = info.inline_freqs.as_ref().filter(|freqs| !freqs.is_empty()) {
            let vals = freqs
                .iter()
                .map(|v| format!("{v:.9e}f"))
                .collect::<Vec<_>>()
                .join(", ");
            w.line("constexpr bool K1_ROPE_INLINE_FREQS = true;");
            w.writeln(format_args!(
                "const float k1_inline_freqs[{}] = {{ {vals} }};",
                freqs.len()
            ));
        } else {
            w.line("constexpr bool K1_ROPE_INLINE_FREQS = false;");
        }
    } else {
        w.line("constexpr bool K1_ROPE      = false;");
        w.line("constexpr uint K1_ROPE_DIMS = 0;");
        w.line("constexpr uint K1_ROPE_HALF = 0;");
        w.line("constexpr float K1_ROPE_BASE_LOG2 = 0.0f;");
        w.line("constexpr float K1_ROPE_SCALE = 1.0f;");
        w.line("constexpr bool K1_ROPE_TRADITIONAL = false;");
        w.line("constexpr bool K1_ROPE_INLINE_FREQS = false;");
    }
}
