//! MSL emit for `FusionKind::KvCacheStoreDecode`.
//!
//! Stores already-prepared current-token K/V rows into the decode KV
//! cache. Unlike `RopeKvCacheStoreDecode`, this kernel does not rotate:
//! upstream trace-preserved ops have already applied any partial RoPE
//! and concatenation. K and V may have different last dimensions, which
//! is required for MLA-style attention (`qk_dim != v_dim`).

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
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
    if !matches!(fusion.attrs, FusionAttrs::KvCacheStoreDecode) {
        bail!("kv_cache_store emitter: attrs mismatch");
    }
    if fusion.inputs.len() < 2 {
        bail!("KvCacheStoreDecode expects K and V inputs");
    }

    let meta_n_kv = meta.num_key_value_heads.max(1);
    let k_layout = current_layout(&fusion.inputs[0].shape, meta_n_kv)?;
    let v_layout = current_layout(&fusion.inputs[1].shape, meta_n_kv)?;
    if k_layout.n_kv != v_layout.n_kv {
        bail!(
            "KvCacheStoreDecode K/V head count mismatch: {} vs {}",
            k_layout.n_kv,
            v_layout.n_kv
        );
    }
    let n_kv = k_layout.n_kv;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .and_then(|_| {
            let trace_m = k_layout.seq.min(v_layout.seq).max(1);
            Some(crate::target::current_chip().active_m(trace_m))
        })
        .unwrap_or_else(|| crate::target::current_chip().active_m_for_unshaped(8))
        .max(1);
    let k_head_major = matches!(k_layout.order, CurrentOrder::HeadSeqDim);
    let v_head_major = matches!(v_layout.order, CurrentOrder::HeadSeqDim);

    w.line(format!(
        "// Fusion #{}: KvCacheStoreDecode (active_m={}, n_kv={}, k_dim={}, v_dim={}, k_seq={}, v_seq={}, k_head_major={}, v_head_major={})",
        fusion.id,
        active_m,
        n_kv,
        k_layout.dim,
        v_layout.dim,
        k_layout.seq,
        v_layout.seq,
        k_head_major,
        v_head_major
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *K            [[buffer(0)]],"));
    w.line(format!("device const {t} *V            [[buffer(1)]],"));
    w.line(format!("device {t}       *cache_keys   [[buffer(2)]],"));
    w.line(format!("device {t}       *cache_values [[buffer(3)]],"));
    w.line("constant uint       &offset       [[buffer(4)]],");
    w.line("constant uint       &capacity     [[buffer(5)]],");
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.writeln(format_args!("constexpr uint N_KV  = {n_kv};"));
    w.writeln(format_args!("constexpr uint ACTIVE_M = {active_m};"));
    w.writeln(format_args!("constexpr uint K_SEQ = {};", k_layout.seq));
    w.writeln(format_args!("constexpr uint V_SEQ = {};", v_layout.seq));
    w.writeln(format_args!("constexpr uint K_DIM = {};", k_layout.dim));
    w.writeln(format_args!("constexpr uint V_DIM = {};", v_layout.dim));
    w.writeln(format_args!(
        "constexpr bool K_HEAD_MAJOR = {};",
        if k_head_major { "true" } else { "false" }
    ));
    w.writeln(format_args!(
        "constexpr bool V_HEAD_MAJOR = {};",
        if v_head_major { "true" } else { "false" }
    ));
    w.line("constexpr uint K_RANGE = N_KV * K_DIM;");
    w.line("constexpr uint TOKEN_RANGE = K_RANGE + N_KV * V_DIM;");
    w.blank();

    w.line("uint g = gid * tg_size + tid;");
    w.line("uint row = g / TOKEN_RANGE;");
    w.line("uint local = g - row * TOKEN_RANGE;");
    w.line("if (row >= ACTIVE_M) return;");
    w.line("if (local < K_RANGE) {");
    w.push_indent();
    w.line("uint h = local / K_DIM;");
    w.line("uint d = local - h * K_DIM;");
    w.line("uint src = K_HEAD_MAJOR ? ((h * K_SEQ + row) * K_DIM + d) : ((row * N_KV + h) * K_DIM + d);");
    w.line("uint dst = h * capacity * K_DIM + (offset + row) * K_DIM + d;");
    w.line("cache_keys[dst] = K[src];");
    w.pop_indent();
    w.line("} else if (local < TOKEN_RANGE) {");
    w.push_indent();
    w.line("uint vg = local - K_RANGE;");
    w.line("uint h = vg / V_DIM;");
    w.line("uint d = vg - h * V_DIM;");
    w.line("uint src = V_HEAD_MAJOR ? ((h * V_SEQ + row) * V_DIM + d) : ((row * N_KV + h) * V_DIM + d);");
    w.line("uint dst = h * capacity * V_DIM + (offset + row) * V_DIM + d;");
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
