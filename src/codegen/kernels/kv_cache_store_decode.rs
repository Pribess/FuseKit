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

fn current_layout(shape: &[i32]) -> Result<(u32, u32, u32)> {
    if shape.len() >= 4 {
        let n = shape.len();
        return Ok((
            shape[n - 3].max(1) as u32,
            shape[n - 2].max(1) as u32,
            shape[n - 1].max(1) as u32,
        ));
    }
    if shape.len() == 3 {
        return Ok((
            shape[0].max(1) as u32,
            shape[1].max(1) as u32,
            shape[2].max(1) as u32,
        ));
    }
    if shape.len() == 2 {
        return Ok((shape[0].max(1) as u32, 1, shape[1].max(1) as u32));
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

    let (n_kv_k, k_seq, k_dim) = current_layout(&fusion.inputs[0].shape)?;
    let (n_kv_v, v_seq, v_dim) = current_layout(&fusion.inputs[1].shape)?;
    if n_kv_k != n_kv_v {
        bail!("KvCacheStoreDecode K/V head count mismatch: {n_kv_k} vs {n_kv_v}");
    }
    let n_kv = n_kv_k;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_tokens_for_shape(&b.shape))
        .unwrap_or_else(|| crate::target::current_chip().active_m(8))
        .max(1);

    w.line(format!(
        "// Fusion #{}: KvCacheStoreDecode (active_m={}, n_kv={}, k_dim={}, v_dim={}, k_seq={}, v_seq={})",
        fusion.id, active_m, n_kv, k_dim, v_dim, k_seq, v_seq
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
    w.writeln(format_args!("constexpr uint K_SEQ = {k_seq};"));
    w.writeln(format_args!("constexpr uint V_SEQ = {v_seq};"));
    w.writeln(format_args!("constexpr uint K_DIM = {k_dim};"));
    w.writeln(format_args!("constexpr uint V_DIM = {v_dim};"));
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
    w.line("uint src = (row * N_KV + h) * K_SEQ * K_DIM + d;");
    w.line("uint dst = h * capacity * K_DIM + (offset + row) * K_DIM + d;");
    w.line("cache_keys[dst] = K[src];");
    w.pop_indent();
    w.line("} else if (local < TOKEN_RANGE) {");
    w.push_indent();
    w.line("uint vg = local - K_RANGE;");
    w.line("uint h = vg / V_DIM;");
    w.line("uint d = vg - h * V_DIM;");
    w.line("uint src = (row * N_KV + h) * V_SEQ * V_DIM + d;");
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
