//! RoPE on K + KV cache store piece.
//!
//! After the layer's K/V projections write `K (n_kv, head_dim)` and
//! `V (n_kv, head_dim)` to scratch (or, in a per-layer megakernel,
//! to SMEM), this piece:
//!   - applies RoPE rotation pair-wise to K,
//!   - writes rotated K into `cache_keys[h, offset, :]`,
//!   - copies V straight into `cache_values[h, offset, :]`.
//!
//! `offset` and `capacity` come from single-u32 device buffers
//! (composer doesn't bind `constant uint &x` references — behaviorally
//! identical to indexing `[0]`).

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::QuantInfo;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece, SmemRegion};

/// RoPE + KV cache store. With `kv_quant: None` the rotated K is
/// written to the cache as the activation dtype (bf16). With
/// `kv_quant: Some(...)` the row is group-wise affine-quantized
/// (same formula as `mx.quantize` so the cache layout matches
/// `QuantizedKVCache`):
///   `scale = (max - min) / (2^bits - 1)`, `bias = min`,
///   `q = round((x - bias) / scale)`,
///   pack `EL_PER_INT` lanes per `uint32` LSB-first.
pub struct RopeKvCacheStore {
    pub theta: f32,
    pub kv_quant: Option<QuantInfo>,
}

impl RopeKvCacheStore {
    pub fn bf16(theta: f32) -> Self {
        Self {
            theta,
            kv_quant: None,
        }
    }
    pub fn quant(theta: f32, bits: u32, group_size: u32) -> Self {
        Self {
            theta,
            kv_quant: Some(QuantInfo { bits, group_size }),
        }
    }
}

impl KernelPiece for RopeKvCacheStore {
    fn name(&self) -> &str {
        "rope_kv_cache_store"
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t: String = ctx.act_ty.into();
        let mut v = vec![
            // K/V come from upstream pieces' scratch (or layer-megakernel
            // SMEM region). Reuse the names the QKV projection piece
            // uses so role-unification stitches them automatically.
            BufferSlot {
                var_name: "K".into(),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: "k_buf".into(),
            },
            BufferSlot {
                var_name: "V".into(),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: "v_buf".into(),
            },
        ];
        match self.kv_quant {
            None => {
                v.push(BufferSlot {
                    var_name: "cache_keys".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Write,
                    role: "layer_cache_k".into(),
                });
                v.push(BufferSlot {
                    var_name: "cache_values".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Write,
                    role: "layer_cache_v".into(),
                });
            }
            Some(_) => {
                // 3-tuple destination per K/V, matching mlx-lm
                // `QuantizedKVCache` storage layout.
                v.push(BufferSlot {
                    var_name: "cache_keys_packed".into(),
                    elem_ty: "uint".into(),
                    access: BufferAccess::Write,
                    role: "layer_cache_k_packed".into(),
                });
                v.push(BufferSlot {
                    var_name: "cache_keys_scales".into(),
                    elem_ty: "half".into(),
                    access: BufferAccess::Write,
                    role: "layer_cache_k_scales".into(),
                });
                v.push(BufferSlot {
                    var_name: "cache_keys_biases".into(),
                    elem_ty: "half".into(),
                    access: BufferAccess::Write,
                    role: "layer_cache_k_biases".into(),
                });
                v.push(BufferSlot {
                    var_name: "cache_values_packed".into(),
                    elem_ty: "uint".into(),
                    access: BufferAccess::Write,
                    role: "layer_cache_v_packed".into(),
                });
                v.push(BufferSlot {
                    var_name: "cache_values_scales".into(),
                    elem_ty: "half".into(),
                    access: BufferAccess::Write,
                    role: "layer_cache_v_scales".into(),
                });
                v.push(BufferSlot {
                    var_name: "cache_values_biases".into(),
                    elem_ty: "half".into(),
                    access: BufferAccess::Write,
                    role: "layer_cache_v_biases".into(),
                });
            }
        }
        v.push(BufferSlot {
            var_name: "offset_buf".into(),
            elem_ty: "uint".into(),
            access: BufferAccess::Read,
            role: "layer_offset".into(),
        });
        v.push(BufferSlot {
            var_name: "capacity_buf".into(),
            elem_ty: "uint".into(),
            access: BufferAccess::Read,
            role: "layer_capacity".into(),
        });
        v
    }

    fn smem_regions(&self, ctx: &EmitCtx) -> Vec<SmemRegion> {
        if self.kv_quant.is_none() {
            return Vec::new();
        }
        // For quant write we need TG-local scratch for the rotated row
        // (so the pack stage can reuse the same values per group reduction),
        // plus per-group min/max reductions across SGs.
        vec![
            SmemRegion {
                name: "qstore_row",
                elem_ty: "float".into(),
                count: format!("{}", ctx.meta.head_dim),
            },
            SmemRegion {
                name: "qstore_red_min",
                elem_ty: "float".into(),
                count: "32".into(),
            },
            SmemRegion {
                name: "qstore_red_max",
                elem_ty: "float".into(),
                count: "32".into(),
            },
        ]
    }

    fn smem_writes(&self) -> Vec<&'static str> {
        if self.kv_quant.is_some() {
            vec!["qstore_row", "qstore_red_min", "qstore_red_max"]
        } else {
            Vec::new()
        }
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        match self.kv_quant {
            None => self.emit_bf16(w, ctx),
            Some(_) => self.emit_quant(w, ctx),
        }
    }
}

impl RopeKvCacheStore {
    fn emit_bf16(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        let t = ctx.act_ty;
        let n_kv = ctx.meta.num_key_value_heads;
        let head_dim = ctx.meta.head_dim;
        let half = head_dim / 2;
        w.writeln(format_args!("constexpr uint RKVS_N_KV_ = {n_kv};"));
        w.writeln(format_args!("constexpr uint RKVS_HEAD_DIM_ = {head_dim};"));
        w.writeln(format_args!("constexpr uint RKVS_HALF_ = {half};"));
        w.writeln(format_args!(
            "constexpr float RKVS_THETA_ = {};",
            self.theta
        ));
        w.line("constexpr uint RKVS_K_RANGE_ = RKVS_N_KV_ * RKVS_HEAD_DIM_;");
        w.line("constexpr uint RKVS_V_RANGE_ = RKVS_K_RANGE_ + RKVS_N_KV_ * RKVS_HEAD_DIM_;");
        w.line("uint rkvs_offset = offset_buf[0];");
        w.line("uint rkvs_capacity = capacity_buf[0];");
        w.line("uint rkvs_g = gid * tg_size + tid;");
        w.line("bool rkvs_is_k = (rkvs_g < RKVS_K_RANGE_);");
        w.line("bool rkvs_is_v = (rkvs_g >= RKVS_K_RANGE_ && rkvs_g < RKVS_V_RANGE_);");
        w.line("uint rkvs_k_idx = rkvs_is_k ? rkvs_g : 0u;");
        w.line("uint rkvs_v_idx = rkvs_is_v ? (rkvs_g - RKVS_K_RANGE_) : 0u;");
        w.line("uint rkvs_kh = rkvs_k_idx / RKVS_HEAD_DIM_;");
        w.line("uint rkvs_kd = rkvs_k_idx % RKVS_HEAD_DIM_;");
        w.line("uint rkvs_pair = (rkvs_kd < RKVS_HALF_) ? rkvs_kd : (rkvs_kd - RKVS_HALF_);");
        w.line(
            "float rkvs_freq = pow(RKVS_THETA_, -2.0f * float(rkvs_pair) / float(RKVS_HEAD_DIM_));",
        );
        w.line("float rkvs_angle = rkvs_freq * float(rkvs_offset);");
        w.line("float rkvs_c = cos(rkvs_angle);");
        w.line("float rkvs_s = sin(rkvs_angle);");
        w.line("uint rkvs_kbase = rkvs_kh * RKVS_HEAD_DIM_;");
        w.line("float rkvs_lo = float(K[rkvs_kbase + rkvs_pair]);");
        w.line("float rkvs_hi = float(K[rkvs_kbase + rkvs_pair + RKVS_HALF_]);");
        w.line("float rkvs_result = (rkvs_kd < RKVS_HALF_) ? (rkvs_lo * rkvs_c - rkvs_hi * rkvs_s) : (rkvs_hi * rkvs_c + rkvs_lo * rkvs_s);");
        // Pair-wise reads done before writes — barrier per kernel.
        w.line("threadgroup_barrier(mem_flags::mem_device);");
        w.line("if (rkvs_is_k) {");
        w.push_indent();
        w.line("uint co = rkvs_kh * rkvs_capacity * RKVS_HEAD_DIM_ + rkvs_offset * RKVS_HEAD_DIM_ + rkvs_kd;");
        w.writeln(format_args!("cache_keys[co] = {t}(rkvs_result);"));
        w.pop_indent();
        w.line("}");
        w.line("if (rkvs_is_v) {");
        w.push_indent();
        w.line("uint vh = rkvs_v_idx / RKVS_HEAD_DIM_;");
        w.line("uint vd = rkvs_v_idx % RKVS_HEAD_DIM_;");
        w.line(
            "uint co = vh * rkvs_capacity * RKVS_HEAD_DIM_ + rkvs_offset * RKVS_HEAD_DIM_ + vd;",
        );
        w.writeln(format_args!("cache_values[co] = {t}(V[rkvs_v_idx]);"));
        w.pop_indent();
        w.line("}");
    }

    /// Quantized cache write. Layout per TG:
    ///   `gid = kh                      ` → K row (rotated)  for kv_head kh
    ///   `gid = kh + N_KV              ` → V row (no rotate) for kv_head kh
    /// Grid = 2 × N_KV threadgroups. Each TG processes ONE row of
    /// HEAD_DIM elements cooperatively.
    fn emit_quant(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        let q = self.kv_quant.unwrap();
        let bits = q.bits;
        let group_size = q.group_size;
        let head_dim = ctx.meta.head_dim;
        let n_kv = ctx.meta.num_key_value_heads;
        let half = head_dim / 2;
        let el_per_int = 32 / bits;
        let n_words = head_dim / el_per_int;
        let n_groups = head_dim / group_size.max(1);
        let mask = (1u32 << bits) - 1;
        w.writeln(format_args!("constexpr uint RKVS_HEAD_DIM_ = {head_dim};"));
        w.writeln(format_args!("constexpr uint RKVS_N_KV_ = {n_kv};"));
        w.writeln(format_args!("constexpr uint RKVS_HALF_ = {half};"));
        w.writeln(format_args!(
            "constexpr float RKVS_THETA_ = {};",
            self.theta
        ));
        w.writeln(format_args!("constexpr uint RKVS_BITS_ = {bits};"));
        w.writeln(format_args!(
            "constexpr uint RKVS_EL_PER_INT_ = {el_per_int};"
        ));
        w.writeln(format_args!("constexpr uint RKVS_N_WORDS_ = {n_words};"));
        w.writeln(format_args!("constexpr uint RKVS_GROUP_ = {group_size};"));
        w.writeln(format_args!("constexpr uint RKVS_N_GROUPS_ = {n_groups};"));
        w.writeln(format_args!("constexpr uint RKVS_MASK_ = {mask};"));
        w.line("uint rkvs_offset = offset_buf[0];");
        w.line("uint rkvs_capacity = capacity_buf[0];");
        w.line("bool rkvs_is_v = (gid >= RKVS_N_KV_);");
        w.line("uint rkvs_kh = rkvs_is_v ? (gid - RKVS_N_KV_) : gid;");
        w.line("uint rkvs_kbase = rkvs_kh * RKVS_HEAD_DIM_;");
        w.blank();
        // Stage 1: load row into SMEM (qstore_row). K does RoPE, V copy.
        w.line("for (uint d = tid; d < RKVS_HEAD_DIM_; d += tg_size) {");
        w.push_indent();
        w.line("float val;");
        w.line("if (rkvs_is_v) {");
        w.push_indent();
        w.line("val = float(V[rkvs_kbase + d]);");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("uint pair = (d < RKVS_HALF_) ? d : (d - RKVS_HALF_);");
        w.line("float freq = pow(RKVS_THETA_, -2.0f * float(pair) / float(RKVS_HEAD_DIM_));");
        w.line("float angle = freq * float(rkvs_offset);");
        w.line("float c = cos(angle); float s = sin(angle);");
        w.line("float lo = float(K[rkvs_kbase + pair]);");
        w.line("float hi = float(K[rkvs_kbase + pair + RKVS_HALF_]);");
        w.line("val = (d < RKVS_HALF_) ? (lo * c - hi * s) : (hi * c + lo * s);");
        w.pop_indent();
        w.line("}");
        w.line("qstore_row[d] = val;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.blank();
        // Stage 2: per-group min/max (cooperative TG reduction).
        w.line("for (uint g = 0u; g < RKVS_N_GROUPS_; ++g) {");
        w.push_indent();
        w.line("uint g_start = g * RKVS_GROUP_;");
        w.line("float my_min = INFINITY; float my_max = -INFINITY;");
        w.line("for (uint d = tid; d < RKVS_GROUP_; d += tg_size) {");
        w.push_indent();
        w.line("float v = qstore_row[g_start + d];");
        w.line("my_min = min(my_min, v); my_max = max(my_max, v);");
        w.pop_indent();
        w.line("}");
        w.line("my_min = simd_min(my_min); my_max = simd_max(my_max);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line("qstore_red_min[simd_id] = my_min;");
        w.line("qstore_red_max[simd_id] = my_max;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        w.line("float v_min = (tid < (tg_size / 32u)) ? qstore_red_min[tid] : INFINITY;");
        w.line("float v_max = (tid < (tg_size / 32u)) ? qstore_red_max[tid] : -INFINITY;");
        w.line("v_min = simd_min(v_min); v_max = simd_max(v_max);");
        w.line("if (simd_lane == 0u) { qstore_red_min[g] = v_min; qstore_red_max[g] = v_max; }");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.pop_indent();
        w.line("}");
        w.blank();
        // Stage 3: pack + write packed words; one thread per word.
        // Per-group scale = (max-min) / MASK, bias = min.
        w.line("uint co_packed = rkvs_kh * rkvs_capacity * RKVS_N_WORDS_ + rkvs_offset * RKVS_N_WORDS_;");
        w.line("uint co_meta   = rkvs_kh * rkvs_capacity * RKVS_N_GROUPS_ + rkvs_offset * RKVS_N_GROUPS_;");
        w.line("for (uint wi = tid; wi < RKVS_N_WORDS_; wi += tg_size) {");
        w.push_indent();
        w.line("uint d_base = wi * RKVS_EL_PER_INT_;");
        w.line("uint g = d_base / RKVS_GROUP_;");
        w.line("float v_min = qstore_red_min[g];");
        w.line("float v_max = qstore_red_max[g];");
        w.line("float scale = (v_max - v_min) / float(RKVS_MASK_);");
        w.line("float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;");
        w.line("uint packed = 0u;");
        w.line("for (uint el = 0u; el < RKVS_EL_PER_INT_; ++el) {");
        w.push_indent();
        w.line("float x = qstore_row[d_base + el];");
        w.line("int qi = int(round((x - v_min) * inv_scale));");
        w.line("uint q = (uint)clamp(qi, 0, int(RKVS_MASK_));");
        w.line("packed |= (q & RKVS_MASK_) << (el * RKVS_BITS_);");
        w.pop_indent();
        w.line("}");
        w.line("if (rkvs_is_v) {");
        w.push_indent();
        w.line("cache_values_packed[co_packed + wi] = packed;");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("cache_keys_packed[co_packed + wi] = packed;");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
        // Per-group scales/biases written by group_idx-th thread.
        w.line("for (uint g = tid; g < RKVS_N_GROUPS_; g += tg_size) {");
        w.push_indent();
        w.line("float v_min = qstore_red_min[g];");
        w.line("float v_max = qstore_red_max[g];");
        w.line("float scale = (v_max - v_min) / float(RKVS_MASK_);");
        w.line("if (rkvs_is_v) {");
        w.push_indent();
        w.line("cache_values_scales[co_meta + g] = half(scale);");
        w.line("cache_values_biases[co_meta + g] = half(v_min);");
        w.pop_indent();
        w.line("} else {");
        w.push_indent();
        w.line("cache_keys_scales[co_meta + g] = half(scale);");
        w.line("cache_keys_biases[co_meta + g] = half(v_min);");
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    }
}
