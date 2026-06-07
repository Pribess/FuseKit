//! SDPA decode piece — single-token scaled-dot-product attention.
//!
//! Three SMEM-coordinated stages:
//!   1. `scores[kv] = (Q · K[kv]) * scale` for kv ∈ [0, kv_len).
//!   2. Numerically-stable softmax: threadgroup max, then exp-and-sum.
//!   3. `out[d] = Σ_kv softmax(scores)[kv] · V[kv, d]`.
//!
//! Reads `kv_len` and `capacity` from single-u32 buffers (composer
//! doesn't bind `constant uint &x [[buffer]]` references yet —
//! functionally identical to indexing `[0]` on a `device const uint*`).
//!
//! Per-layer megakernel uses this as the central piece: Q comes from
//! the preceding RmsNormQkv+Rope pieces, K/V come from the live cache,
//! `out` becomes the consumer of the trailing LinearResidual (o_proj).
//! When composed into a single kernel the `scores` SMEM region (up
//! to `MAX_KV_LEN × 4` bytes = 16 KiB at the cap) is the dominant
//! threadgroup-memory user — leaves room for the per-layer
//! norm_y/q_buf/etc. only if the chip's SMEM budget allows.

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::QuantInfo;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece, SmemRegion};

/// Largest kv_len we statically reserve scratch for in threadgroup
/// memory. Above this, the decode_pipeline falls back to a
/// (still-to-be-written) tiled SDPA piece. Keep in sync with
/// `kernels/sdpa_decode.rs::MAX_KV_LEN`.
pub const MAX_KV_LEN: u32 = 4096;

/// SDPA decode piece. With `kv_quant: None` it reads K/V as activation-
/// dtype (bf16/half/float) — the standard mlx-lm `KVCache` layout.
/// With `kv_quant: Some(QuantInfo { bits, group_size })` it reads the
/// 3-tuple `(packed_uint32, scales, biases)` layout produced by
/// `KVCache.to_quantized(...)` — mlx-lm's `QuantizedKVCache`. The
/// `bits` field is baked into the kernel via constexpr so the
/// dequant inner loop can fold the shift/mask into immediate operands.
pub struct SdpaDecode {
    /// `None` → bf16 K/V cache (default). `Some` → 4- or 8-bit packed.
    pub kv_quant: Option<QuantInfo>,
}

impl SdpaDecode {
    pub fn bf16() -> Self {
        Self { kv_quant: None }
    }
    pub fn quant(bits: u32, group_size: u32) -> Self {
        Self {
            kv_quant: Some(QuantInfo { bits, group_size }),
        }
    }
}

impl KernelPiece for SdpaDecode {
    fn name(&self) -> &str {
        "sdpa_decode"
    }

    fn smem_regions(&self, _ctx: &EmitCtx) -> Vec<SmemRegion> {
        vec![
            SmemRegion {
                name: "scores",
                elem_ty: "float".into(),
                count: format!("{MAX_KV_LEN}"),
            },
            // Composer dedupes by name; `reduce_buf` is also declared
            // by `rmsnorm_reduce` as `sum_sq_partial[32]`. We use a
            // distinct name so the regions don't accidentally alias
            // across the kernel and break either piece's reduction.
            SmemRegion {
                name: "sdpa_reduce_buf",
                elem_ty: "float".into(),
                count: "32".into(),
            },
        ]
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t: String = ctx.act_ty.into();
        let mut v = vec![BufferSlot {
            var_name: "Q".into(),
            elem_ty: t.clone(),
            access: BufferAccess::Read,
            role: "sdpa_q".into(),
        }];
        match self.kv_quant {
            None => {
                v.push(BufferSlot {
                    var_name: "K".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Read,
                    role: "layer_cache_k".into(),
                });
                v.push(BufferSlot {
                    var_name: "V".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Read,
                    role: "layer_cache_v".into(),
                });
            }
            Some(_) => {
                // `QuantizedKVCache.keys` / `.values` are 3-tuples
                // `(packed_uint32, scales, biases)`. Bind each leaf as
                // its own buffer slot — composer + runtime will route
                // them via the role keys below.
                v.push(BufferSlot {
                    var_name: "K_packed".into(),
                    elem_ty: "uint".into(),
                    access: BufferAccess::Read,
                    role: "layer_cache_k_packed".into(),
                });
                v.push(BufferSlot {
                    var_name: "K_scales".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Read,
                    role: "layer_cache_k_scales".into(),
                });
                v.push(BufferSlot {
                    var_name: "K_biases".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Read,
                    role: "layer_cache_k_biases".into(),
                });
                v.push(BufferSlot {
                    var_name: "V_packed".into(),
                    elem_ty: "uint".into(),
                    access: BufferAccess::Read,
                    role: "layer_cache_v_packed".into(),
                });
                v.push(BufferSlot {
                    var_name: "V_scales".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Read,
                    role: "layer_cache_v_scales".into(),
                });
                v.push(BufferSlot {
                    var_name: "V_biases".into(),
                    elem_ty: t.clone(),
                    access: BufferAccess::Read,
                    role: "layer_cache_v_biases".into(),
                });
            }
        }
        v.push(BufferSlot {
            var_name: "kv_len_buf".into(),
            elem_ty: "uint".into(),
            access: BufferAccess::Read,
            role: "layer_kv_len".into(),
        });
        v.push(BufferSlot {
            var_name: "capacity_buf".into(),
            elem_ty: "uint".into(),
            access: BufferAccess::Read,
            role: "layer_capacity".into(),
        });
        v.push(BufferSlot {
            var_name: "sdpa_out".into(),
            elem_ty: t,
            access: BufferAccess::Write,
            role: "sdpa_out".into(),
        });
        v
    }

    fn smem_writes(&self) -> Vec<&'static str> {
        vec!["scores", "sdpa_reduce_buf"]
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        let n_heads = ctx.meta.num_attention_heads;
        let n_kv = ctx.meta.num_key_value_heads;
        let head_dim = ctx.meta.head_dim;
        let heads_per_kv = n_heads / n_kv.max(1);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let t = ctx.act_ty;

        w.writeln(format_args!("constexpr uint SDPA_N_HEADS_ = {n_heads};"));
        w.writeln(format_args!("constexpr uint SDPA_N_KV_HEADS_ = {n_kv};"));
        w.writeln(format_args!("constexpr uint SDPA_HEAD_DIM_ = {head_dim};"));
        w.writeln(format_args!(
            "constexpr uint SDPA_HEADS_PER_KV_ = {heads_per_kv};"
        ));
        w.writeln(format_args!("constexpr float SDPA_SCALE_ = {scale};"));
        if let Some(q) = self.kv_quant {
            let bits = q.bits;
            let group_size = q.group_size;
            let el_per_int = 32 / bits;
            let n_words = head_dim / el_per_int;
            let n_groups = head_dim / group_size.max(1);
            let mask = (1u32 << bits) - 1;
            w.writeln(format_args!("constexpr uint SDPA_KV_BITS_ = {bits};"));
            w.writeln(format_args!(
                "constexpr uint SDPA_KV_EL_PER_INT_ = {el_per_int};"
            ));
            w.writeln(format_args!("constexpr uint SDPA_KV_N_WORDS_ = {n_words};"));
            w.writeln(format_args!(
                "constexpr uint SDPA_KV_GROUP_ = {group_size};"
            ));
            w.writeln(format_args!(
                "constexpr uint SDPA_KV_N_GROUPS_ = {n_groups};"
            ));
            w.writeln(format_args!("constexpr uint SDPA_KV_MASK_ = {mask};"));
        }

        // Composer's `gid` is the TG index; SDPA partitions one TG per
        // query head, so `head_id == gid`. kv_len/capacity come from
        // single-u32 device buffers (composer doesn't bind
        // `constant uint &x` references yet — behaviorally identical).
        w.line("uint head_id = gid;");
        w.line("uint kv_head = head_id / SDPA_HEADS_PER_KV_;");
        w.line("uint kv_len  = kv_len_buf[0];");
        w.line("uint capacity = capacity_buf[0];");
        w.blank();

        // Stage 1: scores = Q · K[kv] · scale per kv position.
        // Vec4 K row loads match the `kernels::sdpa_decode` standalone
        // path — 4× fewer memory transactions vs scalar half loads on
        // Apple GPU.
        w.line("// Stage 1: scores[kv] = (Q · K[kv]) * scale  (thread-per-kv, vec4 K row).");
        w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
        w.push_indent();
        w.writeln(format_args!(
            "device const {t}* q_row = Q + head_id * SDPA_HEAD_DIM_;"
        ));
        w.line("float dot = 0.0f;");
        match self.kv_quant {
            None => {
                w.writeln(format_args!(
                    "device const {t}4* q_row4 = (device const {t}4*)(Q + head_id * SDPA_HEAD_DIM_);"
                ));
                w.writeln(format_args!(
                    "device const {t}4* k_row4 = (device const {t}4*)(K + (kv_head * capacity + kv) * SDPA_HEAD_DIM_);"
                ));
                w.line("for (uint d4 = 0u; d4 < SDPA_HEAD_DIM_ / 4u; ++d4) {");
                w.push_indent();
                w.writeln(format_args!("{t}4 q4 = q_row4[d4];"));
                w.writeln(format_args!("{t}4 k4 = k_row4[d4];"));
                w.line("dot += float(q4.x) * float(k4.x) + float(q4.y) * float(k4.y)");
                w.line("     + float(q4.z) * float(k4.z) + float(q4.w) * float(k4.w);");
                w.pop_indent();
                w.line("}");
            }
            Some(_) => {
                // Quant K dequant inline. Per kv position we read
                // N_WORDS packed uint32s, N_GROUPS (scale, bias) pairs
                // from K_scales/K_biases. EL_PER_INT nibbles (4-bit) or
                // bytes (8-bit) per packed word are unpacked and fused
                // into the dot product against q_row[d].
                w.line("uint kv_off = (kv_head * capacity + kv);");
                w.line("device const uint *kp = K_packed + kv_off * SDPA_KV_N_WORDS_;");
                w.writeln(format_args!(
                    "device const {t} *ks = K_scales + kv_off * SDPA_KV_N_GROUPS_;"
                ));
                w.writeln(format_args!(
                    "device const {t} *kb = K_biases + kv_off * SDPA_KV_N_GROUPS_;"
                ));
                w.line("for (uint wi = 0u; wi < SDPA_KV_N_WORDS_; ++wi) {");
                w.push_indent();
                w.line("uint word = kp[wi];");
                w.line("uint d_base = wi * SDPA_KV_EL_PER_INT_;");
                w.line("uint g = d_base / SDPA_KV_GROUP_;");
                w.line("float s = float(ks[g]);");
                w.line("float b = float(kb[g]);");
                w.line("for (uint el = 0u; el < SDPA_KV_EL_PER_INT_; ++el) {");
                w.push_indent();
                w.line(
                    "float k_val = float((word >> (el * SDPA_KV_BITS_)) & SDPA_KV_MASK_) * s + b;",
                );
                w.line("dot += float(q_row[d_base + el]) * k_val;");
                w.pop_indent();
                w.line("}");
                w.pop_indent();
                w.line("}");
            }
        }
        w.line("scores[kv] = dot * SDPA_SCALE_;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");

        // Stage 2a: threadgroup max(scores).
        w.line("// Stage 2a: max(scores).");
        w.line("float t_max = -INFINITY;");
        w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
        w.push_indent();
        w.line("t_max = max(t_max, scores[kv]);");
        w.pop_indent();
        w.line("}");
        w.line("t_max = simd_max(t_max);");
        w.line("if (simd_lane == 0u) sdpa_reduce_buf[simd_id] = t_max;");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        w.line("float v = (tid < tg_size / 32u) ? sdpa_reduce_buf[tid] : -INFINITY;");
        w.line("v = simd_max(v);");
        w.line("if (simd_lane == 0u) sdpa_reduce_buf[0] = v;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("float global_max = sdpa_reduce_buf[0];");

        // Stage 2b: exp(scores - max) and sum.
        w.line("// Stage 2b: exp(scores - max) + sum.");
        w.line("float t_sum = 0.0f;");
        w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
        w.push_indent();
        w.line("scores[kv] = metal::precise::exp(scores[kv] - global_max);");
        w.line("t_sum += scores[kv];");
        w.pop_indent();
        w.line("}");
        w.line("t_sum = simd_sum(t_sum);");
        w.line("if (simd_lane == 0u) sdpa_reduce_buf[simd_id] = t_sum;");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        w.line("float v = (tid < tg_size / 32u) ? sdpa_reduce_buf[tid] : 0.0f;");
        w.line("v = simd_sum(v);");
        w.line("if (simd_lane == 0u) sdpa_reduce_buf[0] = v;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("float global_sum = sdpa_reduce_buf[0];");

        // Stage 2c: normalise scores in SMEM. Hoists the divide out of
        // Stage 3's inner kv loop.
        w.line("float inv_sum = 1.0f / global_sum;");
        w.line("for (uint kv = tid; kv < kv_len; kv += tg_size) {");
        w.push_indent();
        w.line("scores[kv] *= inv_sum;");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");

        // Stage 3: simdgroup-cooperative weighted sum of V along kv axis.
        // Same emit shape as `kernels::sdpa_decode::emit_legacy_two_pass`'s
        // Stage 3 — each SG owns HEAD_DIM/(sgs_per_tg*4) groups of 4 d's,
        // lanes stripe kv axis, simd_sum collapses. vec4 V loads for the
        // bf16 path; quant path stays scalar (V dequant is per-d-per-kv).
        w.line("// Stage 3: out[d] = Σ softmax(scores)[kv] · V[kv, d] (simdgroup-coop).");
        match self.kv_quant {
            None => {
                w.line("uint sg_per_tg_ = tg_size / 32u;");
                w.line("for (uint d_base = simd_id * 4u; d_base < SDPA_HEAD_DIM_; d_base += sg_per_tg_ * 4u) {");
                w.push_indent();
                w.line("float4 partial = float4(0.0f);");
                w.line("for (uint kv = simd_lane; kv < kv_len; kv += 32u) {");
                w.push_indent();
                w.writeln(format_args!(
                    "device const {t}4* v_row4 = (device const {t}4*)(V + (kv_head * capacity + kv) * SDPA_HEAD_DIM_);"
                ));
                w.writeln(format_args!("{t}4 v4 = v_row4[d_base / 4u];"));
                w.line("float s = scores[kv];");
                w.line("partial.x += s * float(v4.x);");
                w.line("partial.y += s * float(v4.y);");
                w.line("partial.z += s * float(v4.z);");
                w.line("partial.w += s * float(v4.w);");
                w.pop_indent();
                w.line("}");
                w.line("float4 acc;");
                w.line("acc.x = simd_sum(partial.x);");
                w.line("acc.y = simd_sum(partial.y);");
                w.line("acc.z = simd_sum(partial.z);");
                w.line("acc.w = simd_sum(partial.w);");
                w.line("if (simd_lane == 0u) {");
                w.push_indent();
                w.writeln(format_args!(
                    "device {t}4* out4 = (device {t}4*)(sdpa_out + head_id * SDPA_HEAD_DIM_);"
                ));
                w.writeln(format_args!(
                    "out4[d_base / 4u] = {t}4({t}(acc.x), {t}(acc.y), {t}(acc.z), {t}(acc.w));"
                ));
                w.pop_indent();
                w.line("}");
                w.pop_indent();
                w.line("}");
            }
            Some(_) => {
                // Quant path: per-d-per-kv dequant, simdgroup-coop on
                // kv axis. One d per SG; lanes stripe kv. Vec4 V load
                // not applicable (each thread needs only 1 dequanted
                // element per kv).
                w.line("uint sg_per_tg_ = tg_size / 32u;");
                w.line("for (uint d = simd_id; d < SDPA_HEAD_DIM_; d += sg_per_tg_) {");
                w.push_indent();
                w.line("uint wi = d / SDPA_KV_EL_PER_INT_;");
                w.line("uint el = d % SDPA_KV_EL_PER_INT_;");
                w.line("uint g  = d / SDPA_KV_GROUP_;");
                w.line("float partial = 0.0f;");
                w.line("for (uint kv = simd_lane; kv < kv_len; kv += 32u) {");
                w.push_indent();
                w.line("uint kv_off = (kv_head * capacity + kv);");
                w.line("uint word = V_packed[kv_off * SDPA_KV_N_WORDS_ + wi];");
                w.line("float s = float(V_scales[kv_off * SDPA_KV_N_GROUPS_ + g]);");
                w.line("float b = float(V_biases[kv_off * SDPA_KV_N_GROUPS_ + g]);");
                w.line(
                    "float v_val = float((word >> (el * SDPA_KV_BITS_)) & SDPA_KV_MASK_) * s + b;",
                );
                w.line("partial += scores[kv] * v_val;");
                w.pop_indent();
                w.line("}");
                w.line("float acc = simd_sum(partial);");
                w.writeln(format_args!(
                    "if (simd_lane == 0u) sdpa_out[head_id * SDPA_HEAD_DIM_ + d] = {t}(acc);"
                ));
                w.pop_indent();
                w.line("}");
            }
        }
    }
}
