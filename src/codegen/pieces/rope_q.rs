//! RoPE Q rotation piece — applies decode-time rotary embedding to a
//! Q tensor in place.
//!
//! Q has shape `(n_heads, head_dim)`. For each pair `(d, d+half)` we
//! rotate by `angle = θ_d · pos` where `θ_d = THETA^(-2d/head_dim)`.
//! mlx-lm uses the "non-traditional" rotation: the two halves of
//! head_dim rotate together rather than adjacent pairs.
//!
//! ```text
//!   Q[h, d]        = Q[h, d]        · cos - Q[h, d+half] · sin
//!   Q[h, d+half]   = Q[h, d+half]   · cos + Q[h, d]      · sin
//! ```
//!
//! Reads `offset` from the layer's offset-uniform buffer (slot 0
//! u32). The composer doesn't yet distinguish `constant`-address-
//! space uniforms from device buffers; we use a regular `device
//! const uint *` and index `[0]`. Behaviorally identical, slightly
//! more general (uniform updates work the same).

use crate::codegen::msl::MslWriter;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece};

pub struct RopeQ {
    pub theta: f32,
}

impl KernelPiece for RopeQ {
    fn name(&self) -> &str {
        "rope_q"
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t: String = ctx.act_ty.into();
        vec![
            BufferSlot {
                var_name: "Q".into(),
                elem_ty: t,
                access: BufferAccess::Write,
                role: "rope_q".into(),
            },
            BufferSlot {
                var_name: "offset_buf".into(),
                elem_ty: "uint".into(),
                access: BufferAccess::Read,
                role: "layer_offset".into(),
            },
        ]
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        let n_heads = ctx.meta.num_attention_heads;
        let head_dim = ctx.meta.head_dim;
        let half = head_dim / 2;
        w.writeln(format_args!("constexpr uint ROPE_N_HEADS_ = {};", n_heads));
        w.writeln(format_args!(
            "constexpr uint ROPE_HEAD_DIM_ = {};",
            head_dim
        ));
        w.writeln(format_args!("constexpr uint ROPE_HALF_ = {};", half));
        w.writeln(format_args!(
            "constexpr float ROPE_THETA_ = {};",
            self.theta
        ));
        w.line("uint offset = offset_buf[0];");
        w.line("uint g = gid * tg_size + tid;");
        w.line("bool active = (g < ROPE_N_HEADS_ * ROPE_HEAD_DIM_);");
        w.line("uint h = active ? (g / ROPE_HEAD_DIM_) : 0u;");
        w.line("uint d = active ? (g % ROPE_HEAD_DIM_) : 0u;");
        w.line("uint pair_idx = (d < ROPE_HALF_) ? d : (d - ROPE_HALF_);");
        w.line("float freq = pow(ROPE_THETA_, -2.0f * float(pair_idx) / float(ROPE_HEAD_DIM_));");
        w.line("float angle = freq * float(offset);");
        w.line("float c = cos(angle);");
        w.line("float s = sin(angle);");
        w.line("uint base = h * ROPE_HEAD_DIM_;");
        w.line("float lo = float(Q[base + pair_idx]);");
        w.line("float hi = float(Q[base + pair_idx + ROPE_HALF_]);");
        w.line("float result = (d < ROPE_HALF_) ? (lo * c - hi * s) : (hi * c + lo * s);");
        // Pair's two halves are in the same TG only if HEAD_DIM divides
        // TG size. Device barrier here is conservative — the partner
        // load already happened before any writes.
        w.line("threadgroup_barrier(mem_flags::mem_device);");
        w.writeln(format_args!(
            "if (active) Q[g] = {t}(result);",
            t = ctx.act_ty
        ));
    }
}
