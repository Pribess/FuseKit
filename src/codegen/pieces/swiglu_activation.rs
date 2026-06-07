//! SwiGLU activation piece — elementwise SiLU(gate) · up.
//!
//! Pure elementwise op, no reductions or SMEM. One thread per
//! element, looping with stride `tg_size` so any TG size works.
//!
//! This is the smallest viable piece — useful as the validation
//! target for the composer + builds confidence before tackling
//! reduction/heavy pieces.

use crate::codegen::msl::MslWriter;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece};

pub struct SwigluActivation {
    /// Flat element count (product of output shape dims).
    pub n: u32,
}

impl KernelPiece for SwigluActivation {
    fn name(&self) -> &str {
        "swiglu_activation"
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t: String = ctx.act_ty.into();
        vec![
            BufferSlot {
                var_name: "gate".into(),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: "swiglu_gate".into(),
            },
            BufferSlot {
                var_name: "up".into(),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: "swiglu_up".into(),
            },
            BufferSlot {
                var_name: "out".into(),
                elem_ty: t,
                access: BufferAccess::Write,
                role: "swiglu_out".into(),
            },
        ]
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        w.writeln(format_args!("constexpr uint N = {};", self.n));
        w.line("for (uint i = tid; i < N; i += tg_size) {");
        w.push_indent();
        w.line("float g = float(gate[i]);");
        w.line("float u = float(up[i]);");
        w.line("float s = 1.0f / (1.0f + exp(-g));");
        w.writeln(format_args!("out[i] = {}(s * g * u);", ctx.act_ty));
        w.pop_indent();
        w.line("}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::msl::MslWriter;
    use crate::codegen::pieces::compose;
    use crate::fusion::ir::ModelMeta;
    use crate::graph::Dtype;

    #[test]
    fn composer_emits_swiglu_kernel() {
        let meta = ModelMeta {
            model_type: "qwen2".into(),
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 4864,
            vocab_size: 152064,
            quant_group_size: Some(64),
            quant_bits: Some(4),
            mtp_num_hidden_layers: 0,
            num_nextn_predict_layers: 0,
            mtp: crate::fusion::ir::MtpInfo::none(),
            activation_dtype: Dtype::Bf16,
        };
        let ctx = EmitCtx {
            meta: &meta,
            act_ty: "bfloat",
        };
        let pieces: Vec<Box<dyn KernelPiece>> = vec![Box::new(SwigluActivation { n: 4864 })];
        let mut w = MslWriter::new();
        compose("test_swiglu", &pieces, &ctx, &mut w);
        let out = w.finish();

        // Function signature + buffer slots in piece-declared order.
        assert!(out.contains("kernel void test_swiglu("));
        assert!(out.contains("device const bfloat *gate [[buffer(0)]]"));
        assert!(out.contains("device const bfloat *up [[buffer(1)]]"));
        assert!(out.contains("device bfloat *out [[buffer(2)]]"));
        // Body emitted.
        assert!(out.contains("constexpr uint N = 4864;"));
        assert!(out.contains("out[i] = bfloat(s * g * u);"));
        // Standard locals from composer.
        assert!(out.contains("uint simd_lane = tid % 32u;"));
    }
}
