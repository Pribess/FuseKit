//! Residual add piece — elementwise `x[i] = x[i] + delta[i]`.
//!
//! Used both as the trailing residual after `o_proj` and after the
//! MLP `down_proj`. Pure elementwise, no SMEM, one thread per
//! element with `tg_size` stride so any TG size works.

use crate::codegen::msl::MslWriter;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece};

pub struct ResidualAdd {
    /// Flat element count (product of x's shape dims).
    pub n: u32,
}

impl KernelPiece for ResidualAdd {
    fn name(&self) -> &str {
        "residual_add"
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t: String = ctx.act_ty.into();
        vec![
            BufferSlot {
                var_name: "delta".into(),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: "residual_delta".into(),
            },
            // role "x_input" shared with RmsNormReduce: when both
            // pieces appear in one composed kernel, the composer
            // merges them into a single read+write `x` slot. When
            // ResidualAdd is used standalone, the slot is still
            // emitted as a writable `x`.
            BufferSlot {
                var_name: "x".into(),
                elem_ty: t,
                access: BufferAccess::Write,
                role: "x_input".into(),
            },
        ]
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        w.writeln(format_args!("constexpr uint N_RES_ = {};", self.n));
        w.line("for (uint i = tid; i < N_RES_; i += tg_size) {");
        w.push_indent();
        w.writeln(format_args!(
            "x[i] = {t}(float(x[i]) + float(delta[i]));",
            t = ctx.act_ty
        ));
        w.pop_indent();
        w.line("}");
    }
}
