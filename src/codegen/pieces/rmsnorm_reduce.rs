//! RMSNorm reduction piece.
//!
//! Reads `x[HIDDEN]` from device memory, computes
//! `norm_y[i] = x[i] / sqrt(mean(x²) + EPS) * norm_w[i]` cooperatively
//! across the threadgroup, leaves the result in SMEM `norm_y[HIDDEN]`
//! for subsequent pieces (GEMV rows, residual+norm fusion, ...).
//!
//! Producer of: `norm_y`.

use crate::codegen::msl::MslWriter;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece, SmemRegion};

pub struct RmsNormReduce {
    pub eps: f32,
    /// Override the default `act_ty` for the activation buffer. None
    /// → use `ctx.act_ty` (the model's compute precision).
    pub act_ty_override: Option<&'static str>,
}

impl RmsNormReduce {
    pub fn new(eps: f32) -> Self {
        Self {
            eps,
            act_ty_override: None,
        }
    }
}

impl KernelPiece for RmsNormReduce {
    fn name(&self) -> &str {
        "rmsnorm_reduce"
    }

    fn smem_regions(&self, ctx: &EmitCtx) -> Vec<SmemRegion> {
        // Compose-time literal — composer emits SMEM declarations at
        // the top of the kernel, BEFORE any piece body has declared
        // `constexpr uint HIDDEN`. Using a numeric string keeps the
        // SMEM declaration self-contained.
        vec![
            SmemRegion {
                name: "norm_y",
                elem_ty: "float".into(),
                count: format!("{}", ctx.meta.hidden_size),
            },
            SmemRegion {
                name: "sum_sq_partial",
                elem_ty: "float".into(),
                count: "32".into(),
            },
        ]
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t = self.act_ty_override.unwrap_or(ctx.act_ty);
        vec![
            BufferSlot {
                var_name: "x".into(),
                elem_ty: t.into(),
                access: BufferAccess::Read,
                role: "x_input".into(),
            },
            BufferSlot {
                var_name: "norm_w".into(),
                elem_ty: t.into(),
                access: BufferAccess::Read,
                role: "norm_w".into(),
            },
        ]
    }

    fn smem_writes(&self) -> Vec<&'static str> {
        vec!["norm_y", "sum_sq_partial"]
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        w.writeln(format_args!(
            "constexpr uint HIDDEN = {};",
            ctx.meta.hidden_size
        ));
        w.writeln(format_args!("constexpr float EPS_NORM_ = {};", self.eps));
        w.line("// Per-thread partial sum of x².");
        w.line("float sum_sq = 0.0f;");
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.line("float xi = float(x[i]);");
        w.line("sum_sq += xi * xi;");
        w.pop_indent();
        w.line("}");
        w.line("sum_sq = simd_sum(sum_sq);");
        w.line("if (simd_lane == 0u) { sum_sq_partial[simd_id] = sum_sq; }");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("if (simd_id == 0u) {");
        w.push_indent();
        w.line("float v = (tid < (tg_size / 32u)) ? sum_sq_partial[tid] : 0.0f;");
        w.line("v = simd_sum(v);");
        w.line("if (simd_lane == 0u) { sum_sq_partial[0] = v; }");
        w.pop_indent();
        w.line("}");
        w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        w.line("float rms_scale = metal::precise::rsqrt(sum_sq_partial[0] / float(HIDDEN) + EPS_NORM_);");
        w.line("for (uint i = tid; i < HIDDEN; i += tg_size) {");
        w.push_indent();
        w.line("norm_y[i] = float(x[i]) * rms_scale * float(norm_w[i]);");
        w.pop_indent();
        w.line("}");
    }
}
