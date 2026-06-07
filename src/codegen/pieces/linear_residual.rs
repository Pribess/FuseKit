//! LinearResidual piece — GEMV (one output row per simdgroup) +
//! optional bias + per-row residual add into the layer's `x` stream.
//!
//! Used twice per Qwen-style layer:
//!   - o_proj: SDPA out → hidden, written back into `x` (residual)
//!   - down_proj: silu_up → hidden, same residual
//!
//! Reads activation `x_attn_in` (or `silu_up`) and weights from
//! device buffers. Writes the residual sum into the `x_input` role —
//! shared with `RmsNormReduce` and `ResidualAdd` so the layer's
//! input/output residual stream is one buffer across pieces.

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::QuantInfo as Quant;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece};

/// `LinearResidual` piece for one (input, weight, residual) tuple.
/// The role names let the composer route the piece into a layer:
///   - `act_role` identifies the activation source (e.g. `sdpa_out`
///     for o_proj, `silu_up` for down_proj).
///   - The residual destination is always role `"x_input"` so the
///     entire layer's residual stream is one buffer.
pub struct LinearResidual {
    /// Input dim (= reduction axis).
    pub m: u32,
    /// Output dim (= hidden size for residual to fit).
    pub n: u32,
    /// MSL variable name to bind the activation (must match upstream
    /// piece's output var_name when role-sharing).
    pub act_var: String,
    pub act_role: String,
    /// Tag (e.g. `"o"` / `"down"`) — disambiguates buffer slot names
    /// when the piece is reused for both o_proj and down_proj in the
    /// same composed kernel.
    pub tag: String,
    pub quant: Option<Quant>,
    pub has_bias: bool,
}

impl KernelPiece for LinearResidual {
    fn name(&self) -> &str {
        "linear_residual"
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t: String = ctx.act_ty.into();
        let mut v: Vec<BufferSlot> = vec![BufferSlot {
            var_name: self.act_var.clone(),
            elem_ty: t.clone(),
            access: BufferAccess::Read,
            role: self.act_role.clone(),
        }];
        if self.quant.is_some() {
            v.push(BufferSlot {
                var_name: format!("W_{}_packed", self.tag),
                elem_ty: "uint".into(),
                access: BufferAccess::Read,
                role: format!("linres_w_packed_{}", self.tag),
            });
            v.push(BufferSlot {
                var_name: format!("W_{}_s", self.tag),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: format!("linres_w_scales_{}", self.tag),
            });
            v.push(BufferSlot {
                var_name: format!("W_{}_b", self.tag),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: format!("linres_w_biases_{}", self.tag),
            });
        } else {
            v.push(BufferSlot {
                var_name: format!("W_{}", self.tag),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: format!("linres_w_{}", self.tag),
            });
        }
        if self.has_bias {
            v.push(BufferSlot {
                var_name: format!("bias_{}", self.tag),
                elem_ty: t.clone(),
                access: BufferAccess::Read,
                role: format!("linres_bias_{}", self.tag),
            });
        }
        // Residual source — same role as RmsNormReduce's `x` slot so
        // composer merges them into one buffer. var_name MUST match
        // RmsNormReduce's `"x"` because the composer's dedup keeps
        // the first registrant's name; if LinearResidual declares
        // a different name the kernel signature ends up with `x` but
        // this piece's body references `residual`. (When the piece
        // is used standalone the merge doesn't fire and the slot
        // still ends up named `x` because we declare it as `x` here.)
        v.push(BufferSlot {
            var_name: "x".into(),
            elem_ty: t.clone(),
            access: BufferAccess::Read,
            role: "x_input".into(),
        });
        // Output buffer — `y[i] = residual[i] + W @ act`. Separate
        // buffer from `residual` because the trace produces a new
        // tensor for the post-residual value. Role is tagged so
        // multiple LinearResidual pieces (o_proj vs down_proj) in
        // one kernel get distinct outputs.
        v.push(BufferSlot {
            var_name: format!("y_{}", self.tag),
            elem_ty: t,
            access: BufferAccess::Write,
            role: format!("linres_y_{}", self.tag),
        });
        v
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        let t = ctx.act_ty;
        let tag = &self.tag;
        w.writeln(format_args!("constexpr uint LINRES_M_{tag} = {};", self.m));
        w.writeln(format_args!("constexpr uint LINRES_N_{tag} = {};", self.n));
        // Stride-loop: each simdgroup handles `LINRES_N / sgs_total`
        // output rows (ceil), so any grid size covers all rows. In a
        // standalone simdgroup-coop GEMV the host dispatches enough
        // SGs that the loop body runs once; in a LayerBlock megakernel
        // grid=N_HEADS (= 14 TGs × 4 SGs = 56 SGs for Qwen-0.5B) the
        // loop walks ceil(1024/56) ≈ 19 rows per SG.
        w.line(format!(
            "for (uint linres_i_{tag} = gid * sgs_per_tg + simd_id;"
        ));
        w.line(format!("     linres_i_{tag} < LINRES_N_{tag};"));
        w.line(format!("     linres_i_{tag} += sgs_total) {{"));
        w.push_indent();
        w.line("float linres_acc = 0.0f;");
        if let Some(q) = self.quant {
            let bits = q.bits;
            let group_size = q.group_size;
            let packed_row = format!(
                "W_{tag}_packed + linres_i_{tag} * (LINRES_M_{tag} / {})",
                32 / bits
            );
            let scales_row =
                format!("W_{tag}_s + linres_i_{tag} * (LINRES_M_{tag} / {group_size}u)");
            let biases_row =
                format!("W_{tag}_b + linres_i_{tag} * (LINRES_M_{tag} / {group_size}u)");
            let act_var = self.act_var.clone();
            crate::codegen::kernels::emit_int4_gemv_inner_with_unroll2_default(
                w,
                "linres_acc",
                &packed_row,
                &scales_row,
                &biases_row,
                &format!("LINRES_M_{tag}"),
                bits,
                group_size,
                crate::codegen::kernels::qgemv_unroll2_default_for_shape(bits, self.m, self.n),
                |k| format!("float({act_var}[{k}])"),
            );
        } else {
            w.writeln(format_args!(
                "for (uint k = simd_lane; k < LINRES_M_{tag}; k += 32u) {{"
            ));
            w.push_indent();
            w.writeln(format_args!(
                "linres_acc += float({}[k]) * float(W_{tag}[linres_i_{tag} * LINRES_M_{tag} + k]);",
                self.act_var
            ));
            w.pop_indent();
            w.line("}");
        }
        w.line("linres_acc = simd_sum(linres_acc);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if self.has_bias {
            w.writeln(format_args!(
                "linres_acc += float(bias_{tag}[linres_i_{tag}]);"
            ));
        }
        // Separate residual READ + y WRITE — matches the trace's
        // `y = residual + W @ act` semantics. Aliasing residual and
        // y to the same buffer is a host-side decision (mlx's
        // scratch allocator does it for the trace-bound pattern;
        // for LayerBlock the synthesizer must bind them to the
        // right tensors via fusion.inputs/outputs).
        w.writeln(format_args!(
            "y_{tag}[linres_i_{tag}] = {t}(float(x[linres_i_{tag}]) + linres_acc);"
        ));
        w.pop_indent();
        w.line("}");
        w.pop_indent();
        w.line("}");
    }
}
