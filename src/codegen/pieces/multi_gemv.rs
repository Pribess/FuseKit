//! Multi-output GEMV piece.
//!
//! Reads `norm_y` from SMEM (produced by a prior reduction piece) and
//! writes one or more output buffers via simdgroup-cooperative dot
//! products. Each `GemvTarget` describes one (weight, output) pair
//! and an optional post-process step that runs on the just-computed
//! dot product before it's written.
//!
//! This is the piece form of the projection stage that
//! `rmsnorm_qkv`, `rmsnorm_gateup_swiglu`, and (with one target)
//! `final_rmsnorm_lm_head` share. The dispatch logic that maps
//! global simdgroup index `g` into the right output's row index
//! lives here so the rest of the pipeline can stay branch-free.

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::QuantInfo as Quant;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece, SmemRegion};

/// One destination of the multi-output GEMV: which weight to multiply
/// `norm_y` against, where to write, and the optional post-process
/// step (e.g. RoPE rotate before writing Q).
pub struct GemvTarget {
    /// MSL constexpr name holding the output row count, e.g. `"N_Q"`.
    pub count_const: &'static str,
    /// Compile-time row count value used to emit the constexpr declaration.
    pub count_value: u32,
    /// Variable name used in MSL for the weight buffer (e.g. `"W_q"`).
    pub weight_var: String,
    /// Variable name used for output (e.g. `"q"`).
    pub output_var: String,
    /// Optional separate bias buffer variable.
    pub bias_var: Option<String>,
    /// Role identifier (forwarded to BufferSlot for dedup).
    pub weight_role: String,
    pub output_role: String,
    pub bias_role: Option<String>,
    /// For quantized targets: extra buffers and dequant params.
    pub quant_scales_var: Option<String>,
    pub quant_biases_var: Option<String>,
    pub quant_scales_role: Option<String>,
    pub quant_biases_role: Option<String>,
}

pub struct MultiOutputGemv {
    pub targets: Vec<GemvTarget>,
    pub quant: Option<Quant>,
    /// Reduction-axis constexpr name (typically `"HIDDEN"`).
    pub reduction_const: &'static str,
    /// Activation MSL type override (None → use ctx.act_ty).
    pub act_ty_override: Option<&'static str>,
}

impl MultiOutputGemv {
    pub fn new(reduction_const: &'static str, quant: Option<Quant>) -> Self {
        Self {
            targets: Vec::new(),
            quant,
            reduction_const,
            act_ty_override: None,
        }
    }

    pub fn add(mut self, t: GemvTarget) -> Self {
        self.targets.push(t);
        self
    }

    fn act_ty<'a>(&self, ctx: &'a EmitCtx) -> &'a str {
        self.act_ty_override.unwrap_or(ctx.act_ty)
    }

    fn bias_ty<'a>(&self, ctx: &'a EmitCtx) -> &'a str {
        ctx.act_ty
    }
}

impl KernelPiece for MultiOutputGemv {
    fn name(&self) -> &str {
        "multi_gemv"
    }

    fn smem_regions(&self, _ctx: &EmitCtx) -> Vec<SmemRegion> {
        // Reader of `norm_y` — declared by RmsNormReduce (or any
        // producer earlier in the chain). No own SMEM beyond what
        // the reduction piece allocates.
        Vec::new()
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let t = self.act_ty(ctx);
        let bt = self.bias_ty(ctx);
        let mut out: Vec<BufferSlot> = Vec::new();
        for tgt in &self.targets {
            if self.quant.is_some() {
                out.push(BufferSlot {
                    var_name: tgt.weight_var.clone(),
                    elem_ty: "uint".into(),
                    access: BufferAccess::Read,
                    role: tgt.weight_role.clone(),
                });
                if let (Some(sv), Some(sr)) = (&tgt.quant_scales_var, &tgt.quant_scales_role) {
                    out.push(BufferSlot {
                        var_name: sv.clone(),
                        elem_ty: t.into(),
                        access: BufferAccess::Read,
                        role: sr.clone(),
                    });
                }
                if let (Some(bv), Some(br)) = (&tgt.quant_biases_var, &tgt.quant_biases_role) {
                    out.push(BufferSlot {
                        var_name: bv.clone(),
                        elem_ty: t.into(),
                        access: BufferAccess::Read,
                        role: br.clone(),
                    });
                }
            } else {
                out.push(BufferSlot {
                    var_name: tgt.weight_var.clone(),
                    elem_ty: t.into(),
                    access: BufferAccess::Read,
                    role: tgt.weight_role.clone(),
                });
            }
        }
        for tgt in &self.targets {
            if let (Some(bv), Some(br)) = (&tgt.bias_var, &tgt.bias_role) {
                out.push(BufferSlot {
                    var_name: bv.clone(),
                    elem_ty: bt.into(),
                    access: BufferAccess::Read,
                    role: br.clone(),
                });
            }
        }
        for tgt in &self.targets {
            out.push(BufferSlot {
                var_name: tgt.output_var.clone(),
                elem_ty: t.into(),
                access: BufferAccess::Write,
                role: tgt.output_role.clone(),
            });
        }
        out
    }

    fn smem_reads(&self) -> Vec<&'static str> {
        vec!["norm_y"]
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        let t = self.act_ty(ctx);

        // Per-target count constexprs — duplicates are harmless.
        for tgt in &self.targets {
            w.writeln(format_args!(
                "constexpr uint {name} = {val};",
                name = tgt.count_const,
                val = tgt.count_value
            ));
        }

        // Build the cumulative dispatch range string and emit if/elif
        // chain. `g = gid * sgs_per_tg + simd_id` is the global
        // simdgroup index across the union of all outputs.
        w.line("uint g = gid * sgs_per_tg + simd_id;");
        let mut cumulative: Vec<String> = Vec::with_capacity(self.targets.len());
        let mut acc = String::new();
        for (i, tgt) in self.targets.iter().enumerate() {
            if i == 0 {
                acc.push_str(tgt.count_const);
            } else {
                acc = format!("{acc} + {}", tgt.count_const);
            }
            cumulative.push(acc.clone());
        }
        for (i, tgt) in self.targets.iter().enumerate() {
            let head = if i == 0 { "if" } else { "else if" };
            let upper = &cumulative[i];
            let lower = if i == 0 {
                String::new()
            } else {
                cumulative[i - 1].clone()
            };
            w.line(format!("{head} (g < {upper}) {{"));
            w.push_indent();
            // Map global g to per-target row index `i_row`.
            if i == 0 {
                w.line("uint i_row = g;");
            } else {
                w.line(format!("uint i_row = g - ({lower});"));
            }
            self.emit_row_body(w, ctx, tgt, t);
            w.pop_indent();
            w.line("}");
        }
    }
}

impl MultiOutputGemv {
    fn emit_row_body(&self, w: &mut MslWriter, _ctx: &EmitCtx, tgt: &GemvTarget, t: &str) {
        w.line("float acc = 0.0f;");
        let red = self.reduction_const;
        match self.quant {
            Some(q) => {
                let bits = q.bits;
                let group_size = q.group_size;
                let packed_row = format!(
                    "{w} + (i_row) * ({red} / {n})",
                    w = tgt.weight_var,
                    n = 32 / bits
                );
                let scales_var = tgt
                    .quant_scales_var
                    .as_deref()
                    .expect("quant target must have scales");
                let biases_var = tgt
                    .quant_biases_var
                    .as_deref()
                    .expect("quant target must have biases");
                let scales_row = format!("{scales_var} + (i_row) * ({red} / {group_size}u)");
                let biases_row = format!("{biases_var} + (i_row) * ({red} / {group_size}u)");
                crate::codegen::kernels::emit_int4_gemv_inner(
                    w,
                    "acc",
                    &packed_row,
                    &scales_row,
                    &biases_row,
                    red,
                    bits,
                    group_size,
                    |k| format!("norm_y[{k}]"),
                );
            }
            None => {
                w.line(format!(
                    "device const {t}4 *W4_ = (device const {t}4 *)({w} + (i_row) * {red});",
                    w = tgt.weight_var,
                ));
                w.line("threadgroup float4 *ny4_ = (threadgroup float4 *)norm_y;");
                w.writeln(format_args!(
                    "for (uint k4_ = simd_lane; k4_ < {red} / 4u; k4_ += 32u) {{"
                ));
                w.push_indent();
                w.line("acc += dot(ny4_[k4_], float4(W4_[k4_]));");
                w.pop_indent();
                w.line("}");
            }
        }
        w.line("acc = simd_sum(acc);");
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        if let Some(bv) = &tgt.bias_var {
            w.writeln(format_args!("acc += float({bv}[i_row]);"));
        }
        w.writeln(format_args!(
            "{out}[i_row] = {t}(acc);",
            out = tgt.output_var,
            t = t,
        ));
        w.pop_indent();
        w.line("}");
    }
}
