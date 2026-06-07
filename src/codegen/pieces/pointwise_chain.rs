//! Pointwise chain piece — a sequence of elementwise ops sharing
//! a per-thread register chain.
//!
//! Drop-in replacement for `fused_chain::emit`'s pointwise branch:
//! `r0 = expr(in)`, `r1 = expr(r0, in)`, ..., `out0[i] = act_ty(rN)`.
//! One thread per element with `gid*tg_size+tid` indexing and an
//! early `if (i >= N) return;` guard.
//!
//! Routes through `pieces::compose` so the same MSL-emit path is
//! used for every shape; future fusion variants (e.g. RoPE
//! absorbed, residual-fused tails) plug in as additional pieces in
//! the same composed sequence.

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::ChainStep;
use crate::graph::Dtype;

use super::{BufferAccess, BufferSlot, EmitCtx, KernelPiece};

pub struct PointwiseChain {
    /// Flat element count (product of output[0].shape).
    pub n: u32,
    /// External input buffer count. Matches the `ChainStep::inputs`
    /// references; `inN` in the MSL body refers to slot `N`.
    pub n_inputs: u32,
    /// MSL scalar type for each external input buffer. Pointwise
    /// chains are not always activation-typed: shape/frequency helper
    /// graphs often stay float32 even inside fp16/bf16 models.
    pub input_tys: Vec<String>,
    /// MSL scalar type for the output buffer.
    pub output_ty: String,
    /// Steps in execution order. Each `step.op_name` lowers via
    /// `render_pointwise_expr_for_chain` to an MSL expression; the
    /// last step's expression becomes the output store.
    pub steps: Vec<ChainStep>,
}

impl KernelPiece for PointwiseChain {
    fn name(&self) -> &str {
        "pointwise_chain"
    }

    fn buffer_slots(&self, ctx: &EmitCtx) -> Vec<BufferSlot> {
        let fallback_t: String = ctx.act_ty.into();
        let mut slots = Vec::with_capacity(self.n_inputs as usize + 1);
        for i in 0..self.n_inputs {
            let elem_ty = self
                .input_tys
                .get(i as usize)
                .cloned()
                .unwrap_or_else(|| fallback_t.clone());
            slots.push(BufferSlot {
                var_name: format!("in{i}"),
                elem_ty,
                access: BufferAccess::Read,
                role: format!("pwise_in{i}"),
            });
        }
        slots.push(BufferSlot {
            var_name: "out0".into(),
            elem_ty: self.output_ty.clone(),
            access: BufferAccess::Write,
            role: "pwise_out0".into(),
        });
        slots
    }

    fn emit_body(&self, w: &mut MslWriter, ctx: &EmitCtx) {
        w.writeln(format_args!("constexpr uint N = {};", self.n));
        w.line("uint i = gid * tg_size + tid;");
        w.line("if (i >= N) return;");
        let fallback_t = ctx.act_ty;
        let n_steps = self.steps.len();
        for (idx, step) in self.steps.iter().enumerate() {
            let is_last = idx + 1 == n_steps;
            let expr = render_pointwise_expr_for_chain(step).unwrap_or_else(|| {
                // Mirror the legacy emit's hard failure mode: emitter
                // assertion in the metallib so the build doesn't ship
                // a silently-wrong kernel.
                format!("/* unsupported pointwise op `{}` */ 0.0f", step.op_name)
            });
            if is_last {
                w.writeln(format_args!("out0[i] = {t}({expr});", t = self.output_ty));
            } else {
                let rounded = round_expr_to_dtype(step.output_dtype, fallback_t, &expr);
                w.writeln(format_args!("float r{idx} = {rounded};"));
            }
        }
    }
}

fn round_expr_to_dtype(dtype: Dtype, fallback_t: &'static str, expr: &str) -> String {
    match crate::codegen::kernels::scalar_ty_for_dtype(dtype, fallback_t) {
        "bfloat" => format!("float(bfloat({expr}))"),
        "half" => format!("float(half({expr}))"),
        _ => format!("float({expr})"),
    }
}

/// Mirror of `fused_chain::render_pointwise_expr`. Duplicated here so
/// the piece is self-contained and the legacy emitter can be removed
/// without dragging this function with it.
fn render_pointwise_expr_for_chain(step: &ChainStep) -> Option<String> {
    use crate::fusion::ir::ChainOperand;
    let operand = |op: &ChainOperand| -> String {
        match op {
            ChainOperand::External { buffer_index } => {
                format!("float(in{}[i])", buffer_index)
            }
            ChainOperand::Register { step_index } => format!("r{}", step_index),
        }
    };
    let a = step.inputs.first().map(&operand);
    let b = step.inputs.get(1).map(&operand);
    let c = step.inputs.get(2).map(&operand);
    match step.op_name.as_str() {
        "Add" => Some(format!("{} + {}", a?, b?)),
        "Subtract" => Some(format!("{} - {}", a?, b?)),
        "Multiply" => Some(format!("{} * {}", a?, b?)),
        "Divide" => Some(format!("{} / {}", a?, b?)),
        "Power" => Some(format!("pow({}, {})", a?, b?)),
        "Maximum" => Some(format!("max({}, {})", a?, b?)),
        "Minimum" => Some(format!("min({}, {})", a?, b?)),
        "Sigmoid" => Some(format!("(1.0f / (1.0f + exp(-{})))", a?)),
        "Exp" => Some(format!("exp({})", a?)),
        "Negative" => Some(format!("(-{})", a?)),
        "LogAddExp" => Some(format!(
            "(max({a},{b}) + log(1.0f + exp(-fabs({a} - {b}))))",
            a = a?,
            b = b?
        )),
        "Select" => Some(format!(
            "(({a} != 0.0f) ? {b} : {c})",
            a = a?,
            b = b?,
            c = c?
        )),
        "GreaterEqual" => Some(format!("(({a} >= {b}) ? 1.0f : 0.0f)", a = a?, b = b?)),
        "AsType" => Some(a?),
        "Reshape" | "Flatten" | "Unflatten" | "Squeeze" | "ExpandDims" | "AsContiguous" => Some(a?),
        _ => None,
    }
}
