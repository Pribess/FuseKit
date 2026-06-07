//! Decode-time `ArraysCache` sliding-window update.
//!
//! Qwen3/Kimi-style linear attention keeps a short convolution state in
//! `ArraysCache[0]`: `[B, K-1, C]`. At decode time the current token's
//! projected `qkv` row is appended and the oldest state row is dropped.
//! This kernel performs that semantic cache update in-place.

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_arrays_cache_shift_append_decode_f{}", fusion.id)
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    let (state_len, width) = match fusion.attrs {
        FusionAttrs::ArraysCacheShiftAppendDecode {
            state_len, width, ..
        } => (state_len, width),
        _ => bail!("ArraysCacheShiftAppendDecode emitter: attrs mismatch"),
    };
    if fusion.inputs.len() != 1 {
        bail!("ArraysCacheShiftAppendDecode expects one projection input");
    }
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = fusion
        .inputs
        .first()
        .map(|b| crate::target::current_chip().active_tokens_for_shape(&b.shape))
        .unwrap_or_else(|| crate::target::current_chip().active_m(8))
        .max(1)
        .min(state_len.max(1));
    w.line(format!(
        "// Fusion #{}: ArraysCacheShiftAppendDecode(active_m={}, state_len={}, width={})",
        fusion.id, active_m, state_len, width
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *current [[buffer(0)]],"));
    w.line(format!("device {t}       *state   [[buffer(1)]],"));
    if active_m > 1 {
        w.line(format!("device {t}       *mid_state [[buffer(2)]],"));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.writeln(format_args!("constexpr uint STATE_LEN = {state_len}u;"));
    w.writeln(format_args!("constexpr uint WIDTH = {width}u;"));
    w.writeln(format_args!("constexpr uint ACTIVE_M = {active_m}u;"));
    w.line("uint c = gid * tg_size + tid;");
    w.line("if (c >= WIDTH) return;");
    if active_m == 2 && state_len == 3 {
        if active_m > 1 {
            w.line(format!("{t} s1 = state[WIDTH + c];"));
            w.line(format!("{t} s2 = state[2u * WIDTH + c];"));
            w.line(format!("{t} cur0 = current[c];"));
            w.line(format!("{t} cur1 = current[WIDTH + c];"));
            w.line("mid_state[c] = s1;");
            w.line("mid_state[WIDTH + c] = s2;");
            w.line("mid_state[2u * WIDTH + c] = cur0;");
            w.line("state[c] = s2;");
            w.line("state[WIDTH + c] = cur0;");
            w.line("state[2u * WIDTH + c] = cur1;");
            w.pop_indent();
            w.line("}");
            w.blank();
            return Ok(());
        }
    }
    if active_m > 1 {
        w.line("// Snapshot the cache state after the confirmed token (row 0).");
        w.line("for (uint s = 0u; s + 1u < STATE_LEN; ++s) {");
        w.push_indent();
        w.line("mid_state[s * WIDTH + c] = state[(s + 1u) * WIDTH + c];");
        w.pop_indent();
        w.line("}");
        w.line("mid_state[(STATE_LEN - 1u) * WIDTH + c] = current[c];");
    }
    w.line("for (uint s = 0u; s + ACTIVE_M < STATE_LEN; ++s) {");
    w.push_indent();
    w.line("state[s * WIDTH + c] = state[(s + ACTIVE_M) * WIDTH + c];");
    w.pop_indent();
    w.line("}");
    // `current` may be trace-shaped [1, L, C] / [L, C] scratch. The
    // active rows are the first contiguous `ACTIVE_M * WIDTH` elements.
    w.line("for (uint r = 0u; r < ACTIVE_M; ++r) {");
    w.push_indent();
    w.line("state[(STATE_LEN - ACTIVE_M + r) * WIDTH + c] = current[r * WIDTH + c];");
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
