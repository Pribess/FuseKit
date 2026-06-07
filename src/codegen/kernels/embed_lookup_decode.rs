//! MSL emit for `FusionKind::EmbedLookupDecode`.
//!
//! Decode-time embedding lookup. Reads active token ids from a small
//! uniform buffer and copies/dequantizes the corresponding embedding
//! rows into the pre-allocated layer-0 input buffer.
//!
//! ```text
//!   x[row, d] = embed_table[token_id[row], d]
//! ```

use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::EmbedLookupDecode) {
        bail!("embed_lookup_decode emitter: wrong kind {:?}", fusion.kind);
    }
    if !matches!(fusion.attrs, FusionAttrs::EmbedLookupDecode) {
        bail!("embed_lookup_decode emitter: attrs mismatch");
    }

    let hidden = meta.hidden_size;
    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let active_m = crate::target::current_chip().active_m(8).max(1);
    let quant = meta
        .quant_bits
        .zip(meta.quant_group_size)
        .filter(|(bits, group)| matches!(*bits, 4 | 8) && *group > 0);

    w.line(format!(
        "// Fusion #{}: EmbedLookupDecode (active_m = {}, hidden = {}, dt = {})",
        fusion.id, active_m, hidden, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    if quant.is_some() {
        w.line("device const uint  *embed_packed [[buffer(0)]],");
        w.line(format!("device const {t} *embed_scales [[buffer(1)]],"));
        w.line(format!("device const {t} *embed_biases [[buffer(2)]],"));
        w.line("device const uint  *token_ids    [[buffer(3)]],");
        w.line(format!("device {t}       *x            [[buffer(4)]],"));
    } else {
        w.line(format!("device const {t} *embed_table [[buffer(0)]],"));
        w.line("device const uint  *token_ids   [[buffer(1)]],");
        w.line(format!("device {t}       *x           [[buffer(2)]],"));
    }
    w.line("uint tid     [[thread_position_in_threadgroup]],");
    w.line("uint tg_size [[threads_per_threadgroup]],");
    w.line("uint gid     [[threadgroup_position_in_grid]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.writeln(format_args!("constexpr uint ACTIVE_M = {active_m};"));
    w.writeln(format_args!("constexpr uint HIDDEN = {hidden};"));
    w.blank();
    w.line("uint i = gid * tg_size + tid;");
    w.line("if (i >= ACTIVE_M * HIDDEN) return;");
    w.line("uint row = i / HIDDEN;");
    w.line("uint d = i - row * HIDDEN;");
    w.line("uint token_id = token_ids[row];");
    if let Some((bits, group)) = quant {
        let lanes = 32 / bits;
        w.writeln(format_args!("constexpr uint BITS = {bits}u;"));
        w.writeln(format_args!("constexpr uint GROUP = {group}u;"));
        w.writeln(format_args!("constexpr uint LANES = {lanes}u;"));
        w.line("constexpr uint MASK = (1u << BITS) - 1u;");
        w.line("constexpr uint N_WORDS = HIDDEN / LANES;");
        w.line("constexpr uint N_GROUPS = HIDDEN / GROUP;");
        w.line("uint word = embed_packed[token_id * N_WORDS + d / LANES];");
        w.line("uint q = (word >> ((d % LANES) * BITS)) & MASK;");
        w.line("uint g = d / GROUP;");
        w.line("float s = float(embed_scales[token_id * N_GROUPS + g]);");
        w.line("float b = float(embed_biases[token_id * N_GROUPS + g]);");
        w.line(format!("x[row * HIDDEN + d] = {t}(float(q) * s + b);"));
    } else {
        w.line("x[row * HIDDEN + d] = embed_table[token_id * HIDDEN + d];");
    }
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_embed_lookup_decode_f{}", fusion.id)
}
