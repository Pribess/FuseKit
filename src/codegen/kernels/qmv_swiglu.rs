use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta, QmvFanoutTarget};
use crate::task::ir::PartitionPlan;

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_qmv_swiglu_f{}", fusion.id)
}

fn compatible(gate: QmvFanoutTarget, up: QmvFanoutTarget) -> bool {
    gate.k_dim == up.k_dim
        && gate.out_dim == up.out_dim
        && gate.active_rows == up.active_rows
        && gate.bits == up.bits
        && gate.group_size == up.group_size
        && matches!(gate.bits, 4 | 8)
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::QmvSwiglu) {
        bail!("QmvSwiglu emitter called for non-QmvSwiglu fusion");
    }
    let FusionAttrs::QmvSwiglu { gate, up } = fusion.attrs else {
        bail!("QmvSwiglu attrs missing");
    };
    if !compatible(gate, up) {
        bail!("QmvSwiglu gate/up target mismatch");
    }
    if fusion.inputs.len() != 7 || fusion.outputs.len() != 1 {
        bail!("QmvSwiglu malformed inputs/outputs");
    }

    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let qmv_sgs = crate::codegen::kernels::qmv_fast_simdgroups(
        gate.active_rows,
        gate.bits,
        gate.k_dim,
        gate.out_dim,
    );
    let qmv_cols_per_sg = crate::codegen::kernels::qmv_fast_cols_per_simdgroup(
        gate.active_rows,
        gate.bits,
        gate.k_dim,
        gate.out_dim,
    );
    let token_rows = crate::codegen::kernels::qmv_fast_token_rows(
        gate.active_rows,
        gate.bits,
        gate.k_dim,
        gate.out_dim,
    );

    w.line(format!(
        "// Fusion #{}: QmvSwiglu (K={}, M={}, N={}, bits={}, gs={}, sgs={}, cols/sg={}, token_rows={})",
        fusion.id,
        gate.k_dim,
        gate.active_rows,
        gate.out_dim,
        gate.bits,
        gate.group_size,
        qmv_sgs,
        qmv_cols_per_sg,
        token_rows
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *x [[buffer(0)]],"));
    w.line("device const uint *gate_w [[buffer(1)]],");
    w.line(format!("device const {t} *gate_s [[buffer(2)]],"));
    w.line(format!("device const {t} *gate_b [[buffer(3)]],"));
    w.line("device const uint *up_w [[buffer(4)]],");
    w.line(format!("device const {t} *up_s [[buffer(5)]],"));
    w.line(format!("device const {t} *up_b [[buffer(6)]],"));
    w.line(format!("device {t} *out0 [[buffer(7)]],"));
    w.line("uint3 tgid [[threadgroup_position_in_grid]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("#if FUSEKIT_HAVE_MLX_NAX");
    w.push_indent();
    w.writeln(format_args!("constexpr int K = {};", gate.k_dim));
    w.writeln(format_args!("constexpr int N = {};", gate.out_dim));
    w.writeln(format_args!(
        "constexpr int ACTIVE_ROWS = {};",
        gate.active_rows
    ));
    w.writeln(format_args!(
        "fusekit_qmv_swiglu_pair_m_impl<{t}, {}, {}, {}, {}, {}>(gate_w, gate_s, gate_b, up_w, up_s, up_b, x, out0, K, N, ACTIVE_ROWS, tgid, simd_gid, simd_lid);",
        gate.group_size, gate.bits, qmv_sgs, qmv_cols_per_sg, token_rows
    ));
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line(
        "static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit QmvSwiglu requires MLX Metal headers\");",
    );
    w.pop_indent();
    w.line("#endif");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
