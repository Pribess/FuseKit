use anyhow::{bail, Result};

use crate::codegen::msl::MslWriter;
use crate::fusion::ir::{Fusion, FusionAttrs, FusionKind, ModelMeta};
use crate::task::ir::PartitionPlan;

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_qmv_fanout_f{}", fusion.id)
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.saturating_add(b.saturating_sub(1)) / b.max(1)
}

pub fn emit(
    fusion: &Fusion,
    _partition: &PartitionPlan,
    meta: &ModelMeta,
    w: &mut MslWriter,
) -> Result<()> {
    if !matches!(fusion.kind, FusionKind::QmvFanout) {
        bail!("QmvFanout emitter called for non-QmvFanout fusion");
    }
    let FusionAttrs::QmvFanout { targets } = &fusion.attrs else {
        bail!("QmvFanout attrs missing");
    };
    if targets.is_empty() || fusion.inputs.len() != 1 + targets.len() * 3 {
        bail!("QmvFanout malformed inputs/targets");
    }
    if fusion.outputs.len() != targets.len() {
        bail!("QmvFanout output count mismatch");
    }

    let exact = crate::codegen::kernels::qmv_fanout_exact_enabled()
        || targets.iter().any(|target| {
            !crate::codegen::kernels::qmv_fast_shape_is_tail_free(
                target.k_dim,
                target.out_dim,
                target.active_rows,
                target.bits,
            )
        });
    let first = targets[0];
    let qmv_sgs = if exact {
        crate::codegen::kernels::qmv_fanout_exact_simdgroups()
    } else {
        crate::codegen::kernels::qmv_fanout_targets_simdgroups(targets)
    };
    let qmv_cols_per_sg = if exact {
        crate::codegen::kernels::qmv_fanout_exact_cols_per_simdgroup()
    } else {
        crate::codegen::kernels::qmv_fanout_targets_cols_per_simdgroup(targets)
    };
    let target_set_tiling = if exact {
        None
    } else {
        Some((qmv_sgs, qmv_cols_per_sg))
    };
    let target_set_tiling_override =
        !exact && crate::codegen::kernels::qmv_fanout_targets_use_set_tiling(targets);
    let token_rows = crate::codegen::kernels::qmv_fanout_token_rows(targets);
    for target in targets {
        let (sgs, cols) = target_set_tiling.unwrap_or_else(|| {
            (
                crate::codegen::kernels::qmv_fanout_exact_simdgroups(),
                crate::codegen::kernels::qmv_fanout_exact_cols_per_simdgroup(),
            )
        });
        if target.k_dim != first.k_dim
            || target.active_rows != first.active_rows
            || sgs != qmv_sgs
            || cols != qmv_cols_per_sg
        {
            bail!(
                "QmvFanout target tiling mismatch: first=(K={}, M={}, N={}, bits={}, gs={}) target=(K={}, M={}, N={}, bits={}, gs={}) kernel=(sgs={}, cols={}) target_set=(sgs={}, cols={})",
                first.k_dim,
                first.active_rows,
                first.out_dim,
                first.bits,
                first.group_size,
                target.k_dim,
                target.active_rows,
                target.out_dim,
                target.bits,
                target.group_size,
                sgs,
                cols,
                qmv_sgs,
                qmv_cols_per_sg
            );
        }
        if !exact && !target_set_tiling_override {
            let target_sgs = crate::codegen::kernels::qmv_fanout_fast_simdgroups(
                target.active_rows,
                target.bits,
                target.k_dim,
                target.out_dim,
            );
            let target_cols = crate::codegen::kernels::qmv_fanout_fast_cols_per_simdgroup(
                target.active_rows,
                target.bits,
                target.k_dim,
                target.out_dim,
            );
            if target_sgs != qmv_sgs || target_cols != qmv_cols_per_sg {
                bail!(
                    "QmvFanout target tiling mismatch: target=(K={}, M={}, N={}, bits={}, gs={}) target_tiling=(sgs={}, cols={}) target_set=(sgs={}, cols={})",
                    target.k_dim,
                    target.active_rows,
                    target.out_dim,
                    target.bits,
                    target.group_size,
                    target_sgs,
                    target_cols,
                    qmv_sgs,
                    qmv_cols_per_sg
                );
            }
        }
    }

    let t = crate::codegen::kernels::act_ty(meta);
    let fn_name = function_name(fusion);
    let n_inputs = fusion.inputs.len();
    let bits_desc = if targets.iter().all(|target| target.bits == first.bits) {
        first.bits.to_string()
    } else {
        targets
            .iter()
            .map(|target| target.bits.to_string())
            .collect::<Vec<_>>()
            .join("/")
    };
    let group_desc = if targets
        .iter()
        .all(|target| target.group_size == first.group_size)
    {
        first.group_size.to_string()
    } else {
        targets
            .iter()
            .map(|target| target.group_size.to_string())
            .collect::<Vec<_>>()
            .join("/")
    };
    w.line(format!(
        "// Fusion #{}: QmvFanout{} (targets={}, K={}, M={}, bits={}, gs={}, sgs={}, cols/sg={}, token_rows={})",
        fusion.id,
        if exact { " exact" } else { "" },
        targets.len(),
        first.k_dim,
        first.active_rows,
        bits_desc,
        group_desc,
        qmv_sgs,
        qmv_cols_per_sg,
        token_rows
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    w.line(format!("device const {t} *in0 [[buffer(0)]],"));
    for target_idx in 0..targets.len() {
        let base = 1 + target_idx * 3;
        w.line(format!("device const uint *in{base} [[buffer({base})]],"));
        w.line(format!(
            "device const {t} *in{} [[buffer({})]],",
            base + 1,
            base + 1
        ));
        w.line(format!(
            "device const {t} *in{} [[buffer({})]],",
            base + 2,
            base + 2
        ));
    }
    for out_idx in 0..targets.len() {
        w.line(format!(
            "device {t}       *out{out_idx} [[buffer({})]],",
            n_inputs + out_idx
        ));
    }
    w.line("uint3 tgid [[threadgroup_position_in_grid]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();
    w.line("#if FUSEKIT_HAVE_MLX_NAX");
    w.push_indent();
    w.line("uint local_y = tgid.y;");
    let mut target_idx = 0usize;
    while target_idx < targets.len() {
        let target = targets[target_idx];
        let base = 1 + target_idx * 3;
        let cols_per_tg = if exact {
            crate::codegen::kernels::qmv_fanout_exact_cols_per_threadgroup()
        } else {
            crate::codegen::kernels::qmv_fanout_targets_cols_per_threadgroup(targets)
        };
        let tiles = ceil_div(target.out_dim, cols_per_tg);
        let pair2_exact = exact
            && target_idx + 1 < targets.len()
            && crate::codegen::kernels::qmv_fanout_exact_pair2_compatible(
                target,
                targets[target_idx + 1],
            );
        let pair2_m = !exact
            && target_idx + 1 < targets.len()
            && crate::codegen::kernels::qmv_fanout_pair2_m_compatible(
                target,
                targets[target_idx + 1],
            );
        let pair2 = pair2_exact
            || pair2_m
            || (!exact
                && target_idx + 1 < targets.len()
                && crate::codegen::kernels::qmv_fanout_pair2_compatible(
                    target,
                    targets[target_idx + 1],
                ));
        if pair2 {
            let next_base = 1 + (target_idx + 1) * 3;
            w.writeln(format_args!("if (local_y < {tiles}u) {{"));
            w.push_indent();
            w.line("uint3 local_tgid = uint3(tgid.x, local_y, tgid.z);");
            w.line("// same-tile pair path: share x vector load across two sibling qMVs.");
            w.writeln(format_args!("constexpr int K = {};", target.k_dim));
            w.writeln(format_args!("constexpr int N = {};", target.out_dim));
            if pair2_exact {
                w.writeln(format_args!(
                    "fusekit_qmv_fanout2_exact_same_tile_impl<{t}, {}, {}, {}, {}>(in{base}, in{}, in{}, in{next_base}, in{}, in{}, in0, out{target_idx}, out{}, K, N, local_tgid, simd_gid, simd_lid);",
                    target.group_size,
                    target.bits,
                    qmv_sgs,
                    qmv_cols_per_sg,
                    base + 1,
                    base + 2,
                    next_base + 1,
                    next_base + 2,
                    target_idx + 1
                ));
            } else if pair2_m {
                w.writeln(format_args!(
                    "constexpr int ACTIVE_ROWS = {};",
                    target.active_rows
                ));
                w.writeln(format_args!(
                    "fusekit_qmv_fanout2_same_tile_m_impl<{t}, {}, {}, {}, {}, {}>(in{base}, in{}, in{}, in{next_base}, in{}, in{}, in0, out{target_idx}, out{}, K, N, ACTIVE_ROWS, local_tgid, simd_gid, simd_lid);",
                    target.group_size,
                    target.bits,
                    qmv_sgs,
                    qmv_cols_per_sg,
                    token_rows,
                    base + 1,
                    base + 2,
                    next_base + 1,
                    next_base + 2,
                    target_idx + 1
                ));
            } else {
                w.writeln(format_args!(
                    "fusekit_qmv_fanout2_same_tile_impl<{t}, {}, {}, {}, {}>(in{base}, in{}, in{}, in{next_base}, in{}, in{}, in0, out{target_idx}, out{}, K, N, local_tgid, simd_gid, simd_lid);",
                    target.group_size,
                    target.bits,
                    qmv_sgs,
                    qmv_cols_per_sg,
                    base + 1,
                    base + 2,
                    next_base + 1,
                    next_base + 2,
                    target_idx + 1
                ));
            }
            w.line("return;");
            w.pop_indent();
            w.line("}");
            if target_idx + 2 != targets.len() {
                w.writeln(format_args!("local_y -= {tiles}u;"));
            }
            target_idx += 2;
        } else {
            w.writeln(format_args!("if (local_y < {tiles}u) {{"));
            w.push_indent();
            w.line("uint3 local_tgid = uint3(tgid.x, local_y, tgid.z);");
            w.writeln(format_args!("constexpr int K = {};", target.k_dim));
            w.writeln(format_args!("constexpr int N = {};", target.out_dim));
            if exact {
                w.writeln(format_args!(
                    "fusekit_qmv_impl<{t}, {}, {}, {}, {}>(in{base}, in{}, in{}, in0, out{target_idx}, K, N, local_tgid, simd_gid, simd_lid);",
                    target.group_size,
                    target.bits,
                    qmv_sgs,
                    qmv_cols_per_sg,
                    base + 1,
                    base + 2
                ));
            } else {
                let use_m_impl = token_rows > 1
                    || (target.active_rows > 1
                        && std::env::var("FUSEKIT_QMV_FANOUT_M_IMPL_TOKEN1").as_deref() == Ok("1"));
                if use_m_impl {
                    w.writeln(format_args!(
                        "constexpr int ACTIVE_ROWS = {};",
                        target.active_rows
                    ));
                    w.writeln(format_args!(
                        "fusekit_qmv_fast_m_impl<{t}, {}, {}, {}, {}, {}>(in{base}, in{}, in{}, in0, out{target_idx}, K, N, ACTIVE_ROWS, local_tgid, simd_gid, simd_lid);",
                        target.group_size,
                        target.bits,
                        qmv_sgs,
                        qmv_cols_per_sg,
                        token_rows,
                        base + 1,
                        base + 2
                    ));
                } else if qmv_cols_per_sg == 2 {
                    w.writeln(format_args!(
                        "fusekit_qmv_fast2_impl<{t}, {}, {}, {}>(in{base}, in{}, in{}, in0, out{target_idx}, K, N, local_tgid, simd_gid, simd_lid);",
                        target.group_size,
                        target.bits,
                        qmv_sgs,
                        base + 1,
                        base + 2
                    ));
                } else {
                    w.writeln(format_args!(
                        "fusekit_qmv_fast_impl<{t}, {}, {}, {}, {}>(in{base}, in{}, in{}, in0, out{target_idx}, K, N, local_tgid, simd_gid, simd_lid);",
                        target.group_size,
                        target.bits,
                        qmv_sgs,
                        qmv_cols_per_sg,
                        base + 1,
                        base + 2
                    ));
                }
            }
            w.line("return;");
            w.pop_indent();
            w.line("}");
            if target_idx + 1 != targets.len() {
                w.writeln(format_args!("local_y -= {tiles}u;"));
            }
            target_idx += 1;
        }
    }
    w.line("return;");
    w.pop_indent();
    w.line("#else");
    w.push_indent();
    w.line(
        "static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit QmvFanout requires MLX Metal headers\");",
    );
    w.pop_indent();
    w.line("#endif");
    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}
