//! MSL emit for `FusionKind::LinearResidual`.
//!
//! Naive GEMV + per-row residual add. Shape `(M, N)` is read from the
//! fusion's boundary tensors, so each fusion instance bakes its own
//! `M`/`N` constants — no uniforms needed.
//!
//! ```text
//!   out[i] = Σ_k x[k] · W[i, k] + residual[i]   (i ∈ [0, N))
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
    if !matches!(fusion.kind, FusionKind::LinearResidual) {
        bail!("linear_residual emitter: wrong kind {:?}", fusion.kind);
    }
    let quant = match fusion.attrs {
        FusionAttrs::LinearResidual { quant } => quant,
        _ => bail!("linear_residual emitter: attrs mismatch"),
    };

    // MLX's `Linear` keeps weight in `(out, in)` row-major storage and
    // emits `x @ weight.T` as a Transpose-followed-by-Matmul in the
    // trace. The `.T` is a stride view — the underlying memory we
    // receive at runtime is the ORIGINAL (out, in) layout. So we read
    // dims from the activation (M = input dim) and the output
    // boundary (N = output dim) — never trust the traced W shape,
    // which is the transposed view.
    let x = &fusion.inputs[0];
    let y = &fusion.outputs[0];
    let residual_shape = fusion
        .inputs
        .last()
        .map(|b| b.shape.as_slice())
        .unwrap_or(&[]);
    let m = *x
        .shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("x has no shape"))? as u32;
    let n = *y
        .shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("y has no shape"))? as u32;
    let active_rows = crate::target::current_chip()
        .active_rows_before_last(&y.shape)
        .max(1) as u32;

    let t = crate::codegen::kernels::act_ty(meta);
    let mlx_qmv_fast = crate::codegen::kernels::linear_residual_mlx_qmv_fast_shape(fusion);
    let fn_name = function_name(fusion);
    w.line(format!(
        "// Fusion #{}: LinearResidual (M = {}, N = {}, active_rows = {}, quant = {:?}, dt = {})",
        fusion.id, m, n, active_rows, quant, t
    ));
    w.line(format!("kernel void {fn_name}("));
    w.push_indent();
    if quant.is_some() {
        // Quantized buffer order matches what the Python wrapper
        // pushes: x, W_packed, scales, biases, residual, y.
        w.line(format!("device const {t} *x         [[buffer(0)]],"));
        w.line("device const uint   *W_packed  [[buffer(1)]],");
        w.line(format!("device const {t}   *W_scales  [[buffer(2)]],"));
        w.line(format!("device const {t}   *W_biases  [[buffer(3)]],"));
        w.line(format!("device const {t} *residual  [[buffer(4)]],"));
        w.line(format!("device {t}       *y         [[buffer(5)]],"));
    } else {
        w.line(format!("device const {t} *x        [[buffer(0)]],"));
        w.line(format!("device const {t} *W        [[buffer(1)]],"));
        w.line(format!("device const {t} *residual [[buffer(2)]],"));
        w.line(format!("device {t}       *y        [[buffer(3)]],"));
    }
    w.line("uint3 tid     [[thread_position_in_threadgroup]],");
    w.line("uint3 tg_size [[threads_per_threadgroup]],");
    w.line("uint3 tg_pos [[threadgroup_position_in_grid]],");
    w.line("uint simd_gid [[simdgroup_index_in_threadgroup]],");
    w.line("uint simd_lid [[thread_index_in_simdgroup]]");
    w.pop_indent();
    w.line(") {");
    w.push_indent();

    w.writeln(format_args!("constexpr uint M = {m};"));
    w.writeln(format_args!("constexpr uint N = {n};"));
    w.writeln(format_args!("constexpr uint ACTIVE_ROWS = {active_rows};"));
    let residual_vector =
        residual_shape.len() == 1 && residual_shape.last().copied() == Some(n as i32);
    w.writeln(format_args!(
        "constexpr bool RESIDUAL_VECTOR = {};",
        if residual_vector { "true" } else { "false" }
    ));
    w.blank();

    if let Some((k_dim, out_dim, active_rows, bits, group_size)) =
        crate::codegen::kernels::linear_residual_nax_shape(fusion)
    {
        let aligned_n = if out_dim.is_multiple_of(64) {
            "true"
        } else {
            "false"
        };
        let (wm, wn) = if active_rows <= 16 { (1, 4) } else { (2, 2) };
        w.line("#if FUSEKIT_HAVE_MLX_NAX");
        w.push_indent();
        w.writeln(format_args!("constexpr int K_NAX = {k_dim};"));
        w.writeln(format_args!("constexpr int N_NAX = {out_dim};"));
        w.writeln(format_args!("constexpr int M_NAX = {active_rows};"));
        w.line(format!("threadgroup {t} Ws[64 * (64 + 16 / sizeof({t}))];"));
        w.writeln(format_args!(
            "fusekit_qmm_t_nax_residual_tgp_impl<{t}, {group_size}, {bits}, {aligned_n}, 64, 64, 64, {wm}, {wn}>("
        ));
        w.push_indent();
        w.line("W_packed, W_scales, W_biases, x, residual, y, Ws, K_NAX, N_NAX, M_NAX, RESIDUAL_VECTOR, tg_pos, tid.x, simd_gid, simd_lid);");
        w.pop_indent();
        w.pop_indent();
        w.line("#else");
        w.push_indent();
        w.line("static_assert(FUSEKIT_HAVE_MLX_NAX, \"FuseKit LinearResidual NAX requires MLX Metal headers\");");
        w.pop_indent();
        w.line("#endif");
        w.line("return;");
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }

    if let Some((k_dim, out_dim, active_rows, bits, group_size)) = mlx_qmv_fast {
        let qmv_sgs = crate::codegen::kernels::linear_residual_qmv_fast_simdgroups(
            active_rows,
            bits,
            k_dim,
            out_dim,
        );
        let qmv_cols_per_sg = crate::codegen::kernels::linear_residual_qmv_fast_cols_per_simdgroup(
            active_rows,
            bits,
            k_dim,
            out_dim,
        );
        let token_rows = crate::codegen::kernels::linear_residual_qmv_fast_token_rows(
            active_rows,
            bits,
            k_dim,
            out_dim,
        );
        let tail_free =
            crate::codegen::kernels::qmv_fast_shape_is_tail_free(k_dim, out_dim, active_rows, bits);
        if token_rows > 1 {
            let helper = if tail_free {
                "fusekit_qmv_fast_residual_m_impl"
            } else {
                "fusekit_qmv_fast_residual_m_tail_impl"
            };
            w.writeln(format_args!(
                "{helper}<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}, {token_rows}, 1>(W_packed, W_scales, W_biases, x, residual, y, int(M), int(N), int({active_rows}), tg_pos, simd_gid, simd_lid);"
            ));
        } else if active_rows == 1
            && matches!(bits, 4 | 8)
            && k_dim >= 4096
            && tail_free
            && std::env::var("FUSEKIT_LINEAR_RESIDUAL_TGX").as_deref() == Ok("1")
        {
            w.line("threadgroup float fusekit_x_smem[1024];");
            w.writeln(format_args!(
                "fusekit_qmv_fast_residual_tgx_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}, 1>(W_packed, W_scales, W_biases, x, residual, y, fusekit_x_smem, int(M), int(N), tg_pos, tid.x, tg_size.x, simd_gid, simd_lid);"
            ));
        } else {
            w.writeln(format_args!(
                "fusekit_qmv_fast_residual_impl<{t}, {group_size}, {bits}, {qmv_sgs}, {qmv_cols_per_sg}, 1>(W_packed, W_scales, W_biases, x, residual, y, int(M), int(N), tg_pos, simd_gid, simd_lid);"
            ));
        }
        w.line("return;");
        w.pop_indent();
        w.line("}");
        w.blank();
        return Ok(());
    }

    w.line("uint simd_lane = tid.x % 32u;");
    w.line("uint simd_id   = tid.x / 32u;");
    w.line("uint sgs_per_tg = tg_size.x / 32u;");
    let dense_rows = 1u32;
    w.writeln(format_args!("constexpr uint DENSE_ROWS = {dense_rows}u;"));
    w.line("uint flat = (tg_pos.x * sgs_per_tg + simd_id) * DENSE_ROWS;");
    w.line("if (flat >= ACTIVE_ROWS * N) return;");
    w.line("uint row = flat / N;");
    w.line("uint i = flat - row * N;");
    w.line("float acc = 0.0f;");
    if let Some(q) = quant {
        let group_size = q.group_size;
        let bits = q.bits;
        let packed_words = crate::codegen::kernels::quant_packed_words(m as u32, bits);
        let n_groups = (m as u32 / group_size.max(1)).max(1);
        let w_row = format!("W_packed + i * {packed_words}u");
        let s_row = format!("W_scales + i * {n_groups}u");
        let b_row = format!("W_biases + i * {n_groups}u");
        crate::codegen::kernels::emit_int4_gemv_inner(
            w,
            "acc",
            &w_row,
            &s_row,
            &b_row,
            "M",
            bits,
            group_size,
            |off| format!("float(x[row * M + {off}])"),
        );
    } else if dense_rows > 1 && m % 16 == 0 {
        for r in 1..dense_rows {
            w.writeln(format_args!("float acc{r} = 0.0f;"));
        }
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(x + row * M);"
        ));
        for r in 0..dense_rows {
            w.writeln(format_args!(
                "device const {t}4* W4_{r} = (device const {t}4*)(W + (i + {r}u) * M);"
            ));
        }
        w.line("for (uint k4 = simd_lane * 4u; k4 < M / 4u; k4 += 128u) {");
        w.push_indent();
        w.line("float4 xv0 = float4(x4[k4]);");
        w.line("float4 xv1 = float4(x4[k4 + 1u]);");
        w.line("float4 xv2 = float4(x4[k4 + 2u]);");
        w.line("float4 xv3 = float4(x4[k4 + 3u]);");
        for r in 0..dense_rows {
            let acc_name = if r == 0 {
                "acc".to_string()
            } else {
                format!("acc{r}")
            };
            w.writeln(format_args!(
                "{acc_name} += dot(xv0, float4(W4_{r}[k4])) + dot(xv1, float4(W4_{r}[k4 + 1u])) + dot(xv2, float4(W4_{r}[k4 + 2u])) + dot(xv3, float4(W4_{r}[k4 + 3u]));"
            ));
        }
        w.pop_indent();
        w.line("}");
    } else if m % 32 == 0
        && matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && m >= 2048
        && n >= 2048
    {
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(x + row * M);"
        ));
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W + i * M);"
        ));
        w.line("for (uint k4 = simd_lane * 8u; k4 < M / 4u; k4 += 256u) {");
        w.push_indent();
        w.line("acc += dot(float4(x4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(x4[k4+1u]), float4(W4[k4+1u]))");
        w.line("     + dot(float4(x4[k4+2u]), float4(W4[k4+2u]))");
        w.line("     + dot(float4(x4[k4+3u]), float4(W4[k4+3u]))");
        w.line("     + dot(float4(x4[k4+4u]), float4(W4[k4+4u]))");
        w.line("     + dot(float4(x4[k4+5u]), float4(W4[k4+5u]))");
        w.line("     + dot(float4(x4[k4+6u]), float4(W4[k4+6u]))");
        w.line("     + dot(float4(x4[k4+7u]), float4(W4[k4+7u]));");
        w.pop_indent();
        w.line("}");
    } else if m % 16 == 0
        && matches!(crate::target::current_chip(), crate::target::ChipModel::M5)
        && m >= 2048
        && n >= 2048
    {
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(x + row * M);"
        ));
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W + i * M);"
        ));
        w.line("for (uint k4 = simd_lane * 4u; k4 < M / 4u; k4 += 128u) {");
        w.push_indent();
        w.line("acc += dot(float4(x4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(x4[k4+1u]), float4(W4[k4+1u]))");
        w.line("     + dot(float4(x4[k4+2u]), float4(W4[k4+2u]))");
        w.line("     + dot(float4(x4[k4+3u]), float4(W4[k4+3u]));");
        w.pop_indent();
        w.line("}");
    } else if m % 8 == 0 {
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(x + row * M);"
        ));
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W + i * M);"
        ));
        w.line("for (uint k4 = simd_lane * 2u; k4 < M / 4u; k4 += 64u) {");
        w.push_indent();
        w.line("acc += dot(float4(x4[k4]),     float4(W4[k4]))");
        w.line("     + dot(float4(x4[k4+1u]), float4(W4[k4+1u]));");
        w.pop_indent();
        w.line("}");
    } else if m % 4 == 0 {
        w.line(format!(
            "device const {t}4* x4 = (device const {t}4*)(x + row * M);"
        ));
        w.line(format!(
            "device const {t}4* W4 = (device const {t}4*)(W + i * M);"
        ));
        w.line("for (uint k4 = simd_lane; k4 < M / 4u; k4 += 32u) {");
        w.push_indent();
        w.line("acc += dot(float4(x4[k4]), float4(W4[k4]));");
        w.pop_indent();
        w.line("}");
    } else {
        w.line("for (uint k = simd_lane; k < M; k += 32u) {");
        w.push_indent();
        w.line("acc += float(x[row * M + k]) * float(W[i * M + k]);");
        w.pop_indent();
        w.line("}");
    }
    w.line("acc = simd_sum(acc);");
    w.line("uint residual_i = RESIDUAL_VECTOR ? i : (row * N + i);");
    if dense_rows > 1 {
        for r in 1..dense_rows {
            w.writeln(format_args!("acc{r} = simd_sum(acc{r});"));
        }
        w.line("if (simd_lane == 0u) {");
        w.push_indent();
        w.line(format!(
            "y[row * N + i] = {t}(acc + float(residual[residual_i]));"
        ));
        for r in 1..dense_rows {
            w.writeln(format_args!("if (i + {r}u < N) {{"));
            w.push_indent();
            w.writeln(format_args!(
                "uint residual_i{r} = RESIDUAL_VECTOR ? (i + {r}u) : (row * N + i + {r}u);"
            ));
            w.writeln(format_args!(
                "y[row * N + i + {r}u] = {t}(acc{r} + float(residual[residual_i{r}]));"
            ));
            w.pop_indent();
            w.line("}");
        }
        w.pop_indent();
        w.line("}");
    } else {
        w.line(format!(
            "if (simd_lane == 0u) y[row * N + i] = {t}(acc + float(residual[residual_i]));"
        ));
    }

    w.pop_indent();
    w.line("}");
    w.blank();
    Ok(())
}

pub fn function_name(fusion: &Fusion) -> String {
    format!("fusekit_linear_residual_f{}", fusion.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::msl::MslWriter;
    use crate::fusion::ir::{FusionBoundary, ModelMeta, QuantInfo};
    use crate::graph::Dtype;
    use crate::task::ir::PartitionPlan;

    fn bnd(role: &'static str, shape: Vec<i32>, dtype: Dtype) -> FusionBoundary {
        FusionBoundary {
            source: 0,
            role,
            shape,
            dtype,
        }
    }

    fn test_meta(n: u32) -> ModelMeta {
        ModelMeta {
            model_type: "test".into(),
            hidden_size: n,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
            intermediate_size: 1,
            vocab_size: 1,
            activation_dtype: Dtype::Bf16,
            quant_group_size: Some(64),
            quant_bits: Some(4),
            mtp_num_hidden_layers: 0,
            num_nextn_predict_layers: 0,
            mtp: crate::fusion::ir::MtpInfo::none(),
        }
    }

    fn partition() -> PartitionPlan {
        PartitionPlan {
            grid: (1, 1, 1),
            bindings: vec![],
        }
    }

    // Dumps the int4 LinearResidual MSL to /tmp so we can pipe it
    // through `xcrun metal` for a syntax check.
    //   cargo test --release dump_int4_msl -- --nocapture
    #[test]
    fn dump_int4_msl() {
        let (m, n, group, lanes) = (896i32, 896i32, 64i32, 8i32);
        let fusion = Fusion {
            id: 0,
            kind: FusionKind::LinearResidual,
            op_ids: vec![],
            inputs: vec![
                bnd("x", vec![1, 1, m], Dtype::Bf16),
                bnd("W", vec![n, m / lanes], Dtype::U32),
                bnd("W_scales", vec![n, m / group], Dtype::F16),
                bnd("W_biases", vec![n, m / group], Dtype::F16),
                bnd("residual", vec![1, 1, n], Dtype::Bf16),
            ],
            outputs: vec![bnd("y", vec![1, 1, n], Dtype::Bf16)],
            attrs: FusionAttrs::LinearResidual {
                quant: Some(QuantInfo {
                    bits: 4,
                    group_size: 64,
                }),
            },
            parallel_axes: vec![],
            layer_hint: None,
        };
        let mut w = MslWriter::new();
        w.line("#include <metal_stdlib>");
        w.line("using namespace metal;");
        w.blank();
        emit(&fusion, &partition(), &test_meta(n as u32), &mut w).unwrap();
        let src = w.finish();
        let path = "/tmp/fusekit_int4_linres.metal";
        std::fs::write(path, &src).unwrap();
        eprintln!("[dump_int4_msl] wrote {} ({} bytes)", path, src.len());
    }
}
