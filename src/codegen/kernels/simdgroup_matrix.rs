//! simdgroup_matrix MMA helpers for dense bf16/half GEMV (M3+).
//!
//! Apple M3+ has an 8×8 simdgroup_matrix intrinsic that performs a
//! tiled matmul in a single hardware instruction. For dense bf16/half
//! matmul this beats the scalar `dot()` GEMV path on prefill where A
//! has L≥8 rows. For decode L=1 it fills 1 of 8 A-rows = 1/8
//! utilisation, marginal — main payoff is post-#35 prefill.
//!
//! Usage: callers in `emit_chain_dense_matmul` (and future prefill/MTP
//! GEMM emit) branch through `ChipModel::use_simdgroup_gemm(active_rows)`.
//! Default mode is `decode-gemv`; `FUSEKIT_DENSE_MATMUL_MODE=auto` enables
//! GEMM only when active rows fill the 8x8 tile, and
//! `FUSEKIT_DENSE_MATMUL_MODE=simdgroup-gemm` force-enables experiments.
//!
//! Layout assumptions (matches mlx Linear `x @ W.T` after the trace
//! flip):
//!   A: (M, K) row-major; stride between rows = K.
//!   B: (K, N) row-major; stride between rows = N. (Conceptually W^T.)
//!   C: (M, N) row-major; stride = N.
//!
//! Tile shape: 8×8 in (m, k) for A, (k, n) for B, (m, n) for C. M is
//! padded to multiple of 8 by the caller (decode L=1 leaves rows 1..7
//! holding zero contributions, output rows 1..7 ignored).

use crate::codegen::msl::MslWriter;

/// Emit the simdgroup_matrix-based inner loop. The simdgroup writes
/// into the `acc` matrix variable (caller declares `simdgroup_matrix<
/// float, 8, 8> acc(0);` before calling), reads A and B tiles via
/// `simdgroup_load`, fuses via `simdgroup_multiply_accumulate`, leaves
/// the final store to the caller.
///
/// Parameters:
/// - `acc`: variable name of the simdgroup_matrix<float, 8, 8> accumulator
/// - `a_ptr_expr`: pointer expression to start of A row block (e.g.
///   `A + m_base * K`), stride = K (column-major between K columns of A).
/// - `b_ptr_expr`: pointer expression to start of B (k=0, n=n_base).
///   stride = N.
/// - `k_const`: constexpr-like uint expression for K.
/// - `n_stride_expr`: uint expression for the B row stride (= N).
/// - `t`: activation dtype string (`"bfloat"` or `"half"`).
pub fn emit_simdgroup_matmul_inner(
    w: &mut MslWriter,
    acc: &str,
    a_ptr_expr: &str,
    b_ptr_expr: &str,
    k_const: &str,
    a_stride_expr: &str,
    n_stride_expr: &str,
    t: &str,
) {
    w.line("{");
    w.push_indent();
    // K loop. Each iter loads an 8×8 A-tile and 8×8 B-tile and
    // multiply-accumulates into the C accumulator.
    w.writeln(format_args!("for (uint k = 0u; k < {k_const}; k += 8u) {{"));
    w.push_indent();
    w.writeln(format_args!("metal::simdgroup_matrix<{t}, 8, 8> a_tile;"));
    w.writeln(format_args!("metal::simdgroup_matrix<{t}, 8, 8> b_tile;"));
    // A tile: row stride is K-axis (M rows × 8 cols starting at column k).
    // For decode L=1 only the first row carries data; other rows zero-init.
    w.writeln(format_args!(
        "metal::simdgroup_load(a_tile, {a_ptr_expr} + k, {a_stride_expr});"
    ));
    // B tile: row stride is N. Start at (k, n_base).
    w.writeln(format_args!(
        "metal::simdgroup_load(b_tile, {b_ptr_expr} + k * {n_stride_expr}, {n_stride_expr});"
    ));
    w.writeln(format_args!(
        "simdgroup_multiply_accumulate({acc}, a_tile, b_tile, {acc});"
    ));
    w.pop_indent();
    w.line("}");
    w.pop_indent();
    w.line("}");
}

/// Caller-side preamble that declares the accumulator and stores its
/// value to memory at the end. Splitting these out keeps
/// `emit_simdgroup_matmul_inner` focused on the K-axis loop.
pub fn emit_simdgroup_acc_decl(w: &mut MslWriter, acc: &str) {
    w.writeln(format_args!(
        "metal::simdgroup_matrix<float, 8, 8> {acc}(0.0f);"
    ));
}

pub fn emit_simdgroup_acc_store(
    w: &mut MslWriter,
    acc: &str,
    out_ptr_expr: &str,
    n_stride_expr: &str,
) {
    w.writeln(format_args!(
        "metal::simdgroup_store({acc}, {out_ptr_expr}, {n_stride_expr});"
    ));
}
