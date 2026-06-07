# Next session work plan — 7 deferred items

All 7 remaining tasks documented with implementation steps. Priority order
based on ROI vs effort. Tackle ONE item per session — each requires
multi-file coordinated changes.

## Priority 1: #30 Argmax + LM head fuse (2-3h)

**Files to touch:**
- `src/codegen/kernels/fused_chain.rs::emit_reduction_then_heavy_chain` —
  add per-TG (max, idx) tracking after the per-SG simd_sum. SMEM reduce
  inter-SG → lane-0-of-SG-0 writes (max, idx) to a NEW output buffer at
  index `gid`.
- New `src/codegen/kernels/argmax_finalize.rs` — kernel reading per-TG
  (max, idx) buffer (N_TGS × 8 bytes ~ 38KB for vocab=152K), single-TG
  reduce, writes next_token.
- `src/fusion/ir.rs` — add `FusionKind::ArgmaxFinalize` variant.
- `src/fusion/build.rs` — when `FUSEKIT_FUSE_ARGMAX=1`, replace synthetic
  `ArgmaxLogitsDecode` injection with `ArgmaxFinalize`. Also: the LM
  head chain (RMSNorm+Matmul where OUT_DIM == vocab_size) needs an
  extra output declared for the per-TG scratch.
- `src/codegen/task_graph_json.rs` — handle the new FusionKind in
  order_key + scratch allocation.
- `python/fusekit_runtime/dispatch.py` — bind the per-TG scratch to
  the LM head's extra output and to the finalize kernel.

**Detection:** `OUT_DIM == meta.vocab_size` in
`emit_reduction_then_heavy_chain`. The LM head is the only chain that
matches.

**Test:** correctness on Qwen2.5-0.5B-4bit (same first token sampled).
Benchmark expected gain: ~2.5% (saves 1 dispatch + 304KB DRAM
round-trip on argmax read).

## Priority 2: #29 SwiGLU absorption (3-5h)

**Files to touch:**
- `src/fusion/general.rs` — new `try_fanout_consumer_absorb` function:
  detect FusedChain producer (Reduction-fanout) whose all N outputs
  (same shape) flow to a single Pointwise FusedChain consumer with
  no other consumers. Merge into one chain with steps
  `[Reduction, Heavy_1, ..., Heavy_N, tail_step_1, ..., tail_step_M]`.
  Rewrite consumer's External refs to producer outputs → Register
  refs.
- `src/codegen/kernels/fused_chain.rs` — new
  `emit_reduction_aligned_fanout_chain` (or extend existing) where
  each SG owns ONE col, computes ALL N heavies at that col into
  separate accs, then lane-0 applies the pointwise tail using those
  registers and writes ONE output.
- `src/codegen/kernels/mod.rs::kernel_grid_threads_impl` — new grid
  shape: single out_dim (not OUT_TOTAL), each SG owns one col +
  multirow.

**Key difference from existing emit_reduction_fanout_chain:** current
emit has each SG own (col, heavy_step) pair → 2x grid for gate/up.
Aligned version: each SG owns col only → 1x grid + register-resident
heavy outputs.

**Test:** correctness + bench. 24 fewer dispatches/token on Qwen2.5
(per-layer SwiGLU chain absorbed).

## Priority 3: #27 Tile-major weight layout (3-5d)

**Files to touch:**
- `src/commands/patch.rs` — install-time weight conversion: read each
  weight tensor (q4/q8/bf16/f16), tile-pack into
  `(out_dim/T, in_dim/T, T, T)` row-major-within-tile layout. Cache
  next to original or replace.
- `src/codegen/kernels/mod.rs::emit_int4_gemv_inner` — tile-coord
  indexing instead of `out_idx * in_dim + k`.
- `src/codegen/kernels/fused_chain.rs::emit_chain_dense_matmul` —
  same tile-coord indexing.
- `src/target/mod.rs` — `ChipModel::weight_tile_size()` (M5: 32,
  M3/M4: 16, M1/M2: 8).
- `FUSEKIT_TILE_MAJOR=1` opt-in until validated.

**Test:** correctness + bench. Estimated 5-10% on GEMV-bound (most
of decode).

## Priority 4: #28 simdgroup_matrix (M3+) (5-7d)

**Files to touch:**
- `src/codegen/kernels/fused_chain.rs::emit_chain_dense_matmul` —
  branch on `chip.supports_simdgroup_matrix()`. When true and dtype
  is bf16/f16, emit `simdgroup_bfloat8x8` / `simdgroup_half8x8`
  primitives via `simdgroup_load` → `simdgroup_multiply_accumulate` →
  `simdgroup_store`.
- Output padding: ensure OUT_DIM is multiple of 8; pad if needed.
- Accumulator becomes `simdgroup_matrix<float, 8, 8>` instead of
  scalar `float`.

**Test:** correctness on M3/M4/M5 with bf16 dense models. Decode L=1
uses 1/8 of the 8×8 tile — gain is mostly on PREFILL. For pure decode,
may underperform current 4-way vec dot.

## Priority 5: #31 SDPA Stage 1 cooperative GEMV (2-3d)

**Files to touch:**
- `src/codegen/kernels/sdpa_decode.rs` — emit two Stage 1 variants:
  thread-per-kv (current) and simdgroup-cooperative (32 lanes share
  one kv, stride head_dim, simd_sum). Dispatch via runtime cache
  offset (proxy for kv_len).
- Cost model decides crossover threshold.

**Test:** bench at varying kv_len. Expected: flat on 0.5B short-ctx,
+10-20% on long-ctx (kv_len > 4K).

## Priority 6: #12 mlp_dim tiling for silu_up (3-5d, niche)

**Pre-req:** Re-enable `RmsNormGateUpSwigluDown` pattern (currently
commented out at `src/fusion/patterns/mod.rs:212`) — its fusion was
disabled due to single-TG dispatch overhead vs multi-TG split path.

If re-enabling: add SMEM-budget check in
`safety::estimate_smem_bytes::RmsNormGateUpSwigluDown` to fail-safe.
Modify `rmsnorm_gateup_swiglu_down.rs` to tile silu_up: keep T
elements in SMEM, accumulate down_proj output over T-element chunks.

Niche — only affects very-large models (Qwen3.6-27B with
intermediate=25600).

## Priority 7: #7 fused_chain piece migration (multi-week)

**Foundational** for codegen → IR migration. Each chain step becomes
a `piece` in `src/codegen/pieces/` with explicit metadata (input
shapes, SMEM needs, register cost). Composer assembles pieces into
final MSL.

Multi-week effort. Defer until other priorities ship.

## After-each-task checklist

1. `cargo build --release` — clean
2. `cargo test --release --bin fusekit` — all tests pass
3. Patch + bench on Qwen2.5-Coder-0.5B-Instruct-4bit (200 tok)
4. Compare fusekit vs raw mlx-lm at same thermal
5. Update task description + memory entry
6. Mark task `completed`
