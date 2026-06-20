# General Fusion Optimization Log

This file records measured FuseKit general/semantic fusion work so we do not
repeat rejected experiments. Pattern-matcher-only runs are not the default
baseline here. Unless noted otherwise, "fusekit" means `--patterns off`
general fusion, all tasks emitted as custom kernels, and no native fallback.

## Measurement Policy

- Run raw/oMLX baseline first, then FuseKit, to avoid thermal bias favoring
  FuseKit.
- Use oMLX Python:
  `/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11`
- Use this `PYTHONPATH` for direct script calls:
  `/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages`
- Prefer local model paths under `~/.omlx/models/...`; repo ids can
  block at HuggingFace snapshot checking.
- Keep an optimization only if correctness holds and profile/e2e supports it.
- For L512 regressions, inspect late-token profile with
  `FUSEKIT_PROFILE_KERNELS_AT=511 FUSEKIT_PROFILE_COMPACT=1`.
- General fusion policy: propose candidates from dataflow/shape/state-role
  invariants, but admit them by budget. A fusion candidate must fit register
  pressure, threadgroup memory, recompute cost, and parallelism-loss bounds;
  model names or exact layer templates are not valid admission criteria.

## Current Known Good State

Date: 2026-06-19.

Latest restored 4B q4/q8/bf16 artifacts use `--patterns off` general fusion.
Patch coverage for q4/q8 is:

| Model | Custom tasks | Native fallback | Fused/specialized | Source ops fused |
| --- | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3.5-4B-4bit` | 553/553 | 0/553 | 340/553 | 938/1247 |
| `mlx-community/Qwen3.5-4B-8bit` | 553/553 | 0/553 | 340/553 | 938/1247 |
| `mlx-community/Qwen3.5-4B-bf16` | 700/700 | 0/700 | 340/700 | 1131/1587 |

Kept from the 2026-06-18 session:

| Area | Change | Evidence |
| --- | --- | --- |
| q8 dual qMV/tg2 helper | Split `FUSEKIT_QMV_FAST_TG2_PACKS_Q*` from generic qMV packs and default q8 tg2 to packs=4 on M5 | q8 token-511 profile improved from total 39.38 ms to 39.12 ms; swiglu 13.42 ms -> 13.27 ms; reduction-fanout 6.79 ms -> 6.65 ms; correctness passed |
| q4 pair qMV direct x-load | In the q4 two-output SwiGLU helper, accumulate directly from threadgroup `x` instead of storing then rereading `x_thread[]` | q4 token-511 profile total 22.28 ms -> 21.45 ms; q4 L512 raw 46.365 TPS vs FuseKit 47.225 TPS (+1.9%); q8 L512 stayed positive at +2.2%; correctness passed |

Latest same-run L512 benchmark after this kept change:

File: `/private/tmp/fusekit-4b-l512-q4q8bf16-tg2.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Peak delta |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-4bit` | 45.533 | 46.369 | +1.84% | +1.37% |
| `Qwen3.5-4B-8bit` | 25.603 | 26.035 | +1.69% | +0.74% |
| `Qwen3.5-4B-bf16` | 14.054 | 15.053 | +7.11% | +0.51% |

The q4/q8 deltas are smaller than older runs because the raw baseline was much
faster in this run. Use same-run delta plus late-token profile, not historical
peak percentages, when judging a new candidate.

## 2026-06-21 DeepSeek MoE Reference Analysis

Scope: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`, L512 decode,
general fusion only (`--patterns off`).

Current default DeepSeek artifact:

| Item | Value |
| --- | --- |
| Custom tasks | 787/787 |
| Native fallback | 0/787 |
| Fused/specialized tasks | 245/787 |
| Source ops fused | 935/1660 |
| L512 same-run result | raw 73.449 TPS vs FuseKit 73.223 TPS (-0.3%) |
| Late-token total GPU | ~15.0-15.2 ms |

Late-token buckets at token 511:

| Bucket | Count | GPU time |
| --- | ---: | ---: |
| `RoutedQmvSwiglu` | 26 | ~3.84-3.95 ms |
| `RoutedQmvReduce+LinearResidual` | 26 | ~2.84-2.85 ms |
| `SdpaDecode` | 27 | ~1.93-1.95 ms |
| `QmvSwiglu` | 26 | ~1.47 ms |
| `FusedChain reduction-fanout` | 27 | ~1.05 ms |
| `FinalRmsNormLmHead` | 1 | ~0.93-0.98 ms |

Reference source pass:

- DS4 `metal/moe.metal` has q4_K `pair_swiglu` kernels which compute routed
  gate/up and write weighted SwiGLU mid in the same kernel. It also has
  `sum6` kernels that loop over the six selected expert slots and accumulate
  down projection directly into the output.
- DS4 also has selected-slot/address-table paths. Those are not just kernel
  math: the runtime reads the selected expert IDs from GPU to host in some
  modes, wraps or caches six expert buffers, then launches slot-specific
  kernels. This can win for very large DS4 tensors/SSD streaming, but it is not
  directly compatible with FuseKit's no-host-gap decode goal without a
  GPU-resident address-table/cache design.
- MLX's standard quantized qMV keeps a simple 2-simdgroup, 4-output-row helper
  and preserves qdot accumulation order. FuseKit q4/q8 helpers intentionally
  preserve the MLX arithmetic order where prior drift tests showed near-tie
  token flips.

Mapping to FuseKit:

| DS4 idea | FuseKit status | Decision |
| --- | --- | --- |
| Pair gate/up + SwiGLU | Already covered by `RoutedQmvSwiglu` from `GatherQMM` + pointwise semantic tail | Optimize kernel body only; do not add a model-name pattern |
| Down `sum6` | Partially covered by `RoutedQmvReduce+LinearResidual`; it runs six route simdgroups plus shared expert in one TG and sums TG partials | Candidate only if profile shows partial/barrier overhead dominates; previous route-fold attempt was not a win |
| Selected-slot expert views | Not implemented | Do not copy DS4 host-readback path into decode; consider only a GPU-resident address-table or persistent selected-view cache |
| q4_K exact block math | Not directly applicable: DS4 q4_K layout differs from MLX affine q4/group_size=64 layout | Use as microkernel scheduling reference, not as drop-in code |
| Small-M `r1=2..5` variants | Partially covered for MTP/small-M, not relevant to single-token DeepSeek L512 decode | Keep separate from single decode work |

Near-term candidates that are general:

1. Add a semantic `RoutedQmvReduce` `top_k`-fold variant only for the
   no-residual path first, and block it automatically for
   `residual_linear.is_some()` until the residual correctness bug is fixed.
   This removes a known correctness footgun and narrows future experiments.
2. For `RoutedQmvSwiglu`, compare the emitted MLX-affine q4 pair dot body
   against DS4's row2 q4_K structure and implement only shape/quant-metadata
   gated changes that reduce register live range or qparam loads without
   changing qdot order.
3. Investigate GPU-resident selected expert address tables as a separate
   runtime feature. It is general over `GatherQMM` with small `top_k`, but it
   needs a cost model: host readback is rejected for default decode.

Routed fast-qMV lowering check:

- Candidate: recompile the same semantic `RoutedQmvSwiglu` fusion through the
  existing MLX-style `fusekit_qmv_swiglu_pair_m_impl` helper. This is not a
  model-name pattern: admission is still the routed gate/up/SwiGLU fusion with
  q4 `group_size=64`; only the lowering body/grid changes from one output row
  per simdgroup to `sgs=2, cols/sg=4`.
- Tested through an alternate artifact under the model directory
  (`fusekit-routed-fast`) and a `/private/tmp` symlink model root, so the
  installed default `fusekit` hook was not overwritten.
- Correctness passed with explicit `--fusekit-dir`: raw argmax = patched argmax
  = 28427, dispatcher built, forced dispatch, max abs diff `0.04296875`, mean
  abs diff `0.006352317`.

| Variant | Token-511 total GPU | `RoutedQmvSwiglu` bucket | L512 TPS vs own raw |
| --- | ---: | ---: | ---: |
| default artifact | noisy `16.46-17.85 ms` | `4.23-4.28 ms` | `70.046 -> 71.011` (+1.4%) |
| routed fast-qMV candidate | `15.67 ms` single profile, but target bucket larger | `4.36-4.37 ms` | `71.097 -> 69.321` (-2.5%) |

Decision: reject. The candidate did not reduce the target routed gate/up
bucket and regressed L512 wall throughput. Keep the existing `rows/sg=1`
routed lowering. The remaining useful DeepSeek work is still a new MLX-affine
q4 routed qMV body or a structural reduction of the route/down round-trip, not
this helper swap.

Rejected or unsafe for default:

- Forcing flash SDPA on DeepSeek MLA: correctness passed, but token-511 SDPA
  grew from ~1.94 ms to ~4.2 ms and L512 dropped to ~62.5 TPS.
- Route fold on `RoutedQmvReduce+LinearResidual`: previous env test had a
  correctness failure path and no measured e2e win.
- DS4 selected-slot host readback as-is: it adds host synchronization and
  buffer wrapping work that can erase the ICB/runtime gap advantage.

### 2026-06-20 q8 Pair Helper Byte-Accum Hoist Kept

Scope: Qwen3.5-4B q8, L512, general fusion only (`--patterns off`).

- Compared against MLX's installed quantized qMV source. MLX q8 qMV keeps the
  byte-wise `accum += x[i] * w[i]` qdot order. The rejected FuseKit q8 attempts
  changed the body to direct/u32 forms and regressed. This kept change does not
  alter the byte accumulation order.
- In `fusekit_qmv_fast_tg2_partial_impl`, the q8 pair helper now computes the
  two byte-wise accumulators first, then loads/applies the pair's scale/bias.
  This shortens qparam live range in the two-output SwiGLU helper without
  changing q8 layout, model names, or pattern matching.

Correctness:

| Model | same_argmax | max_abs_diff | mean_abs_diff |
| --- | ---: | ---: | ---: |
| `Qwen3.5-4B-8bit` | true | 0.133789 | 0.023144 |

Late-token profile at token 511, before/after this candidate in the same
session:

| q8 bucket | Before | After |
| --- | ---: | ---: |
| Total GPU | 40.56 ms | 38.57 ms |
| SwiGLU qMV | 13.56 ms | 12.85 ms |
| Reduction fanout qMV | 7.36 ms | 6.55 ms |
| MLP down qMV | 6.75 ms | 6.98 ms |
| LM-head qMV | 5.53 ms | 5.08 ms |

Raw-first L512 repeat=3:

| Model | Raw TPS | FuseKit TPS | Delta | Peak delta |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-8bit` | 25.729 | 26.397 | +2.6% | +0.74% |

Decision: keep. The improvement is bits/layout-level for q8 pair qMV helpers
and preserves the MLX qdot arithmetic order.

### 2026-06-20 q4 Single-Output Direct X-Load Rejected

Scope: Qwen3.5-4B q4, L512, general fusion only (`--patterns off`).

- Trial: apply the already-kept q4 two-output direct `x` load idea to
  `fusekit_qmv_fast_tg_partial_impl`, used by single-output helper paths such
  as LM-head.
- Correctness passed (`same_argmax=true`, max_abs_diff 0.171875), but profile
  and wall did not beat the known-good artifact.

| q4 bucket/result | Recent known-good | Trial |
| --- | ---: | ---: |
| Total GPU token 511 | 22.29 ms | 22.37 ms |
| LM-head qMV | 2.91 ms | 2.80 ms |
| Reduction fanout qMV | 3.56 ms | 3.80 ms |
| L512 raw TPS | 45.533 | 45.902 |
| L512 FuseKit TPS | 46.369 | 46.563 |
| Delta | +1.84% | +1.4% |

Decision: reject and remove from source. The local LM-head reduction did not
translate into a stronger e2e result and increased pressure elsewhere.

### 2026-06-19 q8 Dual qGEMV Adjacent Weight-Load Restored

Scope: Qwen3.5-4B q4/q8/bf16, L512, general fusion only (`--patterns off`).

Change:

- Restored the q8 dual gate/up qMV adjacent two-word schedule in
  `emit_dual_qgemv_inner_with_unroll2_default`.
- The rule is model-independent: it applies when `bits == 8`, the large qGEMV
  unroll policy is active, and the quant group is aligned so two adjacent words
  share one scale/bias pair. q4 stays on the strided schedule because prior q4
  A/B regressed with the adjacent live state.
- This fixes a source/log mismatch: the log recorded q8 adjacent dual qGEMV as
  kept, but the current helper default had drifted back to `false`.

Correctness:

| Model | same_argmax | max_abs_diff | mean_abs_diff |
| --- | --- | ---: | ---: |
| `Qwen3.5-4B-4bit` | true | 0.171875 | 0.031358 |
| `Qwen3.5-4B-8bit` | true | 0.125000 | 0.021675 |
| `Qwen3.5-4B-bf16` | true | 0.121094 | 0.021267 |

q8 late-token profile before/after this source fix in the same session:

| q8 token-511 bucket | Before | After |
| --- | ---: | ---: |
| Total GPU | 42.42 ms | 38.82 ms |
| `FusedChain swiglu-absorbed` | 14.19 ms | 12.94 ms |
| `FusedChain reduction-fanout` | 7.47 ms | 6.78 ms |
| `LinearResidual M=9216` | 7.18 ms | 6.96 ms |
| `FinalRmsNormLmHead` | 5.52 ms | 5.19 ms |

Same-run L512 raw-first benchmark, repeat=3:

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-4bit` | 40.601 | 43.380 | +6.8% | 0.527 | 0.365 | 2.487 GB | 2.521 GB |
| `Qwen3.5-4B-8bit` | 23.594 | 25.781 | +9.3% | 1.313 | 0.053 | 4.586 GB | 4.620 GB |
| `Qwen3.5-4B-bf16` | 14.059 | 15.174 | +7.9% | 0.105 | 0.065 | 8.503 GB | 8.546 GB |

Decision:

- Keep. This is a q8 specialized weight/qparam load scheduling rule derived
  from quant metadata and shape alignment, not a model-name or pattern rule.
- It also restores the expected q8 L512 win without reducing q4/bf16.

### 2026-06-19 q4/q8 qMV Direct X-Load Kept

Scope: Qwen3.5-4B q4/q8, L512, general fusion only (`--patterns off`).

Change:

- In `fusekit_qmv_fast_tg2_partial_impl`, the q4 explicit-unpack
  two-output qMV path now loads the normalized threadgroup `x` values and
  immediately accumulates gate/up qMV results. The previous path materialized
  the same values into `x_thread[]` and then reread them for the two rows.
- The rule is gated only by quant metadata (`bits == 4` and the existing q4
  explicit-unpack policy), not model name or pattern matching. q8 keeps its
  existing qdot path.

Correctness:

| Model | same_argmax | max_abs_diff | mean_abs_diff |
| --- | --- | ---: | ---: |
| `Qwen3.5-4B-4bit` | true | 0.171875 | 0.031358 |
| `Qwen3.5-4B-8bit` | true | 0.125000 | 0.021675 |

Late-token profile:

| Model | Total GPU | SwiGLU bucket | Reduction-fanout bucket | LM-head bucket |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-4bit` | 21.45 ms | 6.83 ms | 3.53 ms | 2.84 ms |
| `Qwen3.5-4B-8bit` | 38.03 ms | 12.83 ms | 6.57 ms | 5.11 ms |

Same-run L512 benchmark:

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-4bit` | 46.365 | 47.225 | +1.9% | 0.251 | 0.066 |
| `Qwen3.5-4B-8bit` | 25.943 | 26.509 | +2.2% | 0.008 | 0.027 |

Decision:

- Keep. This is the first q4 L512 microkernel change in this pass that
  reduces the target kernel bucket and survives raw-first e2e benchmarking.

### 2026-06-19 q4 LinearResidual Direct X-Load Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Apply the same direct `x` load idea to q4 `LinearResidual` when the qMV
  helper emits two output rows per simdgroup. This tested whether avoiding
  `x_thread[]` materialization also helps the MLP down-projection.
- This is shape/quant metadata based (`bits == 4`, result rows = 2), not
  model-name or pattern matching.

Result:

| Artifact | Correctness | Total GPU | Down M=9216 GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| accepted q4 direct-x | same_argmax=True | 21.45 ms | 3.59 ms | 47.225 |
| q4 LinearResidual direct-x | same_argmax=True | 21.76 ms | 3.60 ms | 46.055 |

Decision:

- Rejected and removed from source. Unlike the two-output SwiGLU helper, the
  residual helper already benefits from reusing `x_thread[]` across output
  rows; scalarizing the loads increased total GPU time and hurt wall TPS.

### 2026-06-19 q8 Pair qMV Direct X-Load Rejected

Scope: Qwen3.5-4B-8bit, L512, general fusion only (`--patterns off`).

Experiment:

- Replace the q8 two-output SwiGLU helper's qdot+x_thread path with a direct
  per-element byte-weight accumulation from threadgroup `x`. This was scoped
  only to the pair helper, unlike older global q8 qdot-off tests.

Result:

| Artifact | Correctness | Total GPU | SwiGLU GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| accepted q8 baseline | same_argmax=True | 38.03 ms | 12.83 ms | 26.509 |
| q8 pair direct-x | same_argmax=True | 38.52 ms | 12.78 ms | 25.968 |

Decision:

- Rejected and removed from source. q8's existing qdot+x_thread schedule keeps
  lower total GPU time even though the target SwiGLU bucket is close; the
  direct byte path increased surrounding qMV pressure and hurt wall TPS.

Rejected in the 2026-06-18 continuation:

| Experiment | Correctness | Profile / e2e result | Decision |
| --- | --- | --- | --- |
| q4 `LinearResidual` uint32 two-half packed load | pass | total 21.79 ms -> 22.64 ms; M=9216 down 3.61 ms -> 3.68 ms; wall 45.28 -> 44.25 TPS | reject/reverted |
| q8 global `FUSEKIT_QMV_FAST_PACKS_Q8=4` | pass | swiglu/fanout slightly down, but LinearResidual/LM-head regressed; wall no clear win | reject |
| q4 global `FUSEKIT_QMV_FAST_PACKS_Q4=4` | fail | output collapsed to repeated `!` tokens | reject |
| q8 `FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q8_COLS_PER_SG=1` | pass | profile looked lower, but same-run L512 e2e raw 24.118 TPS vs FuseKit 23.370 TPS (-3.10%) | reject/restored |
| q8 `FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q8=0` | pass | profile looked lower, but same-run L512 e2e raw 24.985 TPS vs FuseKit 23.563 TPS (-5.69%) | reject/restored |
| q8 LM-head argmax scan byte loop -> q8 uint32 helper for `token_rows==1` | pass | LM-head 5.19 ms -> 5.24 ms; total GPU no clear win | reject/reverted |
| q4 tg2 packs=1 | pass | total 21.79 ms -> 22.19 ms; swiglu 6.88 ms -> 7.14 ms; wall 45.28 -> 43.66 TPS | reject/restored |
| q4 `LinearResidual` down `FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q4_COLS_PER_SG=4` | pass | M=9216 down 3.61 ms -> 3.75 ms; total 21.79 ms -> 22.28 ms | reject/restored |
| q4 `FUSEKIT_QGEMV_UNROLL2=0` | pass | L512 profile looked slightly better and L32 passed, but code inspection showed q4 M5 already defaults to unroll2=false; default artifact A/B was effectively identical | no code change |
| q4 `FUSEKIT_LINEAR_RESIDUAL_TGX=1` | pass | total 21.79 ms -> 22.03 ms; LinearResidual 5.47 ms -> 6.12 ms; M=9216 down 3.60 ms -> 4.13 ms; wall 43.846 TPS | reject/restored |
| q8 `FUSEKIT_LINEAR_RESIDUAL_TGX=1` | pass | total 39.12 ms -> 39.55 ms; LinearResidual about 9.99 ms -> 10.57 ms; M=9216 down 7.21 ms | reject/restored |
| q4 `FUSEKIT_GEMV_ROWS=2` / `ROWS_PER_SG=2` | pass | total 21.20 ms -> 21.74 ms; swiglu 6.82 ms -> 7.17 ms; wall 45.79 -> 45.26 TPS | reject/restored |
| q8 `FUSEKIT_GEMV_ROWS=2` / `ROWS_PER_SG=2` | pass | total 37.91 ms -> 38.80 ms; swiglu 12.93 ms -> 13.37 ms; wall 25.99 -> 25.60 TPS | reject/restored |
| q4 `FUSEKIT_SWIGLU_CAP=576` | pass | total 21.20 ms -> 21.60 ms; swiglu 6.82 ms -> 6.93 ms; wall 45.79 -> 45.58 TPS | reject/restored |
| q4 `FUSEKIT_FINAL_LM_HEAD_TGS=16` | pass | LM-head 2.78 ms -> 3.25 ms; wall 45.79 -> 44.89 TPS | reject/restored |
| bf16 dense chain K8 unroll disabled | pass | total 66.26 ms -> 67.40 ms; swiglu 22.64 ms -> 23.41 ms; dense fanout 11.50 ms -> 12.06 ms; wall 14.70 -> 14.57 TPS | reject/restored |

Full MLP fusion was inspected with `FUSEKIT_SWIGLU_DOWN_ADMISSION_DEBUG=1`.
For Qwen3.5-4B q4, all 32 MLP candidates were rejected with
`reject_parallelism`: `silu_up` in dtype would fit in smem
(28.8KB under 32KB), but the current full-fusion emitter would collapse the
down projection from 640 parallel TGs to one TG. Recomputing gate/up per down
tile would add about 15GB of gate/up q-weight traffic per layer, so the current
full MLP emitter is intentionally not used for 4B quantized decode.

Older note follows for context.

Date: 2026-06-17.

4B q4/q8/bf16 general fusion artifacts were restored after rejected q8
experiments. Current q8 coverage after patch:

| Model | Custom tasks | Native fallback | Fused/specialized |
| --- | ---: | ---: | ---: |
| `mlx-community/Qwen3.5-4B-8bit` | 577/577 | 0/577 | 332/577 |

Kept changes from this phase:

| Area | Change | Reason |
| --- | --- | --- |
| `src/codegen/msl.rs` | q8 helper `fusekit_q8_dot_u32_load` for qMV fast paths | Same per-element accumulation order, fewer byte loads |
| `src/codegen/kernels/mod.rs` | `QdotPolicy` and qdot disable hook | Allows semantic-sensitive callsites to avoid qdot regrouping |
| `src/codegen/kernels/final_rmsnorm_lm_head.rs` | Disable qdot for direct LM-head argmax | qdot can flip near-tie argmax tokens |
| `src/fusion/general.rs` | Set current chip in general fusion | Budget/tile rules use the real target chip |
| `src/fusion/general.rs` | Detailed swiglu-down admission debug | Explains why full MLP fusion is rejected |
| `src/codegen/kernels/fused_chain.rs` | Fix dual qGEMV helper scope and use inline affine q8 swiglu body | q8 L512 e2e improved; preserves per-element affine form |
| `src/codegen/kernels/fused_chain.rs` | Adjacent-pair dual qGEMV default for q8 swiglu-absorbed chains | Shares qparam loads inside one quant group; q8 profile A/B improved, q4 default stays strided |

## Benchmark History

### Qwen3.5 4B, L32, repeat=3

File from prior run: `/private/tmp/fusekit-4b-l32-u32load-final-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Peak delta |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-4bit` | 34.985 | 41.746 | +19.32% | +0.20% |
| `Qwen3.5-4B-8bit` | 21.382 | 24.072 | +12.58% | +0.04% |
| `Qwen3.5-4B-bf16` | 12.423 | 13.698 | +10.26% | +0.24% |

### Qwen3.5 4B, L512, repeat=3

File from prior run: `/private/tmp/fusekit-4b-l512-u32load-final-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Peak delta | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `Qwen3.5-4B-4bit` | 39.803 | 43.013 | +8.07% | +0.48% | FuseKit sd 1.027 |
| `Qwen3.5-4B-8bit` | 22.583 | 23.174 | +2.62% | +0.30% | Older noisy run |
| `Qwen3.5-4B-bf16` | 13.284 | 14.322 | +7.82% | +0.52% |  |

### Qwen3.5 4B 8bit restored, L512, repeat=3

File: `/private/tmp/fusekit-q8-l512-restored-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Peak delta |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-8bit` | 22.060 | 22.956 | +4.06% | +0.74% |

### Qwen3.5 4B 8bit after rejected q8 experiments, L512, repeat=3

File: `/private/tmp/fusekit-q8-l512-after-rejects-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Peak delta |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B-8bit` | 24.153 | 24.627 | +1.97% | +0.74% |

### Qwen3.5 4B 8bit inline affine swiglu, L32/L512, repeat=3

Files:
`/private/tmp/fusekit-q8-l32-inline-affine-r3.jsonl`,
`/private/tmp/fusekit-q8-l512-inline-affine-r3.jsonl`.

| Tokens | Raw TPS | FuseKit TPS | Delta | Peak delta |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 22.115 | 25.144 | +13.69% | -0.04% |
| 512 | 23.486 | 26.032 | +10.84% | +0.74% |

### Qwen3.5 4B 8bit residual affine trial, L512, repeat=3

File: `/private/tmp/fusekit-q8-l512-linres-affine-r3.jsonl`.

This trial changed only q8 `LinearResidual` single-row qMV to accumulate
`x * (scale * q + bias)` per element. Correctness smoke passed, but the
profile did not show a meaningful down-projection win and e2e absolute FuseKit
TPS was essentially unchanged from the swiglu-only state. Raw TPS was much
higher in this run, so the measured delta collapsed.

| Raw TPS | FuseKit TPS | Delta | Peak delta | Decision |
| ---: | ---: | ---: | ---: | --- |
| 25.721 | 26.071 | +1.36% | +0.74% | reject/reverted |

### Qwen3.5 4B 8bit dual qGEMV adjacent-pair swiglu, L512

Files:
`/private/tmp/fusekit-q8-l512-dual-adjacent-r3.jsonl`.

The kept change makes the swiglu-absorbed dual qGEMV process two adjacent
packed words per lane for q8 when both words stay inside one quant group.
This shares gate/up scale+bias loads for the pair. It is a bits/group-size
rule, not a model-name rule. q4 was tested separately and left on the old
strided schedule by default.

| Variant | Total GPU @ token 511 | Swiglu GPU | Wall TPS | Decision |
| --- | ---: | ---: | ---: | --- |
| old strided pair (`FUSEKIT_QGEMV_ADJACENT_UNROLL2=0` temporary artifact) | 41.60 ms | 13.94 ms | 24.816 | reject |
| adjacent-pair default | 38.39 ms | 12.98 ms | 25.587 | keep |

Same-run raw-vs-FuseKit repeat=3 after enabling adjacent-pair had raw 25.702
TPS and FuseKit 26.061 TPS (+1.40%). The relative delta is low in that run
because raw was unusually high, but the controlled artifact A/B shows the
kernel change itself is materially better than the old dual qGEMV schedule.

q4 A/B showed the opposite, so q4 default does not use adjacent-pair dual
qGEMV:

| q4 Variant | Raw TPS | FuseKit TPS | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| adjacent-pair default trial | 40.076 | 42.522 | +6.10% | reject for q4 |
| old strided pair | 45.711 | 46.276 | +1.24% | keep for q4 default |

### Qwen3.5 4B q4/q8 revisit, L512

Files:
`/private/tmp/fusekit-q4-l512-after-q4-rejects-r3.jsonl`,
`/private/tmp/fusekit-q8-l512-current-qacc-r3b.jsonl`.

q4 experiments tried qparam broadcast and qdot body in the swiglu helper.
Both passed 16-token correctness but regressed the late-token profile, so the
q4 artifact was restored to the default helper path.

q8 swiglu-absorbed dual qGEMV now accumulates `qacc` and `xsum` per packed
word before applying scale/bias. This is a bits=8 quantized affine rule, not
a model-name rule. It passed 16-token and 64-token raw-vs-FuseKit text
checks. The profile win is small: swiglu 13.10 ms -> 13.00 ms while total
GPU stayed roughly flat, so this is a small kept microkernel cleanup rather
than a structural win.

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 39.865 | 43.286 | +8.58% | 0.686 | 0.376 | 2.487 GB | 2.521 GB |
| 8bit | 22.633 | 25.294 | +11.76% | 0.343 | 0.012 | 4.586 GB | 4.620 GB |

The q8 delta is inflated by a low raw run; use the absolute FuseKit TPS and
late-token profile for kernel decisions.

### MTP model smoke history

These were MTP-specific runs and should not be mixed with normal single-decode
q4/q8/bf16 numbers.

| Model | Path | TPS | Delta vs raw | Status |
| --- | --- | ---: | ---: | --- |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | raw-mtp | 35.453 | baseline | ok |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | fusekit-mtp | 60.761 | +71.4% | ok, model/format-specific smoke |
| `corysus/Qwen3.5-4B-oQ4-mtp` | raw-mtp | 57.393 | baseline | ok |
| `corysus/Qwen3.5-4B-oQ4-mtp` | fusekit-mtp | 57.092 | -0.5% | weak/reject |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | raw-mtp | 167.575 | baseline | ok |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | fusekit-mtp | 182.514 | +8.9% | ok but noisy sd |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | raw-mtp | 37.991 | baseline | ok |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | fusekit-mtp | 34.018 | -10.5% | reject/custom MTP path slower |

## Q8 Late-Token Profile

Current restored q8 L512 late-token profile at token 511:

| Kernel group | GPU time |
| --- | ---: |
| `FusedChain swiglu-absorbed` gate/up MLP | 12.87 ms |
| `LinearResidual` down projection | 6.55 ms |
| `FusedChain reduction-fanout` QKV | 6.53 ms |
| `FinalRmsNormLmHead` direct argmax scan | 5.26 ms |
| `RmsNormQkv` | 1.79 ms |
| `sdpa` | 0.37 ms |
| `gated_delta` | 0.49 ms |
| Total GPU | 38.39 ms |

Interpretation: q8 is not losing to fallback. It is qMV weight-streaming
dominated across swiglu, down, QKV fanout, and LM head. Raw MLX q8 is already
strong, so the remaining delta is smaller than q4/bf16.

## Rejected Experiments

Do not repeat these without a new hypothesis and a reason why the underlying
failure has changed.

| Experiment | Result | Decision |
| --- | --- | --- |
| `FUSEKIT_QGEMV_UNROLL2=0` on q8 | 16-token correctness failed | reject |
| `FUSEKIT_QGEMV_ADJACENT_UNROLL2=0` on q8 | 16-token correctness failed | reject |
| Force q8 `FUSEKIT_QGEMV_MLX_QDOT=1` | Near-tie/top-token drift; first decode token can flip | reject |
| q8 qparam broadcast | 16-token correctness failed | reject |
| QKV fanout q8 `FUSEKIT_QMV_FANOUT_Q8_FAST_SGS=2` | total 38.39 ms -> 38.47 ms; fanout 6.53 ms -> 6.59 ms | reject |
| QKV fanout q8 `FUSEKIT_QMV_FANOUT_Q8_COLS_PER_SG=2` | total 38.39 ms -> 41.80 ms; fanout 6.53 ms -> 7.15 ms | reject |
| QKV fanout q8 `FUSEKIT_QMV_FANOUT_Q8_COLS_PER_SG=8` | total 38.39 ms -> 42.52 ms; fanout 6.53 ms -> 7.38 ms | reject |
| LinearResidual q8 `FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q8_COLS_PER_SG=1` | total 38.39 ms -> 44.00 ms; down 6.55 ms -> 7.17 ms | reject |
| LinearResidual q8 `FUSEKIT_LINEAR_RESIDUAL_MLX_QMV_Q8_COLS_PER_SG=4` | total 38.39 ms -> 40.62 ms; down 6.55 ms -> 7.10 ms | reject |
| LM-head q8 `FUSEKIT_FINAL_LM_HEAD_TGS=24` | total 38.39 ms -> 43.12 ms; LM-head 5.26 ms -> 5.77 ms | reject |
| q8 single partial qMV byte loop -> `fusekit_q8_dot_u32_load` | Correctness ok, but total 38.39 ms -> 40.72 ms; LM-head 5.26 ms -> 6.57 ms | reject/reverted |
| q8 helper/x-load `STEEL_PRAGMA_UNROLL` | Correctness ok, but total 38.39 ms -> 40.64 ms; swiglu 12.87 ms -> 13.45 ms | reject/reverted |
| q8 swiglu-absorbed helper bypass before helper-scope fix | Metal compile failed: fallback body closed scope before `acc_g0/acc_u0/out_idx_base` use | reject; fixed separately before retry |
| q8 `LinearResidual` qMV per-element affine accumulation | Correctness ok; late-token down 6.62 ms and FuseKit L512 absolute TPS 26.071, essentially unchanged from swiglu-only 26.032 | reject/reverted |
| q8 explicit affine threshold `output_dim >= 1024` | Correctness ok, but late-token fanout 6.67 ms -> 6.69 ms and total 38.62 ms -> 38.64 ms | reject/reverted |
| q4 swiglu helper qparam broadcast (`FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q4=1`) | Correctness ok, but token-511 total 22.17 ms -> 22.88 ms; swiglu 6.87 ms -> 7.36 ms | reject/reverted artifact |
| q4 swiglu helper qdot body (`FUSEKIT_QMV_FAST_QDOT_Q4=1`) | Correctness ok, but token-511 total 22.17 ms -> 22.42 ms; swiglu/down did not improve | reject/reverted artifact |
| Full swiglu->down single-kernel fusion | Fits dtype SMEM in some cases but collapses down parallelism or adds huge gate/up recompute traffic | reject current form |
| q4/q8 weight repack POC | Peak memory risk and L512 regression; no stable win | reject/remove default |

## Structural Notes

- Current Qwen3.5-4B L512 repeat=3 raw-first benchmark on the general fusion
  path, with pattern matching disabled:

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 44.116 | 45.022 | +2.05% | 0.011 | 0.058 | 2.487 GB | 2.521 GB |
| 8bit | 24.802 | 25.231 | +1.73% | 0.093 | 0.025 | 4.586 GB | 4.620 GB |
| bf16 | 13.676 | 14.572 | +6.56% | 0.004 | 0.014 | 8.503 GB | 8.546 GB |

- Rerun after q4 rejects and q8 qacc/xsum change, same L512 repeat=3
  raw-first suite. This run did not reproduce the earlier q4/q8 win; q4/q8
  should be treated as effectively tied until the late-token profiles explain
  the variance.

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | 45.945 | 45.984 | +0.08% | 0.052 | 0.319 | 2.487 GB | 2.521 GB |
| 8bit | 23.883 | 24.006 | +0.51% | 1.350 | 0.653 | 4.586 GB | 4.620 GB |
| bf16 | 12.375 | 13.402 | +8.30% | 0.048 | 0.066 | 8.503 GB | 8.546 GB |

- Late-token profiling at token 512 confirms the q4/q8 tie is not host
  scheduling. Raw and FuseKit host p50 match the FuseKit GPU total:

| Model | Raw host p50 | FuseKit host p50 | FuseKit GPU total | Main FuseKit buckets at token 512 |
| --- | ---: | ---: | ---: | --- |
| 4bit | 21.97 ms | 22.02 ms | 22.47 ms | swiglu qMV 7.11 ms; MLP down 3.76 ms; QKV fanout 3.58 ms; LM-head 3.00 ms; SDPA 0.37 ms |
| 8bit | 39.83 ms | 39.64 ms | 39.71 ms | swiglu qMV 13.45 ms; QKV fanout 6.80 ms; MLP down 6.75 ms; LM-head 5.47 ms; SDPA 0.37 ms |
| bf16 | 73.05 ms | 68.78 ms | 68.89 ms | swiglu dense 23.51 ms; MLP down 12.01 ms; QKV fanout 11.62 ms; LM-head 10.32 ms; SDPA 0.37 ms |

- Raw MLX synchronized op profile points to the same structure: raw
  `quantized_matmul`/`QuantizedLinear` dominates q4/q8, with MLP
  gate/up/down, linear-attn QKV/out/z, and LM-head as the top signatures.
  Because this profiler synchronizes and nests module/op wrappers, use it
  for rank/shape attribution only, not absolute e2e time.
- Saved q8 Metal captures for manual inspection:
  `/private/tmp/fusekit-q8-raw-token512.gputrace` and
  `/private/tmp/fusekit-q8-fusekit-token512.gputrace`. CLI `xctrace export`
  does not parse these `.gputrace` bundles as normal `.trace` documents.

- q8 default fanout tile is currently `cols_per_sg=4`, `sgs=1`; profile
  confirmed this is better than 2/8 cols and better than 2 SGs.
- q8 LinearResidual default tile is currently `cols_per_sg=2`; profile
  confirmed this is better than 1 or 4.
- q8 LM-head default `n_tgs=32`; profile confirmed this is better than 24.
- `qmv_fast2_impl` is not the current q8 L512 QKV path when default fanout
  uses `cols_per_sg=4`; changing it will not address the measured q8 L512
  bottleneck unless the tiling path changes.
- MLP swiglu/down remains the largest structural opportunity, but naive
  full fusion is bad because it trades a modest activation round-trip for a
  much larger loss of parallelism or extra weight streaming.

## 2026-06-18 q4/q8 qMV Tile Follow-up

General-only patches (`--patterns off`), raw oMLX first, L512, repeat=3,
warmup=1, prompt `x`.

| Model | Raw TPS | FuseKit TPS | Delta | Raw peak | FuseKit peak | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B-4bit | 46.054 | 46.683 | +1.37% | 2.487 GB | 2.521 GB | Kept q4 large single-token qMV `cols_per_sg=8` rule. |
| Qwen3.5-4B-8bit | 24.730 | 25.112 | +1.55% | 4.586 GB | 4.620 GB | q8 swiglu MLX TG qMV helper kept; q8 large qMV `cols_per_sg=8` rejected. |
| Qwen3.5-4B-bf16 | 12.875 | 14.804 | +14.98% | 8.503 GB | 8.546 GB | Unchanged by q4/q8 tile rules. |

Measured and rejected:

- q8 explicit affine regroup in the generated qMV inner: correctness passed,
  but late-token GPU total regressed `38.09 ms -> 38.82 ms`.
- q8 large qMV `cols_per_sg=8`: single late-token profile improved slightly
  (`38.18 ms -> 37.94 ms`), but raw-first repeat=3 regressed e2e to `-0.77%`.
  Removed from defaults.
- q8 LinearResidual `cols_per_sg=4`: correctness passed, but late-token GPU
  total regressed `37.94 ms -> 38.26 ms`. Default remains `2`.

## Next Candidates

Only pursue these after measuring a baseline profile first.

1. q8/q4 qMV microkernel body improvements that preserve arithmetic order.
   Focus on the swiglu-absorbed and LinearResidual qMV paths, not more tile
   knob sweeps.
2. Semantic MLP partial tiling: reduce swiglu intermediate round-trip without
   collapsing down projection parallelism. Needs a cost model for saved
   activation traffic vs extra partial accumulator traffic.
3. Long-context SDPA/KV path only after qMV bottlenecks are stable; current
   q8 L512 profile shows SDPA is not the main cost at 512 tokens.
4. Sampling-aware LM-head/top-k path for non-greedy real usage. Greedy direct
   argmax is already optimized; do not assume greedy-only improvements apply
   to sampling.

## 2026-06-18 q4/q8 AsType Boundary Absorb

General-only patches (`--patterns off`), raw oMLX first, repeat=5, warmup=1,
prompt `x`. This pass rewrites a pointwise/view `FusedChain` external input
from a materialized `AsType(half/bfloat16 -> float32)` output to the original
half/bfloat16 fusion output when that source is externally bindable. It removes
24 standalone AsType kernels without changing the chain's arithmetic model,
because the emitter already reads external operands as `float(inN[i])`.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |
| Qwen3.5-4B-8bit | 32 | match=True |

Patch coverage after the guard:

| Model | Before tasks | After tasks | Removed |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 577 | 553 | 24 AsType tasks |
| Qwen3.5-4B-8bit | 577 | 553 | 24 AsType tasks |

Late-token profile effect:

| Model | Before GPU total | After GPU total | Delta |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 22.59 ms | 21.37 ms | -5.4% |
| Qwen3.5-4B-8bit | 39.48 ms | 38.86 ms | -1.6% |

Raw-first repeat=5 history:

| Run | Model | L | Raw TPS | FuseKit TPS | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| before | Qwen3.5-4B-4bit | 32 | 36.177 | 44.258 | +22.34% |
| before | Qwen3.5-4B-4bit | 512 | 45.213 | 46.081 | +1.92% |
| before | Qwen3.5-4B-8bit | 32 | 22.031 | 24.931 | +13.16% |
| before | Qwen3.5-4B-8bit | 512 | 25.524 | 25.855 | +1.29% |
| final | Qwen3.5-4B-4bit | 32 | 34.310 | 43.096 | +25.61% |
| final | Qwen3.5-4B-4bit | 512 | 42.887 | 44.917 | +4.73% |
| final | Qwen3.5-4B-8bit | 32 | 20.956 | 23.652 | +12.87% |
| final | Qwen3.5-4B-8bit | 512 | 24.186 | 24.710 | +2.17% |

The final run had lower absolute raw and FuseKit TPS than the earlier run,
so the delta is useful but not by itself enough to claim a large absolute
throughput gain. The kernel profile is the stronger evidence for keeping the
AsType absorb.

Additional candidates tested after AsType absorb:

- q8 qMV qdot body (`FUSEKIT_QGEMV_UNROLL2=0`,
  `FUSEKIT_QGEMV_MLX_QDOT=1`): correctness passed, but total GPU regressed
  `38.86 ms -> 39.16 ms`. Rejected.
- q8 fanout `sgs=2`: correctness passed, but total GPU regressed
  `38.86 ms -> 39.16 ms`. Rejected.
- q4 adjacent unroll off: correctness passed, but total GPU regressed
  `21.37 ms -> 21.64 ms`. Rejected.
- q8 LinearResidual `cols_per_sg=1` for large q8 residual shapes:
  correctness passed and one candidate profile hit `38.33 ms`, but the
  final source artifact did not reproduce it (`39.48 ms`) and down-proj
  time regressed. Rejected; default remains `cols_per_sg=2`.

## 2026-06-18 L512 q8 QParam Broadcast

Scope: L512 only, 4B q4/q8, general fusion only (`--patterns off`), raw oMLX
first, repeat=5, warmup=1, prompt `x`.

Initial L512 rerun after the AsType absorb showed q8 had become the active
problem:

| Model | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 45.751 | 46.370 | +1.35% | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 25.596 | 24.206 | -5.43% | 4.586 GB | 4.620 GB |

Late-token q8 profile before this change:

| Bucket | GPU time |
| --- | ---: |
| Total | 38.89 ms |
| FusedChain swiglu-absorbed | 12.94 ms |
| FusedChain reduction-fanout | 6.93 ms |
| LinearResidual M=9216 | 6.83 ms |
| LM-head argmax | 5.23 ms |
| LinearResidual M=4096 | 3.33 ms |

Kept change:

- Enable qparam scale/bias broadcast by default for q8 qMV helper kernels.
  This is bits/shape-level, not model-name matching. q4/q5/q6 defaults stay
  unchanged.
- Rationale: with q8 `group_size=64`, each scale/bias pair is reused by eight
  lanes. Broadcast cuts repeated qparam loads in reduction-fanout and residual
  qMV helpers. q4 stays on direct loads because previous q4 profiles favored
  avoiding the broadcast shuffle.

Rejected in the same session:

- `FUSEKIT_SWIGLU_CAP=512`: correctness passed, but swiglu time regressed
  `12.94 ms -> 13.17 ms` and wall did not improve. Rejected and repatched
  without the cap.

Final no-env q8 artifact:

| Model | L | Raw TPS | FuseKit TPS | Delta | FuseKit sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-8bit | 512 | 25.951 | 26.331 | +1.47% | 0.019 | 4.586 GB | 4.620 GB |

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-8bit | 32 | match=True |

## Useful Commands

Patch general fusion:

```sh
cargo run -- patch mlx-community/Qwen3.5-4B-8bit --force --chip m5 --patterns off
```

Late-token FuseKit profile:

```sh
FUSEKIT_PROFILE_KERNELS_AT=511 FUSEKIT_PROFILE_COMPACT=1 \
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages \
/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11 \
scripts/bench_omlx_engine.py ~/.omlx/models/mlx-community/Qwen3.5-4B-8bit \
--mode engine_fusekit --tokens 512 --repeat 1 --warmup 0 --prompt x --seed 0
```

Raw vs FuseKit repeat benchmark:

```sh
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages \
/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11 \
scripts/bench_omlx_engine_suite.py \
--model ~/.omlx/models/mlx-community/Qwen3.5-4B-8bit \
--tokens 512 --repeat 3 --warmup 1 --prompt x \
--out /private/tmp/fusekit-q8-l512-restored-r3.jsonl
```

## 2026-06-18 Automatic Quant Scratch Policy

Scope: Qwen3.5-4B q4/q8/bf16, L512, general fusion only
(`--patterns off`), raw oMLX first, repeat=3, warmup=1, prompt `x`.

Question: q4/q8/bf16 should not require user knobs. The lowering decision
must be automatic from graph semantic + dtype/quant metadata.

Kept changes:

- Centralize qMV default policy in `src/codegen/kernels/mod.rs` and consume it
  from `src/codegen/msl.rs`, so shape admission and emitted MSL macros agree.
- Make qMV threadgroup-vector helpers accept `threadgroup T*` scratch instead
  of only `threadgroup float*`.
- Use activation-typed norm scratch only for q8 quantized fused chains.
  q4 stays on float scratch because measured L512 throughput regressed with
  bfloat scratch. bf16 dense chains also stay on float scratch.

Rejected:

- q4+q8 activation scratch globally. Correctness passed, but q4 L512 fell to
  +0.81% and bf16 had an unstable negative run. q8 improved, so the policy was
  narrowed to q8-only by quant `bits`.
- q8 explicit/u32 qdot paths. They failed 32-token strict correctness earlier
  and remain rejected.

Correctness after q8-only policy:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |
| Qwen3.5-4B-8bit | 32 | match=True |
| Qwen3.5-4B-bf16 | 32 | match=True |

Final L512 benchmark after q8-only policy:

| Model | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 40.716 | 42.869 | +5.29% | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 22.619 | 23.415 | +3.52% | 4.586 GB | 4.620 GB |
| Qwen3.5-4B-bf16 | 13.061 | 14.526 | +11.22% | 8.503 GB | 8.546 GB |

Interpretation:

- The bf16 win comes from semantic general fusion, especially RMSNorm + MLP
  fanout/SwigLU and dense LinearResidual round-trip reduction.
- q4/q8 run through the same semantic fusion, but the remaining bottleneck is
  quant qMV kernel time in FusedChain/LinearResidual/LM-head.
- q8 benefits from activation-typed norm scratch because it halves scratch
  traffic without the q4-specific helper penalty.
- q4 needs a different quant qMV microkernel improvement; applying the q8
  scratch rule to q4 is a measured regression.

## 2026-06-19 q4 qGEMV Prefetch Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch with `FUSEKIT_QGEMV_PREFETCH=1`.
- This only changes the generated qGEMV body when qparam broadcast and
  unroll2 are both disabled. It is a bits/shape-level emitter option, not a
  model-name rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | Wall TPS |
| --- | ---: | ---: |
| q4 default | 21.20 ms | 45.79 |
| q4 prefetch | 22.30 ms | 45.38 |

Decision:

- Rejected. Prefetch increased late-token GPU time and did not improve wall
  throughput.
- Artifact was restored with plain `cargo run -- patch ... --patterns off`.

## 2026-06-19 q4 LinearResidual u32 Load Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Replace the q4 `LinearResidual` M=1 helper's four `uint16_t` packed loads
  with the shared `fusekit_q4_dot_u32_unshifted` helper used by other q4 qMV
  paths.
- This was a quant-metadata-level microkernel change, not a model-specific
  rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | LinearResidual GPU | Wall TPS |
| --- | ---: | ---: | ---: |
| q4 default | 21.20 ms | 5.38 ms | 45.79 |
| q4 u32 residual | 21.88 ms | 5.56 ms | 45.20 |

Decision:

- Rejected. The u32 helper reduced source duplication but increased the
  residual bucket and total GPU time. The original u16 four-value body remains
  the q4 residual default.

## 2026-06-19 q4 Fanout Pair2 Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch with `FUSEKIT_QMV_FANOUT_PAIR2=1`, forcing same-shape sibling
  fanout qMVs to share the x vector load.
- This is a graph/shape-level QmvFanout rule, not a model-name rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | Reduction-fanout GPU | Wall TPS |
| --- | ---: | ---: | ---: |
| q4 default | 21.20 ms | 3.55 ms | 45.79 |
| q4 pair2 forced | 22.78 ms | 4.10 ms | 45.03 |

Decision:

- Rejected. Shared x-load did not offset q4 register pressure in the large
  decode fanout. The existing guard that disables large q4 pair2 remains.

## 2026-06-19 q8 Fanout Pair2-Off Rejected

Scope: Qwen3.5-4B-8bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch with `FUSEKIT_QMV_FANOUT_PAIR2=0`, disabling same-shape sibling
  fanout qMV pairing for q8.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-8bit | 32 | match=True |

Profile/e2e result:

| Artifact | Total GPU | Reduction-fanout GPU | Wall TPS |
| --- | ---: | ---: | ---: |
| q8 pair2 off | 38.72 ms | 6.90 ms | 25.481 |
| q8 default restored | 40.22 ms | 7.15 ms | 25.489 |

Decision:

- Rejected/no code change. The wall throughput was effectively identical, and
  the profile movement did not isolate a causal bucket improvement. Keep q8
  default pair2 behavior.

## 2026-06-19 FusedChain Fast Rsqrt Rejected

Scope: Qwen3.5-4B q4/q8/bf16, L512, general fusion only (`--patterns off`),
raw-first repeat=3, warmup=1, prompt `x`.

Experiment:

- Add an env-gated FusedChain emitter switch to emit `metal::rsqrt` instead of
  `metal::precise::rsqrt` for FusedChain RMSNorm scale.
- Patch q4/q8/bf16 artifacts with `FUSEKIT_FAST_RSQRT=1`.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |
| Qwen3.5-4B-8bit | 32 | match=True |
| Qwen3.5-4B-bf16 | 32 | match=True |

Benchmark:

File: `/private/tmp/fusekit-fast-rsqrt-4b-l512-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 45.346 | 46.039 | +1.53% | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 25.469 | 25.900 | +1.69% | 4.586 GB | 4.620 GB |
| Qwen3.5-4B-bf16 | 13.970 | 14.956 | +7.06% | 8.503 GB | 8.546 GB |

Decision:

- Rejected/no code change. Correctness passed, but repeat=3 e2e did not beat
  the current known-good baseline (`q4 +1.84%`, `q8 +1.69%`, `bf16 +7.11%`).
- The env switch was removed and artifacts were restored with default
  `precise::rsqrt`.

## 2026-06-19 Dense bf16 Follow-up Rejections

Scope: Qwen3.5-4B-bf16, L512, general fusion only (`--patterns off`).

| Candidate | Correctness | Profile / e2e result | Decision |
| --- | --- | --- | --- |
| Dense `LinearResidual` output rows per simdgroup = 2 | 32-token match=True | token-511 GPU total `~68.5 ms -> 70.73 ms`; `LinearResidual` `18.15 ms -> 18.29 ms`; wall `13.814 TPS` in repeat=1 profile | reject/restored |
| Dense chain activation-typed `norm_y` scratch | 32-token match=True | token-511 profile improved (`~68.5 ms -> 67.45 ms`), but repeat=3 e2e did not beat default: candidate `14.911 TPS` vs restored default `14.967 TPS` | reject/restored |

Both candidates were model-independent shape/dtype experiments, not model-name
patterns. They were rejected because e2e did not preserve the profile win.

## 2026-06-19 q8 TG2 Packs=2 Rejected

Scope: Qwen3.5-4B-8bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch q8 with `FUSEKIT_QMV_FAST_TG2_PACKS_Q8=2`, matching the smaller
  per-thread pack count used by MLX's standalone qmv_fast helper.
- This only changes the bits/chip qMV helper tiling; it is not model-name
  dependent and does not add a graph pattern.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-8bit | 32 | match=True |

Profile/e2e result:

| Artifact | Total GPU | FusedChain GPU | LinearResidual GPU | Wall TPS |
| --- | ---: | ---: | ---: | ---: |
| q8 default | 40.11 ms | 21.06 ms | 10.41 ms | 25.918 |
| q8 packs=2 | 38.97 ms | 20.40 ms | 10.00 ms | 25.945 |

Decision:

- Rejected/no source change. The profile improved, but repeat=3 e2e only moved
  by `+0.10%`, inside run-to-run noise. The default artifact was restored.

## 2026-06-19 q4 QParam Broadcast Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch q4 with `FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q4=1` so qMV helper
  lanes share scale/bias loads through `simd_broadcast`.
- This is a quant metadata policy (`bits=4`, `group_size=64`), not a
  model-name rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | Swiglu qMV GPU | LinearResidual GPU | Wall TPS |
| --- | ---: | ---: | ---: | ---: |
| q4 default | 22.73 ms | 6.94 ms | 5.59 ms | 44.944 |
| q4 qparam broadcast | 22.50 ms | 7.29 ms | 5.59 ms | 44.482 |

Decision:

- Rejected/no source change. The top swiglu qMV kernel and wall throughput both
  regressed, despite some unrelated bucket noise. The default artifact was
  restored.

## 2026-06-19 q4 TG2 Packs=4 Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch q4 with `FUSEKIT_QMV_FAST_TG2_PACKS_Q4=4`, affecting only the
  swiglu-absorbed two-output qMV helper used by fused gate/up.
- This is a bits/helper tiling policy, not a model-name rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | Swiglu qMV GPU | LinearResidual GPU | Wall TPS |
| --- | ---: | ---: | ---: | ---: |
| q4 default | 22.73 ms | 6.94 ms | 5.59 ms | 44.944 |
| q4 tg2 packs=4 | 22.50 ms | 7.38 ms | 5.49 ms | 44.676 |

Decision:

- Rejected/no source change. The targeted swiglu qMV path regressed and wall
  throughput did not improve. The default artifact was restored.

## 2026-06-19 LayerBlock Attention Megakernel Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`), with
`FUSEKIT_LAYER_MEGAKERNEL=1`.

Experiment:

- Enable the structural per-layer `LayerBlock` composer. This emits one
  attention-block megakernel per layer and is admitted by layer/dataflow
  structure, not model names.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-4bit | 32 | match=True |

Profile result:

| Artifact | Tasks | Total GPU | SDPA GPU | Added LayerBlock GPU | Wall TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4 default | 553 | 22.73 ms | 0.37 ms | 0.00 ms | 44.944 |
| q4 LayerBlock | 561 | 24.40 ms | 0.31 ms | ~1.89 ms `other` | 42.295 |

Decision:

- Rejected/no source change. The current LayerBlock path compiles and is
  correct, but it is not a complete replacement for enough existing work.
  It reduces a small SDPA bucket while adding megakernel/barrier work, so L512
  wall throughput regresses. The default artifact was restored.

## 2026-06-19 FusedChain Simple Sigmoid Rejected

Scope: Qwen3.5-4B q4/q8, L512, general fusion only (`--patterns off`).

Experiment:

- Change FusedChain SwiGLU tail from the branch-stable form
  `sigmoid(g) = exp(-abs(g))` reconstruction to direct
  `1 / (1 + exp(-g))`, matching the older standalone qMM SwiGLU helper.
- This was an op-semantic FusedChain emitter change, not a model-name rule.

Correctness/profile result:

| Model | Correctness | Total GPU | Swiglu GPU | Wall TPS | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.5-4B-4bit | 32-token match=True | 22.28 ms -> 22.48 ms | 7.12 ms -> 7.22 ms | 45.27 -> 44.95 | reject |
| Qwen3.5-4B-8bit | 32-token match=False | not profiled | not profiled | not used | reject |

Decision:

- Rejected and reverted. q4 regressed, and q8 drifted immediately:
  raw generated assignment-style text while FuseKit generated input-prompt
  text. q4/q8 artifacts were restored with the stable sigmoid form and q8
  correctness was re-confirmed.

## 2026-06-19 LinearResidual Typed-SMEM TGX Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`), with
`FUSEKIT_LINEAR_RESIDUAL_TGX=1`.

Experiment:

- Revisit the previously rejected LinearResidual TGX path, but stage the
  activation block in threadgroup memory using the model activation dtype
  (`bfloat`) instead of `float`.
- This targets the down projection activation reread cost and is admitted by
  quant shape (`active_rows=1`, bits 4/8, K >= 4096), not model name.

Correctness/profile result:

| Artifact | Correctness | Total GPU | Down LinearResidual GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | match=True | 22.28 ms | 3.62 ms | 45.27 |
| q4 typed-SMEM TGX | match=True | 22.93 ms | 4.14 ms | 43.39 |

Decision:

- Rejected and reverted. Reducing activation global rereads did not pay for
  the extra block copy, barriers, and threadgroup reads. Default q4 artifact
  was restored.

## 2026-06-19 bf16 Dense LinearResidual Rows=2 Rejected

Scope: Qwen3.5-4B-bf16, L512, general fusion only (`--patterns off`).

Experiment:

- Open an env-gated dense `LinearResidual` rows-per-simdgroup override and
  test `FUSEKIT_LINEAR_RESIDUAL_DENSE_ROWS=2`.
- This targets dense down/o-proj activation reuse. It is dtype/shape based,
  not model-name based.

Correctness/profile result:

| Artifact | Correctness | Total GPU | Down LinearResidual GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| bf16 default | match=True | 67.21 ms | 11.90 ms | 14.49 |
| bf16 rows=2 | match=True | 67.41 ms | 12.56 ms | 14.49 |

Decision:

- Rejected and reverted. Extra output-column coalescing increased live ranges
  and made the dense down projection slower. The experiment override was
  removed and the bf16 artifact was restored to rows=1.

## 2026-06-19 q8 QParam Broadcast OFF Rejected

Scope: Qwen3.5-4B-8bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch q8 with `FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q8=0`.
- Motivation: MLX's baseline `qmv_fast_impl` loads scale/bias directly per
  lane, while the current FuseKit q8 helper defaults to `simd_broadcast` for
  qparam sharing. This is a quant metadata policy (`bits=8`, group size 64),
  not a model-name rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-8bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | Swiglu GPU | QKV fanout GPU | Down GPU | LM-head GPU | Wall TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q8 default | 38.56 ms | 13.12 ms | 6.56 ms | 6.65 ms | 5.22 ms | 25.608 |
| q8 broadcast off | 38.70 ms | 12.79 ms | 7.17 ms | 6.74 ms | 5.22 ms | 25.768 |

Decision:

- Rejected and restored default. The targeted swiglu bucket improved, but the
  same qparam policy worsened QKV fanout and down projection enough to increase
  total GPU time. The single-run wall improvement is treated as noise because
  the late-step GPU profile regressed.

## 2026-06-19 Current L512 Same-Run Baseline

Scope: Qwen3.5-4B q4/q8/bf16, L512, repeat=3, warmup=1, general fusion only
(`--patterns off`).

| Model | Raw TPS | FuseKit TPS | Delta | Raw peak | FuseKit peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 45.999 | 46.582 | +1.27% | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 25.608 | 26.157 | +2.14% | 4.586 GB | 4.620 GB |
| Qwen3.5-4B-bf16 | 14.132 | 15.102 | +6.86% | 8.503 GB | 8.546 GB |

Notes:

- q4/q8 are still positive on same-run L512, but not near the earlier
  historical +8-11% runs; those older values were affected by a lower raw
  baseline. Treat this table as the current working baseline.
- q8 late-token profile is weight-streaming dominated:
  swiglu ~13.1 ms, down ~6.6 ms, QKV fanout ~6.6 ms, LM-head ~5.2 ms.
- `silu_up -> down` full fusion admission debug rejected all 32 layers:
  dtype-sized SMEM fits (`~28.8 KiB`) but would collapse down-projection
  parallelism from ~640 threadgroups to a single threadgroup. Any next attempt
  must preserve down parallelism; simple full fusion is structurally rejected.

## 2026-06-19 bf16 LM-head 64-TG Rejected

Scope: Qwen3.5-4B-bf16, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch bf16 with `FUSEKIT_FINAL_LM_HEAD_TGS=64` instead of the dense
  default 128. This is a vocab/hidden/quant policy for direct greedy LM-head
  argmax, not a model-name rule.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-bf16 | 32 | match=True |

Profile result:

| Artifact | Total GPU | LM-head GPU | Wall TPS |
| --- | ---: | ---: | ---: |
| bf16 default 128 TG | 67.45 ms | 9.59 ms | 14.759 |
| bf16 64 TG | 66.66 ms | 9.79 ms | 14.766 |

Decision:

- Rejected and restored default. The changed target bucket (LM-head) regressed;
  the total-GPU improvement came from unrelated bucket noise, so it is not a
  defensible default rule.

## 2026-06-19 q8 SwiGLU Pair QParam Direct-Load Candidate

Scope: Qwen3.5-4B-8bit, L512, general fusion only (`--patterns off`).

Experiment:

- Keep q8 qparam broadcast enabled for fanout/residual/LM-head helpers, but
  switch the two-output `fusekit_qmv_fast_tg2_partial_impl` pair helper to
  direct scale/bias loads when `bits == 8`.
- This isolates the earlier global broadcast-off result: swiglu improved there,
  while QKV fanout and down projection regressed. The new policy is helper-role
  + quant metadata based, not model-name based.

Correctness:

| Model | Tokens | Result |
| --- | ---: | --- |
| Qwen3.5-4B-8bit | 32 | match=True |

Profile result:

| Artifact | Total GPU | Swiglu GPU | QKV fanout GPU | Down GPU | LM-head GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| q8 default | 38.56 ms | 13.12 ms | 6.56 ms | 6.65 ms | 5.22 ms |
| q8 tg2 direct qparam | 37.99 ms | 12.84 ms | 6.49 ms | 6.60 ms | 5.16 ms |

Same-run wall result:

| Raw TPS | FuseKit TPS | Delta | Raw peak | FuseKit peak |
| ---: | ---: | ---: | ---: | ---: |
| 25.736 | 26.172 | +1.69% | 4.586 GB | 4.620 GB |

Decision:

- Keep as a candidate rather than rejecting. The targeted late-token GPU buckets
  improved without the global-broadcast regression. Wall improvement is still
  small and should be rechecked with the full q4/q8/bf16 L512 batch before
  treating this as a final accepted optimization.

## 2026-06-19 bf16 Dense Gate/Up Dual K-Loop Rejected

Scope: Qwen3.5-4B-bf16, L512, general fusion only (`--patterns off`).

Experiment:

- Combine the two dense gate/up GEMV K-loops inside the swiglu-absorbed dense
  `FusedChain`, so each `norm_y` SMEM vector is loaded once and feeds both
  accumulators.
- This is the dense analogue of the q4/q8 two-output qMV helper and is based
  on graph semantics (`two heavy dense projections share the same normalized
  row`), not model names.

Correctness:

| Step | Result |
| --- | --- |
| Initial generated code | failed |
| Cause | generated local `float4 ny4` shadowed the `threadgroup ... *ny4` pointer |
| After renaming locals to `nyv*` | 32-token match=True |

Profile result after fixing correctness:

| Artifact | Total GPU | Swiglu dense GPU | Wall TPS |
| --- | ---: | ---: | ---: |
| bf16 restored default | 66.35 ms | 22.68 ms | 14.756 |
| dense dual K-loop | 67.32 ms | 23.62 ms | 14.698 |

Decision:

- Rejected and reverted. The reduced `norm_y` read/loop overhead did not pay
  for the additional live `float4` values and dual accumulator pressure. The
  bf16 artifact was restored and 32-token correctness re-confirmed.

## 2026-06-19 L512 Raw-First Batch After q8 Pair Candidate

Scope: Qwen3.5-4B q4/q8/bf16, L512, repeat=3, warmup=1, prompt `x`,
general fusion only (`--patterns off`). Baseline was always run first for each
model.

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Raw peak | FuseKit peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 46.072 | 46.630 | +1.21% | 0.082 | 0.097 | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 25.736 | 26.172 | +1.69% | 0.125 | 0.348 | 4.586 GB | 4.620 GB |
| Qwen3.5-4B-bf16 | 14.222 | 15.178 | +6.72% | 0.005 | 0.052 | 8.503 GB | 8.546 GB |

Notes:

- q4/q8 remain small positive wins at L512; q8's helper-local qparam direct
  load improves late-token GPU buckets, but e2e improvement is still modest.
- bf16 stays the strongest L512 win because semantic fusion removes more dense
  pointwise/runtime work while keeping dense GEMV tiling conservative.

## 2026-06-19 q4 FusedChain Activation-Typed SMEM Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Extend the q8-only FusedChain normalized-row SMEM dtype rule to q4, so
  bf16/half activations are stored in activation dtype instead of float after
  RMSNorm.
- This is a quant/dtype semantic rule, not model-name matching. The hypothesis
  was that q4 swiglu qMV would read less SMEM bandwidth while preserving the
  already-rounded activation values.

Correctness/profile result:

| Artifact | Correctness | Total GPU | Swiglu GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | match=True | 21.53 ms | 6.93 ms | 45.41 |
| q4 activation-typed SMEM | match=True | 21.71 ms | 7.00 ms | 45.69 |

Decision:

- Rejected and reverted. Although the stored values are semantically safe, the
  changed SMEM/conversion path increases the swiglu bucket, so the q4 artifact
  was restored to float SMEM for this path.

## 2026-06-19 q4 Reduction-Fanout-Only Rows=2 Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Isolate output-row coalescing to the reduction-fanout/QKV FusedChain only
  instead of using global `FUSEKIT_GEMV_ROWS=2`, which previously widened
  swiglu and regressed.
- This is a structural fanout schedule experiment: multiple quantized
  projections share one normalized row, independent of model name.

Correctness/profile result:

| Artifact | Correctness | Total GPU | Reduction-fanout GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | match=True | 21.53 ms | 3.58 ms | 45.41 |
| q4 reduction-fanout rows=2 | match=True | 21.73 ms | 3.81 ms | 45.20 |

Decision:

- Rejected and reverted. Even when isolated from swiglu, the extra output-row
  live state increases register pressure enough to slow the target QKV bucket.
  The temporary patch-time override was removed and the q4 artifact restored.

## 2026-06-19 q4 Quant LM-Head 2-Row Argmax Scan Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Change quantized `FinalRmsNormLmHead` direct-argmax scan so one simdgroup
  evaluates two vocab rows (`v` and `v + sgs_total`) per loop iteration via the
  two-output qMV helper.
- This mirrors MLX `qmv_fast`'s multi-row idea while preserving the direct
  argmax reduction. It is a vocab/head quant schedule rule, not model-name
  matching. It only applies to the greedy/direct-argmax path.

Correctness/profile result:

| Artifact | Correctness | Total GPU | LM-head scan GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | match=True | 21.53 ms | 2.79 ms | 45.41 |
| q4 2-row LM-head scan | match=True | 22.29 ms | 3.52 ms | 44.33 |

Decision:

- Rejected and reverted. The target LM-head bucket regressed badly: the
  norm_y reuse did not pay for two accumulators, extra live qparam/helper
  state, and the valid-row branch. The q4 artifact was restored and
  correctness re-confirmed.

## 2026-06-19 q8 Materialized-Norm QmvSwiglu Rejected

Scope: Qwen3.5-4B-8bit, L512, general fusion only (`--patterns off`).

Experiment:

- Repatch with `FUSEKIT_Q8_SWIGLU_MATERIALIZE_NORM=1`, which keeps the
  RMSNorm as a separate semantic custom kernel and lowers gate/up/SwiGLU to
  `QmvSwiglu` (`cols/sg=4`) instead of the default single fused
  `RMSNorm -> gate/up -> SwiGLU` FusedChain (`ROWS=1`).
- This is a structural q8 experiment: recover MLX-style output-column
  parallelism at the cost of one extra norm materialization and dispatch.

Correctness/profile result:

| Artifact | Correctness | Total GPU | Gate/up+SwiGLU bucket | Extra RMSNorm | Wall TPS |
| --- | --- | ---: | ---: | ---: | ---: |
| q8 default | match=True | 38.31 ms | 13.06 ms | 0.00 ms | 26.09 |
| q8 materialized norm | match=True | 38.24 ms | 13.20 ms | 0.22 ms | 25.66 |

Follow-up:

- Also tried direct qparam loads inside `QmvSwiglu` for q8, mirroring the
  successful helper-local tg2 direct-qparam candidate. Correctness stayed
  `match=True`, but profile regressed to `38.50 ms` total and `25.50 TPS`.

Decision:

- Rejected and reverted. Recovering 4 output columns per simdgroup does not
  offset the extra RMSNorm materialization/dispatch, and the QmvSwiglu helper
  itself does not beat the default swiglu-absorbed FusedChain on this shape.

## 2026-06-19 q8 RmsNormQkv TG-qMV Helper Accepted

Scope: Qwen3.5-4B q4/q8/bf16, L512, repeat=3, warmup=1, general fusion only
(`--patterns off`), raw oMLX first.

Experiment:

- Route quantized `RmsNormQkv` dot products through the existing
  threadgroup-resident qMV helper instead of the inline generated GEMV body.
- This applies only by shape/metadata: M5, `bits=8`, `group_size=64`,
  `active_rows <= 1`, tail-free hidden dimension, and no decode cache sink.
  It is not model-name or op-pattern matching.

Correctness/profile:

| Artifact | Correctness | Total GPU | RmsNormQkv GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q8 default before | match=True | 38.31 ms | 1.86 ms | 26.09 |
| q8 TG-qMV helper | match=True | 37.89 ms | 1.83 ms | 26.02 single-run |
| q4 TG-qMV helper | match=True | 21.40 ms | 0.969 ms | 45.57 single-run |

q4 was rejected: the same helper slightly increased the target qkv bucket
(`~0.963 ms -> 0.969 ms`) and total GPU time. The default rule was therefore
narrowed to q8 by quant metadata.

Raw-first L512 repeat=3 after baking the q8 rule:

| Model | raw TPS | FuseKit TPS | Delta | raw peak | FuseKit peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 46.035 | 46.680 | +1.40% | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 25.923 | 26.428 | +1.95% | 4.586 GB | 4.620 GB |
| Qwen3.5-4B-bf16 | 14.205 | 15.200 | +7.00% | 8.503 GB | 8.546 GB |

Decision:

- Accepted for q8 only. The rule gives a small but repeatable q8 L512 win,
  keeps q4 and bf16 behavior stable, and preserves 100% custom task coverage
  with native fallback at 0%.

## 2026-06-19 q8 Reduction-Fanout TG-qMV Helper Rejected

Scope: Qwen3.5-4B-8bit, L512 late-token profile, general fusion only
(`--patterns off`).

Experiment:

- Temporarily route the quantized `FusedChain` reduction-fanout dot products
  through the threadgroup-resident qMV helper, analogous to the accepted
  `RmsNormQkv` helper path.
- The candidate was gated by quant metadata and decode shape during the
  experiment, not model name.

Correctness/profile:

| Artifact | Correctness | Total GPU | Reduction-fanout GPU | RmsNormQkv GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: | ---: |
| accepted q8 baseline | match=True | 37.89 ms | 6.55 ms | 1.83 ms | 26.02 |
| reduction-fanout TG-qMV trial | match=True | 38.38 ms | 6.58 ms | 1.83 ms | 26.02 |

Decision:

- Rejected and reverted. Unlike `RmsNormQkv`, the reduction-fanout shape did
  not benefit from the helper; both total GPU time and the target bucket moved
  in the wrong direction.

## 2026-06-19 q8 LinearResidual TGX DType-SMEM Rejected

Scope: Qwen3.5-4B-8bit, L512 late-token profile, general fusion only
(`--patterns off`).

Experiment:

- Revisit the previously rejected `LinearResidual` TGX activation staging path,
  but store the staged activation block in the model activation dtype instead
  of widening it to `float` in threadgroup memory.
- Motivation: the q8 4B MLP down path repeatedly reads the same `silu_up`
  activation vector; reducing staging bandwidth looked like the only plausible
  way to make TGX viable without changing output-row tiling.

Correctness/profile:

| Artifact | Correctness | Total GPU | LinearResidual GPU | Down M=9216 GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: | ---: |
| accepted q8 baseline | match=True | 37.74 ms | 9.78 ms | 6.57 ms | 26.01 |
| TGX dtype-SMEM trial | match=True | 38.90 ms | 10.52 ms | 7.13 ms | 25.24 |

Decision:

- Rejected and reverted. Even after avoiding `float` SMEM, the extra
  threadgroup staging and barriers outweighed any saved activation reads.
  The measured target bucket regressed, so this is not a valid default rule.

## 2026-06-19 q8 LinearResidual NAX M=1 Rejected

Scope: Qwen3.5-4B-8bit, L512 late-token profile, general fusion only
(`--patterns off`).

Experiment:

- Force the existing NAX `qmm_t` LinearResidual path for single-row decode
  (`FUSEKIT_LINEAR_RESIDUAL_NAX_M1=1`) to check whether the MLX-style tiled
  quantized matmul path beats the current qMV helper for large down/o_proj
  projections.

Correctness/profile:

| Artifact | Correctness | Total GPU | LinearResidual GPU | Down M=9216 GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: | ---: |
| accepted q8 baseline | match=True | 37.74 ms | 9.78 ms | 6.57 ms | 26.01 |
| NAX M=1 trial | match=True | 44.98 ms | 16.54 ms | 11.36 ms | 22.25 |

Decision:

- Rejected/restored. For M=1 decode the tiled NAX qmm path is much slower than
  the qMV helper; keep NAX for small-M/MTP-style cases where it is explicitly
  admitted, not single-token decode.

## 2026-06-19 bf16 Dense FusedChain K16 Unroll Rejected

Scope: Qwen3.5-4B-bf16, L512 late-token profile, general fusion only
(`--patterns off`).

Experiment:

- Add an env-gated dense FusedChain inner-loop variant that unrolls 16 `float4`
  chunks per lane instead of the current M5 large-dense default of 8 chunks.
- This keeps output-row parallelism unchanged; only the K-loop body changes.

Correctness/profile:

| Artifact | Correctness | Total GPU | Dense swiglu GPU | Dense QKV fanout GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: | ---: |
| bf16 default | match=True | 66.02 ms | 22.53 ms | 14.71 ms | 14.82 |
| K16 unroll trial | match=True | 67.84 ms | 23.71 ms | 15.11 ms | 14.40 |

Decision:

- Rejected and reverted. The larger K body increased register/live-range
  pressure more than it reduced loop overhead. M5 bf16 decode stays on K8.

## 2026-06-19 Current L512 Final Check

Scope: Qwen3.5-4B q4/q8/bf16, L512, repeat=3, warmup=1, raw oMLX first,
general fusion only (`--patterns off`).

File: `/private/tmp/fusekit-4b-l512-current-final-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Raw peak | FuseKit peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 45.911 | 46.752 | +1.83% | 0.113 | 0.023 | 2.487 GB | 2.521 GB |
| Qwen3.5-4B-8bit | 25.969 | 26.503 | +2.05% | 0.008 | 0.031 | 4.586 GB | 4.620 GB |
| Qwen3.5-4B-bf16 | 14.212 | 15.199 | +6.94% | 0.018 | 0.013 | 8.503 GB | 8.546 GB |

Notes:

- q8 artifact was restored after the rejected TGX and NAX M=1 trials.
- bf16 artifact was restored after the rejected K16 dense-chain trial.
- Remaining q4/q8 L512 gap is still quant qMV weight-streaming time, not
  native fallback: all tested 4B artifacts report 0 native fallback tasks.

## 2026-06-19 q4/q8 SwiGLU TG2 Packs Rejected

Scope: Qwen3.5-4B q4/q8, L512 late-token profile, general fusion only
(`--patterns off`).

Experiment:

- Rebuild q4 `FusedChain` SwiGLU pair helper with
  `FUSEKIT_QMV_FAST_TG2_PACKS_Q4=4` and `=1` instead of the default `2`.
- Rebuild q8 with `FUSEKIT_QMV_FAST_TG2_PACKS_Q8=2` and `=1` instead of
  the default `4`.
- This tests the quant qMV microkernel block size/register-pressure tradeoff
  by bits and shape. It is not model-name matching.

Correctness/profile:

| Artifact | Correctness | Total GPU | SwiGLU GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | same_argmax=True | 21.55 ms | 6.85 ms | 45.67 |
| q4 tg2 packs=4 | same_argmax=True | 21.49 ms | 6.98 ms | 45.31 |
| q4 tg2 packs=1 | not re-run; same generated path | 21.63 ms | 6.88 ms | 45.59 |
| q8 default | same_argmax=True | 37.91 ms | 12.76 ms | 25.90 |
| q8 tg2 packs=2 | same_argmax=True | 38.45 ms | 13.03 ms | 25.86 |
| q8 tg2 packs=1 | not re-run; same generated path | 38.45 ms | 13.08 ms | 25.84 |

Decision:

- Rejected and both q4/q8 artifacts restored to default. q8's wider default
  block is already the best measured point; q4's default is also the best
  balance once the total chain and wall time are considered. The next q4/q8
  work should not revisit tg2 packs without a different kernel body.

## 2026-06-19 q4/q8 Parallel Gate/Up SwiGLU Rejected

Scope: Qwen3.5-4B q4/q8, L512 late-token profile, general fusion only
(`--patterns off`).

Experiment:

- In the quantized `RMSNorm -> gate qMV -> up qMV -> SwiGLU` fused chain,
  split one output element across a pair of simdgroups: one simdgroup computes
  gate, the other computes up, then a threadgroup barrier combines them.
- This was intended to expose more qMV parallelism while still avoiding the
  gate/up intermediate DRAM round-trip. It is shape/quant-metadata based, not
  model-name matching.

Correctness/profile:

| Artifact | Correctness | Total GPU | SwiGLU GPU | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | same_argmax=True | 21.55 ms | 6.85 ms | 45.67 |
| q4 parallel gate/up | same_argmax=True | 22.27 ms | 7.02 ms | 45.55 |
| q8 default | same_argmax=True | 37.91 ms | 12.76 ms | 25.90 |
| q8 parallel gate/up | same_argmax=True | 38.34 ms | 12.79 ms | 25.71 |

Decision:

- Rejected and removed from source. The extra barrier and reduced independent
  output-row scheduling cost more than the gate/up parallelism saves on M5.
  q4/q8 artifacts were restored to the default general fusion path.

## 2026-06-19 q4/q8 Quant qMV Policy Retests Rejected

Scope: Qwen3.5-4B q4/q8, L512 late-token profile, general fusion only
(`--patterns off`).

Experiments:

- q4 MLX-like qMV geometry: force `cols_per_sg=4` instead of the current
  q4 large-projection rule.
- q4 qdot-on: force q4 helper qdot path instead of explicit unpack.
- q8 residual qdot-off: force explicit q8 residual path.
- q8 qparam broadcast-off: disable q8 helper scale/bias broadcast.

Correctness/profile:

| Artifact | Correctness | Total GPU | Main changed bucket | Wall TPS |
| --- | --- | ---: | ---: | ---: |
| q4 default | same_argmax=True | 21.55 ms | swiglu 6.85 ms | 45.67 |
| q4 cols/sg=4 | same_argmax=True | 22.18 ms | swiglu 6.88 ms | 45.39 |
| q4 qdot-on | same_argmax=True | 22.05 ms | swiglu 7.03 ms | 45.38 |
| q8 default | same_argmax=True | 37.91 ms | swiglu 12.76 ms | 25.90 |
| q8 residual qdot-off | same_argmax=True | 38.95 ms | residual 6.65 ms | 25.81 |
| q8 qparam broadcast-off | same_argmax=True | 38.91 ms | LM-head 5.36 ms | 25.67 |

Decision:

- Rejected and artifacts restored. These policies confirm the current q4/q8
  L512 bottleneck is not missing fusion coverage: it is quant qMV weight
  streaming and register/live-range balance. The fused paths already beat raw
  LM-head in the synchronized raw profile; further q4/q8 gains need a new
  qMV microkernel body or a structural change that reduces weight reads, not
  another geometry/qparam knob.

## 2026-06-19 Current L512 q4/q8/bf16 Check

Scope: Qwen3.5-4B q4/q8/bf16, L512, repeat=3, warmup=1, raw oMLX first,
general fusion only (`--patterns off`).

File: `/private/tmp/fusekit-4b-l512-q4q8bf16-after-quant-reject-r3.jsonl`.

| Model | Raw TPS | FuseKit TPS | Delta | Raw sd | FuseKit sd | Raw peak | FuseKit peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 45.625 | 46.167 | +1.19% | 0.136 | 0.066 | 2.491 GB | 2.503 GB |
| Qwen3.5-4B-8bit | 25.671 | 26.160 | +1.91% | 0.040 | 0.039 | 4.591 GB | 4.604 GB |
| Qwen3.5-4B-bf16 | 14.034 | 15.024 | +7.05% | 0.005 | 0.009 | 8.503 GB | 8.546 GB |

Notes:

- q4/q8 artifacts are restored to the default general-fusion path after the
  rejected policy retests.
- All patched artifacts report 0 native fallback tasks during patch coverage.

## 2026-06-19 q4/q8 SwiGLU Grid/Rows Rechecks Rejected

Scope: Qwen3.5-4B q4/q8, L512, general fusion only (`--patterns off`).

Experiments:

- `FUSEKIT_SWIGLU_CAP=256`: reduce per-row SwiGLU threadgroups so one TG
  reuses the normalized RMSNorm row across multiple output strips.
- `FUSEKIT_GEMV_ROWS=2 + FUSEKIT_SWIGLU_ROWS2_TG_QMV=1`: revisit the rejected
  rows-per-simdgroup=2 policy, but use the MLX-style threadgroup qMV helper
  twice instead of the generic multirow qMV fallback.

Correctness/profile:

| Artifact | Correctness | Wall TPS | Decision |
| --- | --- | ---: | --- |
| q4 `SWIGLU_CAP=256` | same_argmax=True on non-tie prompt; tie prompt already differs in default | 45.199 | reject |
| q8 `SWIGLU_CAP=256` | same_argmax=True | 25.127 | reject |
| q4 rows=2 TG-qMV | same_argmax=True | 45.210 | reject |
| q8 rows=2 TG-qMV | same_argmax=True | 25.672 | reject |

Decision:

- Rejected and restored q4/q8 artifacts. Both attempts reduce redundant
  normalized-row work, but lose more from reduced scheduler parallelism and
  larger qMV live ranges. The temporary rows=2 TG-qMV source path was removed.
- This reinforces the current direction: q4/q8 L512 needs a new qMV
  microkernel body or a structural weight-read reduction. More grid caps or
  multi-output rows are not useful on M5 for this shape.

## 2026-06-19 q4/q8 qMV Body Rechecks Rejected

Scope: Qwen3.5-4B q4/q8, L512, general fusion only (`--patterns off`).

Experiments:

- q4 TG2 half-split accumulator: split the pair helper's q4 unpack into low
  four-lane and high four-lane accumulations to shorten temporary live ranges.
- q8 qdot-off/u32 helper: force the order-preserving u32-load dot body instead
  of MLX `qdot` in q8 helper paths.

Correctness/benchmark:

| Artifact | Correctness | Raw TPS | FuseKit TPS | Delta | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| q4 half-split | same_argmax=True | 45.701 | 46.451 | +1.6% | reject |
| q4 default rerun | same_argmax=True | 45.703 | 46.460 | +1.7% | keep default |
| q8 qdot-off/u32 | same_argmax=True | n/a single path | 25.053 | n/a | reject |

Decision:

- q4 half-split was indistinguishable from default in a same-condition
  raw-first A/B, so it was removed.
- q8 qdot-off/u32 regressed wall throughput. Keep q8 `qdot` enabled.
- q4/q8 artifacts were restored to default general fusion.

## 2026-06-19 q4/q8 LM-head Rechecks Rejected

Scope: Qwen3.5-4B q4/q8, L512, general fusion only (`--patterns off`).

Current late-token profiles:

| Model | Total GPU | Chain | LinearResidual | LM-head argmax | QKV | SDPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q4 default | 22.33 ms | 11.05 ms | 5.77 ms | 2.83 ms | 1.20 ms | 0.38 ms |
| q8 default | 39.18 ms | 20.58 ms | 10.09 ms | 5.36 ms | 1.86 ms | 0.35 ms |

Raw synchronized MLX op profiles at the same L512 position show the heavy
raw quantized matmul buckets are already larger than FuseKit's fused active
time:

| Model | Raw gate+up qmatmul | Raw down qmatmul | Raw LM-head qmatmul | FuseKit swiglu | FuseKit down | FuseKit LM-head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q4 | 19.82 ms | 10.06 ms | 2.96 ms | 6.91 ms | 3.70 ms | 2.83 ms |
| q8 | 26.00 ms | 13.16 ms | 5.47 ms | 13.21 ms | 6.73 ms | 5.35 ms |

Experiments:

| Experiment | Correctness | Result | Decision |
| --- | --- | --- | --- |
| q8 `FUSEKIT_FINAL_LM_HEAD_TGS=16` | same_argmax=True | raw 25.706 TPS, fusekit 24.569 TPS, -4.4% | reject |
| q4 `FUSEKIT_FINAL_LM_HEAD_TGS=16` | same_argmax=True | raw 45.816 TPS, fusekit 45.609 TPS, -0.5% | reject |
| q4/q8 quant LM-head `norm_y` SMEM dtype = activation dtype | same_argmax=True on q4/q8 | q4 repeat=3 raw 45.933 TPS, fusekit 46.084 TPS, +0.3%; q8 single raw 25.763 TPS, fusekit 26.164 TPS, +1.6% | reject |

Decision:

- Rejected and restored source/artifacts to default. The activation-dtype SMEM
  path reduced no stable wall time and weakened q4 compared with the default
  +1-2% L512 result.
- q4/q8 remaining L512 bottleneck is not LM-head scan geometry. It is mostly
  quantized MLP weight streaming (`QmvSwiglu` + `LinearResidual`). The next
  non-repeated direction must be a real structural MLP/down fusion or a new
  qMV microkernel body, not more TG/packs/qparam knob sweeps.

## 2026-06-19 q4 Materialized-Norm QmvSwiglu Rejected

Scope: Qwen3.5-4B-4bit, L512, general fusion only (`--patterns off`).

Experiment:

- Temporarily generalize the existing q8 materialized-norm admission to q4:
  keep RMSNorm as a standalone custom kernel and emit gate/up/SwiGLU as
  `QmvSwiglu`.
- This is a graph/quant-metadata structural split, not model-name matching.

Coverage/correctness:

| Artifact | Tasks | Fusions | Native fallback | Correctness |
| --- | ---: | ---: | ---: | --- |
| q4 materialized norm | 585 | `QmvSwiglu=32`, `Single(RMSNorm)=32` | 0/585 | same_argmax=True |
| q4 default restored | 553 | `FusedChain=208` | 0/553 | restored default artifact |

Benchmark:

| Artifact | Raw TPS | FuseKit TPS | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| q4 materialized norm | 45.772 | 45.181 | -1.3% | reject |
| q4 current default batch | 46.158 | 46.709 | +1.2% | keep default |

Decision:

- Rejected and restored. Splitting RMSNorm out of the fused chain increases
  task count and dispatch/round-trip work. The standalone `QmvSwiglu` geometry
  does not recover enough qMV throughput to pay for that, matching the earlier
  q8 materialized-norm rejection.

## 2026-06-19 Greedier General Fusion: Pointwise-Prefix RMSNorm Kept

Scope: Qwen3.5-4B q4/q8/bf16, L512, general fusion only (`--patterns off`).

Change:

- Enable the existing semantic `try_pointwise_reduction_fanout_merge` pass in
  `src/fusion/general.rs`.
- Route pointwise-prefix reduction chains through the existing
  `emit_pointwise_prefix_reduction_chain` emitter in
  `src/codegen/kernels/fused_chain.rs`.
- Fix reduction-body operand indexing so flat-alias view inputs
  (`Reshape`, same element count) are read as `in[row * HIDDEN + i]` instead
  of scalar `in[0]`.

Correctness:

| Model | Correctness | max_abs_diff | mean_abs_diff |
| --- | --- | ---: | ---: |
| Qwen3.5-4B q4 | same_argmax=True | 0.171875 | 0.031358 |
| Qwen3.5-4B q8 | same_argmax=True | 0.125000 | 0.021675 |
| Qwen3.5-4B bf16 | same_argmax=True | 0.121094 | 0.021267 |

Coverage:

| Model | Custom tasks | Native fallback | Fused/specialized | Source ops fused |
| --- | ---: | ---: | ---: | ---: |
| q4 | 553/553 | 0/553 | 340/553 | 938/1247 |
| q8 | 553/553 | 0/553 | 340/553 | 938/1247 |
| bf16 | 700/700 | 0/700 | 340/700 | 1131/1587 |

Benchmark, raw-first repeat=3:

| Model | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4 L512 | 45.953 | 46.519 | +1.2% | 2.487 GB | 2.521 GB |
| q8 L512 | 25.683 | 26.116 | +1.7% | 4.586 GB | 4.620 GB |
| bf16 L512 | 14.082 | 15.120 | +7.4% | 8.503 GB | 8.546 GB |

Profile note:

- q4 L512 token 511 still spends most GPU time in quantized MLP qMV:
  SwiGLU 6.85 ms, down `LinearResidual` 3.67 ms, reduction-fanout 3.59 ms,
  LM-head 2.82 ms. The new pointwise-prefix RMSNorm merge removes small
  reshape/RMSNorm round trips but is not the primary q4/q8 bottleneck.

Decision:

- Keep. This is a general dataflow/shape fusion, not model-name or fixed-pattern
  matching. It improves coverage and gives a small stable q4/q8 gain while
  preserving correctness; bf16 benefits more because dispatch/activation
  overhead is a larger share of the measured path.

## 2026-06-19 L512 Recheck and Greedier Fusion Scan

Scope: Qwen3.5-4B q4/q8/bf16, L512, raw oMLX baseline first, general fusion
only (`--patterns off`). Pattern matcher stayed off.

Benchmark, raw-first repeat=3:

| Model | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4 L512 | 45.736 | 46.415 | +1.5% | 2.487 GB | 2.521 GB |
| q8 L512 | 25.701 | 26.295 | +2.3% | 4.586 GB | 4.620 GB |
| bf16 L512 | 14.069 | 15.122 | +7.5% | 8.503 GB | 8.546 GB |

Patch coverage remains full-custom:

| Model | Custom tasks | Native fallback | Fused/specialized |
| --- | ---: | ---: | ---: |
| q4 | 553/553 | 0/553 | 340/553 |
| q8 | 553/553 | 0/553 | 340/553 |
| bf16 | 700/700 | 0/700 | 340/700 |

Greedier fusion debug on q8:

- `try_qmv_sibling_fanout_merge`: `qmm=65`, `target_ok=65`, `groups=65`,
  `mergeable_groups=0`, `max_group=1`. There are no leftover sibling qMV
  groups after the current RMSNorm fanout and semantic tail lowering.
- `try_reduction_fanout_pointwise_tail_lowering`: finds all 32 MLP
  gate/up pairs and lowers them into the current SwiGLU-producing fused
  chain.
- `SwiGLU -> LinearResidual(down)` admission: all 32 candidates reject with
  `reject_parallelism`. For q8 4B, the full activation fits only in dtype
  SMEM (`28800B < 32768B`) but a single-TG full fusion would collapse down
  projection parallelism from 640 threadgroups to one. Recomputing gate/up
  per down tile would reread about 30GB of gate/up quantized weights per layer
  to save about 189MB of SwiGLU activation traffic, so that lowering is
  structurally wrong for this shape.

Decision:

- No additional plain graph-greedy merge is currently available for q4/q8
  L512. The current general pass already reaches the semantic
  `RMSNorm + gate/up + SwiGLU` point.
- Do not re-run the already rejected TGX, rows=2, materialized-norm, or
  LayerBlock-style attempts without a different parallelism model.
- The next real q4/q8 path must be a new qMV/down microkernel or a multi-stage
  MLP lowering that preserves down-projection parallelism without recomputing
  gate/up weights per down tile.

## 2026-06-20 DeepSeek V2 Lite 4bit Bring-up

Model: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`
(`model_type=deepseek_v2`, q4, MLA split dims q_nope=128, q_pe=64, v=128,
MoE top-6). Downloaded into oMLX model root; final local size is 8.2GB.

Patch result with general fusion only (`--patterns off`):

| Metric | Value |
| --- | ---: |
| custom task kernels | 1850/1850 |
| native fallback tasks | 0/1850 |
| fused/specialized tasks | 298/1850 |
| source ops covered | 2383/2633 |
| source ops fused | 779/2383 |

Bring-up fix:

- Dense `FusedChain` heavy-first emitter had an extra generated `}` after
  function close. DeepSeek V2 emits repeated small dense `[8,2048] x
  `[2048,64]` chains, exposing the syntax bug. Removed the extra close in the
  general dense chain emitter; this is not model-specific.

Correctness:

| Mode | Result |
| --- | --- |
| default FuseKit | `same_argmax=true`, `max_abs_diff=0`, `mean_abs_diff=0` |
| forced strict custom dispatch | `same_argmax=true`, `max_abs_diff=8.1171875`, `mean_abs_diff=1.5609607` |

Runtime note:

- Default FuseKit correctly detects DeepSeek MLA attention state and falls
  back to native dispatch for the session:
  `MLA q_nope/q_pe and asymmetric KV cache bindings are not fully lowered`.
- Forced strict dispatch proves the generated task graph can run, but logits
  drift is too large. Do not use this model as a performance win until MLA
  semantic lowering is implemented.

Short fallback benchmark, L32, repeat=3, raw first:

| Path | TPS | First token | Peak |
| --- | ---: | ---: | ---: |
| raw oMLX | 53.591 | 0.142717s | 8.946GB |
| FuseKit default fallback | 49.830 | 0.152221s | 8.993GB |

Post-budget rebench, L32, repeat=3, raw first:

| Path | TPS | First token | Peak |
| --- | ---: | ---: | ---: |
| raw oMLX | 58.490 | 0.126525s | 8.946GB |
| FuseKit default fallback | 53.477 | 0.146319s | 8.991GB |

Delta: -8.6%. Runtime warning still reports MLA state fallback, so this is not
a valid custom-kernel performance comparison.

Post-budget rebench, L512, repeat=3, raw first:

| Path | TPS | First token | Peak |
| --- | ---: | ---: | ---: |
| raw oMLX | 67.746 | 0.142988s | 9.104GB |
| FuseKit default fallback | 63.871 | 0.155056s | 9.206GB |

Delta: -5.7%. Same warning; the custom dispatch path is not active.

Decision:

- Model is downloaded and patchable.
- Full custom DeepSeek V2 is not correctness-clean yet. Next required work is
  the general DeepSeek MLA lowering: q_nope/q_pe semantic split, asymmetric
  qk/v SDPA emitter, and MLA cache-store/state binding.

### 2026-06-20 Budget Admission Cleanup

The pointwise-tail lowering path already received register/SMEM budgets but did
not enforce them; it used only a saved-activation-vs-heavy-stream heuristic.
Updated `pointwise_tail_absorb_budget_ok` to reject candidates that exceed the
target register or SMEM budget before applying the traffic heuristic.

This does not make DeepSeek MLA correct by itself. It prevents the general pass
from silently admitting a tail fusion solely because it looks semantically
recognizable. Candidate recognition and budget admission are now separated more
cleanly:

- recognition: graph/dataflow/shape/state role;
- admission: register budget, SMEM budget, recompute/round-trip economics,
  preserved parallelism.

Verification: `cargo check` passed.

### 2026-06-20 Post-Budget Rebench: Qwen3.5 4B

Reason: budget admission and `FusedChain` emitter close handling changed after
the previous Qwen benches. Old L32/L512 numbers are no longer authoritative.
All runs below used general fusion only (`--patterns off`), raw oMLX first,
then FuseKit, repeat=3, warmup=1.

Patch/correctness status:

| Model | Patch coverage | Native fallback | Correctness |
| --- | ---: | ---: | --- |
| `mlx-community/Qwen3.5-4B-4bit` | 553/553 | 0/553 | `same_argmax=true`, max diff 0.171875 |
| `mlx-community/Qwen3.5-4B-8bit` | 553/553 | 0/553 | `same_argmax=true`, max diff 0.133789 |
| `mlx-community/Qwen3.5-4B-bf16` | 700/700 | 0/700 | `same_argmax=true`, max diff 0.121094 |

Benchmark summary:

| Model | Tokens | Raw TPS | FuseKit TPS | Delta |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-4bit | 32 | 35.518 | 42.965 | +21.0% |
| Qwen3.5-4B-4bit | 512 | 46.135 | 46.549 | +0.9% |
| Qwen3.5-4B-8bit | 32 | 21.819 | 24.503 | +12.3% |
| Qwen3.5-4B-8bit | 512 | 25.823 | 26.215 | +1.5% |
| Qwen3.5-4B-bf16 | 32 | 12.362 | 13.601 | +10.0% |
| Qwen3.5-4B-bf16 | 512 | 14.170 | 15.161 | +7.0% |

Interpretation:

- Short decode wins are still real; first-token/runtime dispatch reduction is
  visible on all three 4B variants.
- Long decode quant wins are now near flat. q4/q8 L512 no longer reproduce the
  earlier high single-digit/low double-digit wins, so the next optimization
  should focus on late-token GPU kernel time in quantized qMV/FusedChain rather
  than host dispatch.
- bf16 keeps a meaningful L512 win, so dense-path fusion/runtime changes are
  still useful. The weak point is quantized weight streaming, not general
  attachment or ICB coverage.

### 2026-06-20 DeepSeek MLA Drift Fix

Problem:

- Forced custom dispatch on
  `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` produced the same
  argmax but large logits drift: max abs 8.1171875, mean abs 1.56096065.
- Default dispatch fell back because MLA config was hard-guarded as unsupported.

Layer/profiling evidence:

- `scripts/debug_deepseek_layer0.py` showed Q, K, V, cache prefix, and manual
  SDPA matched raw closely, but the custom `SdpaDecode` output diverged
  (`sdpa_buf` max abs about 0.74 before the fix).
- The first real layer output drift was layer 0: max abs 0.4263.

Fixes:

- `SdpaDecode` Q indexing now uses `head_id * Q_SEQ + query_row` for
  head-major Q tensors (`[B,H,T,D]`) and `query_row * N_HEADS + head_id` for
  seq-major Q tensors (`[B,T,H,D]`). The old head-major path incorrectly used
  `OUT_SEQ`, so DeepSeek MLA heads read the wrong Q row when the trace sequence
  was wider than the emitted active sequence.
- `KvCacheStoreDecode` now records whether prepared current K/V is head-major
  or seq-major and stores into cache layout `[H, capacity, D]` accordingly. The
  launch grid uses the semantic sequence axis, not the head axis, for active M.
- The old MLA hard fallback was removed. MLA dispatch is now default when the
  graph declares itself dispatchable and scratch role validation passes. It can
  still be disabled with `FUSEKIT_DISABLE_MLA_DISPATCH=1`.
- `scripts/debug_deepseek_forced_layers.py` no longer prints misleading logits
  when the scan stops early.

Verification:

| Prompt / decode token | Dispatch | Max abs | Mean abs | Argmax | Top-5 |
| --- | --- | ---: | ---: | --- | --- |
| `Hello from fusekit` / 1 | default, no fallback | 0.0390625 | 0.00616983 | same 185 | same order |
| `Write a Python function that sums a list.` / 42 | default, no fallback | 0.0546875 | 0.00824775 | same 6190 | same order |
| `Explain why attention cache layout matters...` / 185 | default, no fallback | 0.0625 | 0.00845294 | same 300 | same order |

Full layer scan after the fix:

- Layer 0 max abs fell from 0.4263 to 0.000977.
- Final logits for the full 27-layer decode were max abs 0.0390625, mean abs
  0.00616983.

Interpretation:

- The semantic MLA drift is fixed; Q/K/V/cache and SDPA consumption are now
  aligned with MLX. The remaining logits difference is small q4 custom-kernel
  numerical drift from different accumulation/order, not a layout/state bug.

Post-guard speed rebench:

Setup: raw oMLX first, then default FuseKit, `--patterns off` patched artifact,
run0 discarded as warmup, 5 measured runs averaged. Prompt:
`Write a concise Python function that parses JSON lines and returns aggregate statistics.`

| Tokens | Raw TPS | FuseKit TPS | Delta | Raw peak | FuseKit peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 56.901 | 56.848 | -0.1% | 8.938GB | 8.986GB |
| 512 | 69.220 | 69.048 | -0.2% | 9.102GB | 9.275GB |

This replaces the earlier fallback-only comparison. Removing the MLA reject
guard moves DeepSeek from invalid fallback regressions (-8.6% L32, -5.7% L512)
to near parity, but the custom path is not yet faster. Next DeepSeek speed work
should profile the active custom graph directly: it still emits 1850 task
kernels, with only 298 fused/specialized tasks and 1552 single-op custom tasks.

### 2026-06-20 DeepSeek Routed MoE Fusion Pass

Goal:

- Keep MLA correctness fixed and move DeepSeek speed work from fallback parity
  to real custom-kernel wins, using general dataflow/shape semantics rather
  than model-name patterns.

Implemented and kept:

- Added `RoutedQmvSwiglu`: two compatible routed `GatherQMM` siblings that
  share activation row, row index, and expert index feed
  `sigmoid(gate) * gate * up`.
- Admission is by op/dataflow/shape:
  same activation source, same expert index, equivalent arange-based route row
  index, compatible q4/q8 affine quant metadata, and isolated SwiGLU consumer.
- DeepSeek patch result: 26 routed gate/up pairs fused, decode task graph
  reduced from 1506/1720-style routed path to 1402/1720 task range depending
  on optional experiment flags, with fallback still 0%.
- Correctness after this fusion:
  JSON prompt decode token 42 max abs 0.0390625, mean abs 0.00631715,
  argmax/top-5 unchanged.

Profile evidence:

| Variant | Token | Total GPU | Main routed buckets |
| --- | ---: | ---: | --- |
| before routed gate/up fusion | late decode | ~16.9ms | `GatherQMM` 78x ~7.13ms |
| routed gate/up fusion on | token 500 | ~17.7ms | `RoutedQmvSwiglu` 26x ~4.24ms, remaining down `GatherQMM` 26x ~2.17ms |

Rejected experiments:

- `RoutedQmvSwiglu` threadgroup-memory staging of the shared activation row:
  correctness held, but token-500 GPU regressed from ~17.7ms to ~18.5ms and
  `RoutedQmvSwiglu` grew from ~4.24ms to ~4.71ms. Rejected because barrier /
  occupancy cost outweighed reduced activation rereads.
- `RoutedQmvReduce` down projection + weighted route sum:
  correctness held and task count fell from 1720 to 1668, but L512 e2e
  regressed. Default is now opt-in only with
  `FUSEKIT_ROUTED_QMV_REDUCE_ABSORB=1`.

Current default benchmark, raw first, run0 discarded:

| Tokens | Raw TPS | FuseKit TPS | Delta | Notes |
| ---: | ---: | ---: | ---: | --- |
| 32 | 62.271 | 57.203 | -8.1% | custom 100%, fallback 0% |
| 512 | 71.943 | 69.258 | -3.7% | custom 100%, fallback 0% |

`RoutedQmvReduce` isolation:

| Variant | L512 FuseKit TPS | Delta vs default |
| --- | ---: | ---: |
| default, reduce off | 69.258 | 0.0% |
| reduce on | 68.601 | -0.9% |

Interpretation:

- General routed gate/up fusion is structurally correct and reduces the
  routed graph, but DeepSeek is still slower than raw oMLX because the custom
  q4 routed kernels themselves are not faster enough than MLX's asynchronous
  quantized path.
- The next useful work is not more route-sum fusion by serializing routes in
  one simdgroup. It should target q4 `RoutedQmvSwiglu` microkernel throughput
  and the remaining late-token buckets: SDPA (~2.1ms), chain (~2.0ms), shared
  expert `QmvSwiglu` (~1.6ms), and linear residual (~1.6ms).

Follow-up isolation after the raw baseline moved:

| Artifact | L32 TPS | L512 TPS | Notes |
| --- | ---: | ---: | --- |
| current default (`RoutedQmvSwiglu` on, reduce off) | 57.203 | 69.258 | tasks 1720, fallback 0 |
| `RoutedQmvSwiglu` off | 55.373 | 66.853 | tasks 1850, `GatherQMM` 78 |

So the later `-8.1%` L32 result was not caused by `RoutedQmvSwiglu`; disabling
it made both L32 and L512 slower. The apparent drop from the previous `-0.1%`
table came mostly from the raw oMLX baseline moving upward:

| Measurement | Raw L32 | FuseKit L32 | Raw L512 | FuseKit L512 |
| --- | ---: | ---: | ---: | ---: |
| earlier post-guard table | 56.901 | 56.848 | 69.220 | 69.048 |
| later rerun | 62.271 | 57.203 | 71.943 | 69.258 |

FuseKit itself was roughly flat/slightly up versus the earlier table; raw oMLX
reran materially faster, especially at L32. Treat the earlier `-0.1%` as a
stale/low raw baseline, not as a regression-free current comparison.

## Generic Pointwise-Tail Admission

Checkpoint before this work:

- `f415c4a Checkpoint general fusion optimization state`

Goal:

- Stop admitting Qwen/DeepSeek SwiGLU-like wins through direct named consumer
  checks. Use graph metadata to collect heavy-output aliases and a pointwise
  tail formula, then lower supported formulas through the existing fast emitter.

Implemented:

- Added `PointwiseTailSource`, `PointwiseTailRole`, and `PointwiseTailMatch`.
- Replaced direct `try_qmv_swiglu_absorb` / `try_routed_qmv_swiglu_absorb`
  admission with:
  - `try_qmv_fanout_pointwise_tail_absorb`
  - `try_routed_qmv_fanout_pointwise_tail_absorb`
- Replaced direct reduction-fanout consumer checks with
  `collect_pointwise_tail_formula`.
- Removed the direct `semantic_silu_gate_up_consumer` helper. The remaining
  tail match is expressed as a formula returned by the collector.
- Kept lowering compatibility with existing fast emitters. This is not a new
  model-name pattern; the current supported formula is still
  `SiluGateUp`, selected from op/dataflow/alias metadata.

Important cost-model fix:

- `pointwise_tail_absorb_budget_ok` was rejecting Qwen small-model MLP tails
  because it compared saved activation traffic against quantized weight stream.
  That was the wrong model for reduction-fanout tails: the producer already
  computes both heavy projections in one kernel, so attaching the pointwise tail
  does not serialize new heavy work. After reg/SMEM checks pass, the merge
  removes gate/up materialization and a standalone pointwise dispatch.

Patch artifacts:

| Model | Patterns | Tasks | Fallback | Notable fused path |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B-Instruct-4bit | off | 364 | 0% | 24 reduction-fanout MLP tails absorbed |
| DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx | off | 1720 | 0% | 26 `RoutedQmvSwiglu`, 26 dense `QmvSwiglu` |

Correctness:

| Model | Check | Result |
| --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct-4bit | raw vs patched decode logits | same argmax, top-5 preserved, max abs 0.0327148, mean abs 0.00475509 |

Bench, Qwen2.5-Coder-0.5B-Instruct-4bit, raw first, warmup discarded:

| Tokens | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 147.270 | 257.725 | +75.0% | 0.289 GB | 0.357 GB |
| 512 | 285.860 | 295.054 | +3.2% | 0.299 GB | 0.350 GB |

Interpretation:

- The Qwen win is now coming from the general reduction-fanout + pointwise-tail
  collector, not from `src/fusion/patterns/*`.
- This is still a staged generalization: the collector is generic over aliases
  and pointwise formula, while the only fast lowered formula today is
  `SiluGateUp`. Extending the formula enum and emitter hooks is the next step
  for non-SwiGLU pointwise tails.

## DeepSeek MLA Decode Round-Trip Cuts

Model:

- `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`
- Compiled with `--patterns off`; no `src/fusion/patterns/*` path.

Implemented in the general path:

- `KvCacheStoreDecode` can read K0/V directly from a shared packed
  `TransposeSplit([0,2,1,3])` source. Admission is by `KvCacheStoreDecode`
  semantic inputs, producer op attrs, axes, shapes, and dtype. It does not use
  model name, layer number, or weight names.
- `RoutedQmvReduce+LinearResidual` can absorb a single downstream `Add` as
  `residual_extra` when the reduce output is the isolated Add input and the
  other Add input has matching shape/dtype. This removes the post-MLP residual
  pointwise dispatch without changing the MoE route semantics.
- `Single(Split)` now dispatches and copies only active-M rows for decode.
  This is generic active-M lowering for split outputs, not a DeepSeek-specific
  rule.

Structural effect:

| Variant | Tasks | Decode pipeline | Split | TransposeSplit | Add | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous current | 705 | 408 | 27 | 54 | 26 | 0% |
| KV TransposeSplit sink | 705 | 381 | 27 | 27 | 26 | 0% |
| + Routed reduce output Add sink | 679 | 355 | 27 | 27 | 0 | 0% |
| + Split active-M grid/work | 679 | 355 | 27 | 27 | 0 | 0% |

Correctness:

| Variant | Check | Result |
| --- | --- | --- |
| finalsplit | raw vs patched decode logits | same argmax, max abs 0.03125, mean abs 0.00426964 |

Profile at token index 511:

| Variant | Kernels | Total GPU | Notable change |
| --- | ---: | ---: | --- |
| previous current | 405 | 16.40 ms | `op:transposesplit` 54 calls, ~0.22 ms in sampled run |
| KV sink | 378 | 15.63 ms | pipeline TransposeSplit 54 -> 27 |
| + Add sink | 352 | 15.21 ms | post-routed Add 26 -> 0 |
| + Split active-M | 352 | 14.63 ms | Split grid 4608 -> 768 threads; `op:split` drops from top profile |

L512 benchmark, raw first, warmup discarded:

| Variant | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| + Add sink | 72.493 | 73.487 | +1.4% | 9.125 GB | 9.208 GB |
| + Split active-M | 71.494 | 72.632 | +1.6% | 9.125 GB | 9.227 GB |
| + SDPA Q TransposeSplit sink | 71.507 | 73.199 | +2.4% | 9.125 GB | 9.222 GB |

Interpretation:

- The kernel-level improvement is larger than the e2e win because the remaining
  dominant work is routed qMV and SDPA. The retained changes still reduce
  dispatch count and late-token GPU time while preserving raw-vs-patched
  logits.
- These are general semantic/dataflow transforms. They are not pattern-matcher
  rules, and they remain active with `--patterns off`.

Additional default-on change:

- `SdpaDecode` now sinks a trace-side Q `TransposeSplit([0,2,1,3])` when both
  split outputs feed only that SDPA. Admission checks SDPA attrs, transpose
  axes, tensor shapes, dtype, and consumer isolation. It does not check model
  names, layer IDs, or weight names.
- Default compile without env now produces 652 tasks, 0 fallback tasks, and
  1205/1660 fused source ops (72.6%). This removed another 27 decode kernels
  versus the active-M variant.
- Correctness: raw vs patched decode logits still match argmax, max abs
  0.03125, mean abs 0.00426964.

Rejected candidates in the same profiling pass:

| Candidate | Evidence | Decision |
| --- | --- | --- |
| `RoutedQmvSwiglu` MLX/NAX fast qMV branch | Correctness OK, but L512 raw/fusekit was 71.680/71.728 TPS (+0.1%), and profiled `RoutedQmvSwiglu` rose to ~4.36 ms from ~4.1 ms. | Do not default. |
| `RoutedQmvReduce+LinearResidual` `cols=2` | Correctness OK, but profiled reduce rose to ~3.0 ms from ~2.9 ms and e2e fell in the sampled run. | Do not default. |

## 2026-06-22 Named MoE Composite Policy Correction

Correction to the previous MoE notes:

- `RoutedQmvSwiglu`, `RoutedQmvReduce`, `SoftmaxTopKRoute`, and
  `WeightedRouteReduce` are not model-name matches, but they are still named
  composite admissions. Under the stricter default-general definition they are
  pattern-like and must not run by default.
- Default general fusion now leaves these named MoE composites off. They are
  only enabled by explicit opt-in through `FUSEKIT_ENABLE_NAMED_MOE_PATTERNS=1`
  or the individual `FUSEKIT_*` knob.
- Recompiled DeepSeek with `--patterns off` and no opt-in env:
  886/886 custom tasks, 0 native fallback tasks, 1023/1660 source ops fused
  (61.6%). Grepping the emitted `task_graph.json` and generated kernels showed
  no `RoutedQmvSwiglu`, `RoutedQmvReduce`, `SoftmaxTopKRoute`, or
  `WeightedRouteReduce`.
- Speed regression check, L512, repeat=3, warmup=1, prompt `x`, old named
  artifact first and no-named artifact second:

| Artifact | Tasks | Named MoE composites | TPS avg | TPS sd | Peak |
| --- | ---: | --- | ---: | ---: | ---: |
| old installed artifact | 787 | on | 69.231 | 0.084 | 9.203 GB |
| no-named default artifact | 886 | off | 69.517 | 0.135 | 9.214 GB |

Result: no measurable speed regression from disabling named MoE composites in
this DeepSeek L512 decode check. The extra tasks are not currently slower than
the old named route in wall TPS; future MoE speedups still need to come from a
generic scheduler/IR transform rather than these named composite admissions.

Next MoE optimization direction:

- Do not reintroduce these as default named composites.
- Build MoE improvements through the generic fusion scheduler: consumer
  fanout, route-index reuse, reduction fan-in, weight-streaming bytes, register
  pressure, and scratch lifetime should decide merge boundaries. Any specialized
  emitted kernel must be derived from generic IR properties and cost model, not
  from a named MoE fusion kind.

## 2026-06-22 DeepSeek No-Named MoE Experiments

Scope: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`, L512 decode,
`--patterns off`, no named MoE composite strings in emitted artifact unless
explicitly stated.

Baseline no-named artifact:

- Artifact: `/private/tmp/deepseek-no-moe-model`
- Coverage: 886/886 custom tasks, 0 fallback tasks.
- L512 repeat=3 after warmup: 70.571 TPS in the first run set, 69.956 TPS in
  the later run set.
- Token 511 profile: 533 kernels, 17.03 ms sampled GPU time. Dominant buckets:
  `Single(GatherQMM)` 6.49 ms, `SdpaDecode` 2.00 ms, `FusedChain` 2.86 ms,
  `LinearResidual` 1.87 ms, `QmvSwiglu` 1.57 ms.

Rejected: `Single(GatherQMM)` MLX `qmv_fast` emitter.

- Implementation: generic `GatherQMM` op-level emitter based on quant metadata
  and shape, env-gated by `FUSEKIT_GATHER_QMM_MLX_QMV=1`.
- Coverage: 52 of 78 `GatherQMM` kernels used the MLX qmv-fast helper; no
  named MoE strings emitted.
- Correctness: argmax matched, max abs 0.03125, mean abs 0.00542.
- Benchmark: baseline 70.571 TPS vs candidate 70.360 TPS, -0.3%.
- Profile: `GatherQMM` only moved 6494 us -> 6430 us. This is too small and
  not stable enough to offset e2e noise. Keep env-gated off.

Rejected as default: generic `ScaledSumReduce`.

- Implementation: general graph lowering of
  `ReduceSum(Multiply(data, Broadcast(scale)))` to `ScaledSumReduce`. It uses
  op semantics, shapes, and single-use dataflow only, not model names or named
  MoE fusion kinds.
- Coverage: tasks dropped 886 -> 834; emitted 26 `ScaledSumReduce` kernels;
  no `RoutedQmv*`, `SoftmaxTopKRoute`, `WeightedRouteReduce`, or
  `weighted_route` strings in artifact.
- Correctness before dtype-rounding fix: argmax matched, max abs 0.046875,
  mean abs 0.00617, but L512 was 67.625 TPS vs 69.956 baseline (-3.3%).
- Semantic fix: preserve the original `Multiply` materialization dtype by
  rounding `data * scale` to the output dtype before accumulating. Correctness
  improved to max abs 0.03125, mean abs 0.00503.
- Benchmark after rounding fix: 68.315 TPS vs 69.956 baseline, -2.3%.
- Profile: the fused reduce bucket itself is smaller (`ScaledSumReduce` ~97 us
  vs old `Broadcast+Multiply+Reduce` ~381 us), but e2e still regresses and the
  generated token stream diverges enough that downstream MoE routing changes.
  Therefore it is opt-in only via `FUSEKIT_SCALED_SUM_REDUCE=1`, not default.

Current conclusion:

- The biggest no-named DeepSeek bottleneck remains generic `Single(GatherQMM)`.
- Replacing it with MLX qmv-fast did not improve the dominant bucket enough.
- Fusing route weighting/reduce is only acceptable if it preserves graph
  materialization semantics and wins e2e; the current implementation does not,
  so it remains off by default.

Default-on: generic `GatherQMM` output-scale sink.

- Implementation: sink `Broadcast(scale) -> Multiply(GatherQMM alias, scale)`
  into the producer store when the `GatherQMM` output and metadata aliases have
  a single multiply consumer. This is an op/dataflow transform: it emits a
  normal `Single(GatherQMM)` kernel with an extra `output_scale` input, not a
  named MoE fusion kind.
- Semantics: preserve original materialization order with
  `out = dtype(dtype(acc) * scale)`, so the old `GatherQMM` half/bfloat store
  and the old `Multiply` half/bfloat output rounding are both represented.
- Coverage: DeepSeek no-named tasks dropped 886 -> 808. Emitted 26
  `GatherQMM output-scale` kernels and no `RoutedQmv*`, `SoftmaxTopKRoute`,
  `WeightedRouteReduce`, or `ScaledSumReduce` strings.
- Correctness: argmax matched, max abs 0.03125, mean abs 0.00448.
- L32 repeat=3 after warmup: baseline 55.776 TPS vs candidate 57.596 TPS,
  +3.3%.
- L512 repeat=3 after warmup in the same noisy run set: baseline 67.764 TPS
  (sd 1.319) vs candidate 68.841 TPS (sd 0.745), +1.6%.
- Token 511 profile: kernel count 533 -> 481; sampled GPU time 17.03 ms ->
  16.23 ms. The old `Broadcast+Multiply` route-weight materialization is gone;
  `Reduce` remains a separate kernel to keep the original reduction order.

Rejected: `GatherQMM` half x-row threadgroup staging.

- Implementation tested: when the routed activation dtype is `half`, stage the
  reused x row in `threadgroup half` instead of `threadgroup float` and cast
  back to float inside the dot loop. This is a dtype/shape-level GatherQMM
  kernel change, not a model-name or MoE-pattern transform.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00448.
- L32 same-run: default 57.891 TPS vs candidate 58.042 TPS, +0.26%.
- L512 same-run: default 70.138 TPS vs candidate 68.692 TPS, -2.1%.
- Sequential token-511 profile: candidate reduced the `GatherQMM` bucket
  slightly, 6377 us -> 6242 us (-2.1%), and total sampled GPU time 15456 us ->
  15234 us (-1.4%), but the wall-clock decode benchmark regressed. Keep float
  x staging; the tested half-staging code was removed rather than left as a
  default knob.

Rejected: `GatherQMM` single-word qparam broadcast.

- Implementation tested: compile with `FUSEKIT_QGEMV_UNROLL2=0` and
  `FUSEKIT_QPARAM_BROADCAST=1`, replacing the default strided two-word qgemv
  body with a single-word loop that loads scale/bias once per quant group lane
  and broadcasts within the simdgroup.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00448.
- L512 same-run: default 71.737 TPS vs candidate 70.245 TPS, -2.1%.
- Sequential token-511 profile: `GatherQMM` bucket worsened, 6377 us -> 7216
  us; `GatherQMM output-scale` worsened 2103 us -> 2695 us. The existing
  strided two-word unroll remains the default.

Rejected: `GatherQMM` single-word qgemv prefetch.

- Implementation tested: compile with `FUSEKIT_QGEMV_UNROLL2=0` and
  `FUSEKIT_QGEMV_PREFETCH=1`, using the generic prefetch body instead of the
  current strided two-word qgemv unroll.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00448.
- L512 same-run: default 69.538 TPS vs candidate 67.351 TPS, -3.1%.
- Conclusion: the existing two-word strided unroll continues to win. The
  prefetch body is not promoted.

Rejected: `GatherQMM` NAX helper.

- Implementation tested: compile with `FUSEKIT_GATHER_QMM_NAX=1`, replacing
  eligible generic `GatherQMM` decode kernels with the MLX NAX qmm helper based
  on op metadata and shape. This emitted 52 `GatherQMM NAX` kernels; the 26
  `GatherQMM output-scale` kernels stayed on the existing default path.
- Correctness: first-token argmax matched, but max abs diff worsened from
  0.03125 to 0.046875 and mean abs diff worsened from 0.00448 to 0.00567.
- L512 repeat=3 after warmup, same-run: default 68.690 TPS vs candidate
  57.084 TPS, -16.9%.
- Observation: the candidate output stream drifted into a repetitive formula
  pattern during the L512 runs. This is not just a dispatch/noise artifact.
- Conclusion: keep NAX out of M=1 `GatherQMM` decode. The current strided qMV
  helper remains the faster and more stable default for this shape.

Rejected: standalone `GatherQMM` q4 qdot body.

- Implementation tested: force the MLX-style qdot body only for generic
  standalone `GatherQMM` with q4/g64 and large K, leaving `GatherQMMReduce`,
  residual projections, and named MoE paths unchanged. This was based on
  quant metadata and shape, not model name or pattern matching.
- Correctness: first-token argmax matched, but max abs diff worsened to
  0.046875 and mean abs diff to 0.00703.
- Profile: token-511 standalone `GatherQMM` improved about 4.74 ms -> 4.21 ms,
  but sampled chain time moved up in the same run and total sampled GPU time
  stayed near 15.8 ms.
- L512 repeat=3 after warmup, same-run: default 70.842 TPS vs candidate
  71.070 TPS, +0.3%. This is inside noise and not worth the larger drift.
- Conclusion: reject and restore the existing strided two-word q4 body.

Rejected/default-off: generic `FusedChain(GatherQMM, GatherQMM, Sigmoid, Multiply, Multiply)`.

- Implementation tested: recognize two sibling generic `GatherQMM` ops feeding
  a pointwise SwiGLU tail and lower them as a normal `FusedChain` with
  `GatherQMM -> GatherQMM -> Sigmoid -> Multiply -> Multiply` steps. This is
  role/shape/dataflow based and emits no `RoutedQmv*`, `SoftmaxTopKRoute`,
  `WeightedRouteReduce`, or `ScaledSumReduce` strings.
- Coverage: DeepSeek no-named tasks 808 -> 730 for the full output-scale +
  reduce + chain artifact; 26 paired GatherQMM+SwiGLU chain kernels emitted.
- Correctness: dispatcher built and first-token argmax matched, but max abs
  diff increased to 0.046875 and mean abs diff to 0.00582.
- L512 repeat=3 after warmup, raw first: raw 71.391 TPS vs candidate 71.647
  TPS, +0.4%. This is inside normal run noise and not enough to justify the
  larger diff.
- Token 511 profile: old `GatherQMM 52 + pointwise tail` path was about
  6.10 ms; fused pair+tail was about 6.06 ms, only ~39 us saved. The kernel
  count reduction did not translate into wall throughput.
- Conclusion: keep the code path for opt-in experiments via
  `FUSEKIT_GATHER_QMM_SWIGLU_CHAIN=1`, but default it off.

Rejected/default-off: generic `GatherQMMReduce` absorbing following `LinearResidual`.

- Implementation tested: when a generic `Single(GatherQMMReduce)` output is the
  isolated residual input of a quantized `LinearResidual`, append the shared
  projection inputs to the same generic reduce fusion. The emitter uses
  `TOP_K` simdgroups for routed down projection plus one simdgroup for shared
  down projection, then stores `moe_reduce + shared_down + residual_extra` in
  the original cast order.
- Coverage: `LinearResidual` tasks dropped 54 -> 28, with no named MoE strings
  and dispatchable task graph.
- Correctness: dispatcher built and first-token argmax matched, but max abs
  diff increased to 0.046875 and mean abs diff to 0.00499.
- Token 511 profile: `LinearResidual` bucket dropped about 1560 us -> 808 us,
  but `GatherQMMReduce` grew about 2185 us -> 2889 us and sampled total GPU
  time worsened to 15.57 ms.
- Conclusion: the extra shared simdgroup increases the reduce kernel's
  occupancy/register pressure enough to erase the saved scratch round-trip.
  Keep as opt-in only via `FUSEKIT_GATHER_QMM_REDUCE_LINEAR=1`.

Rejected: DeepSeek MLA SDPA flash-decoding forced on.

- Implementation tested: compile with `FUSEKIT_FORCE_SDPA_FLASH_DECODING=1` so
  the small decode-head guard does not disable split flash decoding for
  `n_heads=16, qk_dim=192, v_dim=128`.
- Correctness: first-token argmax matched, max abs diff 0.03125, mean abs diff
  0.00599.
- Token 511 profile: `SdpaDecode` worsened about 1986 us -> 2065 us and wall
  TPS fell to 66.626 in the sampled run.
- Conclusion: keep the existing shape rule that disables flash-decoding for
  this small-head decode shape.

Rejected: q4 dual qMV adjacent unroll for `QmvSwiglu`.

- Implementation tested: compile with `FUSEKIT_QGEMV_ADJACENT_UNROLL2=1`,
  forcing the adjacent two-word qgemv schedule currently defaulted only for q8.
- Correctness: first-token argmax matched, max abs diff 0.03125, mean abs diff
  0.00421.
- Token 511 profile: `QmvSwiglu` did not improve, about 1498-1524 us depending
  on paired run noise, and sampled total GPU time was 15.15 ms.
- Conclusion: keep q4 on the existing strided schedule; do not promote the q8
  adjacent schedule to q4.

Rejected: M5 `GatherQMM` rows-per-simdgroup = 2.

- Implementation tested: compile with `FUSEKIT_GATHERQMM_ROWS=2`, computing
  two adjacent expert output rows per simdgroup for generic `GatherQMM`.
- Correctness: first-token argmax matched, but max abs diff increased to
  0.046875 and mean abs diff to 0.00548.
- Token 511 profile: standalone `GatherQMM` did not improve, about 4.23 ms in
  the sampled run, and wall TPS was 66.832.
- Conclusion: M5 stays at `gatherqmm_rows_per_simdgroup() == 1`; the extra
  row accumulator pressure is not worth the activation reuse.

Rejected: DeepSeek `GatherQMM` qparam broadcast, second pass.

- Implementation tested: compile with `FUSEKIT_QPARAM_BROADCAST=1` on the
  no-named general-fusion DeepSeek artifact. This is still a quant metadata
  kernel variant, not a model-name path.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00421025.
- Token 511 profile: sampled total GPU time worsened about 16065 us -> 16654
  us. `GatherQMM` moved 4536 us -> 4584 us, `GatherQMMReduce` 2279 us -> 2402
  us, and `GatherAxisScaleExpand` 365 us -> 503 us.
- Conclusion: keep the existing two-word strided q4 qMV body. The broadcast
  body does not reduce memory traffic enough to offset its scheduling cost.

Rejected: DeepSeek MLA SDPA online softmax.

- Implementation tested: compile with `FUSEKIT_SDPA_FLASH=1`, using the
  online softmax SDPA body rather than the current decode SDPA shape rule. This
  is separate from the previously rejected split flash-decoding force path.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00599063.
- Token 511 profile: sampled total GPU time worsened about 16065 us -> 17074
  us. `SdpaDecode` roughly doubled, 1961 us -> 3930 us.
- Conclusion: for the DeepSeek MLA decode shape (`qk_dim=192`, `v_dim=128`,
  small active-M), the current non-online SDPA kernel is the right default.

Rejected: `ArgTopK` small-k one-simdgroup select and row dispatch correction.

- Implementation tested: for generic `Single(ArgTopKAsc/Desc)`, dispatch by
  logical rows instead of output elements and use a one-simdgroup repeated
  select body for `top_k <= 8`.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00421025.
- Token 511 profile: `ArgTopKDesc` dropped out of the top buckets, but sampled
  total GPU time still worsened about 16065 us -> 16523 us due to higher time
  in the surrounding routed qMV/reduce/residual buckets.
- L512 repeat=3 after warmup, same-run: default 67.164 TPS vs candidate
  64.311 TPS, -4.2%.
- Conclusion: reject both parts for now and restore the older ArgTopK path.
  This route can only be revisited if isolated dispatch-only testing proves a
  measurable e2e win without the serialized select body.

Rejected: DeepSeek `GatherQMMReduce` q4 qdot-force removal.

- Implementation tested: remove the q4/k>=1024 `QdotPolicy::Force` branch in
  generic `Single(GatherQMMReduce)` so the reduce path uses the default q4
  explicit two-word qMV body. This is an op/quant metadata change, not a named
  MoE pattern.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00466487.
- Token 511 profile: sampled total GPU time only moved 15764 us -> 15733 us,
  while the target `GatherQMMReduce` bucket worsened 2262 us -> 2299 us.
- L512 repeat=3 after warmup, same-run: default 69.618 TPS vs candidate
  70.051 TPS, +0.6%. This is below the acceptance threshold and profile does
  not support it as a real reduce-kernel improvement.
- Conclusion: keep the qdot-force branch for q4 `GatherQMMReduce`; the
  candidate was reverted.

Rejected: current-pass DeepSeek `GatherQMM` x-row SMEM disabled.

- Implementation tested: compile the current no-named general-fusion artifact
  with `FUSEKIT_GATHERQMM_X_SMEM=0`, leaving fusion coverage identical at
  782/782 custom tasks and 0 native fallback tasks.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00421025.
- Token 511 profile: `GatherQMMReduce` improved 2348 us -> 2218 us, but the
  larger `GatherQMM` bucket worsened 4222 us -> 4405 us. Sampled total GPU
  time was effectively flat, 15490 us -> 15508 us.
- L512 single run: default 65.418 TPS vs candidate 65.537 TPS, +0.18%, below
  measurement noise.
- Conclusion: keep the current shape rule that stages the routed x row in
  threadgroup memory when it fits and a full threadgroup can reuse the row.

Kept: `GatherQMMReduce` remove dead partial initialization barrier.

- Implementation: in generic `Single(GatherQMMReduce)`, remove the initial
  threadgroup partial zero-fill and barrier. Every route slot read by the final
  reducer is overwritten after the dot product, and the existing post-dot
  barrier remains. This is an op-internal memory/barrier optimization and does
  not introduce a named MoE pattern.
- Correctness: first-token argmax matched; max abs diff 0.03125, mean abs diff
  0.00421025.
- Coverage: unchanged, 782/782 custom tasks, 0 native fallback tasks.
- L512 repeat=3 after warmup, same-run default-vs-candidate: default 70.011
  TPS vs candidate 72.420 TPS, +3.4%.
- L512 raw-first repeat=3 on the candidate model: raw 72.508 TPS vs candidate
  fusekit 72.579 TPS, +0.1%. This means the change recovers the fusekit
  default artifact regression in this DeepSeek run, while raw remains a hard
  ceiling for this path.
- Warm token-511 profile is noisy but consistent with a small reduce-side win:
  candidate `GatherQMMReduce` sampled around 2166-2195 us versus default
  2151-2231 us across adjacent runs; no fallback or task-count change.

Rejected: DeepSeek `GatherQMMReduce` multi-output columns on M5.

- Implementation tested: compile with `FUSEKIT_ROUTED_REDUCE_COLS=2` and
  `FUSEKIT_ROUTED_REDUCE_COLS=4`, which makes one route-reduce simdgroup
  compute adjacent output columns (`OUT_COLS`) instead of the M5 default 1.
  This is an op/quant/shape scheduling change, not a named MoE pattern.
- Coverage: unchanged, 782/782 custom tasks, 0 native fallback tasks.
- Correctness: both kept first-token argmax. `OUT_COLS=2` increased mean abs
  diff from the kept default's 0.00421025 to 0.00438653; `OUT_COLS=4`
  increased it to 0.00459238.
- Token 511 profile: `OUT_COLS=2` reduced the target
  `GatherQMMReduce output-scale` bucket from 2292 us to 2226 us, but sampled
  total GPU stayed flat/worse (15677 us -> 15686 us) and standalone
  `GatherQMM` moved slightly up. `OUT_COLS=4` regressed total GPU to 16327 us
  and `GatherQMMReduce` to 2331 us.
- L512 repeat=3 after warmup, same order default then candidate:
  default 72.072 TPS vs `OUT_COLS=2` 71.868 TPS (-0.3%).
- Conclusion: keep M5 `routed_reduce_cols_per_simdgroup(...) == 1`. The
  adjacent-column reuse is real inside the reduce bucket at 2 columns, but it
  increases enough surrounding pressure/noise that e2e loses.

Rejected/guarded: global SG/TG reduction on DeepSeek routed kernels.

- Implementation tested: compile with `FUSEKIT_TG_SGS=4` to reduce
  simdgroups per threadgroup across routed quantized kernels.
- Coverage: unchanged, 782/782 custom tasks and 0 native fallback tasks.
- Correctness: first-token argmax happened to match, but logit drift became
  unacceptable: max abs diff 2.447265625 and mean abs diff 0.36677.
- Root cause: `GatherQMMReduce` needs one simdgroup per route in the same
  threadgroup. DeepSeek TOP_K=6, so a global 4-SG threadgroup silently skipped
  route slots 4 and 5. The apparently faster reduce bucket was therefore not
  computing the same operation.
- Code action kept: route-reduce threadgroup sizing now clamps against the
  chip default budget, not the global sweep knob, and global `FUSEKIT_TG_SGS`
  no longer overrides `RoutedQmvReduce` or `Single(GatherQMMReduce)`.

Rejected: DeepSeek standalone `GatherQMM` TG=128 default on M5.

- Implementation tested: use op/shape/quant metadata to run only standalone
  generic `Single(GatherQMM)` with 4 SG/TG (128 threads), while keeping
  `GatherQMMReduce` TOP_K-safe at 6 SG/TG (192 threads).
- Coverage: unchanged, 782/782 custom tasks and 0 native fallback tasks.
- Correctness: first-token argmax matched and drift returned to the kept
  baseline, max abs diff 0.03125 and mean abs diff 0.00421025.
- Task graph check: standalone `GatherQMM` changed TG 256 -> 128; reduce
  stayed TG 192.
- Token 511 profile: the candidate run was not a clean targeted win. Sampled
  total GPU time worsened 15364 us -> 20190 us, and unrelated buckets such as
  SDPA and FusedChain also rose, so the profile was dominated by run noise or
  pressure effects rather than an isolated GatherQMM improvement.
- L512 repeat=3 after warmup, same order default then candidate:
  default 73.374 TPS vs candidate 73.608 TPS (+0.3%). This is below the
  acceptance threshold and inside normal run noise.
- Conclusion: reject the standalone TG=128 default and keep TG=256 for
  `GatherQMM`. The correctness guard for reduce override remains.

Kept: generic `GatherQMM` smem-only path when row blocks exactly fill TGs.

- Implementation: when `GatherQMM` x-row threadgroup staging is enabled and
  `ROW_BLOCKS % planned_sgs_per_tg == 0`, emit only the staged-x path. This
  removes the runtime `x_smem_ok` branch and the direct-load fallback body from
  kernels where every launched threadgroup is statically inside one route row.
  Admission is by op shape, quant metadata, and chip launch budget; it does not
  use model names or named MoE fusion kinds.
- Coverage: unchanged, 782/782 custom tasks and 0 native fallback tasks.
- Generated Metal check: candidate artifact had no `x_smem_ok` occurrences;
  the previous artifact had 52 occurrences, matching the 52 standalone
  `GatherQMM` kernels.
- Correctness: first-token argmax matched and drift stayed at the kept
  baseline, max abs diff 0.03125 and mean abs diff 0.00421025.
- Token 511 profile, default then candidate in the same run order:
  total GPU 15930 us -> 15297 us (-4.0%), standalone `GatherQMM` 4400 us ->
  4225 us (-4.0%), `GatherQMMReduce` 2354 us -> 2187 us. The reduce movement
  is likely run coupling/noise because the reduce source did not change, but
  the target `GatherQMM` bucket moved in the intended direction.
- Follow-up launch constantization: the smem-only path now uses compile-time
  `STAGED_SGS_PER_TG`/`STAGED_TG_SIZE` instead of `tg_size / 32` and `tg_size`
  in the staged load loop. Correctness stayed identical; the next token-511
  profile sampled total GPU at 15006 us and standalone `GatherQMM` at 4156 us.
- Follow-up route metadata broadcast: in the same smem-only path, only lane 0
  in each simdgroup loads `row_idx`, `expert_idx`, and optional route scale,
  then `simd_broadcast` distributes them to the lanes. Correctness stayed
  identical. The next token-511 profile sampled standalone `GatherQMM` at
  4031 us; total GPU moved to 15515 us because unrelated SDPA/chain buckets
  rose in that run. Treat this as a target-kernel win, not an independently
  proven e2e win.
- L512 repeat=3 after warmup, same order default then candidate:
  default 67.663 TPS (sd 2.335) vs candidate 68.180 TPS (sd 1.864), +0.8%.
  The wall run was noisy, but it did not contradict the targeted profile win.
- Conclusion: keep as a small kernel-body cleanup because it preserves
  correctness and removes dead branch/code for a general divisible-route
  launch shape. It is not a 10% e2e MoE fix; remaining wins still need deeper
  `GatherQMM` or route/down round-trip work.

Follow-up kept: route metadata broadcast in `GatherQMMReduce`.

- Implementation: in generic `Single(GatherQMMReduce)`, lane 0 of each route
  simdgroup loads `row_idx`, `expert_idx`, and optional `route_weight`, then
  distributes them with `simd_broadcast`. This mirrors the kept standalone
  `GatherQMM` metadata broadcast and is based only on op metadata and TOP_K
  route shape.
- Correctness: first-token argmax/top5 matched. Sampled max abs diff stayed
  0.03125; mean abs diff in the checked run was 0.00545215.
- Token 511 same-condition profile versus the prior no-barrier artifact:
  total GPU 17145 us -> 15245 us, `GatherQMM` 4510 us -> 4289 us, and
  `GatherQMMReduce output-scale` 2707 us -> 2231 us. The absolute numbers are
  noisy, but the targeted reduce bucket moved in the intended direction.
- L512 repeat=3, same order default then candidate: default wall TPS 73.08 vs
  candidate 74.02 (+1.3%); excluding the first run, 74.05 vs 75.60 (+2.1%).
- Decision: keep. This is an op-internal metadata load cleanup, not a named
  MoE pattern, and fallback remains 0/782.

Rejected: generic `GatherQMM pair + SwiGLU tail` default.

- Implementation tested: enable the existing general fanout merger that fuses
  two compatible `GatherQMM` siblings plus a `Sigmoid -> Multiply -> Multiply`
  tail into one `FusedChain`. Admission is by shared activation row, equivalent
  route row index, same expert index, compatible quant/shape, and collected
  pointwise formula. It is not a model-name rule and does not emit named
  `RoutedQmv*` composites.
- Coverage: tasks dropped 782 -> 730 and no fallback appeared.
- Correctness: pair helper default kept argmax/top5 but worsened max abs diff
  to 0.046875. Disabling pair qdot restored max abs diff to 0.03125, but mean
  diff was still higher at 0.00581515 because the fused tail no longer matches
  the separated intermediate materialization exactly.
- Token 511 profile: total GPU improved around 15.2 ms -> 14.6-15.0 ms and
  kernel count dropped 455 -> 403, so the structural idea is real.
- L512 repeat=3 did not preserve the profile win. Pair variants measured below
  the current candidate in wall TPS, and the no-qdot/exact variants were
  unstable across repeats.
- Decision: keep this merger opt-in only. To default it later, the fused tail
  needs exact materialization semantics or a longer-sequence correctness check
  showing no route/token drift.

Rejected: `GatherQMMReduce` half threadgroup partials.

- Implementation tested: store plain `GatherQMMReduce` route partials in
  `threadgroup half` when the output dtype is half, then convert to float for
  the final route reduction. This reduces smem traffic and mirrors the
  materialized dtype more closely than a float scratch.
- Correctness: first-token argmax/top5 matched and max abs diff stayed
  0.03125; mean abs diff stayed at the current candidate level in the checked
  run.
- Token 511 profile improved: total GPU 15245 us -> 14806 us and
  `GatherQMMReduce output-scale` 2231 us -> 2121 us.
- L512 same-run wall did not hold. Current candidate measured 73.15 TPS
  average, 74.62 TPS excluding first run; half-partial measured 72.38 TPS
  average, 73.11 TPS excluding first run, with one low run at 70.93 TPS.
- Decision: reject for default despite the kernel win. It is a good follow-up
  only if a later change explains and removes the wall instability.

Rejected: active-token-1 route index specialization.

- Implementation tested: for decode artifacts where `ACTIVE_TOKENS == 1`,
  simplify generic `GatherQMM` and `GatherQMMReduce` route math at emit time:
  `tok = 0`, `route_idx = route/slot`, and `o_base = gid * OUT_COLS` in the
  applicable reduce path. Multi-token/MTP/prefill shapes kept the old indexing.
- Correctness: first-token argmax/top5 matched; max abs diff and mean abs diff
  stayed at the current candidate level.
- Emitted Metal check: standalone `GatherQMM` no longer emitted
  `route % TOP_K` / `(route / TOP_K) % TOKENS` in the active-token-1 staged
  path, and plain reduce emitted `route_idx = slot`.
- Token 511 profile was not a clean win: standalone `GatherQMM` moved down
  slightly, 4289 us -> 4190 us in the sampled run, but
  `GatherQMMReduce output-scale` and unrelated chain/SDPA buckets rose enough
  to push total sampled GPU time up to 15582 us.
- L512 repeat=3 measured 72.26 TPS average, 73.74 TPS excluding the first run,
  below the kept candidate's recent 73.15/74.62 TPS comparison.
- Decision: reject and remove. The compiler/runtime pressure change is not
  worth the tiny arithmetic simplification.

Correction: route `GatherQMM` op-sequence candidates are not default general fusion.

- Context: the pass must not depend on "turn this pattern on when conditions
  match". The default general path should scan the incoming graph for legal
  candidates, then accept or reject by graph/dataflow legality and a cost model.
- Implementation correction: `GatherQMM pair + SwiGLU tail` and
  `GatherQMMReduce -> LinearResidual` are op-sequence candidate families. They
  are not model-name patterns, but they are still pattern-like under the
  stricter "no named op-sequence templates in default general" rule. They are
  therefore gated behind `FUSEKIT_EXPERIMENTAL_OP_SEQUENCE_FUSIONS=1` and are
  not part of the default general path.
- Cost model: saved activation round-trip bytes plus a small launch credit are
  compared against a qMV/GatherQMM parallelism penalty derived from quantized
  weight stream size. This is deliberately conservative: it lets tiny sinks
  merge, but rejects legal merges that serialize large routed projections.
- DeepSeek-Coder-V2-Lite 4-bit, `--patterns off`, debug compile:
  26 `GatherQMMPairPointwiseTail` candidates were found and rejected
  (`saved=675840B`, `penalty=811008B`, `score=-102400`). 26
  `GatherQMMReduceLinearResidual` candidates were found and rejected
  (`saved=65536B`, `penalty=202752B`, `score=-104448`).
- Coverage after rejection: 782/782 custom tasks, 0 native fallback tasks,
  fused/specialized 245/782, source ops fused 1023/1660. This matches the
  previously kept non-regressing DeepSeek route.
- Correctness against raw MLX/oMLX: argmax matched (`185`), top5 matched,
  dispatcher built, no dispatch failure. Sample drift:
  max abs diff 0.04296875, mean abs diff 0.00648432.
- L512 repeat=3, raw first then FuseKit, same prompt:
  raw wall TPS = 69.684, 72.914, 72.495 (avg 71.698);
  FuseKit wall TPS = 71.990, 74.812, 74.647 (avg 73.816), +2.95%.
  Excluding first runs: raw 72.705 vs FuseKit 74.730, +2.79%.
- Per-kernel profile sample (`--per-kernel`, L128 run): total GPU 13649 us,
  with remaining buckets `GatherQMM` 4122 us, chain 2215 us,
  `GatherQMMReduce output-scale` 2098 us, LinearResidual 1664 us,
  QmvSwiglu 1461 us. The accepted/default MoE work remains the routed
  qMV/GatherQMM kernel bodies and route/down memory traffic, not named MoE
  composites or op-sequence fusion templates.
- Follow-up after moving the two op-sequence families behind
  `FUSEKIT_EXPERIMENTAL_OP_SEQUENCE_FUSIONS=1`: recompiled the installed
  DeepSeek artifact with `--patterns off`. Coverage stayed 782/782 custom
  tasks, 0 fallback, 245/782 fused/specialized. L512 repeat=3, raw first then
  FuseKit, same prompt:
  raw wall TPS = 62.552, 66.488, 68.949 (avg 65.996);
  FuseKit wall TPS = 67.165, 72.077, 72.208 (avg 70.483), +6.8%.
  Excluding first runs: raw 67.719 vs FuseKit 72.143, +6.5%.
- Repeat=5 recheck, same artifact and prompt, raw first then FuseKit:
  raw wall TPS = 63.485, 70.544, 70.862, 70.916, 70.729 (avg 69.307);
  FuseKit wall TPS = 69.232, 73.672, 73.965, 73.865, 73.630 (avg 72.873),
  +5.1%. Excluding first runs: raw 70.763 vs FuseKit 73.783, +4.3%.
- Repeat=5 recheck on 2026-06-23 after the op-sequence gate, raw first then
  FuseKit, same prompt:

| Model | Raw TPS avg | FuseKit TPS avg | Delta | Raw TPS avg excl. first | FuseKit TPS avg excl. first | Delta excl. first | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | 70.681 | 72.892 | +3.1% | 71.535 | 73.726 | +3.1% | 9.102 GB | 9.245 GB |
| `mlx-community/Qwen3.5-4B-4bit` | 45.144 | 46.435 | +2.9% | 45.763 | 46.520 | +1.7% | 2.489 GB | 2.496 GB |
| `mlx-community/Qwen3.5-4B-8bit` | 25.464 | 26.081 | +2.4% | 25.540 | 26.095 | +2.2% | 4.586 GB | 4.597 GB |
| `mlx-community/Qwen3.5-4B-bf16` | 14.055 | 15.075 | +7.3% | 14.101 | 15.095 | +7.1% | 8.503 GB | 8.539 GB |

Gemma4 support and current regression state.

- Problem: Gemma4 bf16 exposed two runtime gaps that the patch coverage table
  did not catch: `BatchRotatingKVCache` offset/write-offset semantics and
  anonymous token-id externals feeding dense embedding `Gather` tasks. The
  q4/q8 artifacts avoided the second issue through quantized embed lowering,
  but bf16 kept the raw `Gather(embed_tokens.weight, input_ids)` graph.
- Fix: bind `BatchRotatingKVCache` cache uniforms per layer, keep RoPE absolute
  offset separate from physical cache write offset, and bind anonymous
  int32/uint32 embedding `Gather` indices to the current token-id uniform. For
  single-token decode with a wider traced token buffer, seed the wider uniform
  by repeating the current token so all traced rows are readable.
- Fix: dense `Matmul`-headed `FusedChain` with M>=8 now has a shape-based
  simdgroup GEMM lowering. It supports both trace `[K,N]` and original MLX
  transposed-weight `[N,K]` layouts by using Metal's `simdgroup_load(...,
  transpose_matrix=true)`; no weight repack or model-name condition is used.
- q4 profile before this fix had Gemma per-layer projection
  `Matmul -> Unflatten -> Multiply -> Unflatten` at 18.9 ms in one decode
  step. After the simdgroup lowering it dropped to 0.67 ms, and total sampled
  GPU time dropped from 48.5 ms to 29.8 ms.
- Patch coverage after the fixes, `--patterns off`:
  q4/q8 `1260/1260` custom tasks, 0 fallback, 509 fused/specialized;
  bf16 `1452/1452` custom tasks, 0 fallback, 718 fused/specialized.
- Correctness spot checks, forced strict dispatch, same prompt/decode token:
  q4/q8/bf16 all built a dispatcher, had `dispatch_failed=false`, and matched
  raw argmax token `6974`. Logit drift remained bounded but nonzero:
  q4 max/mean `0.75/0.113`, q8 `0.625/0.234`, bf16 `0.625/0.190`.
- L512 single-run benchmark, raw first then FuseKit, strict dispatch enabled:

| Model | Raw TPS | FuseKit TPS | Delta | Raw wall | FuseKit wall | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | 38.194 | 35.606 | -6.8% | 13.405 s | 14.379 s | 4.288 GB | 4.335 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 21.891 | 21.456 | -2.0% | 23.389 s | 23.862 s | 8.005 GB | 8.053 GB |
| `mlx-community/gemma-4-e4b-it-bf16` | 11.772 | 9.796 | -16.8% | 43.492 s | 52.265 s | 14.968 GB | 14.981 GB |

- Current diagnosis: unsupported/fallback is fixed; performance is not. bf16
  profile is dominated by dense chains: reduction-fanout
  `RMSNorm -> Matmul -> Matmul` at 34.5 ms, MLP down
  `M=8,K=10240,N=2560` simdgroup GEMM at 24.5 ms, final LM head at 10.3 ms.
  The next work is q4/q8 qMV/LM-head and bf16 dense-chain microkernel quality,
  not more fallback plumbing.

Follow-up, general fusion only, model-independent target/shape rules.

- Rejected and removed: scalar `Power` fast path in `FusedChain` and
  `PointwiseChain`. It was not a clean correctness win and left ad-hoc
  exponent branches in codegen.
- Rejected/restored: dense bf16 K8 unroll disable, dense `M=2` GEMV rows,
  q4 fanout SG=4. Each either regressed late-token profile or did not survive
  same-run e2e.
- Accepted: final RMSNorm+LM-head target threadgroup count now uses one
  shared target/shape/quant helper for both emitter and dispatch grid. M5
  dense large-vocab heads use 256 slices; M5 q4/q8 4B-class heads use 64
  slices. No model-name condition.
- Accepted: standalone q4 qMV uses four simdgroups only when the target,
  active rows, K dimension, output dimension, and column divisibility make it
  valid. This is still shape/quant based.
- Repatched Gemma4 q4/q8/bf16 with `--patterns off`. Coverage:
  q4/q8 `1001/1001` custom tasks, 0 fallback, 426 fused/specialized;
  bf16 `1067/1067` custom tasks, 0 fallback, 593 fused/specialized.
- Correctness spot checks, forced strict dispatch: q4/q8/bf16 all matched raw
  argmax token `6974` with `dispatch_failed=false`.

L512 repeat=3, raw oMLX first, same prompt:

| Model | Raw TPS avg | FuseKit TPS avg | Delta | Raw TPS sd | FuseKit TPS sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | 41.914 | 43.781 | +4.5% | 0.133 | 0.024 | 4.288 GB | 4.360 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 23.625 | 25.169 | +6.5% | 0.065 | 0.010 | 8.006 GB | 8.077 GB |
| `mlx-community/gemma-4-e4b-it-bf16` | 12.970 | 12.858 | -0.9% | 0.019 | 0.045 | 14.968 GB | 15.002 GB |

Interpretation:

- q4/q8 L512 regressions are fixed in the current default general-fusion path.
- bf16 improved from the earlier `-16.8%` single-run state to near parity, but
  remains slightly negative. The remaining work is dense-chain and
  LinearResidual kernel quality, not fallback or LM-head dispatch.
- Rejected follow-up: `FUSEKIT_DENSE_GEMV_ROWS=2` on Gemma4 bf16. Correctness
  matched (`same_argmax=true`), but token-499 profile regressed: total GPU
  `73.4 ms -> 80.5/83.5 ms`, dense-pair `34.0 ms -> 37.5/35.9 ms`, MLP down
  `17.6 ms -> 18.4/17.8 ms`, and e2e L500 was `12.815 TPS`. The installed
  artifact was restored to the default general path.
- Rejected follow-up: dense-pair gate/up joint K-loop helper. It is a
  general semantic optimization, but on M5 it increased register pressure
  enough to lose e2e. Correctness matched, but L512 repeat=3 regressed:
  raw `12.957 TPS` vs FuseKit `12.652 TPS` (`-2.4%`). The helper was removed
  and the installed bf16 artifact was restored.
- Raw synchronized MLX op profile at late token (`warmup=499`, `tokens=1`)
  shows raw MLP linears are not the reason FuseKit is slower: raw grouped
  `gate/up/down` totals were roughly `27.8/26.6/28.0 ms`, while FuseKit's
  fused dense-pair plus MLP-down buckets are roughly `34.0 + 17.6 ms`.
  The remaining bf16 work is therefore SDPA single-op lowering and runtime
  scheduling/noise, not another gate/up/down rows-per-SG tweak.

Additional rejected SDPA candidates:

- Rejected: materialized SDPA softmax `exp -> metal::fast::exp`. It compiled
  and preserved dispatch coverage, but L512 repeat=3 stayed negative:
  raw `12.984 TPS` vs FuseKit `12.872 TPS` (`-0.9%`). No default change kept.
- Rejected: extend forced `OnlineVector` SDPA to `head_dim=256` by making the
  vector path use `D_PER_LANE` registers instead of fixed four registers.
  Correctness matched raw argmax (`6974`), and late-token SDPA itself dropped
  to roughly `0.59 ms` for 35 calls. However L512 repeat=3 still lost:
  raw `12.956 TPS` vs FuseKit `12.884 TPS` (`-0.6%`). The code and installed
  artifact were restored to the default materialized SDPA path.
- Interpretation: SDPA was a visible single-op inefficiency, but cutting it
  is not sufficient for bf16 L512. The dominant and actionable buckets remain
  dense `FusedChain` weight streaming: `dense-pair-pointwise-tail` and the
  following MLP down projection.

Accepted follow-up: Single SDPA decode lowering, general fusion only.

- Change: `Single(ScaledDotProductAttention)` ops that are already rebound to
  semantic Q/K/V cache buffers and have `active_m == 1` now emit through the
  `SdpaDecode` flash-decoding implementation. Admission is by op kind, output
  shape, QK/V dimensions, and target flash-decoding capability; no model name
  or pattern matcher path is used. The task remains `Single(...)` in IR, but
  the MSL body uses the decode online/partial combine path and keeps the
  original Single kernel symbol for task-graph compatibility.
- Correctness issue found and fixed: first attempt emitted the right MSL body
  but returned the simdgroup count as an actual threadgroup size, giving
  `threadgroup=[8,1,1]` for `V_DIM=256`. That wrote only a small prefix of the
  output and collapsed logits. Fixed by returning SG count through the normal
  `kernel_threadgroup_for` path so it becomes `threadgroup=[256,1,1]`.
- Correctness spot checks, strict dispatch, same prompt:
  q4/q8/bf16 all matched raw argmax token `6974`, with
  `dispatch_failed=false`.
- Profile evidence, Gemma4 bf16 L512 token 499:
  `Single(ScaledDotProductAttention)` family dropped from about `7.47 ms` for
  35 calls to about `1.38 ms`; total profiled GPU time dropped from about
  `80.8 ms` to `74.8 ms`. Remaining dominant buckets are still dense MLP:
  `dense-pair-pointwise-tail` about `34.0 ms`, MLP down about `17.6 ms`,
  final LM head about `5.0 ms`.

L512 repeat=3 after Single-SDPA decode lowering, raw oMLX first:

| Model | Raw TPS avg | FuseKit TPS avg | Delta | Raw TPS sd | FuseKit TPS sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | 42.048 | 43.858 | +4.3% | 0.096 | 0.038 | 4.288 GB | 4.371 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 23.768 | 25.454 | +7.1% | 0.054 | 0.016 | 8.006 GB | 8.088 GB |
| `mlx-community/gemma-4-e4b-it-bf16` | 13.086 | 13.119 | +0.2% | 0.015 | 0.106 | 14.968 GB | 15.013 GB |

Interpretation:

- The SDPA lowering is correct and reduces GPU kernel time, but bf16 L512 wall
  gain is small because the remaining dense MLP chains dominate.
- q4/q8 retain meaningful L512 wins and do not regress under the same
  general-fusion-only path.

Rejected follow-up: generic gate-activation-up absorb for Gemma4 q4 GELU MLP.

- Motivation: Gemma4 q4 has 42 layer MLP tails shaped as
  `Power -> Multiply -> Add -> Multiply -> Tanh -> Add -> Multiply -> Multiply -> Multiply`
  after a `RMSNorm -> QuantizedMatmul -> QuantizedMatmul` gate/up fanout. This
  is a semantic tanh-GELU gate, not SwiGLU, so the existing Silu-only tail
  lowering did not apply.
- Implementation tried: generalized the pointwise-tail recognizer to classify
  final `activation(gate) * up` chains by dataflow role, carried scalar
  consumer inputs into the merged FusedChain, and let the quantized fanout
  emitter render the captured pointwise tail in registers. This reduced Gemma4
  q4 tasks from `1001` to `959` and produced 42
  `quant-pair-pointwise-tail` kernels. The change is graph-semantic, not
  model-name based.
- Correctness: q4 strict dispatch matched raw argmax token `6974`.
- Profile result: despite 42 fewer dispatches, late-token q4 profile worsened.
  The merged `quant-pair-pointwise-tail` bucket was about `11.6 ms` for 42
  calls, versus the previous split path's `reduction-fanout` bucket around
  `10.5 ms` plus cheaper separate tail work. `metal::fast::tanh` inside the
  merged tail kept correctness but did not recover the loss.
- E2E result, L512 repeat=3 raw first in the same run: raw `39.120 TPS`,
  FuseKit `39.779 TPS` (`+1.7%`). This is lower than the current default
  q4 result (`+4.3%`), so the GELU-tail absorb is not enabled by default.
  The code path remains opt-in via `FUSEKIT_GATE_ACTIVATION_UP_ABSORB=1` for
  future work, and the installed q4 artifact was restored to `1001` tasks.

Follow-up validation: scalar constants in fusion boundaries.

- Change kept: `FusionBoundary` now carries `scalar_value` from the source
  graph tensor, and task JSON prefers the boundary scalar over a graph lookup.
  This is a general IR fix for merged kernels whose scalar constants are
  remapped into a new input list.
- Change kept: the fanout-tail pointwise renderer lowers `Power(x, 2)` and
  `Power(x, 3)` to explicit multiply chains when the exponent operand is a
  captured scalar constant. Otherwise it falls back to `pow(x, y)`.
- Re-test with `FUSEKIT_GATE_ACTIVATION_UP_ABSORB=1` on
  `mlx-community/gemma-4-e4b-it-4bit`: correctness still matched raw argmax
  token `6974`, strict dispatch, fallback 0, artifact `959` tasks.
- Profile re-test, token 499, L512-ish run:
  default split path total GPU `28.52 ms`, `reduction-fanout` bucket
  `11.23 ms`; merged GELU-tail path total GPU `29.30 ms`,
  `quant-pair-pointwise-tail` bucket `11.75 ms`.
- E2E re-test in that run also moved the wrong way:
  default wall TPS `35.66`, merged wall TPS `34.89`.
- Decision: scalar propagation and constant-power lowering are retained as
  general infrastructure, but GELU-tail absorb remains guarded off. The
  installed Gemma4 q4 artifact was restored to the default `1001` task path.

Accepted follow-up: rotating-cache tail fallback and M5 q4 qMV qdot default.

- Problem found while profiling L512: Gemma4 uses `RotatingKVCache` with a
  fixed physical `max_size=512`. The fast generate path was checking whether
  the entire remaining generation fit in physical KV capacity. With a 12-token
  prompt and L512 generation, it failed immediately after bootstrap
  (`offset=13`, `remaining=511`, `capacity=512`) and fell back to mlx-lm for
  almost the whole run. The fallback edge also passed an empty prompt to
  mlx-lm and crashed on recent mlx-lm.
- Runtime fix: fast generate now pre-sizes the prompt cache request from
  `prompt_len + max_tokens`, but for fixed-size rotating caches it no longer
  requires the whole generation to fit. It runs FuseKit chunks until physical
  capacity is reached, then yields the already-computed next token and
  continues the small tail with mlx-lm using the live `prompt_cache`. This is a
  cache semantic/runtime fix, not a model-name rule.
- Immediate Gemma4 q4 L512 effect with the same prompt:
  the broken immediate-fallback path measured about `33.2 TPS`; after the
  chunk-until-capacity fix it measured `40.8 TPS` in a single run.
- Same-run raw-first repeat=3 after the runtime fix:
  raw oMLX `39.041, 40.092, 39.663 TPS` (avg `39.60 TPS`);
  FuseKit default before qdot `39.423, 40.824, 40.380 TPS` (avg `40.21 TPS`,
  `+1.5%` vs raw).
- q4 qMV qdot candidate: compiling with `FUSEKIT_QMV_FAST_QDOT_Q4=1` preserved
  strict-dispatch correctness on Gemma4 q4 (`raw_argmax = patched_argmax =
  6974`). Token-498 profile improved from total GPU `29.26 ms` to `26.72 ms`,
  with `reduction-fanout` `11.30 -> 10.44 ms` and `Single(qmv_fast)`
  `7.42 -> 7.14 ms`.
- L512 repeat=3 for the qdot candidate:
  `40.964, 41.986, 41.724 TPS` (avg `41.56 TPS`), which is `+3.4%` over the
  pre-qdot FuseKit artifact and `+5.0%` over raw oMLX in the same run set.
- Revalidation after later drift debugging: do **not** make q4 qMV qdot the
  default yet. Forcing `FUSEKIT_QMV_FAST_QDOT_Q4=1` on
  `mlx-community/gemma-4-e4b-it-4bit` with prompt `x` mismatched the
  forced-prefix compare at step 128 (`raw_next=236781`, `fk_next=1205`) and
  the same L512 run measured raw `39.350 TPS` vs FuseKit `33.854 TPS`.
  The source default remains q4 qdot off; this preserves the general
  quant/shape path without accepting a near-tie drift regression.

Rejected follow-up: q4 qparam broadcast.

- Test: build with `FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q4=1` on Gemma4 q4.
- Correctness: forced-prefix `x`/token `236781` matched top1 through 150
  checked steps, but logit differences remained visible.
- Profile: token-448 total GPU `27.96 ms`, reduction-fanout `10.03 ms`,
  standalone qMV `7.18 ms`, direct head argmax `3.16 ms`.
- E2E: same run raw `40.635 TPS`, FuseKit `34.811 TPS` (`-14.3%`).
- Decision: reject as default. It improves one correctness probe but does not
  improve wall time or late-token GPU time enough to justify carrying it.

Rejected follow-up: 512-wide flash-decoding SDPA.

- Motivation: seven Gemma4 q4 decode attention layers are
  `qk_dim=512, v_dim=512` and currently fall back to the legacy SDPA path
  because flash-decoding only supports `v_dim <= 256`.
- Implementation tried: generalize flash-decoding so each thread owns two
  output columns (`tid` and `tid + tg_size`) and gate capability at
  `v_dim <= 512`.
- Correctness: forced-prefix matched until the first multi-chunk boundary, then
  failed at step 127 with all top logits collapsed to `-30.0`. This points at
  the existing multi-TG partial combine/visibility contract, not a model-name
  issue.
- Attempted atomic ordering fix: Metal device atomics in this toolchain only
  compile with `memory_order_relaxed`; `memory_order_acq_rel` is unavailable
  and `memory_order_seq_cst` is rejected for device atomics.
- Decision: revert the 512-wide flash candidate. A correct version needs a
  different combine design, not just a wider V loop.

Accepted follow-up: split single-SDPA decode lowering from flash chunking.

- Root cause: after the multi-chunk SDPA correctness fix, flash-decoding kept
  the `n_heads * MAX_CHUNKS` dispatch shape but only `chunk_id == 0` performed
  the online softmax for `active_chunks > 1`. This preserved correctness but
  left the flash parallelism disabled while still paying chunk over-spawn and,
  for single `ScaledDotProductAttention`, risked falling back to the
  materialized-score prefill emitter if flash was disabled outright.
- Fix: split the predicates. `sdpa_single_decode_enabled` now means the op is
  a semantic decode SDPA and should lower through the decode emitter.
  `sdpa_single_decode_flash_enabled` only controls the unsafe chunked partial
  path and its partial/counter bindings. The default flash gate is false until
  cross-threadgroup partial combine has a safe acquire/release design; decode
  still uses the one-TG-per-head online emitter.
- Correctness: Gemma4 q4 forced-prefix compare with prompt `x` and token
  `236781` repeated 150 steps matched raw top1 through the 128-token boundary.
- Profile, Gemma4 q4 L512 token 448:
  prior safe-flash state total GPU `32.62 ms`, `op:scaleddotproductattention`
  `4.23 ms`; fixed decode-online lowering total GPU `29.31 ms`,
  `op:scaleddotproductattention` `2.18 ms`.
- E2E raw-first repeat=3, Gemma4 q4 L512, prompt `x`:
  raw oMLX avg `39.36 TPS`, FuseKit avg `33.28 TPS` (`-15.5%`). This is still
  slower than raw, but it removes a real SDPA regression and narrows the
  FuseKit long-context gap without changing model-specific behavior.

Rejected follow-up: reduction-fanout pair2 and multirow unroll2.

- Pair2 idea: for two sibling quantized matmuls after a shared RMSNorm, compute
  both output columns in one simdgroup to reuse the normalized row. It was
  implemented behind `FUSEKIT_REDUCTION_FANOUT_PAIR2=1` using only graph
  shape/dataflow checks.
- Pair2 correctness: Gemma4 q4 forced-prefix compare matched through 150 steps.
- Pair2 profile: token 448 total GPU worsened `30.82 -> 34.59 ms`;
  `reduction-fanout` worsened `10.11 -> 11.51 ms`. The added accumulator/live
  range pressure outweighed the activation reuse. The trial code was removed.
- Multirow unroll2 idea: enable the existing two-word unroll for `ROWS=8`
  reduction-fanout qGEMV.
- Multirow unroll2 profile: target bucket improved only
  `10.83 -> 10.51 ms`, but total GPU was effectively worse/noisy
  `32.62 -> 32.84 ms`.
- Multirow unroll2 E2E raw-first repeat=3: raw `39.50 TPS`, FuseKit
  `33.05 TPS` (`-16.3%`). Not promoted.

Accepted follow-up: SDPA flash-decoding drift fix without disabling the fast path.

- Root cause: the previous flash-decoding implementation did per-chunk SDPA
  partial writes and cross-threadgroup combine inside one Metal kernel, using a
  relaxed device atomic counter as the only inter-TG rendezvous. Metal accepted
  only `memory_order_relaxed` for this device atomic path, so the combiner could
  observe the counter before all chunk partial writes were visible. The symptom
  was forced-prefix drift at the first multi-chunk boundary around step 128.
- Fix: keep the general shape-gated flash path, but emit two ordered kernels per
  SDPA decode fusion: `<kernel>_partial` writes `(m, l, o[V])` per
  `(layer,row,head,chunk)` into global scratch, and `<kernel>_combine` reads all
  active chunks and writes the SDPA output. The command/ICB ordering between the
  two dispatches provides the visibility boundary; no model-name or pattern
  match is involved.
- Scope: applies to semantic `SdpaDecode` and single-op
  `ScaledDotProductAttention` decode lowering when `v_dim <= 256`. Existing
  `v_dim=512` layers still stay on the safe non-split decode path until the
  emitter supports multiple V columns per thread without changing the numeric
  contract.
- Correctness: Gemma4 q4 forced-prefix compare with prompt `x` and token
  `236781` repeated 150 steps matched raw top1 through and past the previous
  step-128 failure point.
- Profile, Gemma4 q4 L512 token 448: safe decode-online total GPU `29.31 ms`;
  two-dispatch flash total GPU `28.62 ms`. SDPA-related time is no longer the
  dominant gap; `FusedChain`, standalone qMV, and LM-head remain dominant.
- E2E raw-first repeat=3, Gemma4 q4 L512, prompt `x`: raw oMLX avg
  `41.031 TPS`, FuseKit avg `35.510 TPS` (`-13.5%`). This recovers part of the
  safe fallback regression (`33.28 TPS -> 35.51 TPS`) while preserving
  correctness, but it is not yet a net win over raw.

## 2026-06-28 Gemma4 L512 5% Target Recheck

Scope: Gemma4 E4B q4/q8/bf16, L512, prompt `x`, general fusion only
(`--patterns off`), raw oMLX first, repeat=2 after one warmup per path.

Same-run result:

| Model | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | 40.571 | 39.864 | -1.7% | 4.287 GB | 4.335 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 22.800 | 22.853 | +0.2% | 8.005 GB | 8.053 GB |
| `mlx-community/gemma-4-e4b-it-bf16` | 12.532 | 12.169 | -2.9% | 14.967 GB | 15.010 GB |

Late-token profile summary:

| Model | Total GPU | Top buckets |
| --- | ---: | --- |
| q4 default | 28.05 ms | reduction-fanout 10.26 ms; Single qMV 7.13 ms; LM-head 3.02 ms |
| q8 default | 46.61 ms | reduction-fanout 19.08 ms; Single qMV 13.12 ms; LM-head 5.52 ms |
| bf16 default | 83.23 ms | dense-pair tail 35.23 ms; down projection 18.21 ms; LM-head 10.55 ms |

Rejected candidates:

| Candidate | Result | Decision |
| --- | --- | --- |
| q4 `FUSEKIT_QMV_FAST_QDOT_Q4=1` | forced-prefix top1 matched 150 steps, but L512 raw `40.445` TPS vs FuseKit `40.036` TPS; profile total `28.17` ms vs default `28.05` ms | reject |
| q8 `FUSEKIT_QMV_FAST_QPARAM_BROADCAST_Q8=1` | fanout changed `19.08 -> 19.01` ms, but Single qMV and LM-head worsened; total `46.61 -> 46.71` ms | reject |
| q4 Single qMV `FUSEKIT_QMM_MLX_QMV_FAST_SGS=2` | Single qMV `7.13 -> 7.06` ms, but wall single-run worsened and total only `28.05 -> 27.95` ms | reject |
| q4 LM-head `FUSEKIT_FINAL_LM_HEAD_TGS=64` | runtime/emit must share the same env; with matching env LM-head worsened `3.02 -> 3.28` ms | reject |
| bf16 LM-head `FUSEKIT_FINAL_LM_HEAD_TGS=128` | LM-head `10.55 -> 10.32` ms, but dense/down buckets offset it; total unchanged | reject |
| bf16 LM-head `FUSEKIT_FINAL_LM_HEAD_TGS=512` | LM-head `10.55 -> 10.44` ms and dense-pair worsened; total `83.23 -> 83.55` ms | reject |

Conclusion: no measured 5% win is currently secured on Gemma4 E4B L512. The
remaining path to a real win is not another launch knob; it needs a microkernel
body improvement in the dominant weight-streaming kernels:

- q4/q8: `FusedChain reduction-fanout` and `Single(QuantizedMatmul MLX qmv_fast)`.
- bf16: dense pair-tail MLP and down-projection `heavy-first` kernels.

All temporary artifacts from this recheck were restored to the default
general-fusion build after rejection.

### Accepted q8 LM-head Slice Policy

Scope: `mlx-community/gemma-4-e4b-it-8bit`, L512, prompt `x`, general fusion
only (`--patterns off`), M5.

Change:

- `FinalRmsNormLmHead::target_tgs` now uses 64 threadgroup slices for q8
  mid-hidden large-vocab heads on M5 (`bits=8`, hidden in `2048..=4096`).
- This is shape/quant/target based, not a model-name or pattern-matcher rule.
  It keeps the same direct argmax kernel and changes only vocab scan
  parallelism for the LM-head.

Profile evidence, token 448:

| Variant | Total GPU | LM-head |
| --- | ---: | ---: |
| q8 default `n_tgs=32` | 47.22 ms | 5.38 ms |
| q8 `n_tgs=64` | 44.60 ms | 2.95 ms |

Correctness:

- Forced-prefix compare with prompt `x` and token `236781` repeated 150 steps:
  all raw/FuseKit top1 tokens matched.

Raw-first L512 repeat=3:

| Model | Raw TPS avg | FuseKit TPS avg | Delta | Raw TPS sd | FuseKit TPS sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-8bit` | 22.209 | 23.882 | +7.5% | 0.034 | 0.008 | 8.005 GB | 8.051 GB |

Rejected while searching:

- q4/q8 `FUSEKIT_QGEMM_OUT_COLS=2` did not hold as a wall win. q8 worsened
  total GPU (`47.22 -> 47.88 ms`); q4 reduced one bucket but not enough for
  e2e.
- q4 LM-head qdot forced mode changed argmax on the forced-prefix probe and
  was removed.

### Gemma4 q4/bf16 Follow-up Recheck

Scope: Gemma4 E4B, L512, prompt `x`, general fusion only (`--patterns off`),
raw oMLX first. Artifacts were repatched after each rejected experiment so the
installed q4/q8/bf16 models end in the default general-fusion state.

Baseline after repatching q4/q8/bf16 to current source:

| Model | Raw TPS avg | FuseKit TPS avg | Delta | Raw TPS sd | FuseKit TPS sd | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | 41.435 | 40.723 | -1.7% | 0.085 | 0.027 | 4.287 GB | 4.335 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 23.348 | 23.018 | -1.4% | 0.087 | 0.014 | 8.005 GB | 8.052 GB |
| `mlx-community/gemma-4-e4b-it-bf16` | 12.841 | 12.464 | -2.9% | 0.017 | 0.043 | 14.967 GB | 15.019 GB |

Late-token profile at token 448 with engine chunk disabled for single-step
kernel ranking:

| Model | Total GPU | Dominant buckets |
| --- | ---: | --- |
| q4 default | 26.21 ms | reduction-fanout 9.87 ms; Single qMV 6.89 ms; LM-head 2.98 ms |
| q8 default | 46.01 ms | reduction-fanout 18.67 ms; Single qMV 12.85 ms; LM-head 6.71 ms |
| bf16 default | 80.40 ms | dense-pair tail 34.71 ms; down projection 18.06 ms; LM-head 10.06 ms |

Rejected in this follow-up:

| Candidate | Correctness | Profile / wall result | Decision |
| --- | --- | --- | --- |
| q4 `FUSEKIT_QMV_FAST_QDOT_Q4=1` | forced-prefix top1 matched 150 steps | same-run L512 raw `41.368` TPS vs FuseKit `40.842` TPS (`-1.3%`) | reject and repatch q4 default |
| q8 `FUSEKIT_QMV_FAST_PACKS_Q8=4` | forced-prefix passed | total GPU `46.01 -> 45.51` ms, LM-head `6.71 -> 6.12` ms, qMV buckets unchanged enough to matter | reject as default |
| final-head accumulator-domain argmax compare | forced-prefix passed on q8 | LM-head `6.71 -> 6.60` ms, total essentially unchanged | reject; restore rounded compare |
| bf16 `FUSEKIT_FINAL_LM_HEAD_TGS=512` | not needed; profile-only | LM-head `10.06 -> 10.31` ms, total unchanged | reject and repatch bf16 default |
| bf16 fused-tail `metal::fast::tanh` | forced-prefix passed | profile total `80.40 -> 79.89` ms, but L512 raw `12.852` TPS vs FuseKit `12.486` TPS (`-2.8%`) | reject; restore precise tanh |

Current conclusion: q4/bf16 are not fixed by scalar knobs or small math
substitutions. The remaining profile-grounded work is a structural microkernel
change:

- q4/q8: reduce memory/load cost in `FusedChain reduction-fanout` and
  `Single(QuantizedMatmul MLX qmv_fast)` without changing accumulation enough
  to perturb forced-prefix logits.
- bf16: fuse or retile the dense MLP pair-tail plus following down projection,
  or rewrite those dense GEMV bodies; LM-head and SDPA are secondary at this
  point.

Follow-up candidate: q4 qMV MLX-order accumulation without global qdot.

- Motivation: the only Gemma4 q4 candidate that reached roughly `+5%` was
  global `FUSEKIT_QMV_FAST_QDOT_Q4=1`, but it was not safe enough to default
  because qdot-style regrouping previously caused near-tie greedy drift. The
  default q4 explicit path kept a 32-bit packed load but accumulated eight q4
  values in one expression, which still differs from MLX qdot's four-value
  grouping.
- Change: q4 explicit qMV helpers now keep the 32-bit packed-word load and
  accumulate as two MLX-order four-value groups. The same primitive rule is
  applied to standalone qMV, fanout qMV, multi-row qMV, and threadgroup
  partial helpers. This is quant/shape primitive code, not a model-name or
  pattern-matcher path.
- Local verification: `cargo check` passes.
- Pending verification: patch q4/q8/bf16 artifacts, run q4 forced-prefix
  correctness, then run raw-first L512 repeat>=3. This session could not run
  Metal correctness/profile after the benchmark because sandbox execution has
  no Metal device and the unsandboxed profiling request was denied.

Follow-up candidate: drain already-computed engine chunk tokens in one oMLX
engine tick.

- Motivation: the engine chunk path can compute up to `chunk_capacity` greedy
  tokens into `_fk_engine_queue`, but oMLX step style previously returned one
  queued token per scheduler tick. That reintroduces Python scheduler overhead
  even though the GPU work has already completed.
- Change: `_engine_drain_limit(self)` now defaults to draining the already
  materialized queue for stopless single-request greedy/logprobs-skip fast
  paths. It does not start a new GPU chunk inside the drain loop; it only
  returns tokens already present in `_fk_engine_queue`.
- Scope: runtime semantic condition only. It is not model-name based and does
  not depend on pattern matcher output.
- Local verification: Python bytecode compile passes for
  `python/fusekit_runtime/__init__.py`, `dispatch.py`, and the watch benchmark.
- Pending verification: run raw-first L512 repeat>=3 and compare with
  `FUSEKIT_DISABLE_ENGINE_DRAIN=1` to separate runtime drain win from qMV
  kernel changes.

2026-06-29 continuation notes:

- Added the q4 32-bit packed-load helper to additional full-block decode paths
  that still used two 16-bit loads: M-row qMV and M-row direct-head argmax scan.
  The guard is `token_rows == 1`, so MTP or other multi-token paths keep the
  old one-weight-load, multi-token fanout structure.
- Did not keep the same change for residual qMV/TG residual qMV. That overlaps
  the earlier rejected q4 LinearResidual u32-load experiment, so residual q4
  paths keep the old 16-bit load body until a same-run L512 profile proves
  otherwise.
- Did not keep the analogous q8 M=1 uint32-helper experiment. The log already
  records that q8 LM-head/single-partial byte-loop to uint32 helper regressed
  (`LM-head 5.19 ms -> 5.24 ms`, and earlier single partial total
  `38.39 ms -> 40.72 ms`), so that repeated candidate was removed before
  verification.
- Local verification remains `cargo check` plus Python bytecode compile. Metal
  patch/correctness/benchmark still requires unsandboxed access to the model
  artifact directory and Metal device.

Follow-up cleanup: restore stable fused-tail sigmoid.

- Evidence: the 2026-06-19 "FusedChain Simple Sigmoid" experiment changed
  fused SwiGLU tails from the stable `1 / (1 + exp(abs(g)))` reconstruction to
  direct sigmoid. It regressed q4 (`22.28 ms -> 22.48 ms`) and drifted q8, so
  it was rejected.
- Change: `FusedChain` routed/reduction gated-tail emitters now use the stable
  branch reconstruction again for the in-register `Sigmoid(gate)` tail. This
  is an op-semantic lowering rule, not a model-name condition. Pointwise-only
  standalone `Sigmoid` expressions were left unchanged because historical
  commits already used direct sigmoid there and the rejection was specifically
  the fused tail.
- Local verification: `cargo check` passes.
- Artifact generation check: `fusekit compile ... --out /private/tmp/...`
  still fails at graph capture with `No Metal device available` in this
  sandbox, so correctness/bench remains pending.

Follow-up cleanup: remove rejected fused-tail `metal::fast::tanh`.

- Evidence: the Gemma4 bf16 fused-tail `metal::fast::tanh` candidate preserved
  a forced-prefix check and slightly reduced one profile total, but L512 wall
  regressed (`12.852 TPS` raw vs `12.486 TPS` FuseKit), so the decision was
  "reject; restore precise tanh".
- Change: the remaining FusedChain pointwise-tail renderer now emits
  `tanh(clamp(...))` instead of `metal::fast::tanh(clamp(...))`.
- Local verification: `cargo check` passes and no `metal::fast::tanh` remains
  in the MSL emit sources checked by `rg`.

2026-06-29 Gemma4 q4 L512 +5% search:

- Target: `mlx-community/gemma-4-e4b-it-4bit`, decode L512, raw oMLX first,
  then FuseKit general path. The accepted threshold was +5% over raw.
- Current restored/default artifact coverage: custom task kernels 1001/1001,
  native fallback 0/1001, fused/specialized 426/1001, source ops fused
  1431/2006. Default raw-first L512 before the extra sweeps was raw
  41.09-41.13 TPS and FuseKit about 40.81-40.83 TPS, i.e. roughly -0.7% to
  -0.8%.
- Kept source-level view fast paths from this search only as a local kernel
  cleanup candidate: 4D transpose `[0,2,1,3]` and single-axis unit-stride
  slice now emit simpler index math. Late-token profile showed Slice and
  Transpose buckets drop, but e2e stayed around -0.7%, so this does not meet
  the +5% target by itself.
- Rejected `FUSEKIT_QMV_FAST_QDOT_Q4=1`: raw-first repeat=3 measured raw
  `41.129 TPS` and FuseKit `40.811 TPS` (`-0.8%`). Although raw MLX qmv_fast
  uses qdot, enabling q4 qdot globally did not produce an e2e win and also
  remains risky for near-tie text drift.
- Rejected homogeneous q4 fanout `COLS_PER_SG=8`: one sample improved the
  reduction-fanout bucket slightly, but the second sample regressed the same
  bucket to about 10.95 ms and standalone qMV/other buckets moved enough that
  total GPU time did not improve.
- Rejected MLX tile parity override
  `FUSEKIT_QMM_MLX_QMV_FAST_SGS=2 FUSEKIT_QMM_MLX_QMV_Q4_COLS_PER_SG=4`:
  standalone qMV stayed around 7.0-7.1 ms and chain/RMSNorm noise increased;
  no profile-backed path to +5%.
- Rejected standalone qMV `TOKEN_ROWS=1`: active-row amortization was a
  plausible culprit for Gemma's `[1,8,K]` qMV shapes, but profile samples
  still showed standalone qMV around 7.0-7.35 ms and no total GPU reduction.
- Structural finding: full `SwiGLU -> down_proj` SMEM fusion is not safe for
  Gemma q4 as a simple merge. The existing general admission intentionally
  rejects large down projections because full-SMEM down would collapse the
  many parallel down-projection threadgroups into one row-level threadgroup.
 A useful implementation needs a parallel streaming/partial-reduce design,
 not the current dense full-SMEM emitter generalized blindly to quantized
 LinearResidual.

2026-06-30 Gemma4 q4 L512 recheck after crash recovery:

- Scope: `mlx-community/gemma-4-e4b-it-4bit`, prompt `x`, L512, general
  fusion only (`--patterns off`), raw oMLX first then FuseKit. Default patch
  coverage stayed at custom task kernels `1001/1001`, native fallback
  `0/1001`, fused/specialized `426/1001`, source ops fused `1431/2006`.
- Raw-first repeat=3 current baseline: raw `38.083 TPS` vs FuseKit
  `34.482 TPS` (`-9.5%`). This run was slower than earlier 41 TPS raw/fuse
  runs, but the direction is clear: FuseKit GPU work is too long at L512.
- Late-token profile for the restored default showed total GPU about
  `27.7 ms/token`, dominated by `FusedChain reduction-fanout` (~9.8 ms),
  standalone `Single(QuantizedMatmul MLX qmv_fast)` (~7.1 ms), and
  `FinalRmsNormLmHead` (~3.0 ms). Coverage confirmed this is not native
  fallback; it is custom qMV/reduction-fanout kernel quality.
- Rejected fanout TG2 qdot only (`FUSEKIT_QMV_FAST_TG2_QDOT_Q4=1`):
  forced-prefix correctness passed, but token-448 profile worsened to
  `28.55 ms`; reduction-fanout rose to `10.35 ms` and e2e single sample was
  `34.08 TPS`.
- Rejected fanout grid cap (`FUSEKIT_FANOUT_CAP=128`): token-448 profile
  worsened to `31.74 ms`; reduction-fanout rose to `10.94 ms`. Reducing the
  repeated RMSNorm work lost more qMV parallelism than it saved.
- Rejected generalized materialized-norm gate/up barrier for q4: the semantic
  DAG detector correctly identified 42 GELU-style gate/up tails and produced
  `QmvFanout=42`, `Single(RMSNorm)=156`, but profile worsened to
  `29.03 ms` because `QmvFanout` itself cost `10.70 ms` plus the extra
  RMSNorm cost. The source was restored to the prior q8 opt-in barrier only.
- Rejected q4 dtype SMEM for reduction-fanout: q4 currently stores a bf16-
  rounded normalized row in `float` threadgroup memory. Changing q4 to store
  the row as `bfloat` preserved forced-prefix correctness, but token-448
  profile worsened to `28.65 ms`; reduction-fanout rose to `10.37 ms`.
  Keep q4 norm SMEM as float. q8 remains on dtype SMEM from prior accepted
  work.
- Current conclusion: the remaining +5% target cannot come from more aggressive
  gate/up fusion or materialization on this shape. The next useful work is a
  structural qMV/reduction-fanout microkernel improvement that beats raw MLX
  qmv on `K=2560,N=10240,bits=4` without serializing gate/up or reducing
  output parallelism.

2026-06-30 Gemma4 q4 L512 custom-kernel/fusion diagnosis:

- Scope: `mlx-community/gemma-4-e4b-it-4bit`, general-only patch
  (`--patterns off`), prompt `x`, L512, Python runtime
  `/Applications/oMLX.app/.../python_`.
- Coverage on restored/default artifact stayed fully custom:
  `custom task kernels 1001/1001`, `native fallback 0/1001`,
  `fused/specialized 426/1001`, `source ops fused 1431/2006`.
  The slowdown is therefore not an MLX fallback or unsupported op issue.
- Correctness check after restoring the Tanh stabilization:
  forced-prefix token `236781` matched top1 for four decode steps, and
  layer-level check showed no raw/patched NaNs. Root cause found earlier was
  Metal `tanh` returning NaN in the GELU-style tail for large positive inputs;
  the accepted source fix is `metal::fast::tanh(clamp(x, -20, 20))` for
  general `Tanh` lowering.
- Default late-token profile remains dominated by custom qMV work:
  `FusedChain reduction-fanout` about `10.8 ms`, standalone
  `Single(QuantizedMatmul MLX qmv_fast)` about `8.1 ms`,
  `FinalRmsNormLmHead` about `3.2 ms`.
- Diagnostic reduction-fanout disable: after adding a temporary local guard
  and patching with reduction-fanout disabled, coverage stayed custom but tasks
  rose to `1043` and `QmvFanout=42` appeared. Correctness passed, but profile
  worsened to `32.68 ms` total GPU and e2e sample was `30.39 TPS`.
  Conclusion: the RMSNorm+gate/up reduction-fanout merge is beneficial; the
  remaining issue is q4 microkernel quality inside that fused path.
- Rejected reduction-fanout `ROWS_PER_SG=4`: correctness passed, but profile
  worsened to `34.82 ms`; reduction-fanout rose to `12.72 ms`.
- Rejected reduction-fanout `ROWS_PER_SG=2`: correctness passed, but profile
  worsened to `33.90 ms`; reduction-fanout rose to `12.36 ms`.
  MLX standalone qmv computes four output rows per simdgroup, but that policy
  does not transfer to this fused kernel because the extra accumulators raise
  register pressure enough to lose occupancy.
- Rejected q4 qparam broadcast: correctness passed and one profile reduced
  total GPU to `30.59 ms`, but raw-first repeat=3 measured raw
  `37.799 TPS` vs FuseKit `33.557 TPS` (`-11.2%`), with peak memory rising
  from `4.287 GB` to `4.339 GB`. Kernel-local improvement did not survive e2e.
- Rejected standalone q4 qMV MLX parity cols/SG=4:
  correctness passed, but standalone qMV rose to `8.34 ms` and e2e sample was
  `31.09 TPS`. Keep the current shape policy.
- Rejected final LM-head TGS sweeps: `TGS=64` worsened final head to
  `3.65 ms` and e2e `30.89 TPS`; `TGS=16` also left final head around
  `3.66 ms` and e2e `30.86 TPS`. The current q4 hidden<=4096 default
  `TGS=32` remains the local optimum among tested split counts.
- Current actionable finding: all ops are custom, but the custom q4 qMV family
  is still slower than raw MLX's quantized matmul path at L512. Simple budget
  toggles, row grouping, qparam broadcast, and LM-head split tuning do not
  produce the requested +5%. The next real candidate must rewrite the q4
  weight-streaming microkernel to approach MLX's qdot/load schedule while
  preserving FuseKit's fused RMSNorm/gate-up semantics and near-tie logits.

2026-07-07 Gemma4 q4 L512 async-first runtime and qMV candidate audit:

- Scope: `mlx-community/gemma-4-e4b-it-4bit`, general-only patch
  (`--patterns off`), prompt `x`, L512, repeat=3 unless noted.
- Accepted runtime policy change: engine greedy decode now tries the async
  single-step native path before decode chunk. Chunk remains available when
  forced by env or when async cannot submit. This is a runtime scheduling
  policy, not a model-name rule.
- Evidence: default chunk path used host chunk for long generations
  (`chunk_cap=64`, `gpu->host reason=long_multi_chunk`) and measured about
  `35.97 TPS`. With async-first copied into the model runtime, raw-first final
  comparison measured raw `39.060 TPS` vs FuseKit `40.487 TPS` (`+3.7%`),
  peak `4.287 GB` vs `4.334 GB`.
- Correctness: forced-prefix compare on tokens
  `236781,6974,236781,6974` matched raw top1 for all checked steps. Logit
  drift remains nonzero (`max_abs` up to `2.625` in this short check), so q4
  arithmetic-order changes still require top1/forced-prefix validation.
- Rejected q4 qdot/load-order candidate: changing q4 helper accumulation to
  sequential MLX-like terms and moving M-row/fanout paths to the helper
  worsened e2e. Raw `37.273 TPS` vs candidate `33.405 TPS`; the candidate was
  reverted to the prior explicit q4 schedules.
- Rejected `FUSEKIT_QMV_FANOUT_Q4_COLS_PER_SG=4`: token-448 profile worsened
  from about `27.23 ms` total GPU to `27.77 ms`; reduction-fanout rose
  `10.28 ms -> 10.60 ms`.
- Rejected `FUSEKIT_EXPERIMENTAL_QUANT_PAIR_FANOUT=1`: it correctly emitted
  the general `reduction-quant-pair-fanout` path, but total GPU worsened to
  `28.48 ms`, LM-head also rose, and e2e sample was `36.16 TPS`.
- Rejected `FUSEKIT_QGEMV_UNROLL2=0`: total GPU worsened to `28.33 ms`;
  reduction-fanout rose to `10.72 ms`.
- Rejected `FUSEKIT_QPARAM_BROADCAST=1` for q4: total GPU worsened to
  `28.54 ms`; reduction-fanout rose to `10.93 ms`.
- Rejected `FUSEKIT_QMM_MLX_QMV_FAST_SGS=2`: total GPU worsened to
  `29.38 ms`; standalone qMV and several non-qMV buckets became noisier.
- Current bottleneck after accepted runtime change remains GPU kernel quality,
  not fallback: patch coverage is `custom task kernels 1001/1001`,
  `native fallback 0/1001`, `source ops fused 1431/2006`. Late-token profile
  is still dominated by `FusedChain reduction-fanout` (~`10.28 ms`),
  `Single(QuantizedMatmul MLX qmv_fast)` (~`7.22 ms`), and q4 LM-head
  (~`2.96 ms`).

Follow-up qMV candidate audit:

- Rejected q4 multirow qdot2 with `ROWS=8`. Correctness passed on the
  forced-prefix check, but token-448 profile regressed to `27.50 ms` total GPU
  and the single e2e sample was `38.859 TPS`, below the accepted default
  `~40.5 TPS`. The source was restored to the prior `rows <= 4` qdot gate.
- Rejected Q/K/V `QmvFanout` by bypassing the post-projection RMSNorm barrier.
  The diagnostic emitted `QmvFanout=24` and reduced sampled token-448 GPU time
  to `26.50 ms`, but it raised task count (`1001 -> 1025`), reduced fused
  source coverage (`1431/2006 -> 1359/2006`), and e2e raw-first repeat=3 was
  only raw `37.094 TPS` vs candidate `37.307 TPS` (`+0.6%`). That is worse
  than the accepted default win and not enough to justify weakening the
  correctness/parallelism barrier.
- Rejected standalone-only q4 qdot. The candidate changed only
  `fusekit_qmv_fast_impl` so materialized qMV projections used MLX qdot while
  reduction-fanout and LM-head partial helpers stayed on the stable explicit
  q4 path. It still failed forced-prefix correctness: at decode step 4,
  raw top1 was `236781` and FuseKit top1 was `6974` (`max_abs=1.335938`,
  `mean_abs=0.398242`). The source and installed artifact were restored.
- Rejected q4 qparam-outer accumulation. The candidate kept `ROWS=8` but
  rewrote each packed q4 word as `dot(x, q) * scale + sum(x) * bias`, closer
  to MLX qdot math. Correctness passed for the real raw-greedy-like forced
  prefix (`236781` repeated, 80 decode steps), but token-448 profile did not
  improve the bottleneck: candidate `FusedChain reduction-fanout` was
  `10.080 ms` vs default `10.078 ms`, and `Single qMV` was `7.262 ms` vs
  default `7.155 ms`. Raw-first repeat=3 measured raw `39.507 wall TPS`,
  default FuseKit `40.330` (`+2.1%`), and candidate `40.502` (`+2.5%`).
  The extra `+0.4%` over default is within run noise and not backed by the
  qMV profile, so the env-gated source was removed and the installed artifact
  remains the default general-fusion path.
- Accepted q4 multirow qdot1 lowering. This keeps the MLX qdot
  `scale * dot(x, packed_q) + bias * sum(x)` placement but applies it one
  packed 8-value word at a time rather than qdot2's 16-value live range. It is
  now the default for q4 multirow fanout when `rows <= 8` and qparam broadcast
  is not active. Admission is by quant metadata and row grouping only; no model
  names, layer names, or pattern matcher paths are used.
- Correctness: Gemma4 q4 forced-prefix compare with prompt `x` and token
  `236781` repeated for 80 decode steps matched raw top1 throughout.
- Gemma4 q4 L512 raw-first repeat=3 after the change: raw `39.591 wall TPS`
  vs FuseKit `41.172` (`+4.0%`), peak `4.288 GB` vs `4.328 GB`.
  Compared with the same-run default before this change (`40.330 wall TPS`),
  the qdot1 lowering adds about `+2.1%` absolute wall TPS.
- Cross-model q4 check: `mlx-community/Qwen3.5-4B-4bit`, L512 raw-first
  repeat=3, raw `44.641 wall TPS` vs FuseKit `47.399` (`+6.2%`), peak
  `2.489 GB` vs `2.488 GB` average. This confirms the win is not Gemma-only.
- Profile note: Gemma token-448 profile showed the target
  `FusedChain reduction-fanout` bucket move from about `10.078 ms` to
  `9.956 ms`, while total sampled GPU time stayed noisy (`28.12-28.36 ms`).
  E2E was the deciding signal because the wall win reproduced on both Gemma
  and Qwen with fallback still at 0%.
- Rejected follow-up: forcing the single-row fused qMV path through
  `FUSEKIT_QGEMV_MLX_QDOT=1` on top of qdot1 passed the same 80-step
  forced-prefix top1 check, but token-448 profile regressed badly:
  `FinalRmsNormLmHead` rose to `3.458 ms`, `FusedChain reduction-fanout` rose
  to `10.114 ms`, total GPU rose to `29.397 ms`, and the sampled wall TPS fell
  to `38.000`. The installed Gemma artifact was restored to the default qdot1
  general-fusion path.
- Next structural direction: do not simply remove the barrier. Generalize a
  semantic `qMV -> view/reshape/transpose -> RMSNorm` tail absorb so the
  unnormalized Q/K round-trip can disappear while preserving the post-projection
  RMSNorm semantics. Admission must be by graph op kind, shape, dtype, and
  single-use dataflow, not model name or the old pattern matcher.
