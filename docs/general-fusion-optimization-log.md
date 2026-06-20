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
- Prefer local model paths under `/Users/pribess/.omlx/models/...`; repo ids can
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
scripts/bench_omlx_engine.py /Users/pribess/.omlx/models/mlx-community/Qwen3.5-4B-8bit \
--mode engine_fusekit --tokens 512 --repeat 1 --warmup 0 --prompt x --seed 0
```

Raw vs FuseKit repeat benchmark:

```sh
PYTHONPATH=/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages \
/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11 \
scripts/bench_omlx_engine_suite.py \
--model /Users/pribess/.omlx/models/mlx-community/Qwen3.5-4B-8bit \
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
