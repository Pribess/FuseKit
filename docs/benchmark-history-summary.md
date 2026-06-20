# Benchmark History Summary

> Historical engineering record. For release-facing methodology, qualification
> status, and the concise result table, see [`../BENCHMARKS.md`](../BENCHMARKS.md).
> Some rows below were recovered from logs rather than raw JSONL and must not be
> read as a fresh benchmark of the current working tree.

This file summarizes model-level benchmark records that were recoverable from
the local optimization logs and retained session notes. The authoritative raw
experiment detail remains in `docs/general-fusion-optimization-log.md` and
`docs/dflash-optimization-log.md`.

Unless noted otherwise:

- Baseline is raw oMLX/MLX run first, then FuseKit.
- FuseKit means general fusion with `--patterns off`.
- Metric is wall-clock TPS.
- Delta is `(FuseKit TPS / Raw TPS - 1) * 100`.
- Native fallback is expected to be 0 for valid custom-kernel comparisons.

## Latest Useful Snapshot: Decode

| Model | L | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | 32 | 147.270 | 257.725 | +75.0% | 0.289 GB | 0.357 GB | docs |
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | 512 | 285.860 | 295.054 | +3.2% | 0.299 GB | 0.350 GB | docs |
| `mlx-community/Qwen3.5-4B-4bit` | 32 | 35.518 | 42.965 | +21.0% | n/a | n/a | docs, post-budget |
| `mlx-community/Qwen3.5-4B-4bit` | 512 | 44.641 | 47.399 | +6.2% | 2.489 GB | 2.488 GB | docs, q4 cross-check after qdot1 |
| `mlx-community/Qwen3.5-4B-8bit` | 32 | 21.819 | 24.503 | +12.3% | n/a | n/a | docs, post-budget |
| `mlx-community/Qwen3.5-4B-8bit` | 512 | 25.464 | 26.081 | +2.4% | 4.586 GB | 4.597 GB | docs |
| `mlx-community/Qwen3.5-4B-bf16` | 32 | 12.362 | 13.601 | +10.0% | n/a | n/a | docs, post-budget |
| `mlx-community/Qwen3.5-4B-bf16` | 512 | 14.055 | 15.075 | +7.3% | 8.503 GB | 8.539 GB | docs |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | 32 | 62.271 | 57.203 | -8.1% | n/a | n/a | docs, latest recoverable L32 raw/fuse; raw moved upward |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | 512 | 70.681 | 72.892 | +3.1% | 9.102 GB | 9.245 GB | docs |
| `mlx-community/gemma-4-e4b-it-4bit` | 512 | 39.591 | 41.172 | +4.0% | 4.288 GB | 4.328 GB | docs |
| `mlx-community/gemma-4-e4b-it-8bit` | 512 | 22.209 | 23.882 | +7.5% | 8.005 GB | 8.051 GB | docs, q8 LM-head slice policy |
| `mlx-community/gemma-4-e4b-it-bf16` | 512 | 13.086 | 13.119 | +0.2% | 14.968 GB | 15.013 GB | docs, best recorded near-parity run |
| `mlx-community/gemma-3-4b-it-4bit` | 32 | 38.66 warm avg | 42.80 warm avg | +10.7% | 2.613 GB | 2.660 GB | session-note-only |
| `mlx-community/gemma-3-4b-it-4bit` | 512 | 47.145 | 50.334 | +6.8% | 2.710 GB avg | 2.741 GB avg | session-note-only |

## Latest Useful Snapshot: MTP

These rows are MTP-specific and are not comparable to normal single-decode
rows.

| Model | Path | Mode | MTP | L | TPS avg | TPS sd | Delta vs raw | Wall avg | First-token avg | Peak | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | raw-mtp | raw | yes | 512 | 35.453 | 0.826 | baseline | 14.448009 | 0.304008 | 2.621 GB | smoke |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | fusekit-mtp | fusekit | yes | 512 | 60.761 | 3.543 | +71.4% | 8.449884 | 0.252618 | 2.601 GB | model/format-specific smoke |
| `corysus/Qwen3.5-4B-oQ4-mtp` | raw-mtp | raw | yes | 512 | 57.393 | 1.912 | baseline | 8.929055 | 0.315519 | 2.759 GB | ok |
| `corysus/Qwen3.5-4B-oQ4-mtp` | fusekit-mtp | fusekit | yes | 512 | 57.092 | 1.992 | -0.5% | 8.976554 | 0.218953 | 2.746 GB | weak/reject |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | raw-mtp | raw | yes | 512 | 167.575 | 8.564 | baseline | 3.062000 | 0.209203 | 0.552 GB | ok |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | fusekit-mtp | fusekit | yes | 512 | 182.514 | 24.117 | +8.9% | 2.852034 | 0.076987 | 0.552 GB | ok but noisy |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | raw-mtp | raw | yes | 512 | 37.991 | 0.260 | baseline | 13.477538 | 0.404632 | 5.484 GB | ok |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | fusekit-mtp | fusekit | yes | 512 | 34.018 | 0.803 | -10.5% | 15.057316 | 0.366192 | 5.475 GB | reject/custom MTP path slower |

## Latest Useful Snapshot: Prefill

| Area | Prefill L | Status | Recoverable result |
| --- | ---: | --- | --- |
| raw vs FuseKit prefill TPS | unknown | benchmark harness exists, persisted table missing | no |
| no-logits prefill | unknown | script support exists | no complete local model-level table |
| M=64/chunk prefill variant | M=64 chunk, prompt L unknown | debug helper exists | no complete local model-level table |
| MLXcel comparison | unknown | discussed during session | no complete local table found |

## Qwen3.5 2B / 9B Inventory

| Model / family | Bench type found | L | Raw / baseline TPS | FuseKit / candidate TPS | Delta / speedup | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen3.5-2B-*` | none found | n/a | n/a | n/a | n/a | no local model dir or persisted benchmark row found |
| `mlx-community/Qwen3.5-9B-4bit` + `z-lab/Qwen3.5-9B-DFlash` | DFlash draft smoke | 64 | 26.212 | 28.645 | 1.09x | weak candidate, block 8 |
| `mlx-community/Qwen3.5-9B-4bit` + `z-lab/Qwen3.5-9B-DFlash` | DFlash draft smoke | 64 | 25.947 | 19.264 | 0.74x | reject, block 16 |
| `mlx-community/Qwen3.5-9B-4bit` + `z-lab/Qwen3.5-9B-DFlash` | DFlash block sweep | 64 | n/a | 40.478 | 1.55x | current best DFlash block 4 |
| `fcobos/Qwen3.5-9B-oQ4-mtp` + `z-lab/Qwen3.5-9B-DFlash` | DFlash incompatible target/draft | 64 | 24.570 | 7.787 | 0.32x | reject, acceptance 0 |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | MTP smoke | 512 | 37.991 | 34.018 | -10.5% | reject/custom MTP path slower |

## Latest Snapshot Gaps

| Area | Status |
| --- | --- |
| Gemma4 E4B L32 | no persisted L32 raw/FuseKit table found |
| Gemma3 bf16/q8/q4 variants beyond q4 | no persisted L32/L512 table found |
| Prefill | benchmark script exists, but no complete persisted model-level result table was found |
| Qwen3.5 2B | no local model dir or persisted benchmark row found |
| Qwen3.5 9B normal single-decode | only MTP/DFlash records found; no normal raw-vs-FuseKit decode table found |

## Qwen2.5 0.5B

Patch status recorded for `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit`:

| Patterns | Tasks | Fallback | Notable fused path |
| --- | ---: | ---: | --- |
| off | 364 | 0% | 24 reduction-fanout MLP tails absorbed |

Correctness:

| Check | Result |
| --- | --- |
| raw vs patched decode logits | same argmax, top-5 preserved, max abs 0.0327148, mean abs 0.00475509 |

Bench:

| L | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 147.270 | 257.725 | +75.0% | 0.289 GB | 0.357 GB |
| 512 | 285.860 | 295.054 | +3.2% | 0.299 GB | 0.350 GB |

Session-note-only diagnostic:

| Model | Path | L | Repeat | TPS | Peak | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | raw-mtp flag, MTP inactive | 512 | 5 | 278.556 | 0.330 GB | qwen2 has no MTP heads; FuseKit side run was interrupted in pasted output |

## Qwen3.5 4B

Coverage recorded for restored general artifacts:

| Model | Custom tasks | Native fallback | Fused/specialized | Source ops fused |
| --- | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3.5-4B-4bit` | 553/553 | 0/553 | 340/553 | 938/1247 |
| `mlx-community/Qwen3.5-4B-8bit` | 553/553 | 0/553 | 340/553 | 938/1247 |
| `mlx-community/Qwen3.5-4B-bf16` | 700/700 | 0/700 | 340/700 | 1131/1587 |

Historical model-level benches:

| Date / context | Model | L | Raw TPS | FuseKit TPS | Delta | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| prior u32load final | q4 | 32 | 34.985 | 41.746 | +19.32% | repeat=3 |
| prior u32load final | q8 | 32 | 21.382 | 24.072 | +12.58% | repeat=3 |
| prior u32load final | bf16 | 32 | 12.423 | 13.698 | +10.26% | repeat=3 |
| prior u32load final | q4 | 512 | 39.803 | 43.013 | +8.07% | older/noisy |
| prior u32load final | q8 | 512 | 22.583 | 23.174 | +2.62% | older/noisy |
| prior u32load final | bf16 | 512 | 13.284 | 14.322 | +7.82% | repeat=3 |
| q8 inline affine | q8 | 32 | 22.115 | 25.144 | +13.69% | repeat=3 |
| q8 inline affine | q8 | 512 | 23.486 | 26.032 | +10.84% | raw baseline lower than later runs |
| q4/q8 revisit | q4 | 512 | 39.865 | 43.286 | +8.58% | q8 delta inflated by low raw |
| q4/q8 revisit | q8 | 512 | 22.633 | 25.294 | +11.76% | q8 delta inflated by low raw |
| current structural note | q4 | 512 | 44.116 | 45.022 | +2.05% | repeat=3 |
| current structural note | q8 | 512 | 24.802 | 25.231 | +1.73% | repeat=3 |
| current structural note | bf16 | 512 | 13.676 | 14.572 | +6.56% | repeat=3 |
| q4 rejects + q8 qacc/xsum | q4 | 512 | 45.945 | 45.984 | +0.08% | did not reproduce older q4 win |
| q4 rejects + q8 qacc/xsum | q8 | 512 | 23.883 | 24.006 | +0.51% | noisy |
| q4 rejects + q8 qacc/xsum | bf16 | 512 | 12.375 | 13.402 | +8.30% | repeat=3 |
| qMV tile follow-up | q4 | 512 | 46.054 | 46.683 | +1.37% | kept q4 large single-token qMV cols/sg=8 |
| qMV tile follow-up | q8 | 512 | 24.730 | 25.112 | +1.55% | q8 large cols/sg=8 rejected |
| qMV tile follow-up | bf16 | 512 | 12.875 | 14.804 | +14.98% | unchanged by q4/q8 tile rules |
| AsType absorb before | q4 | 32 | 36.177 | 44.258 | +22.34% | repeat=5 |
| AsType absorb before | q4 | 512 | 45.213 | 46.081 | +1.92% | repeat=5 |
| AsType absorb before | q8 | 32 | 22.031 | 24.931 | +13.16% | repeat=5 |
| AsType absorb before | q8 | 512 | 25.524 | 25.855 | +1.29% | repeat=5 |
| AsType absorb final | q4 | 32 | 34.310 | 43.096 | +25.61% | repeat=5 |
| AsType absorb final | q4 | 512 | 42.887 | 44.917 | +4.73% | repeat=5 |
| AsType absorb final | q8 | 32 | 20.956 | 23.652 | +12.87% | repeat=5 |
| AsType absorb final | q8 | 512 | 24.186 | 24.710 | +2.17% | repeat=5 |
| q8 qparam broadcast final | q8 | 512 | 25.951 | 26.331 | +1.47% | repeat=5 |
| q8-only activation scratch | q4 | 512 | 40.716 | 42.869 | +5.29% | repeat=3 |
| q8-only activation scratch | q8 | 512 | 22.619 | 23.415 | +3.52% | repeat=3 |
| q8-only activation scratch | bf16 | 512 | 13.061 | 14.526 | +11.22% | repeat=3 |
| fast rsqrt candidate | q4 | 512 | 45.346 | 46.039 | +1.53% | rejected |
| fast rsqrt candidate | q8 | 512 | 25.469 | 25.900 | +1.69% | rejected |
| fast rsqrt candidate | bf16 | 512 | 13.970 | 14.956 | +7.06% | rejected |
| current L512 same-run | q4 | 512 | 45.999 | 46.582 | +1.27% | repeat=3 |
| current L512 same-run | q8 | 512 | 25.608 | 26.157 | +2.14% | repeat=3 |
| current L512 same-run | bf16 | 512 | 14.132 | 15.102 | +6.86% | repeat=3 |
| q8 pair candidate batch | q4 | 512 | 46.072 | 46.630 | +1.21% | repeat=3 |
| q8 pair candidate batch | q8 | 512 | 25.736 | 26.172 | +1.69% | repeat=3 |
| q8 pair candidate batch | bf16 | 512 | 14.222 | 15.178 | +6.72% | repeat=3 |
| q8 RmsNormQkv TG-qMV | q4 | 512 | 46.035 | 46.680 | +1.40% | repeat=3 |
| q8 RmsNormQkv TG-qMV | q8 | 512 | 25.923 | 26.428 | +1.95% | repeat=3 |
| q8 RmsNormQkv TG-qMV | bf16 | 512 | 14.205 | 15.200 | +7.00% | repeat=3 |
| current final check | q4 | 512 | 45.911 | 46.752 | +1.83% | repeat=3 |
| current final check | q8 | 512 | 25.969 | 26.503 | +2.05% | repeat=3 |
| current final check | bf16 | 512 | 14.212 | 15.199 | +6.94% | repeat=3 |
| pointwise-prefix RMSNorm | q4 | 512 | 45.953 | 46.519 | +1.2% | repeat=3 |
| pointwise-prefix RMSNorm | q8 | 512 | 25.683 | 26.116 | +1.7% | repeat=3 |
| pointwise-prefix RMSNorm | bf16 | 512 | 14.082 | 15.120 | +7.4% | repeat=3 |
| greedier fusion scan | q4 | 512 | 45.736 | 46.415 | +1.5% | repeat=3 |
| greedier fusion scan | q8 | 512 | 25.701 | 26.295 | +2.3% | repeat=3 |
| greedier fusion scan | bf16 | 512 | 14.069 | 15.122 | +7.5% | repeat=3 |
| post-budget | q4 | 32 | 35.518 | 42.965 | +21.0% | repeat=3 |
| post-budget | q4 | 512 | 46.135 | 46.549 | +0.9% | repeat=3 |
| post-budget | q8 | 32 | 21.819 | 24.503 | +12.3% | repeat=3 |
| post-budget | q8 | 512 | 25.823 | 26.215 | +1.5% | repeat=3 |
| post-budget | bf16 | 32 | 12.362 | 13.601 | +10.0% | repeat=3 |
| post-budget | bf16 | 512 | 14.170 | 15.161 | +7.0% | repeat=3 |
| 2026-06-23 op-sequence gate | q4 | 512 | 45.144 | 46.435 | +2.9% | repeat=5 |
| 2026-06-23 op-sequence gate | q8 | 512 | 25.464 | 26.081 | +2.4% | repeat=5 |
| 2026-06-23 op-sequence gate | bf16 | 512 | 14.055 | 15.075 | +7.3% | repeat=5 |
| q4 qdot1 cross-model check | q4 | 512 | 44.641 | 47.399 | +6.2% | repeat=3, after Gemma4 qdot1 lowering |

## DeepSeek V2 Lite 4bit

Model: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`.

Important distinction: early fallback rows were invalid custom-kernel
comparisons because MLA dispatch was not active. Later rows are full-custom
or no-native-fallback comparisons.

| Context | L | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fallback, short | 32 | 58.490 | 53.477 | -8.6% | 8.946 GB | 8.991 GB | invalid custom comparison; MLA fallback |
| fallback, long | 512 | 67.746 | 63.871 | -5.7% | 9.104 GB | 9.206 GB | invalid custom comparison; MLA fallback |
| post MLA drift fix | 32 | 56.901 | 56.848 | -0.1% | 8.938 GB | 8.986 GB | no fallback |
| post MLA drift fix | 512 | 69.220 | 69.048 | -0.2% | 9.102 GB | 9.275 GB | no fallback |
| routed MoE default | 32 | 62.271 | 57.203 | -8.1% | n/a | n/a | raw baseline moved up |
| routed MoE default | 512 | 71.943 | 69.258 | -3.7% | n/a | n/a | raw baseline moved up |
| no-named + output-scale sink | 512 | 67.764 baseline | 68.841 candidate | +1.6% | n/a | n/a | baseline/candidate comparison |
| DeepSeek round-trip cuts | 512 | 72.493 | 73.487 | +1.4% | 9.125 GB | 9.208 GB | + Add sink |
| DeepSeek round-trip cuts | 512 | 71.494 | 72.632 | +1.6% | 9.125 GB | 9.227 GB | + Split active-M |
| DeepSeek round-trip cuts | 512 | 71.507 | 73.199 | +2.4% | 9.125 GB | 9.222 GB | + SDPA Q TransposeSplit sink |
| budget-based rejection path | 512 | 71.698 | 73.816 | +2.95% | n/a | n/a | repeat=3 |
| op-sequence gated artifact | 512 | 65.996 | 70.483 | +6.8% | n/a | n/a | repeat=3, low raw run |
| repeat=5 recheck | 512 | 69.307 | 72.873 | +5.1% | n/a | n/a | repeat=5 |
| 2026-06-23 repeat=5 | 512 | 70.681 | 72.892 | +3.1% | 9.102 GB | 9.245 GB | current recorded stable table |

Notable DeepSeek candidate records:

| Candidate | Result | Decision |
| --- | --- | --- |
| forcing flash SDPA on MLA | SDPA grew ~1.94 ms -> ~4.2 ms; L512 dropped to ~62.5 TPS | reject |
| old named MoE artifact vs no-named default | old 69.231 TPS, no-named 69.517 TPS | disable named MoE composites by default |
| `GatherQMM` MLX qmv-fast | 70.571 -> 70.360 TPS, -0.3% | reject |
| `ScaledSumReduce` | 69.956 -> 68.315 TPS, -2.3% after rounding fix | opt-in only |
| `GatherQMM` half x-row SMEM | L512 70.138 -> 68.692 TPS, -2.1% | reject |
| `GatherQMM` qparam broadcast | L512 71.737 -> 70.245 TPS, -2.1% | reject |
| `GatherQMM` prefetch body | L512 69.538 -> 67.351 TPS, -3.1% | reject |
| `GatherQMM` NAX helper | L512 68.690 -> 57.084 TPS, -16.9% | reject |
| standalone `GatherQMM` q4 qdot body | L512 70.842 -> 71.070 TPS, +0.3%; drift worsened | reject |
| generic `GatherQMM` pair + SwiGLU tail | raw 71.391 vs candidate 71.647, +0.4%; diff worsened | opt-in only |
| `GatherQMMReduce+LinearResidual` | reduce bucket lower but total GPU worse | opt-in only |
| `ArgTopK` small-k select | 67.164 -> 64.311 TPS, -4.2% | reject |
| remove `GatherQMMReduce` dead barrier | default 70.011 vs candidate 72.420, +3.4%; raw-first candidate 72.508 vs 72.579, +0.1% | keep small cleanup |
| smem-only `GatherQMM` path | default 67.663 vs candidate 68.180, +0.8% | keep small cleanup |
| route metadata broadcast in reduce | default 73.08 vs candidate 74.02, +1.3%; excl first +2.1% | keep |

## Gemma4 E4B

Models:

- `mlx-community/gemma-4-e4b-it-4bit`
- `mlx-community/gemma-4-e4b-it-8bit`
- `mlx-community/gemma-4-e4b-it-bf16`

Coverage after support fixes:

| Model | Custom tasks | Fallback | Fused/specialized |
| --- | ---: | ---: | ---: |
| q4/q8 | 1001/1001 | 0% | 426 |
| bf16 | 1067/1067 | 0% | 593 |

Model-level history:

| Context | Model | L | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict dispatch initial | q4 | 512 | 38.194 | 35.606 | -6.8% | 4.288 GB | 4.335 GB | unsupported/fallback fixed, performance bad |
| strict dispatch initial | q8 | 512 | 21.891 | 21.456 | -2.0% | 8.005 GB | 8.053 GB | initial |
| strict dispatch initial | bf16 | 512 | 11.772 | 9.796 | -16.8% | 14.968 GB | 14.981 GB | initial |
| q4/q8 fixed path | q4 | 512 | 41.914 | 43.781 | +4.5% | 4.288 GB | 4.360 GB | repeat=3 |
| q4/q8 fixed path | q8 | 512 | 23.625 | 25.169 | +6.5% | 8.006 GB | 8.077 GB | repeat=3 |
| q4/q8 fixed path | bf16 | 512 | 12.970 | 12.858 | -0.9% | 14.968 GB | 15.002 GB | repeat=3 |
| Single-SDPA decode lowering | q4 | 512 | 42.048 | 43.858 | +4.3% | 4.288 GB | 4.371 GB | repeat=3 |
| Single-SDPA decode lowering | q8 | 512 | 23.768 | 25.454 | +7.1% | 8.006 GB | 8.088 GB | repeat=3 |
| Single-SDPA decode lowering | bf16 | 512 | 13.086 | 13.119 | +0.2% | 14.968 GB | 15.013 GB | repeat=3 |
| GELU-tail absorb trial | q4 | 512 | 39.120 | 39.779 | +1.7% | n/a | n/a | lower than default; off |
| rotating-cache runtime fix | q4 | 512 | 39.60 avg | 40.21 avg | +1.5% | n/a | n/a | before qdot candidate |
| q4 qdot candidate, first pass | q4 | 512 | 39.60 same-run raw | 41.56 | +5.0% | n/a | n/a | later rejected due forced-prefix drift |
| q4 qdot candidate revalidation | q4 | 512 | 39.350 | 33.854 | -14.0% | n/a | n/a | reject |
| q4 qparam broadcast | q4 | 512 | 40.635 | 34.811 | -14.3% | n/a | n/a | reject |
| safe decode-online after SDPA split | q4 | 512 | 39.36 | 33.28 | -15.5% | n/a | n/a | correctness preserved but slow |
| two-dispatch flash SDPA | q4 | 512 | 41.031 | 35.510 | -13.5% | n/a | n/a | still slower |
| 2026-06-28 recheck | q4 | 512 | 40.571 | 39.864 | -1.7% | 4.287 GB | 4.335 GB | repeat=2 |
| 2026-06-28 recheck | q8 | 512 | 22.800 | 22.853 | +0.2% | 8.005 GB | 8.053 GB | repeat=2 |
| 2026-06-28 recheck | bf16 | 512 | 12.532 | 12.169 | -2.9% | 14.967 GB | 15.010 GB | repeat=2 |
| q8 LM-head slice policy | q8 | 512 | 22.209 | 23.882 | +7.5% | 8.005 GB | 8.051 GB | accepted q8 policy |
| q4/bf16 follow-up baseline | q4 | 512 | 41.435 | 40.723 | -1.7% | 4.287 GB | 4.335 GB | repeat=3 |
| q4/bf16 follow-up baseline | q8 | 512 | 23.348 | 23.018 | -1.4% | 8.005 GB | 8.052 GB | repeat=3 |
| q4/bf16 follow-up baseline | bf16 | 512 | 12.841 | 12.464 | -2.9% | 14.967 GB | 15.019 GB | repeat=3 |
| crash recovery baseline | q4 | 512 | 38.083 | 34.482 | -9.5% | n/a | n/a | restored/default artifact, bad run |
| async-first runtime | q4 | 512 | 39.060 | 40.487 | +3.7% | 4.287 GB | 4.334 GB | accepted runtime policy |
| qparam-outer candidate | q4 | 512 | 39.507 | 40.502 | +2.5% | n/a | n/a | only +0.4 over default; removed |
| q4 multirow qdot1 | q4 | 512 | 39.591 | 41.172 | +4.0% | 4.288 GB | 4.328 GB | accepted; repeat=3 |

## Gemma3

Session-note-only latest record for `mlx-community/gemma-3-4b-it-4bit`.
This was not found in the local markdown log.

Patch/correctness:

| Item | Value |
| --- | --- |
| Custom task kernels | 858/858 |
| Native fallback | 0/858 |
| Source coverage | 1401/1413 |
| Fused/specialized | 311/858 |
| Strict check prompt `""` token 1 | same argmax, max abs 0.390625, mean abs 0.0735533 |
| Strict check prompt `"x"` token 1 | same argmax, max abs 0.53125, mean abs 0.0875103 |

Bench:

| L | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 38.66 | 42.80 | +10.7% | 2.613 GB | 2.660 GB | warm avg of last two runs; all-run avg raw 33.044 vs FuseKit 36.261 (+9.7%) |
| 512 | 47.145 | 50.334 | +6.8% | 2.710 GB avg | 2.741 GB avg | repeat=3 |

## MTP

These are MTP-specific smoke runs and should not be mixed with normal
single-decode q4/q8/bf16 results.

| Model | Path | Mode | MTP | L | TPS avg | TPS sd | Delta vs raw | Wall avg | First-token avg | Peak |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | raw-mtp | raw | yes | 512 | 35.453 | 0.826 | baseline | 14.448009 | 0.304008 | 2.621 GB |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | fusekit-mtp | fusekit | yes | 512 | 60.761 | 3.543 | +71.4% | 8.449884 | 0.252618 | 2.601 GB |
| `corysus/Qwen3.5-4B-oQ4-mtp` | raw-mtp | raw | yes | 512 | 57.393 | 1.912 | baseline | 8.929055 | 0.315519 | 2.759 GB |
| `corysus/Qwen3.5-4B-oQ4-mtp` | fusekit-mtp | fusekit | yes | 512 | 57.092 | 1.992 | -0.5% | 8.976554 | 0.218953 | 2.746 GB |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | raw-mtp | raw | yes | 512 | 167.575 | 8.564 | baseline | 3.062000 | 0.209203 | 0.552 GB |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | fusekit-mtp | fusekit | yes | 512 | 182.514 | 24.117 | +8.9% | 2.852034 | 0.076987 | 0.552 GB |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | raw-mtp | raw | yes | 512 | 37.991 | 0.260 | baseline | 13.477538 | 0.404632 | 5.484 GB |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | fusekit-mtp | fusekit | yes | 512 | 34.018 | 0.803 | -10.5% | 15.057316 | 0.366192 | 5.475 GB |

Notes:

- The `+71.4%` MTPLX row is model/format-specific smoke, not a general MTP
  claim.
- Some later native-MTP/fallback runs were discussed, but no complete local
  table was found beyond the rows above.

## DFlash / Draft Speculative Decoding

These rows are from `dflash_mlx` with `z-lab/*-DFlash` draft models, not the
normal FuseKit single-decode path.

Baseline smoke, max tokens 64, draft quant w4:

| Target | Draft | L | Block | Baseline TPS | DFlash TPS | Speedup | Acceptance | Peak baseline | Peak DFlash | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen3.5-4B-4bit` | `z-lab/Qwen3.5-4B-DFlash` | 64 | 8 | 20.399 | 40.639 | 1.99x | 0.688 | 2.44 GB | 2.79 GB | candidate |
| `mlx-community/Qwen3.5-4B-4bit` | `z-lab/Qwen3.5-4B-DFlash` | 64 | 16 | 19.786 | 35.483 | 1.79x | 0.672 | 2.44 GB | 2.86 GB | slower than block8 |
| `mlx-community/Qwen3.5-9B-4bit` | `z-lab/Qwen3.5-9B-DFlash` | 64 | 8 | 26.212 | 28.645 | 1.09x | 0.703 | 5.13 GB | 5.75 GB | weak candidate |
| `mlx-community/Qwen3.5-9B-4bit` | `z-lab/Qwen3.5-9B-DFlash` | 64 | 16 | 25.947 | 19.264 | 0.74x | 0.688 | 5.13 GB | 5.83 GB | reject |
| `fcobos/Qwen3.5-9B-oQ4-mtp` | `z-lab/Qwen3.5-9B-DFlash` | 64 | 8 | 24.570 | 7.787 | 0.32x | 0.000 | 5.20 GB | 5.87 GB | incompatible target/draft |

Block-size sweep:

| Target | L | Block | DFlash TPS | Speedup | Acceptance | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 64 | 2 | 60.703 | 2.99x | 0.406 | slower than block4 |
| 4B 4bit | 64 | 4 | 70.398 | 3.47x | 0.625 | current best |
| 4B 4bit | 64 | 8 | 45.673 | 2.25x | 0.688 | slower than block4 |
| 4B 4bit | 64 | 16 | 35.483 | 1.79x | 0.672 | reject |
| 9B 4bit | 64 | 2 | 36.522 | 1.39x | 0.422 | slower than block4 |
| 9B 4bit | 64 | 4 | 40.478 | 1.55x | 0.625 | current best |
| 9B 4bit | 64 | 8 | 28.132 | 1.08x | 0.703 | slower than block4 |
| 9B 4bit | 64 | 16 | 19.264 | 0.74x | 0.688 | reject |

Other DFlash rows:

| Target | L | Block | Variant | DFlash TPS | Speedup | Acceptance | Decision |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 4B 4bit | 64 | 8 | `DFLASH_VERIFY_QMM=0` | 45.507 | 2.24x | 0.688 | same as qMM on |
| 4B 4bit | 64 | 8 | `DFLASH_VERIFY_QMM=1` | 45.673 | 2.25x | 0.688 | no clear qMM win |
| 9B 4bit | 64 | 8 | `DFLASH_VERIFY_QMM=0` | 28.103 | 1.07x | 0.703 | same as qMM on |
| 9B 4bit | 64 | 8 | `DFLASH_VERIFY_QMM=1` | 28.132 | 1.08x | 0.703 | no clear qMM win |
| 4B 4bit | 64 | 4 | `mma2big` | 70.089 | 3.45x | 0.625 | no clear win |
| 4B 4bit | 64 | 4 | `mma2big_pipe`, K=4 | 70.172 | 3.46x | 0.625 | no clear win |
| 9B 4bit | 64 | 4 | `mma2big` | 40.549 | 1.55x | 0.625 | within noise |
| 9B 4bit | 64 | 4 | `mma2big_pipe`, K=4 | 40.417 | 1.55x | 0.625 | no clear win |
| 4B 4bit | 64 | 4 | no async draft | 68.993 | 3.39x | 0.625 | slower than async block4 |
| 9B 4bit | 64 | 4 | no async draft | 40.133 | 1.54x | 0.625 | slightly slower |
| 4B 4bit | 64 | 4 | verify cap 3 | 70.815 | 3.48x | 0.562 | too small / not general |
| 4B 4bit | 64 | 4 | verify cap 2 | 58.150 | 2.86x | 0.406 | reject |
| 9B 4bit | 64 | 4 | verify cap 3 | 39.479 | 1.51x | 0.547 | reject |
| 9B 4bit | 64 | 4 | verify cap 2 | 35.968 | 1.38x | 0.438 | reject |
| 4B 4bit | 64 | 8 | adaptive | 45.566 | 2.24x | 0.688 | does not recover block8 |
| 9B 4bit | 64 | 8 | adaptive | 27.656 | 1.06x | 0.703 | slower than block8 qMM |

## Prefill

Prefill benchmark support exists in `scripts/bench_prefill.py`, and several
prefill-related investigations were discussed, including no-logits prefill,
M-chunk variants, `mlxcel` comparisons, and chunking at 256/512/1024 prompt
lengths. However, no complete persisted prefill TPS table was found in
`docs/*.md`, the repo, or `/private/tmp` during this recovery pass.

Recoverable prefill-related facts:

| Item | Prefill L | Status |
| --- | ---: | --- |
| Benchmark harness | user-specified | `scripts/bench_prefill.py` emits `[prefill:run]` and `[prefill:json]` records |
| Isolated debug helper | M=64 chunk, prompt L user-specified | `scripts/debug_prefill_variant_m64.py` compares one FuseKit prefill M chunk against raw oMLX |
| Persisted result files | unknown | none found under repo or `/private/tmp` |
| Model-level raw/FuseKit prefill TPS | unknown | not recoverable from current local logs |
| MLXcel comparison numbers | unknown | discussed, but no complete local table found |

The prefill rows need to be regenerated if they are required for the same
standard as decode: raw first, FuseKit second, repeat count, prompt length,
no-logits setting, peak memory, and correctness check.

## Known Gaps

- Prefill / MLXcel comparison numbers were discussed, but no complete
  model-level TPS table was found in the local markdown logs or `/private/tmp`
  JSONL files. See the Prefill section above.
- Qwen2.5 bf16, Qwen2.5 1.5B bf16, and Qwen3.6 27B appeared in model lists and
  scripts, but exact persisted benchmark result rows were not found in the
  local logs searched in this session.
- Most raw JSONL files referenced by the logs lived under `/private/tmp` and no
  longer exist, so the markdown records above are the recoverable source of
  truth.
