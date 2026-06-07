# Draft Speculative Decoding Optimization Log

This tracks draft-based speculative decoding optimization lines. The measured
implementation in this log is `dflash_mlx` with `z-lab/*-DFlash` draft models,
so rows labeled "DFlash" refer to that implementation, not to the whole
speculative decoding problem. Each item must be kept or rejected from measured
benchmark/profiling data.

## Baseline Smoke Results

All runs used oMLX Python and `dflash_mlx.benchmark` with `--max-tokens 64`,
`--draft-quant w4`, and the JSON-lines prompt.

| Target | Draft | Block | Baseline TPS | DFlash TPS | Speedup | Acceptance | Peak Baseline | Peak DFlash | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mlx-community/Qwen3.5-4B-4bit` | `z-lab/Qwen3.5-4B-DFlash` | 8 | 20.399 | 40.639 | 1.99x | 0.688 | 2.44 GB | 2.79 GB | candidate |
| `mlx-community/Qwen3.5-4B-4bit` | `z-lab/Qwen3.5-4B-DFlash` | 16 | 19.786 | 35.483 | 1.79x | 0.672 | 2.44 GB | 2.86 GB | slower than block8 |
| `mlx-community/Qwen3.5-9B-4bit` | `z-lab/Qwen3.5-9B-DFlash` | 8 | 26.212 | 28.645 | 1.09x | 0.703 | 5.13 GB | 5.75 GB | weak candidate |
| `mlx-community/Qwen3.5-9B-4bit` | `z-lab/Qwen3.5-9B-DFlash` | 16 | 25.947 | 19.264 | 0.74x | 0.688 | 5.13 GB | 5.83 GB | reject |
| `fcobos/Qwen3.5-9B-oQ4-mtp` | `z-lab/Qwen3.5-9B-DFlash` | 8 | 24.570 | 7.787 | 0.32x | 0.000 | 5.20 GB | 5.87 GB | incompatible target/draft |

## Work Items

1. Verify qGEMM/qMM optimization.
   - Measure `DFLASH_VERIFY_QMM` and verify variants against the baseline.
   - Keep only if DFlash TPS improves and acceptance stays unchanged.
   - 2026-06-05: `DFLASH_VERIFY_QMM=0/1` on 4B block8 both measured around
     45.5 TPS, acceptance 0.6875. 9B block8 measured around 28.1 TPS,
     acceptance 0.7031. qMM on/off is not the main lever in this smoke.
   - 2026-06-05: qMM variant forcing was tested through
     `scripts/bench_dflash_monkeypatch.py`. `mma2big` and `mma2big_pipe`
     K_PARTS=4 are within noise of auto on 4B/9B block4; no variant is a
     defensible default from this smoke.

2. Block-size and timing tuning.
   - Block 16 is already slower on 4B and much slower on 9B.
   - Need block-size policy, not a fixed larger block.
   - 2026-06-05: block4 is best in current smoke. 4B block4 reaches 70.4 TPS
     and 9B block4 reaches 40.5 TPS. Block2 loses acceptance and block16 loses
     draft time.

3. Draft-aware materialization fusion.
   - Reduce `draft_greedy -> drafted tokens -> verify input` materialization.
   - Candidate: keep token buffers on-device and avoid host synchronization where possible.
   - 2026-06-05: added `scripts/bench_dflash_monkeypatch.py` to test runtime
     patches without editing installed `dflash_mlx`. Forcing draft launch to
     synchronous mode only reduced 4B block4 from 70.4 to 69.0 TPS and 9B
     block4 from 40.5 to 40.1 TPS. Current async prefetch is directionally
     correct, but not the biggest remaining lever in this short decode smoke.

4. Accept/commit/replay path optimization.
   - 4B block8 spends nontrivial time in `commit`, `verify`, and `replay`.
   - Candidate: reduce cache/state rollback/copy and queue/commit overhead.
   - 2026-06-05: `verify_len_cap=2` and `verify_len_cap=3` were tested on
     block4. Cap2 is clearly slower on 4B/9B because tokens-per-cycle drops.
     Cap3 is a tiny 4B-only win but loses on 9B, so it is not a general default.
   - 2026-06-05: `verify_mode=adaptive` was tested through the monkey-patch
     wrapper on block8. It did not recover block8 to block4 performance.

## Current Hypotheses

For draft speculative decoding on 4B block8, the largest post-prefill costs are
`verify`, `draft`, and `commit`.
For 9B block8 the benchmark is only weakly positive; block16 regresses because
`draft` time explodes, so the draft path must be profiled before increasing M.

## Experiment Log

### Verify qMM Toggle

| Target | Block | Env | DFlash TPS | Speedup | Acceptance | Draft us | Verify us | Commit us | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 8 | `DFLASH_VERIFY_QMM=0` | 45.507 | 2.24x | 0.688 | 12405 | 20705 | 12799 | same as qMM on |
| 4B 4bit | 8 | `DFLASH_VERIFY_QMM=1` | 45.673 | 2.25x | 0.688 | 12040 | 20654 | 12438 | no clear qMM win |
| 9B 4bit | 8 | `DFLASH_VERIFY_QMM=0` | 28.103 | 1.07x | 0.703 | 13891 | 19136 | 18300 | same as qMM on |
| 9B 4bit | 8 | `DFLASH_VERIFY_QMM=1` | 28.132 | 1.08x | 0.703 | 14600 | 20029 | 18608 | no clear qMM win |

Conclusion: verify qMM exists, but on these runs it is not enough. The larger
remaining targets are block policy, draft materialization, and commit/replay.

### Verify qMM Variant

The installed `verify_linear` path resolves variants through `_auto_variant`.
The wrapper can force that resolver without editing installed `dflash_mlx`.

| Target | Block | Forced Variant | DFlash TPS | Speedup | Acceptance | Tokens/Cycle | Draft us | Verify us | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 4 | `mma2big` | 70.089 | 3.45x | 0.625 | 2.667 | 14042 | 23919 | no clear win |
| 4B 4bit | 4 | `mma2big_pipe`, K=4 | 70.172 | 3.46x | 0.625 | 2.667 | 14262 | 24328 | no clear win |
| 9B 4bit | 4 | `mma2big` | 40.549 | 1.55x | 0.625 | 2.667 | 15123 | 23968 | within noise |
| 9B 4bit | 4 | `mma2big_pipe`, K=4 | 40.417 | 1.55x | 0.625 | 2.667 | 15141 | 24172 | no clear win |

Conclusion: qMM variant tuning does not beat the much larger `block_tokens=4`
policy. Keep auto for now unless a longer multi-repeat run shows a stable
model-specific gain.

### Block Size

| Target | Block | DFlash TPS | Speedup | Acceptance | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 2 | 60.703 | 2.99x | 0.406 | slower than block4 |
| 4B 4bit | 4 | 70.398 | 3.47x | 0.625 | current best |
| 4B 4bit | 8 | 45.673 | 2.25x | 0.688 | slower than block4 |
| 4B 4bit | 16 | 35.483 | 1.79x | 0.672 | reject |
| 9B 4bit | 2 | 36.522 | 1.39x | 0.422 | slower than block4 |
| 9B 4bit | 4 | 40.478 | 1.55x | 0.625 | current best |
| 9B 4bit | 8 | 28.132 | 1.08x | 0.703 | slower than block4 |
| 9B 4bit | 16 | 19.264 | 0.74x | 0.688 | reject |

Conclusion: do not increase M blindly. For current Qwen3.5 DFlash smoke,
`block_tokens=4` is the best policy and should be the first default/auto-tune
candidate.

### Draft-Aware Materialization

Monkey-patch command: `scripts/bench_dflash_monkeypatch.py --patch no-async-draft`.
This forces `EagerDraftBackend.draft_greedy(..., async_launch=False)` while
leaving installed `dflash_mlx` files untouched.

| Target | Block | Patch | DFlash TPS | Speedup | Acceptance | Tokens/Cycle | Draft us | Verify us | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 4 | no async draft | 68.993 | 3.39x | 0.625 | 2.667 | 175060 | 24085 | slower than async block4 |
| 9B 4bit | 4 | no async draft | 40.133 | 1.54x | 0.625 | 2.667 | 288088 | 23735 | slightly slower than async block4 |

Reference async block4 results:

| Target | Block | DFlash TPS | Speedup | Acceptance | Tokens/Cycle | Draft us | Verify us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B 4bit | 4 | 70.398 | 3.47x | 0.625 | 2.667 | 14272 | 25053 |
| 9B 4bit | 4 | 40.478 | 1.55x | 0.625 | 2.667 | 17033 | 23722 |

Conclusion: keep async draft prefetch. It overlaps most draft work, but turning
it off does not collapse wall TPS, so the next large gains are not just in this
materialization edge.

### Accept / Commit / Replay

`verify_len_cap` changes how many target tokens are verified per cycle. The
hypothesis was that shorter verify could lower target forward and rollback cost.

| Target | Block | Verify Cap | DFlash TPS | Speedup | Acceptance | Tokens/Cycle | Commit us | Draft us | Verify us | Replay us | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 4 | 3 | 70.815 | 3.48x | 0.562 | 2.286 | 13412 | 16334 | 27843 | 3648 | too small / not general |
| 4B 4bit | 4 | 2 | 58.150 | 2.86x | 0.406 | 1.684 | 27286 | 20170 | 38313 | 3032 | reject |
| 9B 4bit | 4 | 3 | 39.479 | 1.51x | 0.547 | 2.207 | 21614 | 18620 | 30202 | 4189 | reject |
| 9B 4bit | 4 | 2 | 35.968 | 1.38x | 0.438 | 1.778 | 39249 | 21119 | 37257 | 1841 | reject |

Adaptive block policy was tested by forcing `verify_mode=adaptive` through the
wrapper on block8.

| Target | Block | Patch | DFlash TPS | Speedup | Acceptance | Tokens/Cycle | Commit us | Draft us | Verify us | Replay us | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4B 4bit | 8 | adaptive | 45.566 | 2.24x | 0.688 | 3.200 | 12426 | 12213 | 20071 | 4687 | does not recover block8 |
| 9B 4bit | 8 | adaptive | 27.656 | 1.06x | 0.703 | 3.368 | 18489 | 15805 | 19387 | 4347 | slower than block8 qMM |

Conclusion: cap2/cap3 and current adaptive block policy should not replace
block4. The structural issue is not rollback alone; reducing verify length
increases cycles enough that commit/verify totals grow.
