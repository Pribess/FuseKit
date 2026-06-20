# FuseKit benchmarks

This document separates **fresh release checks** from **historical model
measurements**. The distinction matters: the model cache was empty during the
latest source cleanup, so throughput could not be rerun against the current
working tree. No result below is presented as a universal speedup guarantee.

## Release-check status

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | pass |
| Rust workspace tests | 39 passed, 0 failed |
| Release build | pass (`aarch64-apple-darwin`) |
| Embedded Python syntax check | pass |
| Embedded Python probe | pass, CPython 3.11.10 / MLX 0.31.2 / `mlx_lm` import |
| Metal bridge smoke test | pass, 1,024 bf16 elements, max abs error 0.0 |
| Current-source model correctness/throughput | **not rerun: no local model weights available** |

Smoke-test host: Apple M5, 32 GB unified memory, macOS 26.6.2. Build and smoke
checks establish that the compiler, embedded native runtime, and Metal bridge
load; they do not replace a model-level correctness run.

## Historical decode results

The following rows were recovered from the optimization logs. Unless noted:

- raw oMLX/MLX ran first, then FuseKit;
- FuseKit used general fusion (`--patterns off`, now the default compiler path);
- TPS is wall-clock generated-token throughput;
- delta is `(FuseKit / raw - 1) × 100`;
- valid custom-kernel comparisons reported zero native fallback.

| Model | L | Raw TPS | FuseKit TPS | Delta | Peak raw | Peak FuseKit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | 32 | 147.270 | 257.725 | +75.0% | 0.289 GB | 0.357 GB |
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | 512 | 285.860 | 295.054 | +3.2% | 0.299 GB | 0.350 GB |
| `mlx-community/Qwen3.5-4B-4bit` | 32 | 35.518 | 42.965 | +21.0% | n/a | n/a |
| `mlx-community/Qwen3.5-4B-4bit` | 512 | 44.641 | 47.399 | +6.2% | 2.489 GB | 2.488 GB |
| `mlx-community/Qwen3.5-4B-8bit` | 32 | 21.819 | 24.503 | +12.3% | n/a | n/a |
| `mlx-community/Qwen3.5-4B-8bit` | 512 | 25.464 | 26.081 | +2.4% | 4.586 GB | 4.597 GB |
| `mlx-community/Qwen3.5-4B-bf16` | 32 | 12.362 | 13.601 | +10.0% | n/a | n/a |
| `mlx-community/Qwen3.5-4B-bf16` | 512 | 14.055 | 15.075 | +7.3% | 8.503 GB | 8.539 GB |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | 32 | 62.271 | 57.203 | **-8.1%** | n/a | n/a |
| `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | 512 | 70.681 | 72.892 | +3.1% | 9.102 GB | 9.245 GB |
| `mlx-community/gemma-4-e4b-it-4bit` | 512 | 39.591 | 41.172 | +4.0% | 4.288 GB | 4.328 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 512 | 22.209 | 23.882 | +7.5% | 8.005 GB | 8.051 GB |
| `mlx-community/gemma-4-e4b-it-bf16` | 512 | 13.086 | 13.119 | +0.2% | 14.968 GB | 15.013 GB |

### Interpretation

- Short-context wins are often larger because Python/dispatch overhead is a
  greater share of total time.
- Long-context results are generally smaller and more sensitive to thermal
  state and raw MLX kernel changes.
- DeepSeek L32 is a recorded regression. The default should not be described as
  faster for every context length.
- Gemma4 bf16 is effectively parity in the retained historical run.
- Peak-memory changes are small for most 4B rows but are not uniformly zero.

## Correctness evidence retained with those runs

| Model/path | Recorded check |
| --- | --- |
| Qwen2.5 0.5B q4 | same argmax and top-5; max abs logit diff 0.0327148, mean 0.00475509 |
| Qwen3.5 4B q4/q8/bf16 | same argmax in optimization checks; custom task coverage reported 0 native fallback |
| DeepSeek V2 Lite q4 | same argmax after MLA/cache fixes; stable L512 record used full custom dispatch |
| Gemma4 q4 accepted qdot1 path | forced-prefix top-1 matched for 80 decode steps |

These are targeted regression checks, not proofs of bitwise equivalence. Metal
fusion changes operation order, so small floating-point differences are
expected. A release run should archive token sequences plus max/mean logit
error for each advertised model.

## Experimental MTP results

MTP numbers are not comparable to ordinary single-token decode. They are shown
to preserve mixed outcomes rather than select only wins.

| Model | L | Raw MTP TPS | FuseKit MTP TPS | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | 512 | 35.453 | 60.761 | +71.4% | model/format-specific smoke |
| `corysus/Qwen3.5-4B-oQ4-mtp` | 512 | 57.393 | 57.092 | -0.5% | reject/weak |
| `fcobos/Qwen3.5-0.8B-oQ4-mtp` | 512 | 167.575 | 182.514 | +8.9% | noisy |
| `fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp` | 512 | 37.991 | 34.018 | -10.5% | reject |

DFlash and prefill experiments are intentionally excluded from the headline
decode table. DFlash uses a separate draft-model path, while no complete
persisted prefill table was recoverable. See
[`docs/benchmark-history-summary.md`](docs/benchmark-history-summary.md) for
the full historical inventory, including rejected candidates.

## Reproducing a release benchmark

First patch the model with the exact binary being measured:

```bash
cargo build --release --locked
./target/release/fusekit patch mlx-community/Qwen3.5-4B-4bit --force

python3 scripts/bench_suite.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --tokens 32 512 \
  --warmup 1 \
  --repeat 5 \
  --cooldown 2 \
  --save-logs \
  --note "release candidate"
```

The suite:

1. records host, binary, Git revision, MLX, and `mlx-lm` versions;
2. runs raw first and FuseKit second for every `(model, L)` pair;
3. excludes warmup repetitions from averages;
4. checks generated-token counts and required metrics;
5. rejects a FuseKit run if the runtime did not attach or reported fallback;
6. writes JSONL, a Markdown summary, and optional raw logs under
   `benchmark-results/`.

Add `--model` repeatedly to test more models. A benchmark is release evidence
only when the tree is clean, the patched manifest version matches the binary,
correctness passes, and the JSONL/log files are archived with the release.

## Provenance limitation

Most old raw JSONL files were written under `/private/tmp` and no longer exist.
The detailed Markdown optimization logs are therefore the recoverable source
of truth for the historical table. Rows whose only source was unarchived session
notes were omitted from the headline table above. Fresh suite output should
replace these historical rows before a stable release tag.
