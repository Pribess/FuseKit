# FuseKit

FuseKit captures an `mlx-lm` model graph, fuses compatible operations, emits
Apple Metal kernels, and installs a model-local runtime that executes the
decode task graph through an indirect command buffer (ICB).

> **Status: experimental preview.** The compiler/runtime is functional and has
> measured wins on several model families, but model coverage and numerical
> drift still require per-model validation. Do not treat the benchmark table as
> a universal speedup claim.

## Requirements

- Apple-silicon Mac (arm64)
- macOS 26 or newer, matching the current oMLX Python/MLX bundle
- oMLX.app installed in `/Applications`
- Full Xcode Metal toolchain (`xcrun --find metal` must succeed)
- Rust toolchain and Cargo

FuseKit is currently distributed as source/a macOS binary, not as a crates.io
package. The resulting executable dynamically loads oMLX's bundled
`libpython3.11.dylib`, so oMLX must remain installed at runtime.

## Build

```bash
cargo build --release --locked
./target/release/fusekit --version
./target/release/fusekit smoke   # hidden Metal bridge smoke test
```

For a nonstandard oMLX location, set both variables before building:

```bash
export FUSEKIT_MLX_APP="$HOME/Applications/oMLX.app"
export PYO3_PYTHON="$FUSEKIT_MLX_APP/Contents/Python/cpython-3.11/bin/python3.11"
cargo build --release --locked
```

## Quick start

Models are discovered under `~/.omlx/models` by default and are addressed by
their relative path (for example,
`mlx-community/Qwen3.5-4B-4bit`).

```bash
# Inspect the local registry and current hook state.
fusekit ls
fusekit status

# Compile a model and install the runtime hook.
fusekit patch mlx-community/Qwen3.5-4B-4bit

# Compare raw mlx-lm with FuseKit. Run raw first.
fusekit bench mlx-community/Qwen3.5-4B-4bit --raw --tokens 512 --repeat 5
fusekit bench mlx-community/Qwen3.5-4B-4bit       --tokens 512 --repeat 5

# Remove model artifacts. The global hook is removed when no patched models remain.
fusekit unpatch mlx-community/Qwen3.5-4B-4bit
```

`patch` writes `<model>/fusekit/` and installs one
`fusekit.pth` file in the bundled Python's user site-packages. It does not edit
oMLX.app. `unpatch` is the supported cleanup path.

Useful overrides:

| Variable | Purpose |
| --- | --- |
| `FUSEKIT_OMLX_DIR` | Model registry root instead of `~/.omlx/models` |
| `FUSEKIT_MLX_APP` | Nonstandard oMLX.app path |
| `FUSEKIT_PYTHON` | Runtime Python executable override |
| `FUSEKIT_MLX_SITE_PACKAGES` | Runtime MLX/`mlx-lm` site-packages override |
| `FUSEKIT_DISABLE_HOOK=1` | Force raw `mlx-lm` before Python starts |
| `FUSEKIT_STRICT_DISPATCH=1` | Fail rather than silently accepting an unsupported dispatch |

Run `fusekit <command> --help` for chip, active-row, KV-cache, profiling, and
compiler options.

## Benchmarks

The table below consolidates the latest useful records already present in the
optimization logs. Runs were measured on Apple M5 with raw oMLX first, then the
FuseKit general-fusion path (`--patterns off`), using wall-clock generated-token
TPS. Valid custom-kernel comparisons reported zero native fallback.

| Model | L | Raw TPS | FuseKit TPS | Delta |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B 4-bit | 32 | 147.270 | 257.725 | **+75.0%** |
| Qwen2.5-Coder-0.5B 4-bit | 512 | 285.860 | 295.054 | **+3.2%** |
| Qwen3.5-4B 4-bit | 32 | 35.518 | 42.965 | **+21.0%** |
| Qwen3.5-4B 4-bit | 512 | 44.641 | 47.399 | **+6.2%** |
| Qwen3.5-4B 8-bit | 32 | 21.819 | 24.503 | **+12.3%** |
| Qwen3.5-4B 8-bit | 512 | 25.464 | 26.081 | **+2.4%** |
| Qwen3.5-4B bf16 | 32 | 12.362 | 13.601 | **+10.0%** |
| Qwen3.5-4B bf16 | 512 | 14.055 | 15.075 | **+7.3%** |
| DeepSeek-Coder-V2-Lite 4-bit | 32 | 62.271 | 57.203 | **-8.1%** |
| DeepSeek-Coder-V2-Lite 4-bit | 512 | 70.681 | 72.892 | **+3.1%** |
| Gemma-4 E4B 4-bit | 512 | 39.591 | 41.172 | **+4.0%** |
| Gemma-4 E4B 8-bit | 512 | 22.209 | 23.882 | **+7.5%** |
| Gemma-4 E4B bf16 | 512 | 13.086 | 13.119 | **+0.2%** |
| Gemma-3 4B 4-bit¹ | 32 | 38.660 | 42.800 | **+10.7%** |
| Gemma-3 4B 4-bit¹ | 512 | 47.145 | 50.334 | **+6.8%** |

¹ Gemma-3 is retained from the session record; the original JSONL was not
persisted. The other rows are recovered from the repository's optimization
logs. Negative and near-parity rows are intentionally shown.

Experimental records are mixed: MTP ranged from **+71.4%** on one
model/format-specific smoke test to **-10.5%** on the 9B DeepSeek-V4 path.
DFlash's best recorded block-4 runs were **3.47×** for Qwen3.5-4B and **1.55×**
for Qwen3.5-9B, but DFlash is a separate draft-model path and is not comparable
to normal decode. No complete persisted prefill table was recoverable.

These values summarize existing records; they were not rerun during this source
cleanup. See **[BENCHMARKS.md](BENCHMARKS.md)** for peak memory, correctness
evidence, methodology, caveats, and **[the detailed history](docs/benchmark-history-summary.md)**
for individual experiments and rejected candidates.

## Development and release checks

```bash
./scripts/release_check.sh

# Requires a previously patched local model; writes JSONL + Markdown.
python3 scripts/bench_suite.py \
  --model mlx-community/Qwen3.5-4B-4bit \
  --tokens 32 512 --warmup 1 --repeat 5
```

Generated benchmark output goes to `benchmark-results/` and is ignored by Git.
Use `./scripts/build_release.sh` to create a versioned arm64 tarball and SHA-256
checksum under `dist/`.

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmark methodology and results](BENCHMARKS.md)
- [Detailed benchmark history](docs/benchmark-history-summary.md)
- [Fusion algorithms](docs/fusion-algorithms.md)
- [Changelog](CHANGELOG.md)

Optimization logs and `scripts/debug_*` tools are development records, not
stable public APIs.

## Safety and limitations

- Patch only model directories you trust; graph capture imports and executes the
  model through oMLX's Python environment.
- Numerical results are expected to be close to raw MLX, not bit-identical.
  Validate argmax/logits for your model and prompt before relying on output.
- Speedups vary by model, quantization, context length, chip, and thermal state.
  Some measured paths regress and should remain on raw `mlx-lm`.
- MTP, DFlash, prefill variants, and environment tuning knobs are experimental.

## License

Licensed under either Apache-2.0 or MIT, at your option. See
[LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
