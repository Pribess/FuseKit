# Changelog

All notable user-facing changes are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning once a stable public API exists.

## [Unreleased]

## [0.0.1] - 2026-11-02

### Added

- General semantic fusion and Metal code generation for dense, quantized,
  attention, routed-MoE, and pointwise decode paths.
- Model-local Python runtime plus embedded Rust/Metal dispatcher.
- ICB-based decode execution, compiled prefill/MTP variants, profiling, and
  raw-vs-FuseKit benchmark commands.
- Reproducible benchmark suite with warmup exclusion, fallback validation,
  JSONL records, Markdown reports, and environment provenance.
- Release-check and release-archive scripts.

### Changed

- `fusekit bench` and `fusekit profile` now use Python drivers embedded in the
  executable, so a release binary no longer depends on a neighboring
  `scripts/` directory.
- Build discovery honors `FUSEKIT_MLX_APP`, `FUSEKIT_PYTHON_ROOT`, and
  `FUSEKIT_LIBPYTHON` instead of assuming only one source-build path.
- Release documentation now distinguishes fresh build/smoke checks from
  historical model throughput.

### Fixed

- Removed stale documentation that described the runtime as a future stub.
- Removed avoidable compiler warnings and documented the legacy objc cfg probe.
- Added the MIT and Apache-2.0 license texts declared by the manifests.

### Known limitations

- Apple-silicon macOS and oMLX's bundled Python 3.11 are required.
- Current model throughput must be requalified before tagging because model
  weights were unavailable during the release cleanup.
- MTP, DFlash, prefill variants, and tuning environment variables remain
  experimental.
- Numerical output is close to, but not guaranteed bit-identical with, raw MLX.
- `metal-rs` currently pulls `block 0.1.6`, which emits Cargo's upstream
  future-incompatibility notice even though project warnings are clean.
