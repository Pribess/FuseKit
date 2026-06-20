# Scripts

The public CLI embeds the two Python drivers it needs at runtime. Files in this
directory are primarily development and release tooling; they are not installed
next to the binary.

## Release and benchmark tooling

- `release_check.sh` — formatting, Python syntax, Rust tests, focused Clippy,
  release build, and optional Metal smoke test.
- `build_release.sh` — builds a checked arm64 tarball and SHA-256 checksum.
- `bench.py` — fixed-length generation driver embedded by `fusekit bench`.
- `bench_suite.py` — strict raw-first paired benchmark suite; emits JSONL and
  Markdown.
- `bench_prefill.py` — experimental prefill harness.
- `baseline_infer.py` — workload embedded by `fusekit profile`.

## Correctness tooling

- `check_correctness.py` — raw/patched logit comparison.
- `check_layer_correctness.py` — layer-level comparison and diagnostics.
- `check_omlx_engine_correctness.py` — oMLX engine-path comparison.
- `verify_*.py` — focused kernel/pattern verification helpers.

## Internal experiments

`debug_*`, `probe_*`, `profile_*`, `replace_*`, and one-off comparison scripts
are retained to make optimization decisions reproducible. Their arguments and
output formats are unstable. Rejected experiments and measured decisions are
recorded under `docs/*optimization-log.md`.
