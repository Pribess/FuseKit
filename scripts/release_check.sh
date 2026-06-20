#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_smoke=0
if [[ "${1:-}" == "--smoke" ]]; then
  run_smoke=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--smoke]" >&2
  exit 2
fi

PYTHON_BIN="${FUSEKIT_PYTHON:-/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: Python 3.11 not found at $PYTHON_BIN" >&2
  echo "set FUSEKIT_PYTHON to oMLX's bundled interpreter" >&2
  exit 1
fi

echo "==> rustfmt"
cargo fmt --all -- --check

echo "==> Python syntax"
"$PYTHON_BIN" - "$ROOT" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = sorted((root / "python").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
for path in paths:
    compile(path.read_bytes(), str(path), "exec")
print(f"checked {len(paths)} Python files")
PY

echo "==> Rust tests (warnings are errors)"
RUSTFLAGS="${RUSTFLAGS:-} -D warnings" cargo test --workspace --all-targets --locked

echo "==> Clippy correctness/suspicious groups"
cargo clippy --workspace --all-targets --all-features --locked -- \
  -A warnings -D clippy::correctness -D clippy::suspicious

echo "==> Release build"
RUSTFLAGS="${RUSTFLAGS:-} -D warnings" cargo build --release --locked

./target/release/fusekit --version
./target/release/fusekit --help >/dev/null

if [[ $run_smoke -eq 1 ]]; then
  echo "==> Embedded Python probe"
  ./target/release/fusekit py-probe
  echo "==> Metal smoke test"
  ./target/release/fusekit smoke
fi

echo "release checks passed"
