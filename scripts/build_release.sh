#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

"$ROOT/scripts/release_check.sh" --smoke

version="$(./target/release/fusekit --version | awk '{print $2}')"
name="fusekit-v${version}-aarch64-apple-darwin"
stage="$ROOT/dist/$name"
archive="$ROOT/dist/$name.tar.gz"

rm -rf "$stage" "$archive" "$archive.sha256"
mkdir -p "$stage"
cp target/release/fusekit "$stage/"
cp README.md BENCHMARKS.md CHANGELOG.md LICENSE-APACHE LICENSE-MIT "$stage/"
mkdir -p "$stage/docs"
cp docs/architecture.md docs/benchmark-history-summary.md docs/fusion-algorithms.md "$stage/docs/"

# Preserve the linker-generated ad-hoc signature and verify the Mach-O before
# archiving. Official releases can replace this with a Developer ID signature.
codesign --verify "$stage/fusekit"
file "$stage/fusekit" | grep -q 'Mach-O 64-bit executable arm64'

tar -C "$ROOT/dist" -czf "$archive" "$name"
(
  cd "$ROOT/dist"
  shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256"
  shasum -a 256 -c "$(basename "$archive").sha256"
)
rm -rf "$stage"

echo "wrote $archive"
echo "wrote $archive.sha256"
