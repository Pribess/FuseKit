#!/usr/bin/env python3
"""Run dflash_mlx.benchmark with controlled runtime monkey patches.

This keeps experiments reversible: installed dflash_mlx files are not edited.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any


def _patch_no_async_draft() -> None:
    from dflash_mlx.draft_backend import EagerDraftBackend

    original = EagerDraftBackend.draft_greedy

    def draft_greedy_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["async_launch"] = False
        return original(self, *args, **kwargs)

    EagerDraftBackend.draft_greedy = draft_greedy_sync


def _patch_verify_mode_adaptive() -> None:
    import dflash_mlx.benchmark as benchmark

    original_config = benchmark.build_offline_runtime_config
    original_context = benchmark.build_offline_runtime_context

    def build_offline_runtime_config_adaptive(*args: Any, **kwargs: Any) -> Any:
        kwargs["verify_mode"] = "adaptive"
        return original_config(*args, **kwargs)

    def build_offline_runtime_context_adaptive(*args: Any, **kwargs: Any) -> Any:
        kwargs["verify_mode"] = "adaptive"
        return original_context(*args, **kwargs)

    benchmark.build_offline_runtime_config = build_offline_runtime_config_adaptive
    benchmark.build_offline_runtime_context = build_offline_runtime_context_adaptive


def _patch_qmm_variant(variant: str, kparts: int) -> None:
    import dflash_mlx.verify_linear as verify_linear
    import dflash_mlx.verify_qmm as verify_qmm

    if variant == "auto":
        return
    if variant == "mma2big":
        forced = ("mma2big", 1)
    elif variant == "mma2big_pipe":
        forced = ("mma2big_pipe", int(kparts))
    else:
        raise ValueError(f"unsupported qMM variant: {variant}")

    def auto_variant_forced(K: int, N: int) -> tuple[str, int]:
        return forced

    verify_linear._auto_variant = auto_variant_forced
    verify_qmm._auto_variant = auto_variant_forced


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run dflash_mlx.benchmark after reversible monkey patches.",
    )
    parser.add_argument(
        "--patch",
        action="append",
        choices=("no-async-draft", "verify-mode-adaptive"),
        default=[],
        help="Patch to apply before running dflash_mlx.benchmark.",
    )
    parser.add_argument(
        "--qmm-variant",
        choices=("auto", "mma2big", "mma2big_pipe"),
        default="auto",
        help="Force verify qMM variant by patching _auto_variant.",
    )
    parser.add_argument(
        "--qmm-kparts",
        type=int,
        default=8,
        help="K_PARTS used when --qmm-variant=mma2big_pipe.",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to dflash_mlx.benchmark. Prefix with --.",
    )
    args = parser.parse_args(argv)
    forwarded = list(args.benchmark_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    if "no-async-draft" in args.patch:
        _patch_no_async_draft()
    if "verify-mode-adaptive" in args.patch:
        _patch_verify_mode_adaptive()
    _patch_qmm_variant(args.qmm_variant, args.qmm_kparts)

    from dflash_mlx.benchmark import main as benchmark_main

    benchmark_main(forwarded, prog="dflash monkeypatch benchmark")


if __name__ == "__main__":
    main(sys.argv[1:])
