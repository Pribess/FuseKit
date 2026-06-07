"""Capture a late mlx-lm generation window into a Metal .gputrace.

The process must be started with the desired FuseKit hook environment:
raw baselines need FUSEKIT_DISABLE_HOOK=1 before Python startup, while
FuseKit captures should leave the hook enabled.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-tokens", type=int, default=448)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="Repeat the word token forever: token token token",
    )
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    total_tokens = args.warmup_tokens + args.tokens
    print(
        "[capture-mlx-lm] "
        f"hook_disable={os.environ.get('FUSEKIT_DISABLE_HOOK', '')!r} "
        f"model={args.model_dir} warmup={args.warmup_tokens} "
        f"capture_tokens={args.tokens} out={args.out}",
        flush=True,
    )

    started = time.perf_counter()
    model, tokenizer = load(args.model_dir)
    mx.synchronize()
    print(f"[capture-mlx-lm] load_s={time.perf_counter() - started:.3f}", flush=True)

    tokenizer.eos_token_ids = set()

    capture_started = False
    capture_start_time = 0.0
    run_started = time.perf_counter()
    last = None
    for last in stream_generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=total_tokens,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
    ):
        generated_now = int(last.generation_tokens)
        if not capture_started and generated_now >= args.warmup_tokens:
            # Flush prior lazy work so the capture is focused on the late window.
            mx.synchronize()
            mx.metal.start_capture(args.out)
            capture_started = True
            capture_start_time = time.perf_counter()
            print(
                f"[capture-mlx-lm] capture_start tokens={generated_now}",
                flush=True,
            )
        if capture_started and generated_now >= total_tokens:
            mx.synchronize()
            mx.metal.stop_capture()
            dt = time.perf_counter() - capture_start_time
            print(
                f"[capture-mlx-lm] capture_stop tokens={generated_now} "
                f"capture_wall_s={dt:.6f} "
                f"capture_tps={args.tokens / dt if dt else 0:.3f}",
                flush=True,
            )
            break

    mx.synchronize()
    wall_s = time.perf_counter() - run_started
    generated = int(last.generation_tokens if last is not None else 0)
    print(
        f"[capture-mlx-lm] generated_tokens={generated} wall_s={wall_s:.6f} "
        f"wall_tps={generated / wall_s if wall_s else 0:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
