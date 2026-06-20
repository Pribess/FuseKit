"""Capture one oMLX engine decode step with MLX Metal capture.

The capture starts after `--capture-after` tokens have already been yielded,
so `--capture-after 511 --capture-tokens 1` captures the 512th generated token.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path


def _raw_mode_requested(argv: list[str]) -> bool:
    for i, arg in enumerate(argv):
        if arg == "--mode" and i + 1 < len(argv):
            return argv[i + 1] == "engine_raw"
        if arg == "--mode=engine_raw":
            return True
    return False


if _raw_mode_requested(sys.argv[1:]) and not os.environ.get("FUSEKIT_DISABLE_HOOK"):
    env = os.environ.copy()
    env["FUSEKIT_DISABLE_HOOK"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


import mlx.core as mx

from omlx.engine.batched import BatchedEngine
from omlx.scheduler import SchedulerConfig


def _set_no_eos(engine) -> None:
    tokenizer = engine.tokenizer
    for attr, value in (("eos_token_ids", set()), ("eos_token_id", None)):
        try:
            setattr(tokenizer, attr, value)
        except Exception:
            pass
    try:
        scheduler = engine._engine.engine.scheduler
        scheduler._generation_config_eos = set()
        scheduler._output_parser_factory = None
        scheduler._get_stop_tokens = lambda: set()
    except Exception:
        pass


async def run(args: argparse.Namespace) -> int:
    if args.mode == "engine_raw":
        os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
        os.environ["FUSEKIT_DISABLE_FAST"] = "1"
    else:
        os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
        os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    engine = BatchedEngine(
        model_name=args.model,
        scheduler_config=SchedulerConfig(max_num_seqs=1),
        stream_interval=1,
    )
    await engine.start()
    _set_no_eos(engine)
    mx.synchronize()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    capturing = False
    stop_after = args.capture_after + args.capture_tokens
    try:
        async for result in engine.stream_generate(
            args.prompt,
            max_tokens=stop_after,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            stop=[],
            seed=args.seed,
        ):
            generated = int(result.completion_tokens)
            if not capturing and generated >= args.capture_after:
                mx.synchronize()
                mx.metal.start_capture(str(output))
                capturing = True
                print(
                    f"[capture] started after={generated} output={output}",
                    flush=True,
                )
            if capturing and generated >= stop_after:
                mx.synchronize()
                mx.metal.stop_capture()
                capturing = False
                print(f"[capture] stopped generated={generated}", flush=True)
                break
    finally:
        if capturing:
            mx.synchronize()
            mx.metal.stop_capture()
        await engine.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode", choices=("engine_raw", "engine_fusekit"), required=True)
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--capture-after", type=int, default=511)
    parser.add_argument("--capture-tokens", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
