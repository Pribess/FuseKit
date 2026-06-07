"""Capture one oMLX engine generation into a Metal .gputrace.

The process must be started with the desired FuseKit hook environment:
raw baselines need FUSEKIT_DISABLE_HOOK=1 before Python startup, while
FuseKit captures should leave the hook enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import mlx.core as mx

from omlx.engine.batched import BatchedEngine
from omlx.model_settings import ModelSettings
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


async def _generate(engine, prompt: str, tokens: int, seed: int | None) -> tuple[int, float]:
    started = time.perf_counter()
    last = None
    async for last in engine.stream_generate(
        prompt,
        max_tokens=tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=[],
        seed=seed,
    ):
        pass
    mx.synchronize()
    dt = time.perf_counter() - started
    generated = int(last.completion_tokens if last is not None else 0)
    return generated, dt


async def main_async(args: argparse.Namespace) -> int:
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    engine = BatchedEngine(
        model_name=args.model,
        scheduler_config=SchedulerConfig(max_num_seqs=1),
        stream_interval=1,
        model_settings=ModelSettings(mtp_enabled=True) if args.mtp else None,
    )
    await engine.start()
    _set_no_eos(engine)
    mx.synchronize()

    try:
        if args.warmup_tokens > 0:
            generated, dt = await _generate(engine, args.prompt, args.warmup_tokens, args.seed)
            print(
                f"[capture:warmup] generated={generated} wall_s={dt:.6f} "
                f"tps={(generated / dt if dt else 0):.3f}",
                flush=True,
            )

        mx.metal.start_capture(args.out)
        generated, dt = await _generate(engine, args.prompt, args.tokens, args.seed)
        mx.metal.stop_capture()
        print(
            f"[capture:run] generated={generated} wall_s={dt:.6f} "
            f"tps={(generated / dt if dt else 0):.3f} out={args.out}",
            flush=True,
        )
    finally:
        await engine.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mtp", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
