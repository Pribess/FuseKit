"""fusekit benchmark driver.

Runs a fixed-length mlx-lm generation against one model directory. The
Rust CLI controls fusekit/raw/per-kernel mode through environment
variables before Python startup, which is required because the fusekit
hook is installed from a .pth import during interpreter initialization.
"""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="Repeat the word token forever: token token token",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dump-tokens", action="store_true")
    parser.add_argument(
        "--status-every",
        type=int,
        default=0,
        help="Print cumulative and interval wall TPS every N generated tokens.",
    )
    parser.add_argument(
        "--status-synchronize",
        action="store_true",
        help="Synchronize MLX before status timing; useful for profiling, not normal bench.",
    )
    args = parser.parse_args()

    print(
        "[bench] "
        f"hook_disable={os.environ.get('FUSEKIT_DISABLE_HOOK', '')!r} "
        f"dispatch_disable={os.environ.get('FUSEKIT_DISABLE_DISPATCH', '')!r} "
        f"profile_kernels={os.environ.get('FUSEKIT_PROFILE_KERNELS', '')!r} "
        f"fk_chunk={os.environ.get('FK_CHUNK', '')!r}",
        flush=True,
    )

    started = time.perf_counter()
    model, tokenizer = load(args.model_dir)
    mx.synchronize()
    print(f"[bench] load_s={time.perf_counter() - started:.3f}", flush=True)

    # Keep benchmark length fixed; EOS would hide decode cost.
    tokenizer.eos_token_ids = set()

    for rep in range(args.repeat):
        started = time.perf_counter()
        last = None
        tokens = []
        last_status_tokens = 0
        last_status_s = started
        sampler = None
        if args.temperature is not None:
            sampler = make_sampler(
                args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            try:
                sampler.temp = args.temperature
                sampler.temperature = args.temperature
                sampler.top_p = args.top_p
                sampler.top_k = args.top_k
            except Exception:
                pass
        for last in stream_generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.tokens,
            max_kv_size=args.max_kv_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            sampler=sampler,
        ):
            tokens.append(int(last.token))
            generated_now = int(last.generation_tokens)
            if args.status_every and generated_now - last_status_tokens >= args.status_every:
                if args.status_synchronize:
                    mx.synchronize()
                now = time.perf_counter()
                total_s = max(now - started, 1e-9)
                interval_tokens = generated_now - last_status_tokens
                interval_s = max(now - last_status_s, 1e-9)
                print(
                    f"[bench:{rep}:status] tokens={generated_now} "
                    f"wall_s={total_s:.6f} wall_tps={generated_now / total_s:.3f} "
                    f"interval_tokens={interval_tokens} "
                    f"interval_s={interval_s:.6f} "
                    f"interval_tps={interval_tokens / interval_s:.3f}",
                    flush=True,
                )
                last_status_tokens = generated_now
                last_status_s = now
        mx.synchronize()
        wall_s = time.perf_counter() - started
        generated = last.generation_tokens if last is not None else 0
        print(f"[bench:{rep}] prompt_tokens={last.prompt_tokens if last else 0}", flush=True)
        print(f"[bench:{rep}] generated_tokens={generated}", flush=True)
        print(f"[bench:{rep}] stream_tps={last.generation_tps if last else 0:.3f}", flush=True)
        print(f"[bench:{rep}] wall_s={wall_s:.3f}", flush=True)
        print(f"[bench:{rep}] wall_tps={generated / wall_s if wall_s else 0:.3f}", flush=True)
        print(f"[bench:{rep}] peak_gb={last.peak_memory if last else 0:.3f}", flush=True)
        if args.dump_tokens:
            print(f"[bench:{rep}] tokens={tokens}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
