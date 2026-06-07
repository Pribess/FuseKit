"""Probe mx.compile on native-MTP head argmax.

This is intentionally separate from the runtime: MTP cache state is mutable,
so an optimization is only usable if the compiled function both updates cache
state and returns the same greedy draft token as the raw head.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

OMLX_SITE = Path(
    "/Applications/oMLX.app/Contents/Python/framework-mlx-framework/"
    "lib/python3.11/site-packages"
)
if OMLX_SITE.exists():
    sys.path.insert(0, str(OMLX_SITE))

os.environ.setdefault("FUSEKIT_DISABLE_HOOK", "1")
os.environ.setdefault("FUSEKIT_DISABLE_DISPATCH", "1")
os.environ.setdefault("FUSEKIT_DISABLE_FAST", "1")

import mlx.core as mx

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _cache_state(cache) -> list:
    state = []
    for c in cache or []:
        value = getattr(c, "state", None)
        if value is not None:
            state.append(value)
    return state


def _prepare(model, tokenizer, prompt_text: str):
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache = model.make_cache()
    logits = model(prompt, cache=cache)
    first = _argmax_id(logits)

    tok0 = mx.array([[first]], dtype=mx.uint32)
    logits0, hidden0 = model(tok0, cache=cache, return_hidden=True)
    next_main = _argmax_id(logits0)
    mtp_cache = model.make_mtp_cache()
    hidden = hidden0[:, -1:, :]
    next_ids = mx.array([[next_main]], dtype=mx.uint32)
    mx.synchronize()
    return hidden, next_ids, mtp_cache


def _run_raw(model, hidden, next_ids, mtp_cache):
    logits = model.mtp_forward(hidden, next_ids, mtp_cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    mx.eval(tok)
    return tok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--reuse-compiled",
        action="store_true",
        help="Reuse one compiled MTP-head argmax function across repeated cache updates.",
    )
    args = parser.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    model = getattr(model, "_fk_original", model)

    raw_us = []
    compiled_us = []
    mismatches = 0
    compile_error = None

    # Build the compiled function against one cache instance to verify that
    # cache state capture/update works for this model's native-MTP cache.
    hidden_c, next_ids_c, mtp_cache_c = _prepare(model, tokenizer, args.prompt)
    cache_state = _cache_state(mtp_cache_c)

    def compiled_body(hidden, next_ids):
        logits = model.mtp_forward(hidden, next_ids, mtp_cache_c)
        return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

    try:
        compiled = mx.compile(compiled_body, inputs=cache_state, outputs=cache_state)
        for _ in range(args.warmup):
            tok = compiled(hidden_c, next_ids_c)
            mx.eval(tok)
        mx.synchronize()
    except Exception as exc:
        compiled = None
        compile_error = repr(exc)

    if args.reuse_compiled and compiled is not None:
        raw_hidden, raw_next_ids, raw_cache = _prepare(model, tokenizer, args.prompt)
        compiled_hidden, compiled_next_ids, compiled_cache = _prepare(
            model, tokenizer, args.prompt
        )

        # Runtime-equivalent setup: _post_init_mtp creates the first draft with
        # the raw head, which allocates/advances the MTP KV cache once. Compile
        # only after the cache has real state, otherwise KVCache.state is empty
        # and mx.compile cannot track the mutation contract.
        raw_boot = _run_raw(model, raw_hidden, raw_next_ids, raw_cache)
        compiled_boot = _run_raw(model, compiled_hidden, compiled_next_ids, compiled_cache)
        mx.synchronize()
        if int(raw_boot.item()) != int(compiled_boot.item()):
            mismatches += 1
        raw_next_ids = raw_boot.reshape(1, 1)
        compiled_next_ids = compiled_boot.reshape(1, 1)

        compiled_state = _cache_state(compiled_cache)

        def compiled_reuse_body(hidden, next_ids):
            logits = model.mtp_forward(hidden, next_ids, compiled_cache)
            return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

        compiled_reuse = mx.compile(
            compiled_reuse_body, inputs=compiled_state, outputs=compiled_state
        )

        for _ in range(args.repeat):
            t0 = time.perf_counter()
            raw_tok = _run_raw(model, raw_hidden, raw_next_ids, raw_cache)
            mx.synchronize()
            raw_us.append((time.perf_counter() - t0) * 1_000_000.0)

            t0 = time.perf_counter()
            compiled_tok = compiled_reuse(compiled_hidden, compiled_next_ids)
            mx.eval(compiled_tok)
            mx.synchronize()
            compiled_us.append((time.perf_counter() - t0) * 1_000_000.0)

            if int(raw_tok.item()) != int(compiled_tok.item()):
                mismatches += 1
            raw_next_ids = raw_tok.reshape(1, 1)
            compiled_next_ids = compiled_tok.reshape(1, 1)

        def avg(xs: list[float]) -> float:
            return sum(xs) / max(1, len(xs))

        print(
            "[mtp-head-compile-probe] "
            f"repeat={args.repeat} reuse_compiled=True "
            f"raw_avg_us={avg(raw_us):.2f} "
            f"compiled_avg_us={avg(compiled_us):.2f} "
            f"mismatches={mismatches} compile_error={compile_error}",
            flush=True,
        )
        return 0

    for _ in range(args.repeat):
        hidden, next_ids, mtp_cache = _prepare(model, tokenizer, args.prompt)
        t0 = time.perf_counter()
        raw_tok = _run_raw(model, hidden, next_ids, mtp_cache)
        mx.synchronize()
        raw_us.append((time.perf_counter() - t0) * 1_000_000.0)

        if compiled is None:
            continue

        # Rebuild compiled state for an apples-to-apples cache, otherwise the
        # compiled closure would keep advancing its warmup cache.
        hidden_c, next_ids_c, mtp_cache_c = _prepare(model, tokenizer, args.prompt)
        cache_state = _cache_state(mtp_cache_c)

        def compiled_body2(hidden, next_ids):
            logits = model.mtp_forward(hidden, next_ids, mtp_cache_c)
            return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

        compiled2 = mx.compile(compiled_body2, inputs=cache_state, outputs=cache_state)
        t0 = time.perf_counter()
        compiled_tok = compiled2(hidden_c, next_ids_c)
        mx.eval(compiled_tok)
        mx.synchronize()
        compiled_us.append((time.perf_counter() - t0) * 1_000_000.0)
        if int(raw_tok.item()) != int(compiled_tok.item()):
            mismatches += 1

    def avg(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    print(
        "[mtp-head-compile-probe] "
        f"repeat={args.repeat} raw_avg_us={avg(raw_us):.2f} "
        f"compiled_avg_us={avg(compiled_us):.2f} "
        f"mismatches={mismatches} compile_error={compile_error}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
