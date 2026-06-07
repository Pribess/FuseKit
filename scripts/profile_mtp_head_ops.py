"""Synchronized raw oMLX native-MTP head profiler.

Profiles the draft-head call used after each accepted MTP verify cycle:

    model.mtp_forward(hidden[:, -1:, :], next_token, mtp_cache)

FuseKit currently leaves this path on oMLX/MLX, so these buckets tell us
whether compiling the MTP head into FuseKit is worth doing next.
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

import profile_mlx_ops as raw_prof


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _prepare_head_inputs(model, tokenizer, prompt_text: str):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--include-modules", action="store_true")
    args = parser.parse_args()

    print(
        "[mtp-head-op-profile] "
        f"hook_disable={os.environ.get('FUSEKIT_DISABLE_HOOK', '')!r} "
        f"dispatch_disable={os.environ.get('FUSEKIT_DISABLE_DISPATCH', '')!r} "
        f"repeat={args.repeat} warmup={args.warmup}",
        flush=True,
    )

    raw_prof.install_wrappers(args.include_modules)
    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    model = getattr(model, "_fk_original", model)
    raw_prof.annotate_module_names(model)

    for _ in range(args.warmup):
        hidden, next_ids, mtp_cache = _prepare_head_inputs(model, tokenizer, args.prompt)
        out = model.mtp_forward(hidden, next_ids, mtp_cache)
        raw_prof.force_eval(out)
    mx.synchronize()

    for _ in range(args.repeat):
        hidden, next_ids, mtp_cache = _prepare_head_inputs(model, tokenizer, args.prompt)
        raw_prof.PROF.enabled = True
        started = time.perf_counter()
        out = model.mtp_forward(hidden, next_ids, mtp_cache)
        raw_prof.force_eval(out)
        mx.synchronize()
        raw_prof.PROF.enabled = False
        print(
            f"[mtp-head-op-profile] head_wall_us={(time.perf_counter() - started) * 1_000_000.0:.2f}",
            flush=True,
        )
    raw_prof.summarize(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
