"""Probe oMLX native MTP draft agreement against greedy AR tokens."""

from __future__ import annotations

import argparse
import os

import mlx.core as mx

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("FUSEKIT_DISABLE_MTPLX_CONTRACT_FIX", "1")

    model, tokenizer = load_text_model(
        args.model,
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    text = _text_model(model)

    prompt_ids = tokenizer.encode(args.prompt)
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids]), cache=cache)
    first = _argmax_id(logits)

    logits, hidden_pre = model(mx.array([[first]]), cache=cache, return_hidden=True)
    next_token = _argmax_id(logits)
    hidden_pre = hidden_pre[:, -1:, :]

    mtp_cache_pre = model.make_mtp_cache()
    mtp_cache_post = model.make_mtp_cache()
    pre_hits = 0
    post_hits = 0

    print(f"[probe] prompt_tokens={len(prompt_ids)} first={first} next={next_token}")
    for step in range(args.steps):
        hidden_post = text.model.norm(hidden_pre)
        draft_pre = _argmax_id(
            model.mtp_forward(hidden_pre, mx.array([[next_token]]), mtp_cache_pre)
        )
        draft_post = _argmax_id(
            model.mtp_forward(hidden_post, mx.array([[next_token]]), mtp_cache_post)
        )

        logits, next_hidden_pre = model(
            mx.array([[next_token]]),
            cache=cache,
            return_hidden=True,
        )
        target = _argmax_id(logits)
        pre_hits += int(draft_pre == target)
        post_hits += int(draft_post == target)
        print(
            f"[probe:{step}] in={next_token} target={target} "
            f"draft_pre={draft_pre} draft_post={draft_post} "
            f"pre_hit={draft_pre == target} post_hit={draft_post == target}"
        )
        hidden_pre = next_hidden_pre[:, -1:, :]
        next_token = target

    print(
        f"[probe:summary] steps={args.steps} "
        f"pre_hits={pre_hits} post_hits={post_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
