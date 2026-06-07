"""Compare oMLX native-MTP backbone calls against FuseKit dispatch.

This is intentionally a direct model-call probe rather than an engine
benchmark. It keeps two independent caches at the same logical state:
one calls the wrapped model's original backbone, the other calls the
FuseKit wrapper. That isolates post-init and verify-cycle divergence.
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _max_abs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _cmp(name: str, raw, fk) -> None:
    mx.eval(raw, fk)
    print(
        f"[cmp] {name} shape={raw.shape} dtype={raw.dtype} "
        f"max_abs={_max_abs(raw, fk):.6f}",
        flush=True,
    )


def _top(name: str, raw, fk, k: int = 5) -> None:
    raw_1d = raw.reshape(-1, raw.shape[-1])[-1].astype(mx.float32)
    fk_1d = fk.reshape(-1, fk.shape[-1])[-1].astype(mx.float32)
    raw_idx = mx.argpartition(-raw_1d, kth=k - 1)[:k]
    fk_idx = mx.argpartition(-fk_1d, kth=k - 1)[:k]
    mx.eval(raw_idx, fk_idx, raw_1d, fk_1d)
    raw_pairs = sorted(
        ((int(i), float(raw_1d[int(i)].item())) for i in raw_idx.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )
    fk_pairs = sorted(
        ((int(i), float(fk_1d[int(i)].item())) for i in fk_idx.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )
    print(f"[top] {name} raw={raw_pairs} fk={fk_pairs}", flush=True)


def _cache_offsets(cache) -> list[int | None]:
    out = []
    for c in cache:
        if hasattr(c, "_idx"):
            out.append(int(getattr(c, "_idx")))
        elif hasattr(c, "offset"):
            off = getattr(c, "offset")
            try:
                out.append(int(off.item()) if hasattr(off, "item") else int(off))
            except Exception:
                out.append(None)
        else:
            out.append(None)
    return out[:8]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="x")
    args = parser.parse_args()

    model, tokenizer = load_text_model(
        args.model,
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    print(f"[debug] wrapped={raw_model is not model}", flush=True)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array([prompt_ids])
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()

    logits_raw = raw_model(prompt, cache=cache_raw)
    logits_fk = raw_model(prompt, cache=cache_fk)
    first_raw = _argmax_id(logits_raw)
    first_fk = _argmax_id(logits_fk)
    print(
        f"[debug] prompt_tokens={len(prompt_ids)} first_raw={first_raw} "
        f"first_fk={first_fk} offsets_raw={_cache_offsets(cache_raw)} "
        f"offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )

    tok0 = mx.array([[first_raw]], dtype=mx.uint32)
    logits_raw_0, hidden_raw_0 = raw_model(tok0, cache=cache_raw, return_hidden=True)
    logits_fk_0, hidden_fk_0 = model(tok0, cache=cache_fk, return_hidden=True)
    _cmp("post_init.logits", logits_raw_0, logits_fk_0)
    _cmp("post_init.hidden", hidden_raw_0, hidden_fk_0)
    next_main = _argmax_id(logits_raw_0)
    next_main_fk = _argmax_id(logits_fk_0)
    print(
        f"[debug] next_main_raw={next_main} next_main_fk={next_main_fk} "
        f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )

    mtp_cache_raw = raw_model.make_mtp_cache()
    mtp_cache_fk = raw_model.make_mtp_cache()
    mtp_ids = mx.array([[next_main]], dtype=mx.uint32)
    mtp_logits_raw = raw_model.mtp_forward(hidden_raw_0[:, -1:, :], mtp_ids, mtp_cache_raw)
    mtp_logits_fk = raw_model.mtp_forward(hidden_fk_0[:, -1:, :], mtp_ids, mtp_cache_fk)
    _cmp("post_init.mtp_logits", mtp_logits_raw, mtp_logits_fk)
    draft = _argmax_id(mtp_logits_raw)
    draft_fk = _argmax_id(mtp_logits_fk)
    print(f"[debug] draft_raw={draft} draft_fk={draft_fk}", flush=True)

    cache_raw_seq = raw_model.make_cache()
    cache_fk_seq = raw_model.make_cache()
    _ = raw_model(prompt, cache=cache_raw_seq)
    _ = raw_model(prompt, cache=cache_fk_seq)
    _ = raw_model(tok0, cache=cache_raw_seq, return_hidden=True)
    _ = raw_model(tok0, cache=cache_fk_seq, return_hidden=True)
    logits_raw_s0, hidden_raw_s0 = raw_model(
        mx.array([[next_main]], dtype=mx.uint32),
        cache=cache_raw_seq,
        return_hidden=True,
    )
    logits_fk_s0, hidden_fk_s0 = model(
        mx.array([[next_main]], dtype=mx.uint32),
        cache=cache_fk_seq,
        return_hidden=True,
    )
    _cmp("seq_verify0.logits", logits_raw_s0, logits_fk_s0)
    _top("seq_verify0.logits", logits_raw_s0, logits_fk_s0)
    _cmp("seq_verify0.hidden", hidden_raw_s0, hidden_fk_s0)
    logits_raw_s1, hidden_raw_s1 = raw_model(
        mx.array([[draft]], dtype=mx.uint32),
        cache=cache_raw_seq,
        return_hidden=True,
    )
    logits_fk_s1, hidden_fk_s1 = model(
        mx.array([[draft]], dtype=mx.uint32),
        cache=cache_fk_seq,
        return_hidden=True,
    )
    _cmp("seq_verify1.logits", logits_raw_s1, logits_fk_s1)
    _cmp("seq_verify1.hidden", hidden_raw_s1, hidden_fk_s1)
    print(
        f"[debug] seq_argmax_raw={[ _argmax_id(logits_raw_s0), _argmax_id(logits_raw_s1) ]} "
        f"seq_argmax_fk={[ _argmax_id(logits_fk_s0), _argmax_id(logits_fk_s1) ]}",
        flush=True,
    )

    verify = mx.array([[next_main, draft]], dtype=mx.uint32)
    logits_raw_v, hidden_raw_v = raw_model(
        verify,
        cache=cache_raw,
        return_hidden=True,
        n_confirmed=1,
    )
    logits_fk_v, hidden_fk_v = model(
        verify,
        cache=cache_fk,
        return_hidden=True,
        n_confirmed=1,
    )
    _cmp("verify.logits", logits_raw_v, logits_fk_v)
    _cmp("verify.hidden", hidden_raw_v, hidden_fk_v)
    print(
        f"[debug] verify_argmax_raw={[ _argmax_id(logits_raw_v[:, i:i+1, :]) for i in range(2) ]} "
        f"verify_argmax_fk={[ _argmax_id(logits_fk_v[:, i:i+1, :]) for i in range(2) ]} "
        f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
