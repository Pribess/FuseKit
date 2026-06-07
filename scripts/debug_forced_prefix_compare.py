"""Compare raw vs FuseKit logits while forcing a generated token prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _diff(a, b) -> tuple[float, float]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def _top(logits, k: int = 5) -> list[tuple[int, float]]:
    row = logits.reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
    idx = mx.argpartition(-row, kth=k - 1)[:k]
    mx.eval(idx, row)
    return sorted(
        ((int(i), float(row[int(i)].item())) for i in idx.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--tokens", required=True, help="comma/space separated forced tokens")
    ap.add_argument("--stop-after-mismatch", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    print(f"[forced-prefix] wrapped={raw_model is not model}", flush=True)

    forced = _parse_tokens(args.tokens)
    prompt_ids = tokenizer.encode(args.prompt) or [0]
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()

    logits_raw = raw_model(prompt, cache=cache_raw)
    logits_fk = raw_model(prompt, cache=cache_fk)
    mx.eval(logits_raw, logits_fk)
    print(
        "[forced-prefix] "
        f"prompt_tokens={len(prompt_ids)} first_raw={_argmax_id(logits_raw)} "
        f"first_fk={_argmax_id(logits_fk)} offsets_raw={_cache_offsets(cache_raw)} "
        f"offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )

    for i, token_id in enumerate(forced):
        x = mx.array([[token_id]], dtype=mx.uint32)
        logits_raw = raw_model(x, cache=cache_raw)
        logits_fk = model(x, cache=cache_fk)
        mx.eval(logits_raw, logits_fk)
        max_abs, mean_abs = _diff(logits_raw, logits_fk)
        raw_next = _argmax_id(logits_raw)
        fk_next = _argmax_id(logits_fk)
        expected_next = forced[i + 1] if i + 1 < len(forced) else None
        ok = raw_next == fk_next
        print(
            "[forced-prefix] "
            f"i={i:02d} in={token_id} expected_next={expected_next} "
            f"raw_next={raw_next} fk_next={fk_next} match={ok} "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
            f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
            flush=True,
        )
        if not ok:
            print(f"[forced-prefix] raw_top={_top(logits_raw)}", flush=True)
            print(f"[forced-prefix] fk_top={_top(logits_fk)}", flush=True)
            if args.stop_after_mismatch:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
