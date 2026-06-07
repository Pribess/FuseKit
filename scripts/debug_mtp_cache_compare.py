"""Replay MTP cycles and compare raw/FuseKit cache state after each step."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from omlx.model_settings import ModelSettings
from omlx.patches.mlx_lm_mtp import batch_generator as mtp_bg
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _argmax_pos(logits, pos: int) -> int:
    return int(mx.argmax(logits[:, pos, :], axis=-1).item())


def _diff(a, b) -> tuple[float, float]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def _cache_offset(c) -> int:
    if hasattr(c, "_idx"):
        return int(getattr(c, "_idx"))
    if hasattr(c, "offset"):
        off = getattr(c, "offset")
        return int(off.item()) if hasattr(off, "item") else int(off)
    return 0


def _compare_arrays(label: str, a, b) -> tuple[float, float, str] | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return float("inf"), float("inf"), f"{label}: none_mismatch raw={a is None} fk={b is None}"
    if tuple(a.shape) != tuple(b.shape):
        return (
            float("inf"),
            float("inf"),
            f"{label}: shape_mismatch raw_shape={tuple(a.shape)} fk_shape={tuple(b.shape)}",
        )
    max_abs, mean_abs = _diff(a, b)
    return max_abs, mean_abs, (
        f"{label}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} shape={tuple(a.shape)}"
    )


def _cache_diffs(cache_raw, cache_fk, threshold: float) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for li, (cr, cf) in enumerate(zip(cache_raw, cache_fk)):
        kind = type(cr).__name__
        if kind == "ArraysCache":
            for idx, (ar, af) in enumerate(zip(getattr(cr, "cache", []), getattr(cf, "cache", []))):
                res = _compare_arrays(f"layer={li:02d} ArraysCache[{idx}]", ar, af)
                if res is not None and res[0] > threshold:
                    out.append(res)
            rr = getattr(cr, "rollback_state", None)
            rf = getattr(cf, "rollback_state", None)
            if rr is not None or rf is not None:
                if rr is None or rf is None:
                    out.append(
                        (
                            float("inf"),
                            float("inf"),
                            f"layer={li:02d} rollback none_mismatch raw={rr is None} fk={rf is None}",
                        )
                    )
                else:
                    for idx, (ar, af) in enumerate(zip(rr, rf)):
                        res = _compare_arrays(f"layer={li:02d} rollback[{idx}]", ar, af)
                        if res is not None and res[0] > threshold:
                            out.append(res)
        elif hasattr(cr, "keys") and hasattr(cr, "values"):
            ro = _cache_offset(cr)
            fo = _cache_offset(cf)
            if ro != fo:
                out.append((float("inf"), float("inf"), f"layer={li:02d} KV offset raw={ro} fk={fo}"))
            take = min(ro, fo)
            if take > 0:
                res = _compare_arrays(
                    f"layer={li:02d} KV.keys[:{take}]",
                    cr.keys[:, :, :take, :],
                    cf.keys[:, :, :take, :],
                )
                if res is not None and res[0] > threshold:
                    out.append(res)
                res = _compare_arrays(
                    f"layer={li:02d} KV.values[:{take}]",
                    cr.values[:, :, :take, :],
                    cf.values[:, :, :take, :],
                )
                if res is not None and res[0] > threshold:
                    out.append(res)
    out.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return out


def _print_cache(label: str, cache_raw, cache_fk, threshold: float, top: int) -> None:
    diffs = _cache_diffs(cache_raw, cache_fk, threshold)
    print(f"[mtp-cache] {label} diff_count={len(diffs)}", flush=True)
    for _, _, msg in diffs[:top]:
        print(f"[mtp-cache]   {msg}", flush=True)


def _draft_from_mtp(model, hidden, next_main: int, mtp_cache):
    next_ids = mx.array([[next_main]], dtype=mx.uint32)
    logits = model.mtp_forward(hidden, next_ids, mtp_cache)
    mx.eval(logits)
    return _argmax_id(logits), logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)

    prompt_ids = tokenizer.encode(args.prompt) or [0]
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()

    logits_raw = raw_model(prompt, cache=cache_raw)
    logits_fk = raw_model(prompt, cache=cache_fk)
    mx.eval(logits_raw, logits_fk)
    main_raw = _argmax_id(logits_raw)
    main_fk = _argmax_id(logits_fk)
    _print_cache("after_prompt", cache_raw, cache_fk, args.threshold, args.top)

    tok_raw = mx.array([[main_raw]], dtype=mx.uint32)
    tok_fk = mx.array([[main_fk]], dtype=mx.uint32)
    logits_raw_0, hidden_raw_0, _ = mtp_bg._call_backbone(raw_model, tok_raw, cache_raw)
    logits_fk_0, hidden_fk_0, _ = mtp_bg._call_backbone(model, tok_fk, cache_fk)
    mx.eval(logits_raw_0, hidden_raw_0, logits_fk_0, hidden_fk_0)
    next_raw = _argmax_id(logits_raw_0)
    next_fk = _argmax_id(logits_fk_0)
    _print_cache("after_main0", cache_raw, cache_fk, args.threshold, args.top)

    mtp_cache_raw = raw_model.make_mtp_cache()
    mtp_cache_fk = raw_model.make_mtp_cache()
    draft_raw, _ = _draft_from_mtp(raw_model, hidden_raw_0[:, -1:, :], next_raw, mtp_cache_raw)
    draft_fk, _ = _draft_from_mtp(raw_model, hidden_fk_0[:, -1:, :], next_fk, mtp_cache_fk)

    for cycle in range(args.cycles):
        inputs_raw = mx.array([[next_raw, draft_raw]], dtype=mx.uint32)
        inputs_fk = mx.array([[next_fk, draft_fk]], dtype=mx.uint32)
        logits_raw_v, hidden_raw_v, gdn_raw = mtp_bg._call_backbone(
            raw_model, inputs_raw, cache_raw, n_confirmed=1
        )
        logits_fk_v, hidden_fk_v, gdn_fk = mtp_bg._call_backbone(
            model, inputs_fk, cache_fk, n_confirmed=1
        )
        mx.eval(logits_raw_v, hidden_raw_v, logits_fk_v, hidden_fk_v)
        raw_verify = _argmax_pos(logits_raw_v, 0)
        raw_bonus = _argmax_pos(logits_raw_v, 1)
        fk_verify = _argmax_pos(logits_fk_v, 0)
        fk_bonus = _argmax_pos(logits_fk_v, 1)
        raw_accept = raw_verify == draft_raw
        fk_accept = fk_verify == draft_fk
        print(
            "[mtp-cache] "
            f"cycle={cycle:02d} raw_in={[int(x) for x in inputs_raw[0].tolist()]} "
            f"fk_in={[int(x) for x in inputs_fk[0].tolist()]} "
            f"raw_verify={raw_verify} fk_verify={fk_verify} "
            f"raw_bonus={raw_bonus} fk_bonus={fk_bonus} "
            f"raw_accept={raw_accept} fk_accept={fk_accept}",
            flush=True,
        )
        _print_cache(f"cycle={cycle:02d} after_forward", cache_raw, cache_fk, args.threshold, args.top)

        if raw_accept:
            mtp_bg._clear_rollback(cache_raw)
            hidden_for_raw = hidden_raw_v[:, 1:2, :]
            next_for_mtp_raw = raw_bonus
            next_raw = raw_bonus
        else:
            if not mtp_bg._rollback_after_reject(raw_model, cache_raw, gdn_raw, accepted=0, block_size=2):
                raise RuntimeError("raw rollback failed")
            hidden_for_raw = hidden_raw_v[:, 0:1, :]
            next_for_mtp_raw = raw_verify
            next_raw = raw_verify

        if fk_accept:
            mtp_bg._clear_rollback(cache_fk)
            hidden_for_fk = hidden_fk_v[:, 1:2, :]
            next_for_mtp_fk = fk_bonus
            next_fk = fk_bonus
        else:
            if not mtp_bg._rollback_after_reject(model, cache_fk, gdn_fk, accepted=0, block_size=2):
                raise RuntimeError("FuseKit rollback failed")
            hidden_for_fk = hidden_fk_v[:, 0:1, :]
            next_for_mtp_fk = fk_verify
            next_fk = fk_verify
        _print_cache(f"cycle={cycle:02d} after_commit", cache_raw, cache_fk, args.threshold, args.top)

        draft_raw, _ = _draft_from_mtp(raw_model, hidden_for_raw, next_for_mtp_raw, mtp_cache_raw)
        draft_fk, _ = _draft_from_mtp(raw_model, hidden_for_fk, next_for_mtp_fk, mtp_cache_fk)
        if raw_verify != fk_verify or raw_bonus != fk_bonus or raw_accept != fk_accept:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
