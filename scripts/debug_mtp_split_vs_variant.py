"""Compare FuseKit split-M1 MTP verify against the M=2 variant."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
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


@contextmanager
def _env(**values: str | None):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
        return float("inf"), float("inf"), f"{label}: none split={a is None} variant={b is None}"
    if tuple(a.shape) != tuple(b.shape):
        return (
            float("inf"),
            float("inf"),
            f"{label}: shape split={tuple(a.shape)} variant={tuple(b.shape)}",
        )
    max_abs, mean_abs = _diff(a, b)
    return max_abs, mean_abs, (
        f"{label}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} shape={tuple(a.shape)}"
    )


def _cache_diffs(cache_split, cache_variant, threshold: float):
    out = []
    for li, (cs, cv) in enumerate(zip(cache_split, cache_variant)):
        if type(cs).__name__ == "ArraysCache":
            for idx, (a, b) in enumerate(zip(getattr(cs, "cache", []), getattr(cv, "cache", []))):
                res = _compare_arrays(f"layer={li:02d} ArraysCache[{idx}]", a, b)
                if res is not None and res[0] > threshold:
                    out.append(res)
            rs = getattr(cs, "rollback_state", None)
            rv = getattr(cv, "rollback_state", None)
            if rs is not None or rv is not None:
                if rs is None or rv is None:
                    out.append(
                        (
                            float("inf"),
                            float("inf"),
                            f"layer={li:02d} rollback none split={rs is None} variant={rv is None}",
                        )
                    )
                else:
                    for idx, (a, b) in enumerate(zip(rs, rv)):
                        res = _compare_arrays(f"layer={li:02d} rollback[{idx}]", a, b)
                        if res is not None and res[0] > threshold:
                            out.append(res)
        elif hasattr(cs, "keys") and hasattr(cs, "values"):
            so = _cache_offset(cs)
            vo = _cache_offset(cv)
            if so != vo:
                out.append((float("inf"), float("inf"), f"layer={li:02d} KV offset split={so} variant={vo}"))
            take = min(so, vo)
            if take > 0:
                res = _compare_arrays(
                    f"layer={li:02d} KV.keys[:{take}]",
                    cs.keys[:, :, :take, :],
                    cv.keys[:, :, :take, :],
                )
                if res is not None and res[0] > threshold:
                    out.append(res)
                res = _compare_arrays(
                    f"layer={li:02d} KV.values[:{take}]",
                    cs.values[:, :, :take, :],
                    cv.values[:, :, :take, :],
                )
                if res is not None and res[0] > threshold:
                    out.append(res)
    out.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return out


def _print_cache(label: str, cache_split, cache_variant, threshold: float, top: int) -> None:
    if threshold < 0:
        return
    diffs = _cache_diffs(cache_split, cache_variant, threshold)
    print(f"[mtp-split-vs-variant-cache] {label} diff_count={len(diffs)}", flush=True)
    for _, _, msg in diffs[:top]:
        print(f"[mtp-split-vs-variant-cache]   {msg}", flush=True)


def _print_arrays_cache_rows(label: str, cache_split, cache_variant, layer: int) -> None:
    if layer < 0 or layer >= len(cache_split):
        return
    cs = cache_split[layer]
    cv = cache_variant[layer]
    if type(cs).__name__ != "ArraysCache":
        return
    for idx, (a, b) in enumerate(zip(getattr(cs, "cache", []), getattr(cv, "cache", []))):
        if a is None or b is None or len(a.shape) < 3:
            continue
        rows = min(int(a.shape[1]), int(b.shape[1]))
        for row in range(rows):
            max_abs, mean_abs = _diff(a[:, row : row + 1, :], b[:, row : row + 1, :])
            print(
                "[mtp-split-vs-variant-row] "
                f"{label} layer={layer:02d} cache[{idx}] row={row} "
                f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}",
                flush=True,
            )
    rs = getattr(cs, "rollback_state", None)
    rv = getattr(cv, "rollback_state", None)
    if rs is None or rv is None:
        return
    for idx, (a, b) in enumerate(zip(rs, rv)):
        if a is None or b is None or len(a.shape) < 3:
            continue
        rows = min(int(a.shape[1]), int(b.shape[1]))
        for row in range(rows):
            max_abs, mean_abs = _diff(a[:, row : row + 1, :], b[:, row : row + 1, :])
            print(
                "[mtp-split-vs-variant-row] "
                f"{label} layer={layer:02d} rollback[{idx}] row={row} "
                f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}",
                flush=True,
            )


def _draft_from_mtp(model, hidden, next_main: int, mtp_cache):
    ids = mx.array([[next_main]], dtype=mx.uint32)
    logits = model.mtp_forward(hidden, ids, mtp_cache)
    mx.eval(logits)
    return _argmax_id(logits), logits


def _call_split(model, inputs, cache):
    with _env(FUSEKIT_DISABLE_MTP_MN_VARIANT="1"):
        return mtp_bg._call_backbone(model, inputs, cache, n_confirmed=1)


def _call_variant(model, inputs, cache):
    with _env(
        FUSEKIT_DISABLE_MTP_MN_VARIANT=None,
        FUSEKIT_USE_MTP_MN_VARIANT="1",
        FUSEKIT_EXPERIMENTAL_STATEFUL_MTP_MN_VARIANT="1",
    ):
        return mtp_bg._call_backbone(model, inputs, cache, n_confirmed=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--cycles", type=int, default=16)
    ap.add_argument("--cache-threshold", type=float, default=-1.0)
    ap.add_argument("--cache-top", type=int, default=12)
    ap.add_argument("--row-detail-layer", type=int, default=-1)
    ap.add_argument("--stop-after-divergence", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)

    prompt_ids = tokenizer.encode(args.prompt) or [0]
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_split = raw_model.make_cache()
    cache_variant = raw_model.make_cache()
    _ = raw_model(prompt, cache=cache_split)
    _ = raw_model(prompt, cache=cache_variant)

    first = _argmax_id(raw_model(prompt, cache=raw_model.make_cache()))
    tok0 = mx.array([[first]], dtype=mx.uint32)
    with _env(FUSEKIT_DISABLE_MTP_MN_VARIANT="1"):
        logits_s0, hidden_s0, _ = mtp_bg._call_backbone(model, tok0, cache_split)
        logits_v0, hidden_v0, _ = mtp_bg._call_backbone(model, tok0, cache_variant)
    mx.eval(logits_s0, hidden_s0, logits_v0, hidden_v0)
    next_split = _argmax_id(logits_s0)
    next_variant = _argmax_id(logits_v0)

    mtp_cache_split = raw_model.make_mtp_cache()
    mtp_cache_variant = raw_model.make_mtp_cache()
    draft_split, draft_logits_split = _draft_from_mtp(
        raw_model, hidden_s0[:, -1:, :], next_split, mtp_cache_split
    )
    draft_variant, draft_logits_variant = _draft_from_mtp(
        raw_model, hidden_v0[:, -1:, :], next_variant, mtp_cache_variant
    )
    draft_max, draft_mean = _diff(draft_logits_split, draft_logits_variant)
    seq_split = [first, next_split]
    seq_variant = [first, next_variant]
    print(
        "[mtp-split-vs-variant] "
        f"init first={first} split_next={next_split} variant_next={next_variant} "
        f"split_draft={draft_split} variant_draft={draft_variant} "
        f"draft_max={draft_max:.6f} draft_mean={draft_mean:.6f}",
        flush=True,
    )

    for cycle in range(args.cycles):
        in_split = mx.array([[next_split, draft_split]], dtype=mx.uint32)
        in_variant = mx.array([[next_variant, draft_variant]], dtype=mx.uint32)
        logits_s, hidden_s, gdn_s = _call_split(model, in_split, cache_split)
        logits_v, hidden_v, gdn_v = _call_variant(model, in_variant, cache_variant)
        mx.eval(logits_s, hidden_s, logits_v, hidden_v)

        split_verify = _argmax_pos(logits_s, 0)
        variant_verify = _argmax_pos(logits_v, 0)
        split_bonus = _argmax_pos(logits_s, 1)
        variant_bonus = _argmax_pos(logits_v, 1)
        split_accept = split_verify == draft_split
        variant_accept = variant_verify == draft_variant
        logits_max, logits_mean = _diff(logits_s, logits_v)
        hidden_max, hidden_mean = _diff(hidden_s, hidden_v)
        _print_cache(
            f"cycle={cycle:02d} after_forward",
            cache_split,
            cache_variant,
            args.cache_threshold,
            args.cache_top,
        )
        _print_arrays_cache_rows(
            f"cycle={cycle:02d} after_forward",
            cache_split,
            cache_variant,
            args.row_detail_layer,
        )

        if split_accept:
            mtp_bg._clear_rollback(cache_split)
            emit_split = [draft_split, split_bonus]
            next_split = split_bonus
            hidden_for_split = hidden_s[:, 1:2, :]
            next_for_mtp_split = split_bonus
        else:
            if not mtp_bg._rollback_after_reject(
                model, cache_split, gdn_s, accepted=0, block_size=2
            ):
                raise RuntimeError("split rollback failed")
            emit_split = [split_verify]
            next_split = split_verify
            hidden_for_split = hidden_s[:, 0:1, :]
            next_for_mtp_split = split_verify

        if variant_accept:
            mtp_bg._clear_rollback(cache_variant)
            emit_variant = [draft_variant, variant_bonus]
            next_variant = variant_bonus
            hidden_for_variant = hidden_v[:, 1:2, :]
            next_for_mtp_variant = variant_bonus
        else:
            if not mtp_bg._rollback_after_reject(
                model, cache_variant, gdn_v, accepted=0, block_size=2
            ):
                raise RuntimeError("variant rollback failed")
            emit_variant = [variant_verify]
            next_variant = variant_verify
            hidden_for_variant = hidden_v[:, 0:1, :]
            next_for_mtp_variant = variant_verify

        start = len(seq_split)
        seq_split.extend(emit_split)
        seq_variant.extend(emit_variant)
        divergence = None
        for i in range(start, min(len(seq_split), len(seq_variant))):
            if seq_split[i] != seq_variant[i]:
                divergence = i
                break

        draft_split, draft_logits_split = _draft_from_mtp(
            raw_model, hidden_for_split, next_for_mtp_split, mtp_cache_split
        )
        draft_variant, draft_logits_variant = _draft_from_mtp(
            raw_model, hidden_for_variant, next_for_mtp_variant, mtp_cache_variant
        )
        draft_max, draft_mean = _diff(draft_logits_split, draft_logits_variant)
        _print_cache(
            f"cycle={cycle:02d} after_commit",
            cache_split,
            cache_variant,
            args.cache_threshold,
            args.cache_top,
        )
        _print_arrays_cache_rows(
            f"cycle={cycle:02d} after_commit",
            cache_split,
            cache_variant,
            args.row_detail_layer,
        )
        print(
            "[mtp-split-vs-variant] "
            f"cycle={cycle:02d} split_in={[int(x) for x in in_split[0].tolist()]} "
            f"variant_in={[int(x) for x in in_variant[0].tolist()]} "
            f"split_verify={split_verify} variant_verify={variant_verify} "
            f"split_bonus={split_bonus} variant_bonus={variant_bonus} "
            f"split_accept={split_accept} variant_accept={variant_accept} "
            f"emit_split={emit_split} emit_variant={emit_variant} "
            f"draft_split={draft_split} draft_variant={draft_variant} "
            f"logits_max={logits_max:.6f} logits_mean={logits_mean:.6f} "
            f"hidden_max={hidden_max:.6f} hidden_mean={hidden_mean:.6f} "
            f"draft_max={draft_max:.6f} draft_mean={draft_mean:.6f} "
            f"divergence={divergence}",
            flush=True,
        )
        if divergence is not None and args.stop_after_divergence:
            print(f"[mtp-split-vs-variant] split_seq={seq_split}", flush=True)
            print(f"[mtp-split-vs-variant] variant_seq={seq_variant}", flush=True)
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
