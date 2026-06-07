"""Replay native-MTP cycles on raw and FuseKit backbones side by side."""

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


def _top_pos(logits, pos: int, k: int = 5) -> list[tuple[int, float]]:
    row = logits[:, pos, :].reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
    idx = mx.argpartition(-row, kth=k - 1)[:k]
    mx.eval(idx, row)
    return sorted(
        ((int(i), float(row[int(i)].item())) for i in idx.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )


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


def _draft_from_mtp(model, hidden, next_main: int, mtp_cache):
    next_ids = mx.array([[next_main]], dtype=mx.uint32)
    logits = model.mtp_forward(hidden, next_ids, mtp_cache)
    mx.eval(logits)
    return _argmax_id(logits), logits


def _append_and_check(seq: list[int], emitted: list[int], other: list[int]) -> int | None:
    start = len(seq)
    seq.extend(emitted)
    for i in range(start, len(seq)):
        if i < len(other) and seq[i] != other[i]:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--cycles", type=int, default=16)
    ap.add_argument("--stop-after-divergence", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    print(f"[mtp-cycle] wrapped={raw_model is not model}", flush=True)

    prompt_ids = tokenizer.encode(args.prompt) or [0]
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()

    logits_raw = raw_model(prompt, cache=cache_raw)
    logits_fk = raw_model(prompt, cache=cache_fk)
    mx.eval(logits_raw, logits_fk)
    main_raw = _argmax_id(logits_raw)
    main_fk = _argmax_id(logits_fk)

    tok_raw = mx.array([[main_raw]], dtype=mx.uint32)
    tok_fk = mx.array([[main_fk]], dtype=mx.uint32)
    logits_raw_0, hidden_raw_0, _ = mtp_bg._call_backbone(
        raw_model, tok_raw, cache_raw
    )
    logits_fk_0, hidden_fk_0, _ = mtp_bg._call_backbone(model, tok_fk, cache_fk)
    mx.eval(logits_raw_0, hidden_raw_0, logits_fk_0, hidden_fk_0)
    next_raw = _argmax_id(logits_raw_0)
    next_fk = _argmax_id(logits_fk_0)

    mtp_cache_raw = raw_model.make_mtp_cache()
    mtp_cache_fk = raw_model.make_mtp_cache()
    draft_raw, draft_logits_raw = _draft_from_mtp(
        raw_model, hidden_raw_0[:, -1:, :], next_raw, mtp_cache_raw
    )
    draft_fk, draft_logits_fk = _draft_from_mtp(
        raw_model, hidden_fk_0[:, -1:, :], next_fk, mtp_cache_fk
    )
    draft_max, draft_mean = _diff(draft_logits_raw, draft_logits_fk)

    seq_raw = [main_raw, next_raw]
    seq_fk = [main_fk, next_fk]
    print(
        "[mtp-cycle] "
        f"post_init raw_seq={seq_raw} fk_seq={seq_fk} "
        f"raw_draft={draft_raw} fk_draft={draft_fk} "
        f"draft_logits_max={draft_max:.6f} draft_logits_mean={draft_mean:.6f} "
        f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )

    divergence: int | None = None
    for cycle in range(args.cycles):
        inputs_raw = mx.array([[next_raw, draft_raw]], dtype=mx.uint32)
        inputs_fk = mx.array([[next_fk, draft_fk]], dtype=mx.uint32)
        logits_raw_v, hidden_raw_v, gdn_raw = mtp_bg._call_backbone(
            raw_model,
            inputs_raw,
            cache_raw,
            n_confirmed=1,
        )
        logits_fk_v, hidden_fk_v, gdn_fk = mtp_bg._call_backbone(
            model,
            inputs_fk,
            cache_fk,
            n_confirmed=1,
        )
        mx.eval(logits_raw_v, hidden_raw_v, logits_fk_v, hidden_fk_v)

        raw_verify = _argmax_pos(logits_raw_v, 0)
        raw_bonus = _argmax_pos(logits_raw_v, 1)
        fk_verify = _argmax_pos(logits_fk_v, 0)
        fk_bonus = _argmax_pos(logits_fk_v, 1)
        raw_accept = raw_verify == draft_raw
        fk_accept = fk_verify == draft_fk
        logits_max, logits_mean = _diff(logits_raw_v, logits_fk_v)
        hidden_max, hidden_mean = _diff(hidden_raw_v, hidden_fk_v)

        if raw_accept:
            mtp_bg._clear_rollback(cache_raw)
            emitted_raw = [draft_raw, raw_bonus]
            hidden_for_raw = hidden_raw_v[:, 1:2, :]
            next_for_mtp_raw = raw_bonus
            next_raw = raw_bonus
        else:
            ok = mtp_bg._rollback_after_reject(
                raw_model, cache_raw, gdn_raw, accepted=0, block_size=2
            )
            if not ok:
                raise RuntimeError("raw rollback failed")
            emitted_raw = [raw_verify]
            hidden_for_raw = hidden_raw_v[:, 0:1, :]
            next_for_mtp_raw = raw_verify
            next_raw = raw_verify

        if fk_accept:
            mtp_bg._clear_rollback(cache_fk)
            emitted_fk = [draft_fk, fk_bonus]
            hidden_for_fk = hidden_fk_v[:, 1:2, :]
            next_for_mtp_fk = fk_bonus
            next_fk = fk_bonus
        else:
            ok = mtp_bg._rollback_after_reject(
                model, cache_fk, gdn_fk, accepted=0, block_size=2
            )
            if not ok:
                raise RuntimeError("FuseKit rollback failed")
            emitted_fk = [fk_verify]
            hidden_for_fk = hidden_fk_v[:, 0:1, :]
            next_for_mtp_fk = fk_verify
            next_fk = fk_verify

        div_raw = _append_and_check(seq_raw, emitted_raw, seq_fk)
        div_fk = _append_and_check(seq_fk, emitted_fk, seq_raw)
        divergence = div_raw if div_raw is not None else div_fk

        draft_raw, draft_logits_raw = _draft_from_mtp(
            raw_model, hidden_for_raw, next_for_mtp_raw, mtp_cache_raw
        )
        draft_fk, draft_logits_fk = _draft_from_mtp(
            raw_model, hidden_for_fk, next_for_mtp_fk, mtp_cache_fk
        )
        draft_max, draft_mean = _diff(draft_logits_raw, draft_logits_fk)

        print(
            "[mtp-cycle] "
            f"cycle={cycle:02d} raw_in={[int(x) for x in inputs_raw[0].tolist()]} "
            f"fk_in={[int(x) for x in inputs_fk[0].tolist()]} "
            f"raw_verify={raw_verify} fk_verify={fk_verify} "
            f"raw_bonus={raw_bonus} fk_bonus={fk_bonus} "
            f"raw_accept={raw_accept} fk_accept={fk_accept} "
            f"emit_raw={emitted_raw} emit_fk={emitted_fk} "
            f"next_raw={next_raw} next_fk={next_fk} "
            f"draft_raw={draft_raw} draft_fk={draft_fk} "
            f"logits_max={logits_max:.6f} logits_mean={logits_mean:.6f} "
            f"hidden_max={hidden_max:.6f} hidden_mean={hidden_mean:.6f} "
            f"draft_max={draft_max:.6f} draft_mean={draft_mean:.6f} "
            f"divergence={divergence} "
            f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
            flush=True,
        )
        if divergence is not None:
            print(f"[mtp-cycle] raw_seq={seq_raw}", flush=True)
            print(f"[mtp-cycle] fk_seq={seq_fk}", flush=True)
            print(f"[mtp-cycle] raw_verify_top={_top_pos(logits_raw_v, 0)}", flush=True)
            print(f"[mtp-cycle] fk_verify_top={_top_pos(logits_fk_v, 0)}", flush=True)
            print(f"[mtp-cycle] raw_bonus_top={_top_pos(logits_raw_v, 1)}", flush=True)
            print(f"[mtp-cycle] fk_bonus_top={_top_pos(logits_fk_v, 1)}", flush=True)
            if args.stop_after_divergence:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
