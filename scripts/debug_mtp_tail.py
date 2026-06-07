"""Compare raw oMLX and FuseKit MTP tail outputs.

This isolates final hidden, final RMSNorm and tied lm-head logits for a
two-token native-MTP verify call.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.model_settings import ModelSettings
from omlx.patches.mlx_lm_mtp import batch_generator as mtp_bg
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def _advance_cache(model, cache, tokens: list[int]) -> None:
    for tok in tokens:
        out = model(mx.array([[tok]], dtype=mx.uint32), cache=cache, return_hidden=True)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)


def _text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def _as_array(x):
    ptr = getattr(x, "_fk_ptr", None)
    if ptr is None:
        return x
    shape = tuple(int(v) for v in getattr(x, "shape", ()))
    n = math.prod(shape) if shape else 1
    dtype = str(getattr(x, "dtype", ""))
    if "bfloat16" in dtype:
        raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(int(ptr))).copy()
        arr = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return mx.array(arr, dtype=mx.float32)
    if "float16" in dtype:
        raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(int(ptr))).copy()
        return mx.array(raw.view(np.float16).reshape(shape), dtype=mx.float16)
    if "float32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.float32)
    raise TypeError(f"unsupported native scratch dtype: {dtype}")


def _view_rows(x, rows: int):
    x = _as_array(x)
    if x.ndim == 3:
        return x[:, :rows, :]
    if x.ndim == 2:
        return x[:rows, :].reshape(1, rows, -1)
    return x.reshape(1, rows, -1)


def _diff(a, b) -> tuple[float, float, int]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return (
        float(mx.max(d).item()),
        float(mx.mean(d).item()),
        int(mx.sum(d > 0.01).item()),
    )


def _print_cmp(name: str, raw, fk, rows: int) -> None:
    raw_rows = _view_rows(raw, rows).astype(mx.float32)
    fk_rows = _view_rows(fk, rows).astype(mx.float32)
    mx.eval(raw_rows, fk_rows)
    max_abs, mean_abs, gt = _diff(raw_rows, fk_rows)
    print(
        "[mtp-tail] "
        f"{name:<18} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"gt_0_01={gt} raw_shape={tuple(raw_rows.shape)} fk_shape={tuple(fk_rows.shape)}",
        flush=True,
    )


def _top_pos(logits, pos: int, k: int = 5) -> list[tuple[int, float]]:
    row = logits[:, pos, :].reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
    idx = mx.argpartition(-row, kth=k - 1)[:k]
    mx.eval(idx, row)
    return sorted(
        ((int(i), float(row[int(i)].item())) for i in idx.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )


def _tail_sids(variant) -> tuple[int, int, int]:
    tasks = variant._tg["tasks"]
    for task in tasks:
        kernel = str(task.get("kernel", ""))
        if "fusekit_op_quantizedmatmul_" not in kernel:
            continue
        outs = task.get("outputs", [])
        ins = task.get("inputs", [])
        shape = outs[0].get("shape") if outs else None
        if shape and int(shape[-1]) == int(variant._vocab):
            norm_sid = int(ins[0]["scratch_id"])
            logits_sid = int(outs[0]["scratch_id"])
            norm_task_id = None
            for dep in task.get("depends_on", []):
                dep_task = tasks[int(dep)]
                if "fusekit_op_rmsnorm_" in str(dep_task.get("kernel", "")):
                    norm_task_id = int(dep)
                    break
            if norm_task_id is None:
                raise RuntimeError("tail RMSNorm dependency not found")
            norm_task = tasks[norm_task_id]
            hidden_sid = int(norm_task["inputs"][0]["scratch_id"])
            return hidden_sid, norm_sid, logits_sid
    raise RuntimeError("tail quantized lm-head task not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--main", type=int, required=True)
    ap.add_argument("--draft", type=int, required=True)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    text = _text_model(raw_model)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()
    _ = raw_model(prompt, cache=cache_raw)
    _ = raw_model(prompt, cache=cache_fk)
    prefix_tokens = _parse_tokens(args.prefix)
    _advance_cache(raw_model, cache_raw, prefix_tokens)
    _advance_cache(raw_model, cache_fk, prefix_tokens)

    verify = mx.array([[args.main, args.draft]], dtype=mx.uint32)
    logits_raw, hidden_raw, _ = mtp_bg._call_backbone(
        raw_model, verify, cache_raw, n_confirmed=1
    )
    logits_fk, hidden_fk, _ = mtp_bg._call_backbone(
        model, verify, cache_fk, n_confirmed=1
    )
    mx.eval(logits_raw, hidden_raw, logits_fk, hidden_fk)

    variant = model._get_variant_dispatcher(2)
    if variant is None:
        raise RuntimeError("MTP M=2 variant dispatcher was not loaded")
    hidden_sid, norm_sid, logits_sid = _tail_sids(variant)

    raw_norm = text.model.norm(hidden_raw)
    if getattr(text.args, "tie_word_embeddings", False):
        raw_logits_from_hidden = text.model.embed_tokens.as_linear(raw_norm)
    else:
        raw_logits_from_hidden = text.lm_head(raw_norm)
    mx.eval(raw_norm, raw_logits_from_hidden)

    print(
        "[mtp-tail] "
        f"sids hidden={hidden_sid} norm={norm_sid} logits={logits_sid} "
        f"raw_top0={_top_pos(logits_raw, 0)} fk_top0={_top_pos(logits_fk, 0)} "
        f"raw_top1={_top_pos(logits_raw, 1)} fk_top1={_top_pos(logits_fk, 1)}",
        flush=True,
    )
    _print_cmp("hidden_return", hidden_raw, hidden_fk, 2)
    _print_cmp("hidden_scratch", hidden_raw, variant._scratch_buf(hidden_sid), 2)
    _print_cmp("norm_scratch", raw_norm, variant._scratch_buf(norm_sid), 2)
    _print_cmp("logits_return", logits_raw, logits_fk, 2)
    _print_cmp("logits_from_raw", raw_logits_from_hidden, variant._scratch_buf(logits_sid), 2)

    print(
        "[mtp-tail] "
        f"raw_logits_recompute_top0={_top_pos(raw_logits_from_hidden, 0)} "
        f"raw_logits_recompute_top1={_top_pos(raw_logits_from_hidden, 1)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
