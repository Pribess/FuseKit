"""Compare Qwen3.5 linear-attention Conv1d math variants on one MTP input."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def _advance(model, cache, tokens: list[int]) -> None:
    for tok in tokens:
        out = model(mx.array([[tok]], dtype=mx.uint32), cache=cache, return_hidden=True)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)


def _diff(a, b) -> tuple[float, float, int]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return (
        float(mx.max(d).item()),
        float(mx.mean(d).item()),
        int(mx.sum(d > 0.01).item()),
    )


def _manual(conv_input, weight, rows: int, reverse: bool, cast_each: bool, silu: bool = True):
    parts = []
    k = int(weight.shape[1])
    for kk in range(k):
        wk = k - 1 - kk if reverse else kk
        term = conv_input[:, kk : kk + rows, :] * weight[:, wk, 0]
        parts.append(term)
    acc = parts[0]
    if cast_each:
        acc = acc.astype(conv_input.dtype)
    for term in parts[1:]:
        acc = acc + term
        if cast_each:
            acc = acc.astype(conv_input.dtype)
    return nn.silu(acc) if silu else acc


def _manual_order(conv_input, weight, rows: int, order: tuple[int, ...], tree: bool):
    parts = [
        conv_input[:, kk : kk + rows, :] * weight[:, kk, 0]
        for kk in order
    ]
    if tree and len(parts) == 4:
        acc = (parts[0] + parts[1]) + (parts[2] + parts[3])
    else:
        acc = parts[0]
        for term in parts[1:]:
            acc = acc + term
    return nn.silu(acc)


def _manual_f32(conv_input, weight, rows: int, cast_pre_silu: bool):
    x = conv_input.astype(mx.float32)
    w = weight.astype(mx.float32)
    acc = x[:, 0 : 0 + rows, :] * w[:, 0, 0]
    for kk in range(1, int(weight.shape[1])):
        acc = acc + x[:, kk : kk + rows, :] * w[:, kk, 0]
    if cast_pre_silu:
        acc = acc.astype(conv_input.dtype)
    return nn.silu(acc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--main", type=int, required=True)
    ap.add_argument("--draft", type=int, required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    text = getattr(raw_model, "language_model", raw_model)
    core = text.model

    prompt = mx.array([tokenizer.encode(args.prompt)], dtype=mx.uint32)
    cache = raw_model.make_cache()
    raw_model(prompt, cache=cache)
    _advance(raw_model, cache, _parse_tokens(args.prefix))

    layer = core.layers[args.layer]
    hidden = core.embed_tokens(mx.array([[args.main, args.draft]], dtype=mx.uint32))
    for li, (candidate, c) in enumerate(zip(core.layers, cache)):
        if li == args.layer:
            break
        hidden = candidate(hidden, cache=c, n_confirmed=1)
        mx.eval(hidden)

    att = layer.linear_attn
    norm_x = layer.input_layernorm(hidden)
    qkv = att.in_proj_qkv(norm_x)
    conv_input = mx.concatenate([cache[args.layer][0], qkv], axis=1)
    raw_pre = att.conv1d(conv_input)[:, :2, :]
    raw = nn.silu(raw_pre)
    weight = att.conv1d.weight
    variants = {
        "forward_float": _manual(conv_input, weight, 2, reverse=False, cast_each=False),
        "reverse_float": _manual(conv_input, weight, 2, reverse=True, cast_each=False),
        "forward_cast_each": _manual(conv_input, weight, 2, reverse=False, cast_each=True),
        "reverse_cast_each": _manual(conv_input, weight, 2, reverse=True, cast_each=True),
        "f32_silu_f32": _manual_f32(conv_input, weight, 2, cast_pre_silu=False),
        "f32_cast_pre_silu": _manual_f32(conv_input, weight, 2, cast_pre_silu=True),
    }
    mx.eval(raw, *variants.values())
    print(
        f"[conv-debug] dtypes conv_input={conv_input.dtype} weight={weight.dtype} "
        f"raw_pre={raw_pre.dtype} raw={raw.dtype}",
        flush=True,
    )
    pre = _manual(conv_input, weight, 2, reverse=False, cast_each=False, silu=False)
    mx.eval(raw_pre, pre)
    max_abs, mean_abs, gt = _diff(raw_pre, pre[:, :2, :])
    print(
        f"[conv-debug] pre_silu_forward_float max_abs={max_abs:.6f} "
        f"mean_abs={mean_abs:.6f} gt_0_01={gt}",
        flush=True,
    )
    for name, value in variants.items():
        max_abs, mean_abs, gt = _diff(raw, value[:, :2, :])
        print(
            f"[conv-debug] {name} max_abs={max_abs:.6f} "
            f"mean_abs={mean_abs:.6f} gt_0_01={gt}",
            flush=True,
        )
    best = []
    for order in itertools.permutations(range(int(weight.shape[1]))):
        for tree in (False, True):
            value = _manual_order(conv_input, weight, 2, order, tree)
            mx.eval(value)
            max_abs, mean_abs, gt = _diff(raw, value[:, :2, :])
            best.append((max_abs, mean_abs, gt, order, tree))
    best.sort(key=lambda x: (x[0], x[1], x[2]))
    for max_abs, mean_abs, gt, order, tree in best[:8]:
        print(
            f"[conv-debug] order={order} tree={tree} "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} gt_0_01={gt}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
