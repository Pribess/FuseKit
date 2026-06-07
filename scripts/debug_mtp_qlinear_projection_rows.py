"""Compare one MTP verifier QuantizedLinear projection row-by-row."""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model

from debug_mtp_variant_layers import _find_full_layer_sids
from debug_qwen35_linear_layer import find_linear_layer_sids


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def _advance_cache(model, cache, tokens: list[int]) -> None:
    for tok in tokens:
        out = model(mx.array([[tok]], dtype=mx.uint32), cache=cache, return_hidden=True)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)


def _view_rows(x, rows: int):
    x = _as_array(x)
    if x.ndim == 3:
        return x[:, :rows, :]
    if x.ndim == 2:
        return x[:rows, :].reshape(1, rows, -1)
    return x.reshape(1, rows, -1)


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
    if "uint32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_uint32 * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.uint32)
    raise TypeError(f"unsupported native scratch dtype: {dtype}")


def _pick_projection(layer, name: str):
    if name == "qkv":
        return layer.linear_attn.in_proj_qkv
    if name == "z":
        return layer.linear_attn.in_proj_z
    if name == "a":
        return layer.linear_attn.in_proj_a
    if name == "b":
        return layer.linear_attn.in_proj_b
    if name == "out":
        return layer.linear_attn.out_proj
    if name == "q_proj":
        return layer.self_attn.q_proj
    if name == "k_proj":
        return layer.self_attn.k_proj
    if name == "v_proj":
        return layer.self_attn.v_proj
    raise ValueError(f"unsupported projection: {name}")


def _sid_key(name: str) -> str:
    return {
        "qkv": "qkv",
        "z": "z",
        "a": "a_dt",
        "b": "b",
        "out": "attn_proj",
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
    }[name]


def _diff_summary(a, b) -> dict[str, float | int]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return {
        "max_abs": float(mx.max(d).item()),
        "mean_abs": float(mx.mean(d).item()),
        "gt_0_001": int(mx.sum(d > 0.001).item()),
        "gt_0_01": int(mx.sum(d > 0.01).item()),
    }


def _top_diffs(a, b, *, top: int) -> list[dict[str, float | int]]:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    flat = d.reshape(-1)
    n = int(flat.size)
    k = min(top, n)
    # Keep this host-side because k is tiny and the debug script is not timed.
    mx.eval(af, bf, d)
    rows = []
    vals = [(float(flat[i].item()), i) for i in range(n)]
    vals.sort(reverse=True)
    _, _, width = a.shape
    for abs_v, flat_i in vals[:k]:
        token = flat_i // width
        col = flat_i % width
        rows.append(
            {
                "token": int(token),
                "col": int(col),
                "a": float(af.reshape(-1)[flat_i].item()),
                "b": float(bf.reshape(-1)[flat_i].item()),
                "abs": abs_v,
            }
        )
    return rows


def _bf16_scalar(value: float) -> float:
    arr = mx.array([value], dtype=mx.bfloat16)
    return float(arr.astype(mx.float32)[0].item())


def _manual_q4_qmv_impl(x, ql, out_col: int) -> dict[str, float | int]:
    x = x.reshape(-1).astype(mx.float32)
    w = ql.weight[out_col]
    scales = ql.scales[out_col].astype(mx.float32)
    biases = ql.biases[out_col].astype(mx.float32)
    mx.eval(x, w, scales, biases)
    x_list = [float(v.item()) for v in x]
    w_list = [int(v.item()) for v in w]
    s_list = [float(v.item()) for v in scales]
    b_list = [float(v.item()) for v in biases]
    lane_acc = [0.0 for _ in range(32)]
    values_per_thread = 8
    block_size = values_per_thread * 32
    for lane in range(32):
        acc = 0.0
        for k in range(0, len(x_list), block_size):
            base = k + lane * values_per_thread
            vals = x_list[base : base + values_per_thread]
            x_thread = [
                vals[0],
                vals[1] / 16.0,
                vals[2] / 256.0,
                vals[3] / 4096.0,
                vals[4],
                vals[5] / 16.0,
                vals[6] / 256.0,
                vals[7] / 4096.0,
            ]
            x_sum = sum(vals)
            word = w_list[(k // 8) + lane]
            lo = word & 0xFFFF
            hi = word >> 16
            qacc = (
                x_thread[0] * (lo & 0x000F)
                + x_thread[1] * (lo & 0x00F0)
                + x_thread[2] * (lo & 0x0F00)
                + x_thread[3] * (lo & 0xF000)
                + x_thread[4] * (hi & 0x000F)
                + x_thread[5] * (hi & 0x00F0)
                + x_thread[6] * (hi & 0x0F00)
                + x_thread[7] * (hi & 0xF000)
            )
            group = (k // 64) + (lane // 8)
            acc += s_list[group] * qacc + b_list[group] * x_sum
        lane_acc[lane] = acc
    lane_serial = 0.0
    for v in lane_acc:
        lane_serial += v
    return {
        "col": int(out_col),
        "manual_qmv_float": float(lane_serial),
        "manual_qmv_bf16": _bf16_scalar(float(lane_serial)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--projection",
        choices=("qkv", "z", "a", "b", "out", "q_proj", "k_proj", "v_proj"),
        default="qkv",
    )
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--manual-cols", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--main", type=int)
    ap.add_argument("--draft", type=int)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    text = _text_model(raw_model)
    core = text.model

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()

    if args.prefix or args.main is not None or args.draft is not None:
        if args.main is None or args.draft is None:
            raise RuntimeError("--main and --draft are required with --prefix")
        _ = raw_model(prompt, cache=cache_raw)
        _ = raw_model(prompt, cache=cache_fk)
        prefix = _parse_tokens(args.prefix)
        _advance_cache(raw_model, cache_raw, prefix)
        _advance_cache(raw_model, cache_fk, prefix)
        next_main = int(args.main)
        draft = int(args.draft)
    else:
        logits_raw = raw_model(prompt, cache=cache_raw)
        _ = raw_model(prompt, cache=cache_fk)
        first = _argmax_id(logits_raw)

        tok0 = mx.array([[first]], dtype=mx.uint32)
        logits_raw_0, hidden_raw_0 = raw_model(tok0, cache=cache_raw, return_hidden=True)
        _ = raw_model(tok0, cache=cache_fk, return_hidden=True)
        next_main = _argmax_id(logits_raw_0)

        mtp_cache = raw_model.make_mtp_cache()
        mtp_ids = mx.array([[next_main]], dtype=mx.uint32)
        mtp_logits = raw_model.mtp_forward(hidden_raw_0[:, -1:, :], mtp_ids, mtp_cache)
        draft = _argmax_id(mtp_logits)
    verify = mx.array([[next_main, draft]], dtype=mx.uint32)

    hidden = core.embed_tokens(verify)
    if hasattr(core, "fa_idx") and hasattr(core, "ssm_idx"):
        fa_mask = create_attention_mask(hidden, cache_raw[core.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache_raw[core.ssm_idx])
        for li, prev_layer in enumerate(core.layers[: args.layer]):
            mask = ssm_mask if getattr(prev_layer, "is_linear", False) else fa_mask
            hidden = prev_layer(hidden, mask=mask, cache=cache_raw[li], n_confirmed=1)
            mx.eval(hidden)
    layer = core.layers[args.layer]
    ql = _pick_projection(layer, args.projection)
    is_full_projection = args.projection in {"q_proj", "k_proj", "v_proj"}
    if is_full_projection and getattr(layer, "is_linear", False):
        raise RuntimeError(f"layer {args.layer} is linear-attention; choose qkv/z/a/b/out")
    if not is_full_projection and not getattr(layer, "is_linear", False):
        raise RuntimeError(f"layer {args.layer} is full-attention; choose q_proj/k_proj/v_proj")
    raw_norm = layer.input_layernorm(hidden)
    raw_proj_raw_norm = ql(raw_norm)
    raw_proj_split_norm = mx.concatenate(
        [ql(raw_norm[:, i : i + 1, :]) for i in range(2)],
        axis=1,
    )

    logits_fk, hidden_fk = model(
        verify,
        cache=cache_fk,
        return_hidden=True,
        n_confirmed=1,
    )
    mx.eval(logits_fk, hidden_fk)
    variant = model._get_variant_dispatcher(2)
    if variant is None:
        raise RuntimeError("MTP M=2 variant dispatcher was not loaded")
    sids = (
        _find_full_layer_sids(variant._tg, args.layer)
        if is_full_projection
        else find_linear_layer_sids(variant._tg, args.layer, variant._hidden)
    )
    norm_sid = sids["norm_x"]
    proj_sid = sids[_sid_key(args.projection)]
    fk_norm = _view_rows(variant._scratch_buf(norm_sid), 2)
    fk_proj = _view_rows(variant._scratch_buf(proj_sid), 2)
    raw_proj_fk_norm = ql(fk_norm)
    mx.eval(
        raw_norm,
        fk_norm,
        raw_proj_raw_norm,
        raw_proj_split_norm,
        raw_proj_fk_norm,
        fk_proj,
    )

    raw_proj_raw_norm = _view_rows(raw_proj_raw_norm, 2)
    raw_proj_split_norm = _view_rows(raw_proj_split_norm, 2)
    raw_proj_fk_norm = _view_rows(raw_proj_fk_norm, 2)
    raw_norm = _view_rows(raw_norm, 2)
    fk_norm = _view_rows(fk_norm, 2)

    print(
        {
            "layer": args.layer,
            "projection": args.projection,
            "bits": int(ql.bits),
            "group_size": int(ql.group_size),
            "mode": str(ql.mode),
            "verify_tokens": [int(next_main), int(draft)],
            "norm_sid": norm_sid,
            "proj_sid": proj_sid,
            "norm_raw_vs_fk": _diff_summary(raw_norm, fk_norm),
            "raw_proj_raw_norm_vs_fk_proj": _diff_summary(raw_proj_raw_norm, fk_proj),
            "raw_proj_fk_norm_vs_fk_proj": _diff_summary(raw_proj_fk_norm, fk_proj),
            "raw_proj_split_norm_vs_fk_proj": _diff_summary(
                raw_proj_split_norm, fk_proj
            ),
            "raw_proj_raw_norm_vs_raw_proj_split_norm": _diff_summary(
                raw_proj_raw_norm, raw_proj_split_norm
            ),
            "raw_proj_raw_norm_vs_raw_proj_fk_norm": _diff_summary(
                raw_proj_raw_norm, raw_proj_fk_norm
            ),
        },
        flush=True,
    )
    print(
        {
            "top_raw_proj_raw_norm_vs_fk_proj": _top_diffs(
                raw_proj_raw_norm, fk_proj, top=args.top
            )
        },
        flush=True,
    )
    print(
        {
            "top_raw_proj_fk_norm_vs_fk_proj": _top_diffs(
                raw_proj_fk_norm, fk_proj, top=args.top
            )
        },
        flush=True,
    )
    print(
        {
            "top_raw_proj_raw_norm_vs_raw_proj_split_norm": _top_diffs(
                raw_proj_raw_norm, raw_proj_split_norm, top=args.top
            )
        },
        flush=True,
    )
    print(
        {
            "top_norm_raw_vs_fk": _top_diffs(raw_norm, fk_norm, top=args.top),
        },
        flush=True,
    )
    if args.manual_cols:
        cols = [int(c) for c in args.manual_cols.split(",") if c.strip()]
        manual = []
        for token in range(2):
            for col in cols:
                entry = _manual_q4_qmv_impl(raw_norm[:, token : token + 1, :], ql, col)
                entry.update(
                    {
                        "token": token,
                        "raw": float(raw_proj_raw_norm[0, token, col].astype(mx.float32).item()),
                        "fk": float(fk_proj[0, token, col].astype(mx.float32).item()),
                    }
                )
                manual.append(entry)
        print({"manual_q4_qmv_impl": manual}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
