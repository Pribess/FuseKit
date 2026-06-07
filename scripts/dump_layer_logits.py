"""Dump per-layer logit-lens comparisons for raw MLX vs FuseKit decode.

For each decode-layer hidden state, this script applies the original MLX
final norm + lm_head to both raw and FuseKit hidden states. This isolates
where hidden drift becomes visible in vocabulary space without using the
FuseKit final logits kernel as part of the diagnostic.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def text_core(text):
    return getattr(text, "model", text)


def make_cache_for(model_or_text):
    from mlx_lm.models.cache import make_prompt_cache

    if hasattr(model_or_text, "make_cache"):
        return model_or_text.make_cache()
    return make_prompt_cache(model_or_text)


def as_array(x):
    ptr = getattr(x, "_fk_ptr", None)
    if ptr is None:
        return x
    import mlx.core as mx
    import numpy as np

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


def row1(x):
    x = as_array(x)
    if x.ndim == 3:
        return x[:, :1, :].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def run_text_layers(text, inputs, cache, *, capture: bool):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    core = text_core(text)
    hidden = core.embed_tokens(inputs)
    if not (hasattr(core, "fa_idx") and hasattr(core, "ssm_idx")):
        layers = getattr(core, "pipeline_layers", core.layers)
        if cache is None:
            cache = [None] * len(layers)
        mask = create_attention_mask(hidden, cache[0] if cache else None)
        captures = {}
        for i, (layer, c) in enumerate(zip(layers, cache)):
            hidden = layer(hidden, mask, cache=c)
            mx.eval(hidden)
            if capture:
                captures[f"layer_{i:03d}"] = row1(hidden)
        normed = core.norm(hidden)
        logits = core.embed_tokens.as_linear(normed) if getattr(text.args, "tie_word_embeddings", False) else text.lm_head(normed)
        mx.eval(logits)
        if capture:
            captures["logits"] = row1(logits)
        return logits, captures

    fa_mask = create_attention_mask(hidden, cache[core.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache[core.ssm_idx])
    captures = {}
    for i, (layer, c) in enumerate(zip(core.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=c)
        mx.eval(hidden)
        if capture:
            captures[f"layer_{i:03d}"] = row1(hidden)
    normed = core.norm(hidden)
    logits = core.embed_tokens.as_linear(normed) if getattr(text.args, "tie_word_embeddings", False) else text.lm_head(normed)
    mx.eval(logits)
    if capture:
        captures["logits"] = row1(logits)
    return logits, captures


def child_raw(args: argparse.Namespace) -> int:
    os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_FAST"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(model_path(args.model)))
    text = text_model(model)
    cache = make_cache_for(text)

    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    run_text_layers(text, ids, cache, capture=False)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    _, arrays = run_text_layers(text, decode, cache, capture=True)
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(
        json.dumps(
            {
                "mode": "raw",
                "layers": len([k for k in arrays if k.startswith("layer_")]),
            }
        )
    )
    return 0


def child_patched(args: argparse.Namespace) -> int:
    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ.pop("FUSEKIT_ENABLE_FAST", None)
    if args.force_dispatch:
        os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
        os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(model_path(args.model)))
    if args.fusekit_dir:
        import fusekit_runtime as fr

        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)

    text = text_model(model)
    cache = make_cache_for(text)

    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    prefill = model(ids, cache=cache)
    if isinstance(prefill, tuple):
        prefill = prefill[0]
    mx.eval(prefill)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    logits = model(decode, cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)

    disp = getattr(model, "_fk_dispatcher", None)
    arrays = {"logits": row1(logits)}
    if disp is not None:
        for key, sid in sorted(disp._layer_scratch_id.items()):
            name, layer = key
            if name == "out_buf":
                arrays[f"layer_{layer:03d}"] = row1(disp._scratch_buf(sid))
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(
        json.dumps(
            {
                "mode": "patched",
                "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
                "dispatcher_built": disp is not None,
                "strict_dispatch": bool(os.environ.get("FUSEKIT_STRICT_DISPATCH")),
                "layers": len([k for k in arrays if k.startswith("layer_")]),
            }
        )
    )
    return 0


def lm_head_logits(mx, text, hidden_row):
    core = text_core(text)
    hidden = hidden_row.reshape(1, 1, -1)
    normed = core.norm(hidden)
    logits = (
        core.embed_tokens.as_linear(normed)
        if getattr(text.args, "tie_word_embeddings", False)
        else text.lm_head(normed)
    )
    mx.eval(logits)
    return row1(logits)


def compare(mx, a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def topk_summary(x, k: int):
    import numpy as np

    arr = np.array(x).reshape(-1).astype("float32", copy=False)
    k = min(k, arr.size)
    idx = np.argpartition(-arr, k - 1)[:k]
    idx = idx[np.argsort(-arr[idx])]
    return [{"token": int(i), "logit": float(arr[i])} for i in idx]


def parent_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-layer-logits-") as td:
        tmp = Path(td)
        raw_out = tmp / "raw_hidden.safetensors"
        patched_out = tmp / "patched_hidden.safetensors"
        raw_meta = tmp / "raw.json"
        patched_meta = tmp / "patched.json"

        base_cmd = [
            sys.executable,
            __file__,
            "--child",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--decode-token",
            str(args.decode_token),
        ]
        if args.fusekit_dir:
            base_cmd += ["--fusekit-dir", args.fusekit_dir]
        subprocess.run(
            base_cmd + ["--mode", "raw", "--out", str(raw_out), "--meta", str(raw_meta)],
            check=True,
        )
        subprocess.run(
            base_cmd
            + ["--mode", "patched", "--out", str(patched_out), "--meta", str(patched_meta)]
            + (["--force-dispatch"] if args.force_dispatch else []),
            check=True,
        )

        os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
        os.environ["FUSEKIT_DISABLE_FAST"] = "1"
        from mlx_lm import load

        model, _ = load(str(model_path(args.model)))
        text = text_model(model)
        raw = mx.load(str(raw_out))
        patched = mx.load(str(patched_out))

        save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
        raw_logits_dump = {}
        patched_logits_dump = {}
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        keys = sorted(k for k in raw if k.startswith("layer_"))
        if args.max_layers is not None:
            keys = keys[: args.max_layers]
        for key in keys:
            if key not in patched:
                rows.append({"name": key, "missing": "patched"})
                continue
            raw_l = lm_head_logits(mx, text, raw[key])
            patched_l = lm_head_logits(mx, text, patched[key])
            hidden_max, hidden_mean = compare(mx, raw[key], patched[key])
            logit_max, logit_mean = compare(mx, raw_l, patched_l)
            raw_argmax = int(mx.argmax(raw_l).item())
            patched_argmax = int(mx.argmax(patched_l).item())
            mx.eval(raw_l, patched_l)
            rows.append(
                {
                    "name": key,
                    "hidden_max_abs_diff": hidden_max,
                    "hidden_mean_abs_diff": hidden_mean,
                    "logit_max_abs_diff": logit_max,
                    "logit_mean_abs_diff": logit_mean,
                    "raw_argmax": raw_argmax,
                    "patched_argmax": patched_argmax,
                    "same_argmax": raw_argmax == patched_argmax,
                    "raw_top": topk_summary(raw_l.astype(mx.float32), args.topk),
                    "patched_top": topk_summary(patched_l.astype(mx.float32), args.topk),
                }
            )
            if save_dir:
                raw_logits_dump[key] = raw_l
                patched_logits_dump[key] = patched_l

        final_max, final_mean = compare(mx, raw["logits"], patched["logits"])
        final_raw_argmax = int(mx.argmax(raw["logits"]).item())
        final_patched_argmax = int(mx.argmax(patched["logits"]).item())

        summary = {
            "model": args.model,
            "prompt": args.prompt,
            "decode_token": args.decode_token,
            "fusekit_dir": args.fusekit_dir or "",
            "raw_meta": json.loads(raw_meta.read_text()),
            "patched_meta": json.loads(patched_meta.read_text()),
            "rows": rows,
            "final_logits": {
                "max_abs_diff": final_max,
                "mean_abs_diff": final_mean,
                "raw_argmax": final_raw_argmax,
                "patched_argmax": final_patched_argmax,
                "same_argmax": final_raw_argmax == final_patched_argmax,
                "raw_top": topk_summary(raw["logits"].astype(mx.float32), args.topk),
                "patched_top": topk_summary(patched["logits"].astype(mx.float32), args.topk),
            },
        }
        if save_dir:
            raw_logits_path = save_dir / "raw_layer_logits.safetensors"
            patched_logits_path = save_dir / "patched_layer_logits.safetensors"
            summary_path = save_dir / "summary.json"
            mx.save_safetensors(str(raw_logits_path), raw_logits_dump)
            mx.save_safetensors(str(patched_logits_path), patched_logits_dump)
            summary["saved"] = {
                "raw_layer_logits": str(raw_logits_path),
                "patched_layer_logits": str(patched_logits_path),
                "summary": str(summary_path),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def child_main(args: argparse.Namespace) -> int:
    if args.mode == "raw":
        return child_raw(args)
    return child_patched(args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--fusekit-dir")
    ap.add_argument("--force-dispatch", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-layers", type=int)
    ap.add_argument("--save-dir")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", choices=["raw", "patched"], default="raw")
    ap.add_argument("--out")
    ap.add_argument("--meta")
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
