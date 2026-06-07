"""Compare raw vs FuseKit layer outputs with an optional quantized KV cache.

The original version was written for Qwen3.5 Next's mixed FA/SSM stack.  The
helpers below keep that path, but also handle ordinary decoder-only stacks such
as Qwen2 where every layer consumes the same causal attention mask.
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


def layer_list(core):
    return getattr(core, "pipeline_layers", getattr(core, "layers"))


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
    if x.ndim >= 3:
        return x[:, :1, ...].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def make_decode_masks(mx, core, hidden, cache):
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    if hasattr(core, "fa_idx") and hasattr(core, "ssm_idx"):
        return {
            "fa": create_attention_mask(hidden, cache[core.fa_idx]),
            "ssm": create_ssm_mask(hidden, cache[core.ssm_idx]),
        }
    return {"fa": create_attention_mask(hidden, cache[0] if cache else None)}


def layer_mask(core, layer, masks):
    if hasattr(core, "fa_idx") and hasattr(core, "ssm_idx"):
        return masks["ssm"] if getattr(layer, "is_linear", False) else masks["fa"]
    return masks["fa"]


def child_main(args: argparse.Namespace) -> int:
    if args.mode == "raw":
        os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
        os.environ["FUSEKIT_DISABLE_FAST"] = "1"
    else:
        os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
        os.environ.pop("FUSEKIT_DISABLE_FAST", None)
        os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
        os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(str(model_path(args.model)))
    text = text_model(model)
    core = text_core(text)
    layers = layer_list(core)
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    if args.kv_bits is not None:
        from mlx_lm.generate import maybe_quantize_kv_cache

        maybe_quantize_kv_cache(
            cache,
            quantized_kv_start=args.quantized_kv_start,
            kv_group_size=args.kv_group_size,
            kv_bits=args.kv_bits,
        )
        mx.eval([c.state for c in cache])

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    arrays = {}
    if args.mode == "raw":
        hidden = core.embed_tokens(decode)
        masks = make_decode_masks(mx, core, hidden, cache)
        for li, (layer, c) in enumerate(zip(layers, cache)):
            hidden = layer(hidden, mask=layer_mask(core, layer, masks), cache=c)
            arrays[f"out_{li:02d}"] = row1(hidden).astype(mx.float32)
    else:
        logits = model(decode, cache=cache)
        if isinstance(logits, tuple):
            logits = logits[0]
        mx.eval(logits)
        disp = model._fk_dispatcher
        for li in range(len(layers)):
            sid = disp._layer_scratch_id.get(("out_buf", li))
            if sid is not None:
                arrays[f"out_{li:02d}"] = row1(disp._scratch_buf(sid)).astype(mx.float32)
            sid = disp._layer_scratch_id.get(("sdpa_buf", li))
            if sid is not None:
                arrays[f"sdpa_{li:02d}"] = row1(disp._scratch_buf(sid)).astype(mx.float32)

    mx.eval(list(arrays.values()))
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(
        json.dumps(
            {
                "mode": args.mode,
                "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
                "dispatcher_built": getattr(model, "_fk_dispatcher", None) is not None,
            },
            sort_keys=True,
        )
    )
    return 0


def parent_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-q35-quant-layers-") as td:
        tmp = Path(td)
        base = [
            sys.executable,
            __file__,
            "--child",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--decode-token",
            str(args.decode_token),
            "--kv-group-size",
            str(args.kv_group_size),
            "--quantized-kv-start",
            str(args.quantized_kv_start),
        ]
        if args.kv_bits is not None:
            base.extend(["--kv-bits", str(args.kv_bits)])
        raw_out = tmp / "raw.safetensors"
        patched_out = tmp / "patched.safetensors"
        raw_meta = tmp / "raw.json"
        patched_meta = tmp / "patched.json"
        subprocess.run(base + ["--mode", "raw", "--out", str(raw_out), "--meta", str(raw_meta)], check=True)
        subprocess.run(
            base + ["--mode", "patched", "--out", str(patched_out), "--meta", str(patched_meta)],
            check=True,
        )
        raw = mx.load(str(raw_out))
        patched = mx.load(str(patched_out))
        rows = []
        for key in sorted(raw):
            if key not in patched:
                continue
            diff = mx.abs(raw[key] - patched[key])
            mx.eval(diff)
            rows.append(
                {
                    "name": key,
                    "shape": list(raw[key].shape),
                    "max_abs_diff": float(mx.max(diff).item()),
                    "mean_abs_diff": float(mx.mean(diff).item()),
                }
            )
        print(
            json.dumps(
                {
                    "kv_bits": args.kv_bits,
                    "raw_meta": json.loads(raw_meta.read_text()),
                    "patched_meta": json.loads(patched_meta.read_text()),
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--quantized-kv-start", type=int, default=0)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", choices=["raw", "patched"])
    ap.add_argument("--out")
    ap.add_argument("--meta")
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
