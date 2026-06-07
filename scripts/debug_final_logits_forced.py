"""Debug final RMSNorm/LM-head correctness for forced decode dispatch."""

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


def lm_head_logits(text, normed):
    core = text_core(text)
    if getattr(text.args, "tie_word_embeddings", False):
        return core.embed_tokens.as_linear(normed)
    return text.lm_head(normed)


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
                captures[f"layer_{i:03d}"] = row1(hidden).astype(mx.float32)
        normed = core.norm(hidden)
        logits = lm_head_logits(text, normed)
        mx.eval(normed, logits)
        if capture:
            captures["final_norm"] = row1(normed).astype(mx.float32)
            captures["logits"] = row1(logits).astype(mx.float32)
        return logits, captures

    fa_mask = create_attention_mask(hidden, cache[core.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache[core.ssm_idx])
    captures = {}
    for i, (layer, c) in enumerate(zip(core.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=c)
        mx.eval(hidden)
        if capture:
            captures[f"layer_{i:03d}"] = row1(hidden).astype(mx.float32)
    normed = core.norm(hidden)
    logits = lm_head_logits(text, normed)
    mx.eval(normed, logits)
    if capture:
        captures["final_norm"] = row1(normed).astype(mx.float32)
        captures["logits"] = row1(logits).astype(mx.float32)
    return logits, captures


def final_task_sids(disp):
    hidden = disp._hidden
    vocab = disp._vocab
    norm_sid = None
    logits_sid = None
    last_out = None
    for key, sid in disp._layer_scratch_id.items():
        name, layer = key
        if name == "out_buf" and (last_out is None or layer > last_out[0]):
            last_out = (layer, sid)
    for tid in disp._tg["decode_pipeline"]:
        t = disp._tg["tasks"][tid]
        li = t.get("layer_idx")
        outs = [b for b in t.get("outputs", []) if b.get("role") == "scratch"]
        if not outs:
            continue
        shape = outs[0].get("shape") or []
        last = shape[-1] if shape else None
        kernel = t.get("kernel", "")
        if li is None and last == hidden and "rmsnorm" in kernel:
            norm_sid = outs[0]["scratch_id"]
        if li is None and last == vocab:
            logits_sid = outs[0]["scratch_id"]
    return last_out[1] if last_out else None, norm_sid, logits_sid


def child_raw(args: argparse.Namespace) -> int:
    os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_FAST"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(model_path(args.model)))
    text = text_model(model)
    cache = make_cache_for(text)

    ids = tok.encode(args.prompt) or [0]
    ids = mx.array([ids], dtype=mx.uint32)
    run_text_layers(text, ids, cache, capture=False)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    _, arrays = run_text_layers(text, decode, cache, capture=True)
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(json.dumps({"mode": "raw"}))
    return 0


def child_patched(args: argparse.Namespace) -> int:
    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(model_path(args.model)))
    if args.fusekit_dir:
        import fusekit_runtime as fr

        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
    text = text_model(model)
    core = text_core(text)
    cache = make_cache_for(text)

    ids = tok.encode(args.prompt) or [0]
    ids = mx.array([ids], dtype=mx.uint32)
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
    if disp is None:
        raise RuntimeError("dispatcher was not built")
    last_out_sid, norm_sid, logits_sid = final_task_sids(disp)
    if last_out_sid is None or logits_sid is None:
        raise RuntimeError(
            f"missing final sids: out={last_out_sid} norm={norm_sid} logits={logits_sid}"
        )

    last_hidden = disp._scratch_buf(last_out_sid)
    logits_scratch = disp._scratch_buf(logits_sid)
    hidden_3d = row1(last_hidden).reshape(1, 1, -1).astype(last_hidden.dtype)
    norm_from_scratch = core.norm(hidden_3d)
    logits_from_host_norm = lm_head_logits(text, norm_from_scratch)
    mx.eval(norm_from_scratch, logits_from_host_norm)

    arrays = {
        "layer_last": row1(last_hidden).astype(mx.float32),
        "logits": row1(logits_scratch).astype(mx.float32),
        "returned_logits": row1(logits).astype(mx.float32),
        "host_norm_from_layer_last": row1(norm_from_scratch).astype(mx.float32),
        "host_logits_from_host_norm": row1(logits_from_host_norm).astype(mx.float32),
    }
    if norm_sid is not None:
        norm_scratch = disp._scratch_buf(norm_sid)
        norm_scratch_3d = row1(norm_scratch).reshape(1, 1, -1).astype(norm_scratch.dtype)
        logits_from_scratch_norm = lm_head_logits(text, norm_scratch_3d)
        mx.eval(logits_from_scratch_norm)
        arrays["final_norm"] = row1(norm_scratch).astype(mx.float32)
        arrays["host_logits_from_scratch_norm"] = row1(logits_from_scratch_norm).astype(mx.float32)
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(
        json.dumps(
            {
                "mode": "patched",
                "last_out_sid": last_out_sid,
                "norm_sid": norm_sid,
                "logits_sid": logits_sid,
                "strict_dispatch": True,
            },
            sort_keys=True,
        )
    )
    return 0


def compare(a, b):
    import mlx.core as mx

    diff = mx.abs(a - b)
    mx.eval(diff)
    return {
        "max_abs_diff": float(mx.max(diff).item()),
        "mean_abs_diff": float(mx.mean(diff).item()),
    }


def parent_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-final-debug-") as td:
        tmp = Path(td)
        raw_out = tmp / "raw.safetensors"
        patched_out = tmp / "patched.safetensors"
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
            + ["--mode", "patched", "--out", str(patched_out), "--meta", str(patched_meta)],
            check=True,
        )

        raw = mx.load(str(raw_out))
        patched = mx.load(str(patched_out))
        raw_last_layer = sorted(k for k in raw if k.startswith("layer_"))[-1]
        rows = {
            "raw_layer_vs_patched_layer": compare(raw[raw_last_layer], patched["layer_last"]),
            "raw_logits_vs_patched_logits_kernel": compare(raw["logits"], patched["logits"]),
            "host_on_raw_norm_vs_host_on_patched_layer": compare(
                raw["logits"], patched["host_logits_from_host_norm"]
            ),
            "returned_logits_vs_logits_scratch": compare(
                patched["returned_logits"], patched["logits"]
            ),
            "patched_logits_kernel_vs_host_on_patched_layer": compare(
                patched["logits"], patched["host_logits_from_host_norm"]
            ),
        }
        if "final_norm" in patched:
            rows["raw_norm_vs_patched_norm_kernel"] = compare(
                raw["final_norm"], patched["final_norm"]
            )
            rows["patched_norm_kernel_vs_host_norm"] = compare(
                patched["final_norm"], patched["host_norm_from_layer_last"]
            )
            rows["patched_logits_kernel_vs_host_on_scratch_norm"] = compare(
                patched["logits"], patched["host_logits_from_scratch_norm"]
            )
        print(
            json.dumps(
                {
                    "model": args.model,
                    "prompt": args.prompt,
                    "decode_token": args.decode_token,
                    "raw_meta": json.loads(raw_meta.read_text()),
                    "patched_meta": json.loads(patched_meta.read_text()),
                    "raw_last_layer": raw_last_layer,
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
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", choices=["raw", "patched"])
    ap.add_argument("--out")
    ap.add_argument("--meta")
    ap.add_argument("--fusekit-dir")
    args = ap.parse_args()
    if args.child:
        if args.mode == "raw":
            return child_raw(args)
        return child_patched(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
