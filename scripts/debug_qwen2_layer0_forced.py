"""Compare Qwen2 layer-0 raw intermediates against forced fusekit scratch."""

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


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def child_raw(args: argparse.Namespace) -> int:
    os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_FAST"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import scaled_dot_product_attention

    model, tok = load(str(model_path(args.model)))
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    prefill = model(ids, cache=cache)
    if isinstance(prefill, tuple):
        prefill = prefill[0]
    mx.eval(prefill)

    core = model.model
    layer = core.layers[0]
    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    hidden = core.embed_tokens(decode)
    normed = layer.input_layernorm(hidden)
    q = layer.self_attn.q_proj(normed)
    k = layer.self_attn.k_proj(normed)
    v = layer.self_attn.v_proj(normed)
    q = q.reshape(1, 1, layer.self_attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(1, 1, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q_rope = layer.self_attn.rope(q, offset=cache[0].offset)
    k_rope = layer.self_attn.rope(k, offset=cache[0].offset)
    mx.eval(q, k, v, q_rope, k_rope)

    mask = create_attention_mask(hidden, cache[0])
    keys, values = cache[0].update_and_fetch(k_rope, v)
    sdpa = scaled_dot_product_attention(
        q_rope, keys, values, cache=cache[0], scale=layer.self_attn.scale, mask=mask
    )
    sdpa_no_current = scaled_dot_product_attention(
        q_rope,
        keys[:, :, :-1, :],
        values[:, :, :-1, :],
        cache=cache[0],
        scale=layer.self_attn.scale,
        mask=None,
    )
    attn = sdpa.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    o_proj = layer.self_attn.o_proj(attn)
    h_attn = hidden + o_proj
    from mlx_lm.models.qwen2 import swiglu

    post_norm = layer.post_attention_layernorm(h_attn)
    silu_up = swiglu(layer.mlp.gate_proj(post_norm), layer.mlp.up_proj(post_norm))
    mlp = layer.mlp.down_proj(silu_up)
    out = h_attn + mlp
    mx.eval(sdpa, o_proj, h_attn, post_norm, silu_up, mlp, out)

    mx.save_safetensors(
        args.out,
        {
            "q_pre_rope": q.reshape(1, -1).astype(mx.float32),
            "q_rope": q_rope.reshape(1, -1).astype(mx.float32),
            "k_pre_rope": k.reshape(1, -1).astype(mx.float32),
            "k_rope": k_rope.reshape(1, -1).astype(mx.float32),
            "v": v.reshape(1, -1).astype(mx.float32),
            "cache_k_all": keys.reshape(1, -1).astype(mx.float32),
            "cache_v_all": values.reshape(1, -1).astype(mx.float32),
            "sdpa": sdpa.reshape(1, -1).astype(mx.float32),
            "sdpa_no_current": sdpa_no_current.reshape(1, -1).astype(mx.float32),
            "o_proj": o_proj.reshape(1, -1).astype(mx.float32),
            "h_attn": h_attn.reshape(1, -1).astype(mx.float32),
            "post_norm": post_norm.reshape(1, -1).astype(mx.float32),
            "silu_up": silu_up.reshape(1, -1).astype(mx.float32),
            "mlp": mlp.reshape(1, -1).astype(mx.float32),
            "layer_out": out.reshape(1, -1).astype(mx.float32),
            "meta_shape": mx.array(
                [
                    layer.self_attn.n_heads,
                    layer.self_attn.n_kv_heads,
                    q.shape[-1],
                    keys.shape[2],
                ],
                dtype=mx.uint32,
            ),
        },
    )
    return 0


def child_patched(args: argparse.Namespace) -> int:
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"
    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(str(model_path(args.model)))
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    prefill = model(ids, cache=cache)
    if isinstance(prefill, tuple):
        prefill = prefill[0]
    mx.eval(prefill)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    logits = model(decode, cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)

    disp = model._fk_dispatcher
    c0 = cache[0]
    off_after = int(getattr(c0, "offset", getattr(c0, "_idx", 0)))
    off = max(off_after - 1, 0)
    arrays = {}
    for name in ("q_buf", "k_buf", "v_buf", "sdpa_buf", "h_attn", "silu_up", "out_buf"):
        sid = disp._layer_scratch_id[(name, 0)]
        arrays[name] = as_array(disp._scratch_buf(sid)).reshape(1, -1).astype(mx.float32)
    arrays["cache_k_row"] = c0.keys[:, :, off : off + 1, :].reshape(1, -1).astype(mx.float32)
    arrays["cache_v_row"] = c0.values[:, :, off : off + 1, :].reshape(1, -1).astype(mx.float32)
    arrays["cache_k_all"] = c0.keys[:, :, :off_after, :].reshape(1, -1).astype(mx.float32)
    arrays["cache_v_all"] = c0.values[:, :, :off_after, :].reshape(1, -1).astype(mx.float32)
    arrays["uniform_offset"] = mx.array(
        [ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_offset", None)]).value],
        dtype=mx.uint32,
    )
    arrays["uniform_kv_len"] = mx.array(
        [ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_kv_len", None)]).value],
        dtype=mx.uint32,
    )
    arrays["uniform_capacity"] = mx.array(
        [ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_capacity", None)]).value],
        dtype=mx.uint32,
    )
    mx.save_safetensors(args.out, arrays)
    return 0


def parent(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-qwen2-l0-") as td:
        tmp = Path(td)
        raw_path = tmp / "raw.safetensors"
        patched_path = tmp / "patched.safetensors"
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
        ]
        subprocess.run(base + ["--mode", "raw", "--out", str(raw_path)], check=True)
        subprocess.run(base + ["--mode", "patched", "--out", str(patched_path)], check=True)
        raw = mx.load(str(raw_path))
        patched = mx.load(str(patched_path))
        n_heads, n_kv_heads, head_dim, kv_len = [
            int(v.item()) for v in raw["meta_shape"].reshape(-1)
        ]
        q_elems = n_heads * head_dim
        patched_q = patched["q_buf"].reshape(-1)[:q_elems].reshape(1, n_heads, 1, head_dim)
        patched_k = patched["cache_k_all"].reshape(1, n_kv_heads, kv_len, head_dim)
        patched_v = patched["cache_v_all"].reshape(1, n_kv_heads, kv_len, head_dim)
        patched_fast = mx.fast.scaled_dot_product_attention(
            patched_q.astype(mx.float16),
            patched_k.astype(mx.float16),
            patched_v.astype(mx.float16),
            scale=1.0 / math.sqrt(float(head_dim)),
            mask=None,
        ).astype(mx.float32)
        mx.eval(patched_fast)
        pairs = {
            "q_pre_rope_vs_q_buf": ("q_pre_rope", "q_buf"),
            "q_rope_vs_q_buf": ("q_rope", "q_buf"),
            "k_pre_rope_vs_k_buf": ("k_pre_rope", "k_buf"),
            "k_rope_vs_cache_k": ("k_rope", "cache_k_row"),
            "cache_k_all_vs_cache_k_all": ("cache_k_all", "cache_k_all"),
            "v_vs_v_buf": ("v", "v_buf"),
            "v_vs_cache_v": ("v", "cache_v_row"),
            "cache_v_all_vs_cache_v_all": ("cache_v_all", "cache_v_all"),
            "sdpa_vs_sdpa_buf": ("sdpa", "sdpa_buf"),
            "sdpa_no_current_vs_sdpa_buf": ("sdpa_no_current", "sdpa_buf"),
            "h_attn_vs_h_attn": ("h_attn", "h_attn"),
            "silu_up_vs_silu_up": ("silu_up", "silu_up"),
            "mlp_vs_out_minus_h_attn": ("mlp", "out_buf"),
            "layer_out_vs_out_buf": ("layer_out", "out_buf"),
        }
        rows = []
        for name, (rk, pk) in pairs.items():
            lhs = raw[rk].reshape(-1)
            rhs = patched[pk].reshape(-1)
            if name == "mlp_vs_out_minus_h_attn":
                rhs = patched["out_buf"].reshape(-1) - patched["h_attn"].reshape(-1)
            n = min(lhs.size, rhs.size)
            diff = mx.abs(lhs[:n] - rhs[:n])
            mx.eval(diff)
            rows.append(
                {
                    "name": name,
                    "n": int(n),
                    "max_abs_diff": float(mx.max(diff).item()),
                    "mean_abs_diff": float(mx.mean(diff).item()),
                }
            )
        for name, lhs, rhs in (
            ("raw_sdpa_vs_patched_fast", raw["sdpa"].reshape(-1), patched_fast.reshape(-1)),
            ("patched_fast_vs_sdpa_buf", patched_fast.reshape(-1), patched["sdpa_buf"].reshape(-1)),
        ):
            n = min(lhs.size, rhs.size)
            diff = mx.abs(lhs[:n] - rhs[:n])
            mx.eval(diff)
            rows.append(
                {
                    "name": name,
                    "n": int(n),
                    "max_abs_diff": float(mx.max(diff).item()),
                    "mean_abs_diff": float(mx.mean(diff).item()),
                }
            )
        rows.append(
            {
                "name": "patched_uniforms",
                "offset": int(patched["uniform_offset"][0].item()),
                "kv_len": int(patched["uniform_kv_len"][0].item()),
                "capacity": int(patched["uniform_capacity"][0].item()),
            }
        )
        print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", choices=["raw", "patched"])
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.child:
        return child_raw(args) if args.mode == "raw" else child_patched(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
