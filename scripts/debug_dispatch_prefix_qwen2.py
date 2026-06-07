"""Run FuseKit dispatch prefixes and compare layer-0 intermediates.

This executes the real dispatcher-built buffer bindings, but stops after a
chosen task so later scratch reuse cannot hide the first divergent stage.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
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


def row1(x):
    x = as_array(x)
    if x.ndim == 3:
        return x[:, :1, :].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def cache_tensor(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def maybe_quantize_cache(cache, *, kv_bits: int | None, kv_group_size: int, quantized_kv_start: int):
    if kv_bits is None:
        return
    import mlx.core as mx
    from mlx_lm.generate import maybe_quantize_kv_cache

    maybe_quantize_kv_cache(
        cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    mx.eval([c.state for c in cache])


def raw_refs(
    raw_model,
    tok,
    prompt: str,
    decode_token: int,
    layer_idx: int,
    *,
    kv_bits: int | None,
    kv_group_size: int,
    quantized_kv_start: int,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import scaled_dot_product_attention, swiglu

    cache = make_prompt_cache(raw_model)
    ids = mx.array([tok.encode(prompt) or [0]], dtype=mx.uint32)
    y = raw_model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    maybe_quantize_cache(
        cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )

    core = raw_model.model
    layers = getattr(core, "pipeline_layers", core.layers)
    decode = mx.array([[decode_token]], dtype=mx.uint32)
    hidden = core.embed_tokens(decode)
    for li in range(layer_idx):
        hidden = layers[li](hidden, mask=None, cache=cache[li])
        mx.eval(hidden)
    x_in = hidden
    layer = layers[layer_idx]
    normed = layer.input_layernorm(hidden)
    q = layer.self_attn.q_proj(normed)
    k = layer.self_attn.k_proj(normed)
    v = layer.self_attn.v_proj(normed)
    q = q.reshape(1, 1, layer.self_attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(1, 1, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = layer.self_attn.rope(q, offset=cache[0].offset)
    k = layer.self_attn.rope(k, offset=cache[0].offset)
    keys, values = cache[0].update_and_fetch(k, v)
    sdpa = scaled_dot_product_attention(
        q, keys, values, cache=cache[0], scale=layer.self_attn.scale, mask=None
    )
    attn_out = sdpa.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    h_attn = layer.self_attn.o_proj(attn_out) + hidden
    post_norm = layer.post_attention_layernorm(h_attn)
    silu_up = swiglu(layer.mlp.gate_proj(post_norm), layer.mlp.up_proj(post_norm))
    out = layer.mlp.down_proj(silu_up) + h_attn
    mx.eval(q, k, v, sdpa, h_attn, silu_up, out)
    return {
        "x_in": x_in.reshape(1, -1).astype(mx.float32),
        "q_pre": layer.self_attn.q_proj(normed).reshape(1, -1).astype(mx.float32),
        "k_pre": layer.self_attn.k_proj(normed).reshape(1, -1).astype(mx.float32),
        "v_pre": layer.self_attn.v_proj(normed).reshape(1, -1).astype(mx.float32),
        "q_rope": q.reshape(1, -1).astype(mx.float32),
        "k_rope": k.reshape(1, -1).astype(mx.float32),
        "v": v.reshape(1, -1).astype(mx.float32),
        "q_buf": q.reshape(1, -1).astype(mx.float32),
        "cache_k_row": k.reshape(1, -1).astype(mx.float32),
        "cache_v_row": v.reshape(1, -1).astype(mx.float32),
        "sdpa_buf": sdpa.reshape(1, -1).astype(mx.float32),
        "h_attn": h_attn.reshape(1, -1).astype(mx.float32),
        "silu_up": silu_up.reshape(1, -1).astype(mx.float32),
        "out_buf": out.reshape(1, -1).astype(mx.float32),
    }


def run_prefix(
    model,
    tok,
    prompt: str,
    decode_token: int,
    target_task: int,
    layer_idx: int,
    *,
    kv_bits: int | None,
    kv_group_size: int,
    quantized_kv_start: int,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(getattr(model, "_fk_original", model))
    ids = mx.array([tok.encode(prompt) or [0]], dtype=mx.uint32)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    maybe_quantize_cache(
        cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )

    disp = getattr(model, "_fk_dispatcher", None)
    if disp is None:
        import fusekit_runtime.dispatch as dispatch_mod

        disp = dispatch_mod.Dispatcher(
            model._fk_original, model._fk_dir, model._fk_native_handle
        )
        model.__dict__["_fk_dispatcher"] = disp
    c = cache[layer_idx]
    o = disp._cache_decode_offset(c)
    if o is None:
        raise RuntimeError("no decode offset")
    icb_id = disp._ensure_icb(cache)
    if icb_id is None:
        raise RuntimeError("dispatcher did not build")

    decode = mx.array([[decode_token]], dtype=mx.uint32)
    flat = decode.flatten().astype(mx.uint32)
    mx.eval(flat)
    if disp._host_embed_lookup:
        row = disp._embed_module(decode).reshape((disp._hidden,)).astype(disp._act_dt)
        mx.eval(row)
        ctypes.memmove(disp._array_ptr(disp._x_in), disp._array_ptr(row), row.nbytes)
    ctypes.memmove(disp._uniform_ptrs[("next_token_out", None)], disp._array_ptr(flat), 4)
    ctypes.c_uint32.from_address(disp._step_counter_ptr).value = o
    ctypes.c_uint32.from_address(disp._hist_counter_ptr).value = 0
    ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_offset", None)]).value = o
    ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_kv_len", None)]).value = o + 1
    keys_repr = c.keys[0] if isinstance(c.keys, (tuple, list)) else c.keys
    ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_capacity", None)]).value = keys_repr.shape[2]
    for ptr in disp._barrier_counter_ptrs.values():
        ctypes.c_uint32.from_address(ptr).value = 0
    disp._advance_decode_cache(cache, o + 1)

    labels = disp._last_dispatch_labels or []
    idx = next(
        (i for i, label in enumerate(labels) if label.get("task_id") == target_task),
        None,
    )
    if idx is None:
        raise RuntimeError(f"target task {target_task} not in dispatch labels")
    disp._rt.dispatch_batch(disp._handle, disp._last_dispatches[: idx + 1])

    off = o
    arrays = {}
    task = disp._tg["tasks"][target_task]
    if task.get("inputs") and task["inputs"][0].get("role") == "scratch":
        sid = task["inputs"][0].get("scratch_id")
        arrays["x_in"] = row1(disp._scratch_buf(sid)).astype(mx.float32)
    for name in ("q_buf", "k_buf", "v_buf", "sdpa_buf", "h_attn", "silu_up", "out_buf"):
        key = (name, layer_idx)
        if key in disp._layer_scratch_id:
            arrays[name] = row1(disp._scratch_buf(disp._layer_scratch_id[key])).astype(mx.float32)
    keys = cache_tensor(c.keys)
    values = cache_tensor(c.values)
    arrays["cache_k_row"] = keys[:, :, off : off + 1, :].reshape(1, -1).astype(mx.float32)
    arrays["cache_v_row"] = values[:, :, off : off + 1, :].reshape(1, -1).astype(mx.float32)
    return arrays


def host_qkv_refs_for_x(
    raw_model,
    tok,
    prompt: str,
    x_in,
    layer_idx: int,
    *,
    kv_bits: int | None,
    kv_group_size: int,
    quantized_kv_start: int,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(raw_model)
    ids = mx.array([tok.encode(prompt) or [0]], dtype=mx.uint32)
    y = raw_model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    maybe_quantize_cache(
        cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )

    core = raw_model.model
    layers = getattr(core, "pipeline_layers", core.layers)
    layer = layers[layer_idx]
    attn = layer.self_attn
    x = x_in.reshape(1, 1, -1).astype(layer.input_layernorm.weight.dtype)
    normed = layer.input_layernorm(x)
    q = attn.q_proj(normed)
    k = attn.k_proj(normed)
    v = attn.v_proj(normed)
    n_heads = int(attn.n_heads)
    n_kv_heads = int(attn.n_kv_heads)
    head_dim = int(q.shape[-1] // n_heads)
    q = q.reshape(1, 1, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    offset = int(getattr(cache[layer_idx], "offset", getattr(cache[layer_idx], "_idx", 0)))
    q = attn.rope(q, offset=offset)
    k = attn.rope(k, offset=offset)
    mx.eval(q, k, v)
    return {
        "q_buf": q.reshape(1, -1).astype(mx.float32),
        "cache_k_row": k.reshape(1, -1).astype(mx.float32),
        "cache_v_row": v.reshape(1, -1).astype(mx.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Write a concise Python function that parses JSON lines.")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--quantized-kv-start", type=int, default=0)
    ap.add_argument("--fusekit-dir")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    model_dir = model_path(args.model)
    model, tok = load(str(model_dir))
    if args.fusekit_dir:
        import fusekit_runtime as fr

        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
    fk_dir = Path(args.fusekit_dir).expanduser() if args.fusekit_dir else model_dir / "fusekit"
    tg = json.loads((fk_dir / "task_graph.json").read_text())
    layer0 = [
        (idx, t)
        for idx, t in enumerate(tg["tasks"])
        if t.get("layer_idx") == args.layer and idx in set(tg.get("decode_pipeline", []))
    ]
    qkv_task = next((idx for idx, t in layer0 if "fusekit_rmsnorm_qkv_" in t.get("kernel", "")), None)
    qkv_has_sink = False
    if qkv_task is not None:
        qkv_has_sink = any(
            eb.get("role") in ("layer_cache_k", "layer_cache_v")
            for eb in tg["tasks"][qkv_task].get("extra_bindings", [])
        )
    rope_q_task = next((idx for idx, t in layer0 if "rope_q" in t.get("kernel", "").lower()), None)
    rope_kv_task = next((idx for idx, t in layer0 if "kv_cache_store" in t.get("kernel", "").lower()), None)
    sdpa_task = next((idx for idx, t in layer0 if "sdpa" in t.get("kernel", "").lower()), None)
    linear_tasks = [idx for idx, t in layer0 if "linear_residual" in t.get("kernel", "").lower()]
    chain_task = next((idx for idx, t in layer0 if "chain" in t.get("kernel", "").lower()), None)

    targets = []
    if qkv_task is not None:
        targets.append((qkv_task, "x_in", "x_in", "x_in"))
        if qkv_has_sink:
            targets.extend(
                [
                    (qkv_task, "q_buf", "q_rope", "q_buf"),
                    (qkv_task, "cache_k_row", "k_rope", "cache_k_row"),
                    (qkv_task, "cache_v_row", "v", "cache_v_row"),
                ]
            )
        else:
            targets.extend(
                [
                    (qkv_task, "q_buf", "q_pre", "q_pre"),
                    (qkv_task, "k_buf", "k_pre", "k_pre"),
                    (qkv_task, "v_buf", "v_pre", "v_pre"),
                ]
            )
    else:
        if rope_q_task is not None:
            targets.append((rope_q_task, "q_buf", "q_rope", "q_buf"))
        if rope_kv_task is not None:
            targets.extend(
                [
                    (rope_kv_task, "cache_k_row", "k_rope", "cache_k_row"),
                    (rope_kv_task, "cache_v_row", "v", "cache_v_row"),
                ]
            )
    if qkv_task is not None and not qkv_has_sink:
        if rope_q_task is not None:
            targets.append((rope_q_task, "q_buf", "q_rope", "q_buf"))
        if rope_kv_task is not None:
            targets.extend(
                [
                    (rope_kv_task, "cache_k_row", "k_rope", "cache_k_row"),
                    (rope_kv_task, "cache_v_row", "v", "cache_v_row"),
                ]
            )
    if sdpa_task is not None:
        targets.append((sdpa_task, "sdpa_buf", "sdpa_buf", "sdpa_buf"))
    if linear_tasks:
        targets.append((linear_tasks[0], "h_attn", "h_attn", "h_attn"))
    if chain_task is not None:
        targets.append((chain_task, "silu_up", "silu_up", "silu_up"))
    if len(linear_tasks) > 1:
        targets.append((linear_tasks[1], "out_buf", "out_buf", "out_buf"))

    raw_model = getattr(model, "_fk_original", model)
    refs = raw_refs(
        raw_model,
        tok,
        args.prompt,
        args.decode_token,
        args.layer,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )
    rows = []
    if qkv_task is not None and qkv_has_sink:
        qkv_actual = run_prefix(
            model,
            tok,
            args.prompt,
            args.decode_token,
            qkv_task,
            args.layer,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
        )
        if "x_in" in qkv_actual:
            actual_refs = host_qkv_refs_for_x(
                raw_model,
                tok,
                args.prompt,
                qkv_actual["x_in"],
                args.layer,
                kv_bits=args.kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
            )
            for name in ("q_buf", "cache_k_row", "cache_v_row"):
                if not qkv_has_sink and name != "q_buf":
                    continue
                got = qkv_actual.get(name)
                if got is None:
                    continue
                ref = actual_refs[name]
                n = min(got.size, ref.size)
                diff = mx.abs(got.reshape(-1)[:n] - ref.reshape(-1)[:n])
                mx.eval(diff)
                rows.append(
                    {
                        "task": qkv_task,
                        "name": f"actual_x_{name}",
                        "max_abs_diff": float(mx.max(diff).item()),
                        "mean_abs_diff": float(mx.mean(diff).item()),
                        "n": int(n),
                    }
                )
    for task_id, got_name, ref_name, label in targets:
        got = run_prefix(
            model,
            tok,
            args.prompt,
            args.decode_token,
            task_id,
            args.layer,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
        ).get(got_name)
        if got is None:
            rows.append({"task": task_id, "name": label, "missing": True})
            continue
        ref = refs[ref_name]
        n = min(got.size, ref.size)
        diff = mx.abs(got.reshape(-1)[:n] - ref.reshape(-1)[:n])
        mx.eval(diff)
        rows.append(
            {
                "task": task_id,
                "name": label,
                "max_abs_diff": float(mx.max(diff).item()),
                "mean_abs_diff": float(mx.mean(diff).item()),
                "n": int(n),
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
