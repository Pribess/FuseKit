"""Validate QKV decode-sink state through the actual ICB path.

`debug_dispatch_prefix_qwen2.py` uses `dispatch_batch`, which is useful
for kernel math/binding checks but bypasses the failure mode where QKV
decode sinks are correct per-kernel and wrong only when encoded into an
MTLIndirectCommandBuffer. This script executes prefixes of the already
built ICB via `execute_icb_range` and compares semantic side effects at
the same boundaries.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_dispatch_prefix_qwen2 import (  # noqa: E402
    model_path,
    raw_refs,
    row1,
)


def _attach(model, fusekit_dir: str | None):
    if not fusekit_dir:
        return model
    import fusekit_runtime as fr

    return fr.attach(getattr(model, "_fk_original", model), fusekit_dir)


def _dispatcher(model):
    disp = getattr(model, "_fk_dispatcher", None)
    if disp is not None:
        return disp
    import fusekit_runtime.dispatch as dispatch_mod

    disp = dispatch_mod.Dispatcher(
        model._fk_original, model._fk_dir, model._fk_native_handle
    )
    model.__dict__["_fk_dispatcher"] = disp
    return disp


def _seed_decode_state(disp, cache, inputs, offset: int) -> None:
    import mlx.core as mx

    flat = inputs.flatten().astype(mx.uint32)
    mx.eval(flat)
    if disp._host_embed_lookup:
        row = disp._embed_module(inputs).reshape((disp._hidden,)).astype(disp._act_dt)
        mx.eval(row)
        ctypes.memmove(disp._array_ptr(disp._x_in), disp._array_ptr(row), row.nbytes)
    ctypes.memmove(disp._uniform_ptrs[("next_token_out", None)], disp._array_ptr(flat), 4)
    ctypes.c_uint32.from_address(disp._step_counter_ptr).value = offset
    ctypes.c_uint32.from_address(disp._hist_counter_ptr).value = 0
    ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_offset", None)]).value = offset
    ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_kv_len", None)]).value = offset + 1
    c0 = next((c for c in cache if hasattr(c, "keys") and c.keys is not None), None)
    if c0 is None:
        raise RuntimeError("no cache layer with keys")
    keys_repr = c0.keys[0] if isinstance(c0.keys, (tuple, list)) else c0.keys
    ctypes.c_uint32.from_address(disp._uniform_ptrs[("layer_capacity", None)]).value = keys_repr.shape[2]
    for ptr in disp._barrier_counter_ptrs.values():
        ctypes.c_uint32.from_address(ptr).value = 0


def _make_prefilled(model, tok, prompt: str):
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache

    raw_model = getattr(model, "_fk_original", model)
    if getattr(_make_prefilled, "kv_bits", None):
        n_layers = len(getattr(raw_model, "layers", getattr(raw_model.model, "layers", [])))
        cache = [
            QuantizedKVCache(
                group_size=getattr(_make_prefilled, "kv_group_size", 64),
                bits=getattr(_make_prefilled, "kv_bits"),
            )
            for _ in range(n_layers)
        ]
    else:
        cache = make_prompt_cache(raw_model)
    ids = mx.array([tok.encode(prompt) or [0]], dtype=mx.uint32)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    return cache


def _row_cache(cache_layer, offset: int):
    import mlx.core as mx

    keys = cache_layer.keys
    values = cache_layer.values
    if isinstance(keys, (tuple, list)) and isinstance(values, (tuple, list)):
        bits = getattr(cache_layer, "bits", 4)
        group_size = getattr(cache_layer, "group_size", 64)
        k = mx.dequantize(*keys, group_size=group_size, bits=bits)
        v = mx.dequantize(*values, group_size=group_size, bits=bits)
        return {
            "cache_k_row": k[:, :, offset : offset + 1, :].reshape(1, -1).astype(mx.float32),
            "cache_v_row": v[:, :, offset : offset + 1, :].reshape(1, -1).astype(mx.float32),
        }
    if isinstance(keys, (tuple, list)) or isinstance(values, (tuple, list)):
        raise RuntimeError("K/V cache quantization layout mismatch")
    return {
        "cache_k_row": keys[:, :, offset : offset + 1, :].reshape(1, -1).astype(mx.float32),
        "cache_v_row": values[:, :, offset : offset + 1, :].reshape(1, -1).astype(mx.float32),
    }


def _collect_state(disp, cache, layer_idx: int, offset: int):
    import mlx.core as mx

    arrays = {}
    for name in ("q_buf", "k_buf", "v_buf", "sdpa_buf", "h_attn", "silu_up", "out_buf"):
        key = (name, layer_idx)
        if key in disp._layer_scratch_id:
            arrays[name] = row1(disp._scratch_buf(disp._layer_scratch_id[key])).astype(mx.float32)
    arrays.update(_row_cache(cache[layer_idx], offset))
    return arrays


def _compare(mx, got, ref):
    n = min(got.size, ref.size)
    diff = mx.abs(got.reshape(-1)[:n] - ref.reshape(-1)[:n])
    mx.eval(diff)
    return {
        "max_abs_diff": float(mx.max(diff).item()),
        "mean_abs_diff": float(mx.mean(diff).item()),
        "n": int(n),
    }


def _find_layer_tasks(tg: dict, layer_idx: int):
    decode_set = set(tg.get("decode_pipeline", []))
    layer_tasks = [
        (idx, t)
        for idx, t in enumerate(tg["tasks"])
        if idx in decode_set and t.get("layer_idx") == layer_idx
    ]
    qkv = next((idx for idx, t in layer_tasks if "fusekit_rmsnorm_qkv_" in t.get("kernel", "")), None)
    sdpa = next((idx for idx, t in layer_tasks if "sdpa" in t.get("kernel", "").lower()), None)
    return qkv, sdpa


def _dispatch_index_for_task(labels: list, task_id: int) -> int:
    for idx, label in enumerate(labels):
        if isinstance(label, dict) and label.get("task_id") == task_id:
            return idx
    raise RuntimeError(f"task {task_id} not present in dispatch labels")


def run_prefix(args, model, tok, target_task: int, target_name: str):
    import mlx.core as mx

    cache = _make_prefilled(model, tok, args.prompt)
    disp = _dispatcher(model)
    c0 = next((c for c in cache if hasattr(c, "offset")), None)
    if c0 is None:
        raise RuntimeError("no decode cache offset")
    offset = disp._cache_decode_offset(c0)
    if offset is None:
        raise RuntimeError("could not infer decode offset")
    icb_id = disp._ensure_icb(cache)
    if icb_id is None:
        raise RuntimeError("dispatcher did not build ICB")
    if not hasattr(disp._rt, "execute_icb_range"):
        raise RuntimeError("native runtime lacks execute_icb_range; rebuild python_runtime")

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    _seed_decode_state(disp, cache, decode, offset)
    disp._advance_decode_cache(cache, offset + 1)

    labels = disp._last_dispatch_labels or []
    dispatch_idx = _dispatch_index_for_task(labels, target_task)
    gpu_s = float(disp._rt.execute_icb_range(disp._handle, icb_id, 0, dispatch_idx + 1))
    state = _collect_state(disp, cache, args.layer, offset)
    label = labels[dispatch_idx] if dispatch_idx < len(labels) else {}
    return {
        "target": target_name,
        "target_task": target_task,
        "dispatch_index": dispatch_idx,
        "gpu_us": gpu_s * 1_000_000.0,
        "kernel": label.get("kernel"),
        "kind": label.get("kind"),
        "offset": int(offset),
        "state": state,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Write a concise Python function that parses JSON lines.")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kv-bits", type=int, choices=(4, 8))
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--fusekit-dir")
    args = ap.parse_args()
    _make_prefilled.kv_bits = args.kv_bits
    _make_prefilled.kv_group_size = args.kv_group_size

    import mlx.core as mx
    from mlx_lm import load

    model_dir = model_path(args.model)
    model, tok = load(str(model_dir))
    model = _attach(model, args.fusekit_dir)
    fk_dir = Path(args.fusekit_dir).expanduser() if args.fusekit_dir else model_dir / "fusekit"
    tg = json.loads((fk_dir / "task_graph.json").read_text())
    qkv_task, sdpa_task = _find_layer_tasks(tg, args.layer)
    if qkv_task is None:
        raise RuntimeError(f"no QKV task for layer {args.layer}")

    raw_model = getattr(model, "_fk_original", model)
    refs = raw_refs(raw_model, tok, args.prompt, args.decode_token, args.layer)

    rows = []
    for task_id, name in [(qkv_task, "qkv"), (sdpa_task, "sdpa")]:
        if task_id is None:
            continue
        actual = run_prefix(args, model, tok, task_id, name)
        state = actual.pop("state")
        if name == "qkv":
            for got_name, ref_name in (
                ("q_buf", "q_rope"),
                ("cache_k_row", "k_rope"),
                ("cache_v_row", "v"),
            ):
                if got_name in state:
                    row = dict(actual)
                    row["name"] = got_name
                    row.update(_compare(mx, state[got_name], refs[ref_name]))
                    rows.append(row)
        elif name == "sdpa" and "sdpa_buf" in state:
            row = dict(actual)
            row["name"] = "sdpa_buf"
            row.update(_compare(mx, state["sdpa_buf"], refs["sdpa_buf"]))
            rows.append(row)

    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
