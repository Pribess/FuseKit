"""Compare the general QKV decode-sink kernel against MLX for one layer.

This probes only the semantic work the sink absorbs:

  input_layernorm + q/k/v projection + Q RoPE + K RoPE cache store + V cache store

It intentionally bypasses the full dispatcher so drift can be attributed to the
QKV emitter itself rather than later SDPA/residual/MLP stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def as_int(x) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="Write a concise Python function that parses JSON lines.")
    ap.add_argument("--decode-token", type=int, default=1)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model_dir = model_path(args.model)
    fk_dir = model_dir / "fusekit"

    import sys

    sys.path.insert(0, str(fk_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    model, tok = load(str(model_dir))
    handle = rt.load_dispatcher(str(fk_dir / "kernel.metallib"), str(fk_dir / "task_graph.json"))
    tg = json.loads((fk_dir / "task_graph.json").read_text())
    qkv_tasks = [t for t in tg["tasks"] if "fusekit_rmsnorm_qkv_" in t.get("kernel", "")]
    task = qkv_tasks[args.layer]
    kernel = task["kernel"]
    grid = tuple(task["grid_threads"])
    threadgroup = tuple(task["threadgroup"])

    ids = tok.encode(args.prompt) or [0]
    ids = mx.array([ids], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)

    decode_id = mx.array([[args.decode_token]], dtype=mx.uint32)
    h = model.model.embed_tokens(decode_id)
    for li in range(args.layer):
        h = model.model.layers[li](h, mask=None, cache=cache[li])
        mx.eval(h)

    block = model.model.layers[args.layer]
    attn = block.self_attn
    c = cache[args.layer]
    offset = as_int(getattr(c, "offset", getattr(c, "_idx", 0)))
    capacity = int(c.keys.shape[-2])
    dtype = c.keys.dtype

    normed = block.input_layernorm(h)
    q_pre = attn.q_proj(normed)
    k_pre = attn.k_proj(normed)
    v_pre = attn.v_proj(normed)
    mx.eval(q_pre, k_pre, v_pre)

    n_heads = int(attn.n_heads)
    n_kv_heads = int(attn.n_kv_heads)
    head_dim = int(q_pre.shape[-1] // n_heads)
    q_ref = q_pre.reshape(1, 1, n_heads, head_dim).transpose(0, 2, 1, 3)
    k_ref = k_pre.reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_ref = v_pre.reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q_ref = attn.rope(q_ref, offset=offset)
    k_ref = attn.rope(k_ref, offset=offset)
    mx.eval(q_ref, k_ref, v_ref)

    q_buf = mx.zeros((1, 8, n_heads * head_dim), dtype=dtype)
    k_buf = mx.zeros((1, 8, n_kv_heads * head_dim), dtype=dtype)
    v_buf = mx.zeros((1, 8, n_kv_heads * head_dim), dtype=dtype)
    cache_k = mx.zeros_like(c.keys)
    cache_v = mx.zeros_like(c.values)
    offset_arr = mx.array([offset], dtype=mx.uint32)
    capacity_arr = mx.array([capacity], dtype=mx.uint32)
    mx.eval(q_buf, k_buf, v_buf, cache_k, cache_v, offset_arr, capacity_arr)

    def buf(a):
        return (fr.array_data_ptr(a), a.nbytes)

    bufs = [
        buf(mx.contiguous(h).reshape(-1)),
        buf(block.input_layernorm.weight),
        buf(attn.q_proj.weight),
        buf(attn.k_proj.weight),
        buf(attn.v_proj.weight),
        buf(attn.q_proj.scales),
        buf(attn.k_proj.scales),
        buf(attn.v_proj.scales),
        buf(attn.q_proj.biases),
        buf(attn.k_proj.biases),
        buf(attn.v_proj.biases),
        buf(attn.q_proj.bias),
        buf(attn.k_proj.bias),
        buf(attn.v_proj.bias),
        buf(q_buf),
        buf(k_buf),
        buf(v_buf),
        buf(cache_k),
        buf(cache_v),
        buf(offset_arr),
        buf(capacity_arr),
    ]
    rt.dispatch_kernel(handle, kernel, bufs, grid=grid, threadgroup=threadgroup)
    mx.eval(q_buf, k_buf, v_buf, cache_k, cache_v)

    q_got = q_buf[:, :1, :].reshape(1, n_heads, 1, head_dim)
    k_got = cache_k[:, :, offset : offset + 1, :]
    v_got = cache_v[:, :, offset : offset + 1, :]

    def compare(name: str, got, ref) -> dict:
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        mx.eval(diff)
        return {
            f"{name}_max_abs": float(mx.max(diff).item()),
            f"{name}_mean_abs": float(mx.mean(diff).item()),
            f"{name}_gt_1e_2": int(mx.sum(diff > 1e-2).item()),
            f"{name}_gt_1e_1": int(mx.sum(diff > 1e-1).item()),
        }

    out = {
        "model": args.model,
        "layer": args.layer,
        "kernel": kernel,
        "offset": offset,
        "capacity": capacity,
        "grid": list(grid),
        "threadgroup": list(threadgroup),
    }
    out.update(compare("q_rope", q_got, q_ref))
    out.update(compare("k_cache", k_got, k_ref))
    out.update(compare("v_cache", v_got, v_ref))
    out.update(compare("k_scratch_pre", k_buf[:, :1, :].reshape(1, n_kv_heads, 1, head_dim), k_pre.reshape(1, 1, n_kv_heads, head_dim).transpose(0, 2, 1, 3)))
    out.update(compare("v_scratch_pre", v_buf[:, :1, :].reshape(1, n_kv_heads, 1, head_dim), v_ref))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
