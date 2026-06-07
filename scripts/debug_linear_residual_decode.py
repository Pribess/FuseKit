"""Directly validate one decode LinearResidual task against MLX.

The task is selected from the compiled task graph by layer and occurrence:
occurrence 0 is normally attention o_proj + residual, occurrence 1 is MLP
down_proj + residual. This bypasses the full dispatcher and uses the current
task ABI/grid/threadgroup from task_graph.json.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--occurrence", type=int, default=0)
    ap.add_argument("--prompt", default="Write a concise Python function that parses JSON lines.")
    ap.add_argument("--decode-token", type=int, default=1)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import scaled_dot_product_attention, swiglu

    model_dir = model_path(args.model)
    fk_dir = model_dir / "fusekit"

    import sys

    sys.path.insert(0, str(fk_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    model, tok = load(str(model_dir))
    handle = rt.load_dispatcher(str(fk_dir / "kernel.metallib"), str(fk_dir / "task_graph.json"))
    tg = json.loads((fk_dir / "task_graph.json").read_text())
    tasks = [
        t
        for t in tg["tasks"]
        if t.get("layer_idx") == args.layer
        and t.get("kernel", "").startswith("fusekit_linear_residual_")
    ]
    if args.occurrence >= len(tasks):
        raise RuntimeError(f"layer {args.layer} has only {len(tasks)} LinearResidual tasks")
    task = tasks[args.occurrence]
    kernel = task["kernel"]
    grid = tuple(task["grid_threads"])
    threadgroup = tuple(task["threadgroup"])

    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    prefill = model(ids, cache=cache)
    if isinstance(prefill, tuple):
        prefill = prefill[0]
    mx.eval(prefill)

    core = model.model
    h = core.embed_tokens(mx.array([[args.decode_token]], dtype=mx.uint32))
    for li in range(args.layer):
        h = core.layers[li](h, mask=None, cache=cache[li])
        mx.eval(h)

    block = core.layers[args.layer]
    attn = block.self_attn
    normed = block.input_layernorm(h)
    q = attn.q_proj(normed).reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = attn.k_proj(normed).reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = attn.v_proj(normed).reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = attn.rope(q, offset=cache[args.layer].offset)
    k = attn.rope(k, offset=cache[args.layer].offset)
    keys, values = cache[args.layer].update_and_fetch(k, v)
    sdpa = scaled_dot_product_attention(
        q, keys, values, cache=cache[args.layer], scale=attn.scale, mask=None
    )
    attn_out = sdpa.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    h_attn_ref = attn.o_proj(attn_out) + h
    mx.eval(attn_out, h_attn_ref)

    if args.occurrence == 0:
        x = attn_out.reshape(1, -1)
        residual = h.reshape(1, -1)
        ref = h_attn_ref.reshape(1, -1)
        linear = attn.o_proj
    else:
        norm_h = block.post_attention_layernorm(h_attn_ref)
        x = swiglu(block.mlp.gate_proj(norm_h), block.mlp.up_proj(norm_h)).reshape(1, -1)
        residual = h_attn_ref.reshape(1, -1)
        ref = (block.mlp.down_proj(x.reshape(1, 1, -1)) + h_attn_ref).reshape(1, -1)
        linear = block.mlp.down_proj
        mx.eval(x, residual, ref)

    hidden = int(ref.shape[-1])
    active_rows = 8
    x_full = mx.concatenate(
        [x.astype(x.dtype), mx.zeros((active_rows - 1, x.shape[-1]), dtype=x.dtype)],
        axis=0,
    ).reshape(1, active_rows, x.shape[-1])
    residual_full = mx.concatenate(
        [
            residual.astype(residual.dtype),
            mx.zeros((active_rows - 1, residual.shape[-1]), dtype=residual.dtype),
        ],
        axis=0,
    ).reshape(1, active_rows, residual.shape[-1])
    out = mx.zeros((1, active_rows, hidden), dtype=residual.dtype)
    mx.eval(x_full, residual_full, out)

    def buf(a):
        return (fr.array_data_ptr(a), a.nbytes)

    rt.dispatch_kernel(
        handle,
        kernel,
        [
            buf(x_full),
            buf(linear.weight),
            buf(linear.scales),
            buf(linear.biases),
            buf(residual_full),
            buf(out),
        ],
        grid=grid,
        threadgroup=threadgroup,
    )
    mx.eval(out)

    got = out[:, :1, :].reshape(1, -1).astype(mx.float32)
    ref = ref.astype(mx.float32)
    diff = mx.abs(got - ref)
    mx.eval(diff)
    print(
        json.dumps(
            {
                "kernel": kernel,
                "grid": list(grid),
                "threadgroup": list(threadgroup),
                "layer": args.layer,
                "occurrence": args.occurrence,
                "x_shape": list(x_full.shape),
                "weight_shape": list(linear.weight.shape),
                "max_abs_diff": float(mx.max(diff).item()),
                "mean_abs_diff": float(mx.mean(diff).item()),
                "gt_1e_2": int(mx.sum(diff > 1e-2).item()),
                "gt_1e_1": int(mx.sum(diff > 1e-1).item()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
