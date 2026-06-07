"""Per-layer numeric validation for the SDPA decode kernel.

Builds synthetic Q (single-token), K, V tensors in the canonical
contiguous `(n_kv_heads, kv_len, head_dim)` layout the kernel expects,
dispatches `fusekit_sdpa_decode_fNN`, and compares against
`mx.fast.scaled_dot_product_attention` (causal) for the same inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--kv-len", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=0)
    args = ap.parse_args()

    from mlx_lm import load
    model_dir = Path(f"~/.omlx/models/{args.model}").expanduser()
    fusekit_dir = model_dir / "fusekit"

    sys.path.insert(0, str(fusekit_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    print(f"[sdpa] loading {args.model}", file=sys.stderr)
    model, _ = load(str(model_dir))

    handle = rt.load_dispatcher(
        str(fusekit_dir / "kernel.metallib"),
        str(fusekit_dir / "task_graph.json"),
    )
    tg = json.loads((fusekit_dir / "task_graph.json").read_text())
    tasks = [t for t in tg["tasks"] if "fusekit_sdpa_decode_" in t.get("kernel", "")]
    if not tasks:
        raise RuntimeError("no sdpa decode tasks in task_graph.json")
    task = tasks[0]
    kernel = task["kernel"]
    grid = tuple(task["grid_threads"])
    threadgroup = tuple(task["threadgroup"])
    kernels = sorted([k for k in rt.kernel_names(handle) if "sdpa" in k])
    print(
        f"[sdpa] {len(kernels)} SDPA kernels in metallib; probing {kernel} "
        f"grid={grid} tg={threadgroup}",
        file=sys.stderr,
    )

    cfg = model.args
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    print(f"[sdpa] n_heads={n_heads} n_kv={n_kv_heads} head_dim={head_dim} "
          f"kv_len={args.kv_len}", file=sys.stderr)

    def buf(arr):
        return (fr.array_data_ptr(arr), arr.nbytes)

    mx.random.seed(0)
    kv_len = args.kv_len
    capacity = max(kv_len, args.capacity or kv_len)

    # Build canonical-layout tensors. All decode kernels share the
    # same dims so any one works for this validation.
    Q = mx.random.normal((n_heads, head_dim), dtype=mx.float16) * 0.5
    K_live = mx.random.normal((n_kv_heads, kv_len, head_dim), dtype=mx.float16) * 0.5
    V_live = mx.random.normal((n_kv_heads, kv_len, head_dim), dtype=mx.float16) * 0.5
    K = mx.concatenate(
        [K_live, mx.zeros((n_kv_heads, capacity - kv_len, head_dim), dtype=mx.float16)],
        axis=1,
    )
    V = mx.concatenate(
        [V_live, mx.zeros((n_kv_heads, capacity - kv_len, head_dim), dtype=mx.float16)],
        axis=1,
    )
    out = mx.zeros((n_heads, head_dim), dtype=mx.float16)
    kv_len_arr = mx.array([kv_len], dtype=mx.uint32)
    capacity_arr = mx.array([capacity], dtype=mx.uint32)
    mx.eval(Q, K, V, out, kv_len_arr, capacity_arr)

    # `dispatch_kernel` expects (ptr, nbytes) pairs in the kernel's
    # buffer order: Q, K, V, kv_len, capacity, out.
    rt.dispatch_kernel(
        handle,
        kernel,
        [buf(Q), buf(K), buf(V), buf(kv_len_arr), buf(capacity_arr), buf(out)],
        grid=grid,
        threadgroup=threadgroup,
    )
    mx.eval(out)

    # Reference via MLX's fast SDPA. mlx-lm shape convention:
    # (B=1, n_heads, L, head_dim) for Q, (B=1, n_kv, kv_len, head_dim)
    # for K, V.
    Q_ref = Q.reshape(1, n_heads, 1, head_dim)
    K_ref = K_live.reshape(1, n_kv_heads, kv_len, head_dim)
    V_ref = V_live.reshape(1, n_kv_heads, kv_len, head_dim)
    scale = 1.0 / (head_dim ** 0.5)
    ref = mx.fast.scaled_dot_product_attention(Q_ref, K_ref, V_ref, scale=scale)
    ref = ref.reshape(n_heads, head_dim)
    mx.eval(ref)

    diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
    print(f"[sdpa] max_err = {float(diff.max()):.4f}  mean_err = {float(diff.mean()):.4f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
