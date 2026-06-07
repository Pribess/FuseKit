"""Validate one fusekit-emitted MSL kernel against an MLX reference.

Takes a patched model directory, picks a `RmsNormQkv` kernel from
`kernel.metal`, runs both:

  * fusekit kernel via `mx.fast.metal_kernel`
  * MLX reference: separate RMSNorm + 3 Matmul ops

Reports the max abs error. Useful as a smoke test for codegen
correctness before the runtime pipeline is fully wired.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import mlx.core as mx


def extract_kernel_body(metal_source: str, kernel_name: str) -> str | None:
    """Pull the function body of a single kernel from kernel.metal.
    Returns the text between the function's opening `{` and the
    matching closing `}` (excluding both braces)."""
    # Find the kernel function declaration.
    start = metal_source.find(f"kernel void {kernel_name}(")
    if start < 0:
        return None
    # Locate the opening brace of the function body.
    brace_start = metal_source.find("{", start)
    if brace_start < 0:
        return None
    # Walk forward, tracking nesting, until we hit the matching close.
    depth = 0
    i = brace_start
    while i < len(metal_source):
        c = metal_source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return metal_source[brace_start + 1 : i]
        i += 1
    return None


def first_rmsnorm_qkv_kernel(metal_source: str) -> str | None:
    m = re.search(r"kernel void (fusekit_rmsnorm_qkv_f\d+)", metal_source)
    return m.group(1) if m else None


def rewrite_body_for_mx_fast(body: str) -> str:
    """mx.fast.metal_kernel auto-generates the signature and exposes
    thread positions as `thread_position_in_threadgroup.x` etc. Our
    emit uses bare `tid`, `tg_size`, `gid`. Patch the names so the
    body fits mx.fast's signature without us regenerating it."""
    return (
        body.replace("tid", "thread_position_in_threadgroup.x")
        .replace("tg_size", "threads_per_threadgroup.x")
        .replace("gid", "threadgroup_position_in_grid.x")
    )


def reference(x, w_norm, w_q, w_k, w_v, eps):
    """RMSNorm + 3 matmul, computed via standard MLX ops."""
    norm = mx.fast.rms_norm(x, w_norm, eps)
    q = norm @ w_q.T
    k = norm @ w_k.T
    v = norm @ w_v.T
    return q, k, v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", help="path to a fusekit-patched model directory")
    args = p.parse_args()

    metal_path = Path(args.model_dir) / "fusekit" / "kernel.metal"
    if not metal_path.is_file():
        print(f"[verify] no kernel.metal at {metal_path}", file=sys.stderr)
        return 2
    metal_src = metal_path.read_text()

    kname = first_rmsnorm_qkv_kernel(metal_src)
    if not kname:
        print("[verify] no RmsNormQkv kernel in emit", file=sys.stderr)
        return 2
    body = extract_kernel_body(metal_src, kname)
    if body is None:
        print(f"[verify] could not extract body of {kname}", file=sys.stderr)
        return 2
    print(f"[verify] kernel: {kname}", file=sys.stderr)

    # Extract dims by re-running the model's config — every constexpr in
    # the body uses these values, so this also confirms the kernel
    # matches the IR we expected.
    hidden = int(re.search(r"HIDDEN\s*=\s*(\d+)", body).group(1))
    n_q = int(re.search(r"N_Q\s*=\s*(\d+)", body).group(1))
    n_kv = int(re.search(r"N_KV\s*=\s*(\d+)", body).group(1))
    eps = float(re.search(r"EPS\s*=\s*([0-9.e\-+]+)", body).group(1))
    print(f"[verify] HIDDEN={hidden} N_Q={n_q} N_KV={n_kv} EPS={eps}", file=sys.stderr)

    # Build inputs: small magnitudes so bf16 quantization stays mild.
    mx.random.seed(0)
    x = mx.random.normal((hidden,), dtype=mx.bfloat16) * 0.5
    w_norm = mx.random.normal((hidden,), dtype=mx.bfloat16) * 0.1 + 1.0
    w_q = mx.random.normal((n_q, hidden), dtype=mx.bfloat16) * 0.02
    w_k = mx.random.normal((n_kv, hidden), dtype=mx.bfloat16) * 0.02
    w_v = mx.random.normal((n_kv, hidden), dtype=mx.bfloat16) * 0.02
    bias_q = mx.zeros((n_q,), dtype=mx.bfloat16)
    bias_k = mx.zeros((n_kv,), dtype=mx.bfloat16)
    bias_v = mx.zeros((n_kv,), dtype=mx.bfloat16)
    mx.eval(x, w_norm, w_q, w_k, w_v, bias_q, bias_k, bias_v)

    # Build and run our kernel via mx.fast.
    rewritten = rewrite_body_for_mx_fast(body)
    kernel = mx.fast.metal_kernel(
        name="fusekit_rmsnorm_qkv_test",
        input_names=["x", "norm_w", "W_q", "W_k", "W_v", "bias_q", "bias_k", "bias_v"],
        output_names=["q", "k", "v"],
        source=rewritten,
        ensure_row_contiguous=True,
    )
    try:
        q_o, k_o, v_o = kernel(
            inputs=[x, w_norm, w_q, w_k, w_v, bias_q, bias_k, bias_v],
            output_shapes=[(n_q,), (n_kv,), (n_kv,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(q_o, k_o, v_o)
    except Exception as e:
        print(f"[verify] kernel run failed: {e}", file=sys.stderr)
        return 3

    # Reference computation.
    q_ref, k_ref, v_ref = reference(x, w_norm, w_q, w_k, w_v, eps)
    mx.eval(q_ref, k_ref, v_ref)

    # Compare. bf16 with our magnitudes should be within ~0.05 abs.
    for name, got, ref in [("q", q_o, q_ref), ("k", k_o, k_ref), ("v", v_o, v_ref)]:
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        print(f"[verify] {name}: max_err={max_err:.6f} mean_err={mean_err:.6f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
