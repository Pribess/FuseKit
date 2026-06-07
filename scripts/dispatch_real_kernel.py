"""End-to-end validation: dispatch fusekit's RmsNormQkv kernel on a
loaded model's actual layer weights, compare against the MLX reference
(separate RMSNorm + 3 Matmul).

This is the smallest "real model meets real fusekit kernel" test —
proves that:
  1. mx.array buffer pointers extracted via DLPack are usable in our
     per-model metallib's kernel
  2. Our codegen produces numerically equivalent output to MLX's path
  3. Multi-buffer dispatch through pyo3 works against the loaded
     dispatcher's pipeline cache (per-model metallib)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()

    # Bring up oMLX-managed mlx-lm and load the model normally.
    from mlx_lm import load
    model_dir = Path(f"~/.omlx/models/{args.model}").expanduser()
    print(f"[test] loading {args.model}", file=sys.stderr)
    model, _ = load(str(model_dir))

    fusekit_dir = model_dir / "fusekit"
    sys.path.insert(0, str(fusekit_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    # Load dispatcher once; reuse for every layer.
    handle = rt.load_dispatcher(
        str(fusekit_dir / "kernel.metallib"),
        str(fusekit_dir / "task_graph.json"),
    )
    qkv_kernels = [k for k in rt.kernel_names(handle) if k.startswith("fusekit_rmsnorm_qkv_")]
    print(f"[test] {len(qkv_kernels)} RmsNormQkv kernels in metallib",
          file=sys.stderr)
    assert len(qkv_kernels) == len(model.model.layers), \
        f"kernel count ({len(qkv_kernels)}) != layer count ({len(model.model.layers)})"

    def buf(arr):
        return (fr.array_data_ptr(arr), arr.nbytes)

    mx.random.seed(0)
    worst = {"q": 0.0, "k": 0.0, "v": 0.0}

    # Verify each transformer layer end-to-end. Same synthetic input
    # per layer to keep magnitudes consistent across layers.
    for li, layer in enumerate(model.model.layers):
        w_norm = layer.input_layernorm.weight
        w_q = layer.self_attn.q_proj.weight
        w_k = layer.self_attn.k_proj.weight
        w_v = layer.self_attn.v_proj.weight
        bias_q = layer.self_attn.q_proj.bias
        bias_k = layer.self_attn.k_proj.bias
        bias_v = layer.self_attn.v_proj.bias

        hidden = w_norm.shape[0]
        n_q = w_q.shape[0]
        n_kv = w_k.shape[0]
        eps = getattr(layer.input_layernorm, "eps", 1e-6)

        x = mx.random.normal((hidden,), dtype=w_norm.dtype) * 0.5
        mx.eval(x, w_norm, w_q, w_k, w_v, bias_q, bias_k, bias_v)

        q_out = mx.zeros((n_q,), dtype=w_norm.dtype)
        k_out = mx.zeros((n_kv,), dtype=w_norm.dtype)
        v_out = mx.zeros((n_kv,), dtype=w_norm.dtype)
        mx.eval(q_out, k_out, v_out)

        # Each layer's emitted kernel name matches that layer's
        # fusion id; pick by position in the layer list.
        kname = qkv_kernels[li]
        buffers = [
            buf(x), buf(w_norm),
            buf(w_q), buf(w_k), buf(w_v),
            buf(bias_q), buf(bias_k), buf(bias_v),
            buf(q_out), buf(k_out), buf(v_out),
        ]
        rt.dispatch_kernel(handle, kname, buffers,
                           grid=(256, 1, 1), threadgroup=(256, 1, 1))
        mx.eval(q_out, k_out, v_out)

        norm = mx.fast.rms_norm(x, w_norm, eps)
        q_ref = norm @ w_q.T + bias_q
        k_ref = norm @ w_k.T + bias_k
        v_ref = norm @ w_v.T + bias_v
        mx.eval(q_ref, k_ref, v_ref)

        errs = {}
        for name, got, ref in [("q", q_out, q_ref), ("k", k_out, k_ref), ("v", v_out, v_ref)]:
            errs[name] = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
            worst[name] = max(worst[name], errs[name])

        print(
            f"[test] layer {li:2d}: q={errs['q']:.4f}  k={errs['k']:.4f}  v={errs['v']:.4f}",
            file=sys.stderr,
        )

    print(f"[test] worst across all layers: q={worst['q']:.4f}  "
          f"k={worst['k']:.4f}  v={worst['v']:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
