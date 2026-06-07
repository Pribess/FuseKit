"""Per-layer numeric check for the RmsNormGateUp fusekit kernel.

For each transformer layer:
  - extracts post_attention_layernorm + mlp.{gate,up}_proj weights
  - dispatches `fusekit_rmsnorm_gateup_fNN` on synthetic input
  - compares against `mx.fast.rms_norm` + matmul reference
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

    from mlx_lm import load
    model_dir = Path(f"~/.omlx/models/{args.model}").expanduser()
    fusekit_dir = model_dir / "fusekit"

    sys.path.insert(0, str(fusekit_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    print(f"[gateup] loading {args.model}", file=sys.stderr)
    model, _ = load(str(model_dir))

    handle = rt.load_dispatcher(
        str(fusekit_dir / "kernel.metallib"),
        str(fusekit_dir / "task_graph.json"),
    )
    kernels = [k for k in rt.kernel_names(handle) if "gateup" in k]
    print(f"[gateup] {len(kernels)} kernels in metallib", file=sys.stderr)
    assert len(kernels) == len(model.model.layers)

    def buf(arr):
        return (fr.array_data_ptr(arr), arr.nbytes)

    mx.random.seed(0)
    worst = {"gate": 0.0, "up": 0.0}

    for li, layer in enumerate(model.model.layers):
        w_norm = layer.post_attention_layernorm.weight
        w_gate = layer.mlp.gate_proj.weight
        w_up = layer.mlp.up_proj.weight

        hidden = w_norm.shape[0]
        n_int = w_gate.shape[0]
        eps = getattr(layer.post_attention_layernorm, "eps", 1e-6)

        x = mx.random.normal((hidden,), dtype=w_norm.dtype) * 0.5
        gate_out = mx.zeros((n_int,), dtype=w_norm.dtype)
        up_out = mx.zeros((n_int,), dtype=w_norm.dtype)
        mx.eval(x, w_norm, w_gate, w_up, gate_out, up_out)

        rt.dispatch_kernel(
            handle,
            kernels[li],
            [buf(x), buf(w_norm), buf(w_gate), buf(w_up), buf(gate_out), buf(up_out)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(gate_out, up_out)

        norm = mx.fast.rms_norm(x, w_norm, eps)
        gate_ref = norm @ w_gate.T
        up_ref = norm @ w_up.T
        mx.eval(gate_ref, up_ref)

        errs = {
            "gate": float(mx.abs(gate_out.astype(mx.float32) - gate_ref.astype(mx.float32)).max()),
            "up": float(mx.abs(up_out.astype(mx.float32) - up_ref.astype(mx.float32)).max()),
        }
        for k, v in errs.items():
            worst[k] = max(worst[k], v)

        print(f"[gateup] layer {li:2d}: gate={errs['gate']:.4f}  up={errs['up']:.4f}",
              file=sys.stderr)

    print(f"[gateup] worst across all layers: gate={worst['gate']:.4f}  "
          f"up={worst['up']:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
