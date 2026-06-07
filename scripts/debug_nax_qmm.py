"""Compare MLX QuantizedLinear against the FuseKit NAX qmm helper.

This avoids the full decode dispatcher so correctness failures can be
localized to the quantized matmul kernel itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def text_model(model):
    return getattr(getattr(model, "_fk_original", model), "language_model", model)


def layer_norm_eps(norm) -> float:
    for name in ("eps", "variance_epsilon", "_epsilon"):
        if hasattr(norm, name):
            return float(getattr(norm, name))
    return 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fusekit-dir", required=True)
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()

    os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_FAST"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model_dir = model_path(args.model)
    model, _tok = load(str(model_dir))
    core = text_model(model).model
    layer = core.layers[0]
    att = layer.linear_attn
    norm = layer.input_layernorm

    token = mx.array([[args.decode_token]], dtype=mx.uint32)
    x = core.embed_tokens(token)
    x = mx.fast.rms_norm(x, norm.weight, layer_norm_eps(norm))
    x = mx.contiguous(x.astype(att.in_proj_qkv.scales.dtype))
    ref = att.in_proj_qkv(x)

    fusekit_dir = Path(args.fusekit_dir)
    tg = json.loads((fusekit_dir / "task_graph.json").read_text())
    task = None
    for candidate in tg["tasks"]:
        for binding in candidate.get("inputs", []):
            if str(binding.get("source_path", "")).endswith(
                "layers.0.linear_attn.in_proj_qkv.weight"
            ):
                task = candidate
                break
        if task is not None:
            break
    if task is None:
        raise RuntimeError("could not find layer 0 in_proj_qkv QuantizedMatmul task")

    sys.path.insert(0, str(model_dir / "fusekit" / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    handle = rt.load_dispatcher(
        str(fusekit_dir / "kernel.metallib"),
        str(fusekit_dir / "task_graph.json"),
    )

    def buf(arr):
        return (fr.array_data_ptr(arr), arr.nbytes)

    got = mx.zeros(tuple(ref.shape), dtype=ref.dtype)
    mx.eval(got)
    buffers = [
        buf(x),
        buf(att.in_proj_qkv.weight),
        buf(att.in_proj_qkv.scales),
        buf(att.in_proj_qkv.biases),
        buf(got),
    ]
    rt.dispatch_kernel(
        handle,
        task["kernel"],
        buffers,
        tuple(task["grid_threads"]),
        tuple(task["threadgroup"]),
    )
    mx.eval(ref, got)
    diff = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))
    diff_flat = diff.reshape(-1)
    worst_idx = mx.argsort(diff_flat)[-10:][::-1]
    mx.eval(worst_idx)
    worst = []
    flat_ref = ref.reshape(-1)
    flat_got = got.reshape(-1)
    for i in worst_idx:
        idx = int(i.item())
        worst.append(
            {
                "idx": idx,
                "mlx": float(flat_ref[idx].astype(mx.float32).item()),
                "got": float(flat_got[idx].astype(mx.float32).item()),
                "abs": float(diff_flat[idx].item()),
            }
        )
    row = args.row
    print(
        {
            "shape": list(ref.shape),
            "kernel": task["kernel"],
            "row": row,
            "mlx": float(flat_ref[row].astype(mx.float32).item()),
            "nax": float(flat_got[row].astype(mx.float32).item()),
            "max_abs_diff": float(mx.max(diff).item()),
            "mean_abs_diff": float(mx.mean(diff).item()),
            "worst": worst,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
