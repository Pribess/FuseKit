"""Compare one QuantizedLinear projection row against FuseKit scratch."""

from __future__ import annotations

import argparse
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def text_model(model):
    return getattr(getattr(model, "_fk_original", model), "language_model", model)


def row1(x):
    if x.ndim >= 3:
        return x[:, :1, ...].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def scalar_at(mx, arr, row: int) -> float:
    return float(row1(arr)[0, row].astype(mx.float32).item())


def pick_projection(layer, name: str):
    if name == "qkv":
        return layer.linear_attn.in_proj_qkv
    if name == "z":
        return layer.linear_attn.in_proj_z
    if name == "a":
        return layer.linear_attn.in_proj_a
    if name == "b":
        return layer.linear_attn.in_proj_b
    if name == "out":
        return layer.linear_attn.out_proj
    if name == "gate":
        return layer.mlp.gate_proj
    if name == "up":
        return layer.mlp.up_proj
    if name == "down":
        return layer.mlp.down_proj
    raise ValueError(f"unknown projection: {name}")


def projection_sid(disp, layer: int, name: str) -> int:
    from debug_qwen35_linear_layer import find_linear_layer_sids

    key = {
        "qkv": "qkv",
        "z": "z",
        "a": "a_dt",
        "b": "b",
        "out": "attn_proj",
        "gate": "mlp_gate",
        "up": "mlp_up",
        "down": "mlp_down",
    }[name]
    sids = find_linear_layer_sids(disp._tg, layer, disp._hidden)
    return sids[key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fusekit-dir", required=True)
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--decode-token", type=int, default=283)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--projection", choices=("qkv", "z", "a", "b", "out", "gate", "up", "down"), default="b")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--scan-rows", type=int, default=0)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    import fusekit_runtime as fr

    model, tok = load(str(model_path(args.model)))
    model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
    text = text_model(model)
    core = text.model
    cache = text.make_cache()
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
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
    layer = core.layers[args.layer]
    ql = pick_projection(layer, args.projection)
    x = row1(disp._scratch_buf(0)).reshape(1, 1, -1)
    if args.projection in ("gate", "up", "down"):
        raise RuntimeError("this helper currently uses layer input norm scratch only")
    y_mlx = ql(x)
    sid = projection_sid(disp, args.layer, args.projection)
    y_fk = disp._scratch_buf(sid)
    w_deq = mx.dequantize(
        ql.weight[args.row : args.row + 1],
        scales=ql.scales[args.row : args.row + 1],
        biases=ql.biases[args.row : args.row + 1],
        group_size=ql.group_size,
        bits=ql.bits,
        mode=ql.mode,
    ).astype(mx.float32)
    deq_dot = mx.sum(x.reshape(-1).astype(mx.float32) * w_deq.reshape(-1))
    mx.eval(y_mlx, y_fk, w_deq, deq_dot)
    out = {
        "layer": args.layer,
        "projection": args.projection,
        "bits": int(ql.bits),
        "group_size": int(ql.group_size),
        "mode": str(ql.mode),
        "weight_shape": tuple(int(v) for v in ql.weight.shape),
        "scales_shape": tuple(int(v) for v in ql.scales.shape),
        "sid": sid,
        "row": args.row,
        "mlx": scalar_at(mx, y_mlx, args.row),
        "dequant_dot": float(deq_dot.item()),
        "fusekit": scalar_at(mx, y_fk, args.row),
    }
    out["diff_mlx_fk"] = out["mlx"] - out["fusekit"]
    out["diff_deq_fk"] = out["dequant_dot"] - out["fusekit"]
    print(out, flush=True)
    if args.scan_rows:
        y_mlx_f = row1(y_mlx).astype(mx.float32)
        y_fk_f = row1(y_fk).astype(mx.float32)
        diff = mx.abs(y_mlx_f[:, : args.scan_rows] - y_fk_f[:, : args.scan_rows])
        mx.eval(diff)
        rows = []
        for i in range(args.scan_rows):
            rows.append(
                {
                    "row": i,
                    "mlx": float(y_mlx_f[0, i].item()),
                    "fk": float(y_fk_f[0, i].item()),
                    "abs": float(diff[0, i].item()),
                }
            )
        print({"rows": rows}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
