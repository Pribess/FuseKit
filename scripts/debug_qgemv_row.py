"""Compare one q4 QuantizedLinear row against FuseKit scratch output."""

from __future__ import annotations

import argparse
import os
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


def bf16_scalar(mx, value: float) -> float:
    arr = mx.array([value], dtype=mx.bfloat16)
    return float(arr.astype(mx.float32)[0].item())


def manual_q4_row(mx, x, weight, scales, biases, out_row: int) -> tuple[float, float]:
    # Mirror MLX qmv_fast lane partition for bits=4, group_size=64:
    # 2 packed uint32 words per lane per 512-wide block.
    x = x.astype(mx.float32)
    w = weight[out_row]
    s = scales[out_row].astype(mx.float32)
    b = biases[out_row].astype(mx.float32)
    mx.eval(x, w, s, b)
    x_list = [float(v.item()) for v in x]
    w_list = [int(v.item()) for v in w]
    s_list = [float(v.item()) for v in s]
    b_list = [float(v.item()) for v in b]
    lane_acc = [0.0 for _ in range(32)]
    for lane in range(32):
        widx = lane * 2
        while widx < len(w_list):
            group = (widx * 8) // 64
            scale = s_list[group]
            bias = b_list[group]
            qacc = 0.0
            xsum = 0.0
            for p in (0, 1):
                word_index = widx + p
                if word_index >= len(w_list):
                    continue
                word = w_list[word_index]
                base = word_index * 8
                lo = word & 0xFFFF
                hi = word >> 16
                terms = (
                    x_list[base + 0] * float(lo & 0x000F)
                    + (x_list[base + 1] / 16.0) * float(lo & 0x00F0)
                    + (x_list[base + 2] / 256.0) * float(lo & 0x0F00)
                    + (x_list[base + 3] / 4096.0) * float(lo & 0xF000)
                    + x_list[base + 4] * float(hi & 0x000F)
                    + (x_list[base + 5] / 16.0) * float(hi & 0x00F0)
                    + (x_list[base + 6] / 256.0) * float(hi & 0x0F00)
                    + (x_list[base + 7] / 4096.0) * float(hi & 0xF000)
                )
                qacc += terms
                xsum += sum(x_list[base : base + 8])
            lane_acc[lane] += scale * qacc + bias * xsum
            widx += 64
    raw = sum(lane_acc)
    return raw, bf16_scalar(mx, raw)


def manual_q4_row_bf16_dequant(mx, x, weight, scales, biases, out_row: int) -> tuple[float, float]:
    x = x.astype(mx.float32)
    w = weight[out_row]
    s = scales[out_row].astype(mx.float32)
    b = biases[out_row].astype(mx.float32)
    mx.eval(x, w, s, b)
    x_list = [float(v.item()) for v in x]
    w_list = [int(v.item()) for v in w]
    s_list = [float(v.item()) for v in s]
    b_list = [float(v.item()) for v in b]
    acc = 0.0
    for word_index, word in enumerate(w_list):
        group = (word_index * 8) // 64
        scale = s_list[group]
        bias = b_list[group]
        base = word_index * 8
        for off in range(8):
            q = (word >> (off * 4)) & 0xF
            wj = bf16_scalar(mx, float(q) * scale + bias)
            acc += x_list[base + off] * wj
    return acc, bf16_scalar(mx, acc)


def manual_q4_row_orders(mx, x, weight, scales, biases, out_row: int) -> dict[str, float]:
    import numpy as np

    x = x.astype(mx.float32)
    w = weight[out_row]
    s = scales[out_row].astype(mx.float32)
    b = biases[out_row].astype(mx.float32)
    mx.eval(x, w, s, b)
    x_np = np.array([float(v.item()) for v in x], dtype=np.float32)
    w_list = [int(v.item()) for v in w]
    s_np = np.array([float(v.item()) for v in s], dtype=np.float32)
    b_np = np.array([float(v.item()) for v in b], dtype=np.float32)
    vals = np.empty(x_np.shape[0], dtype=np.float32)
    vals_bf16w = np.empty(x_np.shape[0], dtype=np.float32)
    for word_index, word in enumerate(w_list):
        group = (word_index * 8) // 64
        scale = s_np[group]
        bias = b_np[group]
        base = word_index * 8
        for off in range(8):
            q = np.float32((word >> (off * 4)) & 0xF)
            wj = np.float32(q * scale + bias)
            vals[base + off] = np.float32(x_np[base + off] * wj)
            vals_bf16w[base + off] = np.float32(
                x_np[base + off] * np.float32(bf16_scalar(mx, float(wj)))
            )
    serial = np.float32(0.0)
    for v in vals:
        serial = np.float32(serial + v)
    block32 = np.float32(0.0)
    for k in range(0, vals.shape[0], 32):
        part = np.float32(0.0)
        for v in vals[k : k + 32]:
            part = np.float32(part + v)
        block32 = np.float32(block32 + part)
    block32_pair = np.float32(0.0)
    for k in range(0, vals.shape[0], 32):
        part = np.sum(vals[k : k + 32], dtype=np.float32)
        block32_pair = np.float32(block32_pair + part)
    block32_bf16w = np.float32(0.0)
    for k in range(0, vals_bf16w.shape[0], 32):
        part = np.sum(vals_bf16w[k : k + 32], dtype=np.float32)
        block32_bf16w = np.float32(block32_bf16w + part)
    return {
        "np_serial": float(serial),
        "np_serial_bf16": bf16_scalar(mx, float(serial)),
        "np_block32": float(block32),
        "np_block32_bf16": bf16_scalar(mx, float(block32)),
        "np_block32_pair": float(block32_pair),
        "np_block32_pair_bf16": bf16_scalar(mx, float(block32_pair)),
        "np_block32_bf16w": float(block32_bf16w),
        "np_block32_bf16w_bf16": bf16_scalar(mx, float(block32_bf16w)),
    }


def scalar_at(mx, arr, row: int) -> float:
    return float(row1(arr)[0, row].astype(mx.float32).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fusekit-dir", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--scan-rows", type=int, default=0)
    ap.add_argument("--capture")
    args = ap.parse_args()

    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

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
    layer = core.layers[0]
    att = layer.linear_attn
    norm_kernel = row1(disp._scratch_buf(0)).reshape(1, 1, -1)
    if args.capture:
        mx.metal.start_capture(args.capture)
    qkv_mlx = att.in_proj_qkv(norm_kernel)
    if args.capture:
        mx.eval(qkv_mlx)
        mx.metal.stop_capture()
    qkv_fk = disp._scratch_buf(1)
    norm_2d = norm_kernel.reshape(1, -1)
    norm_1d = norm_kernel.reshape(-1)
    norm_2d_contig = mx.contiguous(norm_2d)
    norm_3d_contig = mx.contiguous(norm_kernel)
    qkv_mlx_1d = att.in_proj_qkv(norm_1d)
    qkv_mlx_2d = att.in_proj_qkv(norm_2d)
    qkv_mlx_2d_contig = att.in_proj_qkv(norm_2d_contig)
    qkv_mlx_3d_contig = att.in_proj_qkv(norm_3d_contig)
    deq_row = mx.dequantize(
        att.in_proj_qkv.weight[args.row : args.row + 1],
        scales=att.in_proj_qkv.scales[args.row : args.row + 1],
        biases=att.in_proj_qkv.biases[args.row : args.row + 1],
        group_size=att.in_proj_qkv.group_size,
        bits=att.in_proj_qkv.bits,
        mode=att.in_proj_qkv.mode,
    ).astype(mx.float32)
    deq_dot = mx.sum(norm_kernel.reshape(-1).astype(mx.float32) * deq_row.reshape(-1))
    mx.eval(
        norm_kernel,
        qkv_mlx,
        qkv_mlx_1d,
        qkv_mlx_2d,
        qkv_mlx_2d_contig,
        qkv_mlx_3d_contig,
        qkv_fk,
        deq_row,
        deq_dot,
    )
    raw, rounded = manual_q4_row(
        mx,
        norm_kernel.reshape(-1),
        att.in_proj_qkv.weight,
        att.in_proj_qkv.scales,
        att.in_proj_qkv.biases,
        args.row,
    )
    raw_bf16w, rounded_bf16w = manual_q4_row_bf16_dequant(
        mx,
        norm_kernel.reshape(-1),
        att.in_proj_qkv.weight,
        att.in_proj_qkv.scales,
        att.in_proj_qkv.biases,
        args.row,
    )
    order_vals = manual_q4_row_orders(
        mx,
        norm_kernel.reshape(-1),
        att.in_proj_qkv.weight,
        att.in_proj_qkv.scales,
        att.in_proj_qkv.biases,
        args.row,
    )
    fk = scalar_at(mx, qkv_fk, args.row)
    if args.scan_rows:
        rows = []
        qkv_flat = row1(qkv_mlx)
        qkv_fk_flat = row1(qkv_fk)
        mx.eval(qkv_flat, qkv_fk_flat)
        for r in range(args.scan_rows):
            rr, rb = manual_q4_row(
                mx,
                norm_kernel.reshape(-1),
                att.in_proj_qkv.weight,
                att.in_proj_qkv.scales,
                att.in_proj_qkv.biases,
                r,
            )
            rows.append(
                {
                    "row": r,
                    "manual_bf16": rb,
                    "mlx": float(qkv_flat[0, r].astype(mx.float32).item()),
                    "fk": float(qkv_fk_flat[0, r].astype(mx.float32).item()),
                }
            )
        print({"scan_rows": rows})
    print(
        {
            "row": args.row,
            "manual_float": raw,
            "manual_bf16": rounded,
            "manual_bf16_dequant_float": raw_bf16w,
            "manual_bf16_dequant_bf16": rounded_bf16w,
            "mx_dequant_dot_float": float(deq_dot.item()),
            "mx_dequant_dot_bf16": bf16_scalar(mx, float(deq_dot.item())),
            **order_vals,
            "mlx_qlinear_3d": scalar_at(mx, qkv_mlx, args.row),
            "mlx_qlinear_1d": float(qkv_mlx_1d.reshape(-1)[args.row].astype(mx.float32).item()),
            "mlx_qlinear_2d": scalar_at(mx, qkv_mlx_2d, args.row),
            "mlx_qlinear_2d_contig": scalar_at(mx, qkv_mlx_2d_contig, args.row),
            "mlx_qlinear_3d_contig": scalar_at(mx, qkv_mlx_3d_contig, args.row),
            "fusekit": fk,
            "diff_mlx3d_fk": scalar_at(mx, qkv_mlx, args.row) - fk,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
