"""POC: verify int4 affine dequant formula matches `mx.quantized_matmul`.

If our manual unpack matches mlx's reference, we have the exact formula
to port into an MSL kernel. The kernel just translates this Python
inner loop to threadgroup-cooperative MSL.
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx


def manual_dequant_matmul(x, w_packed, scales, biases, group_size, bits):
    """Dequantize weight and compute (out_dim,) = x (IN,) @ W.T (OUT,IN).

    mlx affine quantization layout (verified against `mx.quantize`):
        weight  uint32  (OUT, IN/8)  -- LSB-first 4-bit packed
        scales  fp16    (OUT, IN/group_size)
        biases  fp16    (OUT, IN/group_size)
        w[i, j] = uint4(weight[i, j/8], j%8) * scales[i, j/gs] + biases[i, j/gs]
    """
    assert bits == 4
    assert w_packed.dtype == mx.uint32
    OUT, IN_W = w_packed.shape
    LANES = 32 // bits  # 8 for bits=4
    IN_DIM = IN_W * LANES
    groups = IN_DIM // group_size

    # Unpack all 4-bit lanes. Shift + mask, broadcast over lane offsets.
    # shifts shape (1, 1, LANES) → uint4 values shape (OUT, IN_W, LANES)
    shifts = mx.arange(LANES, dtype=mx.uint32) * bits
    w32 = w_packed[..., None]                                  # (OUT, IN_W, 1)
    q = (w32 >> shifts) & 0xF                                   # (OUT, IN_W, LANES)
    q = q.reshape(OUT, IN_DIM).astype(mx.float16)               # (OUT, IN_DIM)

    # Broadcast scale/bias per group.
    s_per_dim = mx.repeat(scales, group_size, axis=1)           # (OUT, IN_DIM)
    b_per_dim = mx.repeat(biases, group_size, axis=1)           # (OUT, IN_DIM)
    w = q * s_per_dim + b_per_dim                                # (OUT, IN_DIM)

    return (x[None, :] @ w.T)[0]                                 # (OUT,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dim",  type=int, default=896)
    ap.add_argument("--out-dim", type=int, default=896)
    ap.add_argument("--group",   type=int, default=64)
    args = ap.parse_args()

    mx.random.seed(0)
    w_dense = mx.random.normal((args.out_dim, args.in_dim), dtype=mx.float16) * 0.05
    w_packed, scales, biases = mx.quantize(w_dense, group_size=args.group, bits=4)
    x = mx.random.normal((args.in_dim,), dtype=mx.float16)
    mx.eval(w_packed, scales, biases, x)

    ref = mx.quantized_matmul(
        x[None, :], w_packed, scales, biases,
        transpose=True, group_size=args.group, bits=4,
    )[0]
    ours = manual_dequant_matmul(x, w_packed, scales, biases, args.group, 4)
    mx.eval(ref, ours)

    diff = mx.abs(ref.astype(mx.float32) - ours.astype(mx.float32))
    print(f"[poc] ref  first5 = {ref[:5].tolist()}")
    print(f"[poc] ours first5 = {ours[:5].tolist()}")
    print(f"[poc] max_err = {float(diff.max()):.6f}")
    print(f"[poc] mean_err = {float(diff.mean()):.6f}")
    ok = float(diff.max()) < 0.05  # fp16 rounding tolerance
    print(f"[poc] formula match: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
