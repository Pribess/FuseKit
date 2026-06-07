"""Compare MLX SwiGLU semantics against candidate fused epilogues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def row1(x):
    if x.ndim >= 3:
        return x[:, :1, ...].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.gated_delta import gated_delta_update

    model, tok = load(str(model_path(args.model)))
    text = text_model(model)
    core = text.model
    layer = core.layers[0]
    att = layer.linear_attn
    cache = text.make_cache()

    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    hidden = core.embed_tokens(ids)
    fa_mask = create_attention_mask(hidden, cache[core.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache[core.ssm_idx])
    for lyr, c in zip(core.layers, cache):
        hidden = lyr(hidden, mask=ssm_mask if lyr.is_linear else fa_mask, cache=c)
    mx.eval(hidden)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    x = core.embed_tokens(decode)
    norm_x = layer.input_layernorm(x)
    qkv = att.in_proj_qkv(norm_x)
    z = att.in_proj_z(norm_x).reshape(1, 1, att.num_v_heads, att.head_v_dim)
    b = att.in_proj_b(norm_x)
    a = att.in_proj_a(norm_x)
    conv_input = mx.concatenate([cache[0][0], qkv], axis=1)
    conv = nn.silu(att.conv1d(conv_input))
    q, k, v = [
        t.reshape(1, 1, h, d)
        for t, h, d in zip(
            mx.split(conv, [att.key_dim, 2 * att.key_dim], -1),
            [att.num_k_heads, att.num_k_heads, att.num_v_heads],
            [att.head_k_dim, att.head_k_dim, att.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q_scaled = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k_scaled = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    gd, _ = gated_delta_update(
        q_scaled,
        k_scaled,
        v,
        a,
        b,
        att.A_log,
        att.dt_bias,
        cache[0][1],
        None,
        use_kernel=not att.training,
    )
    h = x + att.out_proj(att.norm(gd, z).reshape(1, 1, -1))
    mlp_norm = layer.post_attention_layernorm(h)
    gate = layer.mlp.gate_proj(mlp_norm)
    up = layer.mlp.up_proj(mlp_norm)

    actual = swiglu(gate, up)
    old_float_epilogue = (
        gate.astype(mx.float32)
        / (1.0 + mx.exp(-gate.astype(mx.float32)))
        * up.astype(mx.float32)
    ).astype(gate.dtype)
    explicit_ops = nn.silu(gate) * up
    precise_ops = (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(
        gate.dtype
    )
    rounded_gate_up = (
        nn.silu(gate.astype(gate.dtype)).astype(gate.dtype) * up.astype(gate.dtype)
    ).astype(gate.dtype)

    vals = {
        "actual_dtype": str(actual.dtype),
        "gate_dtype": str(gate.dtype),
        "up_dtype": str(up.dtype),
    }
    for name, cand in [
        ("explicit_ops", explicit_ops),
        ("old_float_epilogue", old_float_epilogue),
        ("precise_ops", precise_ops),
        ("rounded_gate_up", rounded_gate_up),
    ]:
        diff = mx.abs(row1(actual).astype(mx.float32) - row1(cand).astype(mx.float32))
        mx.eval(diff)
        vals[name] = {
            "max_abs_diff": float(mx.max(diff).item()),
            "mean_abs_diff": float(mx.mean(diff).item()),
            "gt_0": int(mx.sum(diff > 0).item()),
            "gt_0_001": int(mx.sum(diff > 0.001).item()),
            "gt_0_01": int(mx.sum(diff > 0.01).item()),
        }
    print(json.dumps(vals, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
