"""Compare one Qwen3.5 single-decode layer: raw formula vs FuseKit scratch."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model

from debug_qwen35_linear_layer import compare, find_linear_layer_sids, row1


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def _scratch_row(dispatcher, sid: int):
    return row1(dispatcher._scratch_buf(sid))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--decode-token", type=int, required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    text = _text_model(raw_model)
    core = text.model

    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()
    prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    raw_model(prompt, cache=cache_raw)
    raw_model(prompt, cache=cache_fk)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)

    hidden = core.embed_tokens(decode)
    fa_mask = create_attention_mask(hidden, cache_raw[core.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache_raw[core.ssm_idx])
    for li, (layer, c) in enumerate(zip(core.layers, cache_raw)):
        if li == args.layer:
            break
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=c)
        mx.eval(hidden)

    layer = core.layers[args.layer]
    if not layer.is_linear:
        raise RuntimeError(f"layer {args.layer} is not linear-attention")
    att = layer.linear_attn
    mask = ssm_mask

    norm_x = layer.input_layernorm(hidden)
    qkv = att.in_proj_qkv(norm_x)
    z = att.in_proj_z(norm_x).reshape(1, 1, att.num_v_heads, att.head_v_dim)
    b = att.in_proj_b(norm_x)
    a = att.in_proj_a(norm_x)
    a_dt = a + att.dt_bias
    qkv_masked = qkv if mask is None else mx.where(mask[..., None], qkv, 0)
    conv_state = cache_raw[args.layer][0]
    ssm_state = cache_raw[args.layer][1]
    conv_input = mx.concatenate([conv_state, qkv_masked], axis=1)
    conv_silu = nn.silu(att.conv1d(conv_input))
    q_raw, k_raw, v = [
        t[:, :1, :].reshape(1, 1, h, d)
        for t, h, d in zip(
            mx.split(conv_silu, [att.key_dim, 2 * att.key_dim], -1),
            [att.num_k_heads, att.num_k_heads, att.num_v_heads],
            [att.head_k_dim, att.head_k_dim, att.head_v_dim],
        )
    ]
    inv_scale = k_raw.shape[-1] ** -0.5
    q_scaled = (inv_scale**2) * mx.fast.rms_norm(q_raw, None, 1e-6)
    k_scaled = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)
    g = mx.exp(-mx.exp(att.A_log.astype(mx.float32)) * nn.softplus(a_dt))
    b_sigmoid = mx.sigmoid(b)
    gd, _, _ = att._process_chunk(qkv_masked, a, b, conv_state, ssm_state, mask)
    gd_norm = mx.fast.rms_norm(gd, att.norm.weight, att.norm.eps)
    gated = (
        nn.silu(z.astype(mx.float32)) * gd_norm.astype(mx.float32)
    ).astype(gd.dtype)
    gated_flat = gated.reshape(1, 1, -1)
    attn_proj = att.out_proj(gated_flat)
    h_attn = hidden + attn_proj
    post_norm = layer.post_attention_layernorm(h_attn)
    mlp_gate = layer.mlp.gate_proj(post_norm)
    mlp_up = layer.mlp.up_proj(post_norm)
    mlp_silu = swiglu(mlp_gate, mlp_up)
    mlp_down = layer.mlp.down_proj(mlp_silu)
    layer_out = h_attn + mlp_down
    values = {
        "norm_x": norm_x,
        "z": z,
        "qkv": qkv,
        "conv_input": conv_input,
        "conv_silu": conv_silu,
        "q_scaled": q_scaled,
        "k_scaled": k_scaled,
        "v": v,
        "a": a,
        "a_dt": a_dt,
        "b": b,
        "b_sigmoid": b_sigmoid,
        "g": g,
        "gd": gd,
        "gated_flat": gated_flat,
        "attn_proj": attn_proj,
        "h_attn": h_attn,
        "post_norm": post_norm,
        "mlp_gate": mlp_gate,
        "mlp_up": mlp_up,
        "mlp_silu": mlp_silu,
        "mlp_down": mlp_down,
        "layer_out": layer_out,
    }
    mx.eval(*values.values())

    out = model(decode, cache=cache_fk, return_hidden=True)
    mx.eval(*(out if isinstance(out, tuple) else (out,)))
    dispatcher = model._fk_dispatcher
    sids = find_linear_layer_sids(dispatcher._tg, args.layer, dispatcher._hidden)
    sid_by_name = {
        "norm_x": "norm_x",
        "z": "z",
        "qkv": "qkv",
        "conv_input": "conv_input",
        "conv_silu": "conv_silu",
        "q_scaled": "q_scaled",
        "k_scaled": "k_scaled",
        "v": "v",
        "a": "a_dt",
        "b": "b",
        "b_sigmoid": "b_sigmoid",
        "g": "g",
        "gd": "gd",
        "gated_flat": "gated_flat",
        "attn_proj": "attn_proj",
        "h_attn": "h_attn",
        "post_norm": "post_norm",
        "mlp_gate": "mlp_gate",
        "mlp_up": "mlp_up",
        "mlp_silu": "mlp_silu",
        "mlp_down": "mlp_down",
        "layer_out": "layer_out",
    }
    print(f"[single-layer-raw] layer={args.layer} sids={sids}", flush=True)
    for name, sid_name in sid_by_name.items():
        sid = sids.get(sid_name)
        if sid is None:
            print(f"[single-layer-raw] {name:<14} missing_sid={sid_name}", flush=True)
            continue
        stats = compare(mx, row1(values[name]), _scratch_row(dispatcher, sid))
        print(
            "[single-layer-raw] "
            f"{name:<14} max_abs={stats['max_abs_diff']:.6f} "
            f"mean_abs={stats['mean_abs_diff']:.6f} "
            f"gt_0_01={stats['gt_0_01']} gt_0_1={stats['gt_0_1']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
