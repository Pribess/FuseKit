"""Compare Gemma4 layer-0 decode intermediates against FuseKit scratch."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def as_array(x):
    ptr = getattr(x, "_fk_ptr", None)
    if ptr is None:
        return x
    import mlx.core as mx
    import numpy as np

    shape = tuple(int(v) for v in getattr(x, "shape", ()))
    n = math.prod(shape) if shape else 1
    dtype = str(getattr(x, "dtype", ""))
    if "bfloat16" in dtype:
        raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(int(ptr))).copy()
        arr = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return mx.array(arr, dtype=mx.float32)
    if "float16" in dtype:
        raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(int(ptr))).copy()
        return mx.array(raw.view(np.float16).reshape(shape), dtype=mx.float16)
    if "float32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.float32)
    raise TypeError(f"unsupported native scratch dtype: {dtype}")


def current_row_like(x, ref=None):
    x = as_array(x)
    if ref is None:
        ref = x
    ref = as_array(ref)
    if x.ndim == 4:
        ref_shape = tuple(int(v) for v in ref.shape)
        # Decode graphs keep attention tensors either as [B, T, H, D]
        # before SDPA or [B, H, T, D] in cache/SDPA layout. Use the reference
        # layout to decide which axis is the active token row.
        if len(ref_shape) == 4 and ref_shape[1] == 1:
            return x[:, :1, :, :].reshape(1, -1)
        return x[:, :, :1, :].reshape(1, -1)
    if x.ndim >= 3:
        return x[:, :1, ...].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def compare(mx, a, b):
    d = mx.abs(current_row_like(a).astype(mx.float32) - current_row_like(b, a).astype(mx.float32))
    mx.eval(d)
    return {
        "max_abs_diff": float(mx.max(d).item()),
        "mean_abs_diff": float(mx.mean(d).item()),
        "gt_0_01": int(mx.sum(d > 0.01).item()),
        "gt_0_1": int(mx.sum(d > 0.1).item()),
    }


def scalar_int(v) -> int:
    if hasattr(v, "item"):
        return int(v.item())
    return int(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="hello")
    ap.add_argument("--decode-token", type=int, default=42)
    args = ap.parse_args()

    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.base import scaled_dot_product_attention

    model, tok = load(str(model_path(args.model)))
    text = text_model(model)
    core = text.model
    cache = text.make_cache()

    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    logits = model(decode, cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)

    disp = model._fk_dispatcher
    layer = core.layers[0]
    att = layer.self_attn
    c = cache[0]
    offset_after = scalar_int(getattr(c, "offset", getattr(c, "_idx", 0)))
    offset = max(offset_after - 1, 0)

    x_raw = disp._x_in.reshape(1, 1, -1).astype(disp._act_dt)
    x = (x_raw * core.embed_scale).astype(disp._act_dt)
    h = layer.input_layernorm(x)
    q = att.q_proj(h).reshape(1, 1, att.n_heads, att.head_dim)
    k = att.k_proj(h).reshape(1, 1, att.n_kv_heads, att.head_dim)
    v = att.v_proj(h).reshape(1, 1, att.n_kv_heads, att.head_dim)
    q_norm = att.q_norm(q)
    k_norm = att.k_norm(k)
    v_norm = att.v_norm(v)
    q_rope = att.rope(q_norm.transpose(0, 2, 1, 3), offset=offset)
    k_rope = att.rope(k_norm.transpose(0, 2, 1, 3), offset=offset)
    v_t = v_norm.transpose(0, 2, 1, 3)
    keys_live = c.keys[:, :, : offset + 1, :]
    vals_live = c.values[:, :, : offset + 1, :]
    sdpa = scaled_dot_product_attention(
        q_rope,
        keys_live,
        vals_live,
        cache=c,
        scale=att.scale,
        mask=None,
    )
    sdpa_flat = sdpa.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    o_proj = att.o_proj(sdpa_flat)
    h_attn = x + layer.post_attention_layernorm(o_proj)
    pre_ff = layer.pre_feedforward_layernorm(h_attn)
    mlp_gate = layer.mlp.gate_proj(pre_ff)
    mlp_up = layer.mlp.up_proj(pre_ff)
    mlp_act = nn.gelu_approx(mlp_gate) * mlp_up
    mlp_down = layer.mlp.down_proj(mlp_act)
    post_ff = layer.post_feedforward_layernorm(mlp_down)
    h_ff = h_attn + post_ff

    input_embeddings = core.embed_tokens(decode)
    per_embed = core._get_per_layer_inputs(decode, input_embeddings)
    per_embed_unscaled = core.embed_tokens_per_layer(decode).reshape(
        1, 1, core.config.num_hidden_layers, core.hidden_size_per_layer_input
    )
    per_projection = core.per_layer_model_projection(x)
    per_projection = per_projection * core.per_layer_projection_scale
    per_projection = mx.unflatten(
        per_projection,
        -1,
        (core.config.num_hidden_layers, core.hidden_size_per_layer_input),
    )
    per_projection_norm = core.per_layer_projection_norm(per_projection)
    per_all = (per_projection_norm + per_embed) * core.per_layer_input_scale
    row_mean = mx.mean(per_projection.astype(mx.float32).square(), axis=-1, keepdims=True)
    manual_scale = mx.rsqrt(row_mean + core.config.rms_norm_eps)
    manual_normed = (per_projection.astype(mx.float32) * manual_scale).astype(
        disp._act_dt
    )
    manual_weighted = (
        manual_normed * core.per_layer_projection_norm.weight.astype(disp._act_dt)
    ).astype(disp._act_dt)
    per_all_manual_kernel = (
        (manual_weighted.astype(mx.float32) + per_embed.astype(mx.float32))
        * float(core.per_layer_input_scale)
    ).astype(disp._act_dt)
    row_mean_swapped = mx.mean(per_embed.astype(mx.float32).square(), axis=-1, keepdims=True)
    manual_scale_swapped = mx.rsqrt(row_mean_swapped + core.config.rms_norm_eps)
    manual_normed_swapped = (per_embed.astype(mx.float32) * manual_scale_swapped).astype(
        disp._act_dt
    )
    manual_weighted_swapped = (
        manual_normed_swapped * core.per_layer_projection_norm.weight.astype(disp._act_dt)
    ).astype(disp._act_dt)
    per_all_manual_swapped_kernel = (
        (manual_weighted_swapped.astype(mx.float32) + per_projection.astype(mx.float32))
        * float(core.per_layer_input_scale)
    ).astype(disp._act_dt)
    per_embed_norm = core.per_layer_projection_norm(per_embed)
    per_all_swapped = (per_embed_norm + per_projection) * core.per_layer_input_scale
    per_projection_no_weight = mx.fast.rms_norm(
        per_projection, None, core.config.rms_norm_eps
    )
    per_all_no_weight = (per_projection_no_weight + per_embed) * core.per_layer_input_scale
    per_projection_plus_weight = mx.fast.rms_norm(
        per_projection,
        1.0 + core.per_layer_projection_norm.weight,
        core.config.rms_norm_eps,
    )
    per_all_plus_weight = (per_projection_plus_weight + per_embed) * core.per_layer_input_scale
    per0 = per_all[:, :, 0, :]

    pl_gate = layer.per_layer_input_gate(h_ff)
    pl_gate_act = nn.gelu_approx(pl_gate)
    pl_mul = pl_gate_act * per0
    pl_proj = layer.per_layer_projection(pl_mul)
    pl_norm = layer.post_per_layer_input_norm(pl_proj)
    h_per = h_ff + pl_norm
    h_out = h_per * layer.layer_scalar
    mx.eval(
        q,
        k,
        v,
        q_norm,
        k_norm,
        v_norm,
        q_rope,
        k_rope,
        v_t,
        sdpa,
        o_proj,
        h_attn,
        pre_ff,
        mlp_gate,
        mlp_up,
        mlp_act,
        mlp_down,
        post_ff,
        h_ff,
        per_embed,
        per_embed_unscaled,
        per_projection,
        per_projection_norm,
        per_all,
        per_all_manual_kernel,
        per_all_manual_swapped_kernel,
        per_all_swapped,
        per_all_no_weight,
        per_all_plus_weight,
        per0,
        pl_gate,
        pl_mul,
        pl_proj,
        pl_norm,
        h_out,
    )

    sids = {
        "x_scaled": 1,
        "norm": 2,
        "q_proj": 3,
        "q_norm_rope": disp._layer_scratch_id[("q_buf", 0)],
        "k_proj": 5,
        "k_norm": 6,
        "v_proj": 7,
        "v_norm": 8,
        "sdpa": 10,
        "o_proj": 13,
        "h_attn": 14,
        "pre_ff_gate": 15,
        "pre_ff_up": 16,
        "mlp_act": 25,
        "mlp_down": 26,
        "h_ff": 27,
        "per_layer_projection": 39,
        "per_layer_projection_norm": 40,
        "per_layer_all": 43,
        "per_layer0": 44,
        "per_gate": 28,
        "per_mul": 46,
        "per_proj": 47,
        "h_out": 48,
    }
    refs = {
        "x_scaled": x,
        "norm": h,
        "q_proj": q,
        "q_norm_rope": q_rope.transpose(0, 2, 1, 3),
        "k_proj": k,
        "k_norm": k_norm,
        "v_proj": v,
        "v_norm": v_norm,
        "sdpa": sdpa,
        "o_proj": o_proj,
        "h_attn": h_attn,
        "pre_ff_gate": mlp_gate,
        "pre_ff_up": mlp_up,
        "mlp_act": mlp_act,
        "mlp_down": mlp_down,
        "h_ff": h_ff,
        "per_layer_projection": per_projection,
        "per_layer_projection_norm": per_projection_norm,
        "per_layer_all": per_all,
        "per_layer0": per0,
        "per_gate": pl_gate,
        "per_mul": pl_mul,
        "per_proj": pl_proj,
        "h_out": h_out,
    }
    rows = {}
    for name, sid in sids.items():
        scratch = disp._scratch_buf(sid)
        try:
            rows[name] = compare(mx, refs[name], scratch)
        except Exception as exc:
            rows[name] = {
                "error": str(exc),
                "ref_shape": tuple(int(v) for v in as_array(refs[name]).shape),
                "scratch_shape": tuple(int(v) for v in as_array(scratch).shape),
                "ref_current": tuple(int(v) for v in current_row_like(refs[name]).shape),
                "scratch_current": tuple(int(v) for v in current_row_like(scratch, refs[name]).shape),
            }
    try:
        rows["cache_k_current"] = compare(mx, k_rope, c.keys[:, :, offset : offset + 1, :])
    except Exception as exc:
        rows["cache_k_current"] = {"error": str(exc)}
    try:
        rows["cache_v_current"] = compare(mx, v_t, c.values[:, :, offset : offset + 1, :])
    except Exception as exc:
        rows["cache_v_current"] = {"error": str(exc)}
    for name, ref in {
        "per_layer_all_candidate_swapped": per_all_swapped,
        "per_layer_all_candidate_manual_kernel": per_all_manual_kernel,
        "per_layer_all_candidate_manual_swapped_kernel": per_all_manual_swapped_kernel,
        "per_layer_all_candidate_no_weight": per_all_no_weight,
        "per_layer_all_candidate_plus_weight": per_all_plus_weight,
    }.items():
        try:
            rows[name] = compare(mx, ref, disp._scratch_buf(sids["per_layer_all"]))
        except Exception as exc:
            rows[name] = {"error": str(exc)}
    try:
        rows["per_layer_embed_runtime_unscaled"] = compare(
            mx,
            per_embed_unscaled,
            disp._per_layer_embed_buf.reshape(
                1,
                disp._active_m,
                core.config.num_hidden_layers,
                core.hidden_size_per_layer_input,
            ),
        )
    except Exception as exc:
        rows["per_layer_embed_runtime_unscaled"] = {"error": str(exc)}
    try:
        disp_embed = disp._per_layer_embed_module(decode).reshape(
            1,
            1,
            core.config.num_hidden_layers,
            core.hidden_size_per_layer_input,
        )
        mx.eval(disp_embed)
        rows["per_layer_embed_dispatcher_module"] = compare(
            mx,
            per_embed_unscaled,
            disp_embed,
        )
    except Exception as exc:
        rows["per_layer_embed_dispatcher_module"] = {"error": str(exc)}
    try:
        disp._seed_token_derived_inputs(decode, 1)
        rows["per_layer_embed_runtime_after_manual_seed"] = compare(
            mx,
            per_embed_unscaled,
            disp._per_layer_embed_buf.reshape(
                1,
                disp._active_m,
                core.config.num_hidden_layers,
                core.hidden_size_per_layer_input,
            ),
        )
    except Exception as exc:
        rows["per_layer_embed_runtime_after_manual_seed"] = {"error": str(exc)}
    print(
        json.dumps(
            {
                "offset": offset,
                "offset_after": offset_after,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
