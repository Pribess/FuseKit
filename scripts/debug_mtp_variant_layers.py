"""Locate first layer drift in FuseKit MTP M=N verifier variants."""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.activations import swiglu
from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model

from debug_qwen35_linear_layer import find_linear_layer_sids


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _text_model(model):
    inner = getattr(model, "_fk_original", model)
    return getattr(inner, "language_model", inner)


def _argmax_id(logits) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _as_array(x):
    ptr = getattr(x, "_fk_ptr", None)
    if ptr is None:
        return x
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
    if "uint32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_uint32 * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.uint32)
    raise TypeError(f"unsupported native scratch dtype: {dtype}")


def _view_rows(x, rows: int):
    x = _as_array(x)
    if x.ndim == 3:
        return x[:, :rows, :]
    if x.ndim == 2:
        return x[:rows, :].reshape(1, rows, -1)
    return x.reshape(1, rows, -1)


def _diff(a, b) -> tuple[float, float]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def _print_cmp(name: str, raw, fk) -> None:
    mx.eval(raw, fk)
    if raw.shape != fk.shape:
        print(
            "[mtp-layer-debug] "
            f"  {name:<18} shape_mismatch raw_shape={tuple(raw.shape)} "
            f"fk_shape={tuple(fk.shape)}",
            flush=True,
        )
        return
    max_abs, mean_abs = _diff(raw, fk)
    print(
        "[mtp-layer-debug] "
        f"  {name:<18} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"raw_shape={tuple(raw.shape)} fk_shape={tuple(fk.shape)}",
        flush=True,
    )


def _print_linear_internals(
    core,
    layer_idx: int,
    layer,
    x,
    cache_formula,
    ssm_mask,
    variant,
    rows: int,
) -> None:
    sids = find_linear_layer_sids(variant._tg, layer_idx, variant._hidden)
    att = layer.linear_attn
    norm_x = layer.input_layernorm(x)
    qkv = att.in_proj_qkv(norm_x)
    z = att.in_proj_z(norm_x).reshape(
        1, rows, att.num_v_heads, att.head_v_dim
    )
    b = att.in_proj_b(norm_x)
    a = att.in_proj_a(norm_x)
    a_dt = a + att.dt_bias
    qkv_masked = qkv
    if ssm_mask is not None:
        qkv_masked = mx.where(ssm_mask[..., None], qkv, 0)
    conv_state = cache_formula[layer_idx][0]
    ssm_state = cache_formula[layer_idx][1]
    conv_input = mx.concatenate([conv_state, qkv_masked], axis=1)
    conv_silu = nn.silu(att.conv1d(conv_input))
    conv_silu_split = conv_silu
    if rows > 1:
        n_keep = att.conv_kernel_size - 1
        conv_input0 = mx.concatenate([conv_state, qkv_masked[:, :1]], axis=1)
        conv_silu0 = nn.silu(att.conv1d(conv_input0))
        conv_state1 = mx.contiguous(conv_input0[:, -n_keep:])
        conv_input1 = mx.concatenate([conv_state1, qkv_masked[:, 1:2]], axis=1)
        conv_silu1 = nn.silu(att.conv1d(conv_input1))
        conv_silu_split = mx.concatenate([conv_silu0, conv_silu1], axis=1)
    q_raw, k_raw, v = [
        t.reshape(1, rows, h, d)
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
    mask_c = ssm_mask[:, :1] if ssm_mask is not None else None
    mask_d = ssm_mask[:, 1:] if ssm_mask is not None else None
    gd_c, conv_c, ssm_c = att._process_chunk(
        qkv_masked[:, :1],
        a[:, :1],
        b[:, :1],
        conv_state,
        ssm_state,
        mask_c,
    )
    gd_d, _, _ = att._process_chunk(
        qkv_masked[:, 1:],
        a[:, 1:],
        b[:, 1:],
        conv_c,
        ssm_c,
        mask_d,
    )
    gd = mx.concatenate([gd_c, gd_d], axis=1)
    gd_norm = mx.fast.rms_norm(gd, att.norm.weight, att.norm.eps)
    gated = (
        nn.silu(z.astype(mx.float32)) * gd_norm.astype(mx.float32)
    ).astype(gd.dtype)
    h_attn = x + att.out_proj(gated.reshape(1, rows, -1))
    post_norm = layer.post_attention_layernorm(h_attn)
    mlp_gate = layer.mlp.gate_proj(post_norm)
    mlp_up = layer.mlp.up_proj(post_norm)
    mlp_silu = swiglu(mlp_gate, mlp_up)
    layer_out = h_attn + layer.mlp.down_proj(mlp_silu)
    values = {
        "layer_in": x,
        "norm_x": norm_x,
        "qkv": qkv,
        "z": z.reshape(1, rows, -1),
        "a_dt": a,
        "b": b,
        "conv_silu": conv_silu,
        "q_scaled": q_scaled.reshape(1, rows, -1),
        "k_scaled": k_scaled.reshape(1, rows, -1),
        "v": v.reshape(1, rows, -1),
        "g": g,
        "b_sigmoid": b_sigmoid,
        "gd": gd.reshape(1, rows, -1),
        "gd_norm": gd_norm.reshape(1, rows, -1),
        "gated_float": gated.reshape(1, rows, -1),
        "gated_flat": gated.reshape(1, rows, -1),
        "h_attn": h_attn,
        "post_norm": post_norm,
        "mlp_gate": mlp_gate,
        "mlp_up": mlp_up,
        "mlp_silu": mlp_silu,
        "layer_out": layer_out,
    }
    mx.eval(*values.values())
    print(f"[mtp-layer-debug] layer={layer_idx:02d} internals", flush=True)
    for name, raw_value in values.items():
        sid = sids.get(name)
        if sid is None:
            continue
        fk_value = _view_rows(variant._scratch_buf(sid), rows).astype(mx.float32)
        raw_rows = _view_rows(raw_value, rows).astype(mx.float32)
        if raw_rows.shape != fk_value.shape:
            print(
                "[mtp-layer-debug] "
                f"  {name:<18} shape_mismatch raw_shape={tuple(raw_rows.shape)} "
                f"fk_shape={tuple(fk_value.shape)}",
                flush=True,
            )
            continue
        _print_cmp(name, raw_rows, fk_value)
    sid = sids.get("conv_silu")
    if sid is not None:
        fk_value = _view_rows(variant._scratch_buf(sid), rows).astype(mx.float32)
        _print_cmp("conv_silu_split", _view_rows(conv_silu_split, rows), fk_value)


def _find_full_layer_sids(tg: dict, layer_idx: int) -> dict[str, int]:
    tasks = {int(t["id"]): t for t in tg["tasks"]}
    ids = [int(i) for i in tg["decode_pipeline"] if tasks[int(i)].get("layer_idx") == layer_idx]
    sids: dict[str, int] = {}
    hidden_outs: list[int] = []
    for i in ids:
        t = tasks[i]
        k = t.get("kernel", "")
        outs = [b for b in t.get("outputs", []) if b.get("role") == "scratch"]
        ins = [b for b in t.get("inputs", []) if b.get("role") == "scratch"]
        if "fusekit_op_rmsnorm_" in k and ins and outs:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 2560:
                if "layer_in" not in sids:
                    sids["layer_in"] = ins[0]["scratch_id"]
                    sids["norm_x"] = outs[0]["scratch_id"]
                elif "post_norm" not in sids:
                    sids["post_norm"] = outs[0]["scratch_id"]
        elif "fusekit_qmv_fanout_" in k and len(outs) == 3:
            shapes = [o.get("shape") or [] for o in outs]
            if shapes[0] and shapes[0][-1] == 8192:
                sids["q_proj"] = outs[0]["scratch_id"]
                sids["k_proj"] = outs[1]["scratch_id"]
                sids["v_proj"] = outs[2]["scratch_id"]
            elif shapes[0] and shapes[0][-1] == 9216:
                sids["mlp_gate"] = outs[0]["scratch_id"]
                sids["mlp_up"] = outs[1]["scratch_id"]
        elif "fusekit_rmsnorm_rope_qgate_decode_" in k and len(outs) == 2:
            sids["q_rope"] = outs[0]["scratch_id"]
            sids["gate"] = outs[1]["scratch_id"]
        elif "fusekit_sdpa_decode_" in k and len(outs) == 1:
            sids["sdpa"] = outs[0]["scratch_id"]
        elif "fusekit_chain_" in k and len(outs) == 1:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 4096:
                sids["gated_sdpa"] = outs[0]["scratch_id"]
            elif shape and shape[-1] == 9216:
                sids["mlp_silu"] = outs[0]["scratch_id"]
        elif "fusekit_linear_residual_" in k and len(outs) == 1:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 2560:
                hidden_outs.append(outs[0]["scratch_id"])
    if hidden_outs:
        sids["h_attn"] = hidden_outs[0]
        sids["layer_out"] = hidden_outs[-1]
    return sids


def _cache_offset(c) -> int:
    if hasattr(c, "offset"):
        off = getattr(c, "offset")
        return int(off.item()) if hasattr(off, "item") else int(off)
    if hasattr(c, "_idx"):
        return int(getattr(c, "_idx"))
    return 0


def _cache_prefix(arr, length: int):
    if arr is None:
        return None
    return arr[:, :, :length, :]


def _scratch_rows(variant, sid: int, rows: int):
    x = _as_array(variant._scratch_buf(sid))
    if x.ndim == 4:
        return x[:, :, :rows, :].astype(mx.float32)
    return _view_rows(x, rows).astype(mx.float32)


def _print_full_internals(
    layer_idx: int,
    layer,
    x,
    cache_formula,
    cache_fk,
    fa_mask,
    variant,
    rows: int,
) -> None:
    sids = _find_full_layer_sids(variant._tg, layer_idx)
    att = layer.self_attn
    c = cache_formula[layer_idx]
    offset = _cache_offset(c)
    norm_x = layer.input_layernorm(x)
    q_proj = att.q_proj(norm_x)
    q, gate = mx.split(q_proj.reshape(1, rows, att.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(1, rows, -1)
    k_proj = att.k_proj(norm_x)
    v_proj = att.v_proj(norm_x)
    q_norm = att.q_norm(q).transpose(0, 2, 1, 3)
    k_norm = att.k_norm(k_proj.reshape(1, rows, att.num_key_value_heads, -1)).transpose(
        0, 2, 1, 3
    )
    values = v_proj.reshape(1, rows, att.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    q_rope = att.rope(q_norm, offset=offset)
    k_rope = att.rope(k_norm, offset=offset)
    prev_k = _cache_prefix(getattr(c, "keys", None), offset)
    prev_v = _cache_prefix(getattr(c, "values", None), offset)
    keys = k_rope if prev_k is None else mx.concatenate([prev_k, k_rope], axis=2)
    vals = values if prev_v is None else mx.concatenate([prev_v, values], axis=2)
    sdpa = scaled_dot_product_attention(
        q_rope, keys, vals, None, scale=att.scale, mask=fa_mask
    )
    sdpa_flat = sdpa.transpose(0, 2, 1, 3).reshape(1, rows, -1)
    gated = sdpa_flat * mx.sigmoid(gate)
    attn_proj = att.o_proj(gated)
    h_attn = x + attn_proj
    post_norm = layer.post_attention_layernorm(h_attn)
    mlp_gate = layer.mlp.gate_proj(post_norm)
    mlp_up = layer.mlp.up_proj(post_norm)
    mlp_silu = swiglu(mlp_gate, mlp_up)
    layer_out = h_attn + layer.mlp.down_proj(mlp_silu)
    values_by_name = {
        "layer_in": x,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "q_rope": q_rope,
        "gate": gate,
        "sdpa": sdpa_flat,
        "gated_sdpa": gated,
        "h_attn": h_attn,
        "post_norm": post_norm,
        "mlp_gate": mlp_gate,
        "mlp_up": mlp_up,
        "mlp_silu": mlp_silu,
        "layer_out": layer_out,
    }
    mx.eval(*values_by_name.values())
    print(f"[mtp-layer-debug] layer={layer_idx:02d} full internals offset={offset}", flush=True)
    for name, raw_value in values_by_name.items():
        sid = sids.get(name)
        if sid is None:
            continue
        fk_value = _scratch_rows(variant, sid, rows)
        raw_rows = (
            raw_value.astype(mx.float32)
            if raw_value.ndim == 4
            else _view_rows(raw_value, rows).astype(mx.float32)
        )
        _print_cmp(name, raw_rows, fk_value)
    layer_in_sid = sids.get("layer_in")
    if layer_in_sid is not None:
        fk_layer_in = _scratch_rows(variant, layer_in_sid, rows)
        fk_norm_x = layer.input_layernorm(fk_layer_in.astype(x.dtype))
        fk_ref_q_proj = att.q_proj(fk_norm_x)
        fk_ref_k_proj = att.k_proj(fk_norm_x)
        fk_ref_v_proj = att.v_proj(fk_norm_x)
        mx.eval(fk_ref_q_proj, fk_ref_k_proj, fk_ref_v_proj)
        for name, ref_value in (
            ("q_proj_from_fk_input", fk_ref_q_proj),
            ("k_proj_from_fk_input", fk_ref_k_proj),
            ("v_proj_from_fk_input", fk_ref_v_proj),
        ):
            sid = sids.get(name.removesuffix("_from_fk_input"))
            if sid is None:
                continue
            fk_value = _scratch_rows(variant, sid, rows)
            _print_cmp(name, _view_rows(ref_value, rows), fk_value)
    c_fk = cache_fk[layer_idx]
    fk_k = getattr(c_fk, "keys", None)
    fk_v = getattr(c_fk, "values", None)
    if fk_k is not None and fk_v is not None:
        _print_cmp("k_store", k_rope, fk_k[:, :, offset : offset + rows, :].astype(mx.float32))
        _print_cmp("v_store", values, fk_v[:, :, offset : offset + rows, :].astype(mx.float32))


def _cache_offsets(cache) -> list[int | None]:
    out = []
    for c in cache:
        if hasattr(c, "_idx"):
            out.append(int(getattr(c, "_idx")))
        elif hasattr(c, "offset"):
            off = getattr(c, "offset")
            try:
                out.append(int(off.item()) if hasattr(off, "item") else int(off))
            except Exception:
                out.append(None)
        else:
            out.append(None)
    return out[:8]


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def _advance_cache(model, cache, tokens: list[int]) -> None:
    for tok in tokens:
        out = model(mx.array([[tok]], dtype=mx.uint32), cache=cache, return_hidden=True)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--main", type=int)
    ap.add_argument("--draft", type=int)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    text = _text_model(raw_model)
    core = text.model

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    cache_raw = raw_model.make_cache()
    cache_fk = raw_model.make_cache()
    cache_formula = raw_model.make_cache()

    if args.main is not None or args.draft is not None or args.prefix:
        if args.main is None or args.draft is None:
            raise RuntimeError("--main and --draft are required when --prefix is used")
        _ = raw_model(prompt, cache=cache_raw)
        _ = raw_model(prompt, cache=cache_fk)
        _ = raw_model(prompt, cache=cache_formula)
        prefix_tokens = _parse_tokens(args.prefix)
        _advance_cache(raw_model, cache_raw, prefix_tokens)
        _advance_cache(raw_model, cache_fk, prefix_tokens)
        _advance_cache(raw_model, cache_formula, prefix_tokens)
        next_main = int(args.main)
        draft = int(args.draft)
        first = prefix_tokens[0] if prefix_tokens else -1
    else:
        logits_raw = raw_model(prompt, cache=cache_raw)
        _ = raw_model(prompt, cache=cache_fk)
        _ = raw_model(prompt, cache=cache_formula)
        first = _argmax_id(logits_raw)

        tok0 = mx.array([[first]], dtype=mx.uint32)
        logits_raw_0, hidden_raw_0 = raw_model(tok0, cache=cache_raw, return_hidden=True)
        _ = raw_model(tok0, cache=cache_fk, return_hidden=True)
        _ = raw_model(tok0, cache=cache_formula, return_hidden=True)
        next_main = _argmax_id(logits_raw_0)

        mtp_cache = raw_model.make_mtp_cache()
        mtp_ids = mx.array([[next_main]], dtype=mx.uint32)
        mtp_logits = raw_model.mtp_forward(hidden_raw_0[:, -1:, :], mtp_ids, mtp_cache)
        draft = _argmax_id(mtp_logits)
    verify = mx.array([[next_main, draft]], dtype=mx.uint32)
    print(
        "[mtp-layer-debug] "
        f"prompt_tokens={len(prompt_ids)} first={first} next={next_main} draft={draft} "
        f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )

    hidden = core.embed_tokens(verify)
    fa_mask = create_attention_mask(hidden, cache_raw[core.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache_raw[core.ssm_idx])
    raw_rows = []
    raw_inputs = []
    limit = args.max_layers or len(core.layers)
    for li, (layer, c) in enumerate(zip(core.layers[:limit], cache_raw[:limit])):
        raw_inputs.append(_view_rows(hidden, 2).astype(mx.float32))
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=c, n_confirmed=1)
        raw_rows.append(_view_rows(hidden, 2).astype(mx.float32))

    logits_fk, hidden_fk = model(
        verify,
        cache=cache_fk,
        return_hidden=True,
        n_confirmed=1,
    )
    mx.eval(logits_fk, hidden_fk)
    variant = model._get_variant_dispatcher(2)
    if variant is None:
        raise RuntimeError("MTP M=2 variant dispatcher was not loaded")

    for li, raw in enumerate(raw_rows):
        sid = variant._layer_scratch_id.get(("out_buf", li))
        if sid is None:
            print(f"[mtp-layer-debug] layer={li:02d} missing out_buf", flush=True)
            continue
        fk = _view_rows(variant._scratch_buf(sid), 2).astype(mx.float32)
        mx.eval(raw, fk)
        max_abs, mean_abs = _diff(raw, fk)
        status = "DRIFT" if max_abs > args.threshold else "ok"
        layer = core.layers[li]
        kind = "linear" if getattr(layer, "is_linear", False) else "full"
        print(
            "[mtp-layer-debug] "
            f"layer={li:02d} kind={kind:<6} max_abs={max_abs:.6f} "
            f"mean_abs={mean_abs:.6f} status={status}",
            flush=True,
        )
        if max_abs > args.threshold:
            if getattr(layer, "is_linear", False):
                _print_linear_internals(
                    core,
                    li,
                    layer,
                    raw_inputs[li],
                    cache_formula,
                    ssm_mask,
                    variant,
                    2,
                )
            else:
                _print_full_internals(
                    li,
                    layer,
                    raw_inputs[li],
                    cache_formula,
                    cache_fk,
                    fa_mask,
                    variant,
                    2,
                )
            if li == 0:
                sids = find_linear_layer_sids(variant._tg, 0, variant._hidden)
                layer0 = core.layers[0]
                att = layer0.linear_attn
                x = core.embed_tokens(verify)
                ssm_mask0 = create_ssm_mask(x, cache_formula[core.ssm_idx])
                norm_x = layer0.input_layernorm(x)
                qkv = att.in_proj_qkv(norm_x)
                z = att.in_proj_z(norm_x).reshape(
                    1, 2, att.num_v_heads, att.head_v_dim
                )
                b = att.in_proj_b(norm_x)
                a = att.in_proj_a(norm_x)
                a_dt = a + att.dt_bias
                qkv_masked = qkv
                if ssm_mask0 is not None:
                    qkv_masked = mx.where(ssm_mask0[..., None], qkv, 0)
                conv_state = cache_formula[0][0]
                ssm_state = cache_formula[0][1]
                conv_input = mx.concatenate([conv_state, qkv_masked], axis=1)
                conv_silu = nn.silu(att.conv1d(conv_input))
                q_raw, k_raw, v = [
                    t.reshape(1, 2, h, d)
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
                mask_c = ssm_mask0[:, :1] if ssm_mask0 is not None else None
                mask_d = ssm_mask0[:, 1:] if ssm_mask0 is not None else None
                gd_c, conv_c, ssm_c = att._process_chunk(
                    qkv_masked[:, :1],
                    a[:, :1],
                    b[:, :1],
                    conv_state,
                    ssm_state,
                    mask_c,
                )
                gd_d, _, _ = att._process_chunk(
                    qkv_masked[:, 1:],
                    a[:, 1:],
                    b[:, 1:],
                    conv_c,
                    ssm_c,
                    mask_d,
                )
                gd = mx.concatenate([gd_c, gd_d], axis=1)
                gd_norm = mx.fast.rms_norm(gd, att.norm.weight, att.norm.eps)
                gated = (
                    nn.silu(z.astype(mx.float32)) * gd_norm.astype(mx.float32)
                ).astype(gd.dtype)
                h_attn = x + att.out_proj(gated.reshape(1, 2, -1))
                post_norm = layer0.post_attention_layernorm(h_attn)
                mlp_gate = layer0.mlp.gate_proj(post_norm)
                mlp_up = layer0.mlp.up_proj(post_norm)
                mlp_silu = swiglu(mlp_gate, mlp_up)
                layer_out = h_attn + layer0.mlp.down_proj(mlp_silu)
                mx.eval(
                    x,
                    norm_x,
                    qkv,
                    qkv_masked,
                    z,
                    a_dt,
                    b,
                    conv_silu,
                    q_scaled,
                    k_scaled,
                    v,
                    g,
                    b_sigmoid,
                    gd,
                    gated,
                    h_attn,
                    post_norm,
                    layer_out,
                )
                print("[mtp-layer-debug] layer=00 internals", flush=True)
                for name, raw_value in (
                    ("layer_in", x),
                    ("norm_x", norm_x),
                    ("qkv", qkv),
                    ("z", z.reshape(1, 2, -1)),
                    ("a_dt", a_dt),
                    ("b", b),
                    ("conv_silu", conv_silu),
                    ("q_scaled", q_scaled.reshape(1, 2, -1)),
                    ("k_scaled", k_scaled.reshape(1, 2, -1)),
                    ("v", v.reshape(1, 2, -1)),
                    ("g", g),
                    ("b_sigmoid", b_sigmoid),
                    ("gd", gd.reshape(1, 2, -1)),
                    ("gated", gated.reshape(1, 2, -1)),
                    ("h_attn", h_attn),
                    ("post_norm", post_norm),
                    ("layer_out", layer_out),
                ):
                    sid_name = {
                        "layer_in": "layer_in",
                        "norm_x": "norm_x",
                        "qkv": "qkv",
                        "z": "z",
                        "a_dt": "a_dt",
                        "b": "b",
                        "conv_silu": "conv_silu",
                        "q_scaled": "q_scaled",
                        "k_scaled": "k_scaled",
                        "v": "v",
                        "g": "g",
                        "b_sigmoid": "b_sigmoid",
                        "gd": "gd",
                        "gated": "gated_float",
                        "h_attn": "h_attn",
                        "post_norm": "post_norm",
                        "layer_out": "layer_out",
                    }.get(name)
                    sid = sids.get(sid_name or "")
                    if sid is None:
                        continue
                    fk_value = _view_rows(variant._scratch_buf(sid), 2).astype(
                        mx.float32
                    )
                    _print_cmp(name, _view_rows(raw_value, 2), fk_value)
            break

    max_abs, mean_abs = _diff(
        _view_rows(raw_rows[-1], 2),
        _view_rows(hidden_fk, 2).astype(mx.float32),
    )
    print(
        "[mtp-layer-debug] "
        f"final_hidden max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"offsets_raw={_cache_offsets(cache_raw)} offsets_fk={_cache_offsets(cache_fk)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
