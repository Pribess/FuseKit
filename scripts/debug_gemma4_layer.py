"""Compare one Gemma4 decode layer against FuseKit scratch buffers."""

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


def make_cache_for(model_or_text):
    from mlx_lm.models.cache import make_prompt_cache

    if hasattr(model_or_text, "make_cache"):
        return model_or_text.make_cache()
    return make_prompt_cache(model_or_text)


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
    if "uint32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_uint32 * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.uint32)
    raise TypeError(f"unsupported native scratch dtype: {dtype}")


def row_candidates(x):
    x = as_array(x)
    out = []
    if x.ndim == 5:
        for i in range(min(int(x.shape[3]), 16)):
            out.append((f"axis3[{i}]", x[:, :, :, i : i + 1, :].reshape(1, -1)))
        for i in range(min(int(x.shape[1]), 16)):
            out.append((f"axis1[{i}]", x[:, i : i + 1, :, :, :].reshape(1, -1)))
    elif x.ndim == 4:
        for i in range(min(int(x.shape[1]), 16)):
            out.append((f"axis1[{i}]", x[:, i : i + 1, :, :].reshape(1, -1)))
        for i in range(min(int(x.shape[2]), 16)):
            out.append((f"axis2[{i}]", x[:, :, i : i + 1, :].reshape(1, -1)))
    elif x.ndim >= 3:
        for i in range(min(int(x.shape[1]), 16)):
            out.append((f"axis1[{i}]", x[:, i : i + 1, ...].reshape(1, -1)))
    elif x.ndim == 2:
        for i in range(min(int(x.shape[0]), 16)):
            out.append((f"axis0[{i}]", x[i : i + 1, :]))
    else:
        out.append(("scalar", x.reshape(1, -1)))
    return out


def compare(mx, ref, got):
    best = None
    mismatches = []
    for ref_mode, ref_row in row_candidates(ref):
        for got_mode, got_row in row_candidates(got):
            a = ref_row.astype(mx.float32)
            b = got_row.astype(mx.float32)
            if a.shape != b.shape:
                mismatches.append((ref_mode, tuple(int(v) for v in a.shape), got_mode, tuple(int(v) for v in b.shape)))
                continue
            d = mx.abs(a - b)
            mx.eval(d)
            mean = float(mx.mean(d).item())
            row = {
                "max_abs_diff": float(mx.max(d).item()),
                "mean_abs_diff": mean,
                "gt_0_01": int(mx.sum(d > 0.01).item()),
                "gt_0_1": int(mx.sum(d > 0.1).item()),
                "ref_row_mode": ref_mode,
                "got_row_mode": got_mode,
                "ref_shape": tuple(int(v) for v in as_array(ref).shape),
                "got_shape": tuple(int(v) for v in as_array(got).shape),
            }
            if best is None or mean < best["mean_abs_diff"]:
                best = row
    if best is not None:
        return best
    ref0 = row_candidates(ref)[0][1]
    got0 = row_candidates(got)[0][1]
    return {
        "error": "shape mismatch",
        "ref_row": tuple(int(v) for v in ref0.shape),
        "got_row": tuple(int(v) for v in got0.shape),
        "ref_shape": tuple(int(v) for v in as_array(ref).shape),
        "got_shape": tuple(int(v) for v in as_array(got).shape),
        "mismatches": mismatches[:8],
    }


def cache_row(x, row: int):
    x = as_array(x)
    if x.ndim >= 4:
        return x[:, :, row : row + 1, :]
    if x.ndim == 3:
        return x[:, row : row + 1, :]
    return x


def scalar_int(v) -> int:
    if hasattr(v, "item"):
        return int(v.item())
    return int(v)


def run_until_layer(mx, core, inputs, cache, target_layer: int):
    input_embeddings = core.embed_tokens(inputs)
    hidden = input_embeddings
    if hasattr(core, "embed_scale"):
        hidden = hidden * core.embed_scale

    if getattr(core, "hidden_size_per_layer_input", None):
        per_layer_inputs = core._get_per_layer_inputs(inputs, input_embeddings)
        per_layer_inputs = core._project_per_layer_inputs(hidden, per_layer_inputs)
        per_layer_inputs = [
            per_layer_inputs[:, :, i, :] for i, _ in enumerate(core.layers)
        ]
    else:
        per_layer_inputs = [None] * len(core.layers)

    if cache is None:
        cache = [None] * len(core.layers)
    else:
        cache = cache + [None] * (len(core.layers) - len(cache))

    masks = core._make_masks(hidden, cache)
    intermediates = [(None, None)] * len(core.layers)
    for i in range(target_layer):
        layer = core.layers[i]
        c = cache[i]
        shared_kv, offset = intermediates[core.previous_kvs[i]]
        hidden, shared_kv, offset = layer(
            hidden,
            masks[i],
            c,
            per_layer_input=per_layer_inputs[i],
            shared_kv=shared_kv,
            offset=offset,
        )
        intermediates[i] = (shared_kv, offset)
        mx.eval(hidden)
    return hidden, masks, intermediates, per_layer_inputs, cache


def layer_refs(mx, core, inputs, cache, target_layer: int):
    import mlx.nn as nn
    from mlx_lm.models.base import scaled_dot_product_attention

    hidden, masks, intermediates, per_layer_inputs, cache = run_until_layer(
        mx, core, inputs, cache, target_layer
    )
    layer = core.layers[target_layer]
    att = layer.self_attn
    c = cache[target_layer]
    shared_kv, offset = intermediates[core.previous_kvs[target_layer]]
    if offset is None:
        offset = mx.array(c.offset) if c is not None else 0
    offset_i = scalar_int(offset)

    residual = hidden
    norm = layer.input_layernorm(hidden)

    b, l, _ = norm.shape
    q = att.q_proj(norm).reshape(b, l, att.n_heads, att.head_dim)
    q_norm = att.q_norm(q)
    q_trans = q_norm.transpose(0, 2, 1, 3)

    if shared_kv is not None:
        keys, values = shared_kv
        k = None
        v = None
        k_norm = None
        k_trans = None
        v_trans = None
        v_norm = None
    else:
        k = att.k_proj(norm).reshape(b, l, att.n_kv_heads, att.head_dim)
        v = k if att.use_k_eq_v else att.v_proj(norm).reshape(
            b, l, att.n_kv_heads, att.head_dim
        )
        k_norm = att.k_norm(k)
        k_trans = k_norm.transpose(0, 2, 1, 3)
        keys = att.rope(k_trans, offset=offset)
        v_norm = att.v_norm(v)
        v_trans = v_norm.transpose(0, 2, 1, 3)
        values = v_trans

    q_rope = att.rope(q_trans, offset=offset)
    if c is not None:
        keys, values = c.update_and_fetch(keys, values)

    sdpa = scaled_dot_product_attention(
        q_rope, keys, values, cache=c, scale=att.scale, mask=masks[target_layer]
    )
    sdpa_flat = sdpa.transpose(0, 2, 1, 3).reshape(b, l, -1)
    o_proj = att.o_proj(sdpa_flat)
    h_attn_norm = layer.post_attention_layernorm(o_proj)
    h_attn = residual + h_attn_norm

    if getattr(layer, "enable_moe", False):
        raise RuntimeError("Gemma4 MoE layer debug is not implemented here")

    pre_ff = layer.pre_feedforward_layernorm(h_attn)
    gate = layer.mlp.gate_proj(pre_ff)
    up = layer.mlp.up_proj(pre_ff)
    act = nn.gelu_approx(gate) * up
    down = layer.mlp.down_proj(act)
    post_ff = layer.post_feedforward_layernorm(down)
    h_ff = h_attn + post_ff

    refs = {
        "layer_in": hidden,
        "norm": norm,
        "q_proj": q,
        "q_norm": q_norm,
        "q_trans": q_trans,
        "q_rope_raw": q_rope,
        "q_rope": q_rope.reshape(b, att.n_kv_heads, att.n_heads // att.n_kv_heads, l, att.head_dim),
        "sdpa_q": q_rope.reshape(b, att.n_kv_heads, att.n_heads // att.n_kv_heads, l, att.head_dim),
        "sdpa_grouped": sdpa.reshape(b, att.n_kv_heads, att.n_heads // att.n_kv_heads, l, att.head_dim),
        "sdpa_flat": sdpa_flat,
        "o_proj": o_proj,
        "h_attn": h_attn,
        "pre_ff": pre_ff,
        "gate": gate,
        "up": up,
        "act": act,
        "down": down,
        "h_ff": h_ff,
    }
    if k_norm is not None:
        refs["k_proj"] = k
        refs["k_norm"] = k_norm
        refs["k_trans"] = k_trans
    if v_norm is not None:
        refs["v_proj"] = v
        refs["v_norm"] = v_norm
        refs["v_trans"] = v_trans
    if per_layer_inputs[target_layer] is not None:
        per = per_layer_inputs[target_layer]
        pl_gate = layer.per_layer_input_gate(h_ff)
        pl_gate_act = nn.gelu_approx(pl_gate)
        pl_mul = pl_gate_act * per
        pl_proj = layer.per_layer_projection(pl_mul)
        pl_norm = layer.post_per_layer_input_norm(pl_proj)
        refs["per_gate"] = pl_gate
        refs["per_mul"] = pl_mul
        refs["per_proj"] = pl_proj
        refs["h_out"] = (h_ff + pl_norm) * layer.layer_scalar
    else:
        refs["h_out"] = h_ff * layer.layer_scalar

    mx.eval(*refs.values())
    return refs, offset_i


def find_layer_sids(tg: dict, layer: int, hidden: int) -> dict[str, int]:
    tasks = [t for t in tg["tasks"] if t.get("layer_idx") == layer]
    sids: dict[str, int] = {}
    for t in tasks:
        k = str(t.get("kernel", ""))
        ins = t.get("inputs", [])
        outs = t.get("outputs", [])
        out = outs[0] if outs else None
        out_shape = (out.get("shape") or []) if out is not None else []
        in_shape = (ins[0].get("shape") if ins else None) or []
        paths = [str(i.get("source_path", "")) for i in ins]
        if out is not None:
            if any(".self_attn.q_proj.weight" in p for p in paths):
                sids["q_proj"] = out["scratch_id"]
                continue
            if any(".self_attn.k_proj.weight" in p for p in paths):
                sids["k_proj"] = out["scratch_id"]
                continue
            if any(".self_attn.v_proj.weight" in p for p in paths):
                sids["v_proj"] = out["scratch_id"]
                continue
        if "fusekit_rope_q_decode_" in k and ins and "scratch_id" in ins[0]:
            sids["q_rope"] = ins[0]["scratch_id"]
            continue
        if "fusekit_rope_kv_cache_store_decode_" in k:
            if len(ins) > 0 and "scratch_id" in ins[0]:
                sids["k_norm"] = ins[0]["scratch_id"]
            if len(ins) > 1 and "scratch_id" in ins[1]:
                sids["v_norm"] = ins[1]["scratch_id"]
            continue
        if not outs:
            continue
        first_sid = ins[0].get("scratch_id") if ins else None
        if k.startswith("fusekit_op_rmsnorm_"):
            if first_sid == sids.get("q_proj"):
                sids["q_norm"] = out["scratch_id"]
                continue
            if first_sid == sids.get("k_proj"):
                sids["k_norm"] = out["scratch_id"]
                continue
            if first_sid == sids.get("v_proj"):
                sids["v_norm"] = out["scratch_id"]
                continue
        if k.startswith("fusekit_op_transpose_"):
            if first_sid == sids.get("q_norm"):
                sids["q_trans"] = out["scratch_id"]
                continue
            if first_sid == sids.get("k_norm"):
                sids["k_trans"] = out["scratch_id"]
                continue
            if first_sid == sids.get("v_norm"):
                sids["v_trans"] = out["scratch_id"]
                continue
        if k.startswith("fusekit_op_rope_"):
            if first_sid == sids.get("q_trans"):
                sids["q_rope_raw"] = out["scratch_id"]
                continue
            if first_sid == sids.get("k_trans"):
                sids["k_rope"] = out["scratch_id"]
                continue
        if (
            k.startswith("fusekit_chain_")
            and first_sid == sids.get("q_rope_raw")
            and len(out_shape) == 5
        ):
            sids["q_rope"] = out["scratch_id"]
            continue
        if k.startswith("fusekit_op_rmsnorm_") and out_shape[-1:] == [hidden]:
            if "layer_in" not in sids and ins and "scratch_id" in ins[0]:
                sids["layer_in"] = ins[0]["scratch_id"]
                sids["norm"] = out["scratch_id"]
            elif "pre_ff" not in sids:
                sids["pre_ff"] = out["scratch_id"]
            continue
        if "fusekit_sdpa_decode_" in k:
            if ins and "scratch_id" in ins[0]:
                sids["sdpa_q"] = ins[0]["scratch_id"]
            if len(ins) > 1 and "scratch_id" in ins[1]:
                sids["sdpa_k_trace"] = ins[1]["scratch_id"]
            if len(ins) > 2 and "scratch_id" in ins[2]:
                sids["sdpa_v_trace"] = ins[2]["scratch_id"]
            sids["sdpa_grouped"] = out["scratch_id"]
            continue
        if k.startswith("fusekit_op_reshape_") and out_shape[-1:] and out_shape[-1] > hidden:
            sids["sdpa_flat"] = out["scratch_id"]
            continue
        if k.startswith("fusekit_op_quantizedmatmul_") and out_shape[-1:] == [hidden]:
            if in_shape and in_shape[-1] > hidden and "o_proj" not in sids:
                sids["o_proj"] = out["scratch_id"]
            elif in_shape and in_shape[-1] > hidden and "down" not in sids:
                sids["down"] = out["scratch_id"]
            elif in_shape and in_shape[-1] < hidden:
                sids["per_proj"] = out["scratch_id"]
            continue
        if k.startswith("fusekit_chain_") and len(outs) == 1 and out_shape[-1:] == [hidden]:
            if "o_proj" in sids and "h_attn" not in sids:
                sids["h_attn"] = out["scratch_id"]
            elif "down" in sids and "h_ff" not in sids:
                sids["h_ff"] = out["scratch_id"]
            elif "per_proj" in sids:
                sids["h_out"] = out["scratch_id"]
            continue
        if k.startswith("fusekit_chain_") and len(outs) == 2:
            if out_shape[-1:] and out_shape[-1] > hidden:
                sids["gate"] = outs[0]["scratch_id"]
                sids["up"] = outs[1]["scratch_id"]
            continue
        if k.startswith("fusekit_chain_") and len(outs) == 1 and out_shape[-1:] and out_shape[-1] > hidden:
            sids["act"] = out["scratch_id"]
            continue
    return sids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="hello")
    ap.add_argument("--decode-token", type=int, default=42)
    ap.add_argument("--force-prefix-steps", type=int, default=0)
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(model_path(args.model)))
    raw_model = getattr(model, "_fk_original", model)
    text = text_model(raw_model)
    core = text.model

    prompt = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    decode = mx.array([[args.decode_token]], dtype=mx.uint32)

    raw_cache = make_cache_for(text)
    y = raw_model(prompt, cache=raw_cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    if args.force_prefix_steps > 0:
        for _ in range(args.force_prefix_steps):
            y = raw_model(decode, cache=raw_cache)
            if isinstance(y, tuple):
                y = y[0]
            mx.eval(y)
    refs, raw_offset = layer_refs(mx, core, decode, raw_cache, args.layer)

    fk_cache = make_cache_for(text)
    y = model(prompt, cache=fk_cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    if args.force_prefix_steps > 0:
        for _ in range(args.force_prefix_steps):
            y = model(decode, cache=fk_cache)
            if isinstance(y, tuple):
                y = y[0]
            mx.eval(y)
    y = model(decode, cache=fk_cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)

    disp = model._fk_dispatcher
    sids = find_layer_sids(disp._tg, args.layer, int(core.config.hidden_size))
    rows = {}
    for name, sid in sorted(sids.items()):
        if name not in refs:
            continue
        try:
            rows[name] = compare(mx, refs[name], disp._scratch_buf(sid))
        except Exception as exc:
            rows[name] = {"error": str(exc), "scratch_id": sid}
        rows[name]["scratch_id"] = sid

    for prefix in ("q", "k", "v"):
        norm_sid = sids.get(f"{prefix}_norm")
        trans_sid = sids.get(f"{prefix}_trans")
        if norm_sid is None or trans_sid is None:
            continue
        try:
            local = as_array(disp._scratch_buf(norm_sid)).transpose(0, 2, 1, 3)
            rows[f"{prefix}_trans_from_scratch"] = compare(
                mx, local, disp._scratch_buf(trans_sid)
            )
            rows[f"{prefix}_trans_from_scratch"]["scratch_id"] = trans_sid
            rows[f"{prefix}_trans_from_scratch"]["ref_scratch_id"] = norm_sid
        except Exception as exc:
            rows[f"{prefix}_trans_from_scratch"] = {"error": str(exc)}

    try:
        raw_c = raw_cache[args.layer]
        fk_c = fk_cache[args.layer]
        raw_k = raw_c.keys[0] if isinstance(raw_c.keys, (tuple, list)) else raw_c.keys
        raw_v = raw_c.values[0] if isinstance(raw_c.values, (tuple, list)) else raw_c.values
        fk_k = fk_c.keys[0] if isinstance(fk_c.keys, (tuple, list)) else fk_c.keys
        fk_v = fk_c.values[0] if isinstance(fk_c.values, (tuple, list)) else fk_c.values
        rows["cache_k"] = compare(mx, raw_k, fk_k)
        rows["cache_v"] = compare(mx, raw_v, fk_v)
        rows["cache_k_current"] = compare(
            mx, cache_row(raw_k, raw_offset), cache_row(fk_k, raw_offset)
        )
        rows["cache_v_current"] = compare(
            mx, cache_row(raw_v, raw_offset), cache_row(fk_v, raw_offset)
        )
    except Exception as exc:
        rows["cache_compare"] = {"error": str(exc)}

    print(
        json.dumps(
            {
                "layer": args.layer,
                "raw_offset": raw_offset,
                "sids": sids,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
