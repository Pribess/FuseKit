"""Compare one Qwen3.5 full-attention layer against MLX on the same input."""

from __future__ import annotations

import argparse
import json
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


def row1(x):
    if x.ndim >= 3:
        return x[:, :1, ...].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def compare(mx, a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return {
        "max_abs_diff": float(mx.max(d).item()),
        "mean_abs_diff": float(mx.mean(d).item()),
    }


def scalar_int(v) -> int:
    if hasattr(v, "item"):
        return int(v.item())
    return int(v)


def manual_rope(mx, x, *, dims: int, base: float, offset: int):
    head_dim = x.shape[-1]
    half = dims // 2
    x1 = x[..., :half].astype(mx.float32)
    x2 = x[..., half:dims].astype(mx.float32)
    idx = mx.arange(half, dtype=mx.float32)
    inv_freq = mx.exp(-idx * mx.array(float(__import__("math").log(base)) / float(half), dtype=mx.float32))
    theta = mx.array(float(offset), dtype=mx.float32) * inv_freq
    c = mx.cos(theta)
    s = mx.sin(theta)
    first = x1 * c - x2 * s
    second = x1 * s + x2 * c
    rest = x[..., dims:] if dims < head_dim else None
    parts = [first.astype(x.dtype), second.astype(x.dtype)]
    if rest is not None:
        parts.append(rest)
    return mx.concatenate(parts, axis=-1)


def find_layer_sids(tg: dict, layer: int) -> dict:
    tasks = {t["id"]: t for t in tg["tasks"]}
    ids = [i for i in tg["decode_pipeline"] if tasks[i].get("layer_idx") == layer]
    sids: dict[str, int] = {}
    hidden_one: list[int] = []
    linres_hidden: list[int] = []
    for i in ids:
        t = tasks[i]
        k = t.get("kernel", "")
        outs = [b for b in t.get("outputs", []) if b.get("role") == "scratch"]
        ins = [b for b in t.get("inputs", []) if b.get("role") == "scratch"]
        paths = [b.get("source_path", "") for b in t.get("inputs", [])]
        if k.startswith("fusekit_rmsnorm_qkv_") and len(outs) >= 3:
            if ins:
                sids["layer_in"] = ins[0]["scratch_id"]
            for out in outs:
                role = out.get("role_name")
                if role == "q":
                    sids["q_proj"] = out["scratch_id"]
                elif role == "k":
                    sids["k_proj"] = out["scratch_id"]
                elif role == "v":
                    sids["v_proj"] = out["scratch_id"]
            continue
        if f"layers.{layer}.input_layernorm.weight" in " ".join(paths) and ins:
            sids["layer_in"] = ins[0]["scratch_id"]
        if any(f"layers.{layer}.self_attn.q_proj.weight" in p for p in paths) and outs:
            sids["q_proj"] = outs[0]["scratch_id"]
        if any(f"layers.{layer}.self_attn.k_proj.weight" in p for p in paths) and outs:
            sids["k_proj"] = outs[0]["scratch_id"]
        if any(f"layers.{layer}.self_attn.v_proj.weight" in p for p in paths) and outs:
            sids["v_proj"] = outs[0]["scratch_id"]
        if any(f"layers.{layer}.self_attn.o_proj.weight" in p for p in paths) and outs:
            sids["attn_proj"] = outs[0]["scratch_id"]
        if any(f"layers.{layer}.mlp.down_proj.weight" in p for p in paths) and outs:
            sids["mlp_down"] = outs[0]["scratch_id"]
        if "fusekit_chain_" in k and len(outs) == 3:
            if ins:
                sids["layer_in"] = ins[0]["scratch_id"]
            sids["q_proj"] = outs[0]["scratch_id"]
            sids["k_proj"] = outs[1]["scratch_id"]
            sids["v_proj"] = outs[2]["scratch_id"]
        elif "fusekit_rmsnorm_rope_qgate_decode_" in k and len(outs) == 2:
            sids["q_rope"] = outs[0]["scratch_id"]
            sids["gate"] = outs[1]["scratch_id"]
        elif "fusekit_sdpa_decode_" in k and len(outs) == 1:
            sids["sdpa"] = outs[0]["scratch_id"]
        elif k.startswith("fusekit_op_multiply_") and len(outs) == 1:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 4096:
                sids["gated_sdpa"] = outs[0]["scratch_id"]
        elif k.startswith("fusekit_op_add_") and len(outs) == 1:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 4096:
                in_sids = {b["scratch_id"] for b in ins if "scratch_id" in b}
                if "attn_proj" in sids and sids["attn_proj"] in in_sids:
                    sids["h_attn"] = outs[0]["scratch_id"]
                elif "mlp_down" in sids and sids["mlp_down"] in in_sids:
                    sids["layer_out"] = outs[0]["scratch_id"]
        elif "fusekit_linear_residual_" in k and len(outs) == 1:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 4096:
                linres_hidden.append(outs[0]["scratch_id"])
        elif "fusekit_chain_" in k and len(outs) == 1:
            shape = outs[0].get("shape") or []
            if shape and shape[-1] == 4096:
                if len(ins) >= 2 and "sdpa" in sids and "gate" in sids:
                    in_sids = {b["scratch_id"] for b in ins if "scratch_id" in b}
                    if sids["sdpa"] in in_sids and sids["gate"] in in_sids:
                        sids["gated_sdpa"] = outs[0]["scratch_id"]
                        continue
                hidden_one.append(outs[0]["scratch_id"])
    if len(linres_hidden) >= 2:
        sids["h_attn"] = linres_hidden[0]
        sids["layer_out"] = linres_hidden[-1]
    if len(hidden_one) >= 3:
        sids["gated_sdpa"] = hidden_one[0]
        sids["h_attn"] = hidden_one[1]
        sids["layer_out"] = hidden_one[-1]
    return sids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--fusekit-dir")
    args = ap.parse_args()

    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import scaled_dot_product_attention

    model, tok = load(str(model_path(args.model)))
    if args.fusekit_dir:
        import fusekit_runtime as fr

        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
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
    sids = find_layer_sids(disp._tg, args.layer)
    missing = [k for k in ("layer_in", "q_proj", "k_proj", "v_proj", "q_rope", "gate", "sdpa") if k not in sids]
    if missing:
        raise RuntimeError(f"missing scratch ids for layer {args.layer}: {missing}; have={sids}")

    layer = core.layers[args.layer]
    att = layer.self_attn
    c = cache[args.layer]
    offset_after = scalar_int(getattr(c, "offset", getattr(c, "_idx", 0)))
    offset = max(offset_after - 1, 0)

    x = row1(disp._scratch_buf(sids["layer_in"])).reshape(1, 1, -1).astype(disp._act_dt)
    norm_x = layer.input_layernorm(x)
    q_proj = att.q_proj(norm_x)
    q, gate = mx.split(q_proj.reshape(1, 1, att.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(1, 1, -1)
    k_proj = att.k_proj(norm_x)
    v_proj = att.v_proj(norm_x)
    q_rope = att.rope(att.q_norm(q).transpose(0, 2, 1, 3), offset=offset)
    q_norm_t = att.q_norm(q).transpose(0, 2, 1, 3)
    k_norm_t = att.k_norm(k_proj.reshape(1, 1, att.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    k_rope = att.rope(k_norm_t, offset=offset)
    values = v_proj.reshape(1, 1, att.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    keys_live = c.keys[:, :, : offset + 1, :]
    vals_live = c.values[:, :, : offset + 1, :]
    sdpa = scaled_dot_product_attention(q_rope, keys_live, vals_live, cache=c, scale=att.scale, mask=None)
    sdpa_flat = sdpa.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    sig_lazy = mx.sigmoid(gate)
    gated = sdpa_flat * sig_lazy
    sig_materialized = mx.sigmoid(gate)
    mx.eval(sig_materialized)
    gated_materialized_sig = sdpa_flat * sig_materialized
    gated_sig_bf16 = sdpa_flat * sig_materialized.astype(mx.bfloat16)
    attn_proj = att.o_proj(gated)
    h_attn = x + attn_proj
    post_norm = layer.post_attention_layernorm(h_attn)
    mlp_down = layer.mlp(post_norm)
    layer_out = h_attn + mlp_down
    mx.eval(
        q_proj,
        k_proj,
        v_proj,
        q_rope,
        k_rope,
        values,
        sdpa_flat,
        gated,
        gated_materialized_sig,
        gated_sig_bf16,
        attn_proj,
        h_attn,
        post_norm,
        mlp_down,
        layer_out,
    )

    q_rope_s = disp._scratch_buf(sids["q_rope"]).reshape(-1)[: att.num_attention_heads * att.head_dim]
    q_rope_s = q_rope_s.reshape(1, att.num_attention_heads, 1, att.head_dim)
    k_live = c.keys[:, :, offset : offset + 1, :]
    v_live = c.values[:, :, offset : offset + 1, :]

    rows = {
        "q_proj": compare(mx, row1(q_proj), row1(disp._scratch_buf(sids["q_proj"]))),
        "k_proj": compare(mx, row1(k_proj), row1(disp._scratch_buf(sids["k_proj"]))),
        "v_proj": compare(mx, row1(v_proj), row1(disp._scratch_buf(sids["v_proj"]))),
        "q_rope": compare(mx, q_rope, q_rope_s),
        "q_rope_manual_base_1e5": compare(
            mx,
            manual_rope(mx, q_norm_t, dims=att.rope.dims, base=100000.0, offset=offset),
            q_rope_s,
        ),
        "q_rope_manual_base_1e7": compare(
            mx,
            manual_rope(mx, q_norm_t, dims=att.rope.dims, base=10000000.0, offset=offset),
            q_rope_s,
        ),
        "q_rope_manual_base_1e7_cast_norm": compare(
            mx,
            manual_rope(mx, q_norm_t.astype(q_proj.dtype), dims=att.rope.dims, base=10000000.0, offset=offset),
            q_rope_s,
        ),
        "gate": compare(mx, row1(gate), row1(disp._scratch_buf(sids["gate"]))),
        "k_store": compare(mx, k_rope, k_live),
        "v_store": compare(mx, values, v_live),
        "sdpa": compare(mx, row1(sdpa_flat), row1(disp._scratch_buf(sids["sdpa"]))),
    }
    if "gated_sdpa" in sids:
        rows["gated_sdpa"] = compare(mx, row1(gated), row1(disp._scratch_buf(sids["gated_sdpa"])))
        rows["gated_sdpa_materialized_sig"] = compare(
            mx,
            row1(gated_materialized_sig),
            row1(disp._scratch_buf(sids["gated_sdpa"])),
        )
        rows["gated_sdpa_sig_bf16"] = compare(
            mx,
            row1(gated_sig_bf16),
            row1(disp._scratch_buf(sids["gated_sdpa"])),
        )
    if "attn_proj" in sids:
        rows["attn_proj"] = compare(mx, row1(attn_proj), row1(disp._scratch_buf(sids["attn_proj"])))
    if "h_attn" in sids:
        rows["h_attn"] = compare(mx, row1(h_attn), row1(disp._scratch_buf(sids["h_attn"])))
    if "mlp_down" in sids:
        rows["mlp_down"] = compare(mx, row1(mlp_down), row1(disp._scratch_buf(sids["mlp_down"])))
    if "layer_out" in sids:
        rows["layer_out"] = compare(mx, row1(layer_out), row1(disp._scratch_buf(sids["layer_out"])))

    print(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "offset": offset,
                "offset_after": offset_after,
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
