"""Compare one Qwen3.5 linear-attention layer against MLX on same inputs."""

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


def one_token_view(x, *, dtype=None):
    y = x[:, :1, ...] if x.ndim >= 3 else x.reshape(1, 1, -1)
    if dtype is not None:
        y = y.astype(dtype)
    return y


def compare(mx, a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return {
        "max_abs_diff": float(mx.max(d).item()),
        "mean_abs_diff": float(mx.mean(d).item()),
        "gt_0_01": int(mx.sum(d > 0.01).item()),
        "gt_0_1": int(mx.sum(d > 0.1).item()),
    }


def scratch_inputs(task):
    return [b for b in task.get("inputs", []) if b.get("role") == "scratch"]


def scratch_outputs(task):
    return [b for b in task.get("outputs", []) if b.get("role") == "scratch"]


def output_last(task):
    outs = scratch_outputs(task)
    if not outs:
        return None
    shape = outs[0].get("shape") or []
    return shape[-1] if shape else None


def has_path(task, needle: str) -> bool:
    return any(
        needle in (b.get("source_path") or "")
        for b in task.get("inputs", [])
        if b.get("role") == "external"
    )


def first_out(task):
    outs = scratch_outputs(task)
    return outs[0]["scratch_id"] if outs else None


def find_linear_layer_sids(tg: dict, layer: int, hidden: int) -> dict[str, int]:
    tasks = {t["id"]: t for t in tg["tasks"]}
    ids = [i for i in tg["decode_pipeline"] if tasks[i].get("layer_idx") == layer]
    pipeline = [tasks[i] for i in ids]
    sids: dict[str, int] = {}

    prefix = f"layers.{layer}."
    for task in pipeline:
        ins = scratch_inputs(task)
        outs = scratch_outputs(task)
        if (
            has_path(task, prefix + "linear_attn.in_proj_z.weight")
            and has_path(task, prefix + "linear_attn.in_proj_qkv.weight")
            and len(outs) >= 4
        ):
            sids["z"] = outs[0]["scratch_id"]
            sids["qkv"] = outs[1]["scratch_id"]
            sids["a_dt"] = outs[2]["scratch_id"]
            sids["b"] = outs[3]["scratch_id"]
        elif has_path(task, prefix + "input_layernorm.weight") and outs:
            if ins:
                sids["layer_in"] = ins[0]["scratch_id"]
            sids["norm_x"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.in_proj_z.weight") and outs:
            sids["z"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.in_proj_qkv.weight") and outs:
            sids["qkv"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.in_proj_b.weight") and outs:
            sids["b"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.in_proj_a.weight") and outs:
            sids["a_dt"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.conv1d.weight") and ins and outs:
            sids["conv_input"] = ins[0]["scratch_id"]
            if "convolutionsilu" in task.get("kernel", ""):
                sids["conv_silu"] = outs[0]["scratch_id"]
            else:
                sids["conv"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.norm.weight") and outs:
            out_shape = outs[0].get("shape") or []
            if out_shape and out_shape[-1] == hidden:
                # Qwen3.5 fused chain stores the z-gated flat attention
                # input directly, not the intermediate gd_norm.
                sids["gated_flat"] = outs[0]["scratch_id"]
            else:
                sids["gd_norm"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "linear_attn.out_proj.weight") and outs:
            sids["gated_flat"] = ins[0]["scratch_id"] if ins else outs[0]["scratch_id"]
            sids["attn_proj"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "post_attention_layernorm.weight") and outs:
            if output_last(task) == hidden:
                sids["post_norm"] = outs[0]["scratch_id"]
            else:
                sids["mlp_silu"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "mlp.gate_proj.weight") and outs:
            sids["mlp_gate"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "mlp.up_proj.weight") and outs:
            sids["mlp_up"] = outs[0]["scratch_id"]
        elif has_path(task, prefix + "mlp.down_proj.weight") and outs:
            sids["mlp_silu"] = ins[0]["scratch_id"] if ins else outs[0]["scratch_id"]
            sids["mlp_down"] = outs[0]["scratch_id"]

    for task in pipeline:
        k = task.get("kernel", "")
        ins = scratch_inputs(task)
        outs = scratch_outputs(task)
        if not outs:
            continue
        out_sid = outs[0]["scratch_id"]
        if k.startswith("fusekit_op_multiply_") and output_last(task) == 8192:
            sids.setdefault("conv_silu", out_sid)
        elif k.startswith("fusekit_op_split_") and len(outs) == 3:
            if (outs[0].get("shape") or [])[-1] == 2048 and (outs[2].get("shape") or [])[-1] == hidden:
                sids["q_raw"] = outs[0]["scratch_id"]
                sids["k_raw"] = outs[1]["scratch_id"]
                sids["v"] = outs[2]["scratch_id"]
        elif k.startswith("fusekit_op_sigmoid_") and ins:
            if ins[0]["scratch_id"] == sids.get("b"):
                sids["b_sigmoid"] = out_sid
        elif k.startswith("custom_kernel_gated_delta_step_") and len(ins) >= 5 and len(outs) >= 2:
            sids["q_scaled"] = ins[0]["scratch_id"]
            sids["k_scaled"] = ins[1]["scratch_id"]
            sids["v_gd"] = ins[2]["scratch_id"]
            sids["g"] = ins[3]["scratch_id"]
            sids["b_gd"] = ins[4]["scratch_id"]
            sids["gd"] = outs[0]["scratch_id"]
            sids["state_out"] = outs[1]["scratch_id"]
        elif "fusekit_swiglu_act_" in k and len(ins) >= 2:
            last = output_last(task)
            if last == 128:
                sids["z_gate"] = ins[0]["scratch_id"]
                sids["gd_norm_float"] = ins[1]["scratch_id"]
                sids["gated_float"] = out_sid
            elif last is not None and last > hidden:
                sids["mlp_gate_in"] = ins[0]["scratch_id"]
                sids["mlp_up_in"] = ins[1]["scratch_id"]
                sids["mlp_silu"] = out_sid
        elif k.startswith("fusekit_op_add_") and ins and outs:
            in_sids = {b["scratch_id"] for b in ins if "scratch_id" in b}
            if "attn_proj" in sids and sids["attn_proj"] in in_sids:
                sids["h_attn"] = out_sid
            elif "mlp_down" in sids and sids["mlp_down"] in in_sids:
                sids["layer_out"] = out_sid

    return sids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--fusekit-dir")
    ap.add_argument(
        "--mlp-debug-out",
        choices=("swiglu", "gate", "up"),
        default="swiglu",
        help="Expected value stored in fused MLP scratch when compiling with FUSEKIT_DEBUG_RMSGU_SWIGLU_OUT.",
    )
    args = ap.parse_args()

    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.activations import swiglu

    model, tok = load(str(model_path(args.model)))
    if args.fusekit_dir:
        import fusekit_runtime as fr

        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)

    text = text_model(model)
    core = text.model
    layer = core.layers[args.layer]
    if not layer.is_linear:
        raise RuntimeError(f"layer {args.layer} is not a linear-attention layer")

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
    sids = find_linear_layer_sids(disp._tg, args.layer, disp._hidden)
    required = [
        "norm_x",
        "z",
        "qkv",
        "conv_input",
        "q_scaled",
        "k_scaled",
        "v",
        "a_dt",
        "b",
        "b_sigmoid",
        "g",
        "gd",
        "gated_flat",
        "attn_proj",
        "h_attn",
        "mlp_down",
        "layer_out",
    ]
    missing = [k for k in required if k not in sids]
    if "v" not in sids and "v_gd" in sids:
        sids["v"] = sids["v_gd"]
        missing = [k for k in required if k not in sids]
    optional_missing = {"h_attn", "layer_out"}
    hard_missing = [k for k in missing if k not in optional_missing]
    if hard_missing:
        raise RuntimeError(f"missing scratch ids for layer {args.layer}: {hard_missing}; have={sids}")

    att = layer.linear_attn
    if "layer_in" in sids:
        x = one_token_view(disp._scratch_buf(sids["layer_in"]), dtype=disp._act_dt)
    elif args.layer == 0:
        x = one_token_view(disp._x_in, dtype=disp._act_dt)
    else:
        prev_sid = disp._layer_scratch_id.get(("out_buf", args.layer - 1))
        if prev_sid is None:
            raise RuntimeError(f"missing layer input for layer {args.layer}")
        x = one_token_view(disp._scratch_buf(prev_sid), dtype=disp._act_dt)
    norm_x = layer.input_layernorm(x)
    qkv = att.in_proj_qkv(norm_x)
    z = att.in_proj_z(norm_x).reshape(1, 1, att.num_v_heads, att.head_v_dim)
    b = att.in_proj_b(norm_x)
    a = att.in_proj_a(norm_x)
    a_dt = a + att.dt_bias
    conv_input = disp._scratch_buf(sids["conv_input"]).astype(disp._act_dt)
    conv = att.conv1d(conv_input)
    conv_silu = nn.silu(conv)
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
    gd_s = one_token_view(disp._scratch_buf(sids["gd"]))
    gd_norm = mx.fast.rms_norm(gd_s, att.norm.weight, att.norm.eps)
    gated = (nn.silu(z.astype(mx.float32)) * gd_norm.astype(mx.float32)).astype(gd_s.dtype)
    gated_flat = row1(disp._scratch_buf(sids["gated_flat"])).reshape(1, 1, -1).astype(disp._act_dt)
    attn_proj = att.out_proj(gated_flat)
    h_attn = x + attn_proj
    post_norm = layer.post_attention_layernorm(h_attn)
    mlp_gate = layer.mlp.gate_proj(post_norm)
    mlp_up = layer.mlp.up_proj(post_norm)
    mlp_silu = swiglu(mlp_gate, mlp_up)
    mlp_down = layer.mlp.down_proj(mlp_silu)
    layer_out_formula = h_attn + mlp_down
    mlp_silu_down = mlp_silu
    if args.mlp_debug_out == "swiglu" and "mlp_silu" in sids:
        mlp_silu_down = row1(disp._scratch_buf(sids["mlp_silu"])).reshape(1, 1, -1).astype(disp._act_dt)
    mlp_down_from_scratch = layer.mlp.down_proj(mlp_silu_down)
    layer_out_from_scratch_mlp = h_attn + mlp_down_from_scratch
    mx.eval(
        norm_x,
        qkv,
        z,
        b,
        a_dt,
        conv,
        conv_silu,
        q_scaled,
        k_scaled,
        g,
        b_sigmoid,
        gd_norm,
        gated,
        attn_proj,
        h_attn,
        post_norm,
        mlp_gate,
        mlp_up,
        mlp_silu,
        mlp_down,
        mlp_down_from_scratch,
        layer_out_formula,
        layer_out_from_scratch_mlp,
    )

    rows = {
        "norm_x": compare(mx, row1(norm_x), row1(disp._scratch_buf(sids["norm_x"]))),
        "z": compare(mx, row1(z), row1(disp._scratch_buf(sids["z"]))),
        "qkv": compare(mx, row1(qkv), row1(disp._scratch_buf(sids["qkv"]))),
        "q_scaled": compare(mx, row1(q_scaled), row1(disp._scratch_buf(sids["q_scaled"]))),
        "k_scaled": compare(mx, row1(k_scaled), row1(disp._scratch_buf(sids["k_scaled"]))),
        "v": compare(mx, row1(v), row1(disp._scratch_buf(sids["v"]))),
        "a_dt": compare(mx, row1(a_dt), row1(disp._scratch_buf(sids["a_dt"]))),
        "b": compare(mx, row1(b), row1(disp._scratch_buf(sids["b"]))),
        "b_sigmoid": compare(mx, row1(b_sigmoid), row1(disp._scratch_buf(sids["b_sigmoid"]))),
        "g": compare(mx, row1(g), row1(disp._scratch_buf(sids["g"]))),
        "gated_formula": compare(mx, row1(gated), row1(disp._scratch_buf(sids["gated_flat"]))),
        "attn_proj": compare(mx, row1(attn_proj), row1(disp._scratch_buf(sids["attn_proj"]))),
        "mlp_down": compare(mx, row1(mlp_down), row1(disp._scratch_buf(sids["mlp_down"]))),
    }
    if "h_attn" in sids:
        rows["h_attn"] = compare(mx, row1(h_attn), row1(disp._scratch_buf(sids["h_attn"])))
    if "gd_norm" in sids:
        rows["gd_norm"] = compare(mx, row1(gd_norm), row1(disp._scratch_buf(sids["gd_norm"])))
    if args.mlp_debug_out == "swiglu" and "layer_out" in sids:
        rows["layer_out_formula"] = compare(
            mx, row1(layer_out_formula), row1(disp._scratch_buf(sids["layer_out"]))
        )
        rows["layer_out_from_scratch_mlp"] = compare(
            mx, row1(layer_out_from_scratch_mlp), row1(disp._scratch_buf(sids["layer_out"]))
        )
    if "conv" in sids:
        rows["conv"] = compare(mx, row1(conv), row1(disp._scratch_buf(sids["conv"])))
    if "conv_silu" in sids:
        rows["conv_silu"] = compare(mx, row1(conv_silu), row1(disp._scratch_buf(sids["conv_silu"])))
    if "post_norm" in sids:
        rows["post_norm"] = compare(mx, row1(post_norm), row1(disp._scratch_buf(sids["post_norm"])))
    if "mlp_gate" in sids:
        rows["mlp_gate"] = compare(mx, row1(mlp_gate), row1(disp._scratch_buf(sids["mlp_gate"])))
    if "mlp_up" in sids:
        rows["mlp_up"] = compare(mx, row1(mlp_up), row1(disp._scratch_buf(sids["mlp_up"])))
    if "mlp_silu" in sids and sids["mlp_silu"] != sids.get("post_norm"):
        mlp_expected = {
            "swiglu": mlp_silu,
            "gate": mlp_gate,
            "up": mlp_up,
        }[args.mlp_debug_out]
        rows[f"mlp_{args.mlp_debug_out}"] = compare(
            mx, row1(mlp_expected), row1(disp._scratch_buf(sids["mlp_silu"]))
        )

    print(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
                "fusekit_dir": str(getattr(model, "_fk_dir", "")),
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
