"""Qwen3.5 layer-0 intermediate comparison for fusekit debugging."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def flat1(x):
    return x.reshape(1, -1)


def task_scratch_outputs(task):
    return [b for b in task.get("outputs", []) if b.get("role") == "scratch"]


def scratch_binding_shape_last(binding):
    shape = binding.get("shape") or []
    return shape[-1] if shape else None


def find_layer0_mlp_scratch(disp, hidden_size: int):
    tasks_by_id = {t["id"]: t for t in disp._tg["tasks"]}
    pipeline = [
        tasks_by_id[tid]
        for tid in disp._tg["decode_pipeline"]
        if tasks_by_id.get(tid, {}).get("layer_idx") == 0
    ]
    hidden_linears = []
    for task in pipeline:
        if "fusekit_linear_residual_" not in task.get("kernel", ""):
            continue
        outs = task_scratch_outputs(task)
        if len(outs) == 1 and scratch_binding_shape_last(outs[0]) == hidden_size:
            hidden_linears.append(task)
    if len(hidden_linears) < 2:
        raise RuntimeError("could not locate layer0 post-attention/down-proj residual tasks")

    silu_task = None
    for task in pipeline:
        k = task.get("kernel", "")
        outs = task_scratch_outputs(task)
        if not outs:
            continue
        out_last = scratch_binding_shape_last(outs[0])
        if out_last is None or out_last <= hidden_size:
            continue
        if "fusekit_rmsnorm_gateup_swiglu_" in k:
            silu_task = task
            break
        if "fusekit_swiglu_act_" in k:
            silu_task = task
            break
    if silu_task is None:
        raise RuntimeError("could not locate layer0 swiglu output task")

    return {
        "h": task_scratch_outputs(hidden_linears[0])[0]["scratch_id"],
        "silu_up": task_scratch_outputs(silu_task)[0]["scratch_id"],
        "out": task_scratch_outputs(hidden_linears[1])[0]["scratch_id"],
    }


def raw_child(args: argparse.Namespace) -> int:
    os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_FAST"] = "1"

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
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
    a_dt = a + att.dt_bias
    conv_input = mx.concatenate([cache[0][0], qkv], axis=1)
    conv = att.conv1d(conv_input)
    conv_silu = nn.silu(conv)
    q, k, v = [
        t.reshape(1, 1, h, d)
        for t, h, d in zip(
            mx.split(conv_silu, [att.key_dim, 2 * att.key_dim], -1),
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
    gated_norm = att.norm(gd, z)
    attn_out = att.out_proj(gated_norm.reshape(1, 1, -1))
    h = x + attn_out
    mlp_norm = layer.post_attention_layernorm(h)
    mlp_gate = layer.mlp.gate_proj(mlp_norm)
    mlp_up = layer.mlp.up_proj(mlp_norm)
    silu_up = nn.silu(mlp_gate) * mlp_up
    out = h + layer.mlp.down_proj(silu_up)
    gd_norm = mx.fast.rms_norm(gd, att.norm.weight, att.norm.eps)
    arrays = {
        "x": row1(x).astype(mx.float32),
        "norm_x": row1(norm_x).astype(mx.float32),
        "z": row1(z).astype(mx.float32),
        "z_gate": row1(z).astype(mx.float32),
        "qkv": row1(qkv).astype(mx.float32),
        "b": row1(b).astype(mx.float32),
        "b_sigmoid": row1(mx.sigmoid(b)).astype(mx.float32),
        "a": row1(a).astype(mx.float32),
        "a_dt": row1(a_dt).astype(mx.float32),
        "conv_input4": flat1(conv_input[:, :4, :]).astype(mx.float32),
        "conv_current": row1(conv_input[:, -1:, :]).astype(mx.float32),
        "conv": row1(conv).astype(mx.float32),
        "conv_silu": row1(conv_silu).astype(mx.float32),
        "q": row1(q).astype(mx.float32),
        "k": row1(k).astype(mx.float32),
        "q_scaled": row1(q_scaled).astype(mx.float32),
        "k_scaled": row1(k_scaled).astype(mx.float32),
        "v": row1(v).astype(mx.float32),
        "gd": row1(gd).astype(mx.float32),
        "gd_norm": row1(gd_norm).astype(mx.float32),
        "gd_norm_cast": row1(gd_norm).astype(mx.float32),
        "gated_norm": row1(gated_norm).astype(mx.float32),
        "gated_norm_cast": row1(gated_norm).astype(mx.float32),
        "gated_formula": row1(gated_norm).astype(mx.float32),
        "attn_out": row1(attn_out).astype(mx.float32),
        "h": row1(h).astype(mx.float32),
        "mlp_gate": row1(mlp_gate).astype(mx.float32),
        "mlp_up": row1(mlp_up).astype(mx.float32),
        "silu_up": row1(silu_up).astype(mx.float32),
        "out": row1(out).astype(mx.float32),
    }
    if args.norm_debug_from_z:
        arrays["norm_kernel"] = row1(norm_x).astype(mx.float32)
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(json.dumps({"mode": "raw"}))
    return 0


def patched_child(args: argparse.Namespace) -> int:
    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.activations import swiglu

    model, tok = load(str(model_path(args.model)))
    if args.fusekit_dir:
        import fusekit_runtime as fr

        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
    text = text_model(model)
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
    s = lambda sid: row1(disp._scratch_buf(sid)).astype(mx.float32)
    # Current Qwen3.5 oQ4 task graph layout is trace-local for GDN:
    # norm -> z/qkv/a/b -> conv -> gated_delta -> rmsnorm(gd)
    # -> silu(z) * normed -> out_proj residual -> MLP.
    norm_s = disp._scratch_buf(0)
    z_s = disp._scratch_buf(1)
    z_gate_s = disp._scratch_buf(3)
    qkv_s = disp._scratch_buf(7)
    conv_input_s = disp._scratch_buf(8)
    conv_s = disp._scratch_buf(9)
    conv_silu_s = disp._scratch_buf(11)
    q_s = disp._scratch_buf(12)
    k_s = disp._scratch_buf(13)
    v_s = disp._scratch_buf(14)
    q_scaled_s = disp._scratch_buf(18)
    k_scaled_s = disp._scratch_buf(22)
    a_dt_s = disp._scratch_buf(28)
    b_s = disp._scratch_buf(35)
    b_sigmoid_s = disp._scratch_buf(36)
    gd = disp._scratch_buf(39)
    gd_norm_s = disp._scratch_buf(41)
    gd_norm_cast_s = disp._scratch_buf(42)
    gated_s = disp._scratch_buf(4)
    gated_cast_s = disp._scratch_buf(43)
    mlp_sids = find_layer0_mlp_scratch(disp, disp._hidden)
    h_s = disp._scratch_buf(mlp_sids["h"])
    silu_up_s = disp._scratch_buf(mlp_sids["silu_up"])
    out_s = disp._scratch_buf(mlp_sids["out"])
    core = text.model
    layer = core.layers[0]
    att = layer.linear_attn
    norm_x_py = layer.input_layernorm(disp._x_in)
    z_py = att.in_proj_z(norm_x_py).reshape(1, 1, att.num_v_heads, att.head_v_dim)
    qkv_py = att.in_proj_qkv(norm_x_py)
    a_py = att.in_proj_a(norm_x_py)
    a_dt_py = a_py + att.dt_bias
    b_py = att.in_proj_b(norm_x_py)
    gate1 = z_gate_s[:, :1, ...].reshape(1, 1, att.num_v_heads, att.head_v_dim)
    gd4 = gd[:, :1, ...]
    normed = mx.fast.rms_norm(gd4, att.norm.weight, att.norm.eps)
    formula = (
        nn.silu(gate1.astype(mx.float32)) * normed.astype(mx.float32)
    ).astype(gd.dtype)
    kernel_delta = mx.abs(row1(gated_s).astype(mx.float32) - row1(formula).astype(mx.float32))
    norm_delta = mx.abs(row1(norm_s).astype(mx.float32) - row1(norm_x_py).astype(mx.float32))
    z2_delta = mx.abs(row1(z_s).astype(mx.float32) - row1(z_py).astype(mx.float32))
    z_delta = mx.abs(row1(z_gate_s).astype(mx.float32) - row1(z_py).astype(mx.float32))
    qkv_delta = mx.abs(row1(qkv_s).astype(mx.float32) - row1(qkv_py).astype(mx.float32))
    h_for_mlp = h_s[:, :1, :] if h_s.ndim >= 3 else h_s.reshape(1, 1, -1)
    mlp_norm_py = layer.post_attention_layernorm(h_for_mlp)
    mlp_gate_py = layer.mlp.gate_proj(mlp_norm_py)
    mlp_up_py = layer.mlp.up_proj(mlp_norm_py)
    silu_up_py = swiglu(mlp_gate_py, mlp_up_py)
    out_py = h_for_mlp + layer.mlp.down_proj(silu_up_py)
    silu_up_delta = mx.abs(
        row1(silu_up_s).astype(mx.float32) - row1(silu_up_py).astype(mx.float32)
    )
    out_delta = mx.abs(row1(out_s).astype(mx.float32) - row1(out_py).astype(mx.float32))
    silu_up_vs_gate_delta = mx.abs(
        row1(silu_up_s).astype(mx.float32) - row1(mlp_gate_py).astype(mx.float32)
    )
    silu_up_vs_up_delta = mx.abs(
        row1(silu_up_s).astype(mx.float32) - row1(mlp_up_py).astype(mx.float32)
    )
    qkv_kernel_norm_delta = None
    if args.norm_debug_from_z:
        norm_kernel_full = row1(norm_s).reshape(norm_x_py.shape)
        qkv_py_kernel_norm = att.in_proj_qkv(norm_kernel_full)
        qkv_kernel_norm_delta = mx.abs(
            row1(qkv_s).astype(mx.float32)
            - row1(qkv_py_kernel_norm).astype(mx.float32)
        )
    a_scratch = a_dt_s - att.dt_bias
    a_delta = mx.abs(row1(a_scratch).astype(mx.float32) - row1(a_py).astype(mx.float32))
    a_dt_delta = mx.abs(row1(a_dt_s).astype(mx.float32) - row1(a_dt_py).astype(mx.float32))
    b_delta = mx.abs(row1(b_s).astype(mx.float32) - row1(b_py).astype(mx.float32))
    mx.eval(
        norm_delta,
        kernel_delta,
        z2_delta,
        z_delta,
        qkv_delta,
        silu_up_delta,
        silu_up_vs_gate_delta,
        silu_up_vs_up_delta,
        out_delta,
        a_delta,
        a_dt_delta,
        b_delta,
    )
    if qkv_kernel_norm_delta is not None:
        mx.eval(qkv_kernel_norm_delta)
    arrays = {
        "x": row1(disp._x_in).astype(mx.float32),
        "norm_x": row1(norm_s).astype(mx.float32),
        "z": row1(z_s).astype(mx.float32),
        "z_gate": row1(z_gate_s).astype(mx.float32),
        "qkv": row1(qkv_s).astype(mx.float32),
        "a": row1(a_scratch).astype(mx.float32),
        "a_dt": row1(a_dt_s).astype(mx.float32),
        "b": row1(b_s).astype(mx.float32),
        "b_sigmoid": row1(b_sigmoid_s).astype(mx.float32),
        "conv_input4": flat1(conv_input_s[:, :4, :]).astype(mx.float32),
        "conv_current": row1(conv_input_s[:, 3:4, :]).astype(mx.float32),
        "conv": row1(conv_s).astype(mx.float32),
        "conv_silu": row1(conv_silu_s).astype(mx.float32),
        "q": row1(q_s).astype(mx.float32),
        "k": row1(k_s).astype(mx.float32),
        "q_scaled": row1(q_scaled_s).astype(mx.float32),
        "k_scaled": row1(k_scaled_s).astype(mx.float32),
        "v": row1(v_s).astype(mx.float32),
        "gd": row1(gd).astype(mx.float32),
        "gd_norm": row1(gd_norm_s).astype(mx.float32),
        "gd_norm_cast": row1(gd_norm_cast_s).astype(mx.float32),
        "gated_norm": row1(gated_s).astype(mx.float32),
        "gated_norm_cast": row1(gated_cast_s).astype(mx.float32),
        "gated_formula": row1(formula).astype(mx.float32),
        "h": row1(h_s).astype(mx.float32),
        "mlp_gate": row1(mlp_gate_py).astype(mx.float32),
        "mlp_up": row1(mlp_up_py).astype(mx.float32),
        "silu_up": row1(silu_up_s).astype(mx.float32),
        "silu_up_formula": row1(silu_up_py).astype(mx.float32),
        "out": row1(out_s).astype(mx.float32),
        "out_formula": row1(out_py).astype(mx.float32),
    }
    if args.norm_debug_from_z:
        arrays["norm_kernel"] = row1(norm_s).astype(mx.float32)
    mx.save_safetensors(args.out, arrays)
    meta = {
        "mode": "patched",
        "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
        "fusekit_dir": str(getattr(model, "_fk_dir", "")),
        "norm_scratch_python_max_abs": float(mx.max(norm_delta).item()),
        "norm_scratch_python_mean_abs": float(mx.mean(norm_delta).item()),
        "gated_kernel_formula_max_abs": float(mx.max(kernel_delta).item()),
        "gated_kernel_formula_mean_abs": float(mx.mean(kernel_delta).item()),
        "gated_kernel_formula_gt_0_01": int(mx.sum(kernel_delta > 0.01).item()),
        "gated_kernel_formula_gt_0_1": int(mx.sum(kernel_delta > 0.1).item()),
        "gated_kernel_formula_gt_1": int(mx.sum(kernel_delta > 1.0).item()),
        "z_scratch_python_max_abs": float(mx.max(z2_delta).item()),
        "z_scratch_python_mean_abs": float(mx.mean(z2_delta).item()),
        "z_gate_python_max_abs": float(mx.max(z_delta).item()),
        "z_gate_python_mean_abs": float(mx.mean(z_delta).item()),
        "qkv_scratch_python_max_abs": float(mx.max(qkv_delta).item()),
        "qkv_scratch_python_mean_abs": float(mx.mean(qkv_delta).item()),
        "silu_up_scratch_python_max_abs": float(mx.max(silu_up_delta).item()),
        "silu_up_scratch_python_mean_abs": float(mx.mean(silu_up_delta).item()),
        "silu_up_vs_gate_python_max_abs": float(mx.max(silu_up_vs_gate_delta).item()),
        "silu_up_vs_gate_python_mean_abs": float(
            mx.mean(silu_up_vs_gate_delta).item()
        ),
        "silu_up_vs_up_python_max_abs": float(mx.max(silu_up_vs_up_delta).item()),
        "silu_up_vs_up_python_mean_abs": float(mx.mean(silu_up_vs_up_delta).item()),
        "out_scratch_python_max_abs": float(mx.max(out_delta).item()),
        "out_scratch_python_mean_abs": float(mx.mean(out_delta).item()),
        "a_scratch_python_max_abs": float(mx.max(a_delta).item()),
        "a_scratch_python_mean_abs": float(mx.mean(a_delta).item()),
        "a_dt_scratch_python_max_abs": float(mx.max(a_dt_delta).item()),
        "a_dt_scratch_python_mean_abs": float(mx.mean(a_dt_delta).item()),
        "b_scratch_python_max_abs": float(mx.max(b_delta).item()),
        "b_scratch_python_mean_abs": float(mx.mean(b_delta).item()),
    }
    if qkv_kernel_norm_delta is not None:
        meta["qkv_scratch_python_kernel_norm_max_abs"] = float(
            mx.max(qkv_kernel_norm_delta).item()
        )
        meta["qkv_scratch_python_kernel_norm_mean_abs"] = float(
            mx.mean(qkv_kernel_norm_delta).item()
        )
    Path(args.meta).write_text(json.dumps(meta))
    return 0


def child_main(args: argparse.Namespace) -> int:
    if args.mode == "raw":
        return raw_child(args)
    return patched_child(args)


def parent_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-qwen35-l0-") as td:
        tmp = Path(td)
        raw_out = tmp / "raw.safetensors"
        patched_out = tmp / "patched.safetensors"
        raw_meta = tmp / "raw.json"
        patched_meta = tmp / "patched.json"
        base = [
            sys.executable,
            __file__,
            "--child",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--decode-token",
            str(args.decode_token),
        ]
        if args.norm_debug_from_z:
            base += ["--norm-debug-from-z"]
        subprocess.run(base + ["--mode", "raw", "--out", str(raw_out), "--meta", str(raw_meta)], check=True)
        patched_cmd = base + ["--mode", "patched", "--out", str(patched_out), "--meta", str(patched_meta)]
        if args.fusekit_dir:
            patched_cmd += ["--fusekit-dir", args.fusekit_dir]
        subprocess.run(patched_cmd, check=True)
        raw = mx.load(str(raw_out))
        patched = mx.load(str(patched_out))
        rows = []
        for key in sorted(raw):
            if key not in patched:
                rows.append({"name": key, "missing": "patched"})
                continue
            if tuple(raw[key].shape) != tuple(patched[key].shape):
                rows.append({
                    "name": key,
                    "raw_shape": list(raw[key].shape),
                    "patched_shape": list(patched[key].shape),
                    "shape_mismatch": True,
                })
                continue
            diff = mx.abs(raw[key] - patched[key])
            mx.eval(diff)
            rows.append({
                "name": key,
                "shape": list(raw[key].shape),
                "max_abs_diff": float(mx.max(diff).item()),
                "mean_abs_diff": float(mx.mean(diff).item()),
            })
        print(json.dumps({
            "raw_meta": json.loads(raw_meta.read_text()),
            "patched_meta": json.loads(patched_meta.read_text()),
            "rows": rows,
        }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", choices=["raw", "patched"])
    ap.add_argument("--out")
    ap.add_argument("--meta")
    ap.add_argument("--fusekit-dir")
    ap.add_argument("--norm-debug-from-z", action="store_true")
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
