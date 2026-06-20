"""Compare one FuseKit prefill M chunk against raw oMLX.

The script isolates the prefill variant from the surrounding generation loop:
it fills both caches with the same raw prefix, then runs exactly one chunk
through raw oMLX and FuseKit and compares logits plus selected KV rows.
"""

from __future__ import annotations

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import scaled_dot_product_attention, swiglu


def _as_model_dir(path: str) -> str:
    return path


def _unwrap_model(model):
    return getattr(model, "_fk_original", model)


def _diff(a, b) -> dict[str, float | int]:
    aa = a.astype(mx.float32).reshape(-1)
    bb = b.astype(mx.float32).reshape(-1)
    n = min(int(aa.size), int(bb.size))
    d = mx.abs(aa[:n] - bb[:n])
    mx.eval(d)
    return {
        "n": n,
        "max_abs": float(mx.max(d).item()) if n else 0.0,
        "mean_abs": float(mx.mean(d).item()) if n else 0.0,
    }


def _stats(a) -> dict[str, float | int]:
    aa = a.astype(mx.float32).reshape(-1)
    mx.eval(aa)
    return {
        "n": int(aa.size),
        "min": float(mx.min(aa).item()) if int(aa.size) else 0.0,
        "max": float(mx.max(aa).item()) if int(aa.size) else 0.0,
        "mean_abs": float(mx.mean(mx.abs(aa)).item()) if int(aa.size) else 0.0,
    }


def _top(row, k: int = 5) -> list[list[float | int]]:
    rr = row.astype(mx.float32).reshape(-1)
    idx = mx.argpartition(-rr, kth=k - 1)[:k]
    vals = rr[idx]
    order = mx.argsort(-vals)
    idx = idx[order]
    vals = vals[order]
    mx.eval(idx, vals)
    return [[int(i.item()), float(v.item())] for i, v in zip(idx, vals, strict=False)]


def _cache_row(cache_layer, row: int, kind: str):
    arr = cache_layer.keys if kind == "k" else cache_layer.values
    if isinstance(arr, (tuple, list)):
        raise RuntimeError("quantized KV cache is not handled by this debug script")
    return arr[:, :, row : row + 1, :].astype(mx.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--prompt", default="token")
    ap.add_argument("--prompt-repeat", type=int, default=160)
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    prompt = (args.prompt + " ") * args.prompt_repeat
    model, tok = load(_as_model_dir(args.model_dir))
    tok.eos_token_ids = set()
    raw_model = _unwrap_model(model)
    ids = mx.array([tok.encode(prompt) or [0]], dtype=mx.uint32)
    if int(ids.shape[1]) < args.prefix + args.m:
        raise RuntimeError(f"prompt only has {ids.shape[1]} tokens")

    cache_raw = make_prompt_cache(raw_model)
    cache_fk = make_prompt_cache(raw_model)
    cache_formula = make_prompt_cache(raw_model)

    prefix = ids[:, : args.prefix]
    chunk = ids[:, args.prefix : args.prefix + args.m]
    if args.prefix > 0:
        y0 = raw_model(prefix, cache=cache_raw)
        y1 = raw_model(prefix, cache=cache_fk)
        y2 = raw_model(prefix, cache=cache_formula)
        if isinstance(y0, tuple):
            y0 = y0[0]
        if isinstance(y1, tuple):
            y1 = y1[0]
        if isinstance(y2, tuple):
            y2 = y2[0]
        mx.eval(
            y0,
            y1,
            y2,
            [c.state for c in cache_raw],
            [c.state for c in cache_fk],
            [c.state for c in cache_formula],
        )

    core = raw_model.model
    layer = core.layers[args.layer]
    formula_h_attn = None
    formula_layer_out = None
    if args.layer != 0:
        formula_sdpa = None
    else:
        hidden0 = core.embed_tokens(chunk)
        normed = layer.input_layernorm(hidden0)
        q = layer.self_attn.q_proj(normed)
        k = layer.self_attn.k_proj(normed)
        v = layer.self_attn.v_proj(normed)
        q = q.reshape(1, args.m, layer.self_attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(1, args.m, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(1, args.m, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q_rope = layer.self_attn.rope(q, offset=cache_formula[0].offset)
        k_rope = layer.self_attn.rope(k, offset=cache_formula[0].offset)
        mask = create_attention_mask(hidden0, cache_formula[0])
        keys, values = cache_formula[0].update_and_fetch(k_rope, v)
        formula_sdpa = scaled_dot_product_attention(
            q_rope,
            keys,
            values,
            cache=cache_formula[0],
            scale=layer.self_attn.scale,
            mask=mask,
        ).astype(mx.float32)
        attn = formula_sdpa.transpose(0, 2, 1, 3).reshape(1, args.m, -1)
        o_proj = layer.self_attn.o_proj(attn)
        formula_h_attn = hidden0 + o_proj
        post_norm = layer.post_attention_layernorm(formula_h_attn)
        formula_gate = layer.mlp.gate_proj(post_norm)
        formula_up = layer.mlp.up_proj(post_norm)
        formula_silu_up = swiglu(formula_gate, formula_up)
        mlp = layer.mlp.down_proj(formula_silu_up)
        formula_layer_out = formula_h_attn + mlp
        mx.eval(formula_sdpa, formula_h_attn, formula_gate, formula_up, formula_silu_up, mlp, formula_layer_out)

    raw_logits = raw_model(chunk, cache=cache_raw)
    fk_logits = model(chunk, cache=cache_fk)
    if isinstance(raw_logits, tuple):
        raw_logits = raw_logits[0]
    if isinstance(fk_logits, tuple):
        fk_logits = fk_logits[0]
    mx.eval(raw_logits, fk_logits, [c.state for c in cache_raw], [c.state for c in cache_fk])

    rows = {
        "prompt_tokens": int(ids.shape[1]),
        "prefix": args.prefix,
        "m": args.m,
        "raw_argmax_last": int(mx.argmax(raw_logits[:, -1, :], axis=-1).item()),
        "fk_argmax_last": int(mx.argmax(fk_logits[:, -1, :], axis=-1).item()),
        "logits_last": _diff(raw_logits[:, -1, :], fk_logits[:, -1, :]),
        "raw_top": _top(raw_logits[:, -1, :]),
        "fk_top": _top(fk_logits[:, -1, :]),
    }
    for pos in (args.prefix, args.prefix + args.m - 1):
        rows[f"layer{args.layer}_k_row{pos}"] = _diff(
            _cache_row(cache_raw[args.layer], pos, "k"),
            _cache_row(cache_fk[args.layer], pos, "k"),
        )
        rows[f"layer{args.layer}_v_row{pos}"] = _diff(
            _cache_row(cache_raw[args.layer], pos, "v"),
            _cache_row(cache_fk[args.layer], pos, "v"),
        )
    if args.layer == 0 and formula_sdpa is not None:
        disp = model.__dict__["_fk_variant_dispatchers"].get(f"prefill_m{args.m}")
        if disp is not None:
            sid = disp._layer_scratch_id.get(("sdpa_buf", 0))
            if sid is not None:
                fk_sdpa = disp._scratch_buf(sid).astype(mx.float32)
                rows["layer0_sdpa_all"] = _diff(formula_sdpa, fk_sdpa)
                rows["layer0_sdpa_row0"] = _diff(
                    formula_sdpa[:, :, :1, :],
                    fk_sdpa[:, :, :1, :],
                )
                rows["layer0_sdpa_last"] = _diff(
                    formula_sdpa[:, :, args.m - 1 : args.m, :],
                    fk_sdpa[:, :, args.m - 1 : args.m, :],
                )
            for name, ref in (
                ("h_attn", formula_h_attn),
                ("gate_buf", formula_gate),
                ("up_buf", formula_up),
                ("silu_up", formula_silu_up),
                ("out_buf", formula_layer_out),
            ):
                sid = disp._layer_scratch_id.get((name, 0))
                if sid is None and name == "silu_up":
                    for task in getattr(disp, "_decode_tasks", []):
                        if (
                            task.get("layer_idx") == 0
                            and str(task.get("kernel", "")).startswith("fusekit_qmv_swiglu_")
                        ):
                            outs = [b["scratch_id"] for b in task.get("outputs", []) if b.get("role") == "scratch"]
                            if outs:
                                sid = outs[0]
                            break
                if sid is None or ref is None:
                    continue
                got = disp._scratch_buf(sid).astype(mx.float32)
                rows[f"layer0_{name}_all"] = _diff(ref, got)
                rows[f"layer0_{name}_ref_stats"] = _stats(ref)
                rows[f"layer0_{name}_got_stats"] = _stats(got)
                rows[f"layer0_{name}_row0"] = _diff(ref[:, :1, :], got[:, :1, :])
                rows[f"layer0_{name}_last"] = _diff(
                    ref[:, args.m - 1 : args.m, :],
                    got[:, args.m - 1 : args.m, :],
                )
                if name == "silu_up":
                    for cmp_name, cmp_ref in (
                        ("gate", formula_gate),
                        ("up", formula_up),
                        ("swiglu_reversed", swiglu(formula_up, formula_gate)),
                    ):
                        rows[f"layer0_silu_up_vs_{cmp_name}_all"] = _diff(cmp_ref, got)
                        rows[f"layer0_silu_up_vs_{cmp_name}_last"] = _diff(
                            cmp_ref[:, args.m - 1 : args.m, :],
                            got[:, args.m - 1 : args.m, :],
                        )
            for task in getattr(disp, "_decode_tasks", []):
                if task.get("layer_idx") != 0:
                    continue
                if not str(task.get("kernel", "")).startswith("fusekit_qmv_fanout_"):
                    continue
                bufs = disp._bufs_for_task(task, 0, 0, cache_fk)
                rows["layer0_qmv_fanout_direct_buffers"] = len(bufs)
                rows["layer0_qmv_fanout_direct_gpu_s"] = float(
                    disp._rt.dispatch_batch(
                        disp._handle,
                        [
                            (
                                task["kernel"],
                                bufs,
                                tuple(task["grid_threads"]),
                                tuple(task["threadgroup"]),
                            )
                        ],
                    )
                )
                outs = [b["scratch_id"] for b in task.get("outputs", []) if b.get("role") == "scratch"]
                if len(outs) >= 2:
                    direct_gate = disp._scratch_buf(outs[0]).astype(mx.float32)
                    direct_up = disp._scratch_buf(outs[1]).astype(mx.float32)
                    rows["layer0_gate_buf_direct_all"] = _diff(formula_gate, direct_gate)
                    rows["layer0_gate_buf_direct_stats"] = _stats(direct_gate)
                    rows["layer0_up_buf_direct_all"] = _diff(formula_up, direct_up)
                    rows["layer0_up_buf_direct_stats"] = _stats(direct_up)
                break
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
