"""DeepSeek layer-0 raw-vs-fusekit intermediate comparison."""

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


def inner_model(model):
    return getattr(model, "_fk_original", model)


def row1(x):
    if x.ndim == 4:
        return x[:, :, :1, :].reshape(1, -1)
    if x.ndim == 3:
        return x[:, :1, :].reshape(1, -1)
    if x.ndim == 2:
        return x[:1, :]
    return x.reshape(1, -1)


def scratch_slot(x, slot: int):
    if x.ndim == 4:
        return x[:, :, slot : slot + 1, :]
    if x.ndim == 3:
        return x[:, slot : slot + 1, :]
    return x


def raw_child(args: argparse.Namespace) -> int:
    os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_FAST"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(str(model_path(args.model)))
    text = inner_model(model)
    core = text.model
    cache = make_prompt_cache(text)

    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    hidden = core.embed_tokens(ids)
    mask = create_attention_mask(hidden, cache[0])
    for layer, c in zip(core.layers, cache):
        hidden = layer(hidden, mask, cache=c)
    mx.eval(hidden)

    decode = mx.array([[args.decode_token]], dtype=mx.uint32)
    hidden = core.embed_tokens(decode)
    mask = create_attention_mask(hidden, cache[0])
    layer = core.layers[0]
    attn = layer.self_attn
    x = layer.input_layernorm(hidden)
    B, L, _ = x.shape
    offset_before = int(cache[0].offset)

    if attn.q_lora_rank is None:
        q = attn.q_proj(x)
    else:
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x)))
    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe_in = mx.split(q, [attn.qk_nope_head_dim], axis=-1)

    compressed_kv = attn.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe_flat = mx.split(compressed_kv, [attn.kv_lora_rank], axis=-1)
    k_pe_in = k_pe_flat.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv = attn.kv_b_proj(attn.kv_a_layernorm(compressed_kv))
    kv = kv.reshape(B, L, attn.num_heads, -1).transpose(0, 2, 1, 3)
    k_nope, values = mx.split(kv, [attn.qk_nope_head_dim], axis=-1)

    q_pe = attn.rope(q_pe_in, cache[0].offset)
    k_pe_rope = attn.rope(k_pe_in, cache[0].offset)
    k_pe = mx.repeat(k_pe_rope, attn.num_heads, axis=1)
    current_keys = mx.concatenate([k_nope, k_pe], axis=-1)
    keys, values = cache[0].update_and_fetch(current_keys, values)
    queries = mx.concatenate([q_nope, q_pe], axis=-1)
    sdpa = scaled_dot_product_attention(
        queries, keys, values, cache=cache[0], scale=attn.scale, mask=mask
    )
    q_scaled = queries * attn.scale
    manual_scores = mx.matmul(
        q_scaled, keys[:, :, : offset_before + 1, :].transpose(0, 1, 3, 2)
    )
    manual_probs = mx.softmax(manual_scores, axis=-1)
    manual_sdpa = mx.matmul(manual_probs, values[:, :, : offset_before + 1, :])
    attn_out = attn.o_proj(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))
    h_attn = hidden + attn_out
    out = h_attn + layer.mlp(layer.post_attention_layernorm(h_attn))
    normed = core.norm(out)
    logits = text.lm_head(normed)
    mx.eval(queries, current_keys, sdpa, h_attn, out, logits)

    arrays = {
        "q_nope": q_nope.astype(mx.float32),
        "q_pe_in": q_pe_in.astype(mx.float32),
        "q_pe_rope": q_pe.astype(mx.float32),
        "q_buf": queries.astype(mx.float32),
        "q_scaled": q_scaled.astype(mx.float32),
        "k_pe_in": k_pe_in.astype(mx.float32),
        "k_pe_rope": k_pe_rope.astype(mx.float32),
        "k_pe_repeat": k_pe.astype(mx.float32),
        "k_nope": k_nope.astype(mx.float32),
        "k_current": current_keys.astype(mx.float32),
        "k_cache_prefix": keys[:, :, : offset_before + 1, :].astype(mx.float32),
        "v_current": values[:, :, offset_before : offset_before + 1, :].astype(mx.float32),
        "v_cache_prefix": values[:, :, : offset_before + 1, :].astype(mx.float32),
        "manual_sdpa": manual_sdpa.astype(mx.float32),
        "sdpa_buf": sdpa.astype(mx.float32),
        "h_attn": h_attn.astype(mx.float32),
        "out_buf": out.astype(mx.float32),
        "logits": row1(logits).astype(mx.float32),
    }
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(json.dumps({"mode": "raw", "offset_before": offset_before}))
    return 0


def patched_child(args: argparse.Namespace) -> int:
    os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
    os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    os.environ.pop("FUSEKIT_ENABLE_FAST", None)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(str(model_path(args.model)))
    cache = make_prompt_cache(model)
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    offset_before = int(cache[0].offset)
    logits = model(mx.array([[args.decode_token]], dtype=mx.uint32), cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)
    disp = model._fk_dispatcher
    graph = json.loads((model_path(args.model) / "fusekit" / "task_graph.json").read_text())
    k_store_input_sid = None
    for task in graph["tasks"]:
        if task.get("layer_idx") == 0 and "kv_cache_store" in task["kernel"]:
            k_store_input_sid = task["inputs"][0]["scratch_id"]
            break
    sid_by_task_out = {}
    for task in graph["tasks"]:
        if task.get("layer_idx") != 0:
            continue
        outs = [b["scratch_id"] for b in task.get("outputs", []) if b.get("role") == "scratch"]
        if outs:
            sid_by_task_out[task["id"]] = outs
    arrays = {
        "q_nope": disp._scratch_buf(sid_by_task_out[4][0]).astype(mx.float32),
        "q_pe_in": disp._scratch_buf(sid_by_task_out[4][1]).astype(mx.float32),
        "q_pe_rope": disp._scratch_buf(sid_by_task_out[29][0]).astype(mx.float32),
        "q_buf": disp._scratch_buf(sid_by_task_out[30][0]).astype(mx.float32),
        "q_scaled": disp._scratch_buf(disp._layer_scratch_id[("q_buf", 0)]).astype(mx.float32),
        "k_pe_in": disp._scratch_buf(sid_by_task_out[37][0]).astype(mx.float32),
        "k_pe_rope": disp._scratch_buf(sid_by_task_out[38][0]).astype(mx.float32),
        "k_pe_repeat": disp._scratch_buf(sid_by_task_out[41][0]).astype(mx.float32),
        "k_nope": disp._scratch_buf(sid_by_task_out[35][0]).astype(mx.float32),
        "k_pre_store": disp._scratch_buf(k_store_input_sid).astype(mx.float32),
        "v_current": disp._scratch_buf(sid_by_task_out[35][1]).astype(mx.float32),
        "sdpa_buf": disp._scratch_buf(disp._layer_scratch_id[("sdpa_buf", 0)]).astype(mx.float32),
        "h_attn": disp._scratch_buf(disp._layer_scratch_id[("h_attn", 0)]).astype(mx.float32),
        "out_buf": disp._scratch_buf(disp._layer_scratch_id[("out_buf", 0)]).astype(mx.float32),
        "logits": row1(logits).astype(mx.float32),
    }
    arrays["k_current"] = cache[0].keys[:, :, offset_before : offset_before + 1, :].astype(mx.float32)
    arrays["k_cache_prefix"] = cache[0].keys[:, :, : offset_before + 1, :].astype(mx.float32)
    arrays["v_cache_current"] = cache[0].values[:, :, offset_before : offset_before + 1, :].astype(mx.float32)
    arrays["v_cache_prefix"] = cache[0].values[:, :, : offset_before + 1, :].astype(mx.float32)
    q0 = scratch_slot(arrays["q_scaled"], 0).astype(mx.float32)
    manual_scores = mx.matmul(q0, arrays["k_cache_prefix"].transpose(0, 1, 3, 2))
    manual_probs = mx.softmax(manual_scores, axis=-1)
    arrays["manual_sdpa"] = mx.matmul(manual_probs, arrays["v_cache_prefix"]).astype(mx.float32)
    mx.save_safetensors(args.out, arrays)
    Path(args.meta).write_text(json.dumps({
        "mode": "patched",
        "offset_before": offset_before,
        "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
    }))
    return 0


def child_main(args: argparse.Namespace) -> int:
    if args.mode == "raw":
        return raw_child(args)
    return patched_child(args)


def parent_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-deepseek-layer0-") as td:
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
        subprocess.run(base + ["--mode", "raw", "--out", str(raw_out), "--meta", str(raw_meta)], check=True)
        subprocess.run(base + ["--mode", "patched", "--out", str(patched_out), "--meta", str(patched_meta)], check=True)

        raw = mx.load(str(raw_out))
        patched = mx.load(str(patched_out))
        rows = []
        for key in [
            "q_nope",
            "q_pe_in",
            "q_pe_rope",
            "q_buf",
            "q_scaled",
            "k_pe_in",
            "k_pe_rope",
            "k_pe_repeat",
            "k_nope",
            "k_pre_store",
            "k_current",
            "k_cache_prefix",
            "v_current",
            "v_cache_current",
            "v_cache_prefix",
            "manual_sdpa",
            "sdpa_buf",
            "h_attn",
            "out_buf",
            "logits",
        ]:
            if key == "k_pre_store":
                lhs = raw["k_current"]
                rhs = patched[key]
            elif key == "v_cache_current":
                lhs = raw["v_current"]
                rhs = patched[key]
            else:
                lhs = raw[key]
                rhs = patched[key]
            best_slot = None
            if lhs.ndim >= 3 and rhs.ndim == lhs.ndim and lhs.shape[-1] == rhs.shape[-1]:
                lhs_seq = int(lhs.shape[-2])
                rhs_seq = int(rhs.shape[-2])
                if lhs_seq == 1 and rhs_seq > 1:
                    best = None
                    for slot in range(rhs_seq):
                        d = mx.abs(lhs - scratch_slot(rhs, slot))
                        mx.eval(d)
                        mean = float(mx.mean(d).item())
                        item = (mean, slot, d)
                        if best is None or item[0] < best[0]:
                            best = item
                    best_slot = best[1]
                    diff = best[2]
                else:
                    diff = mx.abs(lhs - rhs)
            else:
                diff = mx.abs(lhs - rhs)
            mx.eval(diff)
            row = {
                "name": key,
                "max_abs_diff": float(mx.max(diff).item()),
                "mean_abs_diff": float(mx.mean(diff).item()),
            }
            if best_slot is not None:
                row["best_slot"] = best_slot
            rows.append(row)
        print(json.dumps({
            "model": args.model,
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
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
