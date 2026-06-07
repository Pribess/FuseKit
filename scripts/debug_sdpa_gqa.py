"""Compare MLX fast SDPA with manual GQA mappings for one Qwen2 decode step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import scaled_dot_product_attention

    model, tok = load(str(model_path(args.model)))
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    cache = make_prompt_cache(model)
    prefill = model(ids, cache=cache)
    if isinstance(prefill, tuple):
        prefill = prefill[0]
    mx.eval(prefill)

    core = model.model
    layer = core.layers[0]
    hidden = core.embed_tokens(mx.array([[args.decode_token]], dtype=mx.uint32))
    normed = layer.input_layernorm(hidden)
    q = layer.self_attn.q_proj(normed)
    k = layer.self_attn.k_proj(normed)
    v = layer.self_attn.v_proj(normed)
    q = q.reshape(1, 1, layer.self_attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(1, 1, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, layer.self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = layer.self_attn.rope(q, offset=cache[0].offset)
    k = layer.self_attn.rope(k, offset=cache[0].offset)
    keys, values = cache[0].update_and_fetch(k, v)

    ref = scaled_dot_product_attention(
        q, keys, values, cache=cache[0], scale=layer.self_attn.scale, mask=None
    ).astype(mx.float32)

    qf = q.astype(mx.float32)
    kf = keys.astype(mx.float32)
    vf = values.astype(mx.float32)
    _, n_heads, _, _ = q.shape
    n_kv_heads = keys.shape[1]
    repeats = n_heads // n_kv_heads
    rows = []
    for mode in ("grouped", "interleaved"):
        outs = []
        for head in range(n_heads):
            kv_head = head // repeats if mode == "grouped" else head % n_kv_heads
            scores = mx.sum(qf[:, head : head + 1, :, :] * kf[:, kv_head : kv_head + 1, :, :], axis=-1)
            scores = mx.softmax(scores * layer.self_attn.scale, axis=-1, precise=True)
            out = mx.sum(scores[..., None] * vf[:, kv_head : kv_head + 1, :, :], axis=-2)
            outs.append(out)
        manual = mx.concatenate(outs, axis=1).reshape(ref.shape).astype(mx.float32)
        diff = mx.abs(ref - manual)
        mx.eval(diff)
        rows.append(
            {
                "mode": mode,
                "max_abs_diff": float(mx.max(diff).item()),
                "mean_abs_diff": float(mx.mean(diff).item()),
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
