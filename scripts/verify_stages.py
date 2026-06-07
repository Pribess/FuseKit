"""Stage-by-stage comparison of fusekit kernels against MLX for a
single transformer layer. Drives the same model layer with real
mid-forward state (post-embedding hidden), runs both paths in
parallel, and reports max_err per stage so we can locate divergence.

Usage:
    python3 verify_stages.py <model> [--layer N] [--prompt TEXT]

Stages probed:
    1. RmsNormQkv     (input_layernorm + Q/K/V proj)
    2. SDPA           (after MLX RoPE + cache to make K,V available)
    3. o_proj + resid (LinearResidual)
    4. mlp_block      (full MLP fused, if accepted; else gateup_sw + down separately)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="Hello")
    args = ap.parse_args()

    from mlx_lm import load
    model_dir = Path(f"~/.omlx/models/{args.model}").expanduser()
    fk_dir = model_dir / "fusekit"
    sys.path.insert(0, str(fk_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    print(f"[verify] loading {args.model}", file=sys.stderr)
    model, tok = load(str(model_dir))
    handle = rt.load_dispatcher(
        str(fk_dir / "kernel.metallib"),
        str(fk_dir / "task_graph.json"),
    )
    ks = rt.kernel_names(handle)
    qkv_k = sorted(k for k in ks if "rmsnorm_qkv" in k)[args.layer]
    sdpa_k = sorted(k for k in ks if "sdpa_decode" in k)[args.layer]
    linres = sorted(k for k in ks if "linear_residual" in k)
    mlp_blk_ks = sorted(k for k in ks if "rmsnorm_gateup_swiglu_down" in k)
    gateup_sw_ks = sorted(k for k in ks if k.startswith("fusekit_rmsnorm_gateup_swiglu_f"))
    has_mlp_block = bool(mlp_blk_ks)
    if has_mlp_block:
        mlp_k = mlp_blk_ks[args.layer]
        o_proj_k = linres[args.layer]
        down_k = None
    else:
        mlp_k = None
        gateup_sw_k = gateup_sw_ks[args.layer]
        o_proj_k = linres[2 * args.layer]
        down_k = linres[2 * args.layer + 1]

    block = model.model.layers[args.layer]
    attn = block.self_attn
    mlp = block.mlp
    cfg = model.args
    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    D = cfg.hidden_size
    head_dim = D // n_heads
    n_int = mlp.gate_proj.weight.shape[0]
    dtype = mx.bfloat16

    def buf(a):
        return (fr.array_data_ptr(a), a.nbytes)

    # Drive the layer with a realistic hidden state — run the model up
    # to layer N's input on the prompt's last token.
    ids = mx.array(tok.encode(args.prompt), dtype=mx.uint32).reshape(1, -1)
    from mlx_lm.models import cache as cache_mod
    caches = cache_mod.make_prompt_cache(model)
    # Prefill the prompt to fill caches up to position L-1.
    _ = model(ids, cache=caches)
    mx.eval(_)
    # Now use a fresh "next token" decode step.
    next_id = mx.array([[1]], dtype=mx.uint32)
    h = model.model.embed_tokens(next_id)  # (1, 1, D)
    for li in range(args.layer):
        h = model.model.layers[li](h, mask=None, cache=caches[li])
        mx.eval(h)
    print(f"[verify] layer {args.layer} input h shape={h.shape}", file=sys.stderr)

    cache_li = caches[args.layer]
    mx.eval(h)

    # === Stage 1: RmsNormQkv ===
    # MLX reference
    x_mlx = h
    q_mlx_pre = attn.q_proj(block.input_layernorm(x_mlx))  # (1,1, n_heads*head_dim)
    k_mlx_pre = attn.k_proj(block.input_layernorm(x_mlx))
    v_mlx_pre = attn.v_proj(block.input_layernorm(x_mlx))
    mx.eval(q_mlx_pre, k_mlx_pre, v_mlx_pre)

    # fusekit
    x_flat = mx.contiguous(x_mlx).reshape(-1)
    q_buf = mx.zeros((n_heads * head_dim,), dtype=dtype)
    k_buf = mx.zeros((n_kv_heads * head_dim,), dtype=dtype)
    v_buf = mx.zeros((n_kv_heads * head_dim,), dtype=dtype)
    mx.eval(x_flat, q_buf, k_buf, v_buf,
            block.input_layernorm.weight,
            attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
            attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias)
    rt.dispatch_kernel(handle, qkv_k, [
        buf(x_flat), buf(block.input_layernorm.weight),
        buf(attn.q_proj.weight), buf(attn.k_proj.weight), buf(attn.v_proj.weight),
        buf(attn.q_proj.bias), buf(attn.k_proj.bias), buf(attn.v_proj.bias),
        buf(q_buf), buf(k_buf), buf(v_buf),
    ], grid=(256, 1, 1), threadgroup=(256, 1, 1))
    mx.eval(q_buf, k_buf, v_buf)

    q_diff = mx.abs(q_buf.astype(mx.float32) - q_mlx_pre.reshape(-1).astype(mx.float32))
    k_diff = mx.abs(k_buf.astype(mx.float32) - k_mlx_pre.reshape(-1).astype(mx.float32))
    v_diff = mx.abs(v_buf.astype(mx.float32) - v_mlx_pre.reshape(-1).astype(mx.float32))
    print(f"[stage1 rmsnorm_qkv]  Q max_err={float(q_diff.max()):.4f}  "
          f"K max_err={float(k_diff.max()):.4f}  V max_err={float(v_diff.max()):.4f}",
          file=sys.stderr)

    # Use fusekit Q/K/V going forward. Reshape to (B, n_heads, L, head_dim).
    B, L = 1, 1
    queries = q_buf.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    keys = k_buf.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = v_buf.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    queries = attn.rope(queries, offset=cache_li.offset)
    keys = attn.rope(keys, offset=cache_li.offset)
    keys, values = cache_li.update_and_fetch(keys, values)
    kv_len = keys.shape[2]
    mx.eval(queries, keys, values)

    # === Stage 2: SDPA ===
    # MLX reference
    scale = 1.0 / (head_dim ** 0.5)
    attn_mlx = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale)
    mx.eval(attn_mlx)
    # fusekit
    Q = mx.contiguous(queries.reshape(n_heads, head_dim))
    K = mx.contiguous(keys.reshape(n_kv_heads, kv_len, head_dim))
    V = mx.contiguous(values.reshape(n_kv_heads, kv_len, head_dim))
    sdpa_buf = mx.zeros((n_heads, head_dim), dtype=dtype)
    kv_len_arr = mx.array([kv_len], dtype=mx.uint32)
    mx.eval(Q, K, V, sdpa_buf, kv_len_arr)
    rt.dispatch_kernel(handle, sdpa_k, [
        buf(Q), buf(K), buf(V), buf(kv_len_arr), buf(sdpa_buf),
    ], grid=(n_heads * 64, 1, 1), threadgroup=(64, 1, 1))
    mx.eval(sdpa_buf)
    s_diff = mx.abs(sdpa_buf.astype(mx.float32) - attn_mlx.reshape(n_heads, head_dim).astype(mx.float32))
    print(f"[stage2 sdpa_decode]  max_err={float(s_diff.max()):.4f}  "
          f"mean_err={float(s_diff.mean()):.4f}", file=sys.stderr)

    # === Stage 3: LinearResidual (o_proj + attn residual) ===
    attn_out_mlx = attn_mlx.transpose(0, 2, 1, 3).reshape(B, L, -1)
    h_attn_mlx = attn.o_proj(attn_out_mlx) + x_mlx
    mx.eval(h_attn_mlx)

    sdpa_flat = sdpa_buf.reshape(-1)
    x_resid = mx.contiguous(x_mlx).reshape(-1)
    h_attn = mx.zeros((D,), dtype=dtype)
    mx.eval(sdpa_flat, x_resid, h_attn, attn.o_proj.weight)
    rt.dispatch_kernel(handle, o_proj_k, [
        buf(sdpa_flat), buf(attn.o_proj.weight), buf(x_resid), buf(h_attn),
    ], grid=(256, 1, 1), threadgroup=(256, 1, 1))
    mx.eval(h_attn)

    ha_diff = mx.abs(h_attn.astype(mx.float32) - h_attn_mlx.reshape(-1).astype(mx.float32))
    print(f"[stage3 o_proj+resid] max_err={float(ha_diff.max()):.4f}  "
          f"mean_err={float(ha_diff.mean()):.4f}", file=sys.stderr)

    # === Stage 4: MLP block (or gateup_sw + down separately) ===
    h_attn_3d = h_attn.reshape(1, 1, D)
    norm_h = block.post_attention_layernorm(h_attn_3d)
    silu_up_mlx = mx.sigmoid(mlp.gate_proj(norm_h)) * mlp.gate_proj(norm_h) * mlp.up_proj(norm_h)
    out_mlx = mlp.down_proj(silu_up_mlx) + h_attn_3d
    mx.eval(out_mlx)

    if has_mlp_block:
        out_buf = mx.zeros((D,), dtype=dtype)
        mx.eval(out_buf)
        rt.dispatch_kernel(handle, mlp_k, [
            buf(h_attn), buf(block.post_attention_layernorm.weight),
            buf(mlp.gate_proj.weight), buf(mlp.up_proj.weight),
            buf(mlp.down_proj.weight), buf(out_buf),
        ], grid=(256, 1, 1), threadgroup=(256, 1, 1))
    else:
        silu_up = mx.zeros((n_int,), dtype=dtype)
        out_buf = mx.zeros((D,), dtype=dtype)
        mx.eval(silu_up, out_buf)
        rt.dispatch_kernel(handle, gateup_sw_k, [
            buf(h_attn), buf(block.post_attention_layernorm.weight),
            buf(mlp.gate_proj.weight), buf(mlp.up_proj.weight),
            buf(silu_up),
        ], grid=(256, 1, 1), threadgroup=(256, 1, 1))
        mx.eval(silu_up)
        rt.dispatch_kernel(handle, down_k, [
            buf(silu_up), buf(mlp.down_proj.weight), buf(h_attn), buf(out_buf),
        ], grid=(256, 1, 1), threadgroup=(256, 1, 1))
    mx.eval(out_buf)

    o_diff = mx.abs(out_buf.astype(mx.float32) - out_mlx.reshape(-1).astype(mx.float32))
    print(f"[stage4 mlp_block]    max_err={float(o_diff.max()):.4f}  "
          f"mean_err={float(o_diff.mean()):.4f}", file=sys.stderr)

    # === REPLAY using dispatch_batch (same as fusekit_layer_forward) ===
    # Reset all output buffers and re-run the same SDPA→o_proj→mlp chain
    # through dispatch_batch to see if the batched path diverges from
    # the per-dispatch path.
    sdpa_buf2 = mx.zeros((n_heads, head_dim), dtype=dtype)
    h_attn2 = mx.zeros((D,), dtype=dtype)
    out_buf2 = mx.zeros((D,), dtype=dtype)
    mx.eval(sdpa_buf2, h_attn2, out_buf2)

    if has_mlp_block:
        dispatches = [
            (sdpa_k,
             [buf(Q), buf(K), buf(V), buf(kv_len_arr), buf(sdpa_buf2)],
             (n_heads * 64, 1, 1), (64, 1, 1)),
            (o_proj_k,
             [buf(sdpa_buf2), buf(attn.o_proj.weight), buf(x_resid), buf(h_attn2)],
             (256, 1, 1), (256, 1, 1)),
            (mlp_k,
             [buf(h_attn2), buf(block.post_attention_layernorm.weight),
              buf(mlp.gate_proj.weight), buf(mlp.up_proj.weight),
              buf(mlp.down_proj.weight), buf(out_buf2)],
             (256, 1, 1), (256, 1, 1)),
        ]
    rt.dispatch_batch(handle, dispatches)
    mx.eval(out_buf2)
    batch_diff = mx.abs(out_buf2.astype(mx.float32) - out_mlx.reshape(-1).astype(mx.float32))
    print(f"[batched mlp_block]   max_err={float(batch_diff.max()):.4f}  "
          f"mean_err={float(batch_diff.mean()):.4f}", file=sys.stderr)

    # === dispatch_batch with ONLY mlp_k (one dispatch, no inter-dispatch sync) ===
    out_buf3 = mx.zeros((D,), dtype=dtype)
    mx.eval(out_buf3)
    # Reuse h_attn from per-stage section (already populated)
    rt.dispatch_batch(handle, [(
        mlp_k,
        [buf(h_attn), buf(block.post_attention_layernorm.weight),
         buf(mlp.gate_proj.weight), buf(mlp.up_proj.weight),
         buf(mlp.down_proj.weight), buf(out_buf3)],
        (256, 1, 1), (256, 1, 1),
    )])
    mx.eval(out_buf3)
    single_batch_diff = mx.abs(out_buf3.astype(mx.float32) - out_mlx.reshape(-1).astype(mx.float32))
    print(f"[batch 1-dispatch]    max_err={float(single_batch_diff.max()):.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
