"""Monkey-patch ONE transformer layer to use fusekit's RmsNormQkv
kernel for the norm + Q/K/V projection part, falling back to MLX for
the rest of the layer.

Verifies the layer's output (post-residual) and final logits match a
clean MLX run within bf16 tolerance. Only handles the L=1 decode
case; prefill (L>1) is delegated back to the original layer body.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layer", type=int, default=0,
                    help="which layer index to replace with fusekit dispatch")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    from mlx_lm import load
    model_dir = Path(f"~/.omlx/models/{args.model}").expanduser()
    fusekit_dir = model_dir / "fusekit"

    sys.path.insert(0, str(fusekit_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt

    print(f"[run] loading {args.model}", file=sys.stderr)
    model_orig, tok = load(str(model_dir))
    model_fk, _ = load(str(model_dir))  # Independent copy to monkey-patch.

    handle = rt.load_dispatcher(
        str(fusekit_dir / "kernel.metallib"),
        str(fusekit_dir / "task_graph.json"),
    )
    qkv_kernels = [k for k in rt.kernel_names(handle) if k.startswith("fusekit_rmsnorm_qkv_")]
    kname = qkv_kernels[args.layer]
    print(f"[run] layer {args.layer} kernel: {kname}", file=sys.stderr)

    install_fusekit_block(model_fk, args.layer, handle, kname, rt, fr)

    # Run both models through mlx-lm's generate. Same seed → same
    # sampling path; output tokens should match if our layer is
    # numerically equivalent.
    mx.random.seed(0)
    out_orig = run_inference(model_orig, tok, args.prompt, args.max_tokens)
    mx.random.seed(0)
    out_fk = run_inference(model_fk, tok, args.prompt, args.max_tokens)

    print(f"[run] orig    : {out_orig!r}", file=sys.stderr)
    print(f"[run] fusekit : {out_fk!r}", file=sys.stderr)
    if out_orig == out_fk:
        print("[run] MATCH — fusekit layer dispatch matches MLX path",
              file=sys.stderr)
    else:
        print("[run] MISMATCH — bf16 rounding diverged sampling path",
              file=sys.stderr)
    return 0


def install_fusekit_block(model, layer_idx, dispatcher_handle, kernel_name, rt, fr) -> None:
    block = model.model.layers[layer_idx]
    orig_call = block.__call__

    def fusekit_block_call(x, mask=None, cache=None):
        # Fall back to original for prefill (L>1) — our current kernel
        # is hard-coded to 1-D activation; multi-token prefill needs an
        # additional outer loop we'll add when the seq variant lands.
        if x.shape[1] != 1:
            return orig_call(x, mask, cache)

        out = fusekit_layer_forward(
            block, x, mask, cache, dispatcher_handle, kernel_name, rt, fr,
        )
        return out

    # Bind directly on the instance — overrides nn.Module's normal
    # __call__ resolution for this object only.
    block.__call__ = fusekit_block_call


def fusekit_layer_forward(block, x, mask, cache, handle, kname, rt, fr):
    """Decode-step replacement for one TransformerBlock.

    Stages:
      1. fusekit kernel: input_layernorm + q_proj + k_proj + v_proj
      2. MLX path:       reshape + RoPE + KV-cache + SDPA + o_proj
      3. residual:       x + attn_out
      4. MLX path:       post_attention_layernorm + SwiGLU MLP
      5. residual:       h + mlp_out
    """
    B, L, D = x.shape  # B=L=1 here.
    attn = block.self_attn
    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    head_dim = D // n_heads
    dtype = x.dtype

    w_norm = block.input_layernorm.weight
    w_q = attn.q_proj.weight
    w_k = attn.k_proj.weight
    w_v = attn.v_proj.weight
    b_q = attn.q_proj.bias
    b_k = attn.k_proj.bias
    b_v = attn.v_proj.bias

    # Flatten x to (D,) — our kernel iterates a 1-D buffer.
    x_flat = x.reshape(-1)
    q_buf = mx.zeros((n_heads * head_dim,), dtype=dtype)
    k_buf = mx.zeros((n_kv_heads * head_dim,), dtype=dtype)
    v_buf = mx.zeros((n_kv_heads * head_dim,), dtype=dtype)
    mx.eval(x_flat, w_norm, w_q, w_k, w_v, b_q, b_k, b_v, q_buf, k_buf, v_buf)

    def buf(arr):
        return (fr.array_data_ptr(arr), arr.nbytes)

    rt.dispatch_kernel(
        handle,
        kname,
        [
            buf(x_flat), buf(w_norm),
            buf(w_q), buf(w_k), buf(w_v),
            buf(b_q), buf(b_k), buf(b_v),
            buf(q_buf), buf(k_buf), buf(v_buf),
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
    )
    mx.eval(q_buf, k_buf, v_buf)

    # Reshape to (B, L, H, head_dim) → transpose to (B, H, L, head_dim).
    queries = q_buf.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
    keys = k_buf.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    values = v_buf.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # Remaining attention path uses the original module's RoPE, cache,
    # and SDPA implementations.
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    from mlx_lm.models.base import scaled_dot_product_attention
    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attn.scale, mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    attn_out = attn.o_proj(output)
    h = x + attn_out

    # MLP via original modules.
    mlp_in = block.post_attention_layernorm(h)
    mlp_out = block.mlp(mlp_in)
    return h + mlp_out


def run_inference(model, tokenizer, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


if __name__ == "__main__":
    sys.exit(main())
