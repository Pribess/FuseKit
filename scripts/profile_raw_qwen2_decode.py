"""Synchronized raw mlx-lm Qwen2 decode breakdown.

This script intentionally runs with FUSEKIT disabled and manually reproduces a
single-token Qwen2 decode path using the loaded mlx-lm modules. It synchronizes
around each component to attribute elapsed time to model elements.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generation_stream
from mlx_lm.models.activations import swiglu
from mlx_lm.models import cache as cache_mod
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention


DEFAULT_PROMPT = "Repeat the word token forever: token token token"


@dataclass
class TimerStats:
    total_s: float = 0.0
    count: int = 0


class Profiler:
    def __init__(self) -> None:
        self.stats: dict[str, TimerStats] = defaultdict(TimerStats)

    def time(self, name: str, fn):
        mx.synchronize()
        started = time.perf_counter()
        value = fn()
        mx.eval(value)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        stat = self.stats[name]
        stat.total_s += elapsed
        stat.count += 1
        return value


def encode_prompt(tokenizer, prompt: str) -> mx.array:
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    return mx.array(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def model_cache_state(cache):
    return [c.state for c in cache]


def prefill(model, prompt: mx.array, prompt_cache, prefill_step_size: int) -> mx.array:
    with mx.stream(generation_stream):
        while prompt.size > 1:
            n_to_process = min(prefill_step_size, prompt.size - 1)
            model(prompt[:n_to_process][None], cache=prompt_cache)
            mx.eval(model_cache_state(prompt_cache))
            prompt = prompt[n_to_process:]
            mx.clear_cache()
    mx.synchronize()
    return prompt


def qwen2_decode_one(model, input_tokens: mx.array, prompt_cache, prof: Profiler):
    core = model.model
    h = prof.time("embed", lambda: core.embed_tokens(input_tokens[None]))
    mask = create_attention_mask(h, prompt_cache[0])

    for layer_idx, (layer, c) in enumerate(zip(core.layers, prompt_cache)):
        layer_prefix = f"layer.{layer_idx:02d}"

        x = h
        xn = prof.time("attn.input_rmsnorm", lambda layer=layer, x=x: layer.input_layernorm(x))
        prof.stats[f"{layer_prefix}.attn.input_rmsnorm"].total_s += 0.0

        attn = layer.self_attn
        bsz, seq_len, _ = xn.shape

        q = prof.time("attn.q_proj", lambda attn=attn, xn=xn: attn.q_proj(xn))
        k = prof.time("attn.k_proj", lambda attn=attn, xn=xn: attn.k_proj(xn))
        v = prof.time("attn.v_proj", lambda attn=attn, xn=xn: attn.v_proj(xn))

        q = prof.time(
            "attn.q_reshape",
            lambda q=q, attn=attn: q.reshape(bsz, seq_len, attn.n_heads, -1).transpose(
                0, 2, 1, 3
            ),
        )
        k = prof.time(
            "attn.k_reshape",
            lambda k=k, attn=attn: k.reshape(
                bsz, seq_len, attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3),
        )
        v = prof.time(
            "attn.v_reshape",
            lambda v=v, attn=attn: v.reshape(
                bsz, seq_len, attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3),
        )

        q = prof.time("attn.rope_q", lambda attn=attn, q=q, c=c: attn.rope(q, offset=c.offset))
        k = prof.time("attn.rope_k", lambda attn=attn, k=k, c=c: attn.rope(k, offset=c.offset))
        k, v = prof.time("attn.cache_update_fetch", lambda c=c, k=k, v=v: c.update_and_fetch(k, v))

        y = prof.time(
            "attn.sdpa",
            lambda q=q, k=k, v=v, c=c, attn=attn: scaled_dot_product_attention(
                q,
                k,
                v,
                cache=c,
                scale=attn.scale,
                mask=mask,
            ),
        )
        y = prof.time(
            "attn.out_reshape",
            lambda y=y, bsz=bsz, seq_len=seq_len: y.transpose(0, 2, 1, 3).reshape(
                bsz, seq_len, -1
            ),
        )
        r = prof.time("attn.o_proj", lambda attn=attn, y=y: attn.o_proj(y))
        h = prof.time("residual.attn_add", lambda x=x, r=r: x + r)

        hn = prof.time(
            "mlp.post_rmsnorm",
            lambda layer=layer, h=h: layer.post_attention_layernorm(h),
        )
        gate = prof.time("mlp.gate_proj", lambda layer=layer, hn=hn: layer.mlp.gate_proj(hn))
        up = prof.time("mlp.up_proj", lambda layer=layer, hn=hn: layer.mlp.up_proj(hn))
        act = prof.time("mlp.swiglu", lambda gate=gate, up=up: swiglu(gate, up))
        down = prof.time("mlp.down_proj", lambda layer=layer, act=act: layer.mlp.down_proj(act))
        h = prof.time("residual.mlp_add", lambda h=h, down=down: h + down)

    h = prof.time("final.rmsnorm", lambda core=core, h=h: core.norm(h))
    if model.args.tie_word_embeddings:
        logits = prof.time("final.lm_head", lambda core=core, h=h: core.embed_tokens.as_linear(h))
    else:
        logits = prof.time("final.lm_head", lambda model=model, h=h: model.lm_head(h))
    logits = prof.time("sample.slice_last", lambda logits=logits: logits[:, -1, :])
    logprobs = prof.time(
        "sample.logsumexp",
        lambda logits=logits: logits - mx.logsumexp(logits, keepdims=True),
    )
    sampled = prof.time("sample.argmax", lambda logprobs=logprobs: mx.argmax(logprobs, axis=-1))
    return sampled


def qwen2_decode_one_grouped(model, input_tokens: mx.array, prompt_cache, prof: Profiler):
    core = model.model
    h = prof.time("embed", lambda: core.embed_tokens(input_tokens[None]))
    mask = create_attention_mask(h, prompt_cache[0])

    for layer, c in zip(core.layers, prompt_cache):
        x = h
        xn = prof.time("attn.input_rmsnorm", lambda layer=layer, x=x: layer.input_layernorm(x))
        attn = layer.self_attn
        bsz, seq_len, _ = xn.shape

        def qkv_proj_reshape():
            q = attn.q_proj(xn)
            k = attn.k_proj(xn)
            v = attn.v_proj(xn)
            q = q.reshape(bsz, seq_len, attn.n_heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seq_len, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seq_len, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            return q, k, v

        q, k, v = prof.time("attn.qkv_proj_reshape", qkv_proj_reshape)

        def rope_cache():
            q_rope = attn.rope(q, offset=c.offset)
            k_rope = attn.rope(k, offset=c.offset)
            k_cache, v_cache = c.update_and_fetch(k_rope, v)
            return q_rope, k_cache, v_cache

        q, k, v = prof.time("attn.rope_cache_update", rope_cache)
        y = prof.time(
            "attn.sdpa",
            lambda q=q, k=k, v=v, c=c, attn=attn: scaled_dot_product_attention(
                q,
                k,
                v,
                cache=c,
                scale=attn.scale,
                mask=mask,
            ),
        )

        def o_proj_residual():
            yy = y.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
            rr = attn.o_proj(yy)
            return x + rr

        h = prof.time("attn.o_proj_residual", o_proj_residual)
        hn = prof.time(
            "mlp.post_rmsnorm",
            lambda layer=layer, h=h: layer.post_attention_layernorm(h),
        )

        def mlp_project_act_down():
            gate = layer.mlp.gate_proj(hn)
            up = layer.mlp.up_proj(hn)
            act = swiglu(gate, up)
            return layer.mlp.down_proj(act)

        down = prof.time("mlp.gate_up_swiglu_down", mlp_project_act_down)
        h = prof.time("residual.mlp_add", lambda h=h, down=down: h + down)

    h = prof.time("final.rmsnorm", lambda core=core, h=h: core.norm(h))
    if model.args.tie_word_embeddings:
        logits = prof.time("final.lm_head", lambda core=core, h=h: core.embed_tokens.as_linear(h))
    else:
        logits = prof.time("final.lm_head", lambda model=model, h=h: model.lm_head(h))
    logits = prof.time("sample.slice_last", lambda logits=logits: logits[:, -1, :])
    logprobs = prof.time(
        "sample.logsumexp",
        lambda logits=logits: logits - mx.logsumexp(logits, keepdims=True),
    )
    sampled = prof.time("sample.argmax", lambda logprobs=logprobs: mx.argmax(logprobs, axis=-1))
    return sampled


def qwen2_decode_one_block(model, input_tokens: mx.array, prompt_cache, prof: Profiler):
    core = model.model
    h = prof.time("embed", lambda: core.embed_tokens(input_tokens[None]))
    mask = create_attention_mask(h, prompt_cache[0])

    for layer, c in zip(core.layers, prompt_cache):
        h = prof.time(
            "layer.attn_block",
            lambda layer=layer, c=c, h=h: h
            + layer.self_attn(layer.input_layernorm(h), mask, c),
        )
        h = prof.time(
            "layer.mlp_block",
            lambda layer=layer, h=h: h + layer.mlp(layer.post_attention_layernorm(h)),
        )

    h = prof.time("final.rmsnorm", lambda core=core, h=h: core.norm(h))
    if model.args.tie_word_embeddings:
        logits = prof.time("final.lm_head", lambda core=core, h=h: core.embed_tokens.as_linear(h))
    else:
        logits = prof.time("final.lm_head", lambda model=model, h=h: model.lm_head(h))
    logits = prof.time("sample.slice_last", lambda logits=logits: logits[:, -1, :])
    logprobs = prof.time(
        "sample.logsumexp",
        lambda logits=logits: logits - mx.logsumexp(logits, keepdims=True),
    )
    sampled = prof.time("sample.argmax", lambda logprobs=logprobs: mx.argmax(logprobs, axis=-1))
    return sampled


def print_summary(prof: Profiler) -> None:
    total_s = sum(stat.total_s for stat in prof.stats.values())
    print(
        f"[raw-profile] total_sync_us={total_s * 1_000_000.0:.1f} "
        f"components={len(prof.stats)}",
        flush=True,
    )
    print("[raw-profile] aggregate by component:", flush=True)
    for name, stat in sorted(
        prof.stats.items(), key=lambda item: item[1].total_s, reverse=True
    ):
        if stat.total_s == 0.0:
            continue
        pct = (stat.total_s / total_s * 100.0) if total_s else 0.0
        avg_us = stat.total_s / max(stat.count, 1) * 1_000_000.0
        print(
            "[raw-profile] "
            f"{stat.total_s * 1_000_000.0:10.2f} us "
            f"{pct:6.2f}% count={stat.count:5d} avg={avg_us:9.2f} us component={name}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--mode", choices=("block", "grouped", "fine"), default="block")
    args = parser.parse_args()

    print(
        "[raw-profile] "
        f"hook_disable={os.environ.get('FUSEKIT_DISABLE_HOOK', '')!r} "
        f"tokens={args.tokens} warmup={args.warmup} mode={args.mode}",
        flush=True,
    )

    model, tokenizer = load(args.model_dir)
    tokenizer.eos_token_ids = set()
    mx.synchronize()

    prompt = encode_prompt(tokenizer, args.prompt)
    prompt_cache = cache_mod.make_prompt_cache(model)
    y = prefill(model, prompt, prompt_cache, args.prefill_step_size)

    with mx.stream(generation_stream):
        for _ in range(args.warmup):
            prof = Profiler()
            if args.mode == "fine":
                y = qwen2_decode_one(model, y, prompt_cache, prof)
            elif args.mode == "grouped":
                y = qwen2_decode_one_grouped(model, y, prompt_cache, prof)
            else:
                y = qwen2_decode_one_block(model, y, prompt_cache, prof)
            mx.eval(y)

    prof = Profiler()
    started = time.perf_counter()
    with mx.stream(generation_stream):
        for _ in range(args.tokens):
            if args.mode == "fine":
                y = qwen2_decode_one(model, y, prompt_cache, prof)
            elif args.mode == "grouped":
                y = qwen2_decode_one_grouped(model, y, prompt_cache, prof)
            else:
                y = qwen2_decode_one_block(model, y, prompt_cache, prof)
            mx.eval(y)
    mx.synchronize()
    print(
        f"[raw-profile] measured_wall_s={time.perf_counter() - started:.6f}",
        flush=True,
    )
    print_summary(prof)
    print(f"[raw-profile] last_token={int(y.item())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
