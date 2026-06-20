"""Synchronized raw mlx-lm decode op profiler.

Runs with FuseKit disabled and times selected MLX primitives/modules during
single-token decode. The numbers are intentionally synchronized around each
wrapped call so they are comparable to FuseKit's per-kernel profile buckets.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


OMLX_SITE = Path(
    "/Applications/oMLX.app/Contents/Python/framework-mlx-framework/"
    "lib/python3.11/site-packages"
)
if OMLX_SITE.exists():
    sys.path.insert(0, str(OMLX_SITE))

os.environ.setdefault("FUSEKIT_DISABLE_HOOK", "1")
os.environ.setdefault("FUSEKIT_DISABLE_DISPATCH", "1")
os.environ.setdefault("FUSEKIT_DISABLE_FAST", "1")

import mlx.core as mx


DEFAULT_PROMPT = "Repeat the word token forever: token token token"


def install_gemma4_shared_kv_filter() -> None:
    """Let raw Gemma4 MLX loads ignore shared-layer K/V tensors.

    Some exported Gemma4 checkpoints keep K/V tensors for layers that the
    mlx-lm config marks as KV-shared. FuseKit's runtime installs this filter
    for normal runs, but this profiler starts with the hook disabled so it can
    measure raw MLX. Install the load-only filter here as measurement plumbing.
    """

    try:
        from mlx_lm.models import gemma4_text
    except Exception:
        return

    model_cls = getattr(gemma4_text, "Model", None)
    original = getattr(model_cls, "sanitize", None)
    if model_cls is None or original is None:
        return
    if getattr(original, "_fusekit_profile_gemma4_shared_kv_filter", False):
        return

    def layer_index(key: str) -> int | None:
        marker = ".layers."
        pos = key.find(marker)
        if pos < 0:
            return None
        tail = key[pos + len(marker) :]
        dot = tail.find(".")
        if dot <= 0:
            return None
        try:
            return int(tail[:dot])
        except ValueError:
            return None

    def sanitize(self, weights):
        sanitized = original(self, weights)
        args = getattr(self, "args", None)
        try:
            num_layers = int(getattr(args, "num_hidden_layers", 0) or 0)
            shared_layers = int(getattr(args, "num_kv_shared_layers", 0) or 0)
        except (TypeError, ValueError):
            return sanitized
        first_shared = num_layers - shared_layers
        if num_layers <= 0 or shared_layers <= 0 or first_shared < 0:
            return sanitized

        return {
            key: value
            for key, value in sanitized.items()
            if not (
                (idx := layer_index(key)) is not None
                and idx >= first_shared
                and (
                    ".self_attn.k_proj." in key
                    or ".self_attn.v_proj." in key
                    or ".self_attn.k_norm." in key
                )
            )
        }

    sanitize._fusekit_profile_gemma4_shared_kv_filter = True
    model_cls.sanitize = sanitize


@dataclass
class Stat:
    total_s: float = 0.0
    count: int = 0
    by_sig: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


class Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self.stats: dict[str, Stat] = defaultdict(Stat)

    def record(self, name: str, sig: str, elapsed_s: float) -> None:
        stat = self.stats[name]
        stat.total_s += elapsed_s
        stat.count += 1
        stat.by_sig[sig].append(elapsed_s)


PROF = Profiler()


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def shape_sig(value: Any) -> str:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"{tuple(value.shape)}:{value.dtype}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(shape_sig(v) for v in value) + "]"
    if value is None:
        return "None"
    return type(value).__name__


def call_sig(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts = [shape_sig(a) for a in args[:6]]
    for key in ("transpose", "group_size", "bits", "mode", "sorted_indices", "scale"):
        if key in kwargs:
            parts.append(f"{key}={kwargs[key]}")
    return " ".join(parts)


def module_call_sig(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if not args:
        return call_sig(args, kwargs)

    module = args[0]
    name = getattr(module, "_fusekit_profile_name", module.__class__.__name__)
    parts = [str(name)]
    try:
        weight = module["weight"]
    except Exception:
        weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape"):
        parts.append(f"weight={tuple(weight.shape)}:{getattr(weight, 'dtype', '?')}")
    try:
        has_bias = "bias" in module
    except Exception:
        has_bias = False
    parts.append(f"bias={has_bias}")
    parts.extend(shape_sig(a) for a in args[1:6])
    for key in ("transpose", "group_size", "bits", "mode", "sorted_indices", "scale"):
        if key in kwargs:
            parts.append(f"{key}={kwargs[key]}")
    return " ".join(parts)


def force_eval(value: Any) -> None:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        mx.eval(value)
    elif isinstance(value, (list, tuple)):
        arrays = [v for v in value if hasattr(v, "shape") and hasattr(v, "dtype")]
        if arrays:
            mx.eval(*arrays)


def wrap_call(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not PROF.enabled:
            return fn(*args, **kwargs)
        mx.synchronize()
        started = time.perf_counter()
        out = fn(*args, **kwargs)
        force_eval(out)
        mx.synchronize()
        PROF.record(name, call_sig(args, kwargs), time.perf_counter() - started)
        return out

    return wrapped


def wrap_module_call(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not PROF.enabled:
            return fn(*args, **kwargs)
        mx.synchronize()
        started = time.perf_counter()
        out = fn(*args, **kwargs)
        force_eval(out)
        mx.synchronize()
        PROF.record(name, module_call_sig(args, kwargs), time.perf_counter() - started)
        return out

    return wrapped


def install_wrappers(include_modules: bool) -> None:
    for name in ("gather_qmm", "quantized_matmul", "addmm", "matmul"):
        if hasattr(mx, name):
            setattr(mx, name, wrap_call(f"op:{name}", getattr(mx, name)))

    if hasattr(mx, "fast"):
        for name in ("scaled_dot_product_attention", "rms_norm"):
            if hasattr(mx.fast, name):
                setattr(mx.fast, name, wrap_call(f"op:fast.{name}", getattr(mx.fast, name)))
        if hasattr(mx.fast, "metal_kernel"):
            orig_metal_kernel = mx.fast.metal_kernel

            def metal_kernel_wrapper(*args: Any, **kwargs: Any) -> Any:
                kernel = orig_metal_kernel(*args, **kwargs)
                kernel_name = kwargs.get("name")
                if kernel_name is None and args:
                    kernel_name = args[0]
                return wrap_call(f"op:metal_kernel.{kernel_name}", kernel)

            mx.fast.metal_kernel = metal_kernel_wrapper

    if not include_modules:
        return

    import mlx.nn.layers.linear as linear_mod
    import mlx.nn.layers.embedding as embedding_mod
    import mlx.nn.layers.quantized as quant_mod
    import mlx_lm.models.switch_layers as switch_mod

    linear_mod.Linear.__call__ = wrap_module_call("module:Linear", linear_mod.Linear.__call__)
    embedding_mod.Embedding.as_linear = wrap_module_call(
        "module:Embedding.as_linear", embedding_mod.Embedding.as_linear
    )
    quant_mod.QuantizedLinear.__call__ = wrap_module_call(
        "module:QuantizedLinear", quant_mod.QuantizedLinear.__call__
    )
    quant_mod.QuantizedEmbedding.as_linear = wrap_module_call(
        "module:QuantizedEmbedding.as_linear", quant_mod.QuantizedEmbedding.as_linear
    )
    switch_mod.SwitchLinear.__call__ = wrap_module_call(
        "module:SwitchLinear", switch_mod.SwitchLinear.__call__
    )
    switch_mod.QuantizedSwitchLinear.__call__ = wrap_module_call(
        "module:QuantizedSwitchLinear", switch_mod.QuantizedSwitchLinear.__call__
    )


def annotate_module_names(model: Any) -> None:
    if not hasattr(model, "named_modules"):
        return
    for name, module in model.named_modules():
        setattr(module, "_fusekit_profile_name", name)


def encode_prompt(tokenizer: Any, prompt: str) -> mx.array:
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    return mx.array(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def make_prompt_ids(tokenizer: Any, length: int) -> mx.array:
    text = (
        "Write a concise Python function that parses JSON lines, validates "
        "required fields, and returns aggregate statistics. "
    )
    pieces: list[int] = []
    repeated = text
    while len(pieces) < length:
        pieces = tokenizer.encode(repeated, add_special_tokens=True)
        repeated += text
    return mx.array(pieces[:length], dtype=mx.uint32)


def cache_state(cache: Any) -> list[Any]:
    states: list[Any] = []
    for c in cache:
        try:
            state = c.state
        except Exception:
            continue
        if state is not None:
            states.append(state)
    return states


def prefill(model: Any, prompt: mx.array, prompt_cache: Any, step_size: int) -> mx.array:
    while prompt.size > 1:
        n = min(step_size, prompt.size - 1)
        out = model(prompt[:n][None], cache=prompt_cache)
        force_eval(out)
        mx.eval(*cache_state(prompt_cache))
        prompt = prompt[n:]
        mx.clear_cache()
    mx.synchronize()
    return prompt


def decode_one(model: Any, token: mx.array, prompt_cache: Any) -> mx.array:
    logits = model(token[None], cache=prompt_cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    force_eval(logits)
    last = logits[:, -1, :]
    token = mx.argmax(last, axis=-1).astype(mx.uint32)
    mx.eval(token)
    return token


def summarize(top: int) -> None:
    total_s = sum(stat.total_s for stat in PROF.stats.values())
    print(
        f"[mlx-op-profile] profiled_total_us={total_s * 1_000_000.0:.2f}",
        flush=True,
    )
    rows = sorted(PROF.stats.items(), key=lambda kv: kv[1].total_s, reverse=True)
    print("[mlx-op-profile] aggregate:", flush=True)
    for name, stat in rows[:top]:
        avg_us = stat.total_s / max(stat.count, 1) * 1_000_000.0
        pct = stat.total_s / total_s * 100.0 if total_s else 0.0
        print(
            "[mlx-op-profile] "
            f"{stat.total_s * 1_000_000.0:10.2f} us "
            f"{pct:6.2f}% count={stat.count:5d} avg={avg_us:9.2f} us {name}",
            flush=True,
        )
    print("[mlx-op-profile] top signatures:", flush=True)
    sig_rows: list[tuple[float, str, int, str]] = []
    for name, stat in PROF.stats.items():
        for sig, values in stat.by_sig.items():
            sig_rows.append((sum(values), name, len(values), sig))
    for total, name, count, sig in sorted(sig_rows, reverse=True)[:top]:
        print(
            "[mlx-op-profile] "
            f"{total * 1_000_000.0:10.2f} us count={count:5d} {name} {sig}",
            flush=True,
        )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for name, stat in PROF.stats.items():
        if not name.startswith("module:"):
            continue
        for sig, values in stat.by_sig.items():
            normalized = re.sub(r"\.layers\.\d+\.", ".layers.*.", sig)
            grouped[(name, normalized)].extend(values)
    if grouped:
        print("[mlx-op-profile] grouped module signatures:", flush=True)
    grouped_rows = [
        (sum(values), name, len(values), sig)
        for (name, sig), values in grouped.items()
    ]
    for total, name, count, sig in sorted(grouped_rows, reverse=True)[:top]:
        print(
            "[mlx-op-profile] "
            f"{total * 1_000_000.0:10.2f} us count={count:5d} {name} {sig}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompt-length", type=int, default=0)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--include-modules", action="store_true")
    parser.add_argument(
        "--profile-prefill-only",
        action="store_true",
        help="profile the prompt block forward instead of decode tokens",
    )
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    args = parser.parse_args()

    print(
        "[mlx-op-profile] "
        f"hook_disable={os.environ.get('FUSEKIT_DISABLE_HOOK', '')!r} "
        f"dispatch_disable={os.environ.get('FUSEKIT_DISABLE_DISPATCH', '')!r} "
        f"tokens={args.tokens} warmup={args.warmup}",
        flush=True,
    )

    install_gemma4_shared_kv_filter()
    install_wrappers(args.include_modules)

    from mlx_lm import load
    from mlx_lm.generate import maybe_quantize_kv_cache
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(str(model_path(args.model)))
    annotate_module_names(model)
    tokenizer.eos_token_ids = set()
    mx.synchronize()

    prompt = (
        make_prompt_ids(tokenizer, args.prompt_length)
        if args.prompt_length > 0
        else encode_prompt(tokenizer, args.prompt)
    )
    prompt_cache = make_prompt_cache(model)
    if args.profile_prefill_only:
        mx.synchronize()
        PROF.enabled = True
        started = time.perf_counter()
        token = prefill(model, prompt, prompt_cache, args.prefill_step_size)
        mx.synchronize()
        PROF.enabled = False
        print(
            f"[mlx-op-profile] wall_us={(time.perf_counter() - started) * 1_000_000.0:.2f}",
            flush=True,
        )
        summarize(args.top)
        return 0

    token = prefill(model, prompt, prompt_cache, args.prefill_step_size)
    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=args.quantized_kv_start,
        kv_group_size=args.kv_group_size,
        kv_bits=args.kv_bits,
    )
    mx.eval(*cache_state(prompt_cache))

    for _ in range(args.warmup):
        token = decode_one(model, token, prompt_cache)
    mx.synchronize()

    PROF.enabled = True
    started = time.perf_counter()
    for _ in range(args.tokens):
        token = decode_one(model, token, prompt_cache)
    mx.synchronize()
    PROF.enabled = False
    print(
        f"[mlx-op-profile] wall_us={(time.perf_counter() - started) * 1_000_000.0:.2f}",
        flush=True,
    )
    summarize(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
