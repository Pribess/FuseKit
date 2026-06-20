#!/usr/bin/env python3
"""Benchmark prompt prefill block forwards for raw MLX vs FuseKit.

The parent process spawns one child per mode so `FUSEKIT_DISABLE_HOOK`
is decided before Python startup. That keeps raw and patched measurements
from sharing the same import-time hook state.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


OMLX_PY = "/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11"
OMLX_PYTHONPATH = (
    "/Users/pribess/Documents/workspace/fusekit/python:"
    "/Applications/oMLX.app/Contents/Resources:"
    "/Applications/oMLX.app/Contents/Python/framework-mlx-framework/"
    "lib/python3.11/site-packages"
)
OMLX_MODEL_ROOT = Path("/Users/pribess/.omlx/models")


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return OMLX_MODEL_ROOT / name_or_path


def env_for(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = OMLX_PYTHONPATH
    env.setdefault(
        "PYTHONPYCACHEPREFIX",
        "/Users/pribess/Documents/workspace/fusekit/.pycache-check",
    )
    if mode == "raw":
        env["FUSEKIT_DISABLE_HOOK"] = "1"
        env["FUSEKIT_DISABLE_DISPATCH"] = "1"
        env["FUSEKIT_DISABLE_FAST"] = "1"
    else:
        env.pop("FUSEKIT_DISABLE_HOOK", None)
        env.pop("FUSEKIT_DISABLE_DISPATCH", None)
        env.pop("FUSEKIT_DISABLE_FAST", None)
    return env


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def make_prompt_ids(tokenizer: Any, length: int):
    import mlx.core as mx

    text = (
        "Write a concise Python function that parses JSON lines, validates "
        "required fields, and returns aggregate statistics. "
    )
    pieces: list[int] = []
    repeated = text
    while len(pieces) < length:
        pieces = tokenizer.encode(repeated, add_special_tokens=True)
        repeated += text
    pieces = pieces[:length]
    return mx.array(pieces, dtype=mx.uint32)


def cache_state(cache: Any) -> list[Any]:
    return [c.state for c in cache]


def cache_supports_trim_padding(cache: Any) -> bool:
    allowed = {
        "KVCache",
        "QuantizedKVCache",
        "SimpleKVCache",
        "RotatingKVCache",
        "BatchKVCache",
    }
    return bool(cache) and all(
        c is not None
        and type(c).__name__ in allowed
        and callable(getattr(c, "trim", None))
        for c in cache
    )


def maybe_pad_ids(ids: Any, cache: Any, multiple: int) -> tuple[Any, int]:
    if multiple <= 1 or not cache_supports_trim_padding(cache):
        return ids, 0
    import mlx.core as mx

    n = int(ids.shape[0])
    padded = ((n + multiple - 1) // multiple) * multiple
    extra = padded - n
    if extra <= 0:
        return ids, 0
    return mx.concatenate([ids, mx.zeros((extra,), dtype=ids.dtype)], axis=0), extra


def trim_cache(cache: Any, n: int) -> None:
    if n <= 0:
        return
    for c in cache:
        c.trim(n)


def cache_debug_summary(cache: Any) -> list[dict[str, Any]]:
    import mlx.core as mx

    def arr_stats(a: Any) -> dict[str, Any] | None:
        if not hasattr(a, "shape"):
            return None
        try:
            finite = mx.isfinite(a)
            return {
                "dtype": str(getattr(a, "dtype", "")),
                "min": float(mx.min(a).item()),
                "max": float(mx.max(a).item()),
                "finite": bool(mx.all(finite).item()),
            }
        except Exception as e:
            return {"dtype": str(getattr(a, "dtype", "")), "error": str(e)}

    out = []
    for c in cache:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        key_shape = None
        value_shape = None
        if isinstance(keys, (tuple, list)):
            key_shape = [tuple(getattr(a, "shape", ())) for a in keys]
        elif keys is not None:
            key_shape = tuple(getattr(keys, "shape", ()))
        if isinstance(values, (tuple, list)):
            value_shape = [tuple(getattr(a, "shape", ())) for a in values]
        elif values is not None:
            value_shape = tuple(getattr(values, "shape", ()))
        offset = getattr(c, "offset", None)
        try:
            offset = int(offset)
        except Exception:
            offset = str(offset)
        out.append({
            "type": type(c).__name__,
            "offset": offset,
            "key_shape": key_shape,
            "value_shape": value_shape,
            "key_stats": arr_stats(keys) if not isinstance(keys, (tuple, list)) else None,
            "value_stats": arr_stats(values) if not isinstance(values, (tuple, list)) else None,
        })
    return out


def cache_kv_diff(raw_cache: Any, fk_cache: Any) -> dict[str, Any]:
    import mlx.core as mx

    max_key = 0.0
    max_value = 0.0
    mean_key = 0.0
    mean_value = 0.0
    count = 0
    for raw_c, fk_c in zip(raw_cache, fk_cache):
        raw_keys = getattr(raw_c, "keys", None)
        fk_keys = getattr(fk_c, "keys", None)
        raw_values = getattr(raw_c, "values", None)
        fk_values = getattr(fk_c, "values", None)
        if not hasattr(raw_keys, "shape") or not hasattr(fk_keys, "shape"):
            continue
        if not hasattr(raw_values, "shape") or not hasattr(fk_values, "shape"):
            continue
        kd = mx.abs(raw_keys - fk_keys)
        vd = mx.abs(raw_values - fk_values)
        max_key = max(max_key, float(mx.max(kd).item()))
        max_value = max(max_value, float(mx.max(vd).item()))
        mean_key += float(mx.mean(kd).item())
        mean_value += float(mx.mean(vd).item())
        count += 1
    if count:
        mean_key /= count
        mean_value /= count
    return {
        "layers_compared": count,
        "key_max_abs": max_key,
        "key_mean_abs": mean_key,
        "value_max_abs": max_value,
        "value_mean_abs": mean_value,
    }


def try_native_no_logits(model: Any, tokens: Any, cache: Any):
    candidates = []
    for path in (
        "model",
        "language_model.model",
        "language_model.model.model",
        "text_model.model",
    ):
        cur = model
        ok = True
        for seg in path.split("."):
            try:
                cur = getattr(cur, seg)
            except Exception:
                ok = False
                break
        if ok and cur is not model and callable(cur):
            candidates.append(cur)
    for candidate in candidates:
        try:
            out = candidate(tokens[None], cache=cache)
        except TypeError:
            try:
                out = candidate(tokens[None], cache)
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(out, tuple):
            out = out[0]
        if getattr(out, "shape", None) is None:
            continue
        return out
    return None


def next_token_check(model: Any, ids: Any, pad_to: int) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    inner = getattr(model, "_fk_original", model)
    next_ids = ids[-1:]

    raw_cache = make_prompt_cache(inner)
    raw_out = inner(ids[None], cache=raw_cache)
    if isinstance(raw_out, tuple):
        raw_out = raw_out[0]
    mx.eval(raw_out, cache_state(raw_cache))

    fk_cache_for_cache_check = make_prompt_cache(model)
    fk_ids, fk_pad = maybe_pad_ids(ids, fk_cache_for_cache_check, pad_to)
    fk_out = model(
        fk_ids[None],
        cache=fk_cache_for_cache_check,
        _fusekit_prefill_no_logits=True,
    )
    if isinstance(fk_out, tuple):
        fk_out = fk_out[0]
    trim_cache(fk_cache_for_cache_check, fk_pad)
    mx.eval(fk_out, cache_state(fk_cache_for_cache_check))
    kv_diff = cache_kv_diff(raw_cache, fk_cache_for_cache_check)
    raw_cache_summary = cache_debug_summary(raw_cache)[:1]
    fk_cache_summary = cache_debug_summary(fk_cache_for_cache_check)[:1]

    raw_next = inner(next_ids[None], cache=raw_cache)
    fk_cache_next = inner(next_ids[None], cache=fk_cache_for_cache_check)
    if isinstance(raw_next, tuple):
        raw_next = raw_next[0]
    if isinstance(fk_cache_next, tuple):
        fk_cache_next = fk_cache_next[0]
    raw_logits = raw_next[:, -1, :]
    fk_cache_logits = fk_cache_next[:, -1, :]
    cache_diff = mx.abs(raw_logits - fk_cache_logits)
    raw_argmax = int(mx.argmax(raw_logits.reshape(-1)).item())
    fk_cache_argmax = int(mx.argmax(fk_cache_logits.reshape(-1)).item())
    max_abs_cache = float(mx.max(cache_diff).item())
    mean_abs_cache = float(mx.mean(cache_diff).item())

    fk_cache_for_full_check = make_prompt_cache(model)
    fk_ids, fk_pad = maybe_pad_ids(ids, fk_cache_for_full_check, pad_to)
    fk_out = model(
        fk_ids[None],
        cache=fk_cache_for_full_check,
        _fusekit_prefill_no_logits=True,
    )
    if isinstance(fk_out, tuple):
        fk_out = fk_out[0]
    trim_cache(fk_cache_for_full_check, fk_pad)
    mx.eval(fk_out, cache_state(fk_cache_for_full_check))

    fk_full_next = model(next_ids[None], cache=fk_cache_for_full_check)
    if isinstance(fk_full_next, tuple):
        fk_full_next = fk_full_next[0]
    fk_full_logits = fk_full_next[:, -1, :]
    full_diff = mx.abs(raw_logits - fk_full_logits)
    fk_full_argmax = int(mx.argmax(fk_full_logits.reshape(-1)).item())
    max_abs_full = float(mx.max(full_diff).item())
    mean_abs_full = float(mx.mean(full_diff).item())
    mx.eval(
        raw_next,
        fk_cache_next,
        fk_full_next,
        cache_state(raw_cache),
        cache_state(fk_cache_for_cache_check),
        cache_state(fk_cache_for_full_check),
    )
    return {
        "raw_argmax": raw_argmax,
        "fk_cache_argmax": fk_cache_argmax,
        "fk_full_argmax": fk_full_argmax,
        "cache_argmax_match": raw_argmax == fk_cache_argmax,
        "full_argmax_match": raw_argmax == fk_full_argmax,
        "cache_max_abs": max_abs_cache,
        "cache_mean_abs": mean_abs_cache,
        "full_max_abs": max_abs_full,
        "full_mean_abs": mean_abs_full,
        "raw_cache": raw_cache_summary,
        "fk_cache": fk_cache_summary,
        "kv_diff": kv_diff,
    }


def child_main(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    path = str(model_path(args.model))
    started = time.perf_counter()
    model, tokenizer = load(path)
    mx.synchronize()
    load_s = time.perf_counter() - started
    ids = make_prompt_ids(tokenizer, args.length)
    mx.eval(ids)
    prompt_shape = tuple(ids.shape)
    wrapper = type(model).__name__
    check_result = None
    if args.check_next:
        check_result = next_token_check(model, ids, args.pad_to)
        print("[prefill:check] " + json.dumps(check_result, sort_keys=True), flush=True)

    samples: list[dict[str, float]] = []
    checksum = 0
    total = args.warmup + args.repeat
    for run in range(total):
        cache = make_prompt_cache(model)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        mx.synchronize()
        t0 = time.perf_counter()
        call_kwargs: dict[str, Any] = {"cache": cache}
        run_ids = ids
        pad_extra = 0
        if args.no_logits:
            run_ids, pad_extra = maybe_pad_ids(ids, cache, args.pad_to)
            if args.mode == "raw":
                out = try_native_no_logits(model, run_ids, cache)
                if out is None:
                    raise RuntimeError("raw no-logits backbone path unavailable")
            else:
                call_kwargs["_fusekit_prefill_no_logits"] = True
                out = model(run_ids[None], **call_kwargs)
        else:
            out = model(run_ids[None], **call_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        trim_cache(cache, pad_extra)
        mx.eval(out, cache_state(cache))
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        if args.no_logits:
            tok = 0
        else:
            flat = out[:, -1, :].reshape(-1)
            tok = int(mx.argmax(flat).item())
        checksum ^= tok
        peak_gb = float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else 0.0
        if run >= args.warmup:
            rec = {
                "run": run - args.warmup,
                "wall_ms": elapsed * 1000.0,
                "tok_s": float(args.length) / elapsed if elapsed else 0.0,
                "peak_gb": peak_gb,
                "argmax": tok,
            }
            samples.append(rec)
            print(
                "[prefill:run] "
                f"mode={args.mode} model={args.model} length={args.length} "
                f"run={rec['run']} wall_ms={rec['wall_ms']:.3f} "
                f"tok_s={rec['tok_s']:.3f} peak_gb={rec['peak_gb']:.3f} "
                f"argmax={tok} no_logits={args.no_logits} pad_extra={pad_extra}",
                flush=True,
            )
        mx.clear_cache()

    walls = [r["wall_ms"] for r in samples]
    tps = [r["tok_s"] for r in samples]
    peaks = [r["peak_gb"] for r in samples]
    result = {
        "model": args.model,
        "mode": args.mode,
        "length": args.length,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "load_s": load_s,
        "wrapper": wrapper,
        "dispatcher_built": getattr(model, "_fk_dispatcher", None) is not None,
        "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
        "prompt_shape": prompt_shape,
        "wall_ms_avg": mean(walls),
        "wall_ms_stdev": stdev(walls),
        "tok_s_avg": mean(tps),
        "tok_s_stdev": stdev(tps),
        "peak_gb_avg": mean(peaks),
        "checksum": checksum,
        "samples": samples,
    }
    if check_result is not None:
        result["next_check"] = check_result
    print("[prefill:json] " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def run_child(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    cmd = [
        OMLX_PY,
        __file__,
        "--child",
        "--mode",
        mode,
        "--model",
        args.model,
        "--length",
        str(args.length),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
    ]
    if args.no_logits:
        cmd.append("--no-logits")
        cmd.extend(["--pad-to", str(args.pad_to)])
    if mode == "fusekit" and args.check_next:
        cmd.append("--check-next")
    print(
        "[prefill:path] "
        f"mode={mode} model={args.model} length={args.length} "
        f"repeat={args.repeat} warmup={args.warmup}",
        flush=True,
    )
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env_for(mode),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("[prefill:json] "):
            return json.loads(line[len("[prefill:json] "):])
    raise RuntimeError(f"missing [prefill:json] for mode={mode}")


def print_table(rows: list[dict[str, Any]]) -> None:
    base = rows[0]
    print(
        "model mode length wall_ms_avg wall_ms_sd tok_s_avg tok_s_sd "
        "delta_vs_first peak_gb wrapper dispatcher_built dispatch_failed",
        flush=True,
    )
    for row in rows:
        delta = (
            (base["wall_ms_avg"] - row["wall_ms_avg"]) / base["wall_ms_avg"] * 100.0
            if base["wall_ms_avg"]
            else 0.0
        )
        print(
            f"{row['model']} {row['mode']} {row['length']} "
            f"{row['wall_ms_avg']:.3f} {row['wall_ms_stdev']:.3f} "
            f"{row['tok_s_avg']:.3f} {row['tok_s_stdev']:.3f} "
            f"{delta:.1f}% {row['peak_gb_avg']:.3f} {row['wrapper']}",
            f"{row['dispatcher_built']} {row['dispatch_failed']}",
            flush=True,
        )


def parent_main(args: argparse.Namespace) -> int:
    modes = ["raw", "fusekit"] if args.mode == "both" else [args.mode]
    rows = [run_child(args, mode) for mode in modes]
    print_table(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=["raw", "fusekit", "both"], default="both")
    parser.add_argument("--model", required=True)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--no-logits",
        action="store_true",
        help="for fusekit mode, benchmark prefill chunks that skip logits",
    )
    parser.add_argument(
        "--pad-to",
        type=int,
        default=1,
        help="for fusekit no-logits mode, pad prefill length to this multiple and trim cache",
    )
    parser.add_argument(
        "--check-next",
        action="store_true",
        help="in fusekit mode, compare raw vs fusekit-prefill next-token logits",
    )
    args = parser.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
