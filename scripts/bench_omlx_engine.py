"""Benchmark oMLX's engine backend directly.

This measures the path FuseKit is meant to patch in oMLX:

    BatchedEngine -> AsyncEngineCore -> Scheduler -> BatchGenerator

It intentionally avoids the HTTP server layer so results are not mixed
with SSE, JSON, or client latency.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


def _raw_mode_requested(argv: list[str]) -> bool:
    for i, arg in enumerate(argv):
        if arg == "--mode" and i + 1 < len(argv):
            return argv[i + 1] == "engine_raw"
        if arg == "--mode=engine_raw":
            return True
    return False


if _raw_mode_requested(sys.argv[1:]) and not os.environ.get("FUSEKIT_DISABLE_HOOK"):
    # FUSEKIT_DISABLE_HOOK is consumed during interpreter startup by the
    # fusekit .pth hook. Re-exec before importing mlx/omlx so direct raw
    # runs are the same pure baseline as bench_omlx_engine_suite.py.
    env = os.environ.copy()
    env["FUSEKIT_DISABLE_HOOK"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

import mlx.core as mx

from omlx.engine.batched import BatchedEngine
from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig


_PROFILE_STATS: dict[str, list[float]] | None = None
_PROFILE_LIMIT = 0
_MTP_STATS: dict[str, int] | None = None


def _record_profile(name: str, dt_s: float) -> None:
    if _PROFILE_STATS is None:
        return
    values = _PROFILE_STATS[name]
    if _PROFILE_LIMIT <= 0 or len(values) < _PROFILE_LIMIT:
        values.append(dt_s)


def _record_mtp_stat(name: str, delta: int = 1) -> None:
    if _MTP_STATS is None:
        return
    _MTP_STATS[name] += int(delta)


def _mtp_stat_int(obj, name: str) -> int | None:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _wrap_timed(cls, method_name: str, label: str) -> None:
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            _record_profile(label, time.perf_counter() - t0)

    setattr(cls, method_name, wrapped)


def _install_profile(limit: int) -> None:
    global _PROFILE_STATS, _PROFILE_LIMIT
    if limit == 0:
        return
    _PROFILE_STATS = defaultdict(list)
    _PROFILE_LIMIT = limit

    mlx_generate = importlib.import_module("mlx_lm.generate")
    import omlx.scheduler as omlx_scheduler

    _wrap_timed(omlx_scheduler.Scheduler, "step", "scheduler.step")
    _wrap_timed(
        omlx_scheduler.Scheduler,
        "_process_batch_responses",
        "scheduler.process_batch_responses",
    )
    _wrap_timed(omlx_scheduler.Scheduler, "_cleanup_finished", "scheduler.cleanup_finished")
    _wrap_timed(omlx_scheduler.Scheduler, "_schedule_waiting", "scheduler.schedule_waiting")
    _wrap_timed(mlx_generate.BatchGenerator, "next_generated", "batch.next_generated")
    _wrap_timed(mlx_generate.BatchGenerator, "_next", "batch._next")
    _wrap_timed(mlx_generate.GenerationBatch, "next", "generation.next")
    _wrap_timed(mlx_generate.GenerationBatch, "_step", "generation._step")


def _ensure_profile_sink(limit: int) -> None:
    global _PROFILE_STATS, _PROFILE_LIMIT
    if _PROFILE_STATS is None:
        _PROFILE_STATS = defaultdict(list)
    if limit != 0:
        _PROFILE_LIMIT = limit


def _mx_sync(stream=None) -> None:
    try:
        if stream is None:
            mx.synchronize()
        else:
            mx.synchronize(stream)
    except TypeError:
        mx.synchronize()
    except Exception:
        pass


def _shape_label(value) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "unknown"
    try:
        return "x".join(str(int(x)) for x in shape)
    except Exception:
        return str(shape)


def _install_mtp_call_profile(limit: int, trace_stats: bool) -> None:
    global _MTP_STATS
    if limit == 0 and not trace_stats:
        return
    if trace_stats and _MTP_STATS is None:
        _MTP_STATS = defaultdict(int)
    mtp_bg = importlib.import_module("omlx.patches.mlx_lm_mtp.batch_generator")

    if getattr(mtp_bg, "_fusekit_mtp_call_profiled", False):
        return

    if limit == 0:
        original_mtp = mtp_bg._step_mtp
        original_cycle = mtp_bg._run_verify_cycle
        original_post_init = mtp_bg._post_init_mtp

        def counted_mtp(gen_batch, hidden_at_position, next_main_tok, prev_buf, stats=None):
            _record_mtp_stat("step_mtp_calls")
            return original_mtp(
                gen_batch,
                hidden_at_position,
                next_main_tok,
                prev_buf,
                stats=stats,
            )

        def counted_cycle(gen_batch, state):
            state_stats = getattr(state, "stats", None)
            before_cycles = _mtp_stat_int(state_stats, "cycles")
            before_accepts = _mtp_stat_int(state_stats, "accepts")
            before_rejects = _mtp_stat_int(state_stats, "rejects")
            before_queue = len(getattr(state, "queue", []) or [])
            out = original_cycle(gen_batch, state)
            after_cycles = _mtp_stat_int(state_stats, "cycles")
            after_accepts = _mtp_stat_int(state_stats, "accepts")
            after_rejects = _mtp_stat_int(state_stats, "rejects")
            cycle_delta = (
                max(0, after_cycles - before_cycles)
                if before_cycles is not None and after_cycles is not None
                else 1
            )
            accept_delta = (
                max(0, after_accepts - before_accepts)
                if before_accepts is not None and after_accepts is not None
                else 0
            )
            reject_delta = (
                max(0, after_rejects - before_rejects)
                if before_rejects is not None and after_rejects is not None
                else 0
            )
            after_queue = len(getattr(state, "queue", []) or [])
            _record_mtp_stat("verify_cycles", cycle_delta)
            _record_mtp_stat("accepts", accept_delta)
            _record_mtp_stat("rejects", reject_delta)
            _record_mtp_stat("verify_emitted_est", accept_delta * 2 + reject_delta)
            _record_mtp_stat("queue_delta", max(0, after_queue - before_queue))
            return out

        def counted_post_init(gen_batch):
            _record_mtp_stat("post_init_calls")
            return original_post_init(gen_batch)

        mtp_bg._step_mtp = counted_mtp
        mtp_bg._run_verify_cycle = counted_cycle
        mtp_bg._post_init_mtp = counted_post_init
        mtp_bg._fusekit_mtp_call_profiled = True
        return

    _ensure_profile_sink(limit)

    original_backbone = mtp_bg._call_backbone
    original_mtp = mtp_bg._step_mtp
    original_cycle = mtp_bg._run_verify_cycle
    original_post_init = mtp_bg._post_init_mtp

    def profiled_backbone(model, inputs, cache, n_confirmed=0):
        label = f"mtp.backbone.shape={_shape_label(inputs)}.confirmed={int(n_confirmed or 0)}"
        stream = mtp_bg._get_generation_stream()
        _mx_sync(stream)
        t0 = time.perf_counter()
        out = original_backbone(model, inputs, cache, n_confirmed=n_confirmed)
        if isinstance(out, tuple):
            try:
                mx.eval(*out[:2])
            except Exception:
                pass
        _mx_sync(stream)
        _record_profile(label, time.perf_counter() - t0)
        return out

    def profiled_mtp(gen_batch, hidden_at_position, next_main_tok, prev_buf, stats=None):
        _record_mtp_stat("step_mtp_calls")
        label = (
            f"mtp.head.hidden={_shape_label(hidden_at_position)}"
            f".next={_shape_label(next_main_tok)}"
        )
        stream = mtp_bg._get_generation_stream()
        _mx_sync(stream)
        t0 = time.perf_counter()
        out = original_mtp(gen_batch, hidden_at_position, next_main_tok, prev_buf, stats=stats)
        _mx_sync(stream)
        _record_profile(label, time.perf_counter() - t0)
        return out

    def profiled_cycle(gen_batch, state):
        state_stats = getattr(state, "stats", None)
        before_cycles = _mtp_stat_int(state_stats, "cycles")
        before_accepts = _mtp_stat_int(state_stats, "accepts")
        before_rejects = _mtp_stat_int(state_stats, "rejects")
        before_queue = len(getattr(state, "queue", []) or [])
        stream = mtp_bg._get_generation_stream()
        _mx_sync(stream)
        t0 = time.perf_counter()
        out = original_cycle(gen_batch, state)
        _mx_sync(stream)
        _record_profile("mtp.verify_cycle.total", time.perf_counter() - t0)
        after_cycles = _mtp_stat_int(state_stats, "cycles")
        after_accepts = _mtp_stat_int(state_stats, "accepts")
        after_rejects = _mtp_stat_int(state_stats, "rejects")
        cycle_delta = (
            max(0, after_cycles - before_cycles)
            if before_cycles is not None and after_cycles is not None
            else 1
        )
        accept_delta = (
            max(0, after_accepts - before_accepts)
            if before_accepts is not None and after_accepts is not None
            else 0
        )
        reject_delta = (
            max(0, after_rejects - before_rejects)
            if before_rejects is not None and after_rejects is not None
            else 0
        )
        after_queue = len(getattr(state, "queue", []) or [])
        _record_mtp_stat("verify_cycles", cycle_delta)
        _record_mtp_stat("accepts", accept_delta)
        _record_mtp_stat("rejects", reject_delta)
        _record_mtp_stat("verify_emitted_est", accept_delta * 2 + reject_delta)
        _record_mtp_stat("queue_delta", max(0, after_queue - before_queue))
        return out

    def profiled_post_init(gen_batch):
        _record_mtp_stat("post_init_calls")
        stream = mtp_bg._get_generation_stream()
        _mx_sync(stream)
        t0 = time.perf_counter()
        out = original_post_init(gen_batch)
        _mx_sync(stream)
        _record_profile("mtp.post_init.total", time.perf_counter() - t0)
        return out

    mtp_bg._call_backbone = profiled_backbone
    mtp_bg._step_mtp = profiled_mtp
    mtp_bg._run_verify_cycle = profiled_cycle
    mtp_bg._post_init_mtp = profiled_post_init
    mtp_bg._fusekit_mtp_call_profiled = True


def _scalar_debug(value) -> object:
    try:
        if hasattr(value, "item"):
            return int(value.item())
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return repr(value)


def _array_debug(value) -> object:
    try:
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        pass
    return _scalar_debug(value)


def _install_engine_debug(limit: int) -> None:
    if limit <= 0:
        return
    mlx_generate = importlib.import_module("mlx_lm.generate")
    cls = mlx_generate.GenerationBatch
    original = cls._step
    seen = {"n": 0}

    def wrapped(self, *args, **kwargs):
        if seen["n"] < limit:
            cache = getattr(self, "prompt_cache", None) or []
            cache0 = cache[0] if cache else None
            state_destinations = 0
            for sm in getattr(self, "state_machines", []) or []:
                states = getattr(sm, "_states", {})
                for _trie, destinations in states.values():
                    state_destinations += len(destinations)
            model = getattr(self, "model", None)
            inner = getattr(model, "_fk_original", None)
            fk = getattr(inner, "_fk_lm", None)
            model_type = (
                f"{type(model).__module__}.{type(model).__name__}"
                if model is not None
                else None
            )
            inner_type = (
                f"{type(inner).__module__}.{type(inner).__name__}"
                if inner is not None
                else None
            )
            language_model = getattr(model, "_language_model", None) or getattr(
                model, "language_model", None
            )
            language_type = (
                f"{type(language_model).__module__}.{type(language_model).__name__}"
                if language_model is not None
                else None
            )
            samplers = getattr(self, "samplers", []) or []
            sampler_temps = [
                getattr(s, "temp", None) if s is not None else None for s in samplers
            ]
            fallback = getattr(self, "fallback_sampler", None)
            print(
                "[engine-debug] "
                f"step={seen['n']} "
                f"uids={len(getattr(self, 'uids', []) or [])} "
                f"skip_logprobs={getattr(self, '_fk_skip_logprobs', None)} "
                f"sampler_temps={sampler_temps} "
                f"fallback_temp={getattr(fallback, 'temp', None)} "
                f"logits_any={any(getattr(self, 'logits_processors', []) or [])} "
                f"state_destinations={state_destinations} "
                f"num_tokens={getattr(self, '_num_tokens', None)} "
                f"max_tokens={getattr(self, 'max_tokens', None)} "
                f"next_tokens={_array_debug(getattr(self, '_next_tokens', None))} "
                f"cache0={type(cache0).__name__ if cache0 is not None else None} "
                f"cache0_offset={_scalar_debug(getattr(cache0, 'offset', None)) if cache0 is not None else None} "
                f"cache0_idx={_scalar_debug(getattr(cache0, '_idx', None)) if cache0 is not None else None} "
                f"has_fk={fk is not None} "
                f"model_type={model_type} "
                f"inner_type={inner_type} "
                f"language_type={language_type}",
                flush=True,
            )
            seen["n"] += 1
        return original(self, *args, **kwargs)

    cls._step = wrapped


def _print_profile() -> None:
    if not _PROFILE_STATS:
        return
    for name in sorted(_PROFILE_STATS):
        values = _PROFILE_STATS[name]
        if not values:
            continue
        us = [v * 1_000_000.0 for v in values]
        p50 = statistics.median(us)
        print(
            f"[engine-profile] {name} "
            f"count={len(us)} "
            f"avg_us={statistics.mean(us):.2f} "
            f"p50_us={p50:.2f} "
            f"min_us={min(us):.2f} "
            f"max_us={max(us):.2f} "
            f"total_ms={sum(us) / 1000.0:.2f}",
            flush=True,
        )


def _clear_mtp_stats() -> None:
    if _MTP_STATS is not None:
        _MTP_STATS.clear()
    try:
        import fusekit_runtime

        stats = getattr(fusekit_runtime, "_OMLX_MTP_FASTPATH_STATS", None)
        if isinstance(stats, dict):
            stats.clear()
    except Exception:
        pass


def _print_mtp_stats() -> None:
    if not _MTP_STATS:
        return
    cycles = int(_MTP_STATS.get("verify_cycles", 0) or 0)
    accepts = int(_MTP_STATS.get("accepts", 0) or 0)
    rejects = int(_MTP_STATS.get("rejects", 0) or 0)
    emitted = int(_MTP_STATS.get("verify_emitted_est", 0) or 0)
    accept_rate = (100.0 * accepts / cycles) if cycles else 0.0
    emitted_per_cycle = (emitted / cycles) if cycles else 0.0
    print(
        "[engine-mtp-stats] "
        f"verify_cycles={cycles} "
        f"accepts={accepts} "
        f"rejects={rejects} "
        f"accept_rate={accept_rate:.1f}% "
        f"verify_emitted_est={emitted} "
        f"emitted_per_cycle={emitted_per_cycle:.3f} "
        f"queue_delta={int(_MTP_STATS.get('queue_delta', 0) or 0)} "
        f"step_mtp_calls={int(_MTP_STATS.get('step_mtp_calls', 0) or 0)} "
        f"post_init_calls={int(_MTP_STATS.get('post_init_calls', 0) or 0)}",
        flush=True,
    )


def _model_label(path: str) -> str:
    p = Path(path)
    try:
        idx = p.parts.index("models")
        return "/".join(p.parts[idx + 1 :])
    except ValueError:
        return str(p)


def _set_no_eos(engine) -> None:
    """Keep benchmark length fixed; EOS would hide decode cost."""
    tokenizer = engine.tokenizer
    for attr, value in (
        ("eos_token_ids", set()),
        ("eos_token_id", None),
    ):
        try:
            setattr(tokenizer, attr, value)
        except Exception:
            pass
    try:
        scheduler = engine._engine.engine.scheduler
        scheduler._generation_config_eos = set()
        scheduler._output_parser_factory = None
        scheduler._get_stop_tokens = lambda: set()
    except Exception:
        pass


async def run_model(args: argparse.Namespace) -> int:
    if args.mode == "engine_raw":
        os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
        os.environ["FUSEKIT_DISABLE_FAST"] = "1"
    else:
        os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
        os.environ.pop("FUSEKIT_DISABLE_FAST", None)
    if args.trace_mtp_stats:
        os.environ["FUSEKIT_TRACE_MTP_STATS"] = "1"
    print(
        f"[engine:env] mode={args.mode} "
        f"hook_disable={os.environ.get('FUSEKIT_DISABLE_HOOK', '')!r} "
        f"dispatch_disable={os.environ.get('FUSEKIT_DISABLE_DISPATCH', '')!r} "
        f"fast_disable={os.environ.get('FUSEKIT_DISABLE_FAST', '')!r}",
        flush=True,
    )

    _install_profile(args.profile_steps)
    _install_engine_debug(args.debug_engine_fast)
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)
    prompt = args.prompt
    if args.prompt_repeat > 1:
        prompt = ((args.prompt.rstrip() + " ") * args.prompt_repeat).strip()
    cfg = SchedulerConfig(max_num_seqs=1)
    engine = BatchedEngine(
        model_name=args.model,
        scheduler_config=cfg,
        stream_interval=1,
        model_settings=ModelSettings(mtp_enabled=True) if args.mtp else None,
    )

    load_t0 = time.perf_counter()
    await engine.start()
    mx.synchronize()
    load_s = time.perf_counter() - load_t0
    _set_no_eos(engine)
    _install_mtp_call_profile(args.profile_mtp_calls, args.trace_mtp_stats)

    print(
        f"[engine] model={_model_label(args.model)} "
        f"mode={args.mode} mtp={args.mtp} load_s={load_s:.3f}",
        flush=True,
    )

    rows: list[dict[str, float | int | str]] = []
    total = args.warmup + args.repeat
    try:
        for rep in range(total):
            if args.profile_run_only and rep == args.warmup and _PROFILE_STATS is not None:
                _PROFILE_STATS.clear()
                _clear_mtp_stats()
            try:
                mx.reset_peak_memory()
            except Exception:
                pass

            started = time.perf_counter()
            last = None
            async for last in engine.stream_generate(
                prompt,
                max_tokens=args.tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                stop=[],
                seed=args.seed,
            ):
                pass
            mx.synchronize()
            wall_s = time.perf_counter() - started

            generated = int(last.completion_tokens if last is not None else 0)
            prompt_tokens = int(last.prompt_tokens if last is not None else 0)
            peak_gb = 0.0
            try:
                peak_gb = float(mx.get_peak_memory()) / 1e9
            except Exception:
                pass
            wall_tps = generated / wall_s if wall_s and generated else 0.0

            kind = "warmup" if rep < args.warmup else "run"
            run_idx = rep - args.warmup
            print(
                f"[engine:{kind}:{run_idx if kind == 'run' else rep}] "
                f"prompt_tokens={prompt_tokens} "
                f"generated_tokens={generated} "
                f"wall_s={wall_s:.6f} "
                f"wall_tps={wall_tps:.3f} "
                f"peak_gb={peak_gb:.3f}",
                flush=True,
            )
            if args.dump_text:
                text = getattr(last, "text", "") if last is not None else ""
                print(
                    f"[engine:{kind}:{run_idx if kind == 'run' else rep}] "
                    f"text={text!r}",
                    flush=True,
                )

            if kind == "run":
                rows.append(
                    {
                        "generated": generated,
                        "wall_s": wall_s,
                        "wall_tps": wall_tps,
                        "peak_gb": peak_gb,
                    }
                )
    finally:
        await engine.stop()

    if rows:
        tps = [float(r["wall_tps"]) for r in rows]
        wall = [float(r["wall_s"]) for r in rows]
        peak = [float(r["peak_gb"]) for r in rows]
        print(
            f"[engine:avg] repeat={len(rows)} "
            f"wall_tps_avg={statistics.mean(tps):.3f} "
            f"wall_tps_stdev={(statistics.stdev(tps) if len(tps) > 1 else 0.0):.3f} "
            f"wall_s_avg={statistics.mean(wall):.6f} "
            f"peak_gb_avg={statistics.mean(peak):.3f}",
            flush=True,
        )
    _print_profile()
    _print_mtp_stats()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--mode", choices=("engine_raw", "engine_fusekit"), required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--prompt",
        default="Repeat the word token forever: token token token",
    )
    p.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="repeat --prompt N times before running the engine",
    )
    p.add_argument("--dump-text", action="store_true")
    p.add_argument("--mtp", action="store_true", help="enable oMLX native MTP")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed both MLX and Python RNG before generation",
    )
    p.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="collect timing for up to N scheduler/generator calls (-1 = all)",
    )
    p.add_argument(
        "--profile-mtp-calls",
        type=int,
        default=0,
        help="sync and time oMLX MTP backbone/verify/head calls (-1 = all)",
    )
    p.add_argument(
        "--trace-mtp-stats",
        action="store_true",
        help="print MTP accept/reject and verify-cycle counters",
    )
    p.add_argument(
        "--profile-run-only",
        action="store_true",
        help="clear profile counters after warmup so only measured runs are printed",
    )
    p.add_argument(
        "--debug-engine-fast",
        type=int,
        default=0,
        help="print GenerationBatch fast-path eligibility state for first N calls",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="configure Python logging level, e.g. INFO for oMLX MTP stats",
    )
    args = p.parse_args()
    if args.log_level:
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    return asyncio.run(run_model(args))


if __name__ == "__main__":
    raise SystemExit(main())
