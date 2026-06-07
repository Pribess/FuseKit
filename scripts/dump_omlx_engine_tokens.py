"""Dump generated token IDs from the oMLX BatchedEngine path.

Raw vs FuseKit must be selected by the parent process environment because
FUSEKIT_DISABLE_HOOK is read during Python startup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import Any

import mlx.core as mx

from omlx.engine.batched import BatchedEngine
from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig


_STEPS: list[dict[str, Any]] = []


def _scalar(value) -> int | None:
    try:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)
    except Exception:
        return None


def _array_list(value) -> list[int] | Any:
    try:
        if hasattr(value, "tolist"):
            out = value.tolist()
            if out and isinstance(out[0], list):
                return [int(x) for row in out for x in row]
            if isinstance(out, list):
                return [int(x) for x in out]
            return [int(out)]
    except Exception:
        pass
    scalar = _scalar(value)
    return [] if scalar is None else [scalar]


def _cache_offsets(cache) -> list[int | None]:
    out = []
    for c in cache or []:
        if hasattr(c, "_idx"):
            out.append(_scalar(getattr(c, "_idx")))
        elif hasattr(c, "offset"):
            out.append(_scalar(getattr(c, "offset")))
        else:
            out.append(None)
    return out[:8]


def _install_mtp_trace(limit: int) -> None:
    if limit <= 0:
        return
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp_bg

    if getattr(mtp_bg, "_fk_cycle_dump_installed", False):
        return

    original_post_init = mtp_bg._post_init_mtp
    original_call_backbone = mtp_bg._call_backbone
    original_step_mtp = mtp_bg._step_mtp
    original_run_cycle = mtp_bg._run_verify_cycle
    counters = {"backbone": 0, "cycle": 0, "mtp": 0}

    def traced_post_init(gen_batch):
        out = original_post_init(gen_batch)
        state = getattr(gen_batch, "_omlx_mtp_state", None)
        if state is not None:
            queue = [(int(tok), src) for tok, _lp, src in list(state.queue)]
            print(
                "[mtp-trace-post-init] "
                f"queue={queue} next={_array_list(state.next_main)} "
                f"draft={_array_list(state.draft_tok)} draft_id={state.draft_id} "
                f"offsets={_cache_offsets(getattr(gen_batch, 'prompt_cache', None))}",
                flush=True,
            )
        return out

    def traced_call_backbone(model, inputs, cache, n_confirmed=0):
        out = original_call_backbone(model, inputs, cache, n_confirmed=n_confirmed)
        if n_confirmed and counters["backbone"] < limit:
            logits, hidden, gdn_states = out
            try:
                mx.eval(inputs, logits, hidden)
                verify = int(mx.argmax(logits[:, 0, :], axis=-1).item())
                bonus = int(mx.argmax(logits[:, 1, :], axis=-1).item())
            except Exception:
                verify = -1
                bonus = -1
            print(
                "[mtp-trace-backbone] "
                f"i={counters['backbone']} inputs={_array_list(inputs)} "
                f"n_confirmed={n_confirmed} verify={verify} bonus={bonus} "
                f"hidden_shape={tuple(getattr(hidden, 'shape', ()))} "
                f"gdn={gdn_states is not None} offsets={_cache_offsets(cache)}",
                flush=True,
            )
            counters["backbone"] += 1
        return out

    def traced_step_mtp(gen_batch, hidden_at_position, next_main_tok, prev_buf, stats=None):
        out = original_step_mtp(
            gen_batch,
            hidden_at_position,
            next_main_tok,
            prev_buf,
            stats=stats,
        )
        if counters["mtp"] < limit:
            draft_tok, _draft_lp = out
            print(
                "[mtp-trace-head] "
                f"i={counters['mtp']} next={_array_list(next_main_tok)} "
                f"draft={_array_list(draft_tok)} "
                f"hidden_shape={tuple(getattr(hidden_at_position, 'shape', ()))}",
                flush=True,
            )
            counters["mtp"] += 1
        return out

    def traced_run_cycle(gen_batch, state):
        idx = counters["cycle"]
        before_accepts = int(getattr(state.stats, "accepts", 0))
        before_rejects = int(getattr(state.stats, "rejects", 0))
        before_next = _array_list(state.next_main)
        before_draft = _array_list(state.draft_tok)
        before_draft_id = int(getattr(state, "draft_id", -1))
        out = original_run_cycle(gen_batch, state)
        if idx < limit:
            after_accepts = int(getattr(state.stats, "accepts", 0))
            after_rejects = int(getattr(state.stats, "rejects", 0))
            queue = [(int(tok), src) for tok, _lp, src in list(state.queue)]
            print(
                "[mtp-trace-cycle] "
                f"i={idx} before_next={before_next} before_draft={before_draft} "
                f"before_draft_id={before_draft_id} "
                f"accepted={after_accepts > before_accepts} "
                f"rejected={after_rejects > before_rejects} "
                f"queue={queue} after_next={_array_list(state.next_main)} "
                f"after_draft={_array_list(state.draft_tok)} "
                f"after_draft_id={state.draft_id} "
                f"offsets={_cache_offsets(getattr(gen_batch, 'prompt_cache', None))}",
                flush=True,
            )
        counters["cycle"] += 1
        return out

    mtp_bg._post_init_mtp = traced_post_init
    mtp_bg._call_backbone = traced_call_backbone
    mtp_bg._step_mtp = traced_step_mtp
    mtp_bg._run_verify_cycle = traced_run_cycle
    mtp_bg._fk_cycle_dump_installed = True


def _install_token_capture() -> None:
    from omlx.scheduler import Scheduler

    original = Scheduler._process_batch_responses
    if getattr(original, "_fk_token_capture", False):
        return

    def wrapped(self, responses):
        outputs, finished_ids = original(self, responses)
        for out in outputs:
            _STEPS.append(
                {
                    "new": [int(x) for x in (out.new_token_ids or [])],
                    "output": [int(x) for x in (out.output_token_ids or [])],
                    "completion": int(out.completion_tokens or 0),
                    "finished": bool(out.finished),
                    "finish_reason": out.finish_reason,
                    "text_tail": str(out.output_text or "")[-120:],
                }
            )
        return outputs, finished_ids

    wrapped._fk_token_capture = True
    wrapped._fk_original = original
    Scheduler._process_batch_responses = wrapped


def _set_no_eos(engine) -> None:
    tokenizer = engine.tokenizer
    for attr, value in (("eos_token_ids", set()), ("eos_token_id", None)):
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


async def run(args: argparse.Namespace) -> int:
    _install_token_capture()
    _install_mtp_trace(args.trace_mtp_cycles)
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    prompt = args.prompt
    if args.prompt_repeat > 1:
        prompt = ((args.prompt.rstrip() + " ") * args.prompt_repeat).strip()

    engine = BatchedEngine(
        model_name=args.model,
        scheduler_config=SchedulerConfig(max_num_seqs=1),
        stream_interval=1,
        model_settings=ModelSettings(mtp_enabled=True) if args.mtp else None,
    )
    await engine.start()
    _set_no_eos(engine)
    mx.synchronize()

    started = time.perf_counter()
    last = None
    try:
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
    finally:
        await engine.stop()

    wall_s = time.perf_counter() - started
    final_tokens = _STEPS[-1]["output"] if _STEPS else []
    result = {
        "mode": args.mode,
        "model": args.model,
        "prompt": prompt,
        "mtp": bool(args.mtp),
        "seed": args.seed,
        "requested_tokens": int(args.tokens),
        "generated_tokens": int(last.completion_tokens if last is not None else 0),
        "wall_s": wall_s,
        "wall_tps": (len(final_tokens) / wall_s) if wall_s and final_tokens else 0.0,
        "tokens": final_tokens,
        "text": getattr(last, "text", "") if last is not None else "",
    }
    print("[engine-token-dump] " + json.dumps(result, ensure_ascii=False), flush=True)
    if args.dump_steps:
        for idx, step in enumerate(_STEPS):
            print(
                "[engine-token-step] "
                + json.dumps({"i": idx, **step}, ensure_ascii=False),
                flush=True,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode", default="unknown")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--dump-steps", action="store_true")
    parser.add_argument("--trace-mtp-cycles", type=int, default=0)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
