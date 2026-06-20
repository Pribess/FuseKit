"""fusekit_runtime — mlx-lm side of the fusekit megakernel.

Import side effects:
  Installs a `sys.meta_path` finder that monkey-patches `mlx_lm.utils`'s
  `load()` whenever it's first imported, wrapping the loaded model with
  a `FusekitWrapper` if a `fusekit/` subdir exists next to the weights.

This module is imported automatically at interpreter startup through a
`.pth` file in user site-packages. Importing it manually is harmless —
the finder install is idempotent.

When a compiled artifact is present, `FusekitWrapper` routes supported
prefill/decode work through the embedded Rust/Metal dispatcher and falls
back to mlx-lm only when the runtime cannot prove the fast path safe.
Set `FUSEKIT_DISABLE_HOOK=1` before Python starts to force raw mlx-lm.
"""

from __future__ import annotations

import ctypes
import importlib.abc
import importlib.machinery
import json
import os
import sys
import atexit
import warnings
from pathlib import Path
from typing import Any

__all__ = ["attach", "FusekitWrapper", "install_hook", "array_data_ptr", "array_data_ptrs"]


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in ("", "0", "false", "no", "off")


def _dispatch_enabled() -> bool:
    if _env_flag("FUSEKIT_DISABLE_DISPATCH") or _env_flag("FUSEKIT_DISABLE_FAST"):
        return False
    return True


def _gemma4_layer_index_from_weight_key(key: str) -> int | None:
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


def _install_gemma4_shared_kv_weight_filter() -> None:
    """Allow Gemma4 shared-KV checkpoints with harmless extra K/V tensors.

    Some Gemma4 E*B MLX exports carry K/V tensors for layers that the config
    marks as KV-shared. The mlx-lm model correctly omits those modules, so
    strict loading rejects the extra tensors before raw or FuseKit generation
    can run. Preserve the config semantics and filter only those shared-layer
    K/V tensors during the model's normal sanitize stage.
    """

    try:
        from mlx_lm.models import gemma4_text
    except Exception:
        return

    model_cls = getattr(gemma4_text, "Model", None)
    original = getattr(model_cls, "sanitize", None)
    if model_cls is None or original is None:
        return
    if getattr(original, "_fusekit_gemma4_shared_kv_filter", False):
        return

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

        out = {}
        dropped = 0
        for key, value in sanitized.items():
            layer_idx = _gemma4_layer_index_from_weight_key(key)
            is_shared_kv_extra = (
                layer_idx is not None
                and layer_idx >= first_shared
                and (
                    ".self_attn.k_proj." in key
                    or ".self_attn.v_proj." in key
                    or ".self_attn.k_norm." in key
                )
            )
            if is_shared_kv_extra:
                dropped += 1
                continue
            out[key] = value
        if dropped and _env_flag("FUSEKIT_GEMMA4_LOAD_DEBUG"):
            print(
                f"[fusekit] Gemma4 shared-KV load: ignored {dropped} extra tensors",
                file=sys.stderr,
            )
        return out

    sanitize._fusekit_gemma4_shared_kv_filter = True
    model_cls.sanitize = sanitize


def _prefill_pad_multiple() -> int:
    if _env_flag("FUSEKIT_DISABLE_PADDED_PREFILL"):
        return 1
    raw = os.environ.get("FUSEKIT_PREFILL_PAD_MULTIPLE", "32")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 32
    return max(1, min(value, 256))


def _cache_supports_padded_prefill(cache: Any) -> bool:
    """True when appended pad rows can be removed by cache.trim().

    This is intentionally cache-semantics based. Plain KV caches expose a
    visible offset and `trim(n)` removes the appended rows from future attention.
    Recurrent/stateful caches such as ArraysCache cannot be made correct by
    trimming only KV visibility, so they stay on the normal unpadded path.
    """
    if not cache:
        return False
    allowed = {
        "KVCache",
        "QuantizedKVCache",
        "SimpleKVCache",
        "RotatingKVCache",
        "BatchKVCache",
    }
    for c in cache:
        if c is None or not callable(getattr(c, "trim", None)):
            return False
        if type(c).__name__ not in allowed:
            return False
    return True


def _model_uses_quantized_linear(model: Any) -> bool:
    cached = getattr(model, "_fk_uses_quantized_linear", None)
    if cached is not None:
        return bool(cached)
    root = getattr(model, "_fk_original", model)
    seen: set[int] = set()

    def visit(value: Any, depth: int) -> bool:
        if depth > 8:
            return False
        obj_id = id(value)
        if obj_id in seen:
            return False
        seen.add(obj_id)
        if type(value).__name__ == "QuantizedLinear":
            return True
        if isinstance(value, dict):
            return any(visit(v, depth + 1) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(visit(v, depth + 1) for v in value)
        attrs = getattr(value, "__dict__", None)
        if not attrs:
            return False
        for child in attrs.values():
            if isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            if visit(child, depth + 1):
                return True
        return False

    result = visit(root, 0)
    try:
        model.__dict__["_fk_uses_quantized_linear"] = result
    except Exception:
        pass
    if root is not model:
        try:
            root.__dict__["_fk_uses_quantized_linear"] = result
        except Exception:
            pass
    return result


def _pad_prefill_tokens_if_safe(tokens: Any, cache: Any, model: Any | None = None) -> tuple[Any, int]:
    multiple = _prefill_pad_multiple()
    if multiple <= 1 or not _cache_supports_padded_prefill(cache):
        return tokens, 0
    if (
        model is not None
        and not _env_flag("FUSEKIT_FORCE_PADDED_PREFILL")
        and not _model_uses_quantized_linear(model)
    ):
        return tokens, 0
    try:
        n = int(tokens.shape[0])
    except Exception:
        return tokens, 0
    padded = ((n + multiple - 1) // multiple) * multiple
    extra = padded - n
    if extra <= 0:
        return tokens, 0
    import mlx.core as mx

    pad = mx.zeros((extra,), dtype=tokens.dtype)
    return mx.concatenate([tokens, pad], axis=0), extra


def _trim_prompt_cache_in_place(cache: Any, n: int) -> None:
    if n <= 0:
        return
    for c in cache or []:
        trim = getattr(c, "trim", None)
        if callable(trim):
            trim(n)


class _FusekitSeededTokenList(list):
    """One-token prompt segment carrying a precomputed first generation token.

    oMLX external prefill normally returns the final prompt token; mlx-lm's
    GenerationBatch constructor processes it once and silently primes the first
    generated token.  When FuseKit has already run full-prompt prefill and
    sampled that first generated token, the segment must carry metadata so the
    first GenerationBatch.next() can emit it instead of dropping it.
    """


def _get_attr_path(root: Any, path: str) -> Any | None:
    cur = root
    if not path:
        return cur
    for seg in path.split("."):
        try:
            cur = getattr(cur, seg)
        except Exception:
            return None
    return cur


def _full_prefill_components(model: Any) -> tuple[Any, Any] | None:
    """Find a semantic transformer backbone plus final projection.

    mlx-lm outer language models commonly implement:

        hidden = backbone(tokens, cache)
        logits = lm_head(hidden) or embed_tokens.as_linear(hidden)

    During generation we only need logits for the final real prompt row. This
    helper discovers that structure by module capabilities, not model names, so
    it works for Qwen2/Qwen3-style architectures and falls back for anything
    that does not expose the same contract.
    """
    original = getattr(model, "_fk_original", model)
    if original is None:
        return None

    backbone = None
    for path in (
        "model",
        "language_model.model",
        "language_model.model.model",
        "text_model.model",
    ):
        candidate = _get_attr_path(original, path)
        if candidate is not None and candidate is not original and callable(candidate):
            backbone = candidate
            break
    if backbone is None:
        return None

    holders = [
        original,
        _get_attr_path(original, "language_model"),
        _get_attr_path(original, "model"),
        _get_attr_path(original, "text_model"),
        backbone,
    ]
    tie_word_embeddings = None
    for holder in holders:
        args = getattr(holder, "args", None)
        if args is None:
            continue
        value = getattr(args, "tie_word_embeddings", None)
        if value is not None:
            tie_word_embeddings = bool(value)
            break

    if tie_word_embeddings is not False:
        for holder in (backbone, *holders):
            embed = getattr(holder, "embed_tokens", None)
            as_linear = getattr(embed, "as_linear", None)
            if callable(as_linear):
                return backbone, as_linear
        if tie_word_embeddings is True:
            return None

    for holder in holders:
        lm_head = getattr(holder, "lm_head", None)
        if callable(lm_head):
            return backbone, lm_head
    return None


def _pad_full_prefill_tokens_if_safe(tokens: Any, cache: Any) -> tuple[Any, int]:
    multiple = _prefill_pad_multiple()
    if multiple <= 1 or not _cache_supports_padded_prefill(cache):
        return tokens, 0
    try:
        n = int(tokens.shape[0])
    except Exception:
        return tokens, 0
    padded = ((n + multiple - 1) // multiple) * multiple
    extra = padded - n
    if extra <= 0:
        return tokens, 0
    import mlx.core as mx

    return mx.concatenate([tokens, mx.zeros((extra,), dtype=tokens.dtype)], axis=0), extra


def _try_full_prefill_last_logits(model: Any, tokens: Any, cache: Any) -> Any | None:
    if _env_flag("FUSEKIT_DISABLE_FULL_PREFILL_LAST_LOGITS"):
        return None
    if not _cache_supports_padded_prefill(cache):
        return None
    try:
        real_len = int(tokens.shape[0])
    except Exception:
        return None
    if real_len <= 1:
        return None
    components = _full_prefill_components(model)
    if components is None:
        return None
    backbone, projector = components
    run_tokens, pad_extra = _pad_full_prefill_tokens_if_safe(tokens, cache)
    try:
        hidden = backbone(run_tokens[None], cache=cache)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if getattr(hidden, "shape", None) is None or len(hidden.shape) < 3:
            raise RuntimeError("fusekit full prefill backbone returned invalid hidden state")
        last_hidden = hidden[:, real_len - 1:real_len, :]
        return projector(last_hidden)
    finally:
        _trim_prompt_cache_in_place(cache, pad_extra)


def _sampling_seed_replay_safe(request: Any) -> bool:
    params = getattr(request, "sampling_params", None)
    if params is None:
        return True
    seed = getattr(params, "seed", None)
    if seed is None:
        return True
    # A later oMLX scheduler block re-applies the request seed immediately
    # before BatchGenerator.insert(). For stochastic samplers that would replay
    # the RNG and make the second-token distribution wrong. Greedy is seed-free.
    temp = getattr(params, "temperature", None)
    try:
        return float(temp if temp is not None else 0.0) == 0.0
    except Exception:
        return False


def _engine_full_prefill_seed_candidate(
    scheduler: Any,
    request: Any,
    tokens: list[int],
    existing_cache: Any,
    vlm_embeds: Any,
    stats: dict[str, int] | None = None,
) -> tuple[Any, list[int]] | None:
    """Run oMLX external prefill as a full-prompt prefill when safe.

    This mirrors mlxcel's shape/runtime idea: process all prompt tokens in one
    backbone pass, optionally pad to a tile multiple, trim the cache, project
    only the last real hidden row, and seed GenerationBatch with the first
    generated token.  The path is deliberately semantic/contract based, not
    model-name based.
    """
    def miss(reason: str) -> None:
        if stats is not None:
            stats["engine_full_prefill_seed_miss"] = (
                stats.get("engine_full_prefill_seed_miss", 0) + 1
            )
            key = f"engine_full_prefill_seed_miss_{reason}"
            stats[key] = stats.get(key, 0) + 1
        if os.environ.get("FUSEKIT_DBG_ENGINE_PREFILL_SEED"):
            print(
                f"[fusekit-engine-prefill-seed] miss={reason}",
                file=sys.stderr,
                flush=True,
            )

    if stats is not None:
        stats["engine_full_prefill_seed_attempt"] = (
            stats.get("engine_full_prefill_seed_attempt", 0) + 1
        )

    if _env_flag("FUSEKIT_DISABLE_ENGINE_FULL_PREFILL_SEED"):
        miss("disabled")
        return None
    if vlm_embeds is not None or existing_cache is not None:
        miss("stateful_request")
        return None
    if not tokens or len(tokens) <= 1:
        miss("short_tokens")
        return None
    if getattr(request, "specprefill_indices", None) is not None:
        miss("specprefill")
        return None
    if getattr(request, "vlm_inputs_embeds", None) is not None:
        miss("vlm")
        return None
    if not _sampling_seed_replay_safe(request):
        miss("sampler_seed")
        return None

    # Keep the first implementation on the same safe surface as the existing
    # benchmark path: one row at a time, no SSD boundary snapshot bookkeeping,
    # no prefill hard-limit loop, and no in-flight prompt/generation merge.
    config = getattr(scheduler, "config", None)
    if getattr(config, "paged_cache_block_size", 0) and getattr(scheduler, "block_aware_cache", None) is not None:
        miss("paged_cache")
        return None
    if getattr(scheduler, "_memory_limit_bytes", 0) or getattr(scheduler, "_memory_hard_limit_bytes", 0):
        miss("memory_limit")
        return None
    bg = getattr(scheduler, "batch_generator", None)
    if bg is not None:
        try:
            if len(getattr(bg, "_prompt_batch", [])) or len(getattr(bg, "_generation_batch", [])):
                miss("active_batches")
                return None
        except Exception:
            miss("batch_state")
            return None
        if int(getattr(bg, "prefill_batch_size", 1) or 1) != 1:
            miss("prefill_batch_size")
            return None

    try:
        sampler, logits_processors = scheduler._build_sampler_and_processors(
            request.sampling_params, request
        )
    except Exception:
        miss("sampler_build")
        return None
    if logits_processors:
        miss("logits_processors")
        return None

    try:
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.core as mx

        prompt_cache = make_prompt_cache(scheduler.model)
        if not _cache_supports_padded_prefill(prompt_cache):
            miss("cache_type")
            return None
        if hasattr(scheduler.model, "clear_vlm_position_state"):
            scheduler.model.clear_vlm_position_state()
        token_arr = mx.array(tokens)
        logits = _try_full_prefill_last_logits(scheduler.model, token_arr, prompt_cache)
        if logits is None:
            miss("last_logits")
            return None
        if getattr(logits, "shape", None) is None:
            miss("logits_shape")
            return None
        if len(logits.shape) == 3:
            logits_2d = logits[:, -1, :]
        elif len(logits.shape) == 2:
            logits_2d = logits
        else:
            miss("logits_rank")
            return None
        logprobs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
        first = sampler(logprobs)
        mx.eval(first, logprobs, [c.state for c in prompt_cache])
        first_token = int(first.item())
    except Exception:
        miss("exception")
        return None

    seeded = _FusekitSeededTokenList([first_token])
    seeded._fusekit_seeded_first_token = first_token
    try:
        seeded._fusekit_seeded_first_logprobs = logprobs[0]
    except Exception:
        seeded._fusekit_seeded_first_logprobs = None
    return prompt_cache, seeded


def _mlx_generate_kwargs_are_greedy(kw: dict[str, Any]) -> bool:
    """True only when mlx-lm's public generate_step contract is argmax.

    FuseKit's direct-head path writes the winning token id in the GPU
    graph. That is semantically valid for greedy generation, but not
    for arbitrary sampler/logits-processor combinations. Keep this
    decision at the runtime call boundary; the model graph itself does
    not encode sampling policy.
    """
    if kw.get("logits_processors"):
        return False
    sampler = kw.get("sampler")
    if sampler is None:
        return True
    return _sampler_is_greedy_argmax(sampler, missing_is_greedy=False)


def _sampler_temperature(sampler: Any) -> float | None:
    if sampler is None:
        return None
    for name in ("temp", "temperature"):
        value = getattr(sampler, name, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            return None
    return None


def _sampler_is_greedy_argmax(sampler: Any, *, missing_is_greedy: bool) -> bool:
    temp = _sampler_temperature(sampler)
    if temp is None:
        return missing_is_greedy
    return temp == 0.0


def _direct_head_argmax_scope(model: Any, enabled: bool = True):
    inner = getattr(model, "_fk_original", None)
    prev_model = getattr(model, "_fk_allow_direct_head_argmax", None)
    prev_inner = getattr(inner, "_fk_allow_direct_head_argmax", None) if inner is not None else None
    if enabled:
        try:
            model.__dict__["_fk_allow_direct_head_argmax"] = True
        except Exception:
            pass
        if inner is not None:
            try:
                inner.__dict__["_fk_allow_direct_head_argmax"] = True
            except Exception:
                pass

    def restore() -> None:
        try:
            if prev_model is None:
                model.__dict__.pop("_fk_allow_direct_head_argmax", None)
            else:
                model.__dict__["_fk_allow_direct_head_argmax"] = prev_model
        except Exception:
            pass
        if inner is not None:
            try:
                if prev_inner is None:
                    inner.__dict__.pop("_fk_allow_direct_head_argmax", None)
                else:
                    inner.__dict__["_fk_allow_direct_head_argmax"] = prev_inner
            except Exception:
                pass

    return restore


_OMLX_MTP_GREEDY_PATCHED = False
_OMLX_MTP_GREEDY_FASTPATH_PATCHED = False
_OMLX_MTP_EMIT_LOGPROBS_PATCHED = False
_OMLX_MTP_SAMPLER_ACCEPT_CACHE_PATCHED = False
_OMLX_MTPLX_CONTRACT_PATCHED_CLASSES: set[type] = set()
_OMLX_MTP_STATS_PRINTER_REGISTERED = False
_OMLX_MTP_FASTPATH_STATS: dict[str, int] = {}
_SAMPLE_UTILS_PATCHED = False


def _mtp_fastpath_stat(name: str, delta: int = 1) -> None:
    if not _env_flag("FUSEKIT_TRACE_MTP_STATS"):
        return
    _OMLX_MTP_FASTPATH_STATS[name] = _OMLX_MTP_FASTPATH_STATS.get(name, 0) + int(delta)


def _register_mtp_fastpath_stats_printer() -> None:
    global _OMLX_MTP_STATS_PRINTER_REGISTERED
    if _OMLX_MTP_STATS_PRINTER_REGISTERED or not _env_flag("FUSEKIT_TRACE_MTP_STATS"):
        return

    def _dump() -> None:
        stats = _OMLX_MTP_FASTPATH_STATS
        if not stats:
            return
        cycles = int(stats.get("verify_cycles", 0) or 0)
        accepts = int(stats.get("accepts", 0) or 0)
        rejects = int(stats.get("rejects", 0) or 0)
        emitted = int(stats.get("verify_emitted_est", 0) or 0)
        accept_rate = (100.0 * accepts / cycles) if cycles else 0.0
        emitted_per_cycle = (emitted / cycles) if cycles else 0.0
        print(
            "[fusekit-mtp-stats] "
            f"verify_cycles={cycles} "
            f"accepts={accepts} "
            f"rejects={rejects} "
            f"accept_rate={accept_rate:.1f}% "
            f"verify_emitted_est={emitted} "
            f"emitted_per_cycle={emitted_per_cycle:.3f} "
            f"margin_fallbacks={int(stats.get('margin_fallbacks', 0) or 0)} "
            f"bonus_margin_gated={int(stats.get('bonus_margin_gated', 0) or 0)} "
            f"step_mtp_calls={int(stats.get('step_mtp_calls', 0) or 0)} "
            f"post_init_calls={int(stats.get('post_init_calls', 0) or 0)} "
            f"sampler_cache_hits={int(stats.get('sampler_cache_hits', 0) or 0)} "
            f"sampler_cache_misses={int(stats.get('sampler_cache_misses', 0) or 0)}",
            file=sys.stderr,
        )

    atexit.register(_dump)
    _OMLX_MTP_STATS_PRINTER_REGISTERED = True


def _wrap_sample_utils(module) -> None:
    """Tag mlx-lm sampler callables with their semantic parameters.

    `mlx_lm.sample_utils.make_sampler(temp=0)` returns an anonymous
    argmax lambda with no attributes, so the runtime cannot distinguish
    it from an arbitrary custom sampler. Attaching the construction
    parameters lets FuseKit safely keep the direct-head argmax fast path
    for semantic-greedy sampling while still rejecting non-greedy
    temperature/top-k/top-p samplers.
    """
    global _SAMPLE_UTILS_PATCHED
    if _SAMPLE_UTILS_PATCHED or not hasattr(module, "make_sampler"):
        return
    original_make_sampler = module.make_sampler

    def _patched_make_sampler(*args, **kwargs):
        sampler = original_make_sampler(*args, **kwargs)
        temp = kwargs.get("temp", args[0] if len(args) > 0 else 0.0)
        top_p = kwargs.get("top_p", args[1] if len(args) > 1 else 0.0)
        min_p = kwargs.get("min_p", args[2] if len(args) > 2 else 0.0)
        min_keep = kwargs.get("min_tokens_to_keep", args[3] if len(args) > 3 else 1)
        top_k = kwargs.get("top_k", args[4] if len(args) > 4 else 0)
        try:
            sampler.temp = float(temp)
            sampler.temperature = float(temp)
            sampler.top_p = float(top_p)
            sampler.min_p = float(min_p)
            sampler.min_tokens_to_keep = int(min_keep)
            sampler.top_k = int(top_k)
            sampler._fk_mlx_lm_sampler = True
        except Exception:
            pass
        return sampler

    _patched_make_sampler._fk_original = original_make_sampler
    module.make_sampler = _patched_make_sampler
    _SAMPLE_UTILS_PATCHED = True


def _register_dispatch_stats_printer(wrapper: Any) -> None:
    if not _env_flag("FUSEKIT_TRACE_DISPATCH_STATS"):
        return

    def _dump() -> None:
        stats = getattr(wrapper, "_fk_dispatch_stats", None)
        if not stats:
            return
        calls = int(stats.get("calls", 0) or 0)
        if calls == 0:
            return
        dispatch_hits = int(stats.get("dispatch_hit", 0) or 0)
        mtp_verify = int(stats.get("mtp_verify", 0) or 0)
        variant_hits = int(stats.get("mtp_variant_hit", 0) or 0)
        split_hits = int(stats.get("mtp_split_hit", 0) or 0)
        native = int(stats.get("native_passthrough", 0) or 0)
        hit_pct = (100.0 * dispatch_hits / calls) if calls else 0.0
        mtp_variant_pct = (100.0 * variant_hits / mtp_verify) if mtp_verify else 0.0
        mtp_split_pct = (100.0 * split_hits / mtp_verify) if mtp_verify else 0.0
        native_pct = (100.0 * native / calls) if calls else 0.0
        print(
            "[fusekit-dispatch-stats] "
            f"calls={calls} dispatch_hit={dispatch_hits}({hit_pct:.1f}%) "
            f"eligible={int(stats.get('eligible', 0) or 0)} "
            f"native_passthrough={native}({native_pct:.1f}%) "
            f"single_decode={int(stats.get('single_decode', 0) or 0)} "
            f"single_decode_hit={int(stats.get('single_decode_hit', 0) or 0)} "
            f"prefill_variant={int(stats.get('prefill_variant', 0) or 0)} "
            f"prefill_variant_hit={int(stats.get('prefill_variant_hit', 0) or 0)} "
            f"prefill_variant_missing={int(stats.get('prefill_variant_missing', 0) or 0)} "
            f"prefill_variant_no_output={int(stats.get('prefill_variant_no_output', 0) or 0)} "
            f"prefill_variant_chunk_call={int(stats.get('prefill_variant_chunk_call', 0) or 0)} "
            f"prefill_variant_chunk_attempt={int(stats.get('prefill_variant_chunk_attempt', 0) or 0)} "
            f"prefill_variant_chunk={int(stats.get('prefill_variant_chunk', 0) or 0)} "
            f"prefill_variant_chunk_hit={int(stats.get('prefill_variant_chunk_hit', 0) or 0)} "
            f"prefill_variant_chunk_missing={int(stats.get('prefill_variant_chunk_missing', 0) or 0)} "
            f"prefill_variant_chunk_no_storage={int(stats.get('prefill_variant_chunk_no_storage', 0) or 0)} "
            f"prefill_variant_chunk_no_output={int(stats.get('prefill_variant_chunk_no_output', 0) or 0)} "
            f"prefill_variant_stateful_disabled={int(stats.get('prefill_variant_stateful_disabled', 0) or 0)} "
            f"engine_full_prefill_seed_attempt={int(stats.get('engine_full_prefill_seed_attempt', 0) or 0)} "
            f"engine_full_prefill_seed={int(stats.get('engine_full_prefill_seed', 0) or 0)} "
            f"engine_full_prefill_seed_miss={int(stats.get('engine_full_prefill_seed_miss', 0) or 0)} "
            f"native_prefill_no_logits={int(stats.get('native_prefill_no_logits', 0) or 0)} "
            f"native_prefill_last_logits={int(stats.get('native_prefill_last_logits', 0) or 0)} "
            f"mtp_verify={mtp_verify} "
            f"mtp_variant_attempt={int(stats.get('mtp_variant_attempt', 0) or 0)} "
            f"mtp_variant_hit={variant_hits}({mtp_variant_pct:.1f}%) "
            f"mtp_variant_missing={int(stats.get('mtp_variant_missing', 0) or 0)} "
            f"mtp_variant_no_output={int(stats.get('mtp_variant_no_output', 0) or 0)} "
            f"mtp_split_attempt={int(stats.get('mtp_split_attempt', 0) or 0)} "
            f"mtp_split_hit={split_hits}({mtp_split_pct:.1f}%) "
            f"dispatch_none={int(stats.get('dispatch_none', 0) or 0)} "
            f"mtp_native_passthrough={int(stats.get('mtp_native_passthrough', 0) or 0)} "
            f"dispatch_exception={int(stats.get('dispatch_exception', 0) or 0)}",
            file=sys.stderr,
        )

    atexit.register(_dump)


def _patch_omlx_mtp_greedy_detection() -> None:
    """Optionally reinterpret temp=0 samplers as greedy for MTP.

    Raw oMLX treats a per-request sampler object as non-greedy even when
    its temp field is zero. Some model samplers still produce tokens that
    differ from raw argmax, so the default FuseKit path must preserve the
    original oMLX acceptance semantics. The temp=0 rewrite stays available
    only as an explicit speed experiment.
    """
    global _OMLX_MTP_GREEDY_PATCHED
    if _env_flag("FUSEKIT_DISABLE_MTP_GREEDY_FIX"):
        return
    if not _env_flag("FUSEKIT_ENABLE_MTP_TEMP0_GREEDY_FIX"):
        _patch_omlx_mtp_emit_logprobs_drop()
        _patch_omlx_mtp_sampler_accept_cache()
        _patch_omlx_mtp_greedy_fastpath()
        return
    if _OMLX_MTP_GREEDY_PATCHED:
        _patch_omlx_mtp_emit_logprobs_drop()
        _patch_omlx_mtp_sampler_accept_cache()
        _patch_omlx_mtp_greedy_fastpath()
        return
    try:
        from omlx.patches.mlx_lm_mtp import batch_generator as bg
    except Exception:
        return

    if getattr(bg, "_fk_greedy_patch", False):
        _OMLX_MTP_GREEDY_PATCHED = True
        _patch_omlx_mtp_emit_logprobs_drop(bg)
        _patch_omlx_mtp_sampler_accept_cache(bg)
        _patch_omlx_mtp_greedy_fastpath(bg)
        return
    original = getattr(bg, "_is_greedy", None)
    if not callable(original):
        return

    def _sampler_is_temp0(sampler: Any, *, missing_is_greedy: bool) -> bool:
        if sampler is None:
            return True
        temp = getattr(sampler, "temp", None)
        if temp is None:
            temp = getattr(sampler, "temperature", None)
        if temp is None:
            return missing_is_greedy
        try:
            return float(temp) == 0.0
        except (TypeError, ValueError):
            return False

    def _patched_is_greedy(gen_batch: Any) -> bool:
        try:
            samplers = getattr(gen_batch, "samplers", []) or []
            if samplers and samplers[0] is not None:
                return _sampler_is_temp0(samplers[0], missing_is_greedy=False)
            fallback = getattr(gen_batch, "fallback_sampler", None)
            return _sampler_is_temp0(fallback, missing_is_greedy=True)
        except Exception:
            return original(gen_batch)

    _patched_is_greedy.__name__ = getattr(original, "__name__", "_is_greedy")
    _patched_is_greedy.__doc__ = getattr(original, "__doc__", None)
    _patched_is_greedy._fk_original = original
    bg._is_greedy = _patched_is_greedy
    bg._fk_greedy_patch = True
    _OMLX_MTP_GREEDY_PATCHED = True
    _patch_omlx_mtp_emit_logprobs_drop(bg)
    _patch_omlx_mtp_sampler_accept_cache(bg)
    _patch_omlx_mtp_greedy_fastpath(bg)


def _patch_omlx_mtp_emit_logprobs_drop(bg: Any | None = None) -> None:
    """Avoid carrying full-vocab MTP response logprobs when oMLX will drop them.

    Native MTP still keeps draft/verify distributions in its private state for
    stochastic acceptance and residual sampling. This only strips the response
    payload for requests where Scheduler already marked logprobs as unneeded.
    """
    global _OMLX_MTP_EMIT_LOGPROBS_PATCHED
    if _OMLX_MTP_EMIT_LOGPROBS_PATCHED or _env_flag("FUSEKIT_DISABLE_MTP_DROP_RESPONSE_LOGPROBS"):
        return
    if bg is None:
        try:
            from omlx.patches.mlx_lm_mtp import batch_generator as bg
        except Exception:
            return
    original = getattr(bg, "_emit_response", None)
    if not callable(original) or getattr(bg, "_fk_emit_logprobs_drop_patch", False):
        _OMLX_MTP_EMIT_LOGPROBS_PATCHED = True
        return

    def _patched_emit_response(gen_batch: Any, token_id: int, logprobs_1d: Any, stats: Any = None):
        skip_logprobs = getattr(gen_batch, "_fk_skip_logprobs", False) or getattr(
            bg, "_fk_skip_logprobs_current_step", False
        )
        if skip_logprobs:
            logprobs_1d = None
        return original(gen_batch, token_id, logprobs_1d, stats)

    _patched_emit_response._fk_original = original
    bg._emit_response = _patched_emit_response
    bg._fk_emit_logprobs_drop_patch = True
    _OMLX_MTP_EMIT_LOGPROBS_PATCHED = True


def _patch_omlx_mtp_sampler_accept_cache(bg: Any | None = None, scheduler_module: Any | None = None) -> None:
    """Reuse stochastic sampler filtering for native-MTP acceptance math.

    Native oMLX samples from a filtered distribution, then immediately rebuilds
    the same filtered+temperature-scaled logprob tensor for MTP acceptance. For
    top-p this means a second full-vocab sort per verify/head sample. The patch
    keeps the sampler contract intact, but stores a tiny id-keyed cache of the
    accept-logprob expression that `_accept_lp_for` can reuse for the same MLX
    array object.
    """
    global _OMLX_MTP_SAMPLER_ACCEPT_CACHE_PATCHED
    if _env_flag("FUSEKIT_DISABLE_MTP_SAMPLER_ACCEPT_CACHE"):
        return
    try:
        import mlx.core as mx
        from omlx.utils import sampling as sm
    except Exception:
        return
    if bg is None:
        try:
            from omlx.patches.mlx_lm_mtp import batch_generator as bg
        except Exception:
            bg = None

    original_make_sampler = getattr(sm, "make_sampler", None)
    if callable(original_make_sampler) and not getattr(sm, "_fk_accept_cache_make_sampler_patch", False):
        param_names = (
            "temp",
            "top_p",
            "min_p",
            "min_tokens_to_keep",
            "top_k",
            "xtc_probability",
            "xtc_threshold",
            "xtc_special_tokens",
        )

        def _patched_make_sampler(*args, **kwargs):
            params = {
                "temp": 0.0,
                "top_p": 0.0,
                "min_p": 0.0,
                "min_tokens_to_keep": 1,
                "top_k": 0,
                "xtc_probability": 0.0,
            }
            for name, value in zip(param_names, args):
                params[name] = value
            params.update(kwargs)

            try:
                temp = float(params.get("temp", 0.0) or 0.0)
                top_p = float(params.get("top_p", 0.0) or 0.0)
                min_p = float(params.get("min_p", 0.0) or 0.0)
                min_keep = int(params.get("min_tokens_to_keep", 1) or 1)
                top_k = int(params.get("top_k", 0) or 0)
                xtc_probability = float(params.get("xtc_probability", 0.0) or 0.0)
            except Exception:
                return original_make_sampler(*args, **kwargs)

            # XTC intentionally stays on oMLX's raw path: `_accept_lp_for`
            # does not model its random mask today, so caching it would change
            # native acceptance semantics.
            if temp == 0.0 or xtc_probability > 0.0:
                return original_make_sampler(*args, **kwargs)

            def _cached_sampler(logprobs):
                out = logprobs
                if 0.0 < top_p < 1.0:
                    out = sm.apply_top_p(out, top_p)
                if min_p != 0.0:
                    out = sm.apply_min_p(out, min_p, min_keep)
                if top_k > 0:
                    out = sm.apply_top_k(out, top_k)

                scaled = out * (1.0 / temp)
                cache = list(getattr(_cached_sampler, "_fk_accept_lp_cache", ()) or ())
                cache.append((id(logprobs), getattr(logprobs, "shape", None), scaled))
                _cached_sampler._fk_accept_lp_cache = cache[-4:]
                return mx.random.categorical(scaled)

            _cached_sampler.temp = temp
            _cached_sampler.top_p = top_p
            _cached_sampler.min_p = min_p
            _cached_sampler.top_k = top_k
            _cached_sampler.min_tokens_to_keep = min_keep
            _cached_sampler._fk_accept_cache_sampler = True
            _cached_sampler.__name__ = "fusekit_cached_sampler"
            return _cached_sampler

        _patched_make_sampler._fk_original = original_make_sampler
        sm.make_sampler = _patched_make_sampler
        sm._fk_accept_cache_make_sampler_patch = True

    if scheduler_module is None:
        scheduler_module = sys.modules.get(_OMLX_SCHEDULER_MODULE)
    if scheduler_module is not None:
        try:
            scheduler_module.omlx_make_sampler = sm.make_sampler
        except Exception:
            pass

    if bg is not None and not getattr(bg, "_fk_accept_cache_accept_patch", False):
        original_accept = getattr(bg, "_accept_lp_for", None)
        original_residual = getattr(bg, "_residual_sample", None)
        if callable(original_accept):
            class _AcceptLpView:
                _fk_accept_lp_view = True

                def __init__(self, scaled):
                    self.scaled = scaled
                    self._logz = None
                    self._full = None

                def _fk_logz(self):
                    if self._logz is None:
                        self._logz = mx.logsumexp(
                            self.scaled, axis=-1, keepdims=True
                        )
                    return self._logz

                def _fk_full_accept_lp(self):
                    if self._full is None:
                        self._full = self.scaled - self._fk_logz()
                    return self._full

                def squeeze(self, axis=None):
                    return _AcceptLpView(self.scaled.squeeze(axis))

                def __getitem__(self, idx):
                    shape = getattr(self.scaled, "shape", ())
                    if len(shape) == 1 and isinstance(idx, int):
                        return self.scaled[idx] - self._fk_logz()[0]
                    if (
                        len(shape) == 2
                        and isinstance(idx, tuple)
                        and len(idx) == 2
                        and isinstance(idx[0], int)
                        and isinstance(idx[1], int)
                    ):
                        return self.scaled[idx] - self._fk_logz()[idx[0], 0]
                    return self._fk_full_accept_lp()[idx]

            def _patched_accept_lp_for(sampler, lp):
                cache = getattr(sampler, "_fk_accept_lp_cache", None)
                if cache:
                    key = (id(lp), getattr(lp, "shape", None))
                    for cached_id, cached_shape, cached_scaled in reversed(cache):
                        if key == (cached_id, cached_shape):
                            _mtp_fastpath_stat("sampler_cache_hits")
                            if not _env_flag("FUSEKIT_DISABLE_MTP_LAZY_ACCEPT_LP"):
                                return _AcceptLpView(cached_scaled)
                            return cached_scaled - mx.logsumexp(
                                cached_scaled, axis=-1, keepdims=True
                            )
                _mtp_fastpath_stat("sampler_cache_misses")
                return original_accept(sampler, lp)

            _patched_accept_lp_for._fk_original = original_accept
            bg._accept_lp_for = _patched_accept_lp_for
            if callable(original_residual) and not getattr(bg, "_fk_lazy_accept_residual_patch", False):
                def _patched_residual_sample(verify_lp_2d, draft_lp_1d):
                    if getattr(verify_lp_2d, "_fk_accept_lp_view", False):
                        verify_lp_2d = verify_lp_2d._fk_full_accept_lp()
                    if getattr(draft_lp_1d, "_fk_accept_lp_view", False):
                        draft_lp_1d = draft_lp_1d._fk_full_accept_lp()
                    return original_residual(verify_lp_2d, draft_lp_1d)

                _patched_residual_sample._fk_original = original_residual
                bg._residual_sample = _patched_residual_sample
                bg._fk_lazy_accept_residual_patch = True
            bg._fk_accept_cache_accept_patch = True

    _OMLX_MTP_SAMPLER_ACCEPT_CACHE_PATCHED = True


def _patch_omlx_mtp_greedy_fastpath(bg: Any | None = None) -> None:
    """Skip MTP logprob materialization when the engine will discard it.

    oMLX's MTP loop computes full vocab logprobs even for temp=0 requests,
    then Scheduler drops them when the API request did not ask for logprobs.
    The fast path below is semantic rather than model-pattern based:

      * active MTP batch
      * greedy sampler
      * no logits processors
      * Scheduler marked logprobs as unneeded

    For every other case, including stochastic sampling and requested
    logprobs, the original oMLX MTP implementation remains the source of
    truth.
    """
    global _OMLX_MTP_GREEDY_FASTPATH_PATCHED
    if _OMLX_MTP_GREEDY_FASTPATH_PATCHED or _env_flag("FUSEKIT_DISABLE_MTP_GREEDY_FASTPATH"):
        return
    if bg is None:
        try:
            from omlx.patches.mlx_lm_mtp import batch_generator as bg
        except Exception:
            return
    if getattr(bg, "_fk_greedy_fastpath_patch", False):
        _OMLX_MTP_GREEDY_FASTPATH_PATCHED = True
        return

    original_post_init = getattr(bg, "_post_init_mtp", None)
    original_run_verify = getattr(bg, "_run_verify_cycle", None)
    original_step_mtp = getattr(bg, "_step_mtp", None)
    if not (callable(original_post_init) and callable(original_run_verify) and callable(original_step_mtp)):
        return
    _register_mtp_fastpath_stats_printer()

    fastpath_debug = {"n": 0}

    def _debug_fastpath_miss(reason: str) -> None:
        if not os.environ.get("FUSEKIT_DBG_MTP_FASTPATH"):
            return
        n = fastpath_debug["n"]
        if n >= 12:
            return
        fastpath_debug["n"] = n + 1
        print(f"[fusekit-mtp-fastpath] miss reason={reason}", file=sys.stderr, flush=True)

    def _fast_eligible(gen_batch: Any) -> bool:
        if not _dispatch_enabled():
            _debug_fastpath_miss("dispatch_disabled")
            return False
        skip_logprobs = getattr(gen_batch, "_fk_skip_logprobs", False) or getattr(
            bg, "_fk_skip_logprobs_current_step", False
        )
        if not skip_logprobs:
            _debug_fastpath_miss(
                f"skip_logprobs_false gb={getattr(gen_batch, '_fk_skip_logprobs', None)!r} "
                f"current={getattr(bg, '_fk_skip_logprobs_current_step', None)!r}"
            )
            return False
        try:
            greedy = bool(bg._is_greedy(gen_batch))
            procs = bg._proc_list(gen_batch)
            if not greedy:
                _debug_fastpath_miss("not_greedy")
                return False
            if procs is not None:
                _debug_fastpath_miss("has_logits_processors")
                return False
            return True
        except Exception:
            _debug_fastpath_miss("eligibility_exception")
            return False

    def _argmax_token(logits_2d: Any):
        import mlx.core as mx

        return mx.argmax(logits_2d, axis=-1).astype(mx.uint32)

    def _verify_margin_threshold() -> float:
        raw = os.environ.get("FUSEKIT_MTP_VERIFY_MARGIN")
        if raw is None:
            return 0.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0

    def _bonus_margin_threshold() -> float:
        raw = os.environ.get("FUSEKIT_MTP_BONUS_MARGIN")
        if raw is None:
            return 0.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0

    def _top2_margin(logits_2d: Any) -> float:
        import mlx.core as mx

        vals = mx.topk(logits_2d.astype(mx.float32), k=2, axis=-1)
        mx.eval(vals)
        return float(mx.abs(vals[:, 1] - vals[:, 0]).item())

    def _run_original_verify_with_native_backbone(gen_batch: Any, state: Any) -> None:
        prev_mn = os.environ.get("FUSEKIT_DISABLE_MTP_MN_VARIANT")
        prev_dispatch = os.environ.get("FUSEKIT_DISABLE_DISPATCH")
        os.environ["FUSEKIT_DISABLE_MTP_MN_VARIANT"] = "1"
        os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
        try:
            return original_run_verify(gen_batch, state)
        finally:
            if prev_mn is None:
                os.environ.pop("FUSEKIT_DISABLE_MTP_MN_VARIANT", None)
            else:
                os.environ["FUSEKIT_DISABLE_MTP_MN_VARIANT"] = prev_mn
            if prev_dispatch is None:
                os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
            else:
                os.environ["FUSEKIT_DISABLE_DISPATCH"] = prev_dispatch

    def _clone_mx_array(value: Any) -> Any:
        if value is None or not hasattr(value, "shape"):
            return value
        import mlx.core as mx

        return value + mx.zeros_like(value)

    def _mtp_mn_reject_repair_enabled(model: Any) -> bool:
        if _env_flag("FUSEKIT_DISABLE_MTP_REJECT_REPAIR"):
            return False
        return _env_flag("FUSEKIT_ENABLE_MTP_REJECT_REPAIR")

    def _snapshot_mtp_reject_repair(cache: Any):
        import mlx.core as mx

        snapshots = []
        arrays = []
        for idx, c in enumerate(cache or []):
            if type(c).__name__ == "ArraysCache":
                cloned = []
                for arr in getattr(c, "cache", []):
                    clone = _clone_mx_array(arr)
                    cloned.append(clone)
                    if hasattr(clone, "shape"):
                        arrays.append(clone)
                lengths = _clone_mx_array(getattr(c, "lengths", None))
                left_padding = _clone_mx_array(getattr(c, "left_padding", None))
                if hasattr(lengths, "shape"):
                    arrays.append(lengths)
                if hasattr(left_padding, "shape"):
                    arrays.append(left_padding)
                snapshots.append(("arrays", idx, cloned, lengths, left_padding))
                continue
            if hasattr(c, "offset"):
                try:
                    off = getattr(c, "offset")
                    off = int(off.item()) if hasattr(off, "item") else int(off)
                    snapshots.append(("offset", idx, off))
                except Exception:
                    pass
            elif hasattr(c, "_idx"):
                try:
                    snapshots.append(("idx", idx, int(getattr(c, "_idx"))))
                except Exception:
                    pass
        if arrays:
            mx.eval(*arrays)
        return snapshots

    def _restore_mtp_reject_repair(cache: Any, snapshots: Any) -> bool:
        try:
            for snap in snapshots or []:
                kind = snap[0]
                idx = snap[1]
                c = cache[idx]
                if kind == "arrays":
                    _, _, cloned, lengths, left_padding = snap
                    c.cache = list(cloned)
                    c.lengths = lengths
                    c.left_padding = left_padding
                    if hasattr(c, "rollback_state"):
                        c.rollback_state = None
                elif kind == "offset":
                    _, _, off = snap
                    c.offset = int(off)
                elif kind == "idx":
                    _, _, off = snap
                    setattr(c, "_idx", int(off))
            return True
        except Exception:
            return False

    def _repair_rejected_mtp_state(gen_batch: Any, snapshots: Any, verify_tok: Any):
        import mlx.core as mx

        if not _restore_mtp_reject_repair(gen_batch.prompt_cache, snapshots):
            if os.environ.get("FUSEKIT_DBG_MTP_REPAIR"):
                print("[fusekit-mtp-repair] restore_failed", file=sys.stderr, flush=True)
            return None
        prev = os.environ.get("FUSEKIT_DISABLE_MTP_MN_VARIANT")
        os.environ["FUSEKIT_DISABLE_MTP_MN_VARIANT"] = "1"
        try:
            repaired = bg._call_backbone(
                gen_batch.model,
                bg._ensure_uint32(verify_tok).reshape(1, 1),
                gen_batch.prompt_cache,
            )
            logits0, hidden0, _ = repaired
            mx.eval(logits0, hidden0)
            bg._clear_rollback(gen_batch.prompt_cache)
            _mtp_fastpath_stat("reject_repairs")
            if os.environ.get("FUSEKIT_DBG_MTP_REPAIR"):
                print("[fusekit-mtp-repair] repaired_reject", file=sys.stderr, flush=True)
            return hidden0[:, -1:, :]
        finally:
            if prev is None:
                os.environ.pop("FUSEKIT_DISABLE_MTP_MN_VARIANT", None)
            else:
                os.environ["FUSEKIT_DISABLE_MTP_MN_VARIANT"] = prev

    def _defer_draft_sync_enabled() -> bool:
        return (
            _env_flag("FUSEKIT_ENABLE_MTP_DEFER_DRAFT_SYNC")
            and not _env_flag("FUSEKIT_DISABLE_MTP_DEFER_DRAFT_SYNC")
        )

    def _draft_id(state: Any) -> int:
        draft_id = int(getattr(state, "draft_id", -1) or -1)
        if draft_id >= 0:
            return draft_id
        if getattr(state, "draft_tok", None) is None:
            return -1
        import time

        t0 = time.perf_counter()
        draft_id = int(state.draft_tok.tolist()[0])
        state.draft_id = draft_id
        try:
            state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
        except Exception:
            pass
        return draft_id

    def _patched_step_mtp(
        gen_batch: Any,
        hidden_at_position: Any,
        next_main_tok: Any,
        prev_buf: Any,
        stats: Any = None,
    ):
        if not _fast_eligible(gen_batch):
            return original_step_mtp(gen_batch, hidden_at_position, next_main_tok, prev_buf, stats)

        import time
        import mlx.core as mx

        _mtp_fastpath_stat("step_mtp_calls")
        state = gen_batch._omlx_mtp_state
        t0 = time.perf_counter()
        next_ids = next_main_tok.reshape(1, 1)
        with mx.stream(bg._get_generation_stream()):
            mtp_logits = gen_batch.model.mtp_forward(
                hidden_at_position, next_ids, state.mtp_cache
            )
            mtp_logits_2d = mtp_logits[:, -1, :]
        new_tok = _argmax_token(mtp_logits_2d)
        if _defer_draft_sync_enabled():
            mx.eval(new_tok)
            state.draft_id = -1
        else:
            draft_id_int = int(new_tok.tolist()[0])
            state.draft_id = draft_id_int
        state.draft_accept_lp = None
        if stats is not None:
            stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
        return bg._ensure_uint32(new_tok), None

    def _patched_post_init(gen_batch: Any) -> None:
        if not _fast_eligible(gen_batch):
            return original_post_init(gen_batch)

        import mlx.core as mx

        _mtp_fastpath_stat("post_init_calls")
        if gen_batch._next_tokens is None or not gen_batch.uids:
            return

        main_tok = bg._ensure_uint32(gen_batch._next_tokens)
        with mx.stream(bg._get_generation_stream()):
            logits, hidden, _ = bg._call_backbone(
                gen_batch.model, main_tok[:, None], gen_batch.prompt_cache
            )
        next_main_tok = _argmax_token(logits[:, -1, :])

        mtp_cache = gen_batch.model.make_mtp_cache()
        hidden_at_main = hidden[:, -1:, :]
        next_ids = next_main_tok.reshape(1, 1)
        with mx.stream(bg._get_generation_stream()):
            mtp_logits = gen_batch.model.mtp_forward(hidden_at_main, next_ids, mtp_cache)
        draft_tok = _argmax_token(mtp_logits[:, -1, :])
        mx.eval(main_tok, next_main_tok, draft_tok)

        state = bg._MtpState()
        state.mtp_cache = mtp_cache
        state.next_main = bg._ensure_uint32(next_main_tok)
        state.draft_tok = bg._ensure_uint32(draft_tok)
        state.draft_lp = None
        state.draft_accept_lp = None
        state.draft_id = -1 if _defer_draft_sync_enabled() else int(draft_tok.tolist()[0])
        state.queue.append((int(main_tok.tolist()[0]), None, "init"))
        state.queue.append((int(next_main_tok.tolist()[0]), None, "init"))
        gen_batch._omlx_mtp_state = state

    def _patched_run_verify(gen_batch: Any, state: Any) -> None:
        if not _fast_eligible(gen_batch):
            return original_run_verify(gen_batch, state)
        if state.next_main is None or state.draft_tok is None:
            raise bg._MtpStepFallback("verify cycle entered without next_main / draft")

        import time
        import mlx.core as mx

        inputs = mx.concatenate([state.next_main, state.draft_tok])

        t0 = time.perf_counter()
        model = gen_batch.model
        margin_threshold = _verify_margin_threshold()
        bonus_gate_threshold = _bonus_margin_threshold()
        needs_pre_forward_snapshot = (
            _mtp_mn_reject_repair_enabled(model)
            or margin_threshold > 0.0
            or bonus_gate_threshold > 0.0
        )
        repair_snapshot = (
            _snapshot_mtp_reject_repair(gen_batch.prompt_cache)
            if needs_pre_forward_snapshot
            else None
        )
        inner = getattr(model, "_fk_original", None)
        prev_allow = getattr(model, "_fk_allow_direct_mtp_argmax", None)
        prev_inner_allow = (
            getattr(inner, "_fk_allow_direct_mtp_argmax", None)
            if inner is not None
            else None
        )
        try:
            try:
                model.__dict__["_fk_allow_direct_mtp_argmax"] = True
            except Exception:
                pass
            if inner is not None:
                try:
                    inner.__dict__["_fk_allow_direct_mtp_argmax"] = True
                except Exception:
                    pass
            with mx.stream(bg._get_generation_stream()):
                logits, hidden, gdn_states = bg._call_backbone(
                    model,
                    inputs[None, :],
                    gen_batch.prompt_cache,
                    n_confirmed=1,
                )
                verify_logits = logits[:, 0, :]
                bonus_logits = logits[:, 1, :]
        finally:
            try:
                if prev_allow is None:
                    model.__dict__.pop("_fk_allow_direct_mtp_argmax", None)
                else:
                    model.__dict__["_fk_allow_direct_mtp_argmax"] = prev_allow
            except Exception:
                pass
            if inner is not None:
                try:
                    if prev_inner_allow is None:
                        inner.__dict__.pop("_fk_allow_direct_mtp_argmax", None)
                    else:
                        inner.__dict__["_fk_allow_direct_mtp_argmax"] = prev_inner_allow
                except Exception:
                    pass
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bonus_margin = None
        if margin_threshold > 0.0:
            verify_margin = _top2_margin(verify_logits)
            bonus_margin = _top2_margin(bonus_logits)
            if verify_margin < margin_threshold or bonus_margin < margin_threshold:
                _mtp_fastpath_stat("margin_fallbacks")
                if repair_snapshot is not None:
                    if not _restore_mtp_reject_repair(
                        gen_batch.prompt_cache, repair_snapshot
                    ):
                        raise bg._MtpStepFallback(
                            "cache restore failed before margin fallback"
                        )
                else:
                    if not bg._rollback_after_reject(
                        gen_batch.model,
                        gen_batch.prompt_cache,
                        gdn_states,
                        accepted=0,
                        block_size=2,
                    ):
                        raise bg._MtpStepFallback(
                            "cache rollback failed before margin fallback"
                        )
                state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000
                return _run_original_verify_with_native_backbone(gen_batch, state)

        fk_tokens = getattr(gen_batch.model, "_fk_last_argmax_tokens", None)
        fk_shape = getattr(fk_tokens, "shape", ()) if fk_tokens is not None else ()
        split_tokens_trusted = bool(
            getattr(gen_batch.model, "_fk_mtp_split_tokens_trusted", False)
        )
        if (
            not _env_flag("FUSEKIT_DISABLE_MTP_GPU_ARGMAX_TOKENS")
            and (
                not getattr(gen_batch.model, "_fk_stateful_mtp_graph", False)
                or _env_flag("FUSEKIT_ENABLE_STATEFUL_MTP_DIRECT_ARGMAX")
                or split_tokens_trusted
            )
            and fk_tokens is not None
            and len(fk_shape) >= 1
            and int(fk_shape[0]) >= 2
        ):
            verify_tok = fk_tokens[0:1].astype(mx.uint32)
            bonus_tok = fk_tokens[1:2].astype(mx.uint32)
        else:
            verify_tok = _argmax_token(verify_logits)
            bonus_tok = _argmax_token(bonus_logits)
        mx.eval(verify_tok, bonus_tok)
        draft_id = _draft_id(state)
        verify_id = int(verify_tok.tolist()[0])
        bonus_id = int(bonus_tok.tolist()[0])
        accept = verify_id == draft_id
        state.stats.sample_ms += (time.perf_counter() - t0) * 1000

        hidden_at_confirmed = hidden[:, 0:1, :]
        hidden_at_draft = hidden[:, 1:2, :]

        state.stats.cycles += 1
        _mtp_fastpath_stat("verify_cycles")
        if accept:
            if bonus_gate_threshold > 0.0:
                if bonus_margin is None:
                    bonus_margin = _top2_margin(bonus_logits)
                if bonus_margin < bonus_gate_threshold:
                    _mtp_fastpath_stat("bonus_margin_gated")
                    t0 = time.perf_counter()
                    if repair_snapshot is not None:
                        if not _restore_mtp_reject_repair(
                            gen_batch.prompt_cache, repair_snapshot
                        ):
                            raise bg._MtpStepFallback(
                                "cache restore failed for bonus gate"
                            )
                    else:
                        if not bg._rollback_after_reject(
                            gen_batch.model,
                            gen_batch.prompt_cache,
                            gdn_states,
                            accepted=0,
                            block_size=2,
                        ):
                            raise bg._MtpStepFallback("cache rollback failed for bonus gate")
                    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000
                    return _run_original_verify_with_native_backbone(gen_batch, state)

            state.stats.accepts += 1
            _mtp_fastpath_stat("accepts")
            _mtp_fastpath_stat("verify_emitted_est", 2)
            t0 = time.perf_counter()
            bg._clear_rollback(gen_batch.prompt_cache)
            state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

            new_draft, new_draft_lp = bg._step_mtp(
                gen_batch,
                hidden_at_draft,
                bg._ensure_uint32(bonus_tok),
                prev_buf=None,
                stats=state.stats,
            )
            state.queue.append((draft_id, None, "draft"))
            state.queue.append((bonus_id, None, "bonus"))
            state.next_main = bg._ensure_uint32(bonus_tok)
            state.draft_tok = new_draft
            state.draft_lp = new_draft_lp
            return

        state.stats.rejects += 1
        _mtp_fastpath_stat("rejects")
        _mtp_fastpath_stat("verify_emitted_est")
        t0 = time.perf_counter()
        repaired_hidden = None
        if repair_snapshot is not None:
            repaired_hidden = _repair_rejected_mtp_state(
                gen_batch, repair_snapshot, bg._ensure_uint32(verify_tok)
            )
        if repaired_hidden is not None:
            hidden_at_confirmed = repaired_hidden
        else:
            if not bg._rollback_after_reject(
                gen_batch.model, gen_batch.prompt_cache, gdn_states, accepted=0, block_size=2
            ):
                raise bg._MtpStepFallback("cache layer rejects rollback")
        state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        emit_tok = mx.array([verify_id], dtype=mx.uint32)
        new_draft, new_draft_lp = bg._step_mtp(
            gen_batch,
            hidden_at_confirmed,
            emit_tok,
            prev_buf=None,
            stats=state.stats,
        )
        state.queue.append((verify_id, None, "verify"))
        state.next_main = emit_tok
        state.draft_tok = new_draft
        state.draft_lp = new_draft_lp

    bg._post_init_mtp = _patched_post_init
    bg._run_verify_cycle = _patched_run_verify
    bg._step_mtp = _patched_step_mtp
    bg._fk_greedy_fastpath_patch = True
    bg._fk_greedy_fastpath_originals = {
        "post_init": original_post_init,
        "run_verify": original_run_verify,
        "step_mtp": original_step_mtp,
    }
    if os.environ.get("FUSEKIT_DBG_MTP_FASTPATH"):
        print("[fusekit-mtp-fastpath] installed", file=sys.stderr, flush=True)
    _OMLX_MTP_GREEDY_FASTPATH_PATCHED = True


def _mtplx_contract_path(model: Any) -> Path | None:
    fk_dir = getattr(model, "_fk_dir", None)
    if fk_dir is not None:
        return Path(fk_dir).parent / "mtplx_runtime.json"

    inner = getattr(model, "_fk_original", model)
    for name in ("model_path", "model_dir", "path", "_model_path", "_path"):
        value = getattr(inner, name, None)
        if value:
            path = Path(value)
            if path.is_dir():
                return path / "mtplx_runtime.json"
    return None


def _load_mtplx_contract(model: Any) -> dict | None:
    path = _mtplx_contract_path(model)
    if path is None or not path.is_file():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    arch = str(data.get("arch_id", ""))
    if "mtplx_version" not in data and "mtp" not in arch.lower():
        return None
    return data


def _model_dir(model: Any) -> Path | None:
    fk_dir = getattr(model, "_fk_dir", None)
    if fk_dir is not None:
        path = Path(fk_dir)
        for cur in (path, *path.parents):
            if cur.name == "fusekit":
                return cur.parent
        return path

    inner = getattr(model, "_fk_original", model)
    for name in ("model_path", "model_dir", "path", "_model_path", "_path"):
        value = getattr(inner, name, None)
        if not value:
            continue
        path = Path(value)
        if path.is_dir():
            return path
    return None


def _read_json_file(path: Path) -> dict | None:
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _is_mtp_weight_key(key: str) -> bool:
    return key.startswith("mtp.") or ".mtp." in key


def _has_mtp_weight_keys(model_dir: Path) -> bool:
    index = _read_json_file(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if index else None
    if isinstance(weight_map, dict) and any(_is_mtp_weight_key(k) for k in weight_map):
        return True

    # Header-only scan. Safetensors starts with an 8-byte little-endian
    # JSON header length; no tensor payload is read.
    for path in sorted(model_dir.glob("*.safetensors")):
        try:
            with path.open("rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                if header_len <= 0 or header_len > 128 * 1024 * 1024:
                    continue
                header = json.loads(f.read(header_len))
        except Exception:
            continue
        if isinstance(header, dict) and any(_is_mtp_weight_key(k) for k in header):
            return True
    return False


def _text_config(config: dict) -> dict:
    text = config.get("text_config")
    if isinstance(text, dict):
        return text
    for value in config.values():
        if (
            isinstance(value, dict)
            and isinstance(value.get("model_type"), str)
            and value["model_type"].endswith("_text")
        ):
            return value
    return config


def _config_int(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _model_mtp_metadata(model: Any) -> dict:
    cached = getattr(model, "_fk_mtp_metadata", None)
    if isinstance(cached, dict):
        return cached

    info = {
        "format": "none",
        "declared_layers": 0,
        "max_depth": None,
        "has_weights": False,
        "source": "none",
    }
    model_dir = _model_dir(model)
    if model_dir is not None:
        config = _read_json_file(model_dir / "config.json") or {}
        text_cfg = _text_config(config)
        mtp_layers = _config_int(
            text_cfg.get("mtp_num_hidden_layers"), config.get("mtp_num_hidden_layers")
        )
        nextn_layers = _config_int(
            text_cfg.get("num_nextn_predict_layers"),
            config.get("num_nextn_predict_layers"),
        )
        contract = _load_mtplx_contract(model)
        if contract:
            extra = config.get("mlx_lm_extra_tensors") or {}
            sidecar = extra.get("mtp_file") if isinstance(extra, dict) else None
            has_sidecar = bool(sidecar and (model_dir / str(sidecar)).is_file())
            if not has_sidecar:
                has_sidecar = (model_dir / "mtp.safetensors").is_file()
            max_depth = _config_int(contract.get("mtp_depth_max"))
            info = {
                "format": "mtplx",
                "declared_layers": max(mtp_layers, nextn_layers),
                "max_depth": max_depth or None,
                "has_weights": has_sidecar or _has_mtp_weight_keys(model_dir),
                "source": "mtplx_runtime.json",
            }
        elif nextn_layers > 0:
            info = {
                "format": "next_n",
                "declared_layers": nextn_layers,
                "max_depth": nextn_layers,
                "has_weights": _has_mtp_weight_keys(model_dir),
                "source": "config.num_nextn_predict_layers",
            }
        elif mtp_layers > 0:
            info = {
                "format": "qwen_native",
                "declared_layers": mtp_layers,
                "max_depth": None,
                "has_weights": _has_mtp_weight_keys(model_dir),
                "source": "config.mtp_num_hidden_layers",
            }

    try:
        model.__dict__["_fk_mtp_metadata"] = info
    except Exception:
        pass
    return info


def _trace_mtp_metadata_once(model: Any, info: dict) -> None:
    if not os.environ.get("FUSEKIT_TRACE_MTP_DETECT"):
        return
    if getattr(model, "_fk_mtp_metadata_traced", False):
        return
    try:
        model.__dict__["_fk_mtp_metadata_traced"] = True
    except Exception:
        pass
    print(
        "[fusekit] MTP detect "
        f"format={info.get('format')} layers={info.get('declared_layers')} "
        f"max_depth={info.get('max_depth')} weights={info.get('has_weights')} "
        f"source={info.get('source')}",
        file=sys.stderr,
    )


def _iter_mtp_contract_targets(model: Any):
    inner = getattr(model, "_fk_original", model)
    stack = [inner]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        yield obj
        for name in ("_language_model", "language_model"):
            try:
                child = getattr(obj, name)
            except Exception:
                child = None
            if child is not None:
                stack.append(child)


def _post_norm_hidden_result(model: Any, result: Any) -> Any:
    if not isinstance(result, tuple) or len(result) < 2:
        return result
    hidden = result[1]
    if getattr(hidden, "dtype", None) is None:
        return result
    norm = getattr(getattr(model, "model", None), "norm", None)
    if not callable(norm):
        return result
    hidden = norm(hidden)
    if len(result) == 2:
        return result[0], hidden
    return (result[0], hidden) + tuple(result[2:])


def _patch_mtplx_hidden_contract_class(cls: type) -> None:
    if cls in _OMLX_MTPLX_CONTRACT_PATCHED_CLASSES:
        return
    original = getattr(cls, "__call__", None)
    if not callable(original):
        return

    def _patched_call(self, *args, **kwargs):
        return_hidden = bool(kwargs.get("return_hidden", False))
        result = original(self, *args, **kwargs)
        if (
            return_hidden
            and getattr(self, "_fk_mtplx_hidden_variant", None) == "post_norm"
            and not getattr(self, "_fk_mtplx_returns_post_norm", False)
        ):
            return _post_norm_hidden_result(self, result)
        return result

    _patched_call.__name__ = getattr(original, "__name__", "__call__")
    _patched_call.__doc__ = getattr(original, "__doc__", None)
    _patched_call._fk_original = original
    cls.__call__ = _patched_call
    _OMLX_MTPLX_CONTRACT_PATCHED_CLASSES.add(cls)


def _maybe_shift_mtplx_norm(module: Any, shifted: list[Any]) -> None:
    weight = getattr(module, "weight", None)
    if getattr(weight, "ndim", None) != 1:
        return
    try:
        import mlx.core as mx

        if float(mx.mean(weight).item()) < 0.5:
            module.weight = weight + 1.0
            shifted.append(module.weight)
    except Exception:
        return


def _shift_mtplx_mtp_norms(target: Any) -> int:
    if getattr(target, "_fk_mtplx_norms_shifted", False):
        return 0
    mtp = getattr(target, "mtp", None)
    if mtp is None:
        return 0

    shifted: list[Any] = []
    for name in ("pre_fc_norm_hidden", "pre_fc_norm_embedding", "norm"):
        _maybe_shift_mtplx_norm(getattr(mtp, name, None), shifted)

    for layer in getattr(mtp, "layers", []) or []:
        for name in ("input_layernorm", "post_attention_layernorm"):
            _maybe_shift_mtplx_norm(getattr(layer, name, None), shifted)
        attn = getattr(layer, "self_attn", None)
        for name in ("q_norm", "k_norm"):
            _maybe_shift_mtplx_norm(getattr(attn, name, None), shifted)

    if shifted:
        try:
            import mlx.core as mx

            mx.eval(shifted)
        except Exception:
            pass
    try:
        target._fk_mtplx_norms_shifted = True
    except Exception:
        pass
    return len(shifted)


def _patch_omlx_mtplx_contract(model: Any) -> None:
    """Adapt oMLX native MTP to MTPLX sidecar runtime contracts.

    The MTPLX exported Qwen3.x sidecars declare a post-final-norm backbone
    hidden contract. oMLX's PR-990 compatibility path feeds the MTP head
    the pre-final-norm hidden state. Keep the same MTP path, but normalize
    the returned hidden only for models that carry mtplx_runtime.json.
    """
    if _env_flag("FUSEKIT_DISABLE_MTPLX_CONTRACT_FIX"):
        return
    contract = _load_mtplx_contract(model)
    if not contract:
        return

    hidden_variant = str(
        contract.get("mtp_hidden_variant")
        or contract.get("hidden_variant")
        or "post_norm"
    )
    concat_order = str(
        contract.get("mtp_concat_order")
        or contract.get("concat_order")
        or "embedding_hidden"
    )
    if hidden_variant not in ("pre_norm", "post_norm"):
        return
    if concat_order not in ("embedding_hidden", "hidden_embedding"):
        return

    for target in _iter_mtp_contract_targets(model):
        if not hasattr(target, "mtp_forward") and getattr(target, "mtp", None) is None:
            continue
        try:
            target._fk_mtplx_hidden_variant = hidden_variant
            target._fk_mtplx_concat_order = concat_order
            target._fk_mtplx_returns_post_norm = False
        except Exception:
            continue
        if hidden_variant == "post_norm":
            _patch_mtplx_hidden_contract_class(type(target))
        shifted_norms = _shift_mtplx_mtp_norms(target)
        if os.environ.get("FUSEKIT_TRACE_MTP_CONTRACT"):
            print(
                "[fusekit] MTPLX contract "
                f"hidden={hidden_variant} concat={concat_order} "
                f"norm_shifted={shifted_norms} "
                f"target={type(target).__module__}.{type(target).__name__}",
                file=sys.stderr,
            )


def _model_has_active_mtp(model: Any) -> bool:
    """True when oMLX native MTP is active for this model instance.

    MTP changes the GenerationBatch contract: the loop needs hidden
    states, a separate mtp_forward cache, and reject rollback. FuseKit's
    current fast paths are logits-only decode paths, so they must not
    rewrite MTP generation until the MTP-specific dispatcher is present.
    """
    if model is None:
        return False
    info = _model_mtp_metadata(model)
    _trace_mtp_metadata_once(model, info)
    if info.get("format") != "none" and not info.get("has_weights", False):
        # Config-only MTP declarations are common after stock MLX conversion.
        # Treat them as inactive so we never drive a random or missing draft head.
        return False
    inner = getattr(model, "_fk_original", model)
    try:
        from omlx.patches.mlx_lm_mtp import is_mtp_active

        if not is_mtp_active():
            return False
        # Correctness fixes are part of the native-MTP contract, not only a
        # dispatch optimization. Apply them for active MTP models even when a
        # call later falls through to oMLX.
        _patch_omlx_mtp_greedy_detection()
        _patch_omlx_mtplx_contract(model)
    except Exception:
        pass

    candidates = [inner]
    try:
        candidates.append(getattr(inner, "language_model"))
    except Exception:
        pass
    try:
        candidates.append(getattr(inner, "model"))
    except Exception:
        pass

    has_forward = hasattr(inner, "mtp_forward")
    for candidate in candidates:
        if candidate is not None and getattr(candidate, "mtp", None) is not None:
            return has_forward or hasattr(candidate, "mtp_forward")
    return False


def _cache_has_storage(cache: Any) -> bool:
    """Return true after the native prefill has allocated cache buffers."""
    if cache is None:
        return False
    for c in cache:
        if type(c).__name__ == "ArraysCache":
            for arr in getattr(c, "cache", []):
                if getattr(arr, "dtype", None) is not None:
                    return True
            continue
        keys = getattr(c, "keys", None)
        if isinstance(keys, (tuple, list)):
            if keys and getattr(keys[0], "dtype", None) is not None:
                return True
        elif getattr(keys, "dtype", None) is not None:
            return True
    return False


def _cache_keys_array(keys: Any) -> Any:
    if isinstance(keys, (tuple, list)):
        return keys[0] if keys else None
    return keys


def _cache_array_capacity(arr: Any) -> int | None:
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) < 3:
        return None
    return int(shape[2])


def _cache_capacity(cache: Any) -> int | None:
    """Smallest time-axis capacity across KV-like cache entries."""
    if cache is None:
        return None
    caps: list[int] = []
    for c in cache:
        keys = getattr(c, "keys", None)
        arr = _cache_keys_array(keys)
        cap = _cache_array_capacity(arr)
        if cap is not None:
            max_size = getattr(c, "max_size", None)
            if max_size is not None and "Rotating" in type(c).__name__:
                try:
                    cap = min(int(cap), int(max_size))
                except Exception:
                    pass
            caps.append(cap)
    return min(caps) if caps else None


def _cache_offset(cache: Any) -> int | None:
    if cache is None:
        return None
    offsets = [int(c.offset) for c in cache if hasattr(c, "offset")]
    return max(offsets) if offsets else None


def _engine_cache_offset(cache: Any) -> int | None:
    if cache is None:
        return None
    offsets = []
    for c in cache:
        if type(c).__name__ == "BatchKVCache":
            offsets.append(int(getattr(c, "_idx", 0)))
        elif hasattr(c, "offset"):
            offsets.append(int(c.offset))
    return max(offsets) if offsets else None


def _engine_set_cache_offset(cache: Any, offset: int) -> None:
    if cache is None:
        return
    import mlx.core as mx

    def make_offset_like(current: Any):
        shape = getattr(current, "shape", None)
        if shape is None:
            return offset
        try:
            if len(shape) == 0:
                return mx.array(offset)
            n = int(shape[0]) if len(shape) > 0 else 1
            return mx.array([offset] * max(1, n))
        except Exception:
            return mx.array([offset])

    batch_offset = None
    for c in cache:
        if type(c).__name__ == "BatchKVCache":
            c._idx = offset
            if batch_offset is None:
                batch_offset = make_offset_like(getattr(c, "offset", None))
            c.offset = batch_offset
        elif hasattr(c, "offset"):
            c.offset = make_offset_like(getattr(c, "offset", None))


def _grow_time_axis(arr: Any, target: int):
    import mlx.core as mx

    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) < 3:
        return arr, False
    cur = int(shape[2])
    if target <= cur:
        return arr, False
    extra_shape = list(shape)
    extra_shape[2] = target - cur
    extra = mx.zeros(tuple(extra_shape), dtype=arr.dtype)
    return mx.concatenate([arr, extra], axis=2), True


def _grow_cache_tree(value: Any, target: int):
    if isinstance(value, tuple):
        changed = False
        out = []
        for item in value:
            grown, item_changed = _grow_time_axis(item, target)
            out.append(grown)
            changed = changed or item_changed
        return tuple(out), changed
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            grown, item_changed = _grow_time_axis(item, target)
            out.append(grown)
            changed = changed or item_changed
        return out, changed
    return _grow_time_axis(value, target)


def _cache_storage_arrays(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(_cache_storage_arrays(item))
        return out
    return [value] if getattr(value, "dtype", None) is not None else []


def _ensure_cache_capacity(cache: Any, required: int) -> tuple[bool, str | None]:
    """Grow MLX KV caches before fusekit writes directly into them.

    MLX's normal `update_and_fetch()` path grows KVCache/QuantizedKVCache
    on demand. The fusekit decode kernels bypass that method and write
    into the live buffers by absolute offset, so the host must guarantee
    capacity before any ICB execution.
    """
    if cache is None:
        return False, "no cache"
    if required <= 0:
        return True, None

    import mlx.core as mx

    changed = False
    saw_kv = False
    for c in cache:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        if keys is None or values is None:
            continue
        cap = _cache_array_capacity(_cache_keys_array(keys))
        if cap is None:
            continue
        saw_kv = True
        if cap >= required:
            continue

        max_size = getattr(c, "max_size", None)
        if max_size is not None and required > int(max_size):
            # Rotating/sliding-window caches deliberately cap physical
            # storage at max_size and wrap the write index. `required`
            # is the absolute decode length, so exceeding max_size is
            # valid once the ring buffer is fully allocated.
            if "Rotating" in type(c).__name__:
                if cap >= int(max_size):
                    continue
            else:
                return (
                    False,
                    f"{type(c).__name__} requires {required} slots but max_size is {max_size}",
                )

        step = int(getattr(c, "step", 256) or 256)
        target = ((required + step - 1) // step) * step
        if max_size is not None:
            target = min(target, int(max_size))

        grown_keys, keys_changed = _grow_cache_tree(keys, target)
        grown_values, values_changed = _grow_cache_tree(values, target)
        if keys_changed:
            c.keys = grown_keys
        if values_changed:
            c.values = grown_values
        changed = changed or keys_changed or values_changed

    if not saw_kv:
        return False, "no KV-like cache entries"
    if changed:
        arrays = []
        for c in cache:
            arrays.extend(_cache_storage_arrays(getattr(c, "keys", None)))
            arrays.extend(_cache_storage_arrays(getattr(c, "values", None)))
        mx.eval(arrays)
    return True, None


def _prefill_kv_shape_from_config(model: Any, seq_len: int) -> tuple[int, int, int] | None:
    """Return (n_kv_heads, k_head_dim, v_head_dim) for cache preallocation.

    MLX allocates an empty KVCache lazily from the first K/V tensors passed to
    `update_and_fetch`. FuseKit's prefill variant writes K/V directly into the
    cache, so the host must create the same storage before dispatch. The shape
    comes from semantic config fields, not model names.
    """
    fk_dir = getattr(model, "_fk_dir", None)
    if fk_dir is None:
        return None
    cfg = _read_json_file(Path(fk_dir).parent / "config.json") or {}
    cfg = cfg.get("text_config", cfg)
    try:
        hidden = int(cfg["hidden_size"])
        n_heads = int(cfg["num_attention_heads"])
        n_kv_heads = int(cfg.get("num_key_value_heads") or n_heads)
    except Exception:
        return None
    if n_heads <= 0 or n_kv_heads <= 0:
        return None
    head_dim = int(cfg.get("head_dim") or (hidden // n_heads))
    if head_dim <= 0:
        return None
    v_head_dim = int(cfg.get("v_head_dim") or cfg.get("value_head_dim") or head_dim)
    return n_kv_heads, head_dim, v_head_dim


def _prefill_activation_dtype(model: Any):
    import mlx.core as mx

    fk_dir = getattr(model, "_fk_dir", None)
    cfg = {}
    root_cfg = {}
    if fk_dir is not None:
        root_cfg = _read_json_file(Path(fk_dir).parent / "config.json") or {}
        cfg = root_cfg.get("text_config", root_cfg)
    # Hybrid linear-attention models keep prefill/full-attention state in
    # the model activation dtype even when weights are quantized. Plain
    # qwen2-style quantized models still materialize KV as fp16, so keep
    # the older quantized-linear rule for configs without linear attention.
    has_linear_attention = any(
        str(t) == "linear_attention" for t in cfg.get("layer_types", []) or []
    )
    if _model_uses_quantized_linear(model) and not has_linear_attention:
        return mx.float16
    dtype_name = (
        cfg.get("torch_dtype")
        or cfg.get("dtype")
        or root_cfg.get("torch_dtype")
        or root_cfg.get("dtype")
    )
    if dtype_name in ("bfloat16", "bf16"):
        return mx.bfloat16
    if dtype_name in ("float32", "fp32"):
        return mx.float32
    return mx.float16 if root_cfg.get("quantization") is not None else mx.bfloat16


def _dtype_from_config_name(name: Any, default: Any):
    import mlx.core as mx

    if name in ("bfloat16", "bf16"):
        return mx.bfloat16
    if name in ("float16", "fp16", "half"):
        return mx.float16
    if name in ("float32", "fp32"):
        return mx.float32
    return default


def _prefill_arrays_cache_spec_from_config(
    model: Any,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], Any, Any] | None:
    """Return ArraysCache zero-state shapes for linear-attention layers.

    The layout is config-semantic:
      cache[0] convolution window: [B, conv_kernel - 1, qkv_width]
      cache[1] recurrent state:    [B, value_heads, value_dim, key_dim]

    This covers Qwen3.5/Next-style gated delta linear attention without a
    model-name table.
    """
    fk_dir = getattr(model, "_fk_dir", None)
    if fk_dir is None:
        return None
    root_cfg = _read_json_file(Path(fk_dir).parent / "config.json") or {}
    cfg = root_cfg.get("text_config", root_cfg)
    layer_types = cfg.get("layer_types", []) or []
    if not any(str(t) == "linear_attention" for t in layer_types):
        return None
    try:
        conv_kernel = int(cfg["linear_conv_kernel_dim"])
        key_heads = int(cfg["linear_num_key_heads"])
        value_heads = int(cfg["linear_num_value_heads"])
        key_dim = int(cfg["linear_key_head_dim"])
        value_dim = int(cfg["linear_value_head_dim"])
    except Exception:
        return None
    if min(conv_kernel, key_heads, value_heads, key_dim, value_dim) <= 0:
        return None

    conv_width = 2 * key_heads * key_dim + value_heads * value_dim
    conv_shape = (1, max(conv_kernel - 1, 1), conv_width)
    state_shape = (1, value_heads, value_dim, key_dim)
    act_dtype = _dtype_from_config_name(
        cfg.get("torch_dtype")
        or cfg.get("dtype")
        or root_cfg.get("torch_dtype")
        or root_cfg.get("dtype"),
        _prefill_activation_dtype(model),
    )
    state_dtype = _dtype_from_config_name(cfg.get("mamba_ssm_dtype"), None)
    if state_dtype is None:
        import mlx.core as mx

        state_dtype = mx.float32
    return conv_shape, state_shape, act_dtype, state_dtype


def _prime_empty_kv_cache_for_prefill(model: Any, cache: Any, required: int) -> None:
    """Allocate empty KV cache storage for a compiled prefill variant.

    This mirrors MLX KVCache/QuantizedKVCache first-use allocation without
    calling the model once natively. It is intentionally cache-type based:
    unsupported cache classes remain untouched and therefore fall through to
    the existing native path.
    """
    if cache is None or required <= 0 or _cache_has_storage(cache):
        return
    shape = _prefill_kv_shape_from_config(model, required)
    if shape is None:
        return

    import mlx.core as mx

    n_kv_heads, k_head_dim, v_head_dim = shape
    dtype = _prefill_activation_dtype(model)
    arrays_spec = _prefill_arrays_cache_spec_from_config(model)
    arrays = []
    saw = False
    for c in cache:
        if type(c).__name__ == "ArraysCache" and arrays_spec is not None:
            conv_shape, state_shape, conv_dtype, state_dtype = arrays_spec
            cur = getattr(c, "cache", None)
            if isinstance(cur, list) and len(cur) >= 2 and (cur[0] is None or cur[1] is None):
                cur[0] = mx.zeros(conv_shape, dtype=conv_dtype)
                cur[1] = mx.zeros(state_shape, dtype=state_dtype)
                arrays.extend([cur[0], cur[1]])
                saw = True
            continue
        if getattr(c, "keys", None) is not None or getattr(c, "values", None) is not None:
            continue
        name = type(c).__name__
        step = int(getattr(c, "step", 256) or 256)
        capacity = ((required + step - 1) // step) * step
        if name == "KVCache":
            c.keys = mx.zeros((1, n_kv_heads, capacity, k_head_dim), dtype=dtype)
            c.values = mx.zeros((1, n_kv_heads, capacity, v_head_dim), dtype=dtype)
            arrays.extend([c.keys, c.values])
            saw = True
        elif name == "QuantizedKVCache":
            bits = int(getattr(c, "bits", 8) or 8)
            group_size = int(getattr(c, "group_size", 64) or 64)
            el_per_int = 8 * mx.uint32.size // bits
            base = (1, n_kv_heads, capacity)

            def init_quant(dim: int):
                return (
                    mx.zeros((*base, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*base, dim // group_size), dtype=dtype),
                    mx.zeros((*base, dim // group_size), dtype=dtype),
                )

            c.keys = init_quant(k_head_dim)
            c.values = init_quant(v_head_dim)
            arrays.extend(_cache_storage_arrays(c.keys))
            arrays.extend(_cache_storage_arrays(c.values))
            saw = True
    if saw:
        mx.eval(arrays)


# ---------------------------------------------------------------------
# DLPack helper — extract the host-visible buffer pointer from mx.array
# ---------------------------------------------------------------------
#
# ```text
#   mx.array (Metal-backed, unified memory)
#       │
#       │  view(uint16) for bf16 — MLX's DLPack export rejects bf16
#       │  because it ties dtype validation to numpy, which lacks bf16.
#       │  Same bits, different label; viewing back on the Rust side
#       │  is just a reinterpret_cast.
#       │
#       ▼
#   __dlpack__()  →  capsule "dltensor"  →  DLManagedTensor.data ptr
#       │
#       │  Apple Silicon unified memory: that CPU-mapped pointer is
#       │  the same physical address the GPU sees. Wrapping it with
#       │  `newBufferWithBytesNoCopy:` from Metal gives a zero-copy
#       │  MTLBuffer pointing at the same memory.
#       │
#       ▼
#   ptr (int) ─────►  Rust runtime  ─────►  Metal kernel dispatch


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


# Bind the CPython capsule helpers once at module import.
_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_get_pointer.restype = ctypes.c_void_p
_capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_get_name = ctypes.pythonapi.PyCapsule_GetName
_capsule_get_name.restype = ctypes.c_char_p
_capsule_get_name.argtypes = [ctypes.py_object]


# dtypes mx.array supports that DLPack export rejects. View as the
# numpy-friendly type listed here; the bits are unchanged.
_DLPACK_VIEW_OVERRIDES_BY_NAME = {
    "bfloat16": "uint16",
}


def array_data_ptr(array) -> int:
    """Return the host-visible address of `array`'s backing storage.

    Works for any mx.array dtype; transparently views bf16 (and any
    other DLPack-incompatible types we add later) through a same-bit-
    width integer type first.
    """
    import mlx.core as mx
    dt_name = str(array.dtype).rsplit(".", 1)[-1]
    override = _DLPACK_VIEW_OVERRIDES_BY_NAME.get(dt_name)
    src = array.view(getattr(mx, override)) if override else array
    mx.eval(src)
    cap = src.__dlpack__()
    name = _capsule_get_name(cap)
    raw = _capsule_get_pointer(cap, name)
    mt = _DLManagedTensor.from_address(raw)
    return int(mt.dl_tensor.data)


def array_data_ptrs(arrays) -> list[int]:
    """Vectorized `array_data_ptr` for cold dispatch construction."""
    arrays = list(arrays)
    if not arrays:
        return []
    import mlx.core as mx
    srcs = []
    for array in arrays:
        dt_name = str(array.dtype).rsplit(".", 1)[-1]
        override = _DLPACK_VIEW_OVERRIDES_BY_NAME.get(dt_name)
        srcs.append(array.view(getattr(mx, override)) if override else array)
    mx.eval(*srcs)
    try:
        from . import _fusekit_runtime_native as _native
        return [int(p) for p in _native.array_data_ptrs(srcs)]
    except Exception:
        return [array_data_ptr(array) for array in arrays]


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------


class FusekitWrapper:
    """Drop-in replacement for a wrapped model. Phase 1 forwards every
    call to the original; phase 2 will swap `__call__` for a Metal
    megakernel dispatch.

    Even in passthrough mode the wrapper logs the first few calls'
    shapes so we can see what mlx-lm actually feeds the model — useful
    for confirming our Stage 4 codegen assumptions before we wire
    dispatch.
    """

    def __init__(
        self,
        original: Any,
        fusekit_dir: str,
        *,
        native_handle=None,
        dispatcher_info: str | None = None,
    ):
        self.__dict__["_fk_original"] = original
        self.__dict__["_fk_dir"] = Path(fusekit_dir)
        self.__dict__["_fk_manifest"] = _read_task_graph(fusekit_dir)
        compile_manifest = _read_manifest(fusekit_dir)
        mtp_variants = _manifest_mtp_variant_names(compile_manifest)
        prefill_variants = _manifest_prefill_variant_names(compile_manifest)
        stateful_mtp_graph = _task_graph_has_stateful_mtp(self._fk_manifest)
        stateful_prefill_variants = _stateful_prefill_variant_names(
            self._fk_dir, prefill_variants
        )
        stateful_mtp_mn_ok = (
            not stateful_mtp_graph
            or _env_flag("FUSEKIT_EXPERIMENTAL_STATEFUL_MTP_MN_VARIANT")
            or not _env_flag("FUSEKIT_DISABLE_STATEFUL_MTP_MN_VARIANT")
        )
        self.__dict__["_fk_compile_manifest"] = compile_manifest
        self.__dict__["_fk_mtp_variant_names"] = mtp_variants
        self.__dict__["_fk_prefill_variant_names"] = prefill_variants
        self.__dict__["_fk_stateful_prefill_variants"] = stateful_prefill_variants
        self.__dict__["_fk_has_mtp_variants"] = bool(mtp_variants)
        self.__dict__["_fk_has_prefill_variants"] = bool(prefill_variants)
        self.__dict__["_fk_stateful_mtp_graph"] = stateful_mtp_graph
        self.__dict__["_fk_prefill_variant_auto"] = (
            bool(prefill_variants)
            and not _env_flag("FUSEKIT_DISABLE_PREFILL_VARIANT_AUTO")
        )
        # Native-MTP verification calls the backbone with return_hidden=True.
        # Stateful ArraysCache/GatedDelta graphs use compiled M=N variants by
        # default now that rollback scratch aliasing is guarded in dispatch.py.
        # Keep a kill switch for models that expose a new state primitive.
        self.__dict__["_fk_mtp_single_backbone_auto"] = (
            bool(mtp_variants)
            and _env_flag("FUSEKIT_MTP_SINGLE_BACKBONE")
            and not _env_flag("FUSEKIT_DISABLE_MTP_BACKBONE")
            and not _env_flag("FUSEKIT_DISABLE_MTP_SINGLE_BACKBONE_AUTO")
        )
        # MTP verification feeds logits into speculative acceptance. The
        # compiled M=N verifier is kept opt-in: it is useful for profiling and
        # future small-M kernels, but current oMLX native/split verification is
        # faster on the tested MTP models and avoids extra state drift risk.
        self.__dict__["_fk_mtp_variant_auto"] = (
            bool(mtp_variants)
            and _env_flag("FUSEKIT_AUTO_MTP_MN_VARIANT")
            and stateful_mtp_mn_ok
            and not _env_flag("FUSEKIT_DISABLE_MTP_MN_VARIANT")
        )
        self.__dict__["_fk_native_handle"] = native_handle
        self.__dict__["_fk_dispatcher_info"] = dispatcher_info or ""
        # Bounded opt-in trace; default inference should stay close to
        # raw mlx-lm and not pay stderr overhead on short generations.
        self.__dict__["_fk_log_remaining"] = 3 if _env_flag("FUSEKIT_TRACE_CALLS") else 0
        # Lazy dispatcher — built on the first decode call once the
        # live KV cache exists.
        self.__dict__["_fk_dispatcher"] = None
        self.__dict__["_fk_variant_dispatchers"] = {}
        self.__dict__["_fk_variant_handles"] = {}
        self.__dict__["_fk_dispatch_failed"] = False
        self.__dict__["_fk_dispatch_stats"] = {} if _env_flag("FUSEKIT_TRACE_DISPATCH_STATS") else None
        _register_dispatch_stats_printer(self)

    def _try_native_prefill_no_logits(self, args, kwargs):
        """Run the transformer backbone for a prefill chunk and skip LM head.

        mlx-lm's generation loop discards logits for all prompt chunks except
        the final prompt token. When a FuseKit caller marks such a chunk with
        `_fusekit_prefill_no_logits=True`, we can preserve cache mutation while
        avoiding the tied embedding / lm_head projection. The candidate
        backbone is discovered from common semantic module structure rather
        than a model-name table; if it is absent, callers fall back unchanged.
        """
        if _env_flag("FUSEKIT_DISABLE_NATIVE_PREFILL_NO_LOGITS"):
            return None
        if not args or getattr(args[0], "shape", None) is None:
            return None
        original = self.__dict__.get("_fk_original")
        if original is None:
            return None
        candidates = []
        for path in (
            "model",
            "language_model.model",
            "language_model.model.model",
            "text_model.model",
        ):
            cur = original
            ok = True
            for seg in path.split("."):
                try:
                    cur = getattr(cur, seg)
                except Exception:
                    ok = False
                    break
            if ok and cur is not original and callable(cur):
                candidates.append(cur)

        call_kwargs = {}
        for key in ("cache", "input_embeddings"):
            if key in kwargs:
                call_kwargs[key] = kwargs[key]
        for candidate in candidates:
            try:
                out = candidate(args[0], **call_kwargs)
            except TypeError:
                try:
                    out = candidate(args[0], kwargs.get("cache"))
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(out, tuple):
                out = out[0]
            if getattr(out, "shape", None) is None:
                continue
            import mlx.core as mx

            mx.eval(out)
            return mx.array([[0]], dtype=mx.uint32)
        return None

    def _prefill_variant_chunk_plan(self, seq_len: int) -> list[int] | None:
        if seq_len <= 1 or not _prefill_variant_requested(self, seq_len):
            return None
        plan: list[int] = []
        remaining = seq_len
        while remaining > 0:
            m = _select_prefill_variant_m(self, remaining)
            if m is None or m <= 0:
                return None
            plan.append(m)
            remaining -= m
        return plan if len(plan) > 1 else None

    def _try_run_prefill_variant_chunks(self, inputs, cache, plan, stats):
        if cache is None or not plan:
            return None
        try:
            seq_len = int(inputs.shape[1])
        except Exception:
            return None
        if sum(plan) != seq_len:
            return None

        dispatchers = []
        for m in plan:
            dispatcher = self._get_prefill_variant_dispatcher(m)
            if dispatcher is None:
                if stats is not None:
                    stats["prefill_variant_chunk_missing"] = (
                        stats.get("prefill_variant_chunk_missing", 0) + 1
                    )
                return None
            dispatchers.append((m, dispatcher))

        _prime_empty_kv_cache_for_prefill(self, cache, seq_len)
        if not _cache_has_storage(cache):
            if stats is not None:
                stats["prefill_variant_chunk_no_storage"] = (
                    stats.get("prefill_variant_chunk_no_storage", 0) + 1
                )
            return None

        if stats is not None:
            stats["prefill_variant_chunk"] = stats.get("prefill_variant_chunk", 0) + len(plan)

        offset = 0
        last_out = None
        for m, dispatcher in dispatchers:
            chunk = inputs[:, offset : offset + m]
            last_out = dispatcher.try_run(
                chunk,
                cache,
                return_hidden=False,
                n_confirmed=0,
                prefill_no_logits=True,
            )
            if last_out is None:
                if stats is not None:
                    stats["prefill_variant_chunk_no_output"] = (
                        stats.get("prefill_variant_chunk_no_output", 0) + 1
                    )
                raise RuntimeError(
                    f"fusekit prefill chunk m={m} returned no output after dispatch started"
                )
            offset += m

        if stats is not None:
            stats["prefill_variant_chunk_hit"] = stats.get("prefill_variant_chunk_hit", 0) + 1
        return last_out

    def __call__(self, *args, **kwargs):
        prefill_no_logits = bool(kwargs.pop("_fusekit_prefill_no_logits", False))
        self._log_call(args, kwargs)
        # Decode fast-path: only fires for L=1 inputs with a cache.
        # Prefill and the first (cache-empty) call land on the
        # original model.
        return_hidden = bool(kwargs.get("return_hidden", False))
        n_confirmed = int(kwargs.get("n_confirmed", 0) or 0)
        mtp_active = _model_has_active_mtp(self)
        seq_len = int(args[0].shape[1]) if args and getattr(args[0], "shape", None) is not None and len(args[0].shape) == 2 else 0
        if (
            not prefill_no_logits
            and _ENGINE_EXTERNAL_PREFILL_NO_LOGITS_CURRENT
            and seq_len > 1
            and not return_hidden
            and n_confirmed == 0
            and kwargs.get("cache") is not None
        ):
            prefill_no_logits = True
        mtp_backbone_ok = (
            (
                _env_flag("FUSEKIT_MTP_SINGLE_BACKBONE")
                or self.__dict__.get("_fk_mtp_single_backbone_auto", False)
            )
            and not _env_flag("FUSEKIT_DISABLE_MTP_BACKBONE")
        )
        mtp_mn_ok = (
            (
                self.__dict__.get("_fk_mtp_variant_auto", False)
                or _env_flag("FUSEKIT_USE_MTP_MN_VARIANT")
            )
            and (
                not self.__dict__.get("_fk_stateful_mtp_graph", False)
                or _env_flag("FUSEKIT_EXPERIMENTAL_STATEFUL_MTP_MN_VARIANT")
                or not _env_flag("FUSEKIT_DISABLE_STATEFUL_MTP_MN_VARIANT")
            )
            and not _env_flag("FUSEKIT_DISABLE_MTP_MN_VARIANT")
        )
        single_decode_call = seq_len == 1
        mtp_verify_call = return_hidden and seq_len >= 2 and n_confirmed == 1
        stats = self.__dict__.get("_fk_dispatch_stats")
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            if single_decode_call:
                stats["single_decode"] = stats.get("single_decode", 0) + 1
            if mtp_verify_call:
                stats["mtp_verify"] = stats.get("mtp_verify", 0) + 1
        prefill_variant_name = f"prefill_m{seq_len}"
        stateful_prefill_disabled = (
            prefill_variant_name
            in self.__dict__.get("_fk_stateful_prefill_variants", set())
        )
        prefill_variant_call = (
            seq_len > 1
            and prefill_no_logits
            and not return_hidden
            and n_confirmed == 0
            and _prefill_variant_requested(self, seq_len)
            and prefill_variant_name in self.__dict__.get("_fk_prefill_variant_names", set())
            and not stateful_prefill_disabled
        )
        prefill_chunk_plan = None
        prefill_chunk_call = False
        if (
            prefill_no_logits
            and seq_len > 1
            and not return_hidden
            and n_confirmed == 0
            and not prefill_variant_call
            and _env_flag("FUSEKIT_EXPERIMENTAL_PREFILL_CHUNK_VARIANT")
        ):
            prefill_chunk_plan = self._prefill_variant_chunk_plan(seq_len)
            prefill_chunk_call = prefill_chunk_plan is not None
        if stats is not None and stateful_prefill_disabled:
            stats["prefill_variant_stateful_disabled"] = (
                stats.get("prefill_variant_stateful_disabled", 0) + 1
            )
        if stats is not None and prefill_variant_call:
            stats["prefill_variant"] = stats.get("prefill_variant", 0) + 1
        if stats is not None and prefill_chunk_call:
            stats["prefill_variant_chunk_call"] = stats.get("prefill_variant_chunk_call", 0) + 1
        if (prefill_variant_call or prefill_chunk_call) and kwargs.get("cache") is not None:
            _prime_empty_kv_cache_for_prefill(self, kwargs.get("cache"), seq_len)
        return_hidden_dispatch_ok = (
            not return_hidden
            or single_decode_call
            or mtp_verify_call
            or prefill_variant_call
            or prefill_chunk_call
        )
        mtp_dispatch_ok = (
            not mtp_active
            or prefill_variant_call
            or prefill_chunk_call
            or (
                return_hidden
                and (
                    (single_decode_call and (mtp_backbone_ok or mtp_mn_ok))
                    or (mtp_verify_call and (mtp_backbone_ok or mtp_mn_ok))
                )
            )
        )
        if (
            self._fk_native_handle is not None
            and _dispatch_enabled()
            and not self._fk_dispatch_failed
            and return_hidden_dispatch_ok
            and mtp_dispatch_ok
            and args
            and getattr(args[0], "shape", None) is not None
            and len(args[0].shape) == 2
            and (single_decode_call or mtp_verify_call or prefill_variant_call or prefill_chunk_call)
            and kwargs.get("cache") is not None
            and _cache_has_storage(kwargs.get("cache"))
        ):
            self.__dict__.pop("_fk_last_argmax_tokens", None)
            self.__dict__.pop("_fk_mtp_split_tokens_trusted", None)
            if stats is not None:
                stats["eligible"] = stats.get("eligible", 0) + 1
            try:
                out = None
                hit_dispatcher = None
                if prefill_chunk_call and prefill_chunk_plan is not None:
                    if stats is not None:
                        stats["prefill_variant_chunk_attempt"] = (
                            stats.get("prefill_variant_chunk_attempt", 0) + 1
                        )
                    out = self._try_run_prefill_variant_chunks(
                        args[0],
                        kwargs.get("cache"),
                        prefill_chunk_plan,
                        stats,
                    )
                    if out is not None:
                        if stats is not None:
                            stats["dispatch_hit"] = stats.get("dispatch_hit", 0) + 1
                        return out
                    if out is None and _env_flag("FUSEKIT_STRICT_DISPATCH"):
                        raise RuntimeError("fusekit prefill chunk variant returned no output")
                if prefill_variant_call:
                    if stats is not None:
                        stats["prefill_variant_attempt"] = stats.get("prefill_variant_attempt", 0) + 1
                    variant_dispatcher = self._get_prefill_variant_dispatcher(seq_len)
                    if variant_dispatcher is not None:
                        out = variant_dispatcher.try_run(
                            args[0],
                            kwargs.get("cache"),
                            return_hidden=False,
                            n_confirmed=0,
                            prefill_no_logits=prefill_no_logits,
                        )
                        if out is not None:
                            hit_dispatcher = variant_dispatcher
                        if stats is not None:
                            key = "prefill_variant_hit" if out is not None else "prefill_variant_no_output"
                            stats[key] = stats.get(key, 0) + 1
                    else:
                        if stats is not None:
                            stats["prefill_variant_missing"] = stats.get("prefill_variant_missing", 0) + 1
                    if out is None and _env_flag("FUSEKIT_STRICT_DISPATCH"):
                        raise RuntimeError("fusekit prefill variant returned no output")
                if mtp_verify_call and mtp_mn_ok:
                    if stats is not None:
                        stats["mtp_variant_attempt"] = stats.get("mtp_variant_attempt", 0) + 1
                    variant_dispatcher = self._get_variant_dispatcher(seq_len)
                    if variant_dispatcher is not None:
                        out = variant_dispatcher.try_run(
                            args[0],
                            kwargs.get("cache"),
                            return_hidden=return_hidden,
                            n_confirmed=n_confirmed,
                        )
                        if out is not None:
                            hit_dispatcher = variant_dispatcher
                        if stats is not None:
                            key = "mtp_variant_hit" if out is not None else "mtp_variant_no_output"
                            stats[key] = stats.get(key, 0) + 1
                    else:
                        if stats is not None:
                            stats["mtp_variant_missing"] = stats.get("mtp_variant_missing", 0) + 1
                        if os.environ.get("FUSEKIT_TRACE_MTP_MN"):
                            print(
                                f"[fusekit] no MTP M=N variant for seq_len={seq_len}",
                                file=sys.stderr,
                            )
                    if out is None and _env_flag("FUSEKIT_STRICT_DISPATCH"):
                        raise RuntimeError("fusekit MTP M=N variant returned no output")
                if out is None and not prefill_chunk_call:
                    if self._fk_dispatcher is None:
                        from .dispatch import Dispatcher
                        self.__dict__["_fk_dispatcher"] = Dispatcher(
                            self._fk_original, self._fk_dir, self._fk_native_handle
                        )
                    if stats is not None and mtp_verify_call:
                        stats["mtp_split_attempt"] = stats.get("mtp_split_attempt", 0) + 1
                    if mtp_verify_call and mtp_backbone_ok:
                        out = self._try_run_mtp_split_m1(
                            args[0],
                            kwargs.get("cache"),
                        )
                    else:
                        out = self._fk_dispatcher.try_run(
                            args[0],
                            kwargs.get("cache"),
                            return_hidden=return_hidden,
                            n_confirmed=n_confirmed,
                        )
                    if out is not None:
                        hit_dispatcher = self._fk_dispatcher
                    if stats is not None and out is not None:
                        key = "mtp_split_hit" if mtp_verify_call else "single_decode_hit"
                        stats[key] = stats.get(key, 0) + 1
                if out is not None:
                    tokens = getattr(hit_dispatcher, "last_argmax_tokens", None)
                    if tokens is not None:
                        self.__dict__["_fk_last_argmax_tokens"] = tokens
                    else:
                        self.__dict__.pop("_fk_last_argmax_tokens", None)
                    if stats is not None:
                        stats["dispatch_hit"] = stats.get("dispatch_hit", 0) + 1
                    return out
                if stats is not None:
                    stats["dispatch_none"] = stats.get("dispatch_none", 0) + 1
                if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                    raise RuntimeError("fusekit strict dispatch returned no output")
            except Exception as e:
                if stats is not None:
                    stats["dispatch_exception"] = stats.get("dispatch_exception", 0) + 1
                if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                    raise
                warnings.warn(f"fusekit dispatch failed ({e}); passthrough")
                self.__dict__["_fk_dispatch_failed"] = True
        if (
            prefill_no_logits
            and seq_len > 1
            and not return_hidden
            and n_confirmed == 0
            and kwargs.get("cache") is not None
        ):
            out = self._try_native_prefill_no_logits(args, kwargs)
            if out is not None:
                if stats is not None:
                    stats["native_prefill_no_logits"] = stats.get("native_prefill_no_logits", 0) + 1
                return out
        if stats is not None:
            stats["native_passthrough"] = stats.get("native_passthrough", 0) + 1
            if mtp_verify_call:
                stats["mtp_native_passthrough"] = stats.get("mtp_native_passthrough", 0) + 1
        return self._fk_original(*args, **kwargs)

    @staticmethod
    def _clone_mx_array(value):
        if value is None or not hasattr(value, "shape"):
            return value
        import mlx.core as mx

        # MLX arrays do not expose copy(). Materialize a same-device copy
        # before the following speculative token mutates recurrent state.
        return value + mx.zeros_like(value)

    def _snapshot_mtp_rollback(self, cache):
        import mlx.core as mx

        snapshots = []
        arrays = []
        for c in cache or []:
            if not hasattr(c, "__setitem__"):
                continue
            state = getattr(c, "state", None)
            if not isinstance(state, (list, tuple)) or len(state) < 2:
                continue
            conv = self._clone_mx_array(state[0])
            ssm = self._clone_mx_array(state[1])
            if hasattr(conv, "shape"):
                arrays.append(conv)
            if hasattr(ssm, "shape"):
                arrays.append(ssm)
            snapshots.append((c, conv, ssm))
        if arrays:
            mx.eval(*arrays)
        return snapshots

    @staticmethod
    def _install_mtp_rollback(snapshots) -> None:
        for c, conv, ssm in snapshots:
            try:
                c.rollback_state = (conv, ssm)
            except Exception:
                pass

    def _try_run_mtp_split_m1(self, inputs, cache):
        import mlx.core as mx

        if self._fk_dispatcher is None:
            return None
        if getattr(inputs, "shape", None) is None or len(inputs.shape) != 2:
            return None
        if int(inputs.shape[1]) != 2:
            return None

        first = inputs[:, 0:1]
        second = inputs[:, 1:2]
        out0 = self._fk_dispatcher.try_run(
            first,
            cache,
            return_hidden=True,
            n_confirmed=0,
        )
        if not isinstance(out0, tuple) or len(out0) < 2:
            return None
        tok0 = getattr(self._fk_dispatcher, "last_argmax_tokens", None)
        hidden0 = self._clone_mx_array(out0[1])
        if tok0 is not None:
            tok0 = tok0.reshape(-1)[:1].astype(mx.uint32)
            mx.eval(tok0)
        compact_logits = (
            tok0 is not None
            and _env_flag("FUSEKIT_ENABLE_MTP_SPLIT_COMPACT_LOGITS")
            and not _env_flag("FUSEKIT_DISABLE_MTP_SPLIT_COMPACT_LOGITS")
            and os.environ.get("FUSEKIT_MTP_VERIFY_MARGIN") is None
            and os.environ.get("FUSEKIT_MTP_BONUS_MARGIN") is None
        )
        logits0 = out0[0] if compact_logits else self._clone_mx_array(out0[0])
        mx.eval(hidden0)
        if not compact_logits:
            mx.eval(logits0)
        snapshots = self._snapshot_mtp_rollback(cache)
        out1 = self._fk_dispatcher.try_run(
            second,
            cache,
            return_hidden=True,
            n_confirmed=0,
        )
        if not isinstance(out1, tuple) or len(out1) < 2:
            return None
        tok1 = getattr(self._fk_dispatcher, "last_argmax_tokens", None)
        if tok1 is not None:
            tok1 = tok1.reshape(-1)[:1].astype(mx.uint32)
            mx.eval(tok1)
        logits1 = out1[0] if compact_logits and tok1 is not None else self._clone_mx_array(out1[0])
        hidden1 = self._clone_mx_array(out1[1])
        mx.eval(hidden1)
        if not (compact_logits and tok1 is not None):
            mx.eval(logits1)
        self._install_mtp_rollback(snapshots)
        if tok0 is not None and tok1 is not None:
            tokens = mx.concatenate([tok0, tok1], axis=0).astype(mx.uint32)
            mx.eval(tokens)
            self._fk_dispatcher.last_argmax_tokens = tokens
            self.__dict__["_fk_mtp_split_tokens_trusted"] = True
        else:
            self.__dict__.pop("_fk_mtp_split_tokens_trusted", None)
        if compact_logits and tok1 is not None:
            if len(logits1.shape) == 3:
                logits = mx.broadcast_to(logits1[:, :1, :], (logits1.shape[0], 2, logits1.shape[-1]))
            else:
                logits = mx.broadcast_to(logits1.reshape(1, 1, -1), (1, 2, logits1.shape[-1]))
        else:
            logits = mx.concatenate([logits0, logits1], axis=1)
        hidden = mx.concatenate([hidden0, hidden1], axis=1)
        if compact_logits and tok1 is not None:
            mx.eval(hidden)
        else:
            mx.eval(logits, hidden)
        return logits, hidden

    def _get_named_variant_dispatcher(self, variant_name: str, trace_env: str | None = None):
        root = Path(self._fk_dir)
        if root.parent.name == "variants":
            return None
        cache = self.__dict__["_fk_variant_dispatchers"]
        if variant_name in cache:
            return cache[variant_name]
        variant_dir = root / "variants" / variant_name
        if not (variant_dir / "kernel.metallib").is_file():
            return None
        if not (variant_dir / "task_graph.json").is_file():
            return None
        try:
            from . import _fusekit_runtime_native as _native
            from .dispatch import Dispatcher

            handle = _native.load_dispatcher(
                str(variant_dir / "kernel.metallib"),
                str(variant_dir / "task_graph.json"),
            )
            dispatcher = Dispatcher(self._fk_original, variant_dir, handle)
        except Exception as e:
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise
            warnings.warn(f"fusekit: variant {variant_name} unavailable ({e})")
            return None
        self.__dict__["_fk_variant_handles"][variant_name] = handle
        cache[variant_name] = dispatcher
        if trace_env and os.environ.get(trace_env):
            print(f"[fusekit] loaded variant {variant_name} -> {variant_dir}", file=sys.stderr)
        return dispatcher

    def _get_variant_dispatcher(self, seq_len: int):
        return self._get_named_variant_dispatcher(
            f"mtp_m{seq_len}",
            "FUSEKIT_TRACE_MTP_MN",
        )

    def _get_prefill_variant_dispatcher(self, seq_len: int):
        return self._get_named_variant_dispatcher(
            f"prefill_m{seq_len}",
            "FUSEKIT_TRACE_PREFILL",
        )

    def _log_call(self, args, kwargs) -> None:
        remaining = self.__dict__["_fk_log_remaining"]
        if remaining <= 0:
            return
        self.__dict__["_fk_log_remaining"] = remaining - 1
        parts = []
        for i, a in enumerate(args):
            shape = getattr(a, "shape", None)
            dtype = getattr(a, "dtype", None)
            parts.append(f"arg{i}: shape={shape} dtype={dtype}")
        for k, v in kwargs.items():
            if v is None:
                parts.append(f"{k}=None")
            else:
                parts.append(f"{k}={type(v).__name__}")
        print(f"[fusekit] __call__  {'  '.join(parts)}", file=sys.stderr)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fk_original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._fk_original, name, value)

    def __repr__(self) -> str:
        return f"FusekitWrapper(dir={self._fk_dir}, original={self._fk_original!r})"


def attach(model: Any, fusekit_dir: str) -> Any:
    """Wrap `model` with the fusekit dispatcher. Returns the original
    on any failure so callers can never crash because of fusekit."""
    try:
        dir_path = Path(fusekit_dir)
        if not (dir_path / "kernel.metallib").is_file():
            warnings.warn(f"fusekit: no kernel.metallib in {dir_path}; passthrough")
            return model
        if not (dir_path / "task_graph.json").is_file():
            warnings.warn(f"fusekit: no task_graph.json in {dir_path}; passthrough")
            return model

        # Bring up the Rust-side dispatcher by default. Raw mlx-lm
        # comparison is opt-out via FUSEKIT_DISABLE_HOOK or
        # FUSEKIT_DISABLE_DISPATCH.
        handle = None
        info_line = None
        if not _dispatch_enabled():
            return model
        dispatch_dir = _select_dispatch_dir(model, dir_path)
        try:
            from . import _fusekit_runtime_native as _native
            handle = _native.load_dispatcher(
                str(dispatch_dir / "kernel.metallib"),
                str(dispatch_dir / "task_graph.json"),
            )
            info_line = _native.dispatcher_info(handle)
        except ImportError as e:
            warnings.warn(
                f"fusekit: _fusekit_runtime_native unavailable ({e}); passthrough"
            )
            return model

        wrapped = FusekitWrapper(
            model,
            str(dispatch_dir),
            native_handle=handle,
            dispatcher_info=info_line,
        )
        _model_mtp_metadata(wrapped)
        print(f"[fusekit] attached → {dispatch_dir}", file=sys.stderr)
        if info_line:
            print(f"[fusekit]   {info_line}", file=sys.stderr)
        if handle is not None and os.environ.get("FUSEKIT_SMOKE"):
            # Opt-in smoke dispatch — encodes + commits a dummy kernel
            # launch on each attach so users can verify the runtime
            # pipeline works without waiting for real inference.
            try:
                msg = _native.smoke_dispatch(handle)
                print(f"[fusekit]   {msg}", file=sys.stderr)
            except Exception as e:
                warnings.warn(f"fusekit smoke dispatch failed: {e}")
        return wrapped
    except Exception as e:
        warnings.warn(f"fusekit attach failed: {e}")
        return model


class _FusekitDirProxy:
    def __init__(self, original: Any, fusekit_dir: Path):
        self._fk_original = original
        self._fk_dir = fusekit_dir


def _select_dispatch_dir(model: Any, root_dir: Path) -> Path:
    forced = os.environ.get("FUSEKIT_DISPATCH_VARIANT", "").strip()
    if forced:
        if forced in ("root", "decode", "m1"):
            return root_dir
        candidate = root_dir / "variants" / forced
        if candidate.is_dir():
            return candidate
        warnings.warn(f"fusekit: dispatch variant {forced!r} not found; using root")
        return root_dir

    # MTP M=N variants are selected per call from FusekitWrapper so
    # normal single-token decode keeps using the root M=1 dispatcher.
    return root_dir


def _read_task_graph(fusekit_dir: str) -> dict:
    try:
        with open(os.path.join(fusekit_dir, "task_graph.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_manifest(fusekit_dir: str) -> dict:
    try:
        with open(os.path.join(fusekit_dir, "manifest.json")) as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _manifest_variant_names_with_prefix(manifest: dict, prefix: str) -> set[str]:
    variants = manifest.get("variants", []) if isinstance(manifest, dict) else []
    if not isinstance(variants, list):
        return set()
    names: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        name = str(variant.get("name", ""))
        if name.startswith(prefix):
            names.add(name)
    return names


def _manifest_mtp_variant_names(manifest: dict) -> set[str]:
    return _manifest_variant_names_with_prefix(manifest, "mtp_m")


def _manifest_prefill_variant_names(manifest: dict) -> set[str]:
    return _manifest_variant_names_with_prefix(manifest, "prefill_m")


def _prefill_variant_auto_max_m(model: Any) -> int:
    """Largest prefill variant used automatically.

    M=512/M=1024 variants use the block-NAX SDPA emitter, so the automatic path
    can keep the full prompt graph for medium/long prefill instead of falling
    back to the native no-logits path. The selector still requires the compiled
    variant to exist in the manifest.
    """
    _ = model
    return 1024


def _prefill_variant_requested(model: Any, seq_len: int) -> bool:
    if seq_len <= 1:
        return False
    if _env_flag("FUSEKIT_FORCE_PREFILL_VARIANT"):
        return True
    requested = bool(getattr(model, "_fk_prefill_variant_auto", False)) or _env_flag(
        "FUSEKIT_USE_PREFILL_VARIANT"
    )
    return requested and seq_len <= _prefill_variant_auto_max_m(model)


def _select_prefill_variant_m(model: Any, limit: int) -> int | None:
    if limit <= 1:
        return None
    if not _prefill_variant_requested(model, limit):
        return None
    names = getattr(model, "_fk_prefill_variant_names", None)
    if not names:
        return None
    disabled = getattr(model, "_fk_stateful_prefill_variants", set()) or set()
    best = 0
    for name in names:
        if not isinstance(name, str) or not name.startswith("prefill_m"):
            continue
        if name in disabled:
            continue
        try:
            m = int(name[len("prefill_m"):])
        except ValueError:
            continue
        if m <= limit and m > best:
            best = m
    return best or None


def _task_graph_has_stateful_mtp(task_graph: dict) -> bool:
    """Whether M=N MTP verify must preserve recurrent ArraysCache state."""
    return _task_graph_has_stateful_recurrence(task_graph)


def _task_graph_has_stateful_recurrence(task_graph: dict) -> bool:
    """Whether a graph mutates recurrent state that must be sequenced exactly."""
    tasks = task_graph.get("tasks", []) if isinstance(task_graph, dict) else []
    if not isinstance(tasks, list):
        return False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        kernel = str(task.get("kernel", ""))
        if "gated_delta" in kernel or "arrays_cache_shift_append" in kernel:
            return True
    return False


def _stateful_prefill_variant_names(fusekit_dir: Path, names: set[str]) -> set[str]:
    """Find prefill variants that cannot be used as a full-sequence graph yet.

    Linear-attention/GDN prefill needs a real recurrence emitter over the M
    rows. A decode-shape shift/append graph is correct for one token but wrong
    when run as prefill_mN, so disable those variants by semantic task content.
    """
    if not names:
        return set()
    root = Path(fusekit_dir)
    if root.parent.name == "variants":
        root = root.parent.parent
    disabled: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            continue
        task_graph = _read_task_graph(str(root / "variants" / name))
        if _task_graph_has_stateful_recurrence(task_graph):
            disabled.add(name)
    return disabled


# ---------------------------------------------------------------------
# Import hook
# ---------------------------------------------------------------------
#
#   ┌─ interpreter startup
#   │
#   │  .pth file runs `import fusekit_runtime`
#   │  → install_hook() installs _FusekitMetaPathFinder
#   │
#   ├─ user code runs `from mlx_lm.utils import load`
#   │
#   │  → finder intercepts the import of `mlx_lm.utils`
#   │  → original loader runs, module is created
#   │  → finder monkey-patches module.load to wrap results in attach()
#   │
#   └─ load("/some/model") returns a FusekitWrapper when fusekit/ exists


_TOP_LOAD_MODULE = "mlx_lm"
_TARGET_MODULE = "mlx_lm.utils"
_GENERATE_MODULE = "mlx_lm.generate"
_SAMPLE_UTILS_MODULE = "mlx_lm.sample_utils"
_OMLX_SCHEDULER_MODULE = "omlx.scheduler"
_PATCHED = False
_PATCHED_LOAD_MODULES: set[str] = set()
_GENERATE_PATCHED = False
_ENGINE_PATCHED = False
_OMLX_SCHEDULER_PATCHED = False
_ENGINE_SKIP_LOGPROBS_CURRENT_STEP = False
_ENGINE_EXTERNAL_PREFILL_NO_LOGITS_CURRENT = False


def _extra_tensor_files_for_model_glob(pathname: Any) -> list[str]:
    try:
        raw = os.fspath(pathname)
    except Exception:
        return []
    path = Path(raw)
    if path.name != "model*.safetensors":
        return []

    model_dir = path.parent
    config = _read_json_file(model_dir / "config.json") or {}
    extra = config.get("mlx_lm_extra_tensors")
    candidates: list[Path] = []
    if isinstance(extra, dict):
        mtp_file = extra.get("mtp_file")
        if isinstance(mtp_file, str) and mtp_file:
            candidates.append(model_dir / mtp_file)

    # MTPLX sidecar repos may carry the runtime contract even when older
    # converters did not fill mlx_lm_extra_tensors.
    if (model_dir / "mtplx_runtime.json").is_file():
        candidates.append(model_dir / "mtp.safetensors")

    out: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            key = candidate.resolve()
        except Exception:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        out.append(str(candidate))
    return out


def _wrap_extra_tensor_glob(module) -> None:
    glob_mod = getattr(module, "glob", None)
    original = getattr(glob_mod, "glob", None)
    if original is None or getattr(original, "_fusekit_extra_tensor_patched", False):
        return

    def _patched_glob(pathname, *args, **kwargs):
        result = original(pathname, *args, **kwargs)
        extras = _extra_tensor_files_for_model_glob(pathname)
        if not extras:
            return result
        try:
            merged = list(result)
            for extra in extras:
                if extra not in merged:
                    merged.append(extra)
            return merged
        except Exception:
            return result

    _patched_glob._fusekit_extra_tensor_patched = True
    _patched_glob._fusekit_original = original
    glob_mod.glob = _patched_glob


def _augment_mtplx_mtp_quantization(config: dict) -> None:
    if not isinstance(config, dict):
        return
    policy = config.get("mtplx_policy")
    policy_mtp = policy.get("mtp_quantization") if isinstance(policy, dict) else None
    top_mtp = config.get("mtplx_mtp_quantization")
    if not isinstance(policy_mtp, dict) and not isinstance(top_mtp, dict):
        return

    source = policy_mtp if isinstance(policy_mtp, dict) else top_mtp
    if not isinstance(source, dict):
        return
    linears = source.get("linears")
    if not isinstance(linears, list):
        linears = policy_mtp.get("linears") if isinstance(policy_mtp, dict) else None
    if not isinstance(linears, list):
        return

    bits = _config_int(
        source.get("bits"),
        top_mtp.get("bits") if isinstance(top_mtp, dict) else None,
    )
    group_size = _config_int(
        source.get("group_size"),
        top_mtp.get("group_size") if isinstance(top_mtp, dict) else None,
    )
    mode = str(
        source.get(
            "mode",
            top_mtp.get("mode", "affine") if isinstance(top_mtp, dict) else "affine",
        )
    )
    if bits <= 0 or group_size <= 0:
        return

    quant = config.get("quantization")
    if not isinstance(quant, dict):
        quant = {}
        config["quantization"] = quant

    entry = {"bits": bits, "group_size": group_size, "mode": mode}
    for item in linears:
        if not isinstance(item, str) or not item:
            continue
        base = item[:-7] if item.endswith(".weight") else item
        for path in (base, f"language_model.{base}"):
            quant[path] = dict(entry)


def _wrap_mtplx_load_config(module) -> None:
    original = getattr(module, "load_config", None)
    if original is None or getattr(original, "_fusekit_mtplx_config_patched", False):
        return

    def _patched_load_config(model_path):
        config = original(model_path)
        try:
            _augment_mtplx_mtp_quantization(config)
        except Exception:
            pass
        return config

    _patched_load_config._fusekit_mtplx_config_patched = True
    _patched_load_config._fusekit_original = original
    module.load_config = _patched_load_config


def _candidate_fusekit_dirs(path_or_repo: Any) -> list[Path]:
    if path_or_repo is None:
        return []
    raw = str(path_or_repo)
    if not raw:
        return []
    p = Path(raw).expanduser()
    candidates = [p / "fusekit"]
    if not p.is_absolute():
        roots = []
        env_root = os.environ.get("FUSEKIT_OMLX_DIR")
        if env_root:
            roots.append(Path(env_root).expanduser() / "models")
        roots.append(Path.home() / ".omlx" / "models")
        roots.append(Path.home() / "models")
        for root in roots:
            candidates.append(root / raw / "fusekit")
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except Exception:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _resolve_fusekit_dir(path_or_repo: Any) -> Path | None:
    for candidate in _candidate_fusekit_dirs(path_or_repo):
        if (candidate / "kernel.metallib").is_file() and (
            candidate / "task_graph.json"
        ).is_file():
            return candidate
    return None


def _wrap_load(module) -> None:
    """Monkey-patch `module.load` to call `attach()` after the original
    returns. Idempotent."""
    global _PATCHED
    _wrap_extra_tensor_glob(module)
    _wrap_mtplx_load_config(module)
    module_name = getattr(module, "__name__", repr(module))
    if module_name in _PATCHED_LOAD_MODULES:
        return
    if not hasattr(module, "load"):
        return
    original = module.load
    if getattr(original, "_fusekit_load_patched", False):
        _PATCHED_LOAD_MODULES.add(module_name)
        _PATCHED = True
        return

    def _patched_load(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            path_or_repo = args[0] if args else kwargs.get("path_or_hf_repo")
            fk_dir = _resolve_fusekit_dir(path_or_repo)
            if fk_dir is None:
                return result
            model = result[0]
            if len(result) > 1:
                _prime_tokenizer_cache(result[1])
            wrapped = attach(model, fk_dir)
            return (wrapped,) + tuple(result[1:])
        except Exception as e:
            warnings.warn(f"fusekit hook failed: {e}")
            return result

    _patched_load._fusekit_load_patched = True
    module.load = _patched_load
    _PATCHED_LOAD_MODULES.add(module_name)
    _PATCHED = True


def _prime_tokenizer_cache(tokenizer) -> None:
    """Cache the tokenizer's vocab + detokenizer to skip ~50ms per
    `stream_generate` call.

    `BPEStreamingDetokenizer.__init__` calls `tokenizer.vocab` twice
    (once via `len()`, once via `.items()`); each access goes through
    HF's `get_vocab()` which rebuilds the dict from the rust tokenizer.
    Cache the result on the HF tokenizer instance so the constructor
    becomes a cheap dict copy.

    Also memoize the Detokenizer per-tokenizer-instance and replace the
    `detokenizer` property on the class. The property looks up the
    cached instance via `self._fk_detok` so multiple tokenizers (e.g.
    loading two different models) each keep their own. The class-level
    patch is idempotent via a sentinel flag.
    """
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is None:
        return
    try:
        cached_vocab = inner.get_vocab()
        inner.get_vocab = lambda: cached_vocab
    except Exception:
        return

    cls = type(tokenizer)
    if getattr(cls, "_fk_detokenizer_patched", False):
        return

    def _cached_detokenizer(self):
        cached = getattr(self, "_fk_detok", None)
        if cached is None:
            cached = self._detokenizer_class(self)
            self._fk_detok = cached
        cached.reset()
        return cached

    try:
        cls.detokenizer = property(_cached_detokenizer)
        cls._fk_detokenizer_patched = True
    except Exception:
        pass


def _wrap_generate_step(module) -> None:
    """Monkey-patch `mlx_lm.generate.generate_step` so the fusekit
    fast-path is used when (a) the model has a populated `_fk_lm`
    state with an in-ICB argmax, and (b) generation is semantically
    greedy (default sampler or temp=0 sampler, no logits processors).

    Defensive: inspects the signature; if it doesn't match the
    expected `(prompt, model, *, ..., sampler=None, prompt_cache=None,
    ...)` shape, the original is left alone and a warning is logged.
    Compatible with mlx-lm versions that keep this shape; degrades
    gracefully otherwise.
    """
    global _GENERATE_PATCHED
    if _GENERATE_PATCHED:
        return
    import inspect
    if not hasattr(module, "generate_step"):
        return
    _orig = module.generate_step
    sig = None
    try:
        sig = inspect.signature(_orig)
    except (TypeError, ValueError) as e:
        warnings.warn(
            f"fusekit: cannot inspect mlx_lm.generate.generate_step "
            f"signature ({e}); leaving it unpatched."
        )
        return
    required = {"prompt", "model", "sampler", "prompt_cache"}
    if not required.issubset(sig.parameters):
        warnings.warn(
            "fusekit: mlx_lm.generate.generate_step signature changed "
            f"(missing {required - set(sig.parameters)}); leaving it "
            "unpatched. fusekit will still accelerate model.__call__ via "
            "the standard mlx-lm.generate loop."
        )
        return

    def _patched(prompt, model, **kw):
        # Conditions to use the fast path:
        # 1. Model is fusekit-wrapped and dispatchable.
        # 2. Caller is using the default greedy sampler and no logits
        #    processors, so the in-ICB argmax is semantically equivalent
        #    to mlx-lm's default sampler.
        # 3. Caller did not supply a prompt_cache. The fast path mutates
        #    cache state during bootstrap; when fusekit has to fall back,
        #    we need to be able to rebuild a fresh cache through _orig.
        if not _mlx_generate_kwargs_are_greedy(kw):
            yield from _orig(prompt, model, **kw)
            return
        if kw.get("input_embeddings") is not None:
            # We don't support the embeddings-direct path.
            yield from _orig(prompt, model, **kw)
            return
        if kw.get("prompt_cache") is not None:
            yield from _orig(prompt, model, **kw)
            return
        inner = getattr(model, "_fk_original", None)
        if inner is None:
            if os.environ.get("FUSEKIT_TRACE_GENSTEP"):
                print("[gen] fastpath bypassed: model not wrapped", file=sys.stderr)
            yield from _orig(prompt, model, **kw)
            return
        if _model_has_active_mtp(model):
            yield from _orig(prompt, model, **kw)
            return
        if getattr(model, "_fk_native_handle", None) is None or getattr(
            model, "_fk_dispatch_failed", False
        ):
            yield from _orig(prompt, model, **kw)
            return
        manifest = getattr(model, "_fk_manifest", {}) or {}
        caps = manifest.get("decode_capabilities") or {}
        if caps and not caps.get("dispatchable", True):
            yield from _orig(prompt, model, **kw)
            return
        if os.environ.get("FUSEKIT_TRACE_GENSTEP"):
            print("[gen] fastpath ACTIVE", file=sys.stderr)
        yield from _fast_generate_step(prompt, model, inner, kw, _orig)

    module.generate_step = _patched
    _GENERATE_PATCHED = True
    if os.environ.get("FUSEKIT_TRACE_GENSTEP"):
        print(f"[gen] patched mlx_lm.generate.generate_step", file=sys.stderr)


def _wrap_generation_batch(module) -> None:
    """Use FuseKit's in-ICB argmax on oMLX's BatchGenerator path.

    oMLX's engine drives mlx-lm's `BatchGenerator`, not `generate_step`.
    The normal `GenerationBatch._step` computes logits, normalizes them,
    samples, then asynchronously evaluates the next token. FuseKit decode
    kernels already compute greedy argmax into `next_token_out`; for the
    common engine path (single row, temp=0, no logits processors, no
    requested logprobs) the normalization + sampler work is redundant.
    """
    if not hasattr(module, "GenerationBatch"):
        return
    cls = module.GenerationBatch
    if getattr(cls, "_fk_engine_fast_patched", False):
        return
    original_step = cls._step
    original_next = getattr(cls, "next", None)
    original_extend = getattr(cls, "extend", None)
    original_filter = getattr(cls, "filter", None)
    batch_cls = getattr(module, "BatchGenerator", None)
    prompt_cls = getattr(module, "PromptProcessingBatch", None)
    original_next_generated = (
        getattr(batch_cls, "next_generated", None) if batch_cls is not None else None
    )
    original_insert_segments = (
        getattr(batch_cls, "insert_segments", None) if batch_cls is not None else None
    )
    original_prompt = (
        getattr(prompt_cls, "prompt", None) if prompt_cls is not None else None
    )
    original_generate = (
        getattr(prompt_cls, "generate", None) if prompt_cls is not None else None
    )

    def _engine_style() -> str:
        raw = os.environ.get("FUSEKIT_ENGINE_STYLE", "omlx")
        return raw.strip().lower().replace("_", "-")

    def _is_omlx_step_style() -> bool:
        return _engine_style() in ("omlx", "omlx-step", "single-step", "step")

    def _stopless(self) -> bool:
        for sm in getattr(self, "state_machines", []) or []:
            states = getattr(sm, "_states", {})
            for _trie, destinations in states.values():
                if destinations:
                    return False
        return True

    def _is_engine_fast_eligible(self) -> bool:
        def miss(reason: str) -> bool:
            if os.environ.get("FUSEKIT_DBG_ENGINE_ELIGIBLE"):
                print(
                    "[fusekit-engine-eligible] "
                    f"miss={reason} "
                    f"skip_attr={getattr(self, '_fk_skip_logprobs', None)!r} "
                    f"skip_current={_ENGINE_SKIP_LOGPROBS_CURRENT_STEP!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return False

        if _env_flag("FUSEKIT_DISABLE_ENGINE_FAST"):
            return miss("disabled")
        if _engine_style() in ("off", "disabled", "raw"):
            return miss("style_off")
        if not _is_omlx_step_style() and not _env_flag("FUSEKIT_ENABLE_ENGINE_FAST"):
            return miss("style_not_enabled")
        model = getattr(self, "model", None)
        if _model_has_active_mtp(model):
            return miss("mtp_active")
        if len(getattr(self, "uids", []) or []) != 1:
            return miss("uids")
        samplers = getattr(self, "samplers", []) or []
        if any(samplers):
            if len(samplers) != 1 or not _sampler_is_greedy_argmax(
                samplers[0], missing_is_greedy=False
            ):
                return miss("sampler")
        else:
            sampler = getattr(self, "fallback_sampler", None)
            if not _sampler_is_greedy_argmax(sampler, missing_is_greedy=False):
                return miss("fallback_sampler")
        if any(getattr(self, "logits_processors", []) or []):
            return miss("logits_processors")
        if not (
            getattr(self, "_fk_skip_logprobs", False)
            or _ENGINE_SKIP_LOGPROBS_CURRENT_STEP
        ):
            return miss("logprobs")
        inner = getattr(model, "_fk_original", None)
        if inner is None:
            return miss("not_wrapped")
        if getattr(inner, "_fk_lm", None) is not None:
            return True
        if getattr(model, "_fk_native_handle", None) is None:
            return miss("no_native")
        if getattr(model, "_fk_dispatch_failed", False):
            return miss("dispatch_failed")
        return True

    def _engine_direct_head_argmax_enabled(self) -> bool:
        if not _env_flag("FUSEKIT_ENABLE_DIRECT_HEAD_ARGMAX"):
            return False
        if _env_flag("FUSEKIT_DISABLE_DIRECT_HEAD_ARGMAX"):
            return False
        if _env_flag("FUSEKIT_DISABLE_GPU_ARGMAX_TOKENS"):
            return False
        # Entered only by greedy-argmax call sites. Keep it opt-in:
        # L512 profiles on q4/q8/bf16 show the direct scan/finalize path
        # is not faster than the normal logits + MLX argmax path.
        return _is_engine_fast_eligible(self)

    def _engine_needs_direct_head_rebuild(self) -> bool:
        if not _engine_direct_head_argmax_enabled(self):
            return False
        model = getattr(self, "model", None)
        inner = getattr(model, "_fk_original", None)
        fk = getattr(inner, "_fk_lm", None) if inner is not None else None
        return fk is not None and not bool(fk.get("direct_head_argmax", False))

    def _engine_drain_limit(self=None) -> int:
        if _env_flag("FUSEKIT_DISABLE_ENGINE_DRAIN"):
            return 1
        raw = os.environ.get("FUSEKIT_ENGINE_DRAIN")
        if raw is None:
            gb = getattr(self, "_generation_batch", None)
            queued = getattr(gb, "_fk_engine_queue", None) or []
            return max(1, 1 + len(queued))
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    def _mtp_queue_drain_limit(self=None) -> int:
        if _env_flag("FUSEKIT_DISABLE_MTP_QUEUE_DRAIN"):
            return 1
        raw = os.environ.get("FUSEKIT_MTP_QUEUE_DRAIN")
        if raw is None:
            # Accepted MTP cycles already queued one or more confirmed tokens.
            # Drain only that existing queue in the current scheduler tick; do
            # not start another verify cycle here. This removes avoidable host
            # scheduling round-trips without changing acceptance semantics.
            return 8
        try:
            return max(1, int(raw))
        except ValueError:
            return 8

    def _serve_engine_queue(self, inputs, mx):
        queue = getattr(self, "_fk_engine_queue", None) or []
        if not queue:
            return False
        self._next_tokens = mx.array([int(queue.pop(0))], dtype=mx.uint32)
        self._next_logprobs = [None]
        self._fk_engine_queue = queue
        return True

    def _finish_engine_async(self, mx) -> bool:
        pending = getattr(self, "_fk_engine_async_pending", None)
        if not pending:
            return True

        from . import _fusekit_runtime_native as _native

        if os.environ.get("FUSEKIT_DBG_ENGINE_ASYNC"):
            import time

            started = time.perf_counter()
            gpu_s = float(_native.wait_pending_icb(int(pending["handle"])))
            wait_us = (time.perf_counter() - started) * 1_000_000.0
            age_us = (
                (started - float(pending.get("submitted_at", started))) * 1_000_000.0
            )
            print(
                "[fusekit-engine-async] "
                f"wait_us={wait_us:.2f} gpu_us={gpu_s * 1_000_000.0:.2f} "
                f"age_us={age_us:.2f}",
                file=sys.stderr,
                flush=True,
            )
        else:
            _native.wait_pending_icb(int(pending["handle"]))

        next_token = int(pending["next_token_view"][0])
        self._next_tokens = mx.array([next_token], dtype=mx.uint32)
        self._next_logprobs = [None]
        self._fk_engine_async_pending = None
        return True

    def _drain_engine_fast_state(gb, mx) -> None:
        if gb is None:
            return
        try:
            _finish_engine_async(gb, mx)
        finally:
            gb._fk_engine_queue = []
            gb._fk_engine_async_pending = None
            gb.__dict__.pop("_fk_seeded_first_tokens", None)
            gb.__dict__.pop("_fk_seeded_first_logprobs", None)
            if len(getattr(gb, "uids", []) or []) == 0:
                # The final visible token may have already submitted the next
                # pipelined decode.  Draining that work writes _next_tokens even
                # though GenerationBatch.filter([]) has made the batch idle, so
                # restore the same empty state mlx-lm expects before reuse.
                gb._next_tokens = None
                gb._next_logprobs = []
                gb._current_tokens = None
                gb._current_logprobs = []

    def _reset_idle_engine_fast_state(batch_generator: Any) -> None:
        """Clear stale fast-token state before a new request enters an idle engine."""
        if batch_generator is None:
            return
        try:
            generation_batch = getattr(batch_generator, "_generation_batch", None)
            prompt_batch = getattr(batch_generator, "_prompt_batch", None)
            if generation_batch is None:
                return
            if len(getattr(generation_batch, "uids", []) or []) != 0:
                return
            if prompt_batch is not None and len(getattr(prompt_batch, "uids", []) or []) != 0:
                return
            generation_batch._fk_engine_queue = []
            generation_batch._fk_engine_async_pending = None
            generation_batch._next_tokens = None
            generation_batch._next_logprobs = []
            generation_batch._current_tokens = None
            generation_batch._current_logprobs = []
            generation_batch.__dict__.pop("_fk_seeded_first_tokens", None)
            generation_batch.__dict__.pop("_fk_seeded_first_logprobs", None)
            model = getattr(batch_generator, "model", None)
            if model is not None:
                model.__dict__.pop("_fk_last_argmax_tokens", None)
                model.__dict__.pop("_fk_mtp_split_tokens_trusted", None)
            inner = getattr(model, "_fk_original", None)
            if inner is not None:
                inner.__dict__.pop("_fk_last_argmax_tokens", None)
                inner.__dict__.pop("_fk_mtp_split_tokens_trusted", None)
                inner.__dict__.pop("_fk_lm", None)
            dispatcher = getattr(model, "_fk_dispatcher", None)
            if dispatcher is not None:
                dispatcher._cache_sig = None
                dispatcher._icb_id = None
        except Exception:
            return

    def _ensure_engine_full_capacity(self, remaining: int) -> tuple[bool, str | None, bool]:
        offset = _engine_cache_offset(self.prompt_cache)
        if offset is None:
            return False, "offset_none", False
        before = _cache_capacity(self.prompt_cache)
        ok, reason = _ensure_cache_capacity(self.prompt_cache, offset + max(1, remaining))
        after = _cache_capacity(self.prompt_cache)
        return ok, reason, before is not None and after is not None and int(before) != int(after)

    def _submit_engine_async(self, inputs) -> bool:
        if _env_flag("FUSEKIT_DISABLE_ENGINE_ASYNC"):
            return False
        if "FUSEKIT_ENGINE_ASYNC" in os.environ and not _env_flag("FUSEKIT_ENGINE_ASYNC"):
            return False
        if not _stopless(self):
            return False
        remaining = int(self.max_tokens[0]) - int(self._num_tokens[0])
        if remaining <= 1:
            return False

        model = self.model
        inner = model._fk_original
        fk = getattr(inner, "_fk_lm", None)
        dispatcher = getattr(model, "_fk_dispatcher", None)
        dispatches = getattr(dispatcher, "_last_dispatches", None)
        if fk is None or not dispatches:
            return False
        if _engine_needs_direct_head_rebuild(self):
            return False

        offset = _engine_cache_offset(self.prompt_cache)
        if offset is None:
            return False
        ok, reason, capacity_changed = _ensure_engine_full_capacity(self, remaining)
        if not ok:
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise RuntimeError(
                    "fusekit: engine async cannot guarantee KV cache capacity "
                    f"(offset={offset}: {reason})"
                )
            return False
        if capacity_changed:
            return False
        capacity = _cache_capacity(self.prompt_cache)
        if capacity is None or offset + 1 > capacity:
            return False

        from . import _fusekit_runtime_native as _native

        submit = getattr(_native, "step_decode_shared_state", None)
        if submit is None:
            return False
        if getattr(dispatcher, "_host_embed_lookup", False):
            mx = __import__("mlx.core", fromlist=["core"])
            row = (
                dispatcher._embed_module(inputs)
                .reshape((dispatcher._hidden,))
                .astype(dispatcher._act_dt)
            )
            mx.eval(row)
            ctypes.memmove(
                dispatcher._array_ptr(dispatcher._x_in),
                dispatcher._array_ptr(row),
                dispatcher._x_in.nbytes,
            )
        seed = int(inputs.item())
        submit(
            fk["handle"],
            fk["icb_id"],
            fk["token_id_ptr"],
            seed,
            fk["layer_offset_ptr"],
            offset,
            fk["layer_capacity_ptr"],
            capacity,
            fk["layer_kv_len_ptr"],
            offset + 1,
            fk["step_counter_ptr"],
            fk["hist_counter_ptr"],
            fk.get("barrier_counter_ptrs") or [],
        )
        _engine_set_cache_offset(self.prompt_cache, offset + 1)

        pending = {
            "handle": fk["handle"],
            "next_token_view": fk["next_token_view"],
            "dtype": inputs.dtype,
        }
        if os.environ.get("FUSEKIT_DBG_ENGINE_ASYNC"):
            import time

            pending["submitted_at"] = time.perf_counter()
            print(
                "[fusekit-engine-async] "
                f"submit offset={offset} seed={seed} capacity={capacity}",
                file=sys.stderr,
                flush=True,
            )
        self._fk_engine_async_pending = pending
        self._next_logprobs = [None]
        return True

    def _run_engine_chunk(self, inputs, mx) -> bool:
        def miss(reason: str) -> bool:
            if os.environ.get("FUSEKIT_DBG_ENGINE_CHUNK"):
                n = int(getattr(self, "_fk_engine_chunk_miss_debug", 0) or 0)
                if n < 8:
                    print(
                        "[fusekit-engine-chunk-miss] "
                        f"reason={reason} "
                        f"num_tokens={getattr(self, '_num_tokens', None)} "
                        f"max_tokens={getattr(self, 'max_tokens', None)}",
                        file=sys.stderr,
                        flush=True,
                    )
                self._fk_engine_chunk_miss_debug = n + 1
            return False

        if _env_flag("FUSEKIT_DISABLE_ENGINE_CHUNK"):
            return miss("disabled")
        if not _stopless(self):
            return miss("not_stopless")
        model = self.model
        inner = model._fk_original
        fk = getattr(inner, "_fk_lm", None)
        if fk is None:
            return miss("no_fk")
        if _engine_needs_direct_head_rebuild(self):
            return miss("direct_head_rebuild")
        remaining = int(self.max_tokens[0]) - int(self._num_tokens[0])
        if remaining <= 0:
            return miss(f"remaining={remaining}")
        chunk_cap = int(fk.get("chunk_capacity", 1) or 1)
        chunk = min(chunk_cap, remaining)
        if chunk <= 1:
            return miss(f"chunk={chunk}")
        offset = _engine_cache_offset(self.prompt_cache)
        if offset is None:
            return miss("offset_none")
        ok, reason, capacity_changed = _ensure_engine_full_capacity(self, remaining)
        if not ok:
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise RuntimeError(
                    "fusekit: engine chunk cannot guarantee KV cache capacity "
                    f"(offset={offset}, remaining={remaining}: {reason})"
                )
            return miss(f"capacity_ensure_failed:{reason}")
        if capacity_changed:
            return miss("capacity_grew")
        capacity = _cache_capacity(self.prompt_cache)
        if capacity is None:
            return miss("capacity_none")
        if capacity is None:
            return miss("capacity_none")
        chunk = min(chunk, max(0, int(capacity) - int(offset)))
        if chunk <= 1:
            return miss(f"capacity_guard offset={offset} chunk={chunk} capacity={capacity}")
        chunk_cap = int(fk.get("chunk_capacity", chunk_cap) or chunk_cap)

        from . import _fusekit_runtime_native as _native

        seed = int(inputs.item())
        ctypes.c_uint32.from_address(fk["layer_offset_ptr"]).value = offset
        ctypes.c_uint32.from_address(fk["layer_kv_len_ptr"]).value = offset + 1
        ctypes.c_uint32.from_address(fk["layer_capacity_ptr"]).value = capacity
        barrier_ptrs = fk.get("barrier_counter_ptrs") or []
        chunk_mode = os.environ.get("FUSEKIT_ENGINE_CHUNK_MODE", "auto").lower()
        if chunk_mode == "auto":
            chunk_mode = "gpu" if not barrier_ptrs else "host"
        if (
            chunk_mode in ("gpu", "chain", "gpu-chain")
            and remaining > chunk_cap
            and not _env_flag("FUSEKIT_ENGINE_ALLOW_LONG_GPU_CHAIN")
        ):
            if os.environ.get("FUSEKIT_DBG_ENGINE_CHUNK"):
                print(
                    "[fusekit-engine-chunk] "
                    "downgrade mode=gpu->host reason=long_multi_chunk "
                    f"remaining={remaining} chunk_cap={chunk_cap}",
                    file=sys.stderr,
                    flush=True,
                )
            chunk_mode = "host"
        if chunk_mode in ("gpu", "chain", "gpu-chain"):
            if barrier_ptrs:
                _native.decode_chunk_gpu_chain_reset_barriers(
                    fk["handle"],
                    fk["icb_id"],
                    fk["step_counter_ptr"],
                    fk["hist_counter_ptr"],
                    fk["next_token_ptr"],
                    offset,
                    seed,
                    chunk,
                    barrier_ptrs,
                )
            else:
                _native.decode_chunk_gpu_chain(
                    fk["handle"],
                    fk["icb_id"],
                    fk["step_counter_ptr"],
                    fk["hist_counter_ptr"],
                    fk["next_token_ptr"],
                    offset,
                    seed,
                    chunk,
                )
        else:
            _native.decode_chunk_host_loop(
                fk["handle"],
                fk["icb_id"],
                fk["step_counter_ptr"],
                fk["hist_counter_ptr"],
                fk["token_id_ptr"],
                fk["next_token_ptr"],
                fk["layer_offset_ptr"],
                fk["layer_capacity_ptr"],
                fk["layer_kv_len_ptr"],
                fk["history_ptr"],
                capacity,
                offset,
                seed,
                chunk,
                barrier_ptrs,
            )
        history_view = (ctypes.c_uint32 * int(fk["chunk_capacity"])).from_address(
            fk["history_ptr"]
        )
        history = [int(history_view[i]) for i in range(chunk)]
        if os.environ.get("FUSEKIT_DBG_ENGINE_CHUNK"):
            print(
                "[fusekit-engine-chunk] "
                f"mode={chunk_mode} offset={offset} seed={seed} chunk={chunk} "
                f"history={history} next={int(fk['next_token_view'][0])}",
                file=sys.stderr,
                flush=True,
            )
        _engine_set_cache_offset(self.prompt_cache, offset + chunk)
        self._next_tokens = mx.array([history[0]], dtype=mx.uint32)
        self._next_logprobs = [None]
        self._fk_engine_queue = history[1:]
        return True

    def _engine_prefers_chunk_first(self) -> bool:
        if (
            _env_flag("FUSEKIT_FORCE_ENGINE_CHUNK")
            or "FUSEKIT_ENGINE_CHUNK_MODE" in os.environ
            or _env_flag("FUSEKIT_ENGINE_ALLOW_LONG_GPU_CHAIN")
        ):
            return True
        model = self.model
        inner = getattr(model, "_fk_original", None)
        fk = getattr(inner, "_fk_lm", None) if inner is not None else None
        if not isinstance(fk, dict) or not bool(fk.get("dense_bf16")):
            return False
        hidden = int(fk.get("hidden_size", 0) or 0)
        vocab = int(fk.get("vocab_size", 0) or 0)
        return hidden >= 4096 and vocab >= 200_000

    def _engine_profile_points() -> set[int]:
        raw = os.environ.get("FUSEKIT_PROFILE_KERNELS_AT", "")
        points: set[int] = set()
        if not raw:
            return points
        for part in raw.replace(",", " ").split():
            try:
                points.add(int(part))
            except ValueError:
                pass
        return points

    def _run_engine_profile_step(self, inputs, mx) -> bool:
        points = _engine_profile_points()
        if not points:
            return False
        queue = getattr(self, "_fk_engine_queue", None) or []
        if queue:
            return False
        try:
            token_index = int(self._num_tokens[0])
        except Exception:
            return False
        if token_index not in points:
            return False
        if not _is_engine_fast_eligible(self):
            return False

        model = self.model
        inner = model._fk_original
        dispatcher = getattr(model, "_fk_dispatcher", None)
        fk = getattr(inner, "_fk_lm", None)
        if dispatcher is None or fk is None:
            return False

        remaining = int(self.max_tokens[0]) - token_index
        ok, reason, _capacity_changed = _ensure_engine_full_capacity(self, remaining)
        if not ok:
            if os.environ.get("FUSEKIT_DBG_ENGINE_CHUNK"):
                print(
                    "[fusekit-engine-profile-at-miss] "
                    f"token_index={token_index} reason=capacity:{reason}",
                    file=sys.stderr,
                    flush=True,
                )
            return False

        offset = _engine_cache_offset(self.prompt_cache)
        if offset is None:
            return False
        print(
            "[fusekit-engine-profile-at] "
            f"token_index={token_index} cache_offset={offset}",
            flush=True,
        )
        dispatcher._profile_kernels_once = True
        dispatcher._profile_kernels_skip = 0
        dispatcher._profile_context = {
            "token_index": int(token_index),
            "cache_offset": int(offset),
        }
        direct_head = (
            getattr(inner, "_fk_lm", None) is not None
            and _engine_direct_head_argmax_enabled(self)
        )
        restore_direct = _direct_head_argmax_scope(model, direct_head)
        try:
            logits = model(inputs[:, None], cache=self.prompt_cache)
        finally:
            restore_direct()

        argmax_tokens = getattr(model, "_fk_last_argmax_tokens", None)
        if argmax_tokens is None:
            argmax_tokens = getattr(inner, "_fk_last_argmax_tokens", None)
        if argmax_tokens is not None:
            next_token = int(mx.array(argmax_tokens).reshape(-1)[0].item())
            self._next_tokens = mx.array([next_token], dtype=mx.uint32)
        else:
            if isinstance(logits, tuple):
                logits = logits[0]
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(inputs.dtype)
            mx.eval(next_token)
            self._next_tokens = next_token
        self._next_logprobs = [None]
        return True

    def _patched_prompt(self, tokens):
        mx = __import__("mlx.core", fromlist=["core"])
        debug_prefill = os.environ.get("FUSEKIT_DBG_ENGINE_PREFILL")
        if debug_prefill:
            print(
                "[fusekit-engine-prefill] "
                f"uids={len(getattr(self, 'uids', []) or [])} "
                f"model_wrapped={getattr(self.model, '_fk_original', None) is not None} "
                f"disable={_env_flag('FUSEKIT_DISABLE_ENGINE_PREFILL_NO_LOGITS')}",
                file=sys.stderr,
                flush=True,
            )
        if (
            original_prompt is None
            or _env_flag("FUSEKIT_DISABLE_ENGINE_PREFILL_NO_LOGITS")
            or getattr(self.model, "_fk_original", None) is None
        ):
            return original_prompt(self, tokens)
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return

        # Same cache/token bookkeeping as mlx-lm, but prompt chunks are marked
        # no-logits because their LM-head output is discarded by the engine.
        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)

        if max_padding > 0:
            tokens = module._right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            self.model(
                tokens[:, :n_to_process],
                cache=self.prompt_cache,
                _fusekit_prefill_no_logits=True,
            )
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]

        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

    def _patched_step(self):
        mx = __import__("mlx.core", fromlist=["core"])
        _finish_engine_async(self, mx)
        if not _is_engine_fast_eligible(self):
            return original_step(self)

        model = self.model
        inner = model._fk_original

        cache_offset = _engine_cache_offset(self.prompt_cache)
        cache_capacity = _cache_capacity(self.prompt_cache)
        if (
            cache_offset is not None
            and cache_capacity is not None
            and int(cache_offset) >= int(cache_capacity)
        ):
            return original_step(self)

        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        if not _serve_engine_queue(self, inputs, mx):
            profiled = _run_engine_profile_step(self, inputs, mx)
            ran_fast_step = profiled
            if not ran_fast_step:
                chunk_first = _engine_prefers_chunk_first(self)
                if chunk_first:
                    ran_fast_step = _run_engine_chunk(self, inputs, mx) or _submit_engine_async(
                        self, inputs
                    )
                else:
                    ran_fast_step = _submit_engine_async(self, inputs) or _run_engine_chunk(
                        self, inputs, mx
                    )
            if not ran_fast_step:
                # Calls FusekitWrapper.__call__, which advances the live
                # cache by one token and writes the next greedy token into
                # `next_token_out`.
                remaining = int(self.max_tokens[0]) - int(self._num_tokens[0])
                ok, reason, _capacity_changed = _ensure_engine_full_capacity(
                    self, remaining
                )
                if not ok and _env_flag("FUSEKIT_STRICT_DISPATCH"):
                    raise RuntimeError(
                        "fusekit: engine fallback cannot guarantee KV cache "
                        f"capacity ({reason})"
                    )
                direct_head = (
                    getattr(inner, "_fk_lm", None) is not None
                    and _engine_direct_head_argmax_enabled(self)
                )
                if os.environ.get("FUSEKIT_DBG_ENGINE_DIRECT"):
                    print(
                        "[fusekit-engine-direct] "
                        f"enabled={direct_head} "
                        f"skip_attr={getattr(self, '_fk_skip_logprobs', None)!r} "
                        f"skip_current={_ENGINE_SKIP_LOGPROBS_CURRENT_STEP!r} "
                        f"has_fk={getattr(inner, '_fk_lm', None) is not None} "
                        f"native={getattr(model, '_fk_native_handle', None) is not None}",
                        file=sys.stderr,
                        flush=True,
                    )
                restore_direct = _direct_head_argmax_scope(model, direct_head)
                try:
                    logits = model(inputs[:, None], cache=self.prompt_cache)
                finally:
                    restore_direct()
                fk = getattr(inner, "_fk_lm", None)
                argmax_tokens = getattr(model, "_fk_last_argmax_tokens", None)
                if argmax_tokens is None:
                    argmax_tokens = getattr(inner, "_fk_last_argmax_tokens", None)
                if argmax_tokens is not None:
                    next_token = int(mx.array(argmax_tokens).reshape(-1)[0].item())
                    if os.environ.get("FUSEKIT_DBG_RESPONSE_TOKENS"):
                        print(
                            "[fusekit-step-token] "
                            f"input={int(inputs.item()) if getattr(inputs, 'size', 0) == 1 else inputs.tolist()} "
                            f"next={next_token} source=fk_argmax",
                            file=sys.stderr,
                            flush=True,
                        )
                    self._next_tokens = mx.array([next_token], dtype=mx.uint32)
                else:
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(inputs.dtype)
                    mx.eval(next_token)
                    if os.environ.get("FUSEKIT_DBG_RESPONSE_TOKENS"):
                        print(
                            "[fusekit-step-token] "
                            f"input={int(inputs.item()) if getattr(inputs, 'size', 0) == 1 else inputs.tolist()} "
                            f"next={int(next_token.item()) if getattr(next_token, 'size', 0) == 1 else next_token.tolist()} source=mlx",
                            file=sys.stderr,
                            flush=True,
                        )
                    self._next_tokens = next_token
                self._next_logprobs = [None]

        # Match the original contract: return the token generated by the
        # previous step and append it to the per-sequence token history.
        eval_args = [inputs] + [
            lp for lp in (self._current_logprobs or []) if lp is not None
        ]
        mx.eval(*eval_args)
        out_tokens = inputs.tolist()
        for sti, ti in zip(self.tokens, out_tokens):
            sti.append(ti)
        return out_tokens, self._current_logprobs

    def _patch_seed_lists_after_extend(self, batch, old_len: int) -> None:
        if original_extend is None:
            return
        before_tokens = list(getattr(self, "_fk_seeded_first_tokens", [None] * old_len))
        before_lps = list(getattr(self, "_fk_seeded_first_logprobs", [None] * old_len))
        add_tokens = list(
            getattr(batch, "_fk_seeded_first_tokens", [None] * len(getattr(batch, "uids", []) or []))
        )
        add_lps = list(
            getattr(batch, "_fk_seeded_first_logprobs", [None] * len(getattr(batch, "uids", []) or []))
        )
        merged_tokens = before_tokens + add_tokens
        merged_lps = before_lps + add_lps
        if any(t is not None for t in merged_tokens):
            self.__dict__["_fk_seeded_first_tokens"] = merged_tokens
            self.__dict__["_fk_seeded_first_logprobs"] = merged_lps
        else:
            self.__dict__.pop("_fk_seeded_first_tokens", None)
            self.__dict__.pop("_fk_seeded_first_logprobs", None)

    def _patched_extend(self, batch):
        old_len = len(getattr(self, "uids", []) or [])
        result = original_extend(self, batch)
        _patch_seed_lists_after_extend(self, batch, old_len)
        return result

    def _patched_next(self):
        mx = __import__("mlx.core", fromlist=["core"])
        seeded_tokens = getattr(self, "_fk_seeded_first_tokens", None)
        if (
            original_next is None
            or not seeded_tokens
            or len(seeded_tokens) != len(getattr(self, "uids", []) or [])
            or not all(t is not None for t in seeded_tokens)
        ):
            return original_next(self)

        seeded_logprobs = list(
            getattr(self, "_fk_seeded_first_logprobs", [None] * len(seeded_tokens))
        )
        keep = []
        responses = []
        for i, token in enumerate(seeded_tokens):
            finish_reason = None
            match_sequence = None
            current_state = None
            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            self._matcher_states[i], match_sequence, current_state = (
                self.state_machines[i].match(self._matcher_states[i], int(token))
            )
            if match_sequence is not None and current_state is None:
                finish_reason = "stop"

            if finish_reason is None:
                keep.append(i)
                prompt_cache = None
                all_tokens = None
            else:
                prompt_cache = self.extract_cache(i)
                all_tokens = self.tokens[i]

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=int(token),
                    logprobs=seeded_logprobs[i] if i < len(seeded_logprobs) else None,
                    finish_reason=finish_reason,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=prompt_cache,
                    all_tokens=all_tokens,
                )
            )

        self.__dict__.pop("_fk_seeded_first_tokens", None)
        self.__dict__.pop("_fk_seeded_first_logprobs", None)
        if keep:
            # The seeded first token is returned without calling the normal
            # GenerationBatch._step path. Seed the next decode step explicitly;
            # otherwise a reused oMLX engine can carry a stale `next_token_out`
            # from the previous request into the new request.
            self._next_tokens = mx.array(
                [int(seeded_tokens[i]) for i in keep], dtype=mx.uint32
            )
            self._next_logprobs = [None for _ in keep]
        if len(keep) < len(self.uids) and original_filter is not None:
            original_filter(self, keep)
        return responses

    def _patched_generate(self, tokens):
        generation = original_generate(self, tokens)
        if not tokens or not all(
            isinstance(t, _FusekitSeededTokenList)
            and hasattr(t, "_fusekit_seeded_first_token")
            for t in tokens
        ):
            return generation
        generation.__dict__["_fk_seeded_first_tokens"] = [
            int(getattr(t, "_fusekit_seeded_first_token")) for t in tokens
        ]
        generation.__dict__["_fk_seeded_first_logprobs"] = [
            getattr(t, "_fusekit_seeded_first_logprobs", None) for t in tokens
        ]
        return generation

    def _patched_next_generated(self):
        engine_limit = _engine_drain_limit(self)
        mtp_limit = _mtp_queue_drain_limit()
        if original_next_generated is None:
            return original_next_generated(self)

        mx = __import__("mlx.core", fromlist=["core"])
        with mx.stream(self._stream):
            while True:
                prompt_responses, generation_responses = self._next()
                if not generation_responses and prompt_responses:
                    continue
                break

            responses = list(generation_responses)
            if not responses:
                return responses

            # MTP accept cycles often enqueue already-confirmed tokens. Drain
            # only that existing queue; never start a new verify cycle here.
            while len(responses) < mtp_limit:
                gb = getattr(self, "_generation_batch", None)
                state = getattr(gb, "_omlx_mtp_state", None) if gb is not None else None
                queue = getattr(state, "queue", None) if state is not None else None
                if not queue:
                    break
                _prompt, more = self._next()
                if not more:
                    break
                responses.extend(more)
                if any(getattr(r, "finish_reason", None) is not None for r in more):
                    break

            # Optional throughput mode: keep advancing native MTP cycles within
            # the same engine tick until the response batch is full. This trades
            # per-token streaming granularity for fewer host scheduler round
            # trips; _next still owns stop/max-token semantics.
            if _env_flag("FUSEKIT_MTP_QUEUE_DRAIN_AGGRESSIVE") and not any(
                getattr(r, "finish_reason", None) is not None for r in responses
            ):
                while len(responses) < mtp_limit:
                    _prompt, more = self._next()
                    if not more:
                        break
                    responses.extend(more)
                    if any(getattr(r, "finish_reason", None) is not None for r in more):
                        break

            engine_limit = max(engine_limit, _engine_drain_limit(self))
            # Only drain already queued FuseKit chunk tokens. Do not start
            # another GPU chunk inside the same scheduler tick; the next tick
            # can seed it through the normal GenerationBatch.next contract.
            while len(responses) < engine_limit:
                gb = getattr(self, "_generation_batch", None)
                if gb is None or len(gb) != 1 or not _is_engine_fast_eligible(gb):
                    break
                if not (getattr(gb, "_fk_engine_queue", None) or []):
                    break
                _prompt, more = self._next()
                if not more:
                    break
                responses.extend(more)
                if any(getattr(r, "finish_reason", None) is not None for r in more):
                    break

            if os.environ.get("FUSEKIT_DBG_ENGINE_DRAIN") and len(responses) > 1:
                gb = getattr(self, "_generation_batch", None)
                print(
                    "[fusekit-engine-drain] "
                    f"responses={len(responses)} "
                    f"queued={len(getattr(gb, '_fk_engine_queue', []) or []) if gb is not None else 0}",
                    file=sys.stderr,
                    flush=True,
                )
            if any(getattr(r, "finish_reason", None) is not None for r in responses):
                _drain_engine_fast_state(getattr(self, "_generation_batch", None), mx)
            if os.environ.get("FUSEKIT_DBG_RESPONSE_TOKENS"):
                gb = getattr(self, "_generation_batch", None)
                print(
                    "[fusekit-responses] "
                    f"uids={list(getattr(gb, 'uids', []) or []) if gb is not None else None} "
                    f"tokens={[int(getattr(r, 'token', -1)) for r in responses]} "
                    f"finish={[getattr(r, 'finish_reason', None) for r in responses]}",
                    file=sys.stderr,
                    flush=True,
                )
            return responses

    def _patched_insert_segments(self, *args, **kwargs):
        _reset_idle_engine_fast_state(self)
        return original_insert_segments(self, *args, **kwargs)

    cls._step = _patched_step
    if original_next is not None:
        cls.next = _patched_next
    if original_extend is not None:
        cls.extend = _patched_extend
    if prompt_cls is not None and original_prompt is not None:
        prompt_cls.prompt = _patched_prompt
        prompt_cls._fk_prefill_no_logits_patched = True
    if prompt_cls is not None and original_generate is not None:
        prompt_cls.generate = _patched_generate
        prompt_cls._fk_full_prefill_seed_patched = True
    if batch_cls is not None and original_next_generated is not None:
        batch_cls.next_generated = _patched_next_generated
    if batch_cls is not None and original_insert_segments is not None:
        batch_cls.insert_segments = _patched_insert_segments
    cls._fk_engine_fast_patched = True


def _wrap_omlx_scheduler(module) -> None:
    """Mark when oMLX can safely skip per-token logprobs.

    GenerationBatch does not know whether the API request asked for
    logprobs. Scheduler does, so set a private flag immediately before
    each engine step. If any running request needs logprobs, FuseKit's
    GenerationBatch wrapper automatically falls back to mlx-lm's path.
    """
    global _OMLX_SCHEDULER_PATCHED, _ENGINE_SKIP_LOGPROBS_CURRENT_STEP
    _patch_omlx_mtp_sampler_accept_cache(scheduler_module=module)
    if not hasattr(module, "Scheduler"):
        return
    scheduler_cls = module.Scheduler
    if (
        hasattr(scheduler_cls, "_do_external_prefill")
        and not getattr(scheduler_cls, "_fk_external_prefill_no_logits_patched", False)
    ):
        original_external_prefill = scheduler_cls._do_external_prefill

        def _patched_external_prefill(self, *args, **kwargs):
            global _ENGINE_EXTERNAL_PREFILL_NO_LOGITS_CURRENT
            model = getattr(self, "model", None)
            _reset_idle_engine_fast_state(getattr(self, "batch_generator", None))
            active = (
                getattr(model, "_fk_original", None) is not None
                and not _env_flag("FUSEKIT_DISABLE_ENGINE_PREFILL_NO_LOGITS")
            )
            if active:
                request = args[0] if len(args) >= 1 else kwargs.get("request")
                tokens = args[1] if len(args) >= 2 else kwargs.get("tokens")
                existing_cache = args[2] if len(args) >= 3 else kwargs.get("existing_cache")
                vlm_embeds = kwargs.get("vlm_embeds")
                if request is not None and isinstance(tokens, list):
                    stats = getattr(model, "_fk_dispatch_stats", None)
                    seeded = _engine_full_prefill_seed_candidate(
                        self,
                        request,
                        tokens,
                        existing_cache,
                        vlm_embeds,
                        stats,
                    )
                    if seeded is not None:
                        if stats is not None:
                            stats["engine_full_prefill_seed"] = (
                                stats.get("engine_full_prefill_seed", 0) + 1
                            )
                        return seeded
            prev = _ENGINE_EXTERNAL_PREFILL_NO_LOGITS_CURRENT
            if active:
                _ENGINE_EXTERNAL_PREFILL_NO_LOGITS_CURRENT = True
            try:
                return original_external_prefill(self, *args, **kwargs)
            finally:
                _ENGINE_EXTERNAL_PREFILL_NO_LOGITS_CURRENT = prev

        scheduler_cls._do_external_prefill = _patched_external_prefill
        scheduler_cls._fk_external_prefill_no_logits_patched = True

    if _OMLX_SCHEDULER_PATCHED:
        return
    original = scheduler_cls.step

    def _patched_scheduler_step(self, *args, **kwargs):
        global _ENGINE_SKIP_LOGPROBS_CURRENT_STEP
        bg = getattr(self, "batch_generator", None)
        gb = getattr(bg, "_generation_batch", None)
        skip_logprobs = False
        reqs = []
        try:
            reqs = list(getattr(self, "running", {}).values()) + list(
                getattr(self, "waiting", [])
            )
            skip_logprobs = bool(reqs) and all(
                not getattr(req.sampling_params, "logprobs", False) for req in reqs
            )
        except Exception:
            skip_logprobs = False
        if os.environ.get("FUSEKIT_DBG_SKIP_LOGPROBS"):
            try:
                print(
                    "[fusekit-skip-logprobs] "
                    f"reqs={len(reqs)} "
                    f"running={len(getattr(self, 'running', {}) or {})} "
                    f"waiting={len(getattr(self, 'waiting', []) or [])} "
                    f"skip={skip_logprobs}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
        mtp_bg = None
        try:
            _patch_omlx_mtp_greedy_detection()
            from omlx.patches.mlx_lm_mtp import batch_generator as mtp_bg

            mtp_bg._fk_skip_logprobs_current_step = skip_logprobs
        except Exception:
            mtp_bg = None
        _ENGINE_SKIP_LOGPROBS_CURRENT_STEP = skip_logprobs
        if gb is not None:
            try:
                gb._fk_skip_logprobs = all(
                    not getattr(req.sampling_params, "logprobs", False)
                    for req in getattr(self, "running", {}).values()
                )
            except Exception:
                gb._fk_skip_logprobs = False
        try:
            return original(self, *args, **kwargs)
        finally:
            if mtp_bg is not None:
                try:
                    mtp_bg._fk_skip_logprobs_current_step = False
                except Exception:
                    pass
            # oMLX can return from Scheduler.step before the newly-created
            # GenerationBatch executes its first `_step`. Keep the latest
            # scheduler decision visible; the next Scheduler.step overwrites it
            # before a request with logprobs can enter the fast path.

    scheduler_cls.step = _patched_scheduler_step
    _OMLX_SCHEDULER_PATCHED = True


def _fast_generate_step(prompt, model, inner, kw, _orig):
    """Greedy decode generator that bypasses mlx-lm's per-token Python
    overhead by reading next-token straight from the in-ICB argmax
    uniform. Falls back to `_orig` for any edge case we don't handle.
    """
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from . import _fusekit_runtime_native as _native

    # Prefill via the standard path — handles cache allocation,
    # optional KV quantization, and our patched_call's L>1 MLX fall-back.
    max_tokens = kw.get("max_tokens", 256)
    if max_tokens < 0:
        warnings.warn(
            "fusekit: fast path requires a bounded max_tokens; "
            "falling back to mlx-lm."
        )
        yield from _orig(prompt, model, **kw)
        return
    kv_bits = kw.get("kv_bits")
    kv_group_size = kw.get("kv_group_size", 64)
    quantized_kv_start = kw.get("quantized_kv_start", 0)
    if kv_bits is not None and quantized_kv_start != 0:
        warnings.warn(
            "fusekit: fast path supports quantized KV cache only from "
            "quantized_kv_start=0; falling back to mlx-lm."
        )
        yield from _orig(prompt, model, **kw)
        return
    p = prompt.reshape(-1) if getattr(prompt, "ndim", 1) != 1 else prompt
    total_prompt_tokens = len(p)
    requested_max_kv_size = kw.get("max_kv_size")
    try:
        required_max_kv_size = int(total_prompt_tokens) + int(max_tokens)
    except Exception:
        required_max_kv_size = None
    if required_max_kv_size is not None:
        if requested_max_kv_size is None:
            requested_max_kv_size = required_max_kv_size
        else:
            requested_max_kv_size = max(int(requested_max_kv_size), required_max_kv_size)
    prompt_cache = None
    try:
        from mlx_lm.models import cache as _cache_mod
        from mlx_lm.generate import maybe_quantize_kv_cache
        prompt_cache = _cache_mod.make_prompt_cache(
            model, max_kv_size=requested_max_kv_size)
    except Exception as e:
        warnings.warn(
            f"fusekit: prompt cache construction failed ({e}); "
            "falling back to mlx-lm path.")
        yield from _orig(prompt, model, **kw)
        return
    def quantize_cache():
        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

    # Match mlx-lm.generate_step exactly: prefill every prompt token
    # except the final one, then run the final prompt token through the
    # normal decode step. Full-prompt prefill produces a different KV
    # state for some RoPE/cache paths and can drift later in generation.
    prompt_progress_callback = kw.get("prompt_progress_callback") or (lambda *_: None)
    prefill_step_size = kw.get("prefill_step_size", 2048)
    try:
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        with mx.stream(generation_stream):
            logits = _try_full_prefill_last_logits(model, p, prompt_cache)
            if logits is not None:
                stats = getattr(model, "_fk_dispatch_stats", None)
                if stats is not None:
                    stats["native_prefill_last_logits"] = (
                        stats.get("native_prefill_last_logits", 0) + 1
                    )
                prompt_processed_tokens = total_prompt_tokens
                prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            else:
                while total_prompt_tokens - prompt_processed_tokens > 1:
                    remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
                    n_to_process = min(prefill_step_size, remaining)
                    prefill_m = _select_prefill_variant_m(model, n_to_process)
                    if prefill_m == n_to_process or (
                        prefill_m is not None
                        and _env_flag("FUSEKIT_EXPERIMENTAL_PREFILL_CHUNK_VARIANT")
                    ):
                        n_to_process = prefill_m
                    prefill_tokens = p[:n_to_process]
                    prefill_tokens, padded_prefill_extra = _pad_prefill_tokens_if_safe(
                        prefill_tokens,
                        prompt_cache,
                        model,
                    )
                    model(
                        prefill_tokens[None],
                        cache=prompt_cache,
                        _fusekit_prefill_no_logits=True,
                    )
                    _trim_prompt_cache_in_place(prompt_cache, padded_prefill_extra)
                    quantize_cache()
                    mx.eval([c.state for c in prompt_cache])
                    prompt_processed_tokens += n_to_process
                    prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
                    p = p[n_to_process:]
                    mx.clear_cache()

                logits = model(p[None], cache=prompt_cache)
            quantize_cache()
            mx.eval(logits, [c.state for c in prompt_cache])
        last_logits = logits[:, -1, :]
        # Greedy first token.
        y = mx.argmax(last_logits, axis=-1)
        first_logprobs = last_logits - mx.logsumexp(last_logits, keepdims=True)
        mx.eval(y, first_logprobs)
        token = int(y.item())
        first_token = token
        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
    except Exception as e:
        warnings.warn(
            f"fusekit: prefill failed in fast path ({e}); "
            "falling back to mlx-lm.")
        yield from _orig(prompt, model, **kw)
        return

    offset = _cache_offset(prompt_cache)
    required = None if offset is None else offset + max_tokens
    if required is None:
        reason = "cache has no offset-bearing KV entries"
    else:
        ok, reason = _ensure_cache_capacity(prompt_cache, required)
    if required is None or not ok:
        msg = (
            "fusekit: fast path cannot guarantee KV cache capacity "
            f"after prefill ({reason}); falling back to mlx-lm."
        )
        if _env_flag("FUSEKIT_STRICT_DISPATCH"):
            raise RuntimeError(msg)
        warnings.warn(msg)
        yield from _orig(prompt, model, **kw)
        return

    # Trigger ICB build with one full pass through patched_call —
    # this runs the model on the new token, builds the ICB on first
    # call. After this we have everything cached.
    if max_tokens == 1:
        yield first_token, first_logprobs.squeeze(0)
        return
    try:
        bootstrap = mx.array([[token]], dtype=mx.uint32)
        restore_direct = _direct_head_argmax_scope(
            model, _mlx_generate_kwargs_are_greedy(kw)
        )
        try:
            _ = model(bootstrap, cache=prompt_cache)
        finally:
            restore_direct()
    except Exception as e:
        warnings.warn(
            f"fusekit: ICB bootstrap failed ({e}); falling back.")
        yield from _orig(prompt, model, **kw)
        return

    # Bootstrap may have rebuilt the ICB (new cache → new pointers).
    # Re-fetch `_fk_lm` since `_install_fast_path_state` replaces the
    # whole dict on rebuild; our captured `fk` would otherwise be stale.
    fk = getattr(inner, "_fk_lm", None)
    if fk is None or "next_token_view" not in fk:
        warnings.warn("fusekit: ICB bootstrap did not initialize fast state; falling back.")
        yield from _orig(prompt, model, **kw)
        return
    token = int(fk["next_token_view"][0])

    yield first_token, first_logprobs.squeeze(0)
    n = 1

    # GPU-side feedback chain: pre-submit N decode cmd buffers in a
    # row, no per-step CPU sync. The ICB's step_init kernel advances
    # the offset counter on the GPU, embed_lookup reads its input
    # token from the buffer argmax writes to (same slot), and
    # history_capture appends each step's argmax for streaming. We
    # only `wait_until_completed` on the last cmd buffer, eliminating
    # the inter-cmd GPU-idle gap measured at ~420us/tok.
    last_logits_arr = fk["logits_buf"]
    icb_id = fk["icb_id"]
    handle = fk["handle"]
    token_id_ptr = fk["token_id_ptr"]
    next_token_ptr = fk["next_token_ptr"]
    offset_ptr = fk["layer_offset_ptr"]
    capacity_ptr = fk["layer_capacity_ptr"]
    kv_len_ptr = fk["layer_kv_len_ptr"]
    step_counter_ptr = fk["step_counter_ptr"]
    hist_counter_ptr = fk["hist_counter_ptr"]
    history_ptr = fk["history_ptr"]
    barrier_ptrs = fk.get("barrier_counter_ptrs") or []
    CHUNK = fk["chunk_capacity"]

    capacity = _cache_capacity(prompt_cache)
    step_offset = _cache_offset(prompt_cache)
    remaining = max_tokens - n
    if capacity is None or step_offset is None:
        msg = (
            "fusekit: fast path KV capacity unavailable after bootstrap "
            f"(offset={step_offset}, remaining={remaining}, capacity={capacity}); "
            "falling back to mlx-lm."
        )
        if _env_flag("FUSEKIT_STRICT_DISPATCH"):
            raise RuntimeError(msg)
        warnings.warn(msg)
        # Bootstrap already consumed the first generated token into
        # `prompt_cache` and left the next token in `token` with logits in
        # `last_logits_arr`. Preserve the stream instead of asking mlx-lm to
        # continue from an empty prompt, which raises on recent mlx-lm.
        yield token, last_logits_arr
        n += 1
        if n >= max_tokens:
            return
        fallback_prompt = mx.array([token], dtype=mx.uint32)
        fallback_kw = {**kw, "prompt_cache": prompt_cache, "max_tokens": max_tokens - n}
        yield from _orig(fallback_prompt, model, **fallback_kw)
        return

    ctypes.c_uint32.from_address(capacity_ptr).value = capacity
    history_view = (ctypes.c_uint32 * CHUNK).from_address(history_ptr)
    profile_at_raw = os.environ.get("FUSEKIT_PROFILE_KERNELS_AT", "")
    profile_at: set[int] = set()
    if profile_at_raw:
        for part in profile_at_raw.replace(",", " ").split():
            try:
                profile_at.add(int(part))
            except ValueError:
                pass

    while n < max_tokens:
        if step_offset >= capacity:
            if n + 1 >= max_tokens:
                yield token, last_logits_arr
                return
            msg = (
                "fusekit: fast path reached physical KV capacity "
                f"(offset={step_offset}, capacity={capacity}); "
                "continuing tail with mlx-lm."
            )
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise RuntimeError(msg)
            warnings.warn(msg)
            yield token, last_logits_arr
            n += 1
            if n >= max_tokens:
                return
            fallback_prompt = mx.array([token], dtype=mx.uint32)
            fallback_kw = {**kw, "prompt_cache": prompt_cache, "max_tokens": max_tokens - n}
            yield from _orig(fallback_prompt, model, **fallback_kw)
            return
        if n in profile_at:
            dispatcher = getattr(model, "_fk_dispatcher", None)
            if dispatcher is not None:
                print(
                    "[fusekit-profile-at] "
                    f"token_index={n} cache_offset={step_offset}",
                    flush=True,
                )
                _engine_set_cache_offset(prompt_cache, step_offset)
                dispatcher._profile_kernels_once = True
                dispatcher._profile_kernels_skip = 0
                dispatcher._profile_context = {
                    "token_index": int(n),
                    "cache_offset": int(step_offset),
                }
                profile_in = mx.array([[token]], dtype=mx.uint32)
                restore_direct = _direct_head_argmax_scope(
                    model, _mlx_generate_kwargs_are_greedy(kw)
                )
                try:
                    prof_logits = dispatcher.try_run(profile_in, prompt_cache)
                    prof_tokens = getattr(dispatcher, "last_argmax_tokens", None)
                finally:
                    restore_direct()
                if prof_logits is not None and prof_tokens is not None:
                    yield token, prof_logits
                    n += 1
                    step_offset += 1
                    token = int(prof_tokens.reshape(-1)[0])
                    last_logits_arr = prof_logits
                    continue
                warnings.warn(
                    f"fusekit: profile-at dispatcher miss at token_index={n}; "
                    "continuing with chunk path.",
                    stacklevel=2,
                )

        chunk = min(CHUNK, max_tokens - n)
        chunk = min(chunk, max(1, capacity - step_offset))
        if profile_at:
            future_profile_points = [p for p in profile_at if n < p < n + chunk]
            if future_profile_points:
                chunk = min(future_profile_points) - n
        chunk_mode = os.environ.get("FUSEKIT_GENERATE_CHUNK_MODE", "auto").lower()
        if chunk_mode == "auto":
            chunk_mode = "gpu" if not barrier_ptrs else "host"
        if chunk_mode in (
            "gpu",
            "chain",
            "gpu-chain",
        ):
            if barrier_ptrs:
                _native.decode_chunk_gpu_chain_reset_barriers(
                    handle,
                    icb_id,
                    step_counter_ptr,
                    hist_counter_ptr,
                    next_token_ptr,
                    step_offset,
                    token,
                    chunk,
                    barrier_ptrs,
                )
            else:
                _native.decode_chunk_gpu_chain(
                    handle,
                    icb_id,
                    step_counter_ptr,
                    hist_counter_ptr,
                    next_token_ptr,
                    step_offset,
                    token,
                    chunk,
                )
        else:
            _native.decode_chunk_host_loop(
                handle,
                icb_id,
                step_counter_ptr,
                hist_counter_ptr,
                token_id_ptr,
                next_token_ptr,
                offset_ptr,
                capacity_ptr,
                kv_len_ptr,
                history_ptr,
                capacity,
                step_offset,
                token,
                chunk,
                barrier_ptrs,
            )
        for i in range(chunk):
            yield token, last_logits_arr
            n += 1
            if i + 1 < chunk:
                token = int(history_view[i])
            elif n < max_tokens:
                token = int(history_view[i])
        step_offset += chunk
    for c in prompt_cache:
        c.offset = step_offset


class _PatchingLoader(importlib.abc.Loader):
    """Wraps a real loader; runs `exec_module` as usual then patches
    `load()`. Same instance can be reused — `_wrap_load` is idempotent."""

    def __init__(self, inner: importlib.abc.Loader):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        if os.environ.get("FUSEKIT_TRACE_GENSTEP"):
            print(f"[gen] exec_module fired for {module.__name__}", file=sys.stderr)
        if module.__name__ == _TOP_LOAD_MODULE:
            _wrap_load(module)
        elif module.__name__ == _TARGET_MODULE:
            _wrap_load(module)
        elif module.__name__ == _GENERATE_MODULE:
            sample_utils = sys.modules.get(_SAMPLE_UTILS_MODULE)
            if sample_utils is not None:
                _wrap_sample_utils(sample_utils)
                if hasattr(sample_utils, "make_sampler"):
                    module.make_sampler = sample_utils.make_sampler
            _wrap_generate_step(module)
            _wrap_generation_batch(module)
        elif module.__name__ == _SAMPLE_UTILS_MODULE:
            _wrap_sample_utils(module)
        elif module.__name__ == _OMLX_SCHEDULER_MODULE:
            _wrap_omlx_scheduler(module)


class _FusekitMetaPathFinder(importlib.abc.MetaPathFinder):
    """Only fires for `mlx_lm.utils`. Returns the spec from the normal
    machinery with the loader wrapped so we can post-process the module
    after it's executed."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in (
            _TOP_LOAD_MODULE,
            _TARGET_MODULE,
            _GENERATE_MODULE,
            _SAMPLE_UTILS_MODULE,
            _OMLX_SCHEDULER_MODULE,
        ):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader)
        return spec


def install_hook() -> None:
    """Install the meta-path finder. Idempotent — safe to call from
    multiple .pth files or twice via manual import. Skipped entirely
    when `FUSEKIT_DISABLE_HOOK` is set in the environment so callers
    can drive the dispatcher directly without the layer-level wrapper
    fighting them for `model.__call__`."""
    if os.environ.get("FUSEKIT_DISABLE_HOOK"):
        return
    if any(isinstance(f, _FusekitMetaPathFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _FusekitMetaPathFinder())

    # If mlx_lm / mlx_lm.utils / mlx_lm.generate were already imported before
    # the meta-path finder went in, patch them in place — the .pth
    # auto-import runs after Python startup but `mlx_lm`'s own
    # package init pulls these eagerly on some installs.
    existing_top = sys.modules.get(_TOP_LOAD_MODULE)
    if existing_top is not None:
        _wrap_load(existing_top)
    existing = sys.modules.get(_TARGET_MODULE)
    if existing is not None:
        _wrap_load(existing)
    existing_sample_utils = sys.modules.get(_SAMPLE_UTILS_MODULE)
    if existing_sample_utils is not None:
        _wrap_sample_utils(existing_sample_utils)
    existing_gen = sys.modules.get(_GENERATE_MODULE)
    if existing_gen is not None:
        if existing_sample_utils is not None and hasattr(existing_sample_utils, "make_sampler"):
            existing_gen.make_sampler = existing_sample_utils.make_sampler
        _wrap_generate_step(existing_gen)
        _wrap_generation_batch(existing_gen)
    existing_scheduler = sys.modules.get(_OMLX_SCHEDULER_MODULE)
    if existing_scheduler is not None:
        _wrap_omlx_scheduler(existing_scheduler)


# Auto-install on import. Safe because all the work is idempotent.
_install_gemma4_shared_kv_weight_filter()
install_hook()
