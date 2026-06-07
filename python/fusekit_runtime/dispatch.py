"""Per-step dispatcher driven by the compiled `task_graph.json`.

Auto-installed by `fusekit_runtime.attach` when both `kernel.metallib`
and `task_graph.json` are present alongside the model. The compiler
emits enough metadata (per-task `extra_bindings`, `layer_idx`,
`decode_pipeline` order, captured scalar constants, `source_path` for
every model parameter binding) that this module is the only Python
glue needed — no model-architecture-specific code lives here.
"""

from __future__ import annotations

import ctypes
import heapq
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from . import _ensure_cache_capacity, _env_flag


def _config_uses_mla_attention(cfg: dict) -> bool:
    """Detect MLA-style split Q/KV attention from semantic config fields."""
    return (
        cfg.get("kv_lora_rank") is not None
        and cfg.get("qk_nope_head_dim") is not None
        and cfg.get("qk_rope_head_dim") is not None
        and cfg.get("v_head_dim") is not None
    )


def _barrier_mask(dispatches, labels=None) -> list:
    """Return a per-task barrier flag. Mark `True` whenever the task
    reads or writes a buffer some prior task in the ICB wrote to.
    Adjacent independent tasks (e.g. RoPE-Q and RoPE-K-V-cache-store
    on disjoint scratches) get `False`, letting the GPU scheduler
    overlap them.

    Convention: dispatch tuple layout is
    `(kernel, [(ptr, nbytes), …], grid, threadgroup)`. The kernel
    signature determines which buffer slots are reads vs writes —
    we encode that lookup below per emitter name.
    """
    out = [False] * len(dispatches)
    last_writer = {}  # ptr -> task index
    track_war = bool(os.environ.get("FUSEKIT_SCRATCH_LIVENESS_REUSE"))
    last_access = {} if track_war else None  # ptr -> task index
    for i, (k, bufs, _, _) in enumerate(dispatches):
        label = labels[i] if labels is not None and i < len(labels) else None
        if label is not None and "read_slots" in label and "write_slots" in label:
            reads = label["read_slots"]
            writes = label["write_slots"]
        else:
            reads, writes = _slots_rw(k, len(bufs))
        # Conservative barrier rule: if any of this task's reads or
        # writes overlaps a buffer written by an earlier task in the
        # current ICB, fence both sides of the edge. Metal's indirect
        # `setBarrier` behaves as a command-local ordering marker, so
        # the prior writer must also carry a marker for fast consumers
        # like RoPE reading the step_init offset uniform.
        depends = False
        for s in reads:
            if s >= len(bufs):
                continue
            writer = last_writer.get(bufs[s][0])
            if writer is not None:
                out[writer] = True
                depends = True
        for s in writes:
            if s >= len(bufs):
                continue
            writer = last_writer.get(bufs[s][0])
            if writer is not None:
                out[writer] = True
                depends = True
            if track_war:
                prior = last_access.get(bufs[s][0])
                if prior is not None:
                    out[prior] = True
                    depends = True
        out[i] = depends
        if track_war:
            for s in reads:
                if s >= len(bufs):
                    continue
                last_access[bufs[s][0]] = i
        for s in writes:
            if s >= len(bufs):
                continue
            last_writer[bufs[s][0]] = i
            if track_war:
                last_access[bufs[s][0]] = i
    return out


def _slots_rw(kernel: str, n_slots: int) -> tuple:
    """Per-kernel (reads, writes) slot index sets. Mirrors the MSL
    signature of each emitter — keep aligned with `kernels::mod` in
    the compiler. Unknown kernels default to "everything reads, last
    slot writes" which conservatively asks for a barrier."""
    # step_init: read+write slot 0 (atomic counter); write offset/kv_len
    # uniforms and clears the shared GlobalBarrier counter array.
    if kernel == "fusekit_step_init":
        return ([0, 4], [0, 1, 2, 3])
    # history_capture: read+write slot 0 (counter); read 1; write 2.
    if kernel == "fusekit_history_capture":
        return ([0, 1], [0, 2])
    if "gated_delta_step" in kernel:
        writes = [7, 8]
        if n_slots >= 10:
            writes.append(9)
        return ([0, 1, 2, 3, 4, 5, 6], writes)
    if "fusekit_op_split_" in kernel:
        return ([0], list(range(1, n_slots)))
    if "fusekit_op_rope_" in kernel:
        return (list(range(n_slots - 1)), [n_slots - 1])
    if "fusekit_arrays_cache_shift_append_decode_" in kernel:
        return ([0, 1], [1, 2] if n_slots >= 3 else [1])
    # rmsnorm_qkv_q{4,8}: inputs, q/k/v outputs, six quantized cache
    # buffers, then offset/capacity uniforms.
    if "fusekit_rmsnorm_qkv_" in kernel:
        if "_q4_" in kernel or "_q8_" in kernel:
            n_inputs = n_slots - 11
            if n_inputs <= 0:
                return (list(range(n_slots)), [n_slots - 1])
            return (
                list(range(n_inputs)) + [n_slots - 2, n_slots - 1],
                list(range(n_inputs, n_slots - 2)),
            )
        # dense variant: inputs, q/k/v outputs, dense K/V cache, offset,
        # capacity. The label path below usually supplies the exact
        # slots; this fallback keeps barrier construction conservative.
        return (list(range(n_slots - 3)), [n_slots - 3, n_slots - 2, n_slots - 1])
    # rope_q_decode: slot 0 in/out (read+write), slot 1 = offset uniform read
    if "fusekit_rope_q_decode_" in kernel:
        return ([0, 1], [0])
    # rope_kv_cache_store_decode: read 0,1,4,5; write 2,3 (cache_k/v)
    if "fusekit_rope_kv_cache_store_decode_" in kernel:
        return ([0, 1, 4, 5], [2, 3])
    if "fusekit_kv_cache_store_decode_" in kernel:
        return ([0, 1, 4, 5], [2, 3])
    if "fusekit_rmsnorm_rope_qgate_decode_" in kernel:
        return ([0, 1, 2], [3, 4])
    if "fusekit_rmsnorm_rope_kv_cache_store_decode_" in kernel:
        return ([0, 1, 2, 5, 6], [3, 4])
    # sdpa_decode: read 0..4; write 5
    if "fusekit_sdpa_decode_" in kernel:
        return ([0, 1, 2, 3, 4], [5])
    # linear_residual: read all but last; write last
    if "fusekit_linear_residual_" in kernel:
        return (list(range(n_slots - 1)), [n_slots - 1])
    # rmsnorm_gateup_swiglu: read all but last; write last
    if "fusekit_rmsnorm_gateup_swiglu_" in kernel:
        return (list(range(n_slots - 1)), [n_slots - 1])
    # final_rmsnorm_lm_head: read all but last; write last
    if "fusekit_final_rmsnorm_lm_head_" in kernel:
        return (list(range(n_slots - 1)), [n_slots - 1])
    # embed_lookup_decode: dense reads table+token, quant reads packed
    # table+scale+bias+token.
    if "fusekit_embed_lookup_decode_" in kernel:
        if n_slots >= 5:
            return ([0, 1, 2, 3], [4])
        return ([0, 1], [2])
    # argmax_logits_decode: read 0; write 1
    if "fusekit_argmax_logits_decode_" in kernel:
        return ([0], [1])
    # argmax_logits_scan: read 0 (logits); write 1 (per-TG scratch)
    if "fusekit_argmax_logits_scan_" in kernel:
        return ([0], [1])
    # argmax_logits_finalize: read 0 (per-TG scratch); write 1 (next_tok)
    if "fusekit_argmax_logits_finalize_" in kernel:
        return ([0], [1])
    # qmv head direct argmax: read x + quantized weight triplet; write
    # per-threadgroup argmax scratch. Finalize reduces scratch to next_tok.
    if "fusekit_op_quantizedmatmul_" in kernel and "_argmax_finalize" in kernel:
        return ([0], [1])
    if "fusekit_op_quantizedmatmul_" in kernel and "_argmax" in kernel:
        return ([0, 1, 2, 3], [4])
    # Unknown — be safe.
    return (list(range(n_slots)), [n_slots - 1] if n_slots > 0 else [])


_DTYPE_ALIAS = {
    "bool": "bool_", "uint8": "uint8", "int8": "int8",
    "uint16": "uint16", "int16": "int16",
    "uint32": "uint32", "int32": "int32",
    "uint64": "uint64", "int64": "int64",
    "float16": "float16", "bfloat16": "bfloat16",
    "float32": "float32", "float64": "float64",
    "complex64": "complex64",
}


class _NativeScratchView:
    __slots__ = ("shape", "dtype", "nbytes", "ndim", "_fk_ptr")

    def __init__(self, ptr: int, shape: tuple[int, ...], dtype, nbytes: int):
        self._fk_ptr = int(ptr)
        self.shape = shape
        self.dtype = dtype
        self.nbytes = int(nbytes)
        self.ndim = len(shape)


def _model_root_from_fusekit_dir(fusekit_dir: Path) -> Path:
    path = Path(fusekit_dir).resolve()
    for cur in (path, *path.parents):
        if (cur / "config.json").is_file():
            return cur
        if cur.name == "fusekit" and (cur.parent / "config.json").is_file():
            return cur.parent
    return path.parent


def _mtp_hidden_variant(model: Any) -> str | None:
    """Return the MTP hidden-state contract attached by the wrapper.

    MTPLX models may expect post-final-norm hidden states for the MTP
    head. oMLX represents that by patching model.__call__; FuseKit
    bypasses that Python method when dispatch succeeds, so the dispatcher
    must mirror the same contract explicitly.
    """
    stack = [model, getattr(model, "_fk_original", None)]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        value = getattr(obj, "_fk_mtplx_hidden_variant", None)
        if value in ("pre_norm", "post_norm"):
            return value
        for name in ("language_model", "_language_model", "model"):
            try:
                child = getattr(obj, name)
            except Exception:
                child = None
            if child is not None:
                stack.append(child)
    return None


class Dispatcher:
    """Lazy-initialised per-step dispatcher. The first decode call
    that lands here builds the ICB; subsequent calls reuse it as long
    as the KV cache pointers stay put."""

    def __init__(self, model: Any, fusekit_dir: Path, handle: int) -> None:
        import mlx.core as mx

        self._model = model
        self._dir = Path(fusekit_dir)
        self._model_root = _model_root_from_fusekit_dir(self._dir)
        self._handle = handle
        self._mx = mx
        self._kernel_summaries = self._read_kernel_summaries()

        with open(self._dir / "task_graph.json") as f:
            self._tg = json.load(f)
        if self._tg.get("version") != 2:
            raise RuntimeError(
                f"task_graph.json v{self._tg.get('version')} not supported"
            )
        self._task_by_id = {int(t["id"]): t for t in self._tg["tasks"]}
        self._decode_task_ids = [int(tid) for tid in self._tg["decode_pipeline"]]
        self._decode_tasks = [self._task_by_id[tid] for tid in self._decode_task_ids]
        self._invariant_decode_task_ids_cache: set[int] | None = None
        self._active_m = max(1, int(self._tg.get("active_m", 1) or 1))
        self._stateful_mtp_graph = any(
            "gated_delta" in str(t.get("kernel", ""))
            or "arrays_cache_shift_append" in str(t.get("kernel", ""))
            for t in self._decode_tasks
        )
        self._mtp_hidden_variant = _mtp_hidden_variant(model)

        # Config-derived dimensions. Load before allocating scratch so
        # dense bf16 models that are known-unsafe can stay near native
        # memory usage even when dispatch is enabled for experiments.
        root_cfg = json.load(open(self._model_root / "config.json"))
        cfg = root_cfg.get("text_config", root_cfg)
        self._cfg = cfg
        self._hidden = cfg["hidden_size"]
        self._vocab = cfg["vocab_size"]
        self._act_dt = self._infer_activation_dtype(root_cfg, cfg)
        quantized = (
            root_cfg.get("quantization") is not None
            or cfg.get("quantization") is not None
        )
        self._fk_dense_bf16_disabled: bool = (
            not quantized
            and bool(os.environ.get("FUSEKIT_DISABLE_BF16"))
        )
        self._fk_unsupported_cache: bool = False
        self._fk_unsupported_graph: str | None = None
        self._fk_unsupported_graph_warned: bool = False
        caps = self._tg.get("decode_capabilities") or {}
        force_unsupported_graph = os.environ.get("FUSEKIT_FORCE_UNSUPPORTED_GRAPH")
        if caps and not caps.get("dispatchable", True) and not force_unsupported_graph:
            self._fk_unsupported_graph = self._format_capability_reasons(
                caps.get("reasons") or []
            )
        if (
            _config_uses_mla_attention(cfg)
            and not force_unsupported_graph
            and not _env_flag("FUSEKIT_ENABLE_MLA_DISPATCH")
        ):
            msg = (
                "decode graph uses MLA attention state, but MLA q_nope/q_pe "
                "and asymmetric KV cache bindings are not fully lowered"
            )
            self._fk_unsupported_graph = (
                f"{self._fk_unsupported_graph}; {msg}"
                if self._fk_unsupported_graph
                else msg
            )
        if self._fk_dense_bf16_disabled:
            return

        # Scratch buffers are allocated lazily. Dense bf16 traces often
        # contain materialized weight-transpose tensors that decode
        # dispatch overrides with live parameter bindings; allocating
        # every trace scratch up front doubles peak memory on small
        # models. Keep the schema-indexed list stable, but fill entries
        # only when a dispatched kernel actually binds them.
        self._scratch_specs: list = list(self._tg["scratch_buffers"])
        self._scratch: list = [None] * len(self._scratch_specs)
        self._scratch_arenas: list = []

        # Captured 0-dim trace constants → one-element mx.arrays.
        self._scalar_arrays: dict = {}
        for t in self._tg["tasks"]:
            for b in t["inputs"]:
                if b["role"] != "external":
                    continue
                sv = b.get("scalar")
                if not sv:
                    continue
                tid = b["tensor_id"]
                if tid in self._scalar_arrays:
                    continue
                arr = mx.array(sv["value"], dtype=self._mx_dtype(b["dtype"]))
                mx.eval(arr)
                self._scalar_arrays[tid] = arr

        # Named per-layer scratch derived from trace-bound pattern
        # outputs (rmsnorm_qkv → q_buf/k_buf/v_buf, sdpa → sdpa_buf,
        # the two linear_residuals → h_attn/out_buf, gateup_swiglu →
        # silu_up, final_rmsnorm_lm_head → logits).
        self._layer_scratch_id: dict = {}
        self._global_scratch_id: dict = {}
        self._qmv_argmax_scratch: dict[tuple[int, int], object] = {}
        # Scratch ID aliases — when two different scratch_ids in the
        # task graph point at the same logical buffer (e.g. fusekit's
        # synthetic RopeQDecode declares its output tensor, but the
        # actual write target is layer_scratch.q_buf — same memory).
        # Populated below; consulted in `_buf_for_binding`.
        self._scratch_alias: dict = {}
        # Scratch ids that are semantic cache state rather than
        # runtime-owned temporary storage. Qwen3 linear attention traces
        # expose conv_state as a prefill scratch feeding Concatenate;
        # decode must bind it to ArraysCache[0].
        self._arrays_cache_scratch: dict = {}
        self._arrays_cache_rollback: dict = {}
        self._gated_delta_t_ptr: int | None = None
        self._gated_delta_t1_ptr: int | None = None
        linres_per_layer: dict = {}
        # First pass: walk decode_pipeline in execution order so the
        # o_proj/down_proj split for LinearResidual (first → h_attn,
        # second → out_buf) lines up correctly. This is the original
        # behaviour and is what bf16-cache + non-LayerBlock dispatch
        # depends on.
        pipeline_task_ids = set(self._decode_task_ids)
        self._decode_output_scratch_ids = {
            b["scratch_id"]
            for t in self._decode_tasks
            for b in t.get("outputs", [])
            if b.get("role") == "scratch"
        }

        def _has_input_path_suffix(t, suffix: str) -> bool:
            return any(
                (b.get("source_path") or "").endswith(suffix)
                for b in t.get("inputs", [])
            )

        def _projection_role(t):
            for suffix, role in (
                ("self_attn.q_proj.weight", "q_buf"),
                ("self_attn.k_proj.weight", "k_buf"),
                ("self_attn.v_proj.weight", "v_buf"),
                ("mlp.gate_proj.weight", "gate_buf"),
                ("mlp.up_proj.weight", "up_buf"),
                ("self_attn.o_proj.weight", "h_attn"),
                ("mlp.down_proj.weight", "out_buf"),
            ):
                if _has_input_path_suffix(t, suffix):
                    return role
            return None

        def _projection_roles_for_fanout(t):
            roles = []
            for b in t.get("inputs", []):
                if b.get("role") != "external":
                    continue
                if b.get("role_name") != "W_qmv_fanout":
                    continue
                path = b.get("source_path") or ""
                role = None
                for suffix, candidate in (
                    ("self_attn.q_proj.weight", "q_buf"),
                    ("self_attn.k_proj.weight", "k_buf"),
                    ("self_attn.v_proj.weight", "v_buf"),
                    ("mlp.gate_proj.weight", "gate_buf"),
                    ("mlp.up_proj.weight", "up_buf"),
                    ("self_attn.o_proj.weight", "h_attn"),
                    ("mlp.down_proj.weight", "out_buf"),
                ):
                    if path.endswith(suffix):
                        role = candidate
                        break
                roles.append(role)
            return roles

        for t in self._decode_tasks:
            k = t["kernel"]
            li = t.get("layer_idx")
            outs = [b["scratch_id"] for b in t["outputs"] if b["role"] == "scratch"]
            if "fusekit_rmsnorm_qkv_" in k and li is not None and len(outs) == 3:
                self._layer_scratch_id[("q_buf", li)] = outs[0]
                self._layer_scratch_id[("k_buf", li)] = outs[1]
                self._layer_scratch_id[("v_buf", li)] = outs[2]
            elif "fusekit_sdpa_decode_" in k and li is not None and len(outs) == 1:
                self._layer_scratch_id[("sdpa_buf", li)] = outs[0]
                ins = [b for b in t.get("inputs", []) if b.get("role") == "scratch"]
                if ins:
                    self._layer_scratch_id.setdefault(("q_buf", li), ins[0]["scratch_id"])
            elif "fusekit_linear_residual_" in k and li is not None and len(outs) == 1:
                role = _projection_role(t)
                if role in ("q_buf", "k_buf", "v_buf", "gate_buf", "up_buf"):
                    self._layer_scratch_id[(role, li)] = outs[0]
                    continue
                if role in ("h_attn", "out_buf"):
                    name = role
                else:
                    seen = linres_per_layer.setdefault(li, 0)
                    # First residual projection per layer is o_proj
                    # (writes h_attn); second is down_proj
                    # (writes out_buf). Q/K/V bias projections are
                    # handled by source_path above and do not consume
                    # this residual counter.
                    name = "h_attn" if seen == 0 else "out_buf"
                    linres_per_layer[li] = seen + 1
                self._layer_scratch_id[(name, li)] = outs[0]
            elif "fusekit_op_quantizedmatmul_" in k and li is not None and len(outs) == 1:
                role = _projection_role(t)
                if role in ("q_buf", "k_buf", "v_buf", "gate_buf", "up_buf"):
                    self._layer_scratch_id[(role, li)] = outs[0]
                elif role in ("h_attn", "out_buf"):
                    self._layer_scratch_id[(role, li)] = outs[0]
            elif "fusekit_qmv_fanout_" in k and li is not None:
                roles = _projection_roles_for_fanout(t)
                for role, sid in zip(roles, outs, strict=False):
                    if role in ("q_buf", "k_buf", "v_buf", "gate_buf", "up_buf"):
                        self._layer_scratch_id[(role, li)] = sid
                    elif role in ("h_attn", "out_buf"):
                        self._layer_scratch_id[(role, li)] = sid
            elif "fusekit_op_add_" in k and li is not None and len(outs) == 1:
                out_shapes = [tuple(b.get("shape", []))
                              for b in t["outputs"] if b["role"] == "scratch"]
                if out_shapes and out_shapes[0] and out_shapes[0][-1] == self._hidden:
                    seen = linres_per_layer.setdefault(li, 0)
                    # General fusion may leave residuals as plain Add
                    # tasks. Expose them under the same semantic names
                    # used by LinearResidual/FusedChain.
                    name = "h_attn" if seen == 0 else "out_buf"
                    self._layer_scratch_id[(name, li)] = outs[0]
                    linres_per_layer[li] = seen + 1
            elif "fusekit_rmsnorm_gateup_swiglu_" in k and li is not None and len(outs) == 1:
                self._layer_scratch_id[("silu_up", li)] = outs[0]
            elif "fusekit_final_rmsnorm_lm_head_" in k and len(outs) == 1:
                self._global_scratch_id["logits"] = outs[0]
            elif "fusekit_chain_" in k and li is not None:
                # General-fusion FusedChain output mapping. Chain
                # kernels are produced by the compiler path
                # (FUSEKIT_NO_PATTERNS=1 + FUSEKIT_GENERAL_FUSION=1).
                # For decode-time fast-path, we need to expose chain
                # outputs under the same role aliases the patterns
                # use so SDPA/LinearResidual lookups still resolve.
                #
                # Heuristic by output count + shape:
                #   3 outputs → RMSNorm+QKV fan-out: q, k, v
                #   2 outputs (same shape) → RMSNorm+gate/up fan-out: gate, up
                #   1 output, shape (..., hidden) and a heavy step that
                #     ends in Add → o_proj/down_proj + residual. First
                #     such chain per layer is o_proj (h_attn); second
                #     is down_proj (out_buf). Trace-order mirrors fusion
                #     id order so the decode_pipeline walk sees them
                #     in the correct sequence.
                #   1 output fed by MLP gate/up weights → fused SwiGLU
                #     activation output (silu_up). This is the compiler
                #     path replacement for the legacy
                #     rmsnorm_gateup_swiglu FusionKind.
                out_shapes = [tuple(b.get("shape", []))
                              for b in t["outputs"] if b["role"] == "scratch"]
                source_paths = [
                    b.get("source_path") or ""
                    for b in t.get("inputs", [])
                ]
                has_mlp_gate_up = any(
                    (".mlp.gate_proj." in p or ".mlp.up_proj." in p)
                    for p in source_paths
                )
                if len(outs) == 3:
                    self._layer_scratch_id[("q_buf", li)] = outs[0]
                    self._layer_scratch_id[("k_buf", li)] = outs[1]
                    self._layer_scratch_id[("v_buf", li)] = outs[2]
                elif len(outs) == 2 and len(out_shapes) == 2 and out_shapes[0] == out_shapes[1]:
                    self._layer_scratch_id[("gate_buf", li)] = outs[0]
                    self._layer_scratch_id[("up_buf", li)] = outs[1]
                elif len(outs) == 1:
                    last_dim = out_shapes[0][-1] if out_shapes[0] else 0
                    if has_mlp_gate_up and last_dim != self._hidden:
                        self._layer_scratch_id[("silu_up", li)] = outs[0]
                    elif last_dim == self._hidden:
                        # Heavy+residual chain (o_proj/down_proj+Add).
                        seen = linres_per_layer.setdefault(li, 0)
                        name = "h_attn" if seen == 0 else "out_buf"
                        self._layer_scratch_id[(name, li)] = outs[0]
                        linres_per_layer[li] = seen + 1
            elif "fusekit_chain_" in k and li is None:
                out_shapes = [tuple(b.get("shape", []))
                              for b in t["outputs"] if b["role"] == "scratch"]
                if len(outs) == 1 and out_shapes[0]:
                    if out_shapes[0][-1] == self._vocab:
                        self._global_scratch_id["logits"] = outs[0]
            elif "fusekit_op_concatenate_" in k and li is not None:
                ins = [b for b in t.get("inputs", []) if b.get("role") == "scratch"]
                if len(ins) >= 2:
                    shape0 = tuple(ins[0].get("shape", []))
                    shape1 = tuple(ins[1].get("shape", []))
                    if (
                        len(shape0) == 3
                        and len(shape1) == 3
                        and shape0[0] == shape1[0] == 1
                        and shape0[-1] == shape1[-1]
                        and shape0[1] <= 8
                    ):
                        self._arrays_cache_scratch[(ins[0]["scratch_id"], li)] = 0
            elif len(outs) == 1:
                out_shapes = [tuple(b.get("shape", []))
                              for b in t["outputs"] if b["role"] == "scratch"]
                if (
                    li is None
                    and out_shapes
                    and out_shapes[0]
                    and out_shapes[0][-1] == self._hidden
                    and any(
                        (
                            (b.get("source_path") or "").endswith(".norm.weight")
                            and ".layers." not in (b.get("source_path") or "")
                        )
                        for b in t.get("inputs", [])
                    )
                ):
                    self._global_scratch_id["final_hidden"] = outs[0]
                if out_shapes and out_shapes[0] and out_shapes[0][-1] == self._vocab:
                    self._global_scratch_id["logits"] = outs[0]

        # Second pass: when LayerBlock mode is active, the per-layer
        # SDPA + LinearResidual(o_proj) tasks were filtered out of
        # decode_pipeline (the LayerBlock kernel absorbs them) so the
        # first pass never registered `sdpa_buf`/`h_attn`. Scan the
        # full task list and pick those scratch ids up from the
        # not-dispatched fusions — they still produce the buffers,
        # they just aren't fired at decode time.
        for t in self._tg["tasks"]:
            tid = t["id"]
            if tid in pipeline_task_ids:
                continue
            k = t["kernel"]
            li = t.get("layer_idx")
            outs = [b["scratch_id"] for b in t.get("outputs", []) if b["role"] == "scratch"]
            if "fusekit_sdpa_decode_" in k and li is not None and len(outs) == 1:
                self._layer_scratch_id.setdefault(("sdpa_buf", li), outs[0])
            elif "fusekit_linear_residual_" in k and li is not None and len(outs) == 1:
                # In LayerBlock mode the FIRST linear_residual per layer
                # (o_proj) is the one we skipped — register as h_attn.
                # tasks come out of build_tasks in fusion-id order,
                # which mirrors trace op order so o_proj precedes
                # down_proj. setdefault keeps h_attn pinned to the
                # first match.
                self._layer_scratch_id.setdefault(("h_attn", li), outs[0])

        # RoPE Q decode synthetic — declares the trace's RoPE Q output
        # tensor as its output scratch so downstream chain bindings
        # resolve (Q-scale Multiply, Q reshape to SDPA). At runtime the
        # kernel writes to layer_scratch.q_buf (in-place rotation), so
        # alias the declared scratch_id → q_buf scratch_id.
        import os as _os
        _dbg_alias = _os.environ.get("FUSEKIT_ALIAS_DEBUG")
        for t in self._tg["tasks"]:
            k = t["kernel"]
            li = t.get("layer_idx")
            if "fusekit_rope_q_decode_" in k and li is not None:
                rope_outs = [b["scratch_id"] for b in t.get("outputs", [])
                             if b.get("role") == "scratch"]
                if rope_outs:
                    q_buf = self._layer_scratch_id.get(("q_buf", li))
                    if _dbg_alias:
                        print(f"[alias] rope_q layer={li} declared_out={rope_outs[0]} q_buf={q_buf}", flush=True)
                    if q_buf is not None and rope_outs[0] != q_buf:
                        self._scratch_alias[rope_outs[0]] = q_buf
            elif "fusekit_rope_kv_cache_store_decode_" in k and li is not None:
                # K_rope output (rope-rotated K) is written into the KV
                # cache by the same kernel via ExtraBinding; downstream
                # SDPA reads the cache directly. The declared output
                # tensor isn't read by any further chain in fact (K
                # cache load goes through SDPA's ExtraBinding) but if
                # something does reference it, alias to k_buf.
                rope_outs = [b["scratch_id"] for b in t.get("outputs", [])
                             if b.get("role") == "scratch"]
                if rope_outs:
                    k_buf = self._layer_scratch_id.get(("k_buf", li))
                    if k_buf is not None and rope_outs[0] != k_buf:
                        self._scratch_alias[rope_outs[0]] = k_buf
            elif "fusekit_rmsnorm_rope_qgate_decode_" in k and li is not None:
                outs = [b["scratch_id"] for b in t.get("outputs", [])
                        if b.get("role") == "scratch"]
                if outs:
                    self._layer_scratch_id[("q_buf", li)] = outs[0]

        self._validate_decode_graph()

        # Layer 0's rmsnorm_qkv reads `x` from a trace scratch that the
        # embed lookup synthetic kernel doesn't write to (no IR link).
        # Allocate a dedicated x_in buffer, point the embed kernel's
        # output slot at it (via extra binding), and alias layer-0's
        # rmsnorm_qkv input scratch to the same memory.
        self._x_in = mx.zeros((self._active_m, self._hidden), dtype=self._act_dt)
        mx.eval(self._x_in)

        # Pre-dequantize the embed table — the synthetic
        # embed_lookup_decode kernel reads it as `half`. Locate the
        # embed module via the same multimodal-aware prefix walk used
        # for layers (plain Qwen2: `model.embed_tokens`; multimodal
        # Qwen3.5/3.6: `language_model.model.embed_tokens`).
        embed = None
        for path in ("model.embed_tokens",
                     "language_model.model.embed_tokens",
                     "language_model.embed_tokens"):
            try:
                cur = model
                for seg in path.split("."):
                    cur = getattr(cur, seg)
                embed = cur
                break
            except AttributeError:
                continue
        if embed is None:
            raise RuntimeError(f"could not find embed_tokens on {type(model).__name__}")
        self._host_embed_lookup = False
        self._gpu_quant_embed_lookup = False
        self._embed_module = embed
        self._embed_dequant = None
        if hasattr(embed, "scales") and hasattr(embed, "biases"):
            self._gpu_quant_embed_lookup = self._decode_embed_uses_quant_signature()
            # Old patched models have a dense 3-slot embed signature. For
            # those, keep the correctness fallback: ask MLX for one
            # QuantizedEmbedding row and copy it into `_x_in`. New patches
            # bind packed/scales/biases and dequantize the selected row in
            # the ICB, so the per-token host embedding path disappears.
            self._host_embed_lookup = not self._gpu_quant_embed_lookup
        else:
            self._embed_dequant = embed.weight
            mx.eval(self._embed_dequant)

        self._first_layer_x_scratch = None
        self._first_layer_x_external_tensor_id = None
        for tid in self._tg["decode_pipeline"]:
            t = self._tg["tasks"][tid]
            k = t["kernel"]
            li = t.get("layer_idx")
            # Pattern path: layer 0 rmsnorm_qkv reads x. Compiler path:
            # the layer-0 RMSNorm-headed FusedChain (QKV fan-out, 3
            # outputs) does. Either way, its first scratch input is
            # what the embed_lookup_decode kernel writes to `_x_in`.
            scratch_outs = [o for o in t.get("outputs", []) if o.get("role") == "scratch"]
            scratch_ins = [i for i in t.get("inputs", []) if i.get("role") == "scratch"]
            is_layer0_input_chain = (
                "fusekit_chain_" in k
                and li == 0
                and bool(scratch_outs)
                and bool(scratch_ins)
                and (scratch_ins[0].get("shape") or [None])[-1] == self._hidden
            )
            is_layer0_input_rmsnorm = (
                k.startswith("fusekit_op_rmsnorm_")
                and li == 0
                and bool(scratch_outs)
                and bool(t.get("inputs", []))
                and (t["inputs"][0].get("shape") or [None])[-1] == self._hidden
            )
            if (
                (
                    "fusekit_rmsnorm_qkv_" in k
                    or is_layer0_input_chain
                    or is_layer0_input_rmsnorm
                )
                and li == 0
            ):
                ins = t.get("inputs", [])
                if ins and ins[0]["role"] == "scratch":
                    self._first_layer_x_scratch = ins[0]["scratch_id"]
                elif (
                    ins
                    and ins[0].get("role") == "external"
                    and not ins[0].get("source_path")
                    and (ins[0].get("shape") or [None])[-1] == self._hidden
                ):
                    self._first_layer_x_external_tensor_id = ins[0].get("tensor_id")
                break

        # Allocate 4-byte uniform Metal buffers for every (role, layer)
        # the host writes per step (token_id, layer_offset etc.).
        from . import _fusekit_runtime_native as rt
        self._rt = rt
        self._uniform_ptrs: dict = {}

        # Path-resolution cache for `source_path` lookups.
        self._path_cache: dict = {}

        # Lazy state. The ICB is rebuilt iff the KV cache's
        # underlying buffer pointers move (rare — KVCache step=256
        # holds them stable for 256 tokens).
        self._icb_id: Any = None
        self._cache_sig: Any = None
        self._last_dispatches: list | None = None
        self._last_dispatch_labels: list | None = None
        self._has_argmax_output: bool = any(
            "fusekit_argmax_logits_" in str(t.get("kernel", ""))
            for t in self._tg.get("tasks", [])
        )
        self.last_argmax_tokens = None
        self._last_barriers: list | None = None
        self._dispatch_layout_sig: Any = None
        self._dispatch_rebind_plan: list[tuple[int, int, tuple]] | None = None
        self._last_rebind_failure: str | None = None
        self._profile_build_detail_active = False
        self._profile_scratch_alloc_us = 0.0
        self._profile_scratch_alloc_count = 0
        self._profile_scratch_alloc_bytes = 0
        self._profile_scratch_prealloc_skip_count = 0
        self._profile_scratch_prealloc_skip_bytes = 0
        self._profile_array_ptr_us = 0.0
        self._profile_array_ptr_count = 0
        self._profile_array_ptr_cache_hits = 0
        self._profile_array_ptr_preseed_us = 0.0
        self._profile_array_ptr_preseed_count = 0
        self._array_ptr_build_cache: dict[int, int] | None = None
        self._profile_kernels_once: bool = bool(os.environ.get("FUSEKIT_PROFILE_KERNELS"))
        self._profile_kernels_skip: int = (
            int(os.environ.get("FUSEKIT_PROFILE_KERNELS_SKIP", "0") or "0")
            if str(os.environ.get("FUSEKIT_PROFILE_KERNELS_SKIP", "0") or "0").isdigit()
            else 0
        )
        self._profile_kernels_target_m: int = (
            int(os.environ.get("FUSEKIT_PROFILE_KERNELS_TARGET_M", "0") or "0")
            if str(os.environ.get("FUSEKIT_PROFILE_KERNELS_TARGET_M", "0") or "0").isdigit()
            else 0
        )
        self._profile_context: dict[str, int] = {}
        self._profile_icb_remaining: int = (
            int(os.environ.get("FUSEKIT_PROFILE_ICB", "0") or "0")
            if str(os.environ.get("FUSEKIT_PROFILE_ICB", "0") or "0").isdigit()
            else 0
        )
        self._profile_mtp_state: bool = bool(os.environ.get("FUSEKIT_PROFILE_MTP_STATE"))
        # Set to True once we detect a cache class fusekit doesn't
        # know how to dispatch against (ArraysCache, MambaCache, etc.).
        # All subsequent `try_run` calls bail to native immediately.
        self._fk_unsupported_cache: bool = False
        self._invariant_dispatches_done: bool = False

        # Pipelined-chunk buffers. `chunk_capacity` bounds how many
        # decode cmd buffers the runtime pre-submits per chain.
        #
        # The trade-off: bigger chunks amortise `cmd.commit()` overhead,
        # but when the caller stops mid-chunk (most often: model emits
        # EOS at token X inside a chunk), the GPU has already computed
        # `chunk - (X mod chunk)` extra tokens whose outputs are
        # discarded. With CHUNK=64 and EOS hit ~mid-chunk, this is up
        # to ~30 wasted decode steps ≈ 100ms on a 0.5B-4bit model — the
        # difference between matching mlx-lm and being 20% behind on
        # any prompt that triggers EOS before `max_tokens`.
        #
        # 32 is the sweet spot empirically: cuts EOS-truncation waste
        # roughly in half vs 64 while still amortising commit overhead.
        # Going lower (16, 8) adds per-chunk Python overhead without
        # further reducing the EOS tail. `FK_CHUNK` env still overrides.
        self._chunk_capacity = int(os.environ.get("FK_CHUNK", "64"))
        self._history_nbytes = self._chunk_capacity * 4
        self._history_ptr = self._rt.alloc_uniform(self._handle, self._history_nbytes)
        self._step_counter_ptr = self._rt.alloc_uniform(self._handle, 4)
        self._hist_counter_ptr = self._rt.alloc_uniform(self._handle, 4)
        # Shared atomic counter array for `GlobalBarrier` pieces. The
        # step_init utility clears this array on-GPU at the start of
        # each decode step, so chunked decode can keep all ICB executes
        # bundled in one command buffer even for barrier-bearing models.
        self._barrier_counter_ptrs: dict = {}
        self._barrier_counter_base_ptr: int | None = None
        self._barrier_counter_count_ptr = self._rt.alloc_uniform(self._handle, 4)
        self._barrier_counter_count = 0


    # ── utility helpers ─────────────────────────────────────────

    def _format_capability_reasons(self, reasons: list) -> str:
        if not reasons:
            return "decode graph lacks a supported semantic lowering"
        parts = []
        for r in reasons[:4]:
            loc = []
            if r.get("layer_idx") is not None:
                loc.append(f"layer {r['layer_idx']}")
            if r.get("task_id") is not None:
                loc.append(f"task {r['task_id']}")
            where = f"{' '.join(loc)} " if loc else ""
            code = r.get("code", "unsupported")
            parts.append(f"{where}{code}")
        more = "" if len(reasons) <= 4 else f", +{len(reasons) - 4} more"
        return "decode graph has unsupported capabilities: " + ", ".join(parts) + more

    def _validate_decode_graph(self) -> None:
        """Reject graphs whose runtime scratch roles are not materialized.

        The Python dispatcher can only execute decode pipelines whose
        semantic scratch roles have been reconstructed from the task graph.
        Architectures such as DeepSeek MLA currently lower SDPA/cache inputs
        through additional split/view/pointwise tasks, so pretending they are
        standard Q/K/V scratch buffers would be incorrect. Fall back before
        building the ICB instead of raising from a missing dictionary key.
        """
        missing = []
        seen = set()
        for tid in self._tg["decode_pipeline"]:
            t = self._tg["tasks"][tid]
            li = t.get("layer_idx")
            for eb in t.get("extra_bindings", []):
                role = eb.get("role", "")
                if li is None or not role.startswith("layer_scratch."):
                    continue
                name = role.split(".", 1)[1]
                key = (name, li)
                if key not in self._layer_scratch_id and key not in seen:
                    missing.append(f"layer {li} {role}")
                    seen.add(key)
        if missing:
            head = ", ".join(missing[:4])
            more = "" if len(missing) <= 4 else f", +{len(missing) - 4} more"
            msg = (
                f"decode graph missing runtime scratch bindings: {head}{more}"
            )
            if os.environ.get("FUSEKIT_FORCE_UNSUPPORTED_GRAPH"):
                warnings.warn(
                    f"fusekit: forcing unsupported decode graph despite {msg}",
                    stacklevel=2,
                )
            else:
                self._fk_unsupported_graph = (
                    f"{self._fk_unsupported_graph}; {msg}"
                    if self._fk_unsupported_graph
                    else msg
                )

    def _mx_dtype(self, name: str):
        return getattr(self._mx, _DTYPE_ALIAS.get(name, name))

    def _infer_activation_dtype(self, root_cfg: dict, cfg: dict):
        counts: dict[str, int] = {}
        for spec in self._tg.get("scratch_buffers", []):
            shape = spec.get("shape") or []
            dtype = spec.get("dtype")
            if (
                dtype in ("float16", "bfloat16", "float32")
                and shape
                and shape[-1] == self._hidden
            ):
                elems = 1
                for dim in shape:
                    elems *= int(dim)
                counts[dtype] = counts.get(dtype, 0) + elems
        if counts:
            return self._mx_dtype(max(counts, key=counts.get))

        dtype_name = (
            cfg.get("torch_dtype")
            or cfg.get("dtype")
            or root_cfg.get("torch_dtype")
            or root_cfg.get("dtype")
        )
        if dtype_name in ("bfloat16", "bf16"):
            return self._mx.bfloat16
        if dtype_name in ("float16", "fp16", "half"):
            return self._mx.float16
        if dtype_name in ("float32", "fp32"):
            return self._mx.float32
        return self._mx.float16 if root_cfg.get("quantization") is not None else self._mx.bfloat16

    def _scratch_buf(self, sid: int):
        arr = self._scratch[sid]
        if arr is None:
            profile_t0 = time.perf_counter() if self._profile_build_detail_active else 0.0
            spec = self._scratch_specs[sid]
            shape = tuple(spec["shape"]) if spec["shape"] else (1,)
            arr = self._mx.zeros(shape, dtype=self._mx_dtype(spec["dtype"]))
            self._mx.eval(arr)
            self._scratch[sid] = arr
            if self._profile_build_detail_active:
                self._profile_scratch_alloc_us += (
                    time.perf_counter() - profile_t0
                ) * 1_000_000.0
                self._profile_scratch_alloc_count += 1
                self._profile_scratch_alloc_bytes += int(getattr(arr, "nbytes", 0))
        return arr

    def _decode_scratch_liveness(self, exposed_sids: set[int]) -> tuple[dict[int, int], dict[int, int]]:
        first: dict[int, int] = {}
        last: dict[int, int] = {}

        def mark(sid, idx: int) -> None:
            if sid is None:
                return
            sid = self._scratch_alias.get(sid, sid)
            first.setdefault(sid, idx)
            last[sid] = max(last.get(sid, idx), idx)

        for idx, t in enumerate(self._decode_tasks):
            li = t.get("layer_idx")
            for b in t.get("inputs", []) + t.get("outputs", []):
                if b.get("role") == "scratch":
                    mark(b.get("scratch_id"), idx)
            for eb in t.get("extra_bindings", []):
                role = eb.get("role", "")
                if li is not None and role.startswith("layer_scratch."):
                    name = role.split(".", 1)[1]
                    mark(self._layer_scratch_id.get((name, li)), idx)
                elif role.startswith("global_scratch."):
                    name = role.split(".", 1)[1]
                    mark(self._global_scratch_id.get(name), idx)

        end = len(self._decode_tasks) + 1
        for sid in exposed_sids:
            if sid in first:
                last[sid] = end
        return first, last

    def _assign_scratch_offsets(self, entries: list, first: dict[int, int], last: dict[int, int], *, unit: str) -> tuple[list, int]:
        """Assign arena offsets with conservative lifetime reuse."""
        if not os.environ.get("FUSEKIT_SCRATCH_LIVENESS_REUSE"):
            cursor = 0
            out = []
            for sid, shape, n_elems, elem_bytes, nbytes in entries:
                if unit == "elems":
                    align = max(1, 256 // elem_bytes)
                    cursor = ((cursor + align - 1) // align) * align
                    out.append((sid, shape, cursor, n_elems, elem_bytes, nbytes))
                    cursor += n_elems
                else:
                    cursor = ((cursor + 255) // 256) * 256
                    out.append((sid, shape, cursor, n_elems, elem_bytes, nbytes))
                    cursor += nbytes
            return out, cursor

        end = len(self._decode_tasks) + 1
        align = 256
        if unit == "elems":
            elem_bytes = entries[0][3]
            align = max(1, 256 // elem_bytes)
        active: list[tuple[int, int, int]] = []
        free: list[tuple[int, int]] = []
        cursor = 0
        out = []
        for sid, shape, n_elems, elem_bytes, nbytes in sorted(
            entries, key=lambda e: (first.get(e[0], 0), last.get(e[0], end), e[0])
        ):
            f = first.get(sid, 0)
            l = last.get(sid, end)
            while active and active[0][0] < f:
                _last, off, size = heapq.heappop(active)
                heapq.heappush(free, (size, off))
            need = n_elems if unit == "elems" else nbytes
            need = ((need + align - 1) // align) * align
            chosen = None
            for i, (size, off) in enumerate(free):
                if size >= need:
                    chosen = (i, size, off)
                    break
            if chosen is None:
                off = cursor
                size = need
                cursor += need
            else:
                i, size, off = chosen
                free.pop(i)
                heapq.heapify(free)
            heapq.heappush(active, (l, off, size))
            out.append((sid, shape, off, n_elems, elem_bytes, nbytes))
        return out, cursor

    def _preallocate_decode_scratch(self) -> None:
        """Materialize decode scratch buffers in one MLX eval.

        First-request dispatch construction needs raw Metal buffer pointers for
        small decode task outputs. Allocating/evaluating hundreds of small arrays
        one at a time makes cold M=1/MTP setup host-bound. Large scratch buffers
        stay lazy so preallocation does not pull unused trace buffers into peak
        memory.
        """
        if os.environ.get("FUSEKIT_DISABLE_SCRATCH_PREALLOC"):
            return
        max_bytes = int(os.environ.get("FUSEKIT_SCRATCH_PREALLOC_MAX_BYTES", "262144") or "262144")
        scratch_ids: set[int] = set()
        for t in self._decode_tasks:
            for b in t.get("outputs", []):
                if b.get("role") != "scratch":
                    continue
                sid = b.get("scratch_id")
                if sid is None:
                    continue
                scratch_ids.add(self._scratch_alias.get(sid, sid))

        profile_t0 = time.perf_counter() if self._profile_build_detail_active else 0.0
        if not os.environ.get("FUSEKIT_DISABLE_SCRATCH_ARENA"):
            from . import array_data_ptr

            exposed_sids: set[int] = set()
            for name in ("logits", "final_hidden"):
                sid = self._global_scratch_id.get(name)
                if sid is not None:
                    exposed_sids.add(self._scratch_alias.get(sid, sid))
            try:
                last_layer = len(self._decoder_layers()) - 1
                sid = self._layer_scratch_id.get(("out_buf", last_layer))
                if sid is not None:
                    exposed_sids.add(self._scratch_alias.get(sid, sid))
            except Exception:
                pass
            live_first, live_last = self._decode_scratch_liveness(exposed_sids)

            mlx_groups: dict[str, list] = {}
            native_groups: dict[str, list] = {}
            skipped_bytes = 0
            skipped_count = 0
            for sid in sorted(scratch_ids):
                if sid >= len(self._scratch) or self._scratch[sid] is not None:
                    continue
                spec = self._scratch_specs[sid]
                if spec is None:
                    continue
                shape = tuple(spec["shape"]) if spec["shape"] else (1,)
                n_elems = 1
                for dim in shape:
                    n_elems *= int(dim)
                dtype_name = spec["dtype"]
                elem_bytes = 4 if dtype_name == "float32" else 2
                nbytes = n_elems * elem_bytes
                if max_bytes > 0 and nbytes > max_bytes:
                    skipped_count += 1
                    skipped_bytes += nbytes
                    continue
                use_native_scratch = not _env_flag("FUSEKIT_DISABLE_NATIVE_SCRATCH")
                if self._active_m > 1 and not _env_flag("FUSEKIT_ENABLE_MTP_NATIVE_SCRATCH"):
                    use_native_scratch = False
                if "FUSEKIT_ENABLE_NATIVE_SCRATCH" in os.environ:
                    use_native_scratch = _env_flag("FUSEKIT_ENABLE_NATIVE_SCRATCH")
                groups = native_groups if use_native_scratch and sid not in exposed_sids else mlx_groups
                groups.setdefault(dtype_name, []).append((sid, shape, n_elems, elem_bytes, nbytes))

            arena_count = 0
            arena_bytes = 0
            for dtype_name, entries in native_groups.items():
                offsets, cursor = self._assign_scratch_offsets(
                    entries, live_first, live_last, unit="bytes"
                )
                if cursor <= 0:
                    continue
                base_ptr = self._rt.alloc_uniform(self._handle, cursor)
                self._rt.register_host_buffer(self._handle, base_ptr, cursor)
                self._scratch_arenas.append(base_ptr)
                dtype = self._mx_dtype(dtype_name)
                for sid, shape, offset_bytes, _n_elems, _elem_bytes, nbytes in offsets:
                    view = _NativeScratchView(base_ptr + offset_bytes, shape, dtype, nbytes)
                    self._scratch[sid] = view
                    if self._array_ptr_build_cache is not None:
                        self._array_ptr_build_cache[id(view)] = view._fk_ptr
                arena_count += len(entries)
                arena_bytes += cursor

            for dtype_name, entries in mlx_groups.items():
                elem_bytes = entries[0][3]
                offsets, cursor = self._assign_scratch_offsets(
                    entries, live_first, live_last, unit="elems"
                )
                if cursor <= 0:
                    continue
                arena = self._mx.zeros((cursor,), dtype=self._mx_dtype(dtype_name))
                self._mx.eval(arena)
                base_ptr = array_data_ptr(arena)
                self._rt.register_host_buffer(self._handle, base_ptr, arena.nbytes)
                self._scratch_arenas.append(arena)
                for sid, shape, offset_elems, n_elems, _elem_bytes, _nbytes in offsets:
                    view = arena[offset_elems:offset_elems + n_elems].reshape(shape)
                    self._scratch[sid] = view
                    if self._array_ptr_build_cache is not None:
                        self._array_ptr_build_cache[id(view)] = base_ptr + offset_elems * elem_bytes
                arena_count += len(entries)
                arena_bytes += int(getattr(arena, "nbytes", cursor * elem_bytes))

            if self._profile_build_detail_active:
                self._profile_scratch_prealloc_skip_count += skipped_count
                self._profile_scratch_prealloc_skip_bytes += skipped_bytes
                if arena_count:
                    self._profile_scratch_alloc_us += (
                        time.perf_counter() - profile_t0
                    ) * 1_000_000.0
                    self._profile_scratch_alloc_count += arena_count
                    self._profile_scratch_alloc_bytes += arena_bytes
            return

        pending = []
        pending_bytes = 0
        for sid in sorted(scratch_ids):
            if sid >= len(self._scratch) or self._scratch[sid] is not None:
                continue
            spec = self._scratch_specs[sid]
            if spec is None:
                continue
            shape = tuple(spec["shape"]) if spec["shape"] else (1,)
            n_elems = 1
            for dim in shape:
                n_elems *= int(dim)
            dtype_name = spec["dtype"]
            elem_bytes = 4 if dtype_name == "float32" else 2
            nbytes = n_elems * elem_bytes
            if max_bytes > 0 and nbytes > max_bytes:
                if self._profile_build_detail_active:
                    self._profile_scratch_prealloc_skip_count += 1
                    self._profile_scratch_prealloc_skip_bytes += nbytes
                continue
            arr = self._mx.zeros(shape, dtype=self._mx_dtype(spec["dtype"]))
            self._scratch[sid] = arr
            pending.append(arr)
            pending_bytes += int(getattr(arr, "nbytes", 0))
        if pending:
            self._mx.eval(*pending)
            if self._profile_build_detail_active:
                self._profile_scratch_alloc_us += (
                    time.perf_counter() - profile_t0
                ) * 1_000_000.0
                self._profile_scratch_alloc_count += len(pending)
                self._profile_scratch_alloc_bytes += pending_bytes

    def _seq_view(self, arr, m: int):
        if arr.ndim == 3:
            return arr[:, :m, :]
        if arr.ndim == 2:
            return arr[:m, :].reshape(1, m, -1)
        return None

    def _hidden_result_view(self, pre_norm_hidden, m: int):
        if self._mtp_hidden_variant == "post_norm":
            sid = self._global_scratch_id.get("final_hidden")
            if sid is not None:
                post_norm_hidden = self._scratch_buf(sid)
                view = self._seq_view(post_norm_hidden, m)
                if view is not None:
                    return view
            view = self._seq_view(pre_norm_hidden, m)
            if view is None:
                return None
            norm = getattr(getattr(self._model, "model", None), "norm", None)
            if callable(norm):
                return norm(view)
            return view
        return self._seq_view(pre_norm_hidden, m)

    def _read_kernel_summaries(self) -> dict:
        summaries = {}
        path = self._dir / "kernel.metal"
        try:
            lines = path.read_text().splitlines()
        except Exception:
            return summaries
        pending = None
        for line in lines:
            s = line.strip()
            if s.startswith("// Fusion #"):
                pending = s[3:]
                continue
            if pending and s.startswith("kernel void "):
                name = s[len("kernel void "):].split("(", 1)[0].strip()
                summaries[name] = pending
                pending = None
        return summaries

    # KV cache offsets/lengths/capacities advance in lockstep across
    # layers under mlx-lm's vanilla `KVCache` (every layer's
    # `update_and_fetch` sees the same K shape). We exploit that to
    # collapse the 24 per-layer offset/capacity/kv_len uniforms into a
    # single shared 4-byte slot each — the ICB binds the same pointer
    # at every layer's slot, and the per-step uniform fanout drops from
    # 73 4-byte writes to 4.
    _SHARED_ACROSS_LAYERS = frozenset(
        {"layer_offset", "layer_capacity", "layer_kv_len"}
    )

    def _get_uniform_n(self, role: str, layer_idx, nbytes: int) -> int:
        key = (role, None if role in self._SHARED_ACROSS_LAYERS else layer_idx)
        if nbytes != 4:
            key = key + (nbytes,)
        if key not in self._uniform_ptrs:
            self._uniform_ptrs[key] = self._rt.alloc_uniform(self._handle, nbytes)
        return self._uniform_ptrs[key]

    def _get_uniform(self, role: str, layer_idx) -> int:
        return self._get_uniform_n(role, layer_idx, 4)

    def _read_next_tokens(self, n: int):
        if not self._has_argmax_output:
            return None
        nbytes = max(1, n) * 4
        out_key = ("next_token_out", None) if nbytes == 4 else ("next_token_out", None, nbytes)
        ptr = self._uniform_ptrs.get(out_key)
        if ptr is None:
            key = ("token_id_in", None) if nbytes == 4 else ("token_id_in", None, nbytes)
            ptr = self._uniform_ptrs.get(key)
        if ptr is None or n <= 0:
            return None
        vals = (ctypes.c_uint32 * n).from_address(ptr)
        return self._mx.array([int(vals[i]) for i in range(n)], dtype=self._mx.uint32)

    def _direct_mtp_head_argmax_enabled(self) -> bool:
        if not getattr(self._model, "_fk_allow_direct_mtp_argmax", False):
            return False
        if not _env_flag("FUSEKIT_FORCE_MTP_DIRECT_ARGMAX"):
            min_hidden_raw = os.environ.get("FUSEKIT_MTP_DIRECT_ARGMAX_MIN_HIDDEN", "4096")
            try:
                min_hidden = max(0, int(min_hidden_raw))
            except Exception:
                min_hidden = 4096
            if int(getattr(self, "_hidden", 0) or 0) < min_hidden:
                return False
        return (
            self._active_m > 1
            and (
                not self._stateful_mtp_graph
                or _env_flag("FUSEKIT_ENABLE_STATEFUL_MTP_DIRECT_ARGMAX")
            )
            and not _env_flag("FUSEKIT_DISABLE_MTP_DIRECT_ARGMAX")
            and not _env_flag("FUSEKIT_DISABLE_MTP_GPU_ARGMAX_TOKENS")
        )

    def _direct_head_argmax_enabled(self) -> bool:
        if self._active_m > 1:
            return self._direct_mtp_head_argmax_enabled()
        if not getattr(self._model, "_fk_allow_direct_head_argmax", False):
            return False
        if _env_flag("FUSEKIT_DISABLE_DIRECT_HEAD_ARGMAX"):
            return False
        if _env_flag("FUSEKIT_DISABLE_GPU_ARGMAX_TOKENS"):
            return False
        if not _env_flag("FUSEKIT_FORCE_DIRECT_HEAD_ARGMAX"):
            min_hidden_raw = os.environ.get("FUSEKIT_DIRECT_HEAD_ARGMAX_MIN_HIDDEN", "0")
            try:
                min_hidden = max(0, int(min_hidden_raw))
            except Exception:
                min_hidden = 0
            if int(getattr(self, "_hidden", 0) or 0) < min_hidden:
                return False
        return True

    def _is_direct_mtp_head_qmm_task(self, t, kernel: str, layer_idx) -> bool:
        if not self._direct_head_argmax_enabled():
            return False
        if layer_idx is not None:
            return False
        is_qmm_head = kernel.startswith("fusekit_op_quantizedmatmul_")
        is_final_head = kernel.startswith("fusekit_final_rmsnorm_lm_head_")
        if not (is_qmm_head or is_final_head):
            return False
        if "_argmax" in kernel:
            return False
        outs = t.get("outputs", [])
        if len(outs) != 1 or outs[0].get("role") != "scratch":
            return False
        shape = outs[0].get("shape") or []
        return bool(shape) and int(shape[-1]) == int(self._vocab)

    def _qmv_argmax_scratch_binding(self, n_groups: int) -> tuple[int, int]:
        key = (max(1, int(self._active_m)), max(1, int(n_groups)))
        arr = self._qmv_argmax_scratch.get(key)
        if arr is None:
            arr = self._mx.zeros((key[0] * key[1], 2), dtype=self._mx.uint32)
            self._mx.eval(arr)
            self._qmv_argmax_scratch[key] = arr
        return (self._array_ptr(arr), arr.nbytes)

    def _ensure_barrier_counters(self) -> tuple[int, int, int]:
        count = max(len(self._decoder_layers()), 1)
        nbytes = count * 4
        if self._barrier_counter_base_ptr is None or self._barrier_counter_count < count:
            self._barrier_counter_base_ptr = self._rt.alloc_uniform(self._handle, nbytes)
            self._barrier_counter_count = count
        ctypes.c_uint32.from_address(self._barrier_counter_count_ptr).value = count
        return self._barrier_counter_base_ptr, self._barrier_counter_count_ptr, nbytes

    def _resolve_path(self, path: str):
        if path in self._path_cache:
            return self._path_cache[path]
        cur = self._model
        for seg in path.split("."):
            cur = cur[int(seg)] if seg.isdigit() else getattr(cur, seg)
        self._path_cache[path] = cur
        return cur

    def _decoder_layers(self):
        """Return the `nn.ModuleList` holding decoder layers, handling
        both plain (`self._model.model.layers`) and multimodal
        (`self._model.language_model.model.layers`) layouts. Cached on
        first access."""
        if getattr(self, "_decoder_layers_cache", None) is not None:
            return self._decoder_layers_cache
        candidates = [
            lambda m: m.model.layers,
            lambda m: m.language_model.model.layers,
            lambda m: m.language_model.layers,
        ]
        layers = None
        for getter in candidates:
            try:
                layers = getter(self._model)
                break
            except AttributeError:
                continue
        if layers is None:
            raise RuntimeError(
                "could not find decoder layers list on model "
                f"({type(self._model).__name__})"
            )
        self._decoder_layers_cache = layers
        return layers

    def _decode_embed_uses_quant_signature(self) -> bool:
        for tid in self._tg.get("decode_pipeline", []):
            task = self._tg["tasks"][tid]
            if "fusekit_embed_lookup_decode_" not in task.get("kernel", ""):
                continue
            slots = {
                int(eb.get("slot", -1)): eb.get("role")
                for eb in task.get("extra_bindings", [])
            }
            return (
                slots.get(0) == "param.embed_tokens"
                and slots.get(1) == "param.embed_tokens.scales"
                and slots.get(2) == "param.embed_tokens.biases"
                and slots.get(3) == "token_id_in"
                and slots.get(4) == "global_scratch.x_in"
            )
        return False

    def _layer_attr_prefix(self):
        """Source-path prefix that locates `layer_N.<rest>` for this
        model. For plain Qwen2 this is `model.layers`; for multimodal
        Qwen3.5/3.6 it's `language_model.model.layers`."""
        if getattr(self, "_layer_attr_prefix_cache", None) is not None:
            return self._layer_attr_prefix_cache
        for prefix in ("model.layers", "language_model.model.layers",
                       "language_model.layers"):
            try:
                cur = self._model
                for seg in prefix.split("."):
                    cur = getattr(cur, seg)
                _ = cur[0]  # confirm indexable
                self._layer_attr_prefix_cache = prefix
                return prefix
            except (AttributeError, IndexError, TypeError):
                continue
        raise RuntimeError("could not infer layer attribute prefix")

    def _array_ptr(self, arr) -> int:
        native_ptr = getattr(arr, "_fk_ptr", None)
        if native_ptr is not None:
            return int(native_ptr)
        from . import array_data_ptr
        if self._array_ptr_build_cache is not None:
            key = id(arr)
            cached = self._array_ptr_build_cache.get(key)
            if cached is not None:
                if self._profile_build_detail_active:
                    self._profile_array_ptr_cache_hits += 1
                return cached
            if self._profile_build_detail_active:
                t0 = time.perf_counter()
                ptr = array_data_ptr(arr)
                self._profile_array_ptr_us += (time.perf_counter() - t0) * 1_000_000.0
                self._profile_array_ptr_count += 1
            else:
                ptr = array_data_ptr(arr)
            self._array_ptr_build_cache[key] = ptr
            return ptr
        if self._profile_build_detail_active:
            t0 = time.perf_counter()
            ptr = array_data_ptr(arr)
            self._profile_array_ptr_us += (time.perf_counter() - t0) * 1_000_000.0
            self._profile_array_ptr_count += 1
            return ptr
        return array_data_ptr(arr)

    def _add_preseed_array(self, arrs: list, seen: set[int], arr) -> None:
        if arr is None or not hasattr(arr, "dtype"):
            return
        key = id(arr)
        native_ptr = getattr(arr, "_fk_ptr", None)
        if native_ptr is not None:
            if self._array_ptr_build_cache is not None:
                self._array_ptr_build_cache[key] = int(native_ptr)
            return
        if key in seen:
            return
        if self._array_ptr_build_cache is not None and key in self._array_ptr_build_cache:
            return
        seen.add(key)
        arrs.append(arr)

    def _collect_extra_role_array(self, role: str, layer_idx, cache, arrs: list, seen: set[int]) -> None:
        add = lambda arr: self._add_preseed_array(arrs, seen, arr)
        if layer_idx is not None and role == "layer_cache_k":
            add(cache[layer_idx].keys)
            return
        if layer_idx is not None and role == "layer_cache_v":
            add(cache[layer_idx].values)
            return
        if layer_idx is not None and role.startswith("layer_arrays_cache."):
            idx = int(role.rsplit(".", 1)[1])
            add(cache[layer_idx][idx])
            return
        if layer_idx is not None and role.startswith("layer_arrays_cache_rollback."):
            idx = int(role.rsplit(".", 1)[1])
            add(self._arrays_cache_rollback_arr(layer_idx, idx, cache))
            return
        if layer_idx is not None and role.startswith("layer_scratch."):
            name = role.split(".", 1)[1]
            sid = self._layer_scratch_id.get((name, layer_idx))
            if sid is not None:
                sid = self._scratch_alias.get(sid, sid)
                if sid < len(self._scratch) and self._scratch[sid] is not None:
                    add(self._scratch[sid])
            return
        if role == "global_scratch.x_in":
            add(self._x_in)
            return
        if role.startswith("global_scratch."):
            name = role.split(".", 1)[1]
            sid = self._global_scratch_id.get(name)
            if sid is not None and sid < len(self._scratch) and self._scratch[sid] is not None:
                add(self._scratch[sid])
            return
        if role == "param.embed_tokens":
            if not self._host_embed_lookup:
                add(self._embed_module.weight if self._gpu_quant_embed_lookup else self._embed_dequant)
            return
        if role == "param.embed_tokens.scales":
            add(getattr(self._embed_module, "scales", None))
            return
        if role == "param.embed_tokens.biases":
            add(getattr(self._embed_module, "biases", None))
            return
        if role == "param.final_norm":
            decoder_root = self._decoder_layers().__self__ if hasattr(self._decoder_layers(), "__self__") else None
            if decoder_root is None:
                prefix = self._layer_attr_prefix()
                parent_path = prefix.rsplit(".", 1)[0]
                cur = self._model
                for seg in parent_path.split("."):
                    cur = getattr(cur, seg)
                add(cur.norm.weight)
            else:
                add(decoder_root.norm.weight)
            return
        if role == "param.lm_head":
            if hasattr(self._model, "lm_head"):
                add(self._model.lm_head.weight)
            else:
                prefix = self._layer_attr_prefix()
                parent_path = prefix.rsplit(".", 1)[0]
                cur = self._model
                for seg in parent_path.split("."):
                    cur = getattr(cur, seg)
                add(cur.embed_tokens.weight)
            return
        if role.startswith("layer_param."):
            tail = role[len("layer_param."):]
            sep = tail.find(".")
            if sep < 0:
                return
            obj = self._decoder_layers()[int(tail[:sep])]
            for part in tail[sep + 1:].split("."):
                obj = getattr(obj, part)
            add(obj)

    def _preseed_array_ptr_cache(self, cache) -> None:
        if self._array_ptr_build_cache is None:
            return
        if os.environ.get("FUSEKIT_DISABLE_ARRAY_PTR_PRESEED"):
            return
        from . import array_data_ptrs

        arrs: list = []
        seen: set[int] = set()
        add = lambda arr: self._add_preseed_array(arrs, seen, arr)

        add(self._x_in)
        for arr in self._scratch:
            add(arr)
        for arr in self._scalar_arrays.values():
            add(arr)
        if cache is not None:
            for c in cache:
                for _desc, arr in self._iter_cache_arrays(c):
                    add(arr)

        for t in self._decode_tasks:
            li = t.get("layer_idx")
            for b in t.get("inputs", []):
                role = b.get("role")
                if role == "scratch":
                    sid = b.get("scratch_id")
                    if sid in self._scratch_alias:
                        sid = self._scratch_alias[sid]
                    if sid is not None and sid < len(self._scratch) and self._scratch[sid] is not None:
                        add(self._scratch[sid])
                elif role == "external":
                    path = b.get("source_path")
                    if path:
                        add(self._resolve_path(path))
                    elif b.get("scalar"):
                        add(self._scalar_arrays.get(b["tensor_id"]))
            for b in t.get("outputs", []):
                if b.get("role") != "scratch":
                    continue
                sid = b.get("scratch_id")
                if sid in self._scratch_alias:
                    sid = self._scratch_alias[sid]
                if sid is not None and sid < len(self._scratch) and self._scratch[sid] is not None:
                    add(self._scratch[sid])
            for eb in t.get("extra_bindings", []):
                role = eb.get("role")
                if role:
                    self._collect_extra_role_array(role, li, cache, arrs, seen)

        if not arrs:
            return
        profile_t0 = time.perf_counter() if self._profile_build_detail_active else 0.0
        ptrs = array_data_ptrs(arrs)
        for arr, ptr in zip(arrs, ptrs):
            self._array_ptr_build_cache[id(arr)] = int(ptr)
        if self._profile_build_detail_active:
            self._profile_array_ptr_preseed_us += (
                time.perf_counter() - profile_t0
            ) * 1_000_000.0
            self._profile_array_ptr_preseed_count += len(ptrs)

    def _scalar_int(self, value) -> int | None:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape not in (None, (), (1,)):
            return None
        try:
            if hasattr(value, "item"):
                return int(value.item())
            return int(value)
        except Exception:
            return None

    def _batch_kv_single_ok(self, c) -> bool:
        """BatchKVCache is dispatchable only for a single unpadded row.

        oMLX's engine always uses BatchKVCache, even for one active request.
        The current FuseKit decode kernels take one scalar offset/capacity,
        so true multi-row continuous batching still has to stay on mlx-lm.
        """
        keys = getattr(c, "keys", None)
        if keys is not None:
            shape = getattr(keys, "shape", ())
            if not shape or int(shape[0]) != 1:
                return False
        left_padding = getattr(c, "left_padding", None)
        lp = self._scalar_int(left_padding)
        return lp == 0

    def _cache_decode_offset(self, c) -> int | None:
        if type(c).__name__ == "BatchKVCache":
            if not self._batch_kv_single_ok(c):
                return None
            return int(getattr(c, "_idx", 0))
        return self._scalar_int(getattr(c, "offset", None))

    def _advance_decode_cache(self, cache, next_offset: int, steps: int = 1) -> None:
        batch_offset = None
        for c in cache:
            if type(c).__name__ == "BatchKVCache":
                c._idx = next_offset
                if batch_offset is None:
                    batch_offset = self._mx.array([next_offset])
                c.offset = batch_offset
            elif hasattr(c, "offset"):
                c.offset = next_offset
            elif hasattr(c, "advance"):
                c.advance(steps)

    def _arrays_cache_rollback_arr(self, layer_idx: int, index: int, cache):
        key = (layer_idx, index)
        arr = self._arrays_cache_rollback.get(key)
        live = cache[layer_idx][index]
        if live is None:
            return None
        if (
            arr is None
            # `_ensure_icb()` calls `_prepare_arrays_cache_rollback_buffers()`
            # before this hot path and performs the full storage-pointer alias
            # check there. During each verify step we only need to catch the
            # common reject case where oMLX assigned the rollback array object
            # back into the live cache slot.
            or arr is live
            or getattr(arr, "shape", None) != getattr(live, "shape", None)
            or getattr(arr, "dtype", None) != getattr(live, "dtype", None)
        ):
            arr = self._mx.zeros(live.shape, dtype=live.dtype)
            self._mx.eval(arr)
            self._arrays_cache_rollback[key] = arr
        return arr

    def _same_array_storage(self, a, b) -> bool:
        if a is None or b is None:
            return False
        if a is b:
            return True
        try:
            return self._array_ptr(a) == self._array_ptr(b)
        except Exception:
            return False

    def _prepare_arrays_cache_rollback_buffers(self, cache) -> bool:
        """Ensure speculative rollback scratch never aliases live state.

        On reject, oMLX restores ArraysCache by assigning
        `cache[i] = rollback_state[i]`. If the next M=N verifier reuses that
        same persistent rollback array as its `mid_state` output, the cache
        update kernel receives aliased `state` and `mid_state` pointers and
        corrupts the confirmed-token snapshot. Allocate a fresh scratch buffer
        before ICB construction/rebinding whenever that alias is detected.
        """
        changed = False
        debug_alias = bool(os.environ.get("FUSEKIT_DBG_MTP_ROLLBACK_ALIAS"))
        for li, c in enumerate(cache or []):
            if type(c).__name__ != "ArraysCache":
                continue
            for idx, live in enumerate(getattr(c, "cache", [])):
                if live is None:
                    continue
                key = (li, idx)
                arr = self._arrays_cache_rollback.get(key)
                same_storage = self._same_array_storage(arr, live)
                if (
                    arr is None
                    or same_storage
                    or getattr(arr, "shape", None) != getattr(live, "shape", None)
                    or getattr(arr, "dtype", None) != getattr(live, "dtype", None)
                ):
                    old_ptr = self._array_ptr(arr) if arr is not None else 0
                    live_ptr = self._array_ptr(live)
                    arr = self._mx.zeros(live.shape, dtype=live.dtype)
                    self._mx.eval(arr)
                    self._arrays_cache_rollback[key] = arr
                    changed = True
                    if debug_alias:
                        print(
                            "[fusekit-mtp-alias] "
                            f"layer={li} idx={idx} "
                            f"reason={'alias' if same_storage else 'alloc'} "
                            f"old_rb=0x{old_ptr:x} live=0x{live_ptr:x} "
                            f"new_rb=0x{self._array_ptr(arr):x}",
                            flush=True,
                        )
        return changed

    def _set_mtp_rollback_state(self, cache) -> None:
        if self._active_m <= 1:
            return
        snapshots = []
        for li, c in enumerate(cache):
            if type(c).__name__ != "ArraysCache":
                continue
            conv = self._arrays_cache_rollback.get((li, 0))
            ssm = self._arrays_cache_rollback.get((li, 1))
            if conv is not None and ssm is not None:
                snapshots.extend((conv, ssm))
                c.rollback_state = (conv, ssm)
        if snapshots and not _env_flag("FUSEKIT_DISABLE_MTP_ROLLBACK_EVAL"):
            self._mx.eval(*snapshots)

    def _resolve_extra(self, role: str, layer_idx, cache):
        if role == "token_id_in":
            # Alias to the same slot the argmax writes. With the
            # ICB's intra-step barriers the read happens before the
            # write, and across cmd buffers Metal queue serialisation
            # makes the previous step's argmax visible. Closes the
            # per-token feedback loop on the GPU.
            if self._active_m > 1:
                return (
                    self._get_uniform_n("token_id_in", None, self._active_m * 4),
                    self._active_m * 4,
                )
            return (self._get_uniform("next_token_out", None), 4)
        if role == "next_token_out":
            nbytes = max(1, self._active_m) * 4
            return (self._get_uniform_n(role, None, nbytes), nbytes)
        if role == "layer_offset":
            return (self._get_uniform(role, layer_idx), 4)
        if role == "layer_capacity":
            return (self._get_uniform(role, layer_idx), 4)
        if role == "layer_kv_len":
            return (self._get_uniform(role, layer_idx), 4)
        if role == "layer_cache_k":
            c = cache[layer_idx]
            return (self._array_ptr(c.keys), c.keys.nbytes)
        if role == "layer_cache_v":
            c = cache[layer_idx]
            return (self._array_ptr(c.values), c.values.nbytes)
        if role.startswith("layer_arrays_cache."):
            idx = int(role.rsplit(".", 1)[1])
            arr = cache[layer_idx][idx]
            if arr is None:
                return (0, 0)
            return (self._array_ptr(arr), arr.nbytes)
        if role.startswith("layer_arrays_cache_rollback."):
            idx = int(role.rsplit(".", 1)[1])
            arr = self._arrays_cache_rollback_arr(layer_idx, idx, cache)
            if arr is None:
                return (0, 0)
            return (self._array_ptr(arr), arr.nbytes)
        if role.startswith("layer_scratch."):
            name = role.split(".", 1)[1]
            sid = self._layer_scratch_id[(name, layer_idx)]
            arr = self._scratch_buf(sid)
            return (self._array_ptr(arr), arr.nbytes)
        if role == "global_scratch.x_in":
            return (self._array_ptr(self._x_in), self._x_in.nbytes)
        if role == "global_scratch.logits":
            sid = self._global_scratch_id["logits"]
            arr = self._scratch_buf(sid)
            return (self._array_ptr(arr), arr.nbytes)
        if role == "global_scratch.argmax_scratch":
            # Per-row, per-TG (max_key, idx) scratch for the multi-TG
            # argmax. Lazy-allocate on first bind; covers up to 256 TGs
            # per active row, which is the max `FUSEKIT_ARGMAX_MULTI_TG`
            # accepts (build.rs).
            sid = self._global_scratch_id.get("argmax_scratch")
            if sid is None:
                arr = self._mx.zeros((max(1, self._active_m) * 256, 2), dtype=self._mx.uint32)
                self._mx.eval(arr)
                self._scratch.append(arr)
                self._scratch_specs.append(None)
                sid = len(self._scratch) - 1
                self._global_scratch_id["argmax_scratch"] = sid
            arr = self._scratch_buf(sid)
            return (self._array_ptr(arr), arr.nbytes)
        if role == "global_scratch.sdpa_partials":
            # Flash-Decoding partials, lazy-allocated.
            # Shape: [n_layers, n_heads, MAX_CHUNKS, head_dim + 2] float32.
            # Each (layer, head, chunk) stores m (1 float), l (1 float),
            # o[head_dim] (head_dim floats) — total head_dim+2.
            sid = self._global_scratch_id.get("sdpa_partials")
            if sid is None:
                cfg = self._cfg
                n_layers = cfg["num_hidden_layers"]
                n_heads = cfg["num_attention_heads"]
                head_dim = cfg.get("head_dim", cfg["hidden_size"] // n_heads)
                MAX_CHUNKS = 32  # matches FLASH_DEC_MAX_CHUNKS in sdpa_decode.rs
                arr = self._mx.zeros(
                    (n_layers, self._active_m, n_heads, MAX_CHUNKS, head_dim + 2),
                    dtype=self._mx.float32,
                )
                self._mx.eval(arr)
                self._scratch.append(arr)
                self._scratch_specs.append(None)
                sid = len(self._scratch) - 1
                self._global_scratch_id["sdpa_partials"] = sid
            arr = self._scratch_buf(sid)
            return (self._array_ptr(arr), arr.nbytes)
        if role == "global_scratch.sdpa_counter":
            # Flash-Decoding per-(layer, head) atomic counter. Each
            # active chunk does one atomic_fetch_add; the last incrementor
            # combines and resets to 0. Lazy alloc; init zero.
            sid = self._global_scratch_id.get("sdpa_counter")
            if sid is None:
                cfg = self._cfg
                n_layers = cfg["num_hidden_layers"]
                n_heads = cfg["num_attention_heads"]
                arr = self._mx.zeros(
                    (n_layers, self._active_m, n_heads), dtype=self._mx.uint32
                )
                self._mx.eval(arr)
                self._scratch.append(arr)
                self._scratch_specs.append(None)
                sid = len(self._scratch) - 1
                self._global_scratch_id["sdpa_counter"] = sid
            arr = self._scratch_buf(sid)
            return (self._array_ptr(arr), arr.nbytes)
        if role == "param.embed_tokens":
            if self._host_embed_lookup:
                raise ValueError("param.embed_tokens is host-fed for quantized embeddings")
            arr = self._embed_module.weight if self._gpu_quant_embed_lookup else self._embed_dequant
            return (self._array_ptr(arr), arr.nbytes)
        if role == "param.embed_tokens.scales":
            arr = getattr(self._embed_module, "scales", None)
            if arr is None:
                raise ValueError("param.embed_tokens.scales requested for dense embedding")
            return (self._array_ptr(arr), arr.nbytes)
        if role == "param.embed_tokens.biases":
            arr = getattr(self._embed_module, "biases", None)
            if arr is None:
                raise ValueError("param.embed_tokens.biases requested for dense embedding")
            return (self._array_ptr(arr), arr.nbytes)
        if role == "param.final_norm":
            # Decoder root norm sits one level above the layers list.
            # For multimodal models that's `language_model.model.norm`.
            decoder_root = self._decoder_layers().__self__ if hasattr(self._decoder_layers(), "__self__") else None
            if decoder_root is None:
                # Walk one parent up from the layer-list path.
                prefix = self._layer_attr_prefix()
                parent_path = prefix.rsplit(".", 1)[0]
                cur = self._model
                for seg in parent_path.split("."):
                    cur = getattr(cur, seg)
                arr = cur.norm.weight
            else:
                arr = decoder_root.norm.weight
            return (self._array_ptr(arr), arr.nbytes)
        if role == "param.lm_head":
            # mlx-lm Linear with tied weights stores it at the model
            # root's lm_head attribute; otherwise we fall back to the
            # embed_tokens weight (tied embeddings).
            if hasattr(self._model, "lm_head"):
                arr = self._model.lm_head.weight
            else:
                prefix = self._layer_attr_prefix()
                parent_path = prefix.rsplit(".", 1)[0]
                cur = self._model
                for seg in parent_path.split("."):
                    cur = getattr(cur, seg)
                arr = cur.embed_tokens.weight
            return (self._array_ptr(arr), arr.nbytes)
        if role.startswith("barrier_counters."):
            # `barrier_counters.{N}` binds the shared counter array.
            # The generated LayerBlock uses `N` as the array index, so
            # every barrier gets an independent counter while step_init
            # can clear them all with one tiny GPU-side loop.
            ptr, _count_ptr, nbytes = self._ensure_barrier_counters()
            return (ptr, nbytes)
        if role.startswith("layer_param."):
            # `layer_param.{N}.{relpath}` — e.g. `layer_param.5.self_attn.o_proj.weight`.
            # Resolve to `model.layers[N].<relpath>` via attribute
            # traversal. Synthetic LayerBlock fusion uses this to bind
            # per-layer weights (norms, q/k/v/o/gate/up/down) without
            # an explicit fusion.inputs boundary.
            tail = role[len("layer_param."):]
            sep = tail.find(".")
            if sep < 0:
                raise ValueError(f"malformed layer_param role: {role}")
            layer_n = int(tail[:sep])
            relpath = tail[sep + 1:]
            obj = self._decoder_layers()[layer_n]
            for part in relpath.split("."):
                obj = getattr(obj, part)
            return (self._array_ptr(obj), obj.nbytes)
        raise ValueError(f"unknown extra binding role: {role}")

    def _buf_for_binding(self, b, layer_idx_for_input: int, cache=None):
        if b["role"] == "scratch":
            sid = b["scratch_id"]
            arrays_key = (sid, layer_idx_for_input)
            if (
                cache is not None
                and b.get("role_name") != "out"
                and arrays_key in self._arrays_cache_scratch
            ):
                idx = self._arrays_cache_scratch[arrays_key]
                arr = cache[layer_idx_for_input][idx]
                if arr is None:
                    return (0, 0)
                return (self._array_ptr(arr), arr.nbytes)
            # Synthetic scratch alias (e.g., RopeQDecode declared
            # output → layer_scratch.q_buf). Both ids point at the
            # same underlying buffer; dereference here.
            if sid in self._scratch_alias:
                sid = self._scratch_alias[sid]
            if self._first_layer_x_scratch is not None \
                    and b.get("role_name") != "out" \
                    and sid == self._first_layer_x_scratch \
                    and layer_idx_for_input == 0:
                return (self._array_ptr(self._x_in), self._x_in.nbytes)
            rn = b.get("role_name", "")
            # Q/K/V additive biases land in the trace as Broadcast
            # outputs but the hand-tuned rmsnorm_qkv kernel reads
            # them as raw `bias[head_dim_idx]` — point at the source
            # param directly.
            if rn in ("bias_q", "bias_k", "bias_v"):
                kind = {"bias_q": "q_proj", "bias_k": "k_proj", "bias_v": "v_proj"}[rn]
                arr = self._resolve_path(
                    f"{self._layer_attr_prefix()}.{layer_idx_for_input}.self_attn.{kind}.bias"
                )
                return (self._array_ptr(arr), arr.nbytes)
            arr = self._scratch_buf(sid)
            return (self._array_ptr(arr), arr.nbytes)
        path = b.get("source_path")
        if (
            self._first_layer_x_external_tensor_id is not None
            and b.get("tensor_id") == self._first_layer_x_external_tensor_id
            and layer_idx_for_input == 0
        ):
            return (self._array_ptr(self._x_in), self._x_in.nbytes)
        if path:
            arr = self._resolve_path(path)
            return (self._array_ptr(arr), arr.nbytes)
        if b.get("scalar"):
            arr = self._scalar_arrays[b["tensor_id"]]
            return (self._array_ptr(arr), max(1, arr.nbytes))
        raise RuntimeError(f"task has unbound binding: {b}")

    def _quant_overlay_sdpa(self, li, cache, t):
        """Build buffer list for quantized-cache SDPA dispatch.
        Mirrors `kernels/sdpa_decode_quant.rs` signature: Q,
        K(packed/scales/biases), V(packed/scales/biases), kv_len,
        capacity, sdpa_out."""
        c = cache[li]
        kp, ks, kb = c.keys
        vp, vs, vb = c.values
        slot0_extra = any(
            eb.get("slot") == 0 and eb.get("role") == "layer_scratch.q_buf"
            for eb in t.get("extra_bindings", [])
        )
        q_buf = (
            self._resolve_extra("layer_scratch.q_buf", li, cache)
            if slot0_extra
            else self._buf_for_binding(t["inputs"][0], li, cache)
        )
        bufs = [
            q_buf,
            (self._array_ptr(kp), kp.nbytes),
            (self._array_ptr(ks), ks.nbytes),
            (self._array_ptr(kb), kb.nbytes),
            (self._array_ptr(vp), vp.nbytes),
            (self._array_ptr(vs), vs.nbytes),
            (self._array_ptr(vb), vb.nbytes),
            (self._get_uniform("layer_kv_len", li), 4),
            (self._get_uniform("layer_capacity", li), 4),
        ]
        # sdpa_out from t["outputs"] (Stage 1 trace output)
        for b in t["outputs"]:
            bufs.append(self._buf_for_binding(b, li, cache))
        roles = {eb.get("role") for eb in t.get("extra_bindings", [])}
        if "global_scratch.sdpa_partials" in roles:
            bufs.append(self._resolve_extra("global_scratch.sdpa_partials", li, cache))
        if "global_scratch.sdpa_counter" in roles:
            bufs.append(self._resolve_extra("global_scratch.sdpa_counter", li, cache))
        return bufs

    def _quant_overlay_cache_store(self, li, cache, t):
        """Buffer list for quantized cache_store dispatch.
        Signature: K_in, V_in, cache_keys(packed/scales/biases),
        cache_values(packed/scales/biases), offset, capacity."""
        c = cache[li]
        kp, ks, kb = c.keys
        vp, vs, vb = c.values
        return [
            self._resolve_extra("layer_scratch.k_buf", li, cache),
            self._resolve_extra("layer_scratch.v_buf", li, cache),
            (self._array_ptr(kp), kp.nbytes),
            (self._array_ptr(ks), ks.nbytes),
            (self._array_ptr(kb), kb.nbytes),
            (self._array_ptr(vp), vp.nbytes),
            (self._array_ptr(vs), vs.nbytes),
            (self._array_ptr(vb), vb.nbytes),
            (self._get_uniform("layer_offset", li), 4),
            (self._get_uniform("layer_capacity", li), 4),
        ]

    def _quant_overlay_rmsnorm_rope_cache_store(self, li, cache, t):
        """Buffer list for quantized RmsNorm+RoPE+KV-cache-store.

        Signature mirrors `rmsnorm_rope_kv_cache_store_decode_quant`:
        k_proj, v_proj, k_norm_weight, K(packed/scales/biases),
        V(packed/scales/biases), offset, capacity.
        """
        c = cache[li]
        kp, ks, kb = c.keys
        vp, vs, vb = c.values
        return [
            self._buf_for_binding(t["inputs"][0], li, cache),
            self._buf_for_binding(t["inputs"][1], li, cache),
            self._buf_for_binding(t["inputs"][2], li, cache),
            (self._array_ptr(kp), kp.nbytes),
            (self._array_ptr(ks), ks.nbytes),
            (self._array_ptr(kb), kb.nbytes),
            (self._array_ptr(vp), vp.nbytes),
            (self._array_ptr(vs), vs.nbytes),
            (self._array_ptr(vb), vb.nbytes),
            (self._get_uniform("layer_offset", li), 4),
            (self._get_uniform("layer_capacity", li), 4),
        ]

    def _quant_overlay_rmsnorm_qkv(self, li, cache, t):
        """Buffer list for quantized-cache RmsNormQkv decode sink.

        The semantic task is still `RmsNormQkv`: inputs are trace/model
        params, outputs are q/k/v scratch. Only the decode-sink cache ABI
        changes from dense K/V buffers to QuantizedKVCache's
        `(packed, scales, biases)` triplets for K and V.
        """
        c = cache[li]
        kp, ks, kb = c.keys
        vp, vs, vb = c.values
        bufs = [self._buf_for_binding(b, li, cache) for b in t.get("inputs", [])]
        bufs.extend(self._buf_for_binding(b, li, cache) for b in t.get("outputs", []))
        bufs.extend([
            (self._array_ptr(kp), kp.nbytes),
            (self._array_ptr(ks), ks.nbytes),
            (self._array_ptr(kb), kb.nbytes),
            (self._array_ptr(vp), vp.nbytes),
            (self._array_ptr(vs), vs.nbytes),
            (self._array_ptr(vb), vb.nbytes),
            (self._get_uniform("layer_offset", li), 4),
            (self._get_uniform("layer_capacity", li), 4),
        ])
        return bufs

    def _task_has_qkv_decode_sink(self, t) -> bool:
        """True when an RmsNormQkv task semantically writes KV cache rows.

        Some architectures keep Q/K/V projection and cache store as
        separate graph nodes, while others sink RoPE+KVCacheStore into the
        RmsNormQkv fusion. QuantizedKVCache routing must follow that semantic
        contract, not the broad kernel-name family; otherwise a plain QKV
        projection is incorrectly dispatched as a quantized cache writer.
        """
        if "fusekit_rmsnorm_qkv_" not in t.get("kernel", ""):
            return False
        roles = {eb.get("role") for eb in t.get("extra_bindings", [])}
        return "layer_cache_k" in roles and "layer_cache_v" in roles

    def _gated_delta_overlay(self, li, cache, t):
        """Build the exact mlx CustomKernel signature for gated_delta,
        but bind recurrent state to ArraysCache[1] and force T=1 for
        decode. The trace captured prefill T>1 as a scalar input; using
        that value at decode would read invalid rows from q/k/v."""
        state = cache[li][1]
        if state is None:
            return [(0, 0)]
        if self._gated_delta_t_ptr is None:
            self._gated_delta_t_ptr = self._rt.alloc_uniform(self._handle, 4)
            ctypes.c_int32.from_address(self._gated_delta_t_ptr).value = self._active_m
        bufs = []
        for b in t["inputs"][:5]:
            bufs.append(self._buf_for_binding(b, li, cache))
        bufs.append((self._array_ptr(state), state.nbytes))
        bufs.append((self._gated_delta_t_ptr, 4))
        # y output remains a normal scratch; state_out aliases state_in.
        bufs.append(self._buf_for_binding(t["outputs"][0], li, cache))
        bufs.append((self._array_ptr(state), state.nbytes))
        if self._active_m > 1:
            mid = self._arrays_cache_rollback_arr(li, 1, cache)
            if mid is None:
                bufs.append((0, 0))
            else:
                bufs.append((self._array_ptr(mid), mid.nbytes))
        return bufs

    def _binding_elem_bytes(self, b) -> int:
        dtype = str(b.get("dtype", ""))
        if dtype in ("float32", "uint32", "int32"):
            return 4
        if dtype in ("float16", "bfloat16", "uint16", "int16"):
            return 2
        if dtype in ("uint8", "int8", "bool"):
            return 1
        return 4

    def _row_binding_for_gated_delta(self, b, li, cache, row: int):
        ptr, nbytes = self._buf_for_binding(b, li, cache)
        shape = tuple(int(v) for v in b.get("shape", ()) if int(v) > 0)
        if ptr == 0 or len(shape) < 2 or row <= 0:
            return (ptr, nbytes)
        # GatedDelta tensors are trace-shaped [B=1, T, ...]. Passing a
        # row-offset pointer plus T=1 makes the MLX custom kernel execute
        # the same layout contract as raw oMLX's split n_confirmed path.
        row_elems = 1
        for dim in shape[2:]:
            row_elems *= dim
        offset = row * row_elems * self._binding_elem_bytes(b)
        return (ptr + offset, max(0, nbytes - offset))

    def _gated_delta_split_overlays(self, li, cache, t):
        if self._active_m != 2:
            return None
        state = cache[li][1]
        if state is None:
            return None
        mid = self._arrays_cache_rollback_arr(li, 1, cache)
        if mid is None:
            return None
        if self._gated_delta_t1_ptr is None:
            self._gated_delta_t1_ptr = self._rt.alloc_uniform(self._handle, 4)
            ctypes.c_int32.from_address(self._gated_delta_t1_ptr).value = 1
        ins = t.get("inputs", [])
        outs = t.get("outputs", [])
        if len(ins) < 5 or not outs:
            return None
        y = outs[0]
        y0 = self._row_binding_for_gated_delta(y, li, cache, 0)
        y1 = self._row_binding_for_gated_delta(y, li, cache, 1)
        row0 = [
            self._row_binding_for_gated_delta(b, li, cache, 0) for b in ins[:5]
        ]
        row0.extend([
            (self._array_ptr(state), state.nbytes),
            (self._gated_delta_t1_ptr, 4),
            y0,
            (self._array_ptr(mid), mid.nbytes),
            (self._array_ptr(mid), mid.nbytes),
        ])
        row1 = [
            self._row_binding_for_gated_delta(b, li, cache, 1) for b in ins[:5]
        ]
        row1.extend([
            (self._array_ptr(mid), mid.nbytes),
            (self._gated_delta_t1_ptr, 4),
            y1,
            (self._array_ptr(state), state.nbytes),
            # Preserve the row-0 rollback snapshot. The MLX custom
            # kernel's optional mid-state output is meaningful for a
            # multi-row call, but this split row-1 call would otherwise
            # overwrite `mid` with the post-row-1 state.
            (self._array_ptr(state), state.nbytes),
        ])
        return [row0, row1]

    def _bufs_for_task(self, t, li, x_li, cache):
        extras = {eb["slot"]: eb["role"] for eb in t.get("extra_bindings", [])}
        n_inputs = len(t["inputs"])
        slots = {}
        for slot in range(n_inputs):
            if slot not in extras:
                slots[slot] = self._buf_for_binding(t["inputs"][slot], x_li, cache)
        for slot, role in extras.items():
            slots[int(slot)] = self._resolve_extra(role, li, cache)

        def scratch_id_for_extra(role: str):
            if role.startswith("layer_scratch.") and li is not None:
                name = role.split(".", 1)[1]
                return self._layer_scratch_id.get((name, li))
            if role.startswith("global_scratch."):
                name = role.split(".", 1)[1]
                return self._global_scratch_id.get(name)
            return None

        def normalized_sid(sid):
            while sid in self._scratch_alias:
                sid = self._scratch_alias[sid]
            return sid

        output_cursor = n_inputs
        for out_b in t.get("outputs", []):
            # Some synthetic kernels explicitly bind a semantic scratch
            # output via ExtraBinding (for example Flash-Decoding SDPA
            # binds sdpa_buf at slot 5, then partials/counters at 6/7).
            # Do not append that same output again.
            if out_b.get("role") == "scratch":
                out_sid = normalized_sid(out_b.get("scratch_id"))
                if any(
                    normalized_sid(scratch_id_for_extra(role)) == out_sid
                    for role in extras.values()
                ):
                    continue
            while output_cursor in slots:
                output_cursor += 1
            slots[output_cursor] = self._buf_for_binding(out_b, x_li, cache)
            output_cursor += 1

        if not slots:
            return []
        max_slot = max(slots)
        return [slots.get(slot, (0, 0)) for slot in range(max_slot + 1)]

    def _task_extra_roles(self, t) -> set[str]:
        return {eb.get("role", "") for eb in t.get("extra_bindings", [])}

    def _extra_role_is_invariant(self, role: str) -> bool:
        return role.startswith("param.") or role.startswith("layer_param.")

    def _binding_is_invariant(self, b, invariant_scratch: set[int]) -> bool:
        role = b.get("role")
        if role == "scratch":
            sid = b.get("scratch_id")
            if sid in self._scratch_alias:
                sid = self._scratch_alias[sid]
            return sid in invariant_scratch
        if role == "external":
            return bool(b.get("scalar") or b.get("source_path"))
        return False

    def _invariant_decode_task_ids(self) -> set[int]:
        """Tasks whose outputs do not depend on token/cache/offset state.

        These are mostly trace-captured constant preparation chains such
        as RoPE frequency/YARN vectors. Running them every generated
        token bloats the ICB with hundreds of tiny kernels on models
        like DeepSeek MLA. Hoist them once into their scratch buffers
        before ICB construction.
        """
        if os.environ.get("FUSEKIT_DISABLE_INVARIANT_HOIST"):
            return set()
        if self._invariant_decode_task_ids_cache is not None:
            return self._invariant_decode_task_ids_cache

        invariant_tasks: set[int] = set()
        invariant_scratch: set[int] = set()

        for t in self._decode_tasks:
            tid = int(t["id"])
            # Constness is determined by the task's bindings, not by
            # whether the op is attached to a layer. Layer-local parameter
            # broadcasts (for example per-layer bias/scale expands) are
            # still decode-invariant and should not occupy the hot ICB.
            if t.get("extra_bindings") and not all(
                self._extra_role_is_invariant(r) for r in self._task_extra_roles(t)
            ):
                continue
            if not t.get("outputs"):
                continue
            if not all(
                self._binding_is_invariant(b, invariant_scratch)
                for b in t.get("inputs", [])
            ):
                continue
            if not all(o.get("role") == "scratch" for o in t.get("outputs", [])):
                continue
            invariant_tasks.add(tid)
            for o in t.get("outputs", []):
                sid = o.get("scratch_id")
                if sid in self._scratch_alias:
                    sid = self._scratch_alias[sid]
                invariant_scratch.add(sid)

        self._invariant_decode_task_ids_cache = invariant_tasks
        return invariant_tasks

    def _build_dispatches(self, cache):
        profile_detail = bool(os.environ.get("FUSEKIT_PROFILE_BUILD_DETAIL"))
        if profile_detail:
            detail_t0 = time.perf_counter()
            self._profile_build_detail_active = True
            self._profile_scratch_alloc_us = 0.0
            self._profile_scratch_alloc_count = 0
            self._profile_scratch_alloc_bytes = 0
            self._profile_scratch_prealloc_skip_count = 0
            self._profile_scratch_prealloc_skip_bytes = 0
            self._profile_array_ptr_us = 0.0
            self._profile_array_ptr_count = 0
            self._profile_array_ptr_cache_hits = 0
            self._profile_array_ptr_preseed_us = 0.0
            self._profile_array_ptr_preseed_count = 0
        self._array_ptr_build_cache = {}
        self._preallocate_decode_scratch()
        detail_after_prealloc = time.perf_counter() if profile_detail else 0.0
        kv_quant = self._detect_kv_quant(cache)
        detail_after_kv = time.perf_counter() if profile_detail else 0.0
        self._preseed_array_ptr_cache(cache)
        detail_after_preseed = time.perf_counter() if profile_detail else 0.0
        out = []
        labels = []
        invariant_ids = self._invariant_decode_task_ids()
        detail_after_invariant = time.perf_counter() if profile_detail else 0.0
        invariant_out = []
        # Force-allocate the shared uniforms before the prepend so
        # their pointers are stable when we close the GPU feedback
        # loop. Layer tasks normally populate `_uniform_ptrs` via
        # `_resolve_extra`, but those run AFTER the prepended task.
        offset_uniform_ptr = self._get_uniform("layer_offset", None)
        kv_len_uniform_ptr = self._get_uniform("layer_kv_len", None)
        _ = self._get_uniform("layer_capacity", None)
        next_token_ptr = self._get_uniform("next_token_out", None)
        barrier_ptr, barrier_count_ptr, barrier_nbytes = self._ensure_barrier_counters()
        out.append((
            "fusekit_step_init",
            [
                (self._step_counter_ptr, 4),
                (offset_uniform_ptr, 4),
                (kv_len_uniform_ptr, 4),
                (barrier_ptr, barrier_nbytes),
                (barrier_count_ptr, 4),
            ],
            (1, 1, 1),
            (1, 1, 1),
        ))
        labels.append({
            "task_id": None,
            "fusion_id": None,
            "layer_idx": None,
            "kernel": "fusekit_step_init",
            "kind": "runtime",
            "read_slots": [0, 4],
            "write_slots": [0, 1, 2, 3],
        })
        detail_after_prelude = time.perf_counter() if profile_detail else 0.0
        detail_bufs_us = 0.0
        detail_overlay_us = 0.0
        detail_linres_us = 0.0
        detail_task_count = 0
        detail_embed_skip = 0
        detail_view_skip = 0
        detail_invariant_count = 0
        linres_per_layer: dict = {}
        skip_direct_argmax_tail = 0
        history_capture_fused = False
        for t in self._decode_tasks:
            tid = int(t["id"])
            detail_task_count += 1
            li = t.get("layer_idx")
            x_li = li if li is not None else 0
            kernel = t["kernel"]
            if skip_direct_argmax_tail > 0 and kernel.startswith("fusekit_argmax_logits_"):
                skip_direct_argmax_tail -= 1
                continue
            if self._host_embed_lookup and "fusekit_embed_lookup_decode_" in kernel:
                detail_embed_skip += 1
                continue
            detail_bind_t0 = time.perf_counter() if profile_detail else 0.0
            detail_overlay = False
            direct_head_argmax = self._is_direct_mtp_head_qmm_task(t, kernel, li)
            if direct_head_argmax:
                bufs = [
                    self._buf_for_binding(b, x_li, cache)
                    for b in t.get("inputs", [])
                ]
            elif "gated_delta_step" in kernel and li is not None:
                bufs = self._gated_delta_overlay(li, cache, t)
                detail_overlay = True
            # KV-quant variant routing: swap kernel name + use quant
            # overlay buffer construction. The quant kernels have a
            # different signature (3-tuple K/V cache per slot) so the
            # standard inputs/extras/outputs flow doesn't apply.
            elif kv_quant is not None and li is not None:
                bits, _gs = kv_quant
                if "fusekit_sdpa_decode_f" in kernel:
                    kernel = kernel.replace(
                        "fusekit_sdpa_decode_f",
                        f"fusekit_sdpa_decode_q{bits}_f",
                    )
                    bufs = self._quant_overlay_sdpa(li, cache, t)
                    detail_overlay = True
                elif "fusekit_rmsnorm_qkv_f" in kernel and self._task_has_qkv_decode_sink(t):
                    kernel = kernel.replace(
                        "fusekit_rmsnorm_qkv_f",
                        f"fusekit_rmsnorm_qkv_q{bits}_f",
                    )
                    bufs = self._quant_overlay_rmsnorm_qkv(li, cache, t)
                    detail_overlay = True
                elif "fusekit_rmsnorm_rope_kv_cache_store_decode_f" in kernel:
                    kernel = kernel.replace(
                        "fusekit_rmsnorm_rope_kv_cache_store_decode_f",
                        f"fusekit_rmsnorm_rope_kv_cache_store_decode_q{bits}_f",
                    )
                    bufs = self._quant_overlay_rmsnorm_rope_cache_store(li, cache, t)
                    detail_overlay = True
                elif "fusekit_rope_kv_cache_store_decode_f" in kernel:
                    kernel = kernel.replace(
                        "fusekit_rope_kv_cache_store_decode_f",
                        f"fusekit_rope_kv_cache_store_decode_q{bits}_f",
                    )
                    bufs = self._quant_overlay_cache_store(li, cache, t)
                    detail_overlay = True
                else:
                    bufs = self._bufs_for_task(t, li, x_li, cache)
            else:
                bufs = self._bufs_for_task(t, li, x_li, cache)
            if profile_detail:
                bind_us = (time.perf_counter() - detail_bind_t0) * 1_000_000.0
                if detail_overlay:
                    detail_overlay_us += bind_us
                else:
                    detail_bufs_us += bind_us
            if (
                self._is_inplace_view_kernel(kernel)
                and len(t.get("inputs", [])) == 1
                and len(t.get("outputs", [])) == 1
                and len(bufs) >= 2
                and bufs[0][0] == bufs[-1][0]
            ):
                # Safe view chains are aliased to one scratch id by the
                # compiler. A materializing transpose/reshape cannot run
                # in-place; preserving the alias is the correct action.
                detail_view_skip += 1
                continue
            # LinearResidual `x` slot needs rerouting around intermediate
            # Single ops that don't make it into decode_pipeline.
            #   o_proj (first per layer): bind to SDPA scratch (skip the
            #     transpose+reshape between SDPA and o_proj).
            #   down_proj (second per layer): bind to silu_up scratch
            #     (skip the flatten between rmsnorm_gateup_swiglu and
            #     down_proj — present in bf16 traces but absent in 4bit
            #     because the trace shapes differ by a leading dim).
            if "fusekit_linear_residual_" in t["kernel"] and li is not None:
                detail_linres_t0 = time.perf_counter() if profile_detail else 0.0
                seen = linres_per_layer.setdefault(li, 0)
                x_shape = (
                    t.get("inputs", [{}])[0].get("shape", [])
                    if t.get("inputs")
                    else []
                )
                x_last = x_shape[-1] if x_shape else 0
                x_sid = (
                    t.get("inputs", [{}])[0].get("scratch_id")
                    if t.get("inputs")
                    and t.get("inputs", [{}])[0].get("role") == "scratch"
                    else None
                )
                x_has_pipeline_producer = x_sid in self._decode_output_scratch_ids
                if (
                    seen == 0
                    and not x_has_pipeline_producer
                    and ("sdpa_buf", li) in self._layer_scratch_id
                ):
                    bufs[0] = self._resolve_extra(
                        "layer_scratch.sdpa_buf", li, cache
                    )
                elif (
                    x_last != self._hidden
                    and not x_has_pipeline_producer
                    and ("silu_up", li) in self._layer_scratch_id
                ):
                    bufs[0] = self._resolve_extra(
                        "layer_scratch.silu_up", li, cache
                    )
                linres_per_layer[li] = seen + 1
                if profile_detail:
                    detail_linres_us += (
                        time.perf_counter() - detail_linres_t0
                    ) * 1_000_000.0
            if any(b == (0, 0) for b in bufs):
                raise RuntimeError(
                    f"task {tid} ({kernel}) unbound slot; bufs={bufs}"
                )
            gx, gy, gz = t["grid_threads"]
            tx, ty, tz = t["threadgroup"]
            if (
                kv_quant is not None
                and li is not None
                and "fusekit_rope_kv_cache_store_decode_q" in kernel
            ):
                # Quantized cache-store kernels map one K/V head row to one
                # threadgroup: first N_KV groups write K, next N_KV write V.
                c = cache[li]
                packed_keys = c.keys[0]
                shape = packed_keys.shape
                n_kv = int(shape[1] if len(shape) >= 4 else shape[0])
                gx = self._active_m * 2 * n_kv * tx
            if (
                self._active_m == 2
                and li is not None
                and "gated_delta_step" in kernel
                and not os.environ.get("FUSEKIT_DISABLE_MTP_SPLIT_GATED_DELTA")
            ):
                split_bufs = self._gated_delta_split_overlays(li, cache, t)
                if split_bufs is not None:
                    for part, part_bufs in enumerate(split_bufs):
                        if any(b == (0, 0) for b in part_bufs):
                            raise RuntimeError(
                                f"task {tid} ({kernel}) split gated_delta unbound slot; "
                                f"part={part} bufs={part_bufs}"
                            )
                        out.append((kernel, part_bufs, (gx, gy, gz), (tx, ty, tz)))
                        labels.append({
                            "task_id": tid,
                            "fusion_id": t.get("fusion_id"),
                            "layer_idx": li,
                            "kernel": kernel,
                            "kind": "gated_delta",
                            "split_part": part,
                    })
                    continue
            if kernel.startswith("fusekit_argmax_logits_finalize_"):
                bufs = bufs + [
                    (self._hist_counter_ptr, 4),
                    (self._history_ptr, self._history_nbytes),
                ]
                history_capture_fused = True
            if tid in invariant_ids:
                invariant_out.append((kernel, bufs, (gx, gy, gz), (tx, ty, tz)))
                detail_invariant_count += 1
                continue
            if direct_head_argmax:
                if kernel.startswith("fusekit_final_rmsnorm_lm_head_"):
                    total_tgs = max(1, (int(gx) + max(1, int(tx)) - 1) // max(1, int(tx)))
                    n_groups = max(1, total_tgs // max(1, int(self._active_m)))
                else:
                    n_groups = max(1, (int(gy) + max(1, int(ty)) - 1) // max(1, int(ty)))
                scratch = self._qmv_argmax_scratch_binding(n_groups)
                next_nbytes = max(1, self._active_m) * 4
                next_ptr = self._get_uniform_n("next_token_out", None, next_nbytes)
                scan_kernel = f"{kernel}_argmax"
                finalize_kernel = f"{kernel}_argmax_finalize"
                scan_bufs = bufs + [scratch]
                if kernel.startswith("fusekit_final_rmsnorm_lm_head_"):
                    scan_read_slots = list(range(len(bufs)))
                    scan_write_slots = [len(bufs)]
                else:
                    scan_read_slots = [0, 1, 2, 3]
                    scan_write_slots = [4]
                out.append((scan_kernel, scan_bufs, (gx, gy, gz), (tx, ty, tz)))
                labels.append({
                    "task_id": tid,
                    "fusion_id": t.get("fusion_id"),
                    "layer_idx": li,
                    "kernel": scan_kernel,
                    "kind": "qmv_argmax",
                    "read_slots": scan_read_slots,
                    "write_slots": scan_write_slots,
                })
                out.append((
                    finalize_kernel,
                    [scratch, (next_ptr, next_nbytes)],
                    (max(1, self._active_m) * 256, 1, 1),
                    (256, 1, 1),
                ))
                labels.append({
                    "task_id": tid,
                    "fusion_id": t.get("fusion_id"),
                    "layer_idx": li,
                    "kernel": finalize_kernel,
                    "kind": "qmv_argmax",
                    "read_slots": [0],
                    "write_slots": [1],
                })
                skip_direct_argmax_tail = 2
                continue
            label = {
                "task_id": tid,
                "fusion_id": t.get("fusion_id"),
                "layer_idx": li,
                "kernel": kernel,
                "kind": self._kernel_kind(kernel),
            }
            if "fusekit_chain_" in kernel:
                # FusedChain kernels append every fusion output after
                # the input/extra-binding slots. The generic kernel-name
                # fallback only knows about a single trailing output, so
                # multi-output fanout chains (RMSNorm -> z/qkv/a/b, gate/up,
                # etc.) need task metadata to avoid consumers racing ahead.
                n_writes = len(t.get("outputs", []))
                n_reads = max(0, len(bufs) - n_writes)
                label["read_slots"] = list(range(n_reads))
                label["write_slots"] = list(range(n_reads, len(bufs)))
            elif "fusekit_rmsnorm_qkv_" in kernel:
                n_inputs = len(t.get("inputs", []))
                n_outputs = len(t.get("outputs", []))
                output_slots = list(range(n_inputs, n_inputs + n_outputs))
                if "_q4_" in kernel or "_q8_" in kernel:
                    cache_base = n_inputs + n_outputs
                    read_slots = list(range(n_inputs)) + [cache_base + 6, cache_base + 7]
                    write_slots = output_slots + list(range(cache_base, cache_base + 6))
                else:
                    read_slots = list(range(n_inputs))
                    write_slots = output_slots
                    for eb in t.get("extra_bindings", []):
                        slot = int(eb.get("slot", -1))
                        role = eb.get("role", "")
                        if slot < 0:
                            continue
                        if role in ("layer_cache_k", "layer_cache_v"):
                            write_slots.append(slot)
                        else:
                            read_slots.append(slot)
                label["read_slots"] = sorted(set(read_slots))
                label["write_slots"] = sorted(set(write_slots))
            out.append((kernel, bufs, (gx, gy, gz), (tx, ty, tz)))
            labels.append(label)
        detail_after_loop = time.perf_counter() if profile_detail else 0.0
        if not history_capture_fused:
            # Append history-capture: write the just-produced argmax to
            # `history[atomic_idx]` so the streaming consumer can read
            # all chunk tokens after one wait_until_completed.
            out.append((
                "fusekit_history_capture",
                [
                    (self._hist_counter_ptr, 4),
                    (next_token_ptr, 4),
                    (self._history_ptr, self._history_nbytes),
                ],
                (1, 1, 1),
                (1, 1, 1),
            ))
            labels.append({
                "task_id": None,
                "fusion_id": None,
                "layer_idx": None,
                "kernel": "fusekit_history_capture",
                "kind": "runtime",
                "read_slots": [0, 1],
                "write_slots": [0, 2],
            })
        self._last_dispatch_labels = labels
        if invariant_out and not self._invariant_dispatches_done:
            if os.environ.get("FUSEKIT_TRACE_INVARIANT_HOIST"):
                print(
                    f"[fusekit] hoisting {len(invariant_out)} decode-invariant kernels",
                    file=sys.stderr,
                    flush=True,
                )
            self._rt.dispatch_batch(self._handle, invariant_out)
            self._invariant_dispatches_done = True
        if profile_detail:
            detail_end = time.perf_counter()
            self._profile_build_detail_active = False
            print(
                "[fusekit-build-detail] "
                f"active_m={self._active_m} "
                f"tasks={detail_task_count} "
                f"dispatches={len(out)} "
                f"invariant={detail_invariant_count} "
                f"embed_skip={detail_embed_skip} "
                f"view_skip={detail_view_skip} "
                f"prealloc_us={(detail_after_prealloc - detail_t0) * 1_000_000.0:.2f} "
                f"kv_us={(detail_after_kv - detail_after_prealloc) * 1_000_000.0:.2f} "
                f"array_ptr_preseed_count={self._profile_array_ptr_preseed_count} "
                f"array_ptr_preseed_us={self._profile_array_ptr_preseed_us:.2f} "
                f"invariant_us={(detail_after_invariant - detail_after_preseed) * 1_000_000.0:.2f} "
                f"prelude_us={(detail_after_prelude - detail_after_invariant) * 1_000_000.0:.2f} "
                f"loop_us={(detail_after_loop - detail_after_prelude) * 1_000_000.0:.2f} "
                f"bufs_us={detail_bufs_us:.2f} "
                f"overlay_us={detail_overlay_us:.2f} "
                f"linres_fixup_us={detail_linres_us:.2f} "
                f"scratch_allocs={self._profile_scratch_alloc_count} "
                f"scratch_alloc_mb={self._profile_scratch_alloc_bytes / 1_000_000.0:.2f} "
                f"scratch_alloc_us={self._profile_scratch_alloc_us:.2f} "
                f"scratch_prealloc_skip={self._profile_scratch_prealloc_skip_count} "
                f"scratch_prealloc_skip_mb={self._profile_scratch_prealloc_skip_bytes / 1_000_000.0:.2f} "
                f"array_ptr_calls={self._profile_array_ptr_count} "
                f"array_ptr_cache_hits={self._profile_array_ptr_cache_hits} "
                f"array_ptr_us={self._profile_array_ptr_us:.2f} "
                f"hoist_us={(detail_end - detail_after_loop) * 1_000_000.0:.2f} "
                f"total_us={(detail_end - detail_t0) * 1_000_000.0:.2f}",
                flush=True,
            )
        self._array_ptr_build_cache = None
        return out

    def _kernel_kind(self, kernel: str) -> str:
        if (
            "fusekit_op_quantizedmatmul_" in kernel
            and "_argmax" in kernel
        ):
            return "qmv_argmax"
        if kernel.startswith("fusekit_op_"):
            return self._kernel_family(kernel)
        if kernel.startswith("fusekit_qmv_fanout_"):
            return "qmv_fanout"
        if kernel.startswith("fusekit_qmv_swiglu_"):
            return "qmv_swiglu"
        if kernel.startswith("fusekit_linear_residual_"):
            return "linear_residual"
        if "fusekit_chain_" in kernel:
            return "chain"
        if "gated_delta_step" in kernel:
            return "gated_delta"
        if "arrays_cache_shift_append_decode" in kernel:
            return "arrays_cache"
        if "fusekit_rmsnorm_qkv_" in kernel:
            return "qkv"
        if "sdpa_decode" in kernel:
            return "sdpa"
        if "kv_cache_store" in kernel:
            return "cache_store"
        if "rope_" in kernel:
            return "rope"
        if "embed_lookup" in kernel:
            return "embed"
        if "argmax" in kernel:
            return "argmax"
        return "other"

    def _is_inplace_view_kernel(self, kernel: str) -> bool:
        return any(
            name in kernel
            for name in (
                "fusekit_op_reshape_",
                "fusekit_op_transpose_",
                "fusekit_op_flatten_",
                "fusekit_op_unflatten_",
                "fusekit_op_squeeze_",
                "fusekit_op_expanddims_",
                "fusekit_op_ascontiguous_",
            )
        )

    def _detect_kv_quant(self, cache):
        """Return (bits, group_size) if cache is QuantizedKVCache, else None.
        Detect by inspecting `c.keys` shape: tuple of 3 mx.arrays =
        `(packed_uint32, scales, biases)`."""
        if not cache:
            return None
        c0 = next((c for c in cache if hasattr(c, "keys") and c.keys is not None), None)
        if c0 is None:
            return None
        keys = c0.keys
        if isinstance(keys, (tuple, list)) and len(keys) == 3:
            bits = getattr(c0, "bits", None)
            group_size = getattr(c0, "group_size", None)
            if bits in (4, 8):
                return (bits, group_size or 64)
        return None

    def _graph_has_qkv_decode_sink(self) -> bool:
        cached = getattr(self, "_has_qkv_decode_sink_cache", None)
        if cached is not None:
            return bool(cached)
        has_sink = False
        for t in self._decode_tasks:
            if self._task_has_qkv_decode_sink(t):
                has_sink = True
                break
        self._has_qkv_decode_sink_cache = has_sink
        return has_sink

    def _supports_quantized_qkv_decode_sink(self, kv_quant) -> bool:
        if not kv_quant:
            return False
        bits, group_size = kv_quant
        return bits in (4, 8) and int(group_size or 64) == 64

    # Cache classes whose K/V layout we know how to extract pointers
    # from. Anything else (ArraysCache for state-space layers,
    # ChunkedKVCache, MambaCache, etc.) gets routed to the mlx-lm
    # native path — fusekit can't accelerate them yet and silently
    # disabling lets correctness ride on mlx's own kernels.
    _SUPPORTED_CACHE_TYPES = (
        "KVCache",
        "QuantizedKVCache",
        "RotatingKVCache",
        "ArraysCache",
        "BatchKVCache",
    )

    def _iter_cache_arrays(self, c):
        cls_name = type(c).__name__
        if cls_name == "ArraysCache":
            for idx, arr in enumerate(getattr(c, "cache", [])):
                if hasattr(arr, "dtype"):
                    yield ("arrays", idx), arr
            return
        for name in ("keys", "values"):
            obj = getattr(c, name, None)
            if isinstance(obj, (tuple, list)):
                for idx, arr in enumerate(obj):
                    if hasattr(arr, "dtype"):
                        yield (name, idx), arr
            elif hasattr(obj, "dtype"):
                yield (name, None), obj

    def _cache_layer_layout_sig(self, c):
        cls_name = type(c).__name__
        if cls_name not in self._SUPPORTED_CACHE_TYPES:
            return None
        if cls_name == "BatchKVCache" and not self._batch_kv_single_ok(c):
            return None
        arrays = []
        for desc, arr in self._iter_cache_arrays(c):
            arrays.append((
                desc,
                tuple(int(x) for x in getattr(arr, "shape", ())),
                str(getattr(arr, "dtype", "")),
                int(getattr(arr, "nbytes", 0)),
            ))
        quant = ()
        if cls_name == "QuantizedKVCache":
            quant = (getattr(c, "bits", None), getattr(c, "group_size", None))
        return (cls_name, quant, tuple(arrays))

    def _cache_binding_map(self, cache) -> dict[int, tuple]:
        out: dict[int, tuple] = {}
        for li, c in enumerate(cache):
            for desc, arr in self._iter_cache_arrays(c):
                ptr = self._array_ptr(arr)
                if ptr:
                    out[ptr] = (li, *desc)
        for (li, idx), arr in self._arrays_cache_rollback.items():
            if arr is None:
                continue
            ptr = self._array_ptr(arr)
            if ptr:
                out[ptr] = (li, "arrays_rollback", idx)
        return out

    def _resolve_cache_binding_desc(self, desc: tuple, cache) -> tuple[int, int]:
        li, name, idx = desc
        c = cache[li]
        if name == "arrays":
            arr = getattr(c, "cache", [])[idx]
        elif name == "arrays_rollback":
            arr = self._arrays_cache_rollback.get((li, idx))
        else:
            obj = getattr(c, name)
            arr = obj[idx] if idx is not None else obj
        if arr is None:
            return (0, 0)
        return (self._array_ptr(arr), arr.nbytes)

    def _capture_dispatch_rebind_plan(self, cache, dispatches: list) -> None:
        bindings = self._cache_binding_map(cache)
        plan: list[tuple[int, int, tuple]] = []
        for di, (_kernel, bufs, _grid, _tg) in enumerate(dispatches):
            for bi, (ptr, _nbytes) in enumerate(bufs):
                desc = bindings.get(ptr)
                if desc is not None:
                    plan.append((di, bi, desc))
        self._dispatch_rebind_plan = plan

    def _rebind_dispatch_cache_buffers(self, cache) -> list[tuple[int, int, int, int]] | None:
        self._last_rebind_failure = None
        if self._last_dispatches is None or self._dispatch_rebind_plan is None:
            self._last_rebind_failure = "missing_dispatch_or_plan"
            return None
        updates: list[tuple[int, int, int, int]] = []
        for di, bi, desc in self._dispatch_rebind_plan:
            ptr, nbytes = self._resolve_cache_binding_desc(desc, cache)
            if ptr == 0:
                self._last_rebind_failure = f"unresolved:{desc}"
                return None
            _kernel, bufs, _grid, _tg = self._last_dispatches[di]
            bufs[bi] = (ptr, nbytes)
            updates.append((di, bi, ptr, nbytes))
        return updates

    def _cache_layer_sig(self, c):
        """One slot's signature contribution: returns a tuple whose
        last element is the K-buffer pointer (used by `_ensure_icb` to
        decide whether the ICB needs a rebuild). Returns `None` when
        the cache class isn't one fusekit knows how to dispatch
        against — callers must bail to native in that case."""
        cls_name = type(c).__name__
        if cls_name not in self._SUPPORTED_CACHE_TYPES:
            return None
        if cls_name == "BatchKVCache" and not self._batch_kv_single_ok(c):
            return None
        ptrs = [self._array_ptr(arr) for _desc, arr in self._iter_cache_arrays(c)]
        return (cls_name, *(ptrs or [0]))

    def _ensure_icb(self, cache) -> int | None:
        # Detect cache classes per layer. If ANY layer is in a class we
        # don't dispatch yet, bail this whole session — mixed-cache
        # decode (e.g. Qwen3.5/3.6 hybrid linear+full attention) isn't
        # something the current ICB layout can handle.
        if self._fk_unsupported_cache:
            return None
        rollback_buffers_changed = False
        if self._active_m > 1:
            rollback_buffers_changed = self._prepare_arrays_cache_rollback_buffers(cache)
            if (
                rollback_buffers_changed
                and os.environ.get("FUSEKIT_DISABLE_ROLLBACK_REBIND")
            ):
                self._dispatch_layout_sig = None
                self._last_dispatches = None
                self._last_dispatch_labels = None
                self._last_barriers = None
                self._icb_id = None
                self._cache_sig = None
        per_layer_sigs = [self._cache_layer_sig(c) for c in cache]
        if any(s is None for s in per_layer_sigs):
            unknown = sorted({type(c).__name__ for c, s in zip(cache, per_layer_sigs) if s is None})
            warnings.warn(
                f"fusekit: cache types {unknown} unsupported; "
                "falling back to native dispatch for this session.",
                stacklevel=2,
            )
            self._fk_unsupported_cache = True
            return None
        # All entries are non-None at this point.
        layer_sigs = [s for s in per_layer_sigs if s is not None]
        layer_layout_sigs = tuple(self._cache_layer_layout_sig(c) for c in cache)
        if any(s is None for s in layer_layout_sigs):
            return None
        kv_quant = self._detect_kv_quant(cache)
        if (
            kv_quant
            and self._graph_has_qkv_decode_sink()
            and not self._supports_quantized_qkv_decode_sink(kv_quant)
        ):
            msg = (
                "fusekit: QKV decode sink writes dense KV cache rows, "
                f"but runtime cache is QuantizedKVCache(bits={kv_quant[0]}, "
                f"group_size={kv_quant[1]}); quantized QKV sink emitter supports "
                "only bits in {4, 8} with group_size=64."
            )
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise RuntimeError(msg)
            if not self._fk_unsupported_graph_warned:
                warnings.warn(msg + " Falling back to native dispatch for this session.", stacklevel=2)
                self._fk_unsupported_graph_warned = True
            self._fk_unsupported_cache = True
            return None
        sig_kind = ("quant", kv_quant[0]) if kv_quant else "bf16"
        direct_head_argmax = self._direct_head_argmax_enabled()
        mode_sig = ("direct_head_argmax", int(direct_head_argmax))
        layer_layout_sigs = (mode_sig, layer_layout_sigs)
        live_sig = (mode_sig,) + tuple((sig_kind,) + tuple(s[1:]) for s in layer_sigs)
        if any(ptr == 0 for s in live_sig[1:] for ptr in s[1:]):
            return None  # cache unallocated — caller falls back
        sig = live_sig
        if self._cache_sig == sig and self._icb_id is not None:
            return self._icb_id
        profile_build = bool(os.environ.get("FUSEKIT_PROFILE_BUILD"))
        t0 = time.perf_counter() if profile_build else 0.0
        rebound = False
        rebind_blocked_reason = None
        if (
            self._dispatch_layout_sig == layer_layout_sigs
            and self._last_dispatches is not None
            and self._dispatch_rebind_plan is not None
        ):
            updates = self._rebind_dispatch_cache_buffers(cache)
            rebound = updates is not None
            if not rebound:
                rebind_blocked_reason = self._last_rebind_failure or "updates_none"
            if rebound and self._icb_id is not None and hasattr(self._rt, "update_icb_bindings"):
                self._rt.update_icb_bindings(self._handle, self._icb_id, updates)
                self._cache_sig = sig
                if profile_build:
                    t1 = time.perf_counter()
                    print(
                        "[fusekit-build-profile] "
                        f"rebind=1 dispatches={len(self._last_dispatches)} "
                        f"update_icb_us={(t1 - t0) * 1_000_000.0:.2f} "
                        "reused_icb=1",
                        flush=True,
                    )
                return self._icb_id
            d = self._last_dispatches if rebound else self._build_dispatches(cache)
        else:
            if profile_build:
                if self._dispatch_layout_sig != layer_layout_sigs:
                    rebind_blocked_reason = "layout_changed"
                elif self._last_dispatches is None:
                    rebind_blocked_reason = "missing_last_dispatches"
                elif self._dispatch_rebind_plan is None:
                    rebind_blocked_reason = "missing_rebind_plan"
            d = self._build_dispatches(cache)
        t1 = time.perf_counter() if profile_build else 0.0
        self._last_dispatches = d
        if not rebound:
            self._dispatch_layout_sig = layer_layout_sigs
            self._capture_dispatch_rebind_plan(cache, d)
            self._last_barriers = _barrier_mask(d, self._last_dispatch_labels)
        barriers = self._last_barriers if self._last_barriers is not None else _barrier_mask(d, self._last_dispatch_labels)
        barrier_count = sum(1 for b in barriers if b)
        t2 = time.perf_counter() if profile_build else 0.0
        self._icb_id = self._rt.build_icb(self._handle, d, barriers)
        t3 = time.perf_counter() if profile_build else 0.0
        self._cache_sig = sig
        # Wire up the model state the `_wrap_generate_step` fast path
        # expects so subsequent generate-step calls can pipeline the
        # ICB submit with Python sampling.
        self._install_fast_path_state()
        if profile_build:
            t4 = time.perf_counter()
            kind_counts: dict[str, int] = {}
            kind_barriers: dict[str, int] = {}
            for i, label in enumerate(self._last_dispatch_labels or []):
                kind = str(label.get("kind", "unknown")) if isinstance(label, dict) else "unknown"
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                if i < len(barriers) and barriers[i]:
                    kind_barriers[kind] = kind_barriers.get(kind, 0) + 1
            kind_parts = []
            for kind, count in sorted(kind_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
                kind_parts.append(f"{kind}:{kind_barriers.get(kind, 0)}/{count}")
            print(
                "[fusekit-build-profile] "
                f"rebind={int(rebound)} "
                f"reason={rebind_blocked_reason or 'ok'} "
                f"rollback_changed={int(rollback_buffers_changed)} "
                f"dispatches={len(d)} "
                f"barriers={barrier_count} "
                f"barrier_kinds={','.join(kind_parts)} "
                f"build_dispatches_us={(t1 - t0) * 1_000_000.0:.2f} "
                f"barriers_us={(t2 - t1) * 1_000_000.0:.2f} "
                f"build_icb_us={(t3 - t2) * 1_000_000.0:.2f} "
                f"install_state_us={(t4 - t3) * 1_000_000.0:.2f} "
                f"total_us={(t4 - t0) * 1_000_000.0:.2f}",
                flush=True,
            )
        return self._icb_id

    def _install_fast_path_state(self) -> None:
        """Populate `model._fk_lm` and per-layer `_fk["uniform_ptrs"]`
        on the inner model so `_wrap_generate_step` recognises us
        and switches to the async submit + wait pipeline. Stores raw
        uniform pointers (not ctypes views) so the per-step uniform
        fanout can go through one native FFI call rather than 24×
        Python writes."""
        if self._active_m != 1:
            return
        next_view = (ctypes.c_uint32 * 1).from_address(
            self._uniform_ptrs[("next_token_out", None)]
        )
        logits = self._scratch_buf(self._global_scratch_id["logits"])
        # mlx-lm.generate slices `logits[:, -1, :]` so we need (B, L, V)
        # shape with at least one row of valid data — slice the row we
        # actually compute. Some traces (mlx-lm bf16 models) drop the
        # leading batch dim from the lm_head output, leaving a 2D
        # `(L, V)` scratch. Handle both ranks by slicing the prefill
        # row at axis 0 / axis 1 as needed and reshaping to a fixed
        # (B=1, L=1, V) buffer for the downstream copy.
        if logits.ndim == 3:
            logits_view = logits[:, :1, :]
        elif logits.ndim == 2:
            logits_view = logits[:1, :].reshape(1, 1, -1)
        else:
            raise ValueError(
                f"unexpected logits scratch rank {logits.ndim}, shape {logits.shape}"
            )
        # Shared-across-layers uniforms: a single uint32 each. Layer-
        # per-layer `_fk` is no longer needed by the fast loop, but
        # keep it bound in case downstream code reaches for it.
        offset_ptr = self._uniform_ptrs[("layer_offset", None)]
        capacity_ptr = self._uniform_ptrs[("layer_capacity", None)]
        kv_len_ptr = self._uniform_ptrs[("layer_kv_len", None)]
        layers = self._decoder_layers()
        for li in range(len(layers)):
            layer = layers[li]
            layer.__dict__["_fk"] = {
                "uniform_ptrs": (offset_ptr, capacity_ptr, kv_len_ptr),
            }
        self._model.__dict__["_fk_lm"] = {
            "rt": self._rt,
            "handle": self._handle,
            "icb_id": self._icb_id,
            # token_id_in aliases next_token_out — same buffer.
            "token_id_ptr": self._uniform_ptrs[("next_token_out", None)],
            "next_token_ptr": self._uniform_ptrs[("next_token_out", None)],
            "layer_offset_ptr": offset_ptr,
            "layer_capacity_ptr": capacity_ptr,
            "layer_kv_len_ptr": kv_len_ptr,
            "step_counter_ptr": self._step_counter_ptr,
            "hist_counter_ptr": self._hist_counter_ptr,
            "history_ptr": self._history_ptr,
            "barrier_counter_ptrs": list(self._barrier_counter_ptrs.values()),
            "chunk_capacity": self._chunk_capacity,
            "next_token_view": next_view,
            "logits_buf": logits_view,
            "direct_head_argmax": bool(self._direct_head_argmax_enabled()),
        }

    # ── per-step entry point ────────────────────────────────────

    def try_run(self, inputs, cache, *, return_hidden: bool = False, n_confirmed: int = 0):
        """Run one decode step if the input shape and cache state are
        compatible with the ICB. Returns the logits as a (1, 1, vocab)
        mx.array, or `None` to signal the caller should fall back."""
        self.last_argmax_tokens = None
        if cache is None or len(inputs.shape) != 2:
            return None
        if (
            self._active_m > 1
            and inputs.shape[1] == self._active_m
            and return_hidden
            and n_confirmed == 1
            and _env_flag("FUSEKIT_USE_MTP_MN_VARIANT")
            and not _env_flag("FUSEKIT_DISABLE_MTP_MN_VARIANT")
        ):
            return self._try_run_multi(inputs, cache, return_hidden=True)
        if (
            inputs.shape[1] == 2
            and return_hidden
            and n_confirmed == 1
            and _env_flag("FUSEKIT_MTP_SINGLE_BACKBONE")
            and not _env_flag("FUSEKIT_DISABLE_MTP_BACKBONE")
        ):
            return self._try_run_mtp_verify(inputs, cache)
        return self._try_run_single(inputs, cache, return_hidden=return_hidden)

    def _cache_needs_mtp_mid_rollback(self, cache) -> bool:
        for c in cache:
            state = getattr(c, "state", None)
            if isinstance(state, list) and len(state) >= 2:
                return True
            if type(c).__name__ == "ArraysCache":
                return True
        return False

    def _try_run_multi(self, inputs, cache, *, return_hidden: bool = False):
        m = int(inputs.shape[1])
        if m != self._active_m:
            return None
        # For MTP verify calls with n_confirmed=1 the engine needs the
        # recurrent ArraysCache state after the confirmed first token,
        # regardless of how many speculative rows were evaluated. The
        # emitted cache/custom-kernel bindings snapshot exactly that
        # post-row-0 state, so the same rollback semantic works for
        # M=2 and larger M variants.
        if self._fk_unsupported_graph:
            return None
        if self._fk_unsupported_cache or self._fk_dense_bf16_disabled:
            return None
        c0 = next((c for c in cache if hasattr(c, "offset")), None)
        if c0 is None:
            return None
        o = self._cache_decode_offset(c0)
        if o is None:
            return None
        ok, reason = _ensure_cache_capacity(cache, o + m)
        if not ok:
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise RuntimeError(
                    "fusekit: cannot guarantee KV cache capacity for "
                    f"M={m} decode offset {o} ({reason})"
                )
            return None
        icb_id = self._ensure_icb(cache)
        if icb_id is None:
            return None

        mx = self._mx
        flat = inputs.flatten().astype(mx.uint32)
        mx.eval(flat)
        if self._host_embed_lookup:
            rows = self._embed_module(inputs).reshape((m, self._hidden)).astype(self._act_dt)
            mx.eval(rows)
            ctypes.memmove(self._array_ptr(self._x_in), self._array_ptr(rows), rows.nbytes)
        token_ptr = self._get_uniform_n("token_id_in", None, m * 4)
        ctypes.memmove(token_ptr, self._array_ptr(flat), m * 4)

        ctypes.c_uint32.from_address(self._step_counter_ptr).value = o
        ctypes.c_uint32.from_address(self._hist_counter_ptr).value = 0
        ctypes.c_uint32.from_address(
            self._uniform_ptrs[("layer_offset", None)]
        ).value = o
        ctypes.c_uint32.from_address(
            self._uniform_ptrs[("layer_kv_len", None)]
        ).value = o + 1
        keys_repr = c0.keys[0] if isinstance(c0.keys, (tuple, list)) else c0.keys
        if keys_repr is None:
            return None
        ctypes.c_uint32.from_address(
            self._uniform_ptrs[("layer_capacity", None)]
        ).value = keys_repr.shape[2]
        for ptr in self._barrier_counter_ptrs.values():
            ctypes.c_uint32.from_address(ptr).value = 0
        prof_state = self._profile_mtp_state and self._active_m > 1
        prof_t0 = time.perf_counter() if prof_state else 0.0
        if self._active_m > 1:
            for li, c in enumerate(cache):
                if type(c).__name__ == "ArraysCache":
                    for idx in range(len(getattr(c, "cache", []))):
                        self._arrays_cache_rollback_arr(li, idx, cache)
        prof_t1 = time.perf_counter() if prof_state else 0.0
        self._advance_decode_cache(cache, o + m, steps=m)
        prof_t2 = time.perf_counter() if prof_state else 0.0
        prof_exec0 = time.perf_counter() if prof_state else 0.0
        if self._profile_kernels_once and (
            self._profile_kernels_target_m <= 0 or self._profile_kernels_target_m == m
        ):
            if self._profile_kernels_skip > 0:
                self._profile_kernels_skip -= 1
                self._rt.execute_icb(self._handle, icb_id)
            else:
                self._profile_kernels_once = False
                self._execute_profiled_once()
        elif self._profile_icb_remaining > 0:
            self._profile_icb_remaining -= 1
            started = time.perf_counter()
            gpu_s = float(self._rt.execute_icb(self._handle, icb_id))
            cpu_s = time.perf_counter() - started
            gap_s = max(cpu_s - gpu_s, 0.0)
            print(
                "[fusekit-icb-profile] "
                f"active_m={m} "
                f"cpu_us={cpu_s * 1_000_000.0:.2f} "
                f"gpu_us={gpu_s * 1_000_000.0:.2f} "
                f"gap_us={gap_s * 1_000_000.0:.2f} "
                f"kernels={len(self._last_dispatches or [])}",
                flush=True,
            )
        else:
            self._rt.execute_icb(self._handle, icb_id)
        prof_exec1 = time.perf_counter() if prof_state else 0.0
        self.last_argmax_tokens = self._read_next_tokens(m)
        self._set_mtp_rollback_state(cache)
        if prof_state:
            prof_t3 = time.perf_counter()
            print(
                "[fusekit-mtp-state-profile] "
                f"active_m={m} "
                f"arrays_rollback_us={(prof_t1 - prof_t0) * 1_000_000.0:.2f} "
                f"advance_us={(prof_t2 - prof_t1) * 1_000_000.0:.2f} "
                f"icb_execute_us={(prof_exec1 - prof_exec0) * 1_000_000.0:.2f} "
                f"set_rollback_us={(prof_t3 - prof_exec1) * 1_000_000.0:.2f}",
                flush=True,
            )

        logits = self._scratch_buf(self._global_scratch_id["logits"])
        if logits.ndim == 3:
            logits_view = logits[:, :m, :]
        elif logits.ndim == 2:
            logits_view = logits[:m, :].reshape(1, m, -1)
        else:
            return None
        if not return_hidden:
            return logits_view

        layers = self._decoder_layers()
        last_layer = len(layers) - 1
        sid = self._layer_scratch_id.get(("out_buf", last_layer))
        if sid is None:
            return None
        hidden = self._scratch_buf(sid)
        hidden_view = self._hidden_result_view(hidden, m)
        if hidden_view is None:
            return None
        return logits_view, hidden_view

    def _try_run_mtp_verify(self, inputs, cache):
        """Run oMLX native-MTP verify as two validated single-token
        FuseKit decode steps.

        The PR-990 path calls the backbone with two tokens and
        `n_confirmed=1` so rejection can roll back the speculative second
        token. Our decode kernels are single-token kernels; executing
        token 0 then token 1 preserves the same autoregressive cache
        state. After token 0 we stash the ArraysCache rollback snapshot
        that oMLX's reject path expects.
        """
        if inputs.shape[1] != 2:
            return None
        first = inputs[:, 0:1]
        second = inputs[:, 1:2]
        out0 = self._try_run_single(first, cache, return_hidden=True)
        if out0 is None:
            return None
        logits0, hidden0 = out0
        mx = self._mx
        # `_try_run_single` returns views into reusable scratch buffers.
        # Preserve token-0 outputs before token-1 overwrites those buffers.
        logits0 = logits0.astype(mx.float32).astype(logits0.dtype)
        hidden0 = hidden0.astype(mx.float32).astype(hidden0.dtype)
        mx.eval(logits0, hidden0)
        self._snapshot_mtp_rollback(cache)
        out1 = self._try_run_single(second, cache, return_hidden=True)
        if out1 is None:
            return None
        logits1, hidden1 = out1
        return (
            mx.concatenate([logits0, logits1], axis=1),
            mx.concatenate([hidden0, hidden1], axis=1),
        )

    def _snapshot_mtp_rollback(self, cache) -> None:
        for c in cache:
            state = getattr(c, "state", None)
            if isinstance(state, list) and len(state) >= 2:
                c.rollback_state = (state[0], state[1])

    def _try_run_single(self, inputs, cache, *, return_hidden: bool = False):
        if inputs.shape[1] != 1:
            return None
        if self._fk_unsupported_graph:
            if not self._fk_unsupported_graph_warned:
                warnings.warn(
                    f"fusekit: {self._fk_unsupported_graph}; "
                    "falling back to native dispatch for this session.",
                    stacklevel=2,
                )
                self._fk_unsupported_graph_warned = True
            return None
        if self._fk_unsupported_cache or self._fk_dense_bf16_disabled:
            return None
        c0 = next((c for c in cache if hasattr(c, "offset")), None)
        if c0 is None:
            return None
        o = self._cache_decode_offset(c0)
        if o is None:
            return None
        ok, reason = _ensure_cache_capacity(cache, o + 1)
        if not ok:
            msg = (
                "fusekit: cannot guarantee KV cache capacity for decode "
                f"offset {o} ({reason}); falling back to native dispatch "
                "for this session."
            )
            if _env_flag("FUSEKIT_STRICT_DISPATCH"):
                raise RuntimeError(msg)
            if not self._fk_unsupported_graph_warned:
                warnings.warn(msg, stacklevel=2)
                self._fk_unsupported_graph_warned = True
            self._fk_unsupported_cache = True
            return None
        icb_id = self._ensure_icb(cache)
        if icb_id is None:
            return None
        mx = self._mx
        flat = inputs.flatten().astype(mx.uint32)
        mx.eval(flat)
        if self._host_embed_lookup:
            row = self._embed_module(inputs).reshape((self._hidden,)).astype(self._act_dt)
            mx.eval(row)
            ctypes.memmove(self._array_ptr(self._x_in), self._array_ptr(row), row.nbytes)
        # token_id_in aliases next_token_out — seed the buffer the
        # GPU chain reads from.
        ctypes.memmove(
            self._uniform_ptrs[("next_token_out", None)],
            self._array_ptr(flat), 4,
        )
        # Init the on-GPU counters the ICB's step_init/history_capture
        # kernels advance: step_counter holds the offset where THIS
        # step's K/V will land; hist_counter is reset each call so
        # try_run doesn't accumulate past history[0].
        ctypes.c_uint32.from_address(self._step_counter_ptr).value = o
        ctypes.c_uint32.from_address(self._hist_counter_ptr).value = 0
        # Seed the shared uniforms on the host as well as via
        # step_init. Some trace-preserved decode ops (notably
        # standalone RoPE) read the offset very early in the ICB; a
        # single shared host write avoids relying on GPU-side uniform
        # propagation for correctness while keeping the per-layer
        # fanout collapsed.
        ctypes.c_uint32.from_address(
            self._uniform_ptrs[("layer_offset", None)]
        ).value = o
        ctypes.c_uint32.from_address(
            self._uniform_ptrs[("layer_kv_len", None)]
        ).value = o + 1
        # QuantizedKVCache.keys is (packed, scales, biases) tuple; the
        # `capacity` dim is shape[2] of the packed array (same as the
        # bf16 case's c0.keys.shape[2]).
        keys_repr = c0.keys[0] if isinstance(c0.keys, (tuple, list)) else c0.keys
        if keys_repr is None:
            return None
        ctypes.c_uint32.from_address(
            self._uniform_ptrs[("layer_capacity", None)]
        ).value = keys_repr.shape[2]
        # Reset per-layer barrier counters to 0 so the next ICB execute
        # starts fresh. GlobalBarrier piece atomic_fetch_adds and spins
        # until value == tg_count.
        for ptr in self._barrier_counter_ptrs.values():
            ctypes.c_uint32.from_address(ptr).value = 0
        self._advance_decode_cache(cache, o + 1)
        if self._profile_kernels_once and (
            self._profile_kernels_target_m <= 0 or self._profile_kernels_target_m == 1
        ):
            if self._profile_kernels_skip > 0:
                self._profile_kernels_skip -= 1
                self._rt.execute_icb(self._handle, icb_id)
            else:
                self._profile_kernels_once = False
                self._execute_profiled_once()
        else:
            if self._profile_icb_remaining > 0:
                self._profile_icb_remaining -= 1
                started = time.perf_counter()
                gpu_s = float(self._rt.execute_icb(self._handle, icb_id))
                cpu_s = time.perf_counter() - started
                gap_s = max(cpu_s - gpu_s, 0.0)
                print(
                    "[fusekit-icb-profile] "
                    f"cpu_us={cpu_s * 1_000_000.0:.2f} "
                    f"gpu_us={gpu_s * 1_000_000.0:.2f} "
                    f"gap_us={gap_s * 1_000_000.0:.2f} "
                    f"kernels={len(self._last_dispatches or [])}",
                    flush=True,
                )
            else:
                self._rt.execute_icb(self._handle, icb_id)
        self.last_argmax_tokens = self._read_next_tokens(1)
        # Trace was prefill L=N; sliced row 0 is what mlx-lm samples.
        logits = self._scratch_buf(self._global_scratch_id["logits"])
        if logits.ndim == 3:
            logits_view = logits[:, :1, :]
        else:
            logits_view = logits[:1, :].reshape(1, 1, -1)
        if not return_hidden:
            return logits_view

        layers = self._decoder_layers()
        last_layer = len(layers) - 1
        sid = self._layer_scratch_id.get(("out_buf", last_layer))
        if sid is None:
            return None
        hidden = self._scratch_buf(sid)
        hidden_view = self._hidden_result_view(hidden, 1)
        if hidden_view is None:
            return None
        return logits_view, hidden_view

    def _execute_profiled_once(self) -> None:
        """Execute the current decode pipeline one kernel at a time and
        print GPU time by task. This is a ranking profiler: direct
        per-task command buffers are not identical to ICB execution,
        but they show which generated kernels dominate GPU work."""
        dispatches = self._last_dispatches
        labels = self._last_dispatch_labels or []
        if not dispatches:
            return
        rows = []
        total_gpu_s = 0.0
        for i, d in enumerate(dispatches):
            gpu_s = float(self._rt.dispatch_batch(self._handle, [d]))
            total_gpu_s += gpu_s
            kernel, bufs, grid, tg = d
            label = labels[i] if i < len(labels) else {}
            rows.append({
                "idx": i,
                "gpu_us": gpu_s * 1_000_000.0,
                "kernel": kernel,
                "summary": self._kernel_summaries.get(kernel, ""),
                "kind": label.get("kind", self._kernel_kind(kernel)),
                "task_id": label.get("task_id"),
                "fusion_id": label.get("fusion_id"),
                "layer_idx": label.get("layer_idx"),
                "buffers": len(bufs),
                "bytes": sum(n for _, n in bufs),
                "grid": grid,
                "tg": tg,
            })

        by_kind: dict[str, dict[str, float]] = {}
        for r in rows:
            k = str(r["kind"])
            acc = by_kind.setdefault(k, {"count": 0.0, "gpu_us": 0.0, "bytes": 0.0})
            acc["count"] += 1.0
            acc["gpu_us"] += float(r["gpu_us"])
            acc["bytes"] += float(r["bytes"])
        by_family: dict[str, dict[str, float]] = {}
        for r in rows:
            family = self._kernel_family(str(r["kernel"]))
            acc = by_family.setdefault(family, {"count": 0.0, "gpu_us": 0.0, "bytes": 0.0})
            acc["count"] += 1.0
            acc["gpu_us"] += float(r["gpu_us"])
            acc["bytes"] += float(r["bytes"])
        by_summary: dict[str, dict[str, float]] = {}
        for r in rows:
            summary = str(r.get("summary") or "")
            if not summary:
                continue
            if summary.startswith("Fusion #") and ": " in summary:
                summary = summary.split(": ", 1)[1]
            acc = by_summary.setdefault(summary, {"count": 0.0, "gpu_us": 0.0, "bytes": 0.0})
            acc["count"] += 1.0
            acc["gpu_us"] += float(r["gpu_us"])
            acc["bytes"] += float(r["bytes"])
        if os.environ.get("FUSEKIT_PROFILE_COMPACT"):
            wanted_kinds = (
                "qmv_fanout",
                "linear_residual",
                "other",
                "op:quantizedmatmul",
                "gated_delta",
                "sdpa",
                "chain",
            )
            compact = {
                **getattr(self, "_profile_context", {}),
                "kernels": len(rows),
                "total_gpu_us": total_gpu_s * 1_000_000.0,
                "kind_gpu_us": {
                    k: by_kind.get(k, {}).get("gpu_us", 0.0)
                    for k in wanted_kinds
                },
                "summary_top": [
                    {
                        "summary": summary,
                        "count": int(acc["count"]),
                        "gpu_us": acc["gpu_us"],
                    }
                    for summary, acc in sorted(
                        by_summary.items(),
                        key=lambda kv: kv[1]["gpu_us"],
                        reverse=True,
                    )[:10]
                ],
                "family_top": [
                    {
                        "family": family,
                        "count": int(acc["count"]),
                        "gpu_us": acc["gpu_us"],
                    }
                    for family, acc in sorted(
                        by_family.items(),
                        key=lambda kv: kv[1]["gpu_us"],
                        reverse=True,
                    )[:8]
                ],
            }
            print("[fusekit-profile-json] " + json.dumps(compact), flush=True)
            self._profile_context = {}
            return

        print("[fusekit-profile] per-kernel decode step", flush=True)
        print(f"[fusekit-profile] kernels={len(rows)} total_gpu_us={total_gpu_s * 1_000_000.0:.1f}", flush=True)
        print("[fusekit-profile] aggregate by kind:", flush=True)
        for kind, acc in sorted(by_kind.items(), key=lambda kv: kv[1]["gpu_us"], reverse=True)[:24]:
            mib = acc["bytes"] / (1024.0 * 1024.0)
            print(
                "[fusekit-profile] "
                f"{acc['gpu_us']:9.2f} us  count={int(acc['count']):>4} "
                f"bytes={mib:8.2f} MiB kind={kind}",
                flush=True,
            )
        print("[fusekit-profile] aggregate by kernel family:", flush=True)
        for family, acc in sorted(by_family.items(), key=lambda kv: kv[1]["gpu_us"], reverse=True)[:32]:
            mib = acc["bytes"] / (1024.0 * 1024.0)
            print(
                "[fusekit-profile] "
                f"{acc['gpu_us']:9.2f} us  count={int(acc['count']):>4} "
                f"bytes={mib:8.2f} MiB family={family}",
                flush=True,
            )
        if by_summary:
            print("[fusekit-profile] aggregate by kernel summary:", flush=True)
            for summary, acc in sorted(by_summary.items(), key=lambda kv: kv[1]["gpu_us"], reverse=True)[:32]:
                mib = acc["bytes"] / (1024.0 * 1024.0)
                print(
                    "[fusekit-profile] "
                    f"{acc['gpu_us']:9.2f} us  count={int(acc['count']):>4} "
                    f"bytes={mib:8.2f} MiB summary={summary}",
                    flush=True,
                )
        print("[fusekit-profile] top by gpu_us:", flush=True)
        top_n_raw = os.environ.get("FUSEKIT_PROFILE_TOP", "40")
        try:
            top_n = int(top_n_raw)
        except ValueError:
            top_n = 40
        if top_n <= 0:
            top_n = len(rows)
        for r in sorted(rows, key=lambda x: x["gpu_us"], reverse=True)[:top_n]:
            task = "-" if r["task_id"] is None else str(r["task_id"])
            fusion = "-" if r["fusion_id"] is None else str(r["fusion_id"])
            layer = "-" if r["layer_idx"] is None else str(r["layer_idx"])
            mib = r["bytes"] / (1024.0 * 1024.0)
            print(
                "[fusekit-profile] "
                f"{r['gpu_us']:9.2f} us  kind={r['kind']:<7} "
                f"task={task:>4} fusion={fusion:>4} layer={layer:>3} "
                f"bufs={r['buffers']:>2} bytes={mib:7.2f} MiB "
                f"kernel={r['kernel']}"
                + (f"  // {r['summary']}" if r["summary"] else ""),
                flush=True,
            )

    def _kernel_family(self, kernel: str) -> str:
        if (
            "fusekit_op_quantizedmatmul_" in kernel
            and "_argmax" in kernel
        ):
            return "qmv_argmax"
        if kernel.startswith("fusekit_op_"):
            rest = kernel[len("fusekit_op_"):]
            if "_f" in rest:
                rest = rest.rsplit("_f", 1)[0]
            return "op:" + rest
        if kernel.startswith("fusekit_qmv_fanout_"):
            return "qmv_fanout"
        if kernel.startswith("fusekit_chain_"):
            return "chain"
        if "gated_delta_step" in kernel:
            return "gated_delta"
        if kernel.startswith("fusekit_arrays_cache_shift_append_decode_"):
            return "arrays_cache"
        if kernel.startswith("fusekit_sdpa_decode_"):
            return "sdpa_decode"
        if kernel.startswith("fusekit_rope_"):
            return "rope"
        if kernel.startswith("fusekit_kv_cache_store_"):
            return "cache_store"
        if kernel.startswith("fusekit_argmax_"):
            return "argmax"
        if kernel.startswith("fusekit_step_") or kernel.startswith("fusekit_history_"):
            return "runtime"
        return kernel
