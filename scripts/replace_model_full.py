"""Model-wide ICB hook driven by `task_graph.json`.

Loads the compiled metallib + v2 task graph for a model, builds a
single indirect command buffer covering every fused task, and patches
the model's top-level `__call__` so each generate-step submits the
megakernel in one go (no per-layer dispatch).

This is the runtime-side counterpart of the compiler's v2 schema —
all binding resolution lives here instead of in hand-written layer
templates. The native module (`_fusekit_runtime_native`) provides the
Metal ICB primitives; this script just translates the JSON into
`(kernel, buffers, grid, threadgroup)` tuples and stands the hook up.
"""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

# Suppress the layer-level fusekit hook installed by `fusekit patch`'s
# .pth file. We're driving the dispatcher ourselves at the model level
# and don't want the layer wrapper rewriting `model.__call__` on top
# of our patch.
os.environ.setdefault("FUSEKIT_DISABLE_HOOK", "1")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="model dir under ~/.omlx/models")
    p.add_argument("--prompt", default="Write a short greeting.")
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    model_dir = Path(os.path.expanduser("~/.omlx/models")) / args.model
    fusekit_dir = model_dir / "fusekit"
    metallib = fusekit_dir / "kernel.metallib"
    tg_path = fusekit_dir / "task_graph.json"
    if not metallib.is_file() or not tg_path.is_file():
        sys.stderr.write(f"missing compile artifacts under {fusekit_dir}\n")
        return 2

    sys.path.insert(0, str(fusekit_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt
    import mlx.core as mx
    import mlx_lm

    with open(tg_path) as f:
        tg = json.load(f)
    if tg.get("version") != 2:
        sys.stderr.write(f"task_graph.json v{tg.get('version')} not supported\n")
        return 2

    model, tokenizer = mlx_lm.load(str(model_dir))
    handle = rt.load_dispatcher(str(metallib), str(tg_path))

    # ── allocate scratch buffers as mlx arrays ──────────────────
    # Backing each scratch with mx.zeros keeps lifetime simple — the
    # array stays alive as long as the dispatches list holds the
    # pointer, and mlx tracks the buffer through the unified
    # allocator without us round-tripping through `alloc_uniform`.
    scratch_arrays: list = []
    for s in tg["scratch_buffers"]:
        shape = tuple(s["shape"]) if s["shape"] else (1,)
        dt = getattr(mx, s["dtype"])
        arr = mx.zeros(shape, dtype=dt)
        mx.eval(arr)
        scratch_arrays.append(arr)

    # ── materialise captured scalar constants ───────────────────
    scalar_arrays: dict = {}
    for t in tg["tasks"]:
        for b in t["inputs"]:
            if b["role"] != "external":
                continue
            sv = b.get("scalar")
            if not sv:
                continue
            key = b["tensor_id"]
            if key in scalar_arrays:
                continue
            dt = getattr(mx, b["dtype"])
            arr = mx.array(sv["value"], dtype=dt)
            mx.eval(arr)
            scalar_arrays[key] = arr

    # ── runtime-supplied bindings get host-visible uniforms ─────
    runtime_uniforms: dict = {}
    _dtype_bytes = {
        "bool": 1, "uint8": 1, "int8": 1,
        "uint16": 2, "int16": 2, "float16": 2, "bfloat16": 2,
        "uint32": 4, "int32": 4, "float32": 4,
        "uint64": 8, "int64": 8, "float64": 8, "complex64": 8,
    }
    for t in tg["tasks"]:
        for b in t["inputs"]:
            if b["role"] != "external":
                continue
            if b.get("source_path") or b.get("scalar"):
                continue
            tid = b["tensor_id"]
            if tid in runtime_uniforms:
                continue
            elems = 1
            for d in b["shape"]:
                elems *= max(1, d)
            nbytes = elems * _dtype_bytes.get(b["dtype"], 4)
            ptr = rt.alloc_uniform(handle, nbytes)
            runtime_uniforms[tid] = (ptr, nbytes)

    # ── walk model.<path> to live mlx tensors ───────────────────
    param_cache: dict = {}

    def resolve_path(path: str):
        if path in param_cache:
            return param_cache[path]
        cur = model
        for seg in path.split("."):
            cur = cur[int(seg)] if seg.isdigit() else getattr(cur, seg)
        param_cache[path] = cur
        return cur

    def buf_for(b):
        if b["role"] == "scratch":
            arr = scratch_arrays[b["scratch_id"]]
            return (fr.array_data_ptr(arr), arr.nbytes)
        path = b.get("source_path")
        if path:
            arr = resolve_path(path)
            return (fr.array_data_ptr(arr), arr.nbytes)
        if b.get("scalar"):
            arr = scalar_arrays[b["tensor_id"]]
            return (fr.array_data_ptr(arr), max(1, arr.nbytes))
        return runtime_uniforms[b["tensor_id"]]

    # ── build the dispatches list + ICB ─────────────────────────
    dispatches = []
    for t in tg["tasks"]:
        bufs = [buf_for(b) for b in t["inputs"]] + [buf_for(b) for b in t["outputs"]]
        gx, gy, gz = t["grid_threads"]
        tx, ty, tz = t["threadgroup"]
        dispatches.append((t["kernel"], bufs, (gx, gy, gz), (tx, ty, tz)))

    icb_id = rt.build_icb(handle, dispatches)
    print(f"[fusekit] {len(dispatches)} tasks → ICB ready")

    # ── locate the runtime token uniform + logits output ───────
    # The token-id input is the only runtime-supplied external; its
    # tensor_id is the lowest one in the runtime_uniforms map (Stage 1
    # assigns id 0 to the function input).
    token_tid = min(runtime_uniforms) if runtime_uniforms else None

    # The "final" decode product depends on whether `argmax_logits` is
    # in the graph (then sampling happens on-GPU and the last scratch
    # is a single int32) or not (then the last scratch is the vocab-
    # wide logits). mlx_lm.generate expects logits, so we surface the
    # widest float-typed scratch output of the last few tasks.
    output_arr = None
    for t in reversed(tg["tasks"]):
        for b in t["outputs"]:
            if b["role"] != "scratch":
                continue
            if b["dtype"] not in ("float16", "bfloat16", "float32"):
                continue
            shape = b["shape"]
            if shape and shape[-1] > 1024:
                output_arr = scratch_arrays[b["scratch_id"]]
                break
        if output_arr is not None:
            break

    # ── monkey-patch model.__call__ ─────────────────────────────
    model_cls = type(model)
    original_call = model_cls.__call__

    def patched_call(self, inputs, cache=None, **kwargs):
        if token_tid is not None:
            flat = inputs.flatten().astype(mx.int32)
            mx.eval(flat)
            src = fr.array_data_ptr(flat)
            dst, nbytes = runtime_uniforms[token_tid]
            ctypes.memmove(dst, src, min(flat.nbytes, nbytes))
        rt.execute_icb(handle, icb_id)
        if output_arr is None:
            return original_call(self, inputs, cache=cache, **kwargs)
        return output_arr

    model_cls.__call__ = patched_call
    print("[fusekit] hook installed; generating…")

    text = mlx_lm.generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
