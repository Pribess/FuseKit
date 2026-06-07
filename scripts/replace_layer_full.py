"""Standalone harness for `fusekit_runtime.dispatch.Dispatcher`.

Same dispatcher `fusekit patch` auto-installs into mlx-lm; this
script exists so you can drive it without the auto-hook (useful for
debugging via `FUSEKIT_DUMP_DISPATCHES` and the like). mlx-lm's
`generate()` runs the outer loop; the dispatcher takes over on every
L=1 call once the KV cache is populated.
"""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path


# Suppress the layer-level fusekit hook installed by `fusekit patch`'s
# .pth file — we drive the dispatcher ourselves below.
os.environ.setdefault("FUSEKIT_DISABLE_HOOK", "1")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--prompt", default="Write a short greeting.")
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    model_dir = Path(os.path.expanduser("~/.omlx/models")) / args.model
    fusekit_dir = model_dir / "fusekit"
    if not (fusekit_dir / "kernel.metallib").is_file():
        sys.stderr.write(
            f"missing {fusekit_dir}/kernel.metallib — run `fusekit patch` first\n"
        )
        return 2

    sys.path.insert(0, str(fusekit_dir / "runtime"))
    import fusekit_runtime as fr
    import fusekit_runtime._fusekit_runtime_native as rt
    import mlx.core as mx
    import mlx_lm

    with open(fusekit_dir / "task_graph.json") as f:
        tg = json.load(f)
    if tg.get("version") != 2:
        sys.stderr.write(f"task_graph.json v{tg.get('version')} not supported\n")
        return 2

    model, tokenizer = mlx_lm.load(str(model_dir))
    handle = rt.load_dispatcher(
        str(fusekit_dir / "kernel.metallib"),
        str(fusekit_dir / "task_graph.json"),
    )
    print(
        f"[fusekit-v2] dispatcher up; "
        f"{len(tg['tasks'])} tasks, "
        f"{len(tg['decode_pipeline'])} in decode pipeline",
        file=sys.stderr,
    )

    # mlx-core spells `bool` as `bool_`; everything else is the same.
    _DTYPE = {
        "bool": "bool_", "uint8": "uint8", "int8": "int8",
        "uint16": "uint16", "int16": "int16",
        "uint32": "uint32", "int32": "int32",
        "uint64": "uint64", "int64": "int64",
        "float16": "float16", "bfloat16": "bfloat16",
        "float32": "float32", "float64": "float64",
        "complex64": "complex64",
    }

    def mx_dtype(name: str):
        return getattr(mx, _DTYPE.get(name, name))

    # ── pre-allocate scratch buffers ────────────────────────────
    # Every cross-task tensor mlx's trace emitted becomes one
    # mx.array we own for the dispatcher's lifetime. Reused across
    # generate steps; the kernels read/write them in place.
    scratch_arrays: list = []
    for s in tg["scratch_buffers"]:
        shape = tuple(s["shape"]) if s["shape"] else (1,)
        arr = mx.zeros(shape, dtype=mx_dtype(s["dtype"]))
        mx.eval(arr)
        scratch_arrays.append(arr)

    # ── captured-scalar one-element buffers ─────────────────────
    scalar_arrays: dict = {}
    for t in tg["tasks"]:
        for b in t["inputs"]:
            if b["role"] != "external":
                continue
            sv = b.get("scalar")
            if not sv:
                continue
            tid = b["tensor_id"]
            if tid in scalar_arrays:
                continue
            arr = mx.array(sv["value"], dtype=mx_dtype(b["dtype"]))
            mx.eval(arr)
            scalar_arrays[tid] = arr

    # ── derive per-layer + global named scratch from the trace ──
    # Each pattern emit writes its result into one or more IR-bound
    # output scratch ids. The synthetic kernels (rope_q_decode etc.)
    # expect those same buffers under semantic names (q_buf, …). We
    # walk decode_pipeline once and record the mapping.
    layer_scratch_id: dict = {}     # (name, layer_idx) -> scratch_id
    global_scratch_id: dict = {}    # name -> scratch_id

    seen_linear_residual_per_layer: dict = {}
    for tid in tg["decode_pipeline"]:
        t = tg["tasks"][tid]
        k = t["kernel"]
        li = t.get("layer_idx")
        outs = t.get("outputs", [])
        # We only care about Scratch outputs (model weights / runtime
        # uniforms never sit in this map).
        scratch_outs = [b["scratch_id"] for b in outs if b["role"] == "scratch"]

        if "fusekit_rmsnorm_qkv_" in k and li is not None and len(scratch_outs) == 3:
            layer_scratch_id[("q_buf", li)] = scratch_outs[0]
            layer_scratch_id[("k_buf", li)] = scratch_outs[1]
            layer_scratch_id[("v_buf", li)] = scratch_outs[2]
        elif "fusekit_sdpa_decode_" in k and li is not None and len(scratch_outs) == 1:
            layer_scratch_id[("sdpa_buf", li)] = scratch_outs[0]
        elif "fusekit_linear_residual_" in k and li is not None and len(scratch_outs) == 1:
            seen = seen_linear_residual_per_layer.setdefault(li, 0)
            # First LinearResidual per layer = o_proj (writes h_attn).
            # Second = down_proj (writes out_buf).
            name = "h_attn" if seen == 0 else "out_buf"
            layer_scratch_id[(name, li)] = scratch_outs[0]
            seen_linear_residual_per_layer[li] = seen + 1
        elif "fusekit_rmsnorm_gateup_swiglu_" in k and li is not None and len(scratch_outs) == 1:
            layer_scratch_id[("silu_up", li)] = scratch_outs[0]
        elif "fusekit_final_rmsnorm_lm_head_" in k and len(scratch_outs) == 1:
            global_scratch_id["logits"] = scratch_outs[0]
        elif "fusekit_embed_lookup_decode_" in k:
            # x_in is *bound* to the embed kernel's output slot via
            # extra_bindings; here we just record the slot index so
            # the next layer's rmsnorm_qkv reads from a known
            # scratch. We materialise this lazily below.
            pass

    # x_in is the embed kernel's GlobalScratch(XIn) output and the
    # first layer's rmsnorm_qkv x input. We don't have a trace-
    # output scratch for it — embed_lookup_decode is synthetic, so
    # we allocate a fresh scratch and point both sides at it.
    cfg = json.load(open(model_dir / "config.json"))
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]

    # Activation dtype: pick from the model's compute precision —
    # 4-bit quant runs at fp16, others bf16.
    quant = cfg.get("quantization") is not None
    act_dt = mx.float16 if quant else mx.bfloat16

    x_in_array = mx.zeros((hidden,), dtype=act_dt)
    mx.eval(x_in_array)

    # Quantized models keep `embed_tokens.weight` as packed uint32; the
    # synthetic `embed_lookup_decode` kernel reads it as half (it
    # doesn't dequantize on the fly). Pre-dequantize once at install
    # time so we can hand the kernel a real (vocab, hidden) half buffer.
    embed_module = model.model.embed_tokens
    if hasattr(embed_module, "scales") and hasattr(embed_module, "biases"):
        embed_dequant = mx.dequantize(
            embed_module.weight,
            embed_module.scales,
            embed_module.biases,
            group_size=64,
            bits=4,
        ).astype(act_dt)
    else:
        embed_dequant = embed_module.weight
    mx.eval(embed_dequant)

    # The first layer's rmsnorm_qkv input is supposed to read x_in.
    # In the captured graph that came from the trace's embed op
    # output. We override the IR-bound input for layer-0 rmsnorm_qkv
    # to read from `x_in_array` instead — gives the synthetic
    # embed_lookup_decode and the rest of the pipeline a shared
    # buffer without our needing to rewrite scratch ids in the JSON.
    first_layer_x_scratch = None
    for tid in tg["decode_pipeline"]:
        t = tg["tasks"][tid]
        if "fusekit_rmsnorm_qkv_" in t["kernel"] and t.get("layer_idx") == 0:
            ins = t.get("inputs", [])
            if ins and ins[0]["role"] == "scratch":
                first_layer_x_scratch = ins[0]["scratch_id"]
            break

    # ── runtime uniforms (4-byte buffers, host-writeable) ───────
    # token_id_in / next_token_out are global; layer_offset / layer_kv_len
    # / layer_capacity are per-layer.
    uniform_ptrs: dict = {}

    def get_uniform(role: str, layer_idx) -> int:
        key = (role, layer_idx)
        if key not in uniform_ptrs:
            uniform_ptrs[key] = rt.alloc_uniform(handle, 4)
        return uniform_ptrs[key]

    # ── extra-binding resolver ──────────────────────────────────
    def resolve_extra(role: str, layer_idx, cache):
        if role == "token_id_in":
            return (get_uniform(role, None), 4)
        if role == "next_token_out":
            return (get_uniform(role, None), 4)
        if role == "layer_offset":
            return (get_uniform(role, layer_idx), 4)
        if role == "layer_capacity":
            return (get_uniform(role, layer_idx), 4)
        if role == "layer_kv_len":
            return (get_uniform(role, layer_idx), 4)
        if role == "layer_cache_k":
            c = cache[layer_idx]
            return (fr.array_data_ptr(c.keys), c.keys.nbytes)
        if role == "layer_cache_v":
            c = cache[layer_idx]
            return (fr.array_data_ptr(c.values), c.values.nbytes)
        if role.startswith("layer_scratch."):
            name = role.split(".", 1)[1]
            sid = layer_scratch_id.get((name, layer_idx))
            if sid is None:
                raise RuntimeError(
                    f"no scratch recorded for ({name}, layer {layer_idx})"
                )
            arr = scratch_arrays[sid]
            return (fr.array_data_ptr(arr), arr.nbytes)
        if role == "global_scratch.x_in":
            return (fr.array_data_ptr(x_in_array), x_in_array.nbytes)
        if role == "global_scratch.logits":
            sid = global_scratch_id.get("logits")
            if sid is None:
                raise RuntimeError("no logits scratch recorded")
            arr = scratch_arrays[sid]
            return (fr.array_data_ptr(arr), arr.nbytes)
        if role == "param.embed_tokens":
            # The synthetic embed_lookup_decode kernel reads the
            # embed table as half — feed it the pre-dequantized
            # buffer, not the packed uint32 weight.
            return (fr.array_data_ptr(embed_dequant), embed_dequant.nbytes)
        if role == "param.final_norm":
            arr = model.model.norm.weight
            return (fr.array_data_ptr(arr), arr.nbytes)
        if role == "param.lm_head":
            arr = (
                model.lm_head.weight
                if hasattr(model, "lm_head")
                else model.model.embed_tokens.weight
            )
            return (fr.array_data_ptr(arr), arr.nbytes)
        raise ValueError(f"unknown extra binding role: {role}")

    # ── attribute-path resolution cache ─────────────────────────
    path_cache: dict = {}

    def resolve_path(path: str):
        if path in path_cache:
            return path_cache[path]
        cur = model
        for seg in path.split("."):
            cur = cur[int(seg)] if seg.isdigit() else getattr(cur, seg)
        path_cache[path] = cur
        return cur

    def _buf_for_binding(b, layer_idx_for_input: int):
        """Resolve one IR-bound binding. `layer_idx_for_input` is used
        for the layer-0 rmsnorm_qkv x-input alias and for resolving
        per-layer Q/K/V additive biases that the trace materialises
        via Broadcast → scratch but the hand-tuned kernel really
        wants pointed straight at the param."""
        if b["role"] == "scratch":
            sid = b["scratch_id"]
            # Layer-0 rmsnorm_qkv's first input — point at x_in
            # instead of the trace's scratch so the embed kernel
            # output flows through.
            if first_layer_x_scratch is not None and sid == first_layer_x_scratch \
                    and layer_idx_for_input == 0:
                return (fr.array_data_ptr(x_in_array), x_in_array.nbytes)
            # Q/K/V bias broadcasts come from `nn.Linear.bias` in mlx-lm.
            # The trace materialises them via Broadcast into scratch;
            # the rmsnorm_qkv kernel just dereferences by output index
            # so a (896,)/(128,) param pointer reads identically as
            # long as the first N elements match. Route bias slots
            # back to the underlying parameter.
            rn = b.get("role_name", "")
            if rn in ("bias_q", "bias_k", "bias_v"):
                kind = {"bias_q": "q_proj", "bias_k": "k_proj", "bias_v": "v_proj"}[rn]
                arr = resolve_path(
                    f"model.layers.{layer_idx_for_input}.self_attn.{kind}.bias"
                )
                return (fr.array_data_ptr(arr), arr.nbytes)
            arr = scratch_arrays[sid]
            return (fr.array_data_ptr(arr), arr.nbytes)
        path = b.get("source_path")
        if path:
            arr = resolve_path(path)
            return (fr.array_data_ptr(arr), arr.nbytes)
        if b.get("scalar"):
            arr = scalar_arrays[b["tensor_id"]]
            return (fr.array_data_ptr(arr), max(1, arr.nbytes))
        # Bare runtime input not covered by extra_bindings — should
        # never hit on the decode pipeline (extra_bindings cover
        # token_id et al).
        raise RuntimeError(
            f"task has unbound binding: {b}"
        )

    # ── build dispatches (decode_pipeline order) ────────────────
    def build_dispatches(cache):
        # Count which LinearResidual we're on per layer — the first
        # is o_proj (needs its `x` rerouted to layer.sdpa_buf since
        # the trace's SDPA→o_proj path goes through transpose+
        # reshape view scratches we don't dispatch). The second is
        # down_proj whose `x` chains correctly via gate-up's output.
        linres_idx_per_layer: dict = {}
        out = []
        for tid in tg["decode_pipeline"]:
            t = tg["tasks"][tid]
            li = t.get("layer_idx")
            x_li = li if li is not None else 0  # only matters for rmsnorm_qkv layer 0
            # Lay IR inputs at slots 0..n_in-1.
            bufs = [_buf_for_binding(b, x_li) for b in t["inputs"]]
            # Apply extra_bindings (uniforms / cache pointers / named
            # scratch). These may sit at slot indices BEYOND
            # `n_in` — pad with placeholders, the IR-output placement
            # below uses the final slot count to position itself.
            for eb in t.get("extra_bindings", []):
                slot = eb["slot"]
                while len(bufs) <= slot:
                    bufs.append((0, 0))
                bufs[slot] = resolve_extra(eb["role"], li, cache)
            # IR outputs follow every IR-input + extra slot. Concretely
            # for `SdpaDecode` the MSL signature is
            #   [Q, K, V, kv_len, capacity, out]
            # — output sits at slot 5, *after* the uniform extras, not
            # slot 3 (which is `kv_len`).
            out_base = len(bufs)
            for i, b in enumerate(t["outputs"]):
                bufs.append(_buf_for_binding(b, x_li))
            # Per-layer o_proj override.
            if "fusekit_linear_residual_" in t["kernel"] and li is not None:
                seen = linres_idx_per_layer.setdefault(li, 0)
                if seen == 0:
                    # o_proj — `x` slot 0 → layer.sdpa_buf.
                    bufs[0] = resolve_extra("layer_scratch.sdpa_buf", li, cache)
                linres_idx_per_layer[li] = seen + 1
            if any(b == (0, 0) for b in bufs):
                raise RuntimeError(
                    f"task {tid} ({t['kernel']}) unbound slot; bufs={bufs} "
                    f"extra={t.get('extra_bindings')}"
                )
            gx, gy, gz = t["grid_threads"]
            tx, ty, tz = t["threadgroup"]
            out.append((t["kernel"], bufs, (gx, gy, gz), (tx, ty, tz)))
        return out

    state = {"icb_id": None, "cache_sig": None}

    def _dump_dispatches(d):
        path = os.environ.get("FUSEKIT_DUMP_DISPATCHES")
        if not path:
            return
        serial = []
        for k, bufs, g, tg in d:
            serial.append({"k": k, "bufs": [[int(p), int(n)] for p, n in bufs],
                           "g": list(g), "tg": list(tg)})
        with open(path, "w") as f:
            json.dump(serial, f, indent=2)
        print(f"[dump] wrote {path}: {len(serial)} dispatches", file=sys.stderr)

    def ensure_icb(cache):
        sig = tuple(
            fr.array_data_ptr(c.keys) if c.keys is not None else 0
            for c in cache
        )
        if any(s == 0 for s in sig):
            return None
        if state["cache_sig"] == sig and state["icb_id"] is not None:
            return state["icb_id"]
        d = build_dispatches(cache)
        _dump_dispatches(d)
        state["icb_id"] = rt.build_icb(handle, d)
        state["cache_sig"] = sig
        return state["icb_id"]

    # ── monkey-patch model.__call__ ─────────────────────────────
    orig_call = type(model).__call__

    def patched(self, inputs, cache=None, **kw):
        if cache is None or len(inputs.shape) != 2 or inputs.shape[1] != 1:
            # Prefill (L>1) or first call (cache empty) — mlx
            # fallback handles cache allocation too.
            return orig_call(self, inputs, cache=cache, **kw)
        icb_id = ensure_icb(cache)
        if icb_id is None:
            return orig_call(self, inputs, cache=cache, **kw)
        # Per-step uniform writes.
        flat = inputs.flatten().astype(mx.uint32)
        mx.eval(flat)
        ctypes.memmove(uniform_ptrs[("token_id_in", None)], fr.array_data_ptr(flat), 4)
        for li, c in enumerate(cache):
            o = int(c.offset)
            ctypes.c_uint32.from_address(uniform_ptrs[("layer_offset", li)]).value = o
            ctypes.c_uint32.from_address(uniform_ptrs[("layer_capacity", li)]).value = c.keys.shape[2]
            ctypes.c_uint32.from_address(uniform_ptrs[("layer_kv_len", li)]).value = o + 1
            c.offset = o + 1
        rt.execute_icb(handle, icb_id)
        # Logits scratch is sized for the trace's L=8 prefill shape;
        # decode only fills row 0. Slice that row out for mlx-lm.
        logits_arr = scratch_arrays[global_scratch_id["logits"]]
        if os.environ.get("FUSEKIT_DEBUG_SNIFF"):
            def stat(arr, label, n=6):
                mx.eval(arr)
                flat_v = arr.flatten()[:n]
                full = arr.flatten()
                has_nan = bool(mx.any(mx.isnan(full)))
                has_inf = bool(mx.any(mx.isinf(full)))
                mx_v = float(mx.max(full))
                mn_v = float(mx.min(full))
                vals = [float(x) for x in flat_v]
                tag = "NAN" if has_nan else ("INF" if has_inf else "OK ")
                print(f"  {tag} {label:>20s}  min={mn_v:+8.3f} max={mx_v:+8.3f} "
                      f"first6={['{:+.3f}'.format(v) for v in vals]}",
                      file=sys.stderr)
            o0 = ctypes.c_uint32.from_address(uniform_ptrs[("layer_offset", 0)]).value
            kv0 = ctypes.c_uint32.from_address(uniform_ptrs[("layer_kv_len", 0)]).value
            cap0 = ctypes.c_uint32.from_address(uniform_ptrs[("layer_capacity", 0)]).value
            print(f"[sniff] L0 offset={o0} kv_len={kv0} cap={cap0}", file=sys.stderr)
            stat(x_in_array, "x_in")
            for la in (0, 1, 23):
                for nm in ("out_buf",):
                    sid = layer_scratch_id.get((nm, la))
                    if sid is not None:
                        stat(scratch_arrays[sid], f"L{la}.{nm}")
            stat(logits_arr, "logits")
            row = logits_arr[0, 0, :]
            mx.eval(row)
            top5 = mx.argsort(row)[-5:]
            print(f"  top5: {[(int(i), float(row[i])) for i in top5[::-1]]}",
                  file=sys.stderr)
        return logits_arr[:, :1, :]

    type(model).__call__ = patched
    print("[fusekit-v2] hook installed", file=sys.stderr)

    out = mlx_lm.generate(
        model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
