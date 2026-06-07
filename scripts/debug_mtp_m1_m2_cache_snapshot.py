"""Compare MTP M=1 row state against the M=2 rollback snapshot.

This targets the stateful Qwen3.5/Next MTP path.  For a fixed prompt prefix
and verify pair it runs:

* a single-token FuseKit decode for row 0, capturing the layer qkv scratch and
  live ArraysCache state after that token;
* an M=2 FuseKit verify decode, capturing row-0 qkv scratch plus the
  rollback_state produced for rejection.

If M=2 is batch-invariant, row-0 qkv and the rollback conv/SSM snapshots should
match the M=1 post-row0 state.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.model_settings import ModelSettings
from omlx.patches.mlx_lm_mtp import batch_generator as mtp_bg
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


@contextmanager
def _env(**values: str | None):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def _advance_cache(model, cache, tokens: list[int]) -> None:
    for tok in tokens:
        out = model(mx.array([[tok]], dtype=mx.uint32), cache=cache, return_hidden=True)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)


def _as_array(x):
    ptr = getattr(x, "_fk_ptr", None)
    if ptr is None:
        return x
    shape = tuple(int(v) for v in getattr(x, "shape", ()))
    n = int(np.prod(shape)) if shape else 1
    dtype = str(getattr(x, "dtype", ""))
    if "bfloat16" in dtype:
        raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(int(ptr))).copy()
        arr = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return mx.array(arr, dtype=mx.float32)
    if "float16" in dtype:
        raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(int(ptr))).copy()
        return mx.array(raw.view(np.float16).reshape(shape), dtype=mx.float16)
    if "float32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.float32)
    if "uint32" in dtype:
        arr = np.ctypeslib.as_array((ctypes.c_uint32 * n).from_address(int(ptr))).copy()
        return mx.array(arr.reshape(shape), dtype=mx.uint32)
    raise TypeError(f"unsupported native scratch dtype: {dtype}")


def _row(x, row: int):
    x = _as_array(x)
    if x.ndim == 3:
        return x[:, row : row + 1, :]
    if x.ndim == 2:
        return x[row : row + 1, :].reshape(1, 1, -1)
    return x.reshape(1, -1)[:, row : row + 1]


def _clone(x):
    if x is None or not hasattr(x, "shape"):
        return x
    y = x + mx.zeros_like(x)
    mx.eval(y)
    return y


def _diff(a, b) -> tuple[float, float]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def _print_diff(name: str, a, b) -> None:
    if a is None or b is None:
        print(f"[mtp-m1-m2-snapshot] {name:<24} missing a={a is None} b={b is None}", flush=True)
        return
    if tuple(a.shape) != tuple(b.shape):
        print(
            f"[mtp-m1-m2-snapshot] {name:<24} shape a={tuple(a.shape)} b={tuple(b.shape)}",
            flush=True,
        )
        return
    max_abs, mean_abs = _diff(a, b)
    print(
        f"[mtp-m1-m2-snapshot] {name:<24} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} shape={tuple(a.shape)}",
        flush=True,
    )


def _arrays_cache_layer_indices(cache) -> list[int]:
    return [i for i, c in enumerate(cache) if type(c).__name__ == "ArraysCache"]


def _array_update_input_sid(dispatcher, layer: int) -> int | None:
    for t in dispatcher._tg.get("tasks", []):
        if (
            t.get("layer_idx") == layer
            and str(t.get("kernel", "")).startswith("fusekit_arrays_cache_shift_append_decode")
            and t.get("inputs")
        ):
            b = t["inputs"][0]
            if b.get("role") == "scratch":
                return int(b["scratch_id"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--main", type=int, required=True)
    ap.add_argument("--draft", type=int, required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    prefix = _parse_tokens(args.prefix)

    cache_m1 = raw_model.make_cache()
    cache_m2 = raw_model.make_cache()
    _ = raw_model(prompt, cache=cache_m1)
    _ = raw_model(prompt, cache=cache_m2)
    _advance_cache(raw_model, cache_m1, prefix)
    _advance_cache(raw_model, cache_m2, prefix)

    layer = args.layer
    if layer not in _arrays_cache_layer_indices(cache_m1):
        raise RuntimeError(f"layer {layer} is not an ArraysCache layer")

    row0 = mx.array([[args.main]], dtype=mx.uint32)
    verify = mx.array([[args.main, args.draft]], dtype=mx.uint32)

    with _env(FUSEKIT_DISABLE_MTP_MN_VARIANT="1"):
        out_m1 = mtp_bg._call_backbone(model, row0, cache_m1, n_confirmed=0)
    mx.eval(out_m1[0], out_m1[1])
    disp_m1 = getattr(model, "_fk_dispatcher", None)
    if disp_m1 is None:
        raise RuntimeError("M=1 dispatcher was not created")
    sid_m1 = _array_update_input_sid(disp_m1, layer)
    qkv_m1 = _clone(_row(disp_m1._scratch_buf(sid_m1), 0)) if sid_m1 is not None else None
    conv_m1 = _clone(cache_m1[layer][0])
    ssm_m1 = _clone(cache_m1[layer][1])

    with _env(
        FUSEKIT_DISABLE_MTP_MN_VARIANT=None,
        FUSEKIT_USE_MTP_MN_VARIANT="1",
        FUSEKIT_EXPERIMENTAL_STATEFUL_MTP_MN_VARIANT="1",
    ):
        out_m2 = mtp_bg._call_backbone(model, verify, cache_m2, n_confirmed=1)
    mx.eval(out_m2[0], out_m2[1])
    disp_m2 = model._get_variant_dispatcher(2)
    if disp_m2 is None:
        raise RuntimeError("M=2 dispatcher was not created")
    sid_m2 = _array_update_input_sid(disp_m2, layer)
    qkv_m2 = _clone(_row(disp_m2._scratch_buf(sid_m2), 0)) if sid_m2 is not None else None
    rollback = getattr(cache_m2[layer], "rollback_state", None)
    conv_rb = rollback[0] if rollback is not None else None
    ssm_rb = rollback[1] if rollback is not None else None

    print(
        "[mtp-m1-m2-snapshot] "
        f"prompt_tokens={len(prompt_ids)} prefix={prefix} main={args.main} draft={args.draft} layer={layer}",
        flush=True,
    )
    _print_diff("row0_qkv_m1_vs_m2", qkv_m1, qkv_m2)
    _print_diff("conv_state_m1_vs_rb", conv_m1, conv_rb)
    _print_diff("ssm_state_m1_vs_rb", ssm_m1, ssm_rb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
