"""Locate DeepSeek forced-dispatch drift by comparing per-layer outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _diff(mx, a, b) -> tuple[float, float]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--stop-after", type=float, default=1e-3)
    args = ap.parse_args()

    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"
    os.environ["FUSEKIT_DISABLE_NATIVE_SCRATCH"] = "1"

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import apply_post_load_transforms, load_text_model

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(),
    )
    model = apply_post_load_transforms(model, ModelSettings())
    raw = getattr(model, "_fk_original", model)

    prompt_ids = tokenizer.encode(args.prompt) or [0]
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    decode = mx.array([[args.decode_token]], dtype=mx.uint32)

    cache_raw = make_prompt_cache(raw)
    cache_fk = make_prompt_cache(raw)

    raw_prefill = raw(prompt, cache=cache_raw)
    fk_prefill = raw(prompt, cache=cache_fk)
    mx.eval(raw_prefill, fk_prefill)

    fk_logits = model(decode, cache=cache_fk)
    if isinstance(fk_logits, tuple):
        fk_logits = fk_logits[0]
    mx.eval(fk_logits)
    disp = getattr(model, "_fk_dispatcher", None)
    if disp is None:
        raise RuntimeError("FuseKit dispatcher was not built by forced decode")

    h = raw.model.embed_tokens(decode)
    mask = create_attention_mask(h, cache_raw[0])
    mx.eval(h)

    print(
        f"[deepseek-layer-drift] prompt_tokens={len(prompt_ids)} "
        f"decode_token={args.decode_token}",
        flush=True,
    )
    print(
        f"[deepseek-layer-drift] dispatcher_active_m={getattr(disp, '_active_m', None)} "
        f"scratch_layers={len(getattr(raw.model, 'pipeline_layers', []))}",
        flush=True,
    )

    completed_all_layers = True
    for li, (layer, cache) in enumerate(zip(raw.model.pipeline_layers, cache_raw, strict=False)):
        h = layer(h, mask, cache=cache)
        mx.eval(h)
        sid = disp._layer_scratch_id.get(("out_buf", li))
        if sid is None:
            print(f"[deepseek-layer-drift] layer={li:02d} missing out_buf scratch", flush=True)
            continue
        fk_h = disp._scratch_buf(sid)
        if fk_h.ndim == 3:
            fk_view = fk_h[:, :1, :]
        elif fk_h.ndim == 2:
            fk_view = fk_h[:1, :].reshape(1, 1, -1)
        else:
            print(
                f"[deepseek-layer-drift] layer={li:02d} unexpected scratch shape={fk_h.shape}",
                flush=True,
            )
            continue
        mx.eval(fk_view)
        max_abs, mean_abs = _diff(mx, h, fk_view)
        print(
            f"[deepseek-layer-drift] layer={li:02d} "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
            f"raw_shape={tuple(h.shape)} fk_shape={tuple(fk_view.shape)} sid={sid}",
            flush=True,
        )
        if max_abs > args.stop_after:
            completed_all_layers = False
            break

    if completed_all_layers:
        raw_logits = raw.lm_head(raw.model.norm(h))
        mx.eval(raw_logits)
        max_abs, mean_abs = _diff(mx, raw_logits, fk_logits)
        print(
            f"[deepseek-layer-drift] logits max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}",
            flush=True,
        )
    else:
        print(
            "[deepseek-layer-drift] logits skipped because the layer scan stopped early",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
