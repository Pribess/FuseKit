"""Check whether MTP verify row 0 depends on the speculative row 1 token."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import apply_post_load_transforms, load_text_model
from debug_qwen35_linear_layer import find_linear_layer_sids


def _model_path(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return str(p)
    return str(Path("~/.omlx/models").expanduser() / name_or_path)


def _argmax_pos(logits, pos: int) -> int:
    return int(mx.argmax(logits[:, pos, :], axis=-1).item())


def _diff(a, b) -> tuple[float, float]:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(mx.max(d).item()), float(mx.mean(d).item())


def _top(logits, pos: int, k: int = 5) -> list[tuple[int, float]]:
    row = logits[:, pos, :].reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
    idx = mx.argpartition(-row, kth=k - 1)[:k]
    mx.eval(idx, row)
    return sorted(
        ((int(i), float(row[int(i)].item())) for i in idx.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )


def _row(x, pos: int):
    if x.ndim == 3:
        return x[:, pos : pos + 1, :].reshape(1, -1)
    if x.ndim == 2:
        return x[pos : pos + 1, :]
    return x.reshape(1, -1)


def _snapshot_layer0(variant, names: list[str]):
    sids = find_linear_layer_sids(variant._tg, 0, variant._hidden)
    out = {}
    for name in names:
        sid = sids.get(name)
        if sid is None:
            continue
        arr = _row(variant._scratch_buf(sid), 1).astype(mx.float32) + 0.0
        mx.eval(arr)
        out[name] = arr
    return out


def _snapshot_layer_outs(variant):
    out = {}
    for (name, layer), sid in sorted(variant._layer_scratch_id.items()):
        if name != "out_buf":
            continue
        arr = _row(variant._scratch_buf(sid), 1).astype(mx.float32) + 0.0
        mx.eval(arr)
        out[int(layer)] = arr
    logits = variant._scratch_buf(variant._global_scratch_id["logits"])
    row1 = _row(logits, 1).astype(mx.float32) + 0.0
    mx.eval(row1)
    return out, row1


def _advance_prefix(model, tokenizer, prompt: str, prefix: list[int]):
    prompt_ids = tokenizer.encode(prompt) or [0]
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids], dtype=mx.uint32), cache=cache)
    mx.eval(logits)
    for tok in prefix:
        out = model(mx.array([[tok]], dtype=mx.uint32), cache=cache, return_hidden=True)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    return cache


def _parse_tokens(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", default="x")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--main", type=int, required=True)
    ap.add_argument("--draft-a", type=int, required=True)
    ap.add_argument("--draft-b", type=int, required=True)
    args = ap.parse_args()

    model, tokenizer = load_text_model(
        _model_path(args.model),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    model = apply_post_load_transforms(model, ModelSettings(mtp_enabled=True))
    raw_model = getattr(model, "_fk_original", model)
    prefix = _parse_tokens(args.prefix)

    for label, candidate_model in (("raw", raw_model), ("fusekit", model)):
        cache_a = _advance_prefix(candidate_model, tokenizer, args.prompt, prefix)
        cache_b = _advance_prefix(candidate_model, tokenizer, args.prompt, prefix)
        inp_a = mx.array([[args.main, args.draft_a]], dtype=mx.uint32)
        inp_b = mx.array([[args.main, args.draft_b]], dtype=mx.uint32)
        out_a = candidate_model(inp_a, cache=cache_a, return_hidden=True, n_confirmed=1)
        variant = getattr(candidate_model, "_get_variant_dispatcher", lambda _n: None)(2)
        snap_a = None
        if label == "fusekit" and variant is not None:
            snap_a = _snapshot_layer0(
                variant,
                [
                    "layer_in",
                    "norm_x",
                    "qkv",
                    "conv_silu",
                    "gd",
                    "gated_float",
                    "h_attn",
                    "layer_out",
                ],
            )
            layer_outs_a, logits_row1_a = _snapshot_layer_outs(variant)
        out_b = candidate_model(inp_b, cache=cache_b, return_hidden=True, n_confirmed=1)
        snap_b = None
        layer_outs_b = None
        logits_row1_b = None
        if label == "fusekit" and variant is not None:
            snap_b = _snapshot_layer0(
                variant,
                [
                    "layer_in",
                    "norm_x",
                    "qkv",
                    "conv_silu",
                    "gd",
                    "gated_float",
                    "h_attn",
                    "layer_out",
                ],
            )
            layer_outs_b, logits_row1_b = _snapshot_layer_outs(variant)
        logits_a, hidden_a = out_a
        logits_b, hidden_b = out_b
        mx.eval(logits_a, hidden_a, logits_b, hidden_b)
        logit_max, logit_mean = _diff(logits_a[:, 0:1, :], logits_b[:, 0:1, :])
        hidden_max, hidden_mean = _diff(hidden_a[:, 0:1, :], hidden_b[:, 0:1, :])
        print(
            "[mtp-row0] "
            f"{label} prefix_len={len(prefix)} main={args.main} "
            f"draft_a={args.draft_a} draft_b={args.draft_b} "
            f"row0_argmax_a={_argmax_pos(logits_a, 0)} "
            f"row0_argmax_b={_argmax_pos(logits_b, 0)} "
            f"row1_argmax_a={_argmax_pos(logits_a, 1)} "
            f"row1_argmax_b={_argmax_pos(logits_b, 1)} "
            f"row0_logit_max={logit_max:.6f} row0_logit_mean={logit_mean:.6f} "
            f"row0_hidden_max={hidden_max:.6f} row0_hidden_mean={hidden_mean:.6f}",
            flush=True,
        )
        if logit_max:
            print(f"[mtp-row0] {label} top_a={_top(logits_a, 0)}", flush=True)
            print(f"[mtp-row0] {label} top_b={_top(logits_b, 0)}", flush=True)
        if snap_a is not None and snap_b is not None:
            for name in snap_a:
                max_abs, mean_abs = _diff(snap_a[name], snap_b[name])
                print(
                    "[mtp-row0] "
                    f"fusekit layer0.row1 {name:<12} "
                    f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}",
                    flush=True,
                )
        if layer_outs_b is not None:
            for layer in sorted(layer_outs_a):
                max_abs, mean_abs = _diff(layer_outs_a[layer], layer_outs_b[layer])
                print(
                    "[mtp-row0] "
                    f"fusekit layer={layer:02d}.row1_out "
                    f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}",
                    flush=True,
                )
            max_abs, mean_abs = _diff(logits_row1_a, logits_row1_b)
            print(
                "[mtp-row0] "
                f"fusekit logits.row1 max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
