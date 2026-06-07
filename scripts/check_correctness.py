"""Compare raw MLX logits against fusekit-dispatched logits.

Runs raw and patched paths in separate Python processes so large models
are not loaded twice in one address space. The child process writes the
last-token logits with MLX serialization; the parent reports argmax token,
max/mean absolute error, and whether the patched path actually built a
dispatcher instead of silently falling back.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def child_main(args: argparse.Namespace) -> int:
    if args.mode == "raw":
        os.environ["FUSEKIT_DISABLE_DISPATCH"] = "1"
        os.environ["FUSEKIT_DISABLE_FAST"] = "1"
    else:
        os.environ.pop("FUSEKIT_DISABLE_DISPATCH", None)
        os.environ.pop("FUSEKIT_DISABLE_FAST", None)
        os.environ.pop("FUSEKIT_ENABLE_FAST", None)
        if args.force_dispatch:
            os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
            os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import maybe_quantize_kv_cache
    from mlx_lm.models.cache import make_prompt_cache

    model_dir = str(model_path(args.model))
    model, tok = load(model_dir)
    if args.mode == "patched" and args.fusekit_dir:
        import fusekit_runtime as fr

        # The auto hook may already have wrapped the model from
        # <model>/fusekit. For A/B compiler experiments, unwrap that and
        # attach the explicit dry-compile directory instead.
        model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
    ids = tok.encode(args.prompt)
    if not ids:
        ids = [0]
    ids = mx.array([ids], dtype=mx.uint32)
    cache = make_prompt_cache(model)

    prefill = model(ids, cache=cache)
    if isinstance(prefill, tuple):
        prefill = prefill[0]
    mx.eval(prefill)
    maybe_quantize_kv_cache(
        cache,
        quantized_kv_start=args.quantized_kv_start,
        kv_group_size=args.kv_group_size,
        kv_bits=args.kv_bits,
    )
    mx.eval([c.state for c in cache])

    decode_id = mx.array([[args.decode_token]], dtype=mx.uint32)
    logits = model(decode_id, cache=cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)
    last = logits[:, -1, :].astype(mx.float32)
    mx.eval(last)

    mx.save(args.out, last)
    flat = last.reshape(-1)
    meta = {
        "mode": args.mode,
        "shape": list(last.shape),
        "argmax": int(mx.argmax(flat).item()),
        "wrapped": type(model).__name__,
        "dispatch_failed": bool(getattr(model, "_fk_dispatch_failed", False)),
        "dispatcher_built": getattr(model, "_fk_dispatcher", None) is not None,
        "fusekit_dir": str(getattr(model, "_fk_dir", "")),
        "strict_dispatch": bool(os.environ.get("FUSEKIT_STRICT_DISPATCH")),
    }
    Path(args.meta).write_text(json.dumps(meta, sort_keys=True))
    return 0


def parent_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    with tempfile.TemporaryDirectory(prefix="fusekit-correctness-") as td:
        tmp = Path(td)
        raw_out = tmp / "raw.npy"
        patched_out = tmp / "patched.npy"
        raw_meta = tmp / "raw.json"
        patched_meta = tmp / "patched.json"

        base_cmd = [
            sys.executable,
            __file__,
            "--child",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--decode-token",
            str(args.decode_token),
            "--kv-group-size",
            str(args.kv_group_size),
            "--quantized-kv-start",
            str(args.quantized_kv_start),
        ]
        if args.kv_bits is not None:
            base_cmd += ["--kv-bits", str(args.kv_bits)]
        if args.fusekit_dir:
            base_cmd += ["--fusekit-dir", args.fusekit_dir]
        subprocess.run(
            base_cmd + ["--mode", "raw", "--out", str(raw_out), "--meta", str(raw_meta)],
            check=True,
        )
        subprocess.run(
            base_cmd
            + ["--mode", "patched", "--out", str(patched_out), "--meta", str(patched_meta)]
            + (["--force-dispatch"] if args.force_dispatch else []),
            check=True,
        )

        raw = mx.load(str(raw_out))
        patched = mx.load(str(patched_out))
        diff = mx.abs(raw - patched)
        mx.eval(diff)
        raw_flat = raw.reshape(-1)
        patched_flat = patched.reshape(-1)
        raw_top = mx.argsort(raw_flat)[-5:][::-1]
        patched_top = mx.argsort(patched_flat)[-5:][::-1]
        mx.eval(raw_top, patched_top)
        raw_top_list = [
            {"token": int(i.item()), "logit": float(raw_flat[i].item())}
            for i in raw_top
        ]
        patched_top_list = [
            {"token": int(i.item()), "logit": float(patched_flat[i].item())}
            for i in patched_top
        ]
        rm = json.loads(raw_meta.read_text())
        pm = json.loads(patched_meta.read_text())
        print(json.dumps({
            "model": args.model,
            "prompt": args.prompt,
            "decode_token": args.decode_token,
            "raw_argmax": rm["argmax"],
            "patched_argmax": pm["argmax"],
            "same_argmax": rm["argmax"] == pm["argmax"],
            "max_abs_diff": float(mx.max(diff).item()),
            "mean_abs_diff": float(mx.mean(diff).item()),
            "raw_top5": raw_top_list,
            "patched_top5": patched_top_list,
        "patched_dispatcher_built": pm["dispatcher_built"],
        "patched_dispatch_failed": pm["dispatch_failed"],
        "kv_bits": args.kv_bits,
        "raw_meta": rm,
        "patched_meta": pm,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello from fusekit")
    ap.add_argument("--decode-token", type=int, default=1)
    ap.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--quantized-kv-start", type=int, default=0)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", choices=["raw", "patched"])
    ap.add_argument("--out")
    ap.add_argument("--meta")
    ap.add_argument("--force-dispatch", action="store_true")
    ap.add_argument("--fusekit-dir")
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
