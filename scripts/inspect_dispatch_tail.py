"""Inspect final decode dispatch labels, barriers and buffer pointers."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


def model_path(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return Path("~/.omlx/models").expanduser() / name_or_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fusekit-dir", required=True)
    ap.add_argument("--prompt", default="Write a concise Python function that parses JSON lines.")
    args = ap.parse_args()

    os.environ["FUSEKIT_FORCE_UNSUPPORTED_GRAPH"] = "1"
    os.environ["FUSEKIT_STRICT_DISPATCH"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    import fusekit_runtime as fr

    model, tok = load(str(model_path(args.model)))
    model = fr.attach(getattr(model, "_fk_original", model), args.fusekit_dir)
    cache = make_prompt_cache(getattr(model, "_fk_original", model))
    ids = mx.array([tok.encode(args.prompt) or [0]], dtype=mx.uint32)
    y = model(ids, cache=cache)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    decode = mx.array([[1]], dtype=mx.uint32)
    z = model(decode, cache=cache)
    if isinstance(z, tuple):
        z = z[0]
    mx.eval(z)
    disp = model._fk_dispatcher
    icb = disp._ensure_icb(cache)
    if icb is None:
        raise RuntimeError("icb not built")
    labels = disp._last_dispatch_labels or []
    barriers = disp._last_barriers or []
    dispatches = disp._last_dispatches or []
    for i, (kernel, bufs, grid, tg) in enumerate(dispatches[-16:], start=len(dispatches) - 16):
        label = labels[i] if i < len(labels) else {}
        task_id = label.get("task_id")
        li = label.get("layer_idx")
        reads = label.get("read_slots")
        writes = label.get("write_slots")
        ptrs = [hex(int(p)) for p, _ in bufs]
        print(
            f"{i:03d} barrier={int(bool(barriers[i] if i < len(barriers) else False))} "
            f"task={task_id} li={li} kernel={kernel} reads={reads} writes={writes} "
            f"grid={grid} tg={tg} ptrs={ptrs}"
        )
    print("layer_out_sid", disp._layer_scratch_id.get(("out_buf", 23)))
    print("logits_sid", disp._global_scratch_id.get("logits"))
    print("scratch_alias", dict(sorted(disp._scratch_alias.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
