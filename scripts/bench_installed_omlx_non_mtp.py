#!/usr/bin/env python3
"""Run raw oMLX vs FuseKit engine benchmarks for installed models.

This intentionally does not pass --mtp. Models with MTP weights are still
benchmarked through their normal single-token decode path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


OMLX_APP = Path(os.environ.get("FUSEKIT_MLX_APP", "/Applications/oMLX.app")).expanduser()
PY = os.environ.get(
    "FUSEKIT_PYTHON",
    str(OMLX_APP / "Contents/Python/cpython-3.11/bin/python3.11"),
)
PYTHONPATH = os.pathsep.join(
    [
        str(OMLX_APP / "Contents/Resources"),
        str(
            OMLX_APP
            / "Contents/Python/framework-mlx-framework/lib/python3.11/site-packages"
        ),
    ]
)
OMLX_MODEL_ROOT = Path(
    os.environ.get("FUSEKIT_OMLX_DIR", Path.home() / ".omlx/models")
).expanduser()

MODELS = [
    "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed",
    "corysus/Qwen3.5-4B-oQ4-mtp",
    "fcobos/Qwen3.5-0.8B-oQ4-mtp",
    "fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16",
    "mlx-community/Qwen3.5-4B-4bit",
    "mlx-community/Qwen3.5-4B-8bit",
    "mlx-community/Qwen3.5-4B-bf16",
    "mlx-community/Qwen3.6-27B-4bit",
]

AVG_RE = re.compile(
    r"wall_tps_avg=(?P<wall>[0-9.]+)\s+"
    r"wall_tps_stdev=(?P<stdev>[0-9.]+)\s+"
    r"wall_s_avg=(?P<wall_s>[0-9.]+)\s+"
    r"peak_gb_avg=(?P<peak>[0-9.]+)"
)


def env_for(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/fusekit-pycache"
    if mode == "engine_raw":
        env["FUSEKIT_DISABLE_HOOK"] = "1"
        env.pop("FUSEKIT_ENGINE_STYLE", None)
        env.pop("FUSEKIT_ENABLE_ENGINE_FAST", None)
    else:
        env.pop("FUSEKIT_DISABLE_HOOK", None)
        env["FUSEKIT_ENGINE_STYLE"] = "omlx"
        env["FUSEKIT_ENABLE_ENGINE_FAST"] = "1"
    return env


def bench_model_arg(model: str) -> str:
    local = OMLX_MODEL_ROOT / model
    if local.is_dir():
        return str(local)
    return model


def run(cmd: list[str], env: dict[str, str] | None, timeout: int) -> tuple[int, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    return proc.returncode, f"[elapsed_s={elapsed:.1f}]\n{proc.stdout}"


def parse_avg(output: str) -> dict[str, float] | None:
    for line in output.splitlines():
        m = AVG_RE.search(line)
        if m:
            return {
                "wall_tps": float(m.group("wall")),
                "wall_tps_stdev": float(m.group("stdev")),
                "wall_s": float(m.group("wall_s")),
                "peak_gb": float(m.group("peak")),
            }
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-64", type=int, default=420)
    parser.add_argument("--timeout-512", type=int, default=1200)
    parser.add_argument(
        "--out",
        default="/private/tmp/fusekit-omlx-nonmtp-L64-512-r5.jsonl",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench = Path("scripts/bench_omlx_engine.py")

    with out_path.open("a", encoding="utf-8") as f:
        for model in MODELS:
            print(f"== model {model} ==", flush=True)
            patch_cmd = [
                "./target/debug/fusekit",
                "patch",
                model,
                "--patterns",
                "off",
                "--force",
            ]
            try:
                code, patch_out = run(patch_cmd, env=None, timeout=300)
            except subprocess.TimeoutExpired as exc:
                rec = {
                    "model": model,
                    "phase": "patch",
                    "status": "timeout",
                    "output": (exc.stdout or "") + (exc.stderr or ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"PATCH_TIMEOUT model={model}", flush=True)
                continue
            rec = {
                "model": model,
                "phase": "patch",
                "status": "ok" if code == 0 else "error",
                "returncode": code,
                "output_tail": patch_out[-4000:],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if code != 0:
                print(f"PATCH_ERROR model={model}", flush=True)
                continue

            for tokens in args.tokens:
                timeout = args.timeout_512 if tokens >= 512 else args.timeout_64
                for mode in ("engine_raw", "engine_fusekit"):
                    cmd = [
                        PY,
                        str(bench),
                        "--mode",
                        mode,
                        "--tokens",
                        str(tokens),
                        "--repeat",
                        str(args.repeat),
                        "--warmup",
                        str(args.warmup),
                        "--seed",
                        str(args.seed),
                        bench_model_arg(model),
                    ]
                    print(
                        f"RUN model={model} L={tokens} mode={mode}",
                        flush=True,
                    )
                    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    try:
                        code, output = run(cmd, env=env_for(mode), timeout=timeout)
                        status = "ok" if code == 0 else "error"
                    except subprocess.TimeoutExpired as exc:
                        code = None
                        status = "timeout"
                        output = (exc.stdout or "") + (exc.stderr or "")
                    avg = parse_avg(output)
                    rec = {
                        "model": model,
                        "tokens": tokens,
                        "mode": mode,
                        "status": status,
                        "returncode": code,
                        "started": started,
                        "avg": avg,
                        "output_tail": output[-4000:],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if avg:
                        print(
                            "RESULT "
                            f"model={model} L={tokens} mode={mode} "
                            f"wall_tps={avg['wall_tps']:.3f} "
                            f"stdev={avg['wall_tps_stdev']:.3f} "
                            f"peak={avg['peak_gb']:.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"RESULT_ERROR model={model} L={tokens} "
                            f"mode={mode} status={status}",
                            flush=True,
                        )
    print(f"OUT {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
