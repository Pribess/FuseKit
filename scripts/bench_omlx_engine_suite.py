"""Run oMLX engine raw-vs-FuseKit suites sequentially.

Each mode runs in a fresh Python process because `FUSEKIT_DISABLE_HOOK`
must be present before interpreter startup for the raw baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as stats
import subprocess
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OMLX_APP = Path(os.environ.get("FUSEKIT_MLX_APP", "/Applications/oMLX.app")).expanduser()
OMLX_MODEL_ROOT = Path(
    os.environ.get("FUSEKIT_OMLX_DIR", Path.home() / ".omlx/models")
).expanduser()
DEFAULT_MODEL_IDS = [
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit",
    "mlx-community/Qwen3.5-4B-bf16",
    "mlx-community/Qwen3.5-4B-4bit",
    "mlx-community/Qwen3.5-4B-8bit",
]
DEFAULT_MODELS = [str(OMLX_MODEL_ROOT / model) for model in DEFAULT_MODEL_IDS]

PYTHON = os.environ.get(
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

METRIC_RE = re.compile(
    r"\[engine:run:(\d+)\] "
    r"prompt_tokens=(\d+) generated_tokens=(\d+) "
    r"wall_s=([0-9.]+) wall_tps=([0-9.]+) peak_gb=([0-9.]+)"
)
LOAD_RE = re.compile(r"\[engine\] model=(.*?) mode=(.*?) load_s=([0-9.]+)")


def label(model: str) -> str:
    p = Path(model)
    try:
        idx = p.parts.index("models")
        return "/".join(p.parts[idx + 1 :])
    except ValueError:
        return str(p)


def mean(values: list[float]) -> float | None:
    return stats.mean(values) if values else None


def stdev(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new / old - 1.0) * 100.0


def parse(output: str) -> tuple[float | None, list[dict[str, float]]]:
    load_s = None
    reps: list[dict[str, float]] = []
    for line in output.splitlines():
        lm = LOAD_RE.search(line)
        if lm:
            load_s = float(lm.group(3))
            continue
        mm = METRIC_RE.search(line)
        if not mm:
            continue
        _, prompt_tokens, generated, wall_s, wall_tps, peak_gb = mm.groups()
        reps.append(
            {
                "prompt_tokens": float(prompt_tokens),
                "generated_tokens": float(generated),
                "wall_s": float(wall_s),
                "wall_tps": float(wall_tps),
                "peak_gb": float(peak_gb),
            }
        )
    return load_s, reps


def parse_profile(output: str) -> list[str]:
    return [
        line
        for line in output.splitlines()
        if line.startswith("[engine-profile]")
    ]


def summarize(reps: list[dict[str, float]]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"n": len(reps)}
    for key in ("wall_tps", "wall_s", "peak_gb"):
        values = [rep[key] for rep in reps if key in rep]
        summary[f"{key}_avg"] = mean(values)
        summary[f"{key}_sd"] = stdev(values)
    generated = [rep["generated_tokens"] for rep in reps if "generated_tokens" in rep]
    summary["generated_tokens_min"] = min(generated) if generated else None
    return summary


def run_one(
    workspace: Path,
    model: str,
    tokens: int,
    repeat: int,
    warmup: int,
    mode: str,
    prompt: str,
    profile_steps: int,
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("FUSEKIT_PROFILE_BUILD", None)
    env.pop("FUSEKIT_PROFILE_ICB", None)
    env.pop("FUSEKIT_PROFILE", None)
    env.pop("FUSEKIT_ENABLE_ENGINE_CHUNK", None)
    env.pop("FUSEKIT_ENGINE_CHUNK_MODE", None)
    env.pop("FUSEKIT_ENGINE_DRAIN", None)
    env.pop("FUSEKIT_ENGINE_ALLOW_LONG_GPU_CHAIN", None)
    env.pop("FK_BUNDLE_CMDS", None)
    env.pop("FK_CHAIN_ENCODERS", None)
    if mode == "engine_raw":
        env["FUSEKIT_DISABLE_HOOK"] = "1"
        env.pop("FUSEKIT_ENGINE_STYLE", None)
        env.pop("FUSEKIT_ENABLE_ENGINE_FAST", None)
    else:
        env.pop("FUSEKIT_DISABLE_HOOK", None)
        env["FUSEKIT_ENGINE_STYLE"] = "omlx"
        env["FUSEKIT_ENABLE_ENGINE_FAST"] = "1"

    cmd = [
        PYTHON,
        "scripts/bench_omlx_engine.py",
        model,
        "--mode",
        mode,
        "--tokens",
        str(tokens),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--prompt",
        prompt,
    ]
    if profile_steps:
        cmd.extend(["--profile-steps", str(profile_steps)])
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed_s = time.perf_counter() - started
    load_s, reps = parse(proc.stdout)
    profile = parse_profile(proc.stdout)
    return {
        "model": label(model),
        "model_path": model,
        "tokens": tokens,
        "mode": mode,
        "repeat": repeat,
        "warmup": warmup,
        "rc": proc.returncode,
        "elapsed_s": elapsed_s,
        "load_s": load_s,
        "summary": summarize(reps),
        "reps": reps,
        "profile": profile,
        "tail": proc.stdout[-4000:] if proc.returncode != 0 else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="pass through to bench_omlx_engine.py; 0 disables profiling",
    )
    parser.add_argument(
        "--prompt",
        default="Repeat the word token forever: token token token",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/private/tmp/fusekit-omlx-engine-0p5B-4B-L64-256-r5.jsonl"),
    )
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    print(f"[engine-suite] writing {args.out}", flush=True)
    with args.out.open("w") as out_file:
        for model in models:
            for token_count in args.tokens:
                pair = []
                for mode in ("engine_raw", "engine_fusekit"):
                    print(
                        "[engine-suite] START "
                        f"model={label(model)} L={token_count} mode={mode} "
                        f"repeat={args.repeat} warmup={args.warmup}",
                        flush=True,
                    )
                    rec = run_one(
                        args.workspace,
                        model,
                        token_count,
                        args.repeat,
                        args.warmup,
                        mode,
                        args.prompt,
                        args.profile_steps,
                    )
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_file.flush()
                    pair.append(rec)
                    summary = rec["summary"]
                    if rec["rc"] == 0:
                        print(
                            "[engine-suite] DONE  "
                            f"model={rec['model']} L={token_count} mode={mode} "
                            f"wall={fmt(summary.get('wall_tps_avg'))} "
                            f"wall_sd={fmt(summary.get('wall_tps_sd'))} "
                            f"peak={fmt(summary.get('peak_gb_avg'))} "
                            f"elapsed={fmt(rec['elapsed_s'], 1)}s",
                            flush=True,
                        )
                    else:
                        print(
                            "[engine-suite] FAIL  "
                            f"model={rec['model']} L={token_count} mode={mode} "
                            f"rc={rec['rc']} elapsed={fmt(rec['elapsed_s'], 1)}s",
                            flush=True,
                        )
                        print(rec["tail"], flush=True)
                if len(pair) == 2 and pair[0]["rc"] == 0 and pair[1]["rc"] == 0:
                    raw_summary = pair[0]["summary"]
                    fk_summary = pair[1]["summary"]
                    wall_delta = pct(
                        fk_summary.get("wall_tps_avg"),
                        raw_summary.get("wall_tps_avg"),
                    )
                    peak_delta = pct(
                        fk_summary.get("peak_gb_avg"),
                        raw_summary.get("peak_gb_avg"),
                    )
                    print(
                        "[engine-suite] DELTA "
                        f"model={pair[0]['model']} L={token_count} "
                        f"wall={fmt(wall_delta, 2)}% "
                        f"peak={fmt(peak_delta, 2)}%",
                        flush=True,
                    )
    print("[engine-suite] COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
