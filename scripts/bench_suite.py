"""Run raw-vs-fusekit benchmark suites and write JSONL results.

This is intentionally a thin subprocess driver around `fusekit bench` so it
uses the same CLI path the user runs manually. It keeps runs sequential to
avoid loading multiple large models at once.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
import subprocess
import time
from pathlib import Path


DEFAULT_MODELS = [
    "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16",
    "mlx-community/Qwen3.5-4B-bf16",
    "mlx-community/Qwen3.6-27B-4bit",
]

METRIC_RE = re.compile(
    r"\[bench:(\d+)\] (stream_tps|wall_s|wall_tps|peak_gb|generated_tokens)=(.*)"
)
LOAD_RE = re.compile(r"\[bench\] load_s=(.*)")
ATTACH_RE = re.compile(r"tasks=(?P<tasks>\d+)\s+kernels=(?P<kernels>\d+)")


def parse_bench_output(output: str) -> tuple[float | None, list[dict[str, float]]]:
    reps: dict[int, dict[str, float]] = {}
    load_s = None
    for line in output.splitlines():
        load_match = LOAD_RE.search(line)
        if load_match:
            try:
                load_s = float(load_match.group(1))
            except ValueError:
                pass
        metric_match = METRIC_RE.search(line)
        if not metric_match:
            continue
        idx_raw, key, value_raw = metric_match.groups()
        try:
            value = float(value_raw)
        except ValueError:
            continue
        reps.setdefault(int(idx_raw), {})[key] = value
    return load_s, [reps[i] for i in sorted(reps)]


def mean(values: list[float]) -> float | None:
    return stats.mean(values) if values else None


def stdev(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def summarize(reps: list[dict[str, float]]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"n": len(reps)}
    for key in ("stream_tps", "wall_tps", "wall_s", "peak_gb"):
        values = [rep[key] for rep in reps if key in rep]
        summary[f"{key}_avg"] = mean(values)
        summary[f"{key}_sd"] = stdev(values)
    return summary


def parse_dispatch_info(output: str) -> dict[str, int | bool | None]:
    attached = False
    tasks = None
    kernels = None
    fallback = False
    for line in output.splitlines():
        if "[fusekit] attached" in line:
            attached = True
        match = ATTACH_RE.search(line)
        if match:
            tasks = int(match.group("tasks"))
            kernels = int(match.group("kernels"))
        lowered = line.lower()
        if "falling back" in lowered or "fallback" in lowered:
            fallback = True
    return {
        "attached": attached,
        "tasks": tasks,
        "kernels": kernels,
        "fallback": fallback,
    }


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new / old - 1.0) * 100.0


def run_one(
    workspace: Path,
    binary: str,
    model: str,
    tokens: int,
    repeat: int,
    mode: str,
) -> dict[str, object]:
    cmd = [
        binary,
        "bench",
        model,
        "--tokens",
        str(tokens),
        "--repeat",
        str(repeat),
    ]
    if mode == "raw":
        cmd.append("--raw")
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed_s = time.perf_counter() - started
    load_s, reps = parse_bench_output(proc.stdout)
    dispatch = parse_dispatch_info(proc.stdout)
    return {
        "model": model,
        "tokens": tokens,
        "mode": mode,
        "repeat": repeat,
        "rc": proc.returncode,
        "elapsed_s": elapsed_s,
        "load_s": load_s,
        "summary": summarize(reps),
        "dispatch": dispatch,
        "reps": reps,
        "tail": proc.stdout[-2000:] if proc.returncode != 0 or dispatch["fallback"] else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/Users/pribess/Documents/workspace/fusekit"),
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--bin", default="./target/release/fusekit")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/private/tmp/fusekit-bench-6models-L64-256-r5.jsonl"),
    )
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    print(f"[bench-suite] writing {args.out}", flush=True)
    with args.out.open("w") as out_file:
        for model in models:
            for token_count in args.tokens:
                pair = []
                for mode in ("raw", "patched"):
                    print(
                        "[bench-suite] START "
                        f"model={model} L={token_count} mode={mode} "
                        f"repeat={args.repeat}",
                        flush=True,
                    )
                    rec = run_one(
                        args.workspace,
                        args.bin,
                        model,
                        token_count,
                        args.repeat,
                        mode,
                    )
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_file.flush()
                    pair.append(rec)
                    summary = rec["summary"]
                    if rec["rc"] == 0:
                        print(
                            "[bench-suite] DONE  "
                            f"model={model} L={token_count} mode={mode} "
                            f"stream={fmt(summary.get('stream_tps_avg'))} "
                            f"wall={fmt(summary.get('wall_tps_avg'))} "
                            f"peak={fmt(summary.get('peak_gb_avg'))} "
                            f"fallback={rec['dispatch']['fallback']} "
                            f"elapsed={fmt(rec['elapsed_s'], 1)}s",
                            flush=True,
                        )
                    else:
                        print(
                            "[bench-suite] FAIL  "
                            f"model={model} L={token_count} mode={mode} "
                            f"rc={rec['rc']} elapsed={fmt(rec['elapsed_s'], 1)}s",
                            flush=True,
                        )
                        print(rec["tail"], flush=True)
                if len(pair) == 2 and pair[0]["rc"] == 0 and pair[1]["rc"] == 0:
                    raw_summary = pair[0]["summary"]
                    patched_summary = pair[1]["summary"]
                    stream_delta = pct(
                        patched_summary.get("stream_tps_avg"),
                        raw_summary.get("stream_tps_avg"),
                    )
                    wall_delta = pct(
                        patched_summary.get("wall_tps_avg"),
                        raw_summary.get("wall_tps_avg"),
                    )
                    peak_delta = pct(
                        patched_summary.get("peak_gb_avg"),
                        raw_summary.get("peak_gb_avg"),
                    )
                    print(
                        "[bench-suite] DELTA "
                        f"model={model} L={token_count} "
                        f"stream={fmt(stream_delta, 2)}% "
                        f"wall={fmt(wall_delta, 2)}% "
                        f"peak={fmt(peak_delta, 2)}%",
                        flush=True,
                    )
    print("[bench-suite] COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
