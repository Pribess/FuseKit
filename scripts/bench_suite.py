"""Run reproducible raw-vs-FuseKit benchmark pairs.

The suite drives the public ``fusekit bench`` command, always runs the raw
baseline first, drops configurable warmups, validates that the FuseKit runtime
actually attached, and writes both machine-readable JSONL and a Markdown
summary. Runs stay sequential so only one model is resident at a time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics as stats
import subprocess
import time
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(
    r"\[bench:(\d+)\] (stream_tps|wall_s|wall_tps|peak_gb|generated_tokens)=(.*)"
)
LOAD_RE = re.compile(r"\[bench\] load_s=(.*)")
ATTACH_RE = re.compile(r"tasks=(?P<tasks>\d+)\s+kernels=(?P<kernels>\d+)")
VERSION_RE = re.compile(r"\[bench\] mlx=(?P<mlx>\S+) mlx_lm=(?P<mlx_lm>\S+)")
FALLBACK_RE = re.compile(r"falling back|dispatcher miss|native dispatch fallback", re.I)
REQUIRED_METRICS = {"wall_s", "wall_tps", "peak_gb", "generated_tokens"}


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


def parse_versions(output: str) -> dict[str, str | None]:
    match = VERSION_RE.search(output)
    if not match:
        return {"mlx": None, "mlx_lm": None}
    return match.groupdict()


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
        if FALLBACK_RE.search(line):
            fallback = True
    return {
        "attached": attached,
        "tasks": tasks,
        "kernels": kernels,
        "fallback": fallback,
    }


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


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new / old - 1.0) * 100.0


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def host_metadata(workspace: Path, binary: Path) -> dict[str, Any]:
    revision = command_output(["git", "rev-parse", "HEAD"], workspace)
    dirty = command_output(["git", "status", "--porcelain"], workspace)
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"], workspace),
        "macos": command_output(["sw_vers", "-productVersion"], workspace),
        "binary": str(binary),
        "binary_version": command_output([str(binary), "--version"], workspace),
        "git_revision": revision,
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def validate_run(
    *,
    mode: str,
    token_count: int,
    requested_repeats: int,
    reps: list[dict[str, float]],
    returncode: int,
    dispatch: dict[str, int | bool | None],
    allow_fallback: bool,
) -> list[str]:
    errors: list[str] = []
    if returncode != 0:
        errors.append(f"process exited with {returncode}")
    if len(reps) != requested_repeats:
        errors.append(f"expected {requested_repeats} measured reps, parsed {len(reps)}")
    for index, rep in enumerate(reps):
        missing = sorted(REQUIRED_METRICS - rep.keys())
        if missing:
            errors.append(f"rep {index} missing metrics: {', '.join(missing)}")
        generated = int(rep.get("generated_tokens", -1))
        if generated != token_count:
            errors.append(
                f"rep {index} generated {generated} tokens; expected {token_count}"
            )
    if mode == "fusekit":
        if not dispatch["attached"]:
            errors.append("FuseKit runtime did not attach")
        if dispatch["fallback"] and not allow_fallback:
            errors.append("FuseKit reported a native fallback")
    return errors


def run_one(
    *,
    workspace: Path,
    binary: Path,
    model: str,
    tokens: int,
    repeat: int,
    warmup: int,
    prompt: str,
    mode: str,
    timeout: float | None,
    allow_fallback: bool,
    log_dir: Path | None,
) -> dict[str, Any]:
    total_repeats = repeat + warmup
    command = [
        str(binary),
        "bench",
        model,
        "--tokens",
        str(tokens),
        "--repeat",
        str(total_repeats),
        "--prompt",
        prompt,
    ]
    if mode == "raw":
        command.append("--raw")

    env = os.environ.copy()
    if mode == "fusekit" and not allow_fallback:
        env["FUSEKIT_STRICT_DISPATCH"] = "1"

    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        returncode = proc.returncode
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        captured = exc.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += "\n[bench-suite] process timed out\n"
    elapsed_s = time.perf_counter() - started

    load_s, all_reps = parse_bench_output(output)
    warmup_reps = all_reps[:warmup]
    reps = all_reps[warmup : warmup + repeat]
    dispatch = parse_dispatch_info(output)
    errors = validate_run(
        mode=mode,
        token_count=tokens,
        requested_repeats=repeat,
        reps=reps,
        returncode=returncode,
        dispatch=dispatch,
        allow_fallback=allow_fallback,
    )
    if timed_out and "process timed out" not in errors:
        errors.append("process timed out")

    log_path = None
    if log_dir is not None:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        log_path = log_dir / f"{safe_model}-L{tokens}-{mode}.log"
        log_path.write_text(output)

    return {
        "record_type": "run",
        "model": model,
        "tokens": tokens,
        "mode": mode,
        "repeat": repeat,
        "warmup": warmup,
        "command": command,
        "returncode": returncode,
        "valid": not errors,
        "errors": errors,
        "elapsed_s": elapsed_s,
        "load_s": load_s,
        "summary": summarize(reps),
        "dispatch": dispatch,
        "versions": parse_versions(output),
        "warmup_reps": warmup_reps,
        "reps": reps,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "log_path": str(log_path) if log_path else None,
        "tail": output[-4000:] if errors else "",
    }


def markdown_report(metadata: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# FuseKit benchmark report",
        "",
        f"Generated: `{metadata['created_at']}`",
        "",
        "| Model | Tokens | Raw wall TPS | FuseKit wall TPS | Delta | Raw peak | FuseKit peak | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault((record["model"], record["tokens"]), {})[
            record["mode"]
        ] = record
    for (model, tokens), pair in by_pair.items():
        raw = pair.get("raw")
        fused = pair.get("fusekit")
        if raw and fused and raw["valid"] and fused["valid"]:
            raw_summary = raw["summary"]
            fused_summary = fused["summary"]
            delta = pct(
                fused_summary.get("wall_tps_avg"),
                raw_summary.get("wall_tps_avg"),
            )
            status = "ok"
            raw_tps = fmt(raw_summary.get("wall_tps_avg"))
            fused_tps = fmt(fused_summary.get("wall_tps_avg"))
            raw_peak = fmt(raw_summary.get("peak_gb_avg"))
            fused_peak = fmt(fused_summary.get("peak_gb_avg"))
            delta_text = f"{delta:+.2f}%" if delta is not None else "NA"
        else:
            status = "invalid"
            raw_tps = fused_tps = raw_peak = fused_peak = delta_text = "NA"
        safe_model = model.replace("|", "\\|")
        lines.append(
            f"| `{safe_model}` | {tokens} | {raw_tps} | "
            f"{fused_tps} | {delta_text} | {raw_peak} GB | {fused_peak} GB | {status} |"
        )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Binary: `{metadata.get('binary_version') or 'unknown'}`",
            f"- Git revision: `{metadata.get('git_revision') or 'unknown'}`",
            f"- Dirty tree: `{metadata.get('git_dirty')}`",
            f"- Host: `{metadata.get('chip') or metadata.get('machine')}`",
            f"- macOS: `{metadata.get('macos') or 'unknown'}`",
            f"- MLX: `{next((r['versions']['mlx'] for r in records if r['versions']['mlx']), 'unknown')}`",
            f"- mlx-lm: `{next((r['versions']['mlx_lm'] for r in records if r['versions']['mlx_lm']), 'unknown')}`",
            "",
            "Wall TPS is the primary metric. Each pair runs raw first, then FuseKit; "
            "warmups are excluded from the averages. A row is valid only when all "
            "requested tokens were generated and the FuseKit runtime attached without "
            "a reported fallback.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    default_workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 512])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="Repeat the word token forever: token token token",
    )
    parser.add_argument("--bin", type=Path, default=Path("target/release/fusekit"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Record rather than reject runs that report native fallback.",
    )
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    if args.repeat < 1 or args.warmup < 0 or any(n < 1 for n in args.tokens):
        parser.error("tokens/repeat must be positive and warmup must be non-negative")

    workspace = args.workspace.resolve()
    binary = args.bin if args.bin.is_absolute() else workspace / args.bin
    binary = binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary} (run cargo build --release)")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or workspace / "benchmark-results" / f"fusekit-{stamp}.jsonl"
    out_path = out_path if out_path.is_absolute() else workspace / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.with_suffix(".md")
    log_dir = out_path.with_suffix("").with_name(out_path.stem + "-logs") if args.save_logs else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "record_type": "metadata",
        "schema_version": 1,
        **host_metadata(workspace, binary),
        "settings": {
            "models": args.models,
            "tokens": args.tokens,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "prompt": args.prompt,
            "raw_first": True,
            "allow_fallback": args.allow_fallback,
            "note": args.note,
        },
    }

    records: list[dict[str, Any]] = []
    failed = False
    print(f"[bench-suite] writing {out_path}", flush=True)
    with out_path.open("w") as out_file:
        out_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for model in args.models:
            for token_count in args.tokens:
                pair: list[dict[str, Any]] = []
                for mode in ("raw", "fusekit"):
                    print(
                        "[bench-suite] START "
                        f"model={model} L={token_count} mode={mode} "
                        f"repeat={args.repeat} warmup={args.warmup}",
                        flush=True,
                    )
                    record = run_one(
                        workspace=workspace,
                        binary=binary,
                        model=model,
                        tokens=token_count,
                        repeat=args.repeat,
                        warmup=args.warmup,
                        prompt=args.prompt,
                        mode=mode,
                        timeout=args.timeout,
                        allow_fallback=args.allow_fallback,
                        log_dir=log_dir,
                    )
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    records.append(record)
                    pair.append(record)
                    summary = record["summary"]
                    if record["valid"]:
                        print(
                            "[bench-suite] DONE  "
                            f"model={model} L={token_count} mode={mode} "
                            f"wall={fmt(summary.get('wall_tps_avg'))} "
                            f"sd={fmt(summary.get('wall_tps_sd'))} "
                            f"peak={fmt(summary.get('peak_gb_avg'))} "
                            f"elapsed={fmt(record['elapsed_s'], 1)}s",
                            flush=True,
                        )
                    else:
                        failed = True
                        print(
                            "[bench-suite] INVALID "
                            f"model={model} L={token_count} mode={mode}: "
                            + "; ".join(record["errors"]),
                            flush=True,
                        )
                        if record["tail"]:
                            print(record["tail"], flush=True)
                    if args.cooldown > 0:
                        time.sleep(args.cooldown)

                raw, fused = pair
                if raw["valid"] and fused["valid"]:
                    delta = pct(
                        fused["summary"].get("wall_tps_avg"),
                        raw["summary"].get("wall_tps_avg"),
                    )
                    print(
                        "[bench-suite] DELTA "
                        f"model={model} L={token_count} wall={fmt(delta, 2)}%",
                        flush=True,
                    )

    report_path.write_text(markdown_report(metadata, records))
    print(f"[bench-suite] report {report_path}", flush=True)
    print("[bench-suite] COMPLETE", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
