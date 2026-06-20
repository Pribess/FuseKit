#!/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11
"""Stream oMLX generation text while measuring wall-clock TPS.

stdout is reserved for generated text. Progress and summary lines go to stderr
so the command can be used both interactively and in logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


OMLX_APP = Path(os.environ.get("FUSEKIT_MLX_APP", "/Applications/oMLX.app")).expanduser()
OMLX_RESOURCES = str(OMLX_APP / "Contents/Resources")
OMLX_SITE = str(
    OMLX_APP / "Contents/Python/framework-mlx-framework/lib/python3.11/site-packages"
)
OMLX_MODEL_ROOT = Path(
    os.environ.get("FUSEKIT_OMLX_DIR", Path.home() / ".omlx/models")
).expanduser()
CHILD_ENV = "FUSEKIT_WATCH_CHILD"
MTP_KERNEL_ENV_KEYS = (
    "FUSEKIT_AUTO_MTP_MN_VARIANT",
    "FUSEKIT_USE_MTP_MN_VARIANT",
    "FUSEKIT_MTP_SINGLE_BACKBONE",
    "FUSEKIT_DISABLE_MTP_MN_VARIANT",
    "FUSEKIT_DISABLE_MTP_BACKBONE",
)
MTP_REGRESSION_MODELS = [
    "corysus/Qwen3.5-4B-oQ4-mtp",
    "fcobos/Qwen3.5-9B-DeepSeek-V4-Flash-oQ4-mtp",
]
MTP_CONTROL_MODELS = [
    "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed",
    "fcobos/Qwen3.5-0.8B-oQ4-mtp",
]


def _arg_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default


def _mode_requested(argv: list[str]) -> str:
    return (_arg_value(argv, "--mode", "fusekit") or "fusekit").strip().lower()


def _arg_int(argv: list[str], name: str, default: int) -> int:
    value = _arg_value(argv, name, str(default))
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _split_path_specs(value: str) -> list[str]:
    specs = []
    for chunk in value.replace(";", ",").split(","):
        spec = chunk.strip()
        if spec:
            specs.append(spec)
    return specs


def _split_model_specs(value: str) -> list[str]:
    return _split_path_specs(value)


def _label_for_model(value: str) -> str:
    p = Path(value)
    try:
        idx = p.parts.index("models")
        return "/".join(p.parts[idx + 1 :])
    except ValueError:
        return value


def _installed_model_specs() -> list[tuple[str, str]]:
    out = []
    if not OMLX_MODEL_ROOT.is_dir():
        return out
    for owner in sorted(p for p in OMLX_MODEL_ROOT.iterdir() if p.is_dir()):
        for model in sorted(p for p in owner.iterdir() if p.is_dir()):
            if not (model / "config.json").is_file():
                continue
            label = f"{owner.name}/{model.name}"
            out.append((label, str(model)))
    return out


def _local_model_path(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    return OMLX_MODEL_ROOT / model


def _model_supports_mtp(label: str, model: str) -> bool:
    path = _local_model_path(model)
    name = f"{label} {path.name}".lower()
    if "-mtp" in name or "mtplx" in name:
        return True
    if (path / "mtp.safetensors").is_file():
        return True
    if (path / "model-mtp.safetensors").is_file():
        return True
    if (path / "mtplx_runtime.json").is_file():
        return True
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key in (
        "mtp",
        "mtp_heads",
        "mtp_depth_max",
        "num_nextn_predict_layers",
        "nextn_predict_layers",
    ):
        value = config.get(key)
        if value:
            return True
    return False


def _installed_mtp_model_specs() -> list[tuple[str, str]]:
    return [
        (label, model)
        for label, model in _installed_model_specs()
        if _model_supports_mtp(label, model)
    ]


def _models_arg_specs(argv: list[str]) -> list[tuple[str, str]]:
    specs = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        values: list[str] | None = None
        if arg == "--models":
            if i + 1 < len(argv):
                values = _split_model_specs(argv[i + 1])
                skip = True
        elif arg.startswith("--models="):
            values = _split_model_specs(arg.split("=", 1)[1])
        if values is None:
            continue
        for value in values:
            value_lower = value.strip().lower()
            if value_lower == "all":
                specs.extend(_installed_model_specs())
            elif value_lower == "mtp":
                specs.extend(_installed_mtp_model_specs())
            elif value_lower in ("mtp-regressions", "mtp-minus", "regressions"):
                specs.extend((m, str(_local_model_path(m))) for m in MTP_REGRESSION_MODELS)
            elif value_lower in ("mtp-controls", "controls"):
                specs.extend((m, str(_local_model_path(m))) for m in MTP_CONTROL_MODELS)
            elif value_lower in ("mtp-check", "check"):
                ordered = MTP_REGRESSION_MODELS + MTP_CONTROL_MODELS
                specs.extend((m, str(_local_model_path(m))) for m in ordered)
            else:
                model = value.strip()
                specs.append((_label_for_model(model), model))
    return specs


def _path_specs(argv: list[str]) -> list[str]:
    specs = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg == "--path":
            if i + 1 < len(argv):
                specs.append(argv[i + 1])
                skip = True
            continue
        if arg.startswith("--path="):
            specs.append(arg.split("=", 1)[1])
            continue
        if arg == "--paths":
            if i + 1 < len(argv):
                specs.extend(_split_path_specs(argv[i + 1]))
                skip = True
            continue
        if arg.startswith("--paths="):
            specs.extend(_split_path_specs(arg.split("=", 1)[1]))
            continue
    return specs


def _mtp_kernel_requested(argv: list[str]) -> str | None:
    value = _arg_value(argv, "--mtp-kernel", None)
    if value is None:
        return None
    value = value.strip().lower().replace("_", "-")
    if value in ("", "default"):
        return None
    if value not in ("native", "split", "mn", "auto"):
        raise ValueError("--mtp-kernel must be native, split, mn, auto, or default")
    return value


def _apply_mtp_kernel_env(env: dict[str, str], kernel: str | None) -> None:
    if kernel is None:
        return
    for key in MTP_KERNEL_ENV_KEYS:
        env.pop(key, None)
    if kernel == "native":
        env["FUSEKIT_DISABLE_MTP_MN_VARIANT"] = "1"
        env["FUSEKIT_DISABLE_MTP_BACKBONE"] = "1"
    elif kernel == "split":
        env["FUSEKIT_MTP_SINGLE_BACKBONE"] = "1"
        env["FUSEKIT_DISABLE_MTP_MN_VARIANT"] = "1"
    elif kernel == "mn":
        env["FUSEKIT_USE_MTP_MN_VARIANT"] = "1"
        env["FUSEKIT_DISABLE_MTP_BACKBONE"] = "1"
    elif kernel == "auto":
        env["FUSEKIT_AUTO_MTP_MN_VARIANT"] = "1"


def _parse_path_spec(spec: str) -> tuple[str, str, bool, str | None]:
    label = ""
    body = spec.strip()
    if "=" in body:
        label, body = body.split("=", 1)
        label = label.strip()
    body = body.strip().lower().replace("_", "-")
    mtp = False
    mtp_kernel: str | None = None
    for suffix, kernel in (
        ("-mtp-native", "native"),
        ("-mtp-split", "split"),
        ("-mtp-mn", "mn"),
        ("-mtp-auto", "auto"),
        ("-mtp", "native"),
        (":mtp", "native"),
    ):
        if body.endswith(suffix):
            mtp = True
            mtp_kernel = kernel
            body = body[: -len(suffix)]
            break
    if body not in ("raw", "raw-shim", "fusekit"):
        raise ValueError(
            f"--path must be raw, raw-shim, fusekit, raw-mtp, raw-shim-mtp, "
            f"fusekit-mtp-native, fusekit-mtp-split, fusekit-mtp-mn, "
            f"fusekit-mtp-auto, or label=path; got {spec!r}"
        )
    if mtp_kernel is not None and body != "fusekit":
        mtp_kernel = None
    if not label:
        if mtp and body == "fusekit":
            label = f"{body}-mtp-{mtp_kernel or 'native'}"
        else:
            label = f"{body}{'-mtp' if mtp else ''}"
    return label, body, mtp, mtp_kernel


def _remove_parent_path_args(argv: list[str]) -> list[str]:
    out = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg == "--path":
            skip = True
            continue
        if arg.startswith("--path="):
            continue
        if arg == "--paths":
            skip = True
            continue
        if arg.startswith("--paths="):
            continue
        out.append(arg)
    return out


def _remove_parent_model_args(argv: list[str]) -> list[str]:
    out = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "--models":
            skip = True
            continue
        if arg.startswith("--models="):
            continue
        out.append(arg)
    return out


def _replace_option(argv: list[str], name: str, value: str) -> list[str]:
    out = []
    skip = False
    replaced = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == name:
            out.extend([name, value])
            skip = True
            replaced = True
            continue
        if arg.startswith(name + "="):
            out.append(f"{name}={value}")
            replaced = True
            continue
        out.append(arg)
    if not replaced:
        out.extend([name, value])
    return out


def _set_flag(argv: list[str], name: str, enabled: bool) -> list[str]:
    out = [arg for arg in argv if arg != name]
    if enabled:
        out.append(name)
    return out


def _set_model_option(argv: list[str], model: str) -> list[str]:
    return _replace_option(argv, "--model", model)


def _parse_kv_line(line: str) -> dict[str, str]:
    out = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = value
    return out


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _print_compare_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    baseline_by_model: dict[str, float] = {}
    for row in rows:
        model = str(row.get("model", ""))
        tps = row.get("wall_tps_avg")
        if model not in baseline_by_model and isinstance(tps, float):
            baseline_by_model[model] = float(tps)
    table_rows = []
    for row in rows:
        model = str(row.get("model", ""))
        tps = row.get("wall_tps_avg")
        delta = None
        baseline_tps = baseline_by_model.get(model)
        if isinstance(tps, float) and baseline_tps:
            delta = (tps / baseline_tps - 1.0) * 100.0
        table_rows.append(
            [
                model,
                str(row.get("path", "")),
                str(row.get("mode", "")),
                "yes" if row.get("mtp") else "no",
                str(row.get("status", "ok")),
                _fmt_float(tps if isinstance(tps, float) else None, 3),
                _fmt_float(row.get("wall_tps_stdev") if isinstance(row.get("wall_tps_stdev"), float) else None, 3),
                _fmt_float(delta, 1) + "%" if delta is not None else "NA",
                _fmt_float(row.get("wall_s_avg") if isinstance(row.get("wall_s_avg"), float) else None, 6),
                _fmt_float(row.get("first_token_s_avg") if isinstance(row.get("first_token_s_avg"), float) else None, 6),
                _fmt_float(row.get("peak_gb_avg") if isinstance(row.get("peak_gb_avg"), float) else None, 3),
            ]
        )
    headers = [
        "model",
        "path",
        "mode",
        "mtp",
        "status",
        "tps_avg",
        "tps_sd",
        "delta_vs_first",
        "wall_s_avg",
        "first_s_avg",
        "peak_gb",
    ]
    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\n[watch:compare]", file=sys.stderr, flush=True)
    print(line(headers), file=sys.stderr, flush=True)
    print(line(["-" * w for w in widths]), file=sys.stderr, flush=True)
    for row in table_rows:
        print(line(row), file=sys.stderr, flush=True)


def _run_parent_compare(argv: list[str]) -> int | None:
    if os.environ.get(CHILD_ENV):
        return None
    specs = _path_specs(argv)
    model_specs = _models_arg_specs(argv)
    if not specs and not model_specs:
        return None
    if not specs:
        mode = _mode_requested(argv)
        mtp_suffix = "-mtp" if "--mtp" in argv else ""
        specs = [f"{mode}{mtp_suffix}"]
    try:
        paths = [_parse_path_spec(spec) for spec in specs]
    except ValueError as e:
        print(f"[watch:error] {e}", file=sys.stderr)
        return 2

    parent_repeat = max(1, _arg_int(argv, "--repeat", 1))
    base_argv = _replace_option(
        _remove_parent_model_args(_remove_parent_path_args(argv)),
        "--repeat",
        "1",
    )
    models = model_specs or [(None, None)]
    rows: list[dict[str, object]] = []
    total = len(models) * len(paths) * parent_repeat
    idx = 0
    for model_label, model_arg in models:
        for label, mode, mtp, mtp_kernel in paths:
            if model_label is not None and model_arg is not None and mtp:
                if not _model_supports_mtp(model_label, model_arg):
                    idx += parent_repeat
                    print(
                        f"\n[watch:skip] {idx}/{total} model={model_label} "
                        f"path={label} reason=no_mtp_weights_or_metadata",
                        file=sys.stderr,
                        flush=True,
                    )
                    rows.append(
                        {
                            "model": model_label,
                            "path": label,
                            "mode": mode,
                            "mtp": mtp,
                            "status": "skip:no_mtp",
                        }
                    )
                    continue
            row: dict[str, object] = {
                "model": model_label or "",
                "path": label,
                "mode": mode,
                "mtp": mtp,
                "status": "running",
            }
            values: dict[str, list[float]] = {
                "wall_tps_avg": [],
                "wall_s_avg": [],
                "first_token_s_avg": [],
                "peak_gb_avg": [],
            }
            status = "ok"
            for rep_idx in range(parent_repeat):
                idx += 1
                child_argv = _replace_option(base_argv, "--mode", mode)
                child_argv = _set_flag(child_argv, "--mtp", mtp)
                if mtp_kernel is not None:
                    child_argv = _replace_option(child_argv, "--mtp-kernel", mtp_kernel)
                if model_arg is not None:
                    child_argv = _set_model_option(child_argv, model_arg)
                env = os.environ.copy()
                env[CHILD_ENV] = "1"
                env["FUSEKIT_WATCH_PATH_LABEL"] = (
                    label if parent_repeat == 1 else f"{label}#{rep_idx + 1}"
                )
                if model_label is not None:
                    env["FUSEKIT_WATCH_MODEL_LABEL"] = model_label
                print(
                    f"\n[watch:path] {idx}/{total} "
                    f"model={model_label or '<positional>'} path={label} "
                    f"rep={rep_idx + 1}/{parent_repeat} mode={mode} "
                    f"mtp={mtp} argv={' '.join(child_argv)}",
                    file=sys.stderr,
                    flush=True,
                )
                proc = subprocess.Popen(
                    [sys.executable, sys.argv[0], *child_argv],
                    cwd=os.getcwd(),
                    env=env,
                    stdout=None,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                assert proc.stderr is not None
                for line in proc.stderr:
                    print(line, end="", file=sys.stderr, flush=True)
                    if line.startswith("[watch:avg]"):
                        parsed = _parse_kv_line(line)
                        if not row["model"]:
                            row["model"] = parsed.get("model", "")
                        for key in values:
                            if key in parsed:
                                try:
                                    values[key].append(float(parsed[key]))
                                except ValueError:
                                    pass
                rc = proc.wait()
                if rc != 0:
                    status = f"rc={rc}"
            if values["wall_tps_avg"]:
                row["wall_tps_avg"] = statistics.mean(values["wall_tps_avg"])
                row["wall_tps_stdev"] = (
                    statistics.stdev(values["wall_tps_avg"])
                    if len(values["wall_tps_avg"]) > 1
                    else 0.0
                )
            for key in ("wall_s_avg", "first_token_s_avg", "peak_gb_avg"):
                if values[key]:
                    row[key] = statistics.mean(values[key])
            row["status"] = status
            rows.append(row)

    _print_compare_table(rows)
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


_parent_rc = _run_parent_compare(sys.argv[1:])
if _parent_rc is not None:
    raise SystemExit(_parent_rc)


def _ensure_startup_env() -> None:
    mode = _mode_requested(sys.argv[1:])
    try:
        mtp_kernel = _mtp_kernel_requested(sys.argv[1:])
    except ValueError as e:
        print(f"[watch:error] {e}", file=sys.stderr)
        raise SystemExit(2)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{OMLX_RESOURCES}:{OMLX_SITE}"
    env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/fusekit-pycache")
    if mode == "raw":
        env["FUSEKIT_DISABLE_HOOK"] = "1"
        env["FUSEKIT_DISABLE_DISPATCH"] = "1"
        env["FUSEKIT_DISABLE_FAST"] = "1"
        env["FUSEKIT_DISABLE_MTP_SAMPLER_ACCEPT_CACHE"] = "1"
        env["FUSEKIT_DISABLE_MTP_DROP_RESPONSE_LOGPROBS"] = "1"
        env.pop("FUSEKIT_ENGINE_STYLE", None)
        env.pop("FUSEKIT_ENABLE_ENGINE_FAST", None)
    elif mode == "raw-shim":
        env.pop("FUSEKIT_DISABLE_HOOK", None)
        env["FUSEKIT_DISABLE_DISPATCH"] = "1"
        env["FUSEKIT_DISABLE_FAST"] = "1"
        env["FUSEKIT_DISABLE_MTP_SAMPLER_ACCEPT_CACHE"] = "1"
        env["FUSEKIT_DISABLE_MTP_DROP_RESPONSE_LOGPROBS"] = "1"
        env["FUSEKIT_ENGINE_STYLE"] = "raw"
        env.pop("FUSEKIT_ENABLE_ENGINE_FAST", None)
    else:
        env.pop("FUSEKIT_DISABLE_HOOK", None)
        env.pop("FUSEKIT_DISABLE_DISPATCH", None)
        env.pop("FUSEKIT_DISABLE_FAST", None)
        env.pop("FUSEKIT_DISABLE_MTP_SAMPLER_ACCEPT_CACHE", None)
        env.pop("FUSEKIT_DISABLE_MTP_DROP_RESPONSE_LOGPROBS", None)
        env["FUSEKIT_ENGINE_STYLE"] = "omlx"
        env["FUSEKIT_ENABLE_ENGINE_FAST"] = "1"
        _apply_mtp_kernel_env(env, mtp_kernel)

    changed = any(os.environ.get(k) != v for k, v in env.items() if k in {
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "FUSEKIT_DISABLE_HOOK",
        "FUSEKIT_DISABLE_DISPATCH",
        "FUSEKIT_DISABLE_FAST",
        "FUSEKIT_DISABLE_MTP_SAMPLER_ACCEPT_CACHE",
        "FUSEKIT_DISABLE_MTP_DROP_RESPONSE_LOGPROBS",
        "FUSEKIT_ENGINE_STYLE",
        "FUSEKIT_ENABLE_ENGINE_FAST",
        *MTP_KERNEL_ENV_KEYS,
    })
    for k in (
        "FUSEKIT_DISABLE_HOOK",
        "FUSEKIT_DISABLE_DISPATCH",
        "FUSEKIT_DISABLE_FAST",
        "FUSEKIT_DISABLE_MTP_SAMPLER_ACCEPT_CACHE",
        "FUSEKIT_DISABLE_MTP_DROP_RESPONSE_LOGPROBS",
        "FUSEKIT_ENGINE_STYLE",
        "FUSEKIT_ENABLE_ENGINE_FAST",
        *MTP_KERNEL_ENV_KEYS,
    ):
        if k in os.environ and k not in env:
            changed = True
    if changed:
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_ensure_startup_env()

for path in (OMLX_RESOURCES, OMLX_SITE):
    if path not in sys.path:
        sys.path.insert(0, path)

import mlx.core as mx

from omlx.engine.batched import BatchedEngine
from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig


_MTP_STATS_TRACE_INSTALLED = False


def _install_native_mtp_stats_trace(path_label: str, model_label: str, run_idx: int) -> None:
    if not os.environ.get("FUSEKIT_WATCH_MTP_STATS"):
        return
    global _MTP_STATS_TRACE_INSTALLED
    try:
        import omlx.patches.mlx_lm_mtp.batch_generator as bg
    except Exception as e:
        print(f"[watch:mtp_stats_error] import_failed={e}", file=sys.stderr, flush=True)
        return

    setattr(bg, "_fusekit_watch_path_label", path_label)
    setattr(bg, "_fusekit_watch_model_label", model_label)
    setattr(bg, "_fusekit_watch_run_idx", int(run_idx))
    if _MTP_STATS_TRACE_INSTALLED:
        return

    original = getattr(bg, "_log_mtp_stats", None)
    if original is None:
        print("[watch:mtp_stats_error] missing_log_function=1", file=sys.stderr, flush=True)
        return

    def traced(uid, stats, finish_reason):
        cycles = int(getattr(stats, "cycles", 0) or 0)
        accepts = int(getattr(stats, "accepts", 0) or 0)
        rejects = int(getattr(stats, "rejects", 0) or 0)
        rate = (100.0 * accepts / cycles) if cycles else 0.0
        total_emits = (
            int(getattr(stats, "init_emits", 0) or 0)
            + int(getattr(stats, "draft_emits", 0) or 0)
            + int(getattr(stats, "bonus_emits", 0) or 0)
            + int(getattr(stats, "verify_emits", 0) or 0)
        )
        print(
            "[watch:mtp] "
            f"path={getattr(bg, '_fusekit_watch_path_label', '')} "
            f"model={getattr(bg, '_fusekit_watch_model_label', '')} "
            f"run={getattr(bg, '_fusekit_watch_run_idx', -1)} "
            f"finish={finish_reason} uid={uid} tokens={total_emits} "
            f"cycles={cycles} accepts={accepts} rejects={rejects} "
            f"accept_rate={rate:.1f} "
            f"emits_init={int(getattr(stats, 'init_emits', 0) or 0)} "
            f"emits_draft={int(getattr(stats, 'draft_emits', 0) or 0)} "
            f"emits_bonus={int(getattr(stats, 'bonus_emits', 0) or 0)} "
            f"emits_verify={int(getattr(stats, 'verify_emits', 0) or 0)} "
            f"backbone_ms={float(getattr(stats, 'backbone_ms', 0.0) or 0.0):.1f} "
            f"mtp_ms={float(getattr(stats, 'mtp_head_ms', 0.0) or 0.0):.1f} "
            f"sample_ms={float(getattr(stats, 'sample_ms', 0.0) or 0.0):.1f} "
            f"cache_ms={float(getattr(stats, 'cache_ops_ms', 0.0) or 0.0):.1f}",
            file=sys.stderr,
            flush=True,
        )
        return original(uid, stats, finish_reason)

    bg._log_mtp_stats = traced
    _MTP_STATS_TRACE_INSTALLED = True


def _model_arg(model: str) -> str:
    local = OMLX_MODEL_ROOT / model
    if local.is_dir():
        return str(local)
    return model


def _model_label(model: str) -> str:
    p = Path(model)
    try:
        idx = p.parts.index("models")
        return "/".join(p.parts[idx + 1 :])
    except ValueError:
        return model


def _set_no_eos(engine: BatchedEngine) -> None:
    tokenizer = engine.tokenizer
    for attr, value in (("eos_token_ids", set()), ("eos_token_id", None)):
        try:
            setattr(tokenizer, attr, value)
        except Exception:
            pass
    try:
        scheduler = engine._engine.engine.scheduler
        scheduler._generation_config_eos = set()
        scheduler._output_parser_factory = None
        scheduler._get_stop_tokens = lambda: set()
    except Exception:
        pass


def _response_text(resp) -> str:
    for name in ("text", "output_text"):
        value = getattr(resp, name, None)
        if isinstance(value, str):
            return value
    return ""


def _path_label(args: argparse.Namespace) -> str:
    return os.environ.get(
        "FUSEKIT_WATCH_PATH_LABEL",
        f"{args.mode}{'-mtp' if args.mtp else ''}",
    )


def _active_model_label(model_arg: str) -> str:
    return os.environ.get("FUSEKIT_WATCH_MODEL_LABEL", _model_label(model_arg))


async def run_once(
    args: argparse.Namespace,
    run_idx: int,
    *,
    engine: BatchedEngine | None = None,
    load_s: float | None = None,
) -> dict[str, float | int]:
    prompt = args.prompt
    if args.prompt_repeat > 1:
        prompt = ((args.prompt.rstrip() + " ") * args.prompt_repeat).strip()

    model_arg = _model_arg(args.model)
    owns_engine = engine is None
    if engine is None:
        engine = BatchedEngine(
            model_name=model_arg,
            scheduler_config=SchedulerConfig(max_num_seqs=1),
            stream_interval=1,
            model_settings=ModelSettings(mtp_enabled=True) if args.mtp else None,
        )
        load_t0 = time.perf_counter()
        await engine.start()
        mx.synchronize()
        load_s = time.perf_counter() - load_t0
        if not args.allow_eos:
            _set_no_eos(engine)
    else:
        load_s = float(load_s or 0.0)
    path_label = _path_label(args)
    model_label = _active_model_label(model_arg)

    print(
        f"\n[watch] path={path_label} model={model_label} mode={args.mode} "
        f"mtp={args.mtp} tokens={args.tokens} run={run_idx} load_s={load_s:.3f}",
        file=sys.stderr,
        flush=True,
    )
    _install_native_mtp_stats_trace(path_label, model_label, run_idx)

    try:
        mx.reset_peak_memory()
    except Exception:
        pass
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    started = time.perf_counter()
    first_token_s: float | None = None
    last = None
    prev_text = ""
    last_status_tokens = 0
    try:
        async for last in engine.stream_generate(
            prompt,
            max_tokens=args.tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            stop=[],
            seed=args.seed,
        ):
            now = time.perf_counter()
            completion = int(getattr(last, "completion_tokens", 0) or 0)
            if completion > 0 and first_token_s is None:
                first_token_s = now - started

            text = _response_text(last)
            if text.startswith(prev_text):
                delta = text[len(prev_text) :]
            else:
                delta = text
            if delta:
                print(delta, end="", flush=True)
                prev_text = text

            if args.status_every and completion - last_status_tokens >= args.status_every:
                elapsed = max(now - started, 1e-9)
                print(
                    f"\n[watch:status] path={path_label} model={model_label} "
                    f"mode={args.mode} mtp={args.mtp} run={run_idx} tokens={completion} "
                    f"wall_s={elapsed:.3f} wall_tps={completion / elapsed:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
                last_status_tokens = completion
        mx.synchronize()
    finally:
        if owns_engine:
            await engine.stop()

    wall_s = time.perf_counter() - started
    generated = int(getattr(last, "completion_tokens", 0) or 0)
    prompt_tokens = int(getattr(last, "prompt_tokens", 0) or 0)
    peak_gb = 0.0
    try:
        peak_gb = float(mx.get_peak_memory()) / 1e9
    except Exception:
        pass
    wall_tps = generated / wall_s if wall_s and generated else 0.0
    print("", flush=True)
    print(
        f"[watch:run] path={path_label} model={model_label} mode={args.mode} "
        f"mtp={args.mtp} run={run_idx} prompt_tokens={prompt_tokens} "
        f"generated_tokens={generated} wall_s={wall_s:.6f} "
        f"wall_tps={wall_tps:.3f} first_token_s={(first_token_s or 0.0):.6f} "
        f"peak_gb={peak_gb:.3f}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "generated": generated,
        "wall_s": wall_s,
        "wall_tps": wall_tps,
        "first_token_s": first_token_s or 0.0,
        "peak_gb": peak_gb,
    }


async def run(args: argparse.Namespace) -> int:
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    rows = []
    total = args.warmup + args.repeat
    path_label = _path_label(args)
    model_label = _active_model_label(_model_arg(args.model))
    model_arg = _model_arg(args.model)
    engine = BatchedEngine(
        model_name=model_arg,
        scheduler_config=SchedulerConfig(max_num_seqs=1),
        stream_interval=1,
        model_settings=ModelSettings(mtp_enabled=True) if args.mtp else None,
    )
    load_t0 = time.perf_counter()
    await engine.start()
    mx.synchronize()
    load_s = time.perf_counter() - load_t0
    if not args.allow_eos:
        _set_no_eos(engine)
    try:
        for i in range(total):
            kind = "warmup" if i < args.warmup else "run"
            print(
                f"\n[watch:{kind}] path={path_label} "
                f"index={i if kind == 'warmup' else i - args.warmup}",
                file=sys.stderr,
            )
            row = await run_once(
                args,
                i,
                engine=engine,
                load_s=load_s if i == 0 else 0.0,
            )
            if kind == "run":
                rows.append(row)
    finally:
        await engine.stop()

    if rows:
        tps = [float(r["wall_tps"]) for r in rows]
        wall = [float(r["wall_s"]) for r in rows]
        first = [float(r["first_token_s"]) for r in rows]
        peak = [float(r["peak_gb"]) for r in rows]
        print(
            f"[watch:avg] model={model_label} path={path_label} "
            f"mode={args.mode} mtp={args.mtp} "
            f"repeat={len(rows)} "
            f"wall_tps_avg={statistics.mean(tps):.3f} "
            f"wall_tps_stdev={(statistics.stdev(tps) if len(tps) > 1 else 0.0):.3f} "
            f"wall_s_avg={statistics.mean(wall):.6f} "
            f"first_token_s_avg={statistics.mean(first):.6f} "
            f"peak_gb_avg={statistics.mean(peak):.3f}",
            file=sys.stderr,
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?")
    parser.add_argument("--model", dest="model_option")
    parser.add_argument(
        "--models",
        help=(
            "comma-separated models for parent compare mode; use all to scan "
            "the configured oMLX model root, mtp to scan only MTP-looking models, "
            "mtp-regressions for the two negative MTP models, or mtp-check "
            "for regressions plus controls"
        ),
    )
    parser.add_argument("--mode", choices=("raw", "raw-shim", "fusekit"), default="fusekit")
    parser.add_argument(
        "--path",
        action="append",
        help=(
            "run one or more paths sequentially and compare; examples: "
            "--path raw --path fusekit --path mtp=fusekit-mtp"
        ),
    )
    parser.add_argument(
        "--paths",
        help=(
            "comma-separated paths to run sequentially and compare; "
            "example: --paths raw,fusekit,mtp=fusekit-mtp"
        ),
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="Repeat the word token forever: token token token",
    )
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument(
        "--mtp-kernel",
        choices=("default", "native", "split", "mn", "auto"),
        default="default",
        help=(
            "FuseKit MTP dispatch mode: native keeps oMLX MTP model calls, "
            "split uses custom M=1 kernels twice, mn uses compiled M=N variant, "
            "auto enables runtime auto M=N gating"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--allow-eos", action="store_true")
    parser.add_argument("--status-every", type=int, default=32)
    args = parser.parse_args()
    args.model = args.model_option or args.model
    if not args.model:
        parser.error("model is required unless parent compare mode supplies --models")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
