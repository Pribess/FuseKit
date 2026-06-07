"""Compare raw oMLX engine and FuseKit engine text for fixed-seed decode."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
from pathlib import Path

from bench_omlx_engine_suite import DEFAULT_MODELS, PYTHON, PYTHONPATH, label


TEXT_RE = re.compile(r"\[engine:run:0\] text=(.*)")
TOKENS_RE = re.compile(r"\[engine:run:0\] .*generated_tokens=(\d+)")


def run_one(
    workspace: Path,
    model: str,
    mode: str,
    tokens: int,
    prompt: str,
    mtp: bool,
    seed: int | None,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("FUSEKIT_ENABLE_ENGINE_FAST", None)
    env.pop("FUSEKIT_DISABLE_ENGINE_FAST", None)
    env.pop("FUSEKIT_ENABLE_ENGINE_CHUNK", None)
    env.pop("FUSEKIT_ENGINE_CHUNK_MODE", None)
    env.pop("FUSEKIT_ENGINE_DRAIN", None)
    env.pop("FUSEKIT_ENGINE_ALLOW_LONG_GPU_CHAIN", None)
    env.pop("FK_BUNDLE_CMDS", None)
    env.pop("FK_CHAIN_ENCODERS", None)
    env.pop("FUSEKIT_PROFILE", None)
    env.pop("FUSEKIT_PROFILE_ICB", None)
    if mode == "engine_raw":
        env["FUSEKIT_DISABLE_HOOK"] = "1"
        env.pop("FUSEKIT_ENGINE_STYLE", None)
    else:
        env.pop("FUSEKIT_DISABLE_HOOK", None)
        env["FUSEKIT_ENGINE_STYLE"] = "omlx"
        env["FUSEKIT_ENABLE_ENGINE_FAST"] = "1"

    proc = subprocess.run(
        [
            PYTHON,
            "scripts/bench_omlx_engine.py",
            model,
            "--mode",
            mode,
            "--tokens",
            str(tokens),
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--prompt",
            prompt,
            "--dump-text",
        ]
        + (["--mtp"] if mtp else [])
        + (["--seed", str(seed)] if seed is not None else []),
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = None
    generated = None
    for line in proc.stdout.splitlines():
        tm = TEXT_RE.search(line)
        if tm:
            text = ast.literal_eval(tm.group(1))
        gm = TOKENS_RE.search(line)
        if gm:
            generated = int(gm.group(1))
    return {
        "rc": proc.returncode,
        "text": text,
        "generated": generated,
        "tail": proc.stdout[-3000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/Users/pribess/Documents/workspace/fusekit"),
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--prompt", default="x")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument(
        "--raw-mtp",
        action="store_true",
        help="enable MTP only for the raw baseline; defaults to --mtp",
    )
    parser.add_argument(
        "--fk-mtp",
        action="store_true",
        help="enable MTP only for the FuseKit run; defaults to --mtp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed passed to bench_omlx_engine; required for deterministic oMLX MTP",
    )
    args = parser.parse_args()

    ok_all = True
    raw_mtp = args.mtp or args.raw_mtp
    fk_mtp = args.mtp or args.fk_mtp
    for model in args.models or DEFAULT_MODELS:
        raw = run_one(
            args.workspace,
            model,
            "engine_raw",
            args.tokens,
            args.prompt,
            raw_mtp,
            args.seed,
        )
        fk = run_one(
            args.workspace,
            model,
            "engine_fusekit",
            args.tokens,
            args.prompt,
            fk_mtp,
            args.seed,
        )
        ok = (
            raw["rc"] == 0
            and fk["rc"] == 0
            and raw["generated"] == args.tokens
            and fk["generated"] == args.tokens
            and raw["text"] == fk["text"]
        )
        ok_all = ok_all and ok
        print(
            "[engine-correctness] "
            f"model={label(model)} "
            f"tokens={args.tokens} "
            f"raw_mtp={raw_mtp} "
            f"fk_mtp={fk_mtp} "
            f"seed={args.seed} "
            f"raw_generated={raw['generated']} "
            f"fk_generated={fk['generated']} "
            f"match={ok}",
            flush=True,
        )
        if not ok:
            print("[engine-correctness] raw_tail:")
            print(raw["tail"])
            print("[engine-correctness] fk_tail:")
            print(fk["tail"])
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
