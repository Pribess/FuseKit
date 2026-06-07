"""Baseline MLX-LM inference driver.

Invoked as a subprocess from `fusekit profile-inner` so that xctrace
captures the Metal activity of a real generation run (prefill + decode)
on an oMLX model. fusekit is *not* in the loop here — this measures the
unmodified MLX path, i.e. the baseline we want to beat.

Args:
    model_dir   absolute path to an oMLX model directory
    prompt      text prompt
    max_tokens  number of tokens to decode (int)
"""

import sys
import time

from mlx_lm import generate, load


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: baseline_infer.py <model_dir> <prompt> <max_tokens>", file=sys.stderr)
        return 2

    model_dir = sys.argv[1]
    prompt = sys.argv[2]
    max_tokens = int(sys.argv[3])

    t0 = time.perf_counter()
    model, tokenizer = load(model_dir)
    t_load = time.perf_counter() - t0
    print(f"[baseline] load: {t_load:.2f}s", file=sys.stderr)

    # Apply chat template if the tokenizer has one — matches what mlx-lm
    # serving does and gives realistic ops in the trace.
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        rendered = prompt

    t0 = time.perf_counter()
    # verbose=True streams tokens to stdout as they're generated and
    # prints a tok/s summary at the end. Matches what `mlx_lm.generate`
    # does from the command line.
    generate(
        model,
        tokenizer,
        prompt=rendered,
        max_tokens=max_tokens,
        verbose=True,
    )
    t_gen = time.perf_counter() - t0
    print(f"[baseline] total wall time: {t_gen:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
