#!/usr/bin/env python
"""Long-running speculative decoding session for larger Llama-family models.

Run this file when model loading is the expensive part. It loads target,
assistant, and tokenizer once, then reuses them for every prompt you type.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders.first_principles_speculative_decoder import (
    choose_device,
    effective_max_new_tokens,
    load_models,
    logits_output_path,
    run_prompt,
)


DEFAULT_TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_ASSISTANT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_LLAMA_MAX_NEW_TOKENS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load larger Llama models once and reuse them for many prompts.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer to use. Defaults to --target-model.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_LLAMA_MAX_NEW_TOKENS)
    parser.add_argument("--paragraph", action="store_true", help="Shortcut for a longer 120-token generation.")
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--no-low-cpu-mem-usage", action="store_false", dest="low_cpu_mem_usage")
    parser.add_argument("--show-logits", action="store_true")
    parser.add_argument("--top-k-logits", type=int, default=5)
    parser.add_argument("--logits-out", default=None)
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable live generation progress prints.")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0, help="Print a heartbeat while a model forward is still running.")
    parser.set_defaults(progress=True, low_cpu_mem_usage=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.paragraph and args.max_new_tokens == DEFAULT_LLAMA_MAX_NEW_TOKENS:
        max_new_tokens = 120
    else:
        max_new_tokens = effective_max_new_tokens(args)

    print("Llama speculative decoding session")
    print(f"target model: {args.target_model}")
    print(f"assistant model: {args.assistant_model}")
    print(f"dtype: {args.dtype}")
    print(f"progress: {args.progress}")
    if device.type == "cpu":
        print("Note: CPU-only 8B inference can be very slow and memory-heavy.")

    loaded = load_models(args, device)

    print("\nModels are loaded. Type prompts; type quit/exit or an empty line to stop.")
    print(f"max_new_tokens={max_new_tokens}, draft_len={args.draft_len}, show_logits={args.show_logits}")
    run_index = 1
    while True:
        try:
            prompt = input("\nprompt> ")
        except EOFError:
            print()
            break

        if prompt.strip().lower() in {"", "q", "quit", "exit"}:
            break

        with torch.inference_mode():
            run_prompt(
                args,
                loaded,
                prompt,
                max_new_tokens=max_new_tokens,
                logits_out=logits_output_path(args.logits_out, run_index),
            )
        run_index += 1


if __name__ == "__main__":
    main()
