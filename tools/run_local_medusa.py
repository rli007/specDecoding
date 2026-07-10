#!/usr/bin/env python
"""Run the local first-principles Medusa decoder with public Medusa heads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TextIO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders.first_principles_speculative_decoder import (
    choose_device,
    dtype_from_arg,
    memory_status,
    print_hardware_status,
    timed_operation,
)
from decoders.medusa_speculative_decoder import (
    generate_with_trace,
    load_official_medusa_heads,
    print_trace,
    linear_medusa_choices,
    small_medusa_tree_choices,
)


DEFAULT_BASE_MODEL = "lmsys/vicuna-7b-v1.3"
DEFAULT_MEDUSA_HEADS = "FasterDecoding/medusa-vicuna-7b-v1.3"
DEFAULT_PROMPT = "The professor asked for"
LONGER_SENTENCE_TOKENS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a local base model plus public Medusa heads and run the traceable decoder."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--medusa-heads", default=DEFAULT_MEDUSA_HEADS)
    parser.add_argument("--tokenizer-model", default=None, help="Defaults to --base-model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument(
        "--longer-sentence",
        action="store_true",
        help=f"Shortcut: if --max-new-tokens is left at 2, generate {LONGER_SENTENCE_TOKENS} tokens.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--no-low-cpu-mem-usage", action="store_false", dest="low_cpu_mem_usage")
    parser.add_argument("--medusa-num-heads", type=int, default=None, help="Override head count if checkpoint config is missing.")
    parser.add_argument("--medusa-num-layers", type=int, default=None, help="Override residual layers per head if config is missing.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k Medusa choices per head used by candidate buffers.")
    parser.add_argument(
        "--choice-preset",
        choices=("linear", "small-tree"),
        default="linear",
        help="linear is fastest; small-tree verifies several candidate paths for easier tree inspection.",
    )
    parser.add_argument("--show-logits", action="store_true")
    parser.add_argument("--top-k-logits", type=int, default=5)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-step-text", action="store_false", dest="step_text", help="Disable per-step partial text prints.")
    parser.add_argument("--output-txt", default=None, help="Optional text file for final output and step trace.")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.set_defaults(low_cpu_mem_usage=True, step_text=True)
    return parser.parse_args()


def model_kwargs_from_args(args: argparse.Namespace) -> dict:
    kwargs = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def medusa_choices_for(args: argparse.Namespace, num_heads: int):
    if args.choice_preset == "small-tree":
        return small_medusa_tree_choices(num_heads, args.top_k)
    return linear_medusa_choices(num_heads)


def resolve_max_new_tokens(args: argparse.Namespace) -> int:
    if args.longer_sentence and args.max_new_tokens == 2:
        return LONGER_SENTENCE_TOKENS
    return args.max_new_tokens


def append_log(log_file: TextIO | None, text: str) -> None:
    if log_file is not None:
        print(text, file=log_file, flush=True)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.base_model
    model_kwargs = model_kwargs_from_args(args)
    max_new_tokens = resolve_max_new_tokens(args)
    output_path = Path(args.output_txt).expanduser() if args.output_txt else None

    print("Local Medusa run")
    print(f"base model: {args.base_model}")
    print(f"medusa heads: {args.medusa_heads}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"dtype: {args.dtype}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"choice preset: {args.choice_preset}, top_k={args.top_k}")
    print_hardware_status(device)
    if device.type == "cpu":
        print("Note: CPU-only Vicuna 7B inference will be very slow.")

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.", flush=True)

    print(f"\nLoading base model: {args.base_model}", flush=True)
    if model_kwargs:
        print(f"Model load kwargs: {model_kwargs}", flush=True)
    with timed_operation("base model from_pretrained", torch.device("cpu"), args.heartbeat_seconds):
        target = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    print(f"Moving base model to {device} ...", flush=True)
    with timed_operation("base model to device", device, args.heartbeat_seconds):
        target = target.to(device).eval()
    target.config.output_hidden_states = True
    print(f"Base model ready on {next(target.parameters()).device}.")
    print(f"Memory after base load: {memory_status(device)}")

    print(f"\nLoading Medusa heads: {args.medusa_heads}", flush=True)
    head_dtype = None if args.dtype in {"auto", "none"} else dtype_from_arg(args.dtype)
    with timed_operation("medusa heads load", device, args.heartbeat_seconds):
        medusa_heads = load_official_medusa_heads(
            target,
            args.medusa_heads,
            device=device,
            dtype=head_dtype if isinstance(head_dtype, torch.dtype) else None,
            medusa_num_heads=args.medusa_num_heads,
            medusa_num_layers=args.medusa_num_layers,
        )
    print(f"Medusa heads ready on {next(medusa_heads.parameters()).device}.")
    print(f"Medusa heads: num_heads={medusa_heads.num_heads}")
    print(f"Memory after heads load: {memory_status(device)}")

    choices = medusa_choices_for(args, medusa_heads.num_heads)
    print(f"Medusa choice paths: {choices}")

    print("\nTokenizing prompt...", flush=True)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    print(f"prompt: {args.prompt!r}")
    print(f"prompt token ids: {inputs.input_ids[0].tolist()}")
    print(f"prompt length: {inputs.input_ids.shape[-1]}")

    log_file: TextIO | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = output_path.open("w", encoding="utf-8")
        append_log(log_file, "LOCAL MEDUSA RUN")
        append_log(log_file, f"base_model: {args.base_model}")
        append_log(log_file, f"medusa_heads: {args.medusa_heads}")
        append_log(log_file, f"prompt: {args.prompt!r}")
        append_log(log_file, f"max_new_tokens: {max_new_tokens}")
        append_log(log_file, "")

    prompt_length = inputs.input_ids.shape[-1]

    def on_step(step_trace, generated_ids) -> None:
        if not args.step_text:
            return
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
        line = (
            f"[partial {step_trace.step}] accepted={step_trace.accepted_count} "
            f"appended={step_trace.appended_tokens} new_text={new_text!r}"
        )
        print(line, flush=True)
        append_log(log_file, line)
        append_log(log_file, f"[partial {step_trace.step}] full_text={generated_text!r}")

    try:
        print("\nStarting first-principles Medusa generation...", flush=True)
        with torch.inference_mode():
            output_ids, trace_steps = generate_with_trace(
                target,
                medusa_heads,
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                medusa_choices=choices,
                top_k=args.top_k,
                top_k_logits=args.top_k_logits if args.show_logits else 0,
                progress=args.progress,
                heartbeat_seconds=args.heartbeat_seconds,
                step_callback=on_step,
            )

        print(f"\nGeneration finished after {len(trace_steps)} Medusa step(s).")
        print_trace(trace_steps, tokenizer=tokenizer, show_logits=args.show_logits)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\nFINAL")
        print(f"token ids: {output_ids[0].tolist()}")
        print(f"decoded: {decoded}")

        append_log(log_file, "")
        append_log(log_file, "FINAL")
        append_log(log_file, f"token_ids: {output_ids[0].tolist()}")
        append_log(log_file, f"decoded: {decoded}")
        if output_path is not None:
            print(f"\nSaved Medusa text trace to: {output_path}")
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
