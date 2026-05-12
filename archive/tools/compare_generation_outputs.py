#!/usr/bin/env python
"""Compare target greedy, HF assisted generation, and manual speculative decode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders.first_principles_speculative_decoder import PROMPT, choose_device, generate as speculative_generate


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def print_mismatch(name: str, ref: list[int], other: list[int], tokenizer: AutoTokenizer) -> None:
    idx = first_mismatch(ref, other)
    if idx is None:
        print(f"{name}: no mismatch")
        return

    ref_token = ref[idx] if idx < len(ref) else None
    other_token = other[idx] if idx < len(other) else None
    print(f"{name}: first mismatch index: {idx}")
    print(f"target greedy token: {ref_token}")
    print(f"other token: {other_token}")
    print(f"decoded prefix before mismatch: {tokenizer.decode(ref[:idx], skip_special_tokens=True)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare greedy, HF assisted, and manual speculative outputs.")
    parser.add_argument("--target-model", default="gpt2")
    parser.add_argument("--assistant-model", default="distilgpt2")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device={device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    target = AutoModelForCausalLM.from_pretrained(args.target_model).to(device).eval()
    assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model).to(device).eval()
    assistant.generation_config.num_assistant_tokens = args.draft_len

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        greedy = target.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        assisted = target.generate(
            **inputs,
            assistant_model=assistant,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    manual_ids_tensor = speculative_generate(
        target,
        inputs.input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        draft_model=assistant,
        num_assistant_tokens=args.draft_len,
    )

    greedy_ids = greedy[0].tolist()
    assisted_ids = assisted[0].tolist()
    manual_ids = manual_ids_tensor[0].tolist()

    print("\nA. Normal target greedy generation")
    print(tokenizer.decode(greedy_ids, skip_special_tokens=True))
    print(greedy_ids)

    print("\nB. Hugging Face assisted generation")
    print(tokenizer.decode(assisted_ids, skip_special_tokens=True))
    print(assisted_ids)

    print("\nC. Manual speculative decoding")
    print(tokenizer.decode(manual_ids, skip_special_tokens=True))
    print(manual_ids)

    print("\nComparisons")
    print(f"A == B: {greedy_ids == assisted_ids}")
    print(f"A == C: {greedy_ids == manual_ids}")

    if greedy_ids != assisted_ids:
        print_mismatch("A vs B", greedy_ids, assisted_ids, tokenizer)
    if greedy_ids != manual_ids:
        print_mismatch("A vs C", greedy_ids, manual_ids, tokenizer)


if __name__ == "__main__":
    main()
