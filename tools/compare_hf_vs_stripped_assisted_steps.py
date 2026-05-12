#!/usr/bin/env python
"""Compare HF assisted generation against the stripped-down decoder step by step."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders.stripped_down_llama_assisted_decoder import (  # noqa: E402
    AssistedStepTrace,
    DEFAULT_ASSISTANT_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_ASSISTANT_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TARGET_MODEL,
    assisted_generate,
    choose_device,
    dtype_from_arg,
)


@dataclass
class HfStepTrace:
    step: int
    prefix_length: int
    assistant_budget: float
    draft_tokens: list[int]
    target_selected_tokens: list[int] = field(default_factory=list)
    accepted_assistant_tokens: int | None = None
    appended_tokens: list[int] = field(default_factory=list)
    next_assistant_budget: float | None = None
    output_length: int | None = None


@contextlib.contextmanager
def capture_hf_assisted_steps(records: list[HfStepTrace]):
    """Patch the public candidate-generator hooks that expose each HF step."""
    originals: list[tuple[Any, str, Callable[..., Any]]] = []
    active_records: dict[int, HfStepTrace] = {}

    def replace(obj: Any, name: str, wrapper: Callable[..., Any]) -> None:
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, wrapper)

    original_get_candidates = AssistedCandidateGenerator.get_candidates
    original_update_candidate_strategy = AssistedCandidateGenerator.update_candidate_strategy

    def traced_get_candidates(self: AssistedCandidateGenerator, input_ids: torch.LongTensor):
        candidate_input_ids, candidate_logits = original_get_candidates(self, input_ids)
        prefix_length = input_ids.shape[-1]
        draft_tokens = candidate_input_ids[:, prefix_length:].detach().cpu()[0].tolist()
        record = HfStepTrace(
            step=len(records) + 1,
            prefix_length=prefix_length,
            assistant_budget=float(len(draft_tokens)),
            draft_tokens=draft_tokens,
        )
        records.append(record)
        active_records[id(self)] = record
        return candidate_input_ids, candidate_logits

    def traced_update_candidate_strategy(
        self: AssistedCandidateGenerator,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        num_matches: int,
    ):
        record = active_records.get(id(self), records[-1] if records else None)
        if record is not None:
            record.accepted_assistant_tokens = int(num_matches)
            record.output_length = input_ids.shape[-1]
            record.appended_tokens = input_ids[:, record.prefix_length :].detach().cpu()[0].tolist()
            if scores is not None:
                record.target_selected_tokens = scores.argmax(dim=-1).detach().cpu()[0].tolist()

        result = original_update_candidate_strategy(self, input_ids, scores, num_matches)

        if record is not None:
            record.next_assistant_budget = float(getattr(self, "num_assistant_tokens", 0))
        return result

    replace(AssistedCandidateGenerator, "get_candidates", traced_get_candidates)
    replace(AssistedCandidateGenerator, "update_candidate_strategy", traced_update_candidate_strategy)

    try:
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HF assisted decoding against the stripped decoder.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer to use. Defaults to --target-model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-assistant-tokens", type=int, default=DEFAULT_NUM_ASSISTANT_TOKENS)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    return parser.parse_args()


def make_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True
    return model_kwargs


def configure_assistant_for_comparison(assistant: torch.nn.Module, num_assistant_tokens: int) -> None:
    assistant.generation_config.num_assistant_tokens = num_assistant_tokens
    assistant.generation_config.num_assistant_tokens_schedule = "heuristic"
    assistant.generation_config.assistant_confidence_threshold = 0.0


def decode_token_list(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def fmt_budget(value: float | None) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def print_step_pair(
    tokenizer: AutoTokenizer,
    hf_step: HfStepTrace | None,
    our_step: AssistedStepTrace | None,
) -> None:
    step_number = hf_step.step if hf_step is not None else our_step.step
    print(f"\nSTEP {step_number}")

    if hf_step is None:
        print("HF:   <no step>")
    else:
        print(
            "HF:   "
            f"prefix={hf_step.prefix_length}, "
            f"budget={fmt_budget(hf_step.assistant_budget)} -> {fmt_budget(hf_step.next_assistant_budget)}, "
            f"accepted={hf_step.accepted_assistant_tokens}, "
            f"output_len={hf_step.output_length}"
        )
        print(f"      draft:   {hf_step.draft_tokens} {decode_token_list(tokenizer, hf_step.draft_tokens)!r}")
        print(
            f"      target:  {hf_step.target_selected_tokens} "
            f"{decode_token_list(tokenizer, hf_step.target_selected_tokens)!r}"
        )
        print(f"      append:  {hf_step.appended_tokens} {decode_token_list(tokenizer, hf_step.appended_tokens)!r}")

    if our_step is None:
        print("OURS: <no step>")
    else:
        print(
            "OURS: "
            f"prefix={our_step.prefix_length}, "
            f"budget={fmt_budget(our_step.assistant_budget)} -> {fmt_budget(our_step.next_assistant_budget)}, "
            f"accepted={our_step.accepted_assistant_tokens}, "
            f"output_len={our_step.output_length}"
        )
        print(f"      draft:   {our_step.draft_tokens} {decode_token_list(tokenizer, our_step.draft_tokens)!r}")
        print(
            f"      target:  {our_step.target_selected_tokens} "
            f"{decode_token_list(tokenizer, our_step.target_selected_tokens)!r}"
        )
        print(f"      append:  {our_step.appended_tokens} {decode_token_list(tokenizer, our_step.appended_tokens)!r}")

    if hf_step is not None and our_step is not None:
        print(
            "MATCH "
            f"draft={hf_step.draft_tokens == our_step.draft_tokens}, "
            f"target={hf_step.target_selected_tokens == our_step.target_selected_tokens}, "
            f"accepted={hf_step.accepted_assistant_tokens == our_step.accepted_assistant_tokens}, "
            f"append={hf_step.appended_tokens == our_step.appended_tokens}"
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.target_model
    model_kwargs = make_model_kwargs(args)

    print(f"Using device={device}")
    print(f"target model: {args.target_model}")
    print(f"assistant model: {args.assistant_model}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs).to(device).eval()
    assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model, **model_kwargs).to(device).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    configure_assistant_for_comparison(assistant, args.num_assistant_tokens)
    hf_steps: list[HfStepTrace] = []
    with capture_hf_assisted_steps(hf_steps):
        with torch.inference_mode():
            hf_output = target.generate(
                **inputs,
                assistant_model=assistant,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    configure_assistant_for_comparison(assistant, args.num_assistant_tokens)
    our_steps: list[AssistedStepTrace] = []
    with torch.inference_mode():
        our_output = assisted_generate(
            target,
            assistant,
            inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            num_assistant_tokens=args.num_assistant_tokens,
            verbose=False,
            trace_steps=our_steps,
        )

    print("\nFINAL OUTPUTS")
    print(f"HF ids:   {hf_output[0].tolist()}")
    print(f"OURS ids: {our_output[0].tolist()}")
    print(f"equal:    {hf_output[0].tolist() == our_output[0].tolist()}")
    print(f"HF text:   {tokenizer.decode(hf_output[0], skip_special_tokens=True)!r}")
    print(f"OURS text: {tokenizer.decode(our_output[0], skip_special_tokens=True)!r}")

    print("\nSTEP COMPARISON")
    for index in range(max(len(hf_steps), len(our_steps))):
        hf_step = hf_steps[index] if index < len(hf_steps) else None
        our_step = our_steps[index] if index < len(our_steps) else None
        print_step_pair(tokenizer, hf_step, our_step)


if __name__ == "__main__":
    main()
