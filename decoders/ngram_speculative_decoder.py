#!/usr/bin/env python
"""First-principles n-gram speculative decoding.

This is the prompt-lookup / n-gram variant of speculative decoding. It has no
assistant model. The drafter searches the already generated tokens for a suffix
that appeared earlier, copies the tokens that followed that earlier occurrence,
and lets the target model verify the copied block.

The model call surface is deliberately small: the decoder only calls the target
model forward through `target_predictions_for_draft`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoders.first_principles_speculative_decoder import (
    LogitTopK,
    choose_device,
    dtype_from_arg,
    format_top_logits,
    model_device,
    normalize_eos_token_ids,
    should_stop,
    stop_reason_for,
    target_predictions_for_draft,
    validate_generate_inputs,
)


DEFAULT_PROMPT = "The future of AI is the future of"
DEFAULT_MAX_NEW_TOKENS = 20


@dataclass
class NgramDraft:
    tokens: torch.Tensor
    matched_ngram_size: int
    match_start_index: int | None
    copied_start_index: int | None


@dataclass
class NgramStepTrace:
    step: int
    prefix_length: int
    remaining_new_tokens: int
    requested_draft_tokens: int
    matched_ngram_size: int
    match_start_index: int | None
    copied_start_index: int | None
    draft_tokens: list[int]
    target_predictions: list[int]
    target_top_logits: list[LogitTopK]
    accepted_count: int
    rejected_at: int | None
    appended_tokens: list[int]
    output_length: int
    stop_reason: str | None


def _crop_before_eos(token_ids: list[int], eos_token_ids: set[int]) -> list[int]:
    if not eos_token_ids:
        return token_ids
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[:index]
    return token_ids


def find_prompt_lookup_draft(
    token_ids: list[int],
    num_output_tokens: int,
    max_matching_ngram_size: int,
    min_matching_ngram_size: int = 1,
    eos_token_ids: set[int] | None = None,
) -> tuple[list[int], int, int | None, int | None]:
    """Find a copied continuation for the longest suffix n-gram.

    This mirrors the inspectable Hugging Face prompt-lookup behavior rather
    than vLLM's optimized KMP implementation: try longer suffix n-grams first,
    scan earlier matches from left to right, and copy up to `num_output_tokens`
    tokens that followed the first usable match.
    """
    if num_output_tokens <= 0:
        return [], 0, None, None
    if max_matching_ngram_size <= 0:
        raise ValueError("max_matching_ngram_size must be positive.")
    if min_matching_ngram_size <= 0:
        raise ValueError("min_matching_ngram_size must be positive.")
    if min_matching_ngram_size > max_matching_ngram_size:
        raise ValueError("min_matching_ngram_size cannot exceed max_matching_ngram_size.")

    eos_token_ids = eos_token_ids or set()
    input_length = len(token_ids)
    largest_ngram = min(max_matching_ngram_size, input_length - 1)

    for ngram_size in range(largest_ngram, min_matching_ngram_size - 1, -1):
        suffix = token_ids[-ngram_size:]
        last_possible_match = input_length - ngram_size
        for match_start in range(last_possible_match):
            if token_ids[match_start : match_start + ngram_size] != suffix:
                continue

            copied_start = match_start + ngram_size
            copied_end = min(copied_start + num_output_tokens, input_length)
            if copied_start >= copied_end:
                continue

            copied = _crop_before_eos(token_ids[copied_start:copied_end], eos_token_ids)
            if copied:
                return copied, ngram_size, match_start, copied_start

    return [], 0, None, None


def propose_ngram_draft(
    generated: torch.Tensor,
    num_output_tokens: int,
    max_matching_ngram_size: int,
    min_matching_ngram_size: int,
    eos_token_ids: set[int],
) -> NgramDraft:
    token_ids = [int(token_id) for token_id in generated.detach().cpu()[0].tolist()]
    draft_ids, ngram_size, match_start, copied_start = find_prompt_lookup_draft(
        token_ids,
        num_output_tokens=num_output_tokens,
        max_matching_ngram_size=max_matching_ngram_size,
        min_matching_ngram_size=min_matching_ngram_size,
        eos_token_ids=eos_token_ids,
    )
    draft_tokens = torch.tensor([draft_ids], dtype=generated.dtype, device=generated.device)
    return NgramDraft(
        tokens=draft_tokens,
        matched_ngram_size=ngram_size,
        match_start_index=match_start,
        copied_start_index=copied_start,
    )


def _append_verified_tokens(
    generated: torch.Tensor,
    draft_tokens: torch.Tensor,
    target_predictions: torch.Tensor,
    remaining: int,
    eos_token_ids: set[int],
    min_length: int,
) -> tuple[torch.Tensor, int, int | None, torch.Tensor]:
    accepted_count = 0
    rejected_at: int | None = None
    replacement_token: torch.Tensor | None = None

    for index in range(draft_tokens.shape[-1]):
        draft_token = draft_tokens[:, index : index + 1]
        target_token = target_predictions[:, index : index + 1]
        if int(draft_token.item()) == int(target_token.item()):
            accepted_count += 1
        else:
            rejected_at = index
            replacement_token = target_token
            break

    if draft_tokens.shape[-1] == 0:
        appended = target_predictions[:, :1]
    elif replacement_token is None:
        appended = draft_tokens
        simulated = torch.cat([generated, appended], dim=-1)
        if appended.shape[-1] < remaining and not should_stop(simulated, eos_token_ids, min_length):
            bonus_token = target_predictions[:, draft_tokens.shape[-1] : draft_tokens.shape[-1] + 1]
            appended = torch.cat([appended, bonus_token], dim=-1)
    else:
        accepted_tokens = draft_tokens[:, :accepted_count]
        appended = torch.cat([accepted_tokens, replacement_token], dim=-1)

    appended = appended[:, :remaining]
    generated = torch.cat([generated, appended], dim=-1)
    return generated, accepted_count, rejected_at, appended


def generate(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    num_speculative_tokens: int = 4,
    max_matching_ngram_size: int = 4,
    min_matching_ngram_size: int = 1,
    trace_steps: list[NgramStepTrace] | None = None,
    top_k_logits: int = 0,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
) -> torch.Tensor:
    """Generate greedily with n-gram prompt-lookup speculative proposals."""
    validate_generate_inputs(prompt_token_ids, max_new_tokens, num_speculative_tokens)
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    eos_token_ids = normalize_eos_token_ids(model, eos_token_id)
    prompt_length = generated.shape[-1]

    with torch.inference_mode():
        step = 1
        while generated.shape[-1] - prompt_length < max_new_tokens:
            if should_stop(generated, eos_token_ids, min_length):
                break

            prefix_length = generated.shape[-1]
            remaining = max_new_tokens - (prefix_length - prompt_length)
            requested_draft_tokens = min(num_speculative_tokens, remaining)
            draft = propose_ngram_draft(
                generated,
                num_output_tokens=requested_draft_tokens,
                max_matching_ngram_size=max_matching_ngram_size,
                min_matching_ngram_size=min_matching_ngram_size,
                eos_token_ids=eos_token_ids,
            )

            if progress:
                print(
                    f"[ngram step {step}] prefix_len={prefix_length} "
                    f"match_n={draft.matched_ngram_size} draft={draft.tokens[0].tolist()}",
                    flush=True,
                )

            target_result = target_predictions_for_draft(
                model,
                generated,
                draft.tokens,
                eos_token_ids,
                min_length,
                top_k_logits=top_k_logits,
                progress=progress,
                step=step,
                heartbeat_seconds=heartbeat_seconds,
            )
            generated, accepted_count, rejected_at, appended = _append_verified_tokens(
                generated,
                draft.tokens,
                target_result.predictions,
                remaining,
                eos_token_ids,
                min_length,
            )

            if trace_steps is not None:
                trace_steps.append(
                    NgramStepTrace(
                        step=step,
                        prefix_length=prefix_length,
                        remaining_new_tokens=remaining,
                        requested_draft_tokens=requested_draft_tokens,
                        matched_ngram_size=draft.matched_ngram_size,
                        match_start_index=draft.match_start_index,
                        copied_start_index=draft.copied_start_index,
                        draft_tokens=draft.tokens[0].tolist(),
                        target_predictions=target_result.predictions[0].tolist(),
                        target_top_logits=target_result.top_logits,
                        accepted_count=accepted_count,
                        rejected_at=rejected_at,
                        appended_tokens=appended[0].tolist(),
                        output_length=generated.shape[-1],
                        stop_reason=stop_reason_for(generated, prompt_length, max_new_tokens, eos_token_ids, min_length),
                    )
                )

            if progress:
                print(
                    f"[ngram step {step}] accepted={accepted_count} rejected_at={rejected_at} "
                    f"appended={appended[0].tolist()} output_len={generated.shape[-1]}",
                    flush=True,
                )
            step += 1

    return generated


def generate_with_trace(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    num_speculative_tokens: int = 4,
    max_matching_ngram_size: int = 4,
    min_matching_ngram_size: int = 1,
    top_k_logits: int = 0,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
) -> tuple[torch.Tensor, list[NgramStepTrace]]:
    trace_steps: list[NgramStepTrace] = []
    output_ids = generate(
        model,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        eos_token_id=eos_token_id,
        num_speculative_tokens=num_speculative_tokens,
        max_matching_ngram_size=max_matching_ngram_size,
        min_matching_ngram_size=min_matching_ngram_size,
        trace_steps=trace_steps,
        top_k_logits=top_k_logits,
        progress=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    return output_ids, trace_steps


def print_trace(
    trace_steps: list[NgramStepTrace],
    tokenizer: AutoTokenizer | None = None,
    show_logits: bool = False,
) -> None:
    for item in trace_steps:
        print(f"\nNGRAM STEP {item.step}")
        print(f"prefix length: {item.prefix_length}")
        print(f"remaining new-token budget: {item.remaining_new_tokens}")
        print(f"requested draft tokens: {item.requested_draft_tokens}")
        print(f"matched ngram size: {item.matched_ngram_size}")
        print(f"match start index: {item.match_start_index}")
        print(f"copied start index: {item.copied_start_index}")
        print(f"draft tokens: {item.draft_tokens}")
        if tokenizer is not None:
            print(f"draft text: {tokenizer.decode(item.draft_tokens)!r}")
        print(f"target predictions plus bonus: {item.target_predictions}")
        if show_logits:
            for idx, summary in enumerate(item.target_top_logits):
                print(f"target logits top-k for verify[{idx}]: {format_top_logits(summary, tokenizer)}")
        print(f"accepted count: {item.accepted_count}")
        print(f"rejected at: {item.rejected_at}")
        print(f"appended tokens: {item.appended_tokens}")
        if tokenizer is not None:
            print(f"appended text: {tokenizer.decode(item.appended_tokens)!r}")
        print(f"output length: {item.output_length}")
        print(f"stop reason after step: {item.stop_reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace n-gram prompt-lookup speculative decoding.")
    parser.add_argument("--target-model", default="gpt2")
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer to use. Defaults to --target-model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--max-matching-ngram-size", type=int, default=4)
    parser.add_argument("--min-matching-ngram-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="auto")
    parser.add_argument("--show-logits", action="store_true")
    parser.add_argument("--top-k-logits", type=int, default=5)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs).to(device).eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    output_ids, trace_steps = generate_with_trace(
        model,
        inputs.input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        num_speculative_tokens=args.num_speculative_tokens,
        max_matching_ngram_size=args.max_matching_ngram_size,
        min_matching_ngram_size=args.min_matching_ngram_size,
        top_k_logits=args.top_k_logits if args.show_logits else 0,
        progress=args.progress,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    print(f"prompt token ids: {inputs.input_ids[0].tolist()}")
    print(f"prompt length: {inputs.input_ids.shape[-1]}")
    print_trace(trace_steps, tokenizer, show_logits=args.show_logits)
    print("\nFINAL")
    print(f"token ids: {output_ids[0].tolist()}")
    print(f"decoded: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
