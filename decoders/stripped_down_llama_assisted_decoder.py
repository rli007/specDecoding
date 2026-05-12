#!/usr/bin/env python
"""Stripped-down Llama assisted/speculative decoding.

This mirrors the shape of Hugging Face assisted decoding, but keeps the code
close to the static-cache reference generate function:

1. The assistant drafts a short greedy block.
2. The target verifies that block in one forward pass.
3. Matching assistant tokens are accepted until the first mismatch.
4. The target token at the mismatch/bonus position is appended.

The decode loop does not call `generate()`, candidate-generator helpers, logits
processors, or `prepare_inputs_for_generation`. The runtime model calls are
plain forwards into the assistant and target models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_ASSISTANT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_PROMPT = "The Stanford football team"
DEFAULT_MAX_NEW_TOKENS = 2
DEFAULT_NUM_ASSISTANT_TOKENS = 4


@dataclass
class CandidateState:
    """Small piece of HF's candidate-generator state."""

    num_assistant_tokens: float


@dataclass
class CandidateDraft:
    tokens: torch.Tensor
    logits: torch.Tensor | None
    ended_on_eos: bool


@dataclass
class AssistedStepTrace:
    step: int
    prefix_length: int
    assistant_budget: int
    draft_tokens: list[int]
    target_selected_tokens: list[int]
    accepted_assistant_tokens: int
    appended_tokens: list[int]
    next_assistant_budget: float
    output_length: int


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_from_arg(dtype_arg: str) -> torch.dtype | str | None:
    if dtype_arg == "none":
        return None
    if dtype_arg == "auto":
        return "auto"
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def model_device(model: torch.nn.Module, fallback: torch.Tensor) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback.device


def model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values=None,
):
    """Call the actual model and return logits plus cache."""
    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def normalize_eos_token_ids(eos_token_id: int | Iterable[int] | torch.Tensor | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, torch.Tensor):
        return {int(token) for token in eos_token_id.detach().flatten().tolist()}
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(token) for token in eos_token_id if token is not None}


def process_logits(scores: torch.Tensor, eos_token_ids: set[int]) -> torch.Tensor:
    """Mask EOS so greedy argmax cannot pick it before min_length."""
    if not eos_token_ids:
        return scores

    eos_tensor = torch.tensor(list(eos_token_ids), device=scores.device)
    vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    eos_token_mask = torch.isin(vocab_tensor, eos_tensor)
    return torch.where(eos_token_mask, -math.inf, scores)


def select_next_token(
    logits: torch.Tensor,
    eos_token_ids: set[int],
    current_length: int,
    min_length: int,
) -> torch.Tensor:
    if current_length < min_length - 1:
        logits = process_logits(logits, eos_token_ids)
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def token_is_eos(token: torch.Tensor, eos_token_ids: set[int]) -> bool:
    return bool(eos_token_ids) and int(token.item()) in eos_token_ids


def should_stop(generated: torch.Tensor, eos_token_ids: set[int], min_length: int) -> bool:
    return generated.shape[-1] >= min_length and token_is_eos(generated[:, -1:], eos_token_ids)


def calculate_candidate_token_count(
    state: CandidateState,
    prompt_length: int,
    current_length: int,
    max_new_tokens: int,
) -> int:
    """HF-style assistant budget: reserve one slot for the target's bonus token."""
    max_length = prompt_length + max_new_tokens
    remaining_after_current = max_length - current_length
    return max(0, min(int(state.num_assistant_tokens), remaining_after_current - 1))


def assistant_draft_candidates(
    assistant_model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_candidate_tokens: int,
    eos_token_ids: set[int],
    min_length: int,
) -> CandidateDraft:
    """Draft greedy candidate tokens with manual assistant forwards."""
    if num_candidate_tokens <= 0:
        empty = torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
        return CandidateDraft(tokens=empty, logits=None, ended_on_eos=False)

    assistant_device = model_device(assistant_model, input_ids)
    assistant_input_ids = input_ids.to(assistant_device)

    draft_tokens: list[torch.Tensor] = []
    draft_logits: list[torch.Tensor] = []
    past_key_values = None
    next_input_ids = assistant_input_ids
    current_length = assistant_input_ids.shape[-1]
    ended_on_eos = False

    for _ in range(num_candidate_tokens):
        # First call pre-fills the assistant cache with the current sequence.
        # Later calls feed one draft token and reuse that cache.
        logits, past_key_values = model_forward(assistant_model, next_input_ids, past_key_values)
        next_token = select_next_token(logits, eos_token_ids, current_length, min_length)

        draft_logits.append(logits[:, -1, :].detach())
        draft_tokens.append(next_token)

        current_length += 1
        if token_is_eos(next_token, eos_token_ids):
            ended_on_eos = True
            break

        next_input_ids = next_token

    return CandidateDraft(
        tokens=torch.cat(draft_tokens, dim=-1).to(input_ids.device),
        logits=torch.stack([item.to(input_ids.device) for item in draft_logits], dim=1),
        ended_on_eos=ended_on_eos,
    )


def target_predictions_for_candidates(
    target_model: torch.nn.Module,
    generated: torch.Tensor,
    draft_tokens: torch.Tensor,
    eos_token_ids: set[int],
    min_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify candidate tokens and produce one extra target prediction."""
    target_device = model_device(target_model, generated)
    generated = generated.to(target_device)
    draft_tokens = draft_tokens.to(target_device)

    # HF crops the target cache after verification. This stripped version gets
    # the same semantics by rebuilding the prefix cache up to the token before
    # the verification block.
    prefix_ids = generated[:, :-1]
    verify_ids = torch.cat([generated[:, -1:], draft_tokens], dim=-1)

    past_key_values = None
    if prefix_ids.shape[-1] > 0:
        _, past_key_values = model_forward(target_model, prefix_ids, past_key_values=None)

    logits, _ = model_forward(target_model, verify_ids, past_key_values=past_key_values)

    selected_tokens: list[torch.Tensor] = []
    processed_logits: list[torch.Tensor] = []
    for index in range(logits.shape[1]):
        decision_logits = logits[:, index : index + 1, :]
        current_length = generated.shape[-1] + index
        next_token = select_next_token(decision_logits, eos_token_ids, current_length, min_length)
        selected_tokens.append(next_token)
        processed_logits.append(decision_logits[:, -1, :].detach())

    return (
        torch.cat(selected_tokens, dim=-1).to(generated.device),
        torch.stack(processed_logits, dim=1).to(generated.device),
    )


def count_matching_prefix(candidate_tokens: torch.Tensor, selected_tokens: torch.Tensor) -> int:
    """Count matches until the first target/assistant disagreement."""
    if candidate_tokens.shape[-1] == 0:
        return 0

    target_tokens_for_candidates = selected_tokens[:, : candidate_tokens.shape[-1]]
    mismatches = candidate_tokens != target_tokens_for_candidates
    if not bool(mismatches.any()):
        return candidate_tokens.shape[-1]
    return int(mismatches[0].nonzero(as_tuple=False)[0].item())


def update_candidate_strategy(state: CandidateState, candidate_length: int, n_matches: int) -> None:
    """HF heuristic schedule: increase on full match, decrease on mismatch."""
    if n_matches == candidate_length:
        state.num_assistant_tokens += 2
    else:
        state.num_assistant_tokens = max(1, state.num_assistant_tokens - 1)


def assisted_generate(
    target_model: torch.nn.Module,
    assistant_model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    num_assistant_tokens: int = DEFAULT_NUM_ASSISTANT_TOKENS,
    verbose: bool = True,
    trace_steps: list[AssistedStepTrace] | None = None,
) -> torch.Tensor:
    """Greedy assisted decoding with manual candidate generation."""
    if prompt_token_ids.ndim != 2 or prompt_token_ids.shape[0] != 1:
        raise ValueError("This stripped decoder expects input IDs with shape [1, sequence_length].")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(target_model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    prompt_length = generated.shape[-1]
    eos_token_ids = normalize_eos_token_ids(eos_token_id)
    candidate_state = CandidateState(num_assistant_tokens=float(num_assistant_tokens))

    with torch.no_grad():
        step = 1
        while generated.shape[-1] - prompt_length < max_new_tokens:
            if should_stop(generated, eos_token_ids, min_length):
                break

            current_length = generated.shape[-1]
            candidate_count = calculate_candidate_token_count(
                candidate_state,
                prompt_length,
                current_length,
                max_new_tokens,
            )

            # Candidate generation: assistant drafts up to N tokens.
            draft = assistant_draft_candidates(
                assistant_model,
                generated,
                candidate_count,
                eos_token_ids,
                min_length,
            )

            # Assisted decoding: target verifies the candidate block plus one
            # extra position. With zero candidates, this is normal greedy.
            selected_tokens, target_logits = target_predictions_for_candidates(
                target_model,
                generated,
                draft.tokens,
                eos_token_ids,
                min_length,
            )

            candidate_length = draft.tokens.shape[-1]
            n_matches = count_matching_prefix(draft.tokens, selected_tokens)

            # If the candidate sequence itself reached a stop condition, do not
            # append the extra target bonus token after the final candidate.
            if draft.ended_on_eos and n_matches == candidate_length:
                n_matches -= 1

            valid_tokens = selected_tokens[:, : n_matches + 1]
            remaining = max_new_tokens - (generated.shape[-1] - prompt_length)
            valid_tokens = valid_tokens[:, :remaining]
            generated = torch.cat([generated, valid_tokens.to(generated.device)], dim=-1)

            update_candidate_strategy(candidate_state, candidate_length, n_matches)

            if trace_steps is not None:
                trace_steps.append(
                    AssistedStepTrace(
                        step=step,
                        prefix_length=current_length,
                        assistant_budget=candidate_count,
                        draft_tokens=draft.tokens[0].tolist(),
                        target_selected_tokens=selected_tokens[0].tolist(),
                        accepted_assistant_tokens=n_matches,
                        appended_tokens=valid_tokens[0].tolist(),
                        next_assistant_budget=candidate_state.num_assistant_tokens,
                        output_length=generated.shape[-1],
                    )
                )

            if verbose:
                print(f"\nASSISTED STEP {step}")
                print(f"prefix length: {current_length}")
                print(f"assistant token budget: {candidate_count}")
                print(f"assistant draft tokens: {draft.tokens[0].tolist()}")
                print(f"target selected tokens: {selected_tokens[0].tolist()}")
                print(f"accepted assistant tokens: {n_matches}")
                print(f"appended tokens: {valid_tokens[0].tolist()}")
                print(f"next assistant token budget: {candidate_state.num_assistant_tokens:g}")
                print(f"output length: {generated.shape[-1]}")

            if should_stop(generated, eos_token_ids, min_length):
                break

            step += 1

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stripped-down Llama assisted/speculative decoding.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer to use. Defaults to --target-model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-assistant-tokens", type=int, default=DEFAULT_NUM_ASSISTANT_TOKENS)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Only print the final output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.target_model

    model_kwargs = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True

    print(f"Using device={device}")
    print(f"target model: {args.target_model}")
    print(f"assistant model: {args.assistant_model}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs).to(device).eval()
    assistant_model = AutoModelForCausalLM.from_pretrained(args.assistant_model, **model_kwargs).to(device).eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    output_ids = assisted_generate(
        target_model,
        assistant_model,
        inputs.input_ids,
        max_new_tokens=args.max_new_tokens,
        min_length=args.min_length,
        eos_token_id=tokenizer.eos_token_id,
        num_assistant_tokens=args.num_assistant_tokens,
        verbose=not args.quiet,
    )

    print("\nFINAL")
    print(f"token ids: {output_ids[0].tolist()}")
    print(f"decoded: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
