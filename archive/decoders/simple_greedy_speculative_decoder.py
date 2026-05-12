#!/usr/bin/env python
"""A minimal, clear greedy speculative decoding implementation.

This file does not call target.generate(..., assistant_model=assistant). It
manually implements the accept/reject loop and recomputes full sequences for
clarity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = "The Stanford football team"


@dataclass
class SpecDecodeResult:
    token_ids: list[int]
    text: str


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def argmax_next(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def assistant_draft(
    assistant_model: torch.nn.Module,
    current_ids: torch.Tensor,
    draft_len: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    draft_tokens: list[torch.Tensor] = []
    draft_sequence = current_ids

    for _ in range(draft_len):
        outputs = assistant_model(draft_sequence, use_cache=False)
        next_token = argmax_next(outputs.logits)
        draft_tokens.append(next_token)
        draft_sequence = torch.cat([draft_sequence, next_token], dim=-1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    if not draft_tokens:
        return torch.empty((current_ids.shape[0], 0), dtype=current_ids.dtype, device=current_ids.device)
    return torch.cat(draft_tokens, dim=-1)


def manual_speculative_generate(
    target_model: torch.nn.Module,
    assistant_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    draft_len: int = 4,
    device: torch.device | None = None,
    verbose: bool = True,
) -> SpecDecodeResult:
    device = device or next(target_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()
    prompt_len = generated.shape[-1]
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = target_model(generated, use_cache=True)
        _target_past_key_values = outputs.past_key_values
        next_token = argmax_next(outputs.logits)
        generated = torch.cat([generated, next_token], dim=-1)
        if verbose:
            print("PREFILL")
            print(f"prompt length: {prompt_len}")
            print(f"first target token: {int(next_token.item())} {tokenizer.decode(next_token[0])!r}")
            print(f"target_past_key_values present: {_target_past_key_values is not None}")

        step = 1
        while generated.shape[-1] - prompt_len < max_new_tokens:
            if eos_token_id is not None and int(generated[0, -1].item()) == eos_token_id:
                break

            remaining = max_new_tokens - (generated.shape[-1] - prompt_len)
            current_draft_len = min(draft_len, remaining)
            if current_draft_len <= 0:
                break

            draft_tokens = assistant_draft(assistant_model, generated, current_draft_len, eos_token_id)
            if draft_tokens.shape[-1] == 0:
                break

            candidate_input_ids = torch.cat([generated, draft_tokens], dim=-1)
            target_outputs = target_model(candidate_input_ids, use_cache=False)

            current_len = generated.shape[-1]
            prediction_start = current_len - 1
            prediction_end = prediction_start + draft_tokens.shape[-1]
            target_pred_tokens = torch.argmax(
                target_outputs.logits[:, prediction_start:prediction_end, :],
                dim=-1,
            )

            accepted_count = 0
            rejected_at: int | None = None
            replacement_token: torch.Tensor | None = None

            for idx in range(draft_tokens.shape[-1]):
                draft_token = draft_tokens[:, idx : idx + 1]
                target_pred = target_pred_tokens[:, idx : idx + 1]
                if int(target_pred.item()) == int(draft_token.item()):
                    accepted_count += 1
                else:
                    rejected_at = idx
                    replacement_token = target_pred
                    break

            if verbose:
                print(f"\nSPEC STEP {step}")
                print(f"current sequence length: {current_len}")
                print(f"assistant draft tokens: {draft_tokens[0].tolist()}")
                print(f"assistant draft text: {tokenizer.decode(draft_tokens[0])!r}")
                print(f"target predicted tokens for draft positions: {target_pred_tokens[0].tolist()}")
                print(f"accepted count: {accepted_count}")
                print(f"rejected at index: {rejected_at}")

            if rejected_at is None:
                generated = torch.cat([generated, draft_tokens], dim=-1)
                if generated.shape[-1] - prompt_len < max_new_tokens:
                    extra_token = torch.argmax(target_outputs.logits[:, prediction_end : prediction_end + 1, :], dim=-1)
                    generated = torch.cat([generated, extra_token], dim=-1)
            else:
                accepted = draft_tokens[:, :accepted_count]
                generated = torch.cat([generated, accepted, replacement_token], dim=-1)

            if verbose:
                print(f"new sequence length: {generated.shape[-1]}")

            step += 1

    token_ids = generated[0].tolist()
    return SpecDecodeResult(token_ids=token_ids, text=tokenizer.decode(token_ids, skip_special_tokens=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual greedy speculative decoding.")
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

    result = manual_speculative_generate(
        target,
        assistant,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        device=device,
    )
    print("\nFINAL")
    print(f"token ids: {result.token_ids}")
    print(f"decoded: {result.text}")


if __name__ == "__main__":
    main()
