#!/usr/bin/env python
"""A cached-oriented greedy speculative decoding implementation.

This version exposes prefill, decode, past_key_values, assistant draft, target
verification, and cache_position concepts. It uses target KV cache for the
verification block by caching the accepted prefix except its final token, then
feeding [last_accepted_token] + [draft block]. After each accept/reject update,
it rebuilds the target cache from the accepted sequence. That rollback/rebuild
keeps the code readable while still showing where cache management enters.
"""

from __future__ import annotations

import argparse
import inspect
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from manual_greedy_spec_decode import PROMPT, SpecDecodeResult, assistant_draft, choose_device


def argmax_next(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def prefill(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor]:
    outputs = model(input_ids, use_cache=True)
    return outputs.past_key_values, outputs.logits


def cache_position_for(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(input_ids.shape[-1], device=input_ids.device)


def verify_with_prefix_cache(
    target_model: torch.nn.Module,
    accepted_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return target predictions for each draft position plus one extra position."""
    if accepted_ids.shape[-1] == 1:
        prefix_ids = accepted_ids[:, :0]
        verify_ids = torch.cat([accepted_ids[:, -1:], draft_tokens], dim=-1)
        outputs = target_model(verify_ids, use_cache=True)
        return torch.argmax(outputs.logits, dim=-1)

    prefix_ids = accepted_ids[:, :-1]
    last_plus_draft = torch.cat([accepted_ids[:, -1:], draft_tokens], dim=-1)

    prefix_outputs = target_model(prefix_ids, use_cache=True)
    target_past_key_values = prefix_outputs.past_key_values
    cache_position = torch.arange(prefix_ids.shape[-1], prefix_ids.shape[-1] + last_plus_draft.shape[-1], device=accepted_ids.device)

    verify_kwargs = {
        "past_key_values": target_past_key_values,
        "use_cache": True,
    }
    if "cache_position" in inspect.signature(target_model.forward).parameters:
        verify_kwargs["cache_position"] = cache_position
    verify_outputs = target_model(last_plus_draft, **verify_kwargs)
    return torch.argmax(verify_outputs.logits, dim=-1)


def manual_speculative_generate_cached(
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
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = prompt_ids.clone()
    prompt_len = generated.shape[-1]
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        target_past_key_values, target_logits = prefill(target_model, prompt_ids)
        current_token = argmax_next(target_logits)
        generated = torch.cat([generated, current_token], dim=-1)

        if verbose:
            print("PREFILL")
            print(f"prompt length: {prompt_len}")
            print(f"past_key_values present: {target_past_key_values is not None}")
            print(f"cache_position: {cache_position_for(prompt_ids).tolist()}")
            print(f"current_token: {int(current_token.item())} {tokenizer.decode(current_token[0])!r}")

        step = 1
        while generated.shape[-1] - prompt_len < max_new_tokens:
            if eos_token_id is not None and int(generated[0, -1].item()) == eos_token_id:
                break

            remaining = max_new_tokens - (generated.shape[-1] - prompt_len)
            current_draft_len = min(draft_len, remaining)
            draft_tokens = assistant_draft(assistant_model, generated, current_draft_len, eos_token_id)
            if draft_tokens.shape[-1] == 0:
                break

            # target_preds includes:
            #   index 0: prediction after last accepted token, compared with draft[0]
            #   index 1: prediction after draft[0], compared with draft[1]
            #   ...
            #   final index: extra target token if all draft tokens are accepted
            target_preds = verify_with_prefix_cache(target_model, generated, draft_tokens)

            accepted_count = 0
            rejected_at: int | None = None
            replacement_token: torch.Tensor | None = None

            for idx in range(draft_tokens.shape[-1]):
                draft_token = draft_tokens[:, idx : idx + 1]
                target_pred = target_preds[:, idx : idx + 1]
                if int(target_pred.item()) == int(draft_token.item()):
                    accepted_count += 1
                else:
                    rejected_at = idx
                    replacement_token = target_pred
                    break

            if verbose:
                print(f"\nSPEC STEP {step}")
                print(f"current sequence length: {generated.shape[-1]}")
                print(f"assistant draft tokens: {draft_tokens[0].tolist()}")
                print(f"assistant draft text: {tokenizer.decode(draft_tokens[0])!r}")
                print(f"target predicted tokens for draft positions: {target_preds[0, :draft_tokens.shape[-1]].tolist()}")
                print(f"accepted count: {accepted_count}")
                print(f"rejected at index: {rejected_at}")

            if rejected_at is None:
                generated = torch.cat([generated, draft_tokens], dim=-1)
                if generated.shape[-1] - prompt_len < max_new_tokens:
                    generated = torch.cat([generated, target_preds[:, draft_tokens.shape[-1] : draft_tokens.shape[-1] + 1]], dim=-1)
            else:
                accepted = draft_tokens[:, :accepted_count]
                generated = torch.cat([generated, accepted, replacement_token], dim=-1)

            # Rebuild target cache from the accepted sequence. This is the simple
            # rollback strategy; production implementations surgically crop and
            # update caches instead of recomputing.
            target_past_key_values, _ = prefill(target_model, generated)

            if verbose:
                print(f"target_past_key_values present after update: {target_past_key_values is not None}")
                print(f"new sequence length: {generated.shape[-1]}")

            step += 1

    token_ids = generated[0].tolist()
    return SpecDecodeResult(token_ids=token_ids, text=tokenizer.decode(token_ids, skip_special_tokens=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual cached-style greedy speculative decoding.")
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

    result = manual_speculative_generate_cached(
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
