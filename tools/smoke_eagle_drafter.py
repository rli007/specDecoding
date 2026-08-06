#!/usr/bin/env python
"""Smoke tests for the real EAGLE-1 drafter path.

1. Mechanism + losslessness (no downloads): tiny random target + tiny random
   drafter. A random drafter guesses garbage, so acceptance should be ~0 — but
   greedy EAGLE output must STILL be token-identical to plain greedy decoding,
   because verification only ever keeps tokens the target itself would emit.
   This invariant catches acceptance-logic bugs independent of drafter quality.

2. --real-load: builds the drafter from the published yuhuili checkpoint and
   hard-verifies every weight maps (no missing/unexpected keys), without
   loading the 7B base model.

Usage:
  python tools/smoke_eagle_drafter.py
  python tools/smoke_eagle_drafter.py --real-load
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import LlamaConfig, LlamaForCausalLM

from decoders.eagle_speculative_decoder import (
    EagleDrafterModel,
    EagleOneDrafter,
    generate_with_trace,
    load_official_eagle_drafter,
)


def tiny_config(num_layers: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=211,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
    )


def plain_greedy(model: torch.nn.Module, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    generated = prompt.clone()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(input_ids=generated).logits
            generated = torch.cat([generated, torch.argmax(logits[:, -1:, :], dim=-1)], dim=-1)
    return generated


def mechanism_test() -> None:
    torch.manual_seed(0)
    target = LlamaForCausalLM(tiny_config(num_layers=2)).to(torch.float32).eval()
    drafter_model = EagleDrafterModel(tiny_config(num_layers=1)).to(torch.float32).eval()
    drafter = EagleOneDrafter(drafter_model, lm_head=target.lm_head)

    prompt = torch.randint(0, 211, (1, 8))
    max_new_tokens = 12

    reference = plain_greedy(target, prompt, max_new_tokens)
    output, trace = generate_with_trace(
        target,
        drafter,
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
    )

    assert output.shape[-1] == reference.shape[-1], (
        f"length mismatch: eagle {output.shape[-1]} vs greedy {reference.shape[-1]}"
    )
    assert torch.equal(output, reference), (
        f"LOSSLESSNESS VIOLATED:\n eagle  {output[0].tolist()}\n greedy {reference[0].tolist()}"
    )
    appended = sum(len(step.appended_tokens) for step in trace)
    print(
        f"mechanism: ok — {len(trace)} steps, {appended} tokens, "
        f"{appended / len(trace):.2f} tokens/step (random drafter), output == plain greedy"
    )
    for step in trace[:1]:
        print(f"  step 1 drafter metadata: {step.drafter_metadata}")


def real_load_test() -> None:
    drafter = load_official_eagle_drafter(device="cpu", dtype=torch.float16)
    params = sum(p.numel() for p in drafter.parameters())
    assert isinstance(drafter.layer.input_layernorm, torch.nn.Identity)
    print(f"real checkpoint: ok — {params / 1e6:.0f}M params, every weight mapped, input_layernorm removed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-load", action="store_true", help="Also load the published checkpoint (downloads ~1.6 GB once).")
    args = parser.parse_args()

    mechanism_test()
    if args.real_load:
        real_load_test()


if __name__ == "__main__":
    main()
