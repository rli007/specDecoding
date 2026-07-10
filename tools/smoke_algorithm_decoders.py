#!/usr/bin/env python
"""Smoke-test the algorithm decoders without downloading model weights."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders import eagle_speculative_decoder as eagle
from decoders import medusa_speculative_decoder as medusa
from decoders import ngram_speculative_decoder as ngram


class ToyCacheLayer:
    is_sliding = False

    def __init__(self):
        self.keys = torch.empty((1, 1, 0, 1))
        self.values = torch.empty((1, 1, 0, 1))


class ToyCache:
    def __init__(self):
        self.layers = [ToyCacheLayer()]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.layers[0].keys.shape[-2]


class IncrementToyLM(nn.Module):
    """A tiny causal LM where every token predicts `(token + 1) % vocab`."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dummy = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.generation_config = SimpleNamespace(eos_token_id=None)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        use_cache=False,
        **kwargs,
    ):
        del attention_mask, position_ids, kwargs
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -1000.0,
            device=input_ids.device,
        )
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(dim=-1, index=next_ids.unsqueeze(-1), value=1000.0)

        hidden_states = torch.zeros(
            (batch_size, sequence_length, self.hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        hidden_states[..., 0] = input_ids.float()
        if use_cache:
            if past_key_values is None:
                past_key_values = ToyCache()
            keys = input_ids.float().reshape(batch_size, 1, sequence_length, 1)
            values = keys.clone()
            layer = past_key_values.layers[0]
            layer.keys = torch.cat([layer.keys.to(keys.device), keys], dim=-2)
            layer.values = torch.cat([layer.values.to(values.device), values], dim=-2)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=(hidden_states,), past_key_values=past_key_values)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class IncrementMedusaHeads(nn.Module):
    """Medusa heads that predict token + 2, token + 3, ... from toy hidden states."""

    def __init__(self, num_heads: int = 2, vocab_size: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        logits = torch.full(
            (self.num_heads, batch_size, sequence_length, self.vocab_size),
            -1000.0,
            device=hidden_states.device,
        )
        base_ids = hidden_states[..., 0].long()
        for head_index in range(self.num_heads):
            token_ids = (base_ids + head_index + 2) % self.vocab_size
            logits[head_index].scatter_(dim=-1, index=token_ids.unsqueeze(-1), value=1000.0)
        return logits


def assert_ids(name: str, actual: torch.Tensor, expected: list[int]) -> None:
    actual_ids = actual[0].tolist()
    if actual_ids != expected:
        raise AssertionError(f"{name} produced {actual_ids}, expected {expected}")
    print(f"{name}: ok -> {actual_ids}")


def main() -> None:
    model = IncrementToyLM()
    prompt = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)
    expected = [1, 2, 3, 1, 2, 3, 4, 5]

    ngram_output, ngram_trace = ngram.generate_with_trace(
        model,
        prompt,
        max_new_tokens=3,
        num_speculative_tokens=2,
        max_matching_ngram_size=2,
        min_matching_ngram_size=1,
    )
    assert_ids("ngram", ngram_output, expected)
    assert ngram_trace[0].matched_ngram_size == 2
    assert ngram_trace[0].accepted_count >= 1

    medusa_heads = IncrementMedusaHeads(num_heads=2, vocab_size=model.vocab_size)
    medusa_output, medusa_trace = medusa.generate_with_trace(
        model,
        medusa_heads,
        prompt,
        max_new_tokens=3,
        top_k=1,
        verifier="slow",
    )
    assert_ids("medusa", medusa_output, expected)
    assert medusa_trace[0].selected_path_tokens == [3, 4, 5]

    medusa_tree_output, medusa_tree_trace = medusa.generate_with_trace(
        model,
        medusa_heads,
        prompt,
        max_new_tokens=3,
        top_k=1,
        verifier="tree",
        fallback_to_slow=False,
    )
    assert_ids("medusa-tree", medusa_tree_output, expected)
    assert medusa_tree_trace[0].verification_method == "tree"
    assert medusa_tree_trace[0].cache_updated is True

    eagle_output, eagle_trace = eagle.generate_with_trace(
        model,
        eagle.DebugTargetLogitsDrafter(),
        prompt,
        max_new_tokens=3,
    )
    assert_ids("eagle", eagle_output, expected)
    assert eagle_trace[0].selected_path_tokens == [3]


if __name__ == "__main__":
    main()
