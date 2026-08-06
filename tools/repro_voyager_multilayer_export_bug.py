#!/usr/bin/env python
"""Self-contained repro: torch.export of a multi-layer Llama decode step fails.

A decode-step wrapper (the same shape of wrapper test_codegen.py's llm_decode
uses: precomputed rotary embeddings + a prefilled StaticCache held as a module
attribute + a python loop over decoder layers) exports fine with ONE decoder
layer and fails with TWO OR MORE, during torch.export fake-tensor tracing:

    RuntimeError: Attempting to broadcast a dimension of length <head_dim> at
    -1! Mismatching argument at index 1 had torch.Size([1, 1, 1, head_dim]);
    but expected shape should be broadcastable to [1, 1, head_dim, n_heads]

i.e. inside apply_rotary_pos_emb the query reaches the cos-multiply laid out
as [b, 1, head_dim, n_heads] instead of [b, n_heads, seq, head_dim] from the
second layer onward.

Ruled out by bisection: the checkpoint (fresh random weights fail), head
geometry (12x64, 6x128 and 2x128 all fail), the mask contents (vendored -inf
mask and plain zeros both fail), and voyager's export wrapper (plain
torch.export fails identically). Only the layer count matters.

No downloads needed: builds a tiny random Llama in-process.

Usage: python tools/repro_voyager_multilayer_export_bug.py
"""

from __future__ import annotations

import platform

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

try:
    import voyager_compiler
    from voyager_compiler import export_model

    HAVE_VOYAGER = True
except ImportError:
    HAVE_VOYAGER = False


def build_case(num_layers: int, hidden: int = 256, heads: int = 2):
    """A decode-step wrapper over a tiny random Llama with a prefilled cache."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=hidden // heads,
    )
    model = LlamaForCausalLM(config).to(torch.float16).eval()
    cache = StaticCache(config=config, max_batch_size=1, max_cache_len=64, dtype=torch.float16)
    model(torch.randint(0, 512, (1, 32)), past_key_values=cache, use_cache=True)  # prefill

    embeds = model.model.embed_tokens(torch.randint(0, 512, (1, 1)))
    cache_position = torch.arange(32, 33)
    mask = torch.zeros(1, 1, 1, 64, dtype=torch.float16)
    position_embeddings = model.model.rotary_emb(embeds, cache_position.unsqueeze(0))

    class DecodeStepWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = model.model.layers
            self.static_cache = cache

        def forward(self, hidden_states, attention_mask, position_embeddings, cache_position=None):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=self.static_cache,
                    cache_position=cache_position,
                )[0]
            return hidden_states

    return DecodeStepWrapper(), (embeds, mask, position_embeddings, cache_position)


def main() -> None:
    print(f"python {platform.python_version()} / {platform.platform()}")
    print(f"torch {torch.__version__}, transformers {transformers.__version__}")
    if HAVE_VOYAGER:
        print(f"voyager_compiler at {voyager_compiler.__file__}")
    print()

    for num_layers in (1, 2, 3):
        wrapper, example_args = build_case(num_layers)

        exporters = [("plain torch.export (strict=False)", lambda w, a: torch.export.export(w, a, strict=False))]
        if HAVE_VOYAGER:
            exporters.append(("voyager export_model", lambda w, a: export_model(w, a)))

        for name, exporter in exporters:
            try:
                exporter(wrapper, example_args)
                print(f"{num_layers} layer(s) | {name}: OK")
            except RuntimeError as error:
                last = str(error).strip().splitlines()[-1]
                print(f"{num_layers} layer(s) | {name}: FAILS -> {last[:110]}")


if __name__ == "__main__":
    main()
