#!/usr/bin/env python
"""Trace the real Hugging Face assisted/speculative generation path.

This script intentionally calls Hugging Face's own:

    target.generate(..., assistant_model=assistant, do_sample=False)

The monkey patches below print the path through GenerationMixin, candidate
generation, assistant forwards, and target verification forwards. Some local
variables inside Hugging Face internals are not visible from wrappers; in those
cases the trace prints the closest observable tensor shapes and notes the gap.
"""

from __future__ import annotations

import argparse
import contextlib
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.generation.utils import GenerationMixin


PROMPT = "The Stanford football team"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace HF assisted/speculative generation internals.")
    parser.add_argument("--target-model", default="gpt2")
    parser.add_argument("--assistant-model", default="distilgpt2")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--num-assistant-tokens", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shape_of(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, dict):
        pieces = [f"{key}={shape_of(item)}" for key, item in value.items() if torch.is_tensor(item)]
        return "{" + ", ".join(pieces) + "}"
    if isinstance(value, (tuple, list)):
        pieces = [shape_of(item) for item in value if torch.is_tensor(item)]
        return "[" + ", ".join(pieces) + "]"
    return "None"


def cache_presence(kwargs: dict[str, Any]) -> str:
    pkv = kwargs.get("past_key_values")
    return str(pkv is not None)


def cache_position_text(kwargs: dict[str, Any]) -> str:
    cache_position = kwargs.get("cache_position")
    if cache_position is None:
        return "None"
    if torch.is_tensor(cache_position):
        return f"shape={tuple(cache_position.shape)}, values={cache_position.detach().cpu().tolist()}"
    return repr(cache_position)


@contextlib.contextmanager
def patch_hf_internals(target: torch.nn.Module, assistant: torch.nn.Module, max_new_tokens: int) -> Any:
    originals: list[tuple[Any, str, Callable[..., Any]]] = []

    def replace(obj: Any, name: str, wrapper: Callable[..., Any]) -> None:
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, wrapper)

    original_generate = GenerationMixin.generate

    def traced_generate(self: GenerationMixin, *args: Any, **kwargs: Any) -> Any:
        print("\nENTER: GenerationMixin.generate")
        print(f"assistant_model passed: {kwargs.get('assistant_model') is not None}")
        print(f"do_sample: {kwargs.get('do_sample')}")
        print(f"max_new_tokens: {kwargs.get('max_new_tokens', max_new_tokens)}")
        return original_generate(self, *args, **kwargs)

    original_get_generation_mode = GenerationConfig.get_generation_mode

    def traced_get_generation_mode(self: GenerationConfig, *args: Any, **kwargs: Any) -> Any:
        mode = original_get_generation_mode(self, *args, **kwargs)
        print("\nGENERATION MODE:")
        print(f"mode = {getattr(mode, 'name', mode)}")
        return mode

    original_assisted_decoding = GenerationMixin._assisted_decoding

    def traced_assisted_decoding(self: GenerationMixin, input_ids: torch.LongTensor, *args: Any, **kwargs: Any) -> Any:
        print("\nENTER: _assisted_decoding")
        print(f"input_ids shape: {shape_of(input_ids)}")
        print(f"past_key_values present: {cache_presence(kwargs)}")
        print(f"cache_position: {cache_position_text(kwargs)}")
        return original_assisted_decoding(self, input_ids, *args, **kwargs)

    original_get_candidates = AssistedCandidateGenerator.get_candidates

    def traced_get_candidates(self: AssistedCandidateGenerator, input_ids: torch.LongTensor) -> Any:
        print("\nENTER: candidate_generator.get_candidates")
        print(f"input_ids shape before draft: {shape_of(input_ids)}")
        candidate_input_ids, candidate_logits = original_get_candidates(self, input_ids)
        num_candidate_tokens = candidate_input_ids.shape[-1] - input_ids.shape[-1]
        print(f"candidate_input_ids shape after draft: {shape_of(candidate_input_ids)}")
        print(f"candidate logits shape: {shape_of(candidate_logits)}")
        print(f"num candidate tokens: {num_candidate_tokens}")
        return candidate_input_ids, candidate_logits

    original_update_candidate_strategy = AssistedCandidateGenerator.update_candidate_strategy

    def traced_update_candidate_strategy(
        self: AssistedCandidateGenerator,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        num_matches: int,
    ) -> Any:
        print("\nACCEPTANCE / UPDATE")
        print(f"accepted tokens if accessible: {num_matches}")
        print(f"sequence length after update: {input_ids.shape[-1]}")
        return original_update_candidate_strategy(self, input_ids, scores, num_matches)

    target_forward = target.forward
    assistant_forward = assistant.forward

    def traced_target_forward(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        print("\nTARGET VERIFY FORWARD")
        print(f"input_ids shape: {shape_of(input_ids)}")
        print(f"past_key_values is None: {kwargs.get('past_key_values') is None}")
        print(f"cache_position: {cache_position_text(kwargs)}")
        outputs = target_forward(*args, **kwargs)
        print(f"logits shape: {shape_of(outputs.logits)}")
        return outputs

    def traced_assistant_forward(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        print("\nASSISTANT FORWARD")
        print(f"input_ids shape: {shape_of(input_ids)}")
        print(f"past_key_values is None: {kwargs.get('past_key_values') is None}")
        print(f"cache_position: {cache_position_text(kwargs)}")
        outputs = assistant_forward(*args, **kwargs)
        print(f"logits shape: {shape_of(outputs.logits)}")
        return outputs

    replace(GenerationMixin, "generate", traced_generate)
    replace(GenerationConfig, "get_generation_mode", traced_get_generation_mode)
    replace(GenerationMixin, "_assisted_decoding", traced_assisted_decoding)
    replace(AssistedCandidateGenerator, "get_candidates", traced_get_candidates)
    replace(AssistedCandidateGenerator, "update_candidate_strategy", traced_update_candidate_strategy)
    replace(target, "forward", traced_target_forward)
    replace(assistant, "forward", traced_assistant_forward)

    try:
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device={device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(args.target_model).to(device).eval()
    assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model).to(device).eval()
    assistant.generation_config.num_assistant_tokens = args.num_assistant_tokens

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print("\n=== NORMAL TARGET GREEDY GENERATION ===")
    with torch.inference_mode():
        greedy_ids = target.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"greedy decoded: {tokenizer.decode(greedy_ids[0], skip_special_tokens=True)}")
    print(f"greedy token ids: {greedy_ids[0].tolist()}")

    print("\n=== HUGGING FACE ASSISTED GENERATION TRACE ===")
    with patch_hf_internals(target, assistant, args.max_new_tokens):
        with torch.inference_mode():
            assisted_ids = target.generate(
                **inputs,
                assistant_model=assistant,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    print("\n=== OUTPUT COMPARISON ===")
    print(f"assisted decoded: {tokenizer.decode(assisted_ids[0], skip_special_tokens=True)}")
    print(f"assisted token ids: {assisted_ids[0].tolist()}")
    print(f"token IDs match exactly: {greedy_ids[0].tolist() == assisted_ids[0].tolist()}")
    print("\nNote: some acceptance/cache locals live inside HF stack frames. This script prints the closest")
    print("observable values exposed through wrappers around generation and candidate-generator methods.")


if __name__ == "__main__":
    main()
