#!/usr/bin/env python
"""Print local Hugging Face source paths and useful search terms."""

from __future__ import annotations

import inspect

import transformers.generation.candidate_generator as candidate_generator
import transformers.generation.configuration_utils as configuration_utils
import transformers.generation.utils as generation_utils
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin


def source_path(obj: object) -> str:
    unwrapped = inspect.unwrap(obj)
    return inspect.getsourcefile(unwrapped) or "<source not found>"


def main() -> None:
    print("Local Hugging Face source files")
    print("-------------------------------")
    print(f"transformers.generation.utils: {generation_utils.__file__}")
    print(f"GenerationMixin.generate:      {source_path(GenerationMixin.generate)}")
    print(f"GenerationMixin._assisted_decoding: {source_path(GenerationMixin._assisted_decoding)}")
    print(f"transformers.generation.configuration_utils: {configuration_utils.__file__}")
    print(f"GenerationConfig.get_generation_mode: {source_path(GenerationConfig.get_generation_mode)}")
    print(f"transformers.generation.candidate_generator: {candidate_generator.__file__}")

    print("\nInspect these terms")
    print("-------------------")
    print("In generation/utils.py:")
    for term in [
        "def generate",
        "assistant_model",
        "GenerationMode.ASSISTED_GENERATION",
        "_assisted_decoding",
        "candidate_generator",
        "get_candidates",
        "candidate_input_ids",
        "past_key_values",
        "cache_position",
        "_update_model_kwargs_for_generation",
    ]:
        print(f"  - {term}")

    print("\nIn generation/configuration_utils.py:")
    for term in [
        "get_generation_mode",
        "ASSISTED_GENERATION",
        "assistant_model",
        "prompt_lookup_num_tokens",
    ]:
        print(f"  - {term}")

    print("\nIn candidate_generator files:")
    for term in [
        "AssistedCandidateGenerator",
        "CandidateGenerator",
        "get_candidates",
        "update_candidate_strategy",
    ]:
        print(f"  - {term}")


if __name__ == "__main__":
    main()
