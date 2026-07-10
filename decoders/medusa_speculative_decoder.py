#!/usr/bin/env python
"""First-principles Medusa-style speculative decoding.

Medusa adds several trained prediction heads on top of the target model hidden
state. This file keeps the model forward as a black-box operator, but implements
the Medusa mechanics around that operator: Medusa-head candidate construction,
tree attention verification, posterior acceptance, and KV-cache selection.

The slow path verifier is intentionally still present as a correctness oracle.
It verifies each path with ordinary target forwards and is much easier to
inspect, but it is not the paper's speedup mechanism.

You need trained Medusa heads for useful generations. The `MedusaHeadStack`
class below is the architecture shell; random heads are only good for tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import torch
from torch import nn
from transformers import PretrainedConfig

from decoders.first_principles_speculative_decoder import (
    LogitTopK,
    decision_logits,
    format_top_logits,
    model_device,
    model_vocab_size,
    normalize_eos_token_ids,
    should_stop,
    stop_reason_for,
    summarize_top_logits,
    target_predictions_for_draft,
    timed_operation,
    validate_generate_inputs,
)


@dataclass
class MedusaBuffers:
    choices: list[list[int]]
    tree_indices: torch.Tensor
    retrieve_indices: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    top_k: int


@dataclass
class MedusaCandidates:
    candidate_pool: torch.Tensor
    tree_candidates: torch.Tensor
    candidate_paths: torch.Tensor
    target_next_token: int
    medusa_top_tokens: list[list[int]]


@dataclass
class MedusaVerification:
    path_index: int
    path_tokens: torch.Tensor
    target_predictions: torch.Tensor
    target_top_logits: list[LogitTopK]
    accepted_count: int
    rejected_at: int | None
    appended_tokens: torch.Tensor
    next_target_logits: torch.Tensor | None = None
    next_medusa_logits: torch.Tensor | None = None
    past_key_values: Any | None = None
    cache_updated: bool = False
    verification_method: str = "slow"
    acceptance_mode: str = "greedy"
    fallback_reason: str | None = None


@dataclass
class MedusaStepTrace:
    step: int
    prefix_length: int
    remaining_new_tokens: int
    candidate_path_count: int
    target_next_token: int
    medusa_top_tokens: list[list[int]]
    selected_path_index: int
    selected_path_tokens: list[int]
    target_predictions: list[int]
    target_top_logits: list[LogitTopK]
    accepted_count: int
    rejected_at: int | None
    appended_tokens: list[int]
    output_length: int
    stop_reason: str | None
    verification_method: str = "slow"
    acceptance_mode: str = "greedy"
    tree_node_count: int = 0
    cache_updated: bool = False
    fallback_reason: str | None = None


AcceptanceMode = Literal["greedy", "typical", "nucleus"]
VerifierMode = Literal["tree", "slow"]


VICUNA_7B_STAGE2_CHOICES: list[tuple[int, ...]] = [
    (0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2),
    (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2),
    (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3),
    (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7),
    (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0),
    (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1),
    (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,),
    (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2),
    (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3),
    (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0),
    (7, 0), (1, 4),
]

VICUNA_13B_STAGE2_CHOICES: list[tuple[int, ...]] = [
    (0,), (0, 0), (1,), (0, 0, 0), (0, 1), (1, 0), (2,), (0, 2),
    (0, 0, 1), (0, 1, 0), (3,), (0, 3), (2, 0), (0, 0, 2),
    (0, 0, 0, 0), (0, 4), (1, 0, 0), (1, 1), (4,), (0, 0, 3),
    (0, 5), (0, 2, 0), (5,), (3, 0), (0, 1, 1), (0, 6),
    (0, 0, 4), (0, 0, 0, 1), (0, 7), (0, 0, 5), (1, 2),
    (0, 0, 1, 0), (0, 3, 0), (1, 0, 1), (4, 0), (0, 0, 6),
    (0, 8), (2, 0, 0), (0, 9), (6,), (7,), (2, 1), (5, 0),
    (0, 1, 2), (0, 0, 0, 2), (8,), (0, 4, 0), (0, 1, 0, 0),
    (0, 2, 1), (0, 0, 7), (1, 1, 0), (1, 3), (0, 0, 2, 0),
    (9,), (0, 0, 8), (0, 5, 0), (0, 0, 0, 3), (0, 0, 9),
    (0, 1, 3), (1, 0, 2), (0, 0, 1, 1), (3, 0, 0), (1, 0, 0, 0),
]

ZEPHYR_STAGE2_CHOICES: list[tuple[int, ...]] = [
    (0,), (0, 0), (1,), (0, 1), (2,), (0, 0, 0), (1, 0), (0, 2),
    (3,), (0, 3), (4,), (2, 0), (0, 0, 1), (0, 4), (5,), (0, 5),
    (0, 1, 0), (1, 1), (6,), (0, 0, 2), (3, 0), (0, 6), (7,),
    (0, 7), (0, 8), (0, 0, 3), (1, 0, 0), (0, 9), (0, 2, 0),
    (1, 2), (4, 0), (8,), (9,), (2, 1), (0, 1, 1), (0, 0, 4),
    (0, 0, 0, 0), (5, 0), (0, 3, 0), (1, 3), (0, 0, 5),
    (0, 0, 6), (6, 0), (2, 0, 0), (1, 0, 1), (0, 1, 2),
    (0, 4, 0), (1, 4), (3, 1), (2, 2), (0, 0, 7), (7, 0),
    (0, 2, 1), (0, 0, 8), (0, 1, 3), (0, 5, 0), (1, 5),
    (0, 0, 9), (1, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (4, 1),
    (2, 3),
]


class MedusaResidualBlock(nn.Module):
    """Small residual block used by Medusa heads."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.activation(self.linear(hidden_states))


class MedusaHeadStack(nn.Module):
    """A stack of Medusa prediction heads.

    The shape returned is `[num_heads, batch, sequence, vocab]`, matching the
    official Medusa utility functions.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int, num_layers: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    *[MedusaResidualBlock(hidden_size) for _ in range(num_layers)],
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=0)


def _load_pretrained_config(path_or_repo_id: str) -> PretrainedConfig:
    return PretrainedConfig.from_pretrained(path_or_repo_id)


def infer_medusa_head_shape(
    target_model: torch.nn.Module,
    medusa_heads_path: str,
    medusa_num_heads: int | None = None,
    medusa_num_layers: int | None = None,
) -> tuple[int, int, int, int]:
    """Infer `(hidden_size, vocab_size, num_heads, num_layers)` for Medusa heads."""
    config = getattr(target_model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        output_embeddings = target_model.get_output_embeddings()
        hidden_size = output_embeddings.weight.shape[-1]
    vocab_size = model_vocab_size(target_model)
    if vocab_size is None:
        raise ValueError("Could not infer target model vocabulary size.")

    try:
        medusa_config = _load_pretrained_config(medusa_heads_path)
    except Exception:
        medusa_config = None

    num_heads = medusa_num_heads
    if num_heads is None and medusa_config is not None:
        num_heads = getattr(medusa_config, "medusa_num_heads", None)
    if num_heads is None:
        num_heads = 5

    num_layers = medusa_num_layers
    if num_layers is None and medusa_config is not None:
        num_layers = getattr(medusa_config, "medusa_num_layers", None)
    if num_layers is None:
        num_layers = 1

    return int(hidden_size), int(vocab_size), int(num_heads), int(num_layers)


def _resolve_medusa_head_file(path_or_repo_id: str, filename: str = "medusa_lm_head.pt") -> str:
    path = Path(path_or_repo_id).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidate = path / filename
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Could not find {filename} inside {path}.")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(path_or_repo_id, filename)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _load_state_dict_best_effort(
    medusa_heads: MedusaHeadStack,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Load official Medusa `medusa_lm_head.pt` keys into `MedusaHeadStack`."""
    candidates: list[tuple[nn.Module, dict[str, torch.Tensor]]] = [
        (medusa_heads, state_dict),
        (medusa_heads, _strip_prefix(state_dict, "medusa_head.")),
        (medusa_heads, _strip_prefix(state_dict, "heads.")),
        (medusa_heads.heads, state_dict),
        (medusa_heads.heads, _strip_prefix(state_dict, "medusa_head.")),
        (medusa_heads.heads, _strip_prefix(state_dict, "heads.")),
    ]

    best_missing: list[str] | None = None
    best_unexpected: list[str] | None = None
    best_module: nn.Module | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_score: int | None = None

    for module, candidate_state_dict in candidates:
        result = module.load_state_dict(candidate_state_dict, strict=False)
        score = len(result.missing_keys) + len(result.unexpected_keys)
        if best_score is None or score < best_score:
            best_score = score
            best_missing = list(result.missing_keys)
            best_unexpected = list(result.unexpected_keys)
            best_module = module
            best_state_dict = candidate_state_dict

    if best_module is None or best_state_dict is None or best_missing is None or best_unexpected is None:
        raise RuntimeError("Could not load Medusa head state dict.")

    result = best_module.load_state_dict(best_state_dict, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def _infer_head_shape_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[int | None, int | None]:
    """Infer `(num_heads, num_layers)` from official sequential head keys."""
    head_indices: set[int] = set()
    layer_indices_by_head: dict[int, set[int]] = {}
    for key, value in state_dict.items():
        del value
        parts = key.split(".")
        if parts and parts[0] in {"medusa_head", "heads"}:
            parts = parts[1:]
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        head_index = int(parts[0])
        sequential_index = int(parts[1])
        head_indices.add(head_index)
        layer_indices_by_head.setdefault(head_index, set()).add(sequential_index)

    num_heads = max(head_indices) + 1 if head_indices else None
    num_layers = None
    if layer_indices_by_head:
        # Official heads are [residual block] * N + [linear lm head].
        max_residual_layers = max(max(indices) for indices in layer_indices_by_head.values())
        num_layers = max(1, max_residual_layers)
    return num_heads, num_layers


def load_official_medusa_heads(
    target_model: torch.nn.Module,
    medusa_heads_path: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    medusa_num_heads: int | None = None,
    medusa_num_layers: int | None = None,
    filename: str = "medusa_lm_head.pt",
) -> MedusaHeadStack:
    """Load public FasterDecoding-style Medusa heads for a target model.

    `medusa_heads_path` can be a Hugging Face repo id, a local directory
    containing `medusa_lm_head.pt`, or the checkpoint file itself.
    """
    hidden_size, vocab_size, num_heads, num_layers = infer_medusa_head_shape(
        target_model,
        medusa_heads_path,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
    )
    checkpoint_path = _resolve_medusa_head_file(medusa_heads_path, filename=filename)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected Medusa checkpoint to contain a state dict, got {type(state_dict).__name__}.")

    checkpoint_heads, checkpoint_layers = _infer_head_shape_from_state_dict(state_dict)
    if medusa_num_heads is None and checkpoint_heads is not None and checkpoint_heads != num_heads:
        print(
            f"Using Medusa head count from checkpoint: {checkpoint_heads} "
            f"(config reported {num_heads}).",
            flush=True,
        )
        num_heads = checkpoint_heads
    if medusa_num_layers is None and checkpoint_layers is not None and checkpoint_layers != num_layers:
        print(
            f"Using Medusa layer count from checkpoint: {checkpoint_layers} "
            f"(config reported {num_layers}).",
            flush=True,
        )
        num_layers = checkpoint_layers

    medusa_heads = MedusaHeadStack(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    missing_keys, unexpected_keys = _load_state_dict_best_effort(medusa_heads, state_dict)
    if missing_keys or unexpected_keys:
        print(
            "Loaded Medusa heads with non-strict key match: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}",
            flush=True,
        )

    if device is None:
        device = model_device(target_model, torch.empty(1, dtype=torch.long))
    if dtype is None:
        try:
            dtype = next(target_model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
    return medusa_heads.to(device=device, dtype=dtype).eval()


def linear_medusa_choices(num_heads: int) -> list[list[int]]:
    """Default to one greedy chain through the Medusa heads."""
    return [[0] * depth for depth in range(1, num_heads + 1)]


def small_medusa_tree_choices(num_heads: int, top_k: int) -> list[list[int]]:
    """A compact branching tree useful for tracing."""
    choices: list[list[int]] = []
    first_layer = min(top_k, 3)
    for first in range(first_layer):
        choices.append([first])
    if num_heads >= 2:
        for first in range(min(top_k, 2)):
            for second in range(min(top_k, 2)):
                choices.append([first, second])
    if num_heads >= 3:
        choices.append([0, 0, 0])
        if top_k > 1:
            choices.append([0, 0, 1])
    return choices


def official_medusa_choices(name: str, num_heads: int, top_k: int) -> list[list[int]]:
    """Return official sparse Medusa choice presets clipped to available heads/top-k."""
    normalized = name.lower().replace("_", "-")
    if normalized in {"vicuna-7b-stage2", "official-vicuna-7b", "vicuna-7b"}:
        choices = VICUNA_7B_STAGE2_CHOICES
    elif normalized in {"vicuna-13b-stage2", "official-vicuna-13b", "vicuna-13b"}:
        choices = VICUNA_13B_STAGE2_CHOICES
    elif normalized in {"zephyr-stage2", "official-zephyr", "zephyr"}:
        choices = ZEPHYR_STAGE2_CHOICES
    else:
        raise ValueError(f"Unknown official Medusa choices preset: {name}")

    clipped = [
        list(path)
        for path in choices
        if len(path) <= num_heads and all(choice < top_k for choice in path)
    ]
    if not clipped:
        raise ValueError(
            f"Preset {name!r} has no paths compatible with num_heads={num_heads}, top_k={top_k}."
        )
    return clipped


def _validate_tree_choices(choices: Sequence[Sequence[int]], top_k: int) -> list[list[int]]:
    if not choices:
        raise ValueError("medusa choices must contain at least one path.")
    normalized = [list(path) for path in choices]
    choice_set = {tuple(path) for path in normalized}
    for path in normalized:
        if not path:
            raise ValueError("medusa choice paths must be non-empty.")
        for value in path:
            if value < 0 or value >= top_k:
                raise ValueError(f"choice value {value} is outside [0, {top_k}).")
        for depth in range(1, len(path)):
            prefix = tuple(path[:depth])
            if prefix not in choice_set:
                raise ValueError(f"medusa choices must include prefix path {list(prefix)}.")
    return sorted(normalized, key=lambda item: (len(item), item))


def _pad_path(path: list[int], length: int, pad_value: int = -1) -> list[int]:
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(
    medusa_choices: Sequence[Sequence[int]],
    top_k: int = 10,
    device: torch.device | str = "cpu",
) -> MedusaBuffers:
    """Build tree-index, retrieval, position, and attention buffers."""
    sorted_choices = _validate_tree_choices(medusa_choices, top_k)
    tree_len = len(sorted_choices) + 1

    depth_counts: list[int] = []
    for path in sorted_choices:
        depth = len(path)
        while len(depth_counts) < depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1

    attention_mask = torch.eye(tree_len, dtype=torch.bool)
    attention_mask[:, 0] = True
    start = 0
    for depth, count in enumerate(depth_counts, start=1):
        for offset in range(count):
            path = sorted_choices[start + offset]
            ancestor_indices = [sorted_choices.index(path[:prefix_len]) + 1 for prefix_len in range(1, depth)]
            if ancestor_indices:
                attention_mask[start + offset + 1, ancestor_indices] = True
        start += count

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for depth, count in enumerate(depth_counts):
        for offset in range(count):
            path = sorted_choices[start + offset]
            tree_indices[start + offset + 1] = path[-1] + top_k * depth + 1
            position_ids[start + offset + 1] = depth + 1
        start += count

    retrieve_paths: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for path in reversed(sorted_choices):
        path_indices: list[int] = [0]
        already_seen = tuple(path) in seen
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            path_indices.append(sorted_choices.index(prefix) + 1)
            seen.add(tuple(prefix))
        if not already_seen:
            retrieve_paths.append(path_indices)

    max_len = max(len(path) for path in retrieve_paths)
    retrieve_indices = torch.tensor([_pad_path(path, max_len) for path in retrieve_paths], dtype=torch.long)

    return MedusaBuffers(
        choices=sorted_choices,
        tree_indices=tree_indices.to(device),
        retrieve_indices=retrieve_indices.to(device),
        position_ids=position_ids.to(device),
        attention_mask=attention_mask[None, None].to(device),
        top_k=top_k,
    )


def _extract_forward_result(outputs: Any) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, (tuple, list)):
        logits = outputs[0]
    else:
        logits = outputs
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            hidden_states = outputs[-1]
        else:
            raise ValueError("Target model must return hidden states for Medusa decoding.")
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[-1]
    past_key_values = getattr(outputs, "past_key_values", None)
    return logits, hidden_states, past_key_values


def forward_target_with_hidden_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    use_cache: bool = False,
    progress_label: str | None = None,
    heartbeat_seconds: float = 0.0,
    strict_kwargs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
    """Run the target model and return `(logits, last_hidden_states, cache)`."""
    device = input_ids.device
    context = (
        timed_operation(progress_label, device, heartbeat_seconds)
        if progress_label is not None or heartbeat_seconds > 0
        else None
    )
    if context is None:
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=use_cache,
            )
        except TypeError:
            if strict_kwargs:
                raise
            outputs = model(input_ids, output_hidden_states=True)
    else:
        with context:
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=use_cache,
                )
            except TypeError:
                if strict_kwargs:
                    raise
                outputs = model(input_ids, output_hidden_states=True)
    return _extract_forward_result(outputs)


def forward_target_with_hidden(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the target model and return `(logits, last_hidden_states)`."""
    try:
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    except TypeError:
        outputs = model(input_ids, output_hidden_states=True)
    logits, hidden_states, _ = _extract_forward_result(outputs)
    return logits, hidden_states


def normalize_medusa_logits(raw_logits: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize Medusa-head outputs to `[heads, batch, sequence, vocab]`."""
    if isinstance(raw_logits, (list, tuple)):
        raw_logits = torch.stack(list(raw_logits), dim=0)
    if not isinstance(raw_logits, torch.Tensor):
        raise TypeError("Medusa heads must return a tensor or a list of tensors.")

    if raw_logits.ndim == 4:
        if raw_logits.shape[1] == hidden_states.shape[0]:
            return raw_logits
        if raw_logits.shape[0] == hidden_states.shape[0]:
            return raw_logits.permute(1, 0, 2, 3).contiguous()
    if raw_logits.ndim == 3:
        if raw_logits.shape[1] == hidden_states.shape[0]:
            return raw_logits.unsqueeze(2)
        if raw_logits.shape[0] == hidden_states.shape[0]:
            return raw_logits.permute(1, 0, 2).unsqueeze(2).contiguous()

    raise ValueError(
        "Unsupported Medusa logits shape. Expected [heads,batch,seq,vocab], "
        "[batch,heads,seq,vocab], [heads,batch,vocab], or [batch,heads,vocab]."
    )


def _path_row_to_tensor(row: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    valid = row[row >= 0].to(device=device, dtype=dtype)
    return valid.unsqueeze(0)


def _verified_append(
    generated: torch.Tensor,
    draft_tokens: torch.Tensor,
    target_predictions: torch.Tensor,
    remaining: int,
    eos_token_ids: set[int],
    min_length: int,
) -> tuple[int, int | None, torch.Tensor]:
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

    if replacement_token is None:
        appended = draft_tokens
        simulated = torch.cat([generated, appended], dim=-1)
        if appended.shape[-1] < remaining and not should_stop(simulated, eos_token_ids, min_length):
            appended = torch.cat([appended, target_predictions[:, draft_tokens.shape[-1] : draft_tokens.shape[-1] + 1]], dim=-1)
    else:
        appended = torch.cat([draft_tokens[:, :accepted_count], replacement_token], dim=-1)
    return accepted_count, rejected_at, appended[:, :remaining]


def verify_candidate_paths_slow(
    target_model: torch.nn.Module,
    generated: torch.Tensor,
    candidate_paths: torch.Tensor,
    eos_token_ids: set[int],
    min_length: int,
    remaining: int,
    top_k_logits: int = 0,
    progress: bool = False,
    step: int | None = None,
    heartbeat_seconds: float = 0.0,
) -> MedusaVerification:
    """Verify each Medusa path with ordinary target forwards and pick the best."""
    best: MedusaVerification | None = None
    for path_index, row in enumerate(candidate_paths):
        path_tokens = _path_row_to_tensor(row, generated.dtype, generated.device)
        if path_tokens.shape[-1] == 0:
            continue
        if progress:
            print(f"[medusa step {step}] verifying path {path_index}: {path_tokens[0].tolist()}", flush=True)
        target_result = target_predictions_for_draft(
            target_model,
            generated,
            path_tokens,
            eos_token_ids,
            min_length,
            top_k_logits=top_k_logits,
            progress=False,
            step=step,
            heartbeat_seconds=heartbeat_seconds,
        )
        accepted_count, rejected_at, appended = _verified_append(
            generated,
            path_tokens,
            target_result.predictions,
            remaining,
            eos_token_ids,
            min_length,
        )
        candidate = MedusaVerification(
            path_index=path_index,
            path_tokens=path_tokens,
            target_predictions=target_result.predictions,
            target_top_logits=target_result.top_logits,
            accepted_count=accepted_count,
            rejected_at=rejected_at,
            appended_tokens=appended,
        )
        if best is None or candidate.accepted_count > best.accepted_count:
            best = candidate

    if best is None:
        empty = torch.empty((generated.shape[0], 0), dtype=generated.dtype, device=generated.device)
        target_result = target_predictions_for_draft(target_model, generated, empty, eos_token_ids, min_length)
        return MedusaVerification(
            path_index=-1,
            path_tokens=empty,
            target_predictions=target_result.predictions,
            target_top_logits=target_result.top_logits,
            accepted_count=0,
            rejected_at=None,
            appended_tokens=target_result.predictions[:, :1],
        )
    return best


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _additive_tree_attention_mask(
    prefix_length: int,
    buffers: MedusaBuffers,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the 4D additive mask for Medusa tree attention.

    The left block exposes all prefix KV entries. The right block exposes only
    a tree node's own root/ancestor nodes, matching the official Medusa mask.
    """
    tree_allowed = buffers.attention_mask.to(device=device, dtype=torch.bool)
    tree_len = tree_allowed.shape[-1]
    allowed = torch.zeros((1, 1, tree_len, prefix_length + tree_len), dtype=torch.bool, device=device)
    if prefix_length > 0:
        allowed[:, :, :, :prefix_length] = True
    allowed[:, :, :, prefix_length:] = tree_allowed

    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask


def _safe_retrieve_indices(retrieve_indices: torch.Tensor, pad_index: int) -> torch.Tensor:
    return torch.where(
        retrieve_indices < 0,
        torch.full_like(retrieve_indices, pad_index),
        retrieve_indices,
    )


def _reorder_tree_logits(tree_logits: torch.Tensor, retrieve_indices: torch.Tensor) -> torch.Tensor:
    """Map `[1, tree_len, vocab]` logits into `[paths, path_len, vocab]`."""
    flat_logits = tree_logits[0]
    pad = torch.zeros((1, flat_logits.shape[-1]), dtype=flat_logits.dtype, device=flat_logits.device)
    extended = torch.cat([flat_logits, pad], dim=0)
    safe_retrieve = _safe_retrieve_indices(retrieve_indices, flat_logits.shape[0])
    return extended[safe_retrieve]


def _reorder_tree_medusa_logits(tree_medusa_logits: torch.Tensor, retrieve_indices: torch.Tensor) -> torch.Tensor:
    """Map `[heads, 1, tree_len, vocab]` to `[heads, paths, path_len, vocab]`."""
    flat_logits = tree_medusa_logits[:, 0]
    pad = torch.zeros(
        (flat_logits.shape[0], 1, flat_logits.shape[-1]),
        dtype=flat_logits.dtype,
        device=flat_logits.device,
    )
    extended = torch.cat([flat_logits, pad], dim=1)
    safe_retrieve = _safe_retrieve_indices(retrieve_indices, flat_logits.shape[1])
    return extended[:, safe_retrieve]


def _candidate_path_lengths(candidate_paths: torch.Tensor) -> torch.Tensor:
    return (candidate_paths >= 0).sum(dim=-1)


def _prefix_match_lengths(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] == 0:
        return torch.zeros(mask.shape[0], dtype=torch.long, device=mask.device)
    return torch.cumprod(mask.to(torch.long), dim=-1).sum(dim=-1)


def _nucleus_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cum_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = torch.zeros_like(sorted_remove).scatter(dim=-1, index=sorted_indices, src=sorted_remove)
    filtered_logits = logits.masked_fill(remove, -torch.inf)
    return torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1).squeeze(-1)


def _select_root_token(
    logits: torch.Tensor,
    acceptance_mode: AcceptanceMode,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
    top_p: float,
) -> torch.Tensor:
    """Select the original-LM root token for the next Medusa tree."""
    next_logits = logits[:, -1, :]
    if acceptance_mode == "greedy" or temperature <= 0:
        return torch.argmax(next_logits, dim=-1)
    scaled = next_logits / temperature
    if acceptance_mode == "nucleus":
        return _nucleus_sample(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    threshold = torch.minimum(
        torch.full_like(entropy, posterior_threshold),
        torch.exp(-entropy) * posterior_alpha,
    )
    filtered_logits = scaled.masked_fill(probs < threshold.unsqueeze(-1), -torch.inf)
    return torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1).squeeze(-1)


def generate_medusa_candidates(
    target_logits: torch.Tensor,
    medusa_logits: torch.Tensor,
    buffers: MedusaBuffers,
    acceptance_mode: AcceptanceMode = "greedy",
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    top_p: float = 0.8,
) -> MedusaCandidates:
    """Use target next-token logits and Medusa top-k heads to build paths."""
    if target_logits.shape[0] != 1:
        raise ValueError("This traceable Medusa decoder currently supports batch size 1.")

    num_heads = medusa_logits.shape[0]
    max_depth = int(buffers.position_ids.max().item())
    if max_depth > num_heads:
        raise ValueError(f"Medusa choices require depth {max_depth}, but only {num_heads} heads were provided.")

    top_k = min(buffers.top_k, medusa_logits.shape[-1])
    if top_k != buffers.top_k:
        raise ValueError("top_k is larger than the Medusa head vocabulary.")

    target_next = _select_root_token(
        target_logits,
        acceptance_mode=acceptance_mode,
        temperature=temperature,
        posterior_threshold=posterior_threshold,
        posterior_alpha=posterior_alpha,
        top_p=top_p,
    )
    top_token_ids = torch.topk(medusa_logits[:, 0, -1, :], k=buffers.top_k, dim=-1).indices
    candidate_pool = torch.cat([target_next, top_token_ids.reshape(-1)], dim=0)

    if int(buffers.tree_indices.max().item()) >= candidate_pool.shape[0]:
        raise ValueError("Medusa tree indices exceed the candidate pool. Check choices/top_k/head count.")

    tree_candidates = candidate_pool[buffers.tree_indices]
    pad_index = tree_candidates.shape[0]
    extended = torch.cat(
        [tree_candidates, torch.tensor([-1], dtype=tree_candidates.dtype, device=tree_candidates.device)],
        dim=0,
    )
    safe_retrieve = _safe_retrieve_indices(buffers.retrieve_indices, pad_index)
    candidate_paths = extended[safe_retrieve]
    return MedusaCandidates(
        candidate_pool=candidate_pool,
        tree_candidates=tree_candidates.unsqueeze(0),
        candidate_paths=candidate_paths,
        target_next_token=int(target_next.item()),
        medusa_top_tokens=[[int(token) for token in row] for row in top_token_ids.detach().cpu().tolist()],
    )


def evaluate_posterior(
    path_logits: torch.Tensor,
    candidate_paths: torch.Tensor,
    acceptance_mode: AcceptanceMode,
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    top_p: float = 0.8,
) -> tuple[int, int]:
    """Official-style Medusa posterior evaluation.

    Returns `(best_candidate_index, accepted_future_tokens)`. The accepted
    length does not include the root token predicted by the original LM head.
    """
    valid_future = candidate_paths[:, 1:] >= 0
    if candidate_paths.shape[-1] <= 1:
        return 0, 0

    logits = path_logits[:, :-1, :]
    if acceptance_mode == "greedy" or temperature <= 0:
        predicted = torch.argmax(logits, dim=-1)
        posterior_mask = (candidate_paths[:, 1:] == predicted) & valid_future
        accept_lengths = _prefix_match_lengths(posterior_mask)
        best = int(torch.argmax(accept_lengths).item())
        return best, int(accept_lengths[best].item())

    scaled = logits / temperature
    if acceptance_mode == "nucleus":
        sample_shape = scaled.shape[:-1]
        sampled = _nucleus_sample(scaled.reshape(-1, scaled.shape[-1]), top_p).reshape(sample_shape)
        posterior_mask = (candidate_paths[:, 1:] == sampled) & valid_future
        accept_lengths = _prefix_match_lengths(posterior_mask)
        best = int(torch.argmax(accept_lengths).item())
        return best, int(accept_lengths[best].item())

    probs = torch.softmax(scaled, dim=-1)
    candidate_probs = torch.gather(probs, dim=-1, index=candidate_paths[:, 1:].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    threshold = torch.minimum(
        torch.full_like(entropy, posterior_threshold),
        torch.exp(-entropy) * posterior_alpha,
    )
    posterior_mask = (candidate_probs > threshold) & valid_future
    accept_lengths = _prefix_match_lengths(posterior_mask)
    accept_length = int(accept_lengths.max().item())
    if accept_length == 0:
        return 0, 0

    best_indices = torch.where(accept_lengths == accept_length)[0]
    likelihood = torch.sum(torch.log(candidate_probs[best_indices, :accept_length] + 1e-12), dim=-1)
    best = int(best_indices[torch.argmax(likelihood)].item())
    return best, accept_length


def _copy_selected_tree_cache(
    past_key_values: Any,
    prefix_length: int,
    selected_tree_indices: torch.Tensor,
) -> bool:
    """Copy selected tree KV states into linear continuation positions.

    This mirrors the official Medusa `KVCache.copy(...)` idea for the common
    Transformers `DynamicCache` layout. Returns False when the cache class is
    unfamiliar so the caller can safely fall back to recomputation.
    """
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return False
    if not layers:
        return False

    selected_tree_indices = selected_tree_indices.to(dtype=torch.long)
    accepted_count = int(selected_tree_indices.shape[-1])
    if accepted_count == 0:
        return True

    for layer in layers:
        if getattr(layer, "is_sliding", False):
            return False
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
            return False
        if keys.shape[-2] < prefix_length + int(selected_tree_indices.max().item()) + 1:
            return False

    for layer in layers:
        keys = layer.keys
        values = layer.values
        source_indices = prefix_length + selected_tree_indices.to(keys.device)
        selected_keys = torch.index_select(keys, dim=-2, index=source_indices).clone()
        selected_values = torch.index_select(values, dim=-2, index=source_indices).clone()
        layer.keys[..., prefix_length : prefix_length + accepted_count, :] = selected_keys
        layer.values[..., prefix_length : prefix_length + accepted_count, :] = selected_values
        layer.keys = layer.keys[..., : prefix_length + accepted_count, :]
        layer.values = layer.values[..., : prefix_length + accepted_count, :]
    return True


def verify_candidate_tree(
    target_model: torch.nn.Module,
    medusa_heads: torch.nn.Module,
    generated: torch.Tensor,
    candidates: MedusaCandidates,
    buffers: MedusaBuffers,
    past_key_values: Any | None,
    eos_token_ids: set[int],
    min_length: int,
    remaining: int,
    target_logits: torch.Tensor,
    top_k_logits: int = 0,
    progress: bool = False,
    step: int | None = None,
    heartbeat_seconds: float = 0.0,
    acceptance_mode: AcceptanceMode = "greedy",
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    top_p: float = 0.8,
    use_kv_cache: bool = True,
) -> MedusaVerification:
    """Verify all Medusa candidate paths in one tree-attention forward."""
    del eos_token_ids, min_length
    prefix_length = generated.shape[-1]
    tree_len = candidates.tree_candidates.shape[-1]
    if not use_kv_cache or past_key_values is None:
        raise RuntimeError("tree verifier requires a prefix KV cache; use verifier='slow' for no-cache inspection.")
    if progress:
        print(
            f"[medusa step {step}] tree verify nodes={tree_len} "
            f"paths={candidates.candidate_paths.shape[0]} prefix_len={prefix_length}",
            flush=True,
        )

    dtype = _model_dtype(target_model)
    attention_mask = _additive_tree_attention_mask(prefix_length, buffers, dtype=dtype, device=generated.device)
    position_ids = (buffers.position_ids + prefix_length).unsqueeze(0).to(generated.device)
    tree_logits, hidden_states, next_cache = forward_target_with_hidden_cache(
        target_model,
        candidates.tree_candidates,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        progress_label=f"[step {step}] target tree verify forward" if progress or heartbeat_seconds > 0 else None,
        heartbeat_seconds=heartbeat_seconds,
        strict_kwargs=True,
    )
    tree_medusa_logits = normalize_medusa_logits(medusa_heads(hidden_states), hidden_states)
    path_logits = _reorder_tree_logits(tree_logits, buffers.retrieve_indices)
    path_medusa_logits = _reorder_tree_medusa_logits(tree_medusa_logits, buffers.retrieve_indices)

    best_index, accepted_future = evaluate_posterior(
        path_logits,
        candidates.candidate_paths,
        acceptance_mode=acceptance_mode,
        temperature=temperature,
        posterior_threshold=posterior_threshold,
        posterior_alpha=posterior_alpha,
        top_p=top_p,
    )

    path_length = int(_candidate_path_lengths(candidates.candidate_paths[best_index : best_index + 1])[0].item())
    requested_append_count = min(path_length, accepted_future + 1)
    append_count = min(requested_append_count, remaining)
    path_tokens = candidates.candidate_paths[best_index, :path_length].unsqueeze(0)
    appended = path_tokens[:, :append_count]
    last_state_index = max(append_count - 1, 0)

    selected_tree_indices = buffers.retrieve_indices[best_index, :append_count]
    cache_updated = False
    cache_after = next_cache
    if next_cache is not None:
        cache_updated = _copy_selected_tree_cache(next_cache, prefix_length, selected_tree_indices)
        if not cache_updated:
            cache_after = None

    root_top_logits = summarize_top_logits(target_logits[:, -1, :], top_k_logits)
    selected_top_logits = [
        summarize_top_logits(path_logits[best_index, idx, :].unsqueeze(0), top_k_logits)
        for idx in range(path_length)
    ]
    predictions = torch.cat(
        [
            torch.tensor([[candidates.target_next_token]], dtype=generated.dtype, device=generated.device),
            torch.argmax(path_logits[best_index, :path_length, :], dim=-1, keepdim=True).T.to(generated.dtype),
        ],
        dim=-1,
    )

    next_target_logits = path_logits[best_index, last_state_index : last_state_index + 1, :].unsqueeze(0)
    next_medusa_logits = path_medusa_logits[:, best_index : best_index + 1, last_state_index : last_state_index + 1, :]
    rejected_at = requested_append_count if requested_append_count < path_length else None

    return MedusaVerification(
        path_index=best_index,
        path_tokens=path_tokens,
        target_predictions=predictions,
        target_top_logits=[root_top_logits, *selected_top_logits],
        accepted_count=append_count,
        rejected_at=rejected_at,
        appended_tokens=appended,
        next_target_logits=next_target_logits,
        next_medusa_logits=next_medusa_logits,
        past_key_values=cache_after,
        cache_updated=cache_updated,
        verification_method="tree",
        acceptance_mode=acceptance_mode,
    )


def generate(
    target_model: torch.nn.Module,
    medusa_heads: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    medusa_choices: Sequence[Sequence[int]] | None = None,
    top_k: int = 5,
    trace_steps: list[MedusaStepTrace] | None = None,
    top_k_logits: int = 0,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
    step_callback: Callable[[MedusaStepTrace, torch.Tensor], None] | None = None,
    verifier: VerifierMode = "tree",
    acceptance_mode: AcceptanceMode = "greedy",
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    top_p: float = 0.8,
    use_kv_cache: bool = True,
    fallback_to_slow: bool = True,
) -> torch.Tensor:
    """Generate with Medusa-style heads.

    `verifier="tree"` follows the official Medusa inference mechanics: one
    target forward over the candidate tree, posterior acceptance, and selected
    KV-cache copy when the cache implementation supports it.

    `verifier="slow"` keeps the earlier path-by-path verifier as an inspectable
    correctness oracle.
    """
    validate_generate_inputs(prompt_token_ids, max_new_tokens, num_assistant_tokens=top_k)
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(target_model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    medusa_heads = medusa_heads.to(device).eval()
    eos_token_ids = normalize_eos_token_ids(target_model, eos_token_id)
    prompt_length = generated.shape[-1]
    buffers: MedusaBuffers | None = None
    state_target_logits: torch.Tensor | None = None
    state_medusa_logits: torch.Tensor | None = None
    state_past_key_values: Any | None = None
    state_uses_cache = verifier == "tree" and use_kv_cache

    with torch.inference_mode():
        if verifier == "tree":
            state_target_logits, hidden_states, state_past_key_values = forward_target_with_hidden_cache(
                target_model,
                generated,
                use_cache=state_uses_cache,
                progress_label="[medusa prefill] target forward" if progress or heartbeat_seconds > 0 else None,
                heartbeat_seconds=heartbeat_seconds,
            )
            state_medusa_logits = normalize_medusa_logits(medusa_heads(hidden_states), hidden_states)

        step = 1
        while generated.shape[-1] - prompt_length < max_new_tokens:
            if should_stop(generated, eos_token_ids, min_length):
                break

            prefix_length = generated.shape[-1]
            remaining = max_new_tokens - (prefix_length - prompt_length)
            if verifier == "tree" and (state_target_logits is None or state_medusa_logits is None):
                state_target_logits, hidden_states, state_past_key_values = forward_target_with_hidden_cache(
                    target_model,
                    generated,
                    use_cache=state_uses_cache,
                    progress_label=f"[medusa step {step}] target re-prefill forward"
                    if progress or heartbeat_seconds > 0
                    else None,
                    heartbeat_seconds=heartbeat_seconds,
                )
                state_medusa_logits = normalize_medusa_logits(medusa_heads(hidden_states), hidden_states)

            if verifier == "tree" and state_target_logits is not None and state_medusa_logits is not None:
                target_logits = state_target_logits
                medusa_logits = state_medusa_logits
            else:
                target_logits, hidden_states = forward_target_with_hidden(target_model, generated)
                medusa_logits = normalize_medusa_logits(medusa_heads(hidden_states), hidden_states)

            if buffers is None:
                choices = list(medusa_choices) if medusa_choices is not None else linear_medusa_choices(medusa_logits.shape[0])
                buffers = generate_medusa_buffers(choices, top_k=top_k, device=generated.device)

            candidates = generate_medusa_candidates(
                target_logits,
                medusa_logits,
                buffers,
                acceptance_mode=acceptance_mode,
                temperature=temperature,
                posterior_threshold=posterior_threshold,
                posterior_alpha=posterior_alpha,
                top_p=top_p,
            )
            if progress:
                print(
                    f"[medusa step {step}] prefix_len={prefix_length} "
                    f"paths={candidates.candidate_paths.shape[0]} "
                    f"tree_nodes={candidates.tree_candidates.shape[-1]} "
                    f"target_next={candidates.target_next_token}",
                    flush=True,
                )

            fallback_reason: str | None = None
            if verifier == "tree":
                try:
                    verification = verify_candidate_tree(
                        target_model,
                        medusa_heads,
                        generated,
                        candidates,
                        buffers,
                        state_past_key_values,
                        eos_token_ids,
                        min_length,
                        remaining,
                        target_logits=target_logits,
                        top_k_logits=top_k_logits,
                        progress=progress,
                        step=step,
                        heartbeat_seconds=heartbeat_seconds,
                        acceptance_mode=acceptance_mode,
                        temperature=temperature,
                        posterior_threshold=posterior_threshold,
                        posterior_alpha=posterior_alpha,
                        top_p=top_p,
                        use_kv_cache=state_uses_cache,
                    )
                except Exception as exc:
                    if not fallback_to_slow:
                        raise
                    fallback_reason = f"{type(exc).__name__}: {exc}"
                    if progress:
                        print(f"[medusa step {step}] tree verifier failed; falling back to slow: {fallback_reason}", flush=True)
                    verification = verify_candidate_paths_slow(
                        target_model,
                        generated,
                        candidates.candidate_paths,
                        eos_token_ids,
                        min_length,
                        remaining,
                        top_k_logits=top_k_logits,
                        progress=progress,
                        step=step,
                        heartbeat_seconds=heartbeat_seconds,
                    )
                    verification.fallback_reason = fallback_reason
            else:
                verification = verify_candidate_paths_slow(
                    target_model,
                    generated,
                    candidates.candidate_paths,
                    eos_token_ids,
                    min_length,
                    remaining,
                    top_k_logits=top_k_logits,
                    progress=progress,
                    step=step,
                    heartbeat_seconds=heartbeat_seconds,
                )
            generated = torch.cat([generated, verification.appended_tokens], dim=-1)

            if verifier == "tree" and verification.verification_method == "tree":
                if (
                    verification.cache_updated
                    and verification.next_target_logits is not None
                    and verification.next_medusa_logits is not None
                ):
                    state_target_logits = verification.next_target_logits
                    state_medusa_logits = verification.next_medusa_logits
                    state_past_key_values = verification.past_key_values
                else:
                    state_target_logits = None
                    state_medusa_logits = None
                    state_past_key_values = None
            elif verifier == "tree":
                state_past_key_values = None
                state_target_logits = None
                state_medusa_logits = None

            step_trace = MedusaStepTrace(
                step=step,
                prefix_length=prefix_length,
                remaining_new_tokens=remaining,
                candidate_path_count=candidates.candidate_paths.shape[0],
                target_next_token=candidates.target_next_token,
                medusa_top_tokens=candidates.medusa_top_tokens,
                selected_path_index=verification.path_index,
                selected_path_tokens=verification.path_tokens[0].tolist(),
                target_predictions=verification.target_predictions[0].tolist(),
                target_top_logits=verification.target_top_logits,
                accepted_count=verification.accepted_count,
                rejected_at=verification.rejected_at,
                appended_tokens=verification.appended_tokens[0].tolist(),
                output_length=generated.shape[-1],
                stop_reason=stop_reason_for(generated, prompt_length, max_new_tokens, eos_token_ids, min_length),
                verification_method=verification.verification_method,
                acceptance_mode=verification.acceptance_mode,
                tree_node_count=candidates.tree_candidates.shape[-1],
                cache_updated=verification.cache_updated,
                fallback_reason=verification.fallback_reason or fallback_reason,
            )
            if trace_steps is not None:
                trace_steps.append(step_trace)
            if step_callback is not None:
                step_callback(step_trace, generated)

            if progress:
                print(
                    f"[medusa step {step}] method={verification.verification_method} "
                    f"selected_path={verification.path_index} accepted={verification.accepted_count} "
                    f"cache_updated={verification.cache_updated} "
                    f"appended={verification.appended_tokens[0].tolist()}",
                    flush=True,
                )
            step += 1

    return generated


def generate_with_trace(
    target_model: torch.nn.Module,
    medusa_heads: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    medusa_choices: Sequence[Sequence[int]] | None = None,
    top_k: int = 5,
    top_k_logits: int = 0,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
    step_callback: Callable[[MedusaStepTrace, torch.Tensor], None] | None = None,
    verifier: VerifierMode = "tree",
    acceptance_mode: AcceptanceMode = "greedy",
    temperature: float = 0.0,
    posterior_threshold: float = 0.09,
    posterior_alpha: float = 0.3,
    top_p: float = 0.8,
    use_kv_cache: bool = True,
    fallback_to_slow: bool = True,
) -> tuple[torch.Tensor, list[MedusaStepTrace]]:
    trace_steps: list[MedusaStepTrace] = []
    output_ids = generate(
        target_model,
        medusa_heads,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        eos_token_id=eos_token_id,
        medusa_choices=medusa_choices,
        top_k=top_k,
        trace_steps=trace_steps,
        top_k_logits=top_k_logits,
        progress=progress,
        heartbeat_seconds=heartbeat_seconds,
        step_callback=step_callback,
        verifier=verifier,
        acceptance_mode=acceptance_mode,
        temperature=temperature,
        posterior_threshold=posterior_threshold,
        posterior_alpha=posterior_alpha,
        top_p=top_p,
        use_kv_cache=use_kv_cache,
        fallback_to_slow=fallback_to_slow,
    )
    return output_ids, trace_steps


def print_trace(
    trace_steps: list[MedusaStepTrace],
    tokenizer: Any | None = None,
    show_logits: bool = False,
) -> None:
    for item in trace_steps:
        print(f"\nMEDUSA STEP {item.step}")
        print(f"verification method: {item.verification_method}")
        print(f"acceptance mode: {item.acceptance_mode}")
        print(f"prefix length: {item.prefix_length}")
        print(f"remaining new-token budget: {item.remaining_new_tokens}")
        print(f"candidate paths: {item.candidate_path_count}")
        print(f"tree nodes: {item.tree_node_count}")
        print(f"cache updated: {item.cache_updated}")
        if item.fallback_reason:
            print(f"fallback reason: {item.fallback_reason}")
        print(f"target next token: {item.target_next_token}")
        if tokenizer is not None:
            print(f"target next text: {tokenizer.decode([item.target_next_token])!r}")
        print(f"medusa top tokens by head: {item.medusa_top_tokens}")
        print(f"selected path index: {item.selected_path_index}")
        print(f"selected path tokens: {item.selected_path_tokens}")
        if tokenizer is not None:
            print(f"selected path text: {tokenizer.decode(item.selected_path_tokens)!r}")
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
