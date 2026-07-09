#!/usr/bin/env python
"""First-principles Medusa-style speculative decoding.

Medusa adds several trained prediction heads on top of the target model hidden
state. The official implementation verifies a candidate tree with custom tree
attention. This file keeps the same algorithmic pieces, but verifies each path
with ordinary target-model forwards so the control flow is easy to inspect.

You need trained Medusa heads for useful generations. The `MedusaHeadStack`
class below is the architecture shell; random heads are only good for tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from decoders.first_principles_speculative_decoder import (
    LogitTopK,
    decision_logits,
    model_device,
    normalize_eos_token_ids,
    should_stop,
    stop_reason_for,
    summarize_top_logits,
    target_predictions_for_draft,
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


def forward_target_with_hidden(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the target model and return `(logits, last_hidden_states)`."""
    try:
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    except TypeError:
        outputs = model(input_ids, output_hidden_states=True)

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


def generate_medusa_candidates(
    target_logits: torch.Tensor,
    medusa_logits: torch.Tensor,
    buffers: MedusaBuffers,
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

    target_next = torch.argmax(target_logits[:, -1, :], dim=-1)
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
    safe_retrieve = torch.where(
        buffers.retrieve_indices < 0,
        torch.full_like(buffers.retrieve_indices, pad_index),
        buffers.retrieve_indices,
    )
    candidate_paths = extended[safe_retrieve]
    return MedusaCandidates(
        candidate_pool=candidate_pool,
        tree_candidates=tree_candidates.unsqueeze(0),
        candidate_paths=candidate_paths,
        target_next_token=int(target_next.item()),
        medusa_top_tokens=[[int(token) for token in row] for row in top_token_ids.detach().cpu().tolist()],
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
) -> torch.Tensor:
    """Generate with Medusa-style heads and slow path verification."""
    validate_generate_inputs(prompt_token_ids, max_new_tokens, num_assistant_tokens=top_k)
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(target_model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    medusa_heads = medusa_heads.to(device).eval()
    eos_token_ids = normalize_eos_token_ids(target_model, eos_token_id)
    prompt_length = generated.shape[-1]
    buffers: MedusaBuffers | None = None

    with torch.inference_mode():
        step = 1
        while generated.shape[-1] - prompt_length < max_new_tokens:
            if should_stop(generated, eos_token_ids, min_length):
                break

            prefix_length = generated.shape[-1]
            remaining = max_new_tokens - (prefix_length - prompt_length)
            target_logits, hidden_states = forward_target_with_hidden(target_model, generated)
            medusa_logits = normalize_medusa_logits(medusa_heads(hidden_states), hidden_states)

            if buffers is None:
                choices = list(medusa_choices) if medusa_choices is not None else linear_medusa_choices(medusa_logits.shape[0])
                buffers = generate_medusa_buffers(choices, top_k=top_k, device=generated.device)

            candidates = generate_medusa_candidates(target_logits, medusa_logits, buffers)
            if progress:
                print(
                    f"[medusa step {step}] prefix_len={prefix_length} "
                    f"paths={candidates.candidate_paths.shape[0]} target_next={candidates.target_next_token}",
                    flush=True,
                )

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

            if trace_steps is not None:
                trace_steps.append(
                    MedusaStepTrace(
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
                    )
                )

            if progress:
                print(
                    f"[medusa step {step}] selected_path={verification.path_index} "
                    f"accepted={verification.accepted_count} appended={verification.appended_tokens[0].tolist()}",
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
    )
    return output_ids, trace_steps
