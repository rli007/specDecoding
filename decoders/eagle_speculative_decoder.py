#!/usr/bin/env python
"""First-principles EAGLE-style speculative decoding.

EAGLE uses a trained drafter that consumes target-model hidden states and grows
a candidate tree. The target model then verifies candidate paths and accepts the
longest valid prefix. Official serving implementations do this with optimized
tree attention and cache updates; this file keeps the algorithm visible by
verifying candidate paths with ordinary target forwards.

The `eagle_drafter` is intentionally an interface, not a specific HF class. It
can be any object with `propose_tree(...)`, or any callable that returns token
paths. Exact EAGLE behavior requires trained EAGLE/EAGLE3 drafter weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from decoders.first_principles_speculative_decoder import (
    LogitTopK,
    model_device,
    normalize_eos_token_ids,
    should_stop,
    stop_reason_for,
    target_predictions_for_draft,
    validate_generate_inputs,
)
from decoders.medusa_speculative_decoder import forward_target_with_hidden


@dataclass
class EagleDraftTree:
    candidate_paths: torch.Tensor
    scores: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EagleVerification:
    path_index: int
    path_tokens: torch.Tensor
    target_predictions: torch.Tensor
    target_top_logits: list[LogitTopK]
    accepted_count: int
    rejected_at: int | None
    appended_tokens: torch.Tensor


@dataclass
class EagleStepTrace:
    step: int
    prefix_length: int
    remaining_new_tokens: int
    candidate_path_count: int
    selected_path_index: int
    selected_path_tokens: list[int]
    target_predictions: list[int]
    target_top_logits: list[LogitTopK]
    accepted_count: int
    rejected_at: int | None
    appended_tokens: list[int]
    output_length: int
    stop_reason: str | None
    drafter_metadata: dict[str, Any]


class DebugTargetLogitsDrafter:
    """Debug-only drafter that proposes the target argmax as a one-token path.

    This is not EAGLE. It exists so the EAGLE control flow can be smoke-tested
    without trained EAGLE weights.
    """

    def propose_tree(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        max_depth: int,
        top_k: int,
        max_paths: int,
    ) -> EagleDraftTree:
        del input_ids, hidden_states, max_depth, top_k, max_paths
        next_token = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)
        return EagleDraftTree(candidate_paths=next_token, metadata={"debug_drafter": "target_argmax"})


def _pad_paths(paths: list[list[int]], pad_value: int = -1) -> torch.Tensor:
    if not paths:
        return torch.empty((0, 0), dtype=torch.long)
    max_length = max(len(path) for path in paths)
    padded = [path + [pad_value] * (max_length - len(path)) for path in paths]
    return torch.tensor(padded, dtype=torch.long)


def normalize_eagle_draft_output(
    output: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> EagleDraftTree:
    """Normalize drafter output to padded `[num_paths, path_len]` paths."""
    if isinstance(output, EagleDraftTree):
        output.candidate_paths = output.candidate_paths.to(device=device, dtype=dtype)
        if output.scores is not None:
            output.scores = output.scores.to(device)
        return output

    metadata: dict[str, Any] = {}
    scores = None
    if isinstance(output, dict):
        metadata = {key: value for key, value in output.items() if key not in {"candidate_paths", "paths", "tokens", "scores"}}
        scores = output.get("scores")
        output = output.get("candidate_paths", output.get("paths", output.get("tokens")))

    if isinstance(output, torch.Tensor):
        paths = output
        if paths.ndim == 1:
            paths = paths.unsqueeze(0)
        elif paths.ndim == 3:
            if paths.shape[0] != 1:
                raise ValueError("Batched EAGLE draft paths are not supported in this traceable implementation.")
            paths = paths[0]
        elif paths.ndim != 2:
            raise ValueError("EAGLE draft tensor must have shape [path_len], [num_paths,path_len], or [1,num_paths,path_len].")
        return EagleDraftTree(
            candidate_paths=paths.to(device=device, dtype=dtype),
            scores=scores.to(device) if isinstance(scores, torch.Tensor) else scores,
            metadata=metadata,
        )

    if isinstance(output, list):
        if not output:
            paths = torch.empty((0, 0), dtype=dtype, device=device)
        elif all(isinstance(item, int) for item in output):
            paths = torch.tensor([output], dtype=dtype, device=device)
        else:
            paths = _pad_paths([[int(token) for token in path] for path in output]).to(device=device, dtype=dtype)
        return EagleDraftTree(
            candidate_paths=paths,
            scores=scores.to(device) if isinstance(scores, torch.Tensor) else scores,
            metadata=metadata,
        )

    raise TypeError(
        "EAGLE drafter must return EagleDraftTree, a dict, a tensor, "
        "a token list, or a list of token paths."
    )


def propose_eagle_tree(
    eagle_drafter: Any,
    generated: torch.Tensor,
    hidden_states: torch.Tensor,
    target_logits: torch.Tensor,
    max_depth: int,
    top_k: int,
    max_paths: int,
) -> EagleDraftTree:
    """Call a trained EAGLE-style drafter through a small operator interface."""
    kwargs = {
        "input_ids": generated,
        "hidden_states": hidden_states,
        "target_logits": target_logits,
        "max_depth": max_depth,
        "top_k": top_k,
        "max_paths": max_paths,
    }
    if hasattr(eagle_drafter, "propose_tree"):
        output = eagle_drafter.propose_tree(**kwargs)
    elif callable(eagle_drafter):
        output = eagle_drafter(**kwargs)
    else:
        raise TypeError("eagle_drafter must be callable or expose propose_tree(...).")
    return normalize_eagle_draft_output(output, generated.dtype, generated.device)


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

    if draft_tokens.shape[-1] == 0:
        appended = target_predictions[:, :1]
    elif replacement_token is None:
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
) -> EagleVerification:
    """Verify candidate paths with target forwards and choose max accepted prefix."""
    best: EagleVerification | None = None
    for path_index, row in enumerate(candidate_paths):
        path_tokens = _path_row_to_tensor(row, generated.dtype, generated.device)
        if path_tokens.shape[-1] == 0:
            continue
        if progress:
            print(f"[eagle step {step}] verifying path {path_index}: {path_tokens[0].tolist()}", flush=True)
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
        candidate = EagleVerification(
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
        return EagleVerification(
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
    eagle_drafter: Any,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    max_depth: int = 4,
    top_k: int = 4,
    max_paths: int = 16,
    trace_steps: list[EagleStepTrace] | None = None,
    top_k_logits: int = 0,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
) -> torch.Tensor:
    """Generate with an EAGLE-style hidden-state drafter."""
    validate_generate_inputs(prompt_token_ids, max_new_tokens, num_assistant_tokens=max_depth)
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(target_model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    eos_token_ids = normalize_eos_token_ids(target_model, eos_token_id)
    prompt_length = generated.shape[-1]

    with torch.inference_mode():
        step = 1
        while generated.shape[-1] - prompt_length < max_new_tokens:
            if should_stop(generated, eos_token_ids, min_length):
                break

            prefix_length = generated.shape[-1]
            remaining = max_new_tokens - (prefix_length - prompt_length)
            target_logits, hidden_states = forward_target_with_hidden(target_model, generated)
            draft_tree = propose_eagle_tree(
                eagle_drafter,
                generated,
                hidden_states,
                target_logits,
                max_depth=max_depth,
                top_k=top_k,
                max_paths=max_paths,
            )
            if progress:
                print(
                    f"[eagle step {step}] prefix_len={prefix_length} "
                    f"paths={draft_tree.candidate_paths.shape[0]} max_depth={max_depth}",
                    flush=True,
                )

            verification = verify_candidate_paths_slow(
                target_model,
                generated,
                draft_tree.candidate_paths,
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
                    EagleStepTrace(
                        step=step,
                        prefix_length=prefix_length,
                        remaining_new_tokens=remaining,
                        candidate_path_count=draft_tree.candidate_paths.shape[0],
                        selected_path_index=verification.path_index,
                        selected_path_tokens=verification.path_tokens[0].tolist(),
                        target_predictions=verification.target_predictions[0].tolist(),
                        target_top_logits=verification.target_top_logits,
                        accepted_count=verification.accepted_count,
                        rejected_at=verification.rejected_at,
                        appended_tokens=verification.appended_tokens[0].tolist(),
                        output_length=generated.shape[-1],
                        stop_reason=stop_reason_for(generated, prompt_length, max_new_tokens, eos_token_ids, min_length),
                        drafter_metadata=draft_tree.metadata,
                    )
                )

            if progress:
                print(
                    f"[eagle step {step}] selected_path={verification.path_index} "
                    f"accepted={verification.accepted_count} appended={verification.appended_tokens[0].tolist()}",
                    flush=True,
                )
            step += 1

    return generated


def generate_with_trace(
    target_model: torch.nn.Module,
    eagle_drafter: Any,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    max_depth: int = 4,
    top_k: int = 4,
    max_paths: int = 16,
    top_k_logits: int = 0,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
) -> tuple[torch.Tensor, list[EagleStepTrace]]:
    trace_steps: list[EagleStepTrace] = []
    output_ids = generate(
        target_model,
        eagle_drafter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        eos_token_id=eos_token_id,
        max_depth=max_depth,
        top_k=top_k,
        max_paths=max_paths,
        trace_steps=trace_steps,
        top_k_logits=top_k_logits,
        progress=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    return output_ids, trace_steps
