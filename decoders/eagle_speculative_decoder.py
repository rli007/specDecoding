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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

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


# ---------------------------------------------------------------------------
# EAGLE-1 static draft tree (rank tuples, prefix-closed, same format as the
# Medusa presets). Vendored verbatim from SafeAILab/EAGLE eagle/model/choices.py
# `mc_sim_7b_63` — the paper's default: 25 paths, max depth 5, 26 verify nodes
# including the free root.
# ---------------------------------------------------------------------------

EAGLE_MC_SIM_7B_63_CHOICES: list[list[int]] = [
    [0], [1], [2], [3],
    [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0],
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [1, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
]

DEFAULT_EAGLE_DRAFTER = "yuhuili/EAGLE-Vicuna-7B-v1.3"


@dataclass
class _TreeNode:
    """One node of the static draft tree (excluding the free root)."""

    path: tuple[int, ...]  # rank tuple, e.g. (0, 2)
    parent: int  # index into the node list, -1 = root
    depth: int  # 1-based; root is depth 0
    token: int | None = None  # filled during drafting
    slot: int | None = None  # physical drafter-cache slot, filled during drafting


def _build_tree_nodes(choices: list[list[int]]) -> list[_TreeNode]:
    """Static tree topology from rank tuples; validates prefix closure."""
    by_path: dict[tuple[int, ...], int] = {}
    nodes: list[_TreeNode] = []
    for path in sorted(choices, key=lambda p: (len(p), p)):
        key = tuple(path)
        if len(key) > 1 and key[:-1] not in by_path:
            raise ValueError(f"EAGLE choices are not prefix-closed: {key} lacks parent {key[:-1]}")
        parent = by_path[key[:-1]] if len(key) > 1 else -1
        by_path[key] = len(nodes)
        nodes.append(_TreeNode(path=key, parent=parent, depth=len(key)))
    return nodes


class EagleDrafterModel(nn.Module):
    """Faithful EAGLE-1 drafter (yuhuili checkpoint layout).

    Architecture, confirmed against the published state dict:
    - `embed_tokens`: the drafter's own embedding copy
    - `fc`: Linear(2*hidden -> hidden, bias=True) fusing cat(embedding, feature)
    - one LlamaDecoderLayer whose input_layernorm is REMOVED (the checkpoint has
      no such weight; fc output feeds attention directly)
    - no final norm; logits come from the target's own lm_head applied to the
      drafter's output feature (lm_head is shared, never copied)

    The drafter consumes (feature at position i, token at position i+1) pairs
    and its output at that pair estimates the target's feature at position i+1.
    """

    def __init__(self, config):
        super().__init__()
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRotaryEmbedding,
        )

        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)
        self.layer = LlamaDecoderLayer(config, layer_idx=0)
        self.layer.input_layernorm = nn.Identity()
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        features: torch.Tensor,  # [1, n, hidden]
        token_ids: torch.Tensor,  # [1, n]
        position_ids: torch.Tensor,  # [1, n]
        attention_mask: torch.Tensor,  # [1, 1, n, kv_len] additive
        past_key_values,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.fc(torch.cat([self.embed_tokens(token_ids), features], dim=-1))
        position_embeddings = self.rotary_emb(hidden, position_ids)
        out = self.layer(
            hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        return out[0] if isinstance(out, tuple) else out


def load_official_eagle_drafter(
    drafter_path: str = DEFAULT_EAGLE_DRAFTER,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> EagleDrafterModel:
    """Build the drafter from the published checkpoint; hard-fails on any
    missing/unexpected weight so a silent architecture mismatch is impossible."""
    from huggingface_hub import hf_hub_download
    from transformers import LlamaConfig

    local = Path(drafter_path).expanduser()
    if local.is_dir():
        config_file = local / "config.json"
        weights_file = local / "pytorch_model.bin"
    else:
        config_file = Path(hf_hub_download(drafter_path, "config.json"))
        weights_file = Path(hf_hub_download(drafter_path, "pytorch_model.bin"))

    config = LlamaConfig(**json.loads(config_file.read_text()))
    drafter = EagleDrafterModel(config)

    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    remapped = {key.replace("layers.0.", "layer.", 1): value for key, value in state_dict.items()}
    missing, unexpected = drafter.load_state_dict(remapped, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"EAGLE drafter load mismatch: missing={missing} unexpected={unexpected}")
    return drafter.to(device=device, dtype=dtype).eval()


class EagleOneDrafter:
    """EAGLE-1 static-tree drafting via the shell's propose_tree interface.

    Every step: warm the drafter cache over the (feature, next-token) pairs of
    the current sequence, take the free root token from the target's own
    logits, then expand the static tree depth by depth. Each expansion forward
    processes one depth's frontier with a tree attention mask (a node sees the
    warmed prefix plus its own ancestors only). The drafter cache is rebuilt
    from scratch each step — an inspectability choice, mirrored on the
    hardware side by charging only the algorithmic work (frontier positions),
    like the assisted decoder's cache-rebuild convention.
    """

    def __init__(
        self,
        drafter: EagleDrafterModel,
        lm_head: nn.Module,
        choices: list[list[int]] | None = None,
    ):
        self.drafter = drafter
        self.lm_head = lm_head
        self.choices = choices if choices is not None else EAGLE_MC_SIM_7B_63_CHOICES
        self.nodes_template = _build_tree_nodes(self.choices)

    def _additive_mask(self, allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """[q, kv] boolean -> [1, 1, q, kv] additive float mask."""
        mask = torch.full(allowed.shape, torch.finfo(dtype).min, dtype=dtype, device=allowed.device)
        mask.masked_fill_(allowed, 0.0)
        return mask[None, None]

    def propose_tree(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        max_depth: int,
        top_k: int,
        max_paths: int,
    ) -> EagleDraftTree:
        del max_depth, top_k, max_paths  # tree shape is fixed by self.choices
        from transformers import DynamicCache

        device = hidden_states.device
        dtype = hidden_states.dtype
        length = input_ids.shape[-1]
        root_token = int(torch.argmax(target_logits[:, -1, :], dim=-1).item())

        nodes = [
            _TreeNode(path=node.path, parent=node.parent, depth=node.depth)
            for node in self.nodes_template
        ]
        cache = DynamicCache()

        with torch.inference_mode():
            # Warm-up over the whole current sequence: pair feature f_i with
            # token x_{i+1} (i = 0..L-2), then the root pair (f_{L-1}, root).
            pair_features = torch.cat([hidden_states[:, :-1, :], hidden_states[:, -1:, :]], dim=1)
            pair_tokens = torch.cat(
                [input_ids[:, 1:], torch.tensor([[root_token]], device=device, dtype=input_ids.dtype)],
                dim=1,
            )
            n_pairs = pair_tokens.shape[-1]
            positions = torch.arange(n_pairs, device=device)
            causal = torch.ones(n_pairs, n_pairs, dtype=torch.bool, device=device).tril()
            features_out = self.drafter(
                pair_features.to(dtype),
                pair_tokens,
                positions.unsqueeze(0),
                self._additive_mask(causal, dtype),
                cache,
                positions,
            )
            # Output at the root pair estimates the target's feature for the
            # root token -> its logits rank the depth-1 candidates.
            frontier_features = {-1: features_out[:, -1:, :]}
            frontier_logits = {-1: self.lm_head(features_out[:, -1:, :])}
            slot_of = {-1: n_pairs - 1}  # root pair's physical cache slot
            next_slot = n_pairs

            for depth in range(1, max(node.depth for node in nodes) + 1):
                depth_nodes = [i for i, node in enumerate(nodes) if node.depth == depth]
                if not depth_nodes:
                    break
                # Each node's token: its rank in the PARENT's logits.
                for i in depth_nodes:
                    parent = nodes[i].parent
                    rank = nodes[i].path[-1]
                    parent_logits = frontier_logits[parent][0, 0]
                    nodes[i].token = int(torch.topk(parent_logits, rank + 1).indices[-1].item())

                feats = torch.cat([frontier_features[nodes[i].parent] for i in depth_nodes], dim=1)
                toks = torch.tensor([[nodes[i].token for i in depth_nodes]], device=device, dtype=input_ids.dtype)
                n_front = len(depth_nodes)
                kv_len = next_slot + n_front
                allowed = torch.zeros(n_front, kv_len, dtype=torch.bool, device=device)
                allowed[:, :n_pairs] = True  # every node sees the warmed prefix
                for row, i in enumerate(depth_nodes):
                    nodes[i].slot = next_slot + row
                    allowed[row, nodes[i].slot] = True
                    parent = nodes[i].parent
                    while parent != -1:  # ancestors' physical slots
                        allowed[row, nodes[parent].slot] = True
                        parent = nodes[parent].parent
                pos = torch.full((1, n_front), n_pairs - 1 + depth, device=device, dtype=torch.long)
                cache_pos = torch.arange(next_slot, next_slot + n_front, device=device)
                out = self.drafter(feats.to(dtype), toks, pos, self._additive_mask(allowed, dtype), cache, cache_pos)
                for row, i in enumerate(depth_nodes):
                    frontier_features[i] = out[:, row : row + 1, :]
                    frontier_logits[i] = self.lm_head(out[:, row : row + 1, :])
                next_slot += n_front

        # Candidate paths: free root + the node chain of every choice path.
        paths: list[list[int]] = []
        index_of = {node.path: i for i, node in enumerate(nodes)}
        for choice in self.choices:
            chain = [root_token]
            for d in range(1, len(choice) + 1):
                chain.append(nodes[index_of[tuple(choice[:d])]].token)
            paths.append(chain)

        return EagleDraftTree(
            candidate_paths=_pad_paths(paths).to(device=device),
            metadata={
                "drafter": "eagle1-static-tree",
                "tree_nodes_incl_root": len(nodes) + 1,
                "draft_forward_positions": [n_pairs] + [
                    sum(1 for node in nodes if node.depth == d)
                    for d in range(1, max(node.depth for node in nodes) + 1)
                ],
            },
        )


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
