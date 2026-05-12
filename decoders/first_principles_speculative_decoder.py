#!/usr/bin/env python
"""Traceable first-principles greedy speculative decoding.

This file is intentionally small and linear so it is easy to inspect with a
debugger, print tracing, or torch.profiler. The model forward calls are treated
as black-box operators:

1. The draft model proposes a block of tokens.
2. The target model verifies the accepted prefix plus that block.
3. Python control flow accepts matching draft tokens until the first mismatch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = "The Stanford football team"
DEFAULT_MAX_NEW_TOKENS = 20


@dataclass
class LogitTopK:
    token_ids: list[int]
    values: list[float]


@dataclass
class DraftResult:
    tokens: torch.Tensor
    forward_input_lengths: list[int]
    decision_logits: list[torch.Tensor]
    top_logits: list[LogitTopK]


@dataclass
class TargetResult:
    predictions: torch.Tensor
    decision_logits: list[torch.Tensor]
    top_logits: list[LogitTopK]


@dataclass
class SpeculativeStepTrace:
    step: int
    prefix_length: int
    remaining_new_tokens: int
    requested_draft_tokens: int
    draft_forward_input_lengths: list[int]
    target_forward_input_length: int
    draft_tokens: list[int]
    target_predictions: list[int]
    draft_top_logits: list[LogitTopK]
    target_top_logits: list[LogitTopK]
    accepted_count: int
    rejected_at: int | None
    appended_tokens: list[int]
    output_length: int
    stop_reason: str | None


@dataclass
class LoadedModels:
    tokenizer: AutoTokenizer
    target: torch.nn.Module
    assistant: torch.nn.Module
    device: torch.device


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    if torch.cuda.is_available():
        status["cuda_device"] = torch.cuda.get_device_name(0)
    return status


def print_hardware_status(device: torch.device) -> None:
    status = hardware_status()
    print(
        "Hardware: "
        f"torch={status['torch']}, "
        f"cuda_available={status['cuda_available']}, "
        f"mps_available={status['mps_available']}, "
        f"mps_built={status['mps_built']}",
        flush=True,
    )
    if "cuda_device" in status:
        print(f"CUDA device: {status['cuda_device']}", flush=True)
    print(f"Selected execution device: {device}", flush=True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def memory_status(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        return f"cuda allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"
    if device.type == "mps" and hasattr(torch, "mps"):
        pieces: list[str] = []
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        driver_allocated = getattr(torch.mps, "driver_allocated_memory", None)
        if callable(current_allocated):
            pieces.append(f"mps allocated={current_allocated() / 1024**3:.2f}GiB")
        if callable(driver_allocated):
            pieces.append(f"driver={driver_allocated() / 1024**3:.2f}GiB")
        return " ".join(pieces) if pieces else "mps memory=n/a"
    return "memory=n/a"


def print_memory_status(device: torch.device, label: str) -> None:
    print(f"{label}: {memory_status(device)}", flush=True)


@contextmanager
def timed_operation(label: str, device: torch.device, heartbeat_seconds: float = 0.0):
    start = time.perf_counter()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            elapsed = time.perf_counter() - start
            print(f"{label} still running after {elapsed:.1f}s", flush=True)

    thread: threading.Thread | None = None
    if heartbeat_seconds > 0:
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()

    try:
        yield
        synchronize_device(device)
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=0.2)
        elapsed = time.perf_counter() - start
        print(f"{label} finished in {elapsed:.2f}s", flush=True)


def dtype_from_arg(dtype_arg: str) -> torch.dtype | str | None:
    if dtype_arg == "none":
        return None
    if dtype_arg == "auto":
        return "auto"
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    label: str | None = None,
    heartbeat_seconds: float = 0.0,
) -> torch.Tensor:
    device = input_ids.device
    context = timed_operation(label, device, heartbeat_seconds) if label is not None else null_operation()
    with context:
        try:
            outputs = model(input_ids=input_ids, use_cache=False)
        except TypeError:
            outputs = model(input_ids)
    return extract_logits(outputs)


@contextmanager
def null_operation():
    yield


def model_device(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return input_ids.device


def eos_from_config(model: torch.nn.Module) -> int | list[int] | None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return None
    return generation_config.eos_token_id


def model_vocab_size(model: torch.nn.Module) -> int | None:
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return int(output_embeddings.weight.shape[0])
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    return int(vocab_size) if vocab_size is not None else None


def normalize_eos_token_ids(
    model: torch.nn.Module,
    eos_token_id: int | Iterable[int] | torch.Tensor | None,
) -> set[int]:
    if eos_token_id is None:
        eos_token_id = eos_from_config(model)

    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, torch.Tensor):
        return {int(token) for token in eos_token_id.detach().flatten().tolist()}
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(token) for token in eos_token_id if token is not None}


def process_logits(scores: torch.Tensor, eos_token_ids: set[int]) -> torch.Tensor:
    """Mask EOS so greedy argmax cannot pick it before min_length."""
    if not eos_token_ids:
        return scores

    eos_tensor = torch.tensor(list(eos_token_ids), device=scores.device)
    vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    eos_token_mask = torch.isin(vocab_tensor, eos_tensor)
    return torch.where(eos_token_mask, -math.inf, scores)


def select_next_token(
    logits: torch.Tensor,
    eos_token_ids: set[int],
    current_length: int,
    min_length: int,
) -> torch.Tensor:
    if current_length < min_length - 1:
        logits = process_logits(logits, eos_token_ids)
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def decision_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the [batch, vocab] logits used to choose the next token."""
    return logits[:, -1, :]


def summarize_top_logits(logits: torch.Tensor, top_k: int) -> LogitTopK:
    if top_k <= 0:
        return LogitTopK(token_ids=[], values=[])
    top_k = min(top_k, logits.shape[-1])
    values, token_ids = torch.topk(logits[0], k=top_k)
    return LogitTopK(
        token_ids=[int(token_id) for token_id in token_ids.detach().cpu().tolist()],
        values=[float(value) for value in values.detach().cpu().tolist()],
    )


def is_eos(token: torch.Tensor, eos_token_ids: set[int]) -> bool:
    return bool(eos_token_ids) and int(token.item()) in eos_token_ids


def should_stop(generated: torch.Tensor, eos_token_ids: set[int], min_length: int) -> bool:
    return generated.shape[-1] >= min_length and is_eos(generated[:, -1:], eos_token_ids)


def validate_generate_inputs(prompt_token_ids: torch.Tensor, max_new_tokens: int, num_assistant_tokens: int) -> None:
    if prompt_token_ids.ndim != 2:
        raise ValueError("prompt_token_ids must have shape [batch, sequence_length].")
    if prompt_token_ids.shape[0] != 1:
        raise ValueError("This traceable implementation currently supports batch size 1.")
    if prompt_token_ids.shape[-1] == 0:
        raise ValueError("prompt_token_ids must contain at least one token.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if num_assistant_tokens < 0:
        raise ValueError("num_assistant_tokens must be non-negative.")


def greedy_generate(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain greedy generation with the same token-in/token-out shape."""
    validate_generate_inputs(prompt_token_ids, max_new_tokens, num_assistant_tokens=0)
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    eos_token_ids = normalize_eos_token_ids(model, eos_token_id)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = forward_logits(model, generated)
            next_token = select_next_token(logits, eos_token_ids, generated.shape[-1], min_length)
            generated = torch.cat([generated, next_token], dim=-1)
            if should_stop(generated, eos_token_ids, min_length):
                break

    return generated


def draft_tokens(
    draft_model: torch.nn.Module,
    current_ids: torch.Tensor,
    num_assistant_tokens: int,
    eos_token_ids: set[int],
    min_length: int,
    top_k_logits: int = 0,
    collect_logits: bool = False,
    progress: bool = False,
    step: int | None = None,
    heartbeat_seconds: float = 0.0,
) -> DraftResult:
    """Draft up to num_assistant_tokens greedy tokens from the draft model."""
    token_pieces: list[torch.Tensor] = []
    forward_input_lengths: list[int] = []
    decision_logit_pieces: list[torch.Tensor] = []
    top_logit_summaries: list[LogitTopK] = []
    output_device = current_ids.device
    draft_device = model_device(draft_model, current_ids)
    draft_sequence = current_ids.to(draft_device)

    for _ in range(num_assistant_tokens):
        draft_index = len(token_pieces)
        forward_input_lengths.append(draft_sequence.shape[-1])
        step_text = f"step {step}" if step is not None else "draft"
        if progress:
            print(
                f"[{step_text}] assistant forward {draft_index + 1}/{num_assistant_tokens} "
                f"input_len={draft_sequence.shape[-1]} ...",
                flush=True,
            )
        logits = forward_logits(
            draft_model,
            draft_sequence,
            label=f"[{step_text}] assistant forward {draft_index + 1}/{num_assistant_tokens}"
            if progress or heartbeat_seconds > 0
            else None,
            heartbeat_seconds=heartbeat_seconds,
        )
        next_logits = decision_logits(logits)
        if collect_logits:
            decision_logit_pieces.append(next_logits.detach().cpu().to(torch.float32))
        top_logit_summaries.append(summarize_top_logits(next_logits, top_k_logits))
        next_token = select_next_token(logits, eos_token_ids, draft_sequence.shape[-1], min_length)
        token_pieces.append(next_token)
        if progress:
            print(f"[step {step}] assistant selected token={int(next_token.item())}", flush=True)
        draft_sequence = torch.cat([draft_sequence, next_token], dim=-1)
        if should_stop(draft_sequence, eos_token_ids, min_length):
            break

    if not token_pieces:
        empty = torch.empty((current_ids.shape[0], 0), dtype=current_ids.dtype, device=output_device)
        return DraftResult(
            tokens=empty,
            forward_input_lengths=forward_input_lengths,
            decision_logits=decision_logit_pieces,
            top_logits=top_logit_summaries,
        )

    return DraftResult(
        tokens=torch.cat(token_pieces, dim=-1).to(output_device),
        forward_input_lengths=forward_input_lengths,
        decision_logits=decision_logit_pieces,
        top_logits=top_logit_summaries,
    )


def target_predictions_for_draft(
    target_model: torch.nn.Module,
    generated: torch.Tensor,
    draft_token_ids: torch.Tensor,
    eos_token_ids: set[int],
    min_length: int,
    top_k_logits: int = 0,
    collect_logits: bool = False,
    progress: bool = False,
    step: int | None = None,
    heartbeat_seconds: float = 0.0,
) -> TargetResult:
    """Return target predictions for every draft position plus one bonus token."""
    candidate_input_ids = torch.cat([generated, draft_token_ids], dim=-1)
    step_text = f"step {step}" if step is not None else "target"
    if progress:
        print(f"[{step_text}] target verify forward input_len={candidate_input_ids.shape[-1]} ...", flush=True)
    target_logits = forward_logits(
        target_model,
        candidate_input_ids,
        label=f"[{step_text}] target verify forward" if progress or heartbeat_seconds > 0 else None,
        heartbeat_seconds=heartbeat_seconds,
    )
    if progress:
        print(f"[step {step}] target verify logits shape={tuple(target_logits.shape)}", flush=True)

    prediction_start = generated.shape[-1] - 1
    predictions: list[torch.Tensor] = []
    decision_logit_pieces: list[torch.Tensor] = []
    top_logit_summaries: list[LogitTopK] = []
    for idx in range(draft_token_ids.shape[-1] + 1):
        logits = target_logits[:, prediction_start + idx : prediction_start + idx + 1, :]
        next_logits = decision_logits(logits)
        if collect_logits:
            decision_logit_pieces.append(next_logits.detach().cpu().to(torch.float32))
        top_logit_summaries.append(summarize_top_logits(next_logits, top_k_logits))
        current_length = generated.shape[-1] + idx
        predictions.append(select_next_token(logits, eos_token_ids, current_length, min_length))

    return TargetResult(
        predictions=torch.cat(predictions, dim=-1),
        decision_logits=decision_logit_pieces,
        top_logits=top_logit_summaries,
    )


def stop_reason_for(generated: torch.Tensor, prompt_length: int, max_new_tokens: int, eos_token_ids: set[int], min_length: int) -> str | None:
    if generated.shape[-1] - prompt_length >= max_new_tokens:
        return "max_new_tokens"
    if should_stop(generated, eos_token_ids, min_length):
        return "eos"
    return None


def generate(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    draft_model: torch.nn.Module | None = None,
    num_assistant_tokens: int = 4,
    trace_steps: list[SpeculativeStepTrace] | None = None,
    top_k_logits: int = 0,
    logit_records: list[dict[str, Any]] | None = None,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
) -> torch.Tensor:
    """Greedy speculative decoding from first principles.

    Pass a list as trace_steps to collect one SpeculativeStepTrace per
    speculative iteration. The return value stays a tensor so this mirrors
    reference/static_cache_generation_reference.txt's generate shape.
    """
    validate_generate_inputs(prompt_token_ids, max_new_tokens, num_assistant_tokens)
    if draft_model is None or num_assistant_tokens == 0:
        return greedy_generate(model, prompt_token_ids, max_new_tokens, min_length, eos_token_id)
    if max_new_tokens == 0:
        return prompt_token_ids

    device = model_device(model, prompt_token_ids)
    generated = prompt_token_ids.to(device).clone()
    eos_token_ids = normalize_eos_token_ids(model, eos_token_id)
    prompt_length = generated.shape[-1]

    with torch.inference_mode():
        step = 1
        while generated.shape[-1] - prompt_length < max_new_tokens:
            if should_stop(generated, eos_token_ids, min_length):
                break

            prefix_length = generated.shape[-1]
            remaining = max_new_tokens - (prefix_length - prompt_length)
            requested_draft_tokens = min(num_assistant_tokens, remaining)
            if progress:
                print(
                    f"[step {step}] prefix_len={prefix_length}, "
                    f"remaining={remaining}, drafting={requested_draft_tokens}",
                    flush=True,
                )
            collect_logits = logit_records is not None
            draft_result = draft_tokens(
                draft_model,
                generated,
                requested_draft_tokens,
                eos_token_ids,
                min_length,
                top_k_logits=top_k_logits,
                collect_logits=collect_logits,
                progress=progress,
                step=step,
                heartbeat_seconds=heartbeat_seconds,
            )
            if draft_result.tokens.shape[-1] == 0:
                break

            target_result = target_predictions_for_draft(
                model,
                generated,
                draft_result.tokens,
                eos_token_ids,
                min_length,
                top_k_logits=top_k_logits,
                collect_logits=collect_logits,
                progress=progress,
                step=step,
                heartbeat_seconds=heartbeat_seconds,
            )
            if progress:
                print(
                    f"[step {step}] draft={draft_result.tokens[0].tolist()} "
                    f"target={target_result.predictions[0].tolist()}",
                    flush=True,
                )

            accepted_count = 0
            rejected_at: int | None = None
            replacement_token: torch.Tensor | None = None
            for idx in range(draft_result.tokens.shape[-1]):
                draft_token = draft_result.tokens[:, idx : idx + 1]
                target_token = target_result.predictions[:, idx : idx + 1]
                if int(draft_token.item()) == int(target_token.item()):
                    accepted_count += 1
                else:
                    rejected_at = idx
                    replacement_token = target_token
                    break

            if replacement_token is None:
                appended = draft_result.tokens
                generated = torch.cat([generated, appended], dim=-1)
                if generated.shape[-1] - prompt_length < max_new_tokens and not should_stop(
                    generated,
                    eos_token_ids,
                    min_length,
                ):
                    bonus_token = target_result.predictions[
                        :,
                        draft_result.tokens.shape[-1] : draft_result.tokens.shape[-1] + 1,
                    ]
                    appended = torch.cat([appended, bonus_token], dim=-1)
                    generated = torch.cat([generated, bonus_token], dim=-1)
            else:
                accepted_tokens = draft_result.tokens[:, :accepted_count]
                appended = torch.cat([accepted_tokens, replacement_token], dim=-1)
                generated = torch.cat([generated, appended], dim=-1)

            if logit_records is not None:
                for idx, logits in enumerate(draft_result.decision_logits):
                    logit_records.append(
                        {
                            "model": "assistant",
                            "step": step,
                            "draft_index": idx,
                            "input_length": draft_result.forward_input_lengths[idx],
                            "selected_token_id": int(draft_result.tokens[0, idx].item()),
                            "logits": logits,
                        }
                    )
                for idx, logits in enumerate(target_result.decision_logits):
                    logit_records.append(
                        {
                            "model": "target",
                            "step": step,
                            "verify_index": idx,
                            "input_length": prefix_length + draft_result.tokens.shape[-1],
                            "prediction_prefix_length": prefix_length + idx,
                            "selected_token_id": int(target_result.predictions[0, idx].item()),
                            "logits": logits,
                        }
                    )

            if trace_steps is not None:
                trace_steps.append(
                    SpeculativeStepTrace(
                        step=step,
                        prefix_length=prefix_length,
                        remaining_new_tokens=remaining,
                        requested_draft_tokens=requested_draft_tokens,
                        draft_forward_input_lengths=draft_result.forward_input_lengths,
                        target_forward_input_length=prefix_length + draft_result.tokens.shape[-1],
                        draft_tokens=draft_result.tokens[0].tolist(),
                        target_predictions=target_result.predictions[0].tolist(),
                        draft_top_logits=draft_result.top_logits,
                        target_top_logits=target_result.top_logits,
                        accepted_count=accepted_count,
                        rejected_at=rejected_at,
                        appended_tokens=appended[0].tolist(),
                        output_length=generated.shape[-1],
                        stop_reason=stop_reason_for(generated, prompt_length, max_new_tokens, eos_token_ids, min_length),
                    )
                )

            if progress:
                print(
                    f"[step {step}] accepted={accepted_count}, rejected_at={rejected_at}, "
                    f"appended={appended[0].tolist()}, output_len={generated.shape[-1]}",
                    flush=True,
                )

            step += 1

    return generated


def generate_with_trace(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    min_length: int = 0,
    eos_token_id: int | Iterable[int] | torch.Tensor | None = None,
    draft_model: torch.nn.Module | None = None,
    num_assistant_tokens: int = 4,
    top_k_logits: int = 0,
    logit_records: list[dict[str, Any]] | None = None,
    progress: bool = False,
    heartbeat_seconds: float = 0.0,
) -> tuple[torch.Tensor, list[SpeculativeStepTrace]]:
    trace_steps: list[SpeculativeStepTrace] = []
    output_ids = generate(
        model,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        eos_token_id=eos_token_id,
        draft_model=draft_model,
        num_assistant_tokens=num_assistant_tokens,
        trace_steps=trace_steps,
        top_k_logits=top_k_logits,
        logit_records=logit_records,
        progress=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    return output_ids, trace_steps


def format_top_logits(summary: LogitTopK, tokenizer: AutoTokenizer | None = None) -> str:
    pieces: list[str] = []
    for token_id, value in zip(summary.token_ids, summary.values):
        if tokenizer is None:
            pieces.append(f"{token_id}:{value:.3f}")
        else:
            pieces.append(f"{token_id}:{value:.3f}:{tokenizer.decode([token_id])!r}")
    return "[" + ", ".join(pieces) + "]"


def print_trace(
    trace_steps: list[SpeculativeStepTrace],
    tokenizer: AutoTokenizer | None = None,
    show_logits: bool = False,
) -> None:
    for item in trace_steps:
        print(f"\nSPEC STEP {item.step}")
        print(f"prefix length: {item.prefix_length}")
        print(f"remaining new-token budget: {item.remaining_new_tokens}")
        print(f"requested draft tokens: {item.requested_draft_tokens}")
        print(f"draft forward input lengths: {item.draft_forward_input_lengths}")
        print(f"target verify input length: {item.target_forward_input_length}")
        print(f"draft tokens: {item.draft_tokens}")
        if tokenizer is not None:
            print(f"draft text: {tokenizer.decode(item.draft_tokens)!r}")
        print(f"target predictions plus bonus: {item.target_predictions}")
        if show_logits:
            for idx, summary in enumerate(item.draft_top_logits):
                print(f"assistant logits top-k for draft[{idx}]: {format_top_logits(summary, tokenizer)}")
            for idx, summary in enumerate(item.target_top_logits):
                print(f"target logits top-k for verify[{idx}]: {format_top_logits(summary, tokenizer)}")
        print(f"accepted count: {item.accepted_count}")
        print(f"rejected at: {item.rejected_at}")
        print(f"appended tokens: {item.appended_tokens}")
        if tokenizer is not None:
            print(f"appended text: {tokenizer.decode(item.appended_tokens)!r}")
        print(f"output length: {item.output_length}")
        print(f"stop reason after step: {item.stop_reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace the local first-principles speculative decoder.")
    parser.add_argument("--target-model", default="gpt2")
    parser.add_argument("--assistant-model", default="distilgpt2")
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer to use. Defaults to --target-model.")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--paragraph", action="store_true", help="Shortcut for a longer 120-token generation.")
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="auto")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Ask Transformers to reduce CPU RAM during model loading.")
    parser.add_argument("--show-logits", action="store_true", help="Print top-k next-token logits for assistant and target decisions.")
    parser.add_argument("--top-k-logits", type=int, default=5, help="Number of logits to print per decision when --show-logits is set.")
    parser.add_argument("--logits-out", default=None, help="Optional .pt file path for full next-token logit vectors used by both models.")
    parser.add_argument("--interactive", action="store_true", help="Load weights once, then run prompts until you type quit or an empty line.")
    parser.add_argument("--progress", action="store_true", help="Print one-line progress updates during generation.")
    parser.add_argument("--heartbeat-seconds", type=float, default=0.0, help="Print a heartbeat while a model forward is still running.")
    return parser.parse_args()


def effective_max_new_tokens(args: argparse.Namespace) -> int:
    if args.paragraph and args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS:
        return 120
    return args.max_new_tokens


def load_models(args: argparse.Namespace, device: torch.device) -> LoadedModels:
    print(f"Using device={device}")
    print_hardware_status(device)

    tokenizer_name = args.tokenizer_model or args.target_model
    print(f"Loading tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.", flush=True)

    model_kwargs: dict[str, Any] = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if getattr(args, "attn_implementation", None):
        model_kwargs["attn_implementation"] = args.attn_implementation
    if getattr(args, "low_cpu_mem_usage", False):
        model_kwargs["low_cpu_mem_usage"] = True
    if model_kwargs:
        print(f"Model load kwargs: {model_kwargs}", flush=True)

    heartbeat_seconds = getattr(args, "heartbeat_seconds", 0.0)

    print(f"Loading target model weights: {args.target_model}", flush=True)
    with timed_operation("target from_pretrained", torch.device("cpu"), heartbeat_seconds):
        target = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs)
    print(f"Moving target model to {device} ...", flush=True)
    with timed_operation("target to device", device, heartbeat_seconds):
        target = target.to(device).eval()
    target.config.use_cache = False
    if getattr(target, "generation_config", None) is not None:
        target.generation_config.use_cache = False
    print(f"Target model ready on {next(target.parameters()).device}.", flush=True)
    print_memory_status(device, "After target load")

    print(f"Loading assistant model weights: {args.assistant_model}", flush=True)
    with timed_operation("assistant from_pretrained", torch.device("cpu"), heartbeat_seconds):
        assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model, **model_kwargs)
    print(f"Moving assistant model to {device} ...", flush=True)
    with timed_operation("assistant to device", device, heartbeat_seconds):
        assistant = assistant.to(device).eval()
    assistant.config.use_cache = False
    if getattr(assistant, "generation_config", None) is not None:
        assistant.generation_config.use_cache = False
    print(f"Assistant model ready on {next(assistant.parameters()).device}.", flush=True)
    print_memory_status(device, "After assistant load")

    target_vocab_size = model_vocab_size(target)
    assistant_vocab_size = model_vocab_size(assistant)
    if target_vocab_size is not None and assistant_vocab_size is not None and target_vocab_size != assistant_vocab_size:
        raise ValueError(
            "Target and assistant vocab sizes differ "
            f"({target_vocab_size} vs {assistant_vocab_size}). "
            "This traceable decoder assumes both models share tokenizer/vocab."
        )
    print(f"Model vocab size: {target_vocab_size}", flush=True)
    return LoadedModels(tokenizer=tokenizer, target=target, assistant=assistant, device=device)


def logits_output_path(base_path: str | None, run_index: int) -> str | None:
    if base_path is None:
        return None
    path = Path(base_path)
    if run_index == 1:
        return str(path)
    suffix = path.suffix
    return str(path.with_name(f"{path.stem}_{run_index}{suffix}"))


def run_prompt(
    args: argparse.Namespace,
    loaded: LoadedModels,
    prompt: str,
    max_new_tokens: int,
    logits_out: str | None = None,
) -> torch.Tensor:
    print("\nTokenizing prompt...", flush=True)
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    logit_records: list[dict[str, Any]] | None = [] if logits_out is not None else None
    print(
        f"Starting speculative generation: prompt_len={inputs.input_ids.shape[-1]}, "
        f"max_new_tokens={max_new_tokens}, draft_len={args.draft_len}",
        flush=True,
    )
    output_ids, trace_steps = generate_with_trace(
        loaded.target,
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=loaded.tokenizer.eos_token_id,
        draft_model=loaded.assistant,
        num_assistant_tokens=args.draft_len,
        top_k_logits=args.top_k_logits if args.show_logits else 0,
        logit_records=logit_records,
        progress=args.progress,
        heartbeat_seconds=getattr(args, "heartbeat_seconds", 0.0),
    )
    print(f"Generation finished after {len(trace_steps)} speculative step(s).", flush=True)
    print_memory_status(loaded.device, "After generation")

    print(f"\nprompt token ids: {inputs.input_ids[0].tolist()}")
    print(f"prompt length: {inputs.input_ids.shape[-1]}")
    print(f"max new tokens: {max_new_tokens}")
    print_trace(trace_steps, loaded.tokenizer, show_logits=args.show_logits)

    if logits_out is not None:
        torch.save(
            {
                "target_model": args.target_model,
                "assistant_model": args.assistant_model,
                "prompt": prompt,
                "prompt_token_ids": inputs.input_ids.detach().cpu(),
                "output_ids": output_ids.detach().cpu(),
                "records": logit_records,
            },
            logits_out,
        )
        print(f"\nSaved full decision logits to: {logits_out}")

    print("\nFINAL")
    print(f"token ids: {output_ids[0].tolist()}")
    print(f"decoded: {loaded.tokenizer.decode(output_ids[0], skip_special_tokens=True)}")
    return output_ids


def interactive_loop(args: argparse.Namespace, loaded: LoadedModels, max_new_tokens: int) -> None:
    print("\nInteractive mode: weights are loaded once.")
    print("Type a prompt and press Enter. Type quit/exit, or submit an empty line, to stop.")

    run_index = 1
    while True:
        try:
            prompt = input("\nprompt> ")
        except EOFError:
            print()
            break

        if prompt.strip().lower() in {"", "q", "quit", "exit"}:
            break

        run_prompt(
            args,
            loaded,
            prompt,
            max_new_tokens=max_new_tokens,
            logits_out=logits_output_path(args.logits_out, run_index),
        )
        run_index += 1


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    max_new_tokens = effective_max_new_tokens(args)
    loaded = load_models(args, device)

    if args.interactive:
        interactive_loop(args, loaded, max_new_tokens)
        return

    run_prompt(
        args,
        loaded,
        args.prompt,
        max_new_tokens=max_new_tokens,
        logits_out=args.logits_out,
    )


if __name__ == "__main__":
    main()
