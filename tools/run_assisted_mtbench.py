#!/usr/bin/env python
"""Run the stripped-down draft-model (assisted) decoder on MT-Bench-style questions.

Counterpart of run_medusa_mtbench.py for "normal" speculative decoding: a small
draft model proposes k tokens per step, the target verifies them in one forward.
Besides the usual answers/traces JSONL pair, this runner records **every model
forward** as one component event, so the output directly answers "what ran, how
many times, in what order" for hardware cost modeling:

- <answers stem>.components.csv  one row per model forward, in execution order
- <answers stem>.steps.csv       one row per speculative step (component sequence
                                 plus acceptance counts)

Component names and their hardware meaning:

- draft_prefill        assistant forward over the whole current sequence. The
                       stripped decoder rebuilds the assistant cache each step;
                       HF instead keeps the cache, so on hardware this is a
                       forward over only the newly accepted tokens.
- draft_decode         assistant forward over 1 token (the sequential drafting).
- target_cache_rebuild target forward over the prefix, cache-rebuild artifact of
                       this implementation. HF crops the target cache instead;
                       on hardware this is a cache truncation (DMA), not a forward.
- target_verify        target forward over k+1 tokens (last accepted token plus
                       the k draft tokens). This is the parallel verification.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders.first_principles_speculative_decoder import (
    choose_device,
    memory_status,
    print_hardware_status,
    timed_operation,
)
from decoders.stripped_down_llama_assisted_decoder import (
    ASSISTANT_SCHEDULES,
    AssistedStepTrace,
    assisted_generate,
    dtype_from_arg,
)
from tools.run_medusa_mtbench import (
    DEFAULT_STOP_STRINGS,
    Question,
    append_jsonl,
    build_prompt,
    default_trace_path,
    read_questions,
    trim_answer,
    truncate_file,
    write_run_config,
)


DEFAULT_TARGET_MODEL = "lmsys/vicuna-7b-v1.3"
DEFAULT_ASSISTANT_MODEL = "JackFram/llama-160m"
DEFAULT_QUESTION_FILE = ROOT / "examples" / "mini_mtbench_questions.jsonl"
DEFAULT_ANSWER_FILE = ROOT / "run_logs" / "assisted_mtbench_mini_answers.jsonl"
DEFAULT_MODEL_ID = "stripped-assisted-vicuna-7b-llama-160m"

COMPONENT_FIELDS = (
    "method",
    "question_id",
    "category",
    "turn_index",
    "step",
    "event_index",
    "component",
    "input_positions",
    "cache_len_before",
)
STEP_FIELDS = (
    "method",
    "question_id",
    "category",
    "turn_index",
    "step",
    "component_sequence",
    "draft_calls",
    "target_calls",
    "assistant_budget",
    "accepted_assistant_tokens",
    "appended_tokens",
    "output_length",
)


@dataclass
class ComponentEvent:
    event_index: int
    component: str
    input_positions: int
    cache_len_before: int
    step: int | None = None


@dataclass
class ComponentLog:
    """Ordered log of model forwards for one generation call."""

    events: list[ComponentEvent] = field(default_factory=list)
    _unassigned_from: int = 0

    def record(self, component: str, input_positions: int, cache_len_before: int) -> None:
        self.events.append(
            ComponentEvent(
                event_index=len(self.events),
                component=component,
                input_positions=input_positions,
                cache_len_before=cache_len_before,
            )
        )

    def assign_step(self, step: int) -> list[ComponentEvent]:
        """Tag every event since the previous step boundary with this step number."""
        assigned = self.events[self._unassigned_from :]
        for event in assigned:
            event.step = step
        self._unassigned_from = len(self.events)
        return assigned


def cache_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    try:
        return int(past_key_values.get_seq_length())
    except (AttributeError, TypeError):
        return -1


class LoggedModel:
    """Wrap a causal LM so every forward lands in the component log.

    Only the call shape used by the stripped decoder is supported; the
    component is classified from that shape:
    - assistant: >1 positions = draft_prefill, 1 position = draft_decode
    - target: no cache = target_cache_rebuild, with cache = target_verify
    """

    def __init__(self, model: torch.nn.Module, role: str, log: ComponentLog):
        self._model = model
        self._role = role
        self._log = log

    @property
    def device(self) -> torch.device:
        return self._model.device

    @property
    def config(self):
        return self._model.config

    def __call__(self, *, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = True):
        positions = int(input_ids.shape[-1])
        cache_len = cache_length(past_key_values)
        if self._role == "assistant":
            component = "draft_prefill" if positions > 1 or cache_len == 0 else "draft_decode"
        else:
            component = "target_cache_rebuild" if past_key_values is None else "target_verify"
        self._log.record(component, positions, cache_len)
        return self._model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)


def summarize_components(events: list[ComponentEvent]) -> str:
    """Compact sequence like 'draft_prefill, draft_decode x4, target_verify'."""
    parts: list[str] = []
    for event in events:
        name = event.component
        if parts and parts[-1][0] == name:
            parts[-1][1] += 1
        else:
            parts.append([name, 1])
    return ", ".join(name if count == 1 else f"{name} x{count}" for name, count in parts)


def component_totals(events: list[ComponentEvent]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for event in events:
        entry = totals.setdefault(event.component, {"calls": 0, "positions": 0})
        entry["calls"] += 1
        entry["positions"] += event.input_positions
    return dict(sorted(totals.items()))


def append_csv_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MT-Bench-style answers with the stripped draft-model assisted decoder."
    )
    parser.add_argument("--question-file", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--answers-jsonl", default=str(DEFAULT_ANSWER_FILE))
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="Defaults to the answer path with .traces.jsonl appended before the extension.",
    )
    parser.add_argument("--limit", type=int, default=2, help="Number of questions to run. Use 0 for all.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many questions from the input file.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--tokenizer-model", default=None, help="Defaults to --target-model.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-assistant-tokens", type=int, default=5, help="Draft length k.")
    parser.add_argument(
        "--assistant-schedule",
        choices=ASSISTANT_SCHEDULES,
        default="constant",
        help="constant keeps k fixed (hardware-schedulable); heuristic mirrors HF's dynamic budget.",
    )
    parser.add_argument("--mode", choices=("greedy", "sampling"), default="greedy")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--no-low-cpu-mem-usage", action="store_false", dest="low_cpu_mem_usage")
    parser.add_argument(
        "--prompt-style",
        choices=("vicuna", "plain"),
        default="vicuna",
        help="Conversation prompt formatter. Use vicuna for lmsys/vicuna-* models.",
    )
    parser.add_argument(
        "--stop-string",
        action="append",
        default=None,
        help="Trim generated answer at this string. Can be repeated. Defaults trim USER:/ASSISTANT: markers.",
    )
    parser.add_argument("--no-step-text", action="store_false", dest="step_text")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Read questions and print prompts without loading models.")
    parser.set_defaults(low_cpu_mem_usage=True, step_text=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.target_model
    question_path = Path(args.question_file).expanduser()
    answer_path = Path(args.answers_jsonl).expanduser()
    trace_path = Path(args.trace_jsonl).expanduser() if args.trace_jsonl else default_trace_path(answer_path)
    components_path = answer_path.with_name(f"{answer_path.stem}.components.csv")
    steps_path = answer_path.with_name(f"{answer_path.stem}.steps.csv")
    stop_strings = list(args.stop_string) if args.stop_string is not None else list(DEFAULT_STOP_STRINGS)

    print("Assisted (draft-model) MT-Bench-style generation")
    print(f"question file: {question_path}")
    print(f"answers jsonl: {answer_path}")
    print(f"trace jsonl: {trace_path}")
    print(f"components csv: {components_path}")
    print(f"steps csv: {steps_path}")
    print(f"target model: {args.target_model}")
    print(f"assistant model: {args.assistant_model}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"max_new_tokens per turn: {args.max_new_tokens}")
    print(f"draft length k: {args.num_assistant_tokens} ({args.assistant_schedule} schedule)")
    print(f"mode: {args.mode}")
    print_hardware_status(device)

    questions = read_questions(question_path, limit=args.limit, offset=args.offset)
    if not questions:
        print("No questions selected; exiting.")
        return
    print(f"Loaded {len(questions)} question(s).")

    if args.dry_run:
        print("\nDry run: printing formatted prompts only.")
        for question_index, question in enumerate(questions, start=1):
            print(
                f"\nQUESTION {question_index}/{len(questions)} "
                f"id={question.question_id!r} category={question.category!r}"
            )
            history: list[tuple[str, str]] = []
            for turn_index, user_turn in enumerate(question.turns, start=1):
                history.append(("user", user_turn))
                print(f"\nPROMPT question={question.question_id!r} turn={turn_index}")
                print(build_prompt(history, args.prompt_style))
                history.append(("assistant", "<generated answer would be inserted here>"))
        return

    truncate_file(answer_path)
    truncate_file(trace_path)
    truncate_file(components_path)
    truncate_file(steps_path)
    print(f"run config: {write_run_config(answer_path, args)}")

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.", flush=True)

    model_kwargs: dict[str, Any] = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True

    print(f"\nLoading target model: {args.target_model}", flush=True)
    with timed_operation("target model from_pretrained", torch.device("cpu"), args.heartbeat_seconds):
        target = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs)
    with timed_operation("target model to device", device, args.heartbeat_seconds):
        target = target.to(device).eval()
    print(f"Target ready. Memory: {memory_status(device)}")

    print(f"\nLoading assistant model: {args.assistant_model}", flush=True)
    with timed_operation("assistant model from_pretrained", torch.device("cpu"), args.heartbeat_seconds):
        assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model, **model_kwargs)
    with timed_operation("assistant model to device", device, args.heartbeat_seconds):
        assistant = assistant.to(device).eval()
    print(f"Assistant ready. Memory: {memory_status(device)}")

    if getattr(target.config, "vocab_size", None) != getattr(assistant.config, "vocab_size", None):
        print(
            "WARNING: target and assistant vocab sizes differ "
            f"({target.config.vocab_size} vs {assistant.config.vocab_size}); "
            "assisted decoding assumes a shared tokenizer."
        )

    suite_started = time.perf_counter()
    for question_index, question in enumerate(questions, start=1):
        print(
            f"\nQUESTION {question_index}/{len(questions)} "
            f"id={question.question_id!r} category={question.category!r}",
            flush=True,
        )
        history: list[tuple[str, str]] = []
        answers: list[str] = []
        turn_stats: list[dict[str, Any]] = []

        for turn_index, user_turn in enumerate(question.turns, start=1):
            history.append(("user", user_turn))
            prompt = build_prompt(history, args.prompt_style)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_length = inputs.input_ids.shape[-1]
            print(
                f"[question {question.question_id} turn {turn_index}] "
                f"prompt_tokens={prompt_length} generating...",
                flush=True,
            )

            log = ComponentLog()
            logged_target = LoggedModel(target, "target", log)
            logged_assistant = LoggedModel(assistant, "assistant", log)
            trace_steps: list[AssistedStepTrace] = []
            step_rows: list[dict[str, Any]] = []

            def on_step(step_trace: AssistedStepTrace) -> None:
                step_events = log.assign_step(step_trace.step)
                draft_calls = sum(1 for event in step_events if event.component.startswith("draft"))
                target_calls = sum(1 for event in step_events if event.component.startswith("target"))
                step_rows.append(
                    {
                        "method": args.model_id,
                        "question_id": question.question_id,
                        "category": question.category,
                        "turn_index": turn_index,
                        "step": step_trace.step,
                        "component_sequence": summarize_components(step_events),
                        "draft_calls": draft_calls,
                        "target_calls": target_calls,
                        "assistant_budget": step_trace.assistant_budget,
                        "accepted_assistant_tokens": step_trace.accepted_assistant_tokens,
                        "appended_tokens": len(step_trace.appended_tokens),
                        "output_length": step_trace.output_length,
                    }
                )
                if args.step_text:
                    print(
                        f"[question {question.question_id} turn {turn_index} step {step_trace.step}] "
                        f"{step_rows[-1]['component_sequence']} | "
                        f"accepted={step_trace.accepted_assistant_tokens}/{step_trace.assistant_budget} "
                        f"appended={len(step_trace.appended_tokens)}",
                        flush=True,
                    )

            started = time.perf_counter()
            with torch.inference_mode():
                output_ids = assisted_generate(
                    logged_target,
                    logged_assistant,
                    inputs.input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    num_assistant_tokens=args.num_assistant_tokens,
                    assistant_schedule=args.assistant_schedule,
                    mode=args.mode,
                    verbose=False,
                    trace_steps=trace_steps,
                    step_callback=on_step,
                )
            elapsed = time.perf_counter() - started

            generated_ids = output_ids[0, prompt_length:]
            raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = trim_answer(raw_answer, stop_strings)
            answers.append(answer)
            history.append(("assistant", answer))

            steps = len(trace_steps)
            accepted_total = sum(step.accepted_assistant_tokens for step in trace_steps)
            appended_total = sum(len(step.appended_tokens) for step in trace_steps)
            totals = component_totals(log.events)
            stats: dict[str, Any] = {
                "turn_index": turn_index,
                "prompt_token_count": prompt_length,
                "generated_token_count": int(generated_ids.shape[-1]),
                "elapsed_seconds": elapsed,
                "steps": steps,
                "accepted_assistant_token_count": accepted_total,
                "appended_token_count": appended_total,
                "accepted_assistant_tokens_per_step": accepted_total / steps if steps else 0.0,
                "appended_tokens_per_step": appended_total / steps if steps else 0.0,
                "component_totals": totals,
                "raw_answer": raw_answer,
            }
            turn_stats.append(stats)

            component_rows = [
                {
                    "method": args.model_id,
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn_index": turn_index,
                    "step": event.step,
                    "event_index": event.event_index,
                    "component": event.component,
                    "input_positions": event.input_positions,
                    "cache_len_before": event.cache_len_before,
                }
                for event in log.events
            ]
            append_csv_rows(components_path, COMPONENT_FIELDS, component_rows)
            append_csv_rows(steps_path, STEP_FIELDS, step_rows)

            print(
                f"[question {question.question_id} turn {turn_index}] "
                f"done tokens={stats['generated_token_count']} steps={steps} "
                f"tokens/step={stats['appended_tokens_per_step']:.2f} "
                f"accepted/step={stats['accepted_assistant_tokens_per_step']:.2f}",
                flush=True,
            )
            for component, entry in totals.items():
                print(
                    f"[question {question.question_id} turn {turn_index}] "
                    f"  {component}: {entry['calls']} calls, {entry['positions']} positions",
                    flush=True,
                )
            print(f"[question {question.question_id} turn {turn_index}] answer: {answer!r}", flush=True)

            append_jsonl(
                trace_path,
                {
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn_index": turn_index,
                    "prompt": prompt,
                    "user_turn": user_turn,
                    "answer": answer,
                    "stats": stats,
                    "trace": [asdict(step) for step in trace_steps],
                },
            )

        answer_payload = {
            "question_id": question.question_id,
            "answer_id": uuid.uuid4().hex,
            "model_id": args.model_id,
            "choices": [{"index": 0, "turns": answers}],
            "tstamp": time.time(),
            "decoding_stats": turn_stats,
        }
        append_jsonl(answer_path, answer_payload)

    suite_elapsed = time.perf_counter() - suite_started
    print("\nAssisted MT-Bench-style generation complete.")
    print(f"questions: {len(questions)}")
    print(f"elapsed: {suite_elapsed:.2f}s")
    print(f"answers: {answer_path}")
    print(f"traces: {trace_path}")
    print(f"components: {components_path}")
    print(f"steps: {steps_path}")


if __name__ == "__main__":
    main()
