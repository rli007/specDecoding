#!/usr/bin/env python
"""Run the first-principles EAGLE-1 decoder on MT-Bench-style questions.

Counterpart of run_medusa_mtbench.py / run_assisted_mtbench.py: same harness,
same trace/answer formats, same two component CSVs, so all four methods share
one schema for the run-counts x per-run-latency cost model.

Component semantics per step (from EagleStepTrace.drafter_metadata):

- eagle_draft_warmup   drafter forward over ALL (feature, next-token) pairs of
                       the current sequence. Implementation artifact: the
                       drafter cache is rebuilt each step for inspectability;
                       on hardware the cache is carried and only the newly
                       accepted pairs are ingested. EXCLUDE from hardware
                       costing (same convention as target_cache_rebuild).
- eagle_draft_step     one drafter forward per tree depth over that depth's
                       frontier (widths 4,8,8,3,2 for the default tree). The
                       sequential-but-tiny drafting that IS EAGLE's cost.
- tree_target_forward  one target forward over the 26 tree nodes (cost equals
                       a 26-token decode step; see the Medusa analysis).
- kv_cache_gather      accepted-path KV select-and-compact (DMA, not matmul).
"""

from __future__ import annotations

import argparse
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

from decoders.eagle_speculative_decoder import (
    DEFAULT_EAGLE_DRAFTER,
    EagleOneDrafter,
    EagleStepTrace,
    generate_with_trace,
    load_official_eagle_drafter,
)
from decoders.first_principles_speculative_decoder import (
    choose_device,
    dtype_from_arg,
    memory_status,
    print_hardware_status,
    timed_operation,
)
from tools.run_assisted_mtbench import COMPONENT_FIELDS, STEP_FIELDS, append_csv_rows
from tools.run_medusa_mtbench import (
    DEFAULT_STOP_STRINGS,
    append_jsonl,
    build_prompt,
    default_trace_path,
    read_questions,
    trim_answer,
    truncate_file,
    write_run_config,
)

DEFAULT_TARGET_MODEL = "lmsys/vicuna-7b-v1.3"
DEFAULT_QUESTION_FILE = ROOT / "examples" / "mini_mtbench_questions.jsonl"
DEFAULT_ANSWER_FILE = ROOT / "run_logs" / "eagle_mtbench_mini_answers.jsonl"
DEFAULT_MODEL_ID = "first-principles-eagle1-vicuna-7b-v1.3"


def step_component_events(step: EagleStepTrace, prefix_length: int) -> list[tuple[str, int]]:
    """(component, input_positions) events for one step, in execution order."""
    meta = step.drafter_metadata
    positions = meta.get("draft_forward_positions", [])
    events: list[tuple[str, int]] = []
    if positions:
        events.append(("eagle_draft_warmup", positions[0]))
        events.extend(("eagle_draft_step", width) for width in positions[1:])
    events.append(("tree_target_forward", meta.get("tree_nodes_incl_root", 0)))
    if meta.get("cache_updated"):
        events.append(("kv_cache_gather", len(step.appended_tokens)))
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MT-Bench-style answers with the first-principles EAGLE-1 decoder."
    )
    parser.add_argument("--question-file", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--answers-jsonl", default=str(DEFAULT_ANSWER_FILE))
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=2, help="Number of questions. 0 = all.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--eagle-drafter", default=DEFAULT_EAGLE_DRAFTER)
    parser.add_argument("--tokenizer-model", default=None, help="Defaults to --target-model.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="eager is required: fused kernels may mangle the explicit 4D tree mask.",
    )
    parser.add_argument("--no-low-cpu-mem-usage", action="store_false", dest="low_cpu_mem_usage")
    parser.add_argument("--verifier", choices=("tree", "slow"), default="tree")
    parser.add_argument("--prompt-style", choices=("vicuna", "plain"), default="vicuna")
    parser.add_argument("--stop-string", action="append", default=None)
    parser.add_argument("--no-step-text", action="store_false", dest="step_text")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
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

    print("EAGLE-1 MT-Bench-style generation")
    print(f"target model: {args.target_model}")
    print(f"eagle drafter: {args.eagle_drafter}")
    print(f"verifier: {args.verifier}, max_new_tokens: {args.max_new_tokens}")
    print(f"answers jsonl: {answer_path}")
    print_hardware_status(device)

    questions = read_questions(question_path, limit=args.limit, offset=args.offset)
    if not questions:
        print("No questions selected; exiting.")
        return
    print(f"Loaded {len(questions)} question(s).")

    truncate_file(answer_path)
    truncate_file(trace_path)
    truncate_file(components_path)
    truncate_file(steps_path)
    print(f"run config: {write_run_config(answer_path, args)}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True

    print(f"\nLoading target model: {args.target_model}", flush=True)
    with timed_operation("target model from_pretrained", torch.device("cpu"), args.heartbeat_seconds):
        target = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs)
    with timed_operation("target model to device", device, args.heartbeat_seconds):
        target = target.to(device).eval()
    target.config.output_hidden_states = True
    print(f"Target ready. Memory: {memory_status(device)}")

    print(f"\nLoading EAGLE drafter: {args.eagle_drafter}", flush=True)
    head_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
    with timed_operation("eagle drafter load", device, args.heartbeat_seconds):
        drafter_model = load_official_eagle_drafter(args.eagle_drafter, device=device, dtype=head_dtype)
    drafter = EagleOneDrafter(drafter_model, lm_head=target.lm_head)
    print(f"Drafter ready ({sum(p.numel() for p in drafter_model.parameters()) / 1e6:.0f}M params).")
    print(f"Tree: {len(drafter.choices)} paths, {len(drafter.nodes_template) + 1} nodes incl. free root.")

    suite_started = time.perf_counter()
    for question_index, question in enumerate(questions, start=1):
        print(
            f"\nQUESTION {question_index}/{len(questions)} id={question.question_id!r} category={question.category!r}",
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
                f"[question {question.question_id} turn {turn_index}] prompt_tokens={prompt_length} generating...",
                flush=True,
            )

            started = time.perf_counter()
            with torch.inference_mode():
                output_ids, trace_steps = generate_with_trace(
                    target,
                    drafter,
                    inputs.input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    verifier=args.verifier,
                    progress=args.progress,
                    heartbeat_seconds=args.heartbeat_seconds,
                )
            elapsed = time.perf_counter() - started

            generated_ids = output_ids[0, prompt_length:]
            raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = trim_answer(raw_answer, stop_strings)
            answers.append(answer)
            history.append(("assistant", answer))

            steps = len(trace_steps)
            appended_total = sum(len(step.appended_tokens) for step in trace_steps)
            component_rows: list[dict[str, Any]] = []
            step_rows: list[dict[str, Any]] = []
            event_index = 0
            for step_trace in trace_steps:
                events = step_component_events(step_trace, step_trace.prefix_length)
                sequence_parts: list[list[Any]] = []
                for component, positions in events:
                    component_rows.append(
                        {
                            "method": args.model_id,
                            "question_id": question.question_id,
                            "category": question.category,
                            "turn_index": turn_index,
                            "step": step_trace.step,
                            "event_index": event_index,
                            "component": component,
                            "input_positions": positions,
                            "cache_len_before": step_trace.prefix_length,
                        }
                    )
                    event_index += 1
                    if sequence_parts and sequence_parts[-1][0] == component:
                        sequence_parts[-1][1] += 1
                    else:
                        sequence_parts.append([component, 1])
                step_rows.append(
                    {
                        "method": args.model_id,
                        "question_id": question.question_id,
                        "category": question.category,
                        "turn_index": turn_index,
                        "step": step_trace.step,
                        "component_sequence": ", ".join(
                            name if count == 1 else f"{name} x{count}" for name, count in sequence_parts
                        ),
                        "draft_calls": sum(1 for c, _ in events if c.startswith("eagle_draft")),
                        "target_calls": sum(1 for c, _ in events if c == "tree_target_forward"),
                        "assistant_budget": "",
                        "accepted_assistant_tokens": len(step_trace.appended_tokens) - 1,
                        "appended_tokens": len(step_trace.appended_tokens),
                        "output_length": step_trace.output_length,
                    }
                )
            append_csv_rows(components_path, COMPONENT_FIELDS, component_rows)
            append_csv_rows(steps_path, STEP_FIELDS, step_rows)

            stats: dict[str, Any] = {
                "turn_index": turn_index,
                "prompt_token_count": prompt_length,
                "generated_token_count": int(generated_ids.shape[-1]),
                "elapsed_seconds": elapsed,
                "steps": steps,
                "appended_token_count": appended_total,
                "appended_tokens_per_step": appended_total / steps if steps else 0.0,
                "accept_length_per_step": (appended_total - steps) / steps if steps else 0.0,
                "cache_updated_all_steps": all(
                    step.drafter_metadata.get("cache_updated", False) for step in trace_steps
                ),
                "raw_answer": raw_answer,
            }
            turn_stats.append(stats)
            print(
                f"[question {question.question_id} turn {turn_index}] "
                f"done tokens={stats['generated_token_count']} steps={steps} "
                f"tokens/step={stats['appended_tokens_per_step']:.2f} "
                f"accept_length/step={stats['accept_length_per_step']:.2f} "
                f"tok/s={stats['generated_token_count'] / elapsed:.3f}",
                flush=True,
            )
            if args.step_text:
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
                    "trace": [
                        {
                            "step": step.step,
                            "prefix_length": step.prefix_length,
                            "tree_node_count": step.drafter_metadata.get("tree_nodes_incl_root"),
                            "selected_path_index": step.selected_path_index,
                            "selected_path_tokens": step.selected_path_tokens,
                            "accept_length": len(step.appended_tokens) - 1,
                            "tokens_per_step": len(step.appended_tokens),
                            "appended_tokens": step.appended_tokens,
                            "output_length": step.output_length,
                            "cache_updated": step.drafter_metadata.get("cache_updated"),
                            "draft_forward_positions": step.drafter_metadata.get("draft_forward_positions"),
                            "stop_reason": step.stop_reason,
                        }
                        for step in trace_steps
                    ],
                },
            )

        append_jsonl(
            answer_path,
            {
                "question_id": question.question_id,
                "answer_id": uuid.uuid4().hex,
                "model_id": args.model_id,
                "choices": [{"index": 0, "turns": answers}],
                "tstamp": time.time(),
                "decoding_stats": turn_stats,
            },
        )

    suite_elapsed = time.perf_counter() - suite_started
    print("\nEAGLE MT-Bench-style generation complete.")
    print(f"questions: {len(questions)}, elapsed: {suite_elapsed:.2f}s")
    print(f"answers: {answer_path}")
    print(f"traces: {trace_path}")
    print(f"components: {components_path}")
    print(f"steps: {steps_path}")


if __name__ == "__main__":
    main()
