#!/usr/bin/env python
"""Run plain Vicuna greedy generation on MT-Bench-style questions."""

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

from decoders.first_principles_speculative_decoder import (
    choose_device,
    dtype_from_arg,
    memory_status,
    print_hardware_status,
    timed_operation,
)
from tools.run_local_medusa import DEFAULT_BASE_MODEL, model_kwargs_from_args
from tools.run_medusa_mtbench import (
    DEFAULT_QUESTION_FILE,
    DEFAULT_STOP_STRINGS,
    build_prompt,
    default_trace_path,
    read_questions,
    trim_answer,
)


DEFAULT_ANSWER_FILE = ROOT / "run_logs" / "vicuna_mtbench_baseline_answers.jsonl"
DEFAULT_MODEL_ID = "vicuna-7b-v1.3-greedy-baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MT-Bench-style answers with plain Vicuna greedy decoding.")
    parser.add_argument("--question-file", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--answers-jsonl", default=str(DEFAULT_ANSWER_FILE))
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tokenizer-model", default=None, help="Defaults to --base-model.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--no-low-cpu-mem-usage", action="store_false", dest="low_cpu_mem_usage")
    parser.add_argument("--prompt-style", choices=("vicuna", "plain"), default="vicuna")
    parser.add_argument("--stop-string", action="append", default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-step-text", action="store_false", dest="step_text")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.set_defaults(low_cpu_mem_usage=True, step_text=True)
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def truncate_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def select_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def greedy_generate_cached(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    progress: bool,
    heartbeat_seconds: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    generated = input_ids
    trace: list[dict[str, Any]] = []
    past_key_values = None
    next_input = input_ids

    for step in range(1, max_new_tokens + 1):
        label = f"[baseline step {step}] model forward" if progress or heartbeat_seconds > 0 else None
        started = time.perf_counter()
        with timed_operation(label, input_ids.device, heartbeat_seconds) if label else torch.no_grad():
            outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
        elapsed = time.perf_counter() - started
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        past_key_values = getattr(outputs, "past_key_values", None)
        next_token = select_next_token(logits)
        generated = torch.cat([generated, next_token], dim=-1)
        token_id = int(next_token.item())
        trace.append(
            {
                "step": step,
                "input_length": int(next_input.shape[-1]),
                "token_id": token_id,
                "elapsed_seconds": elapsed,
            }
        )
        if progress:
            print(
                f"[baseline step {step}] input_len={next_input.shape[-1]} "
                f"token={token_id} elapsed={elapsed:.2f}s",
                flush=True,
            )
        if eos_token_id is not None and token_id == int(eos_token_id):
            break
        next_input = next_token

    return generated, trace


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.base_model
    question_path = Path(args.question_file).expanduser()
    answer_path = Path(args.answers_jsonl).expanduser()
    trace_path = Path(args.trace_jsonl).expanduser() if args.trace_jsonl else default_trace_path(answer_path)
    stop_strings = list(args.stop_string) if args.stop_string is not None else list(DEFAULT_STOP_STRINGS)

    print("Vicuna MT-Bench greedy baseline")
    print(f"question file: {question_path}")
    print(f"answers jsonl: {answer_path}")
    print(f"trace jsonl: {trace_path}")
    print(f"base model: {args.base_model}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"max_new_tokens per turn: {args.max_new_tokens}")
    print_hardware_status(device)

    questions = read_questions(question_path, limit=args.limit, offset=args.offset)
    print(f"Loaded {len(questions)} question(s).")
    truncate_file(answer_path)
    truncate_file(trace_path)

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.", flush=True)

    model_kwargs = model_kwargs_from_args(args)
    print(f"\nLoading base model: {args.base_model}", flush=True)
    if model_kwargs:
        print(f"Model load kwargs: {model_kwargs}", flush=True)
    with timed_operation("base model from_pretrained", torch.device("cpu"), args.heartbeat_seconds):
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    print(f"Moving base model to {device} ...", flush=True)
    with timed_operation("base model to device", device, args.heartbeat_seconds):
        model = model.to(device).eval()
    print(f"Base model ready on {next(model.parameters()).device}.")
    print(f"Memory after base load: {memory_status(device)}")

    suite_started = time.perf_counter()
    for question_index, question in enumerate(questions, start=1):
        print(f"\nQUESTION {question_index}/{len(questions)} id={question.question_id!r}", flush=True)
        history: list[tuple[str, str]] = []
        answers: list[str] = []
        turn_stats: list[dict[str, Any]] = []

        for turn_index, user_turn in enumerate(question.turns, start=1):
            history.append(("user", user_turn))
            prompt = build_prompt(history, args.prompt_style)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_length = int(inputs.input_ids.shape[-1])
            print(
                f"[baseline question {question.question_id} turn {turn_index}] "
                f"prompt_tokens={prompt_length} generating...",
                flush=True,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                output_ids, trace = greedy_generate_cached(
                    model,
                    inputs.input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    progress=args.progress,
                    heartbeat_seconds=args.heartbeat_seconds,
                )
            elapsed = time.perf_counter() - started
            generated_ids = output_ids[0, prompt_length:]
            raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = trim_answer(raw_answer, stop_strings)
            answers.append(answer)
            history.append(("assistant", answer))

            if args.step_text:
                print(f"[baseline question {question.question_id} turn {turn_index}] answer: {answer!r}", flush=True)

            stats = {
                "turn_index": turn_index,
                "prompt_token_count": prompt_length,
                "generated_token_count": int(generated_ids.shape[-1]),
                "elapsed_seconds": elapsed,
                "tokens_per_second": int(generated_ids.shape[-1]) / elapsed if elapsed > 0 else None,
                "raw_answer": raw_answer,
            }
            turn_stats.append(stats)
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
                    "trace": trace,
                },
            )
            print(
                f"[baseline question {question.question_id} turn {turn_index}] "
                f"done tokens={stats['generated_token_count']} tok/s={stats['tokens_per_second']:.3f}",
                flush=True,
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
    print("\nVicuna baseline generation complete.")
    print(f"questions: {len(questions)}")
    print(f"elapsed: {suite_elapsed:.2f}s")
    print(f"answers: {answer_path}")
    print(f"traces: {trace_path}")


if __name__ == "__main__":
    main()
