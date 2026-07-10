#!/usr/bin/env python
"""Run the first-principles Medusa decoder on MT-Bench-style questions.

This is a lightweight generation harness, not a full judge harness. It loads
the base model and Medusa heads once, loops over MT-Bench JSONL questions, and
writes:

- an answer JSONL in the usual FastChat-ish shape
- a trace JSONL with per-turn Medusa acceptance/stat details

Use the tiny bundled question file for smoke tests, then point --question-file
at the official FastChat MT-Bench question.jsonl when you are ready.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from decoders.medusa_speculative_decoder import MedusaStepTrace, generate_with_trace, load_official_medusa_heads
from tools.run_local_medusa import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MEDUSA_HEADS,
    medusa_choices_for,
    model_kwargs_from_args,
)


DEFAULT_QUESTION_FILE = ROOT / "examples" / "mini_mtbench_questions.jsonl"
DEFAULT_ANSWER_FILE = ROOT / "run_logs" / "medusa_mtbench_mini_answers.jsonl"
DEFAULT_MODEL_ID = "first-principles-medusa-vicuna-7b-v1.3"
VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
DEFAULT_STOP_STRINGS = ("USER:", "ASSISTANT:")


@dataclass
class Question:
    question_id: str | int
    category: str
    turns: list[str]
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MT-Bench-style answers with the local first-principles Medusa decoder."
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
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--medusa-heads", default=DEFAULT_MEDUSA_HEADS)
    parser.add_argument("--tokenizer-model", default=None, help="Defaults to --base-model.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--no-low-cpu-mem-usage", action="store_false", dest="low_cpu_mem_usage")
    parser.add_argument("--medusa-num-heads", type=int, default=None, help="Override head count if checkpoint config is missing.")
    parser.add_argument("--medusa-num-layers", type=int, default=None, help="Override residual layers per head if config is missing.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k Medusa choices per head used by candidate buffers.")
    parser.add_argument(
        "--choice-preset",
        choices=("linear", "small-tree", "official-vicuna-7b", "official-vicuna-13b", "official-zephyr"),
        default="official-vicuna-7b",
        help="Official presets are sparse Medusa trees; linear is the smallest debugging path.",
    )
    parser.add_argument("--verifier", choices=("tree", "slow"), default="tree")
    parser.add_argument("--acceptance", choices=("greedy", "typical", "nucleus"), default="greedy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--posterior-threshold", type=float, default=0.09)
    parser.add_argument("--posterior-alpha", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--no-kv-cache", action="store_false", dest="use_kv_cache")
    parser.add_argument("--slow-fallback", action="store_true", dest="fallback_to_slow")
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
    parser.add_argument("--top-k-logits", type=int, default=0, help="Store target top-k logits in trace; 0 disables.")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-step-text", action="store_false", dest="step_text")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Read questions and print prompts without loading models.")
    parser.set_defaults(low_cpu_mem_usage=True, step_text=True, use_kv_cache=True, fallback_to_slow=False)
    return parser.parse_args()


def default_trace_path(answer_path: Path) -> Path:
    return answer_path.with_name(f"{answer_path.stem}.traces{answer_path.suffix}")


def read_questions(path: Path, limit: int, offset: int) -> list[Question]:
    if not path.exists():
        raise FileNotFoundError(
            f"Question file not found: {path}\n"
            "For the official MT-Bench file, download FastChat's question.jsonl "
            "and pass it with --question-file."
        )

    questions: list[Question] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            turns = item.get("turns")
            if isinstance(turns, str):
                turns = [turns]
            if not isinstance(turns, list) or not all(isinstance(turn, str) for turn in turns):
                raise ValueError(f"{path}:{line_number} must contain a string list field named 'turns'.")
            question_id = item.get("question_id", line_number)
            category = item.get("category", "unknown")
            questions.append(Question(question_id=question_id, category=category, turns=turns, raw=item))

    selected = questions[offset:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def build_prompt(history: list[tuple[str, str]], prompt_style: str) -> str:
    if prompt_style == "plain":
        lines: list[str] = []
        for role, message in history:
            lines.append(f"{role.title()}: {message}")
        if history and history[-1][0].lower() == "user":
            lines.append("Assistant:")
        return "\n".join(lines)

    prompt = VICUNA_SYSTEM
    for role, message in history:
        normalized_role = role.upper()
        if normalized_role == "USER":
            prompt += f" USER: {message}"
        elif normalized_role == "ASSISTANT":
            prompt += f" ASSISTANT: {message}</s>"
        else:
            raise ValueError(f"Unsupported role: {role}")
    if history and history[-1][0].lower() == "user":
        prompt += " ASSISTANT:"
    return prompt


def trim_answer(text: str, stop_strings: list[str]) -> str:
    trimmed = text
    for stop_string in stop_strings:
        index = trimmed.find(stop_string)
        if index >= 0:
            trimmed = trimmed[:index]
    return trimmed.strip()


def trace_to_json(trace_steps: list[MedusaStepTrace]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in trace_steps:
        result.append(
            {
                "step": step.step,
                "prefix_length": step.prefix_length,
                "remaining_new_tokens": step.remaining_new_tokens,
                "candidate_path_count": step.candidate_path_count,
                "target_next_token": step.target_next_token,
                "medusa_top_tokens": step.medusa_top_tokens,
                "selected_path_index": step.selected_path_index,
                "selected_path_tokens": step.selected_path_tokens,
                "target_predictions": step.target_predictions,
                "target_top_logits": [
                    {"token_ids": item.token_ids, "values": item.values} for item in step.target_top_logits
                ],
                "accepted_count": step.accepted_count,
                "rejected_at": step.rejected_at,
                "appended_tokens": step.appended_tokens,
                "output_length": step.output_length,
                "stop_reason": step.stop_reason,
                "verification_method": step.verification_method,
                "acceptance_mode": step.acceptance_mode,
                "tree_node_count": step.tree_node_count,
                "cache_updated": step.cache_updated,
                "fallback_reason": step.fallback_reason,
            }
        )
    return result


def summarize_trace(trace_steps: list[MedusaStepTrace], generated_token_count: int, elapsed_seconds: float) -> dict[str, Any]:
    medusa_steps = len(trace_steps)
    accepted_total = sum(step.accepted_count for step in trace_steps)
    appended_total = sum(len(step.appended_tokens) for step in trace_steps)
    return {
        "generated_token_count": generated_token_count,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": generated_token_count / elapsed_seconds if elapsed_seconds > 0 else None,
        "medusa_steps": medusa_steps,
        "accepted_token_count": accepted_total,
        "appended_token_count": appended_total,
        "accepted_tokens_per_step": accepted_total / medusa_steps if medusa_steps else 0.0,
        "appended_tokens_per_step": appended_total / medusa_steps if medusa_steps else 0.0,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def truncate_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer_name = args.tokenizer_model or args.base_model
    question_path = Path(args.question_file).expanduser()
    answer_path = Path(args.answers_jsonl).expanduser()
    trace_path = Path(args.trace_jsonl).expanduser() if args.trace_jsonl else default_trace_path(answer_path)
    stop_strings = list(args.stop_string) if args.stop_string is not None else list(DEFAULT_STOP_STRINGS)

    print("Medusa MT-Bench-style generation")
    print(f"question file: {question_path}")
    print(f"answers jsonl: {answer_path}")
    print(f"trace jsonl: {trace_path}")
    print(f"base model: {args.base_model}")
    print(f"medusa heads: {args.medusa_heads}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"max_new_tokens per turn: {args.max_new_tokens}")
    print(f"choice preset: {args.choice_preset}, top_k={args.top_k}")
    print(f"verifier: {args.verifier}, acceptance={args.acceptance}, use_kv_cache={args.use_kv_cache}")
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
        target = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    print(f"Moving base model to {device} ...", flush=True)
    with timed_operation("base model to device", device, args.heartbeat_seconds):
        target = target.to(device).eval()
    target.config.output_hidden_states = True
    print(f"Base model ready on {next(target.parameters()).device}.")
    print(f"Memory after base load: {memory_status(device)}")

    print(f"\nLoading Medusa heads: {args.medusa_heads}", flush=True)
    head_dtype = None if args.dtype in {"auto", "none"} else dtype_from_arg(args.dtype)
    with timed_operation("medusa heads load", device, args.heartbeat_seconds):
        medusa_heads = load_official_medusa_heads(
            target,
            args.medusa_heads,
            device=device,
            dtype=head_dtype if isinstance(head_dtype, torch.dtype) else None,
            medusa_num_heads=args.medusa_num_heads,
            medusa_num_layers=args.medusa_num_layers,
        )
    print(f"Medusa heads ready on {next(medusa_heads.parameters()).device}.")
    print(f"Medusa heads: num_heads={medusa_heads.num_heads}")
    print(f"Memory after heads load: {memory_status(device)}")

    choices = medusa_choices_for(args, medusa_heads.num_heads)
    print(f"Medusa choice paths: {choices}")

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

            def on_step(step_trace: MedusaStepTrace, generated_ids: torch.Tensor) -> None:
                if not args.step_text:
                    return
                new_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
                print(
                    f"[question {question.question_id} turn {turn_index} step {step_trace.step}] "
                    f"accepted={step_trace.accepted_count} appended={step_trace.appended_tokens} "
                    f"partial={trim_answer(new_text, stop_strings)!r}",
                    flush=True,
                )

            started = time.perf_counter()
            with torch.inference_mode():
                output_ids, trace_steps = generate_with_trace(
                    target,
                    medusa_heads,
                    inputs.input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    medusa_choices=choices,
                    top_k=args.top_k,
                    top_k_logits=args.top_k_logits,
                    progress=args.progress,
                    heartbeat_seconds=args.heartbeat_seconds,
                    step_callback=on_step,
                    verifier=args.verifier,
                    acceptance_mode=args.acceptance,
                    temperature=args.temperature,
                    posterior_threshold=args.posterior_threshold,
                    posterior_alpha=args.posterior_alpha,
                    top_p=args.top_p,
                    use_kv_cache=args.use_kv_cache,
                    fallback_to_slow=args.fallback_to_slow,
                )
            elapsed = time.perf_counter() - started
            generated_ids = output_ids[0, prompt_length:]
            raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = trim_answer(raw_answer, stop_strings)
            answers.append(answer)
            history.append(("assistant", answer))

            stats = summarize_trace(trace_steps, int(generated_ids.shape[-1]), elapsed)
            stats.update(
                {
                    "turn_index": turn_index,
                    "prompt_token_count": prompt_length,
                    "raw_answer": raw_answer,
                }
            )
            turn_stats.append(stats)
            print(
                f"[question {question.question_id} turn {turn_index}] "
                f"done tokens={stats['generated_token_count']} "
                f"steps={stats['medusa_steps']} "
                f"accepted/step={stats['accepted_tokens_per_step']:.2f} "
                f"tok/s={stats['tokens_per_second']:.3f}",
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
                    "trace": trace_to_json(trace_steps),
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
    print("\nMT-Bench-style generation complete.")
    print(f"questions: {len(questions)}")
    print(f"elapsed: {suite_elapsed:.2f}s")
    print(f"answers: {answer_path}")
    print(f"traces: {trace_path}")


if __name__ == "__main__":
    main()
