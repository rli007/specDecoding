#!/usr/bin/env python
"""Compare HF assisted generation against the stripped-down decoder step by step."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator
import transformers.generation.utils as generation_utils


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoders.stripped_down_llama_assisted_decoder import (  # noqa: E402
    AssistedStepTrace,
    DEFAULT_ASSISTANT_MODEL,
    DEFAULT_NUM_ASSISTANT_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TARGET_MODEL,
    SamplingDecision,
    TokenProbability,
    assisted_generate,
    choose_device,
    dtype_from_arg,
    summarize_top_probs,
)


# Pick the normal test prompt here. You can add more named prompts below.
DEFAULT_TEST_PROMPT_NAME = "story"
DEFAULT_TEST_MAX_NEW_TOKENS = 16
DEFAULT_TEST_NUM_ASSISTANT_TOKENS = 2

TEST_PROMPTS = {
    "stanford": DEFAULT_PROMPT,
    "short": "The quick brown fox",
    "story": "In a small workshop near the ocean, the engineer",
    "code": "def fibonacci(n):",
}


@dataclass
class HfStepTrace:
    step: int
    mode: str
    prefix_length: int
    assistant_budget: float
    draft_tokens: list[int]
    target_selected_tokens: list[int] = field(default_factory=list)
    accepted_assistant_tokens: int | None = None
    appended_tokens: list[int] = field(default_factory=list)
    next_assistant_budget: float | None = None
    output_length: int | None = None
    assistant_top_probs: list[list[TokenProbability]] = field(default_factory=list)
    target_top_probs: list[list[TokenProbability]] = field(default_factory=list)
    sampling_decisions: list[SamplingDecision] = field(default_factory=list)


class TeeStream:
    """Write output to both the original stream and the run log."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def log_path_for_run(log_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"compare_hf_vs_stripped_assisted_steps_{timestamp}.txt"


def current_logging_handlers() -> list[logging.Handler]:
    handlers = list(logging.getLogger().handlers)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            handlers.extend(logger.handlers)
    return handlers


@contextlib.contextmanager
def tee_run_output(args: argparse.Namespace, prompt: str, device: torch.device):
    if args.no_log:
        yield None
        return

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for_run(log_dir)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = TeeStream(original_stdout, log_file)
        stderr_tee = TeeStream(original_stderr, log_file)
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        handler_streams: list[tuple[logging.StreamHandler, Any]] = []
        for handler in current_logging_handlers():
            if isinstance(handler, logging.StreamHandler) and handler.stream in {original_stdout, original_stderr}:
                handler_streams.append((handler, handler.stream))
                handler.setStream(stderr_tee if handler.stream is original_stderr else stdout_tee)
        try:
            print("RUN LOG")
            print(f"started_at: {datetime.now().isoformat(timespec='seconds')}")
            print(f"log_path: {log_path}")
            print(f"command: {' '.join(sys.argv)}")
            print(f"target_model: {args.target_model}")
            print(f"assistant_model: {args.assistant_model}")
            print(f"tokenizer_model: {args.tokenizer_model or args.target_model}")
            print(f"prompt_name: {args.prompt_name}")
            print(f"prompt: {prompt!r}")
            print(f"mode: {args.mode}")
            print(f"max_new_tokens: {args.max_new_tokens}")
            print(f"num_assistant_tokens: {args.num_assistant_tokens}")
            print(f"top_k_probs: {args.top_k_probs}")
            print(f"seed: {args.seed}")
            print(f"device_arg: {args.device}")
            print(f"selected_device: {device}")
            print(f"dtype: {args.dtype}")
            print(f"attn_implementation: {args.attn_implementation}")
            print(f"low_cpu_mem_usage: {args.low_cpu_mem_usage}")
            print()
            yield log_path
        finally:
            for handler, stream in handler_streams:
                handler.setStream(stream)
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@contextlib.contextmanager
def capture_hf_assisted_steps(
    records: list[HfStepTrace],
    mode: str,
    top_k_probs: int,
    on_step: Callable[[HfStepTrace], None] | None = None,
):
    """Patch the public candidate-generator hooks that expose each HF step."""
    originals: list[tuple[Any, str, Callable[..., Any]]] = []
    active_records: dict[int, HfStepTrace] = {}

    def replace(obj: Any, name: str, wrapper: Callable[..., Any]) -> None:
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, wrapper)

    original_get_candidates = AssistedCandidateGenerator.get_candidates
    original_update_candidate_strategy = AssistedCandidateGenerator.update_candidate_strategy
    original_speculative_sampling = generation_utils._speculative_sampling

    def traced_get_candidates(self: AssistedCandidateGenerator, input_ids: torch.LongTensor):
        candidate_input_ids, candidate_logits = original_get_candidates(self, input_ids)
        prefix_length = input_ids.shape[-1]
        draft_tokens = candidate_input_ids[:, prefix_length:].detach().cpu()[0].tolist()
        record = HfStepTrace(
            step=len(records) + 1,
            mode=mode,
            prefix_length=prefix_length,
            assistant_budget=float(len(draft_tokens)),
            draft_tokens=draft_tokens,
            assistant_top_probs=summarize_top_probs(candidate_logits, top_k_probs),
        )
        records.append(record)
        active_records[id(self)] = record
        return candidate_input_ids, candidate_logits

    def traced_speculative_sampling(
        candidate_input_ids,
        candidate_logits,
        candidate_length,
        new_logits,
        is_done_candidate,
    ):
        record = records[-1] if records else None
        new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
        q = candidate_logits.softmax(dim=-1)
        p = new_logits.softmax(dim=-1)
        positions = torch.arange(candidate_length, device=new_candidate_input_ids.device)
        q_i = q[0, positions, new_candidate_input_ids[0]]
        p_i = p[0, positions, new_candidate_input_ids[0]]
        probability_ratio = p_i / q_i

        random_values = torch.rand_like(probability_ratio)
        accepted_mask = random_values <= probability_ratio
        n_matches = int(((~accepted_mask).cumsum(dim=-1) < 1).sum().item())

        if record is not None:
            record.sampling_decisions = [
                SamplingDecision(
                    index=index,
                    candidate_token=int(new_candidate_input_ids[0, index].item()),
                    assistant_probability=float(q_i[index].item()),
                    target_probability=float(p_i[index].item()),
                    acceptance_ratio=float(probability_ratio[index].item()),
                    random_value=float(random_values[index].item()),
                    accepted=bool(accepted_mask[index].item()),
                )
                for index in range(candidate_length)
            ]

        if is_done_candidate and n_matches == candidate_length:
            n_matches -= 1
            valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
        else:
            gamma = candidate_logits.shape[1]
            p_n_plus_1 = p[:, n_matches, :]
            if n_matches < gamma:
                q_n_plus_1 = q[:, n_matches, :]
                p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
                p_prime.div_(p_prime.sum())
            else:
                p_prime = p_n_plus_1
            sampled_token = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

            if n_matches > 0:
                valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], sampled_token), dim=-1)
            else:
                valid_tokens = sampled_token

        return valid_tokens, n_matches

    def traced_update_candidate_strategy(
        self: AssistedCandidateGenerator,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        num_matches: int,
    ):
        record = active_records.get(id(self), records[-1] if records else None)
        if record is not None:
            record.accepted_assistant_tokens = int(num_matches)
            record.output_length = input_ids.shape[-1]
            record.appended_tokens = input_ids[:, record.prefix_length :].detach().cpu()[0].tolist()
            if scores is not None:
                record.target_selected_tokens = scores.argmax(dim=-1).detach().cpu()[0].tolist()
                record.target_top_probs = summarize_top_probs(scores, top_k_probs)

        result = original_update_candidate_strategy(self, input_ids, scores, num_matches)

        if record is not None:
            record.next_assistant_budget = float(getattr(self, "num_assistant_tokens", 0))
            if on_step is not None:
                on_step(record)
        return result

    replace(AssistedCandidateGenerator, "get_candidates", traced_get_candidates)
    replace(AssistedCandidateGenerator, "update_candidate_strategy", traced_update_candidate_strategy)
    replace(generation_utils, "_speculative_sampling", traced_speculative_sampling)

    try:
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HF assisted decoding against the stripped decoder.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--tokenizer-model", default=None, help="Tokenizer to use. Defaults to --target-model.")
    parser.add_argument("--prompt-name", default=DEFAULT_TEST_PROMPT_NAME, choices=sorted(TEST_PROMPTS))
    parser.add_argument("--prompt", default=None, help="Literal prompt override. If set, ignores --prompt-name.")
    parser.add_argument("--mode", choices=("greedy", "sampling"), default="greedy")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_TEST_MAX_NEW_TOKENS)
    parser.add_argument("--num-assistant-tokens", type=int, default=DEFAULT_TEST_NUM_ASSISTANT_TOKENS)
    parser.add_argument("--top-k-probs", type=int, default=5, help="Top token probabilities to print per decision.")
    parser.add_argument("--seed", type=int, default=1234, help="Torch seed used before each HF/custom run.")
    parser.add_argument("--no-live-steps", action="store_true", help="Only print detailed step output at the end.")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32", "none"), default="float16")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, for example sdpa.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--log-dir", default=str(ROOT / "run_logs"), help="Directory for timestamped text logs.")
    parser.add_argument("--no-log", action="store_true", help="Disable tee logging for this run.")
    return parser.parse_args()


def make_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    dtype = dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True
    return model_kwargs


def configure_sampling_config(model: torch.nn.Module, mode: str) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = mode == "sampling"
    if mode == "sampling":
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 0
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None


def configure_assistant_for_comparison(assistant: torch.nn.Module, num_assistant_tokens: int, mode: str) -> None:
    configure_sampling_config(assistant, mode)
    assistant.generation_config.num_assistant_tokens = num_assistant_tokens
    assistant.generation_config.num_assistant_tokens_schedule = "heuristic"
    assistant.generation_config.assistant_confidence_threshold = 0.0


def model_vocab_size(model: torch.nn.Module) -> int | None:
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return int(output_embeddings.weight.shape[0])
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    return int(vocab_size) if vocab_size is not None else None


def validate_same_vocab(target: torch.nn.Module, assistant: torch.nn.Module) -> None:
    target_vocab_size = model_vocab_size(target)
    assistant_vocab_size = model_vocab_size(assistant)
    if target_vocab_size is None or assistant_vocab_size is None:
        print("vocab size check: skipped")
        return
    print(f"vocab size check: target={target_vocab_size}, assistant={assistant_vocab_size}")
    if target_vocab_size != assistant_vocab_size:
        raise ValueError(
            "Target and assistant vocab sizes differ "
            f"({target_vocab_size} vs {assistant_vocab_size}). "
            "This comparison harness assumes a shared tokenizer/vocab."
        )


def decode_token_list(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def fmt_budget(value: float | None) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def format_top_probs(tokenizer: AutoTokenizer, top_probs: list[TokenProbability]) -> str:
    pieces = []
    for item in top_probs:
        pieces.append(f"{item.token_id}:{item.probability:.4f}:{tokenizer.decode([item.token_id])!r}")
    return "[" + ", ".join(pieces) + "]"


def print_top_prob_block(tokenizer: AutoTokenizer, label: str, top_probs: list[list[TokenProbability]]) -> None:
    if not top_probs:
        return
    for index, position_probs in enumerate(top_probs):
        print(f"      {label}[{index}] top probs: {format_top_probs(tokenizer, position_probs)}")


def print_sampling_decisions(tokenizer: AutoTokenizer, decisions: list[SamplingDecision]) -> None:
    if not decisions:
        return
    print("      sampling acceptance:")
    for decision in decisions:
        status = "accept" if decision.accepted else "reject"
        print(
            f"        draft[{decision.index}] token={decision.candidate_token} "
            f"{tokenizer.decode([decision.candidate_token])!r} "
            f"q={decision.assistant_probability:.6f} "
            f"p={decision.target_probability:.6f} "
            f"p/q={decision.acceptance_ratio:.6f} "
            f"r={decision.random_value:.6f} -> {status}"
        )


def print_single_step(tokenizer: AutoTokenizer, source: str, step: HfStepTrace | AssistedStepTrace) -> None:
    print(
        f"\n{source} STEP {step.step} "
        f"mode={step.mode} "
        f"prefix={step.prefix_length} "
        f"budget={fmt_budget(step.assistant_budget)} -> {fmt_budget(step.next_assistant_budget)} "
        f"accepted={step.accepted_assistant_tokens} "
        f"output_len={step.output_length}"
    )
    print(f"      draft:   {step.draft_tokens} {decode_token_list(tokenizer, step.draft_tokens)!r}")
    print(f"      target:  {step.target_selected_tokens} {decode_token_list(tokenizer, step.target_selected_tokens)!r}")
    print(f"      append:  {step.appended_tokens} {decode_token_list(tokenizer, step.appended_tokens)!r}")
    print_top_prob_block(tokenizer, "assistant", step.assistant_top_probs)
    print_top_prob_block(tokenizer, "target", step.target_top_probs)
    if step.mode == "sampling":
        print_sampling_decisions(tokenizer, step.sampling_decisions)


def print_step_pair(
    tokenizer: AutoTokenizer,
    hf_step: HfStepTrace | None,
    our_step: AssistedStepTrace | None,
) -> None:
    step_number = hf_step.step if hf_step is not None else our_step.step
    print(f"\nSTEP {step_number}")

    if hf_step is None:
        print("HF:   <no step>")
    else:
        print(
            "HF:   "
            f"prefix={hf_step.prefix_length}, "
            f"budget={fmt_budget(hf_step.assistant_budget)} -> {fmt_budget(hf_step.next_assistant_budget)}, "
            f"accepted={hf_step.accepted_assistant_tokens}, "
            f"output_len={hf_step.output_length}"
        )
        print(f"      draft:   {hf_step.draft_tokens} {decode_token_list(tokenizer, hf_step.draft_tokens)!r}")
        print(
            f"      target:  {hf_step.target_selected_tokens} "
            f"{decode_token_list(tokenizer, hf_step.target_selected_tokens)!r}"
        )
        print(f"      append:  {hf_step.appended_tokens} {decode_token_list(tokenizer, hf_step.appended_tokens)!r}")
        print_top_prob_block(tokenizer, "HF assistant", hf_step.assistant_top_probs)
        print_top_prob_block(tokenizer, "HF target", hf_step.target_top_probs)
        if hf_step.mode == "sampling":
            print_sampling_decisions(tokenizer, hf_step.sampling_decisions)

    if our_step is None:
        print("OURS: <no step>")
    else:
        print(
            "OURS: "
            f"prefix={our_step.prefix_length}, "
            f"budget={fmt_budget(our_step.assistant_budget)} -> {fmt_budget(our_step.next_assistant_budget)}, "
            f"accepted={our_step.accepted_assistant_tokens}, "
            f"output_len={our_step.output_length}"
        )
        print(f"      draft:   {our_step.draft_tokens} {decode_token_list(tokenizer, our_step.draft_tokens)!r}")
        print(
            f"      target:  {our_step.target_selected_tokens} "
            f"{decode_token_list(tokenizer, our_step.target_selected_tokens)!r}"
        )
        print(f"      append:  {our_step.appended_tokens} {decode_token_list(tokenizer, our_step.appended_tokens)!r}")
        print_top_prob_block(tokenizer, "OURS assistant", our_step.assistant_top_probs)
        print_top_prob_block(tokenizer, "OURS target", our_step.target_top_probs)
        if our_step.mode == "sampling":
            print_sampling_decisions(tokenizer, our_step.sampling_decisions)

    if hf_step is not None and our_step is not None:
        print(
            "MATCH "
            f"draft={hf_step.draft_tokens == our_step.draft_tokens}, "
            f"target={hf_step.target_selected_tokens == our_step.target_selected_tokens}, "
            f"accepted={hf_step.accepted_assistant_tokens == our_step.accepted_assistant_tokens}, "
            f"append={hf_step.appended_tokens == our_step.appended_tokens}"
        )


def print_step_count_note(args: argparse.Namespace, hf_steps: list[HfStepTrace], our_steps: list[AssistedStepTrace]) -> None:
    step_count = max(len(hf_steps), len(our_steps))
    if step_count > 1:
        return

    max_one_step_append = args.num_assistant_tokens + 1
    print("\nSTEP COUNT NOTE")
    print(
        "Only one assisted step was captured. That can be completely normal: "
        f"one step can append up to num_assistant_tokens + 1 = {max_one_step_append} tokens, "
        f"and this run requested max_new_tokens={args.max_new_tokens}."
    )
    print("To force more steps, increase --max-new-tokens or lower --num-assistant-tokens.")


def run_comparison(args: argparse.Namespace, prompt: str, device: torch.device) -> None:
    tokenizer_name = args.tokenizer_model or args.target_model
    model_kwargs = make_model_kwargs(args)
    print(f"Using device={device}")
    print(f"target model: {args.target_model}")
    print(f"assistant model: {args.assistant_model}")
    print(f"prompt: {prompt!r}")
    print(f"mode: {args.mode}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"initial num_assistant_tokens: {args.num_assistant_tokens}")
    print(f"top_k_probs: {args.top_k_probs}")
    print(f"seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs).to(device).eval()
    assistant = AutoModelForCausalLM.from_pretrained(args.assistant_model, **model_kwargs).to(device).eval()
    configure_sampling_config(target, args.mode)
    configure_sampling_config(assistant, args.mode)
    validate_same_vocab(target, assistant)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"prompt token length: {inputs.input_ids.shape[-1]}")

    do_sample = args.mode == "sampling"
    live_steps = not args.no_live_steps

    configure_assistant_for_comparison(assistant, args.num_assistant_tokens, args.mode)
    hf_steps: list[HfStepTrace] = []
    if live_steps:
        print("\nHF LIVE STEPS")
    torch.manual_seed(args.seed)
    with capture_hf_assisted_steps(
        hf_steps,
        mode=args.mode,
        top_k_probs=args.top_k_probs,
        on_step=(lambda step: print_single_step(tokenizer, "HF", step)) if live_steps else None,
    ):
        with torch.inference_mode():
            hf_output = target.generate(
                **inputs,
                assistant_model=assistant,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

    configure_assistant_for_comparison(assistant, args.num_assistant_tokens, args.mode)
    our_steps: list[AssistedStepTrace] = []
    if live_steps:
        print("\nOURS LIVE STEPS")
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        our_output = assisted_generate(
            target,
            assistant,
            inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            num_assistant_tokens=args.num_assistant_tokens,
            mode=args.mode,
            top_k_probs=args.top_k_probs,
            verbose=False,
            trace_steps=our_steps,
            step_callback=(lambda step: print_single_step(tokenizer, "OURS", step)) if live_steps else None,
        )

    print("\nFINAL OUTPUTS")
    print(f"HF ids:   {hf_output[0].tolist()}")
    print(f"OURS ids: {our_output[0].tolist()}")
    print(f"equal:    {hf_output[0].tolist() == our_output[0].tolist()}")
    print(f"HF text:   {tokenizer.decode(hf_output[0], skip_special_tokens=True)!r}")
    print(f"OURS text: {tokenizer.decode(our_output[0], skip_special_tokens=True)!r}")

    print("\nSTEP COMPARISON" if not live_steps else "\nSTEP MATCH SUMMARY")
    print_step_count_note(args, hf_steps, our_steps)
    for index in range(max(len(hf_steps), len(our_steps))):
        hf_step = hf_steps[index] if index < len(hf_steps) else None
        our_step = our_steps[index] if index < len(our_steps) else None
        if live_steps:
            if hf_step is None or our_step is None:
                print_step_pair(tokenizer, hf_step, our_step)
            else:
                print(
                    f"STEP {index + 1}: "
                    f"draft={hf_step.draft_tokens == our_step.draft_tokens}, "
                    f"accepted={hf_step.accepted_assistant_tokens == our_step.accepted_assistant_tokens}, "
                    f"append={hf_step.appended_tokens == our_step.appended_tokens}"
                )
        else:
            print_step_pair(tokenizer, hf_step, our_step)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    prompt = args.prompt if args.prompt is not None else TEST_PROMPTS[args.prompt_name]

    with tee_run_output(args, prompt, device) as log_path:
        run_comparison(args, prompt, device)
        if log_path is not None:
            print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
