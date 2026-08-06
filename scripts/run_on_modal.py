#!/usr/bin/env python
"""Run the acceptance/run-count benchmark suite on a Modal GPU.

The GPU runs produce the *numerator* of the speedup equation (acceptance and
component run counts, which are device-independent in greedy mode). Cycle
numbers still come from Voyager locally; GPU wall-clock is not a Sphinx number.

One-time setup:
    pip install modal
    modal setup                                    # browser login
    modal secret create huggingface HF_TOKEN=hf_...   # gated Vicuna/Llama weights

Usage:
    modal run scripts/run_on_modal.py --stage smoke
        1 question x 32 tokens for every method; verifies the whole pipeline
        end-to-end (~15 min on top of the first-time model download).

    modal run scripts/run_on_modal.py --stage full
        Real MT-Bench (80 questions, auto-downloaded), 512 tokens/turn:
        Medusa at tree sizes {64,32,16,8,4}, greedy baseline, and draft-model
        assisted decoding (llama-160m, k=5 constant). Roughly 6-10 A10G-hours
        sequentially; pass --parallel to run every config on its own GPU
        (same total cost, much faster wall-clock).

    SUITE_GPU=L4 modal run scripts/run_on_modal.py --stage full
        Override the GPU type (default A10G, 24 GB; T4's 16 GB is too small
        for Vicuna-7B fp16 + Medusa heads).

Results land in the `specdecoding-results` volume, mirrored per stage.
Download them with:
    modal volume get specdecoding-results / run_logs/modal/
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_KIND = os.environ.get("SUITE_GPU", "A10G")
RESULTS_DIR = "/results"
MINI_QUESTIONS = "examples/mini_mtbench_questions.jsonl"
FULL_QUESTIONS = "/tmp/mt_bench/question.jsonl"
QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

app = modal.App("specdecoding-suite")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.*",
        "transformers==5.6.2",  # matches the locally verified cache-surgery layout
        "accelerate",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/repo",
        ignore=["**/.git", "run_logs", "voyager_out", "archive", "data", "**/__pycache__", "**/*.pyc"],
    )
)

hf_cache = modal.Volume.from_name("specdecoding-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("specdecoding-results", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_KIND,
    timeout=8 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": hf_cache, RESULTS_DIR: results_volume},
)
def run_config(name: str, commands: list[list[str]]) -> str:
    import subprocess
    import urllib.request

    env = dict(os.environ, HF_HOME="/cache/huggingface", PYTHONUNBUFFERED="1")

    question_path = Path(FULL_QUESTIONS)
    if not question_path.exists():
        question_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] downloading MT-Bench questions ...", flush=True)
        urllib.request.urlretrieve(QUESTION_URL, question_path)

    for argv in commands:
        print(f"[{name}] running: {' '.join(argv)}", flush=True)
        subprocess.run(argv, cwd="/repo", env=env, check=True)

    results_volume.commit()
    return name


def medusa_config(
    tree_size: int,
    question_file: str,
    limit: int,
    max_new_tokens: int,
    out_dir: str,
    acceptance: str = "greedy",
    temperature: float = 0.0,
) -> tuple[str, list[list[str]]]:
    # Typical acceptance is NOT lossless and NOT comparable to the greedy
    # baseline; keep its outputs under a distinct name so greedy results
    # are never overwritten (CLAUDE.md gotcha 5.2).
    tag = f"medusa_tree{tree_size}" if acceptance == "greedy" else f"medusa_tree{tree_size}_{acceptance}_t{temperature:g}"
    answers = f"{out_dir}/{tag}_answers.jsonl"
    traces = f"{out_dir}/{tag}_answers.traces.jsonl"
    run_cmd = [
        "python", "tools/run_medusa_mtbench.py",
        "--question-file", question_file,
        "--answers-jsonl", answers,
        "--trace-jsonl", traces,
        "--model-id", f"first-principles-medusa-vicuna-7b-v1.3-{tag}",
        "--tree-size", str(tree_size),
        "--limit", str(limit),
        "--max-new-tokens", str(max_new_tokens),
        "--device", "cuda",
        "--dtype", "float16",
        "--attn-implementation", "eager",  # tree mask must be honored verbatim
        "--no-step-text",
        "--progress",
    ]
    if acceptance != "greedy":
        run_cmd += ["--acceptance", acceptance, "--temperature", str(temperature)]
    return (
        tag,
        [
            run_cmd,
            [
                "python", "tools/export_component_log.py", traces,
                "--method", f"medusa-vicuna-7b-{tag}",
                "--out", f"{out_dir}/{tag}",
            ],
        ],
    )


def baseline_config(
    question_file: str, limit: int, max_new_tokens: int, out_dir: str
) -> tuple[str, list[list[str]]]:
    answers = f"{out_dir}/vicuna_baseline_answers.jsonl"
    traces = f"{out_dir}/vicuna_baseline_answers.traces.jsonl"
    return (
        "baseline",
        [
            [
                "python", "tools/run_vicuna_mtbench_baseline.py",
                "--question-file", question_file,
                "--answers-jsonl", answers,
                "--trace-jsonl", traces,
                "--model-id", "vicuna-7b-v1.3-greedy-baseline",
                "--limit", str(limit),
                "--max-new-tokens", str(max_new_tokens),
                "--device", "cuda",
                "--dtype", "float16",
                "--no-step-text",
                "--progress",
            ],
            [
                "python", "tools/export_component_log.py", traces,
                "--method", "vicuna-7b-baseline",
                "--out", f"{out_dir}/baseline",
            ],
        ],
    )


def assisted_config(
    question_file: str, limit: int, max_new_tokens: int, out_dir: str, k: int = 5
) -> tuple[str, list[list[str]]]:
    # The assisted runner writes its component/step CSVs itself.
    return (
        f"assisted_k{k}",
        [
            [
                "python", "tools/run_assisted_mtbench.py",
                "--question-file", question_file,
                "--answers-jsonl", f"{out_dir}/assisted_k{k}_answers.jsonl",
                "--model-id", f"stripped-assisted-vicuna-7b-llama-160m-k{k}",
                "--num-assistant-tokens", str(k),
                "--assistant-schedule", "constant",
                "--limit", str(limit),
                "--max-new-tokens", str(max_new_tokens),
                "--device", "cuda",
                "--dtype", "float16",
                "--no-step-text",
            ],
        ],
    )


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    tree_sizes: str = "64,32,16,8,4",
    limit: int = 0,
    max_new_tokens: int = 512,
    parallel: bool = False,
    typical_temperature: float = 0.7,
):
    """Suite = Medusa (greedy) at each tree size, greedy baseline, assisted k=5,
    plus Medusa with typical acceptance at each tree size when
    typical_temperature > 0 (default 0.7; pass --typical-temperature 0 to skip).
    Typical runs are lossy and land in separate *_typical_t* files; greedy runs
    are the lossless, baseline-comparable ones."""
    if typical_temperature < 0:
        raise ValueError("--typical-temperature must be >= 0 (0 disables typical runs).")

    if stage == "smoke":
        question_file, limit, max_new_tokens = MINI_QUESTIONS, 1, 32
        out_dir = f"{RESULTS_DIR}/smoke"
        sizes = [64]
    elif stage == "full":
        question_file = FULL_QUESTIONS
        out_dir = f"{RESULTS_DIR}/gpu_suite_full"
        sizes = [int(part) for part in tree_sizes.split(",") if part.strip()]
    else:
        raise ValueError("stage must be 'smoke' or 'full'")

    configs = [
        medusa_config(size, question_file, limit, max_new_tokens, out_dir)
        for size in sizes
    ]
    if typical_temperature > 0:
        configs += [
            medusa_config(size, question_file, limit, max_new_tokens, out_dir, "typical", typical_temperature)
            for size in sizes
        ]
    configs.append(baseline_config(question_file, limit, max_new_tokens, out_dir))
    configs.append(assisted_config(question_file, limit, max_new_tokens, out_dir))

    print(f"stage={stage} gpu={GPU_KIND} configs={[name for name, _ in configs]}")
    if parallel:
        for result in run_config.starmap(configs):
            print(f"done: {result}")
    else:
        for name, commands in configs:
            print(f"done: {run_config.remote(name, commands)}")

    print(f"\nAll configs finished. Download results with:")
    print("  modal volume get specdecoding-results / run_logs/modal/")
