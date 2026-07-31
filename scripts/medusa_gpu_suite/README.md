# Medusa GPU Suite

Small wrapper folder for running the local Medusa implementation on a GPU box,
then judging the generated answers with GPT-4o through OpenRouter.

## Quickstart (the one command that matters)

For the full acceptance-measurement run — real MT-Bench, 512 tokens/turn,
Medusa at tree sizes {64, 32, 16, 8, 4}, then the greedy baseline:

```bash
git clone https://github.com/rli007/specDecoding.git && cd specDecoding
python -m pip install -r requirements.txt
huggingface-cli login   # lmsys/vicuna-7b weights are gated
bash scripts/medusa_gpu_suite/04_run_full_mtbench.sh
```

Requirements: one CUDA GPU with >= 24 GB VRAM (the model runs fp16: ~13.5 GB
base weights + ~1.5 GB Medusa heads + KV/activations), ~20 GB of disk for the
HuggingFace cache, no API keys. Expect several hours total (6 model runs x
80 questions x 2 turns x 512 tokens); each run appends results incrementally,
so an interrupted run keeps everything finished so far.

When done, send back the whole `run_logs/gpu_suite_full/` folder (it is
gitignored, so zip/scp it — it will be tens of MB of JSONL).

Setup for the individual wrappers below is the same:

```bash
python -m pip install -r requirements.txt
huggingface-cli login
export OPENROUTER_API_KEY="..."   # only needed for step 3 (judging)
```

## 0. Download the real MT-Bench questions

```bash
bash scripts/medusa_gpu_suite/00_download_mtbench_questions.sh
```

Fetches FastChat's official 80-question `question.jsonl` into
`data/mt_bench/question.jsonl`. The bundled
`examples/mini_mtbench_questions.jsonl` is a 1-question smoke file — fine for
checking the plumbing, useless for acceptance statistics.

## 1. Medusa Tree Run

```bash
bash scripts/medusa_gpu_suite/01_run_medusa_tree.sh
```

Defaults:

- base model: `lmsys/vicuna-7b-v1.3`
- Medusa heads: `FasterDecoding/medusa-vicuna-7b-v1.3`
- device: `cuda`
- dtype: `float16`
- verifier: `tree`
- choice preset: `official-vicuna-7b`
- max new tokens: `32`
- output: `run_logs/gpu_suite/medusa_mtbench_answers.jsonl`
- traces: `run_logs/gpu_suite/medusa_mtbench_answers.traces.jsonl`

The run enables `--progress`, and by default `--verbose-timing`, so traces
include timing for prefill, candidate generation, tree target forward, Medusa
heads, posterior acceptance, cache copy, and step totals. Set
`VERBOSE_TIMING=0` to skip the per-stage device syncs (recommended when the
wall-clock tok/s number matters) and `STEP_TEXT=0` to silence per-step partial
text. `TREE_SIZE=16` truncates the choice tree to 16 nodes (free root
included) using the preset's stored greedy order, which stays prefix-closed
at every cut.

## 2. Plain Vicuna Baseline

```bash
bash scripts/medusa_gpu_suite/02_run_vicuna_baseline.sh
```

This runs normal greedy cached generation with the same question file and token
budget. Use this to compare Medusa speed against ordinary decoding.

## 3. GPT-4o Judge

```bash
bash scripts/medusa_gpu_suite/03_judge_medusa_openrouter.sh
```

This reads the Medusa answer JSONL and writes:

```text
run_logs/gpu_suite/medusa_mtbench_answers.judgments.jsonl
```

It uses OpenRouter's OpenAI-compatible chat completion endpoint with:

```text
JUDGE_MODEL=openai/gpt-4o
```

## Common Overrides

All wrappers are controlled by environment variables:

```bash
LIMIT=5 MAX_NEW_TOKENS=64 bash scripts/medusa_gpu_suite/01_run_medusa_tree.sh
LIMIT=5 MAX_NEW_TOKENS=64 bash scripts/medusa_gpu_suite/02_run_vicuna_baseline.sh
ANSWERS_JSONL=run_logs/gpu_suite/vicuna_baseline_answers.jsonl \
  JUDGMENTS_JSONL=run_logs/gpu_suite/vicuna_baseline_answers.judgments.jsonl \
  bash scripts/medusa_gpu_suite/03_judge_medusa_openrouter.sh
```

Useful knobs:

- `DEVICE=cuda`, `mps`, or `cpu`
- `DTYPE=float16`, `bfloat16`, `float32`, or `auto`
- `ATTN_IMPLEMENTATION=eager` by default; keep this first for tree-mask safety
- `QUESTION_FILE=...` for official MT-Bench questions
- `OUT_DIR=...` to group runs
- `BASE_MODEL=...`
- `MEDUSA_HEADS=...`
- `CHOICE_PRESET=official-vicuna-7b`, `official-vicuna-13b`, `official-zephyr`, `small-tree`, or `linear`
- `TREE_SIZE=N` truncates the preset to N total nodes (Medusa wrapper only)
- `ACCEPTANCE=greedy`, `typical`, or `nucleus` — keep `greedy` for any speedup
  claim; it is the only lossless mode
- `VERBOSE_TIMING=0`, `STEP_TEXT=0` to reduce overhead on long runs

## 4. Full acceptance suite (what to actually run)

```bash
bash scripts/medusa_gpu_suite/04_run_full_mtbench.sh
```

Downloads the questions if missing, runs Medusa at each `TREE_SIZES` entry
(default `64 32 16 8 4`), then the greedy baseline, all at
`MAX_NEW_TOKENS=512` over all 80 questions, into `run_logs/gpu_suite_full/`.
Override the sweep with e.g. `TREE_SIZES="64 8"` or shorten with
`MAX_NEW_TOKENS=256` if time is tight. Per-step `accept_length` and
`tokens_per_step` land in the trace JSONL; the per-turn summary prints
`accept_length/step` (the paper's tau).

For a no-network judge smoke test:

```bash
python tools/judge_openrouter_mtbench.py \
  --answers-jsonl run_logs/gpu_suite/medusa_mtbench_answers.jsonl \
  --dry-run \
  --limit 1
```
