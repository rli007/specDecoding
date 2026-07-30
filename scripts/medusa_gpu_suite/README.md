# Medusa GPU Suite

Small wrapper folder for running the local Medusa implementation on a GPU box,
then judging the generated answers with GPT-4o through OpenRouter.

Run these commands from the repo root:

```bash
python -m pip install -r requirements.txt
huggingface-cli login
export OPENROUTER_API_KEY="..."
```

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

The run enables `--progress` and `--verbose-timing`, so traces include timing
for prefill, candidate generation, tree target forward, Medusa heads,
posterior acceptance, cache copy, and step totals.

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
- `ACCEPTANCE=greedy`, `typical`, or `nucleus`

For a no-network judge smoke test:

```bash
python tools/judge_openrouter_mtbench.py \
  --answers-jsonl run_logs/gpu_suite/medusa_mtbench_answers.jsonl \
  --dry-run \
  --limit 1
```
