#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-run_logs/gpu_suite}"
QUESTION_FILE="${QUESTION_FILE:-examples/mini_mtbench_questions.jsonl}"
ANSWERS_JSONL="${ANSWERS_JSONL:-$OUT_DIR/medusa_mtbench_answers.jsonl}"
JUDGMENTS_JSONL="${JUDGMENTS_JSONL:-$OUT_DIR/medusa_mtbench_answers.judgments.jsonl}"

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-512}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.5}"

args=(
  tools/judge_openrouter_mtbench.py
  --question-file "$QUESTION_FILE"
  --answers-jsonl "$ANSWERS_JSONL"
  --judgments-jsonl "$JUDGMENTS_JSONL"
  --model "$JUDGE_MODEL"
  --limit "$LIMIT"
  --offset "$OFFSET"
  --max-judge-tokens "$MAX_JUDGE_TOKENS"
  --sleep-seconds "$SLEEP_SECONDS"
  --overwrite
)

python "${args[@]}"
