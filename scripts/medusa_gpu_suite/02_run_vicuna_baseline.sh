#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-run_logs/gpu_suite}"
QUESTION_FILE="${QUESTION_FILE:-examples/mini_mtbench_questions.jsonl}"
ANSWERS_JSONL="${ANSWERS_JSONL:-$OUT_DIR/vicuna_baseline_answers.jsonl}"
TRACE_JSONL="${TRACE_JSONL:-$OUT_DIR/vicuna_baseline_answers.traces.jsonl}"

BASE_MODEL="${BASE_MODEL:-lmsys/vicuna-7b-v1.3}"
MODEL_ID="${MODEL_ID:-vicuna-7b-v1.3-greedy-baseline}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
LIMIT="${LIMIT:-1}"
OFFSET="${OFFSET:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-10}"

mkdir -p "$OUT_DIR"

args=(
  tools/run_vicuna_mtbench_baseline.py
  --question-file "$QUESTION_FILE"
  --answers-jsonl "$ANSWERS_JSONL"
  --trace-jsonl "$TRACE_JSONL"
  --model-id "$MODEL_ID"
  --base-model "$BASE_MODEL"
  --limit "$LIMIT"
  --offset "$OFFSET"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --heartbeat-seconds "$HEARTBEAT_SECONDS"
  --progress
)

python "${args[@]}"
