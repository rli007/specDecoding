#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-run_logs/gpu_suite}"
QUESTION_FILE="${QUESTION_FILE:-examples/mini_mtbench_questions.jsonl}"
ANSWERS_JSONL="${ANSWERS_JSONL:-$OUT_DIR/medusa_mtbench_answers.jsonl}"
TRACE_JSONL="${TRACE_JSONL:-$OUT_DIR/medusa_mtbench_answers.traces.jsonl}"

BASE_MODEL="${BASE_MODEL:-lmsys/vicuna-7b-v1.3}"
MEDUSA_HEADS="${MEDUSA_HEADS:-FasterDecoding/medusa-vicuna-7b-v1.3}"
MODEL_ID="${MODEL_ID:-first-principles-medusa-vicuna-7b-v1.3-tree}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
LIMIT="${LIMIT:-1}"
OFFSET="${OFFSET:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TOP_K="${TOP_K:-10}"
CHOICE_PRESET="${CHOICE_PRESET:-official-vicuna-7b}"
TREE_SIZE="${TREE_SIZE:-}"
ACCEPTANCE="${ACCEPTANCE:-greedy}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-10}"
VERBOSE_TIMING="${VERBOSE_TIMING:-1}"
STEP_TEXT="${STEP_TEXT:-1}"

mkdir -p "$OUT_DIR"

args=(
  tools/run_medusa_mtbench.py
  --question-file "$QUESTION_FILE"
  --answers-jsonl "$ANSWERS_JSONL"
  --trace-jsonl "$TRACE_JSONL"
  --model-id "$MODEL_ID"
  --base-model "$BASE_MODEL"
  --medusa-heads "$MEDUSA_HEADS"
  --limit "$LIMIT"
  --offset "$OFFSET"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --choice-preset "$CHOICE_PRESET"
  --top-k "$TOP_K"
  --verifier tree
  --acceptance "$ACCEPTANCE"
  --heartbeat-seconds "$HEARTBEAT_SECONDS"
  --progress
)

if [[ -n "$TREE_SIZE" ]]; then
  args+=(--tree-size "$TREE_SIZE")
fi
if [[ "$VERBOSE_TIMING" == "1" ]]; then
  args+=(--verbose-timing)
fi
if [[ "$STEP_TEXT" != "1" ]]; then
  args+=(--no-step-text)
fi

python "${args[@]}"
