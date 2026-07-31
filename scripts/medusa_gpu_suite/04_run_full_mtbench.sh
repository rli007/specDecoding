#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Full acceptance-measurement run: real MT-Bench (80 questions), long
# generations, one Medusa run per tree size, then the greedy baseline.
# Produces the tau(N) numerator for the speedup-vs-tree-size plot.
#
# Timing flags are off here: acceptance statistics are unaffected, and
# per-stage device syncs would distort the wall-clock tok/s numbers.

OUT_DIR="${OUT_DIR:-run_logs/gpu_suite_full}"
QUESTION_FILE="${QUESTION_FILE:-data/mt_bench/question.jsonl}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TREE_SIZES="${TREE_SIZES:-64 32 16 8 4}"

if [[ ! -f "$QUESTION_FILE" ]]; then
  bash scripts/medusa_gpu_suite/00_download_mtbench_questions.sh
fi

for n in $TREE_SIZES; do
  echo ""
  echo "=== Medusa run: tree_size=$n ==="
  TREE_SIZE="$n" \
  QUESTION_FILE="$QUESTION_FILE" \
  LIMIT="$LIMIT" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  OUT_DIR="$OUT_DIR" \
  ANSWERS_JSONL="$OUT_DIR/medusa_tree${n}_answers.jsonl" \
  TRACE_JSONL="$OUT_DIR/medusa_tree${n}_answers.traces.jsonl" \
  MODEL_ID="first-principles-medusa-vicuna-7b-v1.3-tree${n}" \
  VERBOSE_TIMING=0 \
  STEP_TEXT=0 \
  bash scripts/medusa_gpu_suite/01_run_medusa_tree.sh
done

echo ""
echo "=== Greedy baseline run ==="
QUESTION_FILE="$QUESTION_FILE" \
LIMIT="$LIMIT" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
OUT_DIR="$OUT_DIR" \
ANSWERS_JSONL="$OUT_DIR/vicuna_baseline_answers.jsonl" \
TRACE_JSONL="$OUT_DIR/vicuna_baseline_answers.traces.jsonl" \
bash scripts/medusa_gpu_suite/02_run_vicuna_baseline.sh

echo ""
echo "Full suite complete. Results in $OUT_DIR/ — please send that folder back."
