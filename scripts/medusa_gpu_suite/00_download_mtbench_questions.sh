#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Official FastChat MT-Bench questions: 80 two-turn questions.
# examples/mini_mtbench_questions.jsonl is a 1-question smoke file, NOT this.
DEST="${DEST:-data/mt_bench/question.jsonl}"
URL="https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"

mkdir -p "$(dirname "$DEST")"
curl -fL "$URL" -o "$DEST"

LINES=$(wc -l < "$DEST" | tr -d ' ')
echo "Downloaded $DEST ($LINES questions; expected 80)."
