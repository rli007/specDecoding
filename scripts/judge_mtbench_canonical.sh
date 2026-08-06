#!/usr/bin/env bash
set -euo pipefail

# Canonical MT-Bench quality judging via FastChat's llm_judge (single-answer
# grading, 1-10 per turn). Judges ONLY configs whose text can differ from
# plain decoding: the greedy baseline reference plus the typical-acceptance
# Medusa runs. Greedy Medusa/assisted are character-identical to the baseline
# by construction (verified), so judging them would buy nothing.
#
# One-time setup (done 2026-08-06):
#   - FastChat cloned at ~/Desktop/FastChat
#   - its own venv at ~/Desktop/FastChat/.venv with openai==0.28.1
#     (the judge code uses the legacy OpenAI SDK API)
#   - answer files staged into fastchat/llm_judge/data/mt_bench/model_answer/
#     named by model_id (from run_logs/modal/gpu_suite_limit20/)
#
# Before running:  export OPENAI_API_KEY=sk-...   (keep it in ~/.zshrc, never
# in this repo). Canonical judge is gpt-4 (published MT-Bench tables use it);
# JUDGE_MODEL=gpt-4o is ~10x cheaper but its scores are not comparable to
# published numbers -- fine for relative comparison between our own configs.
#
# FIRST_N=20 because the limit20 suite answered the first 20 questions only;
# raise it after a full 80-question run (and restage the answer files).

FASTCHAT="${FASTCHAT:-$HOME/Desktop/FastChat}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4}"
FIRST_N="${FIRST_N:-20}"

MODELS=(
  vicuna-7b-v1.3-greedy-baseline
  first-principles-medusa-vicuna-7b-v1.3-medusa_tree64_typical_t0.7
  first-principles-medusa-vicuna-7b-v1.3-medusa_tree32_typical_t0.7
  first-principles-medusa-vicuna-7b-v1.3-medusa_tree16_typical_t0.7
  first-principles-medusa-vicuna-7b-v1.3-medusa_tree8_typical_t0.7
  first-principles-medusa-vicuna-7b-v1.3-medusa_tree4_typical_t0.7
)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Get one at platform.openai.com -> API keys," >&2
  echo "then: echo 'export OPENAI_API_KEY=sk-...' >> ~/.zshrc && source ~/.zshrc" >&2
  exit 1
fi

cd "$FASTCHAT/fastchat/llm_judge"
# Python 3.13 skips pip's "__editable__*.pth" pointer files ("hidden .pth"),
# which breaks the editable fastchat install; put the clone on the path directly.
export PYTHONPATH="$FASTCHAT${PYTHONPATH:+:$PYTHONPATH}"
# gen_judgment.py prints its match plan then waits for Enter before spending
# API money; the piped newline auto-confirms so the script runs unattended.
echo "" | "$FASTCHAT/.venv/bin/python" gen_judgment.py \
  --mode single \
  --judge-model "$JUDGE_MODEL" \
  --first-n "$FIRST_N" \
  --parallel 4 \
  --model-list "${MODELS[@]}"

"$FASTCHAT/.venv/bin/python" show_result.py \
  --mode single \
  --judge-model "$JUDGE_MODEL" \
  --model-list "${MODELS[@]}"
