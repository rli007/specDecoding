# Stripped-Down Hugging Face Assisted Decoding

This repo is a learning/debugging sandbox for Hugging Face assisted generation
and speculative decoding. The main goal is to make the hidden logic inside
`generate(..., assistant_model=...)` visible, then reimplement the same behavior
with small, explicit Python loops.

The active model pair is:

```text
target:    meta-llama/Meta-Llama-3.1-8B
assistant: meta-llama/Llama-3.2-1B
```

The Llama repos are gated on Hugging Face. Accept the model license and log in
before loading them locally.

## Core Idea

Hugging Face assisted generation follows this shape:

```text
current tokens
  -> assistant drafts candidate tokens
  -> target verifies candidate block plus one extra position
  -> accept candidates until mismatch/rejection
  -> append accepted tokens plus one target/sample token
  -> update assistant draft budget
  -> repeat
```

This repo keeps that logic close to the stripped-down static-cache generation
style:

```text
prompt -> model forward -> logits/cache -> choose token(s) -> update sequence
```

The active stripped-down implementation calls Hugging Face only to run actual
model forwards. It does not call HF `generate()` inside the local decoder.

## Active Files

- `decoders/stripped_down_llama_assisted_decoder.py`  
  Manual assisted/speculative decoder for the Llama 3.1 8B target and Llama 3.2
  1B assistant. Supports greedy assisted decoding and speculative sampling.

- `tools/compare_hf_vs_stripped_assisted_steps.py`  
  Main test harness. Runs HF assisted generation and the stripped-down decoder
  with the same prompt/models, prints step-by-step comparisons, and writes a
  timestamped log to `run_logs/`.

- `decoders/first_principles_speculative_decoder.py`  
  Older but still useful tensor-level decoder with structured tracing.

- `tools/interactive_llama_speculative_session.py`  
  Interactive Llama session that loads the models once and reuses them across
  prompts.

- `reference/candidate_generation.txt`  
  Hugging Face candidate-generator excerpts.

- `reference/relevant_portions.txt`  
  Hugging Face `_assisted_decoding` excerpts.

- `reference/static_cache_generation_reference.txt`  
  The stripped-down static-cache `generate` reference that inspired the local
  style.

- `reference/recommended_model_pairs.txt`  
  Compatible target/assistant model notes.

Archived earlier experiments live in `archive/`.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login
```

If your IDE uses the wrong Python, run with the Anaconda interpreter:

```bash
/opt/anaconda3/bin/python tools/compare_hf_vs_stripped_assisted_steps.py
```

## Main Comparison

Greedy assisted decoding:

```bash
python tools/compare_hf_vs_stripped_assisted_steps.py --mode greedy
```

Speculative sampling:

```bash
python tools/compare_hf_vs_stripped_assisted_steps.py \
  --mode sampling \
  --top-k-probs 5 \
  --seed 1234
```

Useful knobs:

```bash
--prompt-name story
--prompt "In a small workshop near the ocean, the engineer"
--max-new-tokens 16
--num-assistant-tokens 2
--top-k-probs 5
--no-live-steps
--no-log
```

By default, each run writes a timestamped text log to `run_logs/`.

## Reading The Step Output

Each step shows:

- `draft`: tokens proposed by the assistant
- `target`: target-model diagnostic predictions for the candidate block
- `append`: tokens actually appended to the output
- `accepted`: number of assistant draft tokens accepted
- `budget`: assistant draft length before and after the heuristic update

In greedy mode, acceptance is:

```text
accept while assistant token == target argmax token
```

In sampling mode, acceptance uses the speculative sampling rule:

```text
accept candidate with probability min(1, p_target / q_assistant)
```

The sampling output prints `q`, `p`, `p/q`, the random draw `r`, and the
accept/reject result for each candidate token.

## Greedy Vs Sampling Expectations

Greedy mode should usually match HF step-by-step, because both sides use argmax.

Sampling mode can diverge even with the same seed. HF samples inside
`assistant.generate(...)`, while the local implementation samples from manual
forward calls. On MPS/float16, tiny numerical differences or different RNG
consumption can change a sampled token, especially in the long tail of the
vocabulary. Once the first sampled draft differs, the rest of the sequence will
branch.

The acceptance diagnostics are still useful: they show whether the local
speculative-sampling math agrees with HF for the candidates that were sampled.

## Run The Local Decoder Directly

```bash
python decoders/stripped_down_llama_assisted_decoder.py --mode greedy
python decoders/stripped_down_llama_assisted_decoder.py --mode sampling --top-k-probs 5
```

For a tiny MPS smoke test:

```bash
python tools/compare_hf_vs_stripped_assisted_steps.py \
  --max-new-tokens 2 \
  --num-assistant-tokens 1
```

For a cached GPT-2 smoke test:

```bash
python tools/compare_hf_vs_stripped_assisted_steps.py \
  --target-model gpt2 \
  --assistant-model distilgpt2 \
  --tokenizer-model gpt2 \
  --device cpu \
  --dtype none
```

## Current Limitations

- Batch size is 1.
- Target and assistant are assumed to share tokenizer/vocab.
- The local decoder rebuilds target prefix cache semantics simply rather than
  reproducing every HF cache-cropping detail.
- Beam search and universal assisted decoding with different tokenizers are out
  of scope.
- Sampling can branch from HF because exact sampling RNG/numerics are hard to
  reproduce across implementation paths.
