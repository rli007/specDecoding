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

- `decoders/ngram_speculative_decoder.py`  
  First-principles prompt-lookup / n-gram speculative decoder. This one needs no
  assistant model; it copies continuations from repeated n-grams in the existing
  context and verifies them with target forwards.

- `decoders/medusa_speculative_decoder.py`  
  Medusa-style decoder with inspectable Medusa-head candidate construction and a
  slow path verifier. It does not call HF `generate()`, but useful behavior
  requires trained Medusa heads.

- `decoders/eagle_speculative_decoder.py`  
  EAGLE-style hidden-state drafter loop with a pluggable `propose_tree(...)`
  interface. Exact EAGLE/EAGLE3 behavior requires trained EAGLE draft weights
  paired with the target model.

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

## Extra Algorithm Decoders

The newer algorithm files separate the proposal strategy from target
verification:

```text
ngram:  repeated suffix in context -> copied continuation -> target verifies
Medusa: target hidden state -> Medusa heads -> candidate tree -> target verifies
EAGLE:  target hidden state -> trained EAGLE drafter -> candidate tree -> target verifies
```

Run the tiny no-download smoke test:

```bash
python tools/smoke_algorithm_decoders.py
```

Run n-gram prompt lookup with a real model:

```bash
python decoders/ngram_speculative_decoder.py \
  --target-model gpt2 \
  --prompt "The future of AI is the future of" \
  --max-new-tokens 20 \
  --num-speculative-tokens 4 \
  --max-matching-ngram-size 4 \
  --progress
```

Medusa and EAGLE are implemented as first-principles loops, but they are not
weight loaders. For Medusa, pass a module that maps target hidden states to
`[num_heads, batch, seq, vocab]` Medusa logits. For EAGLE, pass a drafter object
with `propose_tree(input_ids, hidden_states, target_logits, max_depth, top_k,
max_paths)` returning candidate token paths. This keeps the algorithm readable
and avoids hiding the important work inside a framework `generate()` call.

Run the smallest public Medusa setup we have discussed:

```bash
python tools/run_local_medusa.py \
  --base-model lmsys/vicuna-7b-v1.3 \
  --medusa-heads FasterDecoding/medusa-vicuna-7b-v1.3 \
  --prompt "The professor asked for" \
  --max-new-tokens 1 \
  --device mps \
  --dtype float16 \
  --progress
```

That command still loads a 7B Vicuna base model plus the Medusa heads. The
default `--choice-preset linear --top-k 1` verifies one Medusa path and is the
best first local smoke test. After that works, try `--choice-preset small-tree
--top-k 2` to inspect branching behavior; it will be slower because this repo
verifies paths one at a time instead of using optimized Medusa tree attention.

For a longer Medusa-paper-style sequence, keep the fast linear path and log the
partial text after every accepted continuation:

```bash
python tools/run_local_medusa.py \
  --base-model lmsys/vicuna-7b-v1.3 \
  --medusa-heads FasterDecoding/medusa-vicuna-7b-v1.3 \
  --prompt "The professor asked for" \
  --longer-sentence \
  --device mps \
  --dtype float16 \
  --progress \
  --output-txt run_logs/medusa_longer_sentence.txt
```

`--longer-sentence` currently means 12 new tokens. This is still slower than
the paper's implementation because the repo intentionally uses ordinary target
forwards to verify candidate paths instead of Medusa's optimized tree attention.

Run a tiny MT-Bench-style Medusa generation pass:

```bash
python tools/run_medusa_mtbench.py \
  --question-file examples/mini_mtbench_questions.jsonl \
  --limit 1 \
  --max-new-tokens 16 \
  --device mps \
  --dtype float16 \
  --progress
```

This writes FastChat-shaped answers to
`run_logs/medusa_mtbench_mini_answers.jsonl` and detailed Medusa step traces to
`run_logs/medusa_mtbench_mini_answers.traces.jsonl`. The bundled question file
is only a local smoke test. For the real MT-Bench questions, download
FastChat's `question.jsonl` and pass it with `--question-file`. A full
paper-style run also needs an LLM judge, usually through the FastChat judge
scripts, but start with generation-only because this inspectable Medusa verifier
is intentionally much slower than optimized tree attention.

## Current Limitations

- Batch size is 1.
- Target and assistant are assumed to share tokenizer/vocab.
- The local decoder rebuilds target prefix cache semantics simply rather than
  reproducing every HF cache-cropping detail.
- Beam search and universal assisted decoding with different tokenizers are out
  of scope.
- Sampling can branch from HF because exact sampling RNG/numerics are hard to
  reproduce across implementation paths.
- Medusa and EAGLE use slow per-path verification for inspectability instead of
  optimized tree attention.
