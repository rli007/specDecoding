# Trace and Reimplement Hugging Face Speculative Decoding

## What This Project Does

This project traces Hugging Face's assisted/speculative decoding pipeline and
then manually reimplements a minimal greedy version of it.

The default setup is intentionally small:

- Target model: `gpt2`
- Assistant model: `distilgpt2`
- Tokenizer: `gpt2`
- Decoding: greedy only, `do_sample=False`
- Prompt: `The Stanford football team`

## Why

Hugging Face hides speculative decoding inside `generate()`. This project makes
the logic explicit, similar to a static-cache generate file that manually
reimplements normal greedy decoding.

The goal is understanding, not speed.

## HF Pipeline Being Traced

```text
target.generate(..., assistant_model=assistant)
    ↓
GenerationMixin.generate()
    ↓
generation mode selection
    ↓
ASSISTED_GENERATION
    ↓
_assisted_decoding()
    ↓
candidate_generator.get_candidates()
    ↓
assistant drafts candidate tokens
    ↓
target verifies candidate tokens
    ↓
accepted prefix is appended
    ↓
KV cache / model kwargs are updated
    ↓
repeat
```

`tools/trace_huggingface_assisted_generation.py` monkey-patches the local Transformers install where
possible and prints calls through `GenerationMixin.generate`,
`get_generation_mode`, `_assisted_decoding`,
`AssistedCandidateGenerator.get_candidates`, assistant forwards, and target
forwards.

## Manual Speculative Decoding Algorithm

Current accepted tokens:

```text
x
```

Assistant drafts:

```text
d1, d2, d3, d4
```

Target verifies:

```text
target prediction at next position
target prediction after d1
target prediction after d2
target prediction after d3
```

Acceptance:

```text
if target prediction == assistant draft token, accept
else reject at first mismatch and use target token
```

If all draft tokens are accepted, append them and then append one extra target
token if `max_new_tokens` allows.

The tensor-level entry point is `decoders.first_principles_speculative_decoder.generate(...)`:

```python
output_ids = generate(
    model=target_model,
    prompt_token_ids=input_ids,
    max_new_tokens=20,
    eos_token_id=tokenizer.eos_token_id,
    draft_model=assistant_model,
    num_assistant_tokens=4,
)
```

It is shaped like the static-cache `generate` example: token IDs go in, token
IDs come out. The implementation treats model forward calls as black boxes; it
does not call `target.generate(...)` or inspect target/draft internals.

To inspect every speculative step, run:

```bash
python decoders/first_principles_speculative_decoder.py
```

That prints draft forward input lengths, target verification input length,
draft tokens, target predictions, accepted count, rejection location, appended
tokens, and the final output.

For a longer paragraph-sized run:

```bash
python decoders/first_principles_speculative_decoder.py --paragraph
```

To print top-k logits at each assistant and target decision:

```bash
python decoders/first_principles_speculative_decoder.py --max-new-tokens 40 --show-logits --top-k-logits 5
```

To save the full next-token logit vectors used by both models:

```bash
python decoders/first_principles_speculative_decoder.py --paragraph --logits-out spec_logits.pt
```

To avoid reloading model weights between prompts, use interactive mode:

```bash
python decoders/first_principles_speculative_decoder.py --interactive
```

The target and assistant weights load once, then each prompt you type reuses the
same in-memory models. Type `quit`, `exit`, or an empty line to stop. If you use
`--logits-out` in interactive mode, later runs are written with numbered suffixes
such as `spec_logits_2.pt`.

## Larger Llama Session

For a larger Llama-family experiment, use the dedicated session file:

```bash
python tools/interactive_llama_speculative_session.py
```

By default it uses:

```text
target:    meta-llama/Meta-Llama-3-8B
assistant: meta-llama/Llama-3.2-1B
```

This loads the models once and then lets you enter many prompts. To use the
instruction-tuned pair:

```bash
python tools/interactive_llama_speculative_session.py \
  --target-model meta-llama/Meta-Llama-3-8B-Instruct \
  --assistant-model meta-llama/Llama-3.2-1B-Instruct
```

These Meta repos are gated on Hugging Face, so accept the model license and run
`huggingface-cli login` first. On CPU-only machines, 8B inference can be very
slow and memory-heavy.

The Llama session is intentionally conservative by default:

```text
max_new_tokens=2
dtype=float16
progress=True
heartbeat_seconds=5
low_cpu_mem_usage=True
```

For the first MPS test, run:

```bash
python tools/interactive_llama_speculative_session.py --device mps --max-new-tokens 1 --draft-len 1
```

During generation, the session prints before and after every assistant and
target forward. If a forward is slow, it also prints heartbeat lines such as
`still running after 5.0s`, so you can tell whether it is inside the 1B assistant
or 8B target pass. When the tiny run works, increase `--max-new-tokens` and
`--draft-len` gradually.

## Prefill Vs Decode

Prefill:

- full prompt forward pass
- creates KV cache
- produces logits for first generated token

Decode:

- uses previous KV cache
- feeds one new token or a small candidate block
- produces next logits
- updates KV cache

## How This Mirrors The Static-Cache File

The static-cache file manually reimplemented normal greedy generation:

```text
prompt -> prefill -> KV cache -> one-token decode loop
```

This project manually reimplements speculative generation:

```text
prompt -> prefill -> assistant draft -> target verify -> accept/reject -> update sequence
```

## Files

- `decoders/first_principles_speculative_decoder.py`: isolated tensor-level
  speculative decoder with structured per-step tracing.
- `decoders/simple_greedy_speculative_decoder.py`: clear manual speculative
  decoding, including the earlier verbose tokenizer/prompt-oriented demo.
- `decoders/cache_aware_speculative_decoder.py`: cached-style version that
  exposes prefill, decode, `past_key_values`, `cache_position`, and cache
  rebuilds.
- `tools/trace_huggingface_assisted_generation.py`: runs real HF greedy and
  assisted generation, then traces the assisted path.
- `tools/locate_huggingface_generation_source.py`: prints local Transformers
  source paths and search terms.
- `tools/compare_generation_outputs.py`: compares target greedy, HF assisted,
  and manual speculative outputs.
- `tools/interactive_llama_speculative_session.py`: long-running Llama-oriented
  session that loads a larger target and smaller assistant once, then reuses
  them for many prompts.
- `reference/recommended_model_pairs.txt`: compatible target/assistant pairs.
- `reference/static_cache_generation_reference.txt`: reference static-cache
  generation example this repo mirrors.
- `reference/sample_speculative_trace.txt`: saved trace output from an example
  speculative decoding run.

## How To Run

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python tools/trace_huggingface_assisted_generation.py
python tools/locate_huggingface_generation_source.py
python decoders/first_principles_speculative_decoder.py
python decoders/simple_greedy_speculative_decoder.py
python decoders/cache_aware_speculative_decoder.py
python tools/compare_generation_outputs.py
```

If your IDE uses Apple system Python and cannot import `torch`, run with the
Anaconda interpreter:

```bash
/opt/anaconda3/bin/python tools/trace_huggingface_assisted_generation.py
```

## Expected Result

For greedy decoding:

- normal target generation should match HF assisted generation
- manual speculative decoding should match normal target generation

`tools/compare_generation_outputs.py` prints decoded outputs, token IDs, equality checks, and the
first mismatch if something diverges.

## Known Limitations

- Initial manual version recomputes full sequences for clarity.
- Sampling is not supported.
- Batch generation is not supported; the first-principles implementation matches
  the single-batch style in `reference/static_cache_generation_reference.txt`.
- Llama is not supported at first.
- Quantization is not included.
- Static-cache export is not included yet.
- Speed benchmarking is not included yet.
- Some Hugging Face internals are local variables inside stack frames, so the
  trace prints the closest observable information from monkey-patched wrappers.

## Next Steps

After the simple version works:

- add assistant KV cache
- add target KV cache verification without cache rebuilds
- add Llama
- add static cache wrapper
- add quantized KV cache
- connect to Voyager compiler / on-chip execution
- compare against HF `_assisted_decoding` implementation more closely
