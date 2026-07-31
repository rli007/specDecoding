# CLAUDE.md — specDecoding

Context for future sessions. Written 2026-07-28 after a full read of the repo,
the working diff, `run_logs/`, and the Voyager compiler source. Updated
2026-07-29: re-verified the working diff and line references; added notes on
the timing instrumentation and `NOTES_FOR_MENTOR.md`.

---

## 1. Who this is for and what the project is

The owner is a first-year undergrad researcher in a Stanford hardware group
(Raina lab) working on **Sphinx**, a 2 mm² TSMC 7nm edge LLM accelerator
(64×64 systolic Matrix Unit, 128-wide Matrix-Vector Unit, SpMM unit, Vector
Unit, 2.5 MB scratchpad, 8.192 TOPS peak, 1 GHz, tapeout Mar 2026). Sphinx's
three published contributions are mixed-precision codebook quantization
(NF4-in-INT6 + FP8 E5M3 microscaling scales), dynamic activation outlier
filtering via a sparse unit, and 2-bit KIVI KV cache with a fused
low-precision attention path. **Speculative decoding is listed as in-progress
future work.**

The user understands the research and the math but is still learning industry
vocabulary and toolchains. Explain terms; do not assume tribal knowledge.
Prefer building understanding from low-level mechanism upward.

**This repo is not a product.** It is a from-scratch, deliberately inspectable
implementation of speculative decoding methods, built so their tensor
operations can be mapped onto Sphinx and cost-modeled.

### The one-paragraph thesis

On Sphinx, autoregressive decode is **memory-bound**: one token requires
streaming all ~3.4 GB of 4-bit weights from DRAM (~50 ms at ~68 GB/s) but only
13.5 GOP of arithmetic (~1.6 ms at 8.192 TOPS), so the Matrix Unit sits ~3%
utilized. Prefill, which shares one weight load across many tokens, hits ~97%
utilization (presentation slide 15). Speculative decoding makes decode look
like prefill: it converts the sequential token dependency into a batch of
tokens that share a single weight load. The chip has idle MACs and idle
bandwidth headroom in exactly the dimension speculative decoding supplies.

> **The two most important results in this file are in §6: the free-tree-size
> ridge point (N\* ≈ 30 nodes on Sphinx, vs the 63-node tree Medusa ships) and
> the quantization/speculation tension. Read §6 before proposing any work.**

---

## 2. Current state

Active work: **Medusa** with pretrained public heads.

- base model: `lmsys/vicuna-7b-v1.3`
- Medusa heads: `FasterDecoding/medusa-vicuna-7b-v1.3` (5 heads, 1 residual layer each)
- **Decision (2026-07-28): stay on Vicuna-7B**, not Llama-3.1-8B. Pretrained
  Medusa heads exist for Vicuna, so measured acceptance and modeled cycles
  refer to the same system. Note the mismatch with the Sphinx slides, which
  evaluate Llama-3.1-8B.
- Vicuna-7B is **Llama-1 architecture**: MHA, no GQA, hidden 4096, vocab 32000,
  32 layers, intermediate 11008.

### Verified correctness

The Medusa implementation was checked mechanism-by-mechanism against upstream
`FasterDecoding/Medusa` and is **faithful**. Verified: candidate pool
construction (base-LM argmax as free root + top-k per head), `tree_indices`
arithmetic `path[-1] + top_k*depth + 1`, the reversed prefix-dedup path
enumeration for `retrieve_indices`, the `[:, :-1]` / `[:, 1:]` posterior
alignment, `argmax(accept_lengths)` tiebreak, `accept_length + 1` append, KV
select-and-compact, and carrying `logits[best, accept_length]` forward as
next-step state. No mechanism bugs found.

Independent confirmation: greedy Medusa and the plain Vicuna greedy baseline
produce **character-identical** output ("Careful debugging helps software
projects in several ways. First, it allows developers to"). Greedy Medusa is
lossless, as designed. **Do not re-litigate algorithm correctness** — that part
is settled. Effort belongs on metrics, tree sizing, and the Voyager mapping.

### Timing instrumentation (commit de66c17, 2026-07-29): no algorithm changes

The +460/−132 change to `medusa_speculative_decoder.py` and
`run_medusa_mtbench.py` is **pure instrumentation**. It adds a
`--verbose-timing` flag: every stage of a step (prefill forward, tree target
forward, Medusa heads, logit reorder, posterior eval, cache copy, trace
packaging, append/state update) is wrapped in a device-synchronized
`_timed_section`; per-step results land in `MedusaStepTrace.timings` and the
MT-Bench runner emits `timing_totals` per turn. Per gotcha §5.3, these numbers
are for *relative* stage breakdown only — never quote local absolute times.

---

## 3. Repo map

```
decoders/
  first_principles_speculative_decoder.py  shared utils: device/dtype, timing contexts,
                                           top-k logit summaries, EOS/stop logic.
                                           Everything imports from here.
  medusa_speculative_decoder.py            ← LIVE WORK. 1592 lines. See §4.
  stripped_down_llama_assisted_decoder.py  draft-model spec decoding, matched step-for-step
                                           against HF generate(assistant_model=...)
  ngram_speculative_decoder.py             prompt-lookup drafting, no second model
  eagle_speculative_decoder.py             EAGLE shell; needs trained drafter weights

tools/
  run_local_medusa.py                      one prompt, full human-readable trace
  run_medusa_mtbench.py                    MT-Bench loop -> answers.jsonl + traces.jsonl
  run_vicuna_mtbench_baseline.py           plain greedy, same harness (apples-to-apples)
  judge_openrouter_mtbench.py              GPT-4o judge via OpenRouter
  compare_hf_vs_stripped_assisted_steps.py HF-vs-ours differential test
  smoke_algorithm_decoders.py              tiny no-download sanity check

scripts/medusa_gpu_suite/                  CUDA wrappers: medusa -> baseline -> judge
NOTES_FOR_MENTOR.md                        1-page status: N* finding, quant/spec tension, Voyager
                                           plan, 4 open hardware questions
run_logs/                                  GITIGNORED — exists only on this machine, no backup.
                                           Current apples-to-apples set (Jul 23): greedy_rerun,
                                           tree_rerun, baseline_rerun answers + traces
  latest_three_run_analysis.txt            useful prose analysis of the Jul 23 runs
  archive/                                 superseded runs (May 12 differential-test logs,
                                           mini-file smoke run, non-lossless typical run,
                                           first baseline, one-off longer-sentence trace)
examples/mini_mtbench_questions.jsonl      1-question smoke file, NOT real MT-Bench
archive/                                   superseded experiments; also holds the May-era HF
                                           source excerpts (archive/reference/, moved from
                                           top-level reference/ 2026-07-29) and the draft-model
                                           interactive REPL (archive/tools/)
```

### Non-negotiable design rule

**No decoder calls HF `generate()`.** Decoders call the model only for raw
forwards; every intermediate is a named dataclass (`MedusaBuffers`,
`MedusaCandidates`, `MedusaVerification`, `MedusaStepTrace`). This costs lines
of code and is correct for this project: mapping to silicon requires knowing
exactly which tensor ops occur, at what shapes, in what order. Preserve this
when editing.

---

## 4. How Medusa works, mapped to this code

Medusa adds K=5 trained heads on the target's final hidden state. Head *i*
predicts token *t+i+2*. Each head is `[residual block × L, Linear(hidden→vocab)]`
(`MedusaHeadStack`, medusa_speculative_decoder.py:191-227); the residual
`Linear` is zero-init so an untrained head degrades to a copy of `lm_head`.

Per-step loop (`generate()`, medusa_speculative_decoder.py:1183-1500):

1. **Build the candidate tree.** Root = `argmax(lm_head(hidden))` — the base
   model's own next token, **already known correct, so it is free**. Children =
   top-10 tokens from each Medusa head. Pool = 1 + 5×10 = 51 tokens; 64 tree
   nodes select from it. (`generate_medusa_candidates`, :863-914)
   Therefore **`tokens_per_step = accept_length + 1`** and Medusa never does
   worse than plain decoding.

2. **One target forward over all 64 tree nodes** with a hand-built 4D
   additive attention mask: each node sees the prefix and only its own
   ancestors (`_additive_tree_attention_mask`, :761-781; boolean pattern
   built at :503-512). Position IDs encode **tree depth**, not flat index
   (:1063), so siblings share a position — correct, they are alternatives for
   the same slot. **This is the GEMV→GEMM conversion that motivates the whole
   project.**

3. **Medusa heads over the 64 output hidden states** (:1091).

4. **Reorder** flat tree logits into per-path rows via `retrieve_indices`
   (:792-811). Paths are enumerated leaf-to-root, longest-first, so
   `selected_path_index` in traces is NOT a quality rank — matches upstream.

5. **Posterior acceptance** (`evaluate_posterior`, :917-968). Greedy branch is
   `cumprod(candidate == argmax(logits)).sum()`, i.e. length of the leading
   run of matches. Exactly lossless.

6. **Append** `accept_length + 1` tokens.

7. **KV cache surgery** (`_copy_selected_tree_cache`, :971-1013). The forward
   wrote KV for all 64 nodes; copy the accepted path's entries into contiguous
   slots and truncate. Returns `False` on unfamiliar cache layouts so the
   caller falls back to full re-prefill rather than corrupting state
   (:1442-1458). Traces show `cache_updated=true` throughout.
   For hardware: this is a **gather/scatter DMA**, not a matmul — maps to
   neither Matrix Unit, MVU, nor SpMM unit. Three implications:
   - **Cheap.** fp16 KV is ~512 KB/token for Vicuna-7B
     (2 × 32 layers × 32 heads × 128 dim × 2 B); ~3 accepted tokens ≈ 1.5 MB
     read + 1.5 MB written ≈ 45 µs at 68 GB/s, i.e. <0.1% of a 50 ms step. At
     2.25-bit KV it is ~72 KB/token → ~6 µs. Not worth optimizing; worth
     *scheduling* correctly.
   - **The index list is data-dependent** (`retrieve_indices[best_path]`, known
     only after posterior evaluation), so it cannot be baked into a static
     instruction schedule. Needs either a Rocket-issued DMA with runtime
     addresses or an indexed/scatter-gather DMA descriptor. **Open question for
     the mentor: does Sphinx's DMA engine support runtime-indexed gather on the
     token axis?** If not, the static fallback is to copy the whole 64-node
     block and let the next step's mask hide dead slots — ~20× the traffic
     (still only ~1 ms) but fully static.
   - **KIVI interaction (contribution 3).** V is quantized per-token, so its
     per-token scales must be gathered by the same index list. K is quantized
     per-channel in groups, which is why KIVI keeps a small fp16 residual
     buffer of the newest tokens — a channel group cannot be finalized until
     it is full. **Medusa appends 1–5 tokens per step instead of exactly 1, so
     that residual-buffer fill/flush bookkeeping changes.** Real integration
     issue between contribution 3 and speculative decoding; flag it early.

8. **Carry forward** logits + Medusa logits at the last accepted position, so
   no re-prefill is needed.

### The choice presets

`VICUNA_7B_STAGE2_CHOICES` (:148-160) is the official 63-path set from the
Medusa paper. Paths are rank tuples: `(0,)` = head 0's top choice,
`(1,3)` = head 0's 2nd then head 1's 4th. 63 paths + root = 64 nodes.
`_validate_tree_choices` (:465-480) enforces **prefix-closure** — if `(0,1,2)`
is present so must be `(0,1)` and `(0,)` — which is what makes the set a tree
so ancestors physically exist to be attended to. The set is dense where the
model is confident (`(0,0,0,0)` exists) and shallow-wide where it isn't
(`(9,)` exists, `(9,0)` does not).

`--attn-implementation eager` is required: eager attention uses a passed 4D
mask verbatim; fused/flash kernels may assume causality and silently ignore or
reshape a custom tree mask.

---

## 5. Known gotchas — read before interpreting any run

1. **`accepted_count` is misnamed: it means "tokens appended," including the
   free root token** (:1121-1122). So `accepted_tokens_per_step ==
   appended_tokens_per_step` always. The paper's accept length τ =
   `appended - 1`. Translation for existing logs:
   greedy 1.778 appended → τ=0.778; greedy 2.000 → τ=1.000;
   typical 2.667 → τ=1.667.
   **Fixed 2026-07-31 (runner-side):** `run_medusa_mtbench.py` now emits
   per-step `accept_length` + `tokens_per_step` in traces,
   `accept_length_per_step` (= τ) in per-turn stats, and prints
   `accept_length/step` in the done line. The `MedusaStepTrace.accepted_count`
   dataclass field itself is still the misnamed appended count — old trace
   files keep the old semantics.

2. **`--acceptance typical` is a no-op unless `--temperature > 0`** — with
   `temperature <= 0` it falls through to the greedy branch (:936), and
   `--temperature` defaults to `0.0`. Once temperature is nonzero the run is
   **no longer lossless** and no longer comparable to the greedy baseline.
   This is why `latest_three_run_analysis.txt` could not reconcile the typical
   run's divergent output with its recorded settings.
   **For hardware speedup claims, use greedy only.** It is lossless, so
   speedup is apples-to-apples.

3. **Local (MPS) timings are meaningless.** ~50 s/token for 7B fp16 is ~1000×
   off — macOS is swapping a 13.5 GB weight tensor. Every timing number in
   `run_logs/` is dominated by page faults, which is why two "identical" runs
   differed by 1.44×. **Acceptance statistics ARE trustworthy** (deterministic
   and device-independent in greedy mode); timing statistics are not. Get
   acceptance locally, get cycles from Voyager. Do not quote local tok/s.

4. `examples/mini_mtbench_questions.jsonl` is a 1-question smoke file. Real
   MT-Bench = FastChat's `question.jsonl` (80 two-turn questions).

5. 16-token generations are too short for reliable τ: the first step of every
   generation is almost always τ=0 and dominates the average. Use ≥128 tokens
   and multiple questions.

---

## 6. The central hardware finding (order-of-magnitude, to be replaced by Voyager)

Cost model: Vicuna-7B, 6.74B params at ~0.5 bytes/param (Sphinx mixed
precision), ~68 GB/s DRAM, 8.192 TOPS.

| | plain decode (1 tok) | Medusa tree (64 nodes) |
|---|---|---|
| DRAM, weights (read once either way) | 3.4 GB → ~50 ms | 3.4 GB → ~50 ms |
| DRAM, Medusa heads (~739M params ≈ 370 MB) | — | +~5.4 ms |
| Arithmetic | 13.5 GOP → ~1.6 ms | 864 GOP → ~105 ms |
| **Bound by** | **memory, ~50 ms** | **compute, ~105 ms** |

A 64-node tree step costs ~2.1× a decode step and returns ~1.9 tokens
(measured greedy) → **~0.9×, i.e. a slowdown.**

### The free-tree ridge point N\*  ← KEY RESULT, KEEP

A verification step over N tree nodes has two costs, and (with slide-13 double
buffering overlapping them) the step takes the **max**:

    T_memory  = M / BW              M = fixed bytes streamed per step
                                    (weights P·b, plus the KV cache, plus the
                                     Medusa heads) — INDEPENDENT of N, because
                                     each is loaded once and reused by all N
                                     tree nodes
    T_compute = N · 2P / TOPS       LINEAR in N

Tokens are free while T_compute ≤ T_memory. Solving, and taking M ≈ P·b
(weights-dominated, short context):

    N* = M · TOPS / (2P · BW)  ≈  (b · TOPS) / (2 · BW)

Note **P cancels** — the free tree size does not depend on model size.

For Sphinx (TOPS = 8192 GOPS, BW ≈ 68 GB/s):

| weight precision | b (bytes/param) | N\* (free tree nodes) |
|---|---|---|
| fp16 | 2.0 | ~120 |
| int8 | 1.0 | ~60 |
| **nf4_6 (Sphinx)** | **~0.5** | **~30** |

**Below ~30 nodes, tree verification costs the same wall-clock as plain decode —
the extra tokens are free, paid for with otherwise-idle MACs. Above 30 you
start paying. At 64 you pay ~2×, which at the measured ~1.9 tokens/step is a
net slowdown.**

**Sensitivity:** N\* scales **linearly in bandwidth** and **linearly in
bytes/param**. The 68 GB/s figure is an assumed mobile-LPDDR5 number —
substitute the group's real target before quoting N\* to anyone. It is
inversely proportional, so 34 GB/s → N\* ≈ 60; 136 GB/s → N\* ≈ 15.

### Consequence 1: tree size is a hardware parameter, and 63 is an A100 number

Medusa's 63-path tree was tuned on an A100, whose ridge point sits at ~120–400
tokens. Porting it unchanged to a 2 mm² edge accelerator is a category error.
**The mapping from accelerator parameters to optimal tree size is unpublished**
— a real contribution available here.

### Consequence 2: quantization and speculation compete for the SAME slack

Both optimizations consume one resource: **the gap between the memory roofline
and the compute roofline.** Quantization spends it to make the baseline step
faster (cuts `T_memory`, leaves `T_compute` alone). Speculative decoding spends
it to make each step do more work (raises `T_compute` toward `T_memory`).
Whatever one takes, the other cannot have.

So **the two speedups do not multiply.** Illustrative, same chip:

| | baseline step | free tree | ~tokens/step | spec speedup | total vs fp16 |
|---|---|---|---|---|---|
| fp16 | ~200 ms | ~120 nodes | ~3.5 | ~3.5× | ~3.5× |
| nf4_6 | ~50 ms | ~30 nodes | ~2.5 | ~2.5× | ~10× |

Quantization still wins massively in absolute terms — **quantize.** The point
is that a claim like "4× from quantization × 3× from Medusa = 12×" is wrong;
the speculative factor **shrinks as quantization improves**.

**Sphinx's contribution 3 pushes the same way.** The KV cache is also loaded
once and reused by all N tree nodes (QKᵀ reads K once, uses it N times), so KV
bytes sit in `M` alongside weights. Going 16-bit → 2.25-bit KV (slide 38)
shrinks `M` further and lowers N\* again. Contributions 1 and 3 both eat
speculative headroom.

**Why this is paper-shaped:** every speculative-decoding paper reports speedup
on fp16 GPUs; every quantization paper reports memory/latency reduction without
speculation. Nobody has published the interaction curve — *speculative speedup
as a function of weight precision on a fixed accelerator.* Voyager can sweep
`b` (fp16 / int8 / nf4_6) and N independently and read out cycles. That is a
clean 2-D sweep and a novel result.

### Consequence 3: on THIS chip, Medusa should beat draft-model speculation

The same roofline argues for Medusa over a Llama-3.2-1B draft model, and the
reason is specific to memory-bound hardware. A draft model must run *k*
**sequential** decode steps, and each one is itself memory-bound — 1B params at
4-bit ≈ 0.6 GB ≈ 9 ms, and no amount of chip parallelism helps because step
*i+1* needs step *i*.

    draft-model, k=4:  4 × 9 ms (drafting) + ~50 ms (verify 5 tokens)
                       ≈ 86 ms for ~3 tokens  →  ~29 ms/token  →  ~1.7×
    Medusa, N≈30:      0 ms drafting + ~55 ms (tree step)
                       ≈ 55 ms for ~2.5 tokens  →  ~22 ms/token  →  ~2.3×

Medusa's drafting is a *parallel* matmul folded into a step that was already
memory-bound, so it is nearly free; the draft model's is sequential and
memory-bound, so it is nearly as expensive as what it is trying to accelerate.
Medusa also avoids a second resident weight set (0.6 GB) and a second KV cache.

**This is the argument for why Medusa is the right method for Sphinx**, and it is
a hardware argument, not one you will find in the Medusa paper.

These are sketches under stated assumptions, not results. Voyager's scheduled
cycle counts will differ (tiling, buffer pressure, attention GEMMs folded into
the parameter count). The *shape* of the curve — optimum well below 64 —
follows from the roofline and should survive.

### Two available optimizations, both hardware-motivated

**(a) Run the Medusa heads on 1 position, not 64.**
`medusa_speculative_decoder.py:1091` evaluates the heads over all 64 tree
hidden states (faithfully copying upstream), but only **one** position's Medusa
logits are ever used — sliced out at :1163. Reordering to
`forward → lm_head(64) → posterior → medusa_heads(1 position)` is exactly
equivalent.

Per head per position: `4096×4096` residual + `4096×32000` output
= 147.9M MAC = 295.7 MOP. × 5 heads = **1.48 GOP per position.**
- 64 positions: 94.6 GOP → **11.5 ms** on Sphinx
- 1 position: 1.48 GOP → **0.18 ms**

Frame the benefit as **budget, not time.** Setting T_memory = T_compute with
M = 3.4 GB weights + 0.37 GB head weights = 3.77 GB → T_memory ≈ 55.4 ms:

    heads at 64 positions:  N × 1.648 ms + 11.5 ms = 55.4  →  N* ≈ 27
    heads at  1 position:   N × 1.648 ms +  0.18 ms = 55.4  →  N* ≈ 34

**The optimization buys ~25% more free tree nodes**, which converts to
acceptance and then to speedup. In the memory-bound (small-tree) regime it
saves almost no wall-clock directly, because the heads' 370 MB weight *load* is
unchanged — only their arithmetic shrinks.

**(b) Quantize the Medusa heads much harder than the backbone.** ← strong idea

The verification path (`lm_head` + backbone) must be numerically exact, or
greedy Medusa stops being lossless w.r.t. the deployed model. **The speculation
path (Medusa heads) has no such requirement: a bad guess costs acceptance rate,
never correctness.** So the heads can run at 2–3 bit with zero correctness risk
and only a mild τ penalty.

Head weights ≈ 739M params. At nf4_6 (~0.5 B/param) that is ~370 MB — **~11% of
the per-step DRAM budget.** At 2-bit it is ~92 MB, cutting the Medusa memory
overhead to ~2.7% and raising N\* further.

This is precision-by-*role* rather than precision-by-*layer*, which is a
natural extension of the slide-11 sensitivity analysis and looks like a real
contribution. Cheap to test: quantize `MedusaHeadStack` weights in PyTorch,
measure τ on MT-Bench, and confirm output is byte-identical in greedy mode
(it must be).

---

## 7. Voyager compiler export — the next milestone

Repo: https://github.com/jeffreyyu0602/voyager-compiler
Paper: arXiv 2509.15205, "Voyager: An End-to-End Framework for Design-Space
Exploration and Generation of DNN Accelerators". Sphinx is Voyager-generated.
Reference: `test/test_codegen.py`.

**A local clone exists at `~/Desktop/voyager/voyager-compiler`** — checked
2026-07-29: it is behind GitHub main and the CLI has drifted (local:
`--mixed_precision`, `--hardware_unrolling`, `--remove_duplicate`; main:
`--enable_mixed_precision`, `pe_array_size` via a config module,
`--compile_single_layer`, `--attn_implementation`). `git pull` and re-check
`python test/test_codegen.py --help` before trusting any flag list here,
including the step-0 command below. `--remove_duplicate` /
`--compile_single_layer` compiles a SINGLE decoder layer — use it for fast
iteration and scale results by 32 layers.

Pipeline: PyTorch model → PT2E static graph → quantization (`prepare_pt2e` →
calibrate → `convert_pt2e`) → `transform()` (fusion, layout, tiling,
scheduling) → `compile()` (instruction generation) → optional `--report`
(cycle + DRAM-traffic estimate as `.xlsx` + `.perfetto.json`).

### What "hf.export" means and what it constrains

`torch.export` traces a model into a static graph of primitive `aten` ops with
**all Python removed**. HF's `TorchExportableModuleWithStaticCache` (vendored
into `voyager_compiler/llm_utils.py`) makes the KV cache a fixed-size buffer
so shapes are static. Consequences:

1. **No data-dependent control flow.** Medusa's accept/reject branches on
   tensor *values*. **The decode loop cannot be exported.** This is what
   "static graph" means, not a bug to engineer around.
2. **Static shapes.** Hence `StaticCache` with fixed `max_cache_len` and an
   explicit `[1,1,S,max_cache_len]` mask.
3. **You export a *shape of work*, not a program.** `llm_prefill` and
   `llm_decode` in `test_codegen.py:369-499` are the same code path differing
   only in sequence length and mask. The generation loop stays in Python.

This decomposition is correct and maps onto the speedup equation:

    speedup = (tokens per step)  /  (tree-step cycles / decode-step cycles)
              ^ from MT-Bench traces      ^ from Voyager --report

Voyager owns the denominator; existing traces own the numerator. Nothing else
is needed.

### Why this is tractable

    # test_codegen.py llm_decode
    example_args = (inputs_embeds,       # [1, 1, 4096]
                    causal_mask,         # [1, 1, 1, max_cache_len]
                    position_embeddings, cache_position)

    # what Medusa needs
    example_args = (inputs_embeds,       # [1, N, 4096]   N = tree size
                    tree_mask,           # [1, 1, N, max_cache_len]
                    position_embeddings, cache_position)

Structurally identical: different values, different N, same graph topology.
The mask is already a graph input upstream (they even quantize it to int1 at
`test_codegen.py:531-538`), so a non-causal mask needs **no compiler changes**.
`_additive_tree_attention_mask` already produces the mask and
`buffers.position_ids` already produces the depths.

### Sphinx-matching flags

- `--pe_array_size 64,64` — the systolic array
- `--frequency 1.0` — 1 GHz
- `--cache_size` — the 2–2.5 MB scratchpad (slide 42)
- `--dram_bandwidth`, `--dram_access_latency`, `--dram_size` — memory model
- `--double_buffered_l2` — slide 13's ping-pong
- `--enable_mixed_precision` — activates `get_llama_qconfig`
  (`test_codegen.py:117-143`), which **is** the Sphinx scheme: `nf4_6` weights,
  `int6` activations, `bs=64` microscaling blocks, `scale=fp8_e5m3`
- `--outlier_pct` — contribution 2 (outlier filtering / SpMM)
- `--report --report_basename <name>` — the cycle/DRAM estimate

### SHORTCUT: milestone 1 needs no Medusa-specific export code

Voyager does **not** exploit attention-mask sparsity — the mask is a dense
`[1,1,N,max_cache_len]` tensor added to the QKᵀ scores (they even quantize it to
int1 at `test_codegen.py:531-538`), so attention is computed densely either way.
A tree mask and a causal mask of the same shape therefore produce **the same op
set, the same FLOPs, the same DRAM traffic, and the same cycle count.** Only the
values differ.

**Consequence: the cost of an N-node tree-verify step ≈ the cost of an N-token
causal decode-with-cache step.** So the entire N-sweep in §7 can be obtained by
running the stock `llm_decode` path at sequence lengths 1, 4, 8, 16, 32, 64 —
zero new export code. The tree mask and depth-based position IDs only matter for
*functional* equivalence (`--debug` output checks), not for cost modeling.

Do the sweep first, get the curve, then write the real Medusa export branch for
correctness. Verify the sparsity assumption once by diffing reported cycles for
a causal vs tree mask at the same N; if they differ, the assumption is wrong and
the real branch becomes load-bearing.

### Sweep scripts (written 2026-07-29, in this repo)

- `tools/voyager_common.py` — Sphinx defaults (64×64 PE, 1 GHz, 68 GB/s
  ASSUMED bw, 2.5 MB scratchpad, double-buffered L2), the fusion pipeline +
  `get_llama_qconfig` copied from upstream `test_codegen.py` (main), CSV/report
  helpers. `set_qconfig` is imported from the clone's
  `examples/language_modeling` via `--voyager_root`.
- `tools/voyager_milestone1_decode_sweep.py` — N-token decode-step cost sweep
  (defaults: N ∈ {1,2,4,8,16,24,32,40,48,64}, Vicuna-7B, SINGLE decoder layer
  + lm_head for 24 GB-RAM friendliness; `--full_model` for true totals).
  Mirrors upstream `llm_decode` exactly, then calls
  `voyager_compiler.codegen.reporting.report()` → cycles + DRAM bytes per N →
  `decode_sweep.csv`.
- `tools/voyager_milestone3_medusa_heads.py` — MedusaHeadStack exported alone
  at P ∈ {1,64} positions; prices §6(a) directly, and `--head_weight_spec`
  makes the §6(b) 2-bit-heads experiment a one-flag change. Verified
  end-to-end 2026-07-29 with tiny random geometry (export→…→report all work).
- Environment notes: `voyager_compiler` is pip-installed editable from the
  clone (commit 946a222); graphviz `dot` binary installed via brew (compile()
  renders an SVG and hard-fails without it); torch 2.11 moved
  `_annotate_output_qspec` to `torchao.quantization.pt2e.quantizer.utils`.
- **Milestone 1 verified end-to-end 2026-07-29** (real Vicuna, single layer +
  lm_head, N=1): 5,887,604 cycles = 5.888 ms at 1 GHz, 193.6 MB weight
  traffic (consistent with ~333M params at ~0.5 B/param + scales). Two
  non-obvious requirements discovered: (1) `ShapeProp(gm).propagate(...)`
  must run after `convert_pt2e` — it stamps `node.value`, which
  `extract_input_preprocessor` needs; (2) constants folded by `convert_pt2e`
  (e.g. the `index_copy_` KV-write index) carry no memory-space meta and crash
  the new reporting stage — `stamp_unplaced_constants_as_dram()` in
  `voyager_common.py` works around it. **Report the reporting crash to the
  mentor/Jeffrey — likely an untested path on llm_decode.**
- Open question from the smoke run: `dram_kv_bytes` reports 0 — KV-cache
  traffic is either being mis-categorized (folded into activation/weight
  bytes) or not charged. Check before trusting the KV column, especially for
  the KIVI interaction sweep.
- **Full sweep result (2026-07-30, single layer + lm_head, nf4_6,
  `voyager_out/milestone1/decode_sweep.csv`): NO knee in [1,64].** Cycles rise
  linearly at only ~11.5k cycles/token (≈0.2% of baseline per token):
  N=1 → 5.888 ms, N=64 → 6.671 ms = **1.133×**. Weight traffic constant
  (193.6→194.6 MB) as the roofline predicts; the marginal cost matches the
  marginal activation DMA traffic (~0.54 MB/token ÷ 68 GB/s), i.e. the step
  stays memory-bound across the whole range and the added compute hides under
  the weight stream. **This contradicts the §6 sketch's N*≈30 / "64 nodes =
  2.1× = net slowdown" conclusion** — per the compiler, the 63-node tree
  costs 1.13× and yields ~1.9/1.13 ≈ 1.7× speedup even at current weak τ.
  **Probe at N=96/128 (`voyager_out/m1_probe/`) found the knee: N* ≈ 65.**
  N=96 → 9.486 ms (1.61×), N=128 → 12.462 ms (2.12×). The marginal slope
  jumps from ~11.5k cycles/token (below 64: DMA-only, memory-bound) to
  ~90k cycles/token (above: ≈ the 81.3k ideal-compute cycles/token for this
  graph + DMA) — i.e. the cost model and the roofline reconcile exactly, the
  regimes are real, and the two-line fit intersects at **N ≈ 65**. The knee
  sits at ~2× the §6 sketch because the baseline carries ~3 ms of
  N-independent overhead above its 2.9 ms memory floor, which widens the free
  region proportionally (N* ≈ baseline/compute-per-token). In the
  overhead-free limit N* → 2.9 ms/81.3 µs ≈ 36, recovering the sketch's ~30.
  So: **N* is schedule-dependent — between ~36 (ideal overlap) and ~65 (the
  schedule Voyager currently emits); Medusa's stock 63-node tree lands
  almost exactly at the current knee.** Still pending before quoting:
  (a) attribute the ~3 ms overhead via the perfetto trace; (b) confirm
  proportions on `--full_model`.

### Plan

0. **Reproduce the reference unmodified** (~1 day, de-risks everything):
   ```bash
   git clone https://github.com/jeffreyyu0602/voyager-compiler && cd voyager-compiler
   pip install -e . && source setup_shell.sh
   python test/test_codegen.py llm_decode --context_length 512 \
     --pe_array_size 64,64 --frequency 1.0 --enable_mixed_precision \
     --report --report_basename decode_baseline
   ```
   Success = `print_tabular()` output plus `decode_baseline.xlsx` /
   `.perfetto.json`. View the trace at ui.perfetto.dev.
1. Same, with `--model_name_or_path lmsys/vicuna-7b-v1.3`. Record decode
   cycles. **Risk:** Vicuna is Llama-1 (MHA, no GQA); `swap_llama_attention`
   or the RMSNorm→LayerNorm replacement may need mentor help.
2. Add an `llm_medusa_verify` branch: copy `llm_decode`, swap in the tree
   candidates, tree mask, and depth-based position IDs.
3. Export `MedusaHeadStack` separately on `[1, 1, 4096]`. Keeping it separate
   lets you cost 64-position vs 1-position head evaluation (§6).
4. Sweep N ∈ {1, 4, 8, 16, 32, 64}; plot cycles/step vs N; confirm the knee
   near ~30.
5. Get acceptance per tree size from `scripts/medusa_gpu_suite/` (local is
   fine — acceptance is device-independent in greedy mode), then multiply.
   **Deliverable: predicted speedup vs tree size, with the optimum marked.**

---

## 8. Ordered next steps

1. ~~Split `accept_length` from `tokens_per_step` in traces~~ **DONE
   2026-07-31** (runner-side; see gotcha §5.1).
2. Voyager step 0 — reproduce `llm_decode` unmodified. **DONE** (see §7).
3. Stop quoting local absolute timings. `--verbose-timing` is for *relative*
   stage breakdown only.
4. ~~Parameterize tree size~~ **DONE 2026-07-31**: `--tree-size N` (total
   nodes incl. free root) on `run_local_medusa.py` / `run_medusa_mtbench.py`,
   `TREE_SIZE=N` on the GPU-suite wrapper. It truncates the **raw stored
   choice order** (the authors' greedy expected-value selection order), which
   is prefix-closed at every cut — verified for N ∈ {2,4,8,16,32,64}.
   NEVER truncate the `(len, values)`-sorted order `_validate_tree_choices`
   produces: that takes all ten depth-1 siblings first (max accept length 1).
   Best-but-more-work upgrade remains: re-run the authors' greedy node
   selection at each budget from per-head hit-rate calibration.
5. Single-position Medusa-head optimization (§6).
6. Longer MT-Bench runs for reliable τ: **mentor-ready 2026-07-31.**
   `scripts/medusa_gpu_suite/04_run_full_mtbench.sh` = real 80-question
   MT-Bench (auto-downloaded by `00_download_mtbench_questions.sh` to
   `data/mt_bench/question.jsonl`), 512 tokens/turn, Medusa at
   TREE_SIZES {64,32,16,8,4} + greedy baseline, timing/step-text prints off,
   results in `run_logs/gpu_suite_full/`. Needs a ≥24 GB CUDA GPU and
   HF login; no API keys. Judging (03) stays optional/local.

---

## 9. Environment

- Local: macOS / MPS. `transformers` 5.6.2, `torch` 2.11.0 (`requirements.txt`
  floors are much lower: `transformers>=4.45`). The `DynamicCache.layers[i].keys`
  layout that `_copy_selected_tree_cache` relies on is a **recent-transformers**
  detail — if it breaks after an upgrade, that function's `return False`
  guards degrade to re-prefill rather than corrupting state.
- Vicuna/Llama weights are gated: `huggingface-cli login` first.
- Judge needs `OPENROUTER_API_KEY`.
- Everything is committed as of `de66c17` (2026-07-29). **Exception:
  `run_logs/` is gitignored** — the acceptance traces are the measured
  numerator of the speedup equation and have no backup; consider un-ignoring
  them or copying them somewhere durable.
- Batch size 1 throughout. Target and assistant assumed to share a tokenizer.
