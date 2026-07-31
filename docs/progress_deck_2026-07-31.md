# Research Progress deck — 2026-07-31

Draft outline in the style of the July update deck (plain bullets, one idea per
slide, Stanford Engineering footer). Paste into Google Slides; speaker notes
under each slide are for you, not the slide.

---

## Slide 1 — Title

**Research Progress**
Ryan Li

---

## Slide 2 — Recap: where I left off

- Last update: greedy + speculative sampling implemented, matched HF
  assisted-decoding 1-to-1; starting on tree methods
- Since then: full from-scratch **Medusa** decoder on Vicuna-7B with the
  public pretrained heads
- New this month: **Voyager cost model is running** — first measured
  cycle numbers for speculative decoding on the Sphinx configuration

> Notes: one sentence each; the rest of the deck is the Medusa + Voyager story.

---

## Slide 3 — Medusa implementation: faithful and lossless

- No HF `generate()` — every tensor op explicit, for mapping to the chip
- Verified mechanism-by-mechanism against upstream FasterDecoding/Medusa
  (tree indices, tree attention mask, posterior acceptance, KV compaction)
- Greedy Medusa output is **character-identical** to the plain Vicuna
  baseline → lossless by construction, so speedup is apples-to-apples
- Per-step traces logged: acceptance, tree size, cache state, stage timings

> Notes: if asked about typical/sampling acceptance: implemented, but
> temperature>0 breaks losslessness, so all speedup claims are greedy.

---

## Slide 4 — Why Medusa fits Sphinx (the roofline argument)

- Decode streams all ~3.4 GB of weights for ONE token: ~50 ms of memory
  vs ~1.6 ms of compute → Matrix Unit ~3% utilized
- A tree of N candidates shares one weight load → verification is a GEMM,
  like prefill
- Free tree size: **N\* ≈ b·TOPS / (2·BW)** — model size cancels
  - fp16 ≈ 120 nodes, int8 ≈ 60, **nf4_6 ≈ 30** (at assumed 68 GB/s)
- Tree size is a *hardware parameter*: Medusa's 63-node tree was tuned
  on an A100 — the accelerator→tree-size mapping is unpublished

> Notes: emphasize P cancels — free tree size independent of model size.
> Flag: 68 GB/s is an assumption; N* ∝ 1/BW. Ask for the real target.

---

## Slide 5 — Costing it for real: Voyager pipeline is working

- Insight that makes it cheap: Voyager treats the attention mask as a dense
  input tensor → an N-node **tree step costs the same as an N-token causal
  decode step** — the whole sweep runs on the stock `llm_decode` path
- Pipeline verified end-to-end: export → quantize (nf4_6 mixed precision)
  → transform → compile → cycle/DRAM report
- Setup: Vicuna-7B, single decoder layer + lm_head (fits laptop RAM;
  knee position in N is unchanged), Sphinx config: 64×64 PE, 1 GHz,
  68 GB/s, 2.5 MB scratchpad, double-buffered L2
- Found + worked around a bug in Voyager's new reporting stage on the
  decode path (folded constants lack a memory-space label) — will file

> Notes: also fixed a transformers cache-pointer bug in my harness; both
> gotchas documented. Sweep = ~30 s per N once the model is loaded.

---

## Slide 6 — Result: the measured cost curve (the knee is real)

| N (tree nodes) | ms/step | vs N=1 |
|---|---|---|
| 1 | 5.89 | 1.00 |
| 16 | 6.12 | 1.04 |
| 32 | 6.30 | 1.07 |
| 48 | 6.49 | 1.10 |
| **64** | **6.67** | **1.13** |
| 96 | 9.49 | 1.61 |
| 128 | 12.46 | 2.12 |

- Below the knee: slope ≈ marginal DMA traffic only (compute hides under
  the weight stream). Above: slope ≈ full ideal compute per token
- Weight traffic constant (193.6 → 194.6 MB from N=1→64) — the
  shared-weight-load thesis, confirmed by the compiler

> Notes: plot this as the money chart (cycles vs N, knee marked at ~65).

---

## Slide 7 — Reading the knee: N\* is schedule-dependent

- Measured knee ≈ **65**; hand-roofline predicted ~30
- Reconciliation: the baseline step carries ~3 ms of N-independent
  scheduling overhead above its 2.9 ms memory floor — overhead widens the
  free region (N\* ≈ baseline / compute-per-token)
- Overhead-free limit: N\* → ~36, recovering the sketch
- So: **N\* ∈ [~36 ideal schedule, ~65 current schedule]** — and Medusa's
  stock 63-node tree lands almost exactly on the current knee
- Speedup model is now simply: **speedup(N) ≈ tokens_per_step(N) / 1.13**
  (for N ≤ 64)

> Notes: caveats to state proactively: single-layer proportions (lm_head
> over-weighted), KV traffic mis-categorized in the report (bytes counted,
> label wrong), overhead attribution via perfetto still to do, full-model
> run pending on a bigger machine.

---

## Slide 8 — Consequence: quantization and speculation share one budget

- Both spend the same slack: the gap between memory and compute rooflines
- Quantization shrinks T_memory; speculation grows T_compute into it
  → the speedups **do not multiply**; speculative factor shrinks as
  precision drops (fp16: N\*~120 → nf4_6: N\*~30-65)
- 2-bit KV (contribution 3) pushes the same direction
- Nobody has published speculative speedup vs weight precision on a fixed
  accelerator — Voyager sweeps both axes → clean 2-D result

> Notes: this is the paper-shaped claim; the sweep infrastructure for it
> now exists (mixed precision is one flag).

---

## Slide 9 — Acceptance: now the bottleneck (and the honest number)

- Current measured τ ≈ 0.9 (1.9 tokens/step) — **known-unreliable**:
  16-token generations, first step biases the average
- Medusa paper reports ~2.5 tokens/step for these same heads → first fix
  is measuring properly (≥128 tokens, real 80-question MT-Bench)
- Levers, in order: full 63-node tree (now known ~free), longer runs,
  EAGLE drafter (repo shell exists; public weights; τ ≈ 3.5-4),
  Medusa-2-style head training (needs compute)
- Even pessimistically: τ=0.9 → 1.7× lossless; τ=2.5 → 3.1×

---

## Slide 10 — Context: SD reached silicon at ISSCC 2026

- **31.1 (HKUST)**: draft-model SD + stacked ReRAM to make the draft
  model affordable; **31.8 (Tsinghua)**: draft-model SD + lossy verifier
  (prunes target weights per step) + draft/verify overlap
- Both measure our roofline: drafting idles MACs, oversized verification
  idles bandwidth
- Contrast: Medusa **dissolves** the draft bottleneck (no second model,
  drafting folded into the verify step) and keeps the verifier **exact**
- Open niche: no silicon implements self-drafting SD, and neither paper
  publishes the accelerator-parameters → tree-size mapping

---

## Slide 11 — Questions for the group

- What is the real DRAM bandwidth target? (N\* ∝ 1/BW — could move the
  knee 2×)
- Does the DMA support runtime-indexed gather on the KV token axis?
  (accepted-path compaction; static fallback is ~20× traffic but affordable)
- KIVI + variable-length appends (1–5 tokens/step): residual buffer
  bookkeeping — who owns this interaction?
- Voyager: reporting crash on llm_decode (repro + suggested fix ready) and
  ~3 ms/step schedule overhead — worth a look with Jeffrey?

---

## Slide 12 — Next steps

- τ(N) on full MT-Bench, ≥128 tokens, trees truncated in the authors'
  greedy order → combine with cost curve → **speedup-vs-N plot with
  optimum marked** (the deliverable)
- Full-model sweep on a lab machine (`--full_model` is already a flag)
- Adaptive tree size: bound N by the measured knee, adapt within it by
  acceptance feedback (APSD idea, transplanted) — simulate from traces
- 2-bit Medusa heads (speculation path never needs exactness — costs
  acceptance, never correctness)
