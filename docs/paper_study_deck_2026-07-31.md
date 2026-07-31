# Paper Study deck — SD on silicon (ISSCC 2026)

Bullets ready to paste into the Stanford template. One idea per slide,
same style as the July SD-overview deck. `>` lines are speaker notes, not
slide content.

---

## Slide 1 — Title

**Paper Study: Speculative Decoding Reaches Silicon**
ISSCC 2026 31.1 (HKUST) & 31.8 (Tsinghua)
Ryan Li

---

## Slide 2 — Why these two papers

- First **fabricated** speculative-decoding LLM chips — both Feb 2026
- Session 31 (AI accelerators) opens and closes with SD
- Both use draft-model SD; both cite Medusa, neither builds it
- They frame exactly the design space our project sits in

> Note: adjacent prior art is simulation-only (SpecPIM, ASPLOS'24).

---

## Slide 3 — 31.1 (HKUST): overview

- Draft-model SD accelerator, 55nm, with **ReRAM dies stacked on logic**
  (8MB @ 25.6 GB/s private bandwidth for the draft model)
- W4A8 target, INT4 codebook-compressed draft
- 14–136 tok/s; 4.5–7.2× vs their own BF16 SD baseline

---

## Slide 4 — 31.1: their three problems

- Target hard to quantize to 4b → activation outliers
- Draft model doesn't fit on chip → drafting burns DRAM bandwidth
- Long drafts: >90% of draft tokens rejected

> Note: problems 2+3 are our roofline argument, measured in silicon —
> a draft model's sequential decode is itself memory-bound.

---

## Slide 5 — 31.1: their three solutions

- **LRU**: Hadamard rotation smears outliers (orthogonal → math unchanged);
  local/shallow approximation saves 92.7% area vs global rotation
- **RS-PNM + BVQ**: draft weights → learned codebooks → stacked ReRAM;
  drafting never touches external DRAM
- **APSD**: draft length adapts to acceptance feedback; overlaps
  draft & verify

---

## Slide 6 — 31.1: takeaway + caveats

- Takeaway: they spent a **packaging technology** to make the draft
  model affordable
- Caveats: speedup vs own baseline (bundles quant + SD gains);
  55nm; some comparison rows modeled, not measured;
  draft capped at ~300M params by ReRAM capacity → weak acceptance

---

## Slide 7 — 31.8 (Tsinghua): overview

- 28nm, 56.8 mm², 4MB SRAM SD processor; FP16 target / FP8 draft
- 105–685 µs/token; ~10× vs own autoregressive baseline
- Measured the dual waste: drafting idles 76.7% of MACs,
  verification idles 49.7% of bandwidth

> Note: that dual-waste slide is our roofline with silicon numbers.

---

## Slide 8 — 31.8: their three features

- **EDRM**: duplicate tokens in the draft tree share FP exponents →
  reuse exponent logic, −30% MAC energy
- **DBTM**: backprop on the *draft* → gradients rank target weights →
  FP16 / FP8 / **prune** per head during verification
- **DEPC**: accept/reject verdict stabilizes by ~layer 6 →
  start next draft early, overlap with verification

---

## Slide 9 — 31.8: caveats

- **DBTM makes the verifier lossy** — SD guarantee gone
  (they report −0.2 to −0.5% accuracy)
- 10× headline = stacked product of all features vs own baseline
- Needs a distilled draft + on-chip backprop hardware

---

## Slide 10 — Patterns across both chips

- Same roofline, three answers: **hide** the waste (overlap),
  **house** the draft (stacking), or — our route — **remove** the
  draft phase (self-drafting) and size the tree at the knee
- Architecture = static kernels + small CPU controller
  (same decomposition as our Voyager plan)
- Adaptivity is the trend: draft length / precision react to
  runtime feedback

---

## Slide 11 — What this means for our project

- Validation: SD on edge silicon is the current frontier
- Contrast: Medusa has **no draft residency problem** and keeps the
  verifier **exact** (lossy speculator, never lossy verifier)
- Open niche: no fabricated self-drafting SD; no published
  accelerator-parameters → tree-size mapping (our Voyager result)

---

## Slide 12 — What we borrow / what we don't

- Borrow: **adaptive tree size** (APSD idea, bounded by our measured
  knee); **precision-by-role** (2-bit heads, DBTM's lossless cousin)
- Maybe: block-Hadamard rotation as a quantization experiment
  (64-wide rotation = one extra Matrix Unit matmul, no new hardware)
- Skip: EDRM (FP-specific; our datapath is integer), DEPC (no draft
  phase to hide), DBTM (breaks losslessness)
