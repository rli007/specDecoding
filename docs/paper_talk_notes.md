# paper talk notes — isscc 2026 sd chips (31.1 + 31.8)

project the paper, talk from these. one section per figure.

## setup (say first)

- both feb 2026, session 31. first fabricated sd chips
- both draft-model sd. both cite medusa [9], neither builds it
- why we care: same design space as sphinx spec-decoding work

---

## 31.1 hkust — reram-on-logic stacked sd accelerator

### headline

- 55nm logic + 4 stacked reram dies. 2048 f2f bumps
- reram = 8MB, 25.6 GB/s private bw @ 100MHz
- w4a8 target, int4 codebook draft
- 14–136 tok/s. 4.46–7.17x vs their own bf16 sd baseline

### fig 31.1.1 — challenges

- sd recap: dlm drafts, tlm verifies tree in parallel
- latency still weight-EMA dominated. >60% from tlm
- c1: tlm at 4b → activation outliers kill accuracy
- c2: dlm doesn't fit on chip → drafting burns external bw
- c3: long draft len → >90% draft tokens rejected
- c2 + c3 = our roofline argument, measured in silicon
  - draft decode is itself memory bound. sequential. can't hide

### fig 31.1.3 — LRU (fixes c1)

- rotation quantization, quarot family
- H orthogonal: HHt = I → (xH)(HtW) = xW. math unchanged
- HtW folded into weights offline. only xH at runtime
- hadamard entries ±1 → fwht butterfly. adds/subs only, zero mults
- effect: outlier energy smeared across channels → ranges smooth → w4a8 ok
- w4a8 vs bf16 = 4x less tlm weight traffic → 3.82–3.93x speedup
- problem: global fwht on 14336 dim = deep cascade. 4.37x area of 4k-MAC array
- their fix: 2 overlapped shallow local rotations, depth ≤6 (64-pt)
  - + small precomputed non-pow2 hadamard in accumulator. ±1 → mac-free
  - 92.7% area saved. local ≈ global, outliers spread within window only
- sphinx note: 64-wide local rotation = 64x64 matmul = our MU tile size.
  could test as software, no new hw. competes w/ contribution 2 (spmm filter)

### fig 31.1.4 — RS-PNM + BVQ (fixes c2)

- raw dlm too big: 160M @ int4 = 80MB >> 8MB reram
- bvq = block vector quantization. don't store weights, store:
  - codebook of representative weight blocks (learned, int4 qat)
  - per-block index (learned via gumbel softmax)
- codebooks + indices fit reram. 4-chip = 32MB holds everything
- weight "load" = fetch codebook entry over 25.6 GB/s stacked interface
- tile fusion: tokens sharing same CB entry fused → each entry fetched once,
  halves read latency
- net: draft EMA → zero. external dram reserved for tlm
- +1.1–1.46x over w4a8 sd
- vs traditional vq: no big index buffers / multi-port decoders
- cost: qat training required. packaging tech we don't have

### fig 31.1.5 — APSD + WDOS (fixes c3)

- prior art PEARL [14]: parallel draft-and-verify, but INTER-chip
  - draft on chip A, verify on chip B. avoids resource fights
  - problem 1: verify slow → draft chip idles most of the time
  - problem 2: each chip's memory interface serves only its own workload.
    reram bw AND dram bw both underutilized. can't share across chips
- apsd = same idea INTRA-chip. both workloads on one chip
  - works because workloads are complementary:
    draft eats reram bw + little compute. verify eats MACs + dram bw
  - each soaks the other's idle resource
  - needs a referee → WDOS: 4 decoupled instruction queues
    (transceiver / compute / reram load / emac), out-of-order,
    dependency counters. hw scoreboard, not magic
- adaptive part:
  - start conservative: short draft len, sequential
  - streak of full accepts + newest tlm token == first token of the
    concurrent draft → go parallel (draft round k+1 while verifying k)
  - mismatch → discard speculative draft, fall back to short sequential
  - note: parallel drafting is itself a speculation on the prefix
- +1.1–1.29x, 10–14% fewer rejected tokens
- sphinx note: adaptive-size idea transfers to medusa tree (N adapts
  under our measured knee). overlap part doesn't — no draft phase to hide

### caveats (say before moving on)

- speedup vs own baseline. bundles quant gain + sd gain
- 55nm. architecture transfers, absolute numbers don't
- some comparison-table rows modeled (lpddr3 added in sim), not measured
- reram capacity caps dlm at ~300M → weak acceptance pairs. llama3 had
  no small sibling, they trained their own 296M

---

## 31.8 tsinghua — 28nm sd processor

### headline

- 28nm, 56.8mm2, 4MB sram. fp16/bf16 target, fp8 draft
- 105–685 us/token avg. ~10x vs own autoregressive baseline (stacked product)
- measured dual waste:
  - draft decode memory bound → 76.7% MACs idle
  - tree verify compute bound → 49.7% dram bw idle
  - = our roofline slide with silicon numbers

### fig 31.8.3 — EDRM

- observation: 39.5% of draft-tree tokens are duplicates (same word,
  different branch). after embed+attn → near-identical vectors,
  95.7% same fp exponent
- fp mac = mantissa math + exponent math. exponent
  compare/align/normalize = ~70% of accumulator power
- duplicate exponents → do exponent work once, share across PEs.
  mantissa only per-vector
- −30.1% mac energy
- sphinx: doesn't transfer. our MU is integer codebook, block-shared
  fp8 scales. no per-element exponent machinery to dedup

### fig 31.8.4 — DBTM (the spicy one)

- after drafting: run BACKPROP on the draft model, get gradients of
  weight heads + kv
- 86.7% of gradients ~0 → those heads/entries barely matter for verify
- draft distilled from target → assume importance transfers
- per step: top 20% heads fp16, next 20% fp8, rest PRUNED
- −74.9% verify cost, 2.2–2.5x throughput
- BUT: verifier no longer runs the deployed model → sd guarantee gone
  - lossy verifier. −0.2 to −0.5% accuracy reported
- sphinx contrast: our precision-by-role is the mirror image.
  lossy SPECULATOR (2b heads, costs acceptance only), verifier exact.
  guarantee kept

### fig 31.8.5 — DEPC

- observation: accept/reject verdict stabilizes early.
  89.6% of cases: layer 6 onward agrees with final layer
  (cosine sim of hidden states)
- 4 consecutive layers agree → start next draft NOW, parallel with
  rest of verification
- memory-bound draft uses bw that compute-bound verify wastes
- final layer disagrees (<5%) → discard, redraft. restart = lossless
- +1.49–1.89x
- sphinx: no separate draft phase → nothing to overlap. absence = feature

### caveats

- dbtm breaks losslessness (say it plainly)
- 10x = product of all features vs own baseline
- needs distilled draft + on-chip gradient/backprop hw

---

## numbers cheat sheet (context for q&a)

### anchors

- human reading ≈ 5–10 tok/s. >20 tok/s feels instant
- memory-bound decode: tok/s ≈ dram bw / model bytes
  - set by memory interface, NOT transistor count
  - why 55nm 31.1 posts numbers in same band as 7nm sphinx.
    node buys cheap MACs, not bytes

### 31.1 numbers in context

- 14–136 tok/s = usable interactive band for edge. wide range =
  different models / seq lengths
- headline 4.46–7.17x decomposes into their own factors:
  ~3.9x (lru/w4a8 traffic cut) × 1.1–1.46x (bvq) × 1.1–1.29x (apsd)
  - most of headline = QUANTIZATION, not sd
  - sd-attributable ≈ 1.2–1.9x. same band as our τ=0.9 number
- draft capped ~300M by reram capacity → weak acceptance. structural

### 31.8 numbers in context

- 105–685 us/token = 1.5k–9.5k tok/s equiv. suspicious at llm scale
  - fp16 7B step = ~13.6GB traffic → us-scale impossible off-chip
  - implies small eval models, weights maybe resident in 4MB sram
  - CHECK their model table before quoting next to 31.1
- trust the relative numbers: dual waste %, dbtm 2.2–2.5x (lossy),
  depc 1.49–1.89x
- 10x headline = stacked product of everything vs own AR baseline

### us

- baseline: 3.4GB / 68GB/s = 50ms/tok = 20 tok/s (7B nf4_6)
- medusa @ measured 1.13x step cost:
  - τ=0.9 (current, unreliable) → ~34 tok/s = 1.7x
  - τ=2.5 (paper's number for these heads) → ~62 tok/s = 3.1x
- ours = sd-only, lossless, on already-quantized baseline
- if quoted isscc-style (vs fp16 no-sd): ~4x quant × ~2.5x sd ≈ "10x"
  - we reject that multiplication — factors share roofline slack
  - their headline math is the thing our measurement corrects (say nicely)

### roofline (if asked what it means)

- two independent speed limits: compute 8.192 TOPS, memory ~68 GB/s
- step time = max(ops/TOPS, bytes/BW) w/ double buffering overlap
- intensity I = ops per byte fetched. decides which limit binds
- plot throughput vs I: rising slope (bw-limited) + flat ceiling (TOPS)
  = looks like a roof. corner = ridge point, I = TOPS/BW ≈ 120 op/B sphinx
- decode: 2 ops per param, b bytes per param → I = 2/b = 4 @ nf4_6
  - 30x left of ridge → MU 3% utilized
- tree of N: same bytes, Nx ops → I = 2N/b. free until I hits ridge
  - N* = b·TOPS/2BW ≈ 30. our measured knee = ridge point in time domain
- 31.8 dual waste = both sides of ridge measured in silicon:
  draft left of ridge (76.7% MACs idle), verify right (49.7% bw idle)

---

## wrap — our angle (close with this)

- same roofline, three answers:
  - 31.8 hides the waste (overlap)
  - 31.1 houses the draft (stacking)
  - us: remove the draft phase (self-drafting) + size tree at the knee
- both chips: static kernels + small cpu controller = our voyager
  decomposition exactly
- open niche: no fabricated self-drafting sd. no published
  accel-params → tree-size mapping. that's our lane
- borrow: adaptive N (bounded by measured knee ~65), precision-by-role
- skip: edrm (int datapath), depc (no draft phase), dbtm (lossless or bust)
