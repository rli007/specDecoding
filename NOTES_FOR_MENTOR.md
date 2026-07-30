# Speculative decoding on Sphinx — status + one finding

Ryan Li, 2026-07-29

**Implementation.** From-scratch Medusa decoder on `vicuna-7b-v1.3` +
`FasterDecoding/medusa-vicuna-7b-v1.3`, no HF `generate()` so every tensor op is
explicit for mapping. Verified faithful to upstream; greedy output is
character-identical to the plain Vicuna baseline, i.e. lossless. Measured
τ ≈ 0.9 accepted tokens/step — weak number, from 16-token generations, longer
MT-Bench runs in progress. Local wall-clock is unusable (MPS swapping a 14 GB
model); acceptance stats are fine since they're device-independent in greedy mode.

**The finding: our tree should be ~30 nodes, not Medusa's 64.** Weights load once
per step and are reused by all N tree nodes, so `T_memory` is fixed while
`T_compute` scales with N. Tokens are free until they cross:

    N* = (b · TOPS) / (2 · BW) = (0.5 × 8192) / (2 × 68) ≈ 30 nodes

At N=64 we're 2× past that — ~105 ms compute vs ~55 ms memory, so the step costs
2.1× a decode step and returns ~1.9 tokens, i.e. **a net slowdown**. Below ~30
the extra tokens are free, paid for with MACs that sit ~97% idle during decode.
Medusa's 63-path tree was tuned on an A100, whose ridge point is ~120+ tokens.
Tree size is a hardware parameter and the accelerator→tree-size mapping is
unpublished.

**Corollary worth flagging: contributions 1 and 3 eat speculative headroom.**
Quantization cuts `T_memory` and leaves `T_compute` alone; speculation raises
`T_compute` toward `T_memory`. Same slack, so **the speedups don't multiply** —
at fp16, N\* ≈ 120; at nf4_6, ~30. Quantization still wins decisively (~10× vs
~3.5× total), but "4× from quant × 3× from Medusa = 12×" is wrong. 2-bit KV
pushes the same direction. Nobody has published speculative speedup vs weight
precision on a fixed accelerator — Voyager can sweep both axes and read out
cycles.

**Voyager plan.** Medusa's accept/reject is data-dependent so the loop can't be
exported; I export shapes of work and do the loop arithmetic outside. Useful
shortcut: Voyager doesn't exploit mask sparsity (the mask is a dense int1
addend), so a tree mask and a causal mask of equal shape give identical cycles —
**the whole N-sweep runs on the stock `llm_decode` path at seq lengths
1/4/8/16/32/64, no Medusa-specific export code.** I'll verify that once by
diffing causal vs tree at equal N, then sweep and multiply by measured acceptance.

**Questions.**

1. **Is ~68 GB/s the right bandwidth target?** N\* is inversely proportional. At
   34 GB/s it's ~60 and the stock tree is nearly optimal — this one could flip
   the conclusion, so I'd like to pin it before going further.
2. **Does our DMA support runtime-indexed gather on the KV token axis?** The
   accepted-path compaction indexes on a value known only after posterior eval,
   so it can't live in a static schedule. Static fallback (copy all 64, mask the
   dead slots) is ~20× traffic but still ~0.1 ms/step, so affordable either way.
3. **KIVI + variable-length appends.** Medusa appends 1–5 tokens per step, not 1,
   so the per-channel K residual buffer's fill/flush changes and V's per-token
   scales must be gathered with the same index list. Worth resolving early.
4. **Vicuna-7B is Llama-1 (MHA, no GQA)** — will `swap_llama_attention` and the
   RMSNorm→LayerNorm swap handle it, or should I expect to patch?

*All latency numbers are roofline estimates (68 GB/s, ~0.5 B/param, perfect
double-buffering) meant to be replaced by Voyager's scheduled cycles. The shape
of the curve — optimum well below 64 — follows from the roofline alone.*

*Two smaller items, happy to go into detail if useful: upstream evaluates the
Medusa heads at all 64 tree positions but uses one, which is free on an A100 and
~11 ms of wasted compute here (fixing it buys ~25% more free tree nodes); and
the speculation path never needs to be numerically exact — a bad guess costs
acceptance, never correctness — so the heads could run at 2-bit while the
verification path stays at nf4_6.*
