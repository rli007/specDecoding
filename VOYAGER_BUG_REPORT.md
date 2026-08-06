# voyager-compiler bug report — specDecoding integration (2026-08-06)

Found while re-verifying the Vicuna-7B decode-step sweeps and attempting to
compile a llama-160m draft model against voyager-compiler @ `900be4f`
(origin/main, 2026-08-05).

**Environment:** macOS 15.7.3 arm64, Python 3.13.9, torch 2.12.1,
transformers 5.13.0, voyager-compiler `900be4f` installed editable.

---

## Issue 1 — multi-layer LLM decode export fails (blocks `--full_model` and any draft-model compile)

**Severity: high** — single-layer compiles are unaffected, which is why the
reference flow (`--compile_single_layer`) never sees it.

### Summary

Exporting a decode-step wrapper (precomputed rotary embeddings + prefilled
`StaticCache` held as a module attribute + a python loop over decoder layers —
the same wrapper shape as `test_codegen.py`'s `llm_decode`) succeeds with
**one** decoder layer and fails with **two or more**, during torch.export
fake-tensor tracing:

```
RuntimeError: Attempting to broadcast a dimension of length 128 at -1!
Mismatching argument at index 1 had torch.Size([1, 1, 1, 128]); but expected
shape should be broadcastable to [1, 1, 128, 2]
```

The failing op is the `q * cos` multiply inside `apply_rotary_pos_emb`: from
the second layer onward the traced query reaches RoPE laid out as
`[b, 1, head_dim, n_heads]` instead of `[b, n_heads, seq, head_dim]`.

### Reproduction

```
python tools/repro_voyager_multilayer_export_bug.py
```

(self-contained in the specDecoding repo; builds a tiny random Llama
in-process, no downloads). Output on the environment above:

```
1 layer(s) | plain torch.export (strict=False): OK
1 layer(s) | voyager export_model: OK
2 layer(s) | plain torch.export (strict=False): FAILS -> broadcast error above
2 layer(s) | voyager export_model: FAILS -> broadcast error above
```

### Ruled out by bisection

| hypothesis | test | verdict |
|---|---|---|
| old/quirky checkpoint | fresh random-weight models | still fails |
| head geometry | 12×64, 6×128, 2×128 all tried | still fails |
| attention mask contents | vendored −inf mask vs plain zeros | still fails |
| voyager's `export_model` wrapper | plain `torch.export(strict=False)` | **fails identically** |
| layer count | 1 vs 2 vs 3 layers, same everything | **1 OK, ≥2 fails** |

### Notes

- Root cause is therefore in the torch.export × transformers interaction (we
  suspect functionalization of the shared mutable `StaticCache` across
  successive layer iterations), not in voyager code — but it breaks the
  voyager `llm_decode` wrapper pattern specifically, so a fix or workaround
  probably belongs at the wrapper level.
- Possible direction: transformers' own `TorchExportableModuleWithStaticCache`
  (executorch path) exports full multi-layer models routinely; its cache/rope
  threading differs from the llm_decode wrapper. Untested whether adopting its
  pattern avoids the failure.
- Impact on our side: cannot compile any full multi-layer model — in
  particular the 160M draft model needed to cost draft-model speculative
  decoding (currently roofline-bounded instead), and `--full_model` sweeps.

---

## Issue 2 — reporting crashes on constants folded by `convert_pt2e` (workaround in use)

On the `llm_decode`-shaped graph, constants created by convert_pt2e's constant
folding (e.g. the KV-cache write index consumed by `index_copy_`) are never
seen by `plan_memory` and carry no memory-space meta. The reporting stage then
raises when it roots every tensor operand in a memory space.

At N=1 on Vicuna-7B (single layer + lm_head) this affects **232** `get_attr`
nodes. Present since at least `946a222`; unchanged at `900be4f`.

**Workaround we run:** stamp every space-less `get_attr` node as DRAM before
`report()` (`stamp_unplaced_constants_as_dram` in `tools/voyager_common.py`).
Semantically right for this flow and only adds a few bytes of charged traffic.
Suggested fix: default unplaced `get_attr` tensors to DRAM inside reporting,
or have `plan_memory` visit folded constants.

---

## Issue 3 — `dram_kv_bytes` reports 0 on decode graphs

Every decode-step report we generate (N ∈ {1…128}, KV cache present, read by
QKᵀ/PV and written via `index_copy_`) shows `dram_kv_bytes = 0`, with weight
and activation bytes plausible. Either KV traffic is being folded into the
weight/activation categories (mis-labeling only) or it is not charged at all
(cost error). Which is it? This matters for us because the KIVI-interaction
analysis needs the KV column trustworthy.

---

*Contact: Ryan (specDecoding repo). The repro script and workarounds live in
`tools/` there; happy to run candidate fixes against the full sweep.*
