#!/usr/bin/env python
"""Milestone 1: cost of an N-token decode step on Sphinx, swept over N.

Why this prices a Medusa tree step without any Medusa code: Voyager treats the
attention mask as a dense input tensor added to the QK^T scores, so a tree mask
and a causal mask of the same shape produce the same op set, FLOPs, DRAM
traffic, and cycles (CLAUDE.md §7 "SHORTCUT"). An N-node tree-verify step
therefore costs the same as decoding N tokens at once against a warm cache —
which is exactly the stock `llm_decode` shape of work with N tokens in flight.

For each N this script runs the full reference pipeline from voyager-compiler's
test/test_codegen.py (main): export -> graph surgery -> quantize (prepare /
calibrate / convert) -> transform -> compile -> report, and records cycles +
DRAM traffic. N=1 is the plain-decode baseline; the speedup denominator is
cycles(N) / cycles(1).

Defaults are the Sphinx configuration (64x64 PE array, 1 GHz, 64 GB/s ASSUMED
bandwidth, 2.5 MB scratchpad, double-buffered L2, nf4_6 mixed precision) and a
SINGLE decoder layer + lm_head (~1/32 of the model) so it fits in laptop RAM.
Pass --full_model on a big machine for true totals; the knee's position in N
does not depend on layer count.

Example:
    python tools/voyager_milestone1_decode_sweep.py \
        --output_dir voyager_out/milestone1

Expected picture: cycles flat in N while memory-bound, then linear once
compute-bound; roofline predicts the knee near N* ~ 30 at these settings.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_output_qspec as _annotate_output_qspec,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils._pytree import tree_flatten

from voyager_compiler import (
    QuantizationConfig,
    QuantizationSpec,
    ShapeProp,
    TorchExportableModuleWithStaticCache,
    add_compile_args,
    add_quantization_args,
    convert_pt2e,
    export_model,
    extract_input_preprocessor,
    get_default_quantizer,
    prepare_pt2e,
    transform,
)
from voyager_compiler import compile as voyager_compile
from voyager_compiler.codegen import (
    remove_softmax_dtype_cast,
    replace_rmsnorm_with_layer_norm,
)
from voyager_compiler.codegen.reporting import report
from voyager_compiler.hardware_config import AcceleratorConfig

from tools.voyager_common import (
    DEFAULT_VOYAGER_ROOT,
    add_sphinx_defaults,
    build_vector_pipeline,
    get_llama_qconfig,
    load_set_qconfig,
    print_rows,
    schedule_row,
    stamp_unplaced_constants_as_dram,
    write_csv,
)

DEFAULT_NODES = "1,2,4,8,16,24,32,40,48,64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--context_length", type=int, default=512, help="Prompt tokens prefilled into the KV cache.")
    parser.add_argument(
        "--nodes",
        default=DEFAULT_NODES,
        help="Comma-separated N values (tokens per verify step). Dense near the predicted knee ~30.",
    )
    parser.add_argument(
        "--full_model",
        action="store_true",
        help="Compile all decoder layers instead of one. Needs ~16+ GB RAM and much longer compiles.",
    )
    parser.add_argument(
        "--enable_mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sphinx nf4_6/int6 scheme (the b~0.5 bytes/param point). --no-enable_mixed_precision gives the fp16 point.",
    )
    parser.add_argument("--outlier_pct", type=float, default=None, help="Contribution-2 outlier filtering percentage.")
    parser.add_argument(
        "--quantize_attention_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize the additive mask input to int1, as the reference flow does.",
    )
    parser.add_argument("--output_dir", default=str(ROOT / "voyager_out" / "milestone1"))
    parser.add_argument("--csv", default=None, help="CSV path; defaults to <output_dir>/decode_sweep.csv")
    parser.add_argument("--voyager_root", default=str(DEFAULT_VOYAGER_ROOT))
    parser.add_argument("--calibration_runs", type=int, default=2)
    parser.add_argument("--dump_tensors", action="store_true", help="Also dump tensor binaries (slow, big; off by default).")
    add_quantization_args(parser)
    add_compile_args(parser)
    add_sphinx_defaults(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    nodes = sorted({int(part) for part in args.nodes.split(",") if part.strip()})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output_dir / "decode_sweep.csv"
    config = AcceleratorConfig.from_args(args)

    print(f"nodes: {nodes}")
    print(
        f"config: pe={config.pe_array_size} freq={config.frequency}GHz "
        f"bw={config.dram_bandwidth}GB/s scratchpad={config.scratchpad_size}B "
        f"double_buffered_l2={config.double_buffered_l2}"
    )
    print(f"mixed precision: {args.enable_mixed_precision}, single layer: {not args.full_model}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="eager",  # a fused kernel may mangle an explicit 4D mask
    ).eval()

    if not args.full_model:
        # Same trick as the reference llm_kivi branch: keep one decoder layer.
        # Cost scaling to the full model: cycles ~= 32 * layer + lm_head.
        model.model.layers = nn.ModuleList(model.model.layers[:1])
        model.config.num_hidden_layers = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    from datasets import load_dataset

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids = encodings.input_ids[:, : args.context_length]

    # One fixed cache length for every N so the QK^T shape differs only in N.
    # Mirrors the reference: context + 128 headroom, rounded up to the vector
    # lane count.
    block = config.vector_lanes
    raw = args.context_length + 128
    max_cache_len = -(-raw // block) * block
    if max(nodes) > 128:
        raise ValueError("headroom is 128 tokens; raise it before sweeping N > 128")

    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        dtype=model.dtype,
    )
    print(f"prefilling {args.context_length} tokens (max_cache_len={max_cache_len}) ...", flush=True)
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values

    # Real continuation tokens stand in for "speculated" tokens: cycle counts
    # depend only on shapes, but calibration ranges are healthier on real data.
    future_ids = encodings.input_ids[:, args.context_length : args.context_length + max(nodes)]
    pristine_cache = [
        (
            layer.keys.detach().clone(),
            layer.values.detach().clone(),
            layer.cumulative_length.detach().clone(),
        )
        for layer in past_key_values.layers
    ]

    def restore_cache_state() -> None:
        """Reset cache contents AND the write pointer to the post-prefill state.

        transformers' StaticLayer.update ignores the cache_position we pass and
        writes at arange(N) + self.cumulative_length, then advances that pointer
        in place. The pointer is a tensor captured BY REFERENCE into the
        exported graph, so every gm execution (export trace, each calibration
        run, ShapeProp) moves it forward by N — after enough runs the write
        lands past max_cache_len and index_copy_ throws. Resetting the shared
        tensor here rewinds the graph's state too. Upstream test_codegen has
        the same latent drift; it only ever runs N=1, where it never overflows.
        """
        for layer, (keys, values, cumulative) in zip(past_key_values.layers, pristine_cache):
            layer.keys.copy_(keys)
            layer.values.copy_(values)
            layer.cumulative_length.copy_(cumulative)

    hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
    set_qconfig = load_set_qconfig(args.voyager_root) if args.enable_mixed_precision else None

    rows: list[dict] = []
    for n in nodes:
        print(f"\n=== N={n} ===", flush=True)
        restore_cache_state()

        step_ids = future_ids[:, :n]
        inputs_embeds = model.model.embed_tokens(step_ids)
        cache_position = torch.arange(
            args.context_length, args.context_length + n, device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = TorchExportableModuleWithStaticCache._prepare_4d_causal_attention_mask_with_cache_position(
            None,
            sequence_length=n,
            target_length=max_cache_len,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            cache_position=cache_position,
            batch_size=1,
        )
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
        example_args = (inputs_embeds, causal_mask, position_embeddings, cache_position)

        class DecodeStepWrapper(torch.nn.Module):
            """One verify/decode step: decoder layers + lm_head, cache as buffers."""

            def __init__(self) -> None:
                super().__init__()
                self.model = model.model
                self.lm_head = model.lm_head
                self.static_cache = past_key_values
                for i in range(len(self.static_cache.layers)):
                    self.register_buffer(f"key_cache_{i}", self.static_cache.layers[i].keys, persistent=False)
                    self.register_buffer(f"value_cache_{i}", self.static_cache.layers[i].values, persistent=False)

            def forward(self, hidden_states, attention_mask, position_embeddings, cache_position=None):
                for decoder_layer in self.model.layers:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                        past_key_values=self.static_cache,
                        cache_position=cache_position,
                    )
                    hidden_states = layer_outputs[0]
                logits = self.lm_head(hidden_states)
                return logits

        gm = export_model(DecodeStepWrapper(), example_args)
        remove_softmax_dtype_cast(gm)
        replace_rmsnorm_with_layer_norm(
            gm,
            model.model.layers[0].input_layernorm,
            (torch.randn(1, n, hidden_size, dtype=model.dtype),),
        )

        quantizer = get_default_quantizer(
            input_activation=args.activation,
            output_activation=args.output_activation,
            weight=args.weight,
            bias=args.bias,
            force_scale_power_of_two=args.force_scale_power_of_two,
        )
        if args.enable_mixed_precision:
            set_qconfig(quantizer, get_llama_qconfig(args.pe_array_size[1], args.outlier_pct))
            fp8_qspec = QuantizationSpec.from_str("fp8_e4m3,qs=per_tensor_symmetric,qmax=240")
            fp8_qconfig = QuantizationConfig(fp8_qspec, None, None, None)
            quantizer.set_object_type(torch.ops.aten.softmax.int, fp8_qconfig)
            quantizer.set_object_type(torch.ops.aten.layer_norm.default, fp8_qconfig)

        if args.quantize_attention_mask:
            mask_qspec = QuantizationSpec.from_str("int1,qs=per_tensor_symmetric,qmax=1")
            mask_node = next(iter(node for node in gm.graph.nodes if node.target == "attention_mask"))
            _annotate_output_qspec(mask_node, mask_qspec)

        gm = prepare_pt2e(gm, quantizer, example_args)
        for _ in range(args.calibration_runs):
            restore_cache_state()  # each run advances the shared write pointer
            gm(*example_args)
        convert_pt2e(gm, args.bias)

        # Upstream runs ShapeProp here; besides computing a reference output it
        # stamps every node with `.value`, which extract_input_preprocessor
        # (and the outlier-aware transform path) require.
        restore_cache_state()  # ShapeProp executes the graph too
        flatten_args, _ = tree_flatten(example_args)
        ShapeProp(gm).propagate(*flatten_args)

        if args.quantize_attention_mask:
            gm, mask_preprocess_fn = extract_input_preprocessor(gm, "attention_mask")
            embeds, mask, pos, cache_pos = example_args
            example_args = (embeds, mask_preprocess_fn(mask), pos, cache_pos)

        transform(
            gm,
            example_args,
            patterns=build_vector_pipeline(),
            config=config,
            transform_layout=getattr(args, "transform_layout", False),
            transpose_fc=getattr(args, "transpose_fc", False),
            fuse_reshape=not getattr(args, "disable_reshape_fusion", False),
            split_spmm=getattr(args, "split_spmm", False),
            context_len=args.context_length,
            max_gen=128,
        )
        voyager_compile(
            gm,
            example_args,
            config=config,
            output_dir=str(output_dir),
            output_file=f"decode_N{n}",
            dump_tensors=args.dump_tensors,
        )
        stamped = stamp_unplaced_constants_as_dram(gm)
        if stamped:
            print(f"stamped {stamped} folded constant(s) as DRAM for reporting")
        result = report(gm, config, output_dir=str(output_dir), basename=f"decode_N{n}")
        rows.append(schedule_row(f"decode_N{n}", n, result, config))
        ratio = rows[-1]["cycles"] / rows[0]["cycles"]
        print(f"N={n}: {rows[-1]['cycles']:,} cycles ({rows[-1]['ms']:.3f} ms), {ratio:.3f}x vs N={nodes[0]}")
        write_csv(csv_path, rows)  # incrementally, so an interrupted sweep keeps its finished rows

        del gm
        gc.collect()

    print()
    print_rows(rows)
    write_csv(csv_path, rows)

    # Free-token estimate: the largest swept N whose step still costs within 5%
    # of the baseline step. Refine visually from the CSV/perfetto traces.
    baseline = rows[0]["cycles"]
    free = [row["size"] for row in rows if row["cycles"] <= 1.05 * baseline]
    if free:
        print(f"\nlargest ~free tree size in this sweep (cycles within 5% of N={nodes[0]}): N={max(free)}")


if __name__ == "__main__":
    main()
