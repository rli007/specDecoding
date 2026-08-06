#!/usr/bin/env python
"""Milestone 3: cost of the Medusa heads on Sphinx, at P positions.

The stock `llm_decode` graph (milestone 1) contains only the backbone and
lm_head; a real Medusa step also runs the 5 Medusa heads. This script exports
`MedusaHeadStack` as its own graph on hidden states of shape [1, P, 4096] and
prices it with Voyager. Sweeping P (default 1 and 64) prices the §6(a)
optimization directly: upstream Medusa evaluates the heads at all tree
positions but only ever uses one, so cycles(P=64) - cycles(P=1) is pure waste
recoverable by reordering the step.

The head weights (~739M params, ~370 MB at nf4_6) dominate the head cost in
the memory-bound regime; the report splits weight vs activation DRAM traffic
so you can see that. --head_weight_spec makes the §6(b) experiment (quantize
the speculation path harder than the verification path, e.g. 2-bit heads) a
one-flag change — it alters bytes streamed, never correctness, since a bad
guess only costs acceptance.

Total step cost model:
    cost(N) = milestone1_cycles(N) + head_cycles(P)     [P=1 after §6(a)]

Example:
    python tools/voyager_milestone3_medusa_heads.py \
        --output_dir voyager_out/milestone3
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voyager_compiler import (
    add_compile_args,
    add_quantization_args,
    convert_pt2e,
    export_model,
    get_default_quantizer,
    prepare_pt2e,
    transform,
)
from voyager_compiler import compile as voyager_compile
from voyager_compiler.codegen.reporting import report
from voyager_compiler.hardware_config import AcceleratorConfig

from decoders.medusa_speculative_decoder import (
    MedusaHeadStack,
    _load_state_dict_best_effort,
    _resolve_medusa_head_file,
)
from tools.voyager_common import (
    DEFAULT_VOYAGER_ROOT,
    add_sphinx_defaults,
    build_vector_pipeline,
    load_set_qconfig,
    print_rows,
    schedule_row,
    write_csv,
)

DEFAULT_HEADS = "FasterDecoding/medusa-vicuna-7b-v1.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--positions",
        default="1,64",
        help="Comma-separated P values (positions the heads are evaluated at). 1 = §6(a) optimized, 64 = upstream.",
    )
    parser.add_argument("--heads_path", default=DEFAULT_HEADS)
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Skip downloading the checkpoint; cycles depend on shapes, not values, so this only affects calibration scales.",
    )
    # Vicuna-7B head geometry; override for other targets.
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=1, help="Residual blocks per head.")
    parser.add_argument(
        "--enable_mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize head weights/activations with the Sphinx scheme.",
    )
    parser.add_argument(
        "--head_weight_spec",
        default="nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
        help=(
            "Weight quantization spec for the head Linears ({bs} filled from the PE column count). "
            "The speculation path never needs to be exact, so try e.g. "
            "'uint2,bs={bs},qs=group_wise_affine,ax=-1,scale=fp8_e5m3' for the 2-bit-heads experiment."
        ),
    )
    parser.add_argument("--output_dir", default=str(ROOT / "voyager_out" / "milestone3"))
    parser.add_argument("--csv", default=None, help="CSV path; defaults to <output_dir>/heads_sweep.csv")
    parser.add_argument("--voyager_root", default=str(DEFAULT_VOYAGER_ROOT))
    parser.add_argument("--calibration_runs", type=int, default=2)
    parser.add_argument("--dump_tensors", action="store_true")
    add_quantization_args(parser)
    add_compile_args(parser)
    add_sphinx_defaults(parser)
    return parser.parse_args()


def load_heads(args: argparse.Namespace, dtype: torch.dtype) -> MedusaHeadStack:
    heads = MedusaHeadStack(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    if not args.random_weights:
        checkpoint_path = _resolve_medusa_head_file(args.heads_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = _load_state_dict_best_effort(heads, state_dict)
        if missing or unexpected:
            print(f"non-strict head load: missing={len(missing)} unexpected={len(unexpected)}")
    return heads.to(dtype=dtype).eval()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    positions = sorted({int(part) for part in args.positions.split(",") if part.strip()})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output_dir / "heads_sweep.csv"
    config = AcceleratorConfig.from_args(args)

    param_count = args.num_heads * (
        args.num_layers * (args.hidden_size * args.hidden_size + args.hidden_size)
        + args.hidden_size * args.vocab_size
    )
    print(f"positions: {positions}")
    print(f"head stack: {args.num_heads} heads x {args.num_layers} residual layer(s), {param_count / 1e6:.0f}M params")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    heads = load_heads(args, dtype)
    set_qconfig = load_set_qconfig(args.voyager_root) if args.enable_mixed_precision else None

    rows: list[dict] = []
    for p in positions:
        print(f"\n=== P={p} ===", flush=True)
        # Values only steer calibration scales; cycles depend on shapes alone.
        hidden_states = torch.randn(1, p, args.hidden_size, dtype=dtype)
        example_args = (hidden_states,)

        gm = export_model(heads, example_args)

        quantizer = get_default_quantizer(
            input_activation=args.activation,
            output_activation=args.output_activation,
            weight=args.weight,
            bias=args.bias,
            force_scale_power_of_two=args.force_scale_power_of_two,
        )
        if args.enable_mixed_precision:
            bs = args.pe_array_size[1]
            # Mirror the backbone lm_head entry (int6 activations); the weight
            # spec is a flag so the speculation path can go 2-bit (§6(b)).
            set_qconfig(
                quantizer,
                {
                    torch.nn.Linear: [
                        f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                        args.head_weight_spec.format(bs=bs),
                    ],
                },
            )

        gm = prepare_pt2e(gm, quantizer, example_args)
        for _ in range(args.calibration_runs):
            gm(*example_args)
        convert_pt2e(gm, args.bias)

        transform(
            gm,
            example_args,
            patterns=build_vector_pipeline(),
            config=config,
            transform_layout=getattr(args, "transform_layout", False),
            transpose_fc=getattr(args, "transpose_fc", False),
            fuse_reshape=not getattr(args, "disable_reshape_fusion", False),
            split_spmm=getattr(args, "split_spmm", False),
        )
        voyager_compile(
            gm,
            example_args,
            config=config,
            output_dir=str(output_dir),
            output_file=f"medusa_heads_P{p}",
            dump_tensors=args.dump_tensors,
        )
        result = report(gm, config, output_dir=str(output_dir), basename=f"medusa_heads_P{p}")
        rows.append(schedule_row(f"medusa_heads_P{p}", p, result, config))
        print(f"P={p}: {rows[-1]['cycles']:,} cycles ({rows[-1]['ms']:.3f} ms)")

        del gm
        gc.collect()

    print()
    print_rows(rows)
    write_csv(csv_path, rows)

    if len(rows) >= 2:
        saved = rows[-1]["cycles"] - rows[0]["cycles"]
        print(
            f"\n§6(a) reordering recovers {saved:,} cycles/step "
            f"({saved / (config.frequency * 1e6):.3f} ms) by evaluating heads at "
            f"P={rows[0]['size']} instead of P={rows[-1]['size']}."
        )


if __name__ == "__main__":
    main()
