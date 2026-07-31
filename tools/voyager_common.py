#!/usr/bin/env python
"""Shared plumbing for the Voyager export scripts (milestones 1 and 3).

Everything here either mirrors or is copied from voyager-compiler's
`test/test_codegen.py` (main branch) so that our sweep scripts behave exactly
like the reference flow the mentor pointed at. Copied blocks are marked with
their upstream source; if a Voyager update breaks one of these, diff against
the current `test/test_codegen.py` first.

Requires `pip install -e <voyager clone>` (done 2026-07-29 against origin/main,
commit 946a222).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

from voyager_compiler import OpMatcher
from voyager_compiler.codegen.node_info import is_fully_connected
from voyager_compiler.hardware import AcceleratorConfig

DEFAULT_VOYAGER_ROOT = Path.home() / "Desktop" / "voyager" / "voyager-compiler"

SPHINX_PE_ARRAY = (64, 64)  # 64x64 systolic Matrix Unit
SPHINX_FREQUENCY_GHZ = 1.0
# ASSUMED mobile LPDDR5 number, not a measured one. N* scales as 1/bandwidth,
# so this is the single knob most likely to move the conclusion — confirm the
# real target with the mentor before quoting any N* (NOTES_FOR_MENTOR.md Q1).
SPHINX_DRAM_BANDWIDTH_GBS = 68.0
# Slide 42: 2.5 MB on-chip scratchpad. AcceleratorConfig documents
# scratchpad_size as the PER-BUFFER budget under double buffering, while the
# --cache_size CLI help says "total L2"; the two docstrings disagree. We pass
# the full 2.5 MB — if tiles come out impossibly large in the step-0 sanity
# check, halve this. Worth confirming which semantics the tiler uses.
SPHINX_CACHE_SIZE_BYTES = int(2.5 * 1024 * 1024)


def add_sphinx_defaults(parser) -> None:
    """Override Voyager's generic CLI defaults with the Sphinx configuration.

    Call AFTER add_quantization_args/add_compile_args so the flags exist.
    Everything stays overridable from the command line.
    """
    parser.set_defaults(
        pe_array_size=SPHINX_PE_ARRAY,
        frequency=SPHINX_FREQUENCY_GHZ,
        dram_bandwidth=SPHINX_DRAM_BANDWIDTH_GBS,
        cache_size=SPHINX_CACHE_SIZE_BYTES,
        double_buffered_l2=True,  # slide 13 ping-pong
    )


# ---------------------------------------------------------------------------
# Fusion pipeline — copied verbatim from voyager-compiler test/test_codegen.py
# (main). Describes which op chains may fuse into one pass through the Vector
# Unit's pipeline stages (slide 12).
# ---------------------------------------------------------------------------

def _is_bf16_fc(node):
    # BF16 FC are ran on vector unit and thus cannot be fused
    if hasattr(node, "value") and is_fully_connected(node):
        input_node = node.args[0]
        return input_node.meta.get("dtype") is None
    return False


def _is_spmm(node):
    return node.kwargs.get("A_data") is not None


def _can_fuse(node):
    return not _is_spmm(node) and not _is_bf16_fc(node)


def _is_constant_div(node):
    if node.target != torch.ops.aten.div.Tensor:
        return True

    divisor = node.args[1]
    if isinstance(divisor, torch.fx.Node):
        return divisor.value.numel() == 1

    return True


MXU_OPS = ["conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"]
QUANT_OPS = ["quantize", "quantize_mx", "quantize_mx_outlier"]


def build_vector_pipeline():
    return [
        [
            OpMatcher(*MXU_OPS, predicate=_can_fuse),
            OpMatcher("dequantize"),
            OpMatcher("add", "sub", "mul", "div", predicate=_is_constant_div),
            OpMatcher("exp", "abs", "relu"),
            OpMatcher("add", "mul", "div", predicate=_is_constant_div),
            OpMatcher(*QUANT_OPS, "mul", "div"),
        ],
        [
            OpMatcher(*MXU_OPS, predicate=_can_fuse),
            OpMatcher("dequantize"),
            OpMatcher("gelu", "sigmoid", "silu", "tanh", "hardtanh"),
            OpMatcher(*QUANT_OPS, "mul", "div"),
        ],
        # Fused SpMM operation will use the first stage in the pipeline
        [
            OpMatcher(*MXU_OPS, predicate=_is_spmm),
            OpMatcher("dequantize"),
            OpMatcher("exp", "abs", "relu"),
            OpMatcher("add", "mul", "div"),
            OpMatcher(*QUANT_OPS),
        ],
        [
            OpMatcher("layer_norm", "softmax"),
            OpMatcher(*QUANT_OPS),
        ],
    ]


# ---------------------------------------------------------------------------
# Sphinx mixed-precision scheme — copied from test_codegen.py get_llama_qconfig
# (main). Per-op [input_spec, weight_spec] pairs; see CLAUDE.md §7 for the
# string decoding (nf4_6 = NF4 codebook in INT6, microscaling block 64,
# fp8_e5m3 scales).
# ---------------------------------------------------------------------------

def get_llama_qconfig(bs=64, outlier_pct=None):
    if outlier_pct is None:
        return {
            torch.nn.Linear: [
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            ],
            torch.ops.aten.matmul.default: [
                f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"int6,qs=microscaling,bs={bs},ax=-2,scale=fp8_e5m3",
            ],
            (r"lm_head", torch.ops.aten.linear.default, 0): [
                f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            ],
        }
    else:
        return {
            torch.nn.Linear: [
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3,opct={outlier_pct}",
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            ],
            torch.ops.aten.matmul.default: [
                f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"nf4_6,qs=microscaling,bs={bs},ax=-2,scale=fp8_e5m3,othr=6.0",
            ],
        }


def load_set_qconfig(voyager_root: str | Path):
    """Import set_qconfig from the voyager clone's examples/, like upstream does.

    test_codegen.py appends examples/language_modeling to sys.path and imports
    from there; we do the same against the clone location instead of copying
    the function.
    """
    path = Path(voyager_root).expanduser() / "examples" / "language_modeling"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Voyager clone not found at {path.parent.parent}. "
            "Pass --voyager_root pointing at your voyager-compiler checkout."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from quantization_configs import set_qconfig

    return set_qconfig


# ---------------------------------------------------------------------------
# Report handling
# ---------------------------------------------------------------------------

def stamp_unplaced_constants_as_dram(gm) -> int:
    """Give folded constants a DRAM placement so reporting accepts them.

    `report()` requires every tensor input to root in a memory space, but
    constants created by convert_pt2e's constant folding (e.g. the KV-cache
    write index that `index_copy_` uses) are never seen by plan_memory and so
    carry no `space` meta. Constants live in DRAM in this flow, so stamping
    them DRAM is semantically right and only adds a few bytes of charged
    traffic. Apparent gap in voyager's (new) reporting on the llm_decode
    path — worth reporting upstream. Returns the number of nodes stamped.
    """
    count = 0
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.meta.get("space") is None:
            node.meta["space"] = "DRAM"
            count += 1
    return count

def schedule_row(label: str, size: int, result, config: AcceleratorConfig) -> dict:
    """Flatten a voyager ScheduleResult into one CSV/table row."""
    cycles = int(result.total_latency)
    return {
        "label": label,
        "size": size,
        "cycles": cycles,
        "ms": cycles / (config.frequency * 1e6),
        "dram_read_bytes": int(result.dram_read_bytes),
        "dram_write_bytes": int(result.dram_write_bytes),
        "dram_weight_bytes": int(result.dram_weight_bytes),
        "dram_activation_bytes": int(result.dram_activation_bytes),
        "dram_kv_bytes": int(result.dram_kv_bytes),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def print_rows(rows: list[dict]) -> None:
    if not rows:
        return
    baseline = rows[0]["cycles"]
    header = f"{'size':>5} {'cycles':>14} {'ms':>9} {'vs size[0]':>10} {'weightMB':>9} {'kvMB':>7} {'actMB':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['size']:>5} {row['cycles']:>14,} {row['ms']:>9.3f} "
            f"{row['cycles'] / baseline:>10.3f} "
            f"{row['dram_weight_bytes'] / 1e6:>9.1f} "
            f"{row['dram_kv_bytes'] / 1e6:>7.1f} "
            f"{row['dram_activation_bytes'] / 1e6:>7.1f}"
        )
