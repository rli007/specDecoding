#!/usr/bin/env python
"""Export per-component run counts from existing traces.jsonl files.

Produces the same two CSVs as run_assisted_mtbench.py, so all three methods
(plain decode, draft-model assisted, Medusa) share one schema for the
run-count x per-run-latency cost model:

- <out stem>.components.csv  one row per model-component invocation, in order
- <out stem>.steps.csv       one row per decode step with the component sequence

Trace formats are auto-detected per turn record:

- Medusa   (run_medusa_mtbench.py):   steps carry tree_node_count/prefix_length
- baseline (run_vicuna_mtbench_baseline.py): steps carry input_length/token_id

Medusa component semantics (from decoders/medusa_speculative_decoder.py):

- target_prefill        step 1 backbone+lm_head forward over the prompt
- medusa_heads_prefill  heads over all prompt positions; only the last
                        position's logits seed the first candidate pool
- tree_target_forward   backbone+lm_head over the N tree nodes (the GEMV->GEMM
                        conversion; cost equals an N-token decode step)
- medusa_heads          heads over the N tree hidden states; only the accepted
                        position's logits are consumed (see CLAUDE.md 6a)
- kv_cache_gather       accepted-path KV select-and-compact; gather/scatter DMA,
                        not a matmul
- target_reprefill      full re-prefill fallback when the cache surgery bailed
                        (cache_updated=false on the previous step)

Usage:
  python tools/export_component_log.py run_logs/medusa_mtbench_tree_rerun_answers.traces.jsonl \
      --method medusa-vicuna-7b --out run_logs/medusa_tree_rerun
  python tools/export_component_log.py run_logs/vicuna_mtbench_baseline_rerun_answers.traces.jsonl \
      --method vicuna-7b-baseline --out run_logs/baseline_rerun
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterator

COMPONENT_FIELDS = (
    "method",
    "question_id",
    "category",
    "turn_index",
    "step",
    "event_index",
    "component",
    "input_positions",
    "cache_len_before",
)
STEP_FIELDS = (
    "method",
    "question_id",
    "category",
    "turn_index",
    "step",
    "component_sequence",
    "draft_calls",
    "target_calls",
    "assistant_budget",
    "accepted_assistant_tokens",
    "appended_tokens",
    "output_length",
)


def detect_format(step: dict[str, Any]) -> str:
    if "tree_node_count" in step:
        return "medusa"
    if "input_length" in step and "token_id" in step:
        return "baseline"
    raise ValueError(f"Unrecognized trace step keys: {sorted(step.keys())}")


def medusa_turn_events(record: dict[str, Any]) -> Iterator[tuple[int | None, str, int, int]]:
    """Yield (step, component, input_positions, cache_len_before) for one turn."""
    prompt_length = int(record["stats"]["prompt_token_count"])
    steps = record["trace"]
    yield None, "target_prefill", prompt_length, 0
    yield None, "medusa_heads_prefill", prompt_length, 0

    previous_cache_updated = True
    for step in steps:
        step_number = int(step["step"])
        prefix_length = int(step["prefix_length"])
        tree_nodes = int(step["tree_node_count"]) if step.get("tree_node_count") else 0
        appended = len(step.get("appended_tokens") or [])

        if not previous_cache_updated:
            yield step_number, "target_reprefill", prefix_length, 0
            yield step_number, "medusa_heads_prefill", prefix_length, 0

        if tree_nodes:
            yield step_number, "tree_target_forward", tree_nodes, prefix_length
            yield step_number, "medusa_heads", tree_nodes, prefix_length
        if step.get("cache_updated"):
            yield step_number, "kv_cache_gather", appended, prefix_length
        previous_cache_updated = bool(step.get("cache_updated", True))


def baseline_turn_events(record: dict[str, Any]) -> Iterator[tuple[int | None, str, int, int]]:
    cache = 0
    for step in record["trace"]:
        step_number = int(step["step"])
        positions = int(step["input_length"])
        component = "target_prefill" if step_number == 1 else "target_decode"
        yield step_number, component, positions, cache
        cache += positions


def step_summary_rows(
    method: str,
    record: dict[str, Any],
    events: list[dict[str, Any]],
    fmt: str,
) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event["step"] is not None:
            by_step.setdefault(event["step"], []).append(event)

    step_info = {int(step["step"]): step for step in record["trace"]}
    rows: list[dict[str, Any]] = []
    for step_number in sorted(by_step):
        step_events = by_step[step_number]
        sequence_parts: list[list[Any]] = []
        for event in step_events:
            if sequence_parts and sequence_parts[-1][0] == event["component"]:
                sequence_parts[-1][1] += 1
            else:
                sequence_parts.append([event["component"], 1])
        sequence = ", ".join(name if count == 1 else f"{name} x{count}" for name, count in sequence_parts)

        info = step_info.get(step_number, {})
        if fmt == "medusa":
            appended = len(info.get("appended_tokens") or [])
            output_length = info.get("output_length")
        else:
            appended = 1
            output_length = None
        rows.append(
            {
                "method": method,
                "question_id": record.get("question_id"),
                "category": record.get("category"),
                "turn_index": record.get("turn_index"),
                "step": step_number,
                "component_sequence": sequence,
                "draft_calls": 0,
                "target_calls": sum(1 for event in step_events if event["component"].startswith(("target", "tree"))),
                "assistant_budget": "",
                "accepted_assistant_tokens": "",
                "appended_tokens": appended,
                "output_length": output_length,
            }
        )
    return rows


def export(trace_paths: list[Path], method: str, out_stem: Path) -> None:
    components_path = out_stem.with_name(out_stem.name + ".components.csv")
    steps_path = out_stem.with_name(out_stem.name + ".steps.csv")

    all_component_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {}
    turns = 0

    for trace_path in trace_paths:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not record.get("trace"):
                    continue
                fmt = detect_format(record["trace"][0])
                events_iter = medusa_turn_events(record) if fmt == "medusa" else baseline_turn_events(record)

                turn_events: list[dict[str, Any]] = []
                for event_index, (step, component, positions, cache_len) in enumerate(events_iter):
                    turn_events.append(
                        {
                            "method": method,
                            "question_id": record.get("question_id"),
                            "category": record.get("category"),
                            "turn_index": record.get("turn_index"),
                            "step": step,
                            "event_index": event_index,
                            "component": component,
                            "input_positions": positions,
                            "cache_len_before": cache_len,
                        }
                    )
                    entry = totals.setdefault(component, {"calls": 0, "positions": 0})
                    entry["calls"] += 1
                    entry["positions"] += positions

                all_component_rows.extend(turn_events)
                all_step_rows.extend(step_summary_rows(method, record, turn_events, fmt))
                turns += 1

    for path, fieldnames, rows in (
        (components_path, COMPONENT_FIELDS, all_component_rows),
        (steps_path, STEP_FIELDS, all_step_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"method: {method}")
    print(f"turns: {turns}")
    print(f"component rows: {len(all_component_rows)} -> {components_path}")
    print(f"step rows: {len(all_step_rows)} -> {steps_path}")
    print("component totals (calls, positions):")
    for component, entry in sorted(totals.items()):
        print(f"  {component}: {entry['calls']} calls, {entry['positions']} positions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", help="One or more traces.jsonl files (same method).")
    parser.add_argument("--method", required=True, help="Method label written into every row.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output stem; writes <out>.components.csv and <out>.steps.csv.",
    )
    args = parser.parse_args()
    export([Path(p).expanduser() for p in args.traces], args.method, Path(args.out).expanduser())


if __name__ == "__main__":
    main()
