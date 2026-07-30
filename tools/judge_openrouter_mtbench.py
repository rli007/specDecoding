#!/usr/bin/env python
"""Judge MT-Bench-style answer JSONL files with OpenRouter chat completions.

This is intentionally separate from generation. You can run Medusa/baseline
locally on GPUs, then call a stronger remote model such as GPT-4o only for
answer scoring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_medusa_mtbench import DEFAULT_QUESTION_FILE, Question, append_jsonl, read_questions, truncate_file


DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated MT-Bench-style answers with OpenRouter.")
    parser.add_argument("--question-file", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--answers-jsonl", required=True, help="Answer JSONL produced by run_medusa_mtbench.py or baseline.")
    parser.add_argument(
        "--judgments-jsonl",
        default=None,
        help="Defaults to the answer path with .judgments.jsonl appended before the extension.",
    )
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="OpenRouter model slug, for example openai/gpt-4o.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--site-url", default=os.environ.get("OPENROUTER_SITE_URL", "https://github.com/rli007/specDecoding"))
    parser.add_argument("--site-title", default=os.environ.get("OPENROUTER_SITE_TITLE", "specDecoding Medusa Judge"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-judge-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Pause between judge requests.")
    parser.add_argument("--limit", type=int, default=0, help="Limit judged turns. Use 0 for all.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Clear the output judgments file before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Print judge prompts without calling OpenRouter.")
    return parser.parse_args()


def default_judgment_path(answer_path: Path) -> Path:
    return answer_path.with_name(f"{answer_path.stem}.judgments{answer_path.suffix}")


def read_answer_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question_id" not in item:
                raise ValueError(f"{path}:{line_number} is missing question_id.")
            rows.append(item)
    return rows


def question_map(questions: list[Question]) -> dict[str, Question]:
    return {str(question.question_id): question for question in questions}


def iter_answer_turns(
    answer_rows: list[dict[str, Any]],
    questions_by_id: dict[str, Question],
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in answer_rows:
        question_id = str(row["question_id"])
        question = questions_by_id.get(question_id)
        if question is None:
            raise KeyError(f"Could not find question_id={question_id!r} in the question file.")
        choices = row.get("choices") or []
        if not choices:
            continue
        turns = choices[0].get("turns") or []
        for turn_index, answer in enumerate(turns, start=1):
            user_turn = question.turns[turn_index - 1] if turn_index - 1 < len(question.turns) else ""
            items.append(
                {
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn_index": turn_index,
                    "user_turn": user_turn,
                    "answer": answer,
                    "model_id": row.get("model_id"),
                    "answer_id": row.get("answer_id"),
                }
            )
    selected = items[offset:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def build_judge_prompt(item: dict[str, Any]) -> str:
    return (
        "You are judging one assistant answer for an MT-Bench-style evaluation.\n"
        "Score the answer from 1 to 10 using these criteria: relevance to the user request, "
        "factual correctness, completeness, clarity, and whether it avoids unsupported claims.\n\n"
        "Return JSON only with this exact shape:\n"
        '{"score": <number from 1 to 10>, "rationale": "<brief reason>"}\n\n'
        f"Category: {item['category']}\n"
        f"Question id: {item['question_id']}\n"
        f"Turn: {item['turn_index']}\n\n"
        f"User question:\n{item['user_turn']}\n\n"
        f"Assistant answer:\n{item['answer']}\n"
    )


def call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    site_url: str,
    site_title: str,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict but fair evaluation judge. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": site_title,
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_body}") from exc
    elapsed = time.perf_counter() - started
    response_json = json.loads(response_body)
    content = response_json["choices"][0]["message"]["content"]
    return content, elapsed


def parse_judgment(raw_text: str) -> tuple[float | None, str]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        score = payload.get("score")
        rationale = str(payload.get("rationale", ""))
        return float(score) if score is not None else None, rationale
    except json.JSONDecodeError:
        match = re.search(r"(?i)score[^0-9]*(10(?:\.0)?|[1-9](?:\.\d+)?)", text)
        score = float(match.group(1)) if match else None
        return score, text


def main() -> None:
    args = parse_args()
    question_path = Path(args.question_file).expanduser()
    answer_path = Path(args.answers_jsonl).expanduser()
    judgment_path = Path(args.judgments_jsonl).expanduser() if args.judgments_jsonl else default_judgment_path(answer_path)

    questions = read_questions(question_path, limit=0, offset=0)
    answer_rows = read_answer_rows(answer_path)
    items = iter_answer_turns(answer_rows, question_map(questions), args.offset, args.limit)
    if not items:
        print("No answer turns selected; exiting.")
        return

    print("OpenRouter MT-Bench-style judge")
    print(f"question file: {question_path}")
    print(f"answers jsonl: {answer_path}")
    print(f"judgments jsonl: {judgment_path}")
    print(f"judge model: {args.model}")
    print(f"selected turns: {len(items)}")

    if args.overwrite:
        truncate_file(judgment_path)

    api_key = os.environ.get(args.api_key_env)
    if not args.dry_run and not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env}=<your OpenRouter key>.")

    scores: list[float] = []
    for index, item in enumerate(items, start=1):
        prompt = build_judge_prompt(item)
        print(
            f"[judge {index}/{len(items)}] question_id={item['question_id']!r} "
            f"turn={item['turn_index']} model_id={item.get('model_id')!r}",
            flush=True,
        )
        if args.dry_run:
            print(prompt)
            raw_judgment = '{"score": 0, "rationale": "dry run"}'
            request_seconds = 0.0
        else:
            assert api_key is not None
            raw_judgment, request_seconds = call_openrouter(
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_judge_tokens,
                timeout_seconds=args.timeout_seconds,
                site_url=args.site_url,
                site_title=args.site_title,
            )
            if args.sleep_seconds > 0 and index < len(items):
                time.sleep(args.sleep_seconds)

        score, rationale = parse_judgment(raw_judgment)
        if score is not None:
            scores.append(score)
        append_jsonl(
            judgment_path,
            {
                **item,
                "judge_model": args.model,
                "score": score,
                "rationale": rationale,
                "raw_judgment": raw_judgment,
                "request_seconds": request_seconds,
                "tstamp": time.time(),
            },
        )
        score_text = "n/a" if score is None else f"{score:.2f}"
        print(f"[judge {index}/{len(items)}] score={score_text} request_seconds={request_seconds:.2f}", flush=True)

    if scores:
        print(f"average score: {sum(scores) / len(scores):.2f} over {len(scores)} judged turn(s)")
    print(f"judgments: {judgment_path}")


if __name__ == "__main__":
    main()
