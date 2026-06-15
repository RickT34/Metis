#!/usr/bin/env python3
"""Compare original Qwen3-VL and Metis predictions on matched samples."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def first_existing(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def record_key(row: dict[str, Any], fallback: int) -> str:
    value = first_existing(row, ("id", "uid", "index", "idx", "question_id", "sample_id", "input", "question", "prompt"))
    key = normalize_key(value)
    return key if key else str(fallback)


def build_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record_key(row, i): row for i, row in enumerate(rows)}


def parse_correct(row: dict[str, Any]) -> int | None:
    value = first_existing(row, ("accuracy", "answer_score", "acc", "correct", "is_correct", "score"))
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) > 0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "correct", "yes"}:
            return 1
        if lowered in {"0", "false", "incorrect", "wrong", "no"}:
            return 0
    return None


def parse_output(row: dict[str, Any]) -> str:
    value = first_existing(row, ("output", "response", "prediction", "completion", "text"))
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_answer(row: dict[str, Any]) -> str:
    value = first_existing(row, ("answer", "extracted_answer", "final_answer"))
    if value is not None:
        return normalize_key(value)
    match = ANSWER_RE.search(parse_output(row))
    return normalize_key(match.group(1)) if match else ""


def parse_tool_count(row: dict[str, Any]) -> int:
    info = row.get("tool_interact_info")
    if isinstance(info, list):
        return sum(1 for item in info if isinstance(item, dict) and not item.get("finish", False))
    value = first_existing(row, ("tool_calls", "tool_call_count", "Tool Call", "num_turns", "__num_turns__"))
    if isinstance(value, (int, float)):
        return int(value)
    return len(TOOL_CALL_RE.findall(parse_output(row)))


def category(base_correct: int, metis_correct: int) -> str:
    if base_correct and metis_correct:
        return "both_correct"
    if base_correct and not metis_correct:
        return "base_only"
    if not base_correct and metis_correct:
        return "metis_only"
    return "both_wrong"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--metis", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-name", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--metis-name", default="Metis-8B-RL")
    parser.add_argument("--keep-missing-correctness", action="store_true")
    args = parser.parse_args()

    base = build_index(load_records(args.base))
    metis = build_index(load_records(args.metis))
    common = sorted(set(base) & set(metis))
    if not common:
        raise SystemExit("No matched records. Make sure ids/questions align across both files.")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for key in common:
        base_row = base[key]
        metis_row = metis[key]
        base_correct = parse_correct(base_row)
        metis_correct = parse_correct(metis_row)
        if None in (base_correct, metis_correct):
            if not args.keep_missing_correctness:
                skipped += 1
                continue
            base_correct = base_correct or 0
            metis_correct = metis_correct or 0

        rows.append(
            {
                "key": key,
                "category": category(int(base_correct), int(metis_correct)),
                "base_correct": int(base_correct),
                "metis_correct": int(metis_correct),
                "base_tool_calls": parse_tool_count(base_row),
                "metis_tool_calls": parse_tool_count(metis_row),
                "base_answer": parse_answer(base_row),
                "metis_answer": parse_answer(metis_row),
            }
        )

    if not rows:
        raise SystemExit(f"No usable records after dropping missing correctness rows. skipped={skipped}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    counts = {name: 0 for name in ("both_correct", "metis_only", "base_only", "both_wrong")}
    for row in rows:
        counts[row["category"]] += 1

    base_correct_total = sum(row["base_correct"] for row in rows)
    metis_correct_total = sum(row["metis_correct"] for row in rows)
    metis_tool_total = sum(row["metis_tool_calls"] for row in rows)
    metis_tool_called = sum(1 for row in rows if row["metis_tool_calls"] > 0)

    summary = {
        "n": total,
        "base_name": args.base_name,
        "metis_name": args.metis_name,
        "base_accuracy": base_correct_total / total,
        "metis_accuracy": metis_correct_total / total,
        "delta_accuracy": (metis_correct_total - base_correct_total) / total,
        "metis_tool_rate": metis_tool_called / total,
        "metis_avg_tool_calls": metis_tool_total / total,
        **counts,
        "skipped": skipped,
    }

    (args.out_dir / "model_compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "model_compare_records.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "model_compare_records.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "key",
            "category",
            "base_correct",
            "metis_correct",
            "base_tool_calls",
            "metis_tool_calls",
            "base_answer",
            "metis_answer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Qwen3-VL vs Metis Correctness Summary",
        "",
        f"- Base model: `{args.base_name}`",
        f"- Metis model: `{args.metis_name}`",
        f"- Matched samples: {total}",
        f"- Skipped samples: {skipped}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Base accuracy | {pct(summary['base_accuracy'])} |",
        f"| Metis accuracy | {pct(summary['metis_accuracy'])} |",
        f"| Delta accuracy | {pct(summary['delta_accuracy'])} |",
        f"| Metis ToolRate | {pct(summary['metis_tool_rate'])} |",
        f"| Metis AvgToolCalls | {summary['metis_avg_tool_calls']:.2f} |",
        "",
        "| Category | n | Meaning |",
        "|---|---:|---|",
        f"| both_correct | {counts['both_correct']} | both models correct |",
        f"| metis_only | {counts['metis_only']} | Metis fixes a base-model error |",
        f"| base_only | {counts['base_only']} | Metis regresses on a base-correct sample |",
        f"| both_wrong | {counts['both_wrong']} | neither model solves it |",
        "",
        "Inspect `model_compare_records.csv` for sample-level correctness and tool calls.",
    ]
    (args.out_dir / "model_compare_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"matched={len(common)} usable={len(rows)} skipped={skipped}")
    print((args.out_dir / "model_compare_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
