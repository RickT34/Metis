#!/usr/bin/env python3
"""Validate JSONL data for HDPO/Metis evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUESTION_KEYS = ("question", "input", "prompt", "task")
ANSWER_KEYS = ("answer", "ground_truth", "gt", "gts", "label")
ID_KEYS = ("id", "uid", "index", "idx", "question_id", "sample_id")


def first_existing(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"line {line_no}: each row must be a JSON object")
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def image_list(row: dict[str, Any]) -> list[str]:
    images = row.get("images") if "images" in row else row.get("image", [])
    if not images:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        return [str(item) for item in images]
    return [str(images)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    root = args.data_root or args.data.parent
    rows = load_jsonl(args.data)
    seen: set[str] = set()
    errors: list[str] = []
    image_count = 0

    for fallback_idx, row in enumerate(rows):
        line_no = row.pop("_line_no")
        sid = first_existing(row, ID_KEYS)
        sid = str(fallback_idx if sid is None else sid)
        if sid in seen:
            errors.append(f"line {line_no}: duplicate id/key {sid!r}")
        seen.add(sid)

        question = first_existing(row, QUESTION_KEYS)
        if question is None or str(question).strip() == "":
            errors.append(f"line {line_no}: missing question/input/prompt/task")

        answer = first_existing(row, ANSWER_KEYS)
        if answer is None or str(answer).strip() == "":
            errors.append(f"line {line_no}: missing answer/ground_truth/gt/gts/label")

        for image in image_list(row):
            image_count += 1
            if image.startswith(("data:", "http://", "https://")):
                continue
            path = Path(image)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                errors.append(f"line {line_no}: image not found: {image}")

    if errors:
        print(f"invalid: {args.data}")
        print(f"rows={len(rows)} images={image_count} errors={len(errors)}")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors")
        raise SystemExit(1)

    print(f"ok: {args.data}")
    print(f"rows={len(rows)} unique_ids={len(seen)} images={image_count} data_root={root}")


if __name__ == "__main__":
    main()
