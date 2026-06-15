#!/usr/bin/env python3
"""Convert a Hugging Face VLM dataset split into the eval JSONL format.

The output schema is:
{"id": "...", "question": "...", "answer": "...", "images": ["images/xxx.png"]}

The script is intentionally field-name driven so it can handle MathVista and
nearby VLM benchmarks without hard-coding one dataset loader.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def slug(value: Any, fallback: int) -> str:
    text = str(value if value is not None else fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or str(fallback)


def first_existing(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def stringify_answer(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def format_question(question: Any, choices: Any, choices_prefix: str) -> str:
    text = "" if question is None else str(question)
    if not choices:
        return text
    if isinstance(choices, str):
        choices_text = choices
    elif isinstance(choices, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = []
        for idx, choice in enumerate(choices):
            label = labels[idx] if idx < len(labels) else str(idx)
            lines.append(f"{label}. {choice}")
        choices_text = "\n".join(lines)
    else:
        choices_text = json.dumps(choices, ensure_ascii=False)
    return f"{text}\n\n{choices_prefix}\n{choices_text}"


def save_image(value: Any, path: Path) -> str | None:
    if value is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(value, str):
        # Existing path or URL; let downstream evaluator resolve it.
        return value

    if hasattr(value, "save"):
        value.save(path)
        return str(path)

    if isinstance(value, dict):
        if "path" in value and value["path"]:
            return str(value["path"])
        if "bytes" in value and value["bytes"]:
            path.write_bytes(value["bytes"])
            return str(path)

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. AI4Math/MathVista")
    parser.add_argument("--split", default="testmini")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--id-field", default="pid,id,question_id,uid,index")
    parser.add_argument("--question-field", default="query,question,prompt,input")
    parser.add_argument("--answer-field", default="answer,gt,ground_truth,label")
    parser.add_argument("--image-field", default="decoded_image,image,images")
    parser.add_argument("--choices-field", default="choices,options")
    parser.add_argument("--choices-prefix", default="Choices:")
    args = parser.parse_args()

    from datasets import load_dataset

    id_fields = [item.strip() for item in args.id_field.split(",") if item.strip()]
    question_fields = [item.strip() for item in args.question_field.split(",") if item.strip()]
    answer_fields = [item.strip() for item in args.answer_field.split(",") if item.strip()]
    image_fields = [item.strip() for item in args.image_field.split(",") if item.strip()]
    choices_fields = [item.strip() for item in args.choices_field.split(",") if item.strip()]

    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=args.trust_remote_code)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    image_dir = args.image_dir or (args.out.parent / "images")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with args.out.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            row = dict(sample)
            sid = slug(first_existing(row, id_fields), idx)
            question = first_existing(row, question_fields)
            answer = first_existing(row, answer_fields)
            choices = first_existing(row, choices_fields)
            image_value = first_existing(row, image_fields)

            if question is None or answer is None:
                continue

            output_row: dict[str, Any] = {
                "id": sid,
                "question": format_question(question, choices, args.choices_prefix),
                "answer": stringify_answer(answer),
            }

            images: list[str] = []
            if isinstance(image_value, list):
                for image_idx, item in enumerate(image_value):
                    saved = save_image(item, image_dir / f"{sid}_{image_idx}.png")
                    if saved:
                        images.append(saved)
            else:
                saved = save_image(image_value, image_dir / f"{sid}.png")
                if saved:
                    images.append(saved)
            if images:
                output_row["images"] = images

            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"wrote {rows_written} rows to {args.out}")
    print(f"images saved/resolved under {image_dir}")


if __name__ == "__main__":
    main()
