#!/usr/bin/env python3
"""Evaluate Qwen/Metis-style models through an OpenAI-compatible endpoint.

The runner is deliberately small: it reads a JSONL dataset, talks to vLLM's
OpenAI-compatible chat API, optionally executes Metis tool calls, judges answers
with exact matching or a shared LLM judge, and writes JSONL predictions.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import string
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


METIS_TOOL_SYSTEM = """You are an efficient problem-solving agent. Your goal is to answer the user's question accurately while minimizing unnecessary tool usage.

# Tools
You have access to the following tools. Use them ONLY when they provide clear value that reasoning alone cannot.

## Python Code Execution
Write Python code to perform numerical analysis, data processing, or image operations (e.g., cropping, resizing, rotating, color adjustment, contrast enhancement, drawing auxiliary lines).
Format:
<tool_call>{"name": "python", "arguments": {"code": "your code here"}}</tool_call>

## Text Search
Trigger a web search to find relevant textual information. Use specific, targeted queries for best results.
Format:
<tool_call>{"name": "text_search", "arguments": {"query": "your search query"}}</tool_call>

## Image Search
Trigger a visual search using the user's provided image to identify objects, landmarks, products, or other visual content.
Format:
<tool_call>{"name": "image_search", "arguments": {}}</tool_call>

# Decision Guidelines
1. Think before acting: Always reason within <reason>...</reason> before deciding your next action.
2. Choose the right approach:
   - Direct answer: Use when you can confidently answer from your own knowledge or visual inspection.
   - Python: Use for computation, measurement, pixel-level analysis, or image enhancement.
   - Text Search: Use when the question requires factual knowledge you are uncertain about.
   - Image Search: Use when you need to identify something in the image that you cannot recognize.
3. Be purposeful: Each tool call must have a clear objective. Do NOT call tools to confirm what you already know.
4. Be decisive: Once you are confident in your answer, provide it immediately.

# Output Format
Always start with <reason>...</reason>, then choose one:
Option 1 - Use a tool:
<reason>Why this tool call is necessary and what you expect to learn.</reason>
<tool_call>...</tool_call>
Option 2 - Answer directly:
<reason>Your reasoning and final synthesis.</reason>
<answer>Your concise answer.</answer>
"""


NO_TOOL_SYSTEM = """You are a careful multimodal question-answering assistant.
Answer the user's question directly using only the provided input and your internal knowledge.
Do not call or mention external tools. Always use:
<reason>brief reasoning</reason>
<answer>concise final answer</answer>
"""


JUDGE_SYSTEM = """You are a meticulous and impartial AI evaluator. Judge whether the predicted answer is correct based on the question and ground truth answer.
Return exactly one word: CORRECT or INCORRECT."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: expected JSON object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_existing(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def sample_id(row: dict[str, Any], fallback: int) -> str:
    value = first_existing(row, ("id", "uid", "index", "idx", "question_id", "sample_id"))
    return str(fallback if value is None else value)


def question_text(row: dict[str, Any]) -> Any:
    return first_existing(row, ("question", "input", "prompt", "task"))


def ground_truth(row: dict[str, Any]) -> Any:
    return first_existing(row, ("answer", "ground_truth", "gt", "gts", "label"))


def data_uri(path: str, root: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def image_list(row: dict[str, Any]) -> list[str]:
    images = row.get("images") if "images" in row else row.get("image", [])
    if not images:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        return [str(x) for x in images]
    return [str(images)]


def user_content(row: dict[str, Any], root: Path) -> Any:
    question = question_text(row)
    if question is None:
        raise ValueError(f"Missing question/input/prompt/task field: {row}")
    images = image_list(row)
    if not images:
        return str(question)
    content: list[dict[str, Any]] = [{"type": "text", "text": str(question)}]
    for image in images:
        url = image if image.startswith(("data:", "http://", "https://")) else data_uri(image, root)
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def normalize_tool_action(text: str) -> str | None:
    match = TOOL_CALL_RE.search(text)
    if match:
        return match.group(0)
    if "<tool_call>" in text and "</tool_call>" not in text:
        return text[text.index("<tool_call>") :].strip() + "</tool_call>"
    return None


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def extract_choice(text: str) -> str | None:
    match = re.search(r"\b([A-E])\b", text.strip().upper())
    return match.group(1) if match else None


def exact_correct(answer: Any, prediction: str) -> int:
    if answer is None:
        return 0
    answers = answer if isinstance(answer, list) else [answer]
    pred_norm = normalize_answer(prediction)
    pred_choice = extract_choice(prediction)
    for ans in answers:
        ans_norm = normalize_answer(ans)
        if ans_norm and pred_norm == ans_norm:
            return 1
        ans_choice = extract_choice(str(ans))
        if ans_choice and pred_choice and ans_choice == pred_choice:
            return 1
    return 0


def llm_judge_correct(client: OpenAI, model: str, question: str, answer: Any, prediction: str) -> int:
    if answer is None:
        return 0
    prompt = f"""[Question]
{question}

[Ground Truth Answer]
{answer}

[Predicted Answer]
{prediction}

Return exactly one word: CORRECT or INCORRECT."""
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            return int(verdict.startswith("CORRECT"))
        except Exception:
            time.sleep(1)
    return 0


def call_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        kwargs["stop"] = stop
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def call_tool_server(
    tool_url: str,
    trajectory_id: str,
    action: str,
    row: dict[str, Any],
    data_root: Path,
    timeout: int,
) -> dict[str, Any]:
    extra = dict(row.get("extra_info") or {})
    extra.setdefault("index", first_existing(row, ("id", "index", "idx")) or trajectory_id)
    images = image_list(row)
    if images:
        extra["images"] = [
            image if image.startswith(("data:", "http://", "https://")) else data_uri(image, data_root)
            for image in images
        ]

    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [extra],
        "finish": [False],
        "is_last_step": [False],
    }
    resp = requests.post(tool_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    obs = data["observations"][0]
    return {
        "obs": obs.get("obs", obs) if isinstance(obs, dict) else obs,
        "image": obs.get("image", []) if isinstance(obs, dict) else [],
        "metrics": obs.get("metrics", {}) if isinstance(obs, dict) else {},
        "done": bool(data.get("dones", [False])[0]),
        "valid_action": int(bool(data.get("valids", [False])[0])),
        "action": action,
        "finish": False,
        "success": bool(data.get("valids", [False])[0]),
    }


def observation_message(tool_result: dict[str, Any]) -> dict[str, Any]:
    obs = str(tool_result.get("obs", ""))
    images = tool_result.get("image") or []
    if isinstance(images, str):
        images = [images]
    if not images:
        return {"role": "user", "content": f"Tool observation:\n{obs}"}
    content: list[dict[str, Any]] = [{"type": "text", "text": f"Tool observation:\n{obs}"}]
    content.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
    return {"role": "user", "content": content}


def run_one(
    row: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    client: OpenAI,
    judge_client: OpenAI | None,
) -> dict[str, Any]:
    sid = sample_id(row, idx)
    question = question_text(row)
    answer = ground_truth(row)

    if args.mode == "no_tool":
        system = NO_TOOL_SYSTEM
        use_tools = False
        stop = None
    else:
        system = METIS_TOOL_SYSTEM
        use_tools = True
        stop = ["</tool_call>"]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content(row, args.data_root)},
    ]

    output_parts: list[str] = []
    tool_interact_info: list[dict[str, Any]] = []
    trajectory_id = f"{args.model_alias}-{args.mode}-{sid}-{uuid.uuid4().hex[:8]}"

    for turn in range(args.max_turns if use_tools else 1):
        text = call_chat(client, args.model, messages, args.max_tokens_per_turn, args.temperature, stop)
        action = normalize_tool_action(text)
        if action and use_tools and turn < args.max_turns - 1:
            if not text.rstrip().endswith("</tool_call>"):
                text = text.rstrip() + "</tool_call>"
            output_parts.append(text)
            messages.append({"role": "assistant", "content": text})
            tool_result = call_tool_server(args.tool_url, trajectory_id, action, row, args.data_root, args.tool_timeout)
            tool_interact_info.append(tool_result)
            messages.append(observation_message(tool_result))
            if tool_result.get("done"):
                break
            continue
        output_parts.append(text)
        break

    output = "\n".join(output_parts)
    prediction = extract_answer(output)
    if args.judge_mode == "llm":
        assert judge_client is not None
        correct = llm_judge_correct(judge_client, args.judge_model, str(question), answer, prediction)
    else:
        correct = exact_correct(answer, prediction)

    return {
        "id": sid,
        "input": question,
        "gts": answer,
        "output": output,
        "answer": prediction,
        "accuracy": correct,
        "answer_score": correct,
        "tool_calls": len(tool_interact_info),
        "tool_interact_info": tool_interact_info,
        "mode": args.mode,
        "model": args.model,
        "model_alias": args.model_alias,
        "judge_mode": args.judge_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["no_tool", "policy"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-alias", default="model")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--judge-mode", choices=["exact", "llm"], default=os.environ.get("JUDGE_MODE", "exact"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--tool-url", default=os.environ.get("TOOL_SERVER_URL", "http://127.0.0.1:30569/get_observation"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens-per-turn", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tool-timeout", type=int, default=420)
    args = parser.parse_args()

    if args.judge_mode == "llm" and (not args.judge_base_url or not args.judge_model):
        raise SystemExit("JUDGE_MODE=llm requires --judge-base-url and --judge-model.")

    samples = load_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]

    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        done_ids = {str(row.get("id")) for row in load_jsonl(args.output)}

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    judge_client = None
    if args.judge_mode == "llm":
        judge_client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)

    wrote = 0
    for idx, sample in enumerate(samples):
        sid = sample_id(sample, idx)
        if sid in done_ids:
            continue
        print(f"[{args.model_alias}/{args.mode}] {idx + 1}/{len(samples)} id={sid}", flush=True)
        result = run_one(sample, idx, args, client, judge_client)
        append_jsonl(args.output, [result])
        wrote += 1
    print(f"wrote {wrote} new rows to {args.output}")


if __name__ == "__main__":
    main()
