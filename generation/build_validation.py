#!/usr/bin/env python3
"""
Create an OpenAI Batch API JSONL file for validation requests.

This version is adapted to the user's prompt-file format, where each prompt row
looks like:

{
  "prompt": "Write a piece of fiction. Center it on a lonely-hearted train conductor ...",
  "words": [
    {"word": "sigh", "pos": "verb"},
    {"word": "jellyfish", "pos": "noun"},
    {"word": "waste", "pos": "noun"},
    {"word": "write", "pos": "verb"}
  ],
  "features": [],
  "subject": {
    "character": "train conductor",
    "action": "gets snowed in with a group of strangers",
    "place": "a bookstore",
    "adjective": "lonely-hearted",
    "goal": null
  },
  "feature_phrases": [],
  "word_clause": "include 'sigh' as a verb, ...",
  "feature_clause": null,
  "subject_clause": "a lonely-hearted train conductor who gets snowed in with a group of strangers in a bookstore."
}

The completions file should be the JSONL output from a previous OpenAI batch job.
This script joins prompts to completions by row order by default, or by IDs if present.

Usage
-----
python build_validation_jsonl.py \
    --prompts prompts.jsonl \
    --completions batch_output.jsonl \
    --output validation_batch.jsonl \
    --model gpt-5.4

If your prompt file has explicit ids and your completion batch used those ids as
custom_id, the script will use them automatically.

If not, it will fall back to joining by line number / order.

Notes
-----
- This script only creates the JSONL file for upload in the OpenAI dashboard.
- It does not call the API.
- It expects the previous batch output to contain one result per prompt, in the
  same order, unless IDs are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_OUTPUT_TOKENS = 500


# =========================================================
# JSONL utilities
# =========================================================

def load_json(path: Path|str) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        raise TypeError("File must be .json or .jsonl!")
    return rows




def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# =========================================================
# Formatting helpers
# =========================================================

def format_required_words(words: List[Dict[str, Any]]) -> str:
    if not words:
        return "None"
    return "\n".join(f'- {item["word"]}' for item in words)


def format_pos_requirements(words: List[Dict[str, Any]]) -> str:
    if not words:
        return "None"
    return "\n".join(f'- {item["word"]} — {item["pos"]}' for item in words)


def format_features(features: List[str], feature_phrases: Optional[List[str]] = None) -> str:
    combined: List[str] = []
    if features:
        combined.extend(features)
    if feature_phrases:
        combined.extend(feature_phrases)

    if not combined:
        return "None"

    return "\n".join(f"- {feature}" for feature in combined)


def build_subject_text(subject: Dict[str, Any], subject_clause: Optional[str]) -> str:
    """
    Prefer subject_clause if present, since it is the most human-readable summary
    of the intended subject. Fall back to structured subject fields otherwise.
    """
    if isinstance(subject_clause, str) and subject_clause.strip():
        return subject_clause.strip()

    if not isinstance(subject, dict):
        return "None provided"

    adjective = (subject.get("adjective") or "").strip()
    character = (subject.get("character") or "").strip()
    action = (subject.get("action") or "").strip()
    place = (subject.get("place") or "").strip()
    goal = (subject.get("goal") or "").strip()

    parts: List[str] = []

    char_part = " ".join(p for p in [adjective, character] if p).strip()
    if char_part:
        parts.append(char_part)

    if action:
        parts.append(action)

    if place:
        parts.append(f"in {place}")

    if goal:
        parts.append(f"goal: {goal}")

    return ", ".join(parts) if parts else "None provided"


# =========================================================
# Completion extraction
# =========================================================

def extract_story_from_response_obj(obj: Dict[str, Any]) -> Optional[str]:
    """
    Try a few common OpenAI batch output/result shapes.
    """
    response = obj.get("response", {})
    body = response.get("body", {})

    # Shape 1: Responses API output array
    output = body.get("output")
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue

            # Direct output_text item
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                texts.append(item["text"])

            # Message-style nested content
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        texts.append(c["text"])

        text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if text:
            return text

    # Shape 2: direct body.content
    content = body.get("content")
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "output_text" and isinstance(item.get("text"), str):
                texts.append(item["text"])

        text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if text:
            return text

    # Shape 3: direct text-like body fields
    for key in ("text", "output_text"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Shape 4: older chat-style choices fallback
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        texts: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            texts.append(c["text"])

        text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if text:
            return text

    return None


# =========================================================
# Validator prompt construction
# =========================================================

def build_validator_prompt(prompt_row: Dict[str, Any], story: str) -> str:
    words = prompt_row.get("words", [])
    features = prompt_row.get("features", [])
    feature_phrases = prompt_row.get("feature_phrases", [])
    subject = prompt_row.get("subject", {})
    subject_clause = prompt_row.get("subject_clause")
    original_prompt = (prompt_row.get("prompt") or "").strip()

    subject_text = build_subject_text(subject, subject_clause)

    # Since your prompt format does not include sentence bounds, do not enforce them.
    # If later you add them, this can be extended.
    return f"""You are a strict evaluator.

Your task is to check whether a generated story satisfies a set of constraints.
Do NOT rewrite the story or improve it. Only evaluate it.
Be conservative. If a requirement is unclear or only weakly satisfied, mark it as false.

You must check the following:
1. SUBJECT — Does the story clearly match the intended subject?
2. REQUIRED WORDS — Does the story contain each required word exactly?
3. PART OF SPEECH — Is each required word used with the correct part of speech?
4. FEATURES — Does the story contain the requested narrative features?
5. OVERALL — Does the story satisfy all requested constraints?

Return your evaluation ONLY as valid JSON.
Do not include markdown fences.
Do not include commentary.

Use exactly this schema:

{{
  "subject_correct": true,
  "required_words": {{
    "word1": true
  }},
  "pos_correct": {{
    "word1": true
  }},
  "features": {{
    "feature1": true
  }},
  "overall_pass": true,
  "notes": []
}}

Rules:
- "required_words" must contain one boolean entry for every required word listed below.
- "pos_correct" must contain one boolean entry for every required word listed below.
- "features" must contain one boolean entry for every requested feature listed below.
- If there are no requested features, return "features": {{}}.
- "overall_pass" is true only if subject_correct is true, every required word is present, every required word has the correct part of speech, and every requested feature is satisfied.
- "notes" should be a short list of brief failure reasons, or an empty list if all checks pass.

ORIGINAL PROMPT:
{original_prompt}

INTENDED SUBJECT:
{subject_text}

REQUIRED WORDS:
{format_required_words(words)}

PARTS OF SPEECH:
{format_pos_requirements(words)}

REQUESTED FEATURES:
{format_features(features, feature_phrases)}

STORY:
{story}
"""


# =========================================================
# Joining prompt rows with completion rows
# =========================================================

def get_prompt_row_id(prompt_row: Dict[str, Any], idx: int) -> str:
    """
    Prefer explicit id if present; otherwise use row number.
    """
    if "id" in prompt_row and prompt_row["id"] is not None:
        return str(prompt_row["id"])
    return f"row-{idx}"


def get_completion_row_id(completion_row: Dict[str, Any], idx: int) -> str:
    """
    Prefer custom_id if present; otherwise use row number.
    """
    if "custom_id" in completion_row and completion_row["custom_id"] is not None:
        return str(completion_row["custom_id"])
    return f"row-{idx}"


def build_prompt_index(prompt_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(prompt_rows, start=1):
        row_id = get_prompt_row_id(row, idx)
        if row_id in index:
            raise ValueError(f"Duplicate prompt id found: {row_id}")
        index[row_id] = row
    return index


def build_completion_index(completion_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(completion_rows, start=1):
        row_id = get_completion_row_id(row, idx)
        if row_id in index:
            raise ValueError(f"Duplicate completion id/custom_id found: {row_id}")
        index[row_id] = row
    return index


def rows_have_explicit_ids(prompt_rows: List[Dict[str, Any]], completion_rows: List[Dict[str, Any]]) -> bool:
    prompt_has_ids = all("id" in row and row["id"] is not None for row in prompt_rows)
    completion_has_ids = all("custom_id" in row and row["custom_id"] is not None for row in completion_rows)
    return prompt_has_ids and completion_has_ids


# =========================================================
# Batch request construction
# =========================================================

def build_batch_request(
    *,
    validation_id: str,
    validator_prompt: str,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    return {
        "custom_id": validation_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": validator_prompt,
            "max_output_tokens": max_output_tokens,
        },
    }


def create_validation_rows(
    prompt_rows: List[Dict[str, Any]],
    completion_rows: List[Dict[str, Any]],
    *,
    model: str,
    max_output_tokens: int,
    keep_empty: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # If both sides have explicit ids, join by id/custom_id.
    # Otherwise, join by row order.
    if rows_have_explicit_ids(prompt_rows, completion_rows):
        prompt_index = build_prompt_index(prompt_rows)
        completion_index = build_completion_index(completion_rows)

        for idx, (prompt_id, prompt_row) in enumerate(prompt_index.items(), start=1):
            completion_row = completion_index.get(prompt_id)
            if completion_row is None:
                warnings.append(f"Missing completion for prompt id: {prompt_id}")
                continue

            story = extract_story_from_response_obj(completion_row)
            if not story:
                msg = f"Could not extract completion text for prompt id: {prompt_id}"
                if keep_empty:
                    warnings.append(msg + " (kept as empty story)")
                    story = ""
                else:
                    warnings.append(msg + " (skipped)")
                    continue

            validator_prompt = build_validator_prompt(prompt_row, story)
            validation_id = f"validate-{prompt_id}"

            rows.append(
                build_batch_request(
                    validation_id=validation_id,
                    validator_prompt=validator_prompt,
                    model=model,
                    max_output_tokens=max_output_tokens,
                )
            )
    else:
        # Join by order
        if len(prompt_rows) != len(completion_rows):
            warnings.append(
                f"Prompt row count ({len(prompt_rows)}) does not match completion row count ({len(completion_rows)}). "
                f"Will join by order up to min length = {min(len(prompt_rows), len(completion_rows))}."
            )

        n = min(len(prompt_rows), len(completion_rows))
        for idx in range(n):
            prompt_row = prompt_rows[idx]
            completion_row = completion_rows[idx]
            prompt_id = get_prompt_row_id(prompt_row, idx + 1)

            story = extract_story_from_response_obj(completion_row)
            if not story:
                msg = f"Could not extract completion text for prompt row: {idx + 1}"
                if keep_empty:
                    warnings.append(msg + " (kept as empty story)")
                    story = ""
                else:
                    warnings.append(msg + " (skipped)")
                    continue

            validator_prompt = build_validator_prompt(prompt_row, story)
            validation_id = f"validate-{prompt_id}"

            rows.append(
                build_batch_request(
                    validation_id=validation_id,
                    validator_prompt=validator_prompt,
                    model=model,
                    max_output_tokens=max_output_tokens,
                )
            )

    return rows, warnings


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a validation batch JSONL from prompt rows and previous batch completions."
    )
    parser.add_argument("--prompts", required=True, help="Path to prompt JSONL file.")
    parser.add_argument("--completions", required=True, help="Path to previous batch output JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output validation batch JSONL file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name for validation requests (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per validation response (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows even when completion text cannot be extracted; story will be empty.",
    )
    parser.add_argument(
        "--warnings-file",
        default=None,
        help="Optional path to write warnings/skipped-record messages.",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    completions_path = Path(args.completions)
    output_path = Path(args.output)

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    if not completions_path.exists():
        raise FileNotFoundError(f"Completions file not found: {completions_path}")

    prompt_rows = load_json(prompts_path)
    completion_rows = load_json(completions_path)

    rows, warnings = create_validation_rows(
        prompt_rows,
        completion_rows,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        keep_empty=args.keep_empty,
    )

    written = write_jsonl(output_path, rows)

    print(f"Wrote {written} validation batch requests to {output_path}")
    print(f"Loaded {len(prompt_rows)} prompt rows")
    print(f"Loaded {len(completion_rows)} completion rows")
    print(f"Warnings: {len(warnings)}")

    if warnings:
        preview_n = min(10, len(warnings))
        print("\nFirst warnings:")
        for w in warnings[:preview_n]:
            print(f"- {w}")

    if args.warnings_file and len(warnings)>0:
        warnings_path = Path(args.warnings_file)
        warnings_path.write_text("\n".join(warnings), encoding="utf-8")
        print(f"\nWrote warnings to {warnings_path}")


if __name__ == "__main__":
    main()