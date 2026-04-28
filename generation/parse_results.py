#!/usr/bin/env python3
"""
Parse validation batch results and produce:

1. A PASSED .json file containing a list of dicts in this format:
   {
     "story": <generated text from completion output>,
     "instruction": <original prompt row from prompt file>,
     "source": <model name from completion output>
   }

2. A FAILED .jsonl file containing one row per failed example, with prompt,
   completion, validation details, and failure reasons.

3. Optionally, a SUMMARY .json file with pass/fail counts, percentages, and
   failure-reason breakdowns.

Expected inputs
---------------
1. Prompt file (JSONL), one row per instruction, in your prompt format:
   {
     "prompt": "...",
     "words": [...],
     "features": [...],
     "subject": {...},
     "feature_phrases": [...],
     "word_clause": "...",
     "feature_clause": null,
     "subject_clause": "..."
   }

2. Completion batch output file (JSONL), from the generation batch job

3. Validation batch output file (JSONL), from the validation batch job

Joining behavior
----------------
- If prompts have explicit "id" fields and completions have "custom_id", join by ID.
- Otherwise, prompts and completions are joined by row order.
- Validation rows are matched by stripping the "validate-" prefix from their custom_id.

Usage
-----
python parse_validation_results.py \
    --prompts prompts.jsonl \
    --completions generation_batch_output.jsonl \
    --validations validation_batch_output.jsonl \
    --passed-json passed.json \
    --failed-jsonl failed.jsonl \
    --summary-json summary.json

If you do not want a summary file, omit --summary-json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================================================
# JSON / JSONL helpers
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
    

def write_json(path: Path|str, rows: Iterable[Dict[str, Any]]) -> int|None:
    path = Path(path)
    count = None 
    if path.suffix == ".jsonl":
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    elif path.suffix == ".json":
        if isinstance(rows, list):
            count = len(rows)
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        raise TypeError("File must be .json or .jsonl!")
        
    return count


# =========================================================
# ID helpers
# =========================================================

def get_prompt_row_id(prompt_row: Dict[str, Any], idx: int) -> str:
    if "id" in prompt_row and prompt_row["id"] is not None:
        return str(prompt_row["id"])
    return f"row-{idx}"


def get_completion_row_id(completion_row: Dict[str, Any], idx: int) -> str:
    if "custom_id" in completion_row and completion_row["custom_id"] is not None:
        return str(completion_row["custom_id"])
    return f"row-{idx}"


def get_validation_base_id(validation_row: Dict[str, Any], idx: int) -> str:
    custom_id = validation_row.get("custom_id")
    if custom_id is None:
        custom_id = f"row-{idx}"
    custom_id = str(custom_id)

    prefix = "validate-"
    if custom_id.startswith(prefix):
        return custom_id[len(prefix):]
    return custom_id


def rows_have_explicit_ids(prompt_rows: List[Dict[str, Any]], completion_rows: List[Dict[str, Any]]) -> bool:
    prompt_has_ids = all("id" in row and row["id"] is not None for row in prompt_rows)
    completion_has_ids = all("custom_id" in row and row["custom_id"] is not None for row in completion_rows)
    return prompt_has_ids and completion_has_ids


# =========================================================
# Response extraction
# =========================================================

def extract_text_from_response_obj(obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract text from a common OpenAI batch output row.
    """
    response = obj.get("response", {})
    body = response.get("body", {})

    # Responses API output array
    output = body.get("output")
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                texts.append(item["text"])

            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        texts.append(c["text"])

        text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if text:
            return text

    # body.content
    content = body.get("content")
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "output_text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
        text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if text:
            return text

    # direct body text fields
    for key in ("text", "output_text"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # older chat-style choices
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        texts: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if isinstance(message, dict):
                msg_content = message.get("content")
                if isinstance(msg_content, str):
                    texts.append(msg_content)
                elif isinstance(msg_content, list):
                    for c in msg_content:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            texts.append(c["text"])

        text = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if text:
            return text

    return None


def extract_model_from_response_obj(obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract model name from a batch output row.
    """
    response = obj.get("response", {})
    body = response.get("body", {})

    for key in ("model",):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Some shapes may tuck metadata elsewhere
    if isinstance(response.get("model"), str) and response["model"].strip():
        return response["model"].strip()

    return None


# =========================================================
# Validation parsing
# =========================================================

def parse_validation_json(text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "empty_validation_text"

    raw = text.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()

    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"invalid_validation_json: {e}"


def normalize_bool_dict(d: Any) -> Dict[str, bool]:
    if not isinstance(d, dict):
        return {}
    return {str(k): bool(v) for k, v in d.items()}


def normalize_notes(notes: Any) -> List[str]:
    if notes is None:
        return []
    if isinstance(notes, list):
        return [str(x) for x in notes]
    return [str(notes)]


# =========================================================
# Joining
# =========================================================

def join_prompts_and_completions(
    prompt_rows: List[Dict[str, Any]],
    completion_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    joined[id] = {
      "instruction": <prompt row>,
      "completion_row": <raw completion row>,
      "story": <generated text>,
      "source": <model name>
    }
    """
    warnings: List[str] = []
    joined: Dict[str, Dict[str, Any]] = {}

    if rows_have_explicit_ids(prompt_rows, completion_rows):
        prompt_index = {get_prompt_row_id(row, i): row for i, row in enumerate(prompt_rows, start=1)}
        completion_index = {get_completion_row_id(row, i): row for i, row in enumerate(completion_rows, start=1)}

        for pid, prompt_row in prompt_index.items():
            completion_row = completion_index.get(pid)
            if completion_row is None:
                warnings.append(f"Missing completion for prompt id: {pid}")
                continue

            story = extract_text_from_response_obj(completion_row)
            source = extract_model_from_response_obj(completion_row)

            if not story:
                warnings.append(f"Could not extract completion text for prompt id: {pid}")
            if not source:
                warnings.append(f"Could not extract completion model for prompt id: {pid}")

            joined[pid] = {
                "instruction": prompt_row,
                "completion_row": completion_row,
                "story": story,
                "source": source,
            }
    else:
        n = min(len(prompt_rows), len(completion_rows))
        if len(prompt_rows) != len(completion_rows):
            warnings.append(
                f"Prompt row count ({len(prompt_rows)}) != completion row count ({len(completion_rows)}); "
                f"joining by order up to {n}"
            )

        for idx in range(n):
            prompt_row = prompt_rows[idx]
            completion_row = completion_rows[idx]
            pid = get_prompt_row_id(prompt_row, idx + 1)

            story = extract_text_from_response_obj(completion_row)
            source = extract_model_from_response_obj(completion_row)

            if not story:
                warnings.append(f"Could not extract completion text for prompt row: {idx + 1}")
            if not source:
                warnings.append(f"Could not extract completion model for prompt row: {idx + 1}")

            joined[pid] = {
                "instruction": prompt_row,
                "completion_row": completion_row,
                "story": story,
                "source": source,
            }

    return joined, warnings


def build_validation_index(validation_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(validation_rows, start=1):
        vid = get_validation_base_id(row, idx)
        if vid in index:
            raise ValueError(f"Duplicate validation id found after stripping prefix: {vid}")
        index[vid] = row
    return index


# =========================================================
# Failure summarization
# =========================================================

def summarize_failure_reasons(
    subject_correct: bool,
    required_words: Dict[str, bool],
    pos_correct: Dict[str, bool],
    features: Dict[str, bool],
    overall_pass: bool,
    validator_notes: List[str],
    parse_error: Optional[str],
) -> List[str]:
    reasons: List[str] = []

    if parse_error:
        reasons.append(parse_error)
        return reasons

    if not subject_correct:
        reasons.append("subject_incorrect")

    for word, ok in required_words.items():
        if not ok:
            reasons.append(f"missing_required_word:{word}")

    for word, ok in pos_correct.items():
        if not ok:
            reasons.append(f"wrong_pos:{word}")

    for feature, ok in features.items():
        if not ok:
            reasons.append(f"missing_feature:{feature}")

    if not overall_pass and not reasons:
        reasons.append("overall_fail_unspecified")

    reasons.extend(validator_notes)
    return reasons


# =========================================================
# Main merge
# =========================================================

def build_outputs(
    prompt_rows: List[Dict[str, Any]],
    completion_rows: List[Dict[str, Any]],
    validation_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Returns:
      passed_json_rows: list for passed.json
      failed_jsonl_rows: list for failed.jsonl
      warnings
    """
    warnings: List[str] = []

    joined_pc, join_warnings = join_prompts_and_completions(prompt_rows, completion_rows)
    warnings.extend(join_warnings)

    validation_index = build_validation_index(validation_rows)

    passed_json_rows: List[Dict[str, Any]] = []
    failed_jsonl_rows: List[Dict[str, Any]] = []

    for pid, bundle in joined_pc.items():
        instruction = bundle["instruction"]
        story = bundle["story"]
        source = bundle["source"]
        validation_row = validation_index.get(pid)

        validation_text: Optional[str] = None
        validation_obj: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None

        if validation_row is None:
            parse_error = "missing_validation_row"
            warnings.append(f"Missing validation row for id: {pid}")
        else:
            validation_text = extract_text_from_response_obj(validation_row)
            validation_obj, parse_error = parse_validation_json(validation_text)
            if parse_error:
                warnings.append(f"Validation parse issue for id {pid}: {parse_error}")

        if validation_obj is None:
            subject_correct = False
            required_words = {}
            pos_correct = {}
            features = {}
            overall_pass = False
            validator_notes = []
        else:
            subject_correct = bool(validation_obj.get("subject_correct", False))
            required_words = normalize_bool_dict(validation_obj.get("required_words", {}))
            pos_correct = normalize_bool_dict(validation_obj.get("pos_correct", {}))
            features = normalize_bool_dict(validation_obj.get("features", {}))
            overall_pass = bool(validation_obj.get("overall_pass", False))
            validator_notes = normalize_notes(validation_obj.get("notes"))

        failure_reasons = summarize_failure_reasons(
            subject_correct=subject_correct,
            required_words=required_words,
            pos_correct=pos_correct,
            features=features,
            overall_pass=overall_pass,
            validator_notes=validator_notes,
            parse_error=parse_error,
        )

        if overall_pass:
            passed_json_rows.append(
                {
                    "story": story,
                    "instruction": instruction,
                    "source": source,
                }
            )
        else:
            failed_jsonl_rows.append(
                {
                    "id": pid,
                    "story": story,
                    "instruction": instruction,
                    "source": source,
                    "validation": {
                        "raw_text": validation_text,
                        "parsed": validation_obj,
                        "parse_error": parse_error,
                    },
                    "result": {
                        "passed": False,
                        "subject_correct": subject_correct,
                        "required_words": required_words,
                        "pos_correct": pos_correct,
                        "features": features,
                        "failure_reasons": failure_reasons,
                    },
                }
            )

    return passed_json_rows, failed_jsonl_rows, warnings


# =========================================================
# Summary
# =========================================================

def build_summary(
    passed_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    total = len(passed_rows) + len(failed_rows)
    passed = len(passed_rows)
    failed = len(failed_rows)

    pass_pct = (passed / total * 100.0) if total else 0.0
    fail_pct = (failed / total * 100.0) if total else 0.0

    # Count "how many completions failed due to X", not total occurrences of X.
    subject_fail_count = 0
    parse_error_count = 0
    missing_required_word_count = 0
    wrong_pos_count = 0
    missing_feature_count = 0
    unspecified_fail_count = 0

    for row in failed_rows:
        validation = row.get("validation", {})
        result = row.get("result", {})

        if validation.get("parse_error"):
            parse_error_count += 1

        if not result.get("subject_correct", True):
            subject_fail_count += 1

        required_words = result.get("required_words", {}) or {}
        if any(not ok for ok in required_words.values()):
            missing_required_word_count += 1

        pos_correct = result.get("pos_correct", {}) or {}
        if any(not ok for ok in pos_correct.values()):
            wrong_pos_count += 1

        features = result.get("features", {}) or {}
        if any(not ok for ok in features.values()):
            missing_feature_count += 1

        failure_reasons = result.get("failure_reasons", []) or []
        if (
            not validation.get("parse_error")
            and result.get("subject_correct", True)
            and not any(not ok for ok in required_words.values())
            and not any(not ok for ok in pos_correct.values())
            and not any(not ok for ok in features.values())
            and failure_reasons
        ):
            unspecified_fail_count += 1

    return {
        "counts": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "passed_pct": round(pass_pct, 2),
            "failed_pct": round(fail_pct, 2),
            "warnings": len(warnings),
        },
        "failed_due_to": {
            "validation_parse_error": parse_error_count,
            "subject_incorrect": subject_fail_count,
            "missing_required_word": missing_required_word_count,
            "wrong_part_of_speech": wrong_pos_count,
            "missing_feature": missing_feature_count,
            "other_or_unspecified": unspecified_fail_count,
        },
    }
# =========================================================
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse validation results into passed.json and failed.jsonl."
    )
    parser.add_argument("--prompts", required=True, help="Path to prompt JSONL file.")
    parser.add_argument("--completions", required=True, help="Path to generation batch output JSONL file.")
    parser.add_argument("--validations", required=True, help="Path to validation batch output JSONL file.")
    parser.add_argument("--passed-results", required=True, help="Path to output passed .json or .jsonl file.")
    parser.add_argument("--failed-results", required=True, help="Path to output failed .json or .jsonl file.")
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to output summary .json file.",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    completions_path = Path(args.completions)
    validations_path = Path(args.validations)
    passed_path = Path(args.passed_results)
    failed_path = Path(args.failed_results)
    summary_path = Path(args.summary) if args.summary else None

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    if not completions_path.exists():
        raise FileNotFoundError(f"Completions file not found: {completions_path}")
    if not validations_path.exists():
        raise FileNotFoundError(f"Validations file not found: {validations_path}")

    prompt_rows = load_json(prompts_path)
    completion_rows = load_json(completions_path)
    validation_rows = load_json(validations_path)

    passed_rows, failed_rows, warnings = build_outputs(
        prompt_rows=prompt_rows,
        completion_rows=completion_rows,
        validation_rows=validation_rows,
    )

    passed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(passed_path, passed_rows)
    failed_count = write_json(failed_path, failed_rows)

    print(f"Wrote passed JSON:   {passed_path} ({len(passed_rows)} rows)")
    print(f"Wrote failed JSONL:  {failed_path} ({failed_count} rows)")

    if summary_path:
        summary = build_summary(passed_rows, failed_rows, warnings)
        write_json(summary_path, summary)
        print(f"Wrote summary JSON:  {summary_path}")

        counts = summary["counts"]
        print(
            f"Total={counts['total']}  "
            f"Passed={counts['passed']} ({counts['passed_pct']}%)  "
            f"Failed={counts['failed']} ({counts['failed_pct']}%)"
        )


if __name__ == "__main__":
    main()