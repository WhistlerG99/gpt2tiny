import argparse
import json
from typing import List, Dict, Any, Tuple, Iterable
from pathlib import Path

DEFAULT_MODEL = "gpt-5.4"   # replace with the teacher model you want to use
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_OUTPUT_TOKENS=248

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



def build_prompt(subject: str, required_words: List[Dict[str,str]], features: list[str]) -> str:
    word_lines = "\n".join(f"- Use the exact word \"{w['word']}\" as a {w['pos']}" for w in required_words)

    feature_lines = "\n".join(f"- Include {f}" for f in features)

    if sum([subject is None, word_lines == "" , feature_lines==""])>2:
        return None
    
    prompt = f"""Write a very short fictional story.
    
Requirements:"""
    if subject:
        prompt += f"\n- The story should be about {subject}"
    if word_lines:
        prompt += f"\n{word_lines}"
    if feature_lines:
        prompt += f"\n{feature_lines}"   
    prompt += f"""
- Use simple English
- Length: 3 to 5 sentences
- Output only the story
"""
    
    return prompt


def format_prompt_rows(
    prompts: List[Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> Tuple[List[Dict[str, Any]], List[str]]:    
    """
    Each line is one request for the Batch API.
    """    
    prompt_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for i, item in enumerate(prompts):
        prompt = build_prompt(
            subject=item["subject_clause"],
            required_words=item["words"],
            features=item["feature_phrases"],
        )
        if prompt is None:
            warnings.append(f"prompt {i} is missing")
            continue            
        req = {
            "custom_id": f"prompt-{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        prompt_rows.append(req)

    return prompt_rows, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a prompt batch JSONL from openAI batch job."
    )
    parser.add_argument("--prompts", required=True, help="Path to prompt JSON or JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output prompt batch JSONL file.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name for completion requests (default: {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per response (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Model sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling percentile threshold (default: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--warnings-file",
        default=None,
        help="Optional path to write warnings/skipped-record messages.",
    )    
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_path = Path(args.output)

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    
    prompts = load_json(prompts_path)

    prompt_rows, warnings = format_prompt_rows(
        prompts,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(output_path, prompt_rows)

    print(f"Wrote {written} validation batch requests to {output_path}")
    print(f"Loaded {len(prompt_rows)} prompt rows")
    print(f"Warnings: {len(warnings)}")

    if args.warnings_file and len(warnings)>0:
        warnings_path = Path(args.warnings_file)
        warnings_path.write_text("\n".join(warnings), encoding="utf-8")
        print(f"\nWrote warnings to {warnings_path}")

if __name__ == "__main__":
    main()
    






