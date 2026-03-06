#!/usr/bin/env python3
"""
Heuristic reward function for GRPO fine-tuning on TinyStories.

This script computes a reward for (prompt, completion) pairs without using an
LLM judge. It is designed for small story-generation models where we want to
reward:
    - prompt adherence
    - coherence / basic story structure
    - TinyStories-like style
    - appropriate length
    - fluency / cleanliness
    - safety
and penalize:
    - repetition / degeneracy

The output reward is normalized to roughly [-1, 1].

Usage:
    python reward_tinystories.py --prompt "Write a story about a cat who learns to share." \
                                 --completion "Milo was a little cat..."

    python reward_tinystories.py --input_jsonl samples.jsonl --output_jsonl scored.jsonl

Expected JSONL input format:
    {"prompt": "...", "completion": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# -----------------------------
# Basic text utilities
# -----------------------------

WORD_RE = re.compile(r"\b[a-zA-Z']+\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_SPAM_RE = re.compile(r"([!?.,])\1{2,}")
WEIRD_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s.,!?;:'\"()\-\n]")


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as",
    "of", "to", "in", "on", "at", "for", "from", "with", "by", "about", "into",
    "over", "after", "before", "under", "again", "further", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "than", "too",
    "very", "can", "will", "just", "it", "its", "is", "am", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "he", "she", "they", "them", "his", "her", "their", "i", "you", "we", "our",
    "this", "that", "these", "those", "my", "your", "me", "him", "who", "what",
    "when", "where", "why", "how", "write", "story", "short", "little"
}


# TinyStories-ish positive cues
POSITIVE_STYLE_WORDS = {
    "happy", "smiled", "smile", "kind", "gentle", "help", "helped", "friend",
    "friends", "share", "shared", "learned", "learn", "love", "loved", "hug",
    "hugged", "play", "played", "mom", "dad", "mother", "father", "home",
    "toy", "garden", "flower", "puppy", "dog", "cat", "bird", "sun", "day",
    "night", "tree", "book", "ball", "good", "nice"
}

# Strongly undesirable for TinyStories
UNSAFE_WORDS = {
    "blood", "gore", "kill", "killed", "murder", "stab", "stabbed", "gun",
    "rifle", "suicide", "sex", "sexual", "naked", "porn", "hate", "hated",
    "racist", "abuse", "abused", "beer", "vodka", "drugs", "cocaine", "heroin"
}

NEGATIVE_TONE_WORDS = {
    "dead", "death", "horrible", "terrible", "violent", "violence", "scream",
    "screamed", "monster", "evil", "darkness", "destroy", "destroyed"
}

STORY_START_MARKERS = {
    "one", "once", "today", "there", "it", "on", "in"
}
STORY_END_MARKERS = {
    "happy", "smiled", "learned", "sleep", "asleep", "home", "end", "better"
}
TEMPORAL_WORDS = {
    "then", "next", "after", "afterward", "later", "soon", "finally"
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    sents = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]


def unique_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def get_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


# -----------------------------
# Keyword extraction / prompt adherence
# -----------------------------

def extract_prompt_keywords(prompt: str) -> List[str]:
    toks = words(prompt)
    candidates = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    counts = Counter(candidates)
    # Keep the most important-looking prompt words.
    ranked = [w for w, _ in counts.most_common(12)]
    return ranked


def prompt_keyword_coverage(prompt: str, completion: str) -> float:
    prompt_keywords = extract_prompt_keywords(prompt)
    if not prompt_keywords:
        return 0.5

    completion_tokens = set(words(completion))
    matched = 0
    for kw in prompt_keywords:
        if kw in completion_tokens:
            matched += 1

    raw = matched / len(prompt_keywords)

    # Slight boost when early keywords appear; often the first few words matter more.
    first_few = prompt_keywords[:4]
    first_few_match = sum(1 for kw in first_few if kw in completion_tokens)
    first_few_score = safe_div(first_few_match, len(first_few), 0.0)

    score = 0.75 * raw + 0.25 * first_few_score
    return clamp(score)


def prompt_verb_overlap(prompt: str, completion: str) -> float:
    # Cheap approximation: look for action words from prompt that reappear.
    # Not true POS tagging, just a rough heuristic.
    likely_verbs = {
        t for t in words(prompt)
        if t.endswith(("e", "ed", "ing")) or t in {
            "find", "help", "share", "learn", "play", "plant", "save",
            "look", "walk", "run", "jump", "give", "take", "make"
        }
    }
    if not likely_verbs:
        return 0.5

    completion_tokens = set(words(completion))
    matched = sum(1 for v in likely_verbs if v in completion_tokens)
    return clamp(matched / len(likely_verbs))


def score_prompt_adherence(prompt: str, completion: str) -> float:
    kw = prompt_keyword_coverage(prompt, completion)
    verb = prompt_verb_overlap(prompt, completion)
    return clamp(0.8 * kw + 0.2 * verb)


# -----------------------------
# Coherence / story structure
# -----------------------------

def sentence_overlap(a: str, b: str) -> float:
    wa = set(words(a))
    wb = set(words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def score_coherence(completion: str) -> float:
    sents = sentences(completion)
    toks = words(completion)

    if len(toks) < 15:
        return 0.1
    if len(sents) == 0:
        return 0.0
    if len(sents) == 1:
        return 0.35

    overlaps = [
        sentence_overlap(sents[i], sents[i + 1])
        for i in range(len(sents) - 1)
    ]
    avg_overlap = sum(overlaps) / len(overlaps)

    # Too little overlap => random drift.
    # Too much overlap => repetitive sludge.
    if avg_overlap < 0.03:
        flow_score = 0.2
    elif avg_overlap < 0.08:
        flow_score = 0.5
    elif avg_overlap < 0.35:
        flow_score = 1.0
    elif avg_overlap < 0.6:
        flow_score = 0.6
    else:
        flow_score = 0.2

    first_words = set(words(sents[0]))
    last_words = set(words(sents[-1]))
    opening_bonus = 1.0 if first_words & STORY_START_MARKERS else 0.6
    ending_bonus = 1.0 if last_words & STORY_END_MARKERS else 0.6

    temporal_hits = sum(1 for t in words(completion) if t in TEMPORAL_WORDS)
    temporal_score = clamp(temporal_hits / 2.0)

    # Reward simple story shape: setup + progression + ending.
    score = (
        0.50 * flow_score +
        0.20 * opening_bonus +
        0.20 * ending_bonus +
        0.10 * temporal_score
    )
    return clamp(score)


# -----------------------------
# TinyStories style
# -----------------------------

def average_sentence_length(text: str) -> float:
    sents = sentences(text)
    if not sents:
        return 0.0
    sent_lengths = [len(words(s)) for s in sents]
    return sum(sent_lengths) / len(sent_lengths)


def average_word_length(text: str) -> float:
    toks = words(text)
    if not toks:
        return 0.0
    return sum(len(t) for t in toks) / len(toks)


def score_style(completion: str) -> float:
    toks = words(completion)
    if not toks:
        return 0.0

    avg_sent_len = average_sentence_length(completion)
    avg_word_len = average_word_length(completion)

    # TinyStories generally wants short-ish sentences and simple words.
    if avg_sent_len <= 6:
        sent_score = 0.7
    elif avg_sent_len <= 14:
        sent_score = 1.0
    elif avg_sent_len <= 20:
        sent_score = 0.6
    else:
        sent_score = 0.2

    if avg_word_len <= 4.7:
        word_score = 1.0
    elif avg_word_len <= 5.3:
        word_score = 0.8
    elif avg_word_len <= 6.0:
        word_score = 0.5
    else:
        word_score = 0.2

    pos_hits = sum(1 for t in toks if t in POSITIVE_STYLE_WORDS)
    neg_hits = sum(1 for t in toks if t in NEGATIVE_TONE_WORDS)

    pos_score = clamp(pos_hits / max(3.0, len(toks) * 0.03))
    neg_penalty = clamp(neg_hits / max(2.0, len(toks) * 0.02))

    # Mild reward for dialogue; common in children's stories, but not mandatory.
    dialogue_bonus = 1.0 if '"' in completion or "'" in completion else 0.6

    score = (
        0.35 * sent_score +
        0.25 * word_score +
        0.20 * pos_score +
        0.10 * dialogue_bonus +
        0.10 * (1.0 - neg_penalty)
    )
    return clamp(score)


# -----------------------------
# Length
# -----------------------------

def score_length(completion: str, preferred_min: int = 60, preferred_max: int = 180) -> float:
    n = len(words(completion))
    if preferred_min <= n <= preferred_max:
        return 1.0
    if 40 <= n < preferred_min:
        return 0.6 + 0.4 * (n - 40) / max(1, preferred_min - 40)
    if preferred_max < n <= 220:
        return 0.6 + 0.4 * (220 - n) / max(1, 220 - preferred_max)
    if 20 <= n < 40:
        return 0.25 + 0.35 * (n - 20) / 20
    if 220 < n <= 300:
        return 0.25 + 0.35 * (300 - n) / 80
    return 0.0


# -----------------------------
# Safety
# -----------------------------

def score_safety(completion: str) -> float:
    toks = set(words(completion))
    unsafe_hits = sum(1 for w in UNSAFE_WORDS if w in toks)
    if unsafe_hits == 0:
        return 1.0
    if unsafe_hits == 1:
        return 0.3
    return 0.0


# -----------------------------
# Fluency / cleanliness
# -----------------------------

def grammarish_score(completion: str) -> float:
    text = normalize_text(completion)
    if not text:
        return 0.0

    sents = sentences(text)
    toks = words(text)

    if not toks:
        return 0.0

    capitalization_score = 0.0
    if sents:
        capitalized = sum(1 for s in sents if s and s[0].isupper())
        capitalization_score = capitalized / len(sents)

    punctuation_score = 0.0
    ended = sum(1 for s in sents if s[-1] in ".!?")
    if sents:
        punctuation_score = ended / len(sents)

    weird_chars = WEIRD_CHAR_RE.findall(text)
    weird_char_penalty = clamp(len(weird_chars) / max(1.0, len(text) * 0.02))

    punct_spam = len(PUNCT_SPAM_RE.findall(text))
    punct_spam_penalty = clamp(punct_spam / max(1.0, len(sents)))

    score = (
        0.40 * capitalization_score +
        0.40 * punctuation_score +
        0.20 * (1.0 - max(weird_char_penalty, punct_spam_penalty))
    )
    return clamp(score)


# -----------------------------
# Degeneracy / repetition
# -----------------------------

def repeated_ngram_rate(tokens: Sequence[str], n: int) -> float:
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return clamp(repeated / len(ngrams))


def repeated_sentence_rate(text: str) -> float:
    sents = [s.lower() for s in sentences(text)]
    if not sents:
        return 0.0
    counts = Counter(sents)
    repeated = sum(c for c in counts.values() if c > 1)
    return clamp(repeated / len(sents))


def token_repetition_rate(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return clamp(1.0 - unique_ratio(tokens))


def trailing_fragment_penalty(text: str) -> float:
    text = normalize_text(text)
    if not text:
        return 1.0
    # Penalize if it ends without terminal punctuation and is long enough
    if len(words(text)) >= 20 and text[-1] not in ".!?":
        return 0.7
    return 0.0


def score_degeneracy_penalty(completion: str) -> float:
    toks = words(completion)
    if not toks:
        return 1.0

    rep2 = repeated_ngram_rate(toks, 2)
    rep3 = repeated_ngram_rate(toks, 3)
    rep4 = repeated_ngram_rate(toks, 4)
    sent_rep = repeated_sentence_rate(completion)
    tok_rep = token_repetition_rate(toks)
    punct_spam = 1.0 if PUNCT_SPAM_RE.search(completion) else 0.0
    trail = trailing_fragment_penalty(completion)

    penalty = (
        0.20 * rep2 +
        0.20 * rep3 +
        0.20 * rep4 +
        0.15 * sent_rep +
        0.15 * tok_rep +
        0.05 * punct_spam +
        0.05 * trail
    )
    return clamp(penalty)


# -----------------------------
# Main reward logic
# -----------------------------

@dataclass
class RewardBreakdown:
    prompt_adherence: float
    coherence: float
    style: float
    length: float
    safety: float
    fluency: float
    degeneracy_penalty: float
    raw_reward_0_1: float
    reward_minus1_1: float


def compute_reward(
    prompt: str,
    completion: str,
    weights: Dict[str, float] | None = None,
) -> RewardBreakdown:
    """
    Compute a heuristic reward for a prompt/completion pair.

    Default weights:
        0.35 prompt adherence
        0.25 coherence
        0.15 style
        0.10 length
        0.05 safety
        0.10 fluency
       -0.25 degeneracy penalty
    """
    prompt = normalize_text(prompt)
    completion = normalize_text(completion)

    if weights is None:
        weights = {
            "prompt_adherence": 0.35,
            "coherence": 0.25,
            "style": 0.15,
            "length": 0.10,
            "safety": 0.05,
            "fluency": 0.10,
            "degeneracy_penalty": 0.25,
        }

    s_prompt = score_prompt_adherence(prompt, completion)
    s_coherence = score_coherence(completion)
    s_style = score_style(completion)
    s_length = score_length(completion)
    s_safety = score_safety(completion)
    s_fluency = grammarish_score(completion)
    p_deg = score_degeneracy_penalty(completion)

    raw = (
        weights["prompt_adherence"] * s_prompt +
        weights["coherence"] * s_coherence +
        weights["style"] * s_style +
        weights["length"] * s_length +
        weights["safety"] * s_safety +
        weights["fluency"] * s_fluency -
        weights["degeneracy_penalty"] * p_deg
    )

    raw = clamp(raw, 0.0, 1.0)
    reward = 2.0 * raw - 1.0

    return RewardBreakdown(
        prompt_adherence=round(s_prompt, 6),
        coherence=round(s_coherence, 6),
        style=round(s_style, 6),
        length=round(s_length, 6),
        safety=round(s_safety, 6),
        fluency=round(s_fluency, 6),
        degeneracy_penalty=round(p_deg, 6),
        raw_reward_0_1=round(raw, 6),
        reward_minus1_1=round(reward, 6),
    )


# -----------------------------
# Batch file processing
# -----------------------------

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_jsonl(input_path: Path, output_path: Path) -> None:
    def rows():
        for obj in iter_jsonl(input_path):
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            breakdown = compute_reward(prompt, completion)
            out = dict(obj)
            out["reward"] = breakdown.reward_minus1_1
            out["reward_breakdown"] = asdict(breakdown)
            yield out

    write_jsonl(output_path, rows())


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Heuristic TinyStories GRPO reward calculator")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt to score")
    p.add_argument("--completion", type=str, default=None, help="Single completion to score")
    p.add_argument("--input_jsonl", type=str, default=None, help="Input JSONL with prompt/completion fields")
    p.add_argument("--output_jsonl", type=str, default=None, help="Output JSONL with reward fields")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output for single example")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    single_mode = args.prompt is not None or args.completion is not None
    batch_mode = args.input_jsonl is not None or args.output_jsonl is not None

    if single_mode and batch_mode:
        raise SystemExit("Use either single-example mode or JSONL batch mode, not both.")

    if single_mode:
        if args.prompt is None or args.completion is None:
            raise SystemExit("Single-example mode requires both --prompt and --completion.")
        breakdown = compute_reward(args.prompt, args.completion)
        payload = {
            "prompt": args.prompt,
            "completion": args.completion,
            "reward": breakdown.reward_minus1_1,
            "reward_breakdown": asdict(breakdown),
        }
        if args.pretty:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return

    if batch_mode:
        if args.input_jsonl is None or args.output_jsonl is None:
            raise SystemExit("Batch mode requires both --input_jsonl and --output_jsonl.")
        score_jsonl(Path(args.input_jsonl), Path(args.output_jsonl))
        return

    raise SystemExit(
        "Provide either:\n"
        "  --prompt ... --completion ...\n"
        "or\n"
        "  --input_jsonl ... --output_jsonl ..."
    )


if __name__ == "__main__":
    main()