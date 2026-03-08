from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class RewardOutput:
    prompt_texts: List[str]
    completion_texts: List[str]
    rewards: torch.Tensor
    raw_rewards_0_1: torch.Tensor
    prompt_adherence_rewards: torch.Tensor
    coherence_rewards: torch.Tensor
    style_rewards: torch.Tensor
    length_rewards: torch.Tensor
    safety_rewards: torch.Tensor
    fluency_rewards: torch.Tensor
    degeneracy_penalty_rewards: torch.Tensor


@dataclass
class GroupRewardOutput:
    rewards: torch.Tensor
    group_means: torch.Tensor
    group_stds: torch.Tensor
    group_advantages: torch.Tensor
    group_indices: torch.Tensor
    group_sizes: torch.Tensor


class Reward:
    """
    Heuristic reward calculator for TinyStories-style GRPO training.

    Works with:
      - Hugging Face BPE tokenizers
      - Hugging Face SentencePiece tokenizers
      - custom SentencePieceProcessor-like tokenizers exposing DecodeIds()

    Scores directly from:
      - concatenated prompt+generated token ids
      - attention mask
      - prompt lengths
    """

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

    POSITIVE_STYLE_WORDS = {
        "happy", "smiled", "smile", "kind", "gentle", "help", "helped", "friend",
        "friends", "share", "shared", "learned", "learn", "love", "loved", "hug",
        "hugged", "play", "played", "mom", "dad", "mother", "father", "home",
        "toy", "garden", "flower", "puppy", "dog", "cat", "bird", "sun", "day",
        "night", "tree", "book", "ball", "good", "nice"
    }

    UNSAFE_WORDS = {
        "blood", "gore", "kill", "killed", "murder", "stab", "stabbed", "gun",
        "rifle", "suicide", "sex", "sexual", "naked", "porn", "hate", "hated",
        "racist", "abuse", "abused", "beer", "vodka", "drugs", "cocaine", "heroin"
    }

    NEGATIVE_TONE_WORDS = {
        "dead", "death", "horrible", "terrible", "violent", "violence", "scream",
        "screamed", "monster", "evil", "darkness", "destroy", "destroyed"
    }

    STORY_START_MARKERS = {"one", "once", "today", "there", "it", "on", "in"}
    STORY_END_MARKERS = {"happy", "smiled", "learned", "sleep", "asleep", "home", "end", "better"}
    TEMPORAL_WORDS = {"then", "next", "after", "afterward", "later", "soon", "finally"}

    def __init__(
        self,
        tokenizer,
        *,
        preferred_min_words: int = 60,
        preferred_max_words: int = 180,
        reward_weights: Optional[Dict[str, float]] = None,
        clamp_rewards_to_minus1_1: bool = True,
        group_std_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        stop_completion_at_eos: bool = True,
        strip_bos_from_prompt: bool = False,
    ):
        self.tokenizer = tokenizer
        self.preferred_min_words = preferred_min_words
        self.preferred_max_words = preferred_max_words
        self.clamp_rewards_to_minus1_1 = clamp_rewards_to_minus1_1
        self.group_std_eps = group_std_eps
        self.device = device
        self.stop_completion_at_eos = stop_completion_at_eos
        self.strip_bos_from_prompt = strip_bos_from_prompt

        self.eos_token_id = eos_token_id if eos_token_id is not None else self._infer_special_id("eos_token_id")
        self.pad_token_id = pad_token_id if pad_token_id is not None else self._infer_special_id("pad_token_id")
        self.bos_token_id = bos_token_id if bos_token_id is not None else self._infer_special_id("bos_token_id")

        self.reward_weights = reward_weights or {
            "prompt_adherence": 0.35,
            "coherence": 0.25,
            "style": 0.15,
            "length": 0.10,
            "safety": 0.05,
            "fluency": 0.10,
            "degeneracy_penalty": 0.25,
        }

    # =========================================================
    # Public API
    # =========================================================

    @torch.no_grad()
    def score_texts(
        self,
        prompt_texts: Sequence[str],
        completion_texts: Sequence[str],
        *,
        device: Optional[torch.device] = None,
    ) -> RewardOutput:
        if len(prompt_texts) != len(completion_texts):
            raise ValueError("prompt_texts and completion_texts must have the same length")

        device = self._resolve_device(device)

        prompt_scores = []
        coherence_scores = []
        style_scores = []
        length_scores = []
        safety_scores = []
        fluency_scores = []
        degeneracy_penalties = []
        raw_rewards = []
        rewards = []

        normalized_prompts = [self._normalize_text(x) for x in prompt_texts]
        normalized_completions = [self._normalize_text(x) for x in completion_texts]

        for prompt, completion in zip(normalized_prompts, normalized_completions):
            s_prompt = self._score_prompt_adherence(prompt, completion)
            s_coherence = self._score_coherence(completion)
            s_style = self._score_style(completion)
            s_length = self._score_length(completion)
            s_safety = self._score_safety(completion)
            s_fluency = self._score_fluency(completion)
            p_deg = self._score_degeneracy_penalty(completion)

            raw = (
                self.reward_weights["prompt_adherence"] * s_prompt
                + self.reward_weights["coherence"] * s_coherence
                + self.reward_weights["style"] * s_style
                + self.reward_weights["length"] * s_length
                + self.reward_weights["safety"] * s_safety
                + self.reward_weights["fluency"] * s_fluency
                - self.reward_weights["degeneracy_penalty"] * p_deg
            )
            raw = self._clamp(raw, 0.0, 1.0)

            reward = 2.0 * raw - 1.0
            if self.clamp_rewards_to_minus1_1:
                reward = self._clamp(reward, -1.0, 1.0)

            prompt_scores.append(s_prompt)
            coherence_scores.append(s_coherence)
            style_scores.append(s_style)
            length_scores.append(s_length)
            safety_scores.append(s_safety)
            fluency_scores.append(s_fluency)
            degeneracy_penalties.append(p_deg)
            raw_rewards.append(raw)
            rewards.append(reward)

        return RewardOutput(
            prompt_texts=list(normalized_prompts),
            completion_texts=list(normalized_completions),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            raw_rewards_0_1=torch.tensor(raw_rewards, dtype=torch.float32, device=device),
            prompt_adherence_rewards=torch.tensor(prompt_scores, dtype=torch.float32, device=device),
            coherence_rewards=torch.tensor(coherence_scores, dtype=torch.float32, device=device),
            style_rewards=torch.tensor(style_scores, dtype=torch.float32, device=device),
            length_rewards=torch.tensor(length_scores, dtype=torch.float32, device=device),
            safety_rewards=torch.tensor(safety_scores, dtype=torch.float32, device=device),
            fluency_rewards=torch.tensor(fluency_scores, dtype=torch.float32, device=device),
            degeneracy_penalty_rewards=torch.tensor(degeneracy_penalties, dtype=torch.float32, device=device),
        )

    @torch.no_grad()
    def score_from_concat_ids(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
        *,
        skip_special_tokens: bool = True,
        stop_completion_at_eos: Optional[bool] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> RewardOutput:
        """
        Score from concatenated prompt+generated ids, attention mask, and prompt lengths.

        Shapes:
            sequences:      [B, T]
            attention_mask: [B, T]
            prompt_lens:    [B]
        """
        self._validate_concat_inputs(sequences, attention_mask, prompt_lens)

        if stop_completion_at_eos is None:
            stop_completion_at_eos = self.stop_completion_at_eos
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if bos_token_id is None:
            bos_token_id = self.bos_token_id

        prompt_texts: List[str] = []
        completion_texts: List[str] = []

        batch_size = sequences.size(0)

        for i in range(batch_size):
            seq_i = sequences[i]
            mask_i = attention_mask[i]
            prompt_len_i = int(prompt_lens[i].item())

            valid_len = int(mask_i.long().sum().item())
            if valid_len < 0 or valid_len > seq_i.numel():
                raise ValueError(f"Invalid valid_len={valid_len} for row {i}")

            seq_valid = seq_i[:valid_len]

            if prompt_len_i < 0 or prompt_len_i > valid_len:
                raise ValueError(
                    f"Invalid prompt length for row {i}: prompt_len={prompt_len_i}, valid_len={valid_len}"
                )

            prompt_ids = seq_valid[:prompt_len_i]
            completion_ids = seq_valid[prompt_len_i:]

            if self.strip_bos_from_prompt and bos_token_id is not None and prompt_ids.numel() > 0:
                if int(prompt_ids[0].item()) == bos_token_id:
                    prompt_ids = prompt_ids[1:]

            if pad_token_id is not None and completion_ids.numel() > 0:
                completion_ids = completion_ids[completion_ids != pad_token_id]

            if bos_token_id is not None and completion_ids.numel() > 0:
                # Rare, but some pipelines accidentally leave BOS at the start of the generated piece.
                if int(completion_ids[0].item()) == bos_token_id:
                    completion_ids = completion_ids[1:]

            if stop_completion_at_eos and eos_token_id is not None and completion_ids.numel() > 0:
                eos_positions = (completion_ids == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first_eos = int(eos_positions[0].item())
                    completion_ids = completion_ids[:first_eos]

            prompt_text = self.decode(prompt_ids, skip_special_tokens=skip_special_tokens)
            completion_text = self.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            prompt_texts.append(prompt_text)
            completion_texts.append(completion_text)

        return self.score_texts(prompt_texts, completion_texts, device=device)

    @torch.no_grad()
    def compute_group_stats(
        self,
        rewards: torch.Tensor,
        *,
        group_size: Optional[int] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> GroupRewardOutput:
        rewards = rewards.float()
        if rewards.ndim != 1:
            raise ValueError("rewards must be rank-1 [B]")

        if (group_size is None) == (group_indices is None):
            raise ValueError("Provide exactly one of group_size or group_indices")

        device = rewards.device

        if group_indices is None:
            if group_size is None or group_size <= 0:
                raise ValueError("group_size must be a positive integer")
            if rewards.numel() % group_size != 0:
                raise ValueError(
                    f"batch size {rewards.numel()} is not divisible by group_size={group_size}"
                )
            num_groups = rewards.numel() // group_size
            group_indices = torch.arange(num_groups, device=device).repeat_interleave(group_size)
        else:
            if group_indices.ndim != 1 or group_indices.numel() != rewards.numel():
                raise ValueError("group_indices must be rank-1 with same length as rewards")
            group_indices = group_indices.to(device=device, dtype=torch.long)

            unique_groups = torch.unique(group_indices, sorted=True)
            remap = {int(g.item()): i for i, g in enumerate(unique_groups)}
            group_indices = torch.tensor(
                [remap[int(g.item())] for g in group_indices],
                dtype=torch.long,
                device=device,
            )
            num_groups = len(unique_groups)

        group_means = torch.zeros(num_groups, dtype=rewards.dtype, device=device)
        group_stds = torch.zeros(num_groups, dtype=rewards.dtype, device=device)
        group_sizes = torch.zeros(num_groups, dtype=torch.long, device=device)

        for g in range(num_groups):
            mask = group_indices == g
            r_g = rewards[mask]
            group_sizes[g] = r_g.numel()
            group_means[g] = r_g.mean()
            group_stds[g] = r_g.std(unbiased=False)

        advantages = (rewards - group_means[group_indices]) / (group_stds[group_indices] + self.group_std_eps)

        return GroupRewardOutput(
            rewards=rewards,
            group_means=group_means,
            group_stds=group_stds,
            group_advantages=advantages,
            group_indices=group_indices,
            group_sizes=group_sizes,
        )

    @torch.no_grad()
    def compute_grpo_advantages(
        self,
        rewards: torch.Tensor,
        *,
        group_size: Optional[int] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> GroupRewardOutput:
        return self.compute_group_stats(
            rewards=rewards,
            group_size=group_size,
            group_indices=group_indices,
        )

    # =========================================================
    # Tokenizer helpers
    # =========================================================

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        """
        Robust decode for:
          - HF tokenizers: decode(ids, skip_special_tokens=...)
          - sentencepiece.SentencePieceProcessor: DecodeIds(ids)
          - custom tokenizers with decode(ids)

        Returns normalized whitespace text.
        """
        ids = token_ids.detach().cpu().tolist()

        if len(ids) == 0:
            return ""

        text: Optional[str] = None

        # Hugging Face style
        if hasattr(self.tokenizer, "decode"):
            try:
                text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            except TypeError:
                text = self.tokenizer.decode(ids)

        # Raw SentencePieceProcessor style
        elif hasattr(self.tokenizer, "DecodeIds"):
            text = self.tokenizer.DecodeIds(ids)

        if text is None:
            raise ValueError("Tokenizer does not provide a supported decode method")

        return self._normalize_decoded_text(text)

    def _normalize_decoded_text(self, text: str) -> str:
        """
        SentencePiece decode can yield odd leading/trailing whitespace depending on pipeline.
        This cleans it without trying to be clever enough to break punctuation spacing.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self.MULTISPACE_RE.sub(" ", text)
        return text.strip()

    def _infer_special_id(self, attr_name: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        return getattr(self.tokenizer, attr_name, None)

    # =========================================================
    # Validation
    # =========================================================

    def _validate_concat_inputs(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> None:
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape [B, T]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, T]")
        if prompt_lens.ndim != 1:
            raise ValueError("prompt_lens must have shape [B]")
        if sequences.shape != attention_mask.shape:
            raise ValueError(
                f"sequences and attention_mask must have same shape, got "
                f"{tuple(sequences.shape)} vs {tuple(attention_mask.shape)}"
            )
        if sequences.size(0) != prompt_lens.size(0):
            raise ValueError(
                f"Batch mismatch: sequences batch={sequences.size(0)} "
                f"but prompt_lens batch={prompt_lens.size(0)}"
            )

    # =========================================================
    # Heuristic scoring internals
    # =========================================================

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = self.MULTISPACE_RE.sub(" ", text)
        return text

    def _words(self, text: str) -> List[str]:
        return self.WORD_RE.findall(text.lower())

    def _sentences(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []
        return [s.strip() for s in self.SENTENCE_SPLIT_RE.split(text) if s.strip()]

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _safe_div(self, num: float, den: float, default: float = 0.0) -> float:
        return num / den if den else default

    def _extract_prompt_keywords(self, prompt: str) -> List[str]:
        toks = self._words(prompt)
        candidates = [t for t in toks if t not in self.STOPWORDS and len(t) >= 3]
        counts = Counter(candidates)
        return [w for w, _ in counts.most_common(12)]

    def _prompt_keyword_coverage(self, prompt: str, completion: str) -> float:
        prompt_keywords = self._extract_prompt_keywords(prompt)
        if not prompt_keywords:
            return 0.5

        completion_tokens = set(self._words(completion))
        matched = sum(1 for kw in prompt_keywords if kw in completion_tokens)
        raw = matched / len(prompt_keywords)

        first_few = prompt_keywords[:4]
        first_few_match = sum(1 for kw in first_few if kw in completion_tokens)
        first_few_score = self._safe_div(first_few_match, len(first_few), default=0.0)

        return self._clamp(0.75 * raw + 0.25 * first_few_score, 0.0, 1.0)

    def _prompt_verb_overlap(self, prompt: str, completion: str) -> float:
        likely_verbs = {
            t for t in self._words(prompt)
            if t.endswith(("e", "ed", "ing")) or t in {
                "find", "help", "share", "learn", "play", "plant", "save",
                "look", "walk", "run", "jump", "give", "take", "make"
            }
        }
        if not likely_verbs:
            return 0.5

        completion_tokens = set(self._words(completion))
        matched = sum(1 for v in likely_verbs if v in completion_tokens)
        return self._clamp(matched / len(likely_verbs), 0.0, 1.0)

    def _score_prompt_adherence(self, prompt: str, completion: str) -> float:
        kw = self._prompt_keyword_coverage(prompt, completion)
        verb = self._prompt_verb_overlap(prompt, completion)
        return self._clamp(0.8 * kw + 0.2 * verb, 0.0, 1.0)

    def _sentence_overlap(self, a: str, b: str) -> float:
        wa = set(self._words(a))
        wb = set(self._words(b))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def _score_coherence(self, completion: str) -> float:
        sents = self._sentences(completion)
        toks = self._words(completion)

        if len(toks) < 15:
            return 0.1
        if len(sents) == 0:
            return 0.0
        if len(sents) == 1:
            return 0.35

        overlaps = [self._sentence_overlap(sents[i], sents[i + 1]) for i in range(len(sents) - 1)]
        avg_overlap = sum(overlaps) / len(overlaps)

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

        first_words = set(self._words(sents[0]))
        last_words = set(self._words(sents[-1]))
        opening_bonus = 1.0 if first_words & self.STORY_START_MARKERS else 0.6
        ending_bonus = 1.0 if last_words & self.STORY_END_MARKERS else 0.6

        temporal_hits = sum(1 for t in self._words(completion) if t in self.TEMPORAL_WORDS)
        temporal_score = self._clamp(temporal_hits / 2.0, 0.0, 1.0)

        return self._clamp(
            0.50 * flow_score
            + 0.20 * opening_bonus
            + 0.20 * ending_bonus
            + 0.10 * temporal_score,
            0.0,
            1.0,
        )

    def _average_sentence_length(self, text: str) -> float:
        sents = self._sentences(text)
        if not sents:
            return 0.0
        sent_lengths = [len(self._words(s)) for s in sents]
        return sum(sent_lengths) / len(sent_lengths)

    def _average_word_length(self, text: str) -> float:
        toks = self._words(text)
        if not toks:
            return 0.0
        return sum(len(t) for t in toks) / len(toks)

    def _score_style(self, completion: str) -> float:
        toks = self._words(completion)
        if not toks:
            return 0.0

        avg_sent_len = self._average_sentence_length(completion)
        avg_word_len = self._average_word_length(completion)

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

        pos_hits = sum(1 for t in toks if t in self.POSITIVE_STYLE_WORDS)
        neg_hits = sum(1 for t in toks if t in self.NEGATIVE_TONE_WORDS)

        pos_score = self._clamp(pos_hits / max(3.0, len(toks) * 0.03), 0.0, 1.0)
        neg_penalty = self._clamp(neg_hits / max(2.0, len(toks) * 0.02), 0.0, 1.0)

        dialogue_bonus = 1.0 if '"' in completion or "'" in completion else 0.6

        return self._clamp(
            0.35 * sent_score
            + 0.25 * word_score
            + 0.20 * pos_score
            + 0.10 * dialogue_bonus
            + 0.10 * (1.0 - neg_penalty),
            0.0,
            1.0,
        )

    def _score_length(self, completion: str) -> float:
        n = len(self._words(completion))
        lo = self.preferred_min_words
        hi = self.preferred_max_words

        if lo <= n <= hi:
            return 1.0
        if 40 <= n < lo:
            return 0.6 + 0.4 * (n - 40) / max(1, lo - 40)
        if hi < n <= 220:
            return 0.6 + 0.4 * (220 - n) / max(1, 220 - hi)
        if 20 <= n < 40:
            return 0.25 + 0.35 * (n - 20) / 20
        if 220 < n <= 300:
            return 0.25 + 0.35 * (300 - n) / 80
        return 0.0

    def _score_safety(self, completion: str) -> float:
        toks = set(self._words(completion))
        unsafe_hits = sum(1 for w in self.UNSAFE_WORDS if w in toks)
        if unsafe_hits == 0:
            return 1.0
        if unsafe_hits == 1:
            return 0.3
        return 0.0

    def _score_fluency(self, completion: str) -> float:
        text = self._normalize_text(completion)
        if not text:
            return 0.0

        sents = self._sentences(text)
        toks = self._words(text)
        if not toks:
            return 0.0

        capitalization_score = 0.0
        if sents:
            capitalization_score = sum(1 for s in sents if s and s[0].isupper()) / len(sents)

        punctuation_score = 0.0
        if sents:
            punctuation_score = sum(1 for s in sents if s[-1] in ".!?") / len(sents)

        weird_chars = self.WEIRD_CHAR_RE.findall(text)
        weird_char_penalty = self._clamp(len(weird_chars) / max(1.0, len(text) * 0.02), 0.0, 1.0)

        punct_spam = len(self.PUNCT_SPAM_RE.findall(text))
        punct_spam_penalty = self._clamp(punct_spam / max(1.0, len(sents)), 0.0, 1.0)

        return self._clamp(
            0.40 * capitalization_score
            + 0.40 * punctuation_score
            + 0.20 * (1.0 - max(weird_char_penalty, punct_spam_penalty)),
            0.0,
            1.0,
        )

    def _get_ngrams(self, tokens: Sequence[str], n: int) -> List[tuple[str, ...]]:
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _unique_ratio(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _repeated_ngram_rate(self, tokens: Sequence[str], n: int) -> float:
        ngrams = self._get_ngrams(tokens, n)
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(c for c in counts.values() if c > 1)
        return self._clamp(repeated / len(ngrams), 0.0, 1.0)

    def _repeated_sentence_rate(self, text: str) -> float:
        sents = [s.lower() for s in self._sentences(text)]
        if not sents:
            return 0.0
        counts = Counter(sents)
        repeated = sum(c for c in counts.values() if c > 1)
        return self._clamp(repeated / len(sents), 0.0, 1.0)

    def _token_repetition_rate(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        return self._clamp(1.0 - self._unique_ratio(tokens), 0.0, 1.0)

    def _trailing_fragment_penalty(self, text: str) -> float:
        text = self._normalize_text(text)
        if not text:
            return 1.0
        if len(self._words(text)) >= 20 and text[-1] not in ".!?":
            return 0.7
        return 0.0

    def _score_degeneracy_penalty(self, completion: str) -> float:
        toks = self._words(completion)
        if not toks:
            return 1.0

        rep2 = self._repeated_ngram_rate(toks, 2)
        rep3 = self._repeated_ngram_rate(toks, 3)
        rep4 = self._repeated_ngram_rate(toks, 4)
        sent_rep = self._repeated_sentence_rate(completion)
        tok_rep = self._token_repetition_rate(toks)
        punct_spam = 1.0 if self.PUNCT_SPAM_RE.search(completion) else 0.0
        trail = self._trailing_fragment_penalty(completion)

        penalty = (
            0.20 * rep2
            + 0.20 * rep3
            + 0.20 * rep4
            + 0.15 * sent_rep
            + 0.15 * tok_rep
            + 0.05 * punct_spam
            + 0.05 * trail
        )
        return self._clamp(penalty, 0.0, 1.0)

    def _resolve_device(self, device: Optional[torch.device]) -> torch.device:
        if device is not None:
            return device
        if self.device is not None:
            return self.device
        return torch.device("cpu")