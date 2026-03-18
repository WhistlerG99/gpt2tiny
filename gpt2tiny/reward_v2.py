from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import spacy
from sentence_transformers import SentenceTransformer


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class RequiredWord:
    word: str
    pos: str  # noun | verb | adjective


@dataclass
class SubjectSpec:
    character: Optional[str] = None
    adjective: Optional[str] = None
    action: Optional[str] = None
    place: Optional[str] = None
    goal: Optional[str] = None


@dataclass
class RewardWeights:
    words: float = 0.25
    pos: float = 0.15
    subject: float = 0.20
    features: float = 0.25
    format: float = 0.15

    prompt_copy_penalty: float = 0.10
    meta_penalty: float = 0.10
    repetition_penalty: float = 0.10
    stuffing_penalty: float = 0.10


@dataclass
class StoryRewardConfig:
    spacy_model: str = "en_core_web_sm"
    sentence_model: str = "all-MiniLM-L6-v2"

    min_chars: int = 80
    max_chars: int = 2000

    min_sentences: int = 3
    max_sentences: int = 5

    feature_similarity_threshold: float = 0.35
    subject_similarity_threshold: float = 0.35

    max_reasonable_occurrences_per_required_word: int = 3

    allow_propn_for_noun: bool = True
    allow_aux_for_verb: bool = False

    # If True, reward ignores text before prompt_lengths[i]
    # and scores only the generated continuation.
    score_generated_only: bool = True

    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "BadEnding": 1.0,
        "Conflict": 1.0,
        "Dialogue": 1.0,
        "Foreshadowing": 0.5,
        "MoralValue": 0.5,
        "Twist": 0.6,
    })


# ============================================================
# Main reward class
# ============================================================

class StoryReward:
    """
    Reward model for GRPO that can score:
      1. raw strings via score(...)
      2. batched token IDs via score_from_token_ids(...)

    Expected metadata format per example:
    {
        "prompt": "...",
        "words": [
            {"word": "jog", "pos": "verb"},
            {"word": "cane", "pos": "noun"},
            {"word": "polite", "pos": "adjective"},
        ],
        "features": [
            "BadEnding",
            "Dialogue",
            "Twist",
        ],
        "subject": {
            "character": "pirate",
            "adjective": "lonely",
            "action": "goes to the White House",
            "place": "the White House",
            "goal": "recover a stolen heirloom",
        }
    }
    """

    def __init__(
        self,
        tokenizer = None,
        config: Optional[StoryRewardConfig] = None,
        weights: Optional[RewardWeights] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or StoryRewardConfig()
        self.weights = weights or RewardWeights()

        self.nlp = spacy.load(self.config.spacy_model)
        self.st_model = SentenceTransformer(self.config.sentence_model)

        self.conflict_keywords = {
            "fight", "argue", "chase", "hide", "escape", "threat", "threaten",
            "danger", "enemy", "opponent", "war", "struggle", "problem",
            "obstacle", "clash", "betray", "betrayal", "attack", "accuse",
            "prevent", "stop", "lose", "lost", "fear", "afraid", "trap",
            "trapped", "pursue", "pursuit", "rival", "conflict", "wound",
            "injure", "harm", "risk", "risky"
        }

        self.bad_ending_keywords = {
            "dead", "died", "death", "lost", "failed", "failure", "ruined",
            "alone", "broken", "prison", "cry", "cried", "mourning",
            "grief", "regret", "regretted", "bleak", "hopeless", "sad",
            "sorrow", "tragic", "tragedy", "destroyed", "gone", "goodbye"
        }

        self.moral_cue_patterns = [
            r"\blearned that\b",
            r"\bthe lesson\b",
            r"\bfrom then on\b",
            r"\b(realized|realised) that\b",
            r"\bit taught (him|her|them)\b",
            r"\bunderstood that\b",
            r"\bknew then that\b",
        ]

        self.twist_markers = {
            "suddenly", "unexpectedly", "surprisingly", "instead", "however",
            "but", "yet", "turned", "revealed", "reveal", "actually",
            "secretly", "truth", "surprise", "shocked", "shock"
        }

        self.payoff_markers = {
            "later", "finally", "in the end", "turned out", "because of that",
            "that was why", "the same", "again", "at last", "after all"
        }

        self.meta_instruction_patterns = [
            r"\bthis story\b",
            r"\bthe story should\b",
            r"\brequired word",
            r"\brequired words",
            r"\bfeature\b",
            r"\bprompt\b",
            r"\bplot twist\b",
            r"\bdialogue\b",
            r"\bmoral value\b",
        ]

        self.prototype_texts = {
            "conflict": [
                "Someone faces a serious problem.",
                "The main character struggles against an obstacle.",
                "There is danger or opposition in the story.",
            ],
            "moral": [
                "The story conveys a lesson.",
                "The main character learns something important.",
                "The ending contains a moral takeaway.",
            ],
            "twist": [
                "Something unexpected happens.",
                "There is a surprise reveal at the end.",
                "The plot changes in an unexpected way.",
            ],
            "bad_ending": [
                "The ending is unhappy.",
                "The story ends badly for someone.",
                "The conclusion is bleak and unfortunate.",
            ],
        }

    # ========================================================
    # Public API
    # ========================================================

    @torch._dynamo.disable
    def score(self, generated_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        text = self._normalize_text(generated_text)
        prompt = metadata.get("prompt", "") or ""
        required_words = [
            RequiredWord(**rw) if isinstance(rw, dict) else rw
            for rw in metadata.get("words", [])
        ]
        required_features = list(metadata.get("features", []))
        subject_data = metadata.get("subject", None)
        subject = SubjectSpec(**subject_data) if isinstance(subject_data, dict) else None

        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        words_score = self.score_required_words(doc, required_words)
        pos_score = self.score_pos_usage(doc, required_words)
        subject_score = self.score_subject(doc, sentences, subject)
        features_score, feature_breakdown = self.score_features(doc, sentences, required_features)
        format_score = self.score_format(doc, sentences)

        prompt_copy_pen = self.penalty_prompt_copy(text, prompt, required_words)
        meta_pen = self.penalty_meta_language(text)
        repetition_pen = self.penalty_repetition(doc)
        stuffing_pen = self.penalty_keyword_stuffing(doc, required_words)

        # raw_total = (
        #     self.weights.words * words_score
        #     + self.weights.pos * pos_score
        #     + self.weights.subject * subject_score
        #     + self.weights.features * features_score
        #     + self.weights.format * format_score
        #     - self.weights.prompt_copy_penalty * prompt_copy_pen
        #     - self.weights.meta_penalty * meta_pen
        #     - self.weights.repetition_penalty * repetition_pen
        #     - self.weights.stuffing_penalty * stuffing_pen
        # )
        
        has_required_words = len(required_words) > 0
        has_features = len(required_features) > 0
        has_subject = self._has_subject(subject)
        
        positive_sum = 0.0
        active_weight_sum = 0.0
        ########
        if has_required_words:
            positive_sum += self.weights.words * words_score
            active_weight_sum += self.weights.words
        
            positive_sum += self.weights.pos * pos_score
            active_weight_sum += self.weights.pos
        
        if has_subject:
            positive_sum += self.weights.subject * subject_score
            active_weight_sum += self.weights.subject
        
        if has_features:
            positive_sum += self.weights.features * features_score
            active_weight_sum += self.weights.features
        
        # format is always active
        positive_sum += self.weights.format * format_score
        active_weight_sum += self.weights.format
        
        positive_score = positive_sum / max(active_weight_sum, 1e-8)
        
        penalty_total = 0.0
        penalty_total += self.weights.prompt_copy_penalty * prompt_copy_pen
        penalty_total += self.weights.meta_penalty * meta_pen
        penalty_total += self.weights.repetition_penalty * repetition_pen
        
        if has_required_words:
            penalty_total += self.weights.stuffing_penalty * stuffing_pen
        
        raw_total = positive_score - penalty_total
        ########
        total = float(np.clip(raw_total, -1.0, 1.0))

        return {
            "total": total,
            "raw_total": raw_total,
            "components": {
                "words": words_score,
                "pos": pos_score,
                "subject": subject_score,
                "features": features_score,
                "format": format_score,
            },
            "feature_breakdown": feature_breakdown,
            "penalties": {
                "prompt_copy": prompt_copy_pen,
                "meta_language": meta_pen,
                "repetition": repetition_pen,
                "keyword_stuffing": stuffing_pen,
            },
            "debug": {
                "n_sentences": len(sentences),
                "n_chars": len(text),
                "text": text,
            },
        }

    @torch._dynamo.disable
    def score_batch(
        self,
        generated_texts: Sequence[str],
        metadata_list: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(generated_texts) != len(metadata_list):
            raise ValueError("generated_texts and metadata_list must have the same length.")
        return [self.score(t, m) for t, m in zip(generated_texts, metadata_list)]

    @torch._dynamo.disable
    @torch.no_grad()
    def score_from_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: torch.Tensor,
        metadata_list: Sequence[Dict[str, Any]],
        return_breakdown: bool = False,
        skip_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        """
        Scores a batch from concatenated prompt+generated token IDs.

        Args:
            input_ids:
                LongTensor [B, T]
            attention_mask:
                Optional tensor [B, T], 1 for real tokens, 0 for pad.
            prompt_lengths:
                LongTensor [B], number of prompt tokens in each row.
            metadata_list:
                List[dict] of length B
            return_breakdown:
                If True, return full per-example score dicts.
                Else, only scalar rewards.
            skip_special_tokens:
                Passed into tokenizer.decode when supported.

        Returns:
            {
                "rewards": FloatTensor [B],
                "details": Optional[List[dict]]
            }
        """
        if self.tokenizer is None:
            raise ValueError(f"`tokenizer` attribute is `None`. Must be set when class is instantiated")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")

        B, T = input_ids.shape

        if len(metadata_list) != B:
            raise ValueError(
                f"metadata_list length ({len(metadata_list)}) must match batch size ({B})"
            )

        if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != B:
            raise ValueError(
                f"prompt_lengths must have shape [B], got {tuple(prompt_lengths.shape)}"
            )

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must equal input_ids shape {tuple(input_ids.shape)}"
            )

        rewards: List[float] = []
        details: List[Dict[str, Any]] = []

        input_ids_cpu = input_ids.detach().cpu()
        prompt_lengths_cpu = prompt_lengths.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None

        for i in range(B):
            row_ids = input_ids_cpu[i]
            row_prompt_len = int(prompt_lengths_cpu[i].item())

            if attention_mask_cpu is not None:
                row_mask = attention_mask_cpu[i].bool()
                valid_ids = row_ids[row_mask]
            else:
                valid_ids = row_ids

            valid_ids_list = valid_ids.tolist()

            if self.config.score_generated_only:
                start = min(max(row_prompt_len, 0), len(valid_ids_list))
                text_ids = valid_ids_list[start:]
            else:
                text_ids = valid_ids_list

            decoded_text = self._decode_ids(
                token_ids=text_ids,
                skip_special_tokens=skip_special_tokens,
            )

            score_dict = self.score(decoded_text, metadata_list[i])
            rewards.append(float(score_dict["total"]))
            if return_breakdown:
                details.append(score_dict)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        out: Dict[str, Any] = {"rewards": rewards_tensor}
        if return_breakdown:
            out["details"] = details
        return out

    @torch._dynamo.disable
    @torch.no_grad()
    def score_grouped_from_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: torch.Tensor,
        metadata_list: Sequence[Dict[str, Any]],
        group_size: int,
        return_breakdown: bool = False,
        skip_special_tokens: bool = True,
        advantage_eps: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper for GRPO.

        Assumes rows are arranged in groups of completions for the same prompt:
            [prompt1_sample1,
             prompt1_sample2,
             ...,
             prompt1_sampleG,
             prompt2_sample1,
             ... ]

        Returns:
            {
                "rewards": FloatTensor [B],
                "advantages": FloatTensor [B],
                "group_means": FloatTensor [B // G],
                "group_stds": FloatTensor [B // G],
                "details": optional
            }
        """
        _metadata_list = [md for md in metadata_list for _ in range(group_size)]
        out = self.score_from_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            metadata_list=_metadata_list,
            return_breakdown=return_breakdown,
            skip_special_tokens=skip_special_tokens,
        )
        rewards = out["rewards"]

        if rewards.numel() % group_size != 0:
            raise ValueError(
                f"Batch size {rewards.numel()} is not divisible by group_size={group_size}"
            )

        advantages, group_means, group_stds = StoryReward.compute_grpo_advantages(
            rewards=rewards,
            group_size=group_size,
            eps=advantage_eps,
        )

        result: Dict[str, Any] = {
            "rewards": rewards,
            "advantages": advantages,
            "group_means": group_means,
            "group_stds": group_stds,
        }
        if return_breakdown:
            result["details"] = out["details"]
        return result

    @torch._dynamo.disable
    @staticmethod
    def compute_grpo_advantages(
        rewards: torch.Tensor,
        group_size: int,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Group-normalized rewards for GRPO.

        Args:
            rewards: [B]
            group_size: G
        Returns:
            advantages: [B]
            group_means: [B // G]
            group_stds: [B // G]
        """
        if rewards.ndim != 1:
            raise ValueError(f"rewards must have shape [B], got {tuple(rewards.shape)}")
        B = rewards.shape[0]
        if B % group_size != 0:
            raise ValueError(f"Batch size {B} must be divisible by group_size {group_size}")

        grouped = rewards.view(B // group_size, group_size)
        means = grouped.mean(dim=1, keepdim=True)
        stds = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = (grouped - means) / (stds + eps)
        return advantages.reshape(B), means.squeeze(1), stds.squeeze(1)

    # ========================================================
    # Required words
    # ========================================================

    def score_required_words(self, doc, required_words: Sequence[RequiredWord]) -> float:
        if not required_words:
            return np.nan

        scores = []
        for rw in required_words:
            hits = self._find_matching_tokens(doc, rw.word)
            scores.append(1.0 if hits else 0.0)
        return float(np.mean(scores)) if scores else 1.0

    def score_pos_usage(self, doc, required_words: Sequence[RequiredWord]) -> float:
        if not required_words:
            return np.nan

        scores = []
        for rw in required_words:
            hits = self._find_matching_tokens(doc, rw.word)
            if not hits:
                scores.append(0.0)
                continue

            correct = any(self._token_matches_pos(tok, rw.pos) for tok in hits)
            scores.append(1.0 if correct else 0.0)

        return float(np.mean(scores)) if scores else 1.0

    # ========================================================
    # Subject adherence
    # ========================================================

    def score_subject(
        self,
        doc,
        sentences: Sequence[str],
        subject: Optional[SubjectSpec],
    ) -> float:
        if subject is None:
            return np.nan

        sub_scores = []
        sub_weights = []

        if subject.character:
            score = self._lexical_or_semantic_match(
                doc=doc,
                sentences=sentences,
                phrase=subject.character,
                semantic_weight=0.35,
            )
            sub_scores.append(score)
            sub_weights.append(1.0)

        if subject.adjective:
            score = self._score_subject_adjective(doc, subject)
            sub_scores.append(score)
            sub_weights.append(0.6)

        if subject.place:
            score = self._lexical_or_semantic_match(
                doc=doc,
                sentences=sentences,
                phrase=subject.place,
                semantic_weight=0.35,
            )
            sub_scores.append(score)
            sub_weights.append(0.9)

        if subject.action:
            score = self._score_sentence_semantic_match(sentences, subject.action)
            keyword_bonus = self._content_overlap_score(subject.action, " ".join(sentences))
            score = 0.65 * score + 0.35 * keyword_bonus
            sub_scores.append(score)
            sub_weights.append(1.1)

        if subject.goal:
            score = self._score_sentence_semantic_match(sentences, subject.goal)
            keyword_bonus = self._content_overlap_score(subject.goal, " ".join(sentences))
            score = 0.65 * score + 0.35 * keyword_bonus
            sub_scores.append(score)
            sub_weights.append(1.0)

        if not sub_scores:
            return 1.0

        return self._weighted_mean(sub_scores, sub_weights)

    def _score_subject_adjective(self, doc, subject: SubjectSpec) -> float:
        adj = subject.adjective.lower().strip()
        char = (subject.character or "").lower().strip()

        adj_tokens = [
            tok for tok in doc
            if tok.text.lower() == adj or tok.lemma_.lower() == adj
        ]
        if not adj_tokens:
            return 0.0

        base = 0.5

        if char:
            char_positions = [
                tok.i for tok in doc
                if tok.text.lower() == char or tok.lemma_.lower() == char
            ]
            for atok in adj_tokens:
                if any(abs(atok.i - cp) <= 3 for cp in char_positions):
                    base = max(base, 1.0)

        for atok in adj_tokens:
            if atok.pos_ == "ADJ":
                base = max(base, 0.85)

        return float(min(base, 1.0))

    # ========================================================
    # Feature adherence
    # ========================================================

    def score_features(
        self,
        doc,
        sentences: Sequence[str],
        required_features: Sequence[str],
    ) -> Tuple[float, Dict[str, float]]:
        if not required_features:
            return np.nan, {}

        breakdown: Dict[str, float] = {}
        weighted_scores = []
        weighted_weights = []

        for feat in required_features:
            if feat == "BadEnding":
                val = self._score_bad_ending(doc, sentences)
            elif feat == "Conflict":
                val = self._score_conflict(doc, sentences)
            elif feat == "Dialogue":
                val = self._score_dialogue(doc, sentences)
            elif feat == "Foreshadowing":
                val = self._score_foreshadowing(doc, sentences)
            elif feat == "MoralValue":
                val = self._score_moral(doc, sentences)
            elif feat == "Twist":
                val = self._score_twist(doc, sentences)
            else:
                val = 0.0

            breakdown[feat] = val
            w = self.config.feature_weights.get(feat, 1.0)
            weighted_scores.append(val)
            weighted_weights.append(w)

        return self._weighted_mean(weighted_scores, weighted_weights), breakdown

    def _score_bad_ending(self, doc, sentences: Sequence[str]) -> float:
        if not sentences:
            return 0.0

        ending = " ".join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]
        ending_doc = self.nlp(ending)

        keyword_score = self._keyword_fraction(ending_doc, self.bad_ending_keywords)
        semantic_score = self._prototype_similarity(ending, "bad_ending")

        negative_markers = {"dead", "lost", "failed", "alone", "broken", "ruined", "sad", "cry", "prison"}
        positive_markers = {"happy", "joy", "won", "safe", "saved", "smile", "celebrate", "peace"}
        neg = self._keyword_fraction(ending_doc, negative_markers)
        pos = self._keyword_fraction(ending_doc, positive_markers)
        sentiment_proxy = np.clip(0.5 + 0.7 * (neg - pos), 0.0, 1.0)

        return float(np.clip(
            0.35 * keyword_score + 0.40 * semantic_score + 0.25 * sentiment_proxy,
            0.0,
            1.0,
        ))

    def _score_conflict(self, doc, sentences: Sequence[str]) -> float:
        if not sentences:
            return 0.0

        keyword_score = self._keyword_fraction(doc, self.conflict_keywords)
        semantic_score = self._prototype_similarity(" ".join(sentences), "conflict")

        opposition_markers = {"but", "however", "yet", "instead", "against"}
        marker_score = self._keyword_fraction(doc, opposition_markers)

        return float(np.clip(
            0.45 * keyword_score + 0.35 * semantic_score + 0.20 * marker_score,
            0.0,
            1.0,
        ))

    def _score_dialogue(self, doc, sentences: Sequence[str]) -> float:
        text = doc.text
        quote_score = 1.0 if re.search(r'"[^"]{2,}"', text) else 0.0
        single_quote_score = 1.0 if re.search(r"'[^']{2,}'", text) else 0.0

        reporting_verbs = {"say", "ask", "reply", "whisper", "shout", "mutter", "cry", "yell"}
        reporting_score = 0.0
        for tok in doc:
            if tok.lemma_.lower() in reporting_verbs:
                reporting_score = 1.0
                break

        return float(np.clip(max(quote_score, single_quote_score) * 0.8 + reporting_score * 0.2, 0.0, 1.0))

    def _score_foreshadowing(self, doc, sentences: Sequence[str]) -> float:
        if len(sentences) < 3:
            return 0.0

        mid = len(sentences) // 2
        early = " ".join(sentences[:mid])
        late = " ".join(sentences[mid:])

        early_doc = self.nlp(early)
        late_doc = self.nlp(late)

        early_content = {
            tok.lemma_.lower()
            for tok in early_doc
            if tok.is_alpha and not tok.is_stop and tok.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"}
        }
        late_content = {
            tok.lemma_.lower()
            for tok in late_doc
            if tok.is_alpha and not tok.is_stop and tok.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"}
        }

        overlap = len(early_content & late_content) / max(1, len(early_content))
        marker_score = self._phrase_marker_score(late.lower(), self.payoff_markers)

        final_sent = sentences[-1].lower()
        early_nouns = {
            tok.lemma_.lower()
            for tok in early_doc if tok.pos_ in {"NOUN", "PROPN"} and not tok.is_stop
        }
        final_recur = 1.0 if any(n in final_sent for n in early_nouns) else 0.0

        return float(np.clip(
            0.45 * overlap + 0.35 * marker_score + 0.20 * final_recur,
            0.0,
            1.0,
        ))

    def _score_moral(self, doc, sentences: Sequence[str]) -> float:
        if not sentences:
            return 0.0

        end_text = " ".join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]
        cue_score = self._regex_list_score(end_text.lower(), self.moral_cue_patterns)
        semantic_score = self._prototype_similarity(end_text, "moral")
        return float(np.clip(0.5 * cue_score + 0.5 * semantic_score, 0.0, 1.0))

    def _score_twist(self, doc, sentences: Sequence[str]) -> float:
        if not sentences:
            return 0.0

        end_text = " ".join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]
        marker_score = self._keyword_fraction(self.nlp(end_text), self.twist_markers)
        semantic_score = self._prototype_similarity(end_text, "twist")

        if len(sentences) >= 2:
            opening = " ".join(sentences[:2])
            dissimilarity = 1.0 - self._pair_similarity(opening, end_text)
        else:
            dissimilarity = 0.0

        return float(np.clip(
            0.35 * marker_score + 0.45 * semantic_score + 0.20 * dissimilarity,
            0.0,
            1.0,
        ))

    # ========================================================
    # Format / quality
    # ========================================================

    def score_format(self, doc, sentences: Sequence[str]) -> float:
        text = doc.text
        n_chars = len(text)
        n_sent = len(sentences)

        len_score = self._band_score(
            value=n_chars,
            lo=self.config.min_chars,
            hi=self.config.max_chars,
            softness=120.0,
        )
        sent_score = self._band_score(
            value=n_sent,
            lo=self.config.min_sentences,
            hi=self.config.max_sentences,
            softness=1.0,
        )

        clean_score = 1.0
        clean_score -= 0.5 * self.penalty_repetition(doc)
        clean_score -= 0.3 * self._junk_text_penalty(text)
        clean_score = float(np.clip(clean_score, 0.0, 1.0))

        return float(np.clip(
            0.4 * len_score + 0.35 * sent_score + 0.25 * clean_score,
            0.0,
            1.0,
        ))

    # ========================================================
    # Penalties
    # ========================================================

    def penalty_prompt_copy(
        self,
        generated_text: str,
        prompt: str,
        required_words: Sequence[RequiredWord],
    ) -> float:
        if not prompt:
            return 0.0

        gen_tokens = self._simple_content_tokens(generated_text)
        prompt_tokens = self._simple_content_tokens(prompt)

        excluded = {rw.word.lower() for rw in required_words}
        gen_tokens = [t for t in gen_tokens if t not in excluded]
        prompt_tokens = [t for t in prompt_tokens if t not in excluded]

        if not prompt_tokens:
            return 0.0

        prompt_counts = Counter(prompt_tokens)
        gen_counts = Counter(gen_tokens)

        overlap = sum(min(gen_counts[t], prompt_counts[t]) for t in prompt_counts)
        denom = max(1, len(prompt_tokens))
        ratio = overlap / denom

        return float(np.clip((ratio - 0.15) / 0.35, 0.0, 1.0))

    def penalty_meta_language(self, generated_text: str) -> float:
        text = generated_text.lower()
        hits = 0
        for pat in self.meta_instruction_patterns:
            if re.search(pat, text):
                hits += 1
        return float(np.clip(hits / 3.0, 0.0, 1.0))

    def penalty_repetition(self, doc) -> float:
        toks = [t.text.lower() for t in doc if t.is_alpha]
        if len(toks) < 6:
            return 0.0

        counts = Counter(toks)
        repeated_mass = sum(c - 1 for c in counts.values() if c > 2)
        unigram_pen = repeated_mass / max(1, len(toks))

        bigrams = list(zip(toks, toks[1:]))
        bcounts = Counter(bigrams)
        repeated_bigrams = sum(c - 1 for c in bcounts.values() if c > 1)
        bigram_pen = repeated_bigrams / max(1, len(bigrams))

        return float(np.clip(0.5 * unigram_pen + 0.5 * bigram_pen, 0.0, 1.0))

    def penalty_keyword_stuffing(self, doc, required_words: Sequence[RequiredWord]) -> float:
        if not required_words:
            return 0.0

        penalties = []
        for rw in required_words:
            hits = self._find_matching_tokens(doc, rw.word)
            count = len(hits)
            excess = max(0, count - self.config.max_reasonable_occurrences_per_required_word)
            penalties.append(excess / 4.0)

        return float(np.clip(np.mean(penalties), 0.0, 1.0)) if penalties else 0.0

    # ========================================================
    # Helpers
    # ========================================================

    def _find_matching_tokens(self, doc, target_word: str):
        target = target_word.lower().strip()
        matches = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            if tok.text.lower() == target or tok.lemma_.lower() == target:
                matches.append(tok)
        return matches

    def _token_matches_pos(self, tok, expected_pos: str) -> bool:
        expected_pos = expected_pos.lower().strip()

        if expected_pos == "noun":
            allowed = {"NOUN"}
            if self.config.allow_propn_for_noun:
                allowed.add("PROPN")
            return tok.pos_ in allowed

        if expected_pos == "verb":
            allowed = {"VERB"}
            if self.config.allow_aux_for_verb:
                allowed.add("AUX")
            return tok.pos_ in allowed

        if expected_pos == "adjective":
            return tok.pos_ == "ADJ"

        return False

    def _lexical_or_semantic_match(
        self,
        doc,
        sentences: Sequence[str],
        phrase: str,
        semantic_weight: float = 0.35,
    ) -> float:
        phrase = phrase.strip()
        if not phrase:
            return 0.0

        lexical = self._phrase_mention_score(doc.text, phrase)
        semantic = self._score_sentence_semantic_match(sentences, phrase)

        return float(np.clip(
            (1.0 - semantic_weight) * lexical + semantic_weight * semantic,
            0.0,
            1.0,
        ))

    def _phrase_mention_score(self, text: str, phrase: str) -> float:
        t = text.lower()
        p = phrase.lower().strip()

        if p in t:
            return 1.0

        return self._content_overlap_score(phrase, text)

    def _content_overlap_score(self, source_phrase: str, target_text: str) -> float:
        sdoc = self.nlp(source_phrase)
        tdoc = self.nlp(target_text)

        s_terms = {
            tok.lemma_.lower()
            for tok in sdoc
            if tok.is_alpha and not tok.is_stop and tok.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"}
        }
        t_terms = {
            tok.lemma_.lower()
            for tok in tdoc
            if tok.is_alpha and not tok.is_stop
        }

        if not s_terms:
            return 0.0
        return len(s_terms & t_terms) / len(s_terms)

    def _score_sentence_semantic_match(self, sentences: Sequence[str], target_phrase: str) -> float:
        if not sentences or not target_phrase.strip():
            return 0.0

        target_emb = self._embed_texts([target_phrase])
        sent_embs = self._embed_texts(list(sentences))
        sims = self._cosine_matrix(target_emb, sent_embs)[0]
        best = float(np.max(sims))
        return float(np.clip(
            (best - self.config.subject_similarity_threshold)
            / (1.0 - self.config.subject_similarity_threshold),
            0.0,
            1.0,
        ))

    def _prototype_similarity(self, text: str, prototype_key: str) -> float:
        prototypes = self.prototype_texts[prototype_key]
        texts = [text] + prototypes
        embs = self._embed_texts(texts)
        base = embs[:1]
        prot = embs[1:]
        sims = self._cosine_matrix(base, prot)[0]
        best = float(np.max(sims))
        thresh = self.config.feature_similarity_threshold
        return float(np.clip((best - thresh) / (1.0 - thresh), 0.0, 1.0))

    def _pair_similarity(self, text_a: str, text_b: str) -> float:
        embs = self._embed_texts([text_a, text_b])
        sim = self._cosine_matrix(embs[:1], embs[1:])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.st_model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb)

    @staticmethod
    def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b.T)

    @staticmethod
    def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
        values = np.asarray(values, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        denom = float(np.sum(weights))
        if denom <= 0:
            return 0.0
        return float(np.sum(values * weights) / denom)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _keyword_fraction(doc, keywords: Sequence[str]) -> float:
        keywords = set(k.lower() for k in keywords)
        toks = [tok.lemma_.lower() for tok in doc if tok.is_alpha]
        if not toks:
            return 0.0
        hits = sum(1 for t in toks if t in keywords)
        return float(np.clip(hits / 4.0, 0.0, 1.0))

    @staticmethod
    def _phrase_marker_score(text: str, markers: Sequence[str]) -> float:
        hits = sum(1 for m in markers if m in text)
        return float(np.clip(hits / 2.0, 0.0, 1.0))

    @staticmethod
    def _regex_list_score(text: str, patterns: Sequence[str]) -> float:
        hits = sum(1 for p in patterns if re.search(p, text))
        return float(np.clip(hits / 2.0, 0.0, 1.0))

    @staticmethod
    def _band_score(value: float, lo: float, hi: float, softness: float) -> float:
        if lo <= value <= hi:
            return 1.0
        d = (lo - value) if value < lo else (value - hi)
        return float(math.exp(-d / max(softness, 1e-6)))

    @staticmethod
    def _junk_text_penalty(text: str) -> float:
        if not text:
            return 1.0

        penalties = 0.0

        punct_ratio = sum(
            1 for ch in text if not ch.isalnum() and not ch.isspace()
        ) / max(1, len(text))
        if punct_ratio > 0.25:
            penalties += min(1.0, (punct_ratio - 0.25) / 0.25)

        if re.search(r"(.)\1{5,}", text):
            penalties += 0.5

        caps_tokens = re.findall(r"\b[A-Z]{3,}\b", text)
        token_count = max(1, len(text.split()))
        caps_ratio = len(caps_tokens) / token_count
        if caps_ratio > 0.20:
            penalties += min(1.0, caps_ratio)

        return float(np.clip(penalties, 0.0, 1.0))

    @staticmethod
    def _simple_content_tokens(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def _has_subject(self, subject: Optional[SubjectSpec]) -> bool:
        if subject is None:
            return False
        return any([
            bool(subject.character),
            bool(subject.adjective),
            bool(subject.action),
            bool(subject.place),
            bool(subject.goal),
        ])
        
    def _decode_ids(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """
        Tries a couple tokenizer APIs without assuming too much.
        """
        token_ids = list(token_ids)
        if len(token_ids) == 0:
            return ""

        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            return self.tokenizer.decode(token_ids)


# ============================================================
# Example: GRPO usage inside a training step
# ============================================================

def example_grpo_usage():
    """
    Skeleton showing how to use StoryReward inside a GRPO rollout stage.
    """


    # Suppose B = num_prompts * G
    input_ids = torch.tensor([
        [101, 11, 12, 13, 14, 15,   0,  0],
        [101, 11, 12, 13, 16, 17,  18,  0],
        [102, 21, 22, 23, 24, 25,  26, 27],
        [102, 21, 22, 23, 28, 29,   0,  0],
    ], dtype=torch.long)

    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=torch.long)

    prompt_lengths = torch.tensor([4, 4, 4, 4], dtype=torch.long)

    metadata_list = [
        {
            "prompt": "Write a story using the verb 'jog' and include dialogue.",
            "words": [{"word": "jog", "pos": "verb"}],
            "features": ["Dialogue"],
            "subject": {"character": "pirate", "action": "goes to the White House"},
        },
        {
            "prompt": "Write a story using the verb 'jog' and include dialogue.",
            "words": [{"word": "jog", "pos": "verb"}],
            "features": ["Dialogue"],
            "subject": {"character": "pirate", "action": "goes to the White House"},
        },
        {
            "prompt": "Write a story using the noun 'cane' and a bad ending.",
            "words": [{"word": "cane", "pos": "noun"}],
            "features": ["BadEnding"],
            "subject": {"character": "girl", "place": "the mall"},
        },
        {
            "prompt": "Write a story using the noun 'cane' and a bad ending.",
            "words": [{"word": "cane", "pos": "noun"}],
            "features": ["BadEnding"],
            "subject": {"character": "girl", "place": "the mall"},
        },
    ]

    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            vocab = {
                14: '"Run," he said.',
                15: ' Then he jogged away.',
                16: ' The pirate jogged to the gate.',
                17: ' "Stop!" shouted a guard.',
                18: ' He smiled.',
                24: ' She carried a cane.',
                25: ' At night she was alone.',
                26: ' It ended badly.',
                27: '',
                28: ' She dropped the cane and laughed.',
                29: ' Everything was fine.',
            }
            parts = [vocab.get(i, "") for i in ids]
            return " ".join(p for p in parts if p).strip()

    tokenizer = DummyTokenizer()

    reward_fn = StoryReward(tokenizer=tokenizer)
    
    out = reward_fn.score_grouped_from_token_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        metadata_list=metadata_list,
        group_size=2,
        return_breakdown=True,
    )

    rewards = out["rewards"]         # [B]
    advantages = out["advantages"]   # [B]

    print("rewards:", rewards)
    print("advantages:", advantages)
    print("first detail:", out["details"][0])


# ============================================================
# Example: helper for your GRPO module
# ============================================================

class GRPORewardAdapter:
    """
    Thin wrapper to make the call site cleaner inside a training loop.
    """

    def __init__(self, reward_fn: StoryReward) -> None:
        self.reward_fn = reward_fn

    @torch.no_grad()
    def __call__(
        self,
        prompt_and_generated_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: torch.Tensor,
        metadata_list: Sequence[Dict[str, Any]],
        group_size: int,
        return_breakdown: bool = False,
    ) -> Dict[str, Any]:
        return self.reward_fn.score_grouped_from_token_ids(
            input_ids=prompt_and_generated_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            metadata_list=metadata_list,
            group_size=group_size,
            return_breakdown=return_breakdown,
        )


if __name__ == "__main__":
    example_grpo_usage()