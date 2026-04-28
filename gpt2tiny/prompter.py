from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================
# Feature prompt variations
# ============================================================

FEATURE_NAMES: Dict[str, str] = {
    "bad_ending_prompt": "BadEnding",
    "conflict_prompt": "Conflict", 
    "dialogue_prompt": "Dialogue", 
    "foreshadowing_prompt": "Foreshadowing", 
    "moral_value_prompt": "MoralValue",
    "twist_prompt": "Twist"
}

FEATURE_VARIATIONS: Dict[str, List[str]] = {
    "bad_ending_prompt": [
        "the story has a bad ending",
        "the ending should be unhappy",
        "the story concludes on a bleak note",
        "the ending should turn out badly for someone",
        "the narrative should end in an unfortunate way",
        "the conclusion should not be happy",
    ],
    "conflict_prompt": [
        "the story has some form of conflict in it",
        "the narrative should include a conflict",
        "there should be a clear tension or struggle in the story",
        "the story should involve some kind of disagreement, obstacle, or clash",
        "the plot should contain a meaningful conflict",
        "the narrative should feature a problem or opposition that must be faced",
    ],
    "dialogue_prompt": [
        "the story should contain at least one dialogue",
        "include at least one exchange of dialogue",
        "the narrative should have at least one spoken conversation",
        "make sure at least one character speaks in dialogue",
        "the story should include quoted speech at least once",
        "include at least one moment of direct conversation between characters",
    ],
    "foreshadowing_prompt": [
        "the narrative uses foreshadowing or setup and payoff",
        "the story should include foreshadowing or a setup-and-payoff structure",
        "plant an earlier detail that becomes important later",
        "the narrative should hint at something early that pays off later",
        "include setup and payoff somewhere in the story",
        "the story should foreshadow a later event or reveal",
    ],
    "moral_value_prompt": [
        "the story has a moral value",
        "the story should communicate a moral lesson",
        "the narrative should contain a clear moral or takeaway",
        "the story should leave the reader with an ethical lesson",
        "the plot should reflect some moral value",
        "the story should carry a meaningful lesson",
    ],
    "twist_prompt": [
        "something unexpected happens / there is a plot twist",
        "the story should include an unexpected turn",
        "there should be a surprising twist in the plot",
        "something unforeseen should happen in the narrative",
        "the story should take an unexpected direction",
        "include a reveal or event that surprises the reader",
    ],
}


# ============================================================
# POS handling: nouns, verbs, adjectives only
# ============================================================

VALID_POS = {"noun", "verb", "adjective"}


def choose_indefinite_article(word: str) -> str:
    word = word.strip().lower()
    if not word:
        return "a"
    return "an" if word[0] in "aeiou" else "a"


# ============================================================
# Templates
# ============================================================

def template_1(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Write a short story."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_2(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Create a short fictional story."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_3(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Compose a narrative."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_4(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Write a brief tale."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_5(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Generate a story prompt."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_6(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Write an original story."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_7(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Tell a story."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_8(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Create a short narrative."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_9(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Write a piece of fiction."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


def template_10(subject_clause: str, word_clause: str, feature_clause: str) -> str:
    parts = ["Write a short story for a creative writing exercise."]
    if subject_clause:
        parts.append(subject_clause)
    if word_clause:
        parts.append(word_clause)
    if feature_clause:
        parts.append(feature_clause)
    return " ".join(parts)


DEFAULT_TEMPLATES = [
    template_1,
    template_2,
    template_3,
    template_4,
    template_5,
    template_6,
    template_7,
    template_8,
    template_9,
    template_10,
]


# ============================================================
# Word specification
# ============================================================

@dataclass(frozen=True)
class WordSpec:
    word: str
    pos: str

    def __post_init__(self) -> None:
        if not self.word.strip():
            raise ValueError("Word cannot be empty.")
        if self.pos not in VALID_POS:
            valid = ", ".join(sorted(VALID_POS))
            raise ValueError(
                f"Invalid part of speech: {self.pos!r}. Allowed values are: {valid}"
            )

    def to_requirement(self, style: str = "default") -> str:
        word_repr = f"'{self.word}'"
        article = choose_indefinite_article(self.pos)

        if style == "default":
            return f"use the {self.pos} {word_repr}"
        if style == "alt1":
            return f"include {word_repr} as {article} {self.pos}"
        if style == "alt2":
            return f"use {word_repr} in its role as {article} {self.pos}"
        if style == "alt3":
            return f"make sure {word_repr} appears as {article} {self.pos}"
        if style == "alt4":
            return f"work in the {self.pos} {word_repr}"
        if style == "alt5":
            return f"ensure that {word_repr} is used as {article} {self.pos}"
        if style == "alt6":
            return f"have {word_repr} function as {article} {self.pos}"
        return f"use the {self.pos} {word_repr}"


# ============================================================
# Subject specification
# ============================================================

@dataclass(frozen=True)
class StorySubject:
    character: str
    action: str
    place: str
    adjective: Optional[str] = None
    goal: Optional[str] = None

    def to_prompt_fragment(self, rng: random.Random) -> str:
        """
        Produce a subject phrase like:
          - Write about a pirate who goes to the White House.
          - Write about a nervous teacher who searches for a lost key in a museum.
        """
        char = self.character.strip()
        act = self.action.strip()
        place = self.place.strip()
        adj = (self.adjective or "").strip()
        goal = (self.goal or "").strip()

        if adj:
            character_phrase = f"{choose_indefinite_article(adj)} {adj} {char}"
        else:
            character_phrase = f"{choose_indefinite_article(char)} {char}"

        # A couple of structures for variety
        style = rng.choice(["basic", "goal", "place_first"])

        if goal and style == "goal":
            return f"Write about {character_phrase} who {act} in {place} while trying to {goal}."
        if goal and style == "place_first":
            return f"Write about {character_phrase} who goes to {place} in order to {goal}."
        return f"Write about {character_phrase} who {act} in {place}."


# ============================================================
# Generator
# ============================================================

@dataclass
class PromptGenerator:
    templates: List = field(default_factory=lambda: DEFAULT_TEMPLATES.copy())
    feature_variations: Dict[str, List[str]] = field(
        default_factory=lambda: {k: v.copy() for k, v in FEATURE_VARIATIONS.items()}
    )
    rng: random.Random = field(default_factory=random.Random)

    def get_feature_phrase(self, feature_key: str) -> str:
        if feature_key not in self.feature_variations:
            raise KeyError(f"Unknown feature key: {feature_key}")
        return self.rng.choice(self.feature_variations[feature_key])

    def get_feature_phrases(self, features: Sequence[str]) -> List[str]:
        return [self.get_feature_phrase(k) for k in features]

    def generate(
        self,
        selected_words: Optional[Sequence[Tuple[str, str]]] = None,
        features: Optional[Sequence[str]] = None,
        story_subject: Optional[StorySubject] = None,
        template_index: Optional[int] = None,
    ) -> str:
        """
        Generate one prompt.

        A prompt may contain:
          - a story subject
          - words
          - features

        But it may not contain none of the above.
        """
        word_specs = [WordSpec(word=w, pos=pos) for w, pos in (selected_words or [])]
        features = list(features or [])

        if not word_specs and not features and story_subject is None:
            raise ValueError(
                "A prompt cannot be generated with no subject, no words, and no features."
            )

        feature_phrases = self.get_feature_phrases(features)

        template_fn = (
            self.rng.choice(self.templates)
            if template_index is None
            else self.templates[template_index]
        )

        subject_clause, subject_statement  = self._build_subject_clause(story_subject)
        word_clause, word_statement = self._build_word_clause(word_specs)
        feature_clause, feature_statement = self._build_feature_clause(feature_phrases)

        prompt = template_fn(
            subject_clause=subject_clause,
            word_clause=word_clause,
            feature_clause=feature_clause,
        )

        required_words = []
        for word, pos in selected_words:
            required_words.append({"word": word, "pos": pos})
                
        return {
            "prompt": self._cleanup(prompt),
            "words": required_words,
            "features": [] if features is None else list(map(FEATURE_NAMES.get, features)),
            "subject": {} if story_subject is None else story_subject.__dict__,
            "feature_phrases": feature_phrases,
            "word_clause": word_statement,
            "feature_clause": feature_statement,
            "subject_clause": subject_statement,
        }

    def sample_words_by_pos(
        self,
        nouns: Optional[Sequence[str]] = None,
        verbs: Optional[Sequence[str]] = None,
        adjectives: Optional[Sequence[str]] = None,
        min_words: int = 0,
        max_words: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        pools = {
            "noun": self._dedupe_preserve_order(nouns or []),
            "verb": self._dedupe_preserve_order(verbs or []),
            "adjective": self._dedupe_preserve_order(adjectives or []),
        }

        for pos, words in pools.items():
            for word in words:
                WordSpec(word=word, pos=pos)

        all_candidates = [
            (word, pos)
            for pos, words in pools.items()
            for word in words
        ]
        total_available = len(all_candidates)

        if max_words is None:
            max_words = total_available

        if min_words < 0:
            raise ValueError("min_words cannot be negative.")
        if max_words < 0:
            raise ValueError("max_words cannot be negative.")
        if min_words > max_words:
            raise ValueError("min_words cannot be greater than max_words.")
        if min_words > total_available:
            raise ValueError(
                f"min_words={min_words} exceeds the number of available words ({total_available})."
            )

        max_words = min(max_words, total_available)
        k = self.rng.randint(min_words, max_words)

        if k == 0:
            return []

        return self.rng.sample(all_candidates, k)

    def sample_story_subject(
        self,
        subject_characters: Sequence[str],
        subject_actions: Sequence[str],
        subject_places: Sequence[str],
        subject_adjectives: Optional[Sequence[str]] = None,
        subject_goals: Optional[Sequence[str]] = None,
        allow_no_subject: bool = True,
    ) -> Optional[StorySubject]:
        """
        Randomly generate a story subject like:
          - a pirate who goes to the White House
          - a girl who goes to the mall
        """
        characters = self._dedupe_preserve_order(subject_characters)
        actions = self._dedupe_preserve_order(subject_actions)
        places = self._dedupe_preserve_order(subject_places)
        adjectives = self._dedupe_preserve_order(subject_adjectives or [])
        goals = self._dedupe_preserve_order(subject_goals or [])

        if not characters or not actions or not places:
            if allow_no_subject:
                return None
            raise ValueError(
                "To generate a story subject, subject_characters, subject_actions, and subject_places must all be non-empty."
            )

        if allow_no_subject and self.rng.random() < 0.25:
            return None

        return StorySubject(
            character=self.rng.choice(characters),
            action=self.rng.choice(actions),
            place=self.rng.choice(places),
            adjective=self.rng.choice(adjectives) if adjectives and self.rng.random() < 0.6 else None,
            goal=self.rng.choice(goals) if goals and self.rng.random() < 0.5 else None,
        )

    def generate_from_pos_lists(
        self,
        nouns: Optional[Sequence[str]] = None,
        verbs: Optional[Sequence[str]] = None,
        adjectives: Optional[Sequence[str]] = None,
        features: Optional[Sequence[str]] = None,
        subject_characters: Optional[Sequence[str]] = None,
        subject_actions: Optional[Sequence[str]] = None,
        subject_places: Optional[Sequence[str]] = None,
        subject_adjectives: Optional[Sequence[str]] = None,
        subject_goals: Optional[Sequence[str]] = None,
        allow_no_subject: bool = True,
        min_words: int = 0,
        max_words: Optional[int] = None,
        min_features: int = 0,
        max_features: Optional[int] = None,
        template_index: Optional[int] = None,
    ) -> str:
        selected_words = self.sample_words_by_pos(
            nouns=nouns,
            verbs=verbs,
            adjectives=adjectives,
            min_words=min_words,
            max_words=max_words,
        )

        selected_features = self._sample_features(
            features=features or [],
            min_features=min_features,
            max_features=max_features,
        )

        story_subject = self.sample_story_subject(
            subject_characters=subject_characters or [],
            subject_actions=subject_actions or [],
            subject_places=subject_places or [],
            subject_adjectives=subject_adjectives or [],
            subject_goals=subject_goals or [],
            allow_no_subject=allow_no_subject,
        )

        selected_words, selected_features, story_subject = self._ensure_nonempty_prompt_spec(
            selected_words=selected_words,
            selected_features=selected_features,
            story_subject=story_subject,
            nouns=nouns or [],
            verbs=verbs or [],
            adjectives=adjectives or [],
            all_features=features or [],
            subject_characters=subject_characters or [],
            subject_actions=subject_actions or [],
            subject_places=subject_places or [],
            subject_adjectives=subject_adjectives or [],
            subject_goals=subject_goals or [],
        )

        return self.generate(
            selected_words=selected_words,
            features=selected_features,
            story_subject=story_subject,
            template_index=template_index,
        )

    def generate_many_from_pos_lists(
        self,
        n: int,
        nouns: Optional[Sequence[str]] = None,
        verbs: Optional[Sequence[str]] = None,
        adjectives: Optional[Sequence[str]] = None,
        features: Optional[Sequence[str]] = None,
        subject_characters: Optional[Sequence[str]] = None,
        subject_actions: Optional[Sequence[str]] = None,
        subject_places: Optional[Sequence[str]] = None,
        subject_adjectives: Optional[Sequence[str]] = None,
        subject_goals: Optional[Sequence[str]] = None,
        allow_no_subject: bool = True,
        min_words: int = 0,
        max_words: Optional[int] = None,
        min_features: int = 0,
        max_features: Optional[int] = None,
    ) -> List[str]:
        prompts = []
        for _ in range(n):
            prompts.append(
                self.generate_from_pos_lists(
                    nouns=nouns,
                    verbs=verbs,
                    adjectives=adjectives,
                    features=features,
                    subject_characters=subject_characters,
                    subject_actions=subject_actions,
                    subject_places=subject_places,
                    subject_adjectives=subject_adjectives,
                    subject_goals=subject_goals,
                    allow_no_subject=allow_no_subject,
                    min_words=min_words,
                    max_words=max_words,
                    min_features=min_features,
                    max_features=max_features,
                )
            )
        return prompts

    def _sample_features(
        self,
        features: Sequence[str],
        min_features: int = 0,
        max_features: Optional[int] = None,
    ) -> List[str]:
        features = list(features)
        if not features:
            if min_features > 0:
                raise ValueError("min_features > 0, but no feature keys were provided.")
            return []

        if max_features is None:
            max_features = len(features)

        if min_features < 0:
            raise ValueError("min_features cannot be negative.")
        if max_features < 0:
            raise ValueError("max_features cannot be negative.")
        if min_features > max_features:
            raise ValueError("min_features cannot be greater than max_features.")
        if min_features > len(features):
            raise ValueError(
                f"min_features={min_features} exceeds the number of available features ({len(features)})."
            )

        max_features = min(max_features, len(features))
        k = self.rng.randint(min_features, max_features)
        return self.rng.sample(features, k)

    def _ensure_nonempty_prompt_spec(
        self,
        selected_words: List[Tuple[str, str]],
        selected_features: List[str],
        story_subject: Optional[StorySubject],
        nouns: Sequence[str],
        verbs: Sequence[str],
        adjectives: Sequence[str],
        all_features: Sequence[str],
        subject_characters: Sequence[str],
        subject_actions: Sequence[str],
        subject_places: Sequence[str],
        subject_adjectives: Sequence[str],
        subject_goals: Sequence[str],
    ) -> Tuple[List[Tuple[str, str]], List[str], Optional[StorySubject]]:
        """
        Prevent the illegal case where subject, words, and features are all empty.
        """
        if selected_words or selected_features or story_subject is not None:
            return selected_words, selected_features, story_subject

        word_candidates = []
        word_candidates.extend((w, "noun") for w in self._dedupe_preserve_order(nouns))
        word_candidates.extend((w, "verb") for w in self._dedupe_preserve_order(verbs))
        word_candidates.extend((w, "adjective") for w in self._dedupe_preserve_order(adjectives))
        feature_candidates = list(dict.fromkeys(all_features))

        can_make_subject = bool(subject_characters and subject_actions and subject_places)

        choices = []
        if word_candidates:
            choices.append("word")
        if feature_candidates:
            choices.append("feature")
        if can_make_subject:
            choices.append("subject")

        if not choices:
            raise ValueError(
                "Cannot generate a prompt: no subject material, no words, and no features were provided."
            )

        forced = self.rng.choice(choices)

        if forced == "word":
            return [self.rng.choice(word_candidates)], [], None

        if forced == "feature":
            return [], [self.rng.choice(feature_candidates)], None

        forced_subject = StorySubject(
            character=self.rng.choice(self._dedupe_preserve_order(subject_characters)),
            action=self.rng.choice(self._dedupe_preserve_order(subject_actions)),
            place=self.rng.choice(self._dedupe_preserve_order(subject_places)),
            adjective=(
                self.rng.choice(self._dedupe_preserve_order(subject_adjectives))
                if subject_adjectives and self.rng.random() < 0.6
                else None
            ),
            goal=(
                self.rng.choice(self._dedupe_preserve_order(subject_goals))
                if subject_goals and self.rng.random() < 0.5
                else None
            ),
        )
        return [], [], forced_subject

    def _build_subject_clause(self, story_subject: Optional[StorySubject]) -> str:
        if story_subject is None:
            return "", None

        fragment = story_subject.to_prompt_fragment(self.rng)

        # Convert "Write about ..." into a clause that fits after a lead sentence
        variants = [
            fragment,
            fragment.replace("Write about", "Make it about", 1),
            fragment.replace("Write about", "The story should be about", 1),
            fragment.replace("Write about", "Center it on", 1),
        ]
        return self.rng.choice(variants), fragment.replace("Write about", "", 1).strip()

    def _build_word_clause(self, words: List[WordSpec]) -> str:
        if not words:
            return "", None

        word_styles = ["default", "alt1", "alt2", "alt3", "alt4", "alt5", "alt6"]
        requirements = [
            w.to_requirement(style=self.rng.choice(word_styles))
            for w in words
        ]
        joined = self._join_items(requirements)

        starters = [
            f"The story should {joined}.",
            f"Your tale should {joined}.",
            f"Make sure the narrative {joined}.",
            f"In the story, {joined}.",
            f"The narrative must {joined}.",
        ]
        return self.rng.choice(starters), joined

    def _build_feature_clause(self, feature_phrases: List[str]) -> str:
        if not feature_phrases:
            return "", None

        cleaned = [x.rstrip(". ").strip() for x in feature_phrases if x.strip()]
        joined = self._join_items(cleaned)

        starters = [
            f"Additional requirements: {joined}.",
            f"Also, {joined}.",
            f"The story should also satisfy the following: {joined}.",
            f"In addition, {joined}.",
            f"Also ensure that {joined}.",
        ]
        return self.rng.choice(starters), joined

    @staticmethod
    def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def _join_items(items: Sequence[str]) -> str:
        items = [str(x).strip() for x in items if str(x).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    @staticmethod
    def _cleanup(text: str) -> str:
        while "  " in text:
            text = text.replace("  ", " ")
        return (
            text.replace(" .", ".")
            .replace(" ,", ",")
            .replace("..", ".")
            .strip()
        )