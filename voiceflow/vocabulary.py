"""Custom vocabulary manager - learns and applies user-specific words.

Tracks frequently misrecognized words, user corrections, and domain terms.
Passes them to Whisper as hotwords hints for better accuracy.
"""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)


class VocabularyManager:
    """
    Manages user-specific vocabulary for improved transcription.

    Usage:
        vm = VocabularyManager()
        vm.load()

        # Get words to pass to Whisper as hints
        hotwords = vm.get_hotwords()

        # After user corrects a transcription
        vm.learn_correction("avneep", "Avneep")

        # Add domain terms manually
        vm.add_terms(["kubernetes", "postgres", "fastapi"])
    """

    def __init__(self, vocab_path: Path = None):
        self._words = Counter()         # word -> frequency
        self._corrections = {}          # spoken -> correct
        self._domain_terms = set()      # manually added terms
        self._max_hotwords = 50          # Whisper hotword limit
        self._vocab_path = vocab_path

    def load(self):
        """Load vocabulary from disk."""
        if self._vocab_path is None:
            from voiceflow.config import get_config_dir
            self._vocab_path = get_config_dir() / "vocabulary.json"

        if not self._vocab_path.exists():
            logger.debug("No vocabulary file found, starting fresh")
            return

        try:
            with open(self._vocab_path, "r") as f:
                data = json.load(f)

            self._words = Counter(data.get("word_frequency", {}))
            self._corrections = data.get("corrections", {})
            self._domain_terms = set(data.get("domain_terms", []))

            total_words = len(self._words) + len(self._domain_terms)
            logger.info("Loaded vocabulary: %d words, %d corrections, %d domain terms",
                        len(self._words), len(self._corrections), len(self._domain_terms))

        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load vocabulary: %s", e)

    def save(self):
        """Save vocabulary to disk."""
        if self._vocab_path is None:
            return

        try:
            data = {
                "word_frequency": dict(self._words),
                "corrections": self._corrections,
                "domain_terms": sorted(self._domain_terms),
            }
            with open(self._vocab_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error("Failed to save vocabulary: %s", e)

    def get_hotwords(self) -> str:
        """
        Return the top N most frequent/relevant words as a comma-separated string
        for Whisper's hotwords parameter.
        """
        # Combine manual domain terms (highest priority) with frequent corrections
        words = list(self._domain_terms)[:self._max_hotwords]

        # Add corrections (what user actually wanted typed)
        for correction in self._corrections.values():
            if correction not in words and len(words) < self._max_hotwords:
                words.append(correction)

        # Add most frequent words up to the cap
        for word, count in self._words.most_common(self._max_hotwords):
            if word not in words and len(words) < self._max_hotwords:
                words.append(word)

        result = ", ".join(words[:self._max_hotwords])
        logger.debug("Hotwords for Whisper: %s", result[:100] + "..." if len(result) > 100 else result)
        return result

    def learn_correction(self, spoken: str, corrected: str):
        """
        Learn from a user correction.

        Args:
            spoken: What Whisper produced
            corrected: What the user actually wanted
        """
        self._corrections[spoken.lower().strip()] = corrected.strip()

        # Extract new words from the correction
        for word in corrected.split():
            clean = word.lower().strip(".,!?;:\"'()")
            if len(clean) > 2:  # Skip tiny words
                self._words[clean] += 1

        logger.debug("Learned correction: '%s' -> '%s'", spoken[:30], corrected[:30])
        self.save()

    def add_terms(self, terms: List[str]):
        """Add domain-specific terms (technical jargon, names, etc)."""
        for term in terms:
            clean = term.lower().strip()
            if clean:
                self._domain_terms.add(clean)
        if terms:
            logger.info("Added %d domain terms", len(terms))
            self.save()

    def remove_term(self, term: str):
        """Remove a term from domain vocabulary."""
        self._domain_terms.discard(term.lower().strip())
        self.save()

    @property
    def stats(self) -> dict:
        return {
            "learned_corrections": len(self._corrections),
            "domain_terms": len(self._domain_terms),
            "word_frequency_entries": len(self._words),
        }
