"""Tests for the vocabulary manager."""

import json
import tempfile
from pathlib import Path
from voiceflow.vocabulary import VocabularyManager


class TestVocabularyManager:
    def test_init_defaults(self):
        vm = VocabularyManager()
        assert vm._words.total() == 0
        assert len(vm._corrections) == 0
        assert len(vm._domain_terms) == 0

    def test_add_terms(self):
        vm = VocabularyManager()
        vm.add_terms(["kubernetes", "postgres", "fastapi"])
        assert "kubernetes" in vm._domain_terms
        assert "postgres" in vm._domain_terms
        assert len(vm._domain_terms) == 3

    def test_add_terms_deduplicates(self):
        vm = VocabularyManager()
        vm.add_terms(["docker", "docker", "kubernetes"])
        assert len(vm._domain_terms) == 2  # docker, kubernetes

    def test_add_terms_ignores_empty(self):
        vm = VocabularyManager()
        vm.add_terms(["", "  ", "valid"])
        assert "valid" in vm._domain_terms
        assert "" not in vm._domain_terms

    def test_learn_correction(self):
        vm = VocabularyManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            vm._vocab_path = Path(tmpdir) / "vocab.json"
            vm.learn_correction("avneep", "Avneep")
            assert vm._corrections["avneep"] == "Avneep"
            assert "avneep" in vm._words

    def test_learn_correction_extracts_words(self):
        vm = VocabularyManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            vm._vocab_path = Path(tmpdir) / "vocab.json"
            vm.learn_correction("hello world", "Hello world")
            # Should have learned individual words
            assert "hello" in vm._words or "world" in vm._words

    def test_get_hotwords_prioritizes_domain_terms(self):
        vm = VocabularyManager()
        vm._domain_terms = {"kubernetes", "docker"}
        vm._corrections = {"avneep": "Avneep"}
        hotwords = vm.get_hotwords()
        assert "kubernetes" in hotwords
        assert "docker" in hotwords

    def test_get_hotwords_limits_to_max(self):
        vm = VocabularyManager()
        vm._max_hotwords = 5
        vm.add_terms([f"term{i}" for i in range(20)])
        hotwords = vm.get_hotwords()
        # Should be limited to max_hotwords
        words = hotwords.split(", ")
        assert len(words) <= 5

    def test_get_hotwords_empty(self):
        vm = VocabularyManager()
        assert vm.get_hotwords() == ""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            vm = VocabularyManager(vocab_path=vocab_path)
            vm.add_terms(["docker", "kubernetes"])
            vm._corrections = {"avneep": "Avneep"}
            vm._words = __import__("collections").Counter({"test": 5})
            vm.save()
            assert vocab_path.exists()

            # Load into a new instance
            vm2 = VocabularyManager(vocab_path=vocab_path)
            vm2.load()
            assert "docker" in vm2._domain_terms
            assert "avneep" in vm2._corrections
            assert vm2._words["test"] == 5

    def test_stats(self):
        vm = VocabularyManager()
        vm._corrections = {"a": "b", "c": "d"}
        vm._domain_terms = {"x", "y"}
        vm._words = __import__("collections").Counter({"word": 3})
        stats = vm.stats
        assert stats["learned_corrections"] == 2
        assert stats["domain_terms"] == 2
        assert stats["word_frequency_entries"] == 1

    def test_remove_term(self):
        vm = VocabularyManager()
        vm._domain_terms = {"docker", "kubernetes"}
        vm.remove_term("docker")
        assert "docker" not in vm._domain_terms
        assert "kubernetes" in vm._domain_terms
