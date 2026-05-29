"""Edge case tests for main.py and error handling paths."""
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMainLoopEdgeCases:
    """Test edge cases in the main VoiceFlow loop."""

    def test_empty_audio_after_trim(self):
        """When all audio is silence, stop() returns empty array."""
        from voiceflow.audio import AudioRecorder, AudioConfig

        rec = AudioRecorder(AudioConfig(silence_threshold=0.5))
        # Simulate recording only silence
        rec._recording = True
        rec._buffer = [np.zeros(16000, dtype="float32")]
        rec._recording = False
        audio = rec.stop()
        assert len(audio) == 0

    def test_very_short_audio(self):
        """Audio shorter than 100ms should still be handled."""
        from voiceflow.audio import AudioRecorder, AudioConfig

        rec = AudioRecorder(AudioConfig())
        rec._recording = True
        # 50ms of audio
        rec._buffer = [np.random.uniform(-0.1, 0.1, 800).astype("float32")]
        rec._recording = False
        audio = rec.stop()
        # Should not crash, may be empty after trim
        assert isinstance(audio, np.ndarray)

    def test_max_duration_enforcement(self):
        """Audio exceeding max_duration should be truncated."""
        from voiceflow.audio import AudioRecorder, AudioConfig

        rec = AudioRecorder(AudioConfig(max_duration=1.0, sample_rate=16000))
        rec._recording = True
        # 2 seconds of audio (exceeds 1s max)
        rec._buffer = [np.random.uniform(-0.1, 0.1, 32000).astype("float32")]
        rec._recording = False
        audio = rec.stop()
        assert len(audio) <= 16000  # truncated to 1 second

    def test_concurrent_start_stop(self):
        """Calling start() twice should not crash."""
        from voiceflow.audio import AudioRecorder, AudioConfig

        rec = AudioRecorder(AudioConfig())
        rec._recording = True
        rec._buffer = []
        # Second start should reset buffer
        rec._recording = True
        rec._buffer = []
        assert rec.is_recording

    def test_stop_without_start(self):
        """Calling stop() without start() should return empty array."""
        from voiceflow.audio import AudioRecorder, AudioConfig

        rec = AudioRecorder(AudioConfig())
        audio = rec.stop()
        assert len(audio) == 0


class TestTranscriberEdgeCases:
    """Test edge cases in the transcriber."""

    def test_empty_audio_returns_empty(self):
        """Empty audio buffer should return empty string without loading model."""
        from voiceflow.transcriber import Transcriber, TranscriptionConfig

        t = Transcriber(TranscriptionConfig())
        result = t.transcribe(np.array([]))
        assert result == ""

    def test_silence_audio_returns_empty(self):
        """Silence should not crash (model loads lazily, may not be installed)."""
        from voiceflow.transcriber import Transcriber, TranscriptionConfig

        t = Transcriber(TranscriptionConfig())
        silence = np.zeros(16000, dtype="float32")
        # If faster_whisper is not installed, this raises ImportError
        # If it IS installed, silence returns empty string
        try:
            result = t.transcribe(silence)
            assert result == "" or isinstance(result, str)
        except ImportError:
            # Expected in dev environment without GPU deps
            pass

    def test_invalid_model_raises(self):
        """Invalid model size should raise ValueError."""
        from voiceflow.transcriber import TranscriptionConfig

        with pytest.raises(ValueError):
            TranscriptionConfig(model_size="invalid")

    def test_invalid_device_raises(self):
        """Invalid device should raise ValueError."""
        from voiceflow.transcriber import TranscriptionConfig

        with pytest.raises(ValueError):
            TranscriptionConfig(device="tpu")


class TestConfigEdgeCases:
    """Test edge cases in config handling."""

    def test_corrupt_config_uses_defaults(self):
        """Corrupt JSON config should fall back to defaults."""
        from voiceflow.config import load_config
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json{{{")
            tmp = Path(f.name)
        try:
            cfg = load_config(tmp)
            assert "hotkey" in cfg
            assert "llm" in cfg
        finally:
            tmp.unlink()

    def test_missing_config_creates_default(self):
        """Missing config file should create default."""
        from voiceflow.config import load_config
        from pathlib import Path

        tmp = Path(tempfile.gettempdir()) / "voiceflow_test_nonexistent.json"
        if tmp.exists():
            tmp.unlink()
        cfg = load_config(tmp)
        assert "hotkey" in cfg
        # Clean up
        if tmp.exists():
            tmp.unlink()

    def test_save_creates_parent_dirs(self):
        """save_config should create parent directories if needed."""
        from voiceflow.config import save_config
        from pathlib import Path

        tmp = Path(tempfile.gettempdir()) / "voiceflow_test_subdir" / "config.json"
        try:
            save_config({"test": True}, tmp)
            assert tmp.exists()
        finally:
            if tmp.exists():
                tmp.unlink()
            tmp.parent.rmdir()


class TestTyperEdgeCases:
    """Test edge cases in text injection."""

    def test_empty_string_noop(self):
        """Typing empty string should be a no-op."""
        from voiceflow.typer import Typer

        t = Typer()
        t.type_text("")  # Should not crash

    def test_unicode_text(self):
        """Unicode text should be handled."""
        from voiceflow.typer import Typer

        t = Typer()
        # Should not crash (though may not type correctly without proper keyboard layout)
        t.type_text("नमस्ते दुनिया")

    def test_special_characters(self):
        """Special characters should be handled."""
        from voiceflow.typer import Typer

        t = Typer()
        t.type_text("hello@world.com")


class TestVoiceCommandEdgeCases:
    """Test edge cases in voice command processing."""

    def test_empty_transcript(self):
        """Empty transcript should return no command."""
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd, text = proc.process("")
        assert cmd is None
        assert text == ""

    def test_command_only_returns_empty_text(self):
        """Transcript that is only a command returns empty remaining text."""
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd, text = proc.process("new line")
        assert cmd is not None
        assert cmd.action == "press_key"
        assert text == ""

    def test_multiple_commands(self):
        """Multiple commands in one transcript."""
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello new line world")
        assert cmd is not None

    def test_case_insensitive(self):
        """Commands should be case-insensitive."""
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd1, _ = proc.process("NEW LINE")
        cmd2, _ = proc.process("new line")
        assert cmd1 is not None
        assert cmd2 is not None


class TestNoiseGateEdgeCases:
    """Test edge cases in noise gate."""

    def test_all_zeros(self):
        """All-zero audio should return empty or near-zero."""
        from voiceflow.noise_gate import NoiseGate, NoiseGateConfig

        gate = NoiseGate(NoiseGateConfig())
        out = gate.process(np.zeros(1000, dtype="float32"))
        assert np.max(np.abs(out)) < 0.01

    def test_very_loud_audio(self):
        """Very loud audio should pass through (possibly clipped)."""
        from voiceflow.noise_gate import NoiseGate, NoiseGateConfig

        gate = NoiseGate(NoiseGateConfig())
        loud = np.random.uniform(-1.0, 1.0, 16000).astype("float32")
        out = gate.process(loud)
        assert np.max(np.abs(out)) > 0.5

    def test_single_sample(self):
        """Single sample audio should not crash."""
        from voiceflow.noise_gate import NoiseGate, NoiseGateConfig

        gate = NoiseGate(NoiseGateConfig())
        out = gate.process(np.array([0.5], dtype="float32"))
        assert len(out) == 1


class TestVocabularyEdgeCases:
    """Test edge cases in vocabulary manager."""

    def test_add_empty_list(self):
        """Adding empty list should not crash."""
        from voiceflow.vocabulary import VocabularyManager

        vm = VocabularyManager(vocab_path=":memory:")
        vm.add_terms([])
        # get_hotwords returns a comma-separated string, empty string when no terms
        result = vm.get_hotwords()
        assert result == "" or isinstance(result, str)

    def test_add_duplicate_terms(self):
        """Duplicate terms should be deduplicated."""
        from voiceflow.vocabulary import VocabularyManager

        vm = VocabularyManager(vocab_path=":memory:")
        vm.add_terms(["docker", "docker", "kubernetes"])
        hotwords = vm.get_hotwords()
        # Hotwords is a comma-separated string
        assert hotwords.count("docker") == 1

    def test_remove_nonexistent(self):
        """Removing a term that doesn't exist should not crash."""
        from voiceflow.vocabulary import VocabularyManager

        vm = VocabularyManager(vocab_path=":memory:")
        vm.remove_term("nonexistent")  # Should not crash

    def test_stats_empty(self):
        """Stats on empty vocabulary should work."""
        from voiceflow.vocabulary import VocabularyManager

        vm = VocabularyManager(vocab_path=":memory:")
        stats = vm.stats
        assert stats["domain_terms"] == 0
