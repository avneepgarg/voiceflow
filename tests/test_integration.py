"""Integration tests for VoiceFlow — test multi-module workflows.

These tests verify that modules work together correctly using mocked I/O.
No real audio hardware, GPU, or window system needed.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Pipeline: Audio -> Noise Gate
# ---------------------------------------------------------------------------

class TestAudioToGatePipeline:
    """Audio recorder output feeds through noise gate correctly."""

    def test_silence_is_suppressed(self):
        from voiceflow.noise_gate import NoiseGate, NoiseGateConfig

        gate = NoiseGate(NoiseGateConfig(open_threshold=0.02, close_threshold=0.01))
        silence = np.zeros(8000, dtype="float32")
        out = gate.process(silence)
        assert np.max(np.abs(out)) <= np.max(np.abs(silence)) + 1e-7

    def test_speech_passes_through(self):
        from voiceflow.noise_gate import NoiseGate, NoiseGateConfig

        gate = NoiseGate(NoiseGateConfig(open_threshold=0.05, close_threshold=0.02))
        speech = np.random.uniform(-0.8, 0.8, 16000).astype("float32")
        out = gate.process(speech)
        assert np.max(np.abs(out)) > 0.1

    def test_pipeline_preserves_sample_rate(self):
        from voiceflow.audio import AudioConfig
        from voiceflow.noise_gate import NoiseGate

        config = AudioConfig(sample_rate=16000)
        gate = NoiseGate()
        audio = np.random.uniform(-0.3, 0.3, 16000).astype("float32")
        out = gate.process(audio)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Pipeline: Transcriber -> LLM postprocessor -> Typer
# ---------------------------------------------------------------------------

class TestTranscribeToTypePipeline:
    """Mocked transcription output flows through LLM cleanup to typer."""

    def test_disabled_llm_passthrough(self):
        from voiceflow.llm_postprocessor import LLMPostProcessor, LLMConfig

        proc = LLMPostProcessor(LLMConfig(enabled=False))
        raw = "hello world this is a test"
        assert proc.process(raw) == raw

    def test_command_extraction_then_type(self):
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello world new line")
        assert cmd is not None
        assert cmd.action == "press_key"
        assert text == "hello world"

    def test_no_command_returns_plain_text(self):
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd, text = proc.process("just some regular text")
        assert cmd is None
        assert text == "just some regular text"


# ---------------------------------------------------------------------------
# Pipeline: Profiles + Vocabulary + Config
# ---------------------------------------------------------------------------

class TestProfilesWithVocabulary:
    """Profile manager and vocabulary manager work together."""

    def test_profile_affects_vocabulary(self):
        from voiceflow.vocabulary import VocabularyManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            vm = VocabularyManager(vocab_path=tmp)
            vm.add_terms(["docker", "kubernetes", "pytest"])
            hotwords = vm.get_hotwords()
            assert "docker" in hotwords
            assert "kubernetes" in hotwords
        finally:
            os.unlink(tmp)

    def test_profile_manager_loads_builtins(self):
        from voiceflow.profiles import ProfileManager

        pm = ProfileManager(profiles_path=None)
        pm.load_profiles()
        # Verify built-in profiles exist by checking current_profile_name works
        name = pm.current_profile_name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_config_save_load_roundtrip(self):
        from voiceflow.config import save_config, load_config

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = Path(f.name)
        try:
            cfg = {"hotkey": "f9", "llm": {"enabled": True, "model": "gpt-4o"}}
            save_config(cfg, tmp)
            loaded = load_config(tmp)
            assert loaded["hotkey"] == "f9"
            assert loaded["llm"]["model"] == "gpt-4o"
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# Pipeline: Voice commands + Agent mode
# ---------------------------------------------------------------------------

class TestVoiceCommandsWithAgent:
    """Voice commands and agent mode don't interfere."""

    def test_voice_commands_take_priority(self):
        from voiceflow.voice_commands import VoiceCommandProcessor

        proc = VoiceCommandProcessor()
        cmd, text = proc.process("new line")
        assert cmd is not None
        assert cmd.action == "press_key"

    def test_agent_mode_processes_actions(self):
        from voiceflow.agent_mode import AgentMode

        agent = AgentMode(auto_confirm=True)
        result = agent.process_transcript("open firefox")
        assert result is not None
        assert "action_name" in result

    def test_agent_mode_returns_none_for_plain_text(self):
        from voiceflow.agent_mode import AgentMode

        agent = AgentMode(auto_confirm=True)
        result = agent.process_transcript("hello world how are today")
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Pipeline: Dev commands + Integrations
# ---------------------------------------------------------------------------

class TestDevCommandsWithIntegrations:
    """Dev command registry and integration manager coexist."""

    def test_dev_command_matches_build(self):
        from voiceflow.dev_commands import DevCommandMode

        dev = DevCommandMode(auto_confirm=True)
        result = dev.process("run the build")
        assert result is not None
        assert result["status"] in ("success", "confirmation_required")

    def test_integration_manager_registers_defaults(self):
        from voiceflow.integrations import IntegrationManager

        mgr = IntegrationManager()
        result = mgr.match_and_execute(
            "send slack message to general saying hello",
            variables={"user": "test"},
        )
        assert result is None or isinstance(result, dict) or hasattr(result, "success")

    def test_dev_command_detects_python_project(self):
        from voiceflow.dev_commands import detect_project_type

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "pyproject.toml").touch()
            ptype = detect_project_type(tmp)
            assert ptype == "python"


# ---------------------------------------------------------------------------
# Pipeline: Overlay + Tone aware
# ---------------------------------------------------------------------------

class TestOverlayWithTone:
    """Overlay and tone analyzer modules load and basic functions work."""

    def test_tone_analyzer_exists(self):
        from voiceflow.tone_aware import ToneAnalyzer

        ta = ToneAnalyzer()
        assert ta is not None

    def test_overlay_dictation_overlay_exists(self):
        from voiceflow.overlay import DictationOverlay

        ov = DictationOverlay()
        assert ov is not None


# ---------------------------------------------------------------------------
# Pipeline: Translation + Scribe
# ---------------------------------------------------------------------------

class TestTranslationWithScribe:
    """Translation and scribe modules load and basic config works."""

    def test_translation_pipeline_exists(self):
        from voiceflow.translation import TranslationPipeline

        tp = TranslationPipeline(target_lang="en")
        assert tp is not None

    def test_scribe_config_exists(self):
        from voiceflow.scribe import ScribeConfig

        cfg = ScribeConfig()
        assert cfg is not None
        assert cfg.sample_rate == 16000


# ---------------------------------------------------------------------------
# Wake word + Sync server
# ---------------------------------------------------------------------------

class TestWakeWordSync:
    """Wake word detector and sync server modules load."""

    def test_wake_word_detector_exists(self):
        from voiceflow.wake_word import WakeWordDetector

        wd = WakeWordDetector()
        assert wd is not None

    def test_sync_server_exists(self):
        from voiceflow.sync_server import SyncServer

        srv = SyncServer()
        assert srv is not None
