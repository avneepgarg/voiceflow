"""Quick smoke test - verify all modules can be imported and basic functions work.
No heavy dependencies needed. Run: python3 tests/smoke_test.py
"""
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -- {e}")
        failed += 1


def test_audio_module():
    from voiceflow.audio import AudioRecorder, AudioConfig
    rec = AudioRecorder()
    assert rec.config.sample_rate == 16000

    # Silence trimming
    audio = np.zeros(16000, dtype="float32")
    audio[8000:] = 0.5
    trimmed = rec._trim_silence(audio)
    assert len(trimmed) < len(audio)

    # Empty
    empty = rec._trim_silence(np.array([]))
    assert len(empty) == 0


def test_transcriber_config():
    from voiceflow.transcriber import Transcriber, TranscriptionConfig
    config = TranscriptionConfig(model_size="base", device="cpu")
    assert config.model_size == "base"

    # Invalid model should raise
    try:
        TranscriptionConfig(model_size="invalid_model")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
def test_typer():
    from voiceflow.typer import Typer
    t = Typer()
    assert t.delay == 0.005


def test_config():
    from voiceflow.config import load_config, get_config_dir, DEFAULT_CONFIG
    from pathlib import Path
    assert "hotkey" in DEFAULT_CONFIG
    assert "llm" in DEFAULT_CONFIG

    # Test with temp config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json
        json.dump({"hotkey": "f9"}, f)
        tmp_path = f.name

    try:
        cfg = load_config(Path(tmp_path))
        assert cfg["hotkey"] == "f9"
        # Defaults merged
        assert "llm" in cfg
    finally:
        os.unlink(tmp_path)


def test_llm_postprocessor():
    from voiceflow.llm_postprocessor import LLMPostProcessor, LLMConfig
    processor = LLMPostProcessor(LLMConfig(enabled=False))
    text = "hello world um this is a test"
    result = processor.process(text)
    assert result == text  # Disabled = passthrough

    models = LLMPostProcessor.get_available_models()
    assert "openai/gpt-4o-mini" in models


def test_voice_commands():
    from voiceflow.voice_commands import VoiceCommandProcessor
    proc = VoiceCommandProcessor()

    cmd, text = proc.process("hello world new line")
    assert cmd is not None
    assert cmd.action == "press_key"
    assert text == "hello world"

    cmd, text = proc.process("hello world")
    assert cmd is None
    assert text == "hello world"

    cmd, text = proc.process("delete last word")
    assert cmd is not None
    assert cmd.action == "delete_words"


def test_profiles():
    from voiceflow.profiles import ProfileManager, BUILTIN_PROFILES
    pm = ProfileManager(profiles_path=None)
    pm.load_profiles()
    assert len(pm._profiles) >= len(BUILTIN_PROFILES)
    assert "vscode" in pm._profiles
    assert "gmail" in pm._profiles


def test_vocabulary():
    from voiceflow.vocabulary import VocabularyManager
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        vm = VocabularyManager(vocab_path=tmp)
        vm.add_terms(["kubernetes", "fastapi", "postgres"])
        assert "kubernetes" in vm._domain_terms
        hotwords = vm.get_hotwords()
        assert "kubernetes" in hotwords
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    print("VoiceFlow Smoke Tests")
    print("=" * 40)

    check("Audio module", test_audio_module)
    check("Transcriber config", test_transcriber_config)
    check("Typer", test_typer)
    check("Config system", test_config)
    check("LLM post-processor", test_llm_postprocessor)
    check("Voice commands", test_voice_commands)
    check("Profiles", test_profiles)
    check("Vocabulary", test_vocabulary)

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All checks passed!")
