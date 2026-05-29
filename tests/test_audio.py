"""Tests for the audio capture module."""

import numpy as np
from voiceflow.audio import AudioRecorder, AudioConfig


class TestAudioConfig:
    def test_defaults(self):
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.dtype == "float32"
        assert config.max_duration == 120.0

    def test_custom_values(self):
        config = AudioConfig(sample_rate=44100, channels=2)
        assert config.sample_rate == 44100
        assert config.channels == 2


class TestAudioRecorder:
    def test_init_default(self):
        rec = AudioRecorder()
        assert rec.config.sample_rate == 16000
        assert not rec.is_recording

    def test_init_custom_config(self):
        config = AudioConfig(sample_rate=48000)
        rec = AudioRecorder(config)
        assert rec.config.sample_rate == 48000

    def test_not_recording_initially(self):
        rec = AudioRecorder()
        assert rec.is_recording is False

    def test_trim_silence_empty(self):
        rec = AudioRecorder()
        result = rec._trim_silence(np.array([]))
        assert len(result) == 0

    def test_trim_silence_all_silence(self):
        rec = AudioRecorder()
        silent = np.zeros(16000, dtype="float32")
        result = rec._trim_silence(silent)
        assert len(result) == 0

    def test_trim_silence_removes_leading_silence(self):
        rec = AudioRecorder()
        audio = np.zeros(16000, dtype="float32")
        audio[8000:] = 0.5
        result = rec._trim_silence(audio)
        assert len(result) < len(audio)

    def test_trim_silence_keeps_speech(self):
        rec = AudioRecorder()
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype("float32") * 0.5
        result = rec._trim_silence(audio)
        assert len(result) > 0

    def test_enforce_max_duration(self):
        config = AudioConfig(max_duration=1.0)
        rec = AudioRecorder(config)
        audio = np.zeros(48000, dtype="float32")  # 3 seconds
        result = rec._enforce_max_duration(audio)
        assert len(result) == 16000  # truncated to 1 second

    def test_enforce_max_duration_under_limit(self):
        config = AudioConfig(max_duration=120.0)
        rec = AudioRecorder(config)
        audio = np.zeros(16000, dtype="float32")  # 1 second
        result = rec._enforce_max_duration(audio)
        assert len(result) == 16000  # no truncation
