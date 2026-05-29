"""Tests for the noise gate module."""

import numpy as np
import pytest
from voiceflow.noise_gate import NoiseGate, NoiseGateConfig


class TestNoiseGateConfig:
    def test_defaults(self):
        cfg = NoiseGateConfig()
        assert cfg.open_threshold == 0.02
        assert cfg.close_threshold == 0.01
        assert cfg.attack_time == 0.005
        assert cfg.pre_gain == 1.5

    def test_custom(self):
        cfg = NoiseGateConfig(open_threshold=0.05, pre_gain=2.0)
        assert cfg.open_threshold == 0.05
        assert cfg.pre_gain == 2.0


class TestNoiseGate:
    def test_empty_audio(self):
        gate = NoiseGate()
        result = gate.process(np.array([]))
        assert len(result) == 0

    def test_silence_is_attenuated(self):
        """Silent audio should be reduced."""
        gate = NoiseGate()
        silence = np.zeros(16000, dtype=np.float32)
        output = gate.process(silence)
        # Silence should remain zero (or very close)
        assert np.max(np.abs(output)) < 0.001

    def test_loud_audio_passes_through(self):
        """Loud audio should pass through mostly intact."""
        gate = NoiseGate(NoiseGateConfig(open_threshold=0.01))
        loud = np.ones(16000, dtype=np.float32) * 0.5
        output = gate.process(loud)
        # Should be pre-gain * post-gain = 1.5 * 1.0 = 1.5x of 0.5 = 0.75
        # But gate may take a few samples to open
        rms_out = np.sqrt(np.mean(output ** 2))
        assert rms_out > 0.1  # Definitely not silenced

    def test_preserves_shape(self):
        gate = NoiseGate()
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.5
        output = gate.process(audio)
        assert output.shape == audio.shape

    def test_estimate_noise_empty(self):
        gate = NoiseGate()
        assert gate.estimate_noise(np.array([])) == 0.0

    def test_estimate_noise_returns_float(self):
        gate = NoiseGate()
        noise = np.random.randn(16000).astype(np.float32) * 0.005
        floor = gate.estimate_noise(noise)
        assert isinstance(floor, float)
        assert floor >= 0.0

    def test_auto_configure(self):
        gate = NoiseGate()
        noise = np.random.randn(16000).astype(np.float32) * 0.005
        gate.auto_configure(noise)
        assert gate.config.open_threshold > gate.config.close_threshold
        assert gate.config.noise_floor > 0.0

    def test_reset(self):
        gate = NoiseGate()
        audio = np.ones(16000, dtype=np.float32) * 0.5
        gate.process(audio)
        gate.reset()
        assert gate._state == "closed"
        assert gate._gain == 0.0

    def test_typical_speech_audio(self):
        """Simulated speech (sine wave bursts) — should passSpeech segments."""
        gate = NoiseGate(NoiseGateConfig(open_threshold=0.02, attack_time=0.001))
        audio = np.zeros(16000, dtype=np.float32)
        # Add "speech burst" in the middle
        audio[4000:12000] = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000).astype(np.float32) * 0.3
        output = gate.process(audio)
        # Middle section should have more energy than edges
        edge_rms = np.sqrt(np.mean(output[0:1000] ** 2))
        middle_rms = np.sqrt(np.mean(output[6000:10000] ** 2))
        assert middle_rms > edge_rms  # Speech should be louder than silence
