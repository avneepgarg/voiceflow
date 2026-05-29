"""Noise gate -- filter background noise before sending to Whisper.

Simple RMS-based noise gate with configurable thresholds.
No heavy dependencies -- works with numpy only.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class NoiseGateConfig:
    """Noise gate parameters."""
    # Gate opens when RMS exceeds this threshold
    open_threshold: float = 0.02
    # Gate closes when RMS drops below this threshold
    close_threshold: float = 0.01
    # Hysteresis ratio (close = open / hysteresis)
    hysteresis: float = 2.0
    # Attack time (seconds) -- how fast gate opens
    attack_time: float = 0.005
    # Release time (seconds) -- how fast gate closes
    release_time: float = 0.05
    # Minimum noise floor (absolute)
    noise_floor: float = 0.001
    # Pre-gain multiplier (boost quiet signals)
    pre_gain: float = 1.5
    # Post-gain multiplier
    post_gain: float = 1.0


class NoiseGate:
    """
    Applies a noise gate to audio to remove background noise.

    Usage:
        gate = NoiseGate()
        clean_audio = gate.process(audio_array)
        # Or chain with recording:
        noise_db = gate.estimate_noise(audio_array)
    """

    def __init__(self, config: NoiseGateConfig = None):
        self.config = config or NoiseGateConfig()
        self._state = "closed"  # "open", "closed"
        self._gain = 0.0       # current gain (0-1)
        self._sample_rate = 16000

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply noise gate to audio.

        Args:
            audio: numpy float32 array (16kHz, mono)
            sample_rate: sample rate of the audio

        Returns:
            Filtered audio array (same shape as input)
        """
        if len(audio) == 0:
            return audio

        self._sample_rate = sample_rate
        audio = audio.flatten().astype(np.float32)

        # Apply pre-gain
        audio = audio * self.config.pre_gain

        # Calculate per-frame RMS
        frame_size = int(sample_rate * 0.01)  # 10ms frames
        if frame_size < 1:
            frame_size = 1

        output = np.zeros_like(audio)
        attack_coeff = 1.0 - np.exp(-1.0 / (sample_rate * self.config.attack_time))
        release_coeff = 1.0 - np.exp(-1.0 / (sample_rate * self.config.release_time))

        for i in range(0, len(audio), frame_size):
            chunk = audio[i:i + frame_size]
            if len(chunk) == 0:
                break

            rms = np.sqrt(np.mean(chunk ** 2))

            # State machine with hysteresis
            if rms > self.config.open_threshold:
                self._state = "open"
            elif rms < self.config.close_threshold:
                self._state = "closed"

            # Smooth gain transitions
            if self._state == "open":
                self._gain = min(1.0, self._gain + attack_coeff)
            else:
                self._gain = max(0.0, self._gain - release_coeff)

            output[i:i + frame_size] = chunk * self._gain

        # Apply post-gain
        output = output * self.config.post_gain

        # Log stats
        input_rms = np.sqrt(np.mean(audio ** 2))
        output_rms = np.sqrt(np.mean(output ** 2))
        if input_rms > 0:
            reduction_db = 20 * np.log10(output_rms / input_rms) if output_rms > 0 else -60
            logger.debug(
                "Noise gate: input_rms=%.4f output_rms=%.4f reduction=%.1fdb state=%s",
                input_rms, output_rms, reduction_db, self._state,
            )

        return output

    def estimate_noise(self, audio: np.ndarray) -> float:
        """
        Estimate the noise floor of an audio sample.

        Returns:
            RMS energy of the quietest segments (noise floor estimate).
        """
        if len(audio) == 0:
            return 0.0

        audio = audio.flatten().astype(np.float32)

        # Split into 100ms frames
        frame_size = int(self._sample_rate * 0.1)
        if frame_size < 1 or len(audio) < frame_size:
            return float(np.sqrt(np.mean(audio ** 2)))

        n_frames = len(audio) // frame_size
        frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
        rms_values = np.sqrt(np.mean(frames ** 2, axis=1))

        # Use the 10th percentile as noise floor estimate
        noise_floor = float(np.percentile(rms_values, 10))
        logger.debug("Estimated noise floor: %.4f", noise_floor)
        return noise_floor

    def auto_configure(self, audio: np.ndarray):
        """
        Automatically configure thresholds based on a sample of background noise.

        Use this with a few seconds of room silence to set optimal thresholds.
        """
        noise = self.estimate_noise(audio)
        self.config.open_threshold = max(noise * self.config.hysteresis, 0.01)
        self.config.close_threshold = max(noise, 0.005)
        self.config.noise_floor = noise

        logger.info(
            "Auto-configured noise gate: open=%.4f close=%.4f noise_floor=%.4f",
            self.config.open_threshold, self.config.close_threshold, noise,
        )

    def reset(self):
        """Reset gate state."""
        self._state = "closed"
        self._gain = 0.0
