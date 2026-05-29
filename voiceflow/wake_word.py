"""Wake word detection -- activates VoiceFlow via a spoken trigger phrase.

Uses a lightweight energy-based wake word detector (no heavy ML).
For production use, integrate with Porcupine (free for personal use) via
the porcupine_integration() function.

This built-in detector uses a simple spectral fingerprint approach:
records a sample of your voice saying the wake phrase, computes a spectral
signature, and compares incoming audio chunks against it.
"""

import hashlib
import logging
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WakeWordConfig:
    """Wake word detection parameters."""
    wake_phrase: str = "hey voiceflow"
    sensitivity: float = 0.5           # 0.0 (strict) to 1.0 (permissive)
    sample_rate: int = 16000
    listen_duration: float = 0.5       # seconds per detection window
    cooldown_seconds: float = 3.0      # minimum time between activations
    min_trigger_count: int = 2         # consecutive matches to trigger
    training_samples: int = 3          # number of calibration samples


class WakeWordDetector:
    """
    Listens for a wake word to activate VoiceFlow.

    Usage:
        detector = WakeWordDetector()
        detector.train(["sample1.wav", "sample2.wav"])  # or use built-in calibration
        detector.start_listening(on_wake=callback)
    """

    def __init__(self, config: WakeWordConfig = None):
        self.config = config or WakeWordConfig()
        self._signature = None
        self._trained = False
        self._last_trigger = 0.0
        self._consecutive_matches = 0
        self._listener = None
        self._on_wake = None

    def train(self, sample_audio: np.ndarray, sample_rate: int = None):
        """
        Train the wake word detector from a sample of the wake phrase.

        Args:
            sample_audio: numpy float32 array containing the spoken wake phrase
            sample_rate: sample rate (default from config)
        """
        if sample_rate:
            self.config.sample_rate = sample_rate

        self._signature = self._compute_fingerprint(sample_audio)
        self._trained = True
        logger.info("Wake word trained with fingerprint: %s", self._signature[:16])

    def train_from_file(self, filepath: str):
        """Train from a WAV file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(filePath, dtype="float32")
            self.train(audio, sr)
        except ImportError:
            logger.error("soundfile required for load training. Install: pip install soundfile")

    def quick_calibrate(self, audio_chunks: list):
        """
        Calibrate from multiple samples.

        Args:
            audio_chunks: list of numpy float32 arrays, each with a clean
                         recording of the wake phrase
        """
        if not audio_chunks:
            raise ValueError("No calibration samples provided")

        fingerprints = [self._compute_fingerprint(chunk) for chunk in audio_chunks]
        # Average fingerprint
        fp_array = np.array(fingerprints, dtype=np.float64)
        self._signature = fp_array.mean(axis=0).astype(np.float32)
        self._trained = True
        logger.info("Wake word calibrated from %d samples", len(audio_chunks))

    def match(self, audio: np.ndarray) -> float:
        """
        Compare audio chunk against the trained wake word signature.

        Returns:
            Match score from 0.0 (no match) to 1.0 (perfect match).
        """
        if not self._trained:
            return 0.0

        fp = self._compute_fingerprint(audio)
        if fp.shape != self._signature.shape:
            return 0.0

        # Cosine similarity
        dot = np.dot(fp, self._signature)
        norm_fp = np.linalg.norm(fp)
        norm_sig = np.linalg.norm(self._signature)

        if norm_fp < 1e-10 or norm_sig < 1e-10:
            return 0.0

        similarity = float(dot / (norm_fp * norm_sig))
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        similarity = (similarity + 1.0) / 2.0

        return similarity

    def is_triggered(self, audio: np.ndarray) -> bool:
        """
        Check if the audio triggers the wake word.

        Includes cooldown logic and consecutive match requirement.
        """
        now = time.time()
        if now - self._last_trigger < self.config.cooldown_seconds:
            return False

        score = self.match(audio)
        threshold = 1.0 - self.config.sensitivity  # Higher sensitivity = lower threshold

        if score >= threshold:
            self._consecutive_matches += 1
            if self._consecutive_matches >= self.config.min_trigger_count:
                self._last_trigger = now
                self._consecutive_matches = 0
                logger.info("Wake word triggered! (score=%.2f)", score)
                return True
        else:
            self._consecutive_matches = max(0, self._consecutive_matches - 1)

        return False

    def start_listening(self, on_wake: Callable, blocking: bool = True):
        """
        Start listening in the background for the wake word.

        Args:
            on_wake: Callback function called when wake word is detected
            blocking: If True, blocks the calling thread
        """
        self._on_wake = on_wake

        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice required for wake word detection. Install: pip install sounddevice")
            return

        try:
            self._listener = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.config.sample_rate * self.config.listen_duration),
                callback=self._audio_callback,
            )
            self._listener.start()
            logger.info("Wake word detector listening for: '%s'", self.config.wake_phrase)

            if blocking:
                while True:
                    time.sleep(1)

        except Exception as e:
            logger.error("Wake word listener error: %s", e)

    def stop_listening(self):
        """Stop the wake word listener."""
        if self._listener:
            self._listener.stop()
            self._listener.close()
            self._listener = None
            logger.info("Wake word detector stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if not self._trained or self._on_wake is None:
            return

        audio = indata.flatten()
        if self.is_triggered(audio):
            try:
                self._on_wake()
            except Exception as e:
                logger.error("Wake word callback error: %s", e)

    def _compute_fingerprint(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute a spectral fingerprint of the audio.

        Uses MFCC-like features (simplified): mel-spaced energy bins.
        """
        audio = audio.flatten().astype(np.float32)
        if len(audio) < 100:
            return np.zeros(13, dtype=np.float32)

        # Compute power spectrum
        n_fft = min(2048, len(audio))
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.config.sample_rate)

        # Simple mel-spaced binning (13 bands -- like MFCC without DCT)
        n_mels = 13
        mel_points = np.linspace(0, self._hz_to_mel(freqs[-1] if len(freqs) > 0 else 8000), n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        fingerprint = np.zeros(n_mels, dtype=np.float32)
        for i in range(n_mels):
            band_mask = (freqs >= hz_points[i]) & (freqs < hz_points[i + 2])
            if band_mask.any():
                fingerprint[i] = np.mean(spectrum[band_mask])
            else:
                fingerprint[i] = 0.0

        # Log compress
        fingerprint = np.log1p(fingerprint)

        # Normalize (L2)
        norm = np.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint = fingerprint / norm

        return fingerprint

    @staticmethod
    def _hz_to_mel(hz):
        """Convert Hz to mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        """Convert mel scale to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def porcupine_integration(
    access_key: str,
    keyword_paths: list = None,
    model_path: str = None,
    sensitivity: float = 0.5,
    on_wake: Callable = None,
) -> dict:
    """
    Integration with Picovoice Porcupine for production-grade wake word detection.

    Porcupine is free for personal/commercial use with some limitations.
    Requires: pip install pvporcupine pvrecorder

    Args:
        access_key: Picovoice access key (free at console.picovoice.ai)
        keyword_paths: Paths to .ppn keyword files
        model_path: Path to .pv model file
        sensitivity: Detection sensitivity (0.0 to 1.0)
        on_wake: Callback when wake word is detected

    Returns:
        Dict with status and handle info
    """
    try:
        import pvporcuphene
        from pvrecorder import PvRecorder
    except ImportError:
        return {
            "status": "error",
            "error": "porcupine not installed. Run: pip install pvporcupene pvrecorder",
        }

    try:
        handle = pvporcuphene.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            model_path=model_path,
            sensitivities=[sensitivity],
        )

        logger.info("Porcupine wake word engine started")
        return {
            "status": "ok",
            "handle": handle,
            "sample_rate": handle.sample_rate,
            "frame_length": handle.frame_length,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
