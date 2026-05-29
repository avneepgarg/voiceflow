"""Transcription module using faster-whisper with GPU acceleration.

Lazy-loads the Whisper model on first use. Supports CPU and CUDA.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for Whisper transcription."""
    model_size: str = "base"          # tiny, base, small, medium
    device: str = "auto"              # cuda, cpu, or auto
    compute_type: str = "int8"       # int8, int8_float16, float16, float32
    language: str = None              # None = auto-detect. "en", "hi", etc.
    beam_size: int = 5
    vad_filter: bool = True           # Voice Activity Detection (removes silence)
    hotwords: str = ""                # Comma-separated words to boost

    def __post_init__(self):
        """Validate configuration."""
        valid_models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        if self.model_size not in valid_models:
            raise ValueError(
                f"Invalid model_size '{self.model_size}'. Must be one of: {valid_models}"
            )

        valid_devices = ["cpu", "cuda", "auto"]
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. Must be one of: {valid_devices}"
            )


class Transcriber:
    """
    Local Whisper transcription - private, free.

    Usage:
        transcriber = Transcriber(TranscriptionConfig(model_size="base", device="cuda"))
        text = transcriber.transcribe(audio_array)
        # or
        text = transcriber.transcribe_file("recording.wav")
    """

    def __init__(self, config: TranscriptionConfig = None):
        self.config = config or TranscriptionConfig()
        self._model = None

    @property
    def model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self._resolve_device()
            logger.info(
                "Loading Whisper model: %s on %s (%s)",
                self.config.model_size,
                device,
                self.config.compute_type,
            )

            self._model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=self.config.compute_type,
            )
            logger.info("Model loaded successfully")

        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe an audio buffer to text.

        Args:
            audio: numpy float32 array, 16kHz, mono

        Returns:
            Transcribed text string. Empty string if no speech detected.
        """
        if len(audio) == 0:
            logger.debug("Empty audio buffer, returning empty string")
            return ""

        try:
            segments, info = self.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                hotwords=self.config.hotwords or None,
            )

            text = " ".join(segment.text for segment in segments)
            result = text.strip()

            if result:
                logger.info(
                    "Transcribed (lang=%.2f, confidence=%.2f): %s",
                    info.language,
                    info.language_probability,
                    result[:100] + "..." if len(result) > 100 else result,
                )
            else:
                logger.debug("No speech detected in audio")

            return result

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            raise

    def transcribe_file(self, filepath: str) -> str:
        """
        Transcribe an audio file (WAV, MP3, etc).

        Args:
            filepath: path to an audio file

        Returns:
            Transcribed text string.
        """
        try:
            segments, info = self.model.transcribe(
                filepath,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
            )

            text = " ".join(segment.text for segment in segments)
            return text.strip()

        except Exception as e:
            logger.error("Transcription failed for file %s: %s", filepath, e)
            raise

    def _resolve_device(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                logger.info(
                    "CUDA available: %s",
                    torch.cuda.get_device_name(0),
                )
                return "cuda"
        except ImportError:
            pass

        logger.info("CUDA not available, using CPU")
        return "cpu"
