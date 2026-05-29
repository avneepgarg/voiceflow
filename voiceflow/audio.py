"""Audio capture module - records microphone audio while hotkey is held."""

import numpy as np
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio recording."""
    sample_rate: int = 16000       # Whisper expects 16kHz
    channels: int = 1              # Mono
    dtype: str = "float32"
    chunk_duration: float = 0.1    # seconds per callback chunk
    max_duration: float = 120.0    # hard cap at 2 minutes
    silence_threshold: float = 0.01  # trim below this


class AudioRecorder:
    """
    Records audio from the microphone in real-time.

    Usage:
        recorder = AudioRecorder()
        recorder.start()       # begin recording (hotkey press)
        audio = recorder.stop()  # stop and get audio buffer (hotkey release)
    """

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._recording = False
        self._buffer = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        """Begin recording. Call from the hotkey press handler."""
        with self._lock:
            self._recording = True
            self._buffer = []

        try:
            import sounddevice as sd

            chunk_size = int(self.config.sample_rate * self.config.chunk_duration)

            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Recording started (device: %s)", self._stream.device)

        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            self._recording = False
            raise RuntimeError(
                f"Cannot start audio capture. Check microphone permissions and device. Error: {e}"
            )

    def stop(self) -> np.ndarray:
        """
        Stop recording and return the audio buffer.

        Returns:
            numpy float32 array of audio samples (16kHz, mono).
            Empty array if no audio was captured.
        """
        with self._lock:
            self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._buffer:
            logger.warning("No audio captured")
            return np.array([])

        audio = np.concatenate(self._buffer)
        audio = self._trim_silence(audio)
        audio = self._enforce_max_duration(audio)

        logger.info(
            "Recording stopped: %.1fs captured (%d samples)",
            len(audio) / self.config.sample_rate,
            len(audio),
        )
        return audio

    @property
    def is_recording(self) -> bool:
        return self._recording

    def list_devices(self) -> list:
        """List available audio input devices."""
        try:
            import sounddevice as sd

            devices = []
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": d["name"],
                            "channels": d["max_input_channels"],
                            "sample_rate": int(d["default_samplerate"]),
                        }
                    )
            return devices
        except Exception as e:
            logger.error("Failed to list audio devices: %s", e)
            return []

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            logger.warning("Audio stream status: %s", status)
        with self._lock:
            if self._recording:
                self._buffer.append(indata.copy())

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove leading and trailing silence from audio."""
        audio = audio.flatten()
        if len(audio) == 0:
            return audio

        mask = np.abs(audio) > self.config.silence_threshold
        if not mask.any():
            return np.array([])

        start = np.argmax(mask)
        end = len(audio) - np.argmax(mask[::-1])
        trimmed = audio[start:end]

        # Log how much we trimmed
        original_duration = len(audio) / self.config.sample_rate
        trimmed_duration = len(trimmed) / self.config.sample_rate
        if original_duration - trimmed_duration > 0.5:
            logger.debug(
                "Trimmed %.1fs of silence (%.1fs -> %.1fs)",
                original_duration - trimmed_duration,
                original_duration,
                trimmed_duration,
            )

        return trimmed

    def _enforce_max_duration(self, audio: np.ndarray) -> np.ndarray:
        """Cap audio at max_duration to prevent runaway recordings."""
        max_samples = int(self.config.sample_rate * self.config.max_duration)
        if len(audio) > max_samples:
            logger.warning(
                "Audio exceeded max duration (%.1fs), truncating to %.1fs",
                len(audio) / self.config.sample_rate,
                self.config.max_duration,
            )
            return audio[:max_samples]
        return audio
