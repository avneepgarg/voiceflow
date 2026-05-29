"""Tone-Aware Dictation Module.

Detects emotional tone from audio features and adjusts LLM cleanup
accordingly. Uses numpy-only signal processing (no librosa dependency).

Pipeline:
    audio_array -> ToneAnalyzer -> ToneResult -> ToneAwareProcessor
    -> modified system prompt -> LLMPostProcessor

Tone classifications:
    - neutral:  Default, balanced delivery
    - excited:  High energy, fast pace, varied pitch
    - formal:   Measured pace, low variation, careful enunciation
    - casual:   Relaxed energy, conversational rhythm
    - angry:    High energy, sharp attacks, harsh spectral profile
"""

import logging
import enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tone classification enum
# ---------------------------------------------------------------------------

class ToneType(enum.Enum):
    """Supported emotional tone classifications."""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    FORMAL = "formal"
    CASUAL = "casual"
    ANGRY = "angry"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ToneConfig:
    """Configuration for tone detection thresholds.

    All thresholds are derived from empirical testing on 16kHz mono audio.
    Adjust per-user via the VoiceFlow config system.

    Attributes:
        rms_threshold_high:   RMS energy above this = "high energy" (excited/angry).
        rms_threshold_low:    RMS energy below this = "low energy" (formal/casual).
        zcr_threshold_high:   Zero-crossing rate above this = harsh/bright sound.
        zcr_threshold_low:    ZCR below this = smooth/warm sound.
        spectral_threshold:   Spectral centroid dividing line (Hz). Higher = brighter.
        tempo_threshold_fast: BPM above this = fast delivery.
        tempo_threshold_slow: BPM below this = slow/deliberate delivery.
        tone_types:           List of tone types the classifier may return.
        sample_rate:          Assumed sample rate for tempo/ZCR calculations.
        frame_size:           Samples per analysis frame (default 2048 ~128ms @16kHz).
        hop_size:             Samples between frame starts (default 512).
    """
    rms_threshold_high: float = 0.08
    rms_threshold_low: float = 0.02
    zcr_threshold_high: float = 0.15
    zcr_threshold_low: float = 0.05
    spectral_threshold: float = 3000.0
    tempo_threshold_fast: float = 160.0
    tempo_threshold_slow: float = 110.0
    tone_types: list = field(default_factory=lambda: [t.value for t in ToneType])
    sample_rate: int = 16000
    frame_size: int = 2048
    hop_size: int = 512


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ToneResult:
    """Container for a tone analysis result.

    Attributes:
        tone:        The classified ToneType.
        confidence:  0.0-1.0 confidence score (higher = more certain).
        features:    Dict of extracted audio features for debugging/extension.
    """
    tone: str
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ToneResult(tone={self.tone!r}, confidence={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Audio feature extractor
# ---------------------------------------------------------------------------

class ToneAnalyzer:
    """Extracts audio features from a raw numpy array and classifies tone.

    Uses only numpy for signal processing -- no librosa, no scipy.

    Usage:
        analyzer = ToneAnalyzer(ToneConfig())
        result = analyzer.analyze(audio_array)
        print(result.tone, result.confidence)
    """

    def __init__(self, config: Optional[ToneConfig] = None):
        self.config = config or ToneConfig()
        logger.debug("ToneAnalyzer initialised with sample_rate=%d",
                     self.config.sample_rate)

    # ---- public API -------------------------------------------------------

    def analyze(self, audio: np.ndarray) -> ToneResult:
        """Analyse an audio array and return a ToneResult.

        Args:
            audio: 1-D numpy float32 array (mono), typically 16kHz.

        Returns:
            ToneResult with tone classification and feature breakdown.
        """
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio passed to ToneAnalyzer; returning neutral")
            return ToneResult(tone=ToneType.NEUTRAL.value, confidence=0.0)

        audio = self._validate(audio)
        features = self._extract_features(audio)
        tone, confidence = self._classify(features)

        logger.info("Tone detected: %s (confidence=%.2f)", tone, confidence)
        return ToneResult(tone=tone, confidence=confidence, features=features)

    # ---- internal: validation / pre-processing ----------------------------

    @staticmethod
    def _validate(audio: np.ndarray) -> np.ndarray:
        """Flatten to mono, ensure float64, clip to [-1, 1]."""
        audio = audio.flatten().astype(np.float64)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        return audio

    # ---- internal: feature extraction -------------------------------------

    def _extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract all features and return as a dict.

        Keys: rms_energy, rms_variance, zcr_mean, zcr_variance,
              spectral_centroid_mean, spectral_centroid_variance,
              tempo_bpm, energy_entropy, silence_ratio
        """
        frame_size = self.config.frame_size
        hop_size = self.config.hop_size

        # Frame the signal
        frames = self._frame_signal(audio, frame_size, hop_size)

        # RMS energy per frame
        rms_frames = np.array([
            np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0
            for frame in frames
        ])

        # Zero-crossing rate per frame
        zcr_frames = np.array([
            self._zero_crossing_rate(frame) for frame in frames
        ])

        # Spectral centroid per frame (via FFT magnitude)
        centroid_frames = np.array([
            self._spectral_centroid(frame) for frame in frames
        ])

        # Tempo estimation via onset envelope autocorrelation
        tempo = self._estimate_tempo(rms_frames)

        # Energy entropy (how uniformly energy is spread across frames)
        energy_entropy = self._energy_entropy(rms_frames)

        # Silence ratio: fraction of frames below a low-energy gate
        silence_gate = self.config.rms_threshold_low * 0.5
        silence_ratio = float(np.mean(rms_frames < silence_gate))

        features = {
            "rms_energy": float(np.mean(rms_frames)),
            "rms_variance": float(np.var(rms_frames)),
            "zcr_mean": float(np.mean(zcr_frames)),
            "zcr_variance": float(np.var(zcr_frames)),
            "spectral_centroid_mean": (
                float(np.mean(centroid_frames[np.isfinite(centroid_frames)]))
                if np.any(np.isfinite(centroid_frames)) else 0.0
            ),
            "spectral_centroid_variance": (
                float(np.var(centroid_frames[np.isfinite(centroid_frames)]))
                if np.any(np.isfinite(centroid_frames)) else 0.0
            ),
            "tempo_bpm": float(tempo),
            "energy_entropy": float(energy_entropy),
            "silence_ratio": float(silence_ratio),
        }
        logger.debug("Audio features: %s", features)
        return features

    # ---- internal: signal primitives --------------------------------------

    def _frame_signal(
        self, audio: np.ndarray, frame_size: int, hop_size: int
    ) -> list:
        """Split audio into overlapping frames (list of numpy arrays)."""
        frames = []
        for start in range(0, len(audio) - frame_size + 1, hop_size):
            frames.append(audio[start:start + frame_size])
        if not frames:
            frames.append(audio)  # shorter than one frame
        return frames

    @staticmethod
    def _zero_crossing_rate(frame: np.ndarray) -> float:
        """Compute zero-crossing rate of a frame."""
        signs = np.sign(frame)
        signs[signs == 0] = 1  # treat zeros as positive
        crossings = np.abs(np.diff(signs))
        return float(np.sum(crossings) / (2 * len(frame)))

    def _spectral_centroid(self, frame: np.ndarray) -> float:
        """Compute spectral centroid (brightness) of a frame via FFT."""
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / self.config.sample_rate)
        magnitude_sum = np.sum(spectrum)
        if magnitude_sum == 0:
            return 0.0
        return float(np.sum(freqs * spectrum) / magnitude_sum)

    def _estimate_tempo(self, rms_frames: np.ndarray) -> float:
        """Rough BPM estimate from onset-strength autocorrelation.

        Uses the RMS envelope as a proxy for onset strength.
        """
        if len(rms_frames) < 4:
            return 120.0  # default

        # Center the envelope
        env = rms_frames - np.mean(rms_frames)
        n = len(env)

        # Autocorrelation via numpy correlate (full, then keep positive lags)
        autocorr = np.correlate(env, env, mode="full")
        autocorr = autocorr[n - 1:]  # keep lags >= 0

        if len(autocorr) < 2:
            return 120.0

        # Ignore the zero-lag peak (index 0). Look for the peak in a
        # plausible tempo range: 60-240 BPM.
        min_lag = int(self.config.sample_rate * 60.0 / (240.0 * self.config.hop_size))
        max_lag = int(self.config.sample_rate * 60.0 / (60.0 * self.config.hop_size))
        max_lag = min(max_lag, len(autocorr) - 1)
        min_lag = max(min_lag, 1)

        if min_lag >= max_lag:
            return 120.0

        search_region = autocorr[min_lag:max_lag + 1]
        best_lag_offset = int(np.argmax(search_region))
        best_lag = min_lag + best_lag_offset

        # Convert lag (in frames) to BPM
        seconds_per_frame = self.config.hop_size / self.config.sample_rate
        bpm = 60.0 / (best_lag * seconds_per_frame)
        return float(np.clip(bpm, 60.0, 240.0))

    @staticmethod
    def _energy_entropy(rms_frames: np.ndarray) -> float:
        """Shannon entropy of the energy distribution across frames."""
        energy = rms_frames ** 2
        total = np.sum(energy)
        if total == 0:
            return 0.0
        probs = energy / total
        probs = probs[probs > 0]  # avoid log(0)
        return float(-np.sum(probs * np.log2(probs)))

    # ---- internal: classification -----------------------------------------

    def _classify(self, f: Dict[str, float]) -> tuple:
        """Rule-based classifier. Returns (tone_str, confidence).

        Scores each tone based on feature membership, then picks the highest.
        """
        scores: Dict[str, float] = {t: 0.0 for t in self.config.tone_types}

        rms = f["rms_energy"]
        rms_var = f["rms_variance"]
        zcr = f["zcr_mean"]
        centroid = f["spectral_centroid_mean"]
        tempo = f["tempo_bpm"]
        entropy = f["energy_entropy"]
        silence = f["silence_ratio"]

        cfg = self.config

        # --- Excited: high energy, high tempo, high spectral centroid, high variance
        if rms > cfg.rms_threshold_high:
            scores["excited"] += 2.0
        if tempo > cfg.tempo_threshold_fast:
            scores["excited"] += 2.0
        if zcr > cfg.zcr_threshold_low:
            scores["excited"] += 1.0
        if centroid > cfg.spectral_threshold:
            scores["excited"] += 1.0
        if rms_var > 0.003:
            scores["excited"] += 1.0
        if entropy > 7.0:
            scores["excited"] += 1.0

        # --- Angry: high energy + high ZCR (harsh), high spectral centroid
        if rms > cfg.rms_threshold_high:
            scores["angry"] += 2.0
        if zcr > cfg.zcr_threshold_high:
            scores["angry"] += 3.0
        if centroid > cfg.spectral_threshold * 1.2:
            scores["angry"] += 2.0
        if rms_var > 0.004:
            scores["angry"] += 1.0
        if silence < 0.1:
            scores["angry"] += 1.0

        # --- Formal: low-mid energy, slow tempo, low ZCR, high silence ratio
        if rms < cfg.rms_threshold_high:
            scores["formal"] += 1.0
        if tempo < cfg.tempo_threshold_slow:
            scores["formal"] += 2.0
        if zcr < cfg.zcr_threshold_low + 0.03:
            scores["formal"] += 2.0
        if rms_var < 0.002:
            scores["formal"] += 1.5
        if silence > 0.2:
            scores["formal"] += 1.5
        if entropy < 6.0:
            scores["formal"] += 1.0

        # --- Casual: moderate energy, moderate tempo, moderate everything
        if cfg.rms_threshold_low < rms < cfg.rms_threshold_high:
            scores["casual"] += 2.0
        if cfg.tempo_threshold_slow < tempo < cfg.tempo_threshold_fast:
            scores["casual"] += 2.0
        if cfg.zcr_threshold_low < zcr < cfg.zcr_threshold_high:
            scores["casual"] += 1.5
        if 0.1 < silence < 0.3:
            scores["casual"] += 1.0

        # --- Neutral: mid-range on everything, low variance
        if rms_var < 0.001:
            scores["neutral"] += 2.0
        if abs(tempo - 130) < 20:
            scores["neutral"] += 1.5
        if entropy < 6.0:
            scores["neutral"] += 1.0

        # Pick winner
        best_tone = max(scores, key=scores.get)  # noqa: dict is not None
        best_score = scores[best_tone]
        total_score = sum(abs(v) for v in scores.values())

        confidence = best_score / total_score if total_score > 0 else 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        logger.debug("Tone scores: %s", scores)
        return best_tone, confidence


# ---------------------------------------------------------------------------
# LLM prompt adapter
# ---------------------------------------------------------------------------

_TONE_PROMPTS: Dict[str, str] = {
    ToneType.NEUTRAL.value: (
        "Format the text in a clear, neut style. Use proper punctuation "
        "and capitalisation. Maintain a balanced, professional tone."
    ),
    ToneType.EXCITED.value: (
        "Keep the enthusiastic energy of the original speech. "
        "Use exclamation marks where appropriate. Maintain a lively, "
        "engaging tone. Do not over-sentimentalise."
    ),
    ToneType.FORMAL.value: (
        "Format the text formally. Use complete sentences, proper grammar, "
        "and professional language. Avoid contractions and colloquialisms. "
        "This should read like a business communication."
    ),
    ToneType.CASUAL.value: (
        "Keep a casual, conversational tone. It is fine to use contractions "
        "and informal phrasing. Add exclamation marks or ellipses where they "
        "match the natural speech rhythm."
    ),
    ToneType.ANGRY.value: (
        "Preserve the urgent/intense tone of the original speech. "
        "Use short, punchy sentences. Do not soften the message. "
        "Add emphasis where the speaker clearly stressed words. "
        "Use caps sparingly only where the speaker raised their voice significantly."
    ),
}


def get_tone_prompt(tone: str) -> str:
    """Return the LLM instruction suffix for a given tone string.

    Args:
        tone: One of the ToneType string values.

    Returns:
        Instruction string to append to the system prompt, or a neutral
        default if the tone is unrecognised.
    """
    prompt = _TONE_PROMPTS.get(tone)
    if prompt is None:
        logger.warning("Unrecognised tone %r; falling back to neutral", tone)
        prompt = _TONE_PROMPTS[ToneType.NEUTRAL.value]
    return prompt


# ---------------------------------------------------------------------------
# Tone-aware processor (wraps LLMPostProcessor)
# ---------------------------------------------------------------------------

class ToneAwareProcessor:
    """Wraps the transcription + LLM pipeline and adjusts the system prompt
    based on detected tone.

    Designed to work with ``voiceflow.llm_postprocessor.LLMPostProcessor``.
    If the LLM post-processor is available, its ``config.system_prompt`` is
    temporarily overridden with a tone-aware version for each call, then
    restored so that the original config is never mutated.

    Usage:
        llm = LLMPostProcessor(LLMConfig(...))
        processor = ToneAwareProcessor(llm)
        result = processor.process(raw_text, audio_array)
    """

    def __init__(
        self,
        llm_processor: Any = None,
        tone_config: Optional[ToneConfig] = None,
    ):
        """Args:
            llm_processor:  An ``LLMPostProcessor`` instance (or any object
                            with ``.process(text)`` and ``.config.system_prompt``).
                            If None, only the raw transcription is returned.
            tone_config:    Optional ``ToneConfig`` for the tone analyser.
        """
        self._analyzer = ToneAnalyzer(tone_config)
        self._llm = llm_processor
        logger.info(
            "ToneAwareProcessor initialised (llm=%s)",
            "enabled" if self._llm else "disabled",
        )

    # ---- public API -------------------------------------------------------

    def analyze_tone(self, audio_array: np.ndarray) -> ToneResult:
        """Analyse the emotional tone of an audio segment.

        Args:
            audio_array: Raw audio samples as a 1-D numpy float array.

        Returns:
            ToneResult with classification, confidence, and feature map.
        """
        return self._analyzer.analyze(audio_array)

    def process(self, raw_transcription: str, audio_array: np.ndarray) -> str:
        """Process a transcription with tone-aware LLM cleanup.

        If no LLM processor is configured, returns the raw transcription.

        Args:
            raw_transcription: Text from the speech-to-text engine.
            audio_array:       Corresponding audio samples for tone detection.

        Returns:
            Tone-adjusted cleaned-up text (or raw text if no LLM).
        """
        if not self._llm or not raw_transcription:
            return raw_transcription

        tone_result = self.analyze_tone(audio_array)
        tone_prompt = get_tone_prompt(tone_result.tone)

        # Build a tone-aware system prompt without mutating the user's config
        original_prompt = self._llm.config.system_prompt
        self._llm.config.system_prompt = (
            f"{original_prompt}\n\n"
            f"IMPORTANT: {tone_prompt}\n\n"
            f"Detected tone: {tone_result.tone} "
            f"(confidence: {tone_result.confidence:.0%})"
        )

        try:
            cleaned = self._llm.process(raw_transcription)
            logger.info(
                "Tone-aware cleanup done (tone=%s, confidence=%.2f)",
                tone_result.tone,
                tone_result.confidence,
            )
            return cleaned
        except Exception:
            logger.exception("Tone-aware LLM cleanup failed; returning raw text")
            return raw_transcription
        finally:
            # Always restore the original prompt
            self._llm.config.system_prompt = original_prompt

    # ---- convenience helpers ----------------------------------------------

    def get_tone_prompt(self, tone: str) -> str:
        """Return the LLM instruction text for a tone.

        Useful for debugging or UI display.
        """
        return get_tone_prompt(tone)
