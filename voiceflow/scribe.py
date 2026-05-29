"""
VoiceFlow Scribe Mode -- Continuous listening for meetings/interviews.

Records audio continuously, detects speech segments via VAD, guesses speaker
changes using simple audio-feature heuristics, transcribes each segment with
the project's Transcriber, and emits timestamped, speaker-tagged lines.

Exports: to_text(), to_markdown(), to_srt()

Usage
-----
    from voiceflow import ScribeMode, ScribeConfig

    config = ScribeConfig(min_segment_seconds=1.5, max_silence_seconds=2.0)
    with ScribeMode(config) as scribe:
        scribe.start()          # blocks until stop()
        print(scribe.to_markdown())

Or as a standalone listener that processes in real-time::

    scribe = ScribeMode(config)
    scribe.start()  # runs in background; register a callback
    scribe.stop()

Dependencies
------------
Core: numpy, sounddevice, (optional) faster-whisper via voiceflow.transcriber.
No heavy ML deps required -- speaker diarization uses RMS + zero-crossing heuristics.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import struct
import threading
import time
import wave
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScribeConfig:
    """Configuration for ScribeMode.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (Hz). Default 16000 (Whisper-compatible).
    channels : int
        Number of audio channels. Default 1 (mono).
    chunk_duration : float
        Duration of each audio chunk processed by the listener (seconds).
        Smaller values = lower latency but more CPU. Default 0.5.
    vad_rms_threshold : float
        RMS energy above which a chunk is considered speech. Default 0.015.
    min_segment_seconds : float
        Minimum duration for a speech segment to be transcribed. Default 1.5.
    max_silence_seconds : float
        Seconds of silence after which the current speech segment is emitted.
        Default 2.0.
    max_speakers : int
        Maximum number of distinct speakers to track. Default 4.
    speaker_change_threshold : float
        Threshold (0-1) for the speaker-feature distance above which a new
        speaker is guessed. Lower = more sensitive. Default 0.35.
    hotwords : str
        Comma-separated hotwords passed to the Transcriber.
    transcribe : bool
        Whether to transcribe segments automatically. Set False to only
        produce diarized time segments.
    on_segment :
        Optional callback ``(speaker_id: int, start: float, end: float, text: str) -> None``
        invoked each time a segment is finalized and (optionally) transcribed.
    """

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.5
    vad_rms_threshold: float = 0.015
    min_segment_seconds: float = 1.5
    max_silence_seconds: float = 2.0
    max_speakers: int = 4
    speaker_change_threshold: float = 0.35
    hotwords: str = ""
    transcribe: bool = True
    on_segment: Optional[Callable[[int, float, float, str], None]] = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TranscriptEntry:
    """A single transcribed entry."""
    speaker_id: int
    start_time: float   # seconds from session start
    end_time: float
    text: str = ""


@dataclasses.dataclass
class _SpeechSegment:
    """Accumulation buffer for a speech segment-in-progress."""
    start_time: float
    audio_chunks: List[np.ndarray] = dataclasses.field(default_factory=list)
    features: List[np.ndarray] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Lightweight audio features for speaker diarization
# ---------------------------------------------------------------------------

def _compute_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute a compact feature vector from an audio chunk.

    Returns an 8-dim vector:
        [rms_energy, zero_crossing_rate, spectral_centroid,
         spectral_bandwidth, energy_entropy, f0_estimate,
         energy_std, amplitude_max]

    This is deliberately simple -- no heavy ML, just numpy signal processing.
    """
    # Ensure 1-D float32
    audio = audio.flatten().astype(np.float32)

    if len(audio) < 40:
        return np.zeros(8, dtype=np.float32)

    # 1. RMS energy
    rms = np.sqrt(np.mean(audio ** 2))

    # 2. Zero-crossing rate
    zcr = np.mean(np.diff(np.signbit(audio).astype(int)) != 0)

    # FFT-based features
    n_fft = min(2048, len(audio))
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    total_energy = spectrum.sum()
    if total_energy < 1e-12:
        return np.array([rms, zcr, 0, 0, 0, 0, 0, np.max(np.abs(audio))], dtype=np.float32)

    # 3. Spectral centroid (weighted mean frequency)
    centroid = np.sum(freqs * spectrum) / total_energy

    # 4. Spectal bandwidth (weighted std of frequency)
    bandwidth = np.sqrt(np.sum(spectrum * (freqs - centroid) ** 2) / total_energy)

    # 5. energy entropy (across FFT bins)
    p = spectrum / total_energy
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))

    # 6. F0 estimate via autocorrelation (simple pitch detection)
    f0 = _estimate_f0(audio, sample_rate)

    # 7. energy std (short-term variability)
    frame_len = max(len(audio) // 10, 40)
    n_frames = len(audio) // frame_len
    if n_frames > 1:
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        energy_std = float(np.std(frame_rms))
    else:
        energy_std = 0.0

    # 8. amplitude max
    amp_max = float(np.max(np.abs(audio)))

    return np.array(
        [rms, zcr, centroid / 1000.0, bandwidth / 1000.0,
         entropy, f0, energy_std, amp_max],
        dtype=np.float32,
    )


def _estimate_f0(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate fundamental frequency via autocorrelation.

    Normalized to 0-1 range (will be ~0.5 for a typical male voice at 150 Hz,
    ~0.9 for a typical female voice at 250 Hz with 16kHz sample rate).
    """
    # Clamp pitch range to 50-500 Hz
    min_lag = max(int(sample_rate / 500), 1)
    max_lag = min(int(sample_rate / 50), len(audio) // 2)

    if max_lag <= min_lag:
        return 0.0

    audio = audio - np.mean(audio)
    norm = np.dot(audio, audio)
    if norm < 1e-12:
        return 0.0

    # Autocorrelation via numpy correlate
    corr = np.correlate(audio, audio, mode="full")
    corr = corr[len(corr) // 2 :]  # take positive lags

    if len(corr) <= max_lag:
        return 0.0

    # Search for peak in the valid lag range
    search_region = corr[min_lag:max_lag + 1]
    if len(search_region) == 0:
        return 0.0

    peak_idx = np.argmax(search_region)
    lag = min_lag + peak_idx

    f0 = sample_rate / lag

    # Normalize to 0-1 assuming range 50-500 Hz
    normalized = (f0 - 50.0) / 450.0
    return float(np.clip(normalized, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Speaker tracking -- simple running centroid approach
# ---------------------------------------------------------------------------

class _SpeakerTracker:
    """Track speaker identities using running feature centroids.

    Each known speaker has a centroid (mean feature vector).  A new feature
    vector is assigned to the closest speaker within *threshold*.  If no
    speaker is close enough and we haven't reached max_speakers, a new
    speaker is created.
    """

    def __init__(self, max_speakers: int = 4, threshold: float = 0.35):
        self.max_speakers = max_speakers
        self.threshold = threshold
        # speaker_id -> list of feature vectors
        self._centroids: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._next_id = 1

    def assign(self, feature_vec: np.ndarray) -> int:
        """Return the speaker_id closest to *feature_vec*, or create a new one."""
        if not self._centroids:
            # First speaker ever
            sid = self._next_id
            self._next_id += 1
            self._centroids[sid].append(feature_vec.copy())
            return sid

        best_id = -1
        best_dist = float("inf")

        for sid, history in self._centroids.items():
            centroid = np.mean(history, axis=0)
            dist = np.linalg.norm(feature_vec - centroid)
            if dist < best_dist:
                best_dist = dist
                best_id = sid

        # Check threshold or room for new speaker
        if best_dist <= self.threshold:
            self._centroids[best_id].append(feature_vec.copy())
            return best_id

        if len(self._centroids) < self.max_speakers:
            sid = self._next_id
            self._next_id += 1
            self._centroids[sid].append(feature_vec.copy())
            return sid

        # Room full: force assign to closest
        self._centroids[best_id].append(feature_vec.copy())
        return best_id

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def speaker_durations(self, entries: List[TranscriptEntry]) -> Dict[int, float]:
        """Return total speaking time per speaker id."""
        d: Dict[int, float] = defaultdict(float)
        for e in entries:
            d[e.speaker_id] += e.end_time - e.start_time
        return dict(d)


# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------

class _VAD:
    """Simple energy-based VAD with hysteresis.

    State machine:
        SILENT -> (rms > threshold) -> SPEAKING
        SPEAKING -> (rms <= threshold) -> TRAILING
        TRAILING -> (silence_duration > max_silence) -> SILENT
    """

    SILENT = 0
    SPEAKING = 1
    TRAILING = 2

    def __init__(self, rms_threshold: float, max_silence_seconds: float,
                 chunk_duration: float):
        self.rms_threshold = rms_threshold
        self.max_silence_seconds = max_silence_seconds
        self.chunk_duration = chunk_duration
        self.state = self.SILENT
        self._trailing_accum = 0.0  # seconds of silence while TRAILING

    def feed(self, chunk: np.ndarray) -> int:
        """Process an audio chunk. Returns new state."""
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if self.state == self.SILENT:
            if rms > self.rms_threshold:
                self.state = self.SPEAKING
                logger.debug("VAD: SILENT -> SPEAKING (rms=%.4f)", rms)
            return self.state

        elif self.state == self.SPEAKING:
            if rms <= self.rms_threshold:
                self.state = self.TRAILING
                self._trailing_accum = 0.0
                logger.debug("VAD: SPEAKING -> TRAILING (rms=%.4f)", rms)
            return self.state

        elif self.state == self.TRAILING:
            if rms > self.rms_threshold:
                self.state = self.SPEAKING
                self._trailing_accum = 0.0
                logger.debug("VAD: TRAILING -> SPEAKING (rms=%.4f)", rms)
            else:
                self._trailing_accum += self.chunk_duration
                if self._trailing_accum >= self.max_silence_seconds:
                    self.state = self.SILENT
                    logger.debug("VAD: TRAILING -> SILENT (gap=%.1fs)",
                                 self._trailing_accum)
            return self.state

        return self.state

    def reset(self):
        self.state = self.SILENT
        self._trailing_accum = 0.0


# ---------------------------------------------------------------------------
# Continuous Audio Listener
# ---------------------------------------------------------------------------

class ContinuousAudioListener:
    """Records audio chunks in a background thread and delivers them to a callback.

    This is the low-level audio capture used by ScribeMode.  For simple use
    cases it can record chunks into a queue for the calling thread to consume.

    Usage
    -----
        listener = ContinuousAudioListener()
        listener.start()
        # ... in main thread ...
        chunk = listener.get_queue().get(timeout=1.0)
        # ...
        listener.stop()
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._queue: "threading.Queue[Optional[np.ndarray]]" = threading.Queue(maxsize=200)
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def start(self):
        """Start recording in the background thread."""
        with self._lock:
            if self._running:
                logger.warning("ContinuousAudioListener already running")
                return
            self._running = True

        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("ContinuousAudioListener started (sr=%d, chunk=%.1fs)",
                     self.sample_rate, self.chunk_duration)

    def stop(self):
        """Stop recording and join the background thread."""
        with self._lock:
            self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("ContinuousAudioListener thread did not stop in time")
            self._thread = None

        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

        logger.info("ContinuousAudioListener stopped")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_queue(self) -> "threading.Queue[Optional[np.ndarray]]":
        """Return the chunk queue for the consumer thread."""
        return self._queue

    def wait(self, timeout: float = None):
        """Block the calling thread until the listener stops."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # -- internal -----------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.debug("Audio stream status: %s", status)
        try:
            self._queue.put_nowait(indata.copy().astype(np.float32))
        except Exception:
            pass  # queue full, drop chunk

    def _record_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice is required for audio capture. "
                         "Install with: pip install sounddevice")
            with self._lock:
                self._running = False
            return

        chunk_size = int(self.sample_rate * self.chunk_duration)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()

            with self._lock:
                running = self._running
            while running:
                time.sleep(0.1)
                with self._lock:
                    running = self._running

        except Exception as e:
            logger.error("ContinuousAudioListener error: %s", e)
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    # -- context manager ----------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ---------------------------------------------------------------------------
# ScribeMode -- the main class
# ---------------------------------------------------------------------------

class ScribeMode:
    """
    Continuous listening for meetings / interviews with speaker diarization.

    Parameters
    ----------
    config : ScribeConfig
        Configuration for VAD, diarization, transcription, etc.

    Usage as context manager (recommended)::

        with ScribeMode() as scribe:
            scribe.start()          # blocks until stop()
            # ... in another thread or after stop ...
            print(scribe.to_text())

    Usage with manual start/stop::

        scribe = ScribeMode()
        scribe.start(block=False)   # non-blocking
        # ... do other work ...
        scribe.stop()

    The ``on_segment`` callback in ScribeConfig is called in real-time each time
    a speech segment is finalized and transcribed::

        def on_segment(speaker_id, start, end, text):
            print(f"[{format_time(start)}] Speaker {speaker_id}: {text}")

        config = ScribeConfig(on_segment=on_segment)
        scribe = ScribeMode(config)
    """

    def __init__(self, config: Optional[ScribeConfig] = None):
        self.config = config or ScribeConfig()
        self._vad = _VAD(
            rms_threshold=self.config.vad_rms_threshold,
            max_silence_seconds=self.config.max_silence_seconds,
            chunk_duration=self.config.chunk_duration,
        )
        self._speaker_tracker = _SpeakerTracker(
            max_speakers=self.config.max_speakers,
            threshold=self.config.speaker_change_threshold,
        )
        self._listener: Optional[ContinuousAudioListener] = None
        self._entries: List[TranscriptEntry] = []
        self._session_start: float = 0.0
        self._running = False
        self._stop_event = threading.Event()
        self._main_thread: Optional[threading.Thread] = None
        self._current_segment: Optional[_SpeechSegment] = None
        self._total_chunks_processed: int = 0
        self._total_speech_chunks: int = 0

    # -- public controls ----------------------------------------------------

    def start(self, block: bool = True):
        """Start the scribe session.

        If *block* is True, this method blocks until :meth:`stop` is called
        (from any thread) or the context manager exits.  If *block* is False,
        :meth:`stop` must be called later to end the session.
        """
        if self._running:
            logger.warning("ScribeMode already running, ignoring start()")
            return

        self._reset_state()
        self._listener = ContinuousAudioListener(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_duration=self.config.chunk_duration,
        )
        self._listener.start()
        self._session_start = time.monotonic()
        self._running = True
        self._stop_event.clear()
        logger.info("ScribeMode session started")

        self._main_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._main_thread.start()

        if block:
            self._main_thread.join()

    def stop(self):
        """Stop the scribe session.

        Finalizes any in-progress speech segment and stops the audio listener.
        Safe to call from any thread.
        """
        if not self._running:
            return

        logger.info("ScribeMode stopping...")
        self._running = False
        self._stop_event.set()

        # Finalize any in-progress segment
        self._finalize_current_segment()

        if self._listener is not None:
            self._listener.stop()

        if self._main_thread is not None and self._main_thread.is_alive():
            self._main_thread.join(timeout=10.0)

        logger.info("ScribeMode session ended (%d entries)", len(self._entries))

    # -- accessors ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the session is active."""
        return self._running

    @property
    def entries(self) -> List[TranscriptEntry]:
        """All transcript entries collected so far (live view)."""
        return list(self._entries)

    @property
    def duration(self) -> float:
        """Total session wall-clock duration in seconds."""
        if self._session_start == 0:
            return 0.0
        if self._running:
            return time.monotonic() - self._session_start
        # After stop, we don't store end time; estimate from last entry
        if self._entries:
            return self._entries[-1].end_time
        return 0.0

    @property
    def stats(self) -> Dict:
        """Return session statistics as a dict."""
        entries = self._entries
        speaker_durations = self._speaker_tracker.speaker_durations(entries)
        total_speech_time = sum(speaker_durations.values())
        return {
            "session_duration_seconds": round(self.duration, 2),
            "total_entries": len(entries),
            "speakers_detected": self._speaker_tracker.speaker_count,
            "total_speech_seconds": round(total_speech_time, 2),
            "speaker_durations": {
                f"Speaker {sid}": round(dur, 2)
                for sid, dur in sorted(speaker_durations.items())
            },
            "chunks_processed": self._total_chunks_processed,
            "speech_chunks": self._total_speech_chunks,
        }

    # -- export methods -----------------------------------------------------

    def to_text(self) -> str:
        """Return the transcript as plain text with timestamps."""
        lines: List[str] = []
        for e in self._entries:
            ts = _format_timestamp(e.start_time)
            lines.append(f"[{ts}] Speaker {e.speaker_id}: {e.text}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return the transcript as a Markdown document."""
        lines: List[str] = []
        lines.append("# VoiceFlow Scribe Session\n")

        stats = self.stats
        lines.append(f"- **Duration**: {_format_timestamp(stats['session_duration_seconds'])}")
        lines.append(f"- **Speakers detected**: {stats['speakers_detected']}")
        lines.append(f"- **Entries**: {stats['total_entries']}\n")

        lines.append("| Timestamp | Speaker | Text |")
        lines.append("|---|---|---|")
        for e in self._entries:
            ts = _format_timestamp(e.start_time)
            text = e.text.replace("|", "\\|")
            lines.append(f"| {ts} | Speaker {e.speaker_id} | {text} |")

        return "\n".join(lines)

    def to_srt(self) -> str:
        """Return the transcript in SRT subtitle format.

        Format:
            1
            00:00:05,000 --> 00:00:08,500
            Speaker 1: Hello everyone
        """
        lines: List[str] = []
        for idx, e in enumerate(self._entries, start=1):
            start_str = _format_srt_time(e.start_time)
            end_str = _format_srt_time(e.end_time)
            lines.append(str(idx))
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"Speaker {e.speaker_id}: {e.text}")
            lines.append("")  # blank separator
        return "\n".join(lines)

    def save_wav(self, filepath: str):
        """Save the full session audio as a WAV file, if it was recorded."""
        # This is a convenience; ScribeMode doesn't store all audio by default
        # to save memory.  Users needing the full recording should use
        # ContinuousAudioListener directly with their own accumulation.
        logger.info("save_wav: session audio not stored in memory. "
                     "Use ContinuousAudioListener for full audio capture.")

    # -- private ------------------------------------------------------------

    def _reset_state(self):
        self._entries = []
        self._vad = _VAD(
            rms_threshold=self.config.vad_rms_threshold,
            max_silence_seconds=self.config.max_silence_seconds,
            chunk_duration=self.config.chunk_duration,
        )
        self._speaker_tracker = _SpeakerTracker(
            max_speakers=self.config.max_speakers,
            threshold=self.config.speaker_change_threshold,
        )
        self._current_segment = None
        self._total_chunks_processed = 0
        self._total_speech_chunks = 0

    def _process_loop(self):
        """Main loop: reads from audio queue, runs VAD, accumulates segments."""
        assert self._listener is not None
        queue = self._listener.get_queue()

        while self._running and not self._stop_event.is_set():
            try:
                chunk = queue.get(timeout=0.2)
            except Exception:
                continue

            if chunk is None:
                continue

            self._total_chunks_processed += 1
            self._process_chunk(chunk)

    def _process_chunk(self, chunk: np.ndarray):
        """Process a single audio chunk through VAD and segment tracking."""
        now = time.monotonic() - self._session_start
        vad_state = self._vad.feed(chunk)

        if vad_state in (_VAD.SPEAKING, _VAD.TRAILING):
            self._total_speech_chunks += 1

            if self._current_segment is None:
                self._current_segment = _SpeechSegment(start_time=now)

            self._current_segment.audio_chunks.append(chunk)
            feat = _compute_features(chunk, self.config.sample_rate)
            self._current_segment.features.append(feat)

        elif vad_state == _VAD.SILENT:
            # Speech just ended -- finalize if we have a segment
            self._finalize_current_segment()

            # Try to assign speaker for this segment if it's long enough
            pass

        # Check if current segment is long enough and we're back in trailing
        # Actually: finalize only on SILENT transition (handled above)

    def _finalize_current_segment(self):
        """Finalize the current segment: assign speaker, transcribe, emit entry."""
        seg = self._current_segment
        if seg is None:
            return

        self._current_segment = None

        # Check minimum duration
        now = time.monotonic() - self._session_start
        duration = now - seg.start_time
        if duration < self.config.min_segment_seconds:
            logger.debug("Dropping short segment (%.2fs < %.2fs)",
                         duration, self.config.min_segment_seconds)
            return

        # Concatenate audio
        if not seg.audio_chunks:
            return
        audio = np.concatenate(seg.audio_chunks)

        # Compute mean feature vector for speaker assignment
        if seg.features:
            mean_feature = np.mean(seg.features, axis=0)
            speaker_id = self._speaker_tracker.assign(mean_feature)
        else:
            speaker_id = 1

        # Transcribe
        text = ""
        if self.config.transcribe:
            text = self._transcribe_audio(audio)

        # Clamp end_time to last chunk boundary
        end_time = now
        if len(seg.audio_chunks) > 0:
            # More precise: sum chunk durations
            end_time = seg.start_time + len(audio) / self.config.sample_rate

        entry = TranscriptEntry(
            speaker_id=speaker_id,
            start_time=seg.start_time,
            end_time=min(end_time, now),
            text=text,
        )
        self._entries.append(entry)

        # Real-time output
        ts = _format_timestamp(entry.start_time)
        display = f"[{ts}] Speaker {speaker_id}: {text}"
        logger.info(display)
        print(display)  # real-time terminal output

        # Callback
        if self.config.on_segment is not None:
            try:
                self.config.on_segment(speaker_id, entry.start_time,
                                       entry.end_time, text)
            except Exception as e:
                logger.error("on_segment callback error: %s", e)

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe an audio buffer using the project's Transcriber.

        Lazily imports voiceflow.transcriber to avoid forcing a Whisper load
        at import time.
        """
        try:
            from voiceflow.transcriber import Transcriber, TranscriptionConfig

            config = TranscriptionConfig(
                model_size="base",
                device="auto",
                compute_type="int8",
                language=None,
                beam_size=5,
                vad_filter=True,
                hotwords=self.config.hotwords,
            )
            transcriber = Transcriber(config)
            return transcriber.transcribe(audio)
        except ImportError:
            logger.warning("voiceflow.transcriber not available; skipping transcription")
            return ""
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return ""

    def wait(self, timeout: Optional[float] = None):
        """Block until the session ends."""
        if self._main_thread is not None:
            self._main_thread.join(timeout=timeout)

    # -- context manager ----------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _format_srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT format)."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience / CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VoiceFlow Scribe Mode")
    parser.add_argument("--min-segment", type=float, default=1.5,
                        help="Minimum speech segment duration (seconds)")
    parser.add_argument("--max-silence", type=float, default=2.0,
                        help="Max silence before finalizing a segment (seconds)")
    parser.add_argument("--speakers", type=int, default=4,
                        help="Maximum speakers to detect")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Speaker change sensitivity (0.1-1.0)")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Only diarize, don't transcribe")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (.txt, .md, .srt)")
    parser.add_argument("--format", type=str, default="text",
                        choices=["text", "markdown", "srt"],
                        help="Output format")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ScribeConfig(
        min_segment_seconds=args.min_segment,
        max_silence_seconds=args.max_silence,
        max_speakers=args.speakers,
        speaker_change_threshold=args.threshold,
        transcribe=not args.no_transcribe,
    )

    print("\n=== VoiceFlow Scribe Mode ===")
    print("Press Ctrl+C to stop and show transcript.\n")

    scribe = ScribeMode(config)
    try:
        with scribe:
            scribe.start(block=True)
    except KeyboardInterrupt:
        print("\n\n-- Stopping (Ctrl+C) --\n")
        scribe.stop()

    # Show transcript
    print("\n=== TRANSCRIPT ===\n")
    if args.format == "markdown":
        output = scribe.to_markdown()
    elif args.format == "srt":
        output = scribe.to_srt()
    else:
        output = scribe.to_text()
    print(output)

    # Show stats
    print("\n=== SESSION STATS ===")
    for k, v in scribe.stats.items():
        print(f"  {k}: {v}")

    # Save to file
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nSaved to {args.output}")
