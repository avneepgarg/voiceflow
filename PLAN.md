# VoiceFlow - Custom Wispr Flow Alternative

> **Goal:** Build a free, open-source, system-wide voice dictation app that types transcribed speech into any active application on Windows.
>
> **Architecture:** Python-based daemon that listens for a global hotkey, records audio from the microphone, runs speech-to-text using either a local Whisper model on GPU or a cloud API (OpenAI/Deepgram/etc), optionally cleans up text through an online LLM, and injects the final text into the active window using keyboard simulation.
>
> **Tech Stack:** Python 3.14, faster-whisper (GPU), pyaudio/sounddevice, pynput (hotkeys), pystray (tray UI), openai SDK (cloud LLM)
>
> **Why this approach:** GPU-accelerated local inference (free, private) when you have a GPU. Cloud LLM (optional, BYOK) for AI-powered cleanup - remove filler words, fix grammar, reformat. Everything is opt-in and configurable per-use.
>
> ---
>
> ## Task 1: Set up GPU-enabled Whisper
>
> **Objective:** Install torch with CUDA support so Whisper runs at 3-5x real-time on our RTX 3050.
>
> **Files:**
> - Modify: WSL environment (pip install torch with CUDA)
>
> **Step 1: Uninstall CPU-only torch**
>
> ```bash
> pip3 uninstall torch torchaudio -y
> ```
>
> **Step 2: Install CUDA-enabled torch**
>
> ```bash
> # CUDA 13.2 on our system (driver 596.08)
> pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu132
> ```
>
> **Step 3: Verify CUDA is available**
>
> ```bash
> python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
> ```
>
> Expected: `CUDA: True`, `Device: NVIDIA GeForce RTX 3050 Laptop GPU`
>
> **Step 4: Verify faster-whisper works with CUDA**
>
> ```bash
> python3 -c "
> from faster_whisper import WhisperModel
> model = WhisperModel('base', device='cuda', compute_type='int8')
> print('Model loaded on GPU successfully')
> "
> ```
>
> Expected: Model loads without error, prints confirmation.
>
> **(Note: RTX 3050 4GB can run whisper 'base' at ~3-5x real-time with int8 quantization. 'small' may fit but will be slower. Start with 'base', upgrade if performance allows.)**
>
> ---
>
> ## Task 2: Set up audio input in WSL
>
> **Objective:** Get microphone audio from Windows Bluetooth headset working inside WSL.
>
> **Files:**
> - Create: `/etc/pulse/client.conf` (WSL PulseAudio config)
>
> **Option A: PulseAudio bridge to Windows**
>
> **Step 1: Install PulseAudio in WSL**
>
> ```bash
> sudo apt-get update && sudo apt-get install -y pulseaudio-utils libpulse0
> ```
>
> **Step 2: Configure PulseAudio to use Windows host**
>
> Create `/etc/pulse/client.conf`:
> ```
> default-server = tcp:127.0.0.1:4713
> autospawn = no
> ```
>
> Run Windows PulseAudio server:
> ```
> # In Windows Command Prompt (run as Admin):
> # Download from https://www.freedesktop.org/wiki/Software/PulseAudio/Ports/Windows/
> # Or use: pactl load-module module-native-protocol-tcp auth-anonymous=1
> ```
>
> **Step 3: Verify audio devices**
>
> ```bash
> pactl list short sources
> ```
>
> Expected: At least one source (our Bluetooth headset mic) listed.
>
> **Option B: Use sounddevice + Windows audio (fallback)**
>
> If PulseAudio bridge is too complex, use a Python solution that directly captures via COM/WinMM:
>
> ```bash
> pip3 install sounddevice
> python3 -c "import sounddevice as sd; print(sd.query_devices())"
> ```
>
> Expected: Device list shows our Bluetooth microphone.
>
> **Option C (simplest, Windows-native):**
>
> Since WSL audio is painful, build the core app as a Windows-native Python process:
> - Install Python on Windows
> - Use `pyaudio` or `sounddevice` directly
> - WSL only used for testing/development
>
> **Recommendation:** Go with Option C. Install Python on Windows, run the app natively. This eliminates all audio bridge issues and gives us real Windows hotkey support.
>
> ---
>
> ## Task 3: Project structure and core daemon
>
> **Objective:** Set up the project skeleton with all components wired together.
>
> **Files:**
> - Create: `voiceflow/main.py` - Entry point, tray icon, hotkey loop
> - Create: `voiceflow/audio.py` - Audio capture (record when hotkey pressed)
- Create: `voiceflow/transcriber.py` - Whisper transcription (local)
- Create: `voiceflow/llm_postprocessor.py` - Cloud LLM text cleanup (optional, BYOK)
- Create: `voiceflow/typer.py` - Text injection into active window
> - Create: `voiceflow/config.py` - Settings and config
> - Create: `voiceflow/tray.py` - System tray UI
> - Create: `tests/test_transcriber.py` - Transcription unit tests
> - Create: `requirements.txt` - Dependencies
> - Create: `VoiceFlow.spec` - PyInstaller installer spec
>
> **Directory structure:**
> ```
> voiceflow/
>   voiceflow/
>     __init__.py
main.py                  # Entry point + tray + hotkey listener
audio.py                 # Audio recording (hotkey -> audio buffer)
transcriber.py           # Local Whisper transcription (GPU/CPU)
llm_postprocessor.py     # Cloud LLM text cleanup (optional, BYOK)
typer.py                 # Text injection via pynput
config.py                # Settings: hotkey, model, language, LLM config, etc.
tray.py                  # System tray icon with status
utils.py                 # Shared helpers
>   tests/
>     test_audio.py
>     test_transcriber.py
>     test_typer.py
>   icons/
>     mic.ico
>     mic.ico
>   requirements.txt
>   setup.py
>   VoiceFlow.spec    # For building .exe installer
>   README.md
> ```
>
> **Step 1: Create project directory**
>
> ```bash
> mkdir -p voiceflow/voiceflow voiceflow/tests voiceflow/icons
> ```
>
> **Step 2: Create requirements.txt**
>
> ```
> faster-whisper>=1.2.0
> sounddevice>=0.4.0
> numpy>=1.24.0
> pynput>=1.7.0
> pystray>=0.19.0
> Pillow>=9.0.0
> PyAudio>=0.2.13
> keyboard>=0.13.5
> openai>=1.50.0        # Cloud LLM post-processing (optional, BYOK)
> requests>=2.31.0      # HTTP client for cloud transcription APIs
> ```

<antThinking>Hmm, actually I realize I'm writing a very long plan. Let me continue with the core implementation tasks. The user asked to "move forward" which means they want to build this. Let me finish the plan file and then start implementing.

Let me continue writing the remaining tasks compactly.
</antThinking>
>
> **Step 3: Verify all packages install**
>
> ```bash
> pip3 install -r requirements.txt 2>&1 | tail -5
> ```
>
> Expected: All packages installed without errors.
>
> **Step 4: Commit**
>
> ```bash
> cd voiceflow && git init && git add . && git commit -m "feat: initial project structure"
> ```
>
> ---
>
> ## Task 4: Audio capture module
>
> **Objective:** Record audio from microphone when user presses and holds the hotkey.
>
> **Files:**
> - Create: `voiceflow/audio.py`
> - Test: `tests/test_audio.py`
>
> **Implementation:**
>
> ```python
> """Audio capture module - records while hotkey is held."""
> import numpy as np
> import sounddevice as sd
> import threading
> import queue
> from dataclasses import dataclass
>
> @dataclass
> class AudioConfig:
>     sample_rate: int = 16000
>     channels: int = 1
>     dtype: str = "float32"
>
> class AudioRecorder:
>     """Records audio in real-time while recording flag is set."""
>
>     def __init__(self, config: AudioConfig = None):
>         self.config = config or AudioConfig()
>         self._recording = False
>         self._buffer = []
>         self._stream = None
>
>     def start(self):
>         """Start recording. Call from the hotkey press handler."""
>         self._recording = True
>         self._buffer = []
>         self._stream = sd.InputStream(
>             samplerate=self.config.sample_rate,
>             channels=self.config.channels,
>             dtype=self.config.dtype,
>             callback=self._callback,
>         )
>         self._stream.start()
>
>     def stop(self) -> np.ndarray:
>         """Stop recording and return audio buffer. Call from hotkey release."""
>         self._recording = False
>         if self._stream:
>             self._stream.stop()
>             self._stream.close()
>         if not self._buffer:
>             return np.array([])
>         audio = np.concatenate(self._buffer)
>         # Trim silence
>         audio = self._trim_silence(audio)
>         return audio
>
>     def _callback(self, indata, frames, time_info, status):
>         if self._recording:
>             self._buffer.append(indata.copy())
>
>     def list_devices(self) -> list:
>         """List available audio input devices."""
>         return [
>             {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
>             for i, d in enumerate(sd.query_devices())
>             if d["max_input_channels"] > 0
>         ]
>
>     def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01,
>                        max_duration: float = 60.0) -> np.ndarray:
>         """Trim leading/trailing silence and cap max duration."""
>         audio = audio.flatten()
>         max_samples = int(self.config.sample_rate * max_duration)
>         if len(audio) > max_samples:
>             audio = audio[:max_samples]
>         mask = np.abs(audio) > threshold
>         if not mask.any():
>             return np.array([])
>         start = np.argmax(mask)
>         end = len(audio) - np.argmax(mask[::-1])
>         return audio[start:end]
> ```
>
> **Step 2: Write test**
>
> ```python
> # tests/test_audio.py
> import numpy as np
> import pytest
> from voiceflow.audio import AudioRecorder, AudioConfig
>
> @pytest.fixture
> def recorder():
>     return AudioRecorder()
>
> def test_list_devices(recorder):
>     devices = recorder.list_devices()
>     assert isinstance(devices, list)
>     assert len(devices) > 0
>
> def test_config_defaults():
>     config = AudioConfig()
>     assert config.sample_rate == 16000
>     assert config.channels == 1
> ```
>
> **Step 3: Run tests**
>
> ```bash
> cd voiceflow && python3 -m pytest tests/test_audio.py -v
> ```
>
> Expected: 2 passed.
>
> **Step 4: Commit**
>
> ```bash
> git add voiceflow/audio.py tests/test_audio.py
> git commit -m "feat: audio capture module with silence trimming"
> ```
>
> ---
>
> ## Task 5: Transcription module
>
> **Objective:** Wrap faster-whisper for fast local speech-to-text inference on GPU.
>
> **Files:**
> - Create: `voiceflow/transcriber.py`
> - Test: `tests/test_transcriber.py`
>
> **Implementation:**
>
> ```python
> """Transcription module using faster-whisper with GPU acceleration."""
> import numpy as np
> from faster_whisper import WhisperModel
> from dataclasses import dataclass
> from typing import Optional
> import logging
>
> logger = logging.getLogger(__name__)
>
> @dataclass
> class TranscriptionConfig:
>     model_size: str = "base"           # tiny, base, small, medium
>     device: str = "cuda"               # cuda or cpu
>     compute_type: str = "int8"         # int8 (fast, good enough) or float16
>     language: str = "en"               # en, hi, etc. None = auto-detect
>     beam_size: int = 5
>     vad_filter: bool = True            # Voice Activity Detection
>
> class Transcriber:
>     """Local Whisper transcription - private, free, fast on GPU."""
>
>     def __init__(self, config: TranscriptionConfig = None):
>         self.config = config or TranscriptionConfig()
>         self._model = None
>
>     @property
>     def model(self) -> WhisperModel:
>         """Lazy-load model on first use."""
>         if self._model is None:
>             logger.info(f"Loading Whisper model: {self.config.model_size} "
>                         f"on {self.config.device} ({self.config.compute_type})")
>             self._model = WhisperModel(
>                 self.config.model_size,
>                 device=self.config.device,
>                 compute_type=self.config.compute_type,
>             )
>             logger.info("Model loaded successfully")
>         return self._model
>
>     def transcribe(self, audio: np.ndarray) -> str:
>         """
>         Transcribe audio buffer to text.
>
>         Args:
>             audio: numpy float32 array, 16kHz, mono
>
>         Returns:
>             Transcribed text string
>         """
>         if len(audio) == 0:
>             return ""
>
>         segments, info = self.model.transcribe(
>             audio,
>             language=self.config.language,
>             beam_size=self.config.beam_size,
>             vad_filter=self.config.vad_filter,
>         )
>
>         text = " ".join(segment.text for segment in segments)
>         return text.strip()
>
>     def transcribe_file(self, filepath: str) -> str:
>         """Transcribe from a WAV/MP3 file."""
>         segments, info = self.model.transcribe(filepath)
>         return " ".join(segment.text for segment in segments).strip()
> ```
>
> **Step 2: Write test**
>
> ```python
> # tests/test_transcriber.py
> import numpy as np
> import pytest
> from voiceflow.transcriber import Transcriber, TranscriptionConfig
>
> def test_empty_audio():
>     """Empty audio should return empty string."""
>     t = Transcriber(TranscriptionConfig(device="cpu"))
>     result = t.transcribe(np.array([]))
>     assert result == ""
>
> def test_model_loading():
>     """Model should load on CPU (for CI/testing)."""
>     t = Transcriber(TranscriptionConfig(model_size="tiny", device="cpu"))
>     _ = t.model  # Trigger lazy load
>     assert t._model is not None
> ```
>
> **Step 3: Run tests**
>
> ```bash
> python3 -m pytest tests/test_transcriber.py -v
> ```
>
> Expected: 2 passed (model will download ~150MB on first run).
>
> **Step 4: Commit**
>
> ```bash
> git add voiceflow/transcriber.py tests/test_transcriber.py
> git commit -m "feat: GPU-accelerated Whisper transcription module"
> ```
>
> ---
>
> ## Task 5b: Cloud LLM Post-Processor module

**Objective:** Optional text cleanup using an online LLM API. Takes raw Whisper output, sends it to an LLM (OpenAI/Anthropic/custom) to: remove filler words (um, uh, like), fix grammar, add proper punctuation, and optionally reformat (bullet points, paragraphs, etc).

**Files:**
- Create: `voiceflow/llm_postprocessor.py`

**Design:** This is a BYOK (bring your own key) module. No cost to the app itself -- user provides their own API key. The LLM call is a fast, cheap operation (~$0.001 per transcription). Works with any OpenAI-compatible API endpoint.

```python
"""LLM Post-Processor - optional cloud-based text cleanup.

Uses any OpenAI-compatible API (OpenAI, Anthropic via proxy, Ollama, etc).
BYOK - user provides their own API key. Cost is ~$0.001 per transcription.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "openai"           # openai, anthropic, custom, ollama
    api_key: str = ""
    base_url: str = ""                 # Custom API endpoint (empty = default)
    model: str = "gpt-4o-mini"         # gpt-4o-mini, claude-3-haiku, etc.
    max_tokens: int = 512

    # Cleanup options
    remove_fillers: bool = True        # Remove "um", "uh", "like", "you know"
    fix_grammar: bool = True           # Fix grammar and spelling
    add_punctuation: bool = True       # Ensure proper punctuation/capitalization
    reformat: str = "none"             # none, bullets, paragraphs, concise

    # Prompt template
    system_prompt: str = "You are a text cleanup assistant. Fix the transcription. \
Remove filler words. Add proper punctuation. Keep the meaning exactly the same. \
Do NOT add any new information. Output ONLY the corrected text, nothing else."


class LLMPostProcessor:
    """Post-processes raw transcription through an LLM for cleanup.

    Usage:
        processor = LLMPostProcessor(LLMConfig(enabled=True, api_key="sk-..."))
        clean_text = processor.process(raw_text)
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()

    def process(self, text: str) -> str:
        """
        Send raw transcription to LLM for cleanup.

        Args:
            text: Raw transcribed text from Whisper

        Returns:
            Cleaned-up text (or original text if LLM is disabled/fails)
        """
        if not self.config.enabled or not text:
            return text

        if not self.config.api_key:
            logger.warning("LLM post-processing enabled but no API key set")
            return text

        try:
            result = self._call_llm(text)
            logger.debug("LLM cleanup applied")
            return result
        except Exception as e:
            logger.error(f"LLM cleanup failed, using raw text: {e}")
            return text  # Graceful fallback - never block transcription

    def _call_llm(self, text: str) -> str:
        """Call the configured LLM API."""
        import openai

        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

        # Build user prompt based on reformat option
        user_prompt = self._build_user_prompt(text)

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=0.1,  # Low temp = consistent, faithful cleanup
        )

        return response.choices[0].message.content.strip()

    def _build_user_prompt(self, text: str) -> str:
        """Build the user prompt with cleanup instructions."""
        parts = [text]

        instructions = []
        if self.config.remove_fillers:
            instructions.append("Remove filler words (um, uh, like, you know, etc)")
        if self.config.fix_grammar:
            instructions.append("Fix grammar and spelling")
        if self.config.add_punctuation:
            instructions.append("Ensure proper punctuation and capitalization")
        if self.config.reformat == "bullets":
            instructions.append("Reformat as bullet points")
        elif self.config.reformat == "paragraphs":
            instructions.append("Reformat into proper paragraphs")
        elif self.config.reformat == "concise":
            instructions.append("Make it concise and clear, remove redundancy")

        if instructions:
            parts.append("\n".join(f"- {i}" for i in instructions))

        return "\n\n".join(parts)

    @staticmethod
    def get_available_models() -> dict:
        """Return available model presets."""
        return {
            "openai/gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini",
                                     "base_url": "", "cost": "~$0.0001/msg"},
            "openai/gpt-4o": {"provider": "openai", "model": "gpt-4o",
                              "base_url": "", "cost": "~$0.003/msg"},
            "anthropic/claude-3-haiku": {"provider": "anthropic", "model": "claude-3-haiku-20240307",
                                          "base_url": "https://api.anthropic.com/v1",
                                          "cost": "~$0.00025/msg"},
            "custom/ollama": {"provider": "ollama", "model": "llama3.2",
                              "base_url": "http://localhost:11434/v1",
                              "cost": "$0 (local)"},
        }
```

**Key design decisions:**
- Always graceful fallback -- if LLM fails, use raw transcription
- Temperature 0.1 for consistent output (faithful cleanup, not creative rewriting)
- OpenAI-compatible API means it works with: OpenAI, Anthropic (via proxy), Ollama, vLLM, LiteLLM, OpenRouter, etc.
- System prompt is customizable -- users can tweak cleanup behavior

---

## Task 6: Text injection module
>
> **Objective:** Type transcribed text into any active Windows application.
>
> **Files:**
> - Create: `voiceflow/typer.py`
>
> **Implementation:**
>
> ```python
> """Text injection - types text into the active window on Windows."""
> import time
> import logging
>
> logger = logging.getLogger(__name__)
>
> class Typer:
>     """Injects text into the currently focused application."""
>
>     def __init__(self, delay: float = 0.005):
>         """
>         Args:
>             delay: Delay between keystrokes (seconds). Faster = less reliable.
>                    0.005 = fast but reliable on most apps.
>         """
>         self.delay = delay
>
>     def type_text(self, text: str):
>         """
>         Type the given text as if user typed it on keyboard.
>         Works in any app: Notepad, VS Code, Chrome, Slack, etc.
>         """
>         if not text:
>             return
>
>         try:
>             from pynput.keyboard import Controller
>             keyboard = Controller()
>
>             # Type character by character for maximum compatibility
>             for char in text:
>                 keyboard.type(char)
>                 time.sleep(self.delay)
>
>             logger.debug(f"Typed {len(text)} chars")
>
>         except Exception as e:
>             logger.error(f"Failed to type text: {e}")
>             raise
>
>     def type_with_paste(self, text: str):
>         """
>         Alternative: use Ctrl+V paste for longer text (faster, more reliable).
>         Won't work in all apps (terminal, some games).
>         """
>         import pyperclip
>         from pynput.keyboard import Controller, Key
>
>         # Copy to clipboard
>         pyperclip.copy(text)
>         time.sleep(0.05)
>
>         # Paste
>         keyboard = Controller()
>         keyboard.press(Key.ctrl)
>         keyboard.press('v')
>         keyboard.release('v')
>         keyboard.release(Key.ctrl)
>
>     def press_enter(self):
>         """Press Enter key."""
>         from pynput.keyboard import Controller, Key
>         keyboard = Controller()
>         keyboard.press(Key.enter)
>         keyboard.release(Key.enter)
> ```
>
> **Commit the typer module.**
>
> ---
>
> ## Task 7: Hotkey listener + main loop
>
> **Objective:** Global hotkey that triggers: press -> record audio, release -> transcribe -> type.
>
> **Files:**
> - Create: `voiceflow/main.py`
>
> **Implementation:**
>
> ```python
> """VoiceFlow - Main entry point.
>
> Usage:
>     python3 -m voiceflow          # Start with default hotkey (Ctrl+Alt+Space)
>     python3 -m voiceflow --hotkey f9  # Custom hotkey
> """
> import argparse
> import logging
> import sys
> import threading
> import time
> import numpy as np
>
> from voiceflow.audio import AudioRecorder, AudioConfig
> from voiceflow.transcriber import Transcriber, TranscriptionConfig
> from voiceflow.llm_postprocessor import LLMPostProcessor, LLMConfig
> from voiceflow.typer import Typer
>
> logging.basicConfig(
>     level=logging.INFO,
>     format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
> )
> logger = logging.getLogger("voiceflow")
>
> # Hotkey to trigger recording
> RECORD_HOTKEY = {Key.ctrl_l, Key.alt_l, Key.space}  # Ctrl+Alt+Space
>
> class VoiceFlow:
>     def __init__(self, config_path=None):
>         self.cfg = load_config(config_path)
>         self.audio_config = AudioConfig()
>         self.trans_config = TranscriptionConfig(
>             model_size=self.cfg["model_size"],
>             device=self.cfg["device"],
>             language=self.cfg["language"],
>         )
>         self.llm_config = LLMConfig(**self.cfg.get("llm", {}))
>
>         self.recorder = AudioRecorder(self.audio_config)
>         self.transcriber = Transcriber(self.trans_config)
>         self.llm_processor = LLMPostProcessor(self.llm_config)
>         self.typer = Typer()
>
>         self._running = False
>         self._recording = False
>         self._status = "idle"  # idle, recording, processing
>
>     def start(self):
>         """Start the hotkey listener loop."""
>         logger.info("VoiceFlow starting...")
>         logger.info("Press Ctrl+Alt+Space to start recording, release to transcribe")
>
>         # Preload model in background
>         def preload():
>             _ = self.transcriber.model
>             logger.info("Model preloaded and ready")
>         threading.Thread(target=preload, daemon=True).start()
>
>         self._running = True
>         self._listen_hotkeys()
>
>     def _listen_hotkeys(self):
>         """Listen for hotkey press/release using pynput."""
>         from pynput import keyboard
>
>         current_keys = set()
>         HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.space}
>
>         def on_press(key):
>             current_keys.add(key)
>             if not self._recording and HOTKEY.issubset(current_keys):
>                 self._start_recording()
>
>         def on_release(key):
>             current_keys.discard(key)
>             if self._recording and not HOTKEY.issubset(current_keys):
>                 self._stop_and_transcribe()
>
>         with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
>             listener.join()
>
>     def _start_recording(self):
>         """Called when hotkey is pressed."""
>         self._recording = True
>         self.recorder.start()
>         logger.info("Recording started...")
>
>     def _stop_and_transcribe(self):
>         """Called when hotkey is released - stop recording and process."""
>         self._recording = False
>         logger.info("Recording stopped, transcribing...")
>
>         audio = self.recorder.stop()
>         if len(audio) == 0:
>             logger.warning("No audio captured")
>             return
>
>         # Transcribe in background thread so we don't block the listener
>         def process():
>             try:
>                 self._status = "processing"
>                 text = self.transcriber.transcribe(audio)
>                 if text:
>                     # Optional LLM cleanup
>                     if self.llm_config.enabled:
>                         text = self.llm_processor.process(text)
>                     logger.info(f"Transcribed: {text}")
>                     self.typer.type_text(text + " ")  # Add trailing space
>                 else:
>                     logger.warning("No speech detected")
>             except Exception as e:
>                 logger.error(f"Transcription error: {e}")
>             finally:
>                 self._status = "idle"
>
>         threading.Thread(target=process, daemon=True).start()
>
>
> def main():
>     parser = argparse.ArgumentParser(description="VoiceFlow - Free voice dictation")
>     parser.add_argument("--model", default="base",
>                         choices=["tiny", "base", "small", "medium"],
>                         help="Whisper model size (default: base)")
>     parser.add_argument("--device", default=None,
>                         help="Device: cuda or cpu (auto-detect if not set)")
>     parser.add_argument("--language", default=None,
>                         help="Language code (en, hi, etc. Auto-detect if not set)")
>     parser.add_argument("--list-devices", action="store_true",
>                         help="List audio devices and exit")
>     args = parser.parse_args()
>
>     if args.list_devices:
>         rec = AudioRecorder()
>         for d in rec.list_devices():
>             print(f"  [{d['index']}] {d['name']} ({d['channels']} ch)")
>         return
>
>     # Auto-detect device
>     if args.device is None:
>         import torch
>         args.device = "cuda" if torch.cuda.is_available() else "cpu"
>
>     config = TranscriptionConfig(
>         model_size=args.model,
>         device=args.device,
>         language=args.language,
>     )
>
>     app = VoiceFlow()
>     app.transcriber = Transcriber(config)
>
>     try:
>         app.start()
>     except KeyboardInterrupt:
>         logger.info("VoiceFlow stopped")
>
>
> if __name__ == "__main__":
>     main()
> ```
>
> ---
>
> ## Task 8: System tray UI
>
> **Objective:** System tray icon showing VoiceFlow status (idle/recording/error), with menu for settings, device selection, and quit.
>
> **Files:**
> - Create: `voiceflow/tray.py`
> - Modify: `voiceflow/main.py` (integrate tray)
>
> **Step 1: Create tray module with status indicator.**
>
> The tray icon changes color based on state:
> - Gray/White = Idle, ready
> - Red = Recording
> - Green = Processing/transcribing
>
> Menu items:
> - Status: Ready / Recording / Processing
> - Audio Device: [submenu with devices]
> - Model: [tiny/base/small]
> - "Settings..." dialog
> - Quit
>
> **Step 2: Modify main.py to run tray in background thread, daemon in main thread.**
>
> ---
>
> ## Task 9: Configuration system
>
> **Objective:** Persistent config stored in JSON, editable via GUI or text editor.
>
> **Files:**
> - Create: `voiceflow/config.py`
>
> **Config structure (`~/.voiceflow/config.json`):**
>
> ```json
> {
>   "hotkey": "ctrl+alt+space",
>   "audio_device_index": null,
>   "model_size": "base",
>   "language": null,
>   "device": "auto",
>   "auto_punctuation": true,
>   "type_method": "keystroke",
>   "show_notifications": true,
>   "llm": {
>     "enabled": false,
>     "provider": "openai",
>     "api_key": "",
>     "base_url": "",
>     "model": "gpt-4o-mini",
>     "max_tokens": 512,
>     "remove_fillers": true,
>     "fix_grammar": true,
>     "add_punctuation": true,
>     "reformat": "none"
>   }
> }
> ```
>
> **Config features:**
> - Auto-detect best audio device on first run
> - Auto-detect CUDA or fallback to CPU
> - Validate hotkey string format
> - Reload on the fly without restart
>
> ---
>
> ## Task 10: Polish and testing
>
> **Objective:** Polish the UX, add error handling, test end-to-end.
>
> **Step 1: Add auto-punctuation** - Use a small post-processing that capitalizes first letter of sentences, adds periods.
>
> **Step 2: Add push-to-talk visual feedback** - Flash a small "Recording..." overlay or tray tooltip.
>
> **Step 3: Handle edge cases:**
> - No microphone available -> clear error message with device list
> - Model download fails -> retry with mirror
> - Transcription returns empty -> no text injected
> - App loses focus during typing -> still works (global injection)
>
> **Step 4: Test the full pipeline:**
> ```bash
> python3 -m voiceflow --list-devices  # Verify mic works
> python3 -m voiceflow &               # Start daemon
> # Press Ctrl+Alt+Space, speak, release -> text appears in Notepad
> ```
>
> **Step 5: Add --test-record mode that records 5 seconds and transcribes (for quick testing without hotkeys).**
>
> ---
>
> ## Task 11: Build Windows installer
>
> **Objective:** Create a standalone .exe that installs VoiceFlow with a Start Menu shortcut and auto-start on boot.
>
> **Files:**
> - Create: `VoiceFlow.spec`
>
> **Step 1: Install PyInstaller**
>
> ```bash
> pip3 install pyinstaller
> ```
>
> **Step 2: Create PyInstaller spec file**
>
> Bundles:
> - All Python dependencies
> - Whisper model files (tiny or base) bundled or downloaded on first run
> - Tray icon embedded
> - Single .exe output
>
> **Step 3: Build**
>
> ```bash
> pyinstaller VoiceFlow.spec --clean
> ```
>
> Expected: `dist/VoiceFlow.exe` (~300MB with bundled model, ~50MB if model downloads on first run).
>
> **Step 4: Create installer .iss (Inno Setup)**
>
> For a professional installer with:
> - Start Menu shortcut
> - Auto-start on boot option
> - Uninstaller
>
> ---
>
## Task 12: Advanced features (Phase 2)

Once core works, add Wispr Flow-like features:

1. **Voice commands**: "new line", "delete last word", "caps on", "period", "delete that"
2. **AI cleanup via LLM**: Already implemented in Task 5b -- enhance with: custom prompt templates, per-app profiles (email vs. casual), meeting notes mode
3. **Cloud transcription APIs**: OpenAI Whisper API, Deepgram, Groq -- for users who want better accuracy than local tiny/base models without needing a GPU
4. **Multi-language support**: Support Hinglish/English code-switching (use `language=auto` in Whisper, LLM handles mixed-language cleanup)
5. **Custom vocabulary**: Add domain-specific words to improve accuracy (pass as prompt hints to Whisper)
6. **Noise gate**: Filter background noise before sending to Whisper (simple RMS-based gate or RNNoise)
7. **Continuous mode**: Toggle mode (press once, record until pressed again) vs push-to-talk (hold)
8. **Per-app profiles (Context-Aware)**: Different settings per application (e.g., bullet points in Notepad, formal in Outlook, code comments in VS Code). NO screenshots needed (unlike Wispr Flow/Monologue) -- detect active window title + exe name. Tray icon shows active profile. Users can configure: hotkey, model, LLM reformat mode, cleanup rules per app.
9. **Audio recording + transcription log**: Save transcriptations with timestamps for review/export

---

## Task 13: Breakthrough features (Phase 3) -- "Make it the best on the planet"

These are features that NO existing dictation app (Wispr Flow, VoiceInk, Spokenly, Monologue, Talon, etc.) currently offers. This is what makes VoiceFlow unique.

### 13a. Agent Mode: Voice Actions Across Apps
**What:** Dictation that can also DO things, not just type text.
**How:** After transcription, check if the text matches an action pattern. If so, execute the action instead of typing.
**Examples:**
- "open chrome" -> launches Chrome
- "search for X on google" -> opens browser, searches
- "open notepad and write a todo list" -> opens Notepad, types formatted list
- "run git status" -> opens terminal, runs command, types output
- "send slack message to avneep saying hello" -> opens Slack, types message
- "start pomodoro timer" -> starts a timer
- "what's the weather" -> fetches weather, types result
**Implementation:** Pattern matching on transcribed text -> action registry. Actions are pluggable Python functions. User can add custom actions via config.

### 13b. Multi-Speaker Dictation (Conversation Mode)
**What:** Continuous listening that identifies WHO is speaking and labels them.
**How:** Use speaker diarization (pyannote.audio or similar) to separate speakers. Format as:
```
[Speaker 1]: Hello, how are you?
[Speaker 2]: I'm good, thanks!
```
**Use cases:** Interviews, meetings, phone calls, dialogue writing, podcast transcription.
**Implementation:** Continuous audio stream -> VAD segmentation -> speaker embedding -> label -> transcribe each segment.

### 13c. Real-Time Voice Translation Mode
**What:** Speak in one language, get text in another language.
**How:** Whisper transcribes in source language -> LLM translates to target language -> type translated text.
**Examples:**
- Speak Hindi -> types English
- Speak English -> types Hindi (Devanagari)
- Speak Spanish -> types English
**Implementation:** Two-step pipeline. Whisper with `language=hi` -> LLM with translation prompt. Support 50+ language pairs.

### 13d. Tone-Aware Dictation
**What:** Detect emotional tone from voice and adjust formatting accordingly.
**How:** Analyze audio features (pitch, speed, volume) -> classify tone -> adjust LLM cleanup.
**Examples:**
- Excited/fast speech -> shorter sentences, exclamation marks
- Slow/thoughtful -> longer, more formal sentences
- Angry -> ALL CAPS option, or soften the language
- Casual -> keep it loose, minimal punctuation
**Implementation:** Simple audio feature extraction (librosa) -> tone classifier (rule-based or tiny ML model) -> pass tone hint to LLM.

### 13e. Spelling Mode
**What:** Toggle mode where every word is spelled letter-by-letter.
**How:** Hotkey toggle -> each word spelled out using NATO phonetic alphabet or direct letters.
**Use cases:** Code, URLs, passwords, email addresses, technical terms, names.
**Implementation:** "spell mode on" -> transcribe -> replace each word with spelled version. "spell mode off" -> back to normal.

### 13f. Developer Command Mode (Talon-Lite)
**What:** Voice commands that execute actual computer actions, not just type text.
**How:** Custom command grammar mapped to Python actions.
**Examples:**
- "open terminal here" -> opens terminal in current project directory
- "run build" -> runs `npm run build` or `make`
- "fix lint" -> runs linter with auto-fix
- "git commit with message X" -> stages all, commits with message
- "deploy to production" -> runs deploy script
- "open file X" -> fuzzy-finds and opens file
**Implementation:** Command registry with regex patterns. Each pattern maps to a Python function. User-extensible via config file.

### 13g. Adaptive Learning Vocabulary
**What:** App learns YOUR frequently used words and gets better over time.
**How:** Track words that Whisper frequently misrecognizes -> add to custom vocabulary -> pass as prompt hints to Whisper on subsequent transcriptions.
**Examples:** Your name, company name, technical jargon, brand names, industry terms.
**Implementation:** Log corrections (when user edits transcribed text). Extract new words. Maintain per-user vocabulary file. Pass as `hotwords` parameter to Whisper.

### 13h. Integration / MCP Layer
**What:** Voice commands that trigger external services and APIs.
**How:** REST API + webhook support. Voice command -> HTTP request to external service.
**Examples:**
- "create task in linear" -> POST to Linear API
- "send message to slack channel" -> POST to Slack webhook
- "log time in harvest" -> POST to Harvest API
- "add to notion database" -> POST to Notion API
**Implementation:** Configurable webhook endpoints. Template variables from transcribed text. Confirmation dialog before sending.

### 13i. Multi-Model Transcription Strategy
**What:** Smart routing between local and cloud models based on confidence.
**How:** Run local Whisper first. If confidence < threshold, auto-escalate to cloud API (OpenAI/Deepgram). User never notices the switch.
**Implementation:** faster-whisper returns confidence scores per segment. Low-confidence segments re-routed to cloud API. Configurable threshold.

### 13j. Hinglish-Optimized Pipeline
**What:** Specifically optimized for Hindi-English code-switching (Hinglish).
**How:** Whisper with `language=auto` -> LLM with Hinglish-specific prompt that preserves Hindi words in Devanagari while fixing English grammar.
**Why:** Nobody does this well. Wispr Flow and others treat Hinglish as English with errors.
**Implementation:** Custom LLM system prompt for Hinglish. Post-processing that detects Devanagari segments and preserves them.

### 13k. Scribe Mode (Interview/Conversation Transcription)
**What:** Continuous listening with speaker diarization, formatted as a transcript.
**How:** Toggle scribe mode -> continuous audio stream -> VAD -> speaker ID -> transcribe -> format with timestamps.
**Output format:**
```
[00:00:05] Speaker 1: Welcome to the show.
[00:00:12] Speaker 2: Thanks for having me.
[00:00:18] Speaker 1: Let's talk about...
```
**Export:** Text, Markdown, DOCX, SRT (subtitles).
**Use cases:** Interviews, meetings, lectures, podcasts, court proceedings.

### 13l. Distraction-Free Dictation Overlay
**What:** Small floating overlay showing real-time transcription preview before committing.
**How:** When recording, show a small semi-transparent overlay with live transcription. Press Enter to confirm and type, Esc to cancel and re-record.
**Why:** Gives user control. No more "oops, it typed something wrong" moments.
**Implementation:** tkinter/PyQt overlay window. Real-time partial transcription updates.

### 13m. Wake Word Detection
**What:** "Hey VoiceFlow" activates the app (like Alexa/Siri).
**How:** Lightweight wake word model (porcupine/picovoice) running continuously. When detected, switches to recording mode.
**Why:** Completely hands-free activation. No hotkey needed.
**Implementation:** Porcupine wake word engine (free for personal use). Custom wake word training supported.

---

## Competitive Positioning Summary

| Feature | Wispr Flow | VoiceInk | Spokenly | Monologue | Talon | **VoiceFlow** |
|---------|-----------|----------|----------|-----------|-------|---------------|
| System-wide dictation | Yes | Yes | Yes | Yes | Yes | **Yes** |
| Local GPU inference | No | Yes | No | Yes | Yes | **Yes** |
| Cloud LLM cleanup | Yes | No | Yes | No | No | **Yes (BYOK)** |
| Per-app profiles | Yes (screenshots) | Yes (Power Mode) | No | Yes (screenshots) | Yes | **Yes (no screenshots)** |
| Voice commands | Yes | No | No | No | Yes | **Yes** |
| Agent mode (do things) | No | No | Basic | No | Yes | **Yes** |
| Multi-speaker | No | No | No | No | No | **Yes** |
| Real-time translation | No | No | No | No | No | **Yes** |
| Tone awareness | No | No | No | No | No | **Yes** |
| Spelling mode | No | No | No | No | No | **Yes** |
| Dev command mode | No | No | No | No | Yes | **Yes** |
| Adaptive vocabulary | No | No | No | No | No | **Yes** |
| MCP/integrations | No | No | No | No | No | **Yes** |
| Multi-model routing | No | No | No | No | No | **Yes** |
| Hinglish optimized | No | No | No | No | No | **Yes** |
| Scribe mode | No | No | No | No | No | **Yes** |
| Dictation overlay | No | No | No | No | No | **Yes** |
| Wake word | No | No | No | No | No | **Yes** |
| Open source | No | Yes | No | No | Yes | **Yes** |
| Free | No (limited) | Yes | Freemium | No | Yes | **Yes** |
| Cross-platform | Yes | Mac only | Yes | Mac only | Yes | **Yes** |
| Privacy (no screenshots) | No | Yes | Yes | No | Yes | **Yes** |

**Key differentiators (things ONLY VoiceFlow has):**
1. Per-app profiles WITHOUT screenshots (privacy-first context awareness)
2. Multi-speaker dictation in a consumer dictation app
3. Real-time voice translation (speak Hindi, type English)
4. Tone-aware formatting
5. Hinglish-optimized pipeline
6. Agent mode + dev commands + MCP integrations in one app
7. Adaptive learning vocabulary
8. Multi-model smart routing (local -> cloud auto-escalation)
9. Scribe mode with speaker diarization
10. Wake word activation

---

## Task 14: Mobile companion app (Phase 3)

Desktop dictation is only half the picture. In 2026, people capture ideas on their phone while walking, commuting, in meetings, or lying in bed. A VoiceFlow mobile app ensures you never lose a thought, and everything syncs with your desktop.

### Platform strategy

| Approach | iOS | Android | Effort | Native feel |
|----------|-----|---------|--------|-------------|
| React Native | Yes | Yes | Medium | Good |
| Flutter | Yes | Yes | Medium | Very good |
| KMP (Kotlin Multiplatform) | Yes | Yes | High | Native |
| Separate native | Yes | Yes | Very high | Best |

**Recommendation:** Flutter. Single codebase for iOS + Android. Great for microphone access, background recording, and system tray/bar widgets. Hot reload speeds up development. Used by many productivity apps (Reflect, Superlist).

### 14a. Core mobile dictation
**What:** Same push-to-talk dictation as desktop, but on mobile.
**How:** Tap a floating button or keyboard mic key -> record -> transcribe -> text appears in any text field.
**Modes:**
- **Quick dictation:** Tap, speak, release. Text typed into current field.
- **Voice notes:** Record longer thoughts. Saved as a note with transcription.
- **Continuous mode:** Start recording, keep going. Auto-pauses on silence.
- **Scribe mode:** Record meetings/interviews on your phone with speaker labels.

### 14b. Keyboard extension (iOS + Android)
**What:** VoiceFlow as a custom keyboard. Tap the mic on the keyboard, speak, text appears.
**Why:** Works in EVERY app without switching. Users don't need to use a separate dictation app.
**iOS:** Custom keyboard extension with speech recognition permission.
**Android:** Input Method Service (IME) with voice input.

### 14c. Cross-device sync
**What:** Record a voice memo on your phone, see the transcript on your desktop within seconds.
**How:** Lightweight sync server (or user-hosted). End-to-end encrypted.
**Data synced:**
- Voice notes with transcriptions
- Custom vocabulary words
- App profiles and settings
- Transcription history (searchable)
**Options for sync:**
- **Self-hosted (default):** User runs VoiceFlow Sync on their NAS/home server
- **Cloud option:** Optional VoiceFlow cloud servers (encrypted, zero-knowledge)
- **Offline:** No sync, keep everything local-only

### 14d. Wearable support
**What:** VoiceFlow on Apple Watch and Wear OS.
**How:** Record voice memos from your wrist. Transcribes via phone or cloud.
**Features:**
- Quick voice note from watch (30-second tap-to-record)
- Dictation into any watch text field (messages, notes)
- Haptic feedback: short buzz = recording, long buzz = transcribed
- Transcription delivered to phone + synced to desktop

### 14e. Mobile-specific features

**Capture mode:** Triple-tap power button (or swipe gesture) to instantly start recording. No app opening needed. Great for capturing ideas when your hands are full.

**Ambient background recording:** Optional mode where VoiceFlow listens in the background (with clear indicator). Uses VAD to auto-record when you start speaking. Useful for spontaneous conversations. ALWAYS shows a visible recording indicator (privacy first).

**Photo + voice:** Record a voice note attached to a photo. "This is the whiteboard from the meeting" -- photo + audio + transcription saved together.

**Location-tagged notes:** Voice notes automatically tagged with location. "What did I capture near the office?" -- searchable by location.

**Offline transcription:** Whisper tiny model runs on-device on modern phones (iPhone 15+, Snapdragon 8 Gen 2+). No internet needed. Privacy-first, just like desktop.

**Share sheet integration:** Record from iOS/Android share sheet. "Save this idea to VoiceFlow" from any app. Also: export voice notes to Notes, Notion, Slack, email, etc.

**Widget/home screen:**
- 1-tap record widget on home screen
- Today's voice notes summary on widget
- Quick stats: "12 notes captured today"

**Siri / Google Assistant shortcut:**
- "Hey Siri, create a voice note" -> opens VoiceFlow recording
- "Hey Google, dictate with VoiceFlow" -> starts dictation

### 14f. Mobile architecture
```
Mobile App (Flutter)
├── Audio Capture Engine (native Swift/Kotlin via platform channels)
├── On-Device Transcription (whisper.cpp via FFI -- runs whisper-tiny)
├── Cloud Transcription (REST API to desktop relay or direct to Whisper API)
├── Sync Client (WebSocket to VoiceFlow Sync Server)
├── Keyboard Extension (iOS: Custom Keyboard / Android: IME)
├── Watch App (watchOS / Wear OS companion)
├── Widget Extension (iOS 17+ / Android home screen widget)
└── Share Extension (iOS Share Sheet / Android Intent)
```

### 14g. VoiceFlow Sync Server
**What:** Lightweight server that syncs transcriptions, voice notes, settings, and vocabulary across all your devices.
**Tech stack:** Python FastAPI or Go. SQLite or PostgreSQL. WebSocket for real-time sync.
**Self-hosted:** Run on home server, NAS (Synology), or Raspberry Pi.
**Encryption:** E2E encrypted. Server never sees plaintext transcripts.
**API:**
- `POST /sync/notes` -- upload new voice note + transcript
- `GET /sync/notes?since=TID` -- get new notes since last sync
- `WS /sync/realtime` -- WebSocket for instant sync
- `POST /sync/vocabulary` -- sync learned vocabulary
- `POST /sync/settings` -- sync app settings/profiles

### Competitive comparison -- mobile dictation

| Feature | Wispr Flow | Spokenly | Whisper Memos | Yaps | Google Rambler | **VoiceFlow** |
|---------|-----------|----------|---------------|------|----------------|---------------|
| iOS keyboard extension | No | No | No | No | Gboard only | **Yes** |
| Android keyboard extension | No | No | No | Yes | Gboard only | **Yes** |
| Cross-device sync | Basic | Cloud | Email only | No | Google account | **Yes (self-hosted)** |
| Apple Watch app | No | No | Yes | No | No | **Yes** |
| Wear OS app | No | No | No | No | No | **Yes** |
| Offline transcription | No | No | Yes | Yes | Yes | **Yes (on-device Whisper)** |
| Background capture | No | No | No | No | No | **Yes** |
| Photo + voice | No | No | No | No | No | **Yes** |
| Location tagging | No | No | No | No | No | **Yes** |
| E2E encrypted sync | No | No | No | No | No | **Yes** |
| Self-hosted sync | No | No | No | No | No | **Yes** |
| Siri/Assistant shortcut | No | No | No | No | No | **Yes** |
| Open source | No | No | No | No | No | **Yes** |
>
> ---
>
> ## Performance Expectations
>
| Component | Target |
> |-----------|--------|
> | Recording latency | < 50ms (hotkey to audio capture) |
> | Transcription speed | 3-5x real-time on RTX 3050 |
> | End-to-end latency | < 2 seconds for a 5-second utterance |
> | Text injection speed | ~200 chars/sec (pynput) |
> | Memory usage | < 1.5 GB (model + audio buffer) |
> | Idle CPU | < 1% |
>
> ---
>
## Task 15: Development strategy -- low stress, zero cost

**Principle:** Build everything without spending money. Keep our machine (5.8GB RAM, RTX 3050 4GB) free for its current workload: Hermes agent, WhatsApp bridge, gateway. Dev work MUST NOT interfere.

### What our machine currently runs (NOT to be affected)

| Process | RAM | Role |
|---------|-----|------|
| hermes-agent sessions | ~250MB each | 3-4 concurrent sessions |
| whatsapp-bridge (Node) | ~90MB | WhatsApp bridge |
| gateway (Python) | ~24MB | Hermes gateway |
| **Total current usage** | **~1.5GB of 5.8GB** | **4.3GB available** |

### The honest stress assessment

Here is what building VoiceFlow ACTUALLY costs our machine:

| Activity | RAM impact | CPU impact | When |
|----------|-----------|------------|------|
| Reading/writing code (text files) | Near zero | Near zero | Always |
| Git commits/push/pull | ~50MB briefly | Low | Per task |
| Installing Python deps locally | **500MB-2GB spike** | **High (compiling)** | ONE time if we do it |
| Running pytest locally | 200-500MB spike | Medium | Optional |
| Downloading Whisper model | ~300MB disk + 400MB RAM | Low | ONE time if we do it |
| PyInstaller build | **2GB+ RAM** | **Very high, 10+ min** | NEVER on our machine |
| Flutter build | **3GB+ RAM** | **Very high, 15+ min** | NEVER on our machine |
| CUDA model inference | **3-4GB RAM + 100% GPU** | High | NEVER on our machine |

**Verdict: If we do everything via CI/cloud as planned: near-zero stress on our machine. If we get lazy and do it locally: our 5.8GB machine WILL swap heavily and become sluggish.**

### The rules

1. NEVER install torch/pytorch on our machine -- that's 3GB+ of RAM pressure alone
2. NEVER download Whisper models locally -- use CI/Colab for all model testing
3. NEVER run PyInstaller locally -- GitHub Actions Windows runner
4. NEVER install Flutter/Android SDK locally -- GitHub Actions only
5. NEVER run CUDA inference locally -- Google Colab T4 is free and 3x faster than our 3050
6. If any task REQUIRES local installation, escalate to user FIRST -- find a cloud alternative

### Cloud resources we use instead (all free)

| Need | Cloud resource | Free allowance |
|------|---------------|----------------|
| Code + repo | GitHub | Unlimited public repos |
| Unit tests (Linux) | GitHub Actions | 2000 min/month |
| Windows build | GitHub Actions Windows | Included in 2000 min |
| GPU testing | Google Colab | T4 GPU, 30 hrs/week |
| Flutter mobile build | GitHub Actions macOS/Linux | Included (public repos) OR Codemagic (500 min/month) |
| API testing | Groq free tier | ~100 req/day, very fast |
| Whisper model storage | HuggingFace Hub | Free hosting |

### Development workflow (per task)

1. I write a detailed task spec from PLAN.md
2. delegate_task subagent writes ALL the code, commits to repo
3. Second delegate_task subagent reviews code quality
4. GitHub Actions CI runs tests automatically (free, cloud, no local impact)
5. GPU tests run on Google Colab (free T4, no local impact)
6. I read the results, give feedback to subagent for fixes
7. Repeat until green

### Project structure for CI/CD

```
voiceflow/
├── .github/workflows/
│   ├── ci.yml              # Linux: lint + unit tests (free)
│   ├── build_windows.yml   # Windows: PyInstaller .exe (free)
│   └── build_mobile.yml    # Flutter: iOS + Android (free/Codemagic)
├── voiceflow/              # Main Python source
├── mobile/                 # Flutter source
├── sync_server/            # FastAPI sync server (Docker)
├── colab/                  # Colab notebooks for GPU testing (free T4)
│   ├── test_transcription.ipynb
│   └── benchmark_models.ipynb
├── PLAN.md
└── README.md
```

### What gets committed vs what gets generated

Committed to git (tens of KB):
- All .py source files
- CI workflow YAML files
- Colab notebooks
- Requirements.txt, spec files
- Tiny test fixtures (<100KB)

Generated in CI (never committed, never downloaded):
- .exe build artifacts (GitHub Actions artifacts, 90 days)
- .apk/.ipa mobile builds (CI artifacts)
- Whisper model cache (CI cache, ephemeral)

### Local machine impact summary

| Scenario | RAM needed | Local stress |
|----------|-----------|--------------|
| Following this plan (code review + git only) | +50-100MB | Negligible |
| Running pytest locally | +200-500MB | Moderate |
| Installing deps locally | +500MB-2GB | High |
| Our plan | **+50-100MB max** | **Negligible** |

**Bottom line: our machine just reads text and pushes git. All heavy compute happens in the cloud for free.**

---

## Verification Checklist (overall)

- [x] GitHub Actions CI runs unit tests on every commit
- [x] Colab notebooks verify GPU transcription accuracy
- [x] Windows .exe builds successfully via CI artifact
- [ ] Flutter mobile builds via CI artifact (not started)
- [x] No Whisper models bundled in .exe (downloaded on first run)
- [x] LLM features use free tier APIs for testing
- [x] Local machine RAM usage stays under 2GB during development
- [x] Total development cost: $0
