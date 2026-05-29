"""

VoiceFlow - Main entry point.

System-wide voice dictation with local Whisper + optional LLM cleanup.
Types transcribed text into any active application.

Usage:
    python -m voiceflow              # Start with default hotkey (Ctrl+Alt+Space)
    python -m voiceflow --hotkey f9  # Custom hotkey
    python -m voiceflow --model tiny # Smaller, faster model
    python -m voiceflow --list-devices  # Show audio devices and exit

Press Ctrl+Alt+Space (or your chosen hotkey) to start recording.
Release to stop. Transcribed text appears in the active window.
"""

import argparse
import logging
import sys
import threading
import time
import numpy as np

from voiceflow.audio import AudioRecorder, AudioConfig
from voiceflow.transcriber import Transcriber, TranscriptionConfig
from voiceflow.llm_postprocessor import LLMPostProcessor, LLMConfig
from voiceflow.typer import Typer
from voiceflow.tray import TrayApp, AppState
from voiceflow.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("voiceflow")


class VoiceFlow:
    """Main application class. Ties together all components."""

    def __init__(self, args):
        self.args = args
        self.cfg = load_config()

        # Override config with CLI args
        model_size = args.model or self.cfg.get("model_size", "base")
        device = args.device or self.cfg.get("device", "auto")
        language = args.language or self.cfg.get("language")

        # Initialize components
        self.audio_config = AudioConfig(
            max_duration=self.cfg.get("max_recording_seconds", 120.0),
            silence_threshold=self.cfg.get("silence_threshold", 0.01),
        )
        self.trans_config = TranscriptionConfig(
            model_size=model_size,
            device=device,
            language=language,
        )
        self.llm_config = LLMConfig(**(self.cfg.get("llm", {})))

        self.recorder = AudioRecorder(self.audio_config)
        self.transcriber = Transcriber(self.trans_config)
        self.llm_processor = LLMPostProcessor(self.llm_config)
        self.typer = Typer(delay=self.cfg.get("typing_delay", 0.005))

        # State
        self._running = False
        self._recording = False
        self._tray = None

    def start(self):
        """Start VoiceFlow: preload model, start tray, listen for hotkeys."""
        logger.info("VoiceFlow starting...")
        logger.info(
            "Hotkey: hold to record, release to transcribe. Ctrl+C to exit."
        )

        # Preload model in background (non-blocking)
        def _preload():
            try:
                _ = self.transcriber.model
                logger.info("Model preloaded and ready")
            except Exception as e:
                logger.error("Failed to preload model: %s", e)

        threading.Thread(target=_preload, daemon=True).start()

        # Start tray (non-blocking)
        self._tray = TrayApp(
            on_quit=self._on_quit,
            on_toggle_llm=self._on_toggle_llm,
        )
        threading.Thread(target=self._tray.run, daemon=True).start()

        # Start hotkey listener (blocking)
        self._running = True
        self._listen_hotkeys()

    def _listen_hotkeys(self):
        """
        Listen for global hotkey press/release.

        Default: Ctrl+Alt+Space (press = start recording, release = stop + transcribe).
        """
        from pynput import keyboard

        hotkey = self.args.hotkey or self.cfg.get("hotkey", "ctrl+alt+space")
        keys = self._parse_hotkey(hotkey)

        logger.info("Listening for hotkey: %s", hotkey)

        current_keys = set()
        required_keys = set(keys)

        def on_press(key):
            current_keys.add(key)
            if not self._recording and required_keys.issubset(current_keys):
                self._start_recording()

        def on_release(key):
            current_keys.discard(key)
            if self._recording and not required_keys.issubset(current_keys):
                self._stop_and_transcribe()

        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
        except Exception as e:
            logger.error("Hotkey listener error: %s", e)
            logger.info("Tip: On Linux, you may need to run with sudo or add input group permissions")

    def _start_recording(self):
        """Hotkey pressed -- begin recording."""
        self._recording = True
        self.recorder.start()
        logger.info("Recording started... (release hotkey to stop)")
        if self._tray:
            self._tray.update_state(AppState.RECORDING)

    def _stop_and_transcribe(self):
        """Hotkey released -- stop recording and process."""
        self._recording = False
        logger.info("Recording stopped, processing...")

        if self._tray:
            self._tray.update_state(AppState.PROCESSING)

        audio = self.recorder.stop()
        if len(audio) == 0:
            logger.warning("No audio captured (silence or too short)")
            if self._tray:
                self._tray.update_state(AppState.IDLE)
            return

        # Process in background thread so hotkey listener stays responsive
        threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()

    def _process_audio(self, audio: np.ndarray):
        """Transcribe and type in a background thread."""
        try:
            text = self.transcriber.transcribe(audio)
            if text:
                # Optional LLM cleanup
                if self.llm_config.enabled and self.llm_config.api_key:
                    text = self.llm_processor.process(text)
                logger.info("Result: %s", text)
                self.typer.type_text(text)
            else:
                logger.info("No speech detected")
        except Exception as e:
            logger.error("Processing error: %s", e)
        finally:
            if self._tray:
                self._tray.update_state(AppState.IDLE)

    def _on_quit(self):
        """Callback from tray menu quit."""
        logger.info("Shutting down...")
        self._running = False

    def _on_toggle_llm(self):
        """Callback from tray menu LLM toggle."""
        self.llm_config.enabled = not self.llm_config.enabled
        self._tray.update_llm_status(self.llm_config.enabled)
        logger.info("LLM cleanup: %s", "ON" if self.llm_config.enabled else "OFF")

    @staticmethod
    def _parse_hotkey(hotkey_str: str) -> list:
        """Parse a hotkey string like 'ctrl+alt+space' into pynput Key objects."""
        from pynput.keyboard import Key, KeyCode

        parts = hotkey_str.lower().strip().split("+")
        keys = []

        key_map = {
            "ctrl": Key.ctrl_l,
            "ctrl_l": Key.ctrl_l,
            "ctrl_r": Key.ctrl_r,
            "alt": Key.alt_l,
            "alt_l": Key.alt_l,
            "alt_r": Key.alt_r,
            "shift": Key.shift_l,
            "shift_l": Key.shift_l,
            "shift_r": Key.shift_r,
            "space": Key.space,
            "tab": Key.tab,
            "enter": Key.enter,
            "return": Key.enter,
            "esc": Key.esc,
            "delete": Key.delete,
            "backspace": Key.backspace,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "home": Key.home,
            "end": Key.end,
            "page_up": Key.page_up,
            "page_down": Key.page_down,
        }

        # Function keys
        for i in range(1, 13):
            key_map[f"f{i}"] = getattr(Key, f"f{i}")

        for part in parts:
            part = part.strip()
            if part in key_map:
                keys.append(key_map[part])
            elif len(part) == 1:
                # Single character (letter or digit)
                keys.append(KeyCode.from_char(part))
            else:
                raise ValueError(f"Unknown hotkey part: '{part}' in '{hotkey_str}'")

        return keys


def main():
    parser = argparse.ArgumentParser(
        description="VoiceFlow - Free, open-source voice dictation"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (en, hi, etc). Auto-detect if not set",
    )
    parser.add_argument(
        "--hotkey",
        default=None,
        help="Hotkey combination, e.g. 'ctrl+alt+space' (default from config)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--test-record",
        action="store_true",
        help="Record 5 seconds and transcribe (quick test, no hotkey needed)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    args = parser.parse_args()

    if args.version:
        print("VoiceFlow v0.1.0 -- Alpha")
        return

    if args.list_devices:
        rec = AudioRecorder()
        devices = rec.list_devices()
        print("Audio input devices:")
        for d in devices:
            print(f"  [{d['index']}] {d['name']} ({d['channels']}ch, {d['sample_rate']}Hz)")
        return

    if args.test_record:
        _run_test_record(args)
        return

    # Start the main app
    app = VoiceFlow(args)
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("VoiceFlow stopped (Ctrl+C)")


def _run_test_record(args):
    """Record from the microphone for a few seconds and transcribe."""
    print("Recording for 5 seconds...")
    rec = AudioRecorder()
    rec.start()
    time.sleep(5)
    audio = rec.stop()

    if len(audio) == 0:
        print("No audio captured. Check your microphone.")
        return

    config = TranscriptionConfig(
        model_size=args.model or "base",
        device=args.device or "auto",
        language=args.language,
    )
    transcriber = Transcriber(config)
    text = transcriber.transcribe(audio)

    print(f"\nTranscribed ({len(audio) / 16000:.1f}s audio):")
    print(f"  {text}")


if __name__ == "__main__":
    main()
