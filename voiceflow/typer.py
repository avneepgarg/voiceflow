"""Text injection - types transcribed text into the active window.

Works on Windows, macOS, and Linux via pynput keyboard controller.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Typer:
    """
    Injects text into the currently focused application.

    Usage:
        typer = Typer()
        typer.type_text("Hello world")
        typer.press_enter()
    """

    def __init__(self, delay: float = 0.005):
        """
        Args:
            delay: Seconds between keystrokes.
                   0.005 = fast but reliable on most apps.
                   Increase to 0.01 if text gets garbled in some apps.
        """
        self.delay = delay

    def type_text(self, text: str):
        """
        Type text as if the user typed it on the keyboard.
        Works in any application: Notepad, VS Code, Chrome, Slack, etc.

        Args:
            text: The text to type. Empty string is a no-op.
        """
        if not text:
            return

        try:
            from pynput.keyboard import Controller

            keyboard = Controller()

            for char in text:
                keyboard.type(char)
                if self.delay > 0:
                    time.sleep(self.delay)

            logger.debug("Typed %d characters", len(text))

        except Exception as e:
            logger.error("Failed to type text: %s", e)
            raise

    def type_with_paste(self, text: str):
        """
        Paste text via clipboard (Ctrl+V).
        Faster for long text, but doesn't work in all apps (terminals, games).

        Args:
            text: The text to paste. Empty string is a no-op.
        """
        if not text:
            return

        try:
            import pyperclip
            from pynput.keyboard import Controller, Key

            pyperclip.copy(text)
            time.sleep(0.05)

            keyboard = Controller()
            with keyboard.pressed(Key.ctrl):
                keyboard.press("v")
                keyboard.release("v")

            logger.debug("Pasted %d characters via clipboard", len(text))

        except Exception as e:
            logger.error("Failed to paste text: %s", e)
            # Fallback to keystroke method
            logger.info("Falling back to keystroke typing")
            self.type_text(text)

    def press_enter(self):
        """Press the Enter key."""
        from pynput.keyboard import Controller, Key

        keyboard = Controller()
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)

    def press_key(self, key):
        """
        Press a single key.

        Args:
            key: pynput Key enum or string character
        """
        from pynput.keyboard import Controller

        keyboard = Controller()
        keyboard.press(key)
        keyboard.release(key)

    def type_with_formatting(self, text: str, append_space: bool = True):
        """
        Type text with optional trailing space (convenience method).

        Args:
            text: Text to type
            append_space: Whether to add a space after the text
        """
        if append_space and text and not text.endswith(" "):
            text += " "
        self.type_text(text)
