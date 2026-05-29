"""Voice command processor - interprets spoken commands as editing actions.

Commands are patterns matched against transcribed text. If a match is found,
the command is executed instead of (or after) typing the text.

Supported commands:
- "new line" / "new paragraph"  -- press Enter
- "delete last word"             -- delete previous word
- "delete that"                  -- delete last utterance
- "caps on" / "caps off"         -- toggle caps lock
- "cap next"                     -- capitalize next word
- "all caps <text> end caps"     -- type text in ALL CAPS
- "period" / "comma" / "question mark" -- insert punctuation
- "undo that"                    -- Ctrl+Z
- "scratch that"                 -- delete everything typed in this session
- "select all"                   -- Ctrl+A
- "copy that"                    -- Ctrl+C
- "paste that"                   -- Ctrl+V
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VoiceCommand:
    """A recognized voice command."""
    action: str           # The action to execute
    args: Tuple = ()      # Optional arguments
    raw_text: str = ""    # The original spoken text


class VoiceCommandProcessor:
    """
    Processes transcribed text to detect and execute voice commands.

    Usage:
        processor = VoiceCommandProcessor()
        command, remaining_text = processor.process("hello world delete last word")
        if command:
            executor.execute(command)
    """

    # Command patterns: (pattern, action_name, args_extractor)
    # Patterns are matched case-insensitively at the END of the text
    COMMANDS = [
        # Navigation & editing
        (r"\bnew line\b", "press_key", ("enter",)),
        (r"\bnew paragraph\b", "press_key_twice", ("enter",)),
        (r"\bdelete last word\b", "delete_words", (1,)),
        (r"\bdelete last two words\b", "delete_words", (2,)),
        (r"\bdelete last sentence\b", "delete_sentence", ()),
        (r"\bdelete that\b", "delete_last", ()),
        (r"\bscratch that\b", "delete_all_session", ()),
        (r"\bundo that\b", "undo", ()),

        # Capitalization
        (r"\bcaps on\b", "caps_on", ()),
        (r"\bcaps off\b", "caps_off", ()),
        (r"\bcap next\b", "cap_next", ()),
        (r"\ball caps (.+?) end caps\b", "type_caps", ("{1}",)),

        # Punctuation
        (r"\bperiod\b", "insert_text", (".",)),
        (r"\bfull stop\b", "insert_text", (".",)),
        (r"\bcomma\b", "insert_text", (",",)),
        (r"\bquestion mark\b", "insert_text", ("?",)),
        (r"\bexclamation mark\b", "insert_text", ("!",)),
        (r"\bsemicolon\b", "insert_text", (";",)),
        (r"\bcolon\b", "insert_text", (":",)),
        (r"\bopen paren\b", "insert_text", ("(",)),
        (r"\bclose paren\b", "insert_text", (")",)),
        (r"\bdash\b", "insert_text", ("-",)),
        (r"\bquote\b", "insert_text", ('"',)),
        (r"\bnewline\b", "press_key", ("enter",)),

        # Selection & clipboard
        (r"\bselect all\b", "select_all", ()),
        (r"\bcopy that\b", "copy", ()),
        (r"\bpaste that\b", "paste", ()),
        (r"\bcut that\b", "cut", ()),
    ]

    def __init__(self):
        self._caps_mode = False
        self._typed_this_session = []
        self._compiled_commands = self._compile_commands()

    def process(self, text: str) -> Tuple[Optional[VoiceCommand], str]:
        """
        Check if the transcribed text ends with a voice command.

        Args:
            text: Raw transcribed text from Whisper

        Returns:
            Tuple of (command_or_none, remaining_text_to_type)
        """
        import re

        text_lower = text.strip().lower()

        for pattern, action, args in self._compiled_commands:
            match = re.search(pattern, text_lower)
            if match:
                # Extract the text before the command
                remaining = text[: match.start()].strip()

                # Build command args (handle capture groups)
                resolved_args = []
                for arg in args:
                    arg_str = str(arg)
                    if arg_str.startswith("{") and arg_str.endswith("}"):
                        group_num = int(arg_str[1:-1])
                        resolved_args.append(match.group(group_num))
                    else:
                        resolved_args.append(arg)

                command = VoiceCommand(
                    action=action,
                    args=tuple(resolved_args),
                    raw_text=text[match.start(): match.end()],
                )

                logger.debug("Voice command detected: %s %s", action, resolved_args)
                return command, remaining

        return None, text

    def execute(self, command: VoiceCommand, typer) -> bool:
        """
        Execute a voice command using the provided Typer instance.

        Args:
            command: The VoiceCommand to execute
            typer: A Typer instance for text operations

        Returns:
            True if command was executed successfully
        """
        success = True

        try:
            if command.action == "press_key":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                key = self._resolve_key(command.args[0])
                keyboard.press(key)
                keyboard.release(key)

            elif command.action == "press_key_twice":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                key = self._resolve_key(command.args[0])
                keyboard.press(key)
                keyboard.release(key)
                time.sleep(0.05)
                keyboard.press(key)
                keyboard.release(key)

            elif command.action == "delete_words":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                n = command.args[0]
                for _ in range(n):
                    with keyboard.pressed(Key.shift, Key.ctrl):
                        keyboard.press(Key.left)
                        keyboard.release(Key.left)
                    keyboard.press(Key.delete)
                    keyboard.release(Key.delete)

            elif command.action == "delete_sentence":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                with keyboard.pressed(Key.ctrl):
                    keyboard.press(Key.shift)
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                    keyboard.release(Key.shift)
                keyboard.press(Key.delete)
                keyboard.release(Key.delete)

            elif command.action == "delete_last":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                # Delete last "utterance" -- approximate by selecting last 100 chars
                with keyboard.pressed(Key.shift, Key.ctrl):
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)

            elif command.action == "delete_all_session":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                for _ in self._typed_this_session:
                    keyboard.press(Key.backspace)
                    keyboard.release(Key.backspace)
                self._typed_this_session.clear()

            elif command.action == "undo":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                with keyboard.pressed(Key.ctrl):
                    keyboard.press("z")
                    keyboard.release("z")

            elif command.action == "caps_on":
                self._caps_mode = True

            elif command.action == "caps_off":
                self._caps_mode = False

            elif command.action == "cap_next":
                pass  # Handled during text typing

            elif command.action == "type_caps":
                text = command.args[0].upper()
                typer.type_text(text + " ")

            elif command.action == "insert_text":
                typer.type_text(command.args[0])

            elif command.action == "select_all":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                with keyboard.pressed(Key.ctrl):
                    keyboard.press("a")
                    keyboard.release("a")

            elif command.action == "copy":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                with keyboard.pressed(Key.ctrl):
                    keyboard.press("c")
                    keyboard.release("c")

            elif command.action == "paste":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                with keyboard.pressed(Key.ctrl):
                    keyboard.press("v")
                    keyboard.release("v")

            elif command.action == "cut":
                from pynput.keyboard import Controller, Key
                keyboard = Controller()
                with keyboard.pressed(Key.ctrl):
                    keyboard.press("x")
                    keyboard.release("x")

            else:
                logger.warning("Unknown command action: %s", command.action)
                success = False

        except Exception as e:
            logger.error("Failed to execute voice command %s: %s", command.action, e)
            success = False

        return success

    @property
    def caps_mode(self) -> bool:
        return self._caps_mode

    def track_typed(self, text: str):
        """Track text typed in this session (for 'scratch that' command)."""
        self._typed_this_session.append(text)

    def reset_session(self):
        """Clear the session tracking."""
        self._typed_this_session.clear()

    def _compile_commands(self):
        """Compile command patterns into regex objects."""
        import re
        compiled = []
        for pattern, action, args in self.COMMANDS:
            compiled.append((re.compile(pattern, re.IGNORECASE), action, args))
        return compiled

    @staticmethod
    def _resolve_key(key_str: str):
        """Convert a key name string to a pynput Key or character."""
        try:
            from pynput.keyboard import Key
        except ImportError:
            return key_str

        key_map = {
            "enter": Key.enter,
            "tab": Key.tab,
            "space": Key.space,
            "escape": Key.esc,
            "delete": Key.delete,
            "backspace": Key.backspace,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
        }
        return key_map.get(key_str.lower(), key_str)
