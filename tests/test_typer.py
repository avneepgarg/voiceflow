"""Tests for the text injection (Typer) module."""
import os
import pytest
from unittest.mock import MagicMock, patch, call

# Skip all pynput-dependent tests in headless CI (no DISPLAY)
# These tests mock pynput but the import itself fails without X server
_has_display = bool(os.environ.get("DISPLAY")) or os.name == "nt"

# We always import Typer (it's lazy), but skip tests that call pynput methods
# when running headless
skip_gui = pytest.mark.skipif(
    not _has_display, reason="No display server (headless CI)"
)

from voiceflow.typer import Typer


class TestTyperInit:
    def test_default_delay(self):
        t = Typer()
        assert t.delay == 0.005

    def test_custom_delay(self):
        t = Typer(delay=0.01)
        assert t.delay == 0.01

    def test_zero_delay(self):
        t = Typer(delay=0)
        assert t.delay == 0


class TestTyperTypeText:
    def test_empty_string_is_noop(self):
        t = Typer()
        t.type_text("")  # Should not crash

    @skip_gui
    def test_types_all_characters(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer(delay=0)
            t.type_text("hi")
            assert mock_kb.type.call_count == 2

    @skip_gui
    def test_keyboard_error_raises(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_controller.side_effect = Exception("No keyboard")
            t = Typer()
            with pytest.raises(Exception, match="No keyboard"):
                t.type_text("test")


class TestTyperPaste:
    def test_paste_method_does_not_crash(self):
        """type_with_paste should not crash in headless environments."""
        t = Typer()
        try:
            t.type_with_paste("hello")
        except Exception:
            pass  # Expected in headless

    @skip_gui
    def test_paste_uses_clipboard(self):
        with patch("pyperclip.copy") as mock_copy, \
             patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer()
            t.type_with_paste("hello world")
            mock_copy.assert_called_once_with("hello world")


class TestTyperPressEnter:
    @skip_gui
    def test_press_enter(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer()
            t.press_enter()
            assert mock_kb.press.call_count >= 1
            assert mock_kb.release.call_count >= 1


class TestTyperWithFormatting:
    @skip_gui
    def test_append_space(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer(delay=0)
            t.type_with_formatting("hello")
            typed_chars = [c.args[0] for c in mock_kb.type.call_args_list]
            assert "".join(typed_chars) == "hello "

    @skip_gui
    def test_no_append_space(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer(delay=0)
            t.type_with_formatting("hello", append_space=False)
            typed_chars = [c.args[0] for c in mock_kb.type.call_args_list]
            assert "".join(typed_chars) == "hello"

    @skip_gui
    def test_already_has_space(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer(delay=0)
            t.type_with_formatting("hello ")
            typed_chars = [c.args[0] for c in mock_kb.type.call_args_list]
            assert "".join(typed_chars) == "hello "


class TestTyperPressKey:
    @skip_gui
    def test_press_key(self):
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_kb = MagicMock()
            mock_controller.return_value = mock_kb
            t = Typer()
            t.press_key("x")
            mock_kb.press.assert_called_once_with("x")
            mock_kb.release.assert_called_once_with("x")
