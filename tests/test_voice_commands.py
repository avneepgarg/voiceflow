"""Tests for the voice command processor."""

import pytest
from voiceflow.voice_commands import VoiceCommandProcessor, VoiceCommand


class TestVoiceCommandProcessor:
    def test_no_command(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello world this is a test")
        assert cmd is None
        assert text == "hello world this is a test"

    def test_new_line_command(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello world new line")
        assert cmd is not None
        assert cmd.action == "press_key"
        assert text == "hello world"

    def test_delete_last_word(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello world delete last word")
        assert cmd is not None
        assert cmd.action == "delete_words"
        assert text == "hello world"

    def test_period_command(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello world period")
        assert cmd is not None
        assert cmd.action == "insert_text"
        assert cmd.args[0] == "."
        assert text == "hello world"

    def test_comma_command(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello comma world")
        assert cmd is not None
        assert cmd.action == "insert_text"
        assert cmd.args[0] == ","

    def test_caps_on(self):
        proc = VoiceCommandProcessor()
        proc.execute(VoiceCommand("caps_on"), None)
        assert proc.caps_mode is True

    def test_caps_off(self):
        proc = VoiceCommandProcessor()
        proc._caps_mode = True
        proc.execute(VoiceCommand("caps_off"), None)
        assert proc.caps_mode is False

    def test_question_mark(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("what time is it question mark")
        assert cmd is not None
        assert cmd.args[0] == "?"

    def test_type_caps(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("all caps important end caps")
        assert cmd is not None
        assert cmd.action == "type_caps"
        assert cmd.args[0] == "important"

    def test_scratch_that(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("hello scratch that")
        assert cmd is not None
        assert cmd.action == "delete_all_session"
        assert text == "hello"

    def test_select_all(self):
        proc = VoiceCommandProcessor()
        cmd, text = proc.process("select all")
        assert cmd is not None
        assert cmd.action == "select_all"
        assert text == ""
