"""Tests for the configuration module."""

import json
import tempfile
from pathlib import Path
from voiceflow.config import (
    load_config, save_config, get_config_path, get_config_dir,
    reload_config, _deep_merge, DEFAULT_CONFIG,
)


class TestGetConfigPath:
    def test_returns_path(self):
        p = get_config_path()
        assert isinstance(p, Path)
        assert p.name == "config.json"

    def test_config_dir_created(self):
        # Calling get_config_dir should create the directory
        d = get_config_dir()
        assert d.exists()


class TestLoadConfig:
    def test_creates_default_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            cfg = load_config(p)
            assert "hotkey" in cfg
            assert "model_size" in cfg
            assert p.exists()  # Should have created the file

    def test_loads_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            custom = {"hotkey": "f9", "model_size": "tiny"}
            with open(p, "w") as f:
                json.dump(custom, f)
            cfg = load_config(p)
            assert cfg["hotkey"] == "f9"
            assert cfg["model_size"] == "tiny"

    def test_fills_in_missing_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            with open(p, "w") as f:
                json.dump({"hotkey": "f12"}, f)
            cfg = load_config(p)
            assert cfg["hotkey"] == "f12"
            assert "model_size" in cfg  # Should be filled from defaults
            assert "llm" in cfg  # Nested defaults preserved

    def test_handles_corrupt_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            with open(p, "w") as f:
                f.write("not valid json {{{")
            cfg = load_config(p)
            assert "hotkey" in cfg  # Falls back to defaults

    def test_deep_merge_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            user_cfg = {"llm": {"enabled": True, "api_key": "sk-test"}}
            with open(p, "w") as f:
                json.dump(user_cfg, f)
            cfg = load_config(p)
            assert cfg["llm"]["enabled"] is True
            assert cfg["llm"]["api_key"] == "sk-test"
            # Other llm defaults should still be there
            assert "remove_fillers" in cfg["llm"]


class TestSaveConfig:
    def test_saves_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            cfg = {"test": "value", "number": 42}
            save_config(cfg, p)
            with open(p) as f:
                loaded = json.load(f)
            assert loaded["test"] == "value"
            assert loaded["number"] == 42


class TestDeepMerge:
    def test_basic_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"llm": {"enabled": False, "api_key": "", "model": "gpt-4o-mini"}}
        override = {"llm": {"enabled": True}}
        result = _deep_merge(base, override)
        assert result["llm"]["enabled"] is True
        assert result["llm"]["model"] == "gpt-4o-mini"  # Preserved
        assert result["llm"]["api_key"] == ""  # Preserved

    def test_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2, "c": 3}


class TestReloadConfig:
    def test_reloads_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.json"
            save_config({"hotkey": "ctrl+shift+a"}, p)
            cfg = reload_config(p)
            assert cfg["hotkey"] == "ctrl+shift+a"
