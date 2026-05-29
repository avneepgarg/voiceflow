"""Configuration system - persistent JSON-based settings.

All settings stored in ~/.voiceflow/config.json (Windows: %APPDATA%\\VoiceFlow\\config.json).
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default config values
DEFAULT_CONFIG = {
    "hotkey": "ctrl+alt+space",
    "audio_device_index": None,
    "model_size": "base",
    "language": None,
    "device": "auto",
    "auto_punctuation": True,
    "type_method": "keystroke",
    "show_notifications": True,
    "max_recording_seconds": 120.0,
    "silence_threshold": 0.01,
    "typing_delay": 0.005,
    "llm": {
        "enabled": False,
        "provider": "openai",
        "api_key": "",
        "base_url": "",
        "model": "gpt-4o-mini",
        "max_tokens": 512,
        "remove_fillers": True,
        "fix_grammar": True,
        "add_punctuation": True,
        "reformat": "none",
    },
    "profiles": {},
    "vocabulary": [],
}


def get_config_dir() -> Path:
    """Return the platform-appropriate config directory."""
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.uname().sysname == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    config_dir = base / "VoiceFlow"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Return the full path to the config file."""
    return get_config_dir() / "config.json"


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from file. Creates default config if none exists.

    Args:
        config_path: Optional override path. Uses platform default if not set.

    Returns:
        Configuration dictionary (deep-merged with defaults).
    """
    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        logger.info("No config file found, creating default at %s", config_path)
        config = dict(DEFAULT_CONFIG)
        save_config(config, config_path)
        return config

    try:
        with open(config_path, "r") as f:
            user_config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load config (%s), using defaults", e)
        return dict(DEFAULT_CONFIG)

    # Deep merge: user values override defaults
    config = _deep_merge(dict(DEFAULT_CONFIG), user_config)
    logger.debug("Config loaded from %s", config_path)
    return config


def save_config(config: Dict[str, Any], config_path: Path = None):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
        config_path: Optional override path
    """
    if config_path is None:
        config_path = get_config_path()

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.debug("Config saved to %s", config_path)
    except IOError as e:
        logger.error("Failed to save config: %s", e)
        raise


def reload_config(config_path: Path = None) -> Dict[str, Any]:
    """Reload config from disk (useful after external edits)."""
    if config_path is None:
        config_path = get_config_path()
    return load_config(config_path)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base. Override wins."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
