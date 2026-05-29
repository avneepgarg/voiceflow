"""Shared utility functions for VoiceFlow."""

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_platform() -> str:
    """Return the current platform: 'windows', 'macos', or 'linux'."""
    system = platform.system().lower()
    if system == "windows" or system == "nt":
        return "windows"
    elif system == "darwin":
        return "macos"
    return "linux"


def is_windows() -> bool:
    return get_platform() == "windows"


def is_macos() -> bool:
    return get_platform() == "macos"


def is_linux() -> bool:
    return get_platform() == "linux"


def get_app_dir() -> Path:
    """Return the platform-appropriate application data directory."""
    p = get_platform()
    if p == "windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif p == "macos":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    app_dir = base / "VoiceFlow"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_models_dir() -> Path:
    """Return the directory where Whisper models are cached."""
    models_dir = get_app_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def detect_cuda() -> dict:
    """Detect CUDA availability and return device info."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "device_index": 0,
                "memory_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024 * 1024),
            }
    except ImportError:
        pass
    return {"available": False}


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"


def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def open_file_explorer(path: Path):
    """Open the system file explorer at the given path."""
    p = get_platform()
    try:
        if p == "windows":
            subprocess.Popen(["explorer", str(path)])
        elif p == "macos":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        logger.warning("Failed to open file explorer: %s", e)


def ensure_microphone_permission():
    """Check/troubleshoot microphone access."""
    p = get_platform()
    if p == "linux":
        # Check if user is in 'input' and 'audio' groups
        import grp
        try:
            audio_gid = grp.getgrnam("audio").gr_gid
            input_gid = grp.getgrnam("input").gr_gid
            os_gid = os.getgroups()
            missing = []
            if audio_gid not in os_gid:
                missing.append("audio")
            if input_gid not in os_gid:
                missing.append("input")
            if missing:
                logger.warning(
                    "User not in groups: %s. Run: sudo usermod -aG %s %s",
                    ", ".join(missing),
                    ",".join(missing),
                    os.getlogin(),
                )
        except (KeyError, OSError):
            pass
