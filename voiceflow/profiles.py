"""Per-app profiles - automatically switch settings based on active application.

Unlike Wispr Flow and Monologue, this does NOT take screenshots.
Uses only window title and executable name (privacy-first).

Each profile can override: model_size, language, llm_reformat, cleanup rules.
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# Built-in profiles for common apps
BUILTIN_PROFILES = {
    # Email
    "outlook": {
        "name": "Email (Outlook)",
        "match_exe": ["outlook.exe", "msimn.exe"],
        "window_contains": [],
        "llm_reformat": "formal",
        "remove_fillers": True,
        "fix_grammar": True,
        "add_punctuation": True,
        "auto_capitalize": True,
    },
    "gmail": {
        "name": "Email (Gmail)",
        "match_exe": [],
        "window_contains": ["gmail", "mail.google.com"],
        "llm_reformat": "formal",
        "remove_fillers": True,
        "fix_grammar": True,
        "add_punctuation": True,
        "auto_capitalize": True,
    },
    # Messaging
    "slack": {
        "name": "Slack",
        "match_exe": ["slack.exe"],
        "window_contains": ["slack"],
        "llm_reformat": "concise",
        "remove_fillers": True,
        "fix_grammar": False,
        "add_punctuation": False,
        "auto_capitalize": False,
    },
    "teams": {
        "name": "Microsoft Teams",
        "match_exe": ["teams.exe"],
        "window_contains": ["teams"],
        "llm_reformat": "concise",
        "remove_fillers": True,
        "fix_grammar": False,
        "add_punctuation": False,
        "auto_capitalize": False,
    },
    # Code editors
    "vscode": {
        "name": "VS Code / Cursor",
        "match_exe": ["code.exe", "cursor.exe"],
        "window_contains": ["visual studio code", "cursor"],
        "llm_reformat": "none",
        "remove_fillers": False,
        "fix_grammar": False,
        "add_punctuation": False,
        "auto_capitalize": False,
    },
    # Writing
    "notepad": {
        "name": "Notepad / Text Editor",
        "match_exe": ["notepad.exe", "notepad++.exe"],
        "window_contains": ["notepad", "text", "editor"],
        "llm_reformat": "paragraphs",
        "remove_fillers": True,
        "fix_grammar": True,
        "add_punctuation": True,
        "auto_capitalize": True,
    },
    # Web browsers
    "chrome": {
        "name": "Web Browser",
        "match_exe": ["chrome.exe", "firefox.exe", "brave.exe", "msedge.exe"],
        "window_contains": [],
        "llm_reformat": "none",
        "remove_fillers": True,
        "fix_grammar": True,
        "add_punctuation": True,
        "auto_capitalize": True,
    },
}


@dataclass
class AppProfile:
    """Settings override for a specific application."""
    name: str = "Default"
    match_exe: list = field(default_factory=list)        # Executable names to match
    window_contains: list = field(default_factory=list)  # Window title substrings

    # Overrides
    llm_reformat: str = "none"
    remove_fillers: bool = True
    fix_grammar: bool = True
    add_punctuation: bool = True
    auto_capitalize: bool = True
    model_size: str = None        # None = use global setting
    language: str = None          # None = use global setting

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ProfileManager:
    """
    Manages per-app profiles. Detects the active window and applies the
    matching profile.

    Usage:
        pm = ProfileManager()
        pm.load_profiles()
        profile = pm.get_active_profile()
        if profile:
            print(f"Active: {profile.name}")
    """

    def __init__(self, profiles_path: Path = None):
        self._profiles: Dict[str, AppProfile] = {}
        self._profiles_path = profiles_path
        self._current_profile: Optional[AppProfile] = None

    def load_profiles(self):
        """Load built-in + user-defined profiles."""
        # Start with built-ins
        for key, profile_data in BUILTIN_PROFILES.items():
            self._profiles[key] = AppProfile(**profile_data)

        # Load user overrides from config directory
        if self._profiles_path is None:
            from voiceflow.config import get_config_dir
            self._profiles_path = get_config_dir() / "profiles.json"

        if self._profiles_path.exists():
            try:
                with open(self._profiles_path, "r") as f:
                    user_profiles = json.load(f)
                for key, profile_data in user_profiles.items():
                    self._profiles[key] = AppProfile.from_dict(profile_data)
                logger.info("Loaded %d user profiles", len(user_profiles))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load user profiles: %s", e)

        logger.info("Total profiles loaded: %d", len(self._profiles))

    def save_profiles(self):
        """Save user-defined profiles to disk."""
        if self._profiles_path is None:
            return

        # Only save profiles that aren't built-in
        builtin_keys = set(BUILTIN_PROFILES.keys())
        user_profiles = {
            k: v.to_dict()
            for k, v in self._profiles.items()
            if k not in builtin_keys
        }

        try:
            with open(self._profiles_path, "w") as f:
                json.dump(user_profiles, f, indent=2)
        except IOError as e:
            logger.error("Failed to save profiles: %s", e)

    def get_active_profile(self) -> Optional[AppProfile]:
        """
        Detect the currently active window and return its matching profile.

        Returns:
            AppProfile if a match is found, None otherwise.
        """
        try:
            exe, title = self._get_active_window_info()
            if not exe and not title:
                return None

            exe_lower = (exe or "").lower()
            title_lower = (title or "").lower()

            for key, profile in self._profiles.items():
                # Match by executable name
                if profile.match_exe:
                    for exe_pattern in profile.match_exe:
                        if exe_pattern.lower() in exe_lower:
                            self._current_profile = profile
                            return profile

                # Match by window title
                if profile.window_contains:
                    for title_pattern in profile.window_contains:
                        if title_pattern.lower() in title_lower:
                            self._current_profile = profile
                            return profile

            return None

        except Exception as e:
            logger.debug("Failed to detect active window: %s", e)
            return None

    @property
    def current_profile_name(self) -> str:
        if self._current_profile:
            return self._current_profile.name
        return "Default"

    @staticmethod
    def _get_active_window_info() -> tuple:
        """
        Get the active window's executable name and title.

        Returns:
            Tuple of (exe_name_or_None, title_or_None)
        """
        system = os.name

        if system == "nt":  # Windows
            try:
                import ctypes
                import ctypes.wintypes

                user32 = ctypes.windll.user32

                # Get foreground window
                hwnd = user32.GetForegroundWindow()
                if not hwnd:
                    return None, None

                # Get window title
                length = user32.GetWindowTextLengthW(hwnd)
                if length == 0:
                    title = ""
                else:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value

                # Get process name
                pid = ctypes.wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                # Access process to get exe name
                import ctypes.wintypes
                PROCESS_QUERY_INFORMATION = 0x0400
                PROCESS_VM_READ = 0x0010
                h_process = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid.value
                )
                if h_process:
                    buf = ctypes.create_unicode_buffer(260)
                    ctypes.windll.psapi.GetModuleBaseNameW(h_process, None, buf, 260)
                    exe = buf.value
                    ctypes.windll.kernel32.CloseHandle(h_process)
                else:
                    exe = None

                return exe, title

            except Exception:
                return None, None

        else:  # macOS, Linux -- for now return None
            # Future: use AppleScript (macOS) or xdotool/ewmh (Linux)
            logger.debug("Active window detection not implemented for this platform")
            return None, None
