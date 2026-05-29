"""Tests for per-app profiles."""

import pytest
from voiceflow.profiles import AppProfile, ProfileManager, BUILTIN_PROFILES


class TestAppProfile:
    def test_default(self):
        profile = AppProfile()
        assert profile.name == "Default"
        assert profile.llm_reformat == "none"
        assert profile.remove_fillers is True

    def test_to_dict(self):
        profile = AppProfile(name="Test", match_exe=["test.exe"])
        d = profile.to_dict()
        assert d["name"] == "Test"
        assert "test.exe" in d["match_exe"]

    def test_from_dict(self):
        d = {
            "name": "Email",
            "match_exe": ["outlook.exe"],
            "window_contains": ["gmail"],
            "llm_reformat": "formal",
            "remove_fillers": True,
            "fix_grammar": True,
            "add_punctuation": True,
            "auto_capitalize": True,
        }
        profile = AppProfile.from_dict(d)
        assert profile.name == "Email"
        assert profile.llm_reformat == "formal"


class TestProfileManager:
    def test_load_builtin_profiles(self):
        pm = ProfileManager(profiles_path=None)
        pm.load_profiles()
        assert len(pm._profiles) >= len(BUILTIN_PROFILES)

    def test_builtin_profile_keys(self):
        pm = ProfileManager(profiles_path=None)
        pm.load_profiles()
        # Check key builtins exist
        for key in ["outlook", "gmail", "slack", "vscode", "chrome"]:
            assert key in pm._profiles, f"Missing built-in profile: {key}"

    def test_active_profile_default(self):
        pm = ProfileManager(profiles_path=None)
        pm.load_profiles()
        # On test system, may not detect window -- just verify no crash
        profile = pm.get_active_profile()
        # Should return None or a profile, no exception
        assert profile is None or isinstance(profile, AppProfile)

    def test_current_profile_name_default(self):
        pm = ProfileManager(profiles_path=None)
        assert pm.current_profile_name == "Default"
