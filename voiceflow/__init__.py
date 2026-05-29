"""VoiceFlow - Transcription pipeline modules.

Safe imports (no heavy deps): audio, transcriber, typer, config, profiles, vocabulary,
LLM post-processor, voice commands, agent mode, noise gate, wake word.

Modules with optional deps (sounddevice, etc.): scribe, translation, tray.
Import those directly: from voiceflow.scribe import ScribeMode
"""

__version__ = "0.1.0"

# Safe imports -- these work with just numpy
from voiceflow.audio import AudioRecorder, AudioConfig
from voiceflow.transcriber import Transcriber, TranscriptionConfig
from voiceflow.llm_postprocessor import LLMPostProcessor, LLMConfig
from voiceflow.typer import Typer
from voiceflow.voice_commands import VoiceCommandProcessor, VoiceCommand
from voiceflow.profiles import ProfileManager, AppProfile
from voiceflow.vocabulary import VocabularyManager
from voiceflow.agent_mode import AgentMode, Action, ActionRegistry, ActionMatch
from voiceflow.noise_gate import NoiseGate, NoiseGateConfig
from voiceflow.wake_word import WakeWordDetector, WakeWordConfig
from voiceflow.config import load_config, save_config, get_config_path

__all__ = [
    # Version
    "__version__",
    # Audio
    "AudioRecorder", "AudioConfig",
    # Transcription
    "Transcriber", "TranscriptionConfig",
    # LLM
    "LLMPostProcessor", "LLMConfig",
    # Text injection
    "Typer",
    # Voice commands
    "VoiceCommandProcessor", "VoiceCommand",
    # Profiles
    "ProfileManager", "AppProfile",
    # Vocabulary
    "VocabularyManager",
    # Agent mode
    "AgentMode", "Action", "ActionRegistry", "ActionMatch",
    # Noise gate
    "NoiseGate", "NoiseGateConfig",
    # Wake word
    "WakeWordDetector", "WakeWordConfig",
    # Config
    "load_config", "save_config", "get_config_path",
]
