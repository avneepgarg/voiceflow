# VoiceFlow

**Free, open-source, system-wide voice dictation for Windows.**

VoiceFlow types transcribed speech into any active application using local Whisper (GPU-accelerated) with optional LLM cleanup. Think Wispr Flow, but free, private, and open source.

## Features

| Feature | Status |
|---------|--------|
| System-wide dictation (hotkey → record → transcribe → type) | ✅ |
| Local Whisper transcription (GPU via CUDA) | ✅ |
| Cloud LLM text cleanup (OpenAI/Groq, BYOK) | ✅ |
| Per-app profiles (context-aware, no screenshots) | ✅ |
| Voice commands ("new line", "delete word", "caps on") | ✅ |
| Agent mode (voice actions: open apps, search web) | ✅ |
| Developer command mode (run build, git, deploy) | ✅ |
| Noise gate (RMS-based background filtering) | ✅ |
| Custom vocabulary / adaptive learning | ✅ |
| Dictation overlay (preview before committing) | ✅ |
| Wake word ("Hey VoiceFlow") | ✅ |
| Real-time translation (speak Hindi → type English) | ✅ |
| Scribe mode (speaker diarization, timestamps) | ✅ |
| Tone-aware formatting | ✅ |
| MCP integrations (Slack, Linear, Notion) | ✅ |
| System tray with status indicator | ✅ |
| Cross-device sync server | ✅ |
| **Unique differentiator** | Per-app profiles WITHOUT screenshots (privacy-first) |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      VoiceFlow Daemon                     │
│                                                          │
│  Hotkey (Ctrl+Alt+Space)                                 │
│       │                                                  │
│       ▼                                                  │
│  AudioRecorder ──► NoiseGate ──► buffer                  │
│       │                              │                   │
│       ▼                              ▼                   │
│  Release hotkey              Transcriber (Whisper GPU)   │
│       │                              │                   │
│       ▼                              ▼                   │
│  Next recording              VoiceCommandProcessor       │
│       │                        │         │               │
│       │                  commands?    plain text          │
│       │                        │         │               │
│       │                        ▼         ▼               │
│       │                  AgentMode    LLMPostProcessor   │
│       │                  DevCommands  (optional BYOK)    │
│       │                        │         │               │
│       │                        ▼         ▼               │
│       │                      Typer ──► active window     │
│       │                                                  │
│  System Tray: idle ● recording ● processing ● error      │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Windows 10/11 (64-bit)
- Python 3.10+ (for development) OR just the .exe (for users)
- Microphone
- NVIDIA GPU (optional, for local transcription. Falls back to CPU)

### Install

```bash
pip install voiceflow
```

Or download the latest `VoiceFlow.exe` from Releases.

### Configure

VoiceFlow creates a config file on first run:

```json
{
  "hotkey": "ctrl+alt+space",
  "model_size": "base",
  "device": "auto",
  "language": null,
  "typing_delay": 0.005,
  "max_recording_seconds": 120,
  "llm": {
    "enabled": false,
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }
}
```

### Usage

```bash
# Start with defaults
python -m voiceflow

# Custom hotkey
python -m voiceflow --model tiny --hotkey f9

# List audio devices
python -m voiceflow --list-devices

# Quick test (record 5 seconds, transcribe, exit)
python -m voiceflow --test-record
```

**How it works:**
1. Hold `Ctrl+Alt+Space` (or your configured hotkey)
2. Speak
3. Release — text appears in the active window

### LLM Cleanup

Enable AI-powered text cleanup (removes filler words, fixes grammar):

```json
"llm": {
  "enabled": true,
  "api_key": "sk-...",
  "model": "gpt-4o-mini"
}
```

Supports any OpenAI-compatible API (Groq is free and fast):

```json
"llm": {
  "enabled": true,
  "api_key": "gsk_...",
  "base_url": "https://api.groq.com/openai/v1",
  "model": "llama-3.3-70b-versatile"
}
```

## Voice Commands

Say these while dictating:

| Command | Action |
|---------|--------|
| "new line" | Press Enter |
| "delete last word" | Backspace last word |
| "period" or "full stop" | Type "." |
| "comma" | Type "," |
| "question mark" | Type "?" |
| "caps on" ... "caps off" | Caps-lock range |
| "scratch that" | Undo last dictation |
| "select all" | Ctrl+A |

## Per-App Profiles

VoiceFlow detects the active window and applies different settings:

```json
"profiles": {
  "vscode": {
    "hotkey": "ctrl+alt+v",
    "llm_enabled": false,
    "typing_delay": 0.01
  },
  "gmail": {
    "llm_enabled": true,
    "language": "en"
  },
  "slack": {
    "llm_enabled": false
  }
}
```

Profiles match against window title and exe name. No screenshots needed — privacy first.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development workflow.

## Roadmap

- [x] Core dictation engine (Sprint 1)
- [x] All Phase 2 features: voice commands, agent mode, profiles, etc. (Sprint 2)
- [x] CI/CD pipeline (GitHub Actions)
- [ ] Windows .exe build (via CI)
- [ ] Mobile companion app (Flutter, Phase 3)
- [ ] GPU benchmarking on Colab

## License

MIT License. See [LICENSE](LICENSE).
