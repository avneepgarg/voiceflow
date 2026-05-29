# Contributing to VoiceFlow

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- A microphone (for manual testing)
- NVIDIA GPU (optional, for local Whisper testing)

### Clone and Install

```bash
git clone https://github.com/voiceflow/voiceflow.git
cd voiceflow
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_audio.py -v

# Quick smoke test (no GPU needed)
python tests/smoke_test.py
```

### Run Locally

```bash
# List audio devices
python -m voiceflow --list-devices

# Record 5 seconds and transcribe
python -m voiceflow --test-record

# Start the daemon
python -m voiceflow --model tiny
```

## Project Structure

```
voiceflow/
├── voiceflow/              # Main source
│   ├── __init__.py         # Public API
│   ├── __main__.py         # Module entry point
│   ├── main.py             # App daemon, hotkey loop
│   ├── audio.py            # Microphone capture
│   ├── transcriber.py      # Whisper wrapper
│   ├── typer.py            # Text injection
│   ├── config.py           # JSON config
│   ├── tray.py             # System tray
│   ├── llm_postprocessor.py # LLM cleanup
│   ├── voice_commands.py   # Voice editing commands
│   ├── agent_mode.py       # Voice actions
│   ├── dev_commands.py     # Developer commands (Talon-lite)
│   ├── profiles.py         # Per-app profiles
│   ├── vocabulary.py       # Custom vocabulary
│   ├── noise_gate.py       # Background noise filter
│   ├── wake_word.py        # "Hey VoiceFlow"
│   ├── tone_aware.py       # Tone detection
│   ├── overlay.py          # Dictation overlay
│   ├── translation.py      # Voice translation
│   ├── scribe.py           # Scribe mode
│   ├── integrations.py     # MCP/webhooks
│   ├── sync_server.py      # Cross-device sync
│   └── utils.py            # Shared utilities
├── tests/                  # Test suite
│   ├── smoke_test.py       # Quick import checks
│   ├── test_*.py           # Unit tests
│   └── test_integration.py # Multi-module tests
├── .github/workflows/      # CI/CD
│   ├── ci.yml              # Test on push/PR
│   └── build_windows.yml   # .exe build
├── colab/                  # GPU test notebooks
├── mobile/                 # Flutter mobile app (Phase 3)
├── setup.py                # pip install config
├── VoiceFlow.spec          # PyInstaller spec
├── PLAN.md                 # Architecture & task plan
└── README.md
```

## Development Rules

1. **Zero local stress**: Never install torch/pytorch locally. Never download Whisper models locally. Never run PyInstaller locally. All heavy compute happens in the cloud (GitHub Actions, Colab).

2. **Test-first**: Every new feature needs unit tests. Integration tests for multi-module workflows.

3. **No heavy deps in core**: The `voiceflow/__init__.py` safe imports should only require numpy. Heavy deps (sounddevice, pystray, faster-whisper) are lazy-loaded inside functions.

4. **Type hints**: All public functions should have type hints.

5. **Docstrings**: All modules, classes, and public functions should have docstrings.

## CI/CD

Tests run automatically on every push via GitHub Actions (free, cloud).

- **ci.yml**: Runs all 105+ tests on Python 3.10-3.14 (Linux)
- **build_windows.yml**: Builds VoiceFlow.exe on tag push (Windows)

## Release Process

1. Update version in `voiceflow/__init__.py`
2. Tag the release: `git tag v0.1.0`
3. Push tags: `git push --tags`
4. GitHub Actions builds the .exe automatically
5. Download artifact from Actions tab, attach to GitHub Release

## Code Style

- PEP 8
- Sorted imports (stdlib → third-party → local)
- Max line length: 120
- Use f-strings, not % formatting

## Adding a New Module

1. Create `voiceflow/your_module.py`
2. Add tests in `tests/test_your_module.py`
3. Add to `voiceflow/__init__.py` `__all__` if it's a public API
4. Add smoke test entry in `tests/smoke_test.py`
5. Run `python -m pytest tests/ -v` to verify
