# Mobile App (Flutter)

This directory contains the VoiceFlow mobile companion app built with Flutter.

## Structure
```
mobile/
├── lib/
│   ├── main.dart                 # Entry point
│   ├── screens/
│   │   ├── home_screen.dart      # Main screen with record button
│   │   ├── notes_list.dart       # List of voice notes
│   │   ├── settings_screen.dart  # App settings
│   │   └── overlay_screen.dart   # Dictation overlay
│   ├── services/
│   │   ├── audio_service.dart    # Audio capture + VAD
│   │   ├── transcription_service.dart  # Whisper + cloud
│   │   ├── sync_service.dart     # WebSocket sync client
│   │   └── llm_service.dart      # Cloud LLM cleanup
│   ├── models/
│   │   ├── voice_note.dart       # Voice note data model
│   │   └── app_profile.dart      # Per-app profile model
│   └── widgets/
│       ├── record_button.dart    # Big record button
│       ├── note_card.dart        # Voice note display card
│       └── waveform_widget.dart  # Audio waveform visualizer
├── ios/                          # iOS platform files
├── android/                      # Android platform files (includes IME)
├── test/                         # Flutter test suite
└── pubspec.yaml                  # Dependencies
```

## Features
- Tap-to-record voice notes with on-device or cloud transcription
- Custom keyboard extension for system-wide dictation
- Push-to-talk via floating action button
- Continuous / Scribe mode for meetings
- Cross-device sync with VoiceFlow Sync Server
- Home screen widget for 1-tap recording
- Siri/Google Assistant shortcuts
- Offline transcription (whisper.cpp via FFI)

## Build
```bash
flutter build apk          # Android
flutter build ios          # iOS (requires macOS + Xcode)
flutter build appbundle    # Android Play Store
```

## Sync Protocol
- REST API at POST/GET /sync/notes (see sync_server.py)
- WebSocket at WS /sync/realtime for instant push
- End-to-end encryption with AES-256-GCM
