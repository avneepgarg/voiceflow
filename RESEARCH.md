# VoiceFlow Competitive Research Summary

## Feature Landscape Analysis (2026)

### What Wispr Flow does (baseline)
- System-wide dictation via hotkey
- AI cleanup: filler word removal, grammar fix, punctuation
- Context awareness: takes screenshots of active window to understand what app you're in
- Voice commands: delete last word, new line, capitalization, etc.
- Cross-platform: Mac, Windows, iOS, Android
- Cloud-based (audio goes to their servers)

### What VoiceInk does (macOS only)
- Power Mode: auto-detects active app/URL and applies pre-configured settings per app
- Example: Gmail -> email mode, Slack -> casual mode, Cursor -> code mode
- Open source
- This is the MOST requested feature across the industry

### What Spokenly does (cross-platform, newer)
- Agent Mode: voice control to search web, launch apps, run shortcuts
- Basically turns dictation into a voice assistant that can do actions
- This is bleeding-edge (2026)

### What NovaVoice does (April 2026 launch, #1 on Product Hunt)
- "Voice OS" concept - voice as first-class interface for desktop
- Intelligent dictation + cross-app voice commands + AI assistant in one
- Format text instantly + get instant answers by voice

### What Talon / Serenade do (coding-specific)
- Custom command grammar: map any spoken phrase to any action
- Code-specific voice commands: "define function", "add parameter", "build project"
- App control: focus terminal, type text, press keys, run builds
- Talon also supports: eye tracking, noises (clicks, hisses) as input

### What Monologue does (late 2025)
- DeepContext: watches screen while dictating, uses visual context
- Per-mode custom instructions
- Mac only

---

## Features NO existing app has (opportunities for VoiceFlow)

1. **Native context-aware per-app profiles WITHOUT screenshots**
   - Wispr Flow and Monologue take screenshots (privacy concern!)
   - We'll use a different approach: detect active app name (window title), match against profile
   - Privacy advantage: no screenshots ever taken

2. **Agent Mode: Voice actions across apps**
   - Spokenly has this but it's basic
   - We can do it BETTER: robust command palette with app launch, URL open, file search, etc.
   - Works cross-platform (Spokenly is Mac-first)

3. **Multi-speaker dictation** (conversation mode)
   - Speaker diarization: "Speaker 1: ... Speaker 2: ..."
   - USP: NO consumer dictation app does this today
   - Useful for: interviews, meetings, phone calls, dialogue writing

4. **Real-time voice translation mode**
   - Speak in Hindi -> types English (or vice versa)
   - Useful for: bilingual users, content creators, international teams
   - Nothing in dictation space does this

5. **Tone-aware dictation**
   - Detect if you're being casual, formal, angry, excited
   - Adjust LLM cleanup accordingly
   - "Formal email mode" vs "Slack casual mode" that actually adapts

6. **Spelling mode**
   - Say "spell mode" -> next words spelled letter by letter
   - Critical for: code, URLs, passwords, technical terms
   - Quick toggle: "spell mode on/off"

7. **Developer command mode (Talon-lite)**
   - Voice commands that execute actual computer actions
   - "open terminal" -> opens terminal in current project dir
   - "git status" -> runs git status
   - "run build" -> runs the dev server
   - "fix lint" -> runs lint fixer

8. **Learning/adaptive vocabulary**
   - App learns YOUR frequently used words over time
   - Custom word list that Whisper uses as hints
   - Technical terms, names, brand names, industry jargon
   - Gets better the more you use it

9. **Integration/MCP layer**
   - Voice commands that trigger external actions
   - "send slack message to team" -> generates message, asks to confirm, sends
   - "create task in linear" -> creates issue
   - "start timer" -> starts Pomodoro
   - REST API / webhook support

10. **Continuous batch mode**
    - Toggle on once, it keeps listening
    - Auto-detects pauses and transcribes each segment
    - Like a live transcription overlay for conversations/meetings

11. **Multi-model transcription strategy**
    - Fast path: local Whisper tiny (always-on, instant)
    - Accurate path: cloud Whisper/vLLM for formal documents
    - User sets confidence threshold: auto-escalate unclear audio to better model

12. **Hinglish-optimized pipeline**
    - Whisper handles transcription in language=auto
    - LLM cleanup that specifically handles Hinglish (keeps Hindi words, fixes grammar)
    - Nobody does this well today

13. **Scribe mode (interview/conversation)**
    - Continuous listening with speaker diarization
    - Auto-formats as transcript with timestamps
    - Export as text/markdown/docx

14. **Distraction-free dictation overlay**
    - Small overlay window showing "what you said so far"
    - Real-time preview before committing to typing
    - Press Enter to confirm, Esc to re-record

15. **Wake word detection**
    - "Hey VoiceFlow" to activate (like Alexa/Siri)
    - Then continues in push-to-talk
    - Zero false triggers
