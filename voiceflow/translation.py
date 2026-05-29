"""
VoiceFlow Translation -- Real-time voice translation pipeline.

Speak in one language, get text in another. Uses a two-step pipeline:
  1. Whisper transcribes audio in the source language
  2. An LLM translates the transcription to the target language

Quick-start
-----------
    from voiceflow.translation import TranslationPipeline, TranslationMode
    from voiceflow.llm_postprocessor import LLMConfig

    llm_config = LLMConfig(enabled=True, api_key="sk-...", model="gpt-4o-mini")
    translator = TranslationPipeline(
        source_lang="hi",
        target_lang="en",
        llm_config=llm_config,
        mode=TranslationMode.TRANSLATE,
    )
    result = translator.translate_audio(audio_array)
    print(result.translated_text)  # English text from Hindi speech

Mode toggle
-----------
    translator.mode = TranslationMode.NORMAL   # Dictation in source language
    translator.mode = TranslationMode.TRANSLATE  # Translate to target language

Config integration
------------------
    from voiceflow.config import load_config
    cfg = load_config()
    t_cfg = cfg.get("translation", {})
    translator = TranslationPipeline.from_config(t_cfg)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from voiceflow.llm_postprocessor import LLMConfig, LLMPostProcessor
from voiceflow.transcriber import Transcriber, TranscriptionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- 50+ supported language codes with human-readable names
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Dict[str, str] = {
    # Most common
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "pl": "Polish",
    "nl": "Dutch",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ms": "Malay",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sw": "Swahili",
    "tl": "Tagalog",
    "fa": "Persian",
    "af": "Afrikaans",
    "ca": "Catalan",
}

# Preset configurations for common translation pairs
TRANSLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "en->hi": {
        "source_lang": "en",
        "target_lang": "hi",
        "description": "English to Hindi",
    },
    "hi->en": {
        "source_lang": "hi",
        "target_lang": "en",
        "description": "Hindi to English",
    },
    "en->es": {
        "source_lang": "en",
        "target_lang": "es",
        "description": "English to Spanish",
    },
    "es->en": {
        "source_lang": "es",
        "target_lang": "en",
        "description": "Spanish to English",
    },
    "en->fr": {
        "source_lang": "en",
        "target_lang": "fr",
        "description": "English to French",
    },
    "fr->en": {
        "source_lang": "fr",
        "target_lang": "en",
        "description": "French to English",
    },
    "en->de": {
        "source_lang": "en",
        "target_lang": "de",
        "description": "English to German",
    },
    "de->en": {
        "source_lang": "de",
        "target_lang": "en",
        "description": "German to English",
    },
    "en->ja": {
        "source_lang": "en",
        "target_lang": "ja",
        "description": "English to Japanese",
    },
    "ja->en": {
        "source_lang": "ja",
        "target_lang": "en",
        "description": "Japanese to English",
    },
    "en->zh": {
        "source_lang": "en",
        "target_lang": "zh",
        "description": "English to Chinese",
    },
    "zh->en": {
        "source_lang": "zh",
        "target_lang": "en",
        "description": "Chinese to English",
    },
    "en->ko": {
        "source_lang": "en",
        "target_lang": "ko",
        "description": "English to Korean",
    },
    "ko->en": {
        "source_lang": "ko",
        "target_lang": "en",
        "description": "Korean to English",
    },
    "en->pt": {
        "source_lang": "en",
        "target_lang": "pt",
        "description": "English to Portuguese",
    },
    "pt->en": {
        "source_lang": "pt",
        "target_lang": "en",
        "description": "Portuguese to English",
    },
    "hi->ta": {
        "source_lang": "hi",
        "target_lang": "ta",
        "description": "Hindi to Tamil",
    },
    "hi->te": {
        "source_lang": "hi",
        "target_lang": "te",
        "description": "Hindi to Telugu",
    },
    "en->ar": {
        "source_lang": "en",
        "target_lang": "ar",
        "description": "English to Arabic",
    },
    "ar->en": {
        "source_lang": "ar",
        "target_lang": "en",
        "description": "Arabic to English",
    },
    "en->ru": {
        "source_lang": "en",
        "target_lang": "ru",
        "description": "English to Russian",
    },
    "ru->en": {
        "source_lang": "ru",
        "target_lang": "en",
        "description": "Russian to English",
    },
    "en->bn": {
        "source_lang": "en",
        "target_lang": "bn",
        "description": "English to Bengali",
    },
    "en->it": {
        "source_lang": "en",
        "target_lang": "it",
        "description": "English to Italian",
    },
    "it->en": {
        "source_lang": "it",
        "target_lang": "en",
        "description": "Italian to English",
    },
}


# ---------------------------------------------------------------------------
# Mode toggle
# ---------------------------------------------------------------------------

class TranslationMode:
    """Operating modes for the translation pipeline.

    NORMAL:
        Standard dictation -- transcribe in the source language, no translation.
    TRANSLATE:
        Full translation -- transcribe in source language, then translate to target.
    """

    NORMAL: str = "normal"
    TRANSLATE: str = "translate"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """Result of a translation pipeline run.

    Attributes
    ----------
    source_text:
        Raw transcribed text in the source language.
    translated_text:
        Translated text in the target language.  In NORMAL mode this is the
        same as *source_text*.  If translation fails, this falls back to
        *source_text* as well.
    from_lang:
        Source language code (e.g. ``"hi"``).
    to_lang:
        Target language code (e.g. ``"en"``).
    transcription_time:
        Wall-clock seconds spent on Whisper transcription.
    translation_time:
        Wall-clock seconds spent on LLM translation (0.0 in NORMAL mode).
    was_cached:
        Whether the translation was served from the cache.
    mode:
        The mode that was active when this result was produced.
    """

    source_text: str = ""
    translated_text: str = ""
    from_lang: str = ""
    to_lang: str = ""
    transcription_time: float = 0.0
    translation_time: float = 0.0
    was_cached: bool = False
    mode: str = TranslationMode.NORMAL

    @property
    def total_time(self) -> float:
        """Total wall-clock seconds (transcription + translation)."""
        return self.transcription_time + self.translation_time

    @property
    def display_text(self) -> str:
        """Best text to show/type to the user."""
        if self.mode == TranslationMode.NORMAL:
            return self.source_text
        return self.translated_text or self.source_text

    @property
    def formatted_output(self) -> str:
        """Formatted output showing both source and translated text."""
        if self.mode == TranslationMode.NORMAL:
            return self.source_text
        if not self.translated_text:
            return self.source_text
        return f"[{self.from_lang}] {self.source_text}\n[{self.to_lang}] {self.translated_text}"


# ---------------------------------------------------------------------------
# Translation cache
# ---------------------------------------------------------------------------

class TranslationCache:
    """LRU-style cache that avoids re-translating identical segments.

    Keys are ``(source_text, from_lang, to_lang)`` tuples.  A SHA-256 hash
    is used internally so cache keys have a fixed maximum length.

    Parameters
    ----------
    max_size:
        Maximum number of entries.  When the cache is full, the oldest entry
        is evicted (FIFO eviction).
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._store: Dict[str, Tuple[str, float]] = {}  # hash -> (translated, timestamp)
        self._order: List[str] = []  # insertion order for FIFO eviction

    # -- public API ----------------------------------------------------------

    def get(self, source_text: str, from_lang: str, to_lang: str) -> Optional[str]:
        """Return cached translation, or ``None``."""
        key = self._make_key(source_text, from_lang, to_lang)
        entry = self._store.get(key)
        if entry is not None:
            translated, _ = entry
            logger.debug("Translation cache hit for key %s", key[:16])
            return translated
        return None

    def put(self, source_text: str, from_lang: str, to_lang: str, translated: str) -> None:
        """Store a translation in the cache."""
        key = self._make_key(source_text, from_lang, to_lang)
        if key in self._store:
            # Update existing entry; move to end of order
            self._order.remove(key)
        elif len(self._store) >= self.max_size:
            # Evict oldest
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
            logger.debug("Evicted oldest cache entry %s", oldest[:16])

        self._store[key] = (translated, time.monotonic())
        self._order.append(key)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()
        self._order.clear()
        logger.info("Translation cache cleared")

    @property
    def size(self) -> int:
        return len(self._store)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _make_key(source_text: str, from_lang: str, to_lang: str) -> str:
        raw = f"{from_lang}:{to_lang}:{source_text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------

class TranslationPipeline:
    """Two-step voice translation: Whisper transcribe -> LLM translate.

    Parameters
    ----------
    source_lang:
        BCP-47 language code for the input speech (e.g. ``"hi"``).
    target_lang:
        BCP-47 language code for the desired output text (e.g. ``"en"``).
    llm_config:
        Configuration for the LLM used in the translation step.  Reuses
        :class:`LLMConfig` from ``llm_postprocessor.py``.
    mode:
        ``TranslationMode.NORMAL`` for plain dictation or
        ``TranslationMode.TRANSLATE`` for full translation.
    model_size:
        Whisper model size (``"tiny"``, ``"base"``, ``"small"``, etc.).
    device:
        Whisper device (``"auto"``, ``"cpu"``, ``"cuda"``).
    cache_size:
        Maximum translation cache entries.  Set to 0 to disable caching.
    show_both:
        If ``True``, ``formatted_output`` includes both source and translated text.

    Examples
    --------
    Basic usage::

        pipeline = TranslationPipeline("hi", "en", llm_config=my_config)
        result = pipeline.translate_audio(audio_array)
        print(result.translated_text)

    Toggle mode at runtime::

        pipeline.mode = TranslationMode.NORMAL
        result = pipeline.translate_audio(audio_array)
        print(result.source_text)  # Hindi text, no translation
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "en",
        llm_config: Optional[LLMConfig] = None,
        mode: str = TranslationMode.NORMAL,
        model_size: str = "base",
        device: str = "auto",
        cache_size: int = 500,
        show_both: bool = False,
    ) -> None:
        # Validate language codes
        self._validate_lang(source_lang, "source_lang")
        self._validate_lang(target_lang, "target_lang")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.llm_config = llm_config or LLMConfig()
        self.mode = mode
        self.show_both = show_both

        # Build a transcriber locked to the source language so Whisper
        # does not auto-detect (faster, more accurate for translation).
        self._transcriber = Transcriber(
            TranscriptionConfig(
                model_size=model_size,
                device=device,
                language=source_lang,
            )
        )

        # Reuse LLMPostProcessor for the translation LLM call, but we
        # override the system prompt per-request in _build_translation_prompt.
        self._llm_processor = LLMPostProcessor(self.llm_config)

        # Cache
        self._cache = TranslationCache(max_size=cache_size) if cache_size > 0 else None

        logger.info(
            "TranslationPipeline initialised: %s -> %s (mode=%s, model=%s, device=%s)",
            source_lang,
            target_lang,
            mode,
            model_size,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_audio(self, audio: np.ndarray) -> TranslationResult:
        """Transcribe *audio* and optionally translate.

        This is the primary entry point.  It returns a
        :class:`TranslationResult` with timing information.

        Parameters
        ----------
        audio:
            16 kHz mono float32 numpy array.

        Returns
        -------
        TranslationResult
        """
        result = TranslationResult(
            from_lang=self.source_lang,
            to_lang=self.target_lang,
            mode=self.mode,
        )

        # -- Step 1: Transcription --------------------------------------
        t0 = time.monotonic()
        try:
            source_text = self._transcriber.transcribe(audio)
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return result  # empty result, transcription_time stays 0

        t1 = time.monotonic()
        result.source_text = source_text
        result.transcription_time = round(t1 - t0, 4)

        if not source_text:
            logger.debug("Empty transcription, skipping translation")
            result.translated_text = ""
            return result

        # -- Step 2: Translation (only in TRANSLATE mode) ----------------
        if self.mode == TranslationMode.NORMAL:
            result.translated_text = source_text
            return result

        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(source_text, self.source_lang, self.target_lang)
            if cached is not None:
                result.translated_text = cached
                result.was_cached = True
                result.translation_time = 0.0
                logger.debug("Served translation from cache")
                return result

        # Call LLM
        t2 = time.monotonic()
        try:
            translated = self._translate_text(source_text)
        except Exception as exc:
            logger.error("Translation failed, falling back to raw transcription: %s", exc)
            translated = source_text  # graceful fallback

        t3 = time.monotonic()
        result.translated_text = translated
        result.translation_time = round(t3 - t2, 4)

        # Store in cache
        if self._cache is not None and translated != source_text:
            self._cache.put(source_text, self.source_lang, self.target_lang, translated)

        logger.info(
            "Translation complete: %.2fs transcription + %.2fs translation = %.2fs total",
            result.transcription_time,
            result.translation_time,
            result.total_time,
        )

        return result

    def toggle_mode(self) -> str:
        """Toggle between NORMAL and TRANSLATE modes.

        Returns the new mode string.
        """
        if self.mode == TranslationMode.NORMAL:
            self.mode = TranslationMode.TRANSLATE
        else:
            self.mode = TranslationMode.NORMAL
        logger.info("Mode toggled to %s", self.mode)
        return self.mode

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        if self._cache:
            self._cache.clear()

    @property
    def cache_info(self) -> Dict[str, int]:
        """Return cache statistics."""
        if self._cache is None:
            return {"enabled": 0, "size": 0, "max_size": 0}
        return {"enabled": 1, "size": self._cache.size, "max_size": self._cache.max_size}

    # ------------------------------------------------------------------
    # Class methods / presets
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        llm_config: Optional[LLMConfig] = None,
        **kwargs: Any,
    ) -> TranslationPipeline:
        """Create a pipeline from a named preset (e.g. ``"hi->en"``).

        Parameters
        ----------
        preset_key:
            Key into :data:`TRANSLATION_PRESETS`, e.g. ``"en->hi"``.
        llm_config:
            LLM configuration forwarded to the constructor.
        **kwargs:
            Additional keyword arguments forwarded to the constructor.

        Raises
        ------
        KeyError
            If *preset_key* is not a recognised preset.
        """
        if preset_key not in TRANSLATION_PRESETS:
            available = ", ".join(sorted(TRANSLATION_PRESETS.keys()))
            raise KeyError(
                f"Unknown preset '{preset_key}'. Available presets: {available}"
            )
        preset = TRANSLATION_PRESETS[preset_key]
        logger.info("Loading translation preset: %s", preset["description"])
        return cls(
            source_lang=preset["source_lang"],
            target_lang=preset["target_lang"],
            llm_config=llm_config,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TranslationPipeline:
        """Build a pipeline from a configuration dictionary (e.g. from
        ``voiceflow.config``).

        Expected keys::

            {
                "source_lang": "hi",
                "target_lang": "en",
                "mode": "translate",
                "model_size": "base",
                "device": "auto",
                "cache_size": 500,
                "show_both": false,
                "llm": { ... }   # forwarded to LLMConfig
            }
        """
        llm_cfg = LLMConfig(**(config.get("llm", {})))
        return cls(
            source_lang=config.get("source_lang", "en"),
            target_lang=config.get("target_lang", "en"),
            llm_config=llm_cfg,
            mode=config.get("mode", TranslationMode.NORMAL),
            model_size=config.get("model_size", "base"),
            device=config.get("device", "auto"),
            cache_size=config.get("cache_size", 500),
            show_both=config.get("show_both", False),
        )

    @staticmethod
    def list_presets() -> Dict[str, str]:
        """Return a mapping of preset keys to human-readable descriptions."""
        return {key: val["description"] for key, val in TRANSLATION_PRESETS.items()}

    @staticmethod
    def list_languages() -> Dict[str, str]:
        """Return all supported language codes and their names."""
        return dict(SUPPORTED_LANGUAGES)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_text(self, text: str) -> str:
        """Translate *text* from source_lang to target_lang via LLM.

        Falls back to the original text on any error.
        """
        if not self.llm_config.enabled or not self.llm_config.api_key:
            logger.warning(
                "LLM translation requested but LLM is disabled or has no API key. "
                "Returning raw transcription."
            )
            return text

        source_name = SUPPORTED_LANGUAGES.get(self.source_lang, self.source_lang)
        target_name = SUPPORTED_LANGUAGES.get(self.target_lang, self.target_lang)

        system_prompt = (
            f"You are a professional translator. "
            f"Translate the following text from {source_name} to {target_name}. "
            f"Output ONLY the translated text -- no explanations, no notes, "
            f"no romanisation unless the target language uses a non-Latin script "
            f"and the user explicitly asks for it. Preserve the tone and meaning."
        )

        user_prompt = text

        try:
            translated = self._call_llm(system_prompt, user_prompt)
            return translated
        except Exception as exc:
            logger.error("LLM translation call failed: %s", exc)
            return text  # fallback

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Low-level LLM call using the OpenAI-compatible client."""
        import openai

        client = openai.OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url or None,
        )

        response = client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.llm_config.max_tokens,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()

    @staticmethod
    def _validate_lang(code: str, param_name: str) -> None:
        """Raise ``ValueError`` if *code* is not in SUPPORTED_LANGUAGES."""
        if code not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language code '{code}' for {param_name}. "
                f"Use one of: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
            )


# ---------------------------------------------------------------------------
# Convenience function for quick one-off translations
# ---------------------------------------------------------------------------

def quick_translate(
    audio: np.ndarray,
    source_lang: str,
    target_lang: str,
    llm_config: Optional[LLMConfig] = None,
    model_size: str = "base",
    device: str = "auto",
) -> TranslationResult:
    """One-shot translation without explicitly creating a pipeline.

    Example::

        result = quick_translate(audio, "hi", "en", llm_config=my_config)
        print(result.translated_text)
    """
    pipeline = TranslationPipeline(
        source_lang=source_lang,
        target_lang=target_lang,
        llm_config=llm_config,
        mode=TranslationMode.TRANSLATE,
        model_size=model_size,
        device=device,
    )
    return pipeline.translate_audio(audio)
