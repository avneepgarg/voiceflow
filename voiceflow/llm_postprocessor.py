"""LLM Post-Processor - optional cloud-based text cleanup.

Uses any OpenAI-compatible API (OpenAI, Anthropic via proxy, Ollama, etc).
BYOK - user provides their own API key. Cost is ~$0.001 per transcription.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM post-processing."""
    enabled: bool = False
    provider: str = "openai"           # openai, anthropic, custom, ollama
    api_key: str = ""
    base_url: str = ""                 # Custom endpoint. Empty = provider default
    model: str = "gpt-4o-mini"        # Model name
    max_tokens: int = 512

    # Cleanup toggles
    remove_fillers: bool = True
    fix_grammar: bool = True
    add_punctuation: bool = True
    reformat: str = "none"             # none, bullets, paragraphs, concise

    # Custom prompt (advanced)
    system_prompt: str = (
        "You are a text cleanup assistant. Fix the transcription. "
        "Remove filler words. Add proper punctuation. Keep the meaning exactly the same. "
        "Do NOT add any new information. Output ONLY the corrected text, nothing else."
    )


class LLMPostProcessor:
    """
    Post-processes raw transcription through an LLM for cleanup.

    Usage:
        processor = LLMPostProcessor(LLMConfig(enabled=True, api_key="sk-..."))
        clean_text = processor.process(raw_transcription)
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()

    def process(self, text: str) -> str:
        """
        Send raw transcription to LLM for cleanup.

        Args:
            text: Raw transcribed text from Whisper

        Returns:
            Cleaned-up text. Falls back to original text if LLM is
            disabled, has no API key, or fails for any reason.
        """
        if not self.config.enabled or not text:
            return text

        if not self.config.api_key:
            logger.warning("LLM post-processing enabled but no API key configured")
            return text

        try:
            result = self._call_llm(text)
            logger.debug("LLM cleanup applied successfully")
            return result
        except Exception as e:
            logger.error("LLM cleanup failed, using raw text: %s", e)
            return text  # Graceful fallback - never block transcription

    def _call_llm(self, text: str) -> str:
        """Call the configured LLM API via OpenAI-compatible interface."""
        import openai

        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

        user_prompt = self._build_user_prompt(text)

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=0.1,  # Low temp = faithful cleanup, not creative rewriting
        )

        return response.choices[0].message.content.strip()

    def _build_user_prompt(self, text: str) -> str:
        """Build the user prompt with cleanup instructions."""
        parts = [text]

        instructions = []
        if self.config.remove_fillers:
            instructions.append("Remove filler words (um, uh, like, you know, etc)")
        if self.config.fix_grammar:
            instructions.append("Fix grammar and spelling")
        if self.config.add_punctuation:
            instructions.append("Ensure proper punctuation and capitalization")

        reformat_map = {
            "bullets": "Reformat as bullet points",
            "paragraphs": "Reformat into proper paragraphs",
            "concise": "Make it concise and clear, remove redundancy",
        }
        if self.config.reformat in reformat_map:
            instructions.append(reformat_map[self.config.reformat])

        if instructions:
            parts.append("\n".join(f"- {inst}" for inst in instructions))

        return "\n\n".join(parts)

    @staticmethod
    def get_available_models() -> dict:
        """Return available model presets withcost info."""
        return {
            "openai/gpt-4o-mini": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "",
                "cost": "~$0.15/1M input tokens",
            },
            "openai/gpt-4o": {
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "",
                "cost": "~$2.50/1M input tokens",
            },
            "groq/llama-3.3-70b": {
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "base_url": "https://api.groq.com/openai/v1",
                "cost": "Free tier available",
            },
            "anthropic/claude-3-haiku": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "base_url": "https://api.anthropic.com/v1",
                "cost": "~$0.25/1M input tokens",
            },
            "custom/ollama": {
                "provider": "ollama",
                "model": "llama3.2",
                "base_url": "http://localhost:11434/v1",
                "cost": "$0 (local)",
            },
        }
