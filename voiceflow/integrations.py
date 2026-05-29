"""MCP/Integration layer -- Voice commands that trigger external APIs.

Supports Slack webhooks, Linear API, Notion API, and generic webhooks.
BYOK (bring your own key/config). Zero cost to the app itself.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for an external integration."""
    name: str
    provider: str = "generic"          # slack, linear, notion, generic
    webhook_url: str = ""
    api_key: str = ""
    api_base_url: str = ""
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    requires_confirmation: bool = True
    enabled: bool = True


@dataclass
class IntegrationResult:
    """Result of an integration call."""
    success: bool
    integration: str
    status_code: int = 0
    response: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    was_cached: bool = False


@dataclass
class IntegrationAction:
    """A voice-command-triggerable integration action."""
    name: str
    integration: str              # Which integration config to use
    patterns: List[str]           # Voice patterns that trigger this
    description: str = ""
    build_payload: str = ""       # static or "json" or "text"
    requires_confirmation: bool = True


class IntegrationManager:
    """Manages external integrations triggered by voice commands."""

    def __init__(self):
        self._configs: Dict[str, IntegrationConfig] = {}
        self._actions: Dict[str, IntegrationAction] = {}
        self._cache: Dict[str, IntegrationResult] = {}
        self._register_defaults()

    def register_config(self, config: IntegrationConfig):
        """Register an integration configuration."""
        self._configs[config.name] = config
        logger.info("Registered integration config: %s (%s)", config.name, config.provider)

    def register_action(self, action: IntegrationAction):
        """Register a voice-action mapping."""
        self._actions[action.name] = action
        logger.debug("Registered integration action: %s -> %s", action.name, action.integration)

    def match_and_execute(
        self,
        transcript: str,
        variables: Dict[str, str] = None,
        auto_confirm: bool = False,
    ) -> Optional[IntegrationResult]:
        """
        Match transcript against actions, build payload, and call the API.

        Args:
            transcript: Raw voice transcript
            variables: Template variables (user, timestamp, etc)
            auto_confirm: Skip confirmation prompts

        Returns:
            IntegrationResult or None if no action matched
        """
        import re

        transcript_lower = transcript.lower().strip()

        for action_name, action in self._actions.items():
            for pattern in action.patterns:
                match = re.search(pattern, transcript_lower)
                if not match:
                    continue

                # Extract arguments from regex groups
                args = match.groupdict()

                # Check if integration is configured
                if action.integration not in self._configs:
                    logger.warning("Integration '%s' not configured", action.integration)
                    return None

                config = self._configs[action.integration]
                if not config.enabled:
                    logger.debug("Integration '%s' is disabled", action.integration)
                    return None

                # Confirmation
                if action.requires_confirmation and not auto_confirm:
                    return IntegrationResult(
                        success=False,
                        integration=action.integration,
                        error="confirmation_required",
                    )

                # Build payload from template variables
                payload = self._build_payload(action, args, variables or {})

                # Execute
                return self._execute(config, payload)

        return None

    def _execute(self, config: IntegrationConfig, payload: str) -> IntegrationResult:
        """Execute an API call with retries."""
        cache_key = f"{config.name}:{hash(payload)}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.was_cached = True
            return cached

        start = time.time()
        last_error = ""

        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        data = payload.encode("utf-8")

        for attempt in range(1, config.max_retries + 1):
            try:
                req = urllib.request.Request(
                    config.webhook_url or config.api_base_url,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    result = IntegrationResult(
                        success=True,
                        integration=config.name,
                        status_code=resp.status,
                        response=body[:1000],
                        duration_seconds=time.time() - start,
                    )
                    self._cache[cache_key] = result
                    return result

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.read().decode()[:200]}"
                logger.warning("Integration '%s' attempt %d failed: %s",
                               config.name, attempt, last_error)
            except Exception as e:
                last_error = str(e)
                logger.warning("Integration '%s' attempt %d failed: %s",
                               config.name, attempt, last_error)

            if attempt < config.max_retries:
                time.sleep(config.retry_delay_seconds)

        return IntegrationResult(
            success=False,
            integration=config.name,
            error=last_error,
            duration_seconds=time.time() - start,
        )

    @staticmethod
    def _build_payload(
        action: IntegrationAction,
        args: Dict[str, str],
        variables: Dict[str, str],
    ) -> str:
        """Build the API payload from template variables and regex args."""
        # Merge all template sources
        ctx = {}
        ctx.update(variables)
        ctx.update(args)
        ctx["text"] = args.get("raw", "")

        if action.build_payload == "text":
            # Plain text payload
            return ctx.get("text", "")

        # Default: JSON
        payload_dict = {}
        for key, value in ctx.items():
            if isinstance(value, str):
                payload_dict[key] = value

        return json.dumps(payload_dict, ensure_ascii=False)

    def _register_defaults(self):
        """Register built-in integration actions."""
        self.register_action(IntegrationAction(
            name="slack_message",
            integration="slack",
            patterns=[
                r"\bsend\s+(?:a\s+)?slack\s+message\s+(?:to\s+)?(?P<channel>\w+)\s+(?:saying\s+)?(?P<message>.+?)$",
                r"\bslack\s+(?P<channel>\w+)\s+(?P<message>.+?)$",
            ],
            description="Send a message to a Slack channel via webhook.",
            build_payload="json",
        ))

        self.register_action(IntegrationAction(
            name="linear_task",
            integration="linear",
            patterns=[
                r"\bcreate\s+(?:a\s+)?task\s+(?:in\s+)?linear\s+(?:with\s+)?(?:title\s+)?(?P<title>.+?)$",
                r"\blinear\s+task\s+(?P<title>.+?)$",
            ],
            description="Create a task in Linear.",
            build_payload="json",
        ))

        self.register_action(IntegrationAction(
            name="notion_page",
            integration="notion",
            patterns=[
                r"\badd\s+(?:a\s+)?(?:page|note)\s+(?:to\s+)?notion\s+(?:titled\s+)?(?P<title>.+?)$",
                r"\bnotion\s+(?:page|note)\s+(?P<title>.+?)$",
            ],
            description="Add a page to Notion.",
            build_payload="json",
        ))
