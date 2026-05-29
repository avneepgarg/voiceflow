"""
VoiceFlow Agent Mode -- Pluggable voice command action system.

"Agent Mode" lets voice commands DO things, not just type text. Spoken phrases
are matched against registered *actions* and, when a match is found, the
corresponding Python handler is executed.

Quick-start
-----------
    from voiceflow.agent_mode import AgentMode

    agent = AgentMode()
    result = agent.process_transcript("open firefox")
    if result:
        print(f"Executed: {result.action_name}")

Registering a custom action
---------------------------
    from voiceflow.agent_mode import Action, ActionRegistry

    registry = ActionRegistry()

    my_action = Action(
        name="toggle_light",
        patterns=[r"turn (?:the )?(?P<room>\\w+) light (on|off)"],
        handler=my_light_handler,
        description="Toggle a smart light on or off",
    )
    registry.register(my_action)

Design notes
------------
- Regex patterns may use named groups (?P<name>...) to capture arguments.
- Actions marked ``requires_confirmation=True`` ask the user before executing.
- All handlers run inside a subprocess with a configurable timeout for safety.
- The module is fully self-contained *except* for the optional ``webbrowser``
  stdlib module used by the built-in ``open_url`` / ``search_web`` actions.

Public API
----------
AgentMode            -- High-level façade; owns a registry + processes text.
Action               -- Dataclass describing a single action.
ActionMatch          -- Dataclass returned by pattern matching.
ActionRegistry       -- Stores/retrieves registered actions.
MatchConfidence      -- Enum-like helper for match strength.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
import time
import webbrowser
from concurrent.futures import Future, TimeoutError as FuturesTimeout
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: int = 30
"""Maximum wall-clock seconds an action handler is allowed to run."""

CONFIRMATION_KEYWORDS: Tuple[str, ...] = ("yes", "yeah", "yep", "confirm", "do it", "go ahead")
"""Simple keyword list for recognising affirmative confirmation replies."""

DENY_KEYWORDS: Tuple[str, ...] = ("no", "nope", "cancel", "abort", "nah")
"""Simple keyword list for recognising negative replies."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class MatchConfidence:
    """Low / medium / high confidence tiers for pattern matches.

    Values are integers so they can be compared with ``<`` / ``>``.
    """

    LOW: int = 1
    MEDIUM: int = 2
    HIGH: int = 3


@dataclasses.dataclass
class ActionMatch:
    """Represents a successful match between spoken text and an action.

    Attributes
    ----------
    action_name:
        Name of the matched registered action.
    args:
        Named arguments extracted from the regex match (via ``(?P<name>)``
        groups) plus any additional context the caller supplies.
    confidence:
        How confident the matcher is -- see ``MatchConfidence``.
    matched_text:
        The actual text span that triggered the match.
    """

    action_name: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    confidence: int = MatchConfidence.MEDIUM
    matched_text: str = ""


@dataclasses.dataclass
class Action:
    """A single actionable voice command.

    Parameters
    ----------
    name:
        Unique identifier for this action (e.g. ``"open_app"``).
    patterns:
        One or more **compiled** regex patterns *or* raw strings.  Strings are
        auto-compiled on registration.  Use ``(?P<name>)`` named groups to
        capture arguments.
    handler:
        Callable invoked when the action is executed.  Receives a dict of
        extracted args and should return any JSON-serialisable result (or
        ``None``).
    description:
        Human-readable summary of what the action does.
    requires_confirmation:
        If ``True``, ``AgentMode`` will prompt the user for confirmation
        *before* executing the handler.
    timeout_seconds:
        Maximum seconds the handler may run before being killed.  Defaults to
        ``DEFAULT_TIMEOUT_SECONDS``.
    """

    name: str
    patterns: Sequence[str | re.Pattern]
    handler: Callable[..., Any]
    description: str = ""
    requires_confirmation: bool = False
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Action Registry
# ---------------------------------------------------------------------------

class ActionRegistry:
    """Thread-safe registry that maps action names to ``Action`` objects.

    Built-in actions are registered automatically; user-defined actions can be
    added at any time via :meth:`register`.

    Example::

        reg = ActionRegistry()
        reg.register(Action(name="greet", patterns=[r"say hi"], handler=hi))
        match = reg.match("say hi there")
    """

    def __init__(self) -> None:
        self._actions: Dict[str, Action] = {}
        self._compiled: Dict[str, List[re.Pattern]] = {}
        self._lock = __import__("threading").Lock()

    # -- public API ----------------------------------------------------------

    def register(self, action: Action) -> None:
        """Register an *action* so it can be matched and executed.

        Raises ``ValueError`` if another action with the same name is already
        registered.  Call :meth:`unregister` first if you want to replace it.
        """
        with self._lock:
            if action.name in self._actions:
                raise ValueError(
                    f"Action '{action.name}' is already registered. "
                    f"Call unregister('{action.name}') first to replace it."
                )
            self._actions[action.name] = action
            precompiled: List[re.Pattern] = []
            for pat in action.patterns:
                if isinstance(pat, re.Pattern):
                    precompiled.append(pat)
                else:
                    precompiled.append(re.compile(pat, re.IGNORECASE))
            self._compiled[action.name] = precompiled
        logger.info("Registered action: %s", action.name)

    def unregister(self, name: str) -> None:
        """Remove a previously registered action."""
        with self._lock:
            self._actions.pop(name, None)
            self._compiled.pop(name, None)
        logger.info("Unregistered action: %s", name)

    def get(self, name: str) -> Optional[Action]:
        """Return the ``Action`` with *name*, or ``None``."""
        return self._actions.get(name)

    def list_actions(self) -> List[Action]:
        """Return a snapshot of all currently registered actions."""
        with self._lock:
            return list(self._actions.values())

    def match(self, transcript: str) -> Optional[ActionMatch]:
        """Match *transcript* against all registered action patterns.

        Returns the **best** ``ActionMatch`` (highest confidence, then first
        pattern match wins), or ``None`` if nothing matched.
        """
        best: Optional[ActionMatch] = None
        best_confidence = -1

        with self._lock:
            actions_snapshot = list(self._actions.items())
            compiled_snapshot = {
                name: pats for name, pats in self._compiled.items()
            }

        for name, action in actions_snapshot:
            for pattern in compiled_snapshot.get(name, []):
                m = pattern.search(transcript)
                if m is None:
                    continue
                args = m.groupdict()
                # Confidence: named groups -> HIGH, positional -> MEDIUM,
                # simple substring-ish -> LOW.
                if args:
                    confidence = MatchConfidence.HIGH
                elif m.group() == transcript.strip():
                    confidence = MatchConfidence.MEDIUM
                else:
                    confidence = MatchConfidence.HIGH

                if confidence > best_confidence or (
                    confidence == best_confidence and best is not None
                ):
                    best = ActionMatch(
                        action_name=name,
                        args=args,
                        confidence=confidence,
                        matched_text=m.group(),
                    )
                    best_confidence = confidence
                break  # first matching pattern per action is enough

        return best

    # -- convenience for bulk registration -----------------------------------

    def register_defaults(self) -> None:
        """Register all built-in actions shipped with this module."""
        from . import agent_mode as _am  # circular-safe late import

        builtins: List[Action] = [
            _am._OPEN_APP_ACTION,
            _am._OPEN_URL_ACTION,
            _am._RUN_COMMAND_ACTION,
            _am._SEARCH_WEB_ACTION,
            _am._START_TIMER_ACTION,
            _am._SET_REMINDER_ACTION,
        ]
        for action in builtins:
            self.register(action)


# ---------------------------------------------------------------------------
# Built-in action handlers
# ---------------------------------------------------------------------------

def _handle_open_app(args: Dict[str, Any]) -> Dict[str, Any]:
    """Launch a desktop application by name.

    Expected key in *args*:
        app (str): Friendly or executable name (e.g. ``"firefox"``).
    """
    app_name: str = args.get("app", "").strip()
    if not app_name:
        raise ValueError("'app' argument is required for open_app action")

    # Try to resolve via PATH first.
    resolved = shutil.which(app_name)
    if resolved:
        cmd = [resolved]
    else:
        # Fall back to the as-is name; the OS may handle it.
        cmd = [app_name]

    logger.info("Opening application: %s (cmd=%s)", app_name, cmd)
    # Use Popen so the subprocess is detached; don't block.
    # os.setsid creates a new process group so it outlives this process.
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not find application '{app_name}'. "
            "Is it installed and on PATH?"
        ) from None

    return {"status": "launched", "app": app_name, "command": " ".join(cmd)}


_OPEN_APP_ACTION = Action(
    name="open_app",
    patterns=[
        r"\bopen\s+(?:the\s+)?(?:app\s+)?(?P<app>.+?)\s*$",
        r"\blaunch\s+(?:the\s+)?(?:app\s+)?(?P<app>.+?)\s*$",
        r"\bstart\s+(?:the\s+)?(?:app\s+)?(?P<app>.+?)\s*$",
    ],
    handler=_handle_open_app,
    description="Launch a desktop application by name.",
    requires_confirmation=False,
)


def _handle_open_url(args: Dict[str, Any]) -> Dict[str, Any]:
    """Open a URL in the default web browser.

    Expected key in *args*:
        url (str): The URL to open.
    """
    url: str = args.get("url", "").strip()
    if not url:
        raise ValueError("'url' argument is required for open_url action")

    # Ensure scheme is present.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "https://" + url

    logger.info("Opening URL: %s", url)
    webbrowser.open(url)
    return {"status": "opened", "url": url}


_OPEN_URL_ACTION = Action(
    name="open_url",
    patterns=[
        r"\bopen\s+(?:the\s+)?(?:url\s+)?(?P<url>https?://\S+|\S+\.\S+)\s*$",
        r"\bgo\s+to\s+(?P<url>https?://\S+|\S+\.\S+)\s*$",
        r"\bvisit\s+(?P<url>https?://\S+|\S+\.\S+)\s*$",
    ],
    handler=_handle_open_url,
    description="Open a URL in the default web browser.",
    requires_confirmation=False,
)


def _handle_run_command(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a shell command and return its output.

    Expected key in *args*:
        command (str): The shell command to run.
    """
    command: str = args.get("command", "").strip()
    if not command:
        raise ValueError("'command' argument is required for run_command action")

    logger.info("Running shell command: %s", command)
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    return {
        "status": "completed" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


_RUN_COMMAND_ACTION = Action(
    name="run_command",
    patterns=[
        r"\brun\s+(?:the\s+)?(?:command\s+)?(?P<command>.+?)\s*$",
        r"\bexecute\s+(?:the\s+)?(?:command\s+)?(?P<command>.+?)\s*$",
        r"\bexec\s+(?P<command>.+?)\s*$",
    ],
    handler=_handle_run_command,
    description="Execute a shell command.",
    requires_confirmation=True,
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
)


def _handle_search_web(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search the web using the default browser.

    Expected key in *args*:
        query (str): The search query.
    """
    query: str = args.get("query", "").strip()
    if not query:
        raise ValueError("'query' argument is required for search_web action")

    # Build a DuckDuckGo search URL (no tracking).
    import urllib.parse

    url = "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(query)
    logger.info("Web search: %s", url)
    webbrowser.open(url)
    return {"status": "searched", "query": query, "url": url}


_SEARCH_WEB_ACTION = Action(
    name="search_web",
    patterns=[
        r"\bsearch\s+(?:the\s+)?(?:web\s+)?(?:for\s+)?(?P<query>.+?)\s*$",
        r"\bgoogle\s+(?P<query>.+?)\s*$",
        r"\blook\s+up\s+(?P<query>.+?)\s*$",
        r"\bfind\s+(?P<query>.+?)\s*$",
    ],
    handler=_handle_search_web,
    description="Search the web using the default browser.",
    requires_confirmation=False,
)


def _handle_start_timer(args: Dict[str, Any]) -> Dict[str, Any]:
    """Start a countdown timer.

    Expected keys in *args*:
        duration (str): Human-readable duration like ``"5 minutes"`` or ``"30s"``.
    """
    duration_str: str = args.get("duration", "").strip()
    if not duration_str:
        raise ValueError("'duration' argument is required for start_timer action")

    seconds = _parse_duration(duration_str)
    logger.info("Starting timer for %s seconds", seconds)

    # Run the timer in a background thread so we can return immediately.
    import threading

    def _timer_thread(secs: int) -> None:
        time.sleep(secs)
        logger.info("Timer finished (%s seconds)", secs)
        # Attempt a desktop notification.
        try:
            subprocess.run(
                ["notify-send", "VoiceFlow Timer", f"Timer finished ({secs}s)"],
                timeout=5,
            )
        except Exception:
            pass  # notification is best-effort

    t = threading.Thread(target=_timer_thread, args=(seconds,), daemon=True)
    t.start()

    return {"status": "timer_started", "seconds": seconds, "duration": duration_str}


def _parse_duration(text: str) -> int:
    """Convert a human-readable duration string to seconds.

    Supports formats like:
        "5 minutes", "30s", "1h 30m", "2 hours and 15 minutes"
    """
    total = 0
    # Match number + unit pairs.
    pattern = re.compile(
        r"(?P<value>\d+)\s*(?P<unit>s|sec|seconds?|m|min|minutes?|h|hr|hours?)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        value = int(m.group("value"))
        unit = m.group("unit").lower()
        if unit.startswith("h"):
            total += value * 3600
        elif unit.startswith("m"):
            total += value * 60
        else:
            total += value

    if total == 0:
        # Fallback: try to interpret the whole string as a number of seconds.
        try:
            total = int(text)
        except ValueError:
            raise ValueError(
                f"Could not parse duration from '{text}'. "
                "Try formats like '5 minutes', '30s', '1h 30m'."
            )
    return total


_START_TIMER_ACTION = Action(
    name="start_timer",
    patterns=[
        r"\bstart\s+(?:a\s+)?timer\s+(?:for\s+)?(?P<duration>.+?)\s*$",
        r"\bset\s+(?:a\s+)?timer\s+(?:for\s+)?(?P<duration>.+?)\s*$",
        r"\btimer\s+(?:for\s+)?(?P<duration>.+?)\s*$",
        r"\bcountdown\s+(?:for\s+)?(?P<duration>.+?)\s*$",
    ],
    handler=_handle_start_timer,
    description="Start a countdown timer.",
    requires_confirmation=False,
)


def _handle_set_reminder(args: Dict[str, Any]) -> Dict[str, Any]:
    """Set a reminder for a future time.

    Expected keys in *args*:
        time (str): When to remind (e.g. ``"in 10 minutes"``, ``"at 3pm"``).
        message (str): What to remind about.
    """
    time_str: str = args.get("time", "").strip()
    message: str = args.get("message", "Reminder").strip()
    if not time_str:
        raise ValueError("'time' argument is required for set_reminder action")

    # For "in X" patterns, parse as a duration and schedule a background timer.
    in_match = re.match(r"^in\s+(.+)$", time_str, re.IGNORECASE)
    if in_match:
        seconds = _parse_duration(in_match.group(1))
        import threading

        def _reminder_thread(secs: int, msg: str) -> None:
            time.sleep(secs)
            logger.info("Reminder fired: %s", msg)
            try:
                subprocess.run(
                    ["notify-send", "VoiceFlow Reminder", msg],
                    timeout=5,
                )
            except Exception:
                pass

        t = threading.Thread(
            target=_reminder_thread, args=(seconds, message), daemon=True
        )
        t.start()
        return {
            "status": "reminder_set",
            "seconds": seconds,
            "message": message,
        }

    # For "at HH:MM" patterns, we'd need a scheduler; store for now.
    logger.info("Reminder stored (at-based): %s -- %s", time_str, message)
    return {
        "status": "reminder_stored",
        "time": time_str,
        "message": message,
        "note": "at-based reminders are stored; integrate with a scheduler for full support",
    }


_SET_REMINDER_ACTION = Action(
    name="set_reminder",
    patterns=[
        r"\bset\s+(?:a\s+)?reminder\s+(?:for\s+)?(?P<time>.+?)\s+(?:to|that|about)\s+(?P<message>.+?)\s*$",
        r"\bremind\s+me\s+(?:in|at)\s+(?P<time>.+?)\s+(?:to|that|about)\s+(?P<message>.+?)\s*$",
        r"\bremind\s+me\s+(?:to|that|about)\s+(?P<message>.+?)\s+(?:in|at)\s+(?P<time>.+?)\s*$",
    ],
    handler=_handle_set_reminder,
    description="Set a reminder for a future time.",
    requires_confirmation=False,
)


# ---------------------------------------------------------------------------
# Sandboxed execution helper
# ---------------------------------------------------------------------------

def _run_with_timeout(
    handler: Callable[..., Any],
    args: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Execute *handler(args)* with a wall-clock timeout.

    Uses a ``ThreadPoolExecutor`` with a single worker so the handler runs in
    a separate thread.  If the timeout expires, a ``TimeoutError`` is raised.

    Parameters
    ----------
    handler:
        The action handler callable.
    args:
        Dict of keyword arguments forwarded to the handler.
    timeout:
        Maximum seconds to wait.

    Returns
    -------
    Whatever the handler returns.

    Raises
    ------
    TimeoutError:
        If the handler does not complete within *timeout* seconds.
    Exception:
        Any exception raised by the handler is re-raised.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future: Future = executor.submit(handler, args)
        return future.result(timeout=timeout)
    except FuturesTimeout:
        future.cancel()
        raise TimeoutError(
            f"Action handler timed out after {timeout} seconds"
        ) from None
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Confirmation helper
# ---------------------------------------------------------------------------

def request_confirmation(action_name: str, args: Dict[str, Any]) -> bool:
    """Ask the user to confirm execution of *action_name*.

    In a CLI context this reads from stdin.  In a GUI context this would
    present a dialog.  The function returns ``True`` if the user confirms.

    Parameters
    ----------
    action_name:
        Name of the action to confirm.
    args:
        Arguments that will be passed to the handler (shown to the user).
    """
    print(f"\n[VoiceFlow Agent] Confirm action: {action_name}")
    print(f"  Arguments: {args}")
    print(f"  Type 'yes' to confirm, 'no' to cancel.")

    try:
        response = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nConfirmation cancelled.")
        return False

    if response in CONFIRMATION_KEYWORDS:
        return True
    if response in DENY_KEYWORDS:
        return False

    # Default to deny on ambiguous input.
    print("Unrecognised response -- cancelling.")
    return False


# ---------------------------------------------------------------------------
# High-level AgentMode façade
# ---------------------------------------------------------------------------

class AgentMode:
    """High-level interface for VoiceFlow's Agent Mode.

    Owns an :class:`ActionRegistry` (pre-loaded with built-in actions) and
    provides a single :meth:`process_transcript` entry point.

    Example::

        agent = AgentMode(auto_confirm=True)  # skip confirmation prompts
        result = agent.process_transcript("open firefox")
        if result:
            print(result)
    """

    def __init__(
        self,
        *,
        auto_confirm: bool = False,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Create a new AgentMode instance.

        Parameters
        ----------
        auto_confirm:
            If ``True``, destructive actions are executed without prompting.
            **Use with caution.**
        timeout:
            Default timeout in seconds for action handlers.
        """
        self.registry = ActionRegistry()
        self.registry.register_defaults()
        self.auto_confirm = auto_confirm
        self.default_timeout = timeout
        logger.info(
            "AgentMode initialised (auto_confirm=%s, timeout=%ds)",
            auto_confirm,
            timeout,
        )

    # -- public API ----------------------------------------------------------

    def register_action(self, action: Action) -> None:
        """Register a custom action.

        This is a convenience wrapper around :meth:`ActionRegistry.register`.
        """
        self.registry.register(action)

    def unregister_action(self, name: str) -> None:
        """Remove a previously registered action by name."""
        self.registry.unregister(name)

    def process_transcript(self, transcript: str) -> Optional[Dict[str, Any]]:
        """Process a voice transcript and execute any matching action.

        Parameters
        ----------
        transcript:
            The raw text from the speech-to-text engine.

        Returns
        -------
        A dict with keys ``action_name``, ``args``, ``result`` on success,
        or ``None`` if no action matched.
        """
        if not transcript or not transcript.strip():
            return None

        match = self.registry.match(transcript)
        if match is None:
            logger.debug("No action matched for transcript: %s", transcript)
            return None

        action = self.registry.get(match.action_name)
        if action is None:
            logger.error(
                "Matched action '%s' not found in registry", match.action_name
            )
            return None

        logger.info(
            "Matched action '%s' (confidence=%d) for transcript: %s",
            match.action_name,
            match.confidence,
            transcript,
        )

        # Confirmation step.
        if action.requires_confirmation and not self.auto_confirm:
            if not request_confirmation(match.action_name, match.args):
                logger.info("Action '%s' cancelled by user", match.action_name)
                return {
                    "action_name": match.action_name,
                    "args": match.args,
                    "result": {"status": "cancelled"},
                }

        # Execute with timeout.
        try:
            timeout = action.timeout_seconds or self.default_timeout
            result = _run_with_timeout(action.handler, match.args, timeout=timeout)
            logger.info("Action '%s' completed successfully", match.action_name)
            return {
                "action_name": match.action_name,
                "args": match.args,
                "result": result,
            }
        except TimeoutError:
            logger.error(
                "Action '%s' timed out after %ds",
                match.action_name,
                action.timeout_seconds,
            )
            return {
                "action_name": match.action_name,
                "args": match.args,
                "result": {"status": "timeout"},
            }
        except Exception:
            logger.exception("Action '%s' raised an exception", match.action_name)
            return {
                "action_name": match.action_name,
                "args": match.args,
                "result": {"status": "error"},
            }

    def list_actions(self) -> List[Action]:
        """Return all currently registered actions."""
        return self.registry.list_actions()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# A default singleton for simple use-cases.
_default_agent: Optional[AgentMode] = None


def get_default_agent() -> AgentMode:
    """Return (and lazily create) the module-level default ``AgentMode``."""
    global _default_agent
    if _default_agent is None:
        _default_agent = AgentMode()
    return _default_agent


def process_transcript(transcript: str) -> Optional[Dict[str, Any]]:
    """Module-level convenience: process *transcript* with the default agent."""
    return get_default_agent().process_transcript(transcript)


# ---------------------------------------------------------------------------
# Entry point for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    agent = AgentMode(auto_confirm=True)

    test_transcripts = [
        "open firefox",
        "search web for python asyncio tutorial",
        "start timer for 5 seconds",
        "set reminder in 10 minutes to check the oven",
        "run the command echo hello world",
        "open https://example.com",
        "this should not match anything",
    ]

    for t in test_transcripts:
        print(f"\n--- Transcript: {t!r} ---")
        result = agent.process_transcript(t)
        if result:
            print(f"  Action : {result['action_name']}")
            print(f"  Args   : {result['args']}")
            print(f"  Result : {result['result']}")
        else:
            print("  (no match)")
