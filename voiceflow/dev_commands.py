"""Developer command mode (Talon-lite).

Voice commands that execute actual computer actions tailored for developers.
Detects project type, runs builds, git operations, tests, linting, and more.
"""

import os
import re
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project type detection
# ---------------------------------------------------------------------------

PROJECT_MARKERS = {
    "node": ["package.json", "node_modules", "yarn.lock", "pnpm-lock.yaml"],
    "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile", ".venv", "venv"],
    "rust": ["Cargo.toml", "Cargo.lock"],
    "go": ["go.mod", "go.sum"],
    "java": ["pom.xml", "build.gradle", "gradlew"],
    "dotnet": [".csproj", ".sln"],
    "flutter": ["pubspec.yaml"],
    "docker": ["Dockerfile", "docker-compose.yml"],
}

PROJECT_BUILD_COMMANDS = {
    "node": {"build": "npm run build", "test": "npm test", "lint": "npm run lint", "dev": "npm run dev"},
    "python": {"build": "python -m build", "test": "python -m pytest", "lint": "ruff check . --fix", "dev": "python -m http.server"},
    "rust": {"build": "cargo build", "test": "cargo test", "lint": "cargo clippy --fix", "dev": "cargo run"},
    "go": {"build": "go build ./...", "test": "go test ./...", "lint": "gofmt -w .", "dev": "go run ."},
    "java": {"build": "mvn compile", "test": "mvn test", "lint": "mvn checkstyle:check", "dev": "mvn spring-boot:run"},
    "dotnet": {"build": "dotnet build", "test": "dotnet test", "lint": "dotnet format", "dev": "dotnet run"},
    "flutter": {"build": "flutter build", "test": "flutter test", "lint": "flutter analyze", "dev": "flutter run"},
    "docker": {"build": "docker compose build", "test": "docker compose run --rm test", "lint": "hadolint Dockerfile", "dev": "docker compose up"},
}


def detect_project_type(directory: str = None) -> str:
    """Detect the project type from files in the directory."""
    if directory is None:
        directory = os.getcwd()

    for project_type, markers in PROJECT_MARKERS.items():
        for marker in markers:
            if os.path.exists(os.path.join(directory, marker)):
                logger.debug("Detected project type: %s (marker: %s)", project_type, marker)
                return project_type

    return "unknown"


def get_project_commands(directory: str = None) -> Dict[str, str]:
    """Get the build/test/lint commands for the current project."""
    project_type = detect_project_type(directory)
    return PROJECT_BUILD_COMMANDS.get(project_type, {})


# ---------------------------------------------------------------------------
# Dev command registry
# ---------------------------------------------------------------------------

@dataclass
class DevAction:
    """A single developer voice command action."""
    name: str
    patterns: List[str]
    handler: Callable[..., Any]
    description: str
    requires_confirmation: bool = False
    timeout_seconds: int = 60


class DevCommandRegistry:
    """Maps voice patterns to developer action functions."""

    def __init__(self, directory: str = None):
        self.directory = directory or os.getcwd()
        self._actions: Dict[str, DevAction] = {}
        self._register_defaults()

    def register(self, action: DevAction):
        """Register a dev action."""
        self._actions[action.name] = action
        logger.debug("Registered dev action: %s", action.name)

    def match(self, transcript: str) -> Tuple[Optional[DevAction], Dict[str, str]]:
        """Match transcript against dev actions. Returns (action, args)."""
        transcript_lower = transcript.lower().strip()

        for name, action in self._actions.items():
            for pattern in action.patterns:
                match = re.search(pattern, transcript_lower)
                if match:
                    args = match.groupdict()
                    # Also extract any remaining text as "query"
                    args["raw"] = transcript
                    return action, args

        return None, {}

    def execute(self, action: DevAction, args: Dict[str, str]) -> Dict[str, Any]:
        """Execute a dev action with its arguments."""
        try:
            result = action.handler(args, self.directory)
            return {"status": "success", "action": action.name, "result": result}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "action": action.name}
        except Exception as e:
            logger.error("Dev action '%s' failed: %s", action.name, e)
            return {"status": "error", "action": action.name, "error": str(e)}

    def _register_defaults(self):
        """Register all built-in dev commands."""
        self.register(DevAction(
            name="run_build",
            patterns=[
                r"\brun\s+(?:the\s+)?build\b",
                r"\bbuild\s+(?:the\s+)?(?:project|app|code)\b",
                r"\bcompile\s+(?:the\s+)?(?:project|code)\b",
            ],
            handler=self._handle_run_build,
            description="Detect project type and run the build command.",
        ))

        self.register(DevAction(
            name="run_test",
            patterns=[
                r"\brun\s+(?:the\s+)?(?:tests?|test suite)\b",
                r"\btest\s+(?:the\s+)?(?:project|code)\b",
                r"\brun\s+pytest\b",
            ],
            handler=self._handle_run_test,
            description="Run the project's test suite.",
        ))

        self.register(DevAction(
            name="fix_lint",
            patterns=[
                r"\bfix\s+(?:the\s+)?lint(?:ing)?\b",
                r"\brun\s+(?:the\s+)?linter\b",
                r"\bauto[\s-]?fix\b",
            ],
            handler=self._handle_fix_lint,
            description="Run linter with auto-fix.",
        ))

        self.register(DevAction(
            name="git_status",
            patterns=[r"\bgit\s+status\b", r"\bcheck\s+git\b"],
            handler=self._handle_git_status,
            description="Run git status.",
        ))

        self.register(DevAction(
            name="git_commit",
            patterns=[
                r"\bgit\s+commit\s+(?:with\s+)?(?:message\s+)?(?P<message>.+?)(?:\s*$)",
            ],
            handler=self._handle_git_commit,
            description="Stage all changes and commit with message.",
            requires_confirmation=True,
        ))

        self.register(DevAction(
            name="git_push",
            patterns=[r"\bgit\s+push\b", r"\bpush\s+(?:to\s+)?(?:remote|origin)\b"],
            handler=self._handle_git_push,
            description="Push commits to remote.",
            requires_confirmation=True,
        ))

        self.register(DevAction(
            name="open_terminal",
            patterns=[
                r"\bopen\s+(?:a\s+)?(?:new\s+)?(?:terminal|shell|console)\b",
                r"\bopen\s+(?:the\s+)?(?:terminal|shell)\s+(?:here|in\s+this\s+dir(?:ectory)?)\b",
            ],
            handler=self._handle_open_terminal,
            description="Open a terminal in the current directory.",
        ))

        self.register(DevAction(
            name="deploy",
            patterns=[
                r"\bdeploy\s+(?:to\s+)?(?:\w+\s+)?(?:server|production|staging)?\b",
                r"\bdeploy\s+(?:the\s+)?(?:app|project|code)\b",
            ],
            handler=self._handle_deploy,
            description="Run the deploy script for this project.",
            requires_confirmation=True,
            timeout_seconds=120,
        ))

    # --- Handlers ---

    def _handle_run_build(self, args, directory):
        project_type = detect_project_type(directory)
        commands = get_project_commands(directory)
        cmd = commands.get("build", "echo 'No build command configured for this project type'")
        return self._run_shell(cmd, directory)

    def _handle_run_test(self, args, directory):
        commands = get_project_commands(directory)
        cmd = commands.get("test", "echo 'No test command configured'")
        return self._run_shell(cmd, directory)

    def _handle_fix_lint(self, args, directory):
        commands = get_project_commands(directory)
        cmd = commands.get("lint", "echo 'No lint command configured'")
        return self._run_shell(cmd, directory)

    def _handle_git_status(self, args, directory):
        return self._run_shell("git status", directory)

    def _handle_git_commit(self, args, directory):
        message = args.get("message", "VoiceFlow auto-commit")
        self._run_shell("git add -A", directory)
        return self._run_shell(f'git commit -m "{message}"', directory)

    def _handle_git_push(self, args, directory):
        return self._run_shell("git push", directory)

    def _handle_open_terminal(self, args, directory):
        import platform
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "cmd"], cwd=directory)
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", "Terminal", directory])
        else:
            subprocess.Popen(["x-terminal-emulator"], cwd=directory)
        return f"Terminal opened in {directory}"

    def _handle_deploy(self, args, directory):
        # Look for deploy script
        deploy_scripts = ["deploy.sh", "Makefile", "scripts/deploy.sh"]
        for script in deploy_scripts:
            path = os.path.join(directory, script)
            if os.path.exists(path):
                if script.endswith(".sh"):
                    return self._run_shell(f"bash {script}", directory)
                elif script == "Makefile":
                    return self._run_shell("make deploy", directory)
        return "No deploy script found. Looked for: deploy.sh, Makefile, scripts/deploy.sh"

    @staticmethod
    def _run_shell(command: str, cwd: str, timeout: int = 60) -> str:
        """Run a shell command and return stdout."""
        logger.info("Running: %s (in %s)", command, cwd)
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n[stderr] {result.stderr.strip()}"
        if not output:
            output = f"Command completed (exit code {result.returncode})"
        return output


# ---------------------------------------------------------------------------
# High-level interface
# ---------------------------------------------------------------------------

class DevCommandMode:
    """High-level interface for developer voice commands."""

    def __init__(self, directory: str = None, auto_confirm: bool = False):
        self.registry = DevCommandRegistry(directory)
        self.auto_confirm = auto_confirm

    def process(self, transcript: str) -> Optional[Dict[str, Any]]:
        """Process a transcript and execute a dev command if matched."""
        action, args = self.registry.match(transcript)
        if action is None:
            return None

        logger.info("Dev command matched: %s", action.name)

        if action.requires_confirmation and not self.auto_confirm:
            return {
                "status": "confirmation_required",
                "action": action.name,
                "description": action.description,
            }

        return self.registry.execute(action, args)

    @property
    def project_type(self) -> str:
        return detect_project_type(self.registry.directory)
