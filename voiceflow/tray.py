"""System tray icon with status indicator and menu.

States: idle (gray), recording (red), processing (green/yellow).

Menu: Status display, device select, model select, LLM toggle, settings, quit.
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AppState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


class TrayApp:
    """
    System tray application showing VoiceFlow status.

    Usage:
        tray = TrayApp(on_quit=exit_callback, on_toggle_llm=llm_callback)
        tray.update_state(AppState.RECORDING)
        tray.run()  # blocks until quit
    """

    def __init__(self, on_quit=None, on_toggle_llm=None, on_toggle_recording=None):
        self.on_quit = on_quit
        self.on_toggle_llm = on_toggle_llm
        self.on_toggle_recording = on_toggle_recording
        self._state = AppState.IDLE
        self._llm_enabled = False
        self._icon = None
        self._menu = None

    def run(self):
        """Start the tray icon (blocks)."""
        try:
            import pystray
            from PIL import Image, ImageDraw

            self._icon = pystray.Icon(
                "voiceflow",
                icon=self._create_icon(AppState.IDLE),
                title="VoiceFlow - Ready",
                menu=self._build_menu(),
            )
            logger.info("System tray icon started")
            self._icon.run()

        except Exception as e:
            logger.error("Failed to start tray icon: %s", e)
            logger.info("Tray icon requires a GUI. Running without tray.")
            # Don't crash the app if no GUI available

    def update_state(self, state: AppState):
        """Update the tray icon state and tooltip."""
        self._state = state

        if self._icon is None:
            return

        try:
            self._icon.icon = self._create_icon(state)
            self._icon.title = f"VoiceFlow - {state.value.capitalize()}"
            self._icon.menu = self._build_menu()
        except Exception as e:
            logger.debug("Failed to update tray icon: %s", e)

    def update_llm_status(self, enabled: bool):
        """Update LLM status in tray menu."""
        self._llm_enabled = enabled
        if self._icon:
            try:
                self._icon.menu = self._build_menu()
            except Exception as e:
                logger.debug("Failed to update menu: %s", e)

    def stop(self):
        """Stop the tray icon."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass

    def _create_icon(self, state: AppState):
        """Create a simple colored circle icon representing the state."""
        from PIL import Image, ImageDraw

        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        colors = {
            AppState.IDLE: (128, 128, 128, 255),       # Gray
            AppState.RECORDING: (255, 60, 60, 255),     # Red
            AppState.PROCESSING: (80, 200, 120, 255),   # Green
            AppState.ERROR: (255, 165, 0, 255),         # Orange
        }
        color = colors.get(state, colors[AppState.IDLE])

        # Draw filled circle
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color,
            outline=(255, 255, 255, 200),
            width=3,
        )

        return img

    def _build_menu(self):
        """Build the right-click context menu."""
        import pystray

        state_label = f"Status: {self._state.value.capitalize()}"
        llm_label = f"LLM Cleanup: {'On' if self._llm_enabled else 'Off'}"

        items = [
            pystray.MenuItem(state_label, enabled=False),
            pystray.MenuItem(llm_label, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Toggle Recording", self._on_toggle_recording
            ) if self.on_toggle_recording else pystray.MenuItem(
                "Toggle Recording", enabled=False
            ),
            pystray.MenuItem(
                "Toggle LLM", self._on_toggle_click
            ) if self.on_toggle_llm else pystray.MenuItem(
                "Toggle LLM", enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings...", self._on_settings),
            pystray.MenuItem("Quit", self._tray_quit),
        ]

        return pystray.Menu(*items)

    def _tray_quit(self, icon, item):
        if self.on_quit:
            self.on_quit()
        self.stop()

    def _on_toggle_click(self, icon, item):
        if self.on_toggle_llm:
            self.on_toggle_llm()

    def _on_toggle_recording(self, icon, item):
        if self.on_toggle_recording:
            self.on_toggle_recording()

    def _on_settings(self, icon, item):
        logger.info("Settings dialog not yet implemented")
        # Future: open settings dialog
