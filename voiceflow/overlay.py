"""Distraction-Free Dictation Overlay.

A small, semi-transparent floating window that shows live transcription
preview while the user speaks.  Built on tkinter (stdlib only).

Features:
    - 400 px semi-transparent window, bottom-centred by default
    - Green text for confirmed words, yellow for preview/processing
    - Enter -> confirm & type, Escape -> cancel & re-record
    - Configurable position, opacity, font, colours
    - Blocking ``run()`` or non-blocking ``start(callback)`` mode

Usage (blocking):
    overlay = DictationOverlay()
    confirmed_text = overlay.run(initial_text="hello world")

Usage (non-blocking with callback):
    overlay = DictationOverlay()
    overlay.start(lambda text: print(f"Confirmed: {text}"))
    # elsewhere, push partial text:
    overlay.update_text("hello again", confirmed=False)
"""

import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIRMED_COLOUR = "#4AF06E"   # green
_DEFAULT_PREVIEW_COLOUR   = "#F0E04A"   # yellow
_DEFAULT_BG_COLOUR       = "#1E1E2E"   # dark background

_DEFAULT_WIDTH           = 400
_DEFAULT_HEIGHT          = 80
_DEFAULT_OPACITY         = 0.85
_DEFAULT_FONT_FAMILY     = "Segoe UI"
_DEFAULT_FONT_SIZE       = 14
_DEFAULT_PADDING         = 12
_DEFAULT_BOTTOM_MARGIN   = 60   # pixels from bottom of screen
_MAX_DISPLAY_CHARS       = 200  # truncate displayed text


# ---------------------------------------------------------------------------
# Overlay class
# ---------------------------------------------------------------------------

class DictationOverlay:
    """Floating dictation preview window.

    All tkinter calls happen on a single thread.  In *blocking* mode the
    constructor spawns a short-lived Tk mainloop; in *non-blocking* mode a
    dedicated daemon thread runs the mainloop so the caller keeps control.

    Args:
        x:               Window x position (None = auto-centre horizontally).
        y:               Window y position (None = auto bottom-centre).
        width:           Window width in pixels.
        height:          Window height in pixels.
        opacity:         0.0 (fully transparent) to 1.0 (opaque).
        font_family:     Tk font family string.
        font_size:       Font size in points.
        confirmed_colour: Hex colour for confirmed text.
        preview_colour:   Hex colour for preview/processing text.
        bg_colour:       Window background hex colour.
        padding:         Inner padding in pixels.
        bottom_margin:   Distance from bottom of screen in pixels (y=None).
    """

    def __init__(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        opacity: float = _DEFAULT_OPACITY,
        font_family: str = _DEFAULT_FONT_FAMILY,
        font_size: int = _DEFAULT_FONT_SIZE,
        confirmed_colour: str = _DEFAULT_CONFIRMED_COLOUR,
        preview_colour: str = _DEFAULT_PREVIEW_COLOUR,
        bg_colour: str = _DEFAULT_BG_COLOUR,
        padding: int = _DEFAULT_PADDING,
        bottom_margin: int = _DEFAULT_BOTTOM_MARGIN,
    ):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._opacity = max(0.1, min(1.0, opacity))
        self._font = (font_family, font_size)
        self._confirmed_colour = confirmed_colour
        self._preview_colour = preview_colour
        self._bg_colour = bg_colour
        self._padding = padding
        self._bottom_margin = bottom_margin

        # State
        self._result: Optional[str] = None
        self._root = None
        self._callback: Optional[Callable[[str], None]] = None

        logger.debug(
            "DictationOverlay created (w=%d, h=%d, opacity=%.2f)",
            self._width, self._height, self._opacity,
        )

    # ---- public API: blocking mode ----------------------------------------

    def run(self, initial_text: str = "") -> Optional[str]:
        """Show the overlay and block until the user confirms or cancels.

        Args:
            initial_text: Text to display when the overlay opens.

        Returns:
            The confirmed text, or None if the user cancelled.
        """
        self._result = None
        self._callback = None

        self._build(initial_text)
        logger.info("DictationOverlay blocking run started")

        try:
            self._root.mainloop()
        except Exception:
            logger.exception("Overlay mainloop crashed")

        logger.info("DictationOverlay closed (result=%s)",
                     "cancelled" if self._result is None else "confirmed")
        return self._result

    # ---- public API: non-blocking mode ------------------------------------

    def start(
        self, callback: Callable[[str], None], initial_text: str = ""
    ) -> None:
        """Show the overlay on a background thread and invoke callback on confirm.

        Args:
            callback:     Called with the confirmed text (or empty string on
                          callback is dispatched on the tk thread.
            initial_text: Text to display when the overlay opens.
        """
        self._result = None
        self._callback = callback

        thread = threading.Thread(
            target=self._run_nonblocking,
            args=(initial_text,),
            daemon=True,
        )
        thread.start()
        logger.info(
            "DictationOverlay non-blocking start (thread=%s)", thread.name
        )

    def update_text(self, text: str, confirmed: bool = False) -> None:
        """Push new text into the overlay from producer code.

        Thread-safe: schedules the update on the tk event loop.

        Args:
            text:      New display text.
            confirmed: If True, render in confirmed (green) colour.
        """
        if self._root is None:
            logger.warning("update_text called but overlay not built yet")
            return

        colour = self._confirmed_colour if confirmed else self._preview_colour
        display = text[:_MAX_DISPLAY_CHARS]

        def _apply():
            try:
                if self._text_var is not None:
                    self._text_var.set(display)
                if self._label is not None:
                    self._label.configure(fg=colour)
                if self._status_bar is not None:
                    self._status_bar.configure(bg=colour)
            except Exception:
                pass  # root may have been destroyed

        try:
            self._root.after(0, _apply)
        except Exception:
            pass

    def close(self) -> None:
        """Programmatically close the overlay."""
        try:
            if self._root is not None:
                self._root.after(0, self._do_close)
        except Exception:
            pass

    # ---- internal: geometry -----------------------------------------------

    @staticmethod
    def _get_screen_size():
        """Return (screen_width, screen_height) using a temporary Tk root."""
        import tkinter as tk
        probe = tk.Tk()
        probe.withdraw()
        w = probe.winfo_screenwidth()
        h = probe.winfo_screenheight()
        probe.destroy()
        return w, h

    def _geometry_string(self) -> str:
        """Compute the WxH+X+Y geometry string."""
        try:
            screen_w, screen_h = self._get_screen_size()
        except Exception:
            screen_w, screen_h = 1920, 1080

        x = self._x if self._x is not None else (screen_w - self._width) // 2
        y = (
            self._y
            if self._y is not None
            else screen_h - self._height - self._bottom_margin
        )
        return f"{self._width}x{self._height}+{x}+{y}"

    # ---- internal: widget construction ------------------------------------

    def _build(self, initial_text: str) -> None:
        """Create all tkinter widgets."""
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title("VoiceFlow")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", self._opacity)
        self._root.configure(bg=self._bg_colour)
        self._root.geometry(self._geometry_string())

        # Truncated initial display
        display_text = initial_text[:_MAX_DISPLAY_CHARS]
        self._text_var = tk.StringVar(value=display_text)

        self._label = tk.Label(
            self._root,
            textvariable=self._text_var,
            fg=self._preview_colour,
            bg=self._bg_colour,
            font=self._font,
            anchor="w",
            justify="left",
            padx=self._padding,
            pady=self._padding,
            wraplength=self._width - 2 * self._padding,
        )
        self._label.pack(fill="both", expand=True)

        # Thin coloured status bar at bottom
        self._status_bar = tk.Frame(
            self._root, bg=self._preview_colour, height=3
        )
        self._status_bar.pack(fill="x", side="bottom")

        # Hint label (small, bottom-right)
        self._hint_var = tk.StringVar(value="Enter=confirm  Escape=cancel")
        self._hint_label = tk.Label(
            self._root,
            textvariable=self._hint_var,
            fg="#666666",
            bg=self._bg_colour,
            font=(self._font[0], 8),
            anchor="e",
        )
        self._hint_label.place(relx=1.0, rely=1.0, x=-4, y=-4, anchor="se")

        # Key bindings
        self._root.bind("<Return>", self._on_confirm)
        self._root.bind("<KP_Enter>", self._on_confirm)  # numpad Enter
        self._root.bind("<Escape>", self._on_cancel)
        self._root.bind("<Button-1>", lambda _e: self._root.focus_set())

        logger.debug("Overlay widgets built")

    # ---- internal: event handlers -----------------------------------------

    def _on_confirm(self, _event=None) -> None:
        """Enter key: capture text and close."""
        try:
            self._result = self._text_var.get() if self._text_var else ""
        except Exception:
            self._result = ""
        logger.debug("Overlay confirmed: %r", self._result)
        self._do_close()

    def _on_cancel(self, _event=None) -> None:
        """Escape key: set result to None and close."""
        self._result = None
        logger.debug("Overlay cancelled")
        self._do_close()

    def _do_close(self) -> None:
        """Destroy the tk root; safe to call from any thread via after()."""
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass
        self._root = None

    # ---- internal: non-blocking thread entry ------------------------------

    def _run_nonblocking(self, initial_text: str) -> None:
        """Thread target: build UI, run mainloop, then invoke callback."""
        try:
            self._build(initial_text)
            self._root.mainloop()
        except Exception:
            logger.exception("Non-blocking overlay thread crashed")
        finally:
            # Invoke callback on confirm / cancel
            if self._callback is not None:
                try:
                    self._callback(self._result or "")
                except Exception:
                    logger.exception("Overlay callback raised")
            self._root = None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def show_overlay(text: str = "", **kwargs) -> Optional[str]:
    """One-liner blocking overlay.

    Args:
        text:    Initial text to display.
        **kwargs: Forwarded to DictationOverlay constructor.

    Returns:
        Confirmed text or None.
    """
    return DictationOverlay(**kwargs).run(initial_text=text)
