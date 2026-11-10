"""The two environment switches for diagnostics, shared by every entry point.

- ``CV_DEBUG=1``: debug windows (cv2.imshow) and drawing boxes on the card view.
  Off in normal use: drawing and showing frames costs several ms per frame.
- ``CV_PROFILE=1``: per-stage timings printed every 60 frames; ``CV_PROFILE=<N>``
  prints every N frames. Independent of ``CV_DEBUG``, so the production
  configuration (no windows) can be measured too.
"""

from __future__ import annotations

import os

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")
DEFAULT_PROFILE_EVERY = 60
_warned_old_debug = False


def debug_enabled() -> bool:
    global _warned_old_debug
    value = os.getenv("CV_DEBUG")
    if value is None and os.getenv("CV_DEBUG_WINDOWS") is not None:
        value = os.getenv("CV_DEBUG_WINDOWS")
        if not _warned_old_debug:
            print("  NOTE: CV_DEBUG_WINDOWS is now CV_DEBUG; please rename it in .env")
            _warned_old_debug = True
    return (value or "").strip().lower() in _TRUE


def profile_every() -> int:
    """Frames between profiling reports; 0 = profiling off."""
    value = os.getenv("CV_PROFILE", "").strip().lower()
    if value in _FALSE:
        return 0
    if value in _TRUE:
        return DEFAULT_PROFILE_EVERY
    try:
        return max(1, int(value))
    except ValueError:
        return DEFAULT_PROFILE_EVERY
