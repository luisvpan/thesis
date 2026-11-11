"""Tells whether the camera delivered a new color frame.

``get_rgb_frame()`` of the PyKinect2 manager returns the last cached frame when
the sensor has nothing new (PyKinect2 frames arrive asynchronously), so a
stalled color stream looks exactly like a frozen scene. Both hardware managers
expose a counter that only moves when a real frame arrives:

- ``rgb_frame_index``: +1 per new color frame from the camera.
- ``rgb_frame_age_s``: seconds since the last new color frame (inf before the first).

``get_rgb_frame()`` itself is unchanged; callers that do not care ignore these.
"""

from __future__ import annotations

import math
import time


class FrameClock:
    def __init__(self) -> None:
        self.index = 0
        self._last: float | None = None

    def tick(self) -> None:
        self.index += 1
        self._last = time.monotonic()

    @property
    def age_s(self) -> float:
        return math.inf if self._last is None else time.monotonic() - self._last
