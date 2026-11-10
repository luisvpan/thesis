"""Optional per-stage timing, enabled with ``CV_PROFILE`` (see ``cv_system.flags``).

Every N frames (``CV_PROFILE=1``: 60; ``CV_PROFILE=<N>``: N) it prints the average time of each
stage and the real loop rate, so it is visible how much of a frame the detector
takes and where (CPU stages vs the model on the GPU). Disabled, it costs nothing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from cv_system.flags import DEFAULT_PROFILE_EVERY, profile_every


def profiling_enabled() -> bool:
    return profile_every() > 0


class StageTimer:
    def __init__(
        self,
        name: str,
        enabled: bool | None = None,
        every: int | None = None,
        scope: str = "detector",
    ):
        self.name = name
        self.scope = scope  # what the measured stages add up to, for the report
        self.enabled = profiling_enabled() if enabled is None else enabled
        self.every = every or profile_every() or DEFAULT_PROFILE_EVERY
        self._totals: dict[str, float] = defaultdict(float)
        self._frames = 0
        self._window_start = time.perf_counter()

    @contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals[stage] += time.perf_counter() - start

    def add(self, stage: str, seconds: float) -> None:
        """Add time measured by hand (for code with early returns)."""
        if self.enabled:
            self._totals[stage] += seconds

    def frame_done(self) -> str | None:
        """Count a frame; every ``every`` frames print (and return) the report."""
        if not self.enabled:
            return None
        self._frames += 1
        if self._frames < self.every:
            return None
        now = time.perf_counter()
        wall_ms = (now - self._window_start) * 1000 / self._frames
        measured = sum(self._totals.values()) * 1000 / self._frames
        parts = [f"{k}={v * 1000 / self._frames:.1f}" for k, v in self._totals.items()]
        report = (
            f"[profile {self.name}] ms/frame: {' '.join(parts)} | {self.scope}={measured:.1f} "
            f"| loop={wall_ms:.1f} ({1000 / wall_ms:.1f} fps, "
            f"{wall_ms - measured:.1f} ms outside the {self.scope})"
        )
        print(report, flush=True)
        self._totals.clear()
        self._frames = 0
        self._window_start = now
        return report
