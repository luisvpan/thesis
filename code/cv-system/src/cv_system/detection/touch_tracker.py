"""
Touch tracker using ByteTrack for persistent touch IDs and stateful events.

Emits touch_down, touch_move, touch_up events with debouncing to filter noise.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import supervision as sv


@dataclass
class TrackedTouch:
    """A tracked touch event with state."""

    id: int
    x: float
    y: float
    state: Literal["down", "move", "up"]


class TouchTracker:
    """
    Tracks touches using ByteTrack and emits stateful events.

    Converts raw touch points into tracked touches with persistent IDs and
    state transitions (down, move, up). Uses debouncing to filter noise.

    Example:
        tracker = TouchTracker(debounce_frames=3)
        events = tracker.update([(100, 200), (300, 400)])
        # Returns: [TrackedTouch(id=1, x=100, y=200, state="down"), ...]
    """

    def __init__(
        self,
        debounce_frames: int = 3,
        touch_radius: float = 15.0,
        lost_track_buffer: int = 5,
        frame_rate: int = 30,
    ) -> None:
        """
        Initialize the touch tracker.

        Args:
            debounce_frames: Number of consecutive frames required to confirm
                a touch_down or touch_up event. Helps filter noise.
            touch_radius: Radius of synthetic bounding box around touch point.
                Used for ByteTrack matching.
            lost_track_buffer: Number of frames ByteTrack retains lost tracks.
            frame_rate: Expected frame rate for ByteTrack motion prediction.
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=0.3,
            frame_rate=frame_rate,
        )
        self._debounce_frames = debounce_frames
        self._touch_radius = touch_radius

        # State tracking for debouncing
        self._pending_downs: dict[int, tuple[int, float, float]] = {}
        # track_id -> (frames_seen, last_x, last_y)

        self._active_tracks: dict[int, tuple[float, float]] = {}
        # track_id -> (last_x, last_y)

        self._pending_ups: dict[int, tuple[int, float, float]] = {}
        # track_id -> (frames_missing, last_x, last_y)

    def update(self, touches: list[tuple[float, float]]) -> list[TrackedTouch]:
        """
        Update tracker with detected touch points.

        Args:
            touches: List of (x, y) coordinates in projector space.

        Returns:
            List of TrackedTouch events:
            - down: New touch confirmed after debounce_frames
            - move: Active touch position updated
            - up: Touch ended after debounce_frames of absence
        """
        events: list[TrackedTouch] = []

        # 1. Convert points to synthetic bounding boxes for ByteTrack
        if touches:
            r = self._touch_radius
            xyxy = np.array(
                [[x - r, y - r, x + r, y + r] for x, y in touches],
                dtype=np.float32,
            )
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=np.ones(len(touches), dtype=np.float32),
                class_id=np.zeros(len(touches), dtype=int),
            )
        else:
            detections = sv.Detections.empty()

        # 2. Update ByteTrack
        tracked = self._tracker.update_with_detections(detections)
        current_ids: set[int] = set()
        if tracked.tracker_id is not None:
            current_ids = {int(tid) for tid in tracked.tracker_id}

        # 3. Process current tracks
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                track_id = int(tid)
                cx = (tracked.xyxy[i, 0] + tracked.xyxy[i, 2]) / 2
                cy = (tracked.xyxy[i, 1] + tracked.xyxy[i, 3]) / 2

                # Cancel pending_up if track reappears
                if track_id in self._pending_ups:
                    del self._pending_ups[track_id]

                if track_id in self._active_tracks:
                    # Already active → emit move
                    self._active_tracks[track_id] = (cx, cy)
                    events.append(TrackedTouch(id=track_id, x=cx, y=cy, state="move"))

                elif track_id in self._pending_downs:
                    # Pending confirmation
                    frames, _, _ = self._pending_downs[track_id]
                    frames += 1
                    self._pending_downs[track_id] = (frames, cx, cy)

                    if frames >= self._debounce_frames:
                        # Confirmed → emit down
                        del self._pending_downs[track_id]
                        self._active_tracks[track_id] = (cx, cy)
                        events.append(
                            TrackedTouch(id=track_id, x=cx, y=cy, state="down")
                        )
                else:
                    # New track → start pending
                    self._pending_downs[track_id] = (1, cx, cy)

        # 4. Detect lost tracks
        for track_id, (last_x, last_y) in list(self._active_tracks.items()):
            if track_id not in current_ids:
                if track_id not in self._pending_ups:
                    # Start pending up with last known position
                    self._pending_ups[track_id] = (1, last_x, last_y)
                else:
                    frames, lx, ly = self._pending_ups[track_id]
                    self._pending_ups[track_id] = (frames + 1, lx, ly)

                frames, lx, ly = self._pending_ups[track_id]
                if frames >= self._debounce_frames:
                    # Confirmed lost → emit up
                    del self._active_tracks[track_id]
                    del self._pending_ups[track_id]
                    events.append(TrackedTouch(id=track_id, x=lx, y=ly, state="up"))

        # 5. Clean up pending_downs that disappeared before confirmation
        for track_id in list(self._pending_downs.keys()):
            if track_id not in current_ids:
                del self._pending_downs[track_id]

        return events

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracker.reset()
        self._pending_downs.clear()
        self._active_tracks.clear()
        self._pending_ups.clear()
