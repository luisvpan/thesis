"""Touch event JSON schema for Language Runtime communication."""

import logging
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)

TouchEventType = Literal["touch", "touch_down", "touch_move", "touch_up"]


class TouchEvent:
    """
    Touch event structure for Language Runtime WebSocket communication.

    Represents a touch on the table surface, with coordinates in
    projector space (already transformed by CoordinateTransformer).

    Attributes:
        type: Event type - "touch_down", "touch_move", "touch_up", or legacy "touch".
        touch_id: Unique identifier for tracking the same touch across events.
        position: Dict with 'x' and 'y' coordinates in projector space.
        timestamp: ISO 8601 timestamp of when the touch was detected.
    """

    def __init__(
        self,
        x: float,
        y: float,
        timestamp: str,
        event_type: TouchEventType = "touch",
        touch_id: int = 0,
    ):
        """
        Initialize a touch event.

        Args:
            x: X coordinate in projector space (0-1920, will be validated).
            y: Y coordinate in projector space (0-1080, will be validated).
            timestamp: ISO 8601 timestamp string (e.g., "2026-03-25T00:00:00Z").
            event_type: Event type - "touch_down", "touch_move", "touch_up", or "touch".
            touch_id: Unique identifier for tracking the same touch across events.

        Raises:
            ValueError: If coordinates are outside projector bounds or timestamp is invalid.
        """
        # Validate coordinates within projector bounds
        if not (0 <= x < 1920):
            raise ValueError(f"x-coordinate out of bounds: {x} not in [0, 1920)")
        if not (0 <= y < 1080):
            raise ValueError(f"y-coordinate out of bounds: {y} not in [0, 1080)")

        # Validate timestamp format (basic check for ISO 8601)
        try:
            # Parse timestamp to verify it's valid ISO 8601
            datetime.fromisoformat(timestamp)
        except ValueError as e:
            raise ValueError(
                f"Invalid ISO 8601 timestamp: {timestamp}. "
                f"Expected format: YYYY-MM-DDTHH:MM:SS.ffffffZ"
            ) from e

        self.type = event_type
        self.touch_id = touch_id
        self.position = {"x": float(round(x, 2)), "y": float(round(y, 2))}
        self.timestamp = timestamp

        logger.debug(
            f"Created TouchEvent: type={event_type}, id={touch_id}, "
            f"x={x:.2f}, y={y:.2f}"
        )

    @classmethod
    def from_detected_touch(
        cls,
        x: float,
        y: float,
        detected_at: datetime | None = None,
    ) -> "TouchEvent":
        """
        Create TouchEvent from detected touch coordinates.

        Convenience factory for creating touch events from touch detection
        results. Generates timestamp automatically if not provided.

        Args:
            x: X coordinate in projector space.
            y: Y coordinate in projector space.
            detected_at: Detection timestamp. If None, uses current UTC time.

        Returns:
            TouchEvent instance.
        """
        if detected_at is None:
            detected_at = datetime.now(timezone.utc)

        # Format timestamp as ISO 8601 with microseconds
        timestamp_str = detected_at.isoformat(timespec="microseconds")

        return cls(x=x, y=y, timestamp=timestamp_str)

    @classmethod
    def from_tracked_touch(
        cls,
        x: float,
        y: float,
        event_type: TouchEventType,
        touch_id: int,
        detected_at: datetime | None = None,
    ) -> "TouchEvent":
        """
        Create TouchEvent from a tracked touch with state.

        Factory for creating touch events from TouchTracker results.
        Generates timestamp automatically if not provided.

        Args:
            x: X coordinate in projector space.
            y: Y coordinate in projector space.
            event_type: One of "touch_down", "touch_move", "touch_up".
            touch_id: Persistent ID from TouchTracker.
            detected_at: Detection timestamp. If None, uses current UTC time.

        Returns:
            TouchEvent instance with the specified state and ID.
        """
        if detected_at is None:
            detected_at = datetime.now(timezone.utc)

        timestamp_str = detected_at.isoformat(timespec="microseconds")

        return cls(
            x=x,
            y=y,
            timestamp=timestamp_str,
            event_type=event_type,
            touch_id=touch_id,
        )

    def to_dict(self) -> dict:
        """
        Convert TouchEvent to JSON-serializable dictionary.

        Returns:
            Dict with keys: type, touch_id, position, timestamp.
        """
        return {
            "type": self.type,
            "touch_id": self.touch_id,
            "position": self.position,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """
        Convert TouchEvent to JSON string.

        Returns:
            JSON string representation of the touch event.

        Example:
            >>> event = TouchEvent(x=100.0, y=200.0, timestamp="2026-03-25T00:00:00Z")
            >>> event.to_json()
            '{"type": "touch", "position": {"x": 100.0, "y": 200.0}, "timestamp": "2026-03-25T00:00:00Z"}'
        """
        import json

        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TouchEvent(type={self.type}, id={self.touch_id}, "
            f"position=({self.position['x']:.2f}, {self.position['y']:.2f}), "
            f"timestamp={self.timestamp})"
        )
