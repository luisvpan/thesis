"""Unit tests for TouchEvent JSON schema and serialization."""

import json
from datetime import datetime, timezone, timedelta

import pytest

from cv_system.bridge.touch_event import TouchEvent


class TestTouchEventInit:
    """Tests for TouchEvent initialization and validation."""

    def test_init_valid_touch_event(self):
        """Test creating a valid touch event."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(x=100.5, y=200.75, timestamp=timestamp)

        assert event.type == "touch"
        assert event.position["x"] == 100.5
        assert event.position["y"] == 200.75
        assert event.timestamp == timestamp

    def test_init_custom_event_type(self):
        """Test creating touch event with custom event type."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(
            x=100.0, y=200.0, timestamp=timestamp, event_type="touch"
        )

        assert event.type == "touch"

    def test_init_x_at_lower_bound(self):
        """Test x-coordinate at lower bound (0) is valid."""
        event = TouchEvent(x=0.0, y=500.0, timestamp="2026-03-25T00:00:00Z")

        assert event.position["x"] == 0.0

    def test_init_x_at_upper_bound(self):
        """Test x-coordinate at upper bound (1919.99) is valid."""
        event = TouchEvent(x=1919.99, y=500.0, timestamp="2026-03-25T00:00:00Z")

        assert event.position["x"] == 1919.99

    def test_init_x_out_of_bounds_high(self):
        """Test x-coordinate >= 1920 raises ValueError."""
        with pytest.raises(ValueError, match="x-coordinate out of bounds"):
            TouchEvent(x=1920.0, y=500.0, timestamp="2026-03-25T00:00:00Z")

    def test_init_x_out_of_bounds_low(self):
        """Test x-coordinate < 0 raises ValueError."""
        with pytest.raises(ValueError, match="x-coordinate out of bounds"):
            TouchEvent(x=-1.0, y=500.0, timestamp="2026-03-25T00:00:00Z")

    def test_init_y_at_lower_bound(self):
        """Test y-coordinate at lower bound (0) is valid."""
        event = TouchEvent(x=500.0, y=0.0, timestamp="2026-03-25T00:00:00Z")

        assert event.position["y"] == 0.0

    def test_init_y_at_upper_bound(self):
        """Test y-coordinate at upper bound (1079.99) is valid."""
        event = TouchEvent(x=500.0, y=1079.99, timestamp="2026-03-25T00:00:00Z")

        assert event.position["y"] == 1079.99

    def test_init_y_out_of_bounds_high(self):
        """Test y-coordinate >= 1080 raises ValueError."""
        with pytest.raises(ValueError, match="y-coordinate out of bounds"):
            TouchEvent(x=500.0, y=1080.0, timestamp="2026-03-25T00:00:00Z")

    def test_init_y_out_of_bounds_low(self):
        """Test y-coordinate < 0 raises ValueError."""
        with pytest.raises(ValueError, match="y-coordinate out of bounds"):
            TouchEvent(x=500.0, y=-1.0, timestamp="2026-03-25T00:00:00Z")

    def test_init_valid_iso_8601_timestamp(self):
        """Test valid ISO 8601 timestamp is accepted."""
        timestamp = "2026-03-25T12:34:56.789Z"

        event = TouchEvent(x=100.0, y=200.0, timestamp=timestamp)

        assert event.timestamp == timestamp

    def test_init_invalid_timestamp_raises_error(self):
        """Test invalid timestamp format raises ValueError."""
        invalid_timestamps = [
            "2026-03-25",  # Missing time
            "2026-03-25T12:34:56",  # Missing timezone
            "not-a-timestamp",
            "2026-13-25T12:00:00Z",  # Invalid month
        ]

        for timestamp in invalid_timestamps:
            with pytest.raises(ValueError, match="Invalid ISO 8601 timestamp"):
                TouchEvent(x=100.0, y=200.0, timestamp=timestamp)

    def test_init_coordinate_rounding(self):
        """Test coordinates are rounded to 2 decimal places."""
        event = TouchEvent(x=100.5555, y=200.6666, timestamp="2026-03-25T00:00:00Z")

        assert event.position["x"] == 100.56
        assert event.position["y"] == 200.67

    def test_repr(self):
        """Test string representation for debugging."""
        event = TouchEvent(x=100.5, y=200.75, timestamp="2026-03-25T00:00:00Z")

        repr_str = repr(event)

        assert "TouchEvent" in repr_str
        assert "type=touch" in repr_str
        assert "100.50" in repr_str  # x coordinate
        assert "200.75" in repr_str  # y coordinate
        assert "2026-03-25T00:00:00Z" in repr_str


class TestFromDetectedTouch:
    """Tests for TouchEvent.from_detected_touch() factory method."""

    def test_from_detected_touch_with_timestamp(self):
        """Test creating touch event from detected touch with explicit timestamp."""
        detected_at = datetime(2026, 3, 25, 12, 34, 56, tzinfo=timezone.utc)

        event = TouchEvent.from_detected_touch(x=100.5, y=200.75, detected_at=detected_at)

        assert event.type == "touch"
        assert event.position["x"] == 100.5
        assert event.position["y"] == 200.75
        assert event.timestamp == "2026-03-25T12:34:56Z"

    def test_from_detected_touch_auto_timestamp(self):
        """Test creating touch event auto-generates current timestamp."""
        # Capture time before creating event
        before = datetime.now(timezone.utc)

        event = TouchEvent.from_detected_touch(x=500.0, y=600.0)

        # Capture time after creating event
        after = datetime.now(timezone.utc)

        # Parse the generated timestamp
        event_timestamp = datetime.fromisoformat(event.timestamp)

        # Verify timestamp is between before and after (within reasonable window)
        assert before <= event_timestamp <= after

    def test_from_detected_touch_validates_bounds(self):
        """Test factory method validates coordinate bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            TouchEvent.from_detected_touch(x=2000.0, y=500.0)  # x out of bounds


class TestToDict:
    """Tests for TouchEvent.to_dict() method."""

    def test_to_dict_structure(self):
        """Test to_dict() returns correct structure."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(x=100.5, y=200.75, timestamp=timestamp)
        result = event.to_dict()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"type", "position", "timestamp"}
        assert result["type"] == "touch"
        assert isinstance(result["position"], dict)
        assert set(result["position"].keys()) == {"x", "y"}
        assert result["timestamp"] == timestamp

    def test_to_dict_position_values(self):
        """Test to_dict() includes correct position values."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(x=960.25, y=540.5, timestamp=timestamp)
        result = event.to_dict()

        assert result["position"]["x"] == 960.25
        assert result["position"]["y"] == 540.5


class TestToJson:
    """Tests for TouchEvent.to_json() method."""

    def test_to_json_valid_json(self):
        """Test to_json() produces valid JSON string."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(x=100.5, y=200.75, timestamp=timestamp)
        json_str = event.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)

        assert parsed["type"] == "touch"
        assert parsed["position"]["x"] == 100.5
        assert parsed["position"]["y"] == 200.75
        assert parsed["timestamp"] == timestamp

    def test_to_json_precision(self):
        """Test to_json() preserves 2 decimal places."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(x=100.5555, y=200.6666, timestamp=timestamp)
        json_str = event.to_json()

        parsed = json.loads(json_str)

        # Verify 2 decimal places
        assert parsed["position"]["x"] == 100.56
        assert parsed["position"]["y"] == 200.67

    def test_to_json_matches_dict(self):
        """Test to_json() matches to_dict() output."""
        timestamp = "2026-03-25T00:00:00Z"

        event = TouchEvent(x=500.0, y=600.0, timestamp=timestamp)

        dict_result = event.to_dict()
        json_str = event.to_json()

        parsed_json = json.loads(json_str)

        # JSON should serialize dict exactly
        assert parsed_json == dict_result

    def test_to_json_example(self):
        """Test to_json() produces expected JSON format."""
        timestamp = "2026-03-25T12:34:56.789Z"

        event = TouchEvent(x=123.456, y=789.012, timestamp=timestamp)
        json_str = event.to_json()

        # Verify JSON structure matches Language Runtime expectation
        parsed = json.loads(json_str)

        assert parsed["type"] == "touch"
        assert isinstance(parsed["position"], dict)
        assert "x" in parsed["position"]
        assert "y" in parsed["position"]
        assert isinstance(parsed["timestamp"], str)
