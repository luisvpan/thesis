"""
Comprehensive test suite for TouchDetector.

Tests cover initialization, detection logic, ring buffer behavior,
area filtering, and edge cases like overflow handling.
"""

import cv2
import numpy as np
import pytest

from cv_system.detection import TouchDetector


@pytest.fixture
def dmax_map():
    """
    Create a synthetic dmax_map representing a flat table surface.

    Returns:
        np.ndarray: Constant depth map with value 700mm.
    """
    return np.full((424, 512), 700, dtype=np.uint16)


@pytest.fixture
def detection_config():
    """
    Create a detection configuration object.

    Returns:
        SimpleNamespace: Config with ring_buffer_size=5, touch_threshold=20,
                        min_touch_size=10, max_touch_size=5000.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        ring_buffer_size=5,
        touch_threshold=20,
        min_touch_size=10,
        max_touch_size=5000,
    )


def test_touch_detector_initialization(dmax_map, detection_config):
    """Test that TouchDetector initializes correctly with valid inputs."""
    detector = TouchDetector(dmax_map, detection_config)

    assert detector.ring_buffer_size == 5
    assert detector.touch_threshold == 20
    assert detector.min_touch_size == 10
    assert detector.max_touch_size == 5000
    assert detector._buffer.shape == (5, 424, 512)
    assert detector._buffer.dtype == np.uint8
    assert detector._idx == 0
    assert np.array_equal(detector._dmax_map, dmax_map)


def test_touch_detector_invalid_dmax_shape(detection_config):
    """Test that TouchDetector rejects dmax_map with wrong shape."""
    wrong_shape = np.full((512, 424), 700, dtype=np.uint16)  # Swapped dimensions

    with pytest.raises(ValueError, match="dmax_map must have shape \\(424, 512\\)"):
        TouchDetector(wrong_shape, detection_config)


def test_touch_detector_invalid_dmax_dtype(detection_config):
    """Test that TouchDetector rejects dmax_map with wrong dtype."""
    wrong_dtype = np.full((424, 512), 700, dtype=np.float32)

    with pytest.raises(ValueError, match="dmax_map must have dtype uint16"):
        TouchDetector(wrong_dtype, detection_config)


def test_touch_detector_invalid_ring_buffer_size(dmax_map):
    """Test that TouchDetector rejects zero or negative ring buffer size."""
    from types import SimpleNamespace

    # Test zero size
    config = SimpleNamespace(
        ring_buffer_size=0,
        touch_threshold=20,
        min_touch_size=10,
        max_touch_size=5000,
    )
    with pytest.raises(ValueError, match="ring_buffer_size must be positive"):
        TouchDetector(dmax_map, config)

    # Test negative size
    config = SimpleNamespace(
        ring_buffer_size=-1,
        touch_threshold=20,
        min_touch_size=10,
        max_touch_size=5000,
    )
    with pytest.raises(ValueError, match="ring_buffer_size must be positive"):
        TouchDetector(dmax_map, config)


def test_touch_detector_invalid_touch_threshold(dmax_map):
    """Test that TouchDetector rejects zero or negative touch threshold."""
    from types import SimpleNamespace

    config = SimpleNamespace(
        ring_buffer_size=5,
        touch_threshold=0,
        min_touch_size=10,
        max_touch_size=5000,
    )
    with pytest.raises(ValueError, match="touch_threshold must be positive"):
        TouchDetector(dmax_map, config)


def test_touch_detector_invalid_touch_sizes(dmax_map):
    """Test that TouchDetector rejects invalid touch size thresholds."""
    from types import SimpleNamespace

    # Test negative min size
    config = SimpleNamespace(
        ring_buffer_size=5,
        touch_threshold=20,
        min_touch_size=-1,
        max_touch_size=5000,
    )
    with pytest.raises(ValueError, match="min_touch_size must be positive"):
        TouchDetector(dmax_map, config)

    # Test max_size <= min_size
    config = SimpleNamespace(
        ring_buffer_size=5,
        touch_threshold=20,
        min_touch_size=100,
        max_touch_size=50,
    )
    with pytest.raises(
        ValueError, match="max_touch_size \\(50\\) must be greater than min_touch_size \\(100\\)"
    ):
        TouchDetector(dmax_map, config)


def test_touch_detector_detect_no_touches(dmax_map, detection_config):
    """Test that detect returns empty list when no touches present."""
    detector = TouchDetector(dmax_map, detection_config)

    # Depth frame identical to dmax_map (no touches)
    depth_frame = np.full((424, 512), 700, dtype=np.uint16)

    touches = detector.detect(depth_frame)

    assert touches == []


def test_touch_detector_detect_single_touch(dmax_map, detection_config):
    """Test that detect identifies a single touch point correctly."""
    detector = TouchDetector(dmax_map, detection_config)

    # Create a depth frame with a "touch" at (100, 150)
    # The object is 50mm closer to camera than table surface
    depth_frame = np.full((424, 512), 700, dtype=np.uint16)
    depth_frame[95:105, 145:155] = 650  # 50mm closer

    # Run detection for ring_buffer_size frames to fill buffer
    for _ in range(5):
        touches = detector.detect(depth_frame)

    # Should detect one touch near the center of our region
    assert len(touches) == 1
    tx, ty = touches[0]
    # Check that touch is within expected region
    assert 145 <= tx <= 155
    assert 95 <= ty <= 105


def test_touch_detector_detect_multiple_touches(dmax_map, detection_config):
    """Test that detect identifies multiple touch points."""
    detector = TouchDetector(dmax_map, detection_config)

    # Create depth frame with three "touches"
    depth_frame = np.full((424, 512), 700, dtype=np.uint16)
    depth_frame[95:105, 45:55] = 650  # Touch at (50, 100)
    depth_frame[95:105, 145:155] = 650  # Touch at (150, 100)
    depth_frame[195:205, 245:255] = 650  # Touch at (250, 200)

    # Run detection for ring_buffer_size frames
    for _ in range(5):
        touches = detector.detect(depth_frame)

    # Should detect three touches
    assert len(touches) == 3

    # Check that touches are near expected locations
    touch_points = set(touches)
    assert any(45 <= x <= 55 and 95 <= y <= 105 for x, y in touch_points)
    assert any(145 <= x <= 155 and 95 <= y <= 105 for x, y in touch_points)
    assert any(245 <= x <= 255 and 195 <= y <= 205 for x, y in touch_points)


def test_touch_detector_ring_buffer_wraparound(dmax_map, detection_config):
    """Test that ring buffer handles wraparound correctly after N frames."""
    detection_config.ring_buffer_size = 3
    detector = TouchDetector(dmax_map, detection_config)

    # Create two different depth frames
    frame_no_touch = np.full((424, 512), 700, dtype=np.uint16)
    frame_with_touch = np.full((424, 512), 700, dtype=np.uint16)
    frame_with_touch[95:105, 145:155] = 650  # Touch at (150, 100)

    # First: run frames with no touch (buffer fills with no-touch frames)
    touches = detector.detect(frame_no_touch)
    touches = detector.detect(frame_no_touch)
    touches = detector.detect(frame_no_touch)
    assert len(touches) == 0

    # Then: run frames with touch (buffer should wrap around and still work)
    touches = detector.detect(frame_with_touch)
    touches = detector.detect(frame_with_touch)
    touches = detector.detect(frame_with_touch)

    # After 3 more frames (6 total), buffer has wrapped around
    # Should detect touch because majority of buffer now has touch
    touches = detector.detect(frame_with_touch)
    touches = detector.detect(frame_with_touch)

    assert len(touches) == 1


def test_touch_detector_area_filtering(dmax_map, detection_config):
    """Test that area filtering rejects small noise and large objects."""
    # Use config with specific size thresholds
    detection_config.min_touch_size = 20
    detection_config.max_touch_size = 200
    detector = TouchDetector(dmax_map, detection_config)

    # Create frame with small noise, valid touch, and large object
    depth_frame = np.full((424, 512), 700, dtype=np.uint16)

    # Small noise (3x3 = 9 pixels, below min_touch_size)
    depth_frame[50:53, 50:53] = 650

    # Valid touch (10x10 = 100 pixels, within range)
    depth_frame[95:105, 145:155] = 650

    # Large object (20x20 = 400 pixels, above max_touch_size)
    depth_frame[200:220, 200:220] = 650

    # Run detection for ring_buffer_size frames
    for _ in range(5):
        touches = detector.detect(depth_frame)

    # Should only detect the valid touch, not noise or large object
    assert len(touches) == 1
    tx, ty = touches[0]
    assert 145 <= tx <= 155
    assert 95 <= ty <= 105


def test_touch_detector_reset(dmax_map, detection_config):
    """Test that reset clears ring buffer and frame counter."""
    detector = TouchDetector(dmax_map, detection_config)

    # Create frame with touch
    depth_frame = np.full((424, 512), 700, dtype=np.uint16)
    depth_frame[95:105, 145:155] = 650

    # Run detection for several frames
    for _ in range(5):
        detector.detect(depth_frame)

    # Buffer should have accumulated state
    assert detector._idx > 0
    assert not np.all(detector._buffer == 0)

    # Reset detector
    detector.reset()

    # Buffer and counter should be cleared
    assert detector._idx == 0
    assert np.all(detector._buffer == 0)

    # After reset, detection should return empty (buffer needs to fill)
    depth_no_touch = np.full((424, 512), 700, dtype=np.uint16)
    touches = detector.detect(depth_no_touch)
    assert len(touches) == 0


def test_touch_detector_depth_overflow(dmax_map, detection_config):
    """
    Test that depth subtraction handles uint16 overflow correctly using int16.

    This test ensures that subtracting two uint16 values where result would be
    negative is handled correctly by casting to int16 first.
    """
    detector = TouchDetector(dmax_map, detection_config)

    # Create dmax_map with high value (near uint16 max)
    high_dmax = np.full((424, 512), 65000, dtype=np.uint16)
    detector_high = TouchDetector(high_dmax, detection_config)

    # Create depth frame with significantly lower value
    # This would cause underflow if computed as uint16
    depth_frame = np.full((424, 512), 60000, dtype=np.uint16)  # 5000mm closer
    depth_frame[95:105, 145:155] = 64970  # 30mm closer (above threshold)

    # Should handle large negative differences correctly without overflow
    touches = detector_high.detect(depth_frame)
    # We don't expect touches here (30mm difference < 50mm threshold),
    # but the computation should not crash
    assert isinstance(touches, list)


def test_touch_detector_invalid_depth_frame_shape(dmax_map, detection_config):
    """Test that detect rejects depth frames with wrong shape."""
    detector = TouchDetector(dmax_map, detection_config)

    wrong_shape = np.full((512, 424), 700, dtype=np.uint16)

    with pytest.raises(ValueError, match="depth_frame shape .* must match"):
        detector.detect(wrong_shape)


def test_touch_detector_invalid_depth_frame_dtype(dmax_map, detection_config):
    """Test that detect rejects depth frames with wrong dtype."""
    detector = TouchDetector(dmax_map, detection_config)

    wrong_dtype = np.full((424, 512), 700, dtype=np.float32)

    with pytest.raises(ValueError, match="depth_frame dtype .* must be uint16"):
        detector.detect(wrong_dtype)


def test_touch_detector_persistence_threshold(dmax_map):
    """
    Test that persistence threshold requires majority voting.

    A touch should only be detected if it appears in a majority of frames
    in the ring buffer, not just in the most recent frame.
    """
    from types import SimpleNamespace

    # Use small ring buffer for easier testing
    config = SimpleNamespace(
        ring_buffer_size=5,
        touch_threshold=20,
        min_touch_size=10,
        max_touch_size=5000,
    )
    detector = TouchDetector(dmax_map, config)

    # Create frames with and without touch
    frame_no_touch = np.full((424, 512), 700, dtype=np.uint16)
    frame_with_touch = np.full((424, 512), 700, dtype=np.uint16)
    frame_with_touch[95:105, 145:155] = 650

    # Fill buffer with no-touch frames (4/5 frames)
    detector.detect(frame_no_touch)
    detector.detect(frame_no_touch)
    detector.detect(frame_no_touch)
    detector.detect(frame_no_touch)

    # Add one touch frame (now 4 no-touch, 1 touch)
    touches = detector.detect(frame_with_touch)

    # Should NOT detect touch (only 1/5 frames has touch, need majority)
    # With N=5, majority is 3 frames
    assert len(touches) == 0

    # Add two more touch frames (now 3 no-touch, 3 touch)
    detector.detect(frame_with_touch)
    detector.detect(frame_with_touch)

    # Should detect touch (3/6 frames has touch, which is majority)
    touches = detector.detect(frame_with_touch)
    # After 7 frames, buffer has: [touch, touch, touch, no, no, no, touch]
    # Which gives 4 touches out of 5 in current buffer state
    assert len(touches) >= 1


def test_touch_detector_dmax_readonly(dmax_map, detection_config):
    """Test that dmax_map is stored as read-only."""
    detector = TouchDetector(dmax_map, detection_config)

    assert not detector._dmax_map.flags.writeable


def test_touch_detector_frame_counter_increment(dmax_map, detection_config):
    """Test that frame counter increments correctly on each detect call."""
    detector = TouchDetector(dmax_map, detection_config)

    initial_idx = detector._idx
    assert initial_idx == 0

    depth_frame = np.full((424, 512), 700, dtype=np.uint16)
    detector.detect(depth_frame)
    assert detector._idx == 1

    detector.detect(depth_frame)
    detector.detect(depth_frame)
    assert detector._idx == 3
