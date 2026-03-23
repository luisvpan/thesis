"""Tests for dmax_map generation."""

import numpy as np
import pytest

from cv_system.calibration.dmax import compute_depth_stats, generate_dmax_map


@pytest.fixture
def mock_capture_constant_depth() -> callable:
    """Mock capture function that returns constant depth values."""

    def capture() -> np.ndarray:
        return np.full((424, 512), 700, dtype=np.uint16)

    return capture


@pytest.fixture
def mock_capture_varying_depth() -> callable:
    """Mock capture function that returns varying depth values per frame."""

    frame_idx = 0

    def capture() -> np.ndarray:
        nonlocal frame_idx
        # Alternate between 700 and 750 to test mode calculation
        depth = 700 if frame_idx % 2 == 0 else 750
        frame = np.full((424, 512), depth, dtype=np.uint16)
        frame_idx += 1
        return frame

    return capture


def test_generate_dmax_map_basic(mock_capture_constant_depth: callable) -> None:
    """Test basic dmax_map generation with constant depth."""
    dmax_map = generate_dmax_map(
        capture_frame=mock_capture_constant_depth,
        num_frames=10,
        depth_range=(650, 800),
        depth_shape=(424, 512),
    )

    assert dmax_map.shape == (424, 512)
    assert dmax_map.dtype == np.uint16
    # All pixels should be 700 (the constant depth)
    assert np.all(dmax_map == 700)


def test_generate_dmax_map_varying_depth(mock_capture_varying_depth: callable) -> None:
    """Test dmax_map generation with alternating depth values."""
    dmax_map = generate_dmax_map(
        capture_frame=mock_capture_varying_depth,
        num_frames=20,
        depth_range=(650, 800),
        depth_shape=(424, 512),
    )

    # With equal counts of 700 and 750, argmax picks the first occurrence
    # So we expect 700 (since even frames are captured first)
    assert np.all(dmax_map == 700)


def test_generate_dmax_map_depth_range_filtering() -> None:
    """Test that pixels outside depth range are not counted."""

    def capture() -> np.ndarray:
        # Mix of values: some in range (700), some out (600, 850)
        frame = np.full((10, 10), 700, dtype=np.uint16)
        frame[0:3, 0:3] = 600  # Below range
        frame[7:10, 7:10] = 850  # Above range
        return frame

    dmax_map = generate_dmax_map(
        capture_frame=capture,
        num_frames=10,
        depth_range=(650, 800),
        depth_shape=(10, 10),
    )

    # Pixels in range should be 700
    assert np.all(dmax_map[3:7, 3:7] == 700)
    # Pixels out of range should be 0 (invalid)
    assert np.all(dmax_map[0:3, 0:3] == 0)
    assert np.all(dmax_map[7:10, 7:10] == 0)


def test_generate_dmax_map_invalid_depth_shape() -> None:
    """Test that invalid depth_shape raises ValueError."""
    with pytest.raises(ValueError, match="must be 2D"):

        def capture() -> np.ndarray:
            return np.full((100, 50, 50), 700, dtype=np.uint16)

        generate_dmax_map(
            capture_frame=capture,
            num_frames=10,
            depth_shape=(100, 50, 50),  # Invalid: 3D
        )


def test_generate_dmax_map_invalid_depth_range() -> None:
    """Test that invalid depth_range raises ValueError."""

    def capture() -> np.ndarray:
        return np.full((424, 512), 700, dtype=np.uint16)

    with pytest.raises(ValueError, match="min must be less than max"):
        generate_dmax_map(
            capture_frame=capture,
            num_frames=10,
            depth_range=(800, 650),  # Invalid: min >= max
        )


def test_generate_dmax_map_frame_shape_mismatch() -> None:
    """Test that frame shape mismatch raises ValueError."""

    def capture() -> np.ndarray:
        return np.full((512, 424), 700, dtype=np.uint16)  # Wrong shape

    with pytest.raises(ValueError, match="Frame shape mismatch"):
        generate_dmax_map(
            capture_frame=capture,
            num_frames=10,
            depth_shape=(424, 512),
        )


def test_compute_depth_stats_valid() -> None:
    """Test compute_depth_stats with valid dmax_map."""
    dmax_map = np.full((100, 100), 700, dtype=np.uint16)

    stats = compute_depth_stats(dmax_map, depth_range=(650, 800))

    assert stats["mean"] == 700.0
    assert stats["std"] == 0.0
    assert stats["min"] == 700.0
    assert stats["max"] == 700.0
    assert stats["valid_pixel_ratio"] == 1.0


def test_compute_depth_stats_partial_valid() -> None:
    """Test compute_depth_stats with some invalid pixels."""
    dmax_map = np.full((100, 100), 700, dtype=np.uint16)
    dmax_map[0:10, :] = 0  # Invalid pixels

    stats = compute_depth_stats(dmax_map, depth_range=(650, 800))

    assert stats["mean"] == 700.0
    assert stats["valid_pixel_ratio"] == 0.9


def test_compute_depth_stats_all_invalid() -> None:
    """Test compute_depth_stats with all invalid pixels."""
    dmax_map = np.zeros((100, 100), dtype=np.uint16)

    stats = compute_depth_stats(dmax_map, depth_range=(650, 800))

    assert stats["mean"] == 0.0
    assert stats["valid_pixel_ratio"] == 0.0
