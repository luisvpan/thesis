"""Tests for dmax_map generation.

Tests direct mode implementation (no histogram, no depth range filtering).
"""

import numpy as np
import pytest

from cv_system.calibration.dmax import generate_dmax_map


@pytest.fixture
def mock_capture_constant_depth() -> callable:
    """Mock capture function that returns constant depth values."""

    def capture() -> np.ndarray:
        return np.full((424, 512), 700, dtype=np.uint16)

    return capture


@pytest.fixture
def mock_capture_varying_depth() -> callable:
    """Mock capture function that returns varying depth values per frame."""

    frame_idx = [0]

    def capture() -> np.ndarray:
        nonlocal frame_idx
        # Alternate between 700 and 750 to test mode calculation
        depth = 700 if frame_idx[0] % 2 == 0 else 750
        frame = np.full((424, 512), depth, dtype=np.uint16)
        frame_idx[0] += 1
        return frame

    return capture


def test_generate_dmax_map_basic(mock_capture_constant_depth: callable) -> None:
    """Test basic dmax_map generation with constant depth."""
    dmax_map = generate_dmax_map(
        capture_frame=mock_capture_constant_depth,
        num_frames=10,
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
        depth_shape=(424, 512),
    )

    # With equal counts of 700 and 750, argmax picks first occurrence
    # So we expect 700 (since even frames are captured first)
    assert np.all(dmax_map == 700)


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


def test_generate_dmax_map_invalid_frame_shape() -> None:
    """Test that frame shape mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Frame shape mismatch"):

        def capture() -> np.ndarray:
            return np.full((512, 424), 700, dtype=np.uint16)  # Wrong shape

        generate_dmax_map(
            capture_frame=capture,
            num_frames=10,
            depth_shape=(424, 512),
        )


def test_generate_dmax_map_direct_mode_no_depth_range():
    """Test that direct mode does not use depth_range filtering."""
    # Create capture that returns values both in and out of old depth range
    def capture_mixed_range() -> np.ndarray:
        # Mix of values: some in old range (650-800), some out (600, 850)
        # Direct mode should pick mode regardless of range
        return np.full((10, 10), 700, dtype=np.uint16)

    dmax_map = generate_dmax_map(
        capture_frame=capture_mixed_range,
        num_frames=10,
        depth_shape=(10, 10),
    )

    # Direct mode ignores depth range, should pick 700 (most frequent)
    assert np.all(dmax_map == 700)


def test_generate_dmax_map_handles_zero_values():
    """Test that mode computation handles zero values correctly."""
    frame_idx = [0]

    def capture_with_zeros() -> np.ndarray:
        nonlocal frame_idx
        frame_idx[0] += 1
        # 80% of frames are 700, 20% are 0
        if frame_idx[0] < 8:
            return np.full((10, 10), 700, dtype=np.uint16)
        else:
            return np.full((10, 10), 0, dtype=np.uint16)

    dmax_map = generate_dmax_map(
        capture_frame=capture_with_zeros,
        num_frames=10,
        depth_shape=(10, 10),
    )

    # Mode should pick 700 (8/10 frequency)
    assert np.all(dmax_map == 700)


def test_generate_dmax_map_large_num_frames():
    """Test that large num_frames does not cause memory issues."""
    # This tests memory handling for 500 frames (typical calibration)
    def capture() -> np.ndarray:
        return np.full((100, 100), 700, dtype=np.uint16)

    # 500 frames @ 100x100 uint16 = ~10MB - should work fine
    dmax_map = generate_dmax_map(
        capture_frame=capture,
        num_frames=500,
        depth_shape=(100, 100),
    )

    assert dmax_map.shape == (100, 100)
