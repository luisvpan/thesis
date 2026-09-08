"""Tests for homography matrix computation."""

import numpy as np
import pytest

# Skip tests if cv2 is not available (requires system libraries)
pytest.importorskip("cv2", exc_type=ImportError)

from cv_system.calibration.homography import (
    apply_homography,
    compute_homography,
    create_homography_validation_frame,
    validate_homography,
)


def test_compute_homography_basic() -> None:
    """Test basic homography computation with 4 point pairs."""
    # Simple identity-like transformation
    camera = [(100, 100), (700, 100), (700, 500), (100, 500)]
    projector = [(100, 100), (700, 100), (700, 500), (100, 500)]

    H = compute_homography(camera, projector)

    assert H.shape == (3, 3)
    assert H.dtype == np.float32
    # For identity mapping, H should be close to identity matrix
    assert np.allclose(H, np.eye(3, dtype=np.float32), atol=1e-2)


def test_compute_homography_translation() -> None:
    """Test homography with translation."""
    camera = [(100, 100), (200, 100), (200, 200), (100, 200)]
    # Shift everything by (50, 50)
    projector = [(150, 150), (250, 150), (250, 250), (150, 250)]

    H = compute_homography(camera, projector)

    # Test that (150, 150) maps to (200, 200)
    result = apply_homography((150, 150), H)
    assert result[0] == pytest.approx(200.0, abs=1)
    assert result[1] == pytest.approx(200.0, abs=1)


def test_compute_homography_scaling() -> None:
    """Test homography with scaling."""
    camera = [(100, 100), (200, 100), (200, 200), (100, 200)]
    # Scale by 2x
    projector = [(200, 200), (400, 200), (400, 400), (200, 400)]

    H = compute_homography(camera, projector)

    # Test that (150, 150) maps to (300, 300)
    result = apply_homography((150, 150), H)
    assert result[0] == pytest.approx(300.0, abs=2)
    assert result[1] == pytest.approx(300.0, abs=2)


def test_compute_homography_invalid_point_count() -> None:
    """Test that wrong number of points raises ValueError."""
    camera = [(100, 100), (200, 100), (200, 200)]
    projector = [(150, 150), (250, 150), (250, 250), (150, 250)]

    with pytest.raises(ValueError, match="Exactly 4 points required"):
        compute_homography(camera, projector)


def test_compute_homography_invalid_point_format() -> None:
    """Test that malformed point tuples raise ValueError."""
    camera = [(100,), (200,), (200,), (100,)]  # Missing y coordinate
    projector = [(150, 150), (250, 150), (250, 250), (150, 250)]

    with pytest.raises(ValueError, match="must be 4 \\(x, y\\) pairs"):
        compute_homography(camera, projector)


def test_compute_homography_mismatched_counts() -> None:
    """Test that mismatched point counts raise ValueError."""
    camera = [(100, 100), (200, 100), (200, 200), (100, 200)]
    projector = [(150, 150), (250, 150), (250, 250)]  # Only 3 points

    with pytest.raises(ValueError, match="Exactly 4 points required"):
        compute_homography(camera, projector)


def test_apply_homography_basic() -> None:
    """Test applying homography to a point."""
    camera = [(100, 100), (700, 100), (700, 500), (100, 500)]
    projector = [(100, 100), (700, 100), (700, 500), (100, 500)]

    H = compute_homography(camera, projector)

    # Test origin
    result = apply_homography((0, 0), H)
    assert result == pytest.approx((0.0, 0.0), abs=1)

    # Test a specific point
    result = apply_homography((400, 300), H)
    assert result == pytest.approx((400.0, 300.0), abs=2)


def test_apply_homography_with_translation() -> None:
    """Test applying homography with translation."""
    camera = [(0, 0), (100, 0), (100, 100), (0, 100)]
    projector = [(50, 50), (150, 50), (150, 150), (50, 150)]

    H = compute_homography(camera, projector)

    result = apply_homography((50, 50), H)
    assert result == pytest.approx((100.0, 100.0), abs=1)


def test_validate_homography_valid() -> None:
    """Test validation of a valid homography matrix."""
    camera = [(100, 100), (700, 100), (700, 500), (100, 500)]
    projector = [(100, 100), (700, 100), (700, 500), (100, 500)]

    H = compute_homography(camera, projector)
    assert validate_homography(H) is True


def test_validate_homography_invalid_shape() -> None:
    """Test validation rejects wrong shape."""
    H_invalid = np.eye(2, dtype=np.float32)
    assert validate_homography(H_invalid) is False


def test_validate_homography_invalid_dtype() -> None:
    """Test validation rejects wrong dtype."""
    H_invalid = np.eye(3, dtype=np.float64)
    assert validate_homography(H_invalid) is False


def test_validate_homography_singular() -> None:
    """Test validation rejects singular matrix."""
    # Create a singular matrix (determinant = 0)
    H_singular = np.eye(3, dtype=np.float32)
    H_singular[0, 0] = 0
    assert validate_homography(H_singular) is False


def test_homography_roundtrip() -> None:
    """Test that homography preserves rectangle shape."""
    # Define a rectangle in camera space
    camera_rect = [(100, 100), (700, 100), (700, 500), (100, 500)]

    # Map to projector space with translation and scale
    projector_rect = [(200, 200), (1400, 200), (1400, 1000), (200, 1000)]

    H = compute_homography(camera_rect, projector_rect)

    # Transform each corner
    transformed = [apply_homography(p, H) for p in camera_rect]

    # Check that transformed corners match target
    for i, (tx, ty) in enumerate(transformed):
        expected_x, expected_y = projector_rect[i]
        assert tx == pytest.approx(expected_x, abs=2)
        assert ty == pytest.approx(expected_y, abs=2)


def test_create_homography_validation_frame_identity() -> None:
    """Validation frame should be generated and show near-zero error."""
    camera = [(100, 100), (700, 100), (700, 500), (100, 500)]
    projector = [(100, 100), (700, 100), (700, 500), (100, 500)]

    H = compute_homography(camera, projector)
    frame, metrics = create_homography_validation_frame(
        H,
        camera,
        projector,
        frame_size=(800, 600),
    )

    assert frame.shape == (600, 800, 3)
    assert frame.dtype == np.uint8
    assert metrics["rms_error_px"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["mean_error_px"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["max_error_px"] == pytest.approx(0.0, abs=1e-3)


def test_create_homography_validation_frame_requires_four_points() -> None:
    """Validation frame generator should reject invalid point counts."""
    camera = [(0, 0), (1, 0), (1, 1), (0, 1)]
    projector = [(0, 0), (10, 0), (10, 10), (0, 10)]
    H = compute_homography(camera, projector)

    with pytest.raises(ValueError, match="Exactly 4 points required"):
        create_homography_validation_frame(H, camera[:3], projector)
