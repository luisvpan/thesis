"""Tests for CoordinateTransformer module."""

import numpy as np
import pytest

# Skip tests if cv2 is not available
pytest.importorskip("cv2", exc_type=ImportError)

from cv_system.calibration.result import CalibrationResult
from cv_system.transform import CoordinateTransformer


@pytest.fixture
def identity_calibration() -> CalibrationResult:
    """Create a CalibrationResult with identity homography matrix.

    Using identity matrix makes tests predictable since transformed points
    remain unchanged: (x, y) → (x, y) for both directions.
    """
    H = np.eye(3, dtype=np.float32)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"test": True}
    return CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)


@pytest.fixture
def real_calibration() -> CalibrationResult:
    """Create a CalibrationResult with realistic homography matrix.

    This matrix maps camera space (512×424) to projector space (1920×1080)
    with a reasonable scale and offset.
    """
    # Scale: ~3.75x horizontally, ~2.55x vertically
    # Offset: Small shift to account for alignment
    H = np.array(
        [
            [3.75, 0.0, 10.0],
            [0.0, 2.55, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"test": True}
    return CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)


def test_transformer_initialization(identity_calibration: CalibrationResult) -> None:
    """Test that transformer initializes correctly and stores H and H_inv."""
    transformer = CoordinateTransformer(identity_calibration)

    assert transformer.H.shape == (3, 3)
    assert transformer.H.dtype == np.float32
    assert transformer.H_inv.shape == (3, 3)
    assert transformer.H_inv.dtype == np.float32

    # Verify H_inv is actually the inverse
    expected_inv = np.linalg.inv(identity_calibration.H)
    np.testing.assert_allclose(transformer.H_inv, expected_inv, rtol=1e-5)


def test_camera_to_projector_single_point(
    identity_calibration: CalibrationResult,
) -> None:
    """Test transforming a single point from camera to projector space."""
    transformer = CoordinateTransformer(identity_calibration)

    # Single point with required shape (1,1,2)
    point = np.array([[[100.0, 200.0]]], dtype=np.float32)
    result = transformer.camera_to_projector(point)

    assert result.shape == (1, 1, 2)
    assert result.dtype == np.float32
    # With identity matrix, point should be unchanged
    np.testing.assert_allclose(result, point, rtol=1e-5)


def test_projector_to_camera_single_point(
    identity_calibration: CalibrationResult,
) -> None:
    """Test transforming a single point from projector to camera space."""
    transformer = CoordinateTransformer(identity_calibration)

    # Single point with required shape (1,1,2)
    point = np.array([[[960.0, 540.0]]], dtype=np.float32)
    result = transformer.projector_to_camera(point)

    assert result.shape == (1, 1, 2)
    assert result.dtype == np.float32
    # With identity matrix, point should be unchanged
    np.testing.assert_allclose(result, point, rtol=1e-5)


def test_batch_points(identity_calibration: CalibrationResult) -> None:
    """Test transforming multiple points at once."""
    transformer = CoordinateTransformer(identity_calibration)

    # 5 points with shape (5,1,2)
    points = np.array(
        [
            [[0.0, 0.0]],
            [[100.0, 100.0]],
            [[256.0, 212.0]],
            [[511.0, 423.0]],
            [[50.0, 75.0]],
        ],
        dtype=np.float32,
    )

    result = transformer.camera_to_projector(points)

    assert result.shape == (5, 1, 2)
    assert result.dtype == np.float32
    # With identity matrix, all points should be unchanged
    np.testing.assert_allclose(result, points, rtol=1e-5)


def test_batch_points_projector_to_camera(
    identity_calibration: CalibrationResult,
) -> None:
    """Test transforming multiple points from projector to camera space."""
    transformer = CoordinateTransformer(identity_calibration)

    # 3 projector space points
    points = np.array(
        [
            [[0.0, 0.0]],
            [[960.0, 540.0]],
            [[1919.0, 1079.0]],
        ],
        dtype=np.float32,
    )

    result = transformer.projector_to_camera(points)

    assert result.shape == (3, 1, 2)
    assert result.dtype == np.float32
    # With identity matrix, all points should be unchanged
    np.testing.assert_allclose(result, points, rtol=1e-5)


def test_round_trip(identity_calibration: CalibrationResult) -> None:
    """Test that round-trip transformation (camera→projector→camera) is accurate."""
    transformer = CoordinateTransformer(identity_calibration)

    original = np.array([[[256.0, 212.0]]], dtype=np.float32)

    # Transform camera → projector → camera
    projector_point = transformer.camera_to_projector(original)
    back_to_camera = transformer.projector_to_camera(projector_point)

    # Should return to original point within 1px tolerance
    diff = np.abs(back_to_camera - original)
    assert diff.max() < 1.0, f"Round-trip error {diff.max():.2f} exceeds 1px tolerance"


def test_round_trip_real_matrix(real_calibration: CalibrationResult) -> None:
    """Test round-trip with realistic homography matrix."""
    transformer = CoordinateTransformer(real_calibration)

    # Test multiple points as a batch
    test_points = np.array(
        [
            [[100.0, 100.0]],
            [[256.0, 212.0]],
            [[400.0, 300.0]],
        ],
        dtype=np.float32,
    )

    projector_points = transformer.camera_to_projector(test_points)
    back_to_camera = transformer.projector_to_camera(projector_points)

    diff = np.abs(back_to_camera - test_points)
    assert diff.max() < 1.0, f"Round-trip error {diff.max():.2f} exceeds 1px tolerance"


def test_boundary_points_camera(real_calibration: CalibrationResult) -> None:
    """Test boundary points in camera space (corners of 512×424 frame)."""
    transformer = CoordinateTransformer(real_calibration)

    # Camera space corners: (0,0), (511,0), (0,423), (511,423)
    corners = np.array(
        [
            [[0.0, 0.0]],
            [[511.0, 0.0]],
            [[0.0, 423.0]],
            [[511.0, 423.0]],
        ],
        dtype=np.float32,
    )

    result = transformer.camera_to_projector(corners)

    assert result.shape == (4, 1, 2)
    assert result.dtype == np.float32

    # With the test matrix, points should be in projector space
    # and within reasonable bounds (not necessarily exactly 1920×1080)
    assert np.all(result >= 0), "Transformed coordinates should be non-negative"


def test_boundary_points_projector(real_calibration: CalibrationResult) -> None:
    """Test boundary points in projector space (corners of 1920×1080 frame)."""
    transformer = CoordinateTransformer(real_calibration)

    # Projector space corners: (0,0), (1919,0), (0,1079), (1919,1079)
    corners = np.array(
        [
            [[0.0, 0.0]],
            [[1919.0, 0.0]],
            [[0.0, 1079.0]],
            [[1919.0, 1079.0]],
        ],
        dtype=np.float32,
    )

    result = transformer.projector_to_camera(corners)

    assert result.shape == (4, 1, 2)
    assert result.dtype == np.float32


def test_invalid_dtype_raises(identity_calibration: CalibrationResult) -> None:
    """Test that passing float64 array raises ValueError with clear message."""
    transformer = CoordinateTransformer(identity_calibration)

    # Float64 should be rejected
    point = np.array([[[100.0, 200.0]]], dtype=np.float64)

    with pytest.raises(ValueError) as exc_info:
        transformer.camera_to_projector(point)

    assert "float32" in str(exc_info.value)
    assert "float64" in str(exc_info.value)

    # Same test for reverse transformation
    with pytest.raises(ValueError) as exc_info:
        transformer.projector_to_camera(point)

    assert "float32" in str(exc_info.value)


def test_invalid_shape_missing_dimension(
    identity_calibration: CalibrationResult,
) -> None:
    """Test that passing shape (1,2) instead of (1,1,2) raises ValueError."""
    transformer = CoordinateTransformer(identity_calibration)

    # Missing middle dimension: shape (1,2) instead of (1,1,2)
    point = np.array([[100.0, 200.0]], dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        transformer.camera_to_projector(point)

    assert "3D array" in str(exc_info.value)
    assert "(N,1,2)" in str(exc_info.value)


def test_invalid_shape_wrong_coordinates(
    identity_calibration: CalibrationResult,
) -> None:
    """Test that passing wrong number of coordinates raises ValueError."""
    transformer = CoordinateTransformer(identity_calibration)

    # Wrong coordinates: 3 instead of 2 (shape (1,1,3))
    point = np.array([[[100.0, 200.0, 300.0]]], dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        transformer.camera_to_projector(point)

    assert "2 coordinates" in str(exc_info.value)


def test_realistic_transformation(real_calibration: CalibrationResult) -> None:
    """Test that realistic transformation produces expected scale and offset."""
    transformer = CoordinateTransformer(real_calibration)

    # Center of camera space
    center = np.array([[[256.0, 212.0]]], dtype=np.float32)
    result = transformer.camera_to_projector(center)

    # With H matrix: [[3.75, 0, 10], [0, 2.55, 20], [0, 0, 1]]
    # Expected: x' = 3.75*256 + 10 = 970, y' = 2.55*212 + 20 = 560.6
    expected = np.array([[[970.0, 560.6]]], dtype=np.float32)

    np.testing.assert_allclose(result, expected, rtol=1e-3)
