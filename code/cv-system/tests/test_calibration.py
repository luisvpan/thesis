"""Tests for CalibrationResult dataclass."""

import numpy as np
import pytest

from cv_system.calibration.result import CalibrationResult


def test_calibration_result_creation() -> None:
    """Test that CalibrationResult can be created with valid inputs."""
    H = np.eye(3, dtype=np.float32)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500, "depth_range": (650, 800)}

    result = CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)

    assert result.H.shape == (3, 3)
    assert result.dmax_map.shape == (424, 512)
    assert result.metadata == metadata


def test_calibration_result_invalid_H_shape() -> None:
    """Test that CalibrationResult rejects non-3x3 homography matrix."""
    H = np.eye(2, dtype=np.float32)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500}

    with pytest.raises(ValueError, match="must be 3x3"):
        CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)


def test_calibration_result_invalid_H_dtype() -> None:
    """Test that CalibrationResult rejects non-float32 homography matrix."""
    H = np.eye(3, dtype=np.float64)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500}

    with pytest.raises(ValueError, match="must be float32"):
        CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)


def test_calibration_result_invalid_dmax_map_dimensions() -> None:
    """Test that CalibrationResult rejects non-2D dmax_map."""
    H = np.eye(3, dtype=np.float32)
    dmax_map = np.zeros((10, 424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500}

    with pytest.raises(ValueError, match="must be 2D"):
        CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)


def test_calibration_result_immutability() -> None:
    """Test that CalibrationResult fields cannot be reassigned."""
    H = np.eye(3, dtype=np.float32)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500}

    result = CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)

    # frozen=True prevents field reassignment
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        result.metadata = {"new_key": "new_value"}


def test_calibration_result_metadata_mutable_warning() -> None:
    """Test that metadata dict contents can be modified (documented limitation).

    Note: While frozen=True prevents field reassignment, the metadata dict itself
    remains mutable. This is a known trade-off for metadata flexibility.
    """
    H = np.eye(3, dtype=np.float32)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500}

    result = CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)

    # This works but is not recommended - treat CalibrationResult as read-only
    result.metadata["frames_captured"] = 600
    assert result.metadata["frames_captured"] == 600


def test_calibration_result_repr() -> None:
    """Test CalibrationResult string representation."""
    H = np.eye(3, dtype=np.float32)
    dmax_map = np.zeros((424, 512), dtype=np.uint16)
    metadata = {"frames_captured": 500, "depth_range": (650, 800)}

    result = CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)

    repr_str = repr(result)
    assert "CalibrationResult" in repr_str
    assert "H_shape=(3, 3)" in repr_str
    assert "dmax_map_shape=(424, 512)" in repr_str
