"""Tests for Calibrator orchestration."""

import numpy as np
import pytest

# Skip tests if cv2 is not available (requires system libraries)
pytest.importorskip("cv2", exc_type=ImportError)

from cv_system.calibration.calibrator import Calibrator
from cv_system.calibration.result import CalibrationResult


@pytest.fixture
def mock_calibration_config():
    """Create a mock calibration config object."""

    class MockCalibration:
        def __init__(self):
            self.camera_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
            self.projector_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
            self.dmax_num_frames = 10
            self.depth_range_min = 650
            self.depth_range_max = 800

    class MockConfig:
        def __init__(self):
            self.calibration = MockCalibration()

    return MockConfig()


@pytest.fixture
def mock_hardware_manager():
    """Create a mock hardware manager that returns depth frames."""

    class MockHardwareManager:
        def __init__(self):
            self.frame_count = 0

        def get_depth_frame(self):
            # Return constant depth value for testing
            self.frame_count += 1
            return np.full((424, 512), 700, dtype=np.uint16)

    return MockHardwareManager()


def test_calibrator_initialization_valid(
    mock_calibration_config, mock_hardware_manager
):
    """Test that Calibrator initializes with valid config and hardware."""
    calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
    assert calibrator.config is mock_calibration_config
    assert calibrator.hardware_manager is mock_hardware_manager


def test_calibrator_initialization_missing_calibration_attr(mock_hardware_manager):
    """Test that Calibrator rejects config without calibration attribute."""

    class MockConfig:
        pass

    config = MockConfig()

    with pytest.raises(ValueError, match="must have 'calibration' attribute"):
        Calibrator(config, mock_hardware_manager)


def test_calibrator_initialization_missing_corners(mock_hardware_manager):
    """Test that Calibrator rejects config without corner pairs."""

    class MockCalibration:
        pass

    class MockConfig:
        def __init__(self):
            self.calibration = MockCalibration()

    config = MockConfig()

    with pytest.raises(
        ValueError, match="must have camera_corners and projector_corners"
    ):
        Calibrator(config, mock_hardware_manager)


def test_calibrator_initialization_wrong_corner_count(mock_hardware_manager):
    """Test that Calibrator rejects config with wrong number of corners."""

    class MockCalibration:
        def __init__(self):
            self.camera_corners = [(100, 100), (700, 100)]  # Only 2 corners
            self.projector_corners = [(100, 100), (700, 100)]

    class MockConfig:
        def __init__(self):
            self.calibration = MockCalibration()

    config = MockConfig()

    with pytest.raises(ValueError, match="Exactly 4 corner pairs required"):
        Calibrator(config, mock_hardware_manager)


def test_calibrator_run_success(mock_calibration_config, mock_hardware_manager):
    """Test that Calibrator.run() completes successfully."""
    calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
    result = calibrator.run()

    assert isinstance(result, CalibrationResult)
    assert result.H.shape == (3, 3)
    assert result.H.dtype == np.float32
    assert result.dmax_map.shape == (424, 512)
    assert result.dmax_map.dtype == np.uint16
    assert "num_frames" in result.metadata
    assert "depth_range" in result.metadata
    assert "elapsed_seconds" in result.metadata
    assert "stats" in result.metadata


def test_calibrator_run_metadata(mock_calibration_config, mock_hardware_manager):
    """Test that CalibrationResult metadata is populated correctly."""
    calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
    result = calibrator.run()

    assert result.metadata["num_frames"] == 10
    assert result.metadata["depth_range"] == (650, 800)
    assert result.metadata["depth_shape"] == (424, 512)
    assert result.metadata["elapsed_seconds"] > 0
    assert "mean" in result.metadata["stats"]
    assert "std" in result.metadata["stats"]
    assert "valid_pixel_ratio" in result.metadata["stats"]


def test_calibrator_hardware_capture_error(mock_calibration_config):
    """Test that hardware capture errors are handled."""

    class FailingHardwareManager:
        def get_depth_frame(self):
            raise RuntimeError("Hardware error")

    calibrator = Calibrator(mock_calibration_config, FailingHardwareManager())

    with pytest.raises(RuntimeError, match="Failed to generate dmax_map"):
        calibrator.run()


def test_calibrator_invalid_homography():
    """Test that invalid homography is caught during validation."""

    class MockCalibration:
        def __init__(self):
            # Collinear points - homography will fail
            self.camera_corners = [(100, 100), (200, 200), (300, 300), (400, 400)]
            self.projector_corners = [(100, 100), (200, 200), (300, 300), (400, 400)]
            self.dmax_num_frames = 10
            self.depth_range_min = 650
            self.depth_range_max = 800

    class MockConfig:
        def __init__(self):
            self.calibration = MockCalibration()

    class MockHardwareManager:
        def get_depth_frame(self):
            return np.full((424, 512), 700, dtype=np.uint16)

    config = MockConfig()
    hardware = MockHardwareManager()

    calibrator = Calibrator(config, hardware)

    # cv2.getPerspectiveTransform will raise an error for collinear points
    with pytest.raises(RuntimeError, match="Failed to compute homography"):
        calibrator.run()


def test_calibrator_invalid_dmax_shape(mock_calibration_config):
    """Test that invalid dmax_map shape is caught."""

    class MockHardwareManager:
        def get_depth_frame(self):
            # Return wrong shape
            return np.full((512, 424), 700, dtype=np.uint16)

    calibrator = Calibrator(mock_calibration_config, MockHardwareManager())

    with pytest.raises(RuntimeError, match="Failed to generate dmax_map"):
        calibrator.run()


def test_calibrator_low_valid_pixel_ratio(mock_calibration_config):
    """Test that dmax_map with too few valid pixels is rejected."""

    class MockHardwareManager:
        def get_depth_frame(self):
            # Return mostly zeros (outside depth range)
            frame = np.full((424, 512), 0, dtype=np.uint16)
            # Add a few valid pixels in center
            frame[200:224, 250:262] = 700
            return frame

    calibrator = Calibrator(mock_calibration_config, MockHardwareManager())

    with pytest.raises(ValueError, match="too few valid pixels"):
        calibrator.run()
