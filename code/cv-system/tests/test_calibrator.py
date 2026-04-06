"""Tests for Calibrator orchestration."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

# Skip tests if cv2 is not available (requires system libraries)
pytest.importorskip("cv2", exc_type=ImportError)

from cv_system.calibration.calibrator import Calibrator
from cv_system.calibration.result import CalibrationResult


@pytest.fixture
def mock_calibration_config():
    """Create a mock calibration config object."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class MockCalibration:
        def __init__(self):
            # No longer need camera_corners in config
            # Order must match MarkerDetector sorting: [top-left, top-right, bottom-left, bottom-right]
            self.projector_corners = [(100, 100), (700, 100), (100, 500), (700, 500)]
            self.dmax_num_frames = 10

    class MockConfig:
        def __init__(self):
            self.camera = MockCameraConfig()
            self.calibration = MockCalibration()

    return MockConfig()


@pytest.fixture
def mock_hardware_manager():
    """Create a mock hardware manager that returns RGB and depth frames."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class MockHardwareManager:
        def __init__(self):
            self.frame_count = 0
            self.camera_config = MockCameraConfig()

        def get_rgb_frame(self):
            # Return a simulated RGB frame with 4 white squares for testing
            # Simulate 1920x1080 RGB frame with 4 white markers
            # Order: [top-left, top-right, bottom-left, bottom-right]
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Draw white squares at expected positions (will be detected)
            half_size = 50
            corners = [(100, 100), (700, 100), (100, 500), (700, 500)]
            for x, y in corners:
                x1, y1 = max(0, x - half_size), max(0, y - half_size)
                x2, y2 = min(1920, x + half_size), min(1080, y + half_size)
                frame[y1:y2, x1:x2] = [255, 255, 255]
            return frame

        def get_depth_frame(self):
            # Return constant depth value for testing
            self.frame_count += 1
            return np.full((424, 512), 700, dtype=np.uint16)

            ]
            return depth_points

    return MockHardwareManager()


@pytest.fixture
def mock_marker_projector():
    """Create a mock marker projector that doesn't use GUI."""
    with patch("cv_system.calibration.calibrator.MarkerProjector") as mock:
        instance = Mock()
        instance.project_markers.return_value = 1
        instance.destroy_window.return_value = None
        mock.return_value = instance
        yield instance


def test_calibrator_initialization_valid(
    mock_calibration_config, mock_hardware_manager, mock_marker_projector
):
    """Test that Calibrator initializes with valid config and hardware."""
    calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
    assert calibrator.config is mock_calibration_config
    assert calibrator.hardware_manager is mock_hardware_manager
    assert calibrator.marker_detector is not None
    assert calibrator.marker_projector is not None


def test_calibrator_initialization_missing_calibration_attr(
    mock_hardware_manager, mock_marker_projector
):
    """Test that Calibrator rejects config without calibration attribute."""

    class MockConfig:
        pass

    config = MockConfig()


def test_calibrator_initialization_missing_corners(
    mock_hardware_manager, mock_marker_projector
):
    """Test that Calibrator rejects config without projector_corners."""

    class MockCalibration:
        pass

    class MockConfig:
        def __init__(self):
            self.calibration = MockCalibration()

    config = MockConfig()

    with pytest.raises(ValueError, match="must have projector_corners"):
        Calibrator(config, mock_hardware_manager)


def test_calibrator_initialization_wrong_corner_count(
    mock_hardware_manager, mock_marker_projector
):
    """Test that Calibrator rejects config with wrong number of corners."""

    class MockCalibration:
        def __init__(self):
            self.projector_corners = [(100, 100), (700, 100)]  # Only 2 corners
            self.dmax_num_frames = 10

    class MockConfig:
        def __init__(self):
            self.camera = type(
                "MockCameraConfig",
                (),
                {"depth_resolution": (424, 512), "rgb_resolution": (1080, 1920)},
            )()
            self.calibration = MockCalibration()

    config = MockConfig()

    with pytest.raises(ValueError, match="Exactly 4 projector corners required"):
        Calibrator(config, mock_hardware_manager)


def test_calibrator_run_success(
    mock_calibration_config, mock_hardware_manager, mock_marker_projector
):
    """Test that Calibrator.run() completes successfully."""
    calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
    result = calibrator.run()

    assert isinstance(result, CalibrationResult)
    assert result.H.shape == (3, 3)
    assert result.H.dtype == np.float32
    assert result.dmax_map.shape == (424, 512)
    assert result.dmax_map.dtype == np.uint16
    assert len(result.camera_corners) == 4
    assert "num_frames" in result.metadata
    assert result.metadata["depth_shape"] == (424, 512)
    assert result.metadata["elapsed_seconds"] > 0
    # Direct mode calibrator adds method and dmax_compute_time_ms
    assert "method" in result.metadata
    assert result.metadata["method"] == "direct"
    assert "dmax_compute_time_ms" in result.metadata
    assert result.metadata["camera_corners"] == result.camera_corners


def test_calibrator_hardware_capture_error(
    mock_calibration_config, mock_marker_projector
):
    """Test that hardware capture errors are handled."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class FailingHardwareManager:
        def __init__(self):
            self.camera_config = MockCameraConfig()

        def get_rgb_frame(self):
            raise RuntimeError("RGB capture error")

        def get_depth_frame(self):
            raise RuntimeError("Hardware error")

        def map_rgb_to_depth(self, rgb_points):
            return rgb_points

    calibrator = Calibrator(mock_calibration_config, FailingHardwareManager())

    with pytest.raises(RuntimeError, match="Failed to compute homography"):
        calibrator.run()


def test_calibrator_marker_detection_failure(
    mock_calibration_config, mock_marker_projector
):
    """Test that marker detection failures are handled."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class HardwareManagerNoMarkers:
        def __init__(self):
            self.camera_config = MockCameraConfig()

        def get_rgb_frame(self):
            # Return frame with no white markers
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        def get_depth_frame(self):
            return np.full((424, 512), 700, dtype=np.uint16)

        def map_rgb_to_depth(self, rgb_points):
            return [(0, 0)]

    calibrator = Calibrator(mock_calibration_config, HardwareManagerNoMarkers())

    with pytest.raises(RuntimeError, match="Failed to compute homography"):
        calibrator.run()


def test_calibrator_camera_corners_out_of_bounds(
    mock_calibration_config, mock_marker_projector
):
    """Test that out-of-bounds camera corners are rejected."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class HardwareManagerOutOfBounds:
        def __init__(self):
            self.camera_config = MockCameraConfig()

        def get_rgb_frame(self):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Draw white squares
            # Order: [top-left, top-right, bottom-left, bottom-right]
            half_size = 50
            corners = [(100, 100), (700, 100), (100, 500), (700, 500)]
            for x, y in corners:
                x1, y1 = max(0, x - half_size), max(0, y - half_size)
                x2, y2 = min(1920, x + half_size), min(1080, y + half_size)
                frame[y1:y2, x1:x2] = [255, 255, 255]
            return frame

        def get_depth_frame(self):
            return np.full((424, 512), 700, dtype=np.uint16)

        def map_rgb_to_depth(self, rgb_points):
            # Return one coordinate out of bounds
            return [(-10, 100), (100, 100), (100, 100), (100, 100)]

    calibrator = Calibrator(mock_calibration_config, HardwareManagerOutOfBounds())

    with pytest.raises(RuntimeError, match="Failed to compute homography"):
        calibrator.run()


def test_calibrator_invalid_dmax_shape(
    mock_calibration_config, mock_hardware_manager, mock_marker_projector
):
    """Test that invalid dmax_map shape is caught."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class MockHardwareManager:
        def __init__(self):
            self.camera_config = MockCameraConfig()

        def get_rgb_frame(self):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Draw white squares
            # Order: [top-left, top-right, bottom-left, bottom-right]
            half_size = 50
            for x, y in [(100, 100), (700, 100), (100, 500), (700, 500)]:
                x1, y1 = max(0, x - half_size), max(0, y - half_size)
                x2, y2 = min(1920, x + half_size), min(1080, y + half_size)
                frame[y1:y2, x1:x2] = [255, 255, 255]
            return frame

        def get_depth_frame(self):
            # Return wrong shape
            return np.full((512, 424), 700, dtype=np.uint16)

        def map_rgb_to_depth(self, rgb_points):
            return [(int(x * 512 / 1920), int(y * 424 / 1080)) for x, y in rgb_points]

    calibrator = Calibrator(mock_calibration_config, MockHardwareManager())

    with pytest.raises(RuntimeError, match="Failed to generate dmax_map"):
        calibrator.run()


def test_calibrator_low_valid_pixel_ratio(
    mock_calibration_config, mock_hardware_manager, mock_marker_projector
):
    """Test that dmax_map with too few valid pixels is rejected."""

    class MockCameraConfig:
        depth_resolution = (424, 512)
        rgb_resolution = (1080, 1920)

    class MockHardwareManager:
        def __init__(self):
            self.camera_config = MockCameraConfig()

        def get_rgb_frame(self):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # Draw white squares
            # Order: [top-left, top-right, bottom-left, bottom-right]
            half_size = 50
            for x, y in [(100, 100), (700, 100), (100, 500), (700, 500)]:
                x1, y1 = max(0, x - half_size), max(0, y - half_size)
                x2, y2 = min(1920, x + half_size), min(1080, y + half_size)
                frame[y1:y2, x1:x2] = [255, 255, 255]
            return frame

        def get_depth_frame(self):
            # Return mostly zeros (outside depth range)
            frame = np.full((424, 512), 0, dtype=np.uint16)
            # Add a few valid pixels in center
            frame[200:224, 250:262] = 700
            return frame

        def map_rgb_to_depth(self, rgb_points):
            return [(int(x * 512 / 1920), int(y * 424 / 1080)) for x, y in rgb_points]

    calibrator = Calibrator(mock_calibration_config, MockHardwareManager())

    with pytest.raises(ValueError, match="too few valid pixels"):
        calibrator.run()


def test_calibrator_run_success_direct_mode(
    mock_calibration_config, mock_hardware_manager, mock_marker_projector
):
    """Test that Calibrator.run() completes successfully with direct mode."""
    calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
    result = calibrator.run()

    assert isinstance(result, CalibrationResult)
    assert result.H.shape == (3, 3)
    assert result.H.dtype == np.float32
    assert result.dmax_map.shape == (424, 512)
    assert result.dmax_map.dtype == np.uint16
    assert len(result.camera_corners) == 4
    # Direct mode adds "method": "direct"
    assert result.metadata["method"] == "direct"
    assert "dmax_compute_time_ms" in result.metadata
    assert result.metadata["num_frames"] == 10
    assert result.metadata["depth_shape"] == (424, 512)


def test_calibrator_dmax_quality_validation_valid(mock_calibration_config, mock_hardware_manager):
    """Test that dmax_map with sufficient valid pixels passes."""
    with patch("cv_system.calibration.calibrator.generate_dmax_map") as mock_gen:
        # Return dmax_map with sufficient valid pixels (>50%)
        mock_gen.return_value = np.full((424, 512), 700, dtype=np.uint16)

        calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)
        result = calibrator.run()  # Should not raise

        assert result.metadata["num_frames"] == 10


def test_calibrator_dmax_quality_validation_low_valid_ratio(
    mock_calibration_config, mock_hardware_manager
):
    """Test that dmax_map with too few valid pixels is rejected."""
    with patch("cv_system.calibration.calibrator.generate_dmax_map") as mock_gen:
        # Return dmax_map with few valid pixels (<50%)
        dmax_bad = np.full((424, 512), 700, dtype=np.uint16)
        dmax_bad[0:212, :] = 0  # Less than 50% valid

        mock_gen.return_value = dmax_bad

        calibrator = Calibrator(mock_calibration_config, mock_hardware_manager)

        with pytest.raises(ValueError, match="too few valid pixels"):
            calibrator.run()
