"""Unit tests for HardwareManager class.

Tests verify:
- RGB to depth coordinate mapping
- Image registration mode setup
- Coordinate validation
- Error handling for invalid inputs and hardware states
"""

from unittest.mock import patch

import pytest

from cv_system.config import CameraConfig
from cv_system.hardware.manager import HardwareError, HardwareManager


class TestHardwareManagerInitialization:
    """Test HardwareManager initialization."""

    def test_default_initialization(self) -> None:
        """Test that HardwareManager initializes in unconnected state."""
        manager = HardwareManager()

        assert manager.context is None
        assert manager.device is None
        assert manager.depth_stream is None
        assert manager.rgb_stream is None
        assert manager.camera_config is None
        assert manager._initialized is False


class TestMapRgbToDepthSuccess:
    """Test successful RGB to depth coordinate mapping."""

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_map_single_center_pixel(self, mock_cv2, mock_openni2) -> None:
        """Test mapping single center RGB pixel to depth."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        # Map center of 1920x1080 RGB frame
        rgb_points = [(960, 540)]  # Center of RGB frame
        depth_points = manager.map_rgb_to_depth(rgb_points)

        # Expected: scaled to 512x424 depth frame
        # 960 * 512 / 1920 = 256
        # 540 * 424 / 1080 = 212
        assert len(depth_points) == 1
        assert depth_points == [(256, 212)]

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_map_multiple_points(self, mock_cv2, mock_openni2) -> None:
        """Test mapping multiple RGB points to depth."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        rgb_points = [
            (0, 0),  # Top-left corner
            (1919, 0),  # Top-right corner
            (0, 1079),  # Bottom-left corner
            (1919, 1079),  # Bottom-right corner
            (960, 540),  # Center
        ]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        assert len(depth_points) == 5
        # Check scaled coordinates (1920x1080 RGB -> 512x424 depth)
        assert depth_points[0] == (0, 0)  # Top-left
        assert depth_points[1] == (511, 0)  # Top-right
        assert depth_points[2] == (0, 423)  # Bottom-left
        assert depth_points[3] == (511, 423)  # Bottom-right
        assert depth_points[4] == (256, 212)  # Center

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_map_empty_list(self, mock_cv2, mock_openni2) -> None:
        """Test mapping empty list returns empty list."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        depth_points = manager.map_rgb_to_depth([])

        assert depth_points == []

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_map_boundary_values(self, mock_cv2, mock_openni2) -> None:
        """Test mapping at RGB frame boundaries."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        # Test all boundary values (inclusive of 0, exclusive of max)
        rgb_points = [
            (0, 0),  # Min values
            (1, 1),  # Near min
            (1918, 1078),  # Near max
            (1919, 1079),  # Max valid values
        ]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        assert len(depth_points) == 4
        assert depth_points[0] == (0, 0)
        assert depth_points[1] == (0, 0)  # Rounding
        assert depth_points[2] == (511, 423)
        assert depth_points[3] == (511, 423)


class TestMapRgbToDepthValidationErrors:
    """Test validation errors for invalid inputs."""

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_not_initialized_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that mapping without initialization raises HardwareError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = False

        with pytest.raises(HardwareError, match="HardwareManager is not initialized"):
            manager.map_rgb_to_depth([(100, 100)])

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_non_list_input_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that non-list input raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="must be a list"):
            manager.map_rgb_to_depth((100, 100))  # type: ignore

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_non_tuple_point_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that non-tuple point raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="rgb_points\\[0\\] must be an .* tuple"):
            manager.map_rgb_to_depth([100])  # type: ignore

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_wrong_tuple_length_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that tuple with wrong length raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="rgb_points\\[0\\] must be an .* tuple"):
            manager.map_rgb_to_depth([(100, 100, 50)])  # type: ignore

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_non_integer_coordinates_raise_error(self, mock_cv2, mock_openni2) -> None:
        """Test that non-integer coordinates raise ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="coordinates must be integers"):
            manager.map_rgb_to_depth([(100.5, 100)])  # type: ignore

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_negative_x_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that negative x-coordinate raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="x-coordinate out of bounds"):
            manager.map_rgb_to_depth([(-10, 100)])

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_negative_y_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that negative y-coordinate raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="y-coordinate out of bounds"):
            manager.map_rgb_to_depth([(100, -10)])

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_x_at_bound_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that x-coordinate at bound (width) raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="x-coordinate out of bounds"):
            manager.map_rgb_to_depth([(1920, 100)])  # Width is 1920, max is 1919

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_y_at_bound_raises_error(self, mock_cv2, mock_openni2) -> None:
        """Test that y-coordinate at bound (height) raises ValueError."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        with pytest.raises(ValueError, match="y-coordinate out of bounds"):
            manager.map_rgb_to_depth([(100, 1080)])  # Height is 1080, max is 1079


class TestMapRgbToDepthWithCustomResolution:
    """Test mapping with custom camera resolutions."""

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_custom_rgb_resolution(self, mock_cv2, mock_openni2) -> None:
        """Test mapping with custom RGB resolution (1280x720)."""
        manager = HardwareManager()
        config = CameraConfig(rgb_resolution=(720, 1280))
        manager.camera_config = config
        manager._initialized = True

        # Center of 1280x720 RGB frame
        rgb_points = [(640, 360)]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        # 640 * 512 / 1280 = 256
        # 360 * 424 / 720 = 212
        assert depth_points == [(256, 212)]

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_custom_depth_resolution(self, mock_cv2, mock_openni2) -> None:
        """Test mapping with custom depth resolution (640x480)."""
        manager = HardwareManager()
        config = CameraConfig(depth_resolution=(480, 640))
        manager.camera_config = config
        manager._initialized = True

        # Center of 1920x1080 RGB frame
        rgb_points = [(960, 540)]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        # 960 * 640 / 1920 = 320
        # 540 * 480 / 1080 = 240
        assert depth_points == [(320, 240)]

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_both_custom_resolutions(self, mock_cv2, mock_openni2) -> None:
        """Test mapping with both custom resolutions."""
        manager = HardwareManager()
        config = CameraConfig(rgb_resolution=(720, 1280), depth_resolution=(480, 640))
        manager.camera_config = config
        manager._initialized = True

        rgb_points = [(640, 360)]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        # 640 * 640 / 1280 = 320
        # 360 * 480 / 720 = 240
        assert depth_points == [(320, 240)]


class TestMapRgbToDepthOutputBounds:
    """Test that output depth coordinates are within valid bounds."""

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_output_within_depth_bounds(self, mock_cv2, mock_openni2) -> None:
        """Test that all output coordinates are within depth frame bounds."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        # Test various RGB coordinates
        rgb_points = [
            (0, 0),
            (100, 100),
            (500, 300),
            (960, 540),
            (1500, 800),
            (1919, 1079),
        ]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        # Verify all outputs are within depth frame bounds (0-512 x, 0-424 y)
        for depth_x, depth_y in depth_points:
            assert 0 <= depth_x < 512
            assert 0 <= depth_y < 424

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_rgb_corners_map_to_depth_corners(self, mock_cv2, mock_openni2) -> None:
        """Test that RGB corner points map to depth corner points."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        # RGB corners (1920x1080)
        rgb_corners = [
            (0, 0),  # Top-left
            (1919, 0),  # Top-right
            (0, 1079),  # Bottom-left
            (1919, 1079),  # Bottom-right
        ]
        depth_corners = manager.map_rgb_to_depth(rgb_corners)

        # Should map to depth corners (512x424)
        assert depth_corners[0] == (0, 0)  # Top-left
        assert depth_corners[1] == (511, 0)  # Top-right
        assert depth_corners[2] == (0, 423)  # Bottom-left
        assert depth_corners[3] == (511, 423)  # Bottom-right


class TestMapRgbToDepthIntegerRounding:
    """Test integer rounding behavior of coordinate mapping."""

    @patch("cv_system.hardware.manager.openni2")
    @patch("cv_system.hardware.manager.cv2")
    def test_rounding_behavior(self, mock_cv2, mock_openni2) -> None:
        """Test that coordinates are properly rounded to integers."""
        manager = HardwareManager()
        config = CameraConfig()
        manager.camera_config = config
        manager._initialized = True

        # Use coordinates that produce non-integer scaled results
        rgb_points = [
            (100, 100),  # 100 * 512 / 1920 = 26.66... -> 26 or 27
            (500, 300),  # 500 * 512 / 1920 = 133.33... -> 133 or 134
        ]
        depth_points = manager.map_rgb_to_depth(rgb_points)

        # Check that all outputs are integers
        for depth_x, depth_y in depth_points:
            assert isinstance(depth_x, int)
            assert isinstance(depth_y, int)
