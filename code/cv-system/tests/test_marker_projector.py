"""Unit tests for MarkerProjector class.

Tests verify:
- Image creation (correct dimensions, black background)
- White square drawing (correct positions, sizes, and colors)
- Fullscreen window properties (cv2.WINDOW_FULLSCREEN usage)
- Corner validation (exactly 4 corners required)
- Error handling for invalid inputs
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cv_system.calibration.marker_projector import MarkerProjector


@contextmanager
def mock_cv2():
    """Context manager to mock cv2 for lazy import pattern."""
    mock_cv2 = MagicMock()
    mock_cv2.WINDOW_FULLSCREEN = 1
    with patch(
        "builtins.__import__",
        side_effect=lambda name, *args, **kwargs: (
            mock_cv2 if name == "cv2" else __import__(name, *args, **kwargs)
        ),
    ):
        yield mock_cv2


class TestMarkerProjectorInitialization:
    """Test MarkerProjector initialization with various configurations."""

    def test_default_initialization(self) -> None:
        """Test that MarkerProjector initializes with defaults."""
        projector = MarkerProjector()

        assert projector.resolution == (1920, 1080)
        assert projector.marker_size == 100

    def test_custom_resolution(self) -> None:
        """Test initialization with custom resolution."""
        projector = MarkerProjector(resolution=(1280, 720))

        assert projector.resolution == (1280, 720)

    def test_custom_marker_size(self) -> None:
        """Test initialization with custom marker size."""
        projector = MarkerProjector(marker_size=50)

        assert projector.marker_size == 50

    def test_custom_resolution_and_marker_size(self) -> None:
        """Test initialization with both custom parameters."""
        projector = MarkerProjector(resolution=(800, 600), marker_size=80)

        assert projector.resolution == (800, 600)
        assert projector.marker_size == 80

    def test_invalid_resolution_not_tuple(self) -> None:
        """Test that non-tuple resolution raises ValueError."""
        with pytest.raises(ValueError, match="resolution must be a .* tuple"):
            MarkerProjector(resolution="1920,1080")

    def test_invalid_resolution_wrong_length(self) -> None:
        """Test that resolution with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="resolution must be a .* tuple"):
            MarkerProjector(resolution=(1920, 1080, 60))

    def test_invalid_resolution_negative(self) -> None:
        """Test that negative resolution values raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            MarkerProjector(resolution=(-1920, 1080))

    def test_invalid_resolution_zero(self) -> None:
        """Test that zero resolution values raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            MarkerProjector(resolution=(1920, 0))

    def test_invalid_marker_size_not_int(self) -> None:
        """Test that non-integer marker_size raises ValueError."""
        with pytest.raises(ValueError, match="marker_size must be a positive integer"):
            MarkerProjector(marker_size=50.5)

    def test_invalid_marker_size_negative(self) -> None:
        """Test that negative marker_size raises ValueError."""
        with pytest.raises(ValueError, match="marker_size must be a positive integer"):
            MarkerProjector(marker_size=-50)

    def test_invalid_marker_size_zero(self) -> None:
        """Test that zero marker_size raises ValueError."""
        with pytest.raises(ValueError, match="marker_size must be a positive integer"):
            MarkerProjector(marker_size=0)

    def test_repr(self) -> None:
        """Test string representation of MarkerProjector."""
        projector = MarkerProjector(resolution=(1280, 720), marker_size=80)

        repr_str = repr(projector)
        assert "MarkerProjector" in repr_str
        assert "resolution=(1280, 720)" in repr_str
        assert "marker_size=80" in repr_str


class TestCornerValidation:
    """Test corner validation in project_markers method."""

    def test_invalid_corners_not_list(self) -> None:
        """Test that non-list corners raises ValueError."""
        projector = MarkerProjector()

        with pytest.raises(ValueError, match="must be a list of 4 .* tuples"):
            projector.project_markers("not_a_list")

    def test_invalid_corners_wrong_count(self) -> None:
        """Test that wrong number of corners raises ValueError."""
        projector = MarkerProjector()

        with pytest.raises(ValueError, match="Exactly 4 corners required: got 3"):
            projector.project_markers([(0, 0), (0, 100), (100, 0)])  # type: ignore

    def test_invalid_corner_not_tuple(self) -> None:
        """Test that non-tuple corner raises ValueError."""
        projector = MarkerProjector()
        corners = [(0, 0), "not_a_tuple", (100, 0), (0, 100)]

        with pytest.raises(
            ValueError, match="projector_corners\\[1\\] must be an .* tuple"
        ):
            projector.project_markers(corners)

    def test_invalid_corner_wrong_length(self) -> None:
        """Test that corner with wrong length raises ValueError."""
        projector = MarkerProjector()
        corners = [(0, 0, 0), (100, 0), (0, 100), (1920, 1080)]

        with pytest.raises(
            ValueError, match=r"projector_corners\[0\] must be an .* tuple"
        ):
            projector.project_markers(corners)

    def test_invalid_corner_not_int_x(self) -> None:
        """Test that non-integer x-coordinate raises ValueError."""
        projector = MarkerProjector()
        corners = [(0.5, 0), (100, 0), (0, 100), (1920, 1080)]

        with pytest.raises(ValueError, match="coordinates must be integers"):
            projector.project_markers(corners)

    def test_invalid_corner_not_int_y(self) -> None:
        """Test that non-integer y-coordinate raises ValueError."""
        projector = MarkerProjector()
        corners = [(0, 0.5), (100, 0), (0, 100), (1920, 1080)]

        with pytest.raises(ValueError, match="coordinates must be integers"):
            projector.project_markers(corners)

    def test_invalid_corner_x_negative(self) -> None:
        """Test that negative x-coordinate raises ValueError."""
        projector = MarkerProjector()
        corners = [(-10, 0), (100, 0), (0, 100), (1920, 1080)]

        with pytest.raises(ValueError, match="x-coordinate out of bounds"):
            projector.project_markers(corners)

    def test_invalid_corner_y_negative(self) -> None:
        """Test that negative y-coordinate raises ValueError."""
        projector = MarkerProjector()
        corners = [(0, -10), (100, 0), (0, 100), (1920, 1080)]

        with pytest.raises(ValueError, match="y-coordinate out of bounds"):
            projector.project_markers(corners)

    def test_invalid_corner_x_out_of_bounds(self) -> None:
        """Test that x-coordinate >= width raises ValueError."""
        projector = MarkerProjector(resolution=(1920, 1080))
        corners = [(1920, 0), (100, 0), (0, 100), (1919, 1079)]

        with pytest.raises(ValueError, match="x-coordinate out of bounds"):
            projector.project_markers(corners)

    def test_invalid_corner_y_out_of_bounds(self) -> None:
        """Test that y-coordinate >= height raises ValueError."""
        projector = MarkerProjector(resolution=(1920, 1080))
        corners = [(0, 1080), (100, 0), (0, 100), (1919, 1079)]

        with pytest.raises(ValueError, match="y-coordinate out of bounds"):
            projector.project_markers(corners)

    def test_valid_corners_at_bounds(self) -> None:
        """Test that corners at image bounds are accepted."""
        projector = MarkerProjector(resolution=(1920, 1080))
        corners = [(0, 0), (1919, 0), (0, 1079), (1919, 1079)]

        # Should not raise - patch builtins.__import__ to mock cv2
        mock_cv2 = MagicMock()
        mock_cv2.WINDOW_FULLSCREEN = 1
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                mock_cv2 if name == "cv2" else __import__(name, *args, **kwargs)
            ),
        ):
            projector.project_markers(corners)


class TestImageCreation:
    """Test image creation with correct properties."""

    def test_creates_black_image_with_correct_dimensions(self) -> None:
        """Test that project_markers creates a black image with correct dimensions."""
        projector = MarkerProjector(resolution=(800, 600))
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        # Verify imshow was called
        assert len(imshow_call_args) == 1
        image = imshow_call_args[0][0][1]

        # Verify image dimensions
        assert image.shape == (600, 800, 3)  # (height, width, channels)

    def test_creates_black_background(self) -> None:
        """Test that image has black background (all zeros)."""
        projector = MarkerProjector(resolution=(200, 200), marker_size=40)
        corners = [(50, 50), (150, 50), (50, 150), (150, 150)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        # Verify background is black (all pixels should be 0 except markers)
        image = imshow_call_args[0][0][1]

        # Check that at least some pixels are black (outside marker areas)
        # With marker_size=40, markers are 40x40, leaving background visible
        assert np.all(image[0:10, 0:200] == 0)  # Top edge
        assert np.all(image[0:200, 0:10] == 0)  # Left edge
        assert np.all(image[190:200, 0:200] == 0)  # Bottom edge
        assert np.all(image[0:200, 190:200] == 0)  # Right edge
        # Center area should be black (between markers)
        assert np.all(image[75:125, 75:125] == 0)  # Center area between markers

    def test_image_dtype_is_uint8(self) -> None:
        """Test that image has correct dtype (uint8)."""
        projector = MarkerProjector(resolution=(100, 100))
        corners = [(50, 50), (50, 50), (50, 50), (50, 50)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        image = imshow_call_args[0][0][1]
        assert image.dtype == np.uint8


class TestMarkerDrawing:
    """Test white square marker drawing."""

    def test_draws_white_squares_at_corners(self) -> None:
        """Test that white squares are drawn at specified corner positions."""
        projector = MarkerProjector(resolution=(200, 200), marker_size=40)
        corners = [(50, 50), (150, 50), (50, 150), (150, 150)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        image = imshow_call_args[0][0][1]

        # Verify white squares are at expected positions
        # Corner 1 at (50, 50): bounds (30, 30) to (70, 70)
        assert np.all(image[30:70, 30:70] == 255)

        # Corner 2 at (150, 50): bounds (130, 30) to (170, 70)
        assert np.all(image[30:70, 130:170] == 255)

        # Corner 3 at (50, 150): bounds (30, 130) to (70, 170)
        assert np.all(image[130:170, 30:70] == 255)

        # Corner 4 at (150, 150): bounds (130, 130) to (170, 170)
        assert np.all(image[130:170, 130:170] == 255)

    def test_markers_are_white(self) -> None:
        """Test that markers are white (all RGB channels = 255)."""
        projector = MarkerProjector(resolution=(100, 100), marker_size=40)
        corners = [(50, 50), (50, 50), (50, 50), (50, 50)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        image = imshow_call_args[0][0][1]

        # Check that all channels are 255 in the marker region
        marker_region = image[30:70, 30:70]
        assert np.all(marker_region[:, :, 0] == 255)  # R channel
        assert np.all(marker_region[:, :, 1] == 255)  # G channel
        assert np.all(marker_region[:, :, 2] == 255)  # B channel

    def test_marker_size_respects_config(self) -> None:
        """Test that marker size respects marker_size parameter."""
        projector = MarkerProjector(resolution=(200, 200), marker_size=60)
        corners = [(100, 100), (100, 100), (100, 100), (100, 100)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        image = imshow_call_args[0][0][1]

        # With marker_size=60, half_size=30, bounds should be (70, 70) to (130, 130)
        # Verify that white square is 60x60 pixels
        assert np.all(image[70:130, 70:130] == 255)
        assert np.all(image[60:70, 70:130] == 0)  # Top edge (black)
        assert np.all(image[130:140, 70:130] == 0)  # Bottom edge (black)

    def test_markers_clipped_at_image_bounds(self) -> None:
        """Test that markers at image edges are clipped correctly."""
        projector = MarkerProjector(resolution=(100, 100), marker_size=40)
        # Corner at (0, 0) would extend outside image - should be clipped
        corners = [(0, 0), (99, 0), (0, 99), (99, 99)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            projector.project_markers(corners)

        image = imshow_call_args[0][0][1]

        # Corner at (0, 0) with half_size=20 should be clipped to (0, 0) to (20, 20)
        assert np.all(image[0:20, 0:20] == 255)

        # Corner at (99, 99) should be clipped to (79, 79) to (99, 99)
        assert np.all(image[79:99, 79:99] == 255)

        # No index errors should occur


class TestFullscreenWindow:
    """Test fullscreen window behavior."""

    def test_uses_window_fullscreen_flag(self) -> None:
        """Test that cv2.WINDOW_FULLSCREEN flag is used for window creation."""
        projector = MarkerProjector()
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        with mock_cv2() as mock:
            mock.WINDOW_FULLSCREEN = 1  # Typical value
            projector.project_markers(corners)

            # Verify namedWindow was called with WINDOW_FULLSCREEN
            assert mock.namedWindow.called
            args, kwargs = mock.namedWindow.call_args
            assert args[0] == "Calibration Markers"
            assert args[1] == 1  # cv2.WINDOW_FULLSCREEN

    def test_displays_image_in_window(self) -> None:
        """Test that image is displayed using cv2.imshow."""
        projector = MarkerProjector()
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        with mock_cv2() as mock:
            projector.project_markers(corners)

            # Verify imshow was called
            assert mock.imshow.called
            args, kwargs = mock.imshow.call_args
            assert args[0] == "Calibration Markers"
            assert isinstance(args[1], np.ndarray)

    def test_calls_waitKey_to_process_events(self) -> None:
        """Test that waitKey is called to process pending events."""
        projector = MarkerProjector()
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        with mock_cv2() as mock:
            projector.project_markers(corners)

            # Verify waitKey was called
            assert mock.waitKey.called
            args, kwargs = mock.waitKey.call_args
            assert args[0] == 1  # Non-blocking call

    def test_destroy_window_closes_window(self) -> None:
        """Test that destroy_window closes the OpenCV window."""
        projector = MarkerProjector()
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        with mock_cv2() as mock:
            projector.project_markers(corners)
            projector.destroy_window()

            # Verify destroyWindow was called
            assert mock.destroyWindow.called
            args, kwargs = mock.destroyWindow.call_args
            assert args[0] == "Calibration Markers"

    def test_returns_window_handle(self) -> None:
        """Test that project_markers returns a window handle."""
        projector = MarkerProjector()
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        with mock_cv2():
            result = projector.project_markers(corners)

            # Should return some identifier (we return id of window name string)
            assert isinstance(result, int)


class TestCompleteWorkflow:
    """Test complete projection workflow with realistic parameters."""

    def test_complete_projection_with_session_config_corners(self) -> None:
        """Test projection with corners from session.json configuration."""
        projector = MarkerProjector(resolution=(1920, 1080))

        # Corners within valid bounds (0 <= x < 1920, 0 <= y < 1080)
        corners = [(0, 0), (0, 1079), (1919, 0), (1919, 1079)]

        # Track the image passed to imshow
        imshow_call_args = []

        def capture_imshow(*args, **kwargs):
            imshow_call_args.append((args, kwargs))
            return MagicMock()

        with mock_cv2() as mock:
            mock.imshow.side_effect = capture_imshow

            result = projector.project_markers(corners)

            # Verify projection succeeded
            assert isinstance(result, int)

            # Verify image properties
            assert len(imshow_call_args) == 1
            image = imshow_call_args[0][0][1]
            assert image.shape == (1080, 1920, 3)

            # Verify corners are white
            assert np.all(image[0:50, 0:50] == 255)  # (0, 0)
            assert np.all(image[1029:1079, 0:50] == 255)  # (0, 1079)
            assert np.all(image[0:50, 1869:1919] == 255)  # (1919, 0)
            assert np.all(image[1029:1079, 1869:1919] == 255)  # (1919, 1079)


class TestErrorHandling:
    """Test error handling for various failure scenarios."""

    def test_opencv_not_installed(self) -> None:
        """Test that ImportError is raised when OpenCV is not available."""
        projector = MarkerProjector()
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]

        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError, match="OpenCV .* is required"):
                projector.project_markers(corners)

    def test_destroy_window_without_cv2_raises_error(self) -> None:
        """Test that destroy_window raises ImportError when cv2 is missing."""
        projector = MarkerProjector()

        # Remove cv2 from modules
        import sys

        cv2_backup = sys.modules.get("cv2")
        try:
            sys.modules["cv2"] = None

            with pytest.raises(ImportError, match="OpenCV .* is required"):
                projector.destroy_window()
        finally:
            # Restore cv2
            if cv2_backup:
                sys.modules["cv2"] = cv2_backup
