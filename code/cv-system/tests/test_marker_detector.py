"""Unit tests for MarkerDetector class.

Tests verify:
- Input validation (3D RGB frame requirement)
- Detection pipeline (threshold, contours, filtering)
- Centroid extraction accuracy (within 2-3 pixels of actual center)
- Marker filtering (size, aspect ratio, color)
- Marker sorting by position (top-left, top-right, bottom-left, bottom-right)
- Error handling for edge cases
"""

import numpy as np
import pytest

from cv_system.calibration.marker_detector import MarkerDetector


class TestMarkerDetectorInitialization:
    """Test MarkerDetector initialization with various configurations."""

    def test_default_initialization(self) -> None:
        """Test that MarkerDetector initializes with defaults."""
        detector = MarkerDetector()

        assert detector.min_area == 2500
        assert detector.max_area == 40000
        assert detector.aspect_ratio_tolerance == 0.3
        assert detector.threshold_value == 200

    def test_custom_min_area(self) -> None:
        """Test initialization with custom min_area."""
        detector = MarkerDetector(min_area=1000)

        assert detector.min_area == 1000
        assert detector.max_area == 40000

    def test_custom_max_area(self) -> None:
        """Test initialization with custom max_area."""
        detector = MarkerDetector(max_area=50000)

        assert detector.min_area == 2500
        assert detector.max_area == 50000

    def test_custom_aspect_ratio_tolerance(self) -> None:
        """Test initialization with custom aspect_ratio_tolerance."""
        detector = MarkerDetector(aspect_ratio_tolerance=0.2)

        assert detector.aspect_ratio_tolerance == 0.2

    def test_custom_threshold_value(self) -> None:
        """Test initialization with custom threshold_value."""
        detector = MarkerDetector(threshold_value=180)

        assert detector.threshold_value == 180

    def test_all_custom_parameters(self) -> None:
        """Test initialization with all custom parameters."""
        detector = MarkerDetector(
            min_area=1500,
            max_area=30000,
            aspect_ratio_tolerance=0.25,
            threshold_value=190,
        )

        assert detector.min_area == 1500
        assert detector.max_area == 30000
        assert detector.aspect_ratio_tolerance == 0.25
        assert detector.threshold_value == 190

    def test_invalid_min_area_not_int(self) -> None:
        """Test that non-integer min_area raises ValueError."""
        with pytest.raises(ValueError, match="min_area must be a positive integer"):
            MarkerDetector(min_area=2500.5)

    def test_invalid_min_area_negative(self) -> None:
        """Test that negative min_area raises ValueError."""
        with pytest.raises(ValueError, match="min_area must be a positive integer"):
            MarkerDetector(min_area=-100)

    def test_invalid_min_area_zero(self) -> None:
        """Test that zero min_area raises ValueError."""
        with pytest.raises(ValueError, match="min_area must be a positive integer"):
            MarkerDetector(min_area=0)

    def test_invalid_max_area_not_int(self) -> None:
        """Test that non-integer max_area raises ValueError."""
        with pytest.raises(ValueError, match="max_area must be a positive integer"):
            MarkerDetector(max_area=40000.5)

    def test_invalid_max_area_negative(self) -> None:
        """Test that negative max_area raises ValueError."""
        with pytest.raises(ValueError, match="max_area must be a positive integer"):
            MarkerDetector(max_area=-100)

    def test_invalid_max_area_zero(self) -> None:
        """Test that zero max_area raises ValueError."""
        with pytest.raises(ValueError, match="max_area must be a positive integer"):
            MarkerDetector(max_area=0)

    def test_invalid_area_range(self) -> None:
        """Test that max_area <= min_area raises ValueError."""
        with pytest.raises(
            ValueError, match="max_area .* must be greater than min_area"
        ):
            MarkerDetector(min_area=40000, max_area=2500)

    def test_invalid_aspect_ratio_tolerance_not_number(self) -> None:
        """Test that non-number aspect_ratio_tolerance raises ValueError."""
        with pytest.raises(ValueError, match="aspect_ratio_tolerance must be a number"):
            MarkerDetector(aspect_ratio_tolerance="0.3")

    def test_invalid_aspect_ratio_tolerance_negative(self) -> None:
        """Test that negative aspect_ratio_tolerance raises ValueError."""
        with pytest.raises(
            ValueError, match="aspect_ratio_tolerance must be in \\[0, 1\\.0\\]"
        ):
            MarkerDetector(aspect_ratio_tolerance=-0.1)

    def test_invalid_aspect_ratio_tolerance_too_large(self) -> None:
        """Test that aspect_ratio_tolerance > 1.0 raises ValueError."""
        with pytest.raises(
            ValueError, match="aspect_ratio_tolerance must be in \\[0, 1\\.0\\]"
        ):
            MarkerDetector(aspect_ratio_tolerance=1.5)

    def test_invalid_threshold_value_not_int(self) -> None:
        """Test that non-integer threshold_value raises ValueError."""
        with pytest.raises(
            ValueError, match="threshold_value must be an integer in \\[0, 255\\]"
        ):
            MarkerDetector(threshold_value=200.5)

    def test_invalid_threshold_value_negative(self) -> None:
        """Test that negative threshold_value raises ValueError."""
        with pytest.raises(
            ValueError, match="threshold_value must be an integer in \\[0, 255\\]"
        ):
            MarkerDetector(threshold_value=-1)

    def test_invalid_threshold_value_too_large(self) -> None:
        """Test that threshold_value > 255 raises ValueError."""
        with pytest.raises(
            ValueError, match="threshold_value must be an integer in \\[0, 255\\]"
        ):
            MarkerDetector(threshold_value=256)

    def test_repr(self) -> None:
        """Test string representation of MarkerDetector."""
        detector = MarkerDetector(
            min_area=1500,
            max_area=30000,
            aspect_ratio_tolerance=0.25,
            threshold_value=190,
        )

        repr_str = repr(detector)
        assert "MarkerDetector" in repr_str
        assert "min_area=1500" in repr_str
        assert "max_area=30000" in repr_str
        assert "aspect_ratio_tolerance=0.25" in repr_str
        assert "threshold_value=190" in repr_str


class TestInputValidation:
    """Test input validation for detect_markers method."""

    def test_invalid_frame_not_numpy_array(self) -> None:
        """Test that non-numpy array raises ValueError."""
        detector = MarkerDetector()

        with pytest.raises(ValueError, match="rgb_frame must be a numpy array"):
            detector.detect_markers([[1, 2, 3]])  # type: ignore

    def test_invalid_frame_1d(self) -> None:
        """Test that 1D array raises ValueError."""
        detector = MarkerDetector()
        frame = np.zeros(100)

        with pytest.raises(ValueError, match="rgb_frame must be a 3D array"):
            detector.detect_markers(frame)

    def test_invalid_frame_2d(self) -> None:
        """Test that 2D array raises ValueError."""
        detector = MarkerDetector()
        frame = np.zeros((100, 100))

        with pytest.raises(ValueError, match="rgb_frame must be a 3D array"):
            detector.detect_markers(frame)

    def test_invalid_frame_4d(self) -> None:
        """Test that 4D array raises ValueError."""
        detector = MarkerDetector()
        frame = np.zeros((100, 100, 3, 2))

        with pytest.raises(ValueError, match="rgb_frame must be a 3D array"):
            detector.detect_markers(frame)

    def test_invalid_frame_wrong_channels(self) -> None:
        """Test that non-3-channel array raises ValueError."""
        detector = MarkerDetector()
        frame = np.zeros((100, 100, 4))

        with pytest.raises(ValueError, match="rgb_frame must have 3 color channels"):
            detector.detect_markers(frame)

    def test_invalid_frame_zero_height(self) -> None:
        """Test that zero height raises ValueError."""
        detector = MarkerDetector()
        frame = np.zeros((0, 100, 3))

        with pytest.raises(ValueError, match="rgb_frame must have non-zero dimensions"):
            detector.detect_markers(frame)

    def test_invalid_frame_zero_width(self) -> None:
        """Test that zero width raises ValueError."""
        detector = MarkerDetector()
        frame = np.zeros((100, 0, 3))

        with pytest.raises(ValueError, match="rgb_frame must have non-zero dimensions"):
            detector.detect_markers(frame)


class TestSyntheticFrameDetection:
    """Test marker detection on synthetic frames with white squares."""

    def _create_synthetic_frame(
        self,
        width: int,
        height: int,
        marker_centers: list[tuple[int, int]],
        marker_size: int = 100,
    ) -> np.ndarray:
        """Create a synthetic frame with white square markers on black background.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            marker_centers: List of (x, y) center positions for markers.
            marker_size: Size of white square markers.

        Returns:
            RGB frame as numpy array.
        """
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw white squares at marker centers
        for x, y in marker_centers:
            half_size = marker_size // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size)
            y2 = min(height, y + half_size)
            frame[y1:y2, x1:x2] = [255, 255, 255]

        return frame

    def test_detect_four_markers_default_positions(self) -> None:
        """Test detecting 4 markers at default positions."""
        detector = MarkerDetector()

        # Create synthetic frame with 4 markers
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880), (1720, 880)],
            marker_size=100,
        )

        # Detect markers
        markers = detector.detect_markers(frame)

        # Verify exactly 4 markers detected
        assert len(markers) == 4

        # Verify markers are sorted correctly
        # top-left should have smallest x and y
        assert markers[0][0] < markers[1][0]  # top-left.x < top-right.x
        assert markers[0][1] < markers[2][1]  # top-left.y < bottom-left.y
        # top-right should have largest x and small y
        assert markers[1][0] > markers[0][0]  # top-right.x > top-left.x
        assert markers[1][1] < markers[3][1]  # top-right.y < bottom-right.y

    def test_detect_four_markers_centroid_accuracy(self) -> None:
        """Test that centroids are within 2-3 pixels of actual centers."""
        detector = MarkerDetector()

        # Known marker centers
        expected_centers = [(200, 200), (1720, 200), (200, 880), (1720, 880)]

        # Create synthetic frame
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=expected_centers,
            marker_size=100,
        )

        # Detect markers
        detected_markers = detector.detect_markers(frame)

        # Verify each detected centroid is within 3 pixels of expected
        for detected, expected in zip(detected_markers, expected_centers):
            x_error = abs(detected[0] - expected[0])
            y_error = abs(detected[1] - expected[1])
            assert x_error <= 3, f"X error {x_error} > 3 for expected {expected}"
            assert y_error <= 3, f"Y error {y_error} > 3 for expected {expected}"

    def test_detect_markers_small_frame(self) -> None:
        """Test detection on smaller frame (640x480)."""
        detector = MarkerDetector(min_area=500, max_area=5000)

        # Create synthetic frame with 4 markers
        frame = self._create_synthetic_frame(
            width=640,
            height=480,
            marker_centers=[(100, 100), (540, 100), (100, 380), (540, 380)],
            marker_size=50,
        )

        # Detect markers
        markers = detector.detect_markers(frame)

        # Verify exactly 4 markers detected
        assert len(markers) == 4

    def test_detect_markers_offset_positions(self) -> None:
        """Test detection with markers not at extreme corners."""
        detector = MarkerDetector()

        # Create synthetic frame with markers offset from edges
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(400, 300), (1520, 300), (400, 780), (1520, 780)],
            marker_size=100,
        )

        # Detect markers
        markers = detector.detect_markers(frame)

        # Verify exactly 4 markers detected
        assert len(markers) == 4


class TestMarkerFiltering:
    """Test marker filtering logic."""

    def _create_synthetic_frame(
        self,
        width: int,
        height: int,
        marker_centers: list[tuple[int, int]],
        marker_size: int = 100,
    ) -> np.ndarray:
        """Create a synthetic frame with white square markers on black background."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for x, y in marker_centers:
            half_size = marker_size // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size)
            y2 = min(height, y + half_size)
            frame[y1:y2, x1:x2] = [255, 255, 255]

        return frame

    def test_filter_too_small_markers(self) -> None:
        """Test that small markers are filtered out."""
        # Default min_area is 2500 (~50x50)
        detector = MarkerDetector()

        # Create frame with small markers (40x40 = 1600 pixels)
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880), (1720, 880)],
            marker_size=40,  # 40x40 = 1600 < min_area
        )

        # Should fail because all markers are too small
        with pytest.raises(ValueError, match="Detected only .* valid markers"):
            detector.detect_markers(frame)

    def test_filter_too_large_markers(self) -> None:
        """Test that large markers are filtered out."""
        # Default max_area is 40000 (~200x200)
        detector = MarkerDetector()

        # Create frame with large markers (250x250 = 62500 pixels)
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880), (1720, 880)],
            marker_size=250,  # 250x250 = 62500 > max_area
        )

        # Should fail because all markers are too large
        with pytest.raises(ValueError, match="Detected only .* valid markers"):
            detector.detect_markers(frame)

    def test_filter_non_square_rectangles(self) -> None:
        """Test that non-square rectangles are filtered out."""
        # Default aspect_ratio_tolerance is 0.3 (allows 0.7-1.3)
        detector = MarkerDetector()

        # Create frame with rectangular markers (aspect ratio = 2.0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        for x, y in [(200, 200), (1720, 200), (200, 880), (1720, 880)]:
            # Draw 200x100 rectangles (aspect ratio = 2.0)
            x1, y1 = x - 100, y - 50
            x2, y2 = x + 100, y + 50
            frame[y1:y2, x1:x2] = [255, 255, 255]

        # Should fail because markers are not square enough
        with pytest.raises(ValueError, match="Detected only .* valid markers"):
            detector.detect_markers(frame)

    def test_filter_dim_markers(self) -> None:
        """Test that dim (non-white) markers are filtered out."""
        detector = MarkerDetector(threshold_value=200)

        # Create frame with dim gray markers (RGB 180, 180, 180)
        # Grayscale brightness = 180, which is below threshold of 200
        # So no contours are found
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        for x, y in [(200, 200), (1720, 200), (200, 880), (1720, 880)]:
            # Draw gray squares (brightness = 180 < threshold)
            x1, y1 = x - 50, y - 50
            x2, y2 = x + 50, y + 50
            frame[y1:y2, x1:x2] = [180, 180, 180]

        # Should fail because markers are too dim (no contours found after threshold)
        with pytest.raises(ValueError, match="No contours found"):
            detector.detect_markers(frame)

    def test_accept_mixed_valid_and_invalid_markers(self) -> None:
        """Test that only valid markers are returned."""
        detector = MarkerDetector()

        # Create frame with 4 valid and 2 invalid markers
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 4 valid markers (100x100, white)
        for x, y in [(200, 200), (1720, 200), (200, 880), (1720, 880)]:
            x1, y1 = x - 50, y - 50
            x2, y2 = x + 50, y + 50
            frame[y1:y2, x1:x2] = [255, 255, 255]

        # 2 invalid markers (too small)
        for x, y in [(960, 200), (960, 880)]:
            x1, y1 = x - 20, y - 20  # 40x40 = 1600 < min_area
            x2, y2 = x + 20, y + 20
            frame[y1:y2, x1:x2] = [255, 255, 255]

        # Should detect exactly 4 markers (the valid ones)
        markers = detector.detect_markers(frame)
        assert len(markers) == 4


class TestMarkerSorting:
    """Test marker sorting by position."""

    def _create_synthetic_frame(
        self,
        width: int,
        height: int,
        marker_centers: list[tuple[int, int]],
        marker_size: int = 100,
    ) -> np.ndarray:
        """Create a synthetic frame with white square markers on black background."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for x, y in marker_centers:
            half_size = marker_size // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size)
            y2 = min(height, y + half_size)
            frame[y1:y2, x1:x2] = [255, 255, 255]

        return frame

    def test_sort_markers_top_left_top_right_bottom_left_bottom_right(self) -> None:
        """Test that markers are sorted in correct order."""
        detector = MarkerDetector()

        # Create frame with markers at known positions
        # Order in input: bottom-right, top-left, bottom-left, top-right
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(1720, 880), (200, 200), (200, 880), (1720, 200)],
            marker_size=100,
        )

        # Detect and sort markers
        markers = detector.detect_markers(frame)

        # Verify order: top-left, top-right, bottom-left, bottom-right
        assert markers[0][0] < 960 and markers[0][1] < 540  # top-left
        assert markers[1][0] > 960 and markers[1][1] < 540  # top-right
        assert markers[2][0] < 960 and markers[2][1] > 540  # bottom-left
        assert markers[3][0] > 960 and markers[3][1] > 540  # bottom-right

    def test_sort_markers_centered_positions(self) -> None:
        """Test sorting with markers closer to frame center."""
        detector = MarkerDetector()

        # Create frame with markers closer to center
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(400, 300), (1520, 300), (400, 780), (1520, 780)],
            marker_size=100,
        )

        # Detect and sort markers
        markers = detector.detect_markers(frame)

        # Verify order
        assert markers[0] == pytest.approx((400, 300), abs=3)  # top-left
        assert markers[1] == pytest.approx((1520, 300), abs=3)  # top-right
        assert markers[2] == pytest.approx((400, 780), abs=3)  # bottom-left
        assert markers[3] == pytest.approx((1520, 780), abs=3)  # bottom-right


class TestErrorHandling:
    """Test error handling for edge cases."""

    def _create_synthetic_frame(
        self,
        width: int,
        height: int,
        marker_centers: list[tuple[int, int]],
        marker_size: int = 100,
    ) -> np.ndarray:
        """Create a synthetic frame with white square markers on black background."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for x, y in marker_centers:
            half_size = marker_size // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size)
            y2 = min(height, y + half_size)
            frame[y1:y2, x1:x2] = [255, 255, 255]

        return frame

    def test_no_markers_in_frame(self) -> None:
        """Test that error is raised when no markers are present."""
        detector = MarkerDetector()

        # Create empty black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Should fail because no markers are detected
        with pytest.raises(ValueError, match="No contours found"):
            detector.detect_markers(frame)

    def test_too_few_markers(self) -> None:
        """Test that error is raised when fewer than 4 markers are detected."""
        detector = MarkerDetector()

        # Create frame with only 3 markers
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880)],  # Only 3
            marker_size=100,
        )

        # Should fail because only 3 markers are detected
        with pytest.raises(ValueError, match="Detected only .* valid markers"):
            detector.detect_markers(frame)

    def test_more_than_four_valid_markers(self) -> None:
        """Test that 4 largest markers are selected when >4 are detected."""
        detector = MarkerDetector()

        # Create frame with 6 markers (4 large, 2 small)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 4 large markers (100x100)
        for x, y in [(200, 200), (1720, 200), (200, 880), (1720, 880)]:
            x1, y1 = x - 50, y - 50
            x2, y2 = x + 50, y + 50
            frame[y1:y2, x1:x2] = [255, 255, 255]

        # 2 medium markers (80x80) - should be filtered out in favor of 4 largest
        for x, y in [(960, 540), (960, 300)]:
            x1, y1 = x - 40, y - 40
            x2, y2 = x + 40, y + 40
            frame[y1:y2, x1:x2] = [255, 255, 255]

        # Should detect exactly 4 markers (the largest ones)
        markers = detector.detect_markers(frame)
        assert len(markers) == 4

    def test_threshold_too_high(self) -> None:
        """Test that very high threshold still detects pure white markers."""
        detector = MarkerDetector(threshold_value=250)

        # Create frame with white markers (pure white RGB 255, 255, 255)
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880), (1720, 880)],
            marker_size=100,
        )

        # Threshold = 250 is very strict but should still detect pure white
        markers = detector.detect_markers(frame)
        assert len(markers) == 4


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def _create_synthetic_frame(
        self,
        width: int,
        height: int,
        marker_centers: list[tuple[int, int]],
        marker_size: int = 100,
    ) -> np.ndarray:
        """Create a synthetic frame with white square markers on black background."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for x, y in marker_centers:
            half_size = marker_size // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size)
            y2 = min(height, y + half_size)
            frame[y1:y2, x1:x2] = [255, 255, 255]

        return frame

    def test_markers_at_frame_edges(self) -> None:
        """Test detection with markers partially clipped at edges."""
        detector = MarkerDetector()

        # Create frame with markers at very edges
        # Centers are 50 pixels from edge, half-size = 50, so markers touch edges
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(50, 50), (1870, 50), (50, 1030), (1870, 1030)],
            marker_size=100,
        )

        # Should still detect 4 markers
        markers = detector.detect_markers(frame)
        assert len(markers) == 4

    def test_minimal_marker_size(self) -> None:
        """Test detection with markers at near-minimum valid size."""
        detector = MarkerDetector(min_area=2500, max_area=40000)

        # Create frame with markers slightly above minimum valid size
        # 80x80 = 6400 pixels > min_area (2500)
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880), (1720, 880)],
            marker_size=80,
        )

        # Should detect 4 markers
        markers = detector.detect_markers(frame)
        assert len(markers) == 4

    def test_maximal_marker_size(self) -> None:
        """Test detection with markers at maximum valid size."""
        detector = MarkerDetector(min_area=2500, max_area=40000)

        # Create frame with markers at maximum valid size (~200x200)
        frame = self._create_synthetic_frame(
            width=1920,
            height=1080,
            marker_centers=[(200, 200), (1720, 200), (200, 880), (1720, 880)],
            marker_size=200,  # 200x200 = 40000 = max_area
        )

        # Should detect 4 markers
        markers = detector.detect_markers(frame)
        assert len(markers) == 4
