"""Marker Detector for automatic calibration.

This module provides the MarkerDetector class that detects 4 high-contrast
white square markers in RGB frames using threshold and contour detection,
extracting their centroids as camera_corners for homography computation.

The detection pipeline:
1. Convert RGB frame to grayscale
2. Apply threshold to isolate white regions
3. Find contours in the thresholded image
4. Filter contours by size (area), aspect ratio (squareness), and color (white)
5. Extract centroids of valid markers
6. Sort markers by position (top-left, top-right, bottom-left, bottom-right)

Detection parameters are configurable to adapt to different lighting conditions
and camera properties.
"""

import logging

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class MarkerDetector:
    """Detects 4 white square markers in RGB frames for calibration.

    The MarkerDetector processes RGB frames from the camera to find high-contrast
    white square markers projected by MarkerProjector. Detection uses threshold
    and contour analysis, filtering by size, aspect ratio, and color to ensure
    reliable identification.

    Attributes:
        min_area: Minimum contour area in pixels (default: 2500, ~50x50).
        max_area: Maximum contour area in pixels (default: 40000, ~200x200).
        aspect_ratio_tolerance: Max deviation from 1.0 for square detection
            (default: 0.3, allows 0.7-1.3 ratio).
        threshold_value: Threshold value for white detection (default: 200,
            0-255 range, higher = stricter).
    """

    # Default detection parameters
    DEFAULT_MIN_AREA = 2500  # ~50x50 pixels
    DEFAULT_MAX_AREA = 40000  # ~200x200 pixels
    DEFAULT_ASPECT_RATIO_TOLERANCE = 0.3  # Allows 0.7-1.3 ratio
    DEFAULT_THRESHOLD_VALUE = 200  # 0-255 range

    def __init__(
        self,
        min_area: int | None = None,
        max_area: int | None = None,
        aspect_ratio_tolerance: float | None = None,
        threshold_value: int | None = None,
    ) -> None:
        """Initialize the MarkerDetector.

        Args:
            min_area: Minimum contour area in pixels. Defaults to 2500.
            max_area: Maximum contour area in pixels. Defaults to 40000.
            aspect_ratio_tolerance: Maximum deviation from 1.0 for square
                detection. Defaults to 0.3.
            threshold_value: Threshold value for white detection (0-255).
                Higher values are stricter. Defaults to 200.

        Raises:
            ValueError: If any parameter is invalid.
        """
        # Set min_area
        if min_area is None:
            self.min_area = self.DEFAULT_MIN_AREA
        else:
            if not isinstance(min_area, int) or min_area <= 0:
                raise ValueError(f"min_area must be a positive integer: got {min_area}")
            self.min_area = min_area

        # Set max_area
        if max_area is None:
            self.max_area = self.DEFAULT_MAX_AREA
        else:
            if not isinstance(max_area, int) or max_area <= 0:
                raise ValueError(f"max_area must be a positive integer: got {max_area}")
            self.max_area = max_area

        # Validate area range
        if self.max_area <= self.min_area:
            raise ValueError(
                f"max_area ({self.max_area}) must be greater than min_area ({self.min_area})"
            )

        # Set aspect_ratio_tolerance
        if aspect_ratio_tolerance is None:
            self.aspect_ratio_tolerance = self.DEFAULT_ASPECT_RATIO_TOLERANCE
        else:
            if not isinstance(aspect_ratio_tolerance, (int, float)):
                raise ValueError(
                    f"aspect_ratio_tolerance must be a number: got {type(aspect_ratio_tolerance)}"
                )
            if aspect_ratio_tolerance < 0 or aspect_ratio_tolerance > 1.0:
                raise ValueError(
                    f"aspect_ratio_tolerance must be in [0, 1.0]: got {aspect_ratio_tolerance}"
                )
            self.aspect_ratio_tolerance = float(aspect_ratio_tolerance)

        # Set threshold_value
        if threshold_value is None:
            self.threshold_value = self.DEFAULT_THRESHOLD_VALUE
        else:
            if (
                not isinstance(threshold_value, int)
                or threshold_value < 0
                or threshold_value > 255
            ):
                raise ValueError(
                    f"threshold_value must be an integer in [0, 255]: got {threshold_value}"
                )
            self.threshold_value = threshold_value

        logger.info(
            f"MarkerDetector initialized: min_area={self.min_area}, "
            f"max_area={self.max_area}, aspect_ratio_tolerance={self.aspect_ratio_tolerance}, "
            f"threshold_value={self.threshold_value}"
        )

    def detect_markers(self, rgb_frame: np.ndarray) -> list[tuple[int, int]]:
        """Detect 4 white square markers in the RGB frame.

        This method processes the RGB frame to detect white square markers:
        1. Validates the input frame shape (must be height x width x 3)
        2. Converts to grayscale
        3. Applies threshold to isolate white regions
        4. Finds contours and filters by size, aspect ratio, and color
        5. Extracts centroids of valid markers
        6. Sorts markers by position (top-left, top-right, bottom-left, bottom-right)

        Args:
            rgb_frame: RGB image as numpy array with shape (height, width, 3).

        Returns:
            List of 4 (x, y) tuples representing the centroids of detected
            markers, sorted as [top-left, top-right, bottom-left, bottom-right].

        Raises:
            ValueError: If rgb_frame has invalid shape, no markers detected,
                too many markers detected, or threshold fails.
        """
        # Validate input frame shape
        if not isinstance(rgb_frame, np.ndarray):
            raise ValueError(
                f"rgb_frame must be a numpy array: got type {type(rgb_frame)}"
            )

        if rgb_frame.ndim != 3:
            raise ValueError(
                f"rgb_frame must be a 3D array (height, width, 3): "
                f"got shape {rgb_frame.shape} with {rgb_frame.ndim} dimensions"
            )

        if rgb_frame.shape[2] != 3:
            raise ValueError(
                f"rgb_frame must have 3 color channels (RGB): "
                f"got {rgb_frame.shape[2]} channels"
            )

        height, width = rgb_frame.shape[:2]

        if height == 0 or width == 0:
            raise ValueError(
                f"rgb_frame must have non-zero dimensions: got shape {rgb_frame.shape}"
            )

        logger.info(
            f"Validated RGB frame: shape={rgb_frame.shape}, dtype={rgb_frame.dtype}"
        )

        # Import cv2 only when needed (lazy import for CI environments)
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required for marker detection. "
                "Install with: pip install opencv-python"
            ) from e

        # Convert to grayscale
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        logger.debug(f"Converted to grayscale: shape={gray.shape}")

        # Apply threshold to isolate white regions
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        logger.debug(
            f"Applied threshold: value={self.threshold_value}, "
            f"white_pixels={np.count_nonzero(binary)}"
        )

        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        logger.info(f"Found {len(contours)} contours in thresholded image")

        if len(contours) == 0:
            raise ValueError(
                f"No contours found in thresholded image. "
                f"Threshold value {self.threshold_value} may be too high or "
                f"image may not contain white markers."
            )

        # Filter contours by size and aspect ratio
        valid_markers = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                logger.debug(
                    f"Contour {i}: area={area} rejected (outside [{self.min_area}, {self.max_area}])"
                )
                continue

            # Get bounding rectangle for aspect ratio check
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Filter by aspect ratio (should be close to 1.0 for squares)
            min_ratio = 1.0 - self.aspect_ratio_tolerance
            max_ratio = 1.0 + self.aspect_ratio_tolerance
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                logger.debug(
                    f"Contour {i}: aspect_ratio={aspect_ratio:.2f} rejected "
                    f"(outside [{min_ratio:.2f}, {max_ratio:.2f}])"
                )
                continue

            # Check color (should be white in the original RGB frame)
            # Sample the center of the contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_val = cv2.mean(rgb_frame, mask=mask)[:3]
            avg_brightness = sum(mean_val) / 3.0

            # Filter by color (should be bright/white)
            if avg_brightness < self.threshold_value:
                logger.debug(
                    f"Contour {i}: avg_brightness={avg_brightness:.1f} rejected "
                    f"(below threshold {self.threshold_value})"
                )
                continue

            # Extract centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                logger.debug(f"Contour {i}: zero moment, skipped")
                continue

            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            marker_info = {
                "contour_index": i,
                "x": centroid_x,
                "y": centroid_y,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "brightness": avg_brightness,
            }
            valid_markers.append(marker_info)

            logger.debug(
                f"Valid marker {len(valid_markers)}: ({centroid_x}, {centroid_y}), "
                f"area={area}, aspect_ratio={aspect_ratio:.2f}, brightness={avg_brightness:.1f}"
            )

        logger.info(
            f"Filtered to {len(valid_markers)} valid markers from {len(contours)} contours"
        )

        # Validate we found exactly 4 markers
        if len(valid_markers) < 4:
            raise ValueError(
                f"Detected only {len(valid_markers)} valid markers. "
                f"Expected 4 white square markers. "
                f"Adjust detection parameters or verify marker visibility."
            )

        if len(valid_markers) > 4:
            # Sort by area and take the 4 largest (most likely to be the markers)
            valid_markers.sort(key=lambda m: m["area"], reverse=True)
            valid_markers = valid_markers[:4]
            logger.warning(
                f"Detected {len(valid_markers)}+ valid markers. "
                f"Selected the 4 largest by area."
            )

        # Sort markers by position (top-left, top-right, bottom-left, bottom-right)
        sorted_markers = self._sort_markers_by_position(valid_markers)
        camera_corners = [(m["x"], m["y"]) for m in sorted_markers]

        logger.info(
            f"Detected and sorted 4 markers: "
            f"top-left={camera_corners[0]}, top-right={camera_corners[1]}, "
            f"bottom-left={camera_corners[2]}, bottom-right={camera_corners[3]}"
        )

        return camera_corners

    def _sort_markers_by_position(self, markers: list[dict]) -> list[dict]:
        """Sort 4 markers by position.

        Sorts markers as: top-left, top-right, bottom-left, bottom-right.
        This ordering is consistent with homography computation expectations.

        Args:
            markers: List of 4 marker dictionaries with 'x' and 'y' keys.

        Returns:
            Sorted list of marker dictionaries.
        """
        if len(markers) != 4:
            raise ValueError(f"Expected 4 markers to sort, got {len(markers)}")

        # Calculate centroid of all markers
        avg_x = sum(m["x"] for m in markers) / 4.0
        avg_y = sum(m["y"] for m in markers) / 4.0

        # Classify markers into quadrants
        top_markers = [m for m in markers if m["y"] < avg_y]
        bottom_markers = [m for m in markers if m["y"] >= avg_y]

        # Sort top markers by x (left then right)
        top_markers.sort(key=lambda m: m["x"])
        top_left, top_right = top_markers[0], top_markers[1]

        # Sort bottom markers by x (left then right)
        bottom_markers.sort(key=lambda m: m["x"])
        bottom_left, bottom_right = bottom_markers[0], bottom_markers[1]

        logger.debug(
            f"Sorted markers: centroid=({avg_x:.1f}, {avg_y:.1f}), "
            f"top-left=({top_left['x']}, {top_left['y']}), "
            f"top-right=({top_right['x']}, {top_right['y']}), "
            f"bottom-left=({bottom_left['x']}, {bottom_left['y']}), "
            f"bottom-right=({bottom_right['x']}, {bottom_right['y']})"
        )

        return [top_left, top_right, bottom_left, bottom_right]

    def __repr__(self) -> str:
        """Return string representation of MarkerDetector."""
        return (
            f"MarkerDetector(min_area={self.min_area}, max_area={self.max_area}, "
            f"aspect_ratio_tolerance={self.aspect_ratio_tolerance}, "
            f"threshold_value={self.threshold_value})"
        )
