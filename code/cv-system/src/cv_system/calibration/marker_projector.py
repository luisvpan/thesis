"""Marker Projector for automatic calibration.

This module provides the MarkerProjector class that displays 4 high-contrast
white square markers at configured projector_corners using OpenCV fullscreen
windowing. This enables automatic calibration by projecting detectable
reference points.

The projection process:
1. Create a black image at projector resolution (1920x1080)
2. Draw 4 white squares at the specified projector_corners positions
3. Display the image fullscreen using OpenCV's cv2.WINDOW_FULLSCREEN flag
4. The markers remain visible until explicitly destroyed

Marker squares are 100x100 pixels by default, providing high-contrast
reference points that are easily detectable in the Kinect's RGB image.
"""

import logging

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class MarkerProjector:
    """Projects 4 calibration markers on the display for automatic detection.

    The MarkerProjector creates a black image with white square markers at
    configured corner positions and displays it fullscreen. These markers serve
    as reference points for the marker detector to identify in the RGB camera
    image, enabling automatic homography computation.

    Attributes:
        resolution: Projector resolution as (width, height) tuple.
        marker_size: Size of the white square markers in pixels (default: 100).
    """

    # Default projector resolution (1920x1080 HD)
    DEFAULT_RESOLUTION = (1920, 1080)

    # Default marker square size (100x100 pixels)
    DEFAULT_MARKER_SIZE = 100

    # OpenCV window name for the marker projection
    WINDOW_NAME = "Calibration Markers"

    # TODO: consider if it's needed to move this configuration to config file
    # Position to move the window
    WINDOW_DISPLAY_POSITION = (1920, 0)

    def __init__(
        self,
        resolution: tuple[int, int] | None = None,
        marker_size: int | None = None,
    ) -> None:
        """Initialize the MarkerProjector.

        Args:
            resolution: Projector resolution as (width, height). Defaults to
                (1920, 1080) if not specified.
            marker_size: Size of the white square markers in pixels. Defaults
                to 100 if not specified.

        Raises:
            ValueError: If resolution or marker_size is invalid.
        """
        # Set resolution
        if resolution is None:
            self.resolution = self.DEFAULT_RESOLUTION
        else:
            if (
                not isinstance(resolution, (tuple, list))
                or len(resolution) != 2
                or resolution[0] <= 0
                or resolution[1] <= 0
            ):
                raise ValueError(
                    f"resolution must be a (width, height) tuple with positive integers: "
                    f"got {resolution}"
                )
            self.resolution = (int(resolution[0]), int(resolution[1]))

        # Set marker size
        if marker_size is None:
            self.marker_size = self.DEFAULT_MARKER_SIZE
        else:
            if not isinstance(marker_size, int) or marker_size <= 0:
                raise ValueError(
                    f"marker_size must be a positive integer: got {marker_size}"
                )
            self.marker_size = marker_size

        logger.info(
            f"MarkerProjector initialized: resolution={self.resolution}, "
            f"marker_size={self.marker_size}"
        )

    def project_markers(self, projector_points: list[tuple[int, int]]) -> int:
        """Project white square markers at the specified positions.

        This method creates a black image at the projector resolution, draws
        white squares at the specified positions, and displays the image fullscreen.

        Supports any number of points (4 for corners, 9 for 3x3 grid, etc.)

        Args:
            projector_points: List of (x, y) tuples specifying the CENTER positions
                of the white square markers. Each point must be within the image bounds.

        Returns:
            The OpenCV window handle for the fullscreen window.

        Raises:
            ValueError: If projector_points is invalid.
        """
        # Validate input
        if not isinstance(projector_points, list):
            raise ValueError(
                f"projector_points must be a list of (x, y) tuples: "
                f"got type {type(projector_points)}"
            )

        if len(projector_points) < 4:
            raise ValueError(
                f"At least 4 points required: got {len(projector_points)} points"
            )

        # Validate each point
        width, height = self.resolution
        for i, point in enumerate(projector_points):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError(
                    f"projector_points[{i}] must be an (x, y) tuple: got {point}"
                )

            x, y = point

            # Validate coordinates are integers
            if not isinstance(x, (int, np.integer)) or not isinstance(
                y, (int, np.integer)
            ):
                raise ValueError(
                    f"projector_points[{i}] coordinates must be integers: "
                    f"got ({x}, {y})"
                )

            # Validate coordinates are within bounds (with margin for marker size)
            margin = self.marker_size // 2
            if x < margin or x > width - margin:
                raise ValueError(
                    f"projector_points[{i}] x-coordinate out of bounds: "
                    f"{x} not in [{margin}, {width - margin}]"
                )
            if y < margin or y > height - margin:
                raise ValueError(
                    f"projector_points[{i}] y-coordinate out of bounds: "
                    f"{y} not in [{margin}, {height - margin}]"
                )

        logger.info(f"Validated {len(projector_points)} projector points")

        # Create black image at projector resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)

        logger.info(
            f"Created black image: resolution={self.resolution}, shape={image.shape}"
        )

        # Draw white squares at each point (centered on the point)
        half_size = self.marker_size // 2
        for i, (x, y) in enumerate(projector_points):
            x_center, y_center = int(x), int(y)

            # Calculate square bounds (clipped to image bounds)
            x1 = max(0, x_center - half_size)
            y1 = max(0, y_center - half_size)
            x2 = min(width, x_center + half_size)
            y2 = min(height, y_center + half_size)

            print(
                f"Drawing marker {i + 1}/{len(projector_points)} at ({x_center}, {y_center})"
            )

            # Draw magenta square
            image[y1:y2, x1:x2] = [255, 0, 255]

            logger.debug(
                f"Drew marker {i + 1} at ({x_center}, {y_center}), "
                f"bounds=({x1}, {y1}, {x2}, {y2})"
            )

        logger.info(f"Drew {len(projector_points)} white square markers")

        # Import cv2 only when needed (lazy import for CI environments)
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required for marker projection. "
                "Install with: pip install opencv-python"
            ) from e

        # Create fullscreen window
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.WINDOW_NAME, *self.WINDOW_DISPLAY_POSITION)
        cv2.setWindowProperty(
            self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        # Display the image
        cv2.imshow(self.WINDOW_NAME, image)

        logger.info(
            f"Projected markers fullscreen: window='{self.WINDOW_NAME}', "
            f"resolution={self.resolution}, fullscreen_mode=cv2.WINDOW_FULLSCREEN"
        )

        # Return the window name (OpenCV doesn't expose window handles)
        return id(self.WINDOW_NAME)

    def destroy_window(self) -> None:
        """Destroy the fullscreen marker projection window.

        This method closes the OpenCV window displaying the calibration markers.
        Call this when calibration is complete to release the window and
        return to normal display mode.
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required for marker projection. "
                "Install with: pip install opencv-python"
            ) from e

        cv2.destroyWindow(self.WINDOW_NAME)
        logger.info(f"Destroyed marker projection window: '{self.WINDOW_NAME}'")

    def __repr__(self) -> str:
        """Return string representation of MarkerProjector."""
        return (
            f"MarkerProjector(resolution={self.resolution}, "
            f"marker_size={self.marker_size})"
        )
