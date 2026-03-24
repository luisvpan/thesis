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

    def project_markers(self, projector_corners: list[tuple[int, int]]) -> int:
        """Project 4 white square markers at the specified corner positions.

        This method creates a black image at the projector resolution, draws
        4 white squares at the specified corner positions, and displays the
        image fullscreen using OpenCV.

        The window remains open until explicitly destroyed by calling
        destroy_window() or the program exits.

        Args:
            projector_corners: List of 4 (x, y) tuples specifying the center
                positions of the white square markers. Each corner must be
                within the image bounds.

        Returns:
            The OpenCV window handle for the fullscreen window.

        Raises:
            ValueError: If projector_corners is invalid (wrong count, invalid
                format, or out-of-bounds coordinates).
        """
        # Validate input: exactly 4 corners required
        if not isinstance(projector_corners, list):
            raise ValueError(
                f"projector_corners must be a list of 4 (x, y) tuples: "
                f"got type {type(projector_corners)}"
            )

        if len(projector_corners) != 4:
            raise ValueError(
                f"Exactly 4 corners required: got {len(projector_corners)} corners"
            )

        # Validate each corner
        for i, corner in enumerate(projector_corners):
            if not isinstance(corner, (tuple, list)) or len(corner) != 2:
                raise ValueError(
                    f"projector_corners[{i}] must be an (x, y) tuple: got {corner}"
                )

            x, y = corner

            # Validate coordinates are integers
            if not isinstance(x, (int, np.integer)) or not isinstance(
                y, (int, np.integer)
            ):
                raise ValueError(
                    f"projector_corners[{i}] coordinates must be integers: "
                    f"got ({x}, {y})"
                )

            x_int = int(x)
            y_int = int(y)

            # Validate coordinates are within bounds
            width, height = self.resolution
            if x_int < 0 or x_int >= width:
                raise ValueError(
                    f"projector_corners[{i}] x-coordinate out of bounds: "
                    f"{x_int} not in [0, {width})"
                )
            if y_int < 0 or y_int >= height:
                raise ValueError(
                    f"projector_corners[{i}] y-coordinate out of bounds: "
                    f"{y_int} not in [0, {height})"
                )

        logger.info(f"Validated {len(projector_corners)} projector corners")

        # Create black image at projector resolution
        width, height = self.resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)

        logger.info(
            f"Created black image: resolution={self.resolution}, shape={image.shape}"
        )

        # Draw white squares at each corner position
        for i, (x, y) in enumerate(projector_corners):
            x_int, y_int = int(x), int(y)
            half_size = self.marker_size // 2

            # Calculate square bounds (clipped to image bounds)
            x1 = max(0, x_int - half_size)
            y1 = max(0, y_int - half_size)
            x2 = min(width, x_int + half_size)
            y2 = min(height, y_int + half_size)

            # Draw white square
            image[y1:y2, x1:x2] = [255, 255, 255]

            logger.debug(
                f"Drew marker {i + 1} at ({x_int}, {y_int}), "
                f"bounds=({x1}, {y1}, {x2}, {y2})"
            )

        logger.info(f"Drew {len(projector_corners)} white square markers")

        # Import cv2 only when needed (lazy import for CI environments)
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required for marker projection. "
                "Install with: pip install opencv-python"
            ) from e

        # Create fullscreen window
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_FULLSCREEN)

        # Display the image
        cv2.imshow(self.WINDOW_NAME, image)

        # Process pending events to ensure window appears
        cv2.waitKey(1)

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
