"""
Touch detector with ring buffer for temporal filtering.

This module implements a touch detection system that compares depth frames against
a dmax_map (maximum depth map representing the table surface). It uses a ring buffer
to accumulate temporal information and filter out transient noise.
"""

import cv2
import numpy as np


class TouchDetector:
    """
    Detects touch points by comparing depth frames against a calibrated dmax_map.

    The detector uses a ring buffer for temporal filtering to reduce noise and
    improve detection reliability. Objects closer to the camera than the table
    surface (i.e., with depth values significantly lower than dmax_map) are
    identified as potential touch points.

    Attributes:
        dmax_map: Read-only array representing the maximum depth at each pixel (uint16).
        ring_buffer_size: Number of frames to accumulate for temporal filtering.
        touch_threshold: Depth difference threshold (mm) to detect foreground objects.
        min_touch_size: Minimum touch area (pixels) to filter out noise.
        max_touch_size: Maximum touch area (pixels) to filter out large objects.
    """

    def __init__(self, dmax_map: np.ndarray, config):
        """
        Initialize the TouchDetector with a dmax_map and detection configuration.

        Args:
            dmax_map: Maximum depth map array with shape (424, 512) and dtype uint16.
                      This represents the calibrated table surface depth.
            config: DetectionConfig instance containing:
                    - ring_buffer_size: Number of frames in ring buffer (positive integer)
                    - touch_threshold: Depth threshold for touch detection (positive integer)
                    - min_touch_size: Minimum touch area in pixels (positive integer)
                    - max_touch_size: Maximum touch area in pixels (positive integer)

        Raises:
            ValueError: If dmax_map has incorrect shape, dtype, or if config has invalid values.
        """
        # Validate dmax_map
        if dmax_map.dtype != np.uint16:
            raise ValueError(f"dmax_map must have dtype uint16, got {dmax_map.dtype}")
        if dmax_map.shape != (424, 512):
            raise ValueError(
                f"dmax_map must have shape (424, 512), got {dmax_map.shape}"
            )

        # Store dmax_map as read-only
        self._dmax_map = dmax_map.copy()
        self._dmax_map.flags.writeable = False

        # Extract and validate config parameters
        ring_buffer_size = getattr(config, "ring_buffer_size", 5)
        touch_threshold = getattr(config, "touch_threshold", 20)
        min_touch_size = getattr(config, "min_touch_size", 10)
        max_touch_size = getattr(config, "max_touch_size", 5000)

        if ring_buffer_size <= 0:
            raise ValueError(
                f"ring_buffer_size must be positive, got {ring_buffer_size}"
            )
        if touch_threshold <= 0:
            raise ValueError(f"touch_threshold must be positive, got {touch_threshold}")
        if min_touch_size <= 0:
            raise ValueError(f"min_touch_size must be positive, got {min_touch_size}")
        if max_touch_size <= 0:
            raise ValueError(f"max_touch_size must be positive, got {max_touch_size}")
        if max_touch_size <= min_touch_size:
            raise ValueError(
                f"max_touch_size ({max_touch_size}) must be greater than "
                f"min_touch_size ({min_touch_size})"
            )

        self.ring_buffer_size = ring_buffer_size
        self.touch_threshold = touch_threshold
        self.min_touch_size = min_touch_size
        self.max_touch_size = max_touch_size

        # Preallocate ring buffer with shape (N, h, w)
        h, w = dmax_map.shape
        self._buffer = np.zeros((ring_buffer_size, h, w), dtype=np.uint8)
        self._idx = 0

    def detect(self, depth_frame: np.ndarray) -> list[tuple[int, int]]:
        """
        Detect touch points in a depth frame.

        The method computes the depth difference between the current frame and
        the dmax_map, applies a threshold to identify foreground objects, and
        uses temporal filtering via the ring buffer to reduce noise.

        Args:
            depth_frame: Current depth frame with shape (424, 512) and dtype uint16.

        Returns:
            List of (x, y) tuples representing touch centroids in camera coordinates.
            Returns an empty list if no touches are detected.

        Raises:
            ValueError: If depth_frame shape or dtype doesn't match dmax_map.
        """
        # Validate depth_frame
        if depth_frame.shape != self._dmax_map.shape:
            raise ValueError(
                f"depth_frame shape {depth_frame.shape} must match "
                f"dmax_map shape {self._dmax_map.shape}"
            )
        if depth_frame.dtype != np.uint16:
            raise ValueError(f"depth_frame dtype {depth_frame.dtype} must be uint16")

        # Compute depth difference using int16 to avoid overflow
        # Lower depth values mean closer to the camera
        diff = depth_frame.astype(np.int16) - self._dmax_map.astype(np.int16)

        # Create binary mask: pixels significantly closer than dmax
        # Negative diff means current depth is less than dmax (closer to camera)
        mask = (diff < -self.touch_threshold).astype(np.uint8) * 255

        # Update ring buffer
        N = self.ring_buffer_size
        self._buffer[self._idx % N] = mask
        self._idx += 1

        # Accumulate buffer
        accumulated = np.sum(self._buffer, axis=0)

        # Apply persistence threshold: majority voting
        # A pixel is considered a touch if it appears in majority of frames
        # For N frames, need at least (N - N//2) frames to agree
        persistence_threshold = (N - N // 2) * 255
        touch_mask = (accumulated >= persistence_threshold).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            touch_mask, connectivity=8
        )

        # Filter by area and extract centroids
        touches = []
        # stats[0] is the background component, skip it
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_touch_size <= area <= self.max_touch_size:
                # Centroids are in (x, y) format
                cx, cy = centroids[i]
                touches.append((int(cx), int(cy)))

        return touches

    def reset(self) -> None:
        """
        Clear the ring buffer and reset the frame counter.

        This method is useful for resetting the temporal state when
        calibration changes or when starting a new session.
        """
        self._buffer.fill(0)
        self._idx = 0
