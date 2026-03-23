"""Coordinate transformer for bidirectional camera ↔ projector mapping.

This module provides the CoordinateTransformer class, a stateless service that
wraps the homography matrix from CalibrationResult and exposes bidirectional
point transformations using cv2.perspectiveTransform.
"""

import cv2
import numpy as np

from cv_system.calibration.result import CalibrationResult


class CoordinateTransformer:
    """Stateless coordinate transformer for camera ↔ projector mapping.

    This class wraps the homography matrix H from a CalibrationResult and provides
    bidirectional coordinate transformations between camera and projector space.

    Attributes:
        H: 3x3 homography matrix mapping camera coordinates to projector coordinates.
        H_inv: Inverse homography matrix mapping projector coordinates to camera coordinates.

    The transformer is stateless after initialization — all transformations are
    pure mathematical operations using the stored matrices.
    """

    def __init__(self, calibration_result: CalibrationResult) -> None:
        """Initialize transformer from calibration result.

        Args:
            calibration_result: CalibrationResult containing the homography matrix H.

        Raises:
            ValueError: If H is not invertible (determinant is zero or near-zero).
        """
        self._H = calibration_result.H.copy()
        self._H_inv = np.linalg.inv(self._H)

    @property
    def H(self) -> np.ndarray:
        """Read-only access to the homography matrix H (camera → projector)."""
        return self._H

    @property
    def H_inv(self) -> np.ndarray:
        """Read-only access to the inverse homography matrix (projector → camera)."""
        return self._H_inv

    def camera_to_projector(self, point: np.ndarray) -> np.ndarray:
        """Transform point(s) from camera space to projector space.

        Args:
            point: Input point(s) as a numpy array with shape (N, 1, 2) where N is
                the number of points. Must have dtype float32.

        Returns:
            Transformed point(s) with same shape (N, 1, 2) and dtype float32.

        Raises:
            ValueError: If point.dtype is not float32.
            ValueError: If point.ndim != 3.
            ValueError: If point.shape[2] != 2 (expected 2D coordinates).
        """
        # Validate dtype
        if point.dtype != np.float32:
            raise ValueError(
                f"Expected dtype float32, got {point.dtype}. "
                "Use point.astype(np.float32) to convert."
            )

        # Validate dimensions
        if point.ndim != 3:
            raise ValueError(
                f"Expected 3D array (N,1,2) for batch dimension, got {point.ndim}D array. "
                f"Expected shape (N,1,2), got {point.shape}."
            )

        # Validate shape
        if point.shape[2] != 2:
            raise ValueError(
                f"Expected 2 coordinates (x, y) in last dimension, got {point.shape[2]}. "
                f"Expected shape (N,1,2), got {point.shape}."
            )

        # Transform using OpenCV's perspectiveTransform
        result = cv2.perspectiveTransform(point, self._H)

        # Ensure result is float32
        return result.astype(np.float32)

    def projector_to_camera(self, point: np.ndarray) -> np.ndarray:
        """Transform point(s) from projector space to camera space.

        Args:
            point: Input point(s) as a numpy array with shape (N, 1, 2) where N is
                the number of points. Must have dtype float32.

        Returns:
            Transformed point(s) with same shape (N, 1, 2) and dtype float32.

        Raises:
            ValueError: If point.dtype is not float32.
            ValueError: If point.ndim != 3.
            ValueError: If point.shape[2] != 2 (expected 2D coordinates).
        """
        # Validate dtype
        if point.dtype != np.float32:
            raise ValueError(
                f"Expected dtype float32, got {point.dtype}. "
                "Use point.astype(np.float32) to convert."
            )

        # Validate dimensions
        if point.ndim != 3:
            raise ValueError(
                f"Expected 3D array (N,1,2) for batch dimension, got {point.ndim}D array. "
                f"Expected shape (N,1,2), got {point.shape}."
            )

        # Validate shape
        if point.shape[2] != 2:
            raise ValueError(
                f"Expected 2 coordinates (x, y) in last dimension, got {point.shape[2]}. "
                f"Expected shape (N,1,2), got {point.shape}."
            )

        # Transform using inverse homography
        result = cv2.perspectiveTransform(point, self._H_inv)

        # Ensure result is float32
        return result.astype(np.float32)
