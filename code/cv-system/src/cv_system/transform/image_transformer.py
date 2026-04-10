"""Image transformer for bidirectional camera ↔ projector mapping.

This module provides the ImageTransformer class, a stateless service that
wraps the homography matrix from CalibrationResult and exposes bidirectional
image transformations using cv2.warpPerspective.
"""

import cv2
import numpy as np

from cv_system.config import CameraConfig
from cv_system.calibration.result import CalibrationResult


class ImageTransformer:
    """Stateless image transformer for camera ↔ projector mapping.

    This class wraps the homography matrix H from a CalibrationResult and provides
    bidirectional image transformations between camera and projector space.

    Attributes:
        H: 3x3 homography matrix mapping camera coordinates to projector coordinates.
        H_inv: Inverse homography matrix mapping projector coordinates to camera coordinates.

    The transformer is stateless after initialization — all transformations are
    pure mathematical operations using the stored matrices.
    """

    def __init__(self, calibration_result: CalibrationResult, config: CameraConfig) -> None:
        """Initialize transformer from calibration result.

        Args:
            calibration_result: CalibrationResult containing the homography matrix H.

        Raises:
            ValueError: If H is not invertible (determinant is zero or near-zero).
        """
        self._H = calibration_result.rgb_H.copy()
        self._H_inv = np.linalg.inv(self._H)
        #TODO: considerar 2 resoluciones distintas para cámara y proyector, y usar la adecuada en cada transformación
        self._rgb_resolution = (config.rgb_resolution[1], config.rgb_resolution[0])  # (width, height) for cv2 warp perspective functions

    @property
    def H(self) -> np.ndarray:
        """Read-only access to the homography matrix H (camera -> projector)."""
        return self._H

    @property
    def H_inv(self) -> np.ndarray:
        """Read-only access to the inverse homography matrix (projector -> camera)."""
        return self._H_inv
    
    def camera_to_projector(self, image: np.ndarray) -> np.ndarray:
        """Transform image from camera space to projector space.

        Args:
            image: Input image as a numpy array with shape (H, W, 3) where H and W are
                the height and width of the image. Must have dtype float32.

        Returns:
            Transformed image with same shape (H, W, 3) and dtype float32.

        Raises:
            ValueError: If image.dtype is not float32.
            ValueError: If image.ndim != 3.
            ValueError: If image.shape[2] != 3 (expected 3-channel RGB).
        """
        self._validate_image(image)

        # Transform using OpenCV's warpPerspective
        return cv2.warpPerspective(image, self._H, self._rgb_resolution)

    def projector_to_camera(self, image: np.ndarray) -> np.ndarray:
        """Transform image from projector space to camera space.

        Args:
            image: Input image as a numpy array with shape (H, W, 3) where H and W are
                the height and width of the image. Must have dtype float32.

        Returns:
            Transformed image with same shape (H, W, 3) and dtype float32.

        Raises:
            ValueError: If image.dtype is not float32.
            ValueError: If image.ndim != 3.
            ValueError: If image.shape[2] != 3 (expected 3-channel RGB).
        """
        self._validate_image(image)

        # Transform using inverse homography
        return cv2.warpPerspective(image, self._H_inv, self._rgb_resolution)

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate that the input image has the correct shape and dtype.

        Args:
            image: Input image as a numpy array.
        Raises:
            ValueError: If image.dtype is not float32.
            ValueError: If image.ndim != 3. 
            ValueError: If image.shape[2] != 3 (expected 3-channel RGB).
        """
        if image.dtype != np.float32:
            raise ValueError(
                f"Expected dtype float32, got {image.dtype}. "
                "Use image.astype(np.float32) to convert."
            )
        if image.ndim != 3:
            raise ValueError(
                f"Expected 3D array (H, W, 3) for image data, got {image.ndim}D array. "
                f"Expected shape (H, W, 3), got {image.shape}."
            )
        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image, got {image.shape[2]} channels. "
                f"Expected shape (H, W, 3), got {image.shape}."
            )
        