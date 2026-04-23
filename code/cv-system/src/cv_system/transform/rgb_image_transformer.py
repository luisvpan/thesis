"""Image transformer for bidirectional camera ↔ projector mapping.

This module provides the RgbImageTransformer class, a stateless service that
wraps the homography matrix from CalibrationResult and exposes bidirectional
image transformations using cv2.warpPerspective.
"""

import cv2
import numpy as np

from cv_system.config import CameraConfig
from cv_system.calibration.result import CalibrationResult


class RgbImageTransformer:
    """Stateless image transformer for camera ↔ projector mapping.

    This class wraps the homography matrix H from a CalibrationResult and provides
    bidirectional image transformations between camera and projector space.

    Attributes:
        H: 3x3 homography matrix mapping camera coordinates to projector coordinates.
        H_inv: Inverse homography matrix mapping projector coordinates to camera coordinates.

    The transformer is stateless after initialization — all transformations are
    pure mathematical operations using the stored matrices.
    """

    def __init__(
        self,
        calibration_result: CalibrationResult,
        config: CameraConfig,
        projector_corners: list[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize transformer from calibration result.

        Args:
            calibration_result: CalibrationResult containing the homography matrix H.

        Raises:
            ValueError: If H is not invertible (determinant is zero or near-zero).
        """
        self._H = calibration_result.rgb_H.copy()
        # warpPerspective `dsize` is the destination image size (width, height).
        # rgb_H maps Kinect RGB pixels → projector pixels (same space as projector_corners).
        rgb_h, rgb_w = config.rgb_resolution
        proj_h, proj_w = config.projector_resolution
        self._camera_wh = (rgb_w, rgb_h)

        # If calibration corners define a smaller projected ROI, normalize the output
        # space to that ROI so homography dimensions match the projected square area.
        if projector_corners:
            xs = [int(p[0]) for p in projector_corners]
            ys = [int(p[1]) for p in projector_corners]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            roi_w = max_x - min_x
            roi_h = max_y - min_y
            if roi_w <= 0 or roi_h <= 0:
                raise ValueError(
                    f"Invalid projector ROI from corners: width={roi_w}, height={roi_h}"
                )

            translate_to_roi = np.array(
                [[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self._H = (translate_to_roi @ self._H).astype(np.float32)
            self._projector_wh = (roi_w, roi_h)
        else:
            self._projector_wh = (proj_w, proj_h)

        self._H_inv = np.linalg.inv(self._H)

    @property
    def H(self) -> np.ndarray:
        """Read-only access to the homography matrix H (camera -> projector)."""
        return self._H

    @property
    def H_inv(self) -> np.ndarray:
        """Read-only access to the inverse homography matrix (projector -> camera)."""
        return self._H_inv

    @property
    def projector_wh(self) -> tuple[int, int]:
        """Read-only output size (width, height) for projector-space images."""
        return self._projector_wh
    
    def camera_to_projector(self, image: cv2.UMat) -> cv2.UMat:
        """Transform image from camera space to projector space.

        Args:
            image: Input image as cv2.UMat (GPU memory) with 3 channels (BGR).

        Returns:
            Transformed image as cv2.UMat, stays on GPU for efficient chaining.
        """
        # warpPerspective works natively with UMat (GPU accelerated)
        return cv2.warpPerspective(image, self._H, self._projector_wh)

    def projector_to_camera(self, image: cv2.UMat) -> cv2.UMat:
        """Transform image from projector space to camera space.

        Args:
            image: Input image as cv2.UMat (GPU memory) with 3 channels (BGR).

        Returns:
            Transformed image as cv2.UMat, stays on GPU for efficient chaining.
        """
        # warpPerspective works natively with UMat (GPU accelerated)
        return cv2.warpPerspective(image, self._H_inv, self._camera_wh)
        