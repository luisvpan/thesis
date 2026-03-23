"""Calibration result dataclass.

This module defines the immutable CalibrationResult dataclass that contains
the homography matrix, dmax_map, and metadata computed during calibration.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable calibration result containing transformation data.

    Attributes:
        H: 3x3 homography matrix mapping camera coordinates to projector coordinates.
        dmax_map: 2D array where each pixel contains the most frequent depth value
            within the configured range across N calibration frames.
        metadata: Dictionary containing calibration metadata such as number of frames
            captured, depth range used, timestamp, etc.

    The frozen=True decorator ensures this result is immutable after creation,
    preventing accidental modification of the calibration data.
    """

    H: np.ndarray
    dmax_map: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate that arrays have expected shapes and types."""
        # Validate H is 3x3
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")

        # Validate H is float32
        if self.H.dtype != np.float32:
            raise ValueError(f"Homography matrix must be float32, got {self.H.dtype}")

        # Validate dmax_map is 2D
        if self.dmax_map.ndim != 2:
            raise ValueError(
                f"dmax_map must be 2D, got {self.dmax_map.ndim} dimensions"
            )

    def __repr__(self) -> str:
        """Return a concise representation of the calibration result."""
        return (
            f"CalibrationResult(H_shape={self.H.shape}, "
            f"dmax_map_shape={self.dmax_map.shape}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )
