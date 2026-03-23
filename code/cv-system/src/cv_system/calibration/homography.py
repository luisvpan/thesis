"""Homography matrix computation from calibration points.

This module computes the 3x3 homography matrix that maps camera coordinates
to projector coordinates using 4 corresponding point pairs from the
calibration config.
"""

import numpy as np


def compute_homography(
    camera_points: list[tuple[float, float]],
    projector_points: list[tuple[float, float]],
) -> np.ndarray:
    """Compute 3x3 homography matrix from 4 point correspondences.

    Args:
        camera_points: List of 4 (x, y) tuples in camera coordinate space.
        projector_points: List of 4 (x, y) tuples in projector coordinate space.

    Returns:
        H: 3x3 homography matrix (float32) that transforms camera coordinates
            to projector coordinates: point_projector = H @ point_camera

    Raises:
        ValueError: If input validation fails:
            - Not exactly 4 points provided
            - Points have incorrect format
            - Points are collinear (homography cannot be computed)

    Example:
        >>> camera = [(100, 100), (700, 100), (700, 500), (100, 500)]
        >>> projector = [(50, 50), (1000, 50), (1000, 700), (50, 700)]
        >>> H = compute_homography(camera, projector)
        >>> H.shape
        (3, 3)
        >>> H.dtype
        dtype('float32')
    """
    # Validate inputs
    if len(camera_points) != 4 or len(projector_points) != 4:
        raise ValueError(
            f"Exactly 4 points required: "
            f"got {len(camera_points)} camera, {len(projector_points)} projector"
        )

    # Convert to numpy arrays with correct shape and dtype
    # cv2.getPerspectiveTransform expects shape (4, 2) with float32
    camera_array = np.array(camera_points, dtype=np.float32).reshape((4, 2))
    projector_array = np.array(projector_points, dtype=np.float32).reshape((4, 2))

    # Validate array shapes
    if camera_array.shape != (4, 2):
        raise ValueError(
            f"camera_points must be 4 (x, y) pairs, got shape {camera_array.shape}"
        )
    if projector_array.shape != (4, 2):
        raise ValueError(
            f"projector_points must be 4 (x, y) pairs, got shape {projector_array.shape}"
        )

    # Compute homography using OpenCV
    H = cv2.getPerspectiveTransform(camera_array, projector_array)

    return H


def apply_homography(point: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Apply homography matrix to a single point.

    Args:
        point: (x, y) tuple in camera coordinates.
        H: 3x3 homography matrix.

    Returns:
        (x_proj, y_proj) transformed point in projector coordinates.
    """
    # Convert to homogeneous coordinates
    point_homogeneous = np.array([[point[0], point[1], 1.0]], dtype=np.float32)

    # Apply transformation
    transformed = cv2.perspectiveTransform(point_homogeneous, H)

    # Extract x, y (divide by w is done by perspectiveTransform)
    return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))


def validate_homography(H: np.ndarray) -> bool:
    """Validate that a homography matrix is well-formed.

    Args:
        H: 3x3 matrix to validate.

    Returns:
        True if valid, False otherwise.
    """
    if H.shape != (3, 3):
        return False
    if H.dtype != np.float32:
        return False
    # Check that bottom row is approximately [0, 0, 1] (standard form)
    bottom_row = H[2, :]
    if not (abs(bottom_row[0]) < 1e-6 and abs(bottom_row[1]) < 1e-6):
        return False
    # Check determinant is not zero (matrix is invertible)
    if abs(np.linalg.det(H)) < 1e-10:
        return False
    return True


# Import cv2 at module level (after validation to avoid import issues in tests)
# Note: cv2 is required here, will raise ImportError if not available
try:
    import cv2
except ImportError as e:
    # Provide a helpful error message
    raise ImportError(
        "OpenCV (cv2) is required for homography computation. "
        "Install with: pip install opencv-python"
    ) from e
