"""Homography matrix computation from calibration points.

This module computes the 3x3 homography matrix that maps camera coordinates
to projector coordinates using 4 corresponding point pairs from the
calibration config.
"""

import numpy as np


def compute_homography(
    camera_points: list[tuple[int, int]],
    projector_points: list[tuple[int, int]],
) -> np.ndarray:
    """Compute 3x3 homography matrix from N point correspondences.

    For exactly 4 points, uses cv2.getPerspectiveTransform (exact solution).
    For more points, uses cv2.findHomography with least squares (more robust).

    Args:
        camera_points: List of N (x, y) tuples in camera coordinate space (N >= 4).
        projector_points: List of N (x, y) tuples in projector coordinate space (N >= 4).

    Returns:
        H: 3x3 homography matrix (float32) that transforms camera coordinates
            to projector coordinates: point_projector = H @ point_camera

    Raises:
        ValueError: If input validation fails:
            - Less than 4 points provided
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
    # Import cv2 only when needed (lazy import for CI environments)
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for homography computation. "
            "Install with: pip install opencv-python"
        ) from e

    # Validate inputs
    if len(camera_points) < 4 or len(projector_points) < 4:
        raise ValueError(
            f"At least 4 points required: "
            f"got {len(camera_points)} camera, {len(projector_points)} projector"
        )

    if len(camera_points) != len(projector_points):
        raise ValueError(
            f"Number of camera and projector points must match: "
            f"got {len(camera_points)} camera, {len(projector_points)} projector"
        )

    # Validate that each point is a (x, y) tuple
    for i, pt in enumerate(camera_points):
        if not isinstance(pt, (tuple, list)) or len(pt) != 2:
            raise ValueError(
                f"camera_points must be (x, y) pairs: camera_points[{i}] = {pt}"
            )
    for i, pt in enumerate(projector_points):
        if not isinstance(pt, (tuple, list)) or len(pt) != 2:
            raise ValueError(
                f"projector_points must be (x, y) pairs: projector_points[{i}] = {pt}"
            )

    n_points = len(camera_points)

    # Convert to numpy arrays with correct shape and dtype
    camera_array = np.array(camera_points, dtype=np.float32).reshape((n_points, 2))
    projector_array = np.array(projector_points, dtype=np.float32).reshape((n_points, 2))

    if n_points == 4:
        # Exact solution for 4 points
        H = cv2.getPerspectiveTransform(camera_array, projector_array)
    else:
        # Least squares solution for N > 4 points (more robust against noise)
        H, mask = cv2.findHomography(camera_array, projector_array, method=0)  # 0 = regular least squares
        if H is None:
            raise ValueError("Failed to compute homography: points may be collinear")

    # Ensure the result is float32 as documented
    H = H.astype(np.float32)

    return H


def apply_homography(point: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Apply homography matrix to a single point.

    Args:
        point: (x, y) tuple in camera coordinates.
        H: 3x3 homography matrix.

    Returns:
        (x_proj, y_proj) transformed point in projector coordinates.
    """
    # Import cv2 only when needed (lazy import for CI environments)
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for homography computation. "
            "Install with: pip install opencv-python"
        ) from e

    # cv2.perspectiveTransform expects shape (N, 1, 2) with dtype float32
    point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)

    # Apply transformation
    transformed = cv2.perspectiveTransform(point_array, H)

    # Extract x, y from (1, 1, 2) result
    return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))


def validate_homography(H: np.ndarray) -> bool:
    """Validate that a homography matrix is well-formed.

    Args:
        H: 3x3 matrix to validate.

    Returns:
        True if valid, False otherwise.
    """
    print(f"Homography: {H}")
    if H.shape != (3, 3):
        return False
    if H.dtype != np.float32:
        print()
        return False
    # Check determinant is not zero (matrix is invertible)
    if abs(np.linalg.det(H)) < 1e-10:
        print(f"Invalid homography: determinant is zero, got {np.linalg.det(H)}")
        return False
    return True
