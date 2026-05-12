"""Resolution mapper using PyKinect2's native coordinate mapping."""

from __future__ import annotations

import ctypes

import cv2
import numpy as np


class PyKinect2ResolutionMapper:
    """Maps pixel coordinates between RGB and depth frames using Kinect SDK.

    Uses MapColorFrameToDepthSpace for accurate mapping instead of linear scaling.

    Note: PyKinect2HardwareManager returns horizontally flipped frames for natural
    mirror-view. This mapper handles the flip internally - callers can use flipped
    frame coordinates directly.
    """

    def __init__(self, kinect) -> None:
        """
        Args:
            kinect: PyKinectRuntime instance (needed for mapper access)
        """
        self._kinect = kinect
        self._color_width = 1920
        self._color_height = 1080
        self._depth_width = 512
        self._depth_height = 424

        # Cache for color->depth mapping (updated per frame)
        self._depth_space_points = None

    def update_mapping(self, depth_frame: np.ndarray) -> None:
        """Update the color-to-depth mapping with current depth frame.

        Must be called once per frame before using rgb_to_depth.

        Args:
            depth_frame: Depth frame (vertically flipped, same as color frame)
        """
        from pykinect2 import PyKinectV2

        # Give SDK the depth frame AS-IS (flipped).
        # The SDK will compute mappings in the flipped coordinate system,
        # matching the flipped color frame.
        depth_flat = depth_frame.flatten().astype(np.uint16)
        depth_ptr = depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        self._depth_space_points = (
            PyKinectV2._DepthSpacePoint * (self._color_width * self._color_height)
        )()

        self._kinect._mapper.MapColorFrameToDepthSpace(
            self._depth_width * self._depth_height,
            depth_ptr,
            self._color_width * self._color_height,
            self._depth_space_points,
        )

    def rgb_to_depth(self, points: list[tuple[int, int]], debug: bool = False) -> list[tuple[int, int]]:
        """Map RGB coordinates to depth coordinates.

        Uses Kinect SDK's MapColorFrameToDepthSpace for accurate mapping.
        Requires update_mapping() to be called first with current depth frame.

        Args:
            points: List of (x, y) coordinates in RGB space (vertically flipped frame).
            debug: If True, print debug info about the mapping.

        Returns:
            List of (x, y) coordinates in depth space (vertically flipped frame).
            Invalid mappings (where SDK returns -inf) return (-1, -1).
        """
        if self._depth_space_points is None:
            # Fallback to linear scaling if mapping not initialized
            if debug:
                print("[rgb_to_depth] WARNING: SDK mapping not initialized, using linear scaling")
            return [
                (
                    int(x * self._depth_width / self._color_width),
                    int(y * self._depth_height / self._color_height),
                )
                for x, y in points
            ]

        result = []
        for x, y in points:
            # Bounds check
            if not (0 <= x < self._color_width and 0 <= y < self._color_height):
                result.append((-1, -1))
                continue

            # Look up depth coordinate from SDK mapping (row-major: y * width + x)
            # Both color and depth frames are flipped, so SDK mapping is in flipped space
            idx = y * self._color_width + x
            depth_point = self._depth_space_points[idx]
            depth_x = depth_point.x
            depth_y = depth_point.y

            # SDK returns -inf for invalid mappings
            if depth_x < 0 or depth_y < 0 or depth_x >= self._depth_width or depth_y >= self._depth_height:
                if debug:
                    print(f"[rgb_to_depth] rgb({x},{y}) -> INVALID (sdk returned {depth_x:.1f},{depth_y:.1f})")
                result.append((-1, -1))
                continue

            if debug:
                print(f"[rgb_to_depth] rgb({x},{y}) -> depth({int(depth_x)},{int(depth_y)})")

            result.append((int(depth_x), int(depth_y)))

        return result

    def depth_to_rgb(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Map depth coordinates to RGB coordinates.

        Uses simple linear scaling (inverse of depth resolution to color resolution).
        Works with flipped frame coordinates.
        """
        return [
            (
                int(x * self._color_width / self._depth_width),
                int(y * self._color_height / self._depth_height),
            )
            for x, y in points
        ]
