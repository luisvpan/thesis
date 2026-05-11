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
            depth_frame: Depth frame (can be flipped - will be unflipped internally)
        """
        from pykinect2 import PyKinectV2

        # Unflip depth frame for SDK (it expects raw sensor layout)
        depth_unflipped = cv2.flip(depth_frame, 1)

        depth_flat = depth_unflipped.flatten().astype(np.uint16)
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

    def rgb_to_depth(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Map RGB coordinates to depth coordinates.

        Uses simple linear scaling. Both RGB and depth frames are already
        flipped by PyKinect2HardwareManager, so no flip handling needed here.
        """
        # Simple linear scaling (both frames are already flipped consistently)
        return [
            (
                int(x * self._depth_width / self._color_width),
                int(y * self._depth_height / self._color_height),
            )
            for x, y in points
        ]

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
