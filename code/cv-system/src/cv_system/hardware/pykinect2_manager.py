"""Hardware manager using PyKinect2 (Windows Kinect SDK).

This module provides an alternative to the OpenNI2-based HardwareManager,
using the Windows Kinect SDK via PyKinect2 COM bindings.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cv_system.config import CameraConfig
from cv_system.hardware.manager import HardwareError

# Add pykinect2 fork to path if not already installed
_pykinect2_path = Path(__file__).resolve().parents[4] / "pykinect2" / "src"
if _pykinect2_path.exists() and str(_pykinect2_path) not in sys.path:
    sys.path.insert(0, str(_pykinect2_path))

from pykinect2 import PyKinectRuntime, PyKinectV2


class PyKinect2HardwareManager:
    """
    Hardware manager using PyKinect2 (Windows Kinect SDK via COM).

    Drop-in replacement for HardwareManager when using Kinect v2 on Windows.
    Provides the same interface with identical frame formats.
    """

    def __init__(self) -> None:
        """Initialize an empty manager (sensor not yet connected)."""
        self._kinect: PyKinectRuntime.PyKinectRuntime | None = None
        self._initialized = False
        # Cache last frames (PyKinect2 uses async frame arrival)
        self._last_depth: np.ndarray | None = None
        self._last_color: np.ndarray | None = None
        self._last_ir: np.ndarray | None = None

    def initialize(self, config: CameraConfig) -> None:
        """
        Initialize the Kinect v2 sensor.

        Args:
            config: Camera configuration (used for compatibility, not all
                settings may apply to Kinect v2).

        Raises:
            HardwareError: If the sensor cannot be initialized.
        """
        if self._initialized:
            return

        try:
            # Initialize with Color + Depth + Infrared streams
            self._kinect = PyKinectRuntime.PyKinectRuntime(
                PyKinectV2.FrameSourceTypes_Color
                | PyKinectV2.FrameSourceTypes_Depth
                | PyKinectV2.FrameSourceTypes_Infrared
            )
            self._initialized = True
        except Exception as e:
            raise HardwareError(f"Failed to initialize Kinect v2: {e}") from e

    def get_depth_frame(self) -> np.ndarray:
        """
        Capture and return a depth frame.

        Returns:
            Depth frame as (424, 512) uint16 array, horizontally mirrored.
            Values are in millimeters.

        Raises:
            HardwareError: If not initialized or frame acquisition fails.
        """
        if not self._initialized or self._kinect is None:
            raise HardwareError("Hardware not initialized")

        # Check for new frame, update cache if available
        if self._kinect.has_new_depth_frame():
            frame = self._kinect.get_last_depth_frame()
            if frame is not None:
                depth = frame.reshape((424, 512))
                self._last_depth = cv2.flip(depth, 0)

        # If we have a cached frame, return it
        if self._last_depth is not None:
            return self._last_depth

        # First frame - wait with timeout
        timeout = 3.0  # seconds
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                if frame is not None:
                    depth = frame.reshape((424, 512))
                    self._last_depth = cv2.flip(depth, 0)
                    return self._last_depth
            time.sleep(0.001)  # 1ms sleep to avoid busy-wait

        raise HardwareError("Timeout waiting for initial depth frame")

    def get_rgb_frame(self) -> cv2.UMat:
        """
        Capture and return an RGB frame.

        Returns:
            RGB frame as (1080, 1920, 3) BGR UMat (GPU-accelerated).
            Horizontally mirrored to match depth frame.

        Raises:
            HardwareError: If not initialized or frame acquisition fails.
        """
        if not self._initialized or self._kinect is None:
            raise HardwareError("Hardware not initialized")

        # Check for new frame, update cache if available
        if self._kinect.has_new_color_frame():
            frame = self._kinect.get_last_color_frame()
            if frame is not None:
                color = frame.reshape((1080, 1920, 4))
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                self._last_color = cv2.flip(color, 0)

        # If we have a cached frame, return it as UMat
        if self._last_color is not None:
            return cv2.UMat(self._last_color)

        # First frame - wait with timeout
        timeout = 3.0  # seconds
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                if frame is not None:
                    color = frame.reshape((1080, 1920, 4))
                    color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                    self._last_color = cv2.flip(color, 0)
                    return cv2.UMat(self._last_color)
            time.sleep(0.001)

        raise HardwareError("Timeout waiting for initial color frame")

    def get_ir_frame(self) -> np.ndarray:
        """
        Capture and return an infrared frame.

        Returns:
            IR frame as (424, 512) uint16 array, vertically mirrored.
            Values typically range from 0-4000 (sensor-dependent).

        Raises:
            HardwareError: If not initialized or frame acquisition fails.
        """
        if not self._initialized or self._kinect is None:
            raise HardwareError("Hardware not initialized")

        # Check for new frame, update cache if available
        if self._kinect.has_new_infrared_frame():
            frame = self._kinect.get_last_infrared_frame()
            if frame is not None:
                ir = frame.reshape((424, 512))
                self._last_ir = cv2.flip(ir, 0)

        # If we have a cached frame, return it
        if self._last_ir is not None:
            return self._last_ir

        # First frame - wait with timeout
        timeout = 3.0  # seconds
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._kinect.has_new_infrared_frame():
                frame = self._kinect.get_last_infrared_frame()
                if frame is not None:
                    ir = frame.reshape((424, 512))
                    self._last_ir = cv2.flip(ir, 0)
                    return self._last_ir
            time.sleep(0.001)

        raise HardwareError("Timeout waiting for initial infrared frame")

    def shutdown(self) -> None:
        """Shutdown the Kinect sensor and release resources."""
        if self._kinect is not None:
            self._kinect.close()
            self._kinect = None
        self._initialized = False
        self._last_depth = None
        self._last_color = None
        self._last_ir = None

    @property
    def kinect(self) -> PyKinectRuntime.PyKinectRuntime | None:
        """Access to underlying PyKinectRuntime for coordinate mapping."""
        return self._kinect

    def __enter__(self) -> PyKinect2HardwareManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensures cleanup."""
        self.shutdown()
        return False
