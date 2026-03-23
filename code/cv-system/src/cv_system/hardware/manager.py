"""
Hardware Manager for Kinect V2 integration via OpenNI2.

This module handles all OpenNI2 interactions and provides a clean interface
for capturing depth and RGB frames.
"""

import os
from typing import Optional

import cv2
import numpy as np

from cv_system.config import CameraConfig


class HardwareError(RuntimeError):
    """Raised when hardware initialization or operation fails."""

    pass


class HardwareManager:
    """
    Manages Kinect V2 hardware lifecycle and frame capture.

    This is the only module that imports OpenNI2 (per ADR-004).
    """

    def __init__(self) -> None:
        """Initialize an empty HardwareManager (device not yet connected)."""
        self.context: Optional[object] = None
        self.device: Optional[object] = None
        self.depth_stream: Optional[object] = None
        self.rgb_stream: Optional[object] = None
        self.camera_config: Optional[CameraConfig] = None
        self._initialized: bool = False

    def initialize(self, config: CameraConfig) -> None:
        """
        Initialize OpenNI2 context and discover Kinect V2 device.

        Args:
            config: Camera configuration with resolution and FPS settings.

        Raises:
            HardwareError: If OpenNI2 initialization fails, device not found,
                        or streams cannot be created/configured.
        """
        if self._initialized:
            raise HardwareError("HardwareManager is already initialized")

        try:
            import openni2
        except ImportError as e:
            raise HardwareError(
                "OpenNI2 module not found. "
                "Ensure OPENNI2_REDIST_PATH is set in environment."
            ) from e

        try:
            # Check for OPENNI2_REDIST_PATH
            redist_path = os.getenv("OPENNI2_REDIST_PATH")
            if not redist_path:
                raise HardwareError(
                    "OPENNI2_REDIST_PATH environment variable is not set"
                )

            # Initialize OpenNI2 context
            self.context = openni2.initialize()
            self.camera_config = config

            # Open any Kinect V2 device
            self.device = openni2.Device.open_any()

            # Set up depth stream
            self.depth_stream = self.device.create_depth_stream()
            depth_mode = self._find_video_mode(
                self.depth_stream,
                config.depth_resolution[1],  # width
                config.depth_resolution[0],  # height
                config.fps,
            )
            self.depth_stream.set_video_mode(depth_mode)

            # Set up color stream
            self.rgb_stream = self.device.create_color_stream()
            rgb_mode = self._find_video_mode(
                self.rgb_stream,
                config.rgb_resolution[1],  # width
                config.rgb_resolution[0],  # height
                config.fps,
            )
            self.rgb_stream.set_video_mode(rgb_mode)

            # Start both streams
            self.depth_stream.start()
            self.rgb_stream.start()

            self._initialized = True

        except Exception as e:
            # Clean up any partially initialized state
            self._cleanup()
            raise HardwareError(f"Failed to initialize hardware: {e}") from e

    def _find_video_mode(self, stream, width: int, height: int, fps: int) -> object:
        """
        Find the appropriate video mode for a stream.

        Args:
            stream: The OpenNI2 stream object.
            width: Desired width in pixels.
            height: Desired height in pixels.
            fps: Desired frames per second.

        Returns:
            VideoMode object matching the requested parameters.

        Raises:
            HardwareError: If no matching video mode is found.
        """
        for mode in stream.get_sensor_info().get_supported_video_modes():
            if (
                mode.get_resolutionX() == width
                and mode.get_resolutionY() == height
                and mode.get_fps() == fps
            ):
                return mode

        raise HardwareError(
            f"Video mode {width}x{height} @ {fps}fps not supported by device"
        )

    def get_depth_frame(self) -> np.ndarray:
        """Capture and return a depth frame as numpy array.

        Returns:
            numpy array with shape (height, width) and dtype uint16.
            Frame is mirrored horizontally (left-right flipped).

        Raises:
            HardwareError: If hardware is not initialized or frame capture fails.
        """
        if not self._initialized:
            raise HardwareError("HardwareManager is not initialized")

        try:
            frame = self.depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()

            # Reshape to (height, width)
            depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape(
                self.camera_config.depth_resolution
            )

            # Mirror horizontally
            depth_array = cv2.flip(depth_array, 1)

            return depth_array
        except Exception as e:
            raise HardwareError(f"Failed to capture depth frame: {e}") from e

    def get_rgb_frame(self) -> np.ndarray:
        """Capture and return an RGB frame as numpy array.

        Returns:
            numpy array with shape (height, width, 3) and dtype uint8 (BGR format).
            Frame is mirrored horizontally (left-right flipped).

        Raises:
            HardwareError: If hardware is not initialized or frame capture fails.
        """
        if not self._initialized:
            raise HardwareError("HardwareManager is not initialized")

        try:
            frame = self.rgb_stream.read_frame()
            frame_data = frame.get_buffer_as_uint8()

            # Reshape to (height, width, 3)
            rgb_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (*self.camera_config.rgb_resolution, 3)
            )

            # Mirror horizontally
            rgb_array = cv2.flip(rgb_array, 1)

            # Convert RGB to BGR (OpenCV convention)
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            return rgb_array
        except Exception as e:
            raise HardwareError(f"Failed to capture RGB frame: {e}") from e

    def shutdown(self) -> None:
        """Shutdown hardware and release resources."""
        self._cleanup()
        self._initialized = False

    def _cleanup(self) -> None:
        """Internal cleanup helper."""
        if self.depth_stream is not None:
            try:
                self.depth_stream.stop()
            except Exception:
                pass
            self.depth_stream = None

        if self.rgb_stream is not None:
            try:
                self.rgb_stream.stop()
            except Exception:
                pass
            self.rgb_stream = None

        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None

        if self.context is not None:
            try:
                self.context.unload()
                # Import here to avoid undefined name if initialize never ran
                import openni2

                openni2.unload()
            except Exception:
                pass
            self.context = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.shutdown()
        return False  # Propagate exceptions
