"""
Hardware Manager for Kinect V2 integration via OpenNI2.

This module handles all OpenNI2 interactions and provides a clean interface
for capturing depth and RGB frames.
"""

import logging
import os
from types import ModuleType
from typing import Optional, cast
from openni import openni2
from openni.openni2 import VideoStream, Device, VideoMode, OpenNIError

import cv2
import numpy as np

from cv_system.config import CameraConfig

logger = logging.getLogger(__name__)


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
        self.context: Optional[ModuleType] = None
        self.device: Optional[Device] = None
        self.depth_stream: Optional[VideoStream] = None
        self.rgb_stream: Optional[VideoStream] = None
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
            # Check for OPENNI2_REDIST_PATH
            redist_path = os.getenv("OPENNI2_REDIST_PATH")
            if not redist_path:
                raise HardwareError(
                    "OPENNI2_REDIST_PATH environment variable is not set"
                )

            # Initialize OpenNI2 context
            openni2.initialize(redist_path)
            self.context = openni2
            self.camera_config = config

            # Open any Kinect V2 device
            try:
                self.device = openni2.Device.open_any()
            except OpenNIError as e:
                raise HardwareError("No Kinect V2 device found") from e

            # Set up depth stream
            self.depth_stream = self.device.create_depth_stream()
            if self.depth_stream is None:
                raise HardwareError("Failed to create depth stream")
            depth_mode = self._find_video_mode(
                self.depth_stream,
                config.depth_resolution[1],  # width
                config.depth_resolution[0],  # height
                config.fps,
                openni2.PIXEL_FORMAT_DEPTH_1_MM,
            )
            self.depth_stream.set_video_mode(depth_mode)

            # Set up color stream
            self.rgb_stream = self.device.create_color_stream()
            if self.rgb_stream is None:
                raise HardwareError("Failed to create RGB stream")
            # TODO: include pixelFormat in config and check for it in _find_video_mode
            rgb_mode = self._find_video_mode(
                self.rgb_stream,
                config.rgb_resolution[1],  # width
                config.rgb_resolution[0],  # height
                config.fps,
                openni2.PIXEL_FORMAT_RGB888,
            )
            self.rgb_stream.set_video_mode(rgb_mode)

            # Enable depth-to-color registration for coordinate alignment
            # This allows RGB coordinates to directly map to depth frame coordinates
            if self.device.is_image_registration_mode_supported(
                openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            ):
                self.device.set_image_registration_mode(
                    openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
                )
                logger.info(
                    "Enabled depth-to-color image registration - RGB coordinates "
                    "map directly to depth frame"
                )
            else:
                raise HardwareError(
                    "Device does not support depth-to-color image registration"
                )

            # Start both streams
            self.depth_stream.start()
            self.rgb_stream.start()

            self._initialized = True

        except Exception as e:
            # Clean up any partially initialized state
            self._cleanup()
            raise HardwareError(f"Failed to initialize hardware: {e}") from e

    def _find_video_mode(
        self, stream: VideoStream, width: int, height: int, fps: int, pixel_format: int
    ) -> VideoMode:
        """
        Find the appropriate video mode for a stream.

        Args:
            stream: The OpenNI2 stream object.
            width: Desired width in pixels.
            height: Desired height in pixels.
            fps: Desired frames per second.
            pixel_format: The pixel format for the video mode.
        Returns:
            VideoMode object matching the requested parameters.

        Raises:
            HardwareError: If no matching video mode is found.
        """
        sensor_info = stream.get_sensor_info()
        if sensor_info is None:
            raise HardwareError("Failed to get sensor info for stream")
        modes = cast(list[VideoMode], sensor_info.videoModes)
        for mode in modes:
            print(mode)
            if (
                mode.resolutionX == width
                and mode.resolutionY == height
                and mode.fps == fps
                and mode.pixelFormat == pixel_format
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
            assert self.depth_stream is not None  # For type checker
            frame = self.depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()

            # Reshape to (height, width)
            assert self.camera_config is not None  # For type checker
            depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape(
                self.camera_config.depth_resolution
            )

            # Mirror horizontally (OpenCL accelerated)
            depth_array = cv2.flip(cv2.UMat(depth_array), 1).get()

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
            assert self.rgb_stream is not None  # For type checker
            frame = self.rgb_stream.read_frame()
            frame_data = frame.get_buffer_as_uint8()

            # Reshape to (height, width, 3)
            assert self.camera_config is not None  # For type checker
            rgb_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (*self.camera_config.rgb_resolution, 3)
            )

            # Mirror horizontally + convert RGB to BGR (OpenCL accelerated)
            rgb_umat = cv2.UMat(rgb_array)
            rgb_umat = cv2.flip(rgb_umat, 1)
            rgb_umat = cv2.cvtColor(rgb_umat, cv2.COLOR_RGB2BGR)

            return rgb_umat.get()
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
                self.context = cast(openni2, self.context)
                self.context.unload()
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
