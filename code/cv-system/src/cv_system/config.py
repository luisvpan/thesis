"""
Configuration models for the CV system.

All magic numbers from the inherited monolithic code are externalized here
as configuration parameters with type-safe validation via Pydantic.
"""

import json
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""

    pass


class CameraConfig(BaseModel):
    """Hardware parameters for depth/RGB streams and the projector output canvas."""

    # Depth camera resolution (height, width) as returned by OpenNI2
    depth_resolution: Tuple[int, int] = (424, 512)

    # RGB camera resolution (height, width) from the sensor
    rgb_resolution: Tuple[int, int] = (1080, 1920)

    # Projector / "bird view" output size (height, width) for warpPerspective.
    # Must match the pixel space of calibration.projector_corners and the
    # physical display (e.g. 1920×1080), not the Kinect RGB size.
    projector_resolution: Tuple[int, int] = (1080, 1920)

    # Frames per second for both streams
    fps: int = 30

    @field_validator("fps")
    @classmethod
    def fps_must_be_positive(cls, v: int) -> int:
        """Ensure FPS is positive and within reasonable bounds."""
        if v <= 0:
            raise ValueError("fps must be positive")
        if v > 60:
            raise ValueError("fps must not exceed 60")
        return v

    @field_validator("depth_resolution", "rgb_resolution", "projector_resolution")
    @classmethod
    def resolution_must_be_positive(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure resolution dimensions are positive."""
        height, width = v
        if height <= 0 or width <= 0:
            raise ValueError("resolution dimensions must be positive")
        return v


class CalibrationConfig(BaseModel):
    """Parameters for the per-session calibration process."""

    # Number of depth frames to capture for dsurface_map generation (Wilson algorithm)
    dsurface_num_frames: int = 500

    # Minimum count threshold for a histogram bin to be considered valid
    # Wilson algorithm: scan from near to far, find first bin with count >= threshold
    histogram_threshold: int = 3

    # Range in mm around the snapshot reference for per-pixel histograms
    # Each pixel has bins covering [snapshot - range, snapshot + range]
    histogram_range: int = 20

    # Offset to subtract from dsurface to get dmax (in mm)
    # dmax = dsurface - surface_offset ensures the table surface is excluded from touch zone
    # Higher values = touch zone farther from surface (more conservative)
    surface_offset: int = 2

    # Four corner points in camera coordinates (y, x)
    # Format: [(y1, x1), (y2, x2), (y3, x3), (y4, x4)]
    camera_corners: list[Tuple[float, float]] = [
        (0, 0),
        (0, 424),
        (512, 0),
        (512, 424),
    ]

    # Four corner points in projector coordinates (y, x)
    # Format: [(y1, x1), (y2, x2), (y3, x3), (y4, x4)]
    projector_corners: list[Tuple[int, int]] = [
        (0, 0),
        (0, 1080),
        (1920, 0),
        (1920, 1080),
    ]

    @field_validator("dsurface_num_frames")
    @classmethod
    def num_frames_must_be_positive(cls, v: int) -> int:
        """Ensure number of frames is positive."""
        if v <= 0:
            raise ValueError("dsurface_num_frames must be positive")
        return v

    @field_validator("histogram_threshold")
    @classmethod
    def histogram_threshold_must_be_positive(cls, v: int) -> int:
        """Ensure histogram threshold is positive."""
        if v <= 0:
            raise ValueError("histogram_threshold must be positive")
        return v

    @field_validator("histogram_range")
    @classmethod
    def histogram_range_must_be_positive(cls, v: int) -> int:
        """Ensure histogram range is positive."""
        if v <= 0:
            raise ValueError("histogram_range must be positive")
        return v

        if len(v) != 4:
            raise ValueError("must have exactly 4 corner points")
        if len(v[0]) != 2:
            raise ValueError("corner points must be 2-tuples (y, x)")
        return v


class DetectionConfig(BaseModel):
    """Parameters for touch and interaction detection."""

    # Number of frames in the ring buffer for noise filtering
    ring_buffer_size: int = 5

    # Depth difference threshold (in mm) to detect foreground objects
    touch_threshold: int = 20

    # Minimum touch area (in pixels) to filter out noise
    # Increased from 10 to 50 for ToF sensor noise (Kinect V2)
    min_touch_size: int = 50

    # Maximum touch area (in pixels) to filter out large objects (e.g., hand)
    max_touch_size: int = 5000

    # Vibration threshold (in mm) for filtering frame-to-frame noise
    vibration_threshold: int = 15

    # Number of frames required to confirm a touch (touch history)
    touch_history_size: int = 3

    @field_validator("ring_buffer_size")
    @classmethod
    def ring_buffer_size_must_be_positive(cls, v: int) -> int:
        """Ensure ring buffer size is positive."""
        if v <= 0:
            raise ValueError("ring_buffer_size must be positive")
        return v

    @field_validator("touch_threshold")
    @classmethod
    def touch_threshold_must_be_positive(cls, v: int) -> int:
        """Ensure touch threshold is positive."""
        if v <= 0:
            raise ValueError("touch_threshold must be positive")
        return v

    @field_validator("min_touch_size", "max_touch_size")
    @classmethod
    def touch_size_must_be_positive(cls, v: int) -> int:
        """Ensure touch size thresholds are positive."""
        if v <= 0:
            raise ValueError("touch size thresholds must be positive")
        return v

    @field_validator("max_touch_size")
    @classmethod
    def max_must_exceed_min(cls, v: int, info) -> int:
        """Ensure max touch size exceeds min touch size."""
        min_val = info.data.get("min_touch_size", 0)
        if v <= min_val:
            raise ValueError("max_touch_size must be greater than min_touch_size")
        return v


class SessionConfig(BaseModel):
    """Root configuration model for a CV system session."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)


def load_config(path: Path) -> SessionConfig:
    """
    Load and validate a configuration file.

    Args:
        path: Path to the configuration JSON file. Can be relative or absolute.

    Returns:
        SessionConfig: Validated configuration instance.

    Raises:
        ConfigError: If the file doesn't exist, has invalid JSON syntax,
                    or contains values that fail validation.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {path}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read configuration file {path}: {e}") from e

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"Invalid JSON in configuration file {path}: {e.msg} at line {e.lineno}"
        ) from e

    try:
        config = SessionConfig(**data)
    except ValidationError as e:
        errors = e.errors()
        error_summary = "; ".join(
            f"[{err['loc']}] {err['type']}: {err['msg']}" for err in errors[:3]
        )
        if len(errors) > 3:
            error_summary += f" ... and {len(errors) - 3} more errors"
        raise ConfigError(
            f"Configuration validation failed in {path}: {error_summary}"
        ) from e

    return config
