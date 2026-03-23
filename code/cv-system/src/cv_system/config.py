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
    """Hardware parameters for Kinect V2 depth and RGB streams."""

    # Depth camera resolution (height, width) as returned by OpenNI2
    depth_resolution: Tuple[int, int] = (424, 512)

    # RGB camera resolution (height, width) from Kinect V2
    rgb_resolution: Tuple[int, int] = (1080, 1920)

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

    @field_validator("depth_resolution", "rgb_resolution")
    @classmethod
    def resolution_must_be_positive(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure resolution dimensions are positive."""
        height, width = v
        if height <= 0 or width <= 0:
            raise ValueError("resolution dimensions must be positive")
        return v


class CalibrationConfig(BaseModel):
    """Parameters for the per-session calibration process."""

    # Number of depth frames to capture for dmax_map generation
    dmax_num_frames: int = 500

    # Expected depth range for the table surface (in millimeters)
    # Values outside this range are excluded from dmax calculation
    depth_range_min: int = 650
    depth_range_max: int = 800

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
    projector_corners: list[Tuple[float, float]] = [
        (0, 0),
        (0, 1080),
        (1920, 0),
        (1920, 1080),
    ]

    @field_validator("dmax_num_frames")
    @classmethod
    def num_frames_must_be_positive(cls, v: int) -> int:
        """Ensure number of frames is positive."""
        if v <= 0:
            raise ValueError("dmax_num_frames must be positive")
        return v

    @field_validator("depth_range_min", "depth_range_max")
    @classmethod
    def depth_must_be_positive(cls, v: int) -> int:
        """Ensure depth values are positive."""
        if v <= 0:
            raise ValueError("depth values must be positive")
        return v

    @field_validator("depth_range_max")
    @classmethod
    def depth_range_max_must_exceed_min(cls, v: int, info) -> int:
        """Ensure max depth exceeds min depth."""
        min_val = info.data.get("depth_range_min", 0)
        if v <= min_val:
            raise ValueError("depth_range_max must be greater than depth_range_min")
        return v

    @field_validator("camera_corners", "projector_corners")
    @classmethod
    def must_have_four_corners(cls, v: list) -> list:
        """Ensure exactly four corner points are provided."""
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
    min_touch_size: int = 10

    # Maximum touch area (in pixels) to filter out large objects (e.g., hand)
    max_touch_size: int = 5000

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
