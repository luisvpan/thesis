"""Calibration result dataclass.

This module defines immutable CalibrationResult dataclass that contains
the homography matrix, dmax_map, camera_corners, and metadata computed
during calibration.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable calibration result containing transformation data.

    Attributes:
        depth_H: 3x3 homography matrix mapping depth coordinates to projector coordinates.
        rgb_H: 3x3 homography matrix mapping RGB coordinates to projector coordinates.
        dmax_map: 2D array where each pixel contains the most frequent depth value
            across N calibration frames (direct mode, no depth range filtering).
        depth_corners: List of 4 (x, y) tuples representing the detected
            depth corner coordinates in depth space, sorted as
            [top-left, top-right, bottom-left, bottom-right].
        rgb_corners: List of 4 (x, y) tuples representing the detected
            RGB corner coordinates in camera space, sorted as
            [top-left, top-right, bottom-left, bottom-right].
        metadata: Dictionary containing calibration metadata such as number of frames
            captured, generation method, timestamp, etc.

    The frozen=True decorator ensures this result is immutable after creation,
    preventing accidental modification of calibration data.
    """

    depth_H: np.ndarray
    rgb_H: np.ndarray
    dmax_map: np.ndarray
    depth_corners: list[tuple[int, int]] = field(default_factory=list)
    rgb_corners: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that arrays have expected shapes and types."""
        # Validate depth_H is 3x3
        if self.depth_H.shape != (3, 3):
            raise ValueError(f"Depth homography matrix must be 3x3, got {self.depth_H.shape}")

        # Validate rgb_H is 3x3
        if self.rgb_H.shape != (3, 3):
            raise ValueError(f"RGB homography matrix must be 3x3, got {self.rgb_H.shape}")

        # Validate depth_H is float32
        if self.depth_H.dtype != np.float32:
            raise ValueError(f"Depth homography matrix must be float32, got {self.depth_H.dtype}")

        # Validate rgb_H is float32
        if self.rgb_H.dtype != np.float32:
            raise ValueError(f"RGB homography matrix must be float32, got {self.rgb_H.dtype}")

        # Validate dmax_map is 2D
        if self.dmax_map.ndim != 2:
            raise ValueError(
                f"dmax_map must be 2D, got {self.dmax_map.ndim} dimensions"
            )

        # Validate depth_corners has at least 4 points if provided
        if self.depth_corners and len(self.depth_corners) < 4:
            raise ValueError(
                f"depth_corners must have at least 4 points, got {len(self.depth_corners)}"
            )

        # Validate rgb_corners has at least 4 points if provided
        if self.rgb_corners and len(self.rgb_corners) < 4:
            raise ValueError(
                f"rgb_corners must have at least 4 points, got {len(self.rgb_corners)}"
            )

    def __repr__(self) -> str:
        """Return a concise representation of the calibration result."""
        depth_info = (
            f", depth_corners={self.depth_corners}" if self.depth_corners else ""
        )
        rgb_info = (
            f", rgb_corners={self.rgb_corners}" if self.rgb_corners else ""
        )
        return (
            f"CalibrationResult(depth_H_shape={self.depth_H.shape}, "
            f"rgb_H_shape={self.rgb_H.shape}, "
            f"dmax_map_shape={self.dmax_map.shape}{depth_info}{rgb_info}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )

    def save(self, path: Path | str) -> None:
        """Save calibration result to JSON file for persistence and debugging.

        The JSON format is human-readable for manual inspection and debugging.
        Numpy arrays are serialized to nested lists using .tolist().

        Args:
            path: Path to output JSON file. Will add .json extension if not present.

        Raises:
            OSError: If file cannot be written (permission errors, disk full).
            ValueError: If calibration data is invalid (wrong shapes, missing fields).
        """
        # Convert to Path object and ensure .json extension
        output_path = Path(path)
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving calibration result to {output_path}")

        # Validate data before serialization (defensive check)
        if self.depth_H.shape != (3, 3):
            raise ValueError(f"Depth H has invalid shape {self.depth_H.shape}, expected (3, 3)")
        if self.dmax_map.ndim != 2:
            raise ValueError(
                f"dmax_map has invalid dimensions {self.dmax_map.ndim}, expected 2D"
            )

        # Serialize depth_H matrix (3x3 float32) to nested list
        # Using .tolist() for human-readable JSON (can view in text editor)
        depth_H_list = self.depth_H.tolist()

        # Serialize rgb_H matrix (3x3 float32) to nested list
        rgb_H_list = self.rgb_H.tolist()

        # Serialize dmax_map (2D uint16) to list-of-lists
        # Tradeoff: Larger file size than base64, but human-readable for debugging
        dmax_map_list = self.dmax_map.tolist()

        # Serialize depth_corners as list of lists
        depth_corners_list = [list(corner) for corner in self.depth_corners]

        # Serialize rgb_corners as list of lists
        rgb_corners_list = [list(corner) for corner in self.rgb_corners]

        # Build merged metadata with timestamp and version
        # Preserve existing metadata (num_frames, depth_shape, computed_at_ms, etc.)
        # Add persistence metadata
        metadata = dict(self.metadata)  # Copy existing metadata
        metadata.update(
            {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            }
        )

        # Construct JSON structure
        data = {
            "version": metadata.get("version", "1.0"),
            "timestamp": metadata.get("timestamp"),
            "metadata": metadata,
            "depth_H": depth_H_list,
            "rgb_H": rgb_H_list,
            "dmax_map": dmax_map_list,
            "depth_corners": depth_corners_list,
            "rgb_corners": rgb_corners_list,
        }

        # Write JSON file with indentation for human readability
        try:
            output_path.write_text(json.dumps(data, indent=2))
            print(f"  Depth H shape: {self.depth_H.shape}")
            print(f"  RGB H shape: {self.rgb_H.shape}")
            print(f"  dmax_map shape: {self.dmax_map.shape}")
            print(f"  Depth corners: {self.depth_corners}")
            print(f"  RGB corners: {self.rgb_corners}")
            print(f"  Metadata: {list(self.metadata.keys())}")
            print("Calibration result saved successfully")
        except OSError as e:
            raise OSError(
                f"Failed to write calibration file to {output_path}: {e}"
            ) from e

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationResult":
        """Load calibration result from JSON file.

        Deserializes JSON saved by save() method back into a frozen
        CalibrationResult instance with validation.

        Args:
            path: Path to input JSON file.

        Returns:
            CalibrationResult: Frozen instance with loaded calibration data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON is invalid or has wrong shapes, missing fields.
        """
        # Convert to Path object and ensure .json extension
        input_path = Path(path)
        if input_path.suffix != ".json":
            input_path = input_path.with_suffix(".json")

        print(f"Loading calibration result from {input_path}")

        # Check file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Calibration file not found at {input_path}")

        # Read JSON file
        try:
            content = input_path.read_text(encoding="utf-8")
        except OSError as e:
            raise FileNotFoundError(f"Failed to read calibration file: {e}") from e

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in calibration file: {e}") from e

        # Validate required fields exist
        required_fields = ["version", "timestamp", "metadata", "depth_H", "rgb_H", "dmax_map", "depth_corners", "rgb_corners"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in calibration file: {missing_fields}"
            )

        # Deserialize depth_H matrix (3x3 float32)
        try:
            depth_H_list = data["depth_H"]
            depth_H = np.array(depth_H_list, dtype=np.float32)

            if depth_H.shape != (3, 3):
                raise ValueError(f"Depth H has invalid shape {depth_H.shape}, expected (3, 3)")
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to deserialize depth_H matrix: {e}") from e

        # Deserialize rgb_H matrix (3x3 float32)
        try:
            rgb_H_list = data["rgb_H"]
            rgb_H = np.array(rgb_H_list, dtype=np.float32)

            if rgb_H.shape != (3, 3):
                raise ValueError(f"RGB H has invalid shape {rgb_H.shape}, expected (3, 3)")
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to deserialize rgb_H matrix: {e}") from e

        # Deserialize dmax_map (2D uint16)
        try:
            dmax_map_data = data["dmax_map"]
            dmax_map = np.array(dmax_map_data, dtype=np.uint16)

            if dmax_map.ndim != 2:
                raise ValueError(
                    f"dmax_map has invalid dimensions {dmax_map.ndim}, expected 2D"
                )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to deserialize dmax_map: {e}") from e

        # Deserialize depth_corners (list of tuples)
        try:
            corners_list = data["depth_corners"]
            depth_corners = [(x, y) for x, y in corners_list]

            if len(depth_corners) < 4:
                raise ValueError(
                    f"depth_corners must have at least 4 points, got {len(depth_corners)}"
                )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to deserialize depth_corners: {e}") from e

        # Deserialize rgb_corners (list of tuples)
        try:
            corners_list = data["rgb_corners"]
            rgb_corners = [(x, y) for x, y in corners_list]

            if len(rgb_corners) < 4:
                raise ValueError(
                    f"rgb_corners must have at least 4 points, got {len(rgb_corners)}"
                )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to deserialize rgb_corners: {e}") from e

        # Deserialize metadata (preserve all fields)
        try:
            metadata = dict(data["metadata"])
        except KeyError as e:
            raise ValueError(f"Missing metadata field: {e}") from e

        # Create frozen CalibrationResult instance
        # __post_init__ will validate shapes and types
        result = cls(
            depth_H=depth_H, rgb_H=rgb_H, dmax_map=dmax_map, depth_corners=depth_corners, rgb_corners=rgb_corners, metadata=metadata
        )

        # Log successful load
        print(f"  Depth H shape: {result.depth_H.shape}")
        print(f"  RGB H shape: {result.rgb_H.shape}")
        print(f"  dmax_map shape: {result.dmax_map.shape}")
        print(f"  Depth corners: {result.depth_corners}")
        print(f"  RGB corners: {result.rgb_corners}")
        print(f"  Metadata keys: {list(result.metadata.keys())}")
        print("Calibration result loaded successfully")

        return result
