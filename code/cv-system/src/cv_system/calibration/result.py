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
        H: 3x3 homography matrix mapping camera coordinates to projector coordinates.
        dmax_map: 2D array where each pixel contains the most frequent depth value
            across N calibration frames (direct mode, no depth range filtering).
        camera_corners: List of 4 (x, y) tuples representing the detected
            camera corner coordinates in depth space, sorted as
            [top-left, top-right, bottom-left, bottom-right].
        metadata: Dictionary containing calibration metadata such as number of frames
            captured, generation method, timestamp, etc.

    The frozen=True decorator ensures this result is immutable after creation,
    preventing accidental modification of calibration data.
    """

    H: np.ndarray
    dmax_map: np.ndarray
    camera_corners: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that arrays have expected shapes and types."""
        # Validate H is 3x3
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")

        # Validate H is float32
        if self.H.dtype != np.float32:
            raise ValueError(f"Homography matrix must be float32, got {self.H.dtype}")

        # Validate dmax_map is 2D
        if self.dmax_map.ndim != 2:
            raise ValueError(
                f"dmax_map must be 2D, got {self.dmax_map.ndim} dimensions"
            )

        # Validate camera_corners has exactly 4 points if provided
        if self.camera_corners and len(self.camera_corners) != 4:
            raise ValueError(
                f"camera_corners must have exactly 4 points, got {len(self.camera_corners)}"
            )

    def __repr__(self) -> str:
        """Return a concise representation of the calibration result."""
        camera_info = (
            f", camera_corners={self.camera_corners}" if self.camera_corners else ""
        )
        return (
            f"CalibrationResult(H_shape={self.H.shape}, "
            f"dmax_map_shape={self.dmax_map.shape}{camera_info}, "
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
        if self.H.shape != (3, 3):
            raise ValueError(f"H has invalid shape {self.H.shape}, expected (3, 3)")
        if self.dmax_map.ndim != 2:
            raise ValueError(
                f"dmax_map has invalid dimensions {self.dmax_map.ndim}, expected 2D"
            )

        # Serialize H matrix (3x3 float32) to nested list
        # Using .tolist() for human-readable JSON (can view in text editor)
        H_list = self.H.tolist()

        # Serialize dmax_map (2D uint16) to list-of-lists
        # Tradeoff: Larger file size than base64, but human-readable for debugging
        dmax_map_list = self.dmax_map.tolist()

        # Serialize camera_corners as list of lists
        camera_corners_list = [list(corner) for corner in self.camera_corners]

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
                "H": H_list,
                "dmax_map": dmax_map_list,
                "camera_corners": camera_corners_list,
            }

        # Write JSON file with indentation for human readability
        try:
            output_path.write_text(json.dumps(data, indent=2))
            print(f"  H shape: {self.H.shape}")
            print(f"  dmax_map shape: {self.dmax_map.shape}")
            print(f"  Camera corners: {self.camera_corners}")
            print(f"  Metadata: {list(self.metadata.keys())}")
            print("Calibration result saved successfully")
        except OSError as e:
            raise OSError(f"Failed to write calibration file to {output_path}: {e}") from e

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
            raise FileNotFoundError(
                f"Calibration file not found at {input_path}"
            )

        # Read JSON file
        try:
            content = input_path.read_text(encoding="utf-8")
        except OSError as e:
            raise FileNotFoundError(f"Failed to read calibration file: {e}") from e

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in calibration file: {e}"
            ) from e

        # Validate required fields exist
        required_fields = ["version", "timestamp", "metadata", "H", "dmax_map"]
        missing_fields = [
            field for field in required_fields if field not in data
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in calibration file: {missing_fields}"
            )

        # Deserialize H matrix (3x3 float32)
        try:
            H_list = data["H"]
            H = np.array(H_list, dtype=np.float32)

            if H.shape != (3, 3):
                raise ValueError(
                    f"H has invalid shape {H.shape}, expected (3, 3)"
                )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to deserialize H matrix: {e}") from e

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

        # Deserialize camera_corners (list of tuples)
        try:
            corners_list = data["camera_corners"]
            camera_corners = [(x, y) for x, y in corners_list]

            if len(camera_corners) != 4:
                raise ValueError(
                    f"camera_corners must have exactly 4 points, got {len(camera_corners)}"
                )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to deserialize camera_corners: {e}") from e

        # Deserialize metadata (preserve all fields)
        try:
            metadata = dict(data["metadata"])
        except KeyError as e:
            raise ValueError(f"Missing metadata field: {e}") from e

        # Create frozen CalibrationResult instance
        # __post_init__ will validate shapes and types
        result = cls(H=H, dmax_map=dmax_map, camera_corners=camera_corners, metadata=metadata)

        # Log successful load
        print(f"  H shape: {result.H.shape}")
        print(f"  dmax_map shape: {result.dmax_map.shape}")
        print(f"  Camera corners: {result.camera_corners}")
        print(f"  Metadata keys: {list(result.metadata.keys())}")
        print("Calibration result loaded successfully")

        return result
