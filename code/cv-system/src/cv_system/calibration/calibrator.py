"""Calibrator orchestrates the calibration process.

This module provides the Calibrator class that coordinates:
1. Reading calibration configuration (4 corner pairs)
2. Computing homography matrix from config points
3. Generating dmax_map from depth frames
4. Returning immutable CalibrationResult

The calibration process is per-session and requires a hardware manager
to provide depth frame capture capability.
"""

import time

import numpy as np

from cv_system.calibration.dmax import compute_depth_stats, generate_dmax_map
from cv_system.calibration.homography import compute_homography, validate_homography
from cv_system.calibration.result import CalibrationResult


class Calibrator:
    """Orchestrates the calibration process for per-session calibration.

    The calibrator reads calibration configuration, computes the homography
    matrix from 4 corner pairs, generates the dmax_map from N depth frames,
    and returns an immutable CalibrationResult.

    Attributes:
        config: CalibrationConfig containing corner pairs and parameters.
        hardware_manager: HardwareManager instance for frame capture.
    """

    def __init__(
        self,
        config: object,  # CalibrationConfig (circular import avoidance)
        hardware_manager: object,  # HardwareManager (circular import avoidance)
    ) -> None:
        """Initialize the calibrator with config and hardware manager.

        Args:
            config: CalibrationConfig instance with calibration parameters.
            hardware_manager: HardwareManager instance for depth frame capture.

        Raises:
            ValueError: If config or hardware_manager is invalid.
        """
        self.config = config
        self.hardware_manager = hardware_manager

        # Validate config has required attributes
        if not hasattr(config, "calibration"):
            raise ValueError("Config must have 'calibration' attribute")

        calibration = config.calibration
        if not hasattr(calibration, "camera_corners") or not hasattr(
            calibration, "projector_corners"
        ):
            raise ValueError(
                "Calibration config must have camera_corners and projector_corners"
            )

        if (
            len(calibration.camera_corners) != 4
            or len(calibration.projector_corners) != 4
        ):
            raise ValueError(
                "Exactly 4 corner pairs required: "
                f"got {len(calibration.camera_corners)} camera, "
                f"{len(calibration.projector_corners)} projector"
            )

    def run(self) -> CalibrationResult:
        """Run the full calibration process.

        Process:
        1. Compute homography matrix from config corner pairs
        2. Generate dmax_map by capturing N depth frames
        3. Validate results
        4. Return immutable CalibrationResult

        Returns:
            CalibrationResult with homography matrix, dmax_map, and metadata.

        Raises:
            RuntimeError: If calibration fails at any step.
            ValueError: If validation fails.
        """
        print("=" * 60)
        print("Starting calibration process")
        print("=" * 60)

        start_time = time.time()

        # Step 1: Compute homography matrix
        print("\nStep 1: Computing homography matrix...")
        H = self._compute_homography()
        print(f"  Homography matrix computed: {H.shape}, dtype={H.dtype}")

        # Step 2: Generate dmax_map
        print("\nStep 2: Generating dmax_map...")
        dmax_map = self._generate_dmax_map()
        stats = compute_depth_stats(
            dmax_map,
            depth_range=(
                self.config.calibration.depth_range_min,
                self.config.calibration.depth_range_max,
            ),
        )
        print(
            f"  dmax_map stats: mean={stats['mean']:.1f}, "
            f"std={stats['std']:.1f}, valid_ratio={stats['valid_pixel_ratio']:.2%}"
        )

        # Step 3: Validate results
        print("\nStep 3: Validating results...")
        self._validate_results(H, dmax_map)
        print("  Validation passed")

        # Step 4: Create calibration result
        elapsed = time.time() - start_time
        metadata = {
            "num_frames": self.config.calibration.dmax_num_frames,
            "depth_range": (
                self.config.calibration.depth_range_min,
                self.config.calibration.depth_range_max,
            ),
            "depth_shape": dmax_map.shape,
            "elapsed_seconds": elapsed,
            "stats": stats,
        }

        result = CalibrationResult(H=H, dmax_map=dmax_map, metadata=metadata)

        print("\n" + "=" * 60)
        print("Calibration complete")
        print(f"  H shape: {result.H.shape}")
        print(f"  dmax_map shape: {result.dmax_map.shape}")
        print(f"  Elapsed time: {elapsed:.2f}s")
        print("=" * 60)

        return result

    def _compute_homography(self) -> "np.ndarray":
        """Compute homography matrix from config corner pairs.

        Returns:
            3x3 homography matrix (float32).

        Raises:
            RuntimeError: If homography computation fails.
        """
        calibration = self.config.calibration

        try:
            H = compute_homography(
                camera_points=calibration.camera_corners,
                projector_points=calibration.projector_corners,
            )

            # Validate the homography
            if not validate_homography(H):
                raise ValueError("Computed homography matrix is invalid")

            return H

        except Exception as e:
            raise RuntimeError(f"Failed to compute homography: {e}") from e

    def _generate_dmax_map(self) -> "np.ndarray":
        """Generate dmax_map from depth frames using hardware manager.

        Returns:
            2D dmax_map array.

        Raises:
            RuntimeError: If frame capture fails.
        """
        calibration = self.config.calibration

        # Create capture function that delegates to hardware manager
        def capture_frame() -> "np.ndarray":
            try:
                depth_frame = self.hardware_manager.get_depth_frame()
                return depth_frame
            except Exception as e:
                raise RuntimeError(f"Failed to capture depth frame: {e}") from e

        try:
            dmax_map = generate_dmax_map(
                capture_frame=capture_frame,
                num_frames=calibration.dmax_num_frames,
                depth_range=(
                    calibration.depth_range_min,
                    calibration.depth_range_max,
                ),
                depth_shape=(424, 512),  # Kinect V2 depth frame shape
            )
            return dmax_map

        except Exception as e:
            raise RuntimeError(f"Failed to generate dmax_map: {e}") from e

    def _validate_results(self, H: "np.ndarray", dmax_map: "np.ndarray") -> None:
        """Validate calibration results.

        Args:
            H: Homography matrix.
            dmax_map: dmax_map array.

        Raises:
            ValueError: If validation fails.
        """
        # Validate homography
        if not validate_homography(H):
            raise ValueError("Homography matrix validation failed")

        # Validate dmax_map shape
        if dmax_map.shape != (424, 512):
            raise ValueError(
                f"dmax_map has invalid shape {dmax_map.shape}, expected (424, 512)"
            )

        # Check that dmax_map has some valid data
        calibration = self.config.calibration
        valid_mask = (dmax_map >= calibration.depth_range_min) & (
            dmax_map <= calibration.depth_range_max
        )
        valid_ratio = np.sum(valid_mask) / dmax_map.size

        if valid_ratio < 0.5:  # At least 50% of pixels should be valid
            raise ValueError(
                f"dmax_map has too few valid pixels: {valid_ratio:.1%} < 50%"
            )
