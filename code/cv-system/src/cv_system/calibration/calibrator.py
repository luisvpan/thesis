"""Calibrator orchestrates the calibration process.

This module provides the Calibrator class that coordinates:
1. Projecting markers using MarkerProjector
2. Capturing RGB frame and detecting markers with MarkerDetector
3. Mapping detected RGB centroids to depth coordinates
4. Computing homography matrix from detected camera_corners
5. Generating dmax_map from depth frames
6. Returning immutable CalibrationResult

The calibration process is per-session and requires a hardware manager
to provide RGB and depth frame capture capability.
"""

import logging
import time

import cv2
import numpy as np

from cv_system.calibration.dmax import compute_depth_stats, generate_dmax_map
from cv_system.calibration.homography import compute_homography, validate_homography
from cv_system.calibration.marker_detector import MarkerDetector
from cv_system.calibration.marker_projector import MarkerProjector
from cv_system.calibration.result import CalibrationResult
from cv_system.config import SessionConfig
from cv_system.hardware.manager import HardwareManager, HardwareError

logger = logging.getLogger(__name__)


class Calibrator:
    """Orchestrates the calibration process for per-session calibration.

    The calibrator performs automatic marker-based calibration:
    1. Projects 4 calibration markers using MarkerProjector
    2. Captures RGB frame from camera
    3. Detects markers in RGB frame using MarkerDetector
    4. Maps detected RGB centroids to depth coordinates
    5. Computes homography matrix from detected camera_corners
    6. Generates dmax_map from N depth frames using direct mode estimation
    7. Returns immutable CalibrationResult with generation metadata

    Attributes:
        config: SessionConfig containing calibration parameters.
        hardware_manager: HardwareManager instance for frame capture.
        marker_detector: MarkerDetector instance for marker detection.
        marker_projector: MarkerProjector instance for marker projection.
    """

    def __init__(
        self,
        config: SessionConfig,
        hardware_manager: HardwareManager,
    ) -> None:
        """Initialize the calibrator with config and hardware manager.

        Args:
            config: SessionConfig instance with calibration parameters.
            hardware_manager: HardwareManager instance for RGB/depth frame capture.

        Raises:
            ValueError: If config or hardware_manager is invalid.
        """
        self.config = config
        self.hardware_manager = hardware_manager

        # Validate config has required attributes
        if not hasattr(config, "calibration"):
            raise ValueError("Config must have 'calibration' attribute")

        calibration = config.calibration
        if not hasattr(calibration, "projector_corners"):
            raise ValueError("Calibration config must have projector_corners")

        if len(calibration.projector_corners) != 4:
            raise ValueError(
                f"Exactly 4 projector corners required: "
                f"got {len(calibration.projector_corners)} corners"
            )

        # Initialize marker detector with default parameters
        self.marker_detector = MarkerDetector()

        # Initialize marker projector with default resolution
        self.marker_projector = MarkerProjector()

        logger.info(
            f"Calibrator initialized with marker detection: "
            f"projector_corners={calibration.projector_corners}"
        )

    def run(self) -> CalibrationResult:
        """Run the full automatic calibration process.

        Process:
        1. Project 4 calibration markers and detect them in RGB frame
        2. Map detected RGB centroids to depth coordinates
        3. Compute homography matrix from detected camera_corners
        4. Generate dmax_map using direct mode estimation (no histogram, no depth range filtering)
        5. Validate results
        6. Return immutable CalibrationResult with generation metadata

        Returns:
            CalibrationResult with homography matrix, dmax_map, camera_corners,
            and metadata (includes "method": "direct").

        Raises:
            RuntimeError: If calibration fails at any step.
            ValueError: If validation fails.
        """
        print("=" * 60)
        print("Starting automatic calibration process")
        print("=" * 60)

        start_time = time.time()

        # Step 1: Compute homography matrix via marker detection
        print("\nStep 1: Projecting markers and detecting in RGB frame...")
        H, camera_corners = self._compute_homography()
        print(f"  Homography matrix computed: {H.shape}, dtype={H.dtype}")
        print(f"  Detected camera_corners: {camera_corners}")

        # Step 2: Generate dmax_map
        print("\nStep 2: Generating dmax_map...")
        dmax_start_time = time.time()
        dmax_map = self._generate_dmax_map()
        dmax_elapsed_ms = (time.time() - dmax_start_time) * 1000

        # Validate results
        print("\nStep 3: Validating results...")
        self._validate_results(H, dmax_map, camera_corners)

        # Step 4: Create calibration result
        elapsed = time.time() - start_time
        metadata = {
            "method": "direct",  # Direct mode estimation (no histogram)
            "num_frames": self.config.calibration.dmax_num_frames,
            "depth_shape": dmax_map.shape,
            "elapsed_seconds": elapsed,
            "dmax_compute_time_ms": dmax_elapsed_ms,
        }

        result = CalibrationResult(
            H=H, dmax_map=dmax_map, camera_corners=camera_corners, metadata=metadata
        )

        print("\n" + "=" * 60)
        print("Calibration complete")
        print(f"  H shape: {result.H.shape}")
        print(f"  dmax_map shape: {result.dmax_map.shape}")
        print(f"  Elapsed time: {elapsed:.2f}s")
        print("=" * 60)

        return result

    def _compute_homography(self) -> tuple["np.ndarray", list[tuple[int, int]]]:
        """Compute homography matrix via automatic marker detection.

        This method:
        1. Projects 4 calibration markers at configured projector_corners
        2. Captures RGB frame from hardware manager
        3. Detects markers in RGB frame using MarkerDetector
        4. Maps detected RGB centroids to depth coordinates
        5. Computes homography matrix from detected camera_corners

        Returns:
            Tuple of (3x3 homography matrix (float32), camera_corners as list of 4
            (x, y) tuples in depth coordinates).

        Raises:
            RuntimeError: If homography computation fails.
            ValueError: If marker detection fails or RGB frame is invalid.
        """
        calibration = self.config.calibration
        projector_corners = calibration.projector_corners

        try:
            # Step 1: Project markers
            logger.info(f"Projecting markers at {projector_corners}")
            print(f"  Projecting {len(projector_corners)} markers...")
            self.marker_projector.project_markers(projector_corners)
            logger.info("Marker projection complete")
            cv2.waitKey()
            # Step 2: Capture RGB frame
            logger.info("Capturing RGB frame for marker detection")
            print("  Capturing RGB frame...")
            rgb_frame = self.hardware_manager.get_rgb_frame()
            logger.info(
                f"RGB frame captured: shape={rgb_frame.shape}, dtype={rgb_frame.dtype}"
            )

            # Show rgb frame captured
            cv2.namedWindow("Captured RGB Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Captured RGB Frame", rgb_frame)
            cv2.waitKey()

            # Step 3: Detect markers in RGB frame
            logger.info("Detecting markers in RGB frame")
            print("  Detecting markers...")
            rgb_corners = self.marker_detector.detect_markers(rgb_frame)
            logger.info(
                f"Detected {len(rgb_corners)} markers in RGB coordinates: {rgb_corners}"
            )
            print(f"  Detected {len(rgb_corners)} markers at {rgb_corners}")

            # Step 4: Map RGB centroids to depth coordinates
            logger.info("Mapping RGB corners to depth coordinates")
            print("  Mapping RGB coordinates to depth...")
            camera_corners = self.hardware_manager.map_rgb_to_depth(rgb_corners)
            logger.info(f"Mapped to depth coordinates: {camera_corners}")
            print(f"  Mapped to depth coordinates: {camera_corners}")

            # Validate mapped camera_corners are within depth frame bounds
            depth_width, depth_height = (
                self.config.camera.depth_resolution[1],
                self.config.camera.depth_resolution[0],
            )
            for i, (x, y) in enumerate(camera_corners):
                if not (0 <= x < depth_width):
                    raise ValueError(
                        f"camera_corners[{i}] x-coordinate out of bounds: "
                        f"{x} not in [0, {depth_width})"
                    )
                if not (0 <= y < depth_height):
                    raise ValueError(
                        f"camera_corners[{i}] y-coordinate out of bounds: "
                        f"{y} not in [0, {depth_height})"
                    )

            logger.info("Camera_corners within depth frame bounds")

            # Step 5: Compute homography matrix
            H = compute_homography(
                camera_points=camera_corners,
                projector_points=projector_corners,
            )

            # Validate the homography
            if not validate_homography(H):
                raise ValueError("Computed homography matrix is invalid")

            # Clean up marker projection window
            self.marker_projector.destroy_window()
            logger.info("Marker projection window destroyed")

            return H, camera_corners

        except Exception as e:
            # Clean up marker projection window on error
            try:
                self.marker_projector.destroy_window()
            except Exception:
                pass
            raise RuntimeError(f"Failed to compute homography: {e}") from e

    def _generate_dmax_map(self) -> "np.ndarray":
        """Generate dmax_map from depth frames using direct mode estimation.

        Captures N depth frames and computes the per-pixel most frequent
        depth value (mode) along the time axis. No depth range filtering.

        Returns:
            2D dmax_map array (uint16).

        Raises:
            RuntimeError: If frame capture fails.
        """
        calibration = self.config.calibration

        # Create capture function that delegates to hardware manager
        def capture_frame() -> "np.ndarray":
            try:
                depth_frame = self.hardware_manager.get_depth_frame()

                return depth_frame
            except HardwareError as e:
                raise RuntimeError(f"Failed to capture depth frame: {e}") from e

        try:
            dmax_map = generate_dmax_map(
                capture_frame=capture_frame,
                num_frames=calibration.dmax_num_frames,
                depth_shape=(424, 512),  # Kinect V2 depth frame shape
            )
            return dmax_map

        except Exception as e:
            raise RuntimeError(f"Failed to generate dmax_map: {e}") from e

    def _validate_results(
        self,
        H: "np.ndarray",
        dmax_map: "np.ndarray",
        camera_corners: list[tuple[int, int]],
    ) -> None:
        """Validate calibration results.

        Args:
            H: Homography matrix.
            dmax_map: dmax_map array.
            camera_corners: Detected camera corner coordinates in depth space.

        Raises:
            ValueError: If validation fails.
        """
        # Validate homography
        if not validate_homography(H):
            raise ValueError("Homography matrix validation failed")

        # Validate camera_corners are within depth frame bounds
        depth_width, depth_height = (
            self.config.camera.depth_resolution[1],
            self.config.camera.depth_resolution[0],
        )
        for i, (x, y) in enumerate(camera_corners):
            if not (0 <= x < depth_width):
                raise ValueError(
                    f"camera_corners[{i}] x-coordinate out of bounds: "
                    f"{x} not in [0, {depth_width})"
                )
            if not (0 <= y < depth_height):
                raise ValueError(
                    f"camera_corners[{i}] y-coordinate out of bounds: "
                    f"{y} not in [0, {depth_height})"
                )

        logger.info("camera_corners validated within depth frame bounds")

        # Validate dmax_map shape
        if dmax_map.shape != (424, 512):
            raise ValueError(
                f"dmax_map has invalid shape {dmax_map.shape}, expected (424, 512)"
            )

        # Check that dmax_map has some valid data (non-zero pixels)
        # Direct mode dmax: valid pixels are non-zero (no depth range filtering)
        valid_mask = dmax_map > 0  # Valid pixels are non-zero
        valid_ratio = np.sum(valid_mask) / dmax_map.size

        if valid_ratio < 0.5:  # At least 50% of pixels should be valid
            raise ValueError(
                f"dmax_map has too few valid pixels: {valid_ratio:.1%} < 50%"
            )

        logger.info(f"dmax_map validation passed: {valid_ratio:.1%} valid pixels")
