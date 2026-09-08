"""Calibration module for homography computation and dmax_map generation.

This module provides the calibration layer of the CV system:
- CalibrationResult: Immutable dataclass for calibration output
- Calibrator: Orchestrates the calibration process
- compute_homography: Compute 4-point homography matrix
- generate_dmax_map: Generate dmax_map from N depth frames
"""

from cv_system.calibration.calibrator import Calibrator
from cv_system.calibration.dmax import compute_depth_stats, generate_dmax_map, generate_dmax_map_wilson
from cv_system.calibration.homography import (
    apply_homography,
    compute_homography,
    validate_homography,
)
from cv_system.calibration.marker_detector import MarkerDetector
from cv_system.calibration.marker_projector import MarkerProjector
from cv_system.calibration.result import CalibrationResult

__all__ = [
    "CalibrationResult",
    "Calibrator",
    "compute_homography",
    "apply_homography",
    "validate_homography",
    "generate_dmax_map",
    "generate_dmax_map_wilson",
    "compute_depth_stats",
    "MarkerProjector",
    "MarkerDetector",
]
