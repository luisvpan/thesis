"""Calibration module for homography computation and dmax_map generation.

This module provides the calibration layer of the CV system:
- CalibrationResult: Immutable dataclass for calibration output
- Calibrator: Orchestrates the calibration process
"""

from cv_system.calibration.result import CalibrationResult

__all__ = ["CalibrationResult"]
