"""
Detection layer for touch and interaction identification.

This module provides touch detection functionality that compares depth frames
against a calibrated dmax_map using temporal filtering via a ring buffer.
"""

from cv_system.detection.touch_detector import TouchDetector

__all__ = ["TouchDetector"]
