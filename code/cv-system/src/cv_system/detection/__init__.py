"""
Detection layer for touch and interaction identification.

This module provides touch detection functionality that compares depth frames
against a calibrated dmax_map using temporal filtering via a ring buffer.
"""

from cv_system.detection.card_detector import CardDetector, CardDetection
from cv_system.detection.depth_only_touch_detector import DepthOnlyTouchDetector
from cv_system.detection.touch_detector import TouchDetector

__all__ = ["CardDetector", "CardDetection", "DepthOnlyTouchDetector", "TouchDetector"]
