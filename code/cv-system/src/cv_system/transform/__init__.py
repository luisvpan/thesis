"""Coordinate transformer module.

This module provides the CoordinateTransformer class for bidirectional
camera <-> projector coordinate transformations, and the ImageTransformer class
for bidirectional camera <-> projector image transformations.
"""

# ResolutionMapper first: Calibrator imports it from this package while
# depth_coordinate_transformer pulls in calibration (see circular import).
from cv_system.transform.resolution_mapper import ResolutionMapper
from cv_system.transform.depth_coordinate_transformer import DepthCoordinateTransformer
from cv_system.transform.rgb_image_transformer import RgbImageTransformer

__all__ = ["DepthCoordinateTransformer", "RgbImageTransformer", "ResolutionMapper"]
