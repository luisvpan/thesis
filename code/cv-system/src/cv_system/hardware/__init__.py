"""Hardware Manager for Kinect V2 integration.

This is the only module that imports OpenNI2 (per ADR-004).
All other modules access hardware through this interface.
"""

from cv_system.hardware.manager import HardwareManager

__all__ = ["HardwareManager"]
