"""Hardware Manager for Kinect V2 integration.

Provides two implementations:
- HardwareManager: Uses OpenNI2 (cross-platform)
- PyKinect2HardwareManager: Uses Windows Kinect SDK via COM

Select via HARDWARE_MANAGER env var: 'openni2' (default) or 'pykinect2'.
"""

from cv_system.hardware.manager import HardwareError, HardwareManager
from cv_system.hardware.pykinect2_manager import PyKinect2HardwareManager

__all__ = ["HardwareManager", "PyKinect2HardwareManager", "HardwareError"]
