# S02: Hardware Manager

**Goal:** Integrate Kinect V2 hardware via OpenNI2, exposing depth and RGB streams as NumPy arrays with proper lifecycle management.

**Demo:** System initializes Kinect V2, captures and displays 10 depth frames (512x424) and 10 RGB frames (1920x1080) to console (shape + dtype info), then shuts down cleanly. Only HardwareManager imports OpenNI2.

## Must-Haves
- HardwareManager class that initializes Kinect via OpenNI2
- `get_depth_frame()` returns numpy array (424, 512) uint16
- `get_rgb_frame()` returns numpy array (1080, 1920) uint8 (or compatible)
- `initialize()` and `shutdown()` methods manage device lifecycle
- Only hardware/ module imports openni (per ADR-004)
- Proper error handling for OpenNI2 initialization failures

## Tasks

- [x] **T01: OpenNI2 initialization and device discovery**
  Set up OpenNI2 context, discover Kinect V2 device, create depth and color streams.

- [x] **T02: Frame capture and numpy conversion**
  Implement methods to capture frames from OpenNI2 streams and convert to numpy arrays with proper shape/dtype.

- [x] **T03: Lifecycle management and error handling**
  Implement initialize(), shutdown(), and context manager protocol with robust error handling.

## Files Likely Touched
- `src/cv_system/hardware/manager.py` (new)
- `src/cv_system/hardware/__init__.py` (new)
- `pyproject.toml` (verify openni dep, add opencv-python)
- `tests/test_hardware.py` (new, integration tests requiring hardware)
