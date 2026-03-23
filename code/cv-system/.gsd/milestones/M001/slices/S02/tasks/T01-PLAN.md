# T01: OpenNI2 initialization and device discovery

**Slice:** S02
**Milestone:** M001

## Goal
Set up OpenNI2 context, discover Kinect V2 device, create depth and color streams with proper configuration.

## Must-Haves

### Truths
- HardwareManager initializes OpenNI2 context
- Kinect V2 device is discovered and initialized
- Depth stream is created with configured resolution and FPS
- RGB stream is created with configured resolution and FPS
- Initialization failures raise clear exceptions

### Artifacts
- `src/cv_system/hardware/manager.py` — HardwareManager class with initialize() method
- `src/cv_system/hardware/__init__.py` — Exports HardwareManager
- Class has properties: device, depth_stream, rgb_stream

### Key Links
- T02 → uses device, depth_stream, rgb_stream for frame capture
- T03 → adds shutdown() and context manager protocol
- S01 → provides CameraConfig for stream parameters

## Steps
1. Add `opencv-python` and `numpy` to pyproject.toml dependencies
2. Create `src/cv_system/hardware/` directory
3. Create `src/cv_system/hardware/__init__.py` with HardwareManager export
4. Create `src/cv_system/hardware/manager.py` with HardwareManager class
5. In __init__: set context, device, depth_stream, rgb_stream to None
6. Implement initialize(self, config: CameraConfig) method:
   - Import openni2 (only in hardware module per ADR-004)
   - openni2.initialize()
   - Device = openni2.Device.open_any()
   - Set up depth stream: Device.create_depth_stream(), set_video_mode
   - Set up color stream: Device.create_color_stream(), set_video_mode
   - Start streams
7. Add error handling for OpenNI2 initialization failures
8. Run `uv run ruff check src/` to verify linting

## Context
- OpenNI2 API: context initialization, device discovery, stream creation
- Video mode: resolution + FPS configuration
- Use config.camera for resolution and FPS parameters
- Handle case where OPENNI2_REDIST_PATH is not set (needs env var)
