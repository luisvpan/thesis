# T02: Frame capture and numpy conversion

**Slice:** S02
**Milestone:** M001

## Goal
Implement methods to capture frames from OpenNI2 streams and convert to numpy arrays with proper shape/dtype.

## Must-Haves

### Truths
- get_depth_frame() returns numpy array (424, 512) uint16
- get_rgb_frame() returns numpy array (1080, 1920) uint8 (BGR format)
- Frames are flipped horizontally (mirrored) per inherited code pattern
- Shape and dtype match CameraConfig specifications

### Artifacts
- `src/cv_system/hardware/manager.py` — Added get_depth_frame() and get_rgb_frame() methods
- Methods use OpenNI2 frame.read_frame() and buffer extraction
- Arrays are properly shaped as (height, width)

### Key Links
- T01 → provides device, depth_stream, rgb_stream for frame reading
- T03 → uses these methods in lifecycle
- S03 → Calibrator calls these methods for frame capture

## Steps
1. Implement get_depth_frame(self) -> np.ndarray:
   - Call depth_stream.read_frame()
   - Get buffer as uint16: frame.get_buffer_as_uint16()
   - Reshape to camera.depth_resolution (height, width)
   - Apply cv2.flip(frame, 1) to mirror horizontally
   - Return numpy array
2. Implement get_rgb_frame(self) -> np.ndarray:
   - Call rgb_stream.read_frame()
   - Get buffer as uint8: frame.get_buffer_as_uint8()
   - Reshape to camera.rgb_resolution (height, width, 3)
   - Apply cv2.flip(frame, 1) to mirror horizontally
   - Convert RGB to BGR: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
   - Return numpy array
3. Add docstrings to both methods
4. Run `uv run ruff check src/` to verify linting

## Context
- Depth frame: get_buffer_as_uint16, reshape as (height, width)
- RGB frame: get_buffer_as_uint8, reshape as (height, width, 3)
- cv2.flip(frame, 1) mirrors horizontally (left-right)
- cv2.cvtColor RGB→BGR for OpenCV compatibility
- Handle case where stream is not initialized
