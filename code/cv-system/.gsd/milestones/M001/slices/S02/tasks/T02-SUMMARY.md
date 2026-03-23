---
id: T02
parent: S02
milestone: M001
provides:
  - get_depth_frame() method returning numpy array (424, 512) uint16
  - get_rgb_frame() method returning numpy array (1080, 1920, 3) uint8 BGR
  - Horizontal mirroring (cv2.flip) for both stream types
  - Proper buffer extraction and reshaping from OpenNI2 frames
requires:
  - task: T01
    provides: HardwareManager with initialized streams
  - slice: S01
    provides: CameraConfig for resolution references
affects: [S03, S05]
key_files:
  - src/cv_system/hardware/manager.py (added frame capture methods)
key_decisions:
  - cv2.flip(frame, 1) to mirror horizontally (matches inherited code)
  - cv2.cvtColor RGB→BGR for OpenCV compatibility
  - Raise HardwareError on frame capture failures
patterns_established:
  - All frames returned as numpy arrays with consistent dtype
  - Frame shapes always (height, width) or (height, width, 3)
drill_down_paths:
  - .gsd/milestones/M001/slices/S02/tasks/T02-PLAN.md
duration: Implemented with T01
verification_result: pass
completed_at: 2026-03-22T23:55:00Z
---

# T02: Frame capture and numpy conversion

**Frame capture methods returning numpy arrays with proper mirroring.**

## What Happened

Implemented two frame capture methods in HardwareManager:

1. **get_depth_frame()**: Captures depth frame, extracts buffer as uint16, reshapes to (height, width), applies horizontal flip with cv2.flip()
2. **get_rgb_frame()**: Captures RGB frame, extracts buffer as uint8, reshapes to (height, width, 3), applies horizontal flip, converts RGB to BGR with cv2.cvtColor()

Both methods raise HardwareError if hardware is not initialized or frame capture fails. Horizontal mirroring matches the inherited code pattern.

## Deviations
None — integrated into HardwareManager as designed. Methods follow T02-PLAN.md exactly.

## Files Created/Modified
- `src/cv_system/hardware/manager.py` — Added get_depth_frame() and get_rgb_frame() methods (50 lines)
