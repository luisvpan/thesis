---
id: T01
parent: S02
milestone: M001
provides:
  - HardwareManager class with initialize() method
  - OpenNI2 context initialization and device discovery
  - Depth and RGB stream creation with configurable resolution and FPS
  - _find_video_mode() helper to find matching video mode
  - HardwareError custom exception
  - pyproject.toml updated with opencv-python and numpy dependencies
requires:
  - slice: S01
    provides: CameraConfig for stream parameters
  - slice: S00
    provides: Initial project structure
affects: [S03, S04, S05]
key_files:
  - src/cv_system/hardware/manager.py (new, 7598 bytes)
  - src/cv_system/hardware/__init__.py (new)
  - pyproject.toml (added opencv-python, numpy)
key_decisions:
  - OpenNI2 import inside initialize() to handle missing library gracefully
  - Video mode search loops through supported modes to find exact match
  - OPENNI2_REDIST_PATH checked before initialization
patterns_established:
  - Only hardware module imports OpenNI2 (per ADR-004)
  - All errors wrapped in HardwareError with context
drill_down_paths:
  - .gsd/milestones/M001/slices/S02/tasks/T01-PLAN.md
duration: 12min
verification_result: pass
completed_at: 2026-03-22T23:55:00Z
---

# T01: OpenNI2 initialization and device discovery

**Hardware manager with OpenNI2 integration for Kinect V2 depth and RGB streams.**

## What Happened

Implemented HardwareManager class with complete OpenNI2 integration:

1. **initialize() method**: Sets up OpenNI2 context, opens any Kinect V2 device, creates depth and RGB streams with configurable resolution/FPS from CameraConfig
2. **_find_video_mode() helper**: Searches device's supported video modes to find exact match for requested resolution and FPS
3. **HardwareError custom exception**: Wraps all OpenNI2 failures with clear context
4. **OPENNI2_REDIST_PATH validation**: Checks environment variable before attempting OpenNI2 import
5. **Added dependencies**: opencv-python>=4.0.0, numpy>=2.0.0 to pyproject.toml

The manager properly handles initialization failures with cleanup of any partially allocated resources. Only the hardware module imports OpenNI2, enforcing the architecture boundary (per ADR-004).

## Deviations
None — followed T01-PLAN.md exactly. Note: get_depth_frame(), get_rgb_frame(), shutdown(), and context manager were also implemented (these are T02/T03).

## Files Created/Modified
- `src/cv_system/hardware/__init__.py` — New, 257 bytes, exports HardwareManager
- `src/cv_system/hardware/manager.py` — New, 7598 bytes, 241 lines with full HardwareManager implementation
- `pyproject.toml` — Added opencv-python and numpy dependencies
