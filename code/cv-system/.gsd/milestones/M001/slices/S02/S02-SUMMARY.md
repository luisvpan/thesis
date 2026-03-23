# S02: Hardware Manager — Summary

**Completed:** 2026-03-22T23:55:00Z

## Goal
Integrate Kinect V2 hardware via OpenNI2, exposing depth and RGB streams as NumPy arrays with proper lifecycle management.

## What Was Built

Complete Hardware Manager module for Kinect V2 integration:

1. **HardwareManager class** with full lifecycle:
   - `initialize(config)`: Sets up OpenNI2 context, opens device, creates and starts depth/RGB streams
   - `get_depth_frame()`: Returns numpy array (height, width) uint16, mirrored horizontally
   - `get_rgb_frame()`: Returns numpy array (height, width, 3) uint8 BGR, mirrored
   - `shutdown()`: Stops streams, closes device, unloads OpenNI2
   - Context manager protocol: `with HardwareManager() as hw: ...`

2. **OpenNI2 integration**:
   - Context initialization with OPENNI2_REDIST_PATH validation
   - Device discovery via `openni2.Device.open_any()`
   - Video mode search to find matching resolution and FPS
   - Stream creation and configuration

3. **Error handling**:
   - Custom HardwareError exception wraps all failures
   - Partial cleanup on initialization failures
   - Clear error messages for missing library, missing env var, unsupported mode

4. **Architecture boundary enforced**:
   - Only hardware module imports OpenNI2 (per ADR-004)
   - All other modules access hardware via HardwareManager interface

## Must-Haves Verification

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | HardwareManager initializes Kinect via OpenNI2 | ✓ PASS | initialize() method creates context, device, streams |
| 2 | get_depth_frame() returns numpy array (424, 512) uint16 | ✓ PASS | Method implementation uses get_buffer_as_uint16() and reshape |
| 3 | get_rgb_frame() returns numpy array (1080, 1920) uint8 | ✓ PASS | Method implementation uses get_buffer_as_uint8() and reshape |
| 4 | initialize() and shutdown() manage lifecycle | ✓ PASS | shutdown() stops streams, closes device, unloads context |
| 5 | Only hardware module imports openni | ✓ PASS | OpenNI2 import is in manager.py, no other module imports it |

## Files Created/Modified

| File | Description |
|------|-------------|
| `src/cv_system/hardware/manager.py` | HardwareManager class, 241 lines, 7598 bytes |
| `src/cv_system/hardware/__init__.py` | Exports HardwareManager |
| `pyproject.toml` | Added opencv-python>=4.0.0, numpy>=2.0.0 |

## Provides to Downstream Slices

- `HardwareManager` — Main interface for hardware access
- `get_depth_frame()` — Used by Calibrator for dmax_map and by Detection Layer for touch detection
- `get_rgb_frame()` — Used by Calibrator (future marker detection)
- `CameraConfig` parameters used for stream resolution and FPS

## Key Decisions

1. OpenNI2 import inside initialize() for graceful error handling
2. Video mode search loops through all supported modes
3. Horizontal mirroring (cv2.flip) matches inherited code pattern
4. RGB→BGR conversion for OpenCV compatibility
5. Context manager ensures cleanup even on exceptions

## Patterns Established

- All hardware access goes through HardwareManager
- Only hardware module imports OpenNI2 (architecture boundary)
- All errors wrapped in HardwareError with context
- Frame shapes are always (height, width) or (height, width, 3)
- Resource cleanup is robust (continues even if some operations fail)

## Drill Down

Task summaries:
- T01: `.gsd/milestones/M001/slices/S02/tasks/T01-SUMMARY.md`
- T02: `.gsd/milestones/M001/slices/S02/tasks/T02-SUMMARY.md`
- T03: `.gsd/milestones/M001/slices/S02/tasks/T03-SUMMARY.md`

Task plans:
- T01: `.gsd/milestones/M001/slices/S02/tasks/T01-PLAN.md`
- T02: `.gsd/milestones/M001/slices/S02/tasks/T02-PLAN.md`
- T03: `.gsd/milestones/M001/slices/S02/tasks/T03-PLAN.md`
