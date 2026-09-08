# ADR-005: dmax_map as per-session background reference (in-memory with optional persistence)

## Status

Accepted

## Date

2026-03-19 (updated 2026-03-24)

## Context

The CV system needs to distinguish between the base surface (the table) and objects on it (hands, language pieces). The Kinect V2 has inherent noise in its depth measurements that varies frame to frame, so a single reference frame is insufficient. The current code captures 500 frames and computes the per-pixel depth mode to generate a stable dmax_map.

The original code uses a dense histogram approach: it hardcodes `min_depth=650` and `max_depth=800` to preallocate a 3D array of shape `(h, w, max - min)` where the third dimension are histogram bins. This has two problems: the hardcoded range assumes a specific table-to-Kinect distance and breaks if the setup changes, and the memory cost scales with the range width.

### Reference

Khoshelham, K. & Elberink, S.O. (2012). "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications." *Sensors*, 12(2), 1437-1454. Documents that the random error of Kinect depth measurement increases with distance, ranging from a few millimeters up to ~4 cm at the maximum range.

## Decision

The dmax_map is calculated at the start of each session as part of calibration and kept **in memory** throughout the session. It is not persisted to disk as part of the normal flow.

### Direct mode estimation (no histogram, no depth range)

The dense histogram approach is replaced by a **direct approach**: store all N frames in a `(N, h, w)` uint16 array and compute the per-pixel mode directly (via `scipy.stats.mode` or equivalent). This eliminates the need to know or derive the depth range beforehand.

The memory cost is straightforward: for `num_frames_for_depth_mode_estimation = 500` and a ROI of 300x200 pixels at uint16, this is ~60MB of temporary memory during calibration. This is negligible for a machine running a Kinect V2 and OpenCV.

This leaves a single configurable parameter:

- **`num_frames_for_depth_mode_estimation`** (default: 500) — How many depth frames to capture for mode calculation. More frames = more stable dmax_map, but longer calibration time (~17s at 30fps for 500 frames).

There is no depth range parameter, no probe frame step, and no histogram bin configuration.

### Lifecycle

1. At session start, the DMax Generator captures `num_frames_for_depth_mode_estimation` depth frames from the ROI into a `(N, h, w)` array.
2. For each pixel, it computes the depth mode across the N frames.
3. The resulting dmax_map is stored in the `CalibrationResult` as a NumPy `uint16` array (values in millimeters, consistent with `PIXEL_FORMAT_DEPTH_1_MM`).
4. The `(N, h, w)` temporary array is discarded after mode computation.
5. During the session, the Touch Detector compares each frame against the dmax_map.
6. At session end, the dmax_map is discarded.

### Optional persistence

For debugging and development, the `CalibrationResult` (which includes the dmax_map and the homography matrix) can optionally be saved to disk with a `save(path)` method and reloaded with `load(path)`. This allows iterating on touch/piece detection without recalibrating every time the software restarts. This functionality is exclusive to development and is not used in sessions with end users.

### Ring buffer for touch_history

The inherited code accumulates all touch masks in a list that grows indefinitely, causing progressive memory and performance degradation. This is replaced by a preallocated ring buffer of size N (configurable, default 3-5 frames), implemented as a NumPy array of shape `(N, h, w)` with a circular index. The window sum is computed over the full buffer, with constant cost regardless of session duration.

## Consequences

Recalibrating each session adds ~17 seconds to startup (500 frames at 30fps). This cost is acceptable for a system that recalibrates once per session. The single configurable parameter (`num_frames_for_depth_mode_estimation`) controls the tradeoff between stability and startup time. The implementation is simpler than the histogram approach: no range derivation, no bin management, no probe frames. The ~60MB temporary memory cost is negligible. Optional persistence speeds up the development cycle without adding complexity to the production flow.
