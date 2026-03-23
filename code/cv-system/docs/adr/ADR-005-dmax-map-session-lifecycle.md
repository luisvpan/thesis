# ADR-005: dmax_map as per-session background reference (in-memory with optional persistence)

## Status

Accepted

## Date

2026-03-19

## Context

The CV system needs to distinguish between the base surface (the table) and objects on it (hands, language pieces). The Kinect V2 has inherent noise in its depth measurements that varies frame to frame, so a single reference frame is insufficient. The current code captures 500 frames and computes the per-pixel depth mode to generate a stable dmax_map.

The question is whether this map should be persisted to disk between sessions or recalculated each time, and how it relates to the system lifecycle.

### Reference

Khoshelham, K. & Elberink, S.O. (2012). "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications." *Sensors*, 12(2), 1437-1454.

## Decision

The dmax_map is calculated at the start of each session as part of calibration and kept **in memory** throughout the session. It is not persisted to disk as part of the normal flow.

### Lifecycle

1. At session start, the Calibrator captures N depth frames from the ROI (N is configurable, default 500).
2. For each pixel, it computes the depth mode within an expected range (configurable, not hardcoded as 650-800).
3. The resulting dmax_map is stored in the `CalibrationResult` as a NumPy `uint16` array.
4. During the session, the Touch Detector compares each frame against this map.
5. At session end, it is discarded.

### Optional persistence

For debugging and development, the `CalibrationResult` (which includes the dmax_map and the homography matrix) can optionally be saved to disk with a `save(path)` method and reloaded with `load(path)`. This functionality is exclusive to development and is not used in sessions with end users.

### Ring buffer for touch_history

The inherited code accumulates all touch masks in a list that grows indefinitely, causing progressive memory and performance degradation. This is replaced by a preallocated ring buffer of size N (configurable, default 3-5 frames), implemented as a NumPy array of shape `(N, h, w)` with a circular index. The window sum is computed over the full buffer, with constant cost regardless of session duration.

## Consequences

Recalibrating each session adds ~17 seconds to startup (500 frames at 30fps). This cost is acceptable for a system that recalibrates once per session. The depth range and number of frames are no longer magic numbers and become part of the session configuration. Optional persistence speeds up the development cycle without adding complexity to the production flow.
