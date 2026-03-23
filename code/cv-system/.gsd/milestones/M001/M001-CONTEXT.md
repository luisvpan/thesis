# M001: CV Core Pipeline (No WebSocket)

**Gathered:** 2026-03-22
**Status:** Ready for planning

## Implementation Decisions

### Calibration Method
- **Decision:** Config file only (no interactive OpenCV window)
- **Rationale:** Simplifies deployment, calibration is a one-time per-session task
- **Implementation:** User provides camera coordinates (4 corners) in config; projector coordinates (4 corners) in config; system computes homography from these 8 points

### Touch Detection Strategy
- **Decision:** Depth-based only using ring buffer + dmax
- **Rationale:** Fits current constraints (Kinect V2 depth stream at 512x424), simpler implementation, adequate for children's tangible blocks
- **Implementation:**
  - Ring buffer (N frames) to accumulate depth differences
  - dmax_map (per-session reference) to distinguish table surface vs objects
  - Touch = depth difference exceeds threshold AND is within table region

## Agent's Discretion

You may decide on:
- Exact ring buffer size N (start with N=5-10 frames)
- Depth difference threshold values (tune based on testing)
- Coordinate system conventions (camera vs projector origin)
- Error handling for OpenNI2 initialization failures
- Logging verbosity and output format

## Deferred Ideas

These ideas came up but belong to later milestones:
- RGB-based blob detection for touch localization
- YOLO model for tangible piece detection
- Interactive OpenCV calibration UI with real-time visualization
- Performance profiling and optimization
- Multiple calibration profiles for different setups

## Constraints from PROJECT_GOALS.md

- **Hardware:** Kinect V2 (depth 512x424, RGB 1920x1080, 30fps), projector (1920x1080)
- **Latency:** Detection must operate at 30fps for responsive interaction
- **Recalibration:** Once at the start of each session (~17s for 500 dmax_map frames)
- **Architecture:** 4-layer modular architecture (Hardware, Calibration, Transform, Detection)
- **Tech stack:** Python, uv, OpenNI2, OpenCV, numpy
- **Validation:** ruff (lint/format), pytest

## Reference ADRs

- ADR-001: Python as the system language
- ADR-002: uv as the package manager
- ADR-003: 4-point homography for calibration
- ADR-004: 4-layer modular architecture
- ADR-005: dmax_map as in-memory per-session reference
- ADR-008: Ruff as linter and formatter
