# ADR-004: 4-layer modular architecture for the CV system

## Status

Accepted

## Date

2026-03-19 (updated 2026-03-24)

## Context

The codebase inherited from the previous thesis is a monolithic file of ~500 lines (`../../../computer-vision-manager/ultimate_calibrate_area.py`) that mixes Kinect integration, image processing, calibration, touch detection, projector communication, and configuration persistence. This makes any change risky and the code unmaintainable. Magic numbers are scattered throughout the file, coordinate conventions are inconsistent, and there is considerable dead code.

## Decision

We restructure the CV system into **4 layers with separated responsibilities**, plus a communication module:

### Layer 1: Hardware Manager (`hardware/`)

The only module that knows about OpenNI2. Initializes the Kinect V2, configures and exposes the depth (512x424) and RGB (1920x1080) streams as NumPy arrays. Manages the device lifecycle (initialization, teardown). No other module imports OpenNI2 directly.

The Hardware Manager explicitly configures and validates the **pixel format** of each stream: `PIXEL_FORMAT_DEPTH_1_MM` for depth (uint16, values in millimeters) and `PIXEL_FORMAT_RGB888` for color (3 bytes per pixel, RGB order). These formats are exposed as stream metadata so consuming modules know how to interpret the raw data. If the Kinect does not support the requested format, the Hardware Manager fails at initialization with a clear error rather than producing silently corrupt data.

### Layer 2: Calibrator (`calibration/`)

Orchestrates the per-session calibration process. Contains three submodules:

**Marker Projector:** Generates a fullscreen black image at the projector's resolution and draws 4 high-contrast white squares at the configured `projector_corners`. Displays the image on the projector's display using `cv2.namedWindow` + `cv2.setWindowProperty(WINDOW_FULLSCREEN)`. No additional libraries beyond OpenCV are needed.

**Marker Detector:** Captures a frame from the Kinect's RGB stream and detects the 4 projected markers using threshold + contour detection. Extracts centroids as the computed `camera_corners`. These are never configured — they are always the result of detection. Maps RGB coordinates to depth space via the Kinect V2's hardware registration.

**DMax Generator:** Captures `num_frames_for_depth_mode_estimation` depth frames from the ROI, stores them in a temporary `(N, h, w)` array, and computes the per-pixel depth mode directly. No histogram binning, no depth range parameters (see ADR-005).

The Calibrator produces an immutable `CalibrationResult` containing the homography matrix, the dmax_map, and session metadata.

### Layer 3: Coordinate Transformer (`transform/`)

Stateless service post-calibration. Encapsulates the 3x3 homography matrix and exposes bidirectional transformations: `camera_to_projector(point)` and `projector_to_camera(point)` via `cv2.perspectiveTransform`. Constructed from the `CalibrationResult`.

### Layer 4: Detection Layer (`detection/`)

Interaction detection modules for the work surface. The `TouchDetector` compares depth frames against the dmax_map using a sliding window (ring buffer) to filter noise. The `PieceDetector` (future) will detect tangible language pieces in the RGB stream via YOLO. Both use the Coordinate Transformer to report positions in projector coordinates.

### WebSocket Bridge (`bridge/`)

Communicates detection events to the Language Runtime via WebSocket. Serializes positions and events in JSON format.

### Configuration

All magic numbers are externalized into a `SessionConfig` loaded at startup:

- **Hardware Manager:** resolutions, FPS, pixel formats.
- **Calibrator:** `projector_corners` (where to draw markers), `num_frames_for_depth_mode_estimation` (frames for dmax_map).
- **Detection Layer:** sliding window size, touch thresholds.
- **Computed (not configured):** `camera_corners` (detected by Marker Detector), depth range (eliminated — direct mode estimation).

## Consequences

The monolithic file is replaced by ~10 files with clear responsibilities. Each module is testable in isolation. If the depth sensor changes, only the Hardware Manager is affected. If calibration is improved, only the Calibrator changes. Piece detection (YOLO) can be developed independently from touch detection. The architecture diagram in LikeC4 (`../architecture/cv-system-architecture.likec4`) documents the layers and their relationships.
