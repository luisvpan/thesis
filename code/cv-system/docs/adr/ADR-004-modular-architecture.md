# ADR-004: 4-layer modular architecture for the CV system

## Status

Accepted

## Date

2026-03-19

## Context

The codebase inherited from the previous thesis is a monolithic file of ~500 lines (`../../../computer-vision-manager/ultimate_calibrate_area.py`) that mixes Kinect integration, image processing, calibration, touch detection, projector communication, and configuration persistence. This makes any change risky and the code unmaintainable. Magic numbers are scattered throughout the file, coordinate conventions are inconsistent, and there is considerable dead code.

## Decision

We restructure the CV system into **4 layers with separated responsibilities**, plus a communication module:

### Layer 1: Hardware Manager (`hardware/`)

The only module that knows about OpenNI2. Initializes the Kinect V2, configures and exposes the depth (512x424) and RGB (1920x1080) streams as NumPy arrays. Manages the device lifecycle (initialization, teardown). No other module imports OpenNI2 directly.

### Layer 2: Calibrator (`calibration/`)

Orchestrates the per-session calibration process. Contains three submodules: marker generation for the projector, marker detection in the RGB image, and dmax_map generation. Consumes frames from the Hardware Manager. Produces an immutable `CalibrationResult` containing the homography matrix, the dmax_map, and session metadata.

### Layer 3: Coordinate Transformer (`transform/`)

Stateless service post-calibration. Encapsulates the 3x3 homography matrix and exposes bidirectional transformations: `camera_to_projector(point)` and `projector_to_camera(point)` via `cv2.perspectiveTransform`. Constructed from the `CalibrationResult`.

### Layer 4: Detection Layer (`detection/`)

Interaction detection modules for the work surface. The `TouchDetector` compares depth frames against the dmax_map using a sliding window (ring buffer) to filter noise. The `PieceDetector` (future) will detect tangible language pieces in the RGB stream via YOLO. Both use the Coordinate Transformer to report positions in projector coordinates.

### WebSocket Bridge (`bridge/`)

Communicates detection events to the Language Runtime via WebSocket. Serializes positions and events in JSON format.

### Configuration

All magic numbers are externalized into a `SessionConfig` loaded at startup. Hardware parameters (resolutions, FPS) are configured in the Hardware Manager. Calibration parameters (number of frames for dmax, depth range) are passed to the Calibrator. Detection parameters (sliding window size, thresholds) are passed to the Detection Layer.

## Consequences

The monolithic file is replaced by ~10 files with clear responsibilities. Each module is testable in isolation. If the depth sensor changes, only the Hardware Manager is affected. If calibration is improved, only the Calibrator changes. Piece detection (YOLO) can be developed independently from touch detection. The architecture diagram in LikeC4 (`../architecture/cv-system-architecture.likec4`) documents the layers and their relationships.
