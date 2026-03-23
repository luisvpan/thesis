# M001: CV Core Pipeline (No WebSocket)

**Vision:** End-to-end computer vision pipeline from Kinect V2 depth/RGB streams through to touch detection on the work surface, with per-session calibration and coordinate transformation. WebSocket communication deferred to future milestone.

**Success Criteria:**
- System initializes Kinect V2 and captures depth (512x424) and RGB (1920x1080) streams at 30fps
- Calibration process reads config file, computes 4-point homography matrix, generates dmax_map (500 frames)
- Coordinate transformer correctly maps points between camera and projector spaces using homography
- Touch detector identifies objects/hands on table surface using ring buffer + dmax_map comparison
- All modules are testable, configuration externalized, and no magic numbers remain in code

---

## Slices

- [x] **S01: Configuration Foundation** `risk:low` `depends:[]`
  > After this: System can load session configuration from JSON file with type-safe validation using Pydantic models. Configuration includes hardware parameters (resolutions, FPS), calibration parameters (dmax frames, depth range), and detection parameters (ring buffer size, thresholds).

- [x] **S02: Hardware Manager** `risk:medium` `depends:[S01]`
  > After this: Kinect V2 initializes via OpenNI2 and exposes depth stream (512x424, 30fps) and RGB stream (1920x1080, 30fps) as NumPy arrays. Module handles device lifecycle (init, teardown) and is the only place OpenNI2 is imported.

- [ ] **S03: Calibrator** `risk:high` `depends:[S01, S02]`
  > After this: System reads 4 corner pairs (camera + projector coordinates) from config, computes 4-point homography matrix using cv2.getPerspectiveTransform, captures 500 depth frames to generate dmax_map (per-pixel depth mode), and returns immutable CalibrationResult with H, dmax_map, and metadata.

- [ ] **S04: Coordinate Transformer** `risk:low` `depends:[S03]`
  > After this: Stateless service wraps homography matrix H and exposes bidirectional transformations: camera_to_projector(point) and projector_to_camera(point) using cv2.perspectiveTransform. All coordinates in projector space (1920x1080) for downstream consumption.

- [ ] **S05: Detection Layer + Integration** `risk:medium` `depends:[S03, S04]`
  > After this: Touch detector compares each depth frame against dmax_map using ring buffer (N=5-10 frames) to identify foreground objects. Ring buffer accumulates depth differences, thresholded to produce touch masks. Main module wires all layers together into a runnable session loop.

## Boundary Map

### S01 → S02
Produces:
  - `src/cv_system/config.py` — SessionConfig dataclass, CameraConfig, CalibrationConfig, DetectionConfig
  - `config/session.json` — Example config file with all parameters

Consumes: nothing (leaf node)

### S02 → S03
Produces:
  - `src/cv_system/hardware/manager.py` — HardwareManager class
  - `src/cv_system/hardware/__init__.py` — Exports: HardwareManager
  - Interfaces: get_depth_frame(), get_rgb_frame(), initialize(), shutdown()

Consumes from S01:
  - config.py → CameraConfig (depth_resolution, rgb_resolution, fps)

### S03 → S04
Produces:
  - `src/cv_system/calibration/calibrator.py` — Calibrator class
  - `src/cv_system/calibration/result.py` — CalibrationResult dataclass (H, dmax_map, metadata)
  - `src/cv_system/calibration/__init__.py` — Exports: Calibrator, CalibrationResult

Consumes from S01:
  - config.py → CalibrationConfig (num_dmax_frames, depth_range_min, depth_range_max, marker_coords)

Consumes from S02:
  - hardware/manager.py → get_depth_frame(), get_rgb_frame()

### S04 → S05
Produces:
  - `src/cv_system/transform/transformer.py` — CoordinateTransformer class
  - `src/cv_system/transform/__init__.py` — Exports: CoordinateTransformer
  - Interfaces: camera_to_projector(point), projector_to_camera(point)

Consumes from S03:
  - calibration/result.py → CalibrationResult (homography matrix H)

### S05 → (none, terminal slice)
Produces:
  - `src/cv_system/detection/touch_detector.py` — TouchDetector class
  - `src/cv_system/detection/__init__.py` — Exports: TouchDetector
  - `src/cv_system/main.py` — Session orchestration (init → calibrate → detect loop)
  - Tests for all layers

Consumes from S03:
  - calibration/result.py → CalibrationResult (dmax_map)

Consumes from S04:
  - transform/transformer.py → camera_to_projector(point)
