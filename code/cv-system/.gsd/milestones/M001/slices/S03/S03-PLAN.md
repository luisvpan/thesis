# S03: Calibrator

**Goal:** Implement per-session calibration process: read 4 corner pairs from config, compute 4-point homography matrix, generate dmax_map (500 frames), return immutable CalibrationResult.

**Demo:** System reads marker coords from config, generates dmax_map by capturing 500 depth frames, computes homography matrix H, and returns CalibrationResult. Print shows H shape (3,3), dmax_map shape (h,w), and metadata.

## Must-Haves
- Calibrator class that orchestrates calibration process
- Reads 4 marker corner pairs (camera + projector) from CalibrationConfig
- Computes homography matrix H (3x3) using cv2.getPerspectiveTransform
- Captures N depth frames (configurable, default 500) to compute dmax_map
- dmax_map uses per-pixel mode within depth range (configurable, default 650-800)
- CalibrationResult dataclass is immutable (frozen=True) with H, dmax_map, metadata
- All marker detection logic uses RGB stream (per ADR-003), but we're using config file (no detection needed)

## Tasks

- [ ] **T01: CalibrationResult dataclass**
  Define immutable dataclass with homography matrix, dmax_map, and metadata fields.

- [ ] **T02: DMax map generation**
  Implement frame capture loop, histogram accumulation, and mode calculation for dmax_map.

- [ ] **T03: Homography matrix computation**
  Implement 4-point homography calculation using cv2.getPerspectiveTransform from config coords.

- [ ] **T04: Calibrator orchestration**
  Wire dmax generation and homography computation together in Calibrator class with proper error handling.

## Files Likely Touched
- `src/cv_system/calibration/result.py` (new)
- `src/cv_system/calibration/calibrator.py` (new)
- `src/cv_system/calibration/__init__.py` (new)
- `pyproject.toml` (verify opencv-python, numpy deps)
- `tests/test_calibration.py` (new)
