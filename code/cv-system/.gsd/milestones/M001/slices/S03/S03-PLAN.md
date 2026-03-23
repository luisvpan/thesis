# S03: Calibrator

**Goal:** System reads 4 corner pairs from config, computes 4-point homography matrix using cv2.getPerspectiveTransform, captures 500 depth frames to generate dmax_map (per-pixel depth mode), and returns immutable CalibrationResult with H, dmax_map, and metadata.

**Demo:** Run `Calibrator.run()` with mock hardware; it computes H, generates dmax_map from N frames, validates results, and returns CalibrationResult with correct shapes and types.

## Must-Haves

- System reads 4 corner point pairs (camera + projector coordinates) from configuration
- Computes 4-point homography matrix using cv2.getPerspectiveTransform
- Captures N=500 depth frames to generate dmax_map via histogram-based mode calculation
- Returns immutable CalibrationResult dataclass containing H (3x3 float32), dmax_map (424, 512 uint16), and metadata
- All calibration parameters externalized to CalibrationConfig (no magic numbers)

## Proof Level

- This slice proves: **contract**
- Real runtime required: no
- Human/UAT required: no

## Verification

- `uv run pytest tests/test_calibrator.py -v` — All calibrator tests pass
- `uv run ruff check src/cv_system/config.py src/cv_system/calibration/` — No lint errors
- `uv run ruff format src/cv_system/config.py src/cv_system/calibration/ --check` — Code formatted

## Observability / Diagnostics

- Runtime signals: Calibration progress printed to stdout (step 1, 2, 3, frame capture progress)
- Inspection surfaces: CalibrationResult.metadata contains stats dict with mean, std, valid_pixel_ratio
- Failure visibility: ValueError with descriptive messages (which attribute missing, invalid corner count, validation failed)
- Redaction constraints: none (no PII in calibration data)

## Integration Closure

- Upstream surfaces consumed: SessionConfig.calibration (from S01), HardwareManager.get_depth_frame() (from S02)
- New wiring introduced in this slice: Calibrator class connects config.calibration attributes to hardware manager
- What remains before the milestone is truly usable end-to-end: S04 (Coordinate Transformer) and S05 (Detection Layer + main loop)

## Tasks

- [ ] **T01: Fix config attribute names to match calibrator expectations** `est:15m`
  - Why: Calibrator expects `camera_corners`, `projector_corners`, `dmax_num_frames` but config provides `marker_camera_coords`, `marker_projector_coords`, `num_dmax_frames`. Fixing this mismatch unblocks the calibrator.
  - Files: `src/cv_system/config.py`, `config/session.json`
  - Do:
    1. In `CalibrationConfig`, rename `marker_camera_coords` → `camera_corners`
    2. In `CalibrationConfig`, rename `marker_projector_coords` → `projector_corners`
    3. In `CalibrationConfig`, rename `num_dmax_frames` → `dmax_num_frames`
    4. Update validator function names from `marker_*` to `camera_*/projector_*/dmax_*`
    5. Update `config/session.json` field names to match new attributes
  - Verify: `uv run ruff check src/cv_system/config.py --fix` passes
  - Done when: All three attributes renamed in config.py and session.json, no ruff errors

- [ ] **T02: Validate calibrator integration with fixed config** `est:10m`
  - Why: Verify that the attribute name fix allows calibrator to initialize and run successfully with actual config.
  - Files: `src/cv_system/config.py`, `tests/test_calibrator.py`, `config/session.json`
  - Do:
    1. Run `uv run pytest tests/test_calibrator.py -v` to verify all tests pass
    2. Run `uv run ruff check src/ && uv run ruff format src/ --check` for code quality
    3. Verify test_calibrator.py tests use correct attribute names in mock fixtures
  - Verify: `uv run pytest tests/test_calibrator.py -v` returns exit code 0 (all tests pass)
  - Done when: All calibrator tests pass, no lint/format errors

## Files Likely Touched

- `src/cv_system/config.py`
- `config/session.json`
- `tests/test_calibrator.py` (read-only, to verify fixtures match)
