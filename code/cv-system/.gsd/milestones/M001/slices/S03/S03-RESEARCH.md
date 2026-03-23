# S03: Calibrator — Research

**Researched:** 2026-03-23
**Domain:** Computer Vision Calibration — Homography + DMax Map
**Confidence:** HIGH

## Summary

The calibrator orchestration layer is already implemented but has a critical compatibility issue: attribute names in the calibrator don't match the CalibrationConfig from S01. The core logic is sound — homography computation using `cv2.getPerspectiveTransform`, dmax_map generation via histogram-based mode calculation — but the bridge between S01 config and S03 calibrator needs fixing.

**Primary fix:** Rename config attributes to match the calibrator's expectations:
- `marker_camera_coords` → `camera_corners`
- `marker_projector_coords` → `projector_corners`
- `num_dmax_frames` → `dmax_num_frames`

Or alternatively: Update the calibrator to use the existing attribute names from S01's CalibrationConfig.

## Recommendation

Update S01's `CalibrationConfig` to use the attribute names expected by the S03 calibrator. This is the minimal fix since:
1. The calibrator implementation and tests are already complete
2. The attribute names `camera_corners` and `projector_corners` are more self-documenting
3. Only one file needs to change (config.py)

Alternative: Update calibrator.py to use `marker_camera_coords`, `marker_projector_coords`, `num_dmax_frames`. This keeps S01 unchanged but requires changes to calibrator.py and test fixtures.

## Implementation Landscape

### Key Files

- `src/cv_system/config.py` — **NEEDS UPDATE**: Change CalibrationConfig field names to match calibrator expectations
- `src/cv_system/calibration/calibrator.py` — Calibrator orchestration, expects `camera_corners`, `projector_cores`, `dmax_num_frames`
- `src/cv_system/calibration/homography.py` — Homography computation using cv2.getPerspectiveTransform, already correct
- `src/cv_system/calibration/dmax.py` — DMax map generation via histogram mode, already correct
- `src/cv_system/calibration/result.py` — Immutable CalibrationResult dataclass, already correct
- `src/cv_system/calibration/__init__.py` — Module exports, already correct
- `config/session.json` — Example config, will need to match updated field names
- `tests/test_calibrator.py` — Tests already written and passing with mock fixtures

### Build Order

1. **Update CalibrationConfig field names** in `src/cv_system/config.py`
   - Change `marker_camera_coords` → `camera_corners`
   - Change `marker_projector_coords` → `projector_corners`
   - Change `num_dmax_frames` → `dmax_num_frames`
   - Update validator function names to match
   - This unblocks the calibrator to work with actual config

2. **Update config/session.json** to use new field names

3. **Run validation**: `uv run ruff check src/ && uv run ruff format src/ --check && uv run pytest tests/test_calibrator.py`

4. **Integration test**: Run a full calibration with mock hardware to verify end-to-end

### Verification Approach

- Unit tests: `uv run pytest tests/test_calibrator.py -v` — Should pass all existing tests
- Integration test: Create a simple script that loads config from session.json, creates a Calibrator with mock hardware, and calls `run()` — Should complete without errors
- Lint/format: `uv run ruff check src/ && uv run ruff format src/ --check` — Should pass

## Common Pitfalls

- **Attribute name mismatch** — The core issue: calibrator expects `camera_corners` but config provides `marker_camera_coords`. Verify names match exactly.
- **Coordinate format** — Config stores corners as (y, x) but calibrator expects (x, y). Check the homography.py code — it expects the format from config, so this should work if the arrays are passed through correctly.
- **Float32 requirement** — cv2.getPerspectiveTransform returns float64 but calibrator validates for float32. The conversion happens in compute_homography() — verify this is working.
- **DMax histogram memory** — The histogram array is (depth_width, height, width) which could be large for wide depth ranges. Config limits this to 650-800 (151 bins), which is reasonable.

## Open Risks

- **Low risk** — The implementation is already complete and tested. The only change is attribute name alignment.

## Sources

- OpenCV getPerspectiveTransform documentation (from get_library_docs) — Confirms the function expects 4 (x, y) point pairs in float32 format
- S01 Summary — Confirms CalibrationConfig structure and field names
- S02 Summary — Confirms HardwareManager.get_depth_frame() returns (424, 512) uint16 array
- M001-RESEARCH — Describes the dmax_map algorithm using histogram-based mode calculation

## Forward Intelligence from S01 and S02

- **From S01 Summary:** The config module uses Pydantic v2 validators (@field_validator, not @v1 @validator). Follow this pattern when updating field names.
- **From S02 Summary:** HardwareManager frames are already mirrored horizontally. The calibrator doesn't need to flip frames again.
- **From inherited code:** The dmax_map algorithm uses a 3D histogram (depth_width, height, width) and computes mode via np.argmax. This is already implemented correctly in dmax.py.
