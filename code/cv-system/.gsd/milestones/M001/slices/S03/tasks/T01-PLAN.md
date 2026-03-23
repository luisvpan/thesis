---
estimated_steps: 5
estimated_files: 2
skills_used:
  - lint
  - test
---

# T01: Fix config attribute names to match calibrator expectations

**Slice:** S03 — Calibrator
**Milestone:** M001

## Description

Fix the attribute name mismatch between the configuration module and the calibrator. The calibrator expects `camera_corners`, `projector_corners`, and `dmax_num_frames`, but CalibrationConfig currently provides `marker_camera_coords`, `marker_projector_coords`, and `num_dmax_frames`. This task renames these attributes in both the Pydantic model and the example JSON config file.

## Steps

1. Read `src/cv_system/config.py` to locate the CalibrationConfig class
2. Rename the field `marker_camera_coords` to `camera_corners` in CalibrationConfig
3. Rename the field `marker_projector_coords` to `projector_corners` in CalibrationConfig
4. Rename the field `num_dmax_frames` to `dmax_num_frames` in CalibrationConfig
5. Update the @field_validator function name `marker_camera_coords` → `camera_corners`
6. Update the @field_validator function name `marker_projector_coords` → `projector_corners`
7. Update the @field_validator function name `num_dmax_frames` → `dmax_num_frames`
8. Update all docstrings and comments to reflect the new attribute names
9. Read `config/session.json` and update field names to match
10. Run `uv run ruff check src/cv_system/config.py --fix` to apply auto-fixes
11. Run `uv run ruff format src/cv_system/config.py --check` to verify formatting

## Must-Haves

- [ ] CalibrationConfig.camera_corners field exists and is a list of (y, x) tuples
- [ ] CalibrationConfig.projector_corners field exists and is a list of (y, x) tuples
- [ ] CalibrationConfig.dmax_num_frames field exists and is an int
- [ ] Old attribute names (marker_camera_coords, marker_projector_coords, num_dmax_frames) are removed
- [ ] All @field_validator decorators reference the new field names
- [ ] config/session.json uses the new field names (camera_corners, projector_corners, dmax_num_frames)
- [ ] Code passes ruff lint with no errors
- [ ] Code passes ruff format check

## Verification

- Run `uv run ruff check src/cv_system/config.py` — returns exit code 0 (no errors)
- Run `uv run ruff format src/cv_system/config.py --check` — returns exit code 0 (no formatting changes needed)
- Run `python -c "from cv_system.config import CalibrationConfig; c = CalibrationConfig(); assert hasattr(c.calibration, 'camera_corners'); assert hasattr(c.calibration, 'projector_corners'); assert hasattr(c.calibration, 'dmax_num_frames')"` — imports successfully and attributes exist

## Observability Impact

- Signals added/changed: None (this is a refactor, not behavioral change)
- How a future agent inspects this: Read CalibrationConfig class definition to see field names
- Failure state exposed: Validation errors from Pydantic if field types or validators are incorrect

## Inputs

- `src/cv_system/config.py` — Existing CalibrationConfig class with outdated field names
- `config/session.json` — Example config file with outdated field names

## Expected Output

- `src/cv_system/config.py` — Updated CalibrationConfig with camera_corners, projector_corners, dmax_num_fields
- `config/session.json` — Updated JSON with matching field names
