---
id: T01
parent: S03
milestone: M001
provides:
  - CalibrationConfig with renamed attributes: camera_corners, projector_corners, dmax_num_frames
  - Updated @field_validator decorators matching new field names
  - config/session.json with matching field names
key_files:
  - src/cv_system/config.py
  - config/session.json
key_decisions:
  - None (refactor only, no new decisions)
patterns_established:
  - Pydantic @field_validator decorator must reference exact field names from model
observability_surfaces:
  - Validation errors from Pydantic if field names mismatch
  - Runtime attribute checks via hasattr() for field presence verification
duration: 10min
verification_result: passed
completed_at: 2026-03-23T06:51:00Z
blocker_discovered: false
---

# T01: Fix config attribute names to match calibrator expectations

**Renamed CalibrationConfig fields and updated all references to align with calibrator's expected attribute names.**

## What Happened

Fixed an attribute name mismatch between the configuration module and the calibrator expectations. The calibrator requires `camera_corners`, `projector_corners`, and `dmax_num_frames`, but the CalibrationConfig class was providing `marker_camera_coords`, `marker_projector_coords`, and `num_dmax_frames`.

**Changes made:**
- Renamed `marker_camera_coords` → `camera_corners` in CalibrationConfig field definition
- Renamed `marker_projector_coords` → `projector_corners` in CalibrationConfig field definition
- Renamed `num_dmax_frames` → `dmax_num_frames` in CalibrationConfig field definition
- Updated `@field_validator` decorator to reference `"dmax_num_frames"` instead of `"num_dmax_frames"`
- Updated `@field_validator` decorator to reference `"camera_corners", "projector_corners"` instead of `"marker_camera_coords", "marker_projector_coords"`
- Updated error message in validator from `"num_dmax_frames must be positive"` to `"dmax_num_frames must be positive"`
- Updated config/session.json to use new field names

This was a pure refactor with no behavioral changes—the field types, validation logic, and default values all remain identical. Only the attribute names changed to match what the calibrator expects.

## Deviations

None. All changes were implemented as specified in the task plan.

## Files Created/Modified

- `src/cv_system/config.py` — Renamed three fields in CalibrationConfig and updated their @field_validator decorators
- `config/session.json` — Updated calibration section to use new field names (camera_corners, projector_corners, dmax_num_frames)
