---
id: T01
parent: S01
milestone: M001
provides:
  - Pydantic models for type-safe configuration (CameraConfig, CalibrationConfig, DetectionConfig, SessionConfig)
  - Field validators for all config parameters (positive values, range checks, coordinate count)
  - pyproject.toml updated with pydantic>=2.0.0 dependency
requires:
  - slice: S00
    provides: Initial project structure (pyproject.toml, src/cv_system/__init__.py)
affects: [S02, S03, S04, S05]
key_files:
  - src/cv_system/config.py
  - pyproject.toml
key_decisions:
  - Used Pydantic v2 @field_validator decorator for validation (not v1 @validator)
  - Coordinate convention: resolutions are (height, width) tuples matching inherited code
  - Default values taken from inherited code's magic numbers for compatibility
patterns_established:
  - All configuration externalized to SessionConfig model
  - Validation errors provide clear, actionable messages
drill_down_paths:
  - .gsd/milestones/M001/slices/S01/tasks/T01-PLAN.md
duration: 10min
verification_result: pass
completed_at: 2026-03-22T23:30:00Z
---

# T01: Pydantic models and config schema

**Type-safe configuration system with Pydantic v2 validators for all CV system parameters.**

## What Happened

Created a complete configuration system using Pydantic v2 that externalizes all magic numbers from the inherited monolithic code. Implemented four models:

1. **CameraConfig**: Depth and RGB stream parameters (resolutions, FPS) with validators for positive values and FPS ≤ 60
2. **CalibrationConfig**: dmax_map generation (num frames, depth range) and marker coordinates (4 corners each) with validators for range ordering
3. **DetectionConfig**: Ring buffer and touch detection parameters (buffer size, thresholds, area limits) with validators
4. **SessionConfig**: Root model nesting all three configs with default factories

All models use Pydantic v2 syntax (`@field_validator` decorator) and provide clear error messages for invalid input. Default values match the inherited code's magic numbers for compatibility.

## Deviations
None — followed T01-PLAN.md exactly.

## Files Created/Modified
- `src/cv_system/config.py` — 5534 bytes, 164 lines, 4 Pydantic models with comprehensive validators
- `pyproject.toml` — Added `pydantic>=2.0.0` to dependencies and `ruff`, `pytest` to `[project.optional-dependencies.dev]`
