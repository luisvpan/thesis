# T01: Pydantic models and config schema

**Slice:** S01
**Milestone:** M001

## Goal
Define Pydantic models for all configuration sections with proper types, defaults, and validation rules.

## Must-Haves

### Truths
- Pydantic models can be instantiated with valid data from the inherited code's magic numbers
- Invalid data (e.g., negative resolution, FPS > 60) raises ValidationError
- Nested models structure (SessionConfig → CameraConfig/CalibrationConfig/DetectionConfig) works

### Artifacts
- `src/cv_system/config.py` — Contains SessionConfig, CameraConfig, CalibrationConfig, DetectionConfig Pydantic models (min 100 lines)
- Each model has field types (int, tuple[int, int], str, etc.)
- Each model has validators (e.g., @field_validator for positive FPS, non-negative coordinates)

### Key Links
- T02 → imports SessionConfig from config.py
- T03 → imports all config models for loading

## Steps
1. Add `pydantic` to pyproject.toml dependencies
2. Create `src/cv_system/config.py`
3. Define CameraConfig with: depth_resolution (tuple), rgb_resolution (tuple), fps (int)
4. Define CalibrationConfig with: num_dmax_frames (int), depth_range_min (int), depth_range_max (int), marker_camera_coords (list of tuples), marker_projector_coords (list of tuples)
5. Define DetectionConfig with: ring_buffer_size (int), touch_threshold (int), min_touch_size (int), max_touch_size (int)
6. Define SessionConfig as root model with: camera (CameraConfig), calibration (CalibrationConfig), detection (DetectionConfig)
7. Add field validators: ensure fps > 0, resolutions have positive dims, depth_range_min < depth_range_max, ring_buffer_size > 0
8. Run `uv run ruff check src/` to verify linting

## Context
- Magic numbers from inherited code:
  - depth_camera_resolution = (512, 424)
  - color_camera_resolution = (1080, 1920)
  - depth_camera_fps = 30
  - num_frames for dmax = 500
  - min_depth = 650, max_depth = 800
- Use Pydantic v2 (check uv's available version)
- Use `@field_validator` decorator for validation (Pydantic v2 syntax)
- Use `tuple[int, int]` for resolution types (h, w) or (y, x) — check inherited code convention
