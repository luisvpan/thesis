# T03: Example config and environment integration

**Slice:** S01
**Milestone:** M001

## Goal
Create example config file with realistic defaults, add environment variable for config path, and wire config loading into the main entry point.

## Must-Haves

### Truths
- Example config file can be loaded successfully by load_config()
- All config values match the inherited code's magic numbers
- CONFIG_PATH env var defaults to "config/session.json"
- Running `uv run cv-system` loads config and prints it (or uses it)

### Artifacts
- `config/session.json` — Example config with realistic values (validates against SessionConfig)
- `.env.example` — Contains CONFIG_PATH=config/session.json (new or updated)
- `src/cv_system/main.py` — Updated to load config and use it (or at least load it)

### Key Links
- T01 → provides SessionConfig model that validates the example config
- T02 → provides load_config() function used in main.py

## Steps
1. Create `config/session.json` with CameraConfig, CalibrationConfig, DetectionConfig sections
2. Use magic numbers from inherited code:
   - depth_resolution: [512, 424] or [424, 512] (check inherited convention)
   - rgb_resolution: [1080, 1920] or [1920, 1080]
   - fps: 30
   - num_dmax_frames: 500
   - depth_range_min: 650, depth_range_max: 800
   - ring_buffer_size: 5 (reasonable default)
   - touch_threshold: 20 (reasonable default)
   - marker_camera_coords: [[0, 0], [0, 424], [512, 0], [512, 424]] (example corners)
   - marker_projector_coords: [[0, 0], [0, 1080], [1920, 0], [1920, 1080]] (example corners)
3. Update `.env.example` to include CONFIG_PATH=config/session.json
4. Update `src/cv_system/main.py` to:
   - Import load_config, SessionConfig
   - Load config from env var or default path
   - Print loaded config (for demo)
5. Run `uv run ruff check src/` and `uv run ruff format src/` to verify linting
6. Test: `uv run cv-system` should load and print config

## Context
- Check inherited code for coordinate order convention: is it (width, height) or (height, width)?
- OpenCV uses (height, width) for array shapes but (x, y) for points
- The example config doesn't need perfect marker coords — those will be calibrated per session
- Make sure config directory exists or create it in main.py if needed
