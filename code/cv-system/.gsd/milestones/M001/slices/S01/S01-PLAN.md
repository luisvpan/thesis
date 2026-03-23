# S01: Configuration Foundation

**Goal:** Implement type-safe configuration loading with Pydantic models, externalizing all magic numbers from the inherited monolithic code.

**Demo:** User creates `config/session.json` with hardware, calibration, and detection parameters. Running the code validates the config and prints the loaded values. Invalid config raises clear Pydantic ValidationError.

## Must-Haves
- `SessionConfig` Pydantic model with CameraConfig, CalibrationConfig, DetectionConfig nested models
- Config file loads from `config/session.json` (or path provided via env var)
- All magic numbers from inherited code are now config parameters
- Validation errors provide clear, actionable messages
- Example config file with realistic default values

## Tasks

- [x] **T01: Pydantic models and config schema**
  Define Pydantic models for all configuration sections with proper types, defaults, and validation.

- [x] **T02: Config loader implementation**
  Implement load function that reads JSON file, validates with Pydantic, and returns SessionConfig.

- [x] **T03: Example config and environment integration**
  Create example config file, add OPENNI2_REDIST_PATH to .env, wire config loading into __init__.py.

## Files Likely Touched
- `src/cv_system/config.py` (new)
- `config/session.json` (new, example)
- `pyproject.toml` (add pydantic, pydantic-settings deps)
- `.env.example` (add CONFIG_PATH)
- `tests/test_config.py` (new)
