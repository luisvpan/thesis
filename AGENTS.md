# CV System

## Build & Run

```bash
# Install dependencies
uv sync

# Install dev dependencies (includes ruff, pytest)
uv sync --all-extras

# Run the CV system
uv run cv-system
```

### Environment Setup

Before first run, configure the required environment variables:

```bash
cp .env.example .env
# Edit .env:
#   OPENNI2_REDIST_PATH=/path/to/OpenNI2/Redist
#   LANGUAGE_RUNTIME_WS_URL=ws://localhost:3000/cv
```

## Validation

Run after every change to get immediate feedback:

```bash
# Lint
uv run ruff check src/

# Lint with auto-fix
uv run ruff check src/ --fix

# Format
uv run ruff format src/

# Format check (dry run, no changes)
uv run ruff format src/ --check

# Tests (all)
uv run pytest

# Tests (specific module)
uv run pytest tests/test_calibration.py

# Tests (verbose)
uv run pytest -v

# All validations in sequence
uv run ruff check src/ && uv run ruff format src/ --check && uv run pytest
```

## Operational Notes

### Project Structure

```
cv-system/
  src/cv_system/
    hardware/       # Layer 1: Kinect V2 integration (OpenNI2)
    calibration/    # Layer 2: Homography + dmax_map
    transform/      # Layer 3: Coordinate mapping service
    detection/      # Layer 4: Touch + pieces (future)
    bridge/         # WebSocket communication
    config.py       # Session configuration loader
    main.py         # Entry point
```

### Critical Commands

- Always run `uv run ruff check src/` before committing — catches unused imports, undefined names
- Run `uv run ruff format src/` to auto-format — no style debates
- Run `uv run pytest` after changes — catches regressions
- Use `uv run ruff check src/ --fix` for safe auto-fixes (unused imports, etc.)

### Common Pitfalls

- OpenNI2 is NOT a pip package — must be installed as system dependency
- If `import openni` fails, check `OPENNI2_REDIST_PATH` in `.env`
- The Kinect streams depth as (424, 512) not (512, 424) — watch reshape order
- `cv2.getPerspectiveTransform` expects `float32` arrays, not `int`
- `dmax_map` is per-session — never hardcode depth ranges

### Codebase Patterns

#### Coordinate Transformation

```python
# CORRECT: Use the homography matrix
import cv2
import numpy as np

point_camera = np.array([[[x, y]]], dtype=np.float32)
point_projector = cv2.perspectiveTransform(point_camera, H)

# WRONG: Do not use linear scale + translation
# x_proj = x_min + (x - x_cam_min) * sx  # <- insufficient DOF
```

#### Ring Buffer for Touch Detection

```python
# CORRECT: Fixed-size ring buffer
buffer = np.zeros((N, h, w), dtype=np.uint8)
idx = 0

# Each frame:
buffer[idx % N] = current_mask
idx += 1
accumulated = np.sum(buffer, axis=0)

# WRONG: Unbounded list append
# touch_history.append(mask)  # <- memory leak
```

#### Git & Commitment Rules
- Follow the [Conventional Commits 1.0.0](https://www.conventionalcommits.org) specification.
- Perform atomic commits. Each logical change must be committed separately before moving to the next task.
- Use lowercase for the description. Do not end the subject line with a period.
- Add 'Assisted by: Ralph (gsd-pi Agent)' to the commit message body in the footer section.
