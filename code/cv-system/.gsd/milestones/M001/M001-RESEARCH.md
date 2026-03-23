# M001: CV Core Pipeline — Research

**Researched:** 2026-03-22
**Domain:** Computer Vision with Kinect V2, OpenCV, OpenNI2
**Confidence:** HIGH

## Summary

The CV system needs 4 layers: Hardware Manager (Kinect V2 via OpenNI2), Calibrator (4-point homography + dmax_map), Coordinate Transformer (stateless mapping), and Detection Layer (touch detection with ring buffer). The existing monolithic code in `../computer-vision-manager/ultimate_calibrate_area.py` (424 lines) shows working patterns but mixes concerns and has magic numbers.

**Primary recommendation:** Build the 4-layer architecture as specified in ADR-004, starting with a clean implementation that externalizes configuration. Use config file for calibration (4 corner pairs in projector and camera space), depth-based touch detection only (ring buffer + dmax), and skip WebSocket for this milestone.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Homography calculation | Manual matrix solving | `cv2.getPerspectiveTransform` | OpenCV is battle-tested, handles edge cases |
| Ring buffer | Python list with append | Preallocated NumPy array `(N, h, w)` | Fixed memory, no GC overhead, constant-time indexing |
| Depth mode calculation | Sort + loop | `np.argmax` on histogram | Vectorized, 100x faster |
| Coordinate mapping | Linear scale + translation | Full homography matrix | Corrects perspective distortion (ADR-003) |
| Stream reading | Manual buffer management | OpenNI2's `depth_stream.read_frame()` | Driver handles frame synchronization |
| Configuration | JSON parsing by hand | Pydantic models | Type validation, defaults, documentation |

## Common Pitfalls

### Pitfall 1: Mixing depth and RGB coordinate spaces
**What goes wrong:** Depth stream is (424, 512) but RGB is (1080, 1920). Using depth coordinates in RGB context causes crashes.
**Why it happens:** Kinect V2 has different resolutions for each sensor.
**How to avoid:** Always annotate which coordinate space a point/rect is in. Use explicit conversion functions with type hints.
**Warning signs:** "IndexError: index 512 is out of bounds for axis 0 with size 424"

### Pitfall 2: Unbounded memory growth in touch detection
**What goes wrong:** Accumulating all touch masks in a list causes O(N) memory usage and progressive slowdown.
**Why it happens:** Inherited code pattern: `touch_history.append(mask)`.
**How to avoid:** Preallocate ring buffer: `buffer = np.zeros((N, h, w), dtype=np.uint8)`. Use circular index: `buffer[idx % N] = current_mask`.
**Warning signs:** Memory usage growing linearly with session duration, frame rate dropping over time.

### Pitfall 3: Using int instead of float32 for homography
**What goes wrong:** `cv2.getPerspectiveTransform` returns float64 but code converts to int, losing precision.
**Why it happens:** Copy-paste from legacy code that used 2-point transform.
**How to avoid:** Always store H as `np.float64` or `np.float32`. Convert to float32 for `cv2.perspectiveTransform`.
**Warning signs:** "TypeError: Expected Ptr<cv::Mat> for argument 'M'"

### Pitfall 4: Hardcoding depth ranges
**What goes wrong:** `min_depth = 650, max_depth = 800` works in one setup but fails in others.
**Why it happens:** Different table heights and camera mounting angles.
**How to avoid:** Externalize to `SessionConfig`. Allow calibration to auto-discover valid depth range.
**Warning signs:** dmax_map has all zeros or all same value.

### Pitfall 5: Forgetting to flip frames
**What goes wrong:** Detected touch coordinates are mirrored left-to-right.
**Why it happens:** Kinect V2 depth stream is mirrored; RGB stream is not.
**How to avoid:** `cv2.flip(frame, 1)` for depth stream immediately after reading. Document this in code comments.
**Warning signs:** Touch appears on opposite side of table.

## Relevant Code

### Existing Implementation
- `../computer-vision-manager/ultimate_calibrate_area.py` — 424 lines, monolithic
  - Shows working patterns: OpenNI2 initialization, dmax_map calculation, depth frame processing
  - Contains anti-patterns: magic numbers, unbounded list append, mixed concerns
  - Uses PySimpleGUI for calibration UI (we're replacing with config file)

### Project Structure
```
cv-system/
  src/cv_system/
    __init__.py  — Currently just prints "Hello from cv-system!"
  pyproject.toml — Has openni dependency
  docs/adr/     — Architecture decisions (read ADR-003, ADR-004, ADR-005)
  .env.example   — OPENNI2_REDIST_PATH template
```

### Dependencies
- `openni>=2.3.0` — Kinect V2 driver bindings (NOT a pip package, system dependency)
- Need to add: `opencv-python`, `numpy`, `pydantic`, `pytest`
- Dev dependencies (in uv sync --all-extras): `ruff`, `pytest`

## Sources

### Internal Documentation
- ADR-001: Python as the system language — Confirmed, using Python 3.12
- ADR-002: uv as the package manager — Confirmed, using uv
- ADR-003: 4-point homography — Using cv2.getPerspectiveTransform, config file for markers
- ADR-004: 4-layer modular architecture — Confirmed, implement Hardware, Calibrator, Transform, Detection
- ADR-005: dmax_map session lifecycle — In-memory per session, optional persistence for dev
- ADR-008: Ruff linter — Use `uv run ruff check src/` and `uv run ruff format src/`

### External References
- OpenCV Homography Tutorial: https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html (HIGH confidence)
- Torralba et al. (2024) "Foundations of Computer Vision", Ch. 41: Homographies — Mathematical basis (MEDIUM confidence)
- OpenNI2 Documentation: https://structure.io/openni (LOW confidence, limited docs available)
- Khoshelham & Elberink (2012) "Accuracy and Resolution of Kinect Depth Data" — Depth noise analysis (MEDIUM confidence)

## Architecture Diagram Reference

See `docs/architecture/cv-system-architecture.likec4` for visual representation of layers and data flow. Key relationships:
- Hardware Manager → Calibrator (provides RGB + depth frames)
- Calibrator → CalibrationResult (produces H + dmax_map)
- CalibrationResult → Coordinate Transformer (provides H)
- CalibrationResult → Detection Layer (provides dmax_map)
- Detection Layer → Coordinate Transformer (converts camera → projector coordinates)

## Implementation Order

Based on dependencies and risk:
1. **Configuration** (no dependencies, low risk) — SessionConfig, dataclasses
2. **Hardware Manager** (no dependencies, medium risk) — OpenNI2 integration, depth + RGB streams
3. **Calibrator** (depends on Hardware Manager, high risk) — Homography + dmax_map generation
4. **Coordinate Transformer** (depends on CalibrationResult, low risk) — Stateless mapping service
5. **Detection Layer** (depends on Transformer + dmax_map, medium risk) — Touch detection with ring buffer
6. **Main Entry** (wires everything together, low risk) — Session lifecycle

## Open Questions (to be resolved during planning)

1. Ring buffer size N for touch detection? Suggested: 5-10 frames (empirically tuned)
2. Depth difference threshold for foreground? Suggested: 20-30mm (tunable via config)
3. Config file format? JSON, YAML, or TOML? Suggested: JSON (simple, Pydantic support)
4. Number of frames for dmax_map? Suggested: 500 (from existing code, ~17s at 30fps)
5. Depth range auto-discovery or config? Suggested: Config with fallback to auto-discovery
