# S05: Detection Layer + Integration

**Goal:** Implement touch detection using ring buffer + dmax_map comparison, and wire all layers together into a runnable session loop.

**Demo:** System initializes, calibrates, enters detection loop. When object/hand is placed on table, touch detector identifies touch points, converts to projector coordinates, and prints to console. Runs at 30fps without memory leaks.

## Must-Haves
- TouchDetector class with ring buffer (N frames) for depth difference accumulation
- Compares each depth frame against dmax_map, thresholded to produce touch mask
- Uses ring buffer to filter noise (requires N consecutive frames above threshold)
- Converts touch points from camera to projector coordinates
- Main entry point wires: config → hardware → calibrator → transformer → detector → loop
- No memory leaks (ring buffer is preallocated, no unbounded appends)
- Runs at target FPS (30) with reasonable latency

## Tasks

- [ ] **T01: TouchDetector with ring buffer**
  Implement ring buffer accumulation, depth comparison, thresholding, and touch point extraction.

- [ ] **T02: Coordinate integration**
  Wire CoordinateTransformer to convert touch points from camera to projector space.

- [ ] **T03: Main session orchestration**
  Wire all layers together: init hardware → calibrate → create transformer/detector → detection loop → shutdown.

- [ ] **T04: End-to-end tests**
  Test full pipeline with mock hardware (if available) or integration tests.

## Files Likely Touched
- `src/cv_system/detection/touch_detector.py` (new)
- `src/cv_system/detection/__init__.py` (new)
- `src/cv_system/main.py` (update with full session loop)
- `tests/test_detection.py` (new)
- `tests/test_integration.py` (new)
