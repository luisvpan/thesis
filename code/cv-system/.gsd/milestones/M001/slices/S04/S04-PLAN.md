# S04: Coordinate Transformer

**Goal:** Implement stateless service that wraps homography matrix H and exposes bidirectional camera ↔ projector coordinate transformations.

**Demo:** Given CalibrationResult with homography H, transformer converts camera point (100, 100) to projector space, then back to camera, verifying round-trip accuracy within 1px tolerance.

## Must-Haves
- CoordinateTransformer class that wraps homography matrix H (3x3, float64)
- `camera_to_projector(point)` converts camera coordinates to projector space
- `projector_to_camera(point)` converts projector coordinates to camera space
- Uses cv2.perspectiveTransform for all transformations
- Handles single points (1,1,2) and batch points (N,1,2)
- Stateless (no mutable internal state)
- Type hints for input/output shapes

## Tasks

- [ ] **T01: CoordinateTransformer class with bidirectional methods**
  Implement stateless wrapper around homography matrix with camera_to_projector and projector_to_camera methods.

- [ ] **T02: Batch processing and edge case handling**
  Add support for batch points, handle out-of-bounds coordinates, add validation.

- [ ] **T03: Tests and documentation**
  Write unit tests for single/batch transformations, round-trip accuracy, edge cases.

## Files Likely Touched
- `src/cv_system/transform/transformer.py` (new)
- `src/cv_system/transform/__init__.py` (new)
- `tests/test_transform.py` (new)
