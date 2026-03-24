# ADR-003: Homography-based calibration for camera-projector transformation

## Status

Accepted

## Date

2026-03-19 (updated 2026-03-23)

## Context

The system needs to map points between the Kinect V2 depth camera coordinate space (512x424) and the projector space (1920x1080). The inherited code uses a 2-point linear transformation (scale + translation, 4 degrees of freedom) that cannot correct rotation, skew, or perspective distortion. This produces an offset between what the camera sees and what the projector displays, particularly at the edges of the work area.

2D transformations are organized in a hierarchy of degrees of freedom (DOF):

- **Translation:** 2 DOF (displacement in X, Y). Requires 1 point.
- **Similarity:** 4 DOF (uniform scale + rotation + translation). Requires 2 points.
- **Affine:** 6 DOF (non-uniform scale + rotation + skew + translation). Requires 3 points. Preserves parallel lines.
- **Homography (projective):** 8 DOF (full projective transformation). Requires 4 points. Preserves straight lines but not parallelism.

The physical setup (Kinect and projector mounted at different heights and angles above the table) introduces perspective distortion that only a projective transformation can correct.

### References

- Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press. Ch. 2: Projective Geometry and Transformations of 2D.
- Torralba, A., Isola, P. & Freeman, W.T. (2024). *Foundations of Computer Vision*, MIT Press. Ch. 41: Homographies. Available at https://visionbook.mit.edu/homography.html
- OpenCV. *Basic concepts of the homography explained with code*. https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html
- Guo, Y., Chu, S., Liu, Z., Qiu, C., Luo, H. & Tan, J. (2018). "A real-time interactive system of surface reconstruction and dynamic projection mapping with RGB-depth sensor and projector." *International Journal of Distributed Sensor Networks*, 14(7).

## Decision

We replace the 2-point transformation with a **4-point homography** using `cv2.getPerspectiveTransform`.

### What is configured vs. what is computed

**`projector_corners` are configured.** These are the 4 positions (in projector pixel coordinates) where the calibration markers will be drawn. They define the usable work area on the projected surface. They are known a priori because we choose where to project them (e.g., the corners of the desired work rectangle within the projector's 1920x1080 viewport). These are the `dst_points` for `cv2.getPerspectiveTransform`.

**`camera_corners` are computed, never configured.** These are the 4 positions where the Kinect's camera actually sees the projected markers. They are detected automatically during the calibration process. Hardcoding them would defeat the purpose of calibration — the homography would be static and unable to adapt to the physical setup. These are the `src_points` for `cv2.getPerspectiveTransform`.

This is why `cv2.getPerspectiveTransform` (exact 4-point solution) is sufficient and RANSAC is unnecessary: we control the `dst_points` (projector) precisely, and the `src_points` (camera) are detected from high-contrast markers with no ambiguity.

### Calibration flow

1. **Marker Projector** creates a fullscreen black image at the projector's resolution and draws 4 high-contrast white squares at the configured `projector_corners`. This image is displayed via `cv2.namedWindow` + `cv2.setWindowProperty(WINDOW_FULLSCREEN)` on the projector's display. No additional libraries are needed — OpenCV handles windowing natively.

2. **Marker Detector** captures a frame from the Kinect's RGB stream (not depth) and detects the 4 white squares using threshold + contour detection. The high contrast (white on black) makes detection robust and simple. The detector extracts the centroids of the 4 detected markers — these are the `camera_corners` in RGB coordinates.

3. The RGB coordinates are mapped to the depth space using the Kinect V2's hardware registration (coordinate mapping between the color and depth sensors).

4. With the 4 correspondence pairs (`camera_corners` → `projector_corners`), the homography matrix H (3x3) is computed via `cv2.getPerspectiveTransform`.

5. Any subsequent point transformation is performed with `cv2.perspectiveTransform(point, H)`.

### RANSAC ruled out

RANSAC was ruled out because it is unnecessary in this scenario: the `projector_corners` are controlled precisely, and the `camera_corners` are detected from high-contrast markers with no competing features. With exactly 4 points, `cv2.getPerspectiveTransform` solves the system deterministically. If marker detection proves unreliable in the future, more than 4 points can be projected and `cv2.findHomography` with least squares (without RANSAC) can be used instead.

## Consequences

Calibration requires projecting and detecting 4 markers instead of 2, which implies a more robust detection pipeline (4 squares instead of 2). Detection is performed on the Kinect's RGB image, which requires the Hardware Manager to expose the RGB stream in addition to depth (at least during calibration). The marker projection uses OpenCV's native windowing (`cv2.namedWindow` + fullscreen) and requires no additional dependencies. The resulting transformation automatically corrects rotation, skew, perspective, and non-uniform scale, eliminating the offset that existed with the linear transformation.
