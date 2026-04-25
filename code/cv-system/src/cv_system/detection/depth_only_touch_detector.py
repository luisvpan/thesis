"""
Depth-only touch detector using depth shell approach.

This detector identifies touches by finding objects within a thin depth "shell"
above the calibrated surface (dmax_map). Unlike TouchDetector which uses MediaPipe
for hand detection, this detector works purely with depth data and can detect
any object (finger, stylus, etc.) that enters the touch zone.
"""

import cv2
import numpy as np

from cv_system.config import DetectionConfig
from cv_system.transform import DepthCoordinateTransformer, ResolutionMapper


class DepthOnlyTouchDetector:
    """
    Detects touches using depth shell approach.

    Touch is detected when an object's depth falls within the range:
        dmin < depth < dmax
    where dmin = dmax - touch_shell_thickness.

    This approach:
    - Works without RGB/hand detection
    - Detects any object entering the touch zone
    - Uses temporal filtering to reduce sensor noise

    Coordinate flow:
        depth_frame (depth space)
            -> touch_mask creation (dmin < z < dmax)
            -> noise filtering (vibration, morphology)
            -> connected components analysis
            -> centroid extraction (depth space)
            -> ResolutionMapper.depth_to_rgb -> (x, y) in RGB/camera space
            -> DepthCoordinateTransformer.camera_to_projector -> projector space
    """

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_coordinate_transformer: DepthCoordinateTransformer,
        resolution_mapper: ResolutionMapper,
        config: DetectionConfig,
        depth_corners: list[tuple[int, int]],
        *,
        show_debug: bool = False,
    ) -> None:
        """
        Initialize the depth-only touch detector.

        Args:
            dmax_map: Calibrated maximum depth map (surface depth) in depth space.
            depth_coordinate_transformer: Transformer for camera <-> projector mapping.
            resolution_mapper: Mapper for RGB <-> depth resolution scaling.
            config: Detection configuration parameters.
            depth_corners: List of 4 (x, y) corners defining the calibrated area
                in depth space [top-left, top-right, bottom-left, bottom-right].
            show_debug: If True, display debug visualization windows.
        """
        # dmax_map already has surface_offset applied during calibration
        # dmin = dmax - touch_threshold (defines the shell thickness)
        # Touch zone: dmin < depth < dmax (strict inequalities)
        self._dmax_map = dmax_map.astype(np.int32)
        self._dmin_map = (self._dmax_map - config.touch_threshold).astype(np.int32)
        self._coordinate_transformer = depth_coordinate_transformer
        self._resolution_mapper = resolution_mapper
        self._show_debug = show_debug

        # Configuration
        self._vibration_threshold = config.vibration_threshold
        self._min_touch_size = config.min_touch_size
        self._max_touch_size = config.max_touch_size

        # Create area mask from depth_corners
        self._area_mask = np.zeros(dmax_map.shape, dtype=np.uint8)
        if depth_corners and len(depth_corners) == 4:
            # depth_corners: [top-left, top-right, bottom-left, bottom-right]
            # Reorder for cv2.fillPoly: TL -> TR -> BR -> BL (clockwise)
            polygon = np.array([
                depth_corners[0],  # top-left
                depth_corners[1],  # top-right
                depth_corners[3],  # bottom-right
                depth_corners[2],  # bottom-left
            ], dtype=np.int32)
            cv2.fillPoly(self._area_mask, [polygon], 255)
        else:
            # If no corners, use entire frame
            self._area_mask[:] = 255

        # Ring buffer for temporal median filtering
        self._ring_buffer_size = config.ring_buffer_size
        self._ring_buffer: list[np.ndarray] = []

        # Vibration filter state
        self._previous_depth: np.ndarray | None = None

        # Touch history for temporal persistence
        self._touch_history: list[np.ndarray] = []
        self._touch_history_size = config.touch_history_size

        # Temporal smoothing for touch position stability
        self._last_touch: tuple[float, float] | None = None
        self._smoothing_alpha = 0.7  # 0=all history, 1=only new position

        # Store defects for debug visualization
        self._last_defects: list[tuple[int, int]] = []


    def detect(
        self, depth_frame: np.ndarray
    ) -> tuple[list[tuple[float, float]], bool]:
        """
        Detect touches from depth frame.

        Args:
            depth_frame: Raw depth frame from HardwareManager (depth space, uint16).

        Returns:
            Tuple of (touches, objects_detected):
            - touches: List of (x, y) touch positions in projector coordinates.
            - objects_detected: True if any objects were detected in the touch zone.
        """
        # Convert to int32 for safe arithmetic
        depth_int = depth_frame.astype(np.int32)

        # 1. Vibration filter - restore pixels that change too abruptly
        if self._previous_depth is not None:
            roi_diff = cv2.absdiff(
                depth_int.astype(np.uint16), self._previous_depth.astype(np.uint16)
            )
            _, vibration_mask = cv2.threshold(
                roi_diff.astype(np.uint8), self._vibration_threshold, 255, cv2.THRESH_BINARY
            )
            vibration_mask = cv2.medianBlur(vibration_mask, ksize=5)
            depth_int[vibration_mask > 0] = self._previous_depth[vibration_mask > 0]
        self._previous_depth = depth_int.copy()

        # Use depth directly (temporal filtering was adding noise instead of removing it)
        depth_filtered = depth_int

        # 2. Create depth shell mask (dmin < z < dmax) - strict inequalities
        # Objects at dmax or beyond are excluded (surface/table)
        # Objects at dmin or closer are excluded (hovering too high)
        touch_mask = ((depth_filtered > self._dmin_map) & (depth_filtered < self._dmax_map))
        touch_mask = touch_mask.astype(np.uint8) * 255

        # Save RAW mask before filtering (for debug comparison)
        touch_mask_raw = touch_mask.copy()

        # 3. Noise filtering - simplified (Wilson calibration handles most noise)
        # Only medianBlur to clean up flying pixels at object boundaries
        touch_mask = cv2.medianBlur(touch_mask, ksize=5)

        # 4. Apply area mask to ignore pixels outside calibrated region
        touch_mask = touch_mask & self._area_mask

        # 5. Touch history - require persistence across multiple frames
        self._touch_history.append(touch_mask.copy())
        if len(self._touch_history) > self._touch_history_size:
            self._touch_history.pop(0)

        if len(self._touch_history) >= self._touch_history_size:
            # Sum all masks and threshold to require majority presence
            accumulated = np.sum(self._touch_history, axis=0)
            threshold = (self._touch_history_size - 1) * 255 // self._touch_history_size
            _, touch_mask = cv2.threshold(
                accumulated.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY
            )

        # 6. Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            touch_mask, connectivity=8
        )

        touches_projector: list[tuple[float, float]] = []

        # Debug: show all component areas
        if self._show_debug and num_labels > 1:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            print(f"[DepthOnly] Components: {num_labels-1}, areas: {sorted(areas, reverse=True)[:5]}, filter: {self._min_touch_size}-{self._max_touch_size}")

        # 6. Find the BEST component (valid size + highest depth variance)
        # This handles cases where noise forms one huge connected component
        best_component_idx = None
        best_variance = -1.0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Skip if outside size bounds
            if area < self._min_touch_size or area > self._max_touch_size:
                continue

            # Get depth variance for this component
            component_mask = labels == i
            ys, xs = np.where(component_mask)
            depths = depth_filtered[ys, xs]
            depth_variance = float(np.var(depths))

            # Track the component with highest variance (most likely real hand)
            if depth_variance > best_variance:
                best_variance = depth_variance
                best_component_idx = i

        if self._show_debug:
            valid_count = sum(1 for i in range(1, num_labels)
                            if self._min_touch_size <= stats[i, cv2.CC_STAT_AREA] <= self._max_touch_size)
            print(f"[DepthOnly] Valid components: {valid_count}, best variance: {best_variance:.1f}")

        # Store defect points for debug visualization
        self._last_defects: list[tuple[int, int]] = []

        # Process the best component - detect fingertips using convex hull + defects
        h, w = depth_filtered.shape
        if best_component_idx is not None and best_variance >= 3.0:
            component_mask = (labels == best_component_idx).astype(np.uint8) * 255

            # Find contour of the touch region
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours and len(contours[0]) >= 5:
                contour = contours[0]

                # Convex hull with indices (not points)
                hull_indices = cv2.convexHull(contour, returnPoints=False)

                # Find convexity defects (valleys between fingers)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)

                    fingertips: list[tuple[int, int]] = []

                    if defects is not None:
                        for defect in defects:
                            start_idx, end_idx, far_idx, distance = defect[0]

                            start = tuple(contour[start_idx][0])
                            end = tuple(contour[end_idx][0])
                            far = tuple(contour[far_idx][0])

                            # Store defect point for visualization
                            self._last_defects.append(far)

                            # Calculate angle at the far point (valley)
                            start_arr = np.array(start, dtype=np.float32)
                            end_arr = np.array(end, dtype=np.float32)
                            far_arr = np.array(far, dtype=np.float32)

                            a = np.linalg.norm(start_arr - far_arr)
                            b = np.linalg.norm(end_arr - far_arr)
                            c = np.linalg.norm(start_arr - end_arr)

                            # Law of cosines to get angle at far point
                            cos_angle = (a**2 + b**2 - c**2) / (2 * a * b + 1e-6)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)

                            # Angle <= 90° indicates finger gap
                            # distance > 5000 filters small noise (distance is scaled by 256)
                            if angle <= np.pi / 2 and distance > 5000:
                                # The START point of each valid defect is a fingertip
                                fx, fy = start

                                # Validate: must be close to surface
                                if 0 <= fy < h and 0 <= fx < w:
                                    dist_to_surface = int(self._dmax_map[fy, fx]) - int(depth_filtered[fy, fx])
                                    if dist_to_surface < 25:  # mm from surface
                                        fingertips.append((fx, fy))

                    # Remove duplicates (fingertips too close together)
                    unique_fingertips: list[tuple[int, int]] = []
                    for fp in fingertips:
                        is_duplicate = False
                        for ufp in unique_fingertips:
                            if np.linalg.norm(np.array(fp) - np.array(ufp)) < 20:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            unique_fingertips.append(fp)

                    # Transform to projector coordinates
                    for fx, fy in unique_fingertips[:5]:
                        proj_point = self._transform_to_projector(float(fx), float(fy))
                        touches_projector.append(proj_point)

        objects_detected = len(touches_projector) > 0

        # Debug visualization
        if self._show_debug:
            self._show_debug_windows(depth_frame, touch_mask_raw, touch_mask, touches_projector)

        return touches_projector, objects_detected

    def _transform_to_projector(
        self, depth_x: float, depth_y: float
    ) -> tuple[float, float]:
        """
        Transform a point from depth space to projector space.

        Args:
            depth_x: X coordinate in depth space.
            depth_y: Y coordinate in depth space.

        Returns:
            (x, y) coordinates in projector space.
        """
        # depth -> RGB (resolution scaling)
        rgb_points = self._resolution_mapper.depth_to_rgb(
            [(int(depth_x), int(depth_y))]
        )
        rgb_x, rgb_y = rgb_points[0]

        # RGB/camera -> projector (homography)
        camera_point = np.array([[rgb_x, rgb_y]], dtype=np.float32)
        proj_point = self._coordinate_transformer.camera_to_projector(camera_point)

        return (float(proj_point[0, 0]), float(proj_point[0, 1]))

    def _show_debug_windows(
        self,
        depth_frame: np.ndarray,
        touch_mask_raw: np.ndarray,
        touch_mask: np.ndarray,
        touches: list[tuple[float, float]],
    ) -> None:
        """Display debug visualization windows."""
        h, w = depth_frame.shape
        depth_int = depth_frame.astype(np.int32)

        # === Color-coded depth zones visualization ===
        # This shows WHERE objects are relative to the touch shell:
        # - RED: Objects ABOVE touch zone (closer than dmin) - hand hovering
        # - GREEN: Objects IN touch zone (dmin < z < dmax) - valid touch
        # - BLUE: Objects AT/BELOW surface (z >= dmax) - table surface
        # - BLACK: No depth data (z=0)
        zones_vis = np.zeros((h, w, 3), dtype=np.uint8)

        # Create masks for each zone
        valid_depth = depth_int > 0
        above_zone = valid_depth & (depth_int < self._dmin_map)  # Closer than dmin (hand above)
        in_zone = valid_depth & (depth_int >= self._dmin_map) & (depth_int < self._dmax_map)  # Touch zone
        at_surface = valid_depth & (depth_int >= self._dmax_map)  # At or below surface

        # Apply area mask
        above_zone = above_zone & (self._area_mask > 0)
        in_zone = in_zone & (self._area_mask > 0)
        at_surface = at_surface & (self._area_mask > 0)

        # Color code: BGR format
        zones_vis[above_zone] = (0, 0, 255)    # RED = above touch zone (hand hovering)
        zones_vis[in_zone] = (0, 255, 0)       # GREEN = in touch zone (valid touch)
        zones_vis[at_surface] = (255, 0, 0)    # BLUE = at surface (table)

        # Count pixels in each zone for debugging
        above_count = np.sum(above_zone)
        in_count = np.sum(in_zone)
        surface_count = np.sum(at_surface)

        # Add legend
        cv2.putText(zones_vis, f"RED=above({above_count}px)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(zones_vis, f"GREEN=touch({in_count}px)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(zones_vis, f"BLUE=surface({surface_count}px)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Show center depth info
        cy, cx = h // 2, w // 2
        current_z = int(depth_frame[cy, cx])
        dmax_z = int(self._dmax_map[cy, cx])
        dmin_z = int(self._dmin_map[cy, cx])
        shell_info = f"Center: z={current_z} shell=[{dmin_z}-{dmax_z}]"
        cv2.putText(zones_vis, shell_info, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw area mask contour
        area_contours, _ = cv2.findContours(self._area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(zones_vis, area_contours, -1, (255, 255, 0), 1)

        # Draw hand silhouette (white contour from touch_mask)
        if np.any(touch_mask > 0):
            hand_contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(zones_vis, hand_contours, -1, (255, 255, 255), 2)

        # Draw convexity defects (valleys between fingers) as magenta circles
        for dx, dy in self._last_defects:
            cv2.circle(zones_vis, (dx, dy), 5, (255, 0, 255), -1)  # Magenta filled

        # === Original depth visualization ===
        depth_vis = cv2.normalize(
            depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        cv2.putText(depth_vis, shell_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.drawContours(depth_vis, area_contours, -1, (255, 255, 0), 1)

        # Draw all fingertip touches on both visualizations
        for i, (proj_x, proj_y) in enumerate(touches):
            # Transform back to depth space for visualization
            rgb_point = self._coordinate_transformer.projector_to_camera(
                np.array([[proj_x, proj_y]], dtype=np.float32)
            )
            depth_points = self._resolution_mapper.rgb_to_depth(
                [(int(rgb_point[0, 0]), int(rgb_point[0, 1]))]
            )
            dx, dy = depth_points[0]

            # Yellow filled circle with white border for fingertips
            cv2.circle(zones_vis, (dx, dy), 8, (0, 255, 255), -1)  # Yellow filled
            cv2.circle(zones_vis, (dx, dy), 8, (255, 255, 255), 2)  # White border
            cv2.circle(depth_vis, (dx, dy), 8, (0, 255, 255), -1)  # Yellow filled

            # Show depth info at first touch point only (avoid clutter)
            if i == 0 and 0 <= dy < h and 0 <= dx < w:
                tz = int(depth_frame[dy, dx])
                tdmax = int(self._dmax_map[dy, dx])
                tdmin = int(self._dmin_map[dy, dx])
                info = f"z={tz} [{tdmin}-{tdmax}]"
                cv2.putText(depth_vis, info, (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Touch mask visualization (colorized)
        touch_mask_color = cv2.applyColorMap(touch_mask, cv2.COLORMAP_JET)

        cv2.namedWindow("DepthOnly - Depth Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("DepthOnly - Depth Frame", depth_vis)

        cv2.namedWindow("DepthOnly - Touch Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("DepthOnly - Touch Mask", touch_mask_color)

        cv2.namedWindow("DepthOnly - Depth Zones", cv2.WINDOW_NORMAL)
        cv2.imshow("DepthOnly - Depth Zones", zones_vis)

        # === RAW vs FILTERED comparison ===
        comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)

        # RAW (left) - red color
        raw_color = np.zeros((h, w, 3), dtype=np.uint8)
        raw_color[touch_mask_raw > 0] = (0, 0, 255)  # Red

        # FILTERED (right) - green color
        filtered_color = np.zeros((h, w, 3), dtype=np.uint8)
        filtered_color[touch_mask > 0] = (0, 255, 0)  # Green

        comparison[:, :w] = raw_color
        comparison[:, w + 10:] = filtered_color

        # Labels and pixel counts
        raw_count = np.sum(touch_mask_raw > 0)
        filtered_count = np.sum(touch_mask > 0)
        reduction = (1 - filtered_count / raw_count) * 100 if raw_count > 0 else 0

        cv2.putText(comparison, f"RAW ({raw_count}px)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(comparison, f"FILTERED ({filtered_count}px)", (w + 20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(comparison, f"Reduction: {reduction:.1f}%", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.namedWindow("DepthOnly - RAW vs FILTERED", cv2.WINDOW_NORMAL)
        cv2.imshow("DepthOnly - RAW vs FILTERED", comparison)

        cv2.waitKey(1)
