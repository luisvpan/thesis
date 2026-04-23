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
        self._dmax_map = dmax_map.astype(np.int32)
        self._dmin_map = (dmax_map - config.touch_threshold).astype(np.int32)
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

        # Temporal smoothing for touch position stability
        self._last_touch: tuple[float, float] | None = None
        self._smoothing_alpha = 0.7  # 0=all history, 1=only new position

        # Position locking to prevent jumps between fingers
        self._max_jump_distance = 30.0  # Max allowed movement per frame (depth pixels)
        self._frames_without_touch = 0
        self._lock_reset_frames = 5  # Frames without touch to reset lock

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

        # 1. Temporal median filter using ring buffer
        self._ring_buffer.append(depth_int.copy())
        if len(self._ring_buffer) > self._ring_buffer_size:
            self._ring_buffer.pop(0)

        # Compute temporal median across buffer (reduces ToF noise significantly)
        if len(self._ring_buffer) >= 3:
            stacked = np.stack(self._ring_buffer, axis=0)
            depth_filtered = np.median(stacked, axis=0).astype(np.int32)
        else:
            depth_filtered = depth_int

        # 2. Create depth shell mask (dmin < z < dmax)
        # Note: Objects exactly at dmax are indistinguishable from the table surface
        touch_mask = ((depth_filtered > self._dmin_map) & (depth_filtered < self._dmax_map))
        touch_mask = touch_mask.astype(np.uint8) * 255

        # 3. Noise filtering pipeline - aggressive for ToF noise
        touch_mask = cv2.medianBlur(touch_mask, ksize=7)
        touch_mask = cv2.GaussianBlur(touch_mask, (9, 9), 0)
        _, touch_mask = cv2.threshold(touch_mask, 180, 255, cv2.THRESH_BINARY)

        # Erosion to break noise bridges, then dilate to restore hand size
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        # Erode to break thin bridges (less aggressive)
        touch_mask = cv2.erode(touch_mask, kernel_medium, iterations=1)

        # Dilate to restore solid areas (hand)
        touch_mask = cv2.dilate(touch_mask, kernel_medium, iterations=1)

        # Final opening to clean up remaining noise
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 4. Apply area mask to ignore pixels outside calibrated region
        touch_mask = touch_mask & self._area_mask

        # 5. Connected components analysis
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

        # Process the best component if found
        if best_component_idx is not None and best_variance >= 3.0:
            component_mask = labels == best_component_idx
            ys, xs = np.where(component_mask)
            depths = depth_filtered[ys, xs]
            area = stats[best_component_idx, cv2.CC_STAT_AREA]

            # Check compactness
            component_mask_uint8 = component_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            valid_component = True
            if contours:
                perimeter = cv2.arcLength(contours[0], closed=True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)
                    if compactness < 0.08:
                        valid_component = False
                        if self._show_debug:
                            print(f"[DepthOnly] Rejected: low compactness ({compactness:.3f})")

            if valid_component:
                # Find touch point: pixel closest to surface (dmax)
                dmax_values = self._dmax_map[ys, xs]
                distance_to_surface = dmax_values - depths

                min_idx = np.argmin(distance_to_surface)
                cx, cy = float(xs[min_idx]), float(ys[min_idx])

                # Position locking + smoothing to prevent jumps
                if self._last_touch is not None:
                    prev_x, prev_y = self._last_touch
                    distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)

                    if distance > self._max_jump_distance:
                        # Jump too large - keep previous position (lock)
                        cx, cy = prev_x, prev_y
                        if self._show_debug:
                            print(f"[DepthOnly] Jump rejected: {distance:.1f}px > {self._max_jump_distance}")
                    else:
                        # Valid movement - apply smoothing
                        cx = self._smoothing_alpha * cx + (1 - self._smoothing_alpha) * prev_x
                        cy = self._smoothing_alpha * cy + (1 - self._smoothing_alpha) * prev_y

                self._last_touch = (cx, cy)
                self._frames_without_touch = 0

                # Transform to projector coordinates
                proj_point = self._transform_to_projector(cx, cy)
                touches_projector.append(proj_point)

        # Reset lock if no touch for several frames
        if len(touches_projector) == 0:
            self._frames_without_touch += 1
            if self._frames_without_touch >= self._lock_reset_frames:
                self._last_touch = None

        objects_detected = len(touches_projector) > 0

        # Debug visualization
        if self._show_debug:
            self._show_debug_windows(depth_frame, touch_mask, touches_projector)

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
        contours, _ = cv2.findContours(self._area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(zones_vis, contours, -1, (255, 255, 0), 1)

        # === Original depth visualization ===
        depth_vis = cv2.normalize(
            depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        cv2.putText(depth_vis, shell_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.drawContours(depth_vis, contours, -1, (255, 255, 0), 1)

        # Draw touch centroids on both visualizations
        for proj_x, proj_y in touches:
            # Transform back to depth space for visualization
            rgb_point = self._coordinate_transformer.projector_to_camera(
                np.array([[proj_x, proj_y]], dtype=np.float32)
            )
            depth_points = self._resolution_mapper.rgb_to_depth(
                [(int(rgb_point[0, 0]), int(rgb_point[0, 1]))]
            )
            dx, dy = depth_points[0]
            cv2.circle(depth_vis, (dx, dy), 5, (0, 255, 0), -1)
            cv2.circle(zones_vis, (dx, dy), 5, (255, 255, 255), 2)

            # Show depth info at touch point
            if 0 <= dy < h and 0 <= dx < w:
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

        cv2.waitKey(1)
