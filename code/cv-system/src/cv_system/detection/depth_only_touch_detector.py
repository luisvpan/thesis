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
from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch
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
            -> fingertip extraction: point closest to dmax (depth space)
            -> DepthCoordinateTransformer.camera_to_projector -> projector space
            (depth_H maps depth coordinates directly to projector coordinates)
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

        # Position locking to prevent jumps between fingers
        self._max_jump_distance = 30.0  # Max allowed movement per frame (depth pixels)
        self._frames_without_touch = 0
        self._lock_reset_frames = 5  # Frames without touch to reset lock

        # Touch tracker for persistent IDs and stateful events
        self._touch_tracker = TouchTracker(
            debounce_frames=3,
            touch_radius=15.0,
            lost_track_buffer=5,
        )

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        """
        Detect touches from depth frame.

        Args:
            depth_frame: Raw depth frame from HardwareManager (depth space, uint16).
            rgb_frame: RGB frame (unused, for API compatibility).
            ir_frame: IR frame (unused, for API compatibility).

        Returns:
            Tuple of (tracked_touches, objects_detected):
            - tracked_touches: List of TrackedTouch events (down/move/up).
            - objects_detected: True if any objects were detected in the touch zone.
        """
        # rgb_frame and ir_frame unused - DepthOnlyTouchDetector uses only depth
        # Convert UMat to numpy if needed
        if isinstance(depth_frame, cv2.UMat):
            depth_frame = depth_frame.get()

        # Convert to int32 for safe arithmetic
        depth_int = depth_frame.astype(np.int32)

        # Use depth directly (temporal filtering was adding noise)
        depth_filtered = depth_int

        # Create depth shell mask (dmin < z < dmax)
        touch_mask = ((depth_filtered > self._dmin_map) & (depth_filtered < self._dmax_map))
        touch_mask = touch_mask.astype(np.uint8) * 255

        # Save raw mask for debug visualization
        touch_mask_raw = touch_mask.copy()

        # Simplified noise filtering - just median blur
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
            valid_count = sum(1 for a in areas if self._min_touch_size <= a <= self._max_touch_size)
            print(f"[DepthOnly] Components: {num_labels-1}, valid: {valid_count}, areas: {sorted(areas, reverse=True)[:5]}")

        # Process ALL valid components - find the point closest to surface (fingertip)
        h, w = depth_filtered.shape
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Skip if outside size bounds
            if area < self._min_touch_size or area > self._max_touch_size:
                continue

            # Find point closest to dmax (fingertip) instead of centroid
            # This is more accurate per Wilson 2010 - the centroid is displaced
            # from the actual touch point, while the closest point to surface
            # represents the fingertip
            component_mask = (labels == i)

            # Distance to surface: dmax - depth (smaller = closer to surface)
            distance_to_surface = self._dmax_map - depth_filtered
            distance_to_surface = np.where(component_mask, distance_to_surface, np.iinfo(np.int32).max)

            # Find pixel with minimum distance (closest to surface = fingertip)
            min_idx = np.argmin(distance_to_surface)
            cy, cx = np.unravel_index(min_idx, distance_to_surface.shape)

            # Validate it's within bounds
            if 0 <= cy < h and 0 <= cx < w:
                # Transform to projector coordinates
                proj_point = self._transform_to_projector(float(cx), float(cy))
                touches_projector.append(proj_point)

        objects_detected = len(touches_projector) > 0

        # Debug visualization
        if self._show_debug:
            self._show_debug_windows(depth_frame, touch_mask_raw, touch_mask, touches_projector)

        # Track touches for persistent IDs and stateful events
        tracked_touches = self._touch_tracker.update(touches_projector)

        return tracked_touches, objects_detected

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
        # depth -> projector directly (depth_H maps depth coordinates to projector)
        depth_point = np.array([[depth_x, depth_y]], dtype=np.float32)
        proj_point = self._coordinate_transformer.camera_to_projector(depth_point)

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

        # Draw hand silhouette (white contour)
        if np.any(touch_mask > 0):
            hand_contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(zones_vis, hand_contours, -1, (255, 255, 255), 2)

        # === Original depth visualization ===
        depth_vis = cv2.normalize(
            depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        cv2.putText(depth_vis, shell_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.drawContours(depth_vis, area_contours, -1, (255, 255, 0), 1)

        # Draw fingertip touches (yellow with white border)
        for i, (proj_x, proj_y) in enumerate(touches):
            # Transform back to depth space for visualization
            # projector -> depth directly (depth_H_inv maps projector to depth)
            depth_point = self._coordinate_transformer.projector_to_camera(
                np.array([[proj_x, proj_y]], dtype=np.float32)
            )
            dx, dy = int(depth_point[0, 0]), int(depth_point[0, 1])

            # Yellow filled circle with white border
            cv2.circle(zones_vis, (dx, dy), 8, (0, 255, 255), -1)  # Yellow filled
            cv2.circle(zones_vis, (dx, dy), 8, (255, 255, 255), 2)  # White border
            cv2.circle(depth_vis, (dx, dy), 8, (0, 255, 255), -1)  # Yellow on depth too

            # Show depth info at first touch point only (avoid clutter)
            if i == 0 and 0 <= dy < h and 0 <= dx < w:
                tz = int(depth_frame[dy, dx])
                tdmax = int(self._dmax_map[dy, dx])
                tdmin = int(self._dmin_map[dy, dx])
                info = f"z={tz} [{tdmin}-{tdmax}]"
                cv2.putText(depth_vis, info, (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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
