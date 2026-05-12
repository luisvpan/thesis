"""
Touch detector based on DIRECT paper (CMU, 2016).

"DIRECT: Making Touch Tracking on Ordinary Surfaces Practical with Hybrid
Depth-Infrared Sensing" by Robert Xiao et al.

This implementation follows the exact algorithm from the original source code:
https://github.com/nneonneo/direct-handtracking

Key insight: Uses hierarchical flood-fill (arm→hand→finger→tip) with different
edge constraints at each stage. IR edges constrain the tip fill when depth
becomes unreliable due to mixed pixels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from cv_system.detection.direct_background import DIRECTBackgroundModel
from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch
from cv_system.transform import DepthCoordinateTransformer


# Zone constants (matching original DIRECT code)
ZONE_ERROR = 0
ZONE_NOISE = 1
ZONE_LOW = 2
ZONE_MID = 3
ZONE_HIGH = 4


@dataclass
class FingerData:
    """Data for a detected finger."""
    tip_x: float
    tip_y: float
    touch_z: float  # z-score for touch detection
    pixels: List[int]  # pixel indices
    touched: bool = False
    touch_age: int = 0


class DIRECTTouchDetector:
    """
    Touch detector using DIRECT algorithm (depth + IR fusion).

    Pipeline:
        1. Classify pixels into zones based on distance from surface
        2. Build edge map (IR Canny + depth discontinuities)
        3. Hierarchical flood-fill: ARM → HAND → FINGER → TIP
        4. Compute fingertip as max-distance pixel from finger base
        5. Touch detection using z-score with hysteresis
    """

    # Parameters from original DIRECT code
    ARM_MIN_SIZE = 100  # px
    HAND_MIN_SIZE = 10  # px
    FINGER_MIN_SIZE = 10  # px
    FINGER_MIN_DIST = 5  # px
    TIP_MAX_DIST = 30  # px

    # Touch detection thresholds (diff in mm)
    TOUCHZ_ENTER = 10.0  # diff below = touching
    TOUCHZ_EXIT = 25.0  # diff above = not touching
    TOUCHZ_WINDOW = 8  # pixels to average for touch detection

    # Edge detection thresholds
    EDGE_DEPTHREL_THRESH = 50  # mm - smoothness
    EDGE_DEPTHABS_THRESH = 100  # mm - height

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_transformer: DepthCoordinateTransformer,
        config,
        *,
        show_debug: bool = False,
    ) -> None:
        self._depth_transformer = depth_transformer
        self._show_debug = show_debug
        self._touch_threshold = config.touch_threshold

        self._h, self._w = dmax_map.shape

        # Dynamic background model (starts from scratch)
        self._bg_model = DIRECTBackgroundModel((self._h, self._w))

        # Touch state tracking (for hysteresis)
        self._finger_states: Dict[int, bool] = {}  # finger_id -> touched
        self._next_finger_id = 0

        # Touch tracker for persistent IDs
        self._touch_tracker = TouchTracker(
            debounce_frames=2,
            touch_radius=25.0,
            lost_track_buffer=3,
        )

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        """
        Detect touches using DIRECT algorithm.

        Args:
            depth_frame: Raw depth frame (uint16, 424x512).
            rgb_frame: RGB frame (unused).
            ir_frame: IR frame (uint16, 424x512). Required for DIRECT.

        Returns:
            Tuple of (tracked_touches, hands_detected).
        """
        if ir_frame is None:
            # Fallback without IR
            return [], False

        # Update background model
        self._bg_model.update(depth_frame)

        # Check if model is ready for detection
        if not self._bg_model.is_ready:
            if self._show_debug:
                self._draw_calibration_progress(ir_frame)
            return [], False

        h, w = depth_frame.shape

        # Get background statistics
        bg_mean = self._bg_model.mean
        bg_stddev = self._bg_model.stddev

        # Step 1: Build diff and zone maps
        zones, diff_map, z_score_map = self._classify_zones(
            depth_frame, bg_mean, bg_stddev
        )

        # Step 2: Build edge maps
        ir_edges, depth_rel_edges, depth_abs_edges = self._build_edge_maps(
            ir_frame, diff_map
        )

        # Step 3: Detect touches via hierarchical flood-fill
        fingers = self._detect_touches(
            zones, diff_map, z_score_map, ir_edges, depth_rel_edges, depth_abs_edges
        )

        hands_detected = len(fingers) > 0

        # Step 4: Convert to projector coordinates
        touches_projector: list[tuple[float, float]] = []
        for finger in fingers:
            if finger.touched:
                point = np.array([[finger.tip_x, finger.tip_y]], dtype=np.float32)
                proj_point = self._depth_transformer.camera_to_projector(point)
                touches_projector.append(
                    (float(proj_point[0, 0]), float(proj_point[0, 1]))
                )
                # Debug: compare with Hybrid
                if self._show_debug:
                    print(f"[DIRECT] depth({finger.tip_x},{finger.tip_y}) -> proj({proj_point[0,0]:.0f},{proj_point[0,1]:.0f})")

        # Debug visualization
        if self._show_debug:
            self._draw_debug(
                depth_frame, ir_frame, zones, diff_map,
                ir_edges, depth_rel_edges, depth_abs_edges, fingers
            )
            self._draw_background_debug()

        # Track touches for persistent IDs
        tracked_touches = self._touch_tracker.update(touches_projector)

        return tracked_touches, hands_detected

    def _classify_zones(
        self,
        depth_frame: np.ndarray,
        bg_mean: np.ndarray,
        bg_stddev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify each pixel into a zone based on distance from surface.

        Zones (from original DIRECT):
            ZONE_ERROR: diff < -10mm (below surface, invalid)
            ZONE_NOISE: z < 0.7 (within noise, fingertip merged with surface)
            ZONE_LOW: diff < 12mm (finger close to surface)
            ZONE_MID: diff < 60mm (hand)
            ZONE_HIGH: diff >= 60mm (arm, far above surface)
        """
        # diff = how far above the surface (positive = above)
        diff_map = bg_mean.astype(np.int32) - depth_frame.astype(np.int32)

        # Handle invalid depth (0)
        diff_map[depth_frame == 0] = 0

        # z-score = diff / stddev (avoid division by zero)
        safe_stddev = np.maximum(bg_stddev, 1e-6)
        z_score_map = diff_map.astype(np.float32) / safe_stddev

        # Classify into zones
        # Note: Thresholds increased from original DIRECT (12/60mm) to adapt to our setup
        zones = np.full((self._h, self._w), ZONE_HIGH, dtype=np.uint8)
        zones[diff_map < 80] = ZONE_MID   # Original: 60mm
        zones[diff_map < 25] = ZONE_LOW   # Original: 12mm
        zones[z_score_map < 0.7] = ZONE_NOISE
        zones[diff_map < -10] = ZONE_ERROR
        zones[depth_frame == 0] = ZONE_ERROR

        return zones, diff_map, z_score_map

    def _build_edge_maps(
        self, ir_frame: np.ndarray, diff_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build edge maps from IR and depth data.

        Returns:
            ir_edges: Canny edges from IR image
            depth_rel_edges: Smoothness discontinuities (>50mm local diff)
            depth_abs_edges: Height discontinuities (near >100mm objects)
        """
        # IR Canny edges
        # Original uses ir/64 then Canny(4000,8000,7) on 16-bit
        # We convert to 8-bit: ir/16 gives 0-255 range, then Canny(60,120)
        ir_8bit = (np.clip(ir_frame, 0, 4000) / 16).astype(np.uint8)
        ir_edges = cv2.Canny(ir_8bit, 60, 120)

        # Fill small gaps in IR edges (simplified fillIrCannyHoles)
        kernel = np.ones((3, 3), np.uint8)
        ir_edges = cv2.dilate(ir_edges, kernel, iterations=1)
        ir_edges = cv2.erode(ir_edges, kernel, iterations=1)

        # Depth relative edges (smoothness check)
        # Pixels where local neighborhood has >50mm variation
        diff_float = diff_map.astype(np.float32)
        diff_smooth = cv2.GaussianBlur(diff_float, (5, 5), 0)
        depth_rel_edges = (np.abs(diff_float - diff_smooth) > self.EDGE_DEPTHREL_THRESH).astype(np.uint8) * 255

        # Depth absolute edges (near high objects)
        # Pixels that are low (<100mm) but adjacent to high (>100mm) pixels
        high_mask = (diff_map > self.EDGE_DEPTHABS_THRESH).astype(np.uint8)
        kernel7 = np.ones((7, 7), np.uint8)
        high_dilated = cv2.dilate(high_mask, kernel7)
        depth_abs_edges = ((high_dilated > 0) & (diff_map <= self.EDGE_DEPTHABS_THRESH)).astype(np.uint8) * 255

        return ir_edges, depth_rel_edges, depth_abs_edges

    def _detect_touches(
        self,
        zones: np.ndarray,
        diff_map: np.ndarray,
        z_score_map: np.ndarray,
        ir_edges: np.ndarray,
        depth_rel_edges: np.ndarray,
        depth_abs_edges: np.ndarray,
    ) -> List[FingerData]:
        """
        Detect fingers using hierarchical flood-fill.

        Pipeline: For each unvisited ZONE_HIGH pixel:
            ARM → HAND → FINGER → TIP → compute fingertip
        """
        h, w = self._h, self._w
        visited = np.zeros((h, w), dtype=np.uint8)
        fingers: List[FingerData] = []

        # Combined edge masks for each stage
        edge_hand = (ir_edges > 0) | (depth_rel_edges > 0)
        edge_finger = (ir_edges > 0) | (depth_abs_edges > 0)
        edge_tip = ir_edges > 0

        # Find all ARM starting points (ZONE_HIGH pixels)
        for y in range(h):
            for x in range(w):
                if zones[y, x] != ZONE_HIGH or visited[y, x]:
                    continue

                # Flood ARM
                arm_pixels, hand_seeds = self._flood_arm(
                    x, y, zones, visited
                )

                if len(arm_pixels) < self.ARM_MIN_SIZE:
                    continue

                # For each hand seed, flood HAND
                for hx, hy in hand_seeds:
                    if visited[hy, hx]:
                        continue

                    hand_pixels, finger_seeds = self._flood_hand(
                        hx, hy, zones, visited, edge_hand
                    )

                    if len(hand_pixels) < self.HAND_MIN_SIZE:
                        continue

                    # For each finger seed, flood FINGER
                    for fx, fy in finger_seeds:
                        if visited[fy, fx]:
                            continue

                        finger_result = self._flood_finger(
                            fx, fy, zones, visited, edge_finger, edge_tip,
                            diff_map, z_score_map
                        )

                        if finger_result is not None:
                            fingers.append(finger_result)

        return fingers

    def _flood_arm(
        self, start_x: int, start_y: int, zones: np.ndarray, visited: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Flood-fill ARM stage (ZONE_HIGH pixels).

        Returns:
            arm_pixels: List of (x, y) in the arm
            hand_seeds: List of (x, y) ZONE_MID neighbors to seed hand fill
        """
        h, w = self._h, self._w
        arm_pixels: List[Tuple[int, int]] = []
        hand_seeds: Set[Tuple[int, int]] = set()

        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = 1

        while queue:
            x, y = queue.popleft()
            arm_pixels.append((x, y))

            # 4-way connectivity
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    zone = zones[ny, nx]
                    if zone == ZONE_HIGH:
                        visited[ny, nx] = 1
                        queue.append((nx, ny))
                    elif zone == ZONE_MID:
                        hand_seeds.add((nx, ny))

        return arm_pixels, list(hand_seeds)

    def _flood_hand(
        self,
        start_x: int,
        start_y: int,
        zones: np.ndarray,
        visited: np.ndarray,
        edge_mask: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Flood-fill HAND stage (ZONE_MID pixels), respecting edges.

        Returns:
            hand_pixels: List of (x, y) in the hand
            finger_seeds: List of (x, y) ZONE_LOW neighbors to seed finger fill
        """
        h, w = self._h, self._w
        hand_pixels: List[Tuple[int, int]] = []
        finger_seeds: Set[Tuple[int, int]] = set()

        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = 1

        while queue:
            x, y = queue.popleft()
            hand_pixels.append((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    # Stop at edges
                    if edge_mask[ny, nx]:
                        continue

                    zone = zones[ny, nx]
                    if zone >= ZONE_MID:  # MID or HIGH
                        visited[ny, nx] = 1
                        queue.append((nx, ny))
                    elif zone == ZONE_LOW:
                        finger_seeds.add((nx, ny))

        return hand_pixels, list(finger_seeds)

    def _flood_finger(
        self,
        start_x: int,
        start_y: int,
        zones: np.ndarray,
        visited: np.ndarray,
        edge_finger: np.ndarray,
        edge_tip: np.ndarray,
        diff_map: np.ndarray,
        z_score_map: np.ndarray,
    ) -> Optional[FingerData]:
        """
        Flood-fill FINGER + TIP stages with distance tracking.

        Returns:
            FingerData if valid finger detected, None otherwise.
        """
        h, w = self._h, self._w

        # Track distance from roots (pixels adjacent to MID/HIGH)
        distance = np.full((h, w), -1, dtype=np.int32)
        finger_pixels: List[int] = []  # Linear indices
        tip_seeds: List[Tuple[int, int, int]] = []  # (x, y, dist)
        roots: List[Tuple[int, int]] = []

        # Flood FINGER (ZONE_LOW)
        queue = deque([(start_x, start_y, 0)])
        visited[start_y, start_x] = 1
        distance[start_y, start_x] = 0

        while queue:
            x, y, dist = queue.popleft()
            idx = y * w + x
            finger_pixels.append(idx)

            # Check if this is a root (adjacent to MID/HIGH)
            is_root = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if zones[ny, nx] >= ZONE_MID:
                        is_root = True
                        break
            if is_root:
                roots.append((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    # Stop at finger edges
                    if edge_finger[ny, nx]:
                        continue

                    zone = zones[ny, nx]
                    if zone == ZONE_LOW:  # Only ZONE_LOW (not MID or HIGH)
                        visited[ny, nx] = 1
                        distance[ny, nx] = dist + 1
                        queue.append((nx, ny, dist + 1))
                    elif zone == ZONE_NOISE:
                        tip_seeds.append((nx, ny, dist + 1))

        # Flood TIP (ZONE_NOISE) - only IR edges constrain
        tip_pixels: List[int] = []
        for sx, sy, sdist in tip_seeds:
            if visited[sy, sx]:
                continue

            tip_queue = deque([(sx, sy, sdist)])
            visited[sy, sx] = 1
            distance[sy, sx] = sdist
            overfill = False

            local_tip_pixels: List[int] = []

            while tip_queue:
                x, y, dist = tip_queue.popleft()

                if dist > self.TIP_MAX_DIST:
                    overfill = True
                    break

                idx = y * w + x
                local_tip_pixels.append(idx)

                # Check root
                is_root = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if zones[ny, nx] >= ZONE_MID:
                            is_root = True
                            break
                if is_root:
                    roots.append((x, y))

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        # Only IR edges stop tip fill
                        if edge_tip[ny, nx]:
                            continue

                        visited[ny, nx] = 1
                        distance[ny, nx] = dist + 1
                        tip_queue.append((nx, ny, dist + 1))

            if not overfill:
                tip_pixels.extend(local_tip_pixels)
                finger_pixels.extend(local_tip_pixels)

        # Check minimum size
        if len(finger_pixels) < self.FINGER_MIN_SIZE:
            return None

        # Reflood from roots to get correct distances
        if roots:
            distance.fill(-1)
            reflood_queue = deque()
            for rx, ry in roots:
                distance[ry, rx] = 0
                reflood_queue.append((rx, ry, 0))

            pixel_set = set(finger_pixels)
            while reflood_queue:
                x, y, dist = reflood_queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    nidx = ny * w + nx
                    if nidx in pixel_set and distance[ny, nx] < 0:
                        distance[ny, nx] = dist + 1
                        reflood_queue.append((nx, ny, dist + 1))

        # Compute finger metrics (use diff_map in mm for touch detection)
        return self._compute_finger_metrics(
            finger_pixels, distance, diff_map, w
        )

    def _compute_finger_metrics(
        self,
        pixels: List[int],
        distance: np.ndarray,
        diff_map: np.ndarray,
        w: int,
    ) -> Optional[FingerData]:
        """
        Compute fingertip position and touch metric.

        Fingertip = pixel with maximum distance from roots.
        Touch z = average diff (mm) of top 8 distance pixels.

        Note: Original DIRECT uses diff in mm, not z-score.
        """
        if not pixels:
            return None

        # Get distances for each pixel
        pixel_dists = []
        for idx in pixels:
            y, x = idx // w, idx % w
            d = distance[y, x]
            if d >= 0:
                pixel_dists.append((idx, d))

        if not pixel_dists:
            return None

        # Sort by distance
        pixel_dists.sort(key=lambda p: p[1])

        # Check minimum distance
        max_dist = pixel_dists[-1][1]
        if max_dist < self.FINGER_MIN_DIST:
            return None

        # Fingertip = max distance pixel
        tip_idx = pixel_dists[-1][0]
        tip_y, tip_x = tip_idx // w, tip_idx % w

        # Touch metric = average diff (mm) of top TOUCHZ_WINDOW pixels
        # Original DIRECT uses diff in mm, not z-score
        start = max(0, len(pixel_dists) - self.TOUCHZ_WINDOW)
        top_pixels = pixel_dists[start:]

        total_diff = 0.0
        count = 0
        for idx, _ in top_pixels:
            y, x = idx // w, idx % w
            total_diff += diff_map[y, x]
            count += 1

        touch_z = total_diff / count if count > 0 else 0.0

        # Determine touch state with hysteresis
        finger_id = self._next_finger_id
        self._next_finger_id += 1

        prev_touched = self._finger_states.get(finger_id, False)
        if prev_touched and touch_z > self.TOUCHZ_EXIT:
            touched = False
        elif not prev_touched and touch_z < self.TOUCHZ_ENTER:
            touched = True
        else:
            touched = prev_touched

        self._finger_states[finger_id] = touched

        return FingerData(
            tip_x=float(tip_x),
            tip_y=float(tip_y),
            touch_z=touch_z,
            pixels=pixels,
            touched=touched,
        )

    def _draw_debug(
        self,
        depth_frame: np.ndarray,
        ir_frame: np.ndarray,
        zones: np.ndarray,
        diff_map: np.ndarray,
        ir_edges: np.ndarray,
        depth_rel_edges: np.ndarray,
        depth_abs_edges: np.ndarray,
        fingers: List[FingerData],
    ) -> None:
        """Draw debug visualization windows."""
        h, w = self._h, self._w

        # Zone visualization
        zone_colors = {
            ZONE_ERROR: (0, 0, 128),    # Dark red
            ZONE_NOISE: (128, 128, 0),  # Cyan-ish
            ZONE_LOW: (0, 255, 255),    # Yellow
            ZONE_MID: (255, 128, 0),    # Blue-ish
            ZONE_HIGH: (255, 0, 0),     # Blue
        }
        debug_zones = np.zeros((h, w, 3), dtype=np.uint8)
        for zone_val, color in zone_colors.items():
            debug_zones[zones == zone_val] = color

        # Edge visualization
        debug_edges = np.zeros((h, w, 3), dtype=np.uint8)
        debug_edges[:, :, 2] = ir_edges  # Red = IR
        debug_edges[:, :, 1] = depth_rel_edges  # Green = depth rel
        debug_edges[:, :, 0] = depth_abs_edges  # Blue = depth abs

        # Depth with fingertips
        debug_depth = cv2.normalize(
            depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        debug_depth = cv2.cvtColor(debug_depth, cv2.COLOR_GRAY2BGR)

        # Draw fingers
        for finger in fingers:
            color = (0, 255, 0) if finger.touched else (0, 255, 255)
            cv2.circle(
                debug_depth,
                (int(finger.tip_x), int(finger.tip_y)),
                8 if finger.touched else 6,
                color,
                -1 if finger.touched else 2,
            )
            # Show touch_z
            cv2.putText(
                debug_depth,
                f"z:{finger.touch_z:.1f}",
                (int(finger.tip_x) + 10, int(finger.tip_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Info
        touch_count = sum(1 for f in fingers if f.touched)
        cv2.putText(
            debug_depth,
            f"Touches: {touch_count} Fingers: {len(fingers)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.namedWindow("DIRECT - Zones", cv2.WINDOW_NORMAL)
        cv2.imshow("DIRECT - Zones", debug_zones)
        cv2.namedWindow("DIRECT - Edges", cv2.WINDOW_NORMAL)
        cv2.imshow("DIRECT - Edges", debug_edges)
        cv2.namedWindow("DIRECT - Depth", cv2.WINDOW_NORMAL)
        cv2.imshow("DIRECT - Depth", debug_depth)
        cv2.waitKey(1)

    def _draw_calibration_progress(self, ir_frame: np.ndarray) -> None:
        """Show calibration progress while model initializes."""
        h, w = self._h, self._w
        stable_pct = self._bg_model.stable_percentage

        # Create progress image
        progress_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Show IR frame dimmed as background
        ir_8bit = (np.clip(ir_frame, 0, 4000) / 16).astype(np.uint8)
        progress_img[:, :, 0] = ir_8bit // 4
        progress_img[:, :, 1] = ir_8bit // 4
        progress_img[:, :, 2] = ir_8bit // 4

        # Draw progress text
        cv2.putText(
            progress_img,
            f"Calibrating: {stable_pct:.0f}%",
            (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            progress_img,
            "Keep surface clear",
            (w // 2 - 100, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # Draw progress bar
        bar_x, bar_y = w // 4, h // 2 + 70
        bar_w, bar_h = w // 2, 20
        cv2.rectangle(progress_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        filled_w = int(bar_w * stable_pct / 100)
        cv2.rectangle(progress_img, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        cv2.namedWindow("DIRECT - Depth", cv2.WINDOW_NORMAL)
        cv2.imshow("DIRECT - Depth", progress_img)
        cv2.waitKey(1)

    def _draw_background_debug(self) -> None:
        """Draw background model state visualization."""
        h, w = self._h, self._w
        debug_bg = np.zeros((h, w, 3), dtype=np.uint8)

        # R = stability (255=stable, 0=unstable)
        debug_bg[:, :, 2] = (self._bg_model.stable_mask * 255).astype(np.uint8)

        # G = mean (normalized to 0-255 for typical desk distance 500-1500mm)
        mean_norm = np.clip((self._bg_model.mean - 500) / 1000 * 255, 0, 255)
        debug_bg[:, :, 1] = mean_norm.astype(np.uint8)

        # B = stddev * 50 (typical 2-5mm -> 100-250)
        stddev_vis = np.clip(self._bg_model.stddev * 50, 0, 255)
        debug_bg[:, :, 0] = stddev_vis.astype(np.uint8)

        # Add stability percentage text
        stable_pct = self._bg_model.stable_percentage
        cv2.putText(
            debug_bg,
            f"Stable: {stable_pct:.0f}%",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.namedWindow("DIRECT - Background", cv2.WINDOW_NORMAL)
        cv2.imshow("DIRECT - Background", debug_bg)

    def close(self) -> None:
        """Release resources."""
        pass
