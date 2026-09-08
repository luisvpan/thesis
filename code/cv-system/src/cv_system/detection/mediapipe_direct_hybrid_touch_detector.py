"""
Hybrid touch detector: DIRECT detection + MediaPipe positioning.

- DIRECT: Does ALL the touch detection (flood-fill, zones, hysteresis)
- MediaPipe: Provides precise fingertip position when DIRECT detects a touch

This gives the best of both worlds:
- DIRECT's robust touch detection (knows WHEN you're touching)
- MediaPipe's precise finger localization (knows WHERE the finger is)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cv_system.detection.direct_background import DIRECTBackgroundModel
from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch
from cv_system.transform import DepthCoordinateTransformer, ResolutionMapper


# Zone constants (from DIRECT)
ZONE_ERROR = 0
ZONE_NOISE = 1
ZONE_LOW = 2
ZONE_MID = 3
ZONE_HIGH = 4


@dataclass
class FingerData:
    """Data for a detected finger (from DIRECT algorithm)."""
    tip_x: float
    tip_y: float
    touch_z: float
    pixels: List[int]
    touched: bool = False


class MediapipeDIRECTHybridTouchDetector:
    """
    Hybrid detector: DIRECT for touch detection, MediaPipe for position.

    Pipeline:
        1. DIRECT detects fingers via hierarchical flood-fill
        2. MediaPipe detects hands in RGB
        3. For each DIRECT touch, find nearest MediaPipe fingertip
        4. Report MediaPipe position (or fallback to DIRECT)
    """

    # MediaPipe landmark indices
    INDEX_FINGER_TIP = 8

    # DIRECT parameters
    ARM_MIN_SIZE = 100
    HAND_MIN_SIZE = 10
    FINGER_MIN_SIZE = 10
    FINGER_MIN_DIST = 5
    TIP_MAX_DIST = 30
    TOUCHZ_ENTER = 10.0
    TOUCHZ_EXIT = 25.0
    TOUCHZ_WINDOW = 8
    EDGE_DEPTHREL_THRESH = 50
    EDGE_DEPTHABS_THRESH = 100

    # Maximum distance to match DIRECT finger with MediaPipe (in depth pixels)
    MAX_MATCH_DISTANCE = 50

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_coordinate_transformer: DepthCoordinateTransformer,
        resolution_mapper: ResolutionMapper,
        config,
        *,
        rgb_H: np.ndarray | None = None,
        show_debug: bool = False,
    ) -> None:
        self._depth_transformer = depth_coordinate_transformer
        self._resolution_mapper = resolution_mapper
        self._config = config
        self._show_debug = show_debug
        self._h, self._w = dmax_map.shape

        # RGB → Projector homography (for MediaPipe points)
        self._rgb_H = rgb_H

        # DIRECT background model
        self._bg_model = DIRECTBackgroundModel((self._h, self._w))

        # MediaPipe HandLandmarker
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_detector = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()

        # Touch tracker
        self._touch_tracker = TouchTracker(
            debounce_frames=2,
            touch_radius=25.0,
            lost_track_buffer=3,
        )

        # Touch state for hysteresis
        self._finger_states: dict[int, bool] = {}
        self._next_finger_id = 0

        # MediaPipe state
        self._last_mp_image: mp.Image | None = None
        self._last_rgb_frame: np.ndarray | None = None
        self._last_hands: list[list[tuple[float, float, float]]] = []
        self._last_fingers: List[FingerData] = []

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        """Detect touches using DIRECT + MediaPipe hybrid approach."""
        # Handle UMat
        if isinstance(depth_frame, cv2.UMat):
            depth_frame = depth_frame.get()
        if rgb_frame is not None and isinstance(rgb_frame, cv2.UMat):
            rgb_frame = rgb_frame.get()
        if ir_frame is not None and isinstance(ir_frame, cv2.UMat):
            ir_frame = ir_frame.get()

        # IR required for DIRECT
        if ir_frame is None:
            return [], False

        # Update background
        self._bg_model.update(depth_frame)
        if not self._bg_model.is_ready:
            if self._show_debug:
                self._draw_calibration_progress(ir_frame)
            return [], False

        # 1. DIRECT detection (full algorithm)
        bg_mean = self._bg_model.mean
        bg_stddev = self._bg_model.stddev

        zones, diff_map, z_score_map = self._classify_zones(depth_frame, bg_mean, bg_stddev)
        ir_edges, depth_rel_edges, depth_abs_edges = self._build_edge_maps(ir_frame, diff_map)
        fingers = self._detect_fingers(zones, diff_map, z_score_map, ir_edges, depth_rel_edges, depth_abs_edges)

        self._last_fingers = fingers

        # 2. MediaPipe detection
        hands = []
        if rgb_frame is not None:
            hands = self._detect_hands(rgb_frame)
            self._last_rgb_frame = rgb_frame
            self._last_hands = hands

        # 3. Match DIRECT touches with MediaPipe positions
        touches_projector: list[tuple[float, float]] = []
        for finger in fingers:
            if not finger.touched:
                continue

            # Try to find matching MediaPipe fingertip (in RGB space)
            mp_rgb = self._find_nearest_mediapipe_rgb(
                finger.tip_x, finger.tip_y, hands, rgb_frame.shape if rgb_frame is not None else None
            )

            if mp_rgb is not None and self._rgb_H is not None:
                # Use MediaPipe RGB position → transform directly via rgb_H
                rgb_x, rgb_y = mp_rgb
                rgb_point = np.array([[[rgb_x, rgb_y]]], dtype=np.float32)
                proj_point = cv2.perspectiveTransform(rgb_point, self._rgb_H)
                proj_x, proj_y = float(proj_point[0, 0, 0]), float(proj_point[0, 0, 1])
                used_mediapipe = True
            else:
                # Fallback: use DIRECT position → transform via depth homography
                depth_point = np.array([[finger.tip_x, finger.tip_y]], dtype=np.float32)
                proj_point = self._depth_transformer.camera_to_projector(depth_point)
                proj_x, proj_y = float(proj_point[0, 0]), float(proj_point[0, 1])
                used_mediapipe = False

            touches_projector.append((proj_x, proj_y))

            # Debug: show coordinate flow
            if self._show_debug:
                if used_mediapipe:
                    print(f"[Hybrid] DIRECT({finger.tip_x:.0f},{finger.tip_y:.0f}) | MP_RGB({mp_rgb[0]:.0f},{mp_rgb[1]:.0f}) --rgb_H--> proj({proj_x:.0f},{proj_y:.0f})")
                else:
                    print(f"[Hybrid] DIRECT({finger.tip_x:.0f},{finger.tip_y:.0f}) --depth_H--> proj({proj_x:.0f},{proj_y:.0f}) [no MP match]")

        # Debug
        if self._show_debug:
            self._draw_debug(depth_frame, ir_frame, zones, diff_map, fingers, hands)

        # Track
        tracked = self._touch_tracker.update(touches_projector)
        return tracked, len(fingers) > 0

    def _detect_hands(self, rgb_frame: np.ndarray) -> list[list[tuple[float, float, float]]]:
        """Detect hands using MediaPipe."""
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        self._last_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._mp_detector.detect_for_video(self._last_mp_image, timestamp_ms)

        hands = []
        for hand_landmarks in result.hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            hands.append(landmarks)
        return hands

    def _find_nearest_mediapipe_rgb(
        self,
        direct_x: float,
        direct_y: float,
        hands: list[list[tuple[float, float, float]]],
        rgb_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """Find nearest MediaPipe fingertip to DIRECT's position, return RGB coords.

        Returns the RGB pixel coordinates of the nearest MediaPipe INDEX_FINGER_TIP.
        Matching is done in depth space (comparing with DIRECT's depth position).
        """
        if not hands or rgb_shape is None:
            return None

        rgb_h, rgb_w = rgb_shape[:2]
        best_rgb = None
        best_dist = float("inf")

        for hand in hands:
            tip = hand[self.INDEX_FINGER_TIP]
            rgb_x = int(tip[0] * rgb_w)
            rgb_y = int(tip[1] * rgb_h)

            # Map to depth coordinates for distance calculation
            depth_points = self._resolution_mapper.rgb_to_depth([(rgb_x, rgb_y)])
            if not depth_points:
                continue
            depth_x, depth_y = depth_points[0]
            if depth_x < 0 or depth_y < 0:
                continue

            # Calculate distance in depth space
            dist = np.sqrt((depth_x - direct_x) ** 2 + (depth_y - direct_y) ** 2)
            if dist < best_dist and dist < self.MAX_MATCH_DISTANCE:
                best_dist = dist
                best_rgb = (rgb_x, rgb_y)  # Return RGB coords, not depth

        return best_rgb

    # ===== DIRECT Algorithm (copied from direct_touch_detector.py) =====

    def _classify_zones(
        self, depth_frame: np.ndarray, bg_mean: np.ndarray, bg_stddev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify pixels into zones based on distance from surface."""
        diff_map = bg_mean.astype(np.int32) - depth_frame.astype(np.int32)
        diff_map[depth_frame == 0] = 0

        safe_stddev = np.maximum(bg_stddev, 1e-6)
        z_score_map = diff_map.astype(np.float32) / safe_stddev

        zones = np.full((self._h, self._w), ZONE_HIGH, dtype=np.uint8)
        zones[diff_map < 80] = ZONE_MID
        zones[diff_map < 25] = ZONE_LOW
        zones[z_score_map < 0.7] = ZONE_NOISE
        zones[diff_map < -10] = ZONE_ERROR
        zones[depth_frame == 0] = ZONE_ERROR

        return zones, diff_map, z_score_map

    def _build_edge_maps(
        self, ir_frame: np.ndarray, diff_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build edge maps from IR and depth data."""
        ir_8bit = (np.clip(ir_frame, 0, 4000) / 16).astype(np.uint8)
        ir_edges = cv2.Canny(ir_8bit, 60, 120)

        kernel = np.ones((3, 3), np.uint8)
        ir_edges = cv2.dilate(ir_edges, kernel, iterations=1)
        ir_edges = cv2.erode(ir_edges, kernel, iterations=1)

        diff_float = diff_map.astype(np.float32)
        diff_smooth = cv2.GaussianBlur(diff_float, (5, 5), 0)
        depth_rel_edges = (np.abs(diff_float - diff_smooth) > self.EDGE_DEPTHREL_THRESH).astype(np.uint8) * 255

        high_mask = (diff_map > self.EDGE_DEPTHABS_THRESH).astype(np.uint8)
        kernel7 = np.ones((7, 7), np.uint8)
        high_dilated = cv2.dilate(high_mask, kernel7)
        depth_abs_edges = ((high_dilated > 0) & (diff_map <= self.EDGE_DEPTHABS_THRESH)).astype(np.uint8) * 255

        return ir_edges, depth_rel_edges, depth_abs_edges

    def _detect_fingers(
        self,
        zones: np.ndarray,
        diff_map: np.ndarray,
        z_score_map: np.ndarray,
        ir_edges: np.ndarray,
        depth_rel_edges: np.ndarray,
        depth_abs_edges: np.ndarray,
    ) -> List[FingerData]:
        """Detect fingers using hierarchical flood-fill."""
        h, w = self._h, self._w
        visited = np.zeros((h, w), dtype=np.uint8)
        fingers: List[FingerData] = []

        edge_hand = (ir_edges > 0) | (depth_rel_edges > 0)
        edge_finger = (ir_edges > 0) | (depth_abs_edges > 0)
        edge_tip = ir_edges > 0

        for y in range(h):
            for x in range(w):
                if zones[y, x] != ZONE_HIGH or visited[y, x]:
                    continue

                arm_pixels, hand_seeds = self._flood_arm(x, y, zones, visited)
                if len(arm_pixels) < self.ARM_MIN_SIZE:
                    continue

                for hx, hy in hand_seeds:
                    if visited[hy, hx]:
                        continue

                    hand_pixels, finger_seeds = self._flood_hand(hx, hy, zones, visited, edge_hand)
                    if len(hand_pixels) < self.HAND_MIN_SIZE:
                        continue

                    for fx, fy in finger_seeds:
                        if visited[fy, fx]:
                            continue

                        finger = self._flood_finger(
                            fx, fy, zones, visited, edge_finger, edge_tip, diff_map, z_score_map
                        )
                        if finger is not None:
                            fingers.append(finger)

        return fingers

    def _flood_arm(
        self, start_x: int, start_y: int, zones: np.ndarray, visited: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Flood-fill ARM stage."""
        h, w = self._h, self._w
        arm_pixels: List[Tuple[int, int]] = []
        hand_seeds: Set[Tuple[int, int]] = set()

        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = 1

        while queue:
            x, y = queue.popleft()
            arm_pixels.append((x, y))

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
        self, start_x: int, start_y: int, zones: np.ndarray, visited: np.ndarray, edge_mask: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Flood-fill HAND stage."""
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
                    if edge_mask[ny, nx]:
                        continue
                    zone = zones[ny, nx]
                    if zone >= ZONE_MID:
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
    ) -> FingerData | None:
        """Flood-fill FINGER + TIP stages."""
        h, w = self._h, self._w
        distance = np.full((h, w), -1, dtype=np.int32)
        finger_pixels: List[int] = []
        tip_seeds: List[Tuple[int, int, int]] = []
        roots: List[Tuple[int, int]] = []

        queue = deque([(start_x, start_y, 0)])
        visited[start_y, start_x] = 1
        distance[start_y, start_x] = 0

        while queue:
            x, y, dist = queue.popleft()
            idx = y * w + x
            finger_pixels.append(idx)

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
                    if edge_finger[ny, nx]:
                        continue
                    zone = zones[ny, nx]
                    if zone == ZONE_LOW:
                        visited[ny, nx] = 1
                        distance[ny, nx] = dist + 1
                        queue.append((nx, ny, dist + 1))
                    elif zone == ZONE_NOISE:
                        tip_seeds.append((nx, ny, dist + 1))

        # Flood TIP
        tip_pixels: List[int] = []
        for sx, sy, sdist in tip_seeds:
            if visited[sy, sx]:
                continue

            tip_queue = deque([(sx, sy, sdist)])
            visited[sy, sx] = 1
            distance[sy, sx] = sdist
            overfill = False
            local_tip: List[int] = []

            while tip_queue:
                x, y, dist = tip_queue.popleft()
                if dist > self.TIP_MAX_DIST:
                    overfill = True
                    break

                idx = y * w + x
                local_tip.append(idx)

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
                        if edge_tip[ny, nx]:
                            continue
                        visited[ny, nx] = 1
                        distance[ny, nx] = dist + 1
                        tip_queue.append((nx, ny, dist + 1))

            if not overfill:
                tip_pixels.extend(local_tip)
                finger_pixels.extend(local_tip)

        if len(finger_pixels) < self.FINGER_MIN_SIZE:
            return None

        # Reflood from roots
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

        return self._compute_finger_metrics(finger_pixels, distance, diff_map, w)

    def _compute_finger_metrics(
        self, pixels: List[int], distance: np.ndarray, diff_map: np.ndarray, w: int
    ) -> FingerData | None:
        """Compute fingertip position and touch metric."""
        if not pixels:
            return None

        pixel_dists = []
        for idx in pixels:
            y, x = idx // w, idx % w
            d = distance[y, x]
            if d >= 0:
                pixel_dists.append((idx, d))

        if not pixel_dists:
            return None

        pixel_dists.sort(key=lambda p: p[1])
        max_dist = pixel_dists[-1][1]
        if max_dist < self.FINGER_MIN_DIST:
            return None

        tip_idx = pixel_dists[-1][0]
        tip_y, tip_x = tip_idx // w, tip_idx % w

        start = max(0, len(pixel_dists) - self.TOUCHZ_WINDOW)
        top_pixels = pixel_dists[start:]

        total_diff = 0.0
        count = 0
        for idx, _ in top_pixels:
            y, x = idx // w, idx % w
            total_diff += diff_map[y, x]
            count += 1

        touch_z = total_diff / count if count > 0 else 0.0

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

    # ===== Debug Visualization =====

    def _draw_calibration_progress(self, ir_frame: np.ndarray) -> None:
        """Show calibration progress."""
        h, w = self._h, self._w
        stable_pct = self._bg_model.stable_percentage

        progress_img = np.zeros((h, w, 3), dtype=np.uint8)
        ir_8bit = (np.clip(ir_frame, 0, 4000) / 16).astype(np.uint8)
        progress_img[:, :, 0] = ir_8bit // 4
        progress_img[:, :, 1] = ir_8bit // 4
        progress_img[:, :, 2] = ir_8bit // 4

        cv2.putText(progress_img, f"Calibrating: {stable_pct:.0f}%", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(progress_img, "Keep surface clear", (w // 2 - 100, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(progress_img, "DIRECT + MediaPipe Hybrid", (w // 2 - 120, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        bar_x, bar_y = w // 4, h // 2 + 90
        bar_w, bar_h = w // 2, 20
        cv2.rectangle(progress_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        filled_w = int(bar_w * stable_pct / 100)
        cv2.rectangle(progress_img, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        cv2.namedWindow("Hybrid - Depth", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - Depth", progress_img)
        cv2.waitKey(1)

    def _draw_debug(
        self,
        depth_frame: np.ndarray,
        ir_frame: np.ndarray,
        zones: np.ndarray,
        diff_map: np.ndarray,
        fingers: List[FingerData],
        hands: list,
    ) -> None:
        """Draw debug visualization."""
        h, w = self._h, self._w

        # 1. Depth with DIRECT fingers
        debug_depth = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        debug_depth = cv2.cvtColor(debug_depth, cv2.COLOR_GRAY2BGR)

        for finger in fingers:
            color = (0, 255, 0) if finger.touched else (0, 255, 255)
            cv2.circle(debug_depth, (int(finger.tip_x), int(finger.tip_y)), 8 if finger.touched else 6, color, -1 if finger.touched else 2)
            cv2.putText(debug_depth, f"z:{finger.touch_z:.1f}", (int(finger.tip_x) + 10, int(finger.tip_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw MediaPipe tips in depth space
        mp_depth_points = []
        if hands and self._last_rgb_frame is not None:
            rgb_h, rgb_w = self._last_rgb_frame.shape[:2]
            for hand in hands:
                tip = hand[self.INDEX_FINGER_TIP]
                rgb_x = int(tip[0] * rgb_w)
                rgb_y = int(tip[1] * rgb_h)
                depth_points = self._resolution_mapper.rgb_to_depth([(rgb_x, rgb_y)])
                if depth_points:
                    dx, dy = depth_points[0]
                    if dx >= 0 and dy >= 0:
                        cv2.circle(debug_depth, (dx, dy), 6, (255, 0, 255), 2)  # Magenta for MediaPipe
                        mp_depth_points.append((dx, dy))

        touch_count = sum(1 for f in fingers if f.touched)
        cv2.putText(debug_depth, f"DIRECT: {len(fingers)} touches:{touch_count} | MP: {len(hands)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_depth, "Green=DIRECT Magenta=MediaPipe", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.namedWindow("Hybrid - Depth", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - Depth", debug_depth)

        # 2. MediaPipe hand visualization (RGB)
        self._draw_mediapipe_debug(hands)

        # 3. Combined visualization
        self._draw_combined_debug(depth_frame, ir_frame, fingers, hands, mp_depth_points)

        cv2.waitKey(1)

    def _draw_mediapipe_debug(self, hands: list) -> None:
        """Draw MediaPipe hand landmarks on RGB."""
        if self._last_rgb_frame is None:
            return

        # Scale down RGB for display
        scale = 0.5
        rgb_display = cv2.resize(self._last_rgb_frame, None, fx=scale, fy=scale)
        h, w = rgb_display.shape[:2]

        # Hand connections
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),  # Palm
        ]

        for hand_idx, landmarks in enumerate(hands):
            hand_color = (255, 100, 100) if hand_idx == 0 else (100, 100, 255)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start[0] * w), int(start[1] * h)
                x2, y2 = int(end[0] * w), int(end[1] * h)
                cv2.line(rgb_display, (x1, y1), (x2, y2), hand_color, 2)

            # Draw landmarks
            for i, lm in enumerate(landmarks):
                x, y = int(lm[0] * w), int(lm[1] * h)
                if i in (4, 8, 12, 16, 20):  # Fingertips
                    cv2.circle(rgb_display, (x, y), 6, (0, 255, 255), -1)
                else:
                    cv2.circle(rgb_display, (x, y), 3, hand_color, -1)

            # Highlight INDEX_FINGER_TIP
            tip = landmarks[self.INDEX_FINGER_TIP]
            tip_x, tip_y = int(tip[0] * w), int(tip[1] * h)
            cv2.circle(rgb_display, (tip_x, tip_y), 10, (255, 0, 255), 2)
            cv2.putText(rgb_display, "IDX", (tip_x + 12, tip_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.putText(rgb_display, f"MediaPipe: {len(hands)} hands", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.namedWindow("Hybrid - MediaPipe", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - MediaPipe", rgb_display)

    def _draw_combined_debug(
        self,
        depth_frame: np.ndarray,
        ir_frame: np.ndarray,
        fingers: List[FingerData],
        hands: list,
        mp_depth_points: list,
    ) -> None:
        """Draw combined visualization showing DIRECT + MediaPipe + matching."""
        h, w = self._h, self._w

        # Use IR as base (better contrast)
        ir_8bit = (np.clip(ir_frame, 0, 4000) / 16).astype(np.uint8)
        combined = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)

        # Draw DIRECT fingers
        for finger in fingers:
            fx, fy = int(finger.tip_x), int(finger.tip_y)
            if finger.touched:
                cv2.circle(combined, (fx, fy), 12, (0, 255, 0), 3)  # Green thick circle
                cv2.circle(combined, (fx, fy), 4, (0, 255, 0), -1)  # Green dot
            else:
                cv2.circle(combined, (fx, fy), 8, (0, 200, 200), 2)  # Yellow circle

        # Draw MediaPipe tips
        for dx, dy in mp_depth_points:
            cv2.circle(combined, (dx, dy), 10, (255, 0, 255), 2)  # Magenta circle
            cv2.drawMarker(combined, (dx, dy), (255, 0, 255), cv2.MARKER_CROSS, 8, 2)

        # Draw matching lines (DIRECT touch -> nearest MediaPipe)
        for finger in fingers:
            if not finger.touched:
                continue
            fx, fy = int(finger.tip_x), int(finger.tip_y)

            # Find nearest MediaPipe
            best_mp = None
            best_dist = float("inf")
            for dx, dy in mp_depth_points:
                dist = np.sqrt((dx - fx) ** 2 + (dy - fy) ** 2)
                if dist < best_dist and dist < self.MAX_MATCH_DISTANCE:
                    best_dist = dist
                    best_mp = (dx, dy)

            if best_mp is not None:
                # Draw line from DIRECT to MediaPipe
                cv2.line(combined, (fx, fy), best_mp, (0, 255, 255), 2)
                # Draw final position (MediaPipe) with white
                cv2.circle(combined, best_mp, 6, (255, 255, 255), -1)

        # Legend
        touch_count = sum(1 for f in fingers if f.touched)
        cv2.putText(combined, f"DIRECT:{len(fingers)} touch:{touch_count} MP:{len(hands)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Green=DIRECT Magenta=MP White=Final Yellow=Line",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.namedWindow("Hybrid - Combined", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - Combined", combined)

    def reset_background(self) -> None:
        """Reset background model."""
        self._bg_model = DIRECTBackgroundModel((self._h, self._w))
        self._finger_states.clear()
