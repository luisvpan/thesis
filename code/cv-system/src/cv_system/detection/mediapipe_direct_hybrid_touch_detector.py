"""
Hybrid touch detector combining MediaPipe hand detection with DIRECT-style touch analysis.

This detector combines the strengths of both approaches:
- MediaPipe: Robust hand detection and precise fingertip localization via ML landmarks
- DIRECT: Sophisticated touch detection using z-score hysteresis and dynamic background

Key advantages:
- Only detects touches from confirmed HANDS (ignores cards, blocks, other objects)
- Precise fingertip position from ML landmarks
- Robust touch detection using depth analysis with hysteresis
- Optional IR edge refinement for sub-pixel accuracy
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cv_system.detection.direct_background import DIRECTBackgroundModel
from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch
from cv_system.transform import DepthCoordinateTransformer


@dataclass
class HybridTouchPoint:
    """Represents a touch point from the hybrid detector."""

    x: float  # World X coordinate (mm)
    y: float  # World Y coordinate (mm)
    z: float  # Depth at touch point (mm)
    pixel_x: int  # Depth image X
    pixel_y: int  # Depth image Y
    touching: bool  # Is finger touching surface
    diff_mm: float  # Distance from background (mm)
    hand_index: int  # Which hand (0 or 1)


class MediapipeDIRECTHybridTouchDetector:
    """
    Hybrid touch detector: MediaPipe hand detection + DIRECT-style touch analysis.

    Pipeline:
        1. MediaPipe detects hands in RGB frame
        2. Map fingertip landmarks to depth space
        3. For each fingertip, compute diff from background (DIRECT-style)
        4. Apply hysteresis for stable touch detection
        5. Optionally refine position using IR edges
        6. Transform to projector space via TouchTracker
    """

    # MediaPipe landmark indices
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_DIP = 7
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20
    THUMB_TIP = 4

    # Touch detection thresholds (in mm, matching DIRECT)
    TOUCHZ_ENTER = 10.0  # diff below this = touching
    TOUCHZ_EXIT = 25.0  # diff above this = not touching

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_coordinate_transformer: DepthCoordinateTransformer,
        resolution_mapper,
        config,
        *,
        show_debug: bool = False,
        use_ir_refinement: bool = True,
    ) -> None:
        """
        Initialize the hybrid touch detector.

        Args:
            dmax_map: Calibrated maximum depth map (for initial reference).
            depth_coordinate_transformer: Transformer for depth <-> projector mapping.
            resolution_mapper: Mapper for RGB <-> depth coordinate conversion.
            config: Detection configuration.
            show_debug: If True, display debug visualization windows.
            use_ir_refinement: If True, refine fingertip position using IR edges.
        """
        self._dmax_map = dmax_map.astype(np.float32)
        self._depth_transformer = depth_coordinate_transformer
        self._resolution_mapper = resolution_mapper
        self._config = config
        self._show_debug = show_debug
        self._use_ir_refinement = use_ir_refinement
        self._shape = dmax_map.shape

        # DIRECT-style background model
        self._bg_model = DIRECTBackgroundModel(dmax_map.shape)

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
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()

        # Touch tracker for persistent IDs and stateful events
        self._touch_tracker = TouchTracker(
            debounce_frames=3,
            touch_radius=15.0,
            lost_track_buffer=5,
        )

        # Per-finger touch state for hysteresis (hand_index, finger_index) -> bool
        self._touch_states: dict[tuple[int, int], bool] = {}

        # IR correction parameters
        self._ir_roi_size = 15
        self._ir_canny_low = 50
        self._ir_canny_high = 150

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        """
        Detect touches using hybrid MediaPipe + DIRECT approach.

        Args:
            depth_frame: Depth frame (uint16, 424x512, mm values).
            rgb_frame: RGB frame (BGR, 1080x1920) for MediaPipe.
            ir_frame: Optional IR frame for fingertip refinement.

        Returns:
            touches: List of TrackedTouch with persistent IDs and events.
            hands_detected: True if any hands were detected (even if not touching).
        """
        # 1. Update background model
        self._bg_model.update(depth_frame)

        # Check if background is ready
        if not self._bg_model.is_ready:
            if self._show_debug:
                self._draw_calibration_status()
            return [], False

        # 2. MediaPipe hand detection (requires RGB)
        if rgb_frame is None:
            if self._show_debug:
                self._draw_status("No RGB frame")
            return [], False

        hands = self._detect_hands(rgb_frame)

        if not hands:
            # No hands = clear touch states and return
            self._touch_states.clear()
            if self._show_debug:
                self._draw_debug(depth_frame, ir_frame, [], [])
            return [], False

        # 3. Process each hand's index finger
        touch_points: list[HybridTouchPoint] = []
        bg_mean = self._bg_model.mean
        bg_stddev = self._bg_model.stddev

        for hand_idx, hand_landmarks in enumerate(hands):
            # Get index finger TIP and DIP in depth space
            tip_depth = self._landmark_to_depth(
                hand_landmarks[self.INDEX_FINGER_TIP], rgb_frame.shape
            )
            dip_depth = self._landmark_to_depth(
                hand_landmarks[self.INDEX_FINGER_DIP], rgb_frame.shape
            )

            if tip_depth is None or dip_depth is None:
                continue

            tip_x, tip_y = tip_depth
            h, w = depth_frame.shape

            # Bounds check
            if not (0 <= tip_x < w and 0 <= tip_y < h):
                continue

            # 4. Compute diff from background (DIRECT-style)
            current_depth = float(depth_frame[tip_y, tip_x])
            background_depth = float(bg_mean[tip_y, tip_x])

            if current_depth <= 0 or background_depth <= 0:
                continue

            diff_mm = background_depth - current_depth  # positive = closer to camera

            # 5. Apply hysteresis for touch detection
            finger_key = (hand_idx, self.INDEX_FINGER_TIP)
            prev_touching = self._touch_states.get(finger_key, False)
            is_touching = self._apply_hysteresis(diff_mm, prev_touching)
            self._touch_states[finger_key] = is_touching

            # 6. Optional IR refinement
            final_tip = tip_depth
            if is_touching and self._use_ir_refinement and ir_frame is not None:
                final_tip = self._refine_with_ir(tip_depth, dip_depth, ir_frame)

            final_x, final_y = final_tip

            # 7. Transform to world coordinates
            world_x, world_y = self._depth_transformer.depth_to_world(
                final_x, final_y, current_depth
            )

            touch_points.append(
                HybridTouchPoint(
                    x=world_x,
                    y=world_y,
                    z=current_depth,
                    pixel_x=final_x,
                    pixel_y=final_y,
                    touching=is_touching,
                    diff_mm=diff_mm,
                    hand_index=hand_idx,
                )
            )

        # 8. Filter to only touching points and track
        touching_points = [p for p in touch_points if p.touching]

        # Convert to format expected by tracker
        touch_positions = [(p.x, p.y) for p in touching_points]
        tracked = self._touch_tracker.update(touch_positions)

        if self._show_debug:
            self._draw_debug(depth_frame, ir_frame, touch_points, tracked)

        return tracked, len(hands) > 0

    def _detect_hands(
        self, rgb_frame: np.ndarray
    ) -> list[list[tuple[float, float, float]]]:
        """
        Detect hands using MediaPipe.

        Returns:
            List of hands, each containing 21 landmarks as (x, y, z) normalized.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Get timestamp for video mode
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)

        # Detect
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        hands = []
        for hand_landmarks in result.hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            hands.append(landmarks)

        return hands

    def _landmark_to_depth(
        self,
        landmark: tuple[float, float, float],
        rgb_shape: tuple[int, ...],
    ) -> tuple[int, int] | None:
        """
        Convert MediaPipe landmark (normalized) to depth image coordinates.

        Args:
            landmark: (x, y, z) normalized landmark from MediaPipe.
            rgb_shape: Shape of RGB frame (height, width, channels).

        Returns:
            (x, y) in depth image coordinates, or None if out of bounds.
        """
        rgb_h, rgb_w = rgb_shape[:2]

        # Denormalize to RGB pixel coordinates
        rgb_x = int(landmark[0] * rgb_w)
        rgb_y = int(landmark[1] * rgb_h)

        # Map RGB to depth coordinates
        depth_x, depth_y = self._resolution_mapper.rgb_to_depth(rgb_x, rgb_y)

        # Bounds check
        depth_h, depth_w = self._shape
        if not (0 <= depth_x < depth_w and 0 <= depth_y < depth_h):
            return None

        return (int(depth_x), int(depth_y))

    def _apply_hysteresis(self, diff_mm: float, prev_touching: bool) -> bool:
        """
        Apply hysteresis for stable touch detection.

        Args:
            diff_mm: Distance from background in mm (positive = closer).
            prev_touching: Previous touch state.

        Returns:
            Current touch state.
        """
        if prev_touching:
            # Currently touching: need diff > EXIT threshold to release
            return diff_mm <= self.TOUCHZ_EXIT
        else:
            # Not touching: need diff < ENTER threshold to activate
            return diff_mm >= self.TOUCHZ_ENTER

    def _refine_with_ir(
        self,
        tip_depth: tuple[int, int],
        dip_depth: tuple[int, int],
        ir_frame: np.ndarray,
    ) -> tuple[int, int]:
        """
        Refine fingertip position using IR edge detection.

        The depth sensor has edge erosion. IR edges are sharper.
        Find the IR edge point most aligned with finger direction.

        Args:
            tip_depth: TIP position in depth space.
            dip_depth: DIP position for direction calculation.
            ir_frame: IR frame (uint16).

        Returns:
            Refined fingertip position.
        """
        tip_x, tip_y = tip_depth
        dip_x, dip_y = dip_depth
        h, w = ir_frame.shape

        # Create ROI around TIP
        roi_half = self._ir_roi_size // 2
        x1 = max(0, tip_x - roi_half)
        y1 = max(0, tip_y - roi_half)
        x2 = min(w, tip_x + roi_half + 1)
        y2 = min(h, tip_y + roi_half + 1)

        if x2 - x1 < 3 or y2 - y1 < 3:
            return tip_depth

        # Extract and process IR ROI
        ir_roi = ir_frame[y1:y2, x1:x2]
        ir_roi_8bit = np.clip(ir_roi / 16, 0, 255).astype(np.uint8)
        ir_blurred = cv2.GaussianBlur(ir_roi_8bit, (3, 3), 0)
        edges = cv2.Canny(ir_blurred, self._ir_canny_low, self._ir_canny_high)

        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))

        if len(edge_points) == 0:
            return tip_depth

        # Convert to global coordinates
        edge_global = [(x1 + col, y1 + row) for row, col in edge_points]

        # Find edge point most aligned with finger direction
        dir_x = tip_x - dip_x
        dir_y = tip_y - dip_y
        dir_len = np.sqrt(dir_x * dir_x + dir_y * dir_y)

        if dir_len < 1:
            return tip_depth

        dir_x /= dir_len
        dir_y /= dir_len

        best_point = tip_depth
        best_score = -float("inf")

        for ex, ey in edge_global:
            # Vector from DIP to edge point
            vec_x = ex - dip_x
            vec_y = ey - dip_y
            vec_len = np.sqrt(vec_x * vec_x + vec_y * vec_y)

            if vec_len < 1:
                continue

            # Alignment score (dot product)
            alignment = (vec_x * dir_x + vec_y * dir_y) / vec_len

            # Distance along finger direction
            distance = vec_x * dir_x + vec_y * dir_y

            # Combined score: prefer aligned points farther along finger
            score = alignment * 0.5 + distance * 0.01

            if score > best_score and alignment > 0.7:
                best_score = score
                best_point = (ex, ey)

        return best_point

    def _draw_calibration_status(self) -> None:
        """Draw calibration progress during background model initialization."""
        h, w = self._shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

        pct = self._bg_model.stable_percentage
        cv2.putText(
            img,
            f"Calibrating: {pct:.0f}%",
            (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "Keep surface clear",
            (w // 2 - 90, h // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            img,
            "MediaPipe + DIRECT Hybrid",
            (w // 2 - 100, h // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
        )

        cv2.namedWindow("Hybrid - Status", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - Status", img)
        cv2.waitKey(1)

    def _draw_status(self, text: str) -> None:
        """Draw status message."""
        h, w = self._shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            img,
            text,
            (w // 2 - 60, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.namedWindow("Hybrid - Status", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - Status", img)
        cv2.waitKey(1)

    def _draw_debug(
        self,
        depth_frame: np.ndarray,
        ir_frame: np.ndarray | None,
        touch_points: list[HybridTouchPoint],
        tracked: list[TrackedTouch],
    ) -> None:
        """Draw debug visualization."""
        bg_mean = self._bg_model.mean

        # Diff visualization
        diff = bg_mean - depth_frame.astype(np.float32)
        diff_vis = np.clip((diff + 50) / 100 * 255, 0, 255).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

        # Draw all touch points (detected hands)
        for tp in touch_points:
            color = (0, 255, 0) if tp.touching else (0, 165, 255)  # Green/Orange
            cv2.circle(diff_color, (tp.pixel_x, tp.pixel_y), 8, color, 2)

            label = f"{tp.diff_mm:.1f}mm"
            if tp.touching:
                label += " TOUCH"
            cv2.putText(
                diff_color,
                label,
                (tp.pixel_x + 10, tp.pixel_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Draw tracked touches (with IDs)
        for t in tracked:
            # Convert world back to pixel for visualization
            px = int(t.x / 1000 * diff_color.shape[1])  # Approximate
            py = int(t.y / 1000 * diff_color.shape[0])
            cv2.putText(
                diff_color,
                f"ID:{t.id}",
                (10, 20 + t.id * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Add info
        cv2.putText(
            diff_color,
            f"Hands: {len(touch_points)} | Touching: {sum(1 for p in touch_points if p.touching)}",
            (10, diff_color.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.namedWindow("Hybrid - Depth Diff", cv2.WINDOW_NORMAL)
        cv2.imshow("Hybrid - Depth Diff", diff_color)

        # IR visualization if available
        if ir_frame is not None:
            ir_vis = np.clip(ir_frame / 16, 0, 255).astype(np.uint8)
            ir_color = cv2.cvtColor(ir_vis, cv2.COLOR_GRAY2BGR)

            for tp in touch_points:
                color = (0, 255, 0) if tp.touching else (0, 165, 255)
                cv2.circle(ir_color, (tp.pixel_x, tp.pixel_y), 5, color, 1)

            cv2.namedWindow("Hybrid - IR", cv2.WINDOW_NORMAL)
            cv2.imshow("Hybrid - IR", ir_color)

        cv2.waitKey(1)

    def reset_background(self) -> None:
        """Reset background model to recalibrate."""
        self._bg_model = DIRECTBackgroundModel(self._shape)
        self._touch_states.clear()

    def set_thresholds(self, enter_mm: float, exit_mm: float) -> None:
        """
        Adjust touch detection thresholds.

        Args:
            enter_mm: Diff threshold to activate touch (default: 10mm).
            exit_mm: Diff threshold to deactivate touch (default: 25mm).
        """
        self.TOUCHZ_ENTER = enter_mm
        self.TOUCHZ_EXIT = exit_mm
