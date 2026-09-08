"""
Touch detector using MediaPipe HandLandmarker with depth validation.

Hybrid approach: Uses MediaPipe to locate the index finger, then searches
within that region in depth space for touch points using dmin/dmax validation.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch
from cv_system.transform import DepthCoordinateTransformer


class TouchDetector:
    """
    Hybrid touch detector: MediaPipe hand segmentation + depth shell validation.

    Coordinate flow:
        RGB frame -> MediaPipe -> 21 hand landmarks
            -> map all landmarks to depth space
            -> create convex hull of hand in depth space
            -> search within hull for pixels where dmin < z < dmax
            -> if found, touch point = index finger TIP position
            -> (optional) correct TIP position using IR edge detection
            -> transform touch point to projector space
            -> TouchTracker -> TrackedTouch events (down/move/up)
    """

    INDEX_FINGER_TIP = 8  # Fingertip landmark
    INDEX_FINGER_DIP = 7  # DIP joint (for direction calculation)

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_coordinate_transformer: DepthCoordinateTransformer,
        resolution_mapper,
        config,
        *,
        show_debug: bool = False,
    ) -> None:
        """
        Initialize the touch detector.

        Args:
            dmax_map: Calibrated maximum depth map (surface depth with offset).
            depth_coordinate_transformer: Transformer for depth <-> projector mapping.
            resolution_mapper: Mapper for RGB <-> depth coordinate conversion.
            config: Detection configuration with touch_threshold.
            show_debug: If True, display debug visualization windows.
        """
        # Touch zone: dmin < z < dmax (same as DepthOnlyTouchDetector)
        self._dmax_map = dmax_map.astype(np.int32)
        self._dmin_map = (self._dmax_map - config.touch_threshold).astype(np.int32)
        self._depth_transformer = depth_coordinate_transformer
        self._resolution_mapper = resolution_mapper
        self._show_debug = show_debug


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

        # IR correction parameters
        self._ir_roi_size = 15  # Size of ROI around fingertip for IR correction
        self._ir_canny_low = 50
        self._ir_canny_high = 150

    def _correct_fingertip_with_ir(
        self,
        tip_depth: tuple[int, int],
        dip_depth: tuple[int, int],
        ir_frame: np.ndarray,
        debug_ir: np.ndarray | None = None,
    ) -> tuple[int, int]:
        """
        Correct fingertip position using IR edge detection (ESPOL paper method).

        The depth sensor has erosion at object edges. IR has sharper edges.
        This method finds the IR edge point most aligned with finger direction.

        Args:
            tip_depth: TIP position in depth space (x, y).
            dip_depth: DIP position in depth space (for direction calculation).
            ir_frame: IR frame (uint16, 512x424).
            debug_ir: Optional IR visualization to draw on.

        Returns:
            Corrected fingertip position (x, y) in depth space.
        """
        tip_x, tip_y = tip_depth
        dip_x, dip_y = dip_depth
        h, w = ir_frame.shape

        # Calculate Pmid (midpoint between TIP and DIP)
        mid_x = (tip_x + dip_x) // 2
        mid_y = (tip_y + dip_y) // 2

        # Create ROI around TIP
        roi_half = self._ir_roi_size // 2
        x1 = max(0, tip_x - roi_half)
        y1 = max(0, tip_y - roi_half)
        x2 = min(w, tip_x + roi_half + 1)
        y2 = min(h, tip_y + roi_half + 1)

        if x2 - x1 < 3 or y2 - y1 < 3:
            return tip_depth  # ROI too small

        # Extract and process IR ROI
        ir_roi = ir_frame[y1:y2, x1:x2]

        # Normalize to 8-bit for Canny
        ir_roi_8bit = np.clip(ir_roi / 16, 0, 255).astype(np.uint8)

        # Apply Gaussian blur and Canny edge detection
        ir_blurred = cv2.GaussianBlur(ir_roi_8bit, (3, 3), 0)
        edges = cv2.Canny(ir_blurred, self._ir_canny_low, self._ir_canny_high)

        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))  # (row, col) format

        if len(edge_points) == 0:
            return tip_depth  # No edges found

        # Convert to global coordinates (x, y format)
        edge_global = [(x1 + col, y1 + row) for row, col in edge_points]

        # Find edge point most aligned with finger direction
        # Direction vector from mid to tip
        dir_x = tip_x - mid_x
        dir_y = tip_y - mid_y
        dir_len = np.sqrt(dir_x**2 + dir_y**2)

        if dir_len < 1:
            return tip_depth  # No clear direction

        dir_x /= dir_len
        dir_y /= dir_len

        best_point = tip_depth
        best_score = -1.0

        for ex, ey in edge_global:
            # Vector from TIP to edge point
            vec_x = ex - tip_x
            vec_y = ey - tip_y
            vec_len = np.sqrt(vec_x**2 + vec_y**2)

            if vec_len < 1:
                continue

            # Dot product (alignment with finger direction)
            alignment = (vec_x * dir_x + vec_y * dir_y) / vec_len

            # We want points in the direction of the finger (alignment close to 1)
            # and close to TIP
            if alignment > 0.7:  # Within ~45 degrees of finger direction
                score = alignment - (vec_len / self._ir_roi_size) * 0.3
                if score > best_score:
                    best_score = score
                    best_point = (ex, ey)

        # Debug visualization
        if debug_ir is not None:
            # Draw ROI rectangle
            cv2.rectangle(debug_ir, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # Draw edges in ROI
            for ex, ey in edge_global:
                cv2.circle(debug_ir, (ex, ey), 1, (255, 0, 0), -1)

            # Draw original TIP (red)
            cv2.circle(debug_ir, tip_depth, 4, (0, 0, 255), -1)

            # Draw corrected TIP (green)
            if best_point != tip_depth:
                cv2.circle(debug_ir, best_point, 4, (0, 255, 0), -1)
                cv2.line(debug_ir, tip_depth, best_point, (0, 255, 0), 1)

            # Draw direction from mid to tip
            cv2.line(debug_ir, (mid_x, mid_y), tip_depth, (255, 255, 0), 1)

        return best_point

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        """
        Detect touches from depth frame and raw RGB frame.

        Args:
            depth_frame: Raw depth frame (depth space, uint16, 424x512).
            rgb_frame: Raw RGB frame from camera (1920x1080 BGR).
            ir_frame: Optional IR frame for fingertip correction (uint16, 424x512).

        Returns:
            Tuple of (tracked_touches, hands_detected):
            - tracked_touches: List of TrackedTouch events (down/move/up).
            - hands_detected: True if any hands were detected in the frame.
        """
        # Convert to RGB for MediaPipe
        rgb_np = rgb_frame.get() if isinstance(rgb_frame, cv2.UMat) else rgb_frame
        rgb_rgb = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        rgb_h, rgb_w = rgb_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_rgb)

        # VIDEO mode: synchronous detection, thread-safe
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        touches_projector: list[tuple[float, float]] = []
        hands_detected = bool(result and result.hand_landmarks)

        # Prepare debug visualizations
        debug_rgb = None
        depth_vis = None
        debug_ir = None
        if self._show_debug:
            debug_rgb = rgb_np.copy()
            depth_vis = cv2.normalize(
                depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            if ir_frame is not None:
                # Normalize IR to 8-bit and colorize
                ir_8bit = np.clip(ir_frame / 16, 0, 255).astype(np.uint8)
                debug_ir = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)

        depth_h, depth_w = depth_frame.shape

        # Precompute touch zone mask
        touch_zone = (
            (depth_frame > self._dmin_map)
            & (depth_frame < self._dmax_map)
            & (depth_frame > 0)
        )

        if hands_detected:
            for hand_landmarks in result.hand_landmarks:
                # 1. Get ALL 21 landmarks in RGB space
                all_rgb_points = []
                for lm in hand_landmarks:
                    lm_rgb_x = int(lm.x * rgb_w)
                    lm_rgb_y = int(lm.y * rgb_h)
                    all_rgb_points.append((lm_rgb_x, lm_rgb_y))

                # Draw landmarks on RGB view
                if self._show_debug:
                    for pt in all_rgb_points:
                        cv2.circle(debug_rgb, pt, 5, (0, 255, 0), -1)

                # 2. Map all landmarks to depth space
                all_depth_points = self._resolution_mapper.rgb_to_depth(all_rgb_points)

                # Filter valid points (within depth frame bounds)
                valid_depth_points = [
                    (x, y) for x, y in all_depth_points
                    if 0 <= x < depth_w and 0 <= y < depth_h
                ]

                if len(valid_depth_points) < 3:
                    # Need at least 3 points for convex hull
                    if self._show_debug:
                        cv2.putText(debug_rgb, "NOT ENOUGH POINTS",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    continue

                # 3. Create convex hull in depth space
                depth_points_array = np.array(valid_depth_points, dtype=np.int32)
                hull = cv2.convexHull(depth_points_array)

                # 4. Create hand mask
                hand_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
                cv2.fillConvexPoly(hand_mask, hull, 255)

                # Draw hull on depth view
                if self._show_debug:
                    cv2.drawContours(depth_vis, [hull], 0, (255, 0, 255), 2)

                # 5. Check for touch within hand mask
                hand_touch = (hand_mask > 0) & touch_zone

                if not np.any(hand_touch):
                    # No touch
                    if self._show_debug:
                        cv2.putText(debug_rgb, "NO TOUCH",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                    continue

                # 6. Touch detected! Get TIP position for output
                tip = hand_landmarks[self.INDEX_FINGER_TIP]
                tip_rgb_x = int(tip.x * rgb_w)
                tip_rgb_y = int(tip.y * rgb_h)
                tip_depth = self._resolution_mapper.rgb_to_depth([(tip_rgb_x, tip_rgb_y)])[0]
                tip_depth_x, tip_depth_y = tip_depth

                # Clamp to valid range
                tip_depth_x = max(0, min(depth_w - 1, tip_depth_x))
                tip_depth_y = max(0, min(depth_h - 1, tip_depth_y))

                # 7. Correct TIP position using IR edge detection (if available)
                if ir_frame is not None:
                    dip = hand_landmarks[self.INDEX_FINGER_DIP]
                    dip_rgb_x = int(dip.x * rgb_w)
                    dip_rgb_y = int(dip.y * rgb_h)
                    dip_depth = self._resolution_mapper.rgb_to_depth(
                        [(dip_rgb_x, dip_rgb_y)]
                    )[0]
                    dip_depth_x = max(0, min(depth_w - 1, dip_depth[0]))
                    dip_depth_y = max(0, min(depth_h - 1, dip_depth[1]))

                    corrected = self._correct_fingertip_with_ir(
                        (tip_depth_x, tip_depth_y),
                        (dip_depth_x, dip_depth_y),
                        ir_frame,
                        debug_ir,
                    )
                    tip_depth_x, tip_depth_y = corrected

                # Transform TIP to projector space
                touch_depth_point = np.array(
                    [[tip_depth_x, tip_depth_y]], dtype=np.float32
                )
                touch_proj_point = self._depth_transformer.camera_to_projector(
                    touch_depth_point
                )
                touch_proj_x = float(touch_proj_point[0, 0])
                touch_proj_y = float(touch_proj_point[0, 1])

                touches_projector.append((touch_proj_x, touch_proj_y))

                # Debug visualization
                if self._show_debug:
                    # Show TOUCH in green
                    cv2.putText(debug_rgb, "TOUCH!",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

                    # Highlight touch area in depth view
                    touch_overlay = depth_vis.copy()
                    touch_overlay[hand_touch] = (0, 255, 0)
                    cv2.addWeighted(touch_overlay, 0.5, depth_vis, 0.5, 0, depth_vis)

                    # Mark TIP position
                    cv2.circle(depth_vis, (tip_depth_x, tip_depth_y), 10, (0, 0, 255), -1)
                    cv2.circle(depth_vis, (tip_depth_x, tip_depth_y), 10, (255, 255, 255), 2)

        if self._show_debug:
            cv2.namedWindow("TouchDetector - RGB", cv2.WINDOW_NORMAL)
            cv2.imshow("TouchDetector - RGB", debug_rgb)
            cv2.namedWindow("TouchDetector - Depth", cv2.WINDOW_NORMAL)
            cv2.imshow("TouchDetector - Depth", depth_vis)
            if debug_ir is not None:
                cv2.namedWindow("TouchDetector - IR", cv2.WINDOW_NORMAL)
                cv2.imshow("TouchDetector - IR", debug_ir)
            cv2.waitKey(1)

        # Track touches for persistent IDs and stateful events
        tracked_touches = self._touch_tracker.update(touches_projector)

        return tracked_touches, hands_detected

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()
