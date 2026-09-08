"""
Touch detector with MediaPipe (CPU) + depth validation on torch-directml.

MediaPipe HandLandmarker uses its own TFLite runtime and stays on CPU.
Depth touch-zone masks (dmin/dmax) run on the DirectML device.
"""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cv_system.detection.dml_depth_ops import DmlDepthTouchOps
from cv_system.detection.touch_detector import TouchDetector
from cv_system.detection.touch_tracker import TrackedTouch
from cv_system.transform import DepthCoordinateTransformer


class DmlTouchDetector(TouchDetector):
    """
    Same pipeline as TouchDetector, with GPU-accelerated depth touch-zone checks.

    Set CV_TORCH_BACKEND=directml and install torch-directml.
    """

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_coordinate_transformer: DepthCoordinateTransformer,
        resolution_mapper,
        config,
        *,
        show_debug: bool = False,
    ) -> None:
        # Skip TouchDetector.__init__ MediaPipe setup — duplicate below for clarity
        self._dmax_map = dmax_map.astype(np.int32)
        self._dmin_map = (self._dmax_map - config.touch_threshold).astype(np.int32)
        self._depth_transformer = depth_coordinate_transformer
        self._resolution_mapper = resolution_mapper
        self._show_debug = show_debug

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

        from cv_system.detection.touch_tracker import TouchTracker

        self._touch_tracker = TouchTracker(
            debounce_frames=3,
            touch_radius=15.0,
            lost_track_buffer=5,
        )

        self._ir_roi_size = 15
        self._ir_canny_low = 50
        self._ir_canny_high = 150

        self._dml_ops = DmlDepthTouchOps(self._dmin_map, self._dmax_map)
        print(
            "  DmlTouchDetector: MediaPipe on CPU, depth masks on "
            f"{self._dml_ops._device}"
        )

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        rgb_np = rgb_frame.get() if isinstance(rgb_frame, cv2.UMat) else rgb_frame
        rgb_rgb = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        rgb_h, rgb_w = rgb_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_rgb)

        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        touches_projector: list[tuple[float, float]] = []
        hands_detected = bool(result and result.hand_landmarks)

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
                ir_8bit = np.clip(ir_frame / 16, 0, 255).astype(np.uint8)
                debug_ir = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)

        depth_h, depth_w = depth_frame.shape
        touch_zone = self._dml_ops.touch_zone_mask(depth_frame)
        touch_zone_np = touch_zone.cpu().numpy()

        if hands_detected:
            for hand_landmarks in result.hand_landmarks:
                all_rgb_points = []
                for lm in hand_landmarks:
                    all_rgb_points.append((int(lm.x * rgb_w), int(lm.y * rgb_h)))

                if self._show_debug:
                    for pt in all_rgb_points:
                        cv2.circle(debug_rgb, pt, 5, (0, 255, 0), -1)

                all_depth_points = self._resolution_mapper.rgb_to_depth(all_rgb_points)
                valid_depth_points = [
                    (x, y) for x, y in all_depth_points if 0 <= x < depth_w and 0 <= y < depth_h
                ]

                if len(valid_depth_points) < 3:
                    if self._show_debug:
                        cv2.putText(
                            debug_rgb,
                            "NOT ENOUGH POINTS",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            3,
                        )
                    continue

                depth_points_array = np.array(valid_depth_points, dtype=np.int32)
                hull = cv2.convexHull(depth_points_array)
                hand_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
                cv2.fillConvexPoly(hand_mask, hull, 255)

                if self._show_debug:
                    cv2.drawContours(depth_vis, [hull], 0, (255, 0, 255), 2)

                if not self._dml_ops.hand_has_touch(hand_mask, touch_zone):
                    if self._show_debug:
                        cv2.putText(
                            debug_rgb,
                            "NO TOUCH",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 255),
                            4,
                        )
                    continue

                tip = hand_landmarks[self.INDEX_FINGER_TIP]
                tip_rgb_x = int(tip.x * rgb_w)
                tip_rgb_y = int(tip.y * rgb_h)
                tip_depth = self._resolution_mapper.rgb_to_depth([(tip_rgb_x, tip_rgb_y)])[0]
                tip_depth_x = max(0, min(depth_w - 1, tip_depth[0]))
                tip_depth_y = max(0, min(depth_h - 1, tip_depth[1]))

                if ir_frame is not None:
                    dip = hand_landmarks[self.INDEX_FINGER_DIP]
                    dip_rgb_x = int(dip.x * rgb_w)
                    dip_rgb_y = int(dip.y * rgb_h)
                    dip_depth = self._resolution_mapper.rgb_to_depth([(dip_rgb_x, dip_rgb_y)])[0]
                    dip_depth_x = max(0, min(depth_w - 1, dip_depth[0]))
                    dip_depth_y = max(0, min(depth_h - 1, dip_depth[1]))
                    tip_depth_x, tip_depth_y = self._correct_fingertip_with_ir(
                        (tip_depth_x, tip_depth_y),
                        (dip_depth_x, dip_depth_y),
                        ir_frame,
                        debug_ir,
                    )

                touch_depth_point = np.array([[tip_depth_x, tip_depth_y]], dtype=np.float32)
                touch_proj_point = self._depth_transformer.camera_to_projector(touch_depth_point)
                touches_projector.append(
                    (float(touch_proj_point[0, 0]), float(touch_proj_point[0, 1]))
                )

                if self._show_debug:
                    cv2.putText(
                        debug_rgb,
                        "TOUCH!",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        4,
                    )
                    hand_touch = (hand_mask > 0) & touch_zone_np
                    touch_overlay = depth_vis.copy()
                    touch_overlay[hand_touch] = (0, 255, 0)
                    cv2.addWeighted(touch_overlay, 0.5, depth_vis, 0.5, 0, depth_vis)
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

        return self._touch_tracker.update(touches_projector), hands_detected
