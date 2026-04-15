import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
from enum import IntEnum, verify, UNIQUE
from typing import Optional

from cv_system.transform import RgbImageTransformer, DepthCoordinateTransformer, ResolutionMapper


@verify(UNIQUE)
class HandLandmark(IntEnum):
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20


class TouchDetector:
    """
    Detects touches and returns positions in projector coordinates.

    Internally transforms the RGB frame to bird view (projector space)
    before running MediaPipe. Detected landmark positions are then mapped
    back to camera/depth space to validate touch against the dmax_map.

    Coordinate flow:
        rgb (camera space)
            -> RgbImageTransformer       -> rgb_bird (projector space)
            -> MediaPipe                 -> landmarks (projector space, normalized)
            -> DepthCoordinateTransformer.projector_to_camera
                                         -> (x, y) in rgb/camera space
            -> ResolutionMapper.rgb_to_depth
                                         -> (cx, cy) in depth space
            -> dmax_map[cy, cx]          -> touch validation
        touch confirmed -> landmark position (projector space) returned directly
    """

    latest_result: Optional[vision.HandLandmarkerResult] = None

    def __init__(
        self,
        dmax_map: np.ndarray,
        rgb_image_transformer: RgbImageTransformer,
        depth_coordinate_transformer: DepthCoordinateTransformer,
        resolution_mapper: ResolutionMapper,
        config,
    ) -> None:
        self._dmax_map = dmax_map
        self._image_transformer = rgb_image_transformer
        self._coordinate_transformer = depth_coordinate_transformer
        self._resolution_mapper = resolution_mapper
        self.touch_threshold = getattr(config, "touch_threshold", 20)
        self.latest_result = None
        self.FINGER_TIPS = [HandLandmark.INDEX_FINGER_TIP.value]

        def result_callback(
            result: vision.HandLandmarkerResult,
            _output_image: mp.Image,
            _timestamp_ms: int,
        ) -> None:
            self.latest_result = result

        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback,
            num_hands=2,
            min_hand_detection_confidence=0.15,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(
        self, depth_frame: np.ndarray, rgb_frame: np.ndarray
    ) -> list[tuple[float, float]]:
        """
        Detect touches from raw frames and return projector coordinates.

        Args:
            depth_frame: Raw depth frame from HardwareManager (depth space, uint16).
            rgb_frame: Raw RGB frame from HardwareManager (camera space, uint8 BGR).

        Returns:
            List of (x, y) touch positions in projector coordinates.
        """
        # Transform RGB to bird view (projector space) for MediaPipe
        rgb_float = rgb_frame.astype(np.float32) / 255.0
        rgb_bird = self._image_transformer.camera_to_projector(rgb_float)
        rgb_h, rgb_w = rgb_bird.shape[:2]

        rgb_bird_uint8 = (rgb_bird * 255).astype(np.uint8)
        rgb_bird_mp = cv2.cvtColor(rgb_bird_uint8, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_bird_mp)

        timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, timestamp_ms)

        touches_projector = []
        debug_img = rgb_bird_uint8.copy()

        if self.latest_result and self.latest_result.hand_landmarks:
            for hand_landmarks in self.latest_result.hand_landmarks:
                # Draw all landmarks for debug
                for lm in hand_landmarks:
                    gx = int(lm.x * rgb_w)
                    gy = int(lm.y * rgb_h)
                    cv2.circle(debug_img, (gx, gy), 2, (0, 255, 0), -1)

                for idx in self.FINGER_TIPS:
                    lm = hand_landmarks[idx]

                    # Landmark in projector space (denormalized)
                    proj_x = lm.x * rgb_w
                    proj_y = lm.y * rgb_h

                    # projector -> camera/rgb space
                    proj_point = np.array([[proj_x, proj_y]], dtype=np.float32)
                    camera_point = self._coordinate_transformer.projector_to_camera(proj_point)
                    cam_x = int(camera_point[0, 0])
                    cam_y = int(camera_point[0, 1])

                    # camera/rgb space -> depth space
                    depth_points = self._resolution_mapper.rgb_to_depth([(cam_x, cam_y)])
                    cx, cy = depth_points[0]

                    if not (0 <= cx < self._dmax_map.shape[1] and 0 <= cy < self._dmax_map.shape[0]):
                        continue

                    # Sample depth with small area to reduce sensor noise
                    roi_size = 1
                    z_roi = depth_frame[
                        max(0, cy - roi_size) : cy + roi_size + 1,
                        max(0, cx - roi_size) : cx + roi_size + 1,
                    ]
                    valid_z = z_roi[z_roi > 0]
                    current_z = int(np.median(valid_z)) if valid_z.size > 0 else 0

                    surface_z = int(self._dmax_map[cy, cx])
                    diff = surface_z - current_z

                    is_touching = -10 <= diff <= self.touch_threshold and current_z > 0
                    if is_touching:
                        touches_projector.append((proj_x, proj_y))

                    # Debug overlay
                    color = (0, 0, 255) if is_touching else (0, 255, 255)
                    cv2.circle(debug_img, (int(proj_x), int(proj_y)), 4, color, -1)
                    debug_text = (
                        "NO DATA (Z=0)"
                        if current_z == 0
                        else f"Z:{current_z} M:{surface_z} D:{diff}"
                    )
                    cv2.putText(
                        debug_img,
                        debug_text,
                        (int(proj_x) + 10, int(proj_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

        cv2.namedWindow("Kinect V2 - Livestream AI Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("Kinect V2 - Livestream AI Debug", debug_img)
        cv2.waitKey(1)

        return touches_projector