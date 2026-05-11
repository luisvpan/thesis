"""
Touch detector based on ESPOL paper.

"Fingertip Detection Approach on Depth Image Sequences for Interactive
Projection System" (Cadena et al., 2016)

Uses classical CV techniques: contours, K-means clustering, K-curvature
algorithm, IR correction, and hysteresis for touch stability.
No ML dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch
from cv_system.transform import DepthCoordinateTransformer


class Fingertip(NamedTuple):
    """Detected fingertip with associated points."""

    position: tuple[int, int]  # Ptest - fingertip position
    midpoint: tuple[int, int]  # Pmid - midpoint between k-neighbors
    contour_idx: int  # Index in the contour


@dataclass
class TouchState:
    """Hysteresis state for a single touch point."""

    is_touching: bool = False
    fingertip_id: int = 0


class ESPOLTouchDetector:
    """
    Touch detector based on ESPOL paper methodology.

    Coordinate flow:
        depth_frame → background subtraction → binary mask
            → morphological filtering → contour extraction
            → K-means arm/hand separation → K-curvature fingertip detection
            → IR correction → hysteresis touch detection
            → transform to projector space → TouchTracker
    """

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_transformer: DepthCoordinateTransformer,
        config,
        *,
        k_curvature: int = 10,
        angle_threshold: float = 90.0,
        min_contour_area: int = 500,
        max_contour_area: int = 15000,
        morph_kernel_size: int = 5,
        hysteresis_lower: int = 20,
        hysteresis_upper: int = 5,
        ir_roi_size: int = 10,
        show_debug: bool = False,
    ) -> None:
        """
        Initialize the ESPOL touch detector.

        Args:
            dmax_map: Calibrated maximum depth map (surface depth with offset).
            depth_transformer: Transformer for depth <-> projector mapping.
            config: Detection configuration.
            k_curvature: Number of neighbors for K-curvature algorithm.
            angle_threshold: Maximum angle (degrees) to consider as fingertip.
            min_contour_area: Minimum contour area to consider.
            max_contour_area: Maximum contour area to consider.
            morph_kernel_size: Kernel size for morphological operations.
            hysteresis_lower: Touch ON threshold (mm below dmax).
            hysteresis_upper: Touch OFF threshold (mm below dmax).
            ir_roi_size: Size of ROI for IR correction.
            show_debug: If True, display debug visualization windows.
        """
        self._dmax_map = dmax_map.astype(np.int32)
        self._depth_transformer = depth_transformer
        self._show_debug = show_debug

        # Algorithm parameters
        self._k = k_curvature
        self._angle_threshold = angle_threshold
        self._min_contour_area = min_contour_area
        self._max_contour_area = max_contour_area
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self._hysteresis_lower = hysteresis_lower
        self._hysteresis_upper = hysteresis_upper
        self._ir_roi_size = ir_roi_size

        # Hysteresis state per fingertip (by approximate position)
        self._touch_states: dict[int, TouchState] = {}

        # Touch tracker for persistent IDs
        self._touch_tracker = TouchTracker(
            debounce_frames=1,  # Minimal debounce, hysteresis handles stability
            touch_radius=20.0,
            lost_track_buffer=3,
        )

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[TrackedTouch], bool]:
        """
        Detect touches using ESPOL algorithm.

        Args:
            depth_frame: Raw depth frame (uint16, 424x512).
            rgb_frame: RGB frame (unused, for API compatibility).
            ir_frame: Optional IR frame for fingertip correction (uint16, 424x512).

        Returns:
            Tuple of (tracked_touches, hands_detected).
        """
        # rgb_frame is unused - ESPOL uses only depth + IR
        h, w = depth_frame.shape

        # Debug visualization
        debug_depth = None
        debug_ir = None
        if self._show_debug:
            debug_depth = cv2.normalize(
                depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            debug_depth = cv2.cvtColor(debug_depth, cv2.COLOR_GRAY2BGR)
            if ir_frame is not None:
                ir_8bit = np.clip(ir_frame / 16, 0, 255).astype(np.uint8)
                debug_ir = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)

        # Step 1: Background subtraction
        binary_mask = self._background_subtract(depth_frame)

        # Step 2: Morphological filtering
        binary_mask = self._morphological_filter(binary_mask)

        # Step 3: Find hand contours
        contours = self._find_hand_contours(binary_mask)
        hands_detected = len(contours) > 0

        if self._show_debug:
            cv2.drawContours(debug_depth, contours, -1, (0, 255, 0), 1)

        touches_projector: list[tuple[float, float]] = []

        for contour_idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Step 4: Separate arm from hand using K-means
            hand_points, arm_points, cm, ch = self._separate_arm_hand(contour)

            if self._show_debug:
                # Show contour info near its centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(debug_depth, f"#{contour_idx} A:{area:.0f}",
                                (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw centers
                if cm is not None and np.any(cm):
                    cv2.circle(debug_depth, (int(cm[0]), int(cm[1])), 6, (255, 0, 0), -1)
                if ch is not None and np.any(ch):
                    cv2.circle(debug_depth, (int(ch[0]), int(ch[1])), 6, (0, 0, 255), -1)

            if hand_points is None or len(hand_points) < 3:
                continue

            # Step 5: Detect fingertips using K-curvature
            fingertips = self._detect_fingertips_kcurvature(contour, cm, ch)

            if self._show_debug and fingertips:
                cv2.putText(debug_depth, f"Tips:{len(fingertips)}",
                            (int(cm[0]) - 20, int(cm[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            for fingertip in fingertips:
                tip_x, tip_y = fingertip.position
                mid_x, mid_y = fingertip.midpoint

                # Step 6: IR correction (if available)
                if ir_frame is not None:
                    corrected = self._correct_with_ir(
                        fingertip.position, fingertip.midpoint, ir_frame, debug_ir
                    )
                    tip_x, tip_y = corrected

                # Step 7: Check touch with hysteresis
                is_touch = self._check_touch_hysteresis(
                    (tip_x, tip_y), fingertip.midpoint, depth_frame
                )

                if is_touch:
                    # Transform to projector space
                    touch_point = np.array([[tip_x, tip_y]], dtype=np.float32)
                    proj_point = self._depth_transformer.camera_to_projector(touch_point)
                    touches_projector.append(
                        (float(proj_point[0, 0]), float(proj_point[0, 1]))
                    )

                    if self._show_debug:
                        cv2.circle(debug_depth, (tip_x, tip_y), 8, (0, 255, 0), -1)
                        cv2.line(
                            debug_depth,
                            (tip_x, tip_y),
                            (mid_x, mid_y),
                            (255, 255, 0),
                            2,
                        )
                else:
                    if self._show_debug:
                        cv2.circle(debug_depth, (tip_x, tip_y), 5, (0, 255, 255), 2)

        if self._show_debug:
            cv2.namedWindow("ESPOL - Depth", cv2.WINDOW_NORMAL)
            cv2.imshow("ESPOL - Depth", debug_depth)
            if debug_ir is not None:
                cv2.namedWindow("ESPOL - IR", cv2.WINDOW_NORMAL)
                cv2.imshow("ESPOL - IR", debug_ir)
            cv2.waitKey(1)

        # Track touches for persistent IDs
        tracked_touches = self._touch_tracker.update(touches_projector)

        return tracked_touches, hands_detected

    def _background_subtract(self, depth_frame: np.ndarray) -> np.ndarray:
        """
        Subtract background (dmax_map) to get objects above the table.

        Objects closer to the camera have lower depth values.
        We want pixels where: 0 < depth < dmax (objects above table).
        """
        # Valid depth and above the table surface
        mask = (depth_frame > 0) & (depth_frame < self._dmax_map)
        return mask.astype(np.uint8) * 255

    def _morphological_filter(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply erosion + dilation to remove salt-and-pepper noise."""
        # Erosion removes small noise
        eroded = cv2.erode(binary_mask, self._morph_kernel, iterations=1)
        # Dilation restores object size
        dilated = cv2.dilate(eroded, self._morph_kernel, iterations=1)
        return dilated

    def _find_hand_contours(self, binary_mask: np.ndarray) -> list:
        """Find and filter contours by area and shape."""
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Filter by area and aspect ratio
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self._min_contour_area <= area <= self._max_contour_area):
                continue

            # Filter by aspect ratio - hands/arms are elongated
            # Legs/body tend to be wider than tall
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1)

            # Arms are elongated (aspect > 1.5), reject very square/wide shapes
            if aspect < 1.3:
                continue

            filtered.append(contour)

        return filtered

    def _separate_arm_hand(
        self, contour: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
        """
        Separate arm from hand using K-means clustering (K=2).

        The cluster closer to the image center is assumed to be the hand.

        Returns:
            (hand_points, arm_points, center_of_mass, hand_center)
        """
        if len(contour) < 10:
            return None, None, np.array([0, 0]), np.array([0, 0])

        # Approximate contour to polygon (less aggressive to keep more vertices)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 4:
            # Use original contour points if approximation is too aggressive
            approx = contour

        # Get vertices
        vertices = approx.reshape(-1, 2).astype(np.float32)

        # Center of mass of entire contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, None, np.array([0, 0]), np.array([0, 0])
        cm = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])

        # K-means clustering to separate arm from hand
        if len(vertices) < 4:
            # Not enough vertices for K-means, use all as hand
            return vertices, np.array([]), cm, cm

        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vertices)
            centers = kmeans.cluster_centers_

            # Determine which cluster is "hand" (closer to center of mass)
            dist0 = np.linalg.norm(centers[0] - cm)
            dist1 = np.linalg.norm(centers[1] - cm)

            if dist0 < dist1:
                hand_label = 0
            else:
                hand_label = 1

            hand_points = vertices[labels == hand_label]
            arm_points = vertices[labels != hand_label]
            ch = centers[hand_label]  # Hand center

            return hand_points, arm_points, cm, ch
        except Exception:
            # K-means failed, use all vertices as hand
            return vertices, np.array([]), cm, cm

    def _detect_fingertips_kcurvature(
        self, contour: np.ndarray, cm: np.ndarray, ch: np.ndarray
    ) -> list[Fingertip]:
        """
        Detect fingertips using K-curvature algorithm.

        For each point Pi, calculate the angle between vectors:
        Pi→Pi+k and Pi→Pi-k

        If angle < threshold, it's a fingertip candidate.
        Additional validation: distance(CH-Ptest) < distance(CM-Ptest)
        """
        contour_points = contour.reshape(-1, 2)
        n = len(contour_points)

        if n < self._k * 2 + 1:
            return []

        fingertips = []
        angle_threshold_rad = np.deg2rad(self._angle_threshold)

        for i in range(n):
            # Get k neighbors forward and backward
            pi = contour_points[i]
            pi_plus_k = contour_points[(i + self._k) % n]
            pi_minus_k = contour_points[(i - self._k) % n]

            # Vectors from Pi to neighbors
            v1 = pi_plus_k - pi
            v2 = pi_minus_k - pi

            # Calculate angle between vectors
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)

            if len1 < 1 or len2 < 1:
                continue

            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            # Check if angle is sharp enough
            if angle < angle_threshold_rad:
                # Calculate Pmid (midpoint between neighbors)
                pmid = (pi_plus_k + pi_minus_k) / 2

                # Validate: the fingertip should point outward from hand center
                # distance(CM-Pmid) < distance(CM-Ptest) ensures tip is farther than base
                dist_cm_pmid = np.linalg.norm(cm - pmid)
                dist_cm_ptest = np.linalg.norm(cm - pi)

                # Relaxed validation: only check that tip is farther than midpoint
                if dist_cm_pmid < dist_cm_ptest:
                    fingertips.append(
                        Fingertip(
                            position=(int(pi[0]), int(pi[1])),
                            midpoint=(int(pmid[0]), int(pmid[1])),
                            contour_idx=i,
                        )
                    )

        # Filter fingertips: keep farthest from CM (actual fingers, not elbow)
        return self._filter_nearby_fingertips(fingertips, contour_points, cm)

    def _filter_nearby_fingertips(
        self, fingertips: list[Fingertip], contour_points: np.ndarray, cm: np.ndarray
    ) -> list[Fingertip]:
        """Filter fingertips: remove duplicates and keep only the farthest from CM."""
        if len(fingertips) == 0:
            return fingertips

        # Calculate distance from CM for each fingertip
        distances = []
        for ft in fingertips:
            dist = np.linalg.norm(np.array(ft.position) - cm)
            distances.append((dist, ft))

        # Sort by distance (farthest first)
        distances.sort(key=lambda x: -x[0])

        # Keep only fingertips that are in the outer 50% of distances
        if len(distances) > 1:
            max_dist = distances[0][0]
            min_dist = distances[-1][0]
            threshold = min_dist + (max_dist - min_dist) * 0.5

            distances = [(d, ft) for d, ft in distances if d >= threshold]

        # Remove duplicates that are too close
        filtered = []
        for dist, ft in distances:
            is_duplicate = False
            for existing in filtered:
                if np.linalg.norm(
                    np.array(ft.position) - np.array(existing.position)
                ) < 25:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(ft)

        return filtered

    def _correct_with_ir(
        self,
        fingertip: tuple[int, int],
        midpoint: tuple[int, int],
        ir_frame: np.ndarray,
        debug_ir: np.ndarray | None = None,
    ) -> tuple[int, int]:
        """
        Correct fingertip position using IR edge detection.

        Creates ROI around fingertip, finds Canny edges in IR,
        selects edge point most aligned with finger direction.
        """
        tip_x, tip_y = fingertip
        mid_x, mid_y = midpoint
        h, w = ir_frame.shape

        # ROI bounds
        half = self._ir_roi_size // 2
        x1 = max(0, tip_x - half)
        y1 = max(0, tip_y - half)
        x2 = min(w, tip_x + half + 1)
        y2 = min(h, tip_y + half + 1)

        if x2 - x1 < 3 or y2 - y1 < 3:
            return fingertip

        # Extract and process ROI
        ir_roi = ir_frame[y1:y2, x1:x2]
        ir_8bit = np.clip(ir_roi / 16, 0, 255).astype(np.uint8)
        ir_blurred = cv2.GaussianBlur(ir_8bit, (3, 3), 0)
        edges = cv2.Canny(ir_blurred, 50, 150)

        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) == 0:
            return fingertip

        # Direction from midpoint to tip
        dir_x = tip_x - mid_x
        dir_y = tip_y - mid_y
        dir_len = np.sqrt(dir_x**2 + dir_y**2)

        if dir_len < 1:
            return fingertip

        dir_x /= dir_len
        dir_y /= dir_len

        # Find edge point most aligned with finger direction
        best_point = fingertip
        best_score = -1.0

        for row, col in edge_points:
            ex, ey = x1 + col, y1 + row

            vec_x = ex - tip_x
            vec_y = ey - tip_y
            vec_len = np.sqrt(vec_x**2 + vec_y**2)

            if vec_len < 1:
                continue

            alignment = (vec_x * dir_x + vec_y * dir_y) / vec_len

            if alignment > 0.7:
                score = alignment - (vec_len / self._ir_roi_size) * 0.3
                if score > best_score:
                    best_score = score
                    best_point = (ex, ey)

        # Debug visualization
        if debug_ir is not None:
            cv2.rectangle(debug_ir, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.circle(debug_ir, fingertip, 3, (0, 0, 255), -1)
            if best_point != fingertip:
                cv2.circle(debug_ir, best_point, 3, (0, 255, 0), -1)

        return best_point

    def _check_touch_hysteresis(
        self,
        fingertip: tuple[int, int],
        midpoint: tuple[int, int],
        depth_frame: np.ndarray,
    ) -> bool:
        """
        Check if fingertip is touching using hysteresis.

        Ptouch = midpoint between fingertip and Pmid
        Prom = average depth of 5 points around Ptouch

        Touch ON when Prom < dmax - hysteresis_lower
        Touch OFF when Prom > dmax - hysteresis_upper
        """
        tip_x, tip_y = fingertip
        mid_x, mid_y = midpoint
        h, w = depth_frame.shape

        # Calculate Ptouch (midpoint between tip and mid)
        ptouch_x = (tip_x + mid_x) // 2
        ptouch_y = (tip_y + mid_y) // 2

        # Clamp to valid range
        ptouch_x = max(1, min(w - 2, ptouch_x))
        ptouch_y = max(1, min(h - 2, ptouch_y))

        # Get Prom (average of 5 points)
        depths = [
            depth_frame[ptouch_y, ptouch_x],
            depth_frame[ptouch_y - 1, ptouch_x],
            depth_frame[ptouch_y + 1, ptouch_x],
            depth_frame[ptouch_y, ptouch_x - 1],
            depth_frame[ptouch_y, ptouch_x + 1],
        ]
        valid_depths = [d for d in depths if d > 0]

        if not valid_depths:
            return False

        prom = np.mean(valid_depths)
        dmax_at_point = self._dmax_map[ptouch_y, ptouch_x]

        # Hysteresis state lookup (by grid position)
        grid_key = (ptouch_x // 20, ptouch_y // 20)  # 20px grid

        if grid_key not in self._touch_states:
            self._touch_states[grid_key] = TouchState()

        state = self._touch_states[grid_key]
        lower_threshold = dmax_at_point - self._hysteresis_lower
        upper_threshold = dmax_at_point - self._hysteresis_upper

        if not state.is_touching:
            # Need to go below lower threshold to activate
            if prom < lower_threshold:
                state.is_touching = True
        else:
            # Need to go above upper threshold to deactivate
            if prom > upper_threshold:
                state.is_touching = False

        return state.is_touching

    def close(self) -> None:
        """Release resources."""
        pass
