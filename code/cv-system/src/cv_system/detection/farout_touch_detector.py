"""
FarOut Touch Detector - Based on Shen et al. 2021.

Implements touch detection using the "finger denting" phenomenon where
touching fingers appear as dark spots (farther from sensor) due to
multipath reflections at longer ranges (1.5-3.5m).

Reference:
    Shen, V., Spann, J., & Harrison, C. (2021). FarOut Touch: Extending the
    Range of ad hoc Touch Sensing with Depth Cameras. In Symposium on
    Spatial User Interaction (SUI '21).

Note: This approach works best at 1.5-3.5m range. At shorter ranges
(< 1.5m), the denting effect is weak and detection may be unreliable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FarOutTouch:
    """Represents a detected touch point from FarOut detector."""

    x: float  # World X coordinate (mm)
    y: float  # World Y coordinate (mm)
    z: float  # Depth value at touch point (mm)
    touching: bool  # Whether finger is touching surface
    confidence: float  # Detection confidence (0-1)
    pixel_x: int  # Image X coordinate
    pixel_y: int  # Image Y coordinate


class FarOutTouchDetector:
    """
    FarOut Touch detector implementation.

    Uses temporal averaging, background subtraction, hand finding,
    and "dent" detection to locate touch points. Designed for
    longer range (1.5-3.5m) but can work at shorter ranges with
    inverted logic.
    """

    # Temporal averaging
    TEMPORAL_FRAMES = 5

    # Hand finding thresholds (distance from surface in mm)
    WRIST_DIST_MIN = 30  # 3cm
    WRIST_DIST_MAX = 120  # 12cm
    ARM_DIST_MIN = 80  # 8cm
    ARM_DIST_MAX = 250  # 25cm

    # Touch detection
    TOUCH_DEPTH_THRESHOLD = 10  # mm - depth threshold for touch
    HAND_PATCH_SIZE = 80  # pixels
    MIN_BLOB_SIZE = 36  # 6x6 pixels minimum blob area
    TOP_PIXELS = 50  # Number of extreme pixels to consider

    # Convergence for stable detection
    CONVERGENCE_FRAMES = 5
    CONVERGENCE_TOLERANCE = 10  # pixels

    # Calibration
    CALIBRATION_FRAMES = 20

    def __init__(
        self,
        dmax_map: np.ndarray,
        depth_transformer: Any,
        config: Any,
        *,
        show_debug: bool = False,
        use_denting: bool = True,
    ) -> None:
        """
        Initialize FarOut Touch detector.

        Args:
            dmax_map: Maximum depth map (background reference)
            depth_transformer: Coordinate transformer for depth-to-world
            config: Configuration object
            show_debug: Whether to show debug windows
            use_denting: If True, look for "dent" (FarOut style at long range).
                        If False, look for "bump" (short range style).
        """
        self._dmax_map = dmax_map.astype(np.float32)
        self._depth_transformer = depth_transformer
        self._config = config
        self._show_debug = show_debug
        self._use_denting = use_denting
        self._shape = dmax_map.shape

        # Temporal buffer for averaging
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.TEMPORAL_FRAMES)

        # Background model (super-resolved from first N frames)
        self._background: np.ndarray | None = None
        self._bg_frames: list[np.ndarray] = []
        self._calibrating = True

        # Touch state and history for convergence
        self._last_touch: FarOutTouch | None = None
        self._touch_history: deque[tuple[float, float, bool]] = deque(
            maxlen=self.CONVERGENCE_FRAMES
        )

    def detect(
        self,
        depth_frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> tuple[list[FarOutTouch], bool]:
        """
        Detect touches using FarOut Touch approach.

        Args:
            depth_frame: Current depth frame (uint16, mm values)
            rgb_frame: Optional RGB frame (unused)
            ir_frame: Optional IR frame (unused - FarOut uses depth only)

        Returns:
            touches: List of detected FarOutTouch objects
            has_touch: True if any finger is touching
        """
        # 1. Add to temporal buffer
        self._frame_buffer.append(depth_frame.astype(np.float32))

        if len(self._frame_buffer) < self.TEMPORAL_FRAMES:
            if self._show_debug:
                self._draw_status(
                    f"Buffering: {len(self._frame_buffer)}/{self.TEMPORAL_FRAMES}"
                )
            return [], False

        # 2. Temporal averaging (reduces noise by factor of sqrt(N))
        avg_frame = np.mean(np.stack(list(self._frame_buffer)), axis=0)

        # 3. Calibration phase (capture background without user)
        if self._calibrating:
            self._bg_frames.append(avg_frame.copy())
            if len(self._bg_frames) >= self.CALIBRATION_FRAMES:
                self._background = np.mean(np.stack(self._bg_frames), axis=0)
                self._calibrating = False
                self._bg_frames = []  # Free memory
            if self._show_debug:
                pct = len(self._bg_frames) / self.CALIBRATION_FRAMES * 100
                self._draw_status(f"Calibrating: {pct:.0f}%")
            return [], False

        assert self._background is not None

        # 4. Background subtraction
        # diff > 0 means object is closer to camera than background
        diff = self._background - avg_frame

        # 5. Find hand region using wrist/forearm contours
        hand_patch, patch_offset = self._find_hand_region(diff)

        if hand_patch is None:
            self._touch_history.clear()
            if self._show_debug:
                self._draw_debug(diff, None, None, None)
            return [], False

        # 6. DCT denoising (simplified as Gaussian blur)
        denoised = self._dct_denoise(hand_patch)

        # 7. Find touch - either "dent" (long range) or "bump" (short range)
        touch_point = self._find_touch(denoised, patch_offset, avg_frame)

        if self._show_debug:
            self._draw_debug(diff, hand_patch, denoised, touch_point)

        if touch_point is not None and touch_point.touching:
            return [touch_point], True

        return [], False

    def _find_hand_region(
        self, diff: np.ndarray
    ) -> tuple[np.ndarray | None, tuple[int, int] | None]:
        """
        Find hand using wrist/forearm contours following FarOut paper.

        Returns 80x80 patch around probable finger location by
        extrapolating from arm→wrist direction.

        Args:
            diff: Background-subtracted depth difference

        Returns:
            patch: Hand region patch or None
            offset: (x, y) offset of patch in original image
        """
        h, w = diff.shape

        # Find wrist region (3-12cm from surface)
        wrist_mask = (diff > self.WRIST_DIST_MIN) & (diff < self.WRIST_DIST_MAX)
        wrist_mask = wrist_mask.astype(np.uint8)

        # Find arm region (8-25cm from surface)
        arm_mask = (diff > self.ARM_DIST_MIN) & (diff < self.ARM_DIST_MAX)
        arm_mask = arm_mask.astype(np.uint8)

        # Find contours
        wrist_contours, _ = cv2.findContours(
            wrist_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        arm_contours, _ = cv2.findContours(
            arm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not wrist_contours or not arm_contours:
            return None, None

        # Get largest contours
        wrist_cnt = max(wrist_contours, key=cv2.contourArea)
        arm_cnt = max(arm_contours, key=cv2.contourArea)

        wrist_area = cv2.contourArea(wrist_cnt)
        arm_area = cv2.contourArea(arm_cnt)

        if wrist_area < 50 or arm_area < 100:
            return None, None

        # Get centroids
        wrist_M = cv2.moments(wrist_cnt)
        arm_M = cv2.moments(arm_cnt)

        if wrist_M["m00"] == 0 or arm_M["m00"] == 0:
            return None, None

        wrist_cx = int(wrist_M["m10"] / wrist_M["m00"])
        wrist_cy = int(wrist_M["m01"] / wrist_M["m00"])
        arm_cx = int(arm_M["m10"] / arm_M["m00"])
        arm_cy = int(arm_M["m01"] / arm_M["m00"])

        # Extrapolate finger position beyond wrist
        dx = wrist_cx - arm_cx
        dy = wrist_cy - arm_cy
        length = np.sqrt(dx * dx + dy * dy)

        if length < 10:
            return None, None

        # Extend 50% beyond wrist to find fingertip region
        finger_cx = int(wrist_cx + dx * 0.5)
        finger_cy = int(wrist_cy + dy * 0.5)

        # Extract patch
        half = self.HAND_PATCH_SIZE // 2
        x1 = max(0, finger_cx - half)
        y1 = max(0, finger_cy - half)
        x2 = min(w, finger_cx + half)
        y2 = min(h, finger_cy + half)

        patch = diff[y1:y2, x1:x2]

        if patch.size == 0 or patch.shape[0] < 20 or patch.shape[1] < 20:
            return None, None

        return patch, (x1, y1)

    def _dct_denoise(self, patch: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """
        DCT-based denoising as described in FarOut Touch paper.

        Simplified implementation using Gaussian blur which achieves
        similar frequency-domain filtering effect.

        Args:
            patch: Input patch to denoise
            sigma: Gaussian blur sigma

        Returns:
            Denoised patch
        """
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            return patch

        # Simple Gaussian blur as approximation of DCT denoising
        # Full DCT implementation would be more complex
        denoised = cv2.GaussianBlur(patch.astype(np.float32), (5, 5), sigma)

        return denoised

    def _find_touch(
        self,
        patch: np.ndarray,
        offset: tuple[int, int],
        avg_frame: np.ndarray,
    ) -> FarOutTouch | None:
        """
        Find touch point using "dent" or "bump" detection.

        At long range (1.5-3.5m): FarOut looks for "dent" - pixels that
        appear FARTHER than background due to multipath reflections.

        At short range (<1.5m): We can optionally look for "bump" -
        pixels that appear CLOSER than background.

        Args:
            patch: Denoised hand patch
            offset: (x, y) offset of patch in full image
            avg_frame: Temporally averaged depth frame

        Returns:
            FarOutTouch object or None
        """
        if patch is None:
            return None

        ox, oy = offset
        flat = patch.flatten()

        if len(flat) < self.TOP_PIXELS:
            return None

        if self._use_denting:
            # FarOut style: find farthest pixels (most negative diff = "dent")
            sorted_indices = np.argsort(flat)
            extreme_indices = sorted_indices[: self.TOP_PIXELS]
        else:
            # Short range style: find closest pixels (most positive diff = "bump")
            sorted_indices = np.argsort(flat)[::-1]
            extreme_indices = sorted_indices[: self.TOP_PIXELS]

        # Create mask of extreme pixels
        mask = np.zeros(patch.shape, dtype=np.uint8)
        for idx in extreme_indices:
            y_local = idx // patch.shape[1]
            x_local = idx % patch.shape[1]
            mask[y_local, x_local] = 255

        # Find blobs in the extreme pixels
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter by size and find the one farthest from patch center
        # (most likely fingertip extends away from wrist)
        valid_blobs: list[tuple[int, int, float, float]] = []
        patch_center = (patch.shape[1] // 2, patch.shape[0] // 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.MIN_BLOB_SIZE:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.sqrt(
                        (cx - patch_center[0]) ** 2 + (cy - patch_center[1]) ** 2
                    )
                    valid_blobs.append((cx, cy, dist, area))

        if not valid_blobs:
            return None

        # Select blob farthest from center (most likely fingertip)
        best = max(valid_blobs, key=lambda b: b[2])
        cx, cy, dist_from_center, area = best

        # Convert back to full image coordinates
        img_x = ox + cx
        img_y = oy + cy

        # Get depth value at touch point
        touch_diff = patch[cy, cx]

        # Determine if touching based on detection mode
        if self._use_denting:
            # FarOut: touch = when dent is present (negative diff)
            is_touching = touch_diff < -self.TOUCH_DEPTH_THRESHOLD
        else:
            # Short range: touch = when bump is very close to surface
            is_touching = 0 < touch_diff < self.TOUCH_DEPTH_THRESHOLD

        # Add to history for convergence check
        self._touch_history.append((float(img_x), float(img_y), is_touching))

        # Check convergence (need CONVERGENCE_FRAMES consistent frames)
        if len(self._touch_history) < self.CONVERGENCE_FRAMES:
            return None

        xs = [t[0] for t in self._touch_history]
        ys = [t[1] for t in self._touch_history]
        touches = [t[2] for t in self._touch_history]

        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        median_x = np.median(xs)
        median_y = np.median(ys)

        # If mean and median diverge, detection is unstable
        if (
            abs(mean_x - median_x) > self.CONVERGENCE_TOLERANCE
            or abs(mean_y - median_y) > self.CONVERGENCE_TOLERANCE
        ):
            return None

        # Majority vote for touching state
        final_touching = sum(touches) > len(touches) // 2

        # Get depth at final position
        final_img_x = int(mean_x)
        final_img_y = int(mean_y)

        if 0 <= final_img_y < avg_frame.shape[0] and 0 <= final_img_x < avg_frame.shape[1]:
            depth_at_touch = float(avg_frame[final_img_y, final_img_x])
        else:
            depth_at_touch = 0.0

        # Convert to world coordinates
        world_x, world_y = self._depth_transformer.depth_to_world(
            final_img_x, final_img_y, depth_at_touch
        )

        return FarOutTouch(
            x=world_x,
            y=world_y,
            z=depth_at_touch,
            touching=final_touching,
            confidence=min(area / 100.0, 1.0),
            pixel_x=final_img_x,
            pixel_y=final_img_y,
        )

    def _draw_status(self, text: str) -> None:
        """Draw status message during calibration/buffering."""
        h, w = self._shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            img,
            text,
            (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "FarOut Touch",
            (w // 2 - 70, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
        )
        cv2.namedWindow("FarOut - Status", cv2.WINDOW_NORMAL)
        cv2.imshow("FarOut - Status", img)
        cv2.waitKey(1)

    def _draw_debug(
        self,
        diff: np.ndarray,
        hand_patch: np.ndarray | None,
        denoised: np.ndarray | None,
        touch: FarOutTouch | None,
    ) -> None:
        """Draw debug visualization windows."""
        h, w = self._shape

        # Diff visualization (normalized to -100 to +100 mm range)
        diff_vis = np.clip((diff + 100) / 200 * 255, 0, 255).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

        # Draw touch point if detected
        if touch is not None:
            color = (0, 255, 0) if touch.touching else (0, 0, 255)
            cv2.circle(diff_color, (touch.pixel_x, touch.pixel_y), 10, color, 2)
            label = f"z={touch.z:.0f}mm"
            if touch.touching:
                label += " TOUCH"
            cv2.putText(
                diff_color,
                label,
                (touch.pixel_x + 15, touch.pixel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Add mode indicator
        mode = "Denting (long range)" if self._use_denting else "Bump (short range)"
        cv2.putText(
            diff_color,
            f"Mode: {mode}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.namedWindow("FarOut - Diff", cv2.WINDOW_NORMAL)
        cv2.imshow("FarOut - Diff", diff_color)

        # Hand patch visualization if available
        if hand_patch is not None:
            # Normalize patch to -50 to +50 mm range for better visibility
            patch_vis = np.clip((hand_patch + 50) / 100 * 255, 0, 255).astype(np.uint8)
            patch_color = cv2.applyColorMap(patch_vis, cv2.COLORMAP_JET)

            # Resize for visibility
            scale = 4
            patch_large = cv2.resize(
                patch_color,
                (patch_color.shape[1] * scale, patch_color.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.namedWindow("FarOut - Hand Patch", cv2.WINDOW_NORMAL)
            cv2.imshow("FarOut - Hand Patch", patch_large)

        cv2.waitKey(1)

    def reset_calibration(self) -> None:
        """Reset calibration to recapture background."""
        self._calibrating = True
        self._background = None
        self._bg_frames = []
        self._frame_buffer.clear()
        self._touch_history.clear()

    def set_detection_mode(self, use_denting: bool) -> None:
        """
        Switch between denting (long range) and bump (short range) modes.

        Args:
            use_denting: If True, look for "dent" (pixels farther than background).
                        If False, look for "bump" (pixels closer than background).
        """
        self._use_denting = use_denting
        self._touch_history.clear()
