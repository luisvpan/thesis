"""Card detection with YOLO on RGB mapped to projector (bird) view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cv_system.transform import RgbImageTransformer


@dataclass(frozen=True)
class CardDetection:
    """Single detection in projector image coordinates."""

    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class CardDetector:
    """
    Runs YOLO on the RGB frame warped to projector space (same input space as MediaPipe in TouchDetector).

    Depth is not used.
    """

    def __init__(self, rgb_image_transformer: RgbImageTransformer, model_path: str | Path) -> None:
        self._image_transformer = rgb_image_transformer
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {path.resolve()}")
        self._model = YOLO(str(path))

    def detect(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, list[CardDetection]]:
        """
        Args:
            rgb_frame: Raw BGR frame from the camera (uint8).

        Returns:
            Annotated BGR image in projector resolution and list of detections.
        """
        rgb_float = rgb_frame.astype(np.float32) / 255.0
        rgb_bird = self._image_transformer.camera_to_projector(rgb_float)
        rgb_bird_uint8 = (np.clip(rgb_bird, 0.0, 1.0) * 255).astype(np.uint8)

        results = self._model.predict(rgb_bird_uint8, verbose=False)
        annotated = rgb_bird_uint8.copy()
        detections: list[CardDetection] = []

        names = self._model.names
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                detections.append(
                    CardDetection(
                        class_id=cls_id,
                        label=label,
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
                self._draw_detection(annotated, label, conf, x1, y1, x2, y2)

        return annotated, detections

    @staticmethod
    def _draw_detection(
        img: np.ndarray,
        label: str,
        conf: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 220, 0), 2)
        pct = conf * 100.0
        text = f"{label} {pct:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        x1i, y1i = p1[0], p1[1]
        y_text_baseline = y1i - pad
        y_text_top = y1i - th - baseline - pad
        if y_text_top < 0:
            y_text_baseline = y1i + th + baseline + pad
        cv2.rectangle(
            img,
            (x1i, y_text_baseline - th - baseline - 2),
            (x1i + tw + pad, y_text_baseline + 2),
            (0, 220, 0),
            -1,
        )
        cv2.putText(
            img,
            text,
            (x1i + 2, y_text_baseline),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
