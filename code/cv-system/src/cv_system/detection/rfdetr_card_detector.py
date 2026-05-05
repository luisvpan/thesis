"""RF-DETR ONNX card detector (DirectML / CPU fallback)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

from cv_system.detection.card_detector import CardDetection, CardDetector
from cv_system.transform import RgbImageTransformer


class RFDETRCardDetector(CardDetector):
    """
    Runs RF-DETR ONNX (inference_model.onnx) on the RGB frame.

    Drop-in replacement for CardDetector — same constructor signature.

    RF-DETR ONNX output format (differs completely from YOLO):
      - input  'input'  [1, 3, H, W]   float32, values in [0, 1]
      - output 'dets'   [1, Q, 4]      normalized (cx, cy, w, h) in [0, 1]
      - output 'labels' [1, Q, C]      per-class logits → sigmoid for confidence

    where Q = 300 DETR queries, C = num_classes.

    Class names are NOT embedded in the ONNX metadata. They are loaded
    automatically from data.yaml / data.yml placed next to the model file,
    or supplied via the optional class_names= keyword argument.
    """

    def __init__(
        self,
        rgb_image_transformer: RgbImageTransformer,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        *,
        class_names: dict[int, str] | None = None,
    ) -> None:
        # Store before super().__init__ calls _init_onnx → _load_onnx_class_names
        self._provided_class_names = class_names
        super().__init__(
            rgb_image_transformer,
            model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    # ------------------------------------------------------------------ #
    #  Overrides                                                           #
    # ------------------------------------------------------------------ #

    def _load_onnx_class_names(self, path: Path) -> dict[int, str]:
        """Load class names from data.yaml next to the model, or from class_names=."""
        if self._provided_class_names is not None:
            return self._provided_class_names

        for yaml_name in ("data.yaml", "data.yml"):
            yaml_path = Path(path).parent / yaml_name
            if yaml_path.exists():
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, list):
                    return {i: n for i, n in enumerate(names)}
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}

        raise FileNotFoundError(
            f"RF-DETR ONNX has no embedded class names. "
            f"Place data.yaml next to {path} or pass class_names={{...}} explicitly."
        )

    def _detect_onnx(self, image: cv2.UMat) -> list[CardDetection]:
        """Run RF-DETR inference and decode (cx,cy,w,h) + logit outputs."""
        _, _, target_h, target_w = self._input_shape
        orig_h, orig_w = image.get().shape[:2]

        # --- Preprocess (same pipeline as YOLO) ---
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (target_w, target_h)).get()
        normalized = resized.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

        # --- Inference ---
        dets_raw, labels_raw = self._session.run(
            self._output_names, {self._input_name: batched}
        )
        # dets_raw:   [1, Q, 4]   (cx, cy, w, h) normalized
        # labels_raw: [1, Q, C]   logits

        boxes_cxcywh = dets_raw[0]   # [Q, 4]
        logits = labels_raw[0]       # [Q, C]

        # --- Decode class and confidence via sigmoid ---
        probs = 1.0 / (1.0 + np.exp(-logits))   # sigmoid, [Q, C]
        class_ids = np.argmax(probs, axis=1)      # [Q]
        confidences = np.max(probs, axis=1)       # [Q]

        # --- Confidence filter ---
        mask = confidences >= self._conf_threshold
        if not mask.any():
            self._tracker.update_with_detections(
                self._empty_sv_detections()
            )
            return self._collect_lost_detections()

        boxes_cxcywh = boxes_cxcywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # --- Convert (cx,cy,w,h) normalized → (x1,y1,x2,y2) pixel ---
        cx = np.clip(boxes_cxcywh[:, 0], 0.0, 1.0) * orig_w
        cy = np.clip(boxes_cxcywh[:, 1], 0.0, 1.0) * orig_h
        bw = np.clip(boxes_cxcywh[:, 2], 0.0, 1.0) * orig_w
        bh = np.clip(boxes_cxcywh[:, 3], 0.0, 1.0) * orig_h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # --- NMS (suppress duplicate DETR queries for the same object) ---
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self._conf_threshold,
            self._iou_threshold,
        )
        if len(indices) == 0:
            self._tracker.update_with_detections(self._empty_sv_detections())
            return self._collect_lost_detections()

        nms_idx = [
            idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            for idx in indices
        ]
        boxes_xyxy = boxes_xyxy[nms_idx]
        confidences = confidences[nms_idx]
        class_ids = class_ids[nms_idx]

        # --- ByteTrack ---
        sv_det = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=confidences,
            class_id=class_ids.astype(int),
        )
        tracked = self._tracker.update_with_detections(sv_det)

        detections: list[CardDetection] = []
        for i in range(len(tracked)):
            label = self._names.get(int(tracked.class_id[i]), str(tracked.class_id[i]))
            track_id = (
                int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            )
            detections.append(
                CardDetection(
                    class_id=int(tracked.class_id[i]),
                    label=label,
                    confidence=float(tracked.confidence[i]),
                    x1=float(tracked.xyxy[i, 0]),
                    y1=float(tracked.xyxy[i, 1]),
                    x2=float(tracked.xyxy[i, 2]),
                    y2=float(tracked.xyxy[i, 3]),
                    track_id=track_id,
                )
            )

        detections.extend(self._collect_lost_detections())
        return detections

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_sv_detections() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )
