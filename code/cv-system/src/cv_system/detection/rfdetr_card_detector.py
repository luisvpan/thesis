"""Card detector for RF-DETR models exported to ONNX."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from cv_system.detection.card_detector import CardDetection, CardDetector


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply softmax along the specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class RFDETRCardDetector(CardDetector):
    """
    Card detector for RF-DETR models exported to ONNX.

    RF-DETR outputs two tensors:
    - dets: [batch, N, 4] - bounding boxes in cxcywh normalized format (0-1)
    - labels: [batch, N, num_classes] - class logits (requires softmax)

    Input resolution varies by model (nano=384, small=560, etc.) and is read from the model.
    """

    def _detect_onnx(self, image: cv2.UMat) -> list[CardDetection]:
        """Run detection using ONNX Runtime for RF-DETR models."""
        # Get target size from model input shape
        _, _, target_h, target_w = self._input_shape
        if not isinstance(target_h, int) or target_h <= 0:
            target_h = 384  # Default for RF-DETR nano
        if not isinstance(target_w, int) or target_w <= 0:
            target_w = 384

        # Convert UMat to numpy for ONNX preprocessing
        image_np = image.get() if isinstance(image, cv2.UMat) else image
        orig_h, orig_w = image_np.shape[:2]

        # Preprocess: BGR->RGB, resize, normalize
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (int(target_w), int(target_h)))
        normalized = resized.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(transposed, axis=0)  # Add batch dim

        # Run inference - RF-DETR has two outputs: dets and labels
        outputs = self._session.run(self._output_names, {self._input_name: batched})

        # Find dets and labels outputs by name or position
        dets_idx = next((i for i, name in enumerate(self._output_names) if "det" in name.lower()), 0)
        labels_idx = next((i for i, name in enumerate(self._output_names) if "label" in name.lower()), 1)

        dets = outputs[dets_idx][0]      # [N, 4] - cxcywh normalized
        labels = outputs[labels_idx][0]  # [N, num_classes] - logits

        # Apply softmax to get class probabilities
        probs = softmax(labels, axis=-1)

        # Get best class and confidence for each detection
        class_ids = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # Filter by confidence threshold
        mask = confidences >= self._conf_threshold
        dets = dets[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(dets) == 0:
            return []

        # Convert cxcywh normalized (0-1) to xyxy in original image pixels
        cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]

        # Scale to original image size
        cx = cx * orig_w
        cy = cy * orig_h
        w = w * orig_w
        h = h * orig_h

        # Convert to corner format
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Create supervision Detections and update tracker
        sv_detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences.astype(np.float32),
            class_id=class_ids.astype(int),
        )
        tracked = self._tracker.update_with_detections(sv_detections)

        # Build CardDetection objects with track IDs
        detections: list[CardDetection] = []
        for i in range(len(tracked)):
            label = self._names.get(int(tracked.class_id[i]), str(tracked.class_id[i]))
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
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

        return detections
