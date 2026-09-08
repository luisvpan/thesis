"""Card detector for YOLO models exported with end-to-end NMS."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from cv_system.detection.card_detector import CardDetection, CardDetector


class E2ECardDetector(CardDetector):
    """
    Card detector for YOLO models exported with end-to-end NMS.

    These models output [batch, max_detections, 6] where 6 = [x1, y1, x2, y2, conf, class_id].
    NMS is already applied by the model, so no post-processing NMS is needed.
    """

    def _detect_onnx(self, image: cv2.UMat) -> list[CardDetection]:
        """Run detection using ONNX Runtime for end-to-end models."""
        # Get target size from model input shape (handle dynamic shapes)
        _, _, target_h, target_w = self._input_shape
        if not isinstance(target_h, int) or target_h <= 0:
            target_h = 640
        if not isinstance(target_w, int) or target_w <= 0:
            target_w = 640

        # Convert UMat to numpy for ONNX preprocessing
        image_np = image.get() if isinstance(image, cv2.UMat) else image
        orig_h, orig_w = image_np.shape[:2]

        # Preprocess: BGR->RGB, resize, normalize
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (int(target_w), int(target_h)))
        normalized = resized.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(transposed, axis=0)  # Add batch dim

        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: batched})
        output = outputs[0]  # [batch, max_detections, 6]

        predictions = output[0]  # [max_detections, 6]

        # Filter out padding (zero or negative confidence entries)
        valid_mask = predictions[:, 4] > 0
        predictions = predictions[valid_mask]

        if len(predictions) == 0:
            return []

        # Extract: [x1, y1, x2, y2, confidence, class_id]
        boxes = predictions[:, :4].copy()
        confidences = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)

        # Filter by confidence threshold
        conf_mask = confidences >= self._conf_threshold
        boxes = boxes[conf_mask]
        confidences = confidences[conf_mask]
        class_ids = class_ids[conf_mask]

        if len(boxes) == 0:
            return []

        # Scale boxes from model input size to original image size
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        boxes[:, 0] *= scale_x  # x1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 3] *= scale_y  # y2

        # Create supervision Detections and update tracker (no NMS needed)
        sv_detections = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=confidences,
            class_id=class_ids,
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
