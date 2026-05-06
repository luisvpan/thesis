"""Card detection with YOLO on RGB mapped to projector (bird) view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import supervision as sv

from cv_system.transform import RgbImageTransformer

if TYPE_CHECKING:
    import onnxruntime as ort
    from ultralytics import YOLO


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
    track_id: int = -1  # -1 means no tracking (e.g., ONNX backend)


class CardDetector:
    """
    Runs YOLO on the RGB frame warped to projector space.

    Supports two backends:
    - PyTorch (.pt): Uses ultralytics YOLO directly (CPU or CUDA if available)
    - ONNX (.onnx): Uses ONNX Runtime with DirectML for AMD GPU acceleration
    """

    def __init__(
        self,
        rgb_image_transformer: RgbImageTransformer,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._image_transformer = rgb_image_transformer
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path.resolve()}")

        self._use_onnx = path.suffix.lower() == ".onnx"

        if self._use_onnx:
            self._init_onnx(path)
        else:
            self._init_pytorch(path)

    def _init_pytorch(self, path: Path) -> None:
        """Initialize PyTorch/ultralytics backend."""
        from ultralytics import YOLO

        self._model: YOLO = YOLO(str(path))
        self._names: dict[int, str] = self._model.names

    def _init_onnx(self, path: Path) -> None:
        """Initialize ONNX Runtime with DirectML backend."""
        import onnxruntime as ort

        # Log available providers
        available = ort.get_available_providers()
        print(f"  ONNX available providers: {available}")

        # Request DirectML with CPU fallback
        providers = [
            ("DmlExecutionProvider", {"device_id": 1}),
            "CPUExecutionProvider",
        ]
        self._session: ort.InferenceSession = ort.InferenceSession(
            str(path), providers=providers
        )

        # Get input details
        input_info = self._session.get_inputs()[0]
        self._input_name: str = input_info.name
        self._input_shape: tuple[int, ...] = tuple(input_info.shape)  # [1, 3, H, W]

        # Get output details
        self._output_names: list[str] = [o.name for o in self._session.get_outputs()]

        # Load class names from ONNX metadata
        self._names = self._load_onnx_class_names(path)

        # Log which provider is ACTUALLY being used
        actual = self._session.get_providers()
        print(f"  ONNX session providers: {actual}")
        if "DmlExecutionProvider" in actual:
            print("  DirectML GPU acceleration active")
        else:
            print("  WARNING: DirectML not available, using CPU fallback!")

        # Initialize ByteTrack for persistent object IDs
        # Tuned for card tracking with hand movement
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.20,  # Lower = easier to create tracks
            lost_track_buffer=45,             # ~1.5s at 30fps to re-find lost tracks
            minimum_matching_threshold=0.5,   # Lower IoU = tolerate more movement
            frame_rate=30,
        )

    def _load_onnx_class_names(self, path: Path) -> dict[int, str]:
        """Load class names from ONNX model metadata."""
        metadata = self._session.get_modelmeta().custom_metadata_map

        # ultralytics stores names as JSON in metadata
        if "names" in metadata:
            names_json = metadata["names"]
            # Format: "{0: 'class0', 1: 'class1', ...}"
            # Parse as Python dict (it's actually Python repr, not JSON)
            try:
                # Try JSON first
                names = json.loads(names_json.replace("'", '"'))
                return {int(k): v for k, v in names.items()}
            except json.JSONDecodeError:
                # Fallback: eval (safe for this format)
                names = eval(names_json)  # noqa: S307
                return {int(k): v for k, v in names.items()}

        # Fallback: generic class names
        return {i: f"class_{i}" for i in range(100)}

    def detect(self, rgb_bird: cv2.UMat) -> tuple[np.ndarray, list[CardDetection]]:
        """
        Detect cards in the pre-transformed bird view image.

        Args:
            rgb_bird: BGR image as UMat already transformed to projector space.

        Returns:
            Annotated BGR image (numpy) in projector resolution and list of detections.
        """
        # Run detection
        if self._use_onnx:
            detections = self._detect_onnx(rgb_bird)
        else:
            detections = self._detect_pytorch(rgb_bird)

        # Draw detections on numpy copy
        annotated = rgb_bird.get().copy()
        for d in detections:
            self._draw_detection(
                annotated, d.label, d.confidence, d.x1, d.y1, d.x2, d.y2
            )

        return annotated, detections

    def _detect_pytorch(self, image: cv2.UMat) -> list[CardDetection]:
        """Run detection with tracking using PyTorch/ultralytics backend."""
        # ultralytics requires numpy array
        image_np = image.get()
        results = self._model.track(image_np, persist=True, verbose=False)
        detections: list[CardDetection] = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self._names.get(cls_id, str(cls_id))
                track_id = int(box.id[0]) if box.id is not None else -1
                detections.append(
                    CardDetection(
                        class_id=cls_id,
                        label=label,
                        confidence=conf,
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        track_id=track_id,
                    )
                )

        return detections

    def _detect_onnx(self, image: cv2.UMat) -> list[CardDetection]:
        """Run detection using ONNX Runtime with DirectML."""
        # Get target size from model input shape (handle dynamic shapes)
        _, _, target_h, target_w = self._input_shape
        # ONNX models may have dynamic shapes (-1 or strings), default to 640
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

        # Parse YOLOv8 output format: [1, 4+num_classes, num_predictions]
        # Transpose to [1, num_predictions, 4+num_classes]
        output = outputs[0]
        # Debug: print output shape on first call
        if not hasattr(self, "_debug_printed"):
            print(f"  ONNX output shape: {output.shape}")
            self._debug_printed = True

        if output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        predictions = output[0]  # Remove batch dim: [num_predictions, 4+num_classes]

        # Extract boxes and class scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        class_scores = predictions[:, 4:]

        # Debug: check if scores are already probabilities or raw logits
        if not hasattr(self, "_debug_scores"):
            print(f"  Class scores range: min={class_scores.min():.3f}, max={class_scores.max():.3f}")
            self._debug_scores = True

        # Only apply sigmoid if scores look like logits (outside 0-1 range)
        if class_scores.min() < 0 or class_scores.max() > 1:
            class_scores = 1 / (1 + np.exp(-class_scores))

        # Get best class for each prediction
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Debug: print max confidence on first few frames
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 5:
            print(f"  Max confidence: {confidences.max():.3f}, threshold: {self._conf_threshold}")
            self._debug_count += 1

        # Filter by confidence
        mask = confidences >= self._conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert from center format to corner format
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # Scale boxes back to original image size
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y

        # Apply NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            self._conf_threshold,
            self._iou_threshold,
        )

        if len(indices) == 0:
            return []

        # Filter arrays by NMS indices
        nms_indices = [idx[0] if isinstance(idx, (list, np.ndarray)) else idx for idx in indices]
        nms_boxes = boxes_for_nms[nms_indices]
        nms_confidences = confidences[nms_indices]
        nms_class_ids = class_ids[nms_indices]

        # Create supervision Detections and update tracker
        sv_detections = sv.Detections(
            xyxy=nms_boxes,
            confidence=nms_confidences,
            class_id=nms_class_ids.astype(int),
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
