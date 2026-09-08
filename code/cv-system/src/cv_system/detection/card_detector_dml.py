"""YOLO card detection via PyTorch + torch-directml (AMD GPU on Windows)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops

from cv_system.detection.card_detector import CardDetection, CardDetector
from cv_system.gpu_config import dml_safe_no_grad, get_torch_device, patch_ultralytics_for_directml
from cv_system.transform import RgbImageTransformer


class DmlCardDetector(CardDetector):
    """
    YOLO .pt inference on DirectML (torch-directml).

    Ultralytics' high-level predict/track uses torch.inference_mode(), which fails on
    DirectML. This backend runs letterbox → forward (no_grad) → NMS on CPU → ByteTrack.
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
        if path.suffix.lower() != ".pt":
            raise ValueError(
                f"DmlCardDetector requires a .pt model, got {path.suffix}. "
                "Use CardDetector with .onnx for ONNX Runtime + DirectML."
            )

        patch_ultralytics_for_directml()
        self._device = get_torch_device()
        self._model: YOLO = YOLO(str(path))
        self._names: dict[int, str] = self._model.names
        self._model.model.to(self._device)
        self._model.model.eval()

        stride = max(int(self._model.model.stride.max()), 32)
        self._letterbox = LetterBox((640, 640), auto=False, stride=stride)
        self._imgsz = (640, 640)

        self._init_byte_tracker()
        self._use_onnx = False  # use _detect_pytorch branch in detect()

        print(f"  DmlCardDetector: YOLO on {self._device} (ByteTrack)")

    def _detect_pytorch(self, image: cv2.UMat) -> list[CardDetection]:
        image_np = image.get() if isinstance(image, cv2.UMat) else image
        orig_h, orig_w = image_np.shape[:2]

        im_lb = self._letterbox(image=image_np)
        im_chw = im_lb.transpose((2, 0, 1))[::-1]
        im_chw = np.ascontiguousarray(im_chw)
        tensor = torch.from_numpy(im_chw).to(self._device).float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        with dml_safe_no_grad():
            raw = self._model.model(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        raw = raw.cpu()

        detections_list = nms.non_max_suppression(
            raw,
            self._conf_threshold,
            self._iou_threshold,
        )
        pred = detections_list[0]
        if pred is None or len(pred) == 0:
            return self._track_detections(sv.Detections.empty())

        pred[:, :4] = ops.scale_boxes(tensor.shape[2:], pred[:, :4], (orig_h, orig_w))

        xyxy = pred[:, :4].numpy().astype(np.float32)
        confidences = pred[:, 4].numpy()
        class_ids = pred[:, 5].numpy().astype(int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
        )
        return self._track_detections(sv_detections)
