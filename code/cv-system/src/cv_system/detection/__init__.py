"""
Detection layer for touch and interaction identification.

This module provides touch detection functionality that compares depth frames
against a calibrated dmax_map using temporal filtering via a ring buffer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cv_system.detection.card_detector import CardDetector, CardDetection
from cv_system.detection.e2e_card_detector import E2ECardDetector
from cv_system.detection.rfdetr_card_detector import RFDETRCardDetector
from cv_system.detection.depth_only_touch_detector import DepthOnlyTouchDetector
from cv_system.detection.touch_detector import TouchDetector
from cv_system.detection.espol_touch_detector import ESPOLTouchDetector
from cv_system.detection.direct_touch_detector import DIRECTTouchDetector
from cv_system.detection.farout_touch_detector import FarOutTouchDetector, FarOutTouch
from cv_system.detection.mediapipe_direct_hybrid_touch_detector import (
    MediapipeDIRECTHybridTouchDetector,
    HybridTouchPoint,
)
from cv_system.detection.touch_tracker import TouchTracker, TrackedTouch

CardMethod = Literal["simple", "e2e", "rfdetr"]
TouchMethod = Literal["depth_only", "mediapipe", "espol", "direct", "farout", "hybrid"]

CARD_DETECTORS: dict[CardMethod, type[CardDetector]] = {
    "simple": CardDetector,
    "e2e": E2ECardDetector,
    "rfdetr": RFDETRCardDetector,
}


def detect_card_method(model_path: str | Path) -> CardMethod:
    """
    Auto-detect the appropriate card detector based on model file.

    Heuristics:
    - .pt files: always "simple" (ultralytics handles everything internally)
    - .onnx files:
      - 2 outputs named "dets" + "labels" → "rfdetr"
      - 1 output with shape[-1] == 6 → "e2e" (end-to-end with NMS)
      - Otherwise → "simple" (raw YOLO output)
    """
    path = Path(model_path)

    if path.suffix.lower() == ".pt":
        return "simple"

    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()
        output_names = [o.name.lower() for o in outputs]

        # RF-DETR: has 'dets' and 'labels' outputs
        if any("det" in name for name in output_names) and any("label" in name for name in output_names):
            return "rfdetr"

        # YOLO e2e: single output with shape [batch, N, 6]
        if len(outputs) == 1:
            shape = outputs[0].shape
            if len(shape) >= 2 and shape[-1] == 6:
                return "e2e"

        # Default: raw YOLO output
        return "simple"

    # Unknown extension, default to simple
    return "simple"


__all__ = [
    "CardDetector",
    "CardDetection",
    "CardMethod",
    "CARD_DETECTORS",
    "E2ECardDetector",
    "RFDETRCardDetector",
    "DepthOnlyTouchDetector",
    "TouchDetector",
    "ESPOLTouchDetector",
    "DIRECTTouchDetector",
    "FarOutTouchDetector",
    "FarOutTouch",
    "MediapipeDIRECTHybridTouchDetector",
    "HybridTouchPoint",
    "TouchMethod",
    "TouchTracker",
    "TrackedTouch",
    "detect_card_method",
]
