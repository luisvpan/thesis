"""
Hybrid DIRECT + MediaPipe touch detector with torch-directml depth helpers.

Same logic as MediapipeDIRECTHybridTouchDetector. MediaPipe and DIRECT flood-fill
stay on CPU; optional DirectML device is used when depth tensor ops are added later.
"""

from __future__ import annotations

from cv_system.detection.mediapipe_direct_hybrid_touch_detector import (
    MediapipeDIRECTHybridTouchDetector,
)
from cv_system.gpu_config import get_torch_device


class MediapipeDIRECTHybridTouchDetectorDml(MediapipeDIRECTHybridTouchDetector):
    """DIRECT + MediaPipe hybrid with DirectML-backed PyTorch available for depth ops."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._torch_device = get_torch_device()
        print(
            "  MediapipeDIRECTHybridTouchDetectorDml: DIRECT+MediaPipe on CPU, "
            f"PyTorch device {self._torch_device} ready for depth tensors"
        )
