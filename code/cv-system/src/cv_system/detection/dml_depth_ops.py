"""Depth touch-zone masks on torch-directml."""

from __future__ import annotations

import numpy as np
import torch

from cv_system.gpu_config import get_torch_device


class DmlDepthTouchOps:
    """GPU-backed depth zone checks (MediaPipe landmarks still run on CPU)."""

    def __init__(self, dmin_map: np.ndarray, dmax_map: np.ndarray) -> None:
        self._device = get_torch_device()
        self._dmin = torch.from_numpy(dmin_map.astype(np.int32)).to(self._device)
        self._dmax = torch.from_numpy(dmax_map.astype(np.int32)).to(self._device)

    def touch_zone_mask(self, depth_frame: np.ndarray) -> torch.Tensor:
        depth = torch.from_numpy(depth_frame).to(self._device)
        return (depth > self._dmin) & (depth < self._dmax) & (depth > 0)

    def hand_has_touch(self, hand_mask: np.ndarray, touch_zone: torch.Tensor) -> bool:
        mask = torch.from_numpy(hand_mask > 0).to(self._device)
        return bool(torch.any(mask & touch_zone).item())
