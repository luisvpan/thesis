"""Square model input from a frame, and boxes back to frame coordinates.

``letterbox`` reproduces Ultralytics' LetterBox (center padding, pad value 114),
the preprocessing YOLO models are trained with. ``stretch`` resizes ignoring the
aspect ratio, as Roboflow's "Stretch to" preprocessing did for legacy models.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Geometry:
    """How a frame was mapped into the model input: input = frame * scale + pad."""

    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float


def letterbox(
    image: np.ndarray, size: int, pad_value: int = 114
) -> tuple[np.ndarray, Geometry]:
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    resized = (
        image
        if (new_w, new_h) == (w, h)
        else cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    )
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    return out, Geometry(new_w / w, new_h / h, left, top)


def stretch(image: np.ndarray, size: int) -> tuple[np.ndarray, Geometry]:
    h, w = image.shape[:2]
    out = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return out, Geometry(size / w, size / h, 0.0, 0.0)


def to_model_input(
    image: np.ndarray, size: int, mode: str, pad_value: int = 114
) -> tuple[np.ndarray, Geometry]:
    if mode == "letterbox":
        return letterbox(image, size, pad_value)
    if mode == "stretch":
        return stretch(image, size)
    raise ValueError(f"unknown resize mode {mode!r}")


def to_frame_coords(xyxy: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Boxes (N, 4) in model-input pixels -> frame pixels."""
    out = np.asarray(xyxy, dtype=np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - geometry.pad_x) / geometry.scale_x
    out[:, [1, 3]] = (out[:, [1, 3]] - geometry.pad_y) / geometry.scale_y
    return out
