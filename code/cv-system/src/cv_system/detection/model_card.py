"""Per-model inference contract, read from the ``model_card.json`` that
``code/vision-model`` exports next to each model.

Models trained before that pipeline have no card. They were trained on Roboflow
exports preprocessed with "Stretch to 640x640", so their input must be STRETCHED
to a square; models from the pipeline are trained with LETTERBOX (aspect ratio
kept, padded). Feeding a model the other preprocessing deforms what it sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ResizeMode = Literal["letterbox", "stretch"]
CARD_FILENAME = "model_card.json"


@dataclass(frozen=True)
class ModelCard:
    resize: ResizeMode = "stretch"  # legacy default: models without a card
    pad_value: int = 114
    imgsz: int | None = None  # None: take it from the model itself
    conf_threshold: float | None = None  # None: the detector's default
    name: str | None = None
    source: Path | None = None  # card file, or None for the legacy default

    @property
    def is_legacy(self) -> bool:
        return self.source is None


def find_model_card(model_path: str | Path) -> Path | None:
    """``<stem>.model_card.json`` next to the model, else ``model_card.json`` in its folder."""
    path = Path(model_path)
    for candidate in (
        path.with_name(f"{path.stem}.{CARD_FILENAME}"),
        path.with_name(CARD_FILENAME),
    ):
        if candidate.is_file():
            return candidate
    return None


def load_model_card(model_path: str | Path) -> ModelCard:
    """Card for ``model_path``; the legacy (stretch) contract if there is none."""
    card_path = find_model_card(model_path)
    if card_path is None:
        return ModelCard()
    data = json.loads(card_path.read_text(encoding="utf-8"))
    inp = data.get("input", {})
    resize = inp.get("resize", "letterbox")
    if resize not in ("letterbox", "stretch"):
        raise ValueError(f"{card_path}: unsupported input.resize {resize!r}")
    conf = data.get("conf_threshold")
    return ModelCard(
        resize=resize,
        pad_value=int(inp.get("pad_value", 114)),
        imgsz=int(inp["imgsz"]) if inp.get("imgsz") else None,
        conf_threshold=float(conf) if conf is not None else None,
        name=data.get("name"),
        source=card_path,
    )
