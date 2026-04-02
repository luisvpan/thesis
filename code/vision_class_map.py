"""
Diccionario clase YOLO ↔ nombre (data.yaml) ↔ valor numérico para cartas dígito.

El índice en `names` coincide con `class_id` del modelo.
Solo las clases nombradas one..nine tienen `semantic_number` 1..9.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_YAML = ROOT / "config" / "dataflow_augmented.yaml"

# Nombre de clase (minúsculas, como en yaml) → dígito mostrado en el canvas
WORD_TO_DIGIT: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def load_class_names(yaml_path: Path | None = None) -> list[str]:
    path = yaml_path or Path(os.environ.get("VISION_DATA_YAML", str(DEFAULT_YAML)))
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el dataset YAML: {path}")
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    names = data.get("names")
    if not names or not isinstance(names, list):
        raise ValueError("YAML debe contener 'names:' como lista")
    return [str(x).strip() for x in names]


def class_id_to_label(class_id: int, names: list[str]) -> str:
    if class_id < 0 or class_id >= len(names):
        return f"unknown({class_id})"
    return names[class_id]


def semantic_number_for_label(label: str) -> int | None:
    """Si la clase es un dígito (one..nine), devuelve 1..9; si no, None."""
    key = label.strip().lower()
    return WORD_TO_DIGIT.get(key)


def resolve_detection(class_id: int, names: list[str]) -> tuple[str, int | None]:
    """
    Devuelve (label, semantic_number).
    semantic_number solo para clases one..nine; el resto None.
    """
    label = class_id_to_label(class_id, names)
    return label, semantic_number_for_label(label)
