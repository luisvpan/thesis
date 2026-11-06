"""Read the raw Roboflow export (COCO JSON format).

COCO is used as the raw format because, unlike the YOLO export, it keeps what
Roboflow knows about each image: the original filename (``extra.name``), its
tags (``extra.user_tags``) and its metadata (``extra.user_metadata``). Session
information assigned in Roboflow travels through those fields.

Expected layout (as Roboflow exports it)::

    data/raw/
      train/_annotations.coco.json + images
      valid/_annotations.coco.json + images
      test/_annotations.coco.json  + images

The export's train/valid/test folders carry no meaning here; ``prepare``
re-splits everything by session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ANNOTATIONS_FILE = "_annotations.coco.json"


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels, COCO convention (top-left x, y, width, height)."""

    class_name: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class RawImage:
    path: Path
    original_name: str
    width: int
    height: int
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    boxes: tuple[Box, ...] = ()


def class_names(categories: list[dict]) -> dict[int, str]:
    """Category id -> name, without Roboflow's parent category.

    Roboflow adds one category named after the project (``supercategory: none``)
    that every real class points to as its supercategory. It is not a class.
    """
    parents = {c["supercategory"] for c in categories}
    return {
        c["id"]: c["name"]
        for c in categories
        if not (c.get("supercategory") == "none" and c["name"] in parents)
    }


def load_split(split_dir: Path) -> tuple[list[RawImage], list[str]]:
    """Images and ordered class names of one exported folder."""
    data = json.loads((split_dir / ANNOTATIONS_FILE).read_text(encoding="utf-8"))
    names = class_names(data["categories"])

    boxes: dict[int, list[Box]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] not in names:
            continue
        x, y, w, h = ann["bbox"]
        boxes.setdefault(ann["image_id"], []).append(Box(names[ann["category_id"]], x, y, w, h))

    images = []
    for img in data["images"]:
        extra = img.get("extra") or {}
        images.append(
            RawImage(
                path=split_dir / img["file_name"],
                original_name=extra.get("name") or img["file_name"],
                width=int(img["width"]),
                height=int(img["height"]),
                tags=tuple(extra.get("user_tags") or ()),
                metadata=dict(extra.get("user_metadata") or {}),
                boxes=tuple(boxes.get(img["id"], ())),
            )
        )
    ordered = [names[i] for i in sorted(names)]
    return images, ordered


def load_export(raw_dir: Path) -> tuple[list[RawImage], list[str]]:
    """All images of the export and the class list (must agree across folders)."""
    images: list[RawImage] = []
    classes: list[str] | None = None
    for ann_file in sorted(raw_dir.glob(f"*/{ANNOTATIONS_FILE}")):
        split_images, split_classes = load_split(ann_file.parent)
        if classes is not None and split_classes != classes:
            raise ValueError(f"class list differs in {ann_file.parent}")
        classes = split_classes
        images.extend(split_images)
    if classes is None:
        raise FileNotFoundError(f"no {ANNOTATIONS_FILE} under {raw_dir}")
    return images, classes


def to_yolo_line(box: Box, class_id: int, width: int, height: int) -> str:
    """COCO pixel box -> YOLO normalized ``class cx cy w h``."""
    cx = (box.x + box.w / 2) / width
    cy = (box.y + box.h / 2) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {box.w / width:.6f} {box.h / height:.6f}"
