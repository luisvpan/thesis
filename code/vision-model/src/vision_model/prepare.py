"""Build the train/val/test dataset (YOLO format) from the raw COCO export, split by session.

The export's own train/valid/test folders are IGNORED: Roboflow splits by image,
which leaks near-duplicate frames across splits. Every image is assigned to its
capture session (see ``sessions.py``) and whole sessions go to one split, as
listed in ``params.yaml`` (``split.val_sessions`` / ``split.test_sessions``).

Classes in ``data.drop_classes`` are removed and the remaining ones re-numbered
contiguously, keeping the export's order.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vision_model.params import load_params
from vision_model.raw import load_export, to_yolo_line
from vision_model.sessions import UNTIMED_SESSION, assign_sessions

SPLITS = ("train", "val", "test")


def build_class_list(classes: list[str], drop: list[str]) -> list[str]:
    """Export classes minus the dropped ones, order preserved."""
    dropped = set(drop)
    return [c for c in classes if c not in dropped]


def split_for_session(session_id: str, split_cfg: dict) -> str:
    if session_id in split_cfg.get("test_sessions", []):
        return "test"
    if session_id in split_cfg.get("val_sessions", []):
        return "val"
    if session_id == UNTIMED_SESSION:
        return split_cfg.get("untimed_to", "train")
    return "train"


def prepared_stem(session_id: str, original_name: str) -> str:
    """Output filename stem: ``<session>__<original stem>``, so the session is visible
    when browsing data/prepared or Ultralytics' plots."""
    return f"{session_id}__{Path(original_name).stem}"


def unique_stem(stem: str, seen: set[str]) -> str:
    """Old predict names can repeat even within a session ("untimed"): suffix duplicates."""
    candidate, n = stem, 1
    while candidate in seen:
        candidate = f"{stem}__{n}"
        n += 1
    seen.add(candidate)
    return candidate


def main() -> None:
    params = load_params()
    data_cfg, split_cfg = params["data"], params["split"]
    raw_dir = Path(data_cfg["raw_dir"])
    out_dir = Path(data_cfg["prepared_dir"])

    images, export_classes = load_export(raw_dir)
    classes = build_class_list(export_classes, data_cfg.get("drop_classes", []))
    class_id = {name: i for i, name in enumerate(classes)}
    sessions = assign_sessions(images, data_cfg["session_gap_minutes"])
    known = set(sessions.values())

    val_s = {str(s) for s in split_cfg.get("val_sessions") or []}
    test_s = {str(s) for s in split_cfg.get("test_sessions") or []}
    if not val_s:
        sys.exit(
            "split.val_sessions must be set. "
            "Run `uv run vm-sessions` to list the sessions and pick them."
        )
    if missing := (val_s | test_s) - known:
        sys.exit(f"Sessions in params.yaml not found in the data: {sorted(missing)}")
    if overlap := val_s & test_s:
        sys.exit(f"Sessions assigned to both val and test: {sorted(overlap)}")
    if not test_s:
        print("WARNING: split.test_sessions is empty: there will be no test metrics.")

    # A class with no training examples cannot be learned: fail before writing anything.
    split_cfg = {**split_cfg, "val_sessions": val_s, "test_sessions": test_s}
    train_classes = {
        box.class_name
        for image in images
        if split_for_session(sessions[image.path], split_cfg) == "train"
        for box in image.boxes
    }
    if untrainable := sorted(set(classes) - train_classes):
        sys.exit(
            f"{len(untrainable)} classes have NO objects in train with this split: {untrainable}. "
            "Move the sessions that contain them to train, or add them to data.drop_classes."
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)

    manifest: list[dict] = []
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dropped_boxes = 0
    seen: set[str] = set()

    for image in images:
        session_id = sessions[image.path]
        split = split_for_session(session_id, split_cfg)
        stem = unique_stem(prepared_stem(session_id, image.original_name), seen)

        shutil.copy2(image.path, out_dir / split / "images" / f"{stem}{image.path.suffix.lower()}")
        lines = []
        for box in image.boxes:
            if box.class_name not in class_id:
                dropped_boxes += 1
                continue
            lines.append(to_yolo_line(box, class_id[box.class_name], image.width, image.height))
            class_counts[split][box.class_name] += 1
        (out_dir / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        manifest.append(
            {
                "file": stem,
                "original_name": image.original_name,
                "session": session_id,
                "split": split,
                "size": [image.width, image.height],
                "objects": len(lines),
            }
        )

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": dict(enumerate(classes)),
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary = {
        "images_per_split": dict(Counter(m["split"] for m in manifest)),
        "sessions_per_split": {
            split: sorted({m["session"] for m in manifest if m["split"] == split})
            for split in SPLITS
        },
        "objects_per_class": {split: dict(class_counts[split]) for split in SPLITS},
        "classes_missing": {
            split: sorted(set(classes) - set(class_counts[split])) for split in SPLITS
        },
        "dropped_boxes": dropped_boxes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps({"summary": summary, "images": manifest}, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary["images_per_split"]))
    for split in ("val", "test") if test_s else ("val",):
        if missing_classes := summary["classes_missing"][split]:
            print(
                f"WARNING: {len(missing_classes)} classes with no objects in {split} "
                f"(they will not be evaluated): {missing_classes}"
            )


if __name__ == "__main__":
    main()
