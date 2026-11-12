"""Build the dataset (YOLO format) and the group k-fold lists from the raw COCO export.

The export's own train/valid/test folders are IGNORED: Roboflow splits by image,
which leaks near-duplicate frames across splits. Every image is assigned to its
capture session (see ``sessions.py``); folds and test are made of whole sessions
(see ``folds.py`` and ``params.yaml`` -> ``split``).

Classes in ``data.drop_classes`` are removed and the remaining ones re-numbered
contiguously, keeping the export's order.

Output (``data/prepared``)::

    images/<session>__<name>.jpg, labels/<session>__<name>.txt   every image, once
    fold<k>_train.txt, fold<k>_val.txt     image lists of fold k
    final_train.txt                        everything except test
    test.txt                               test sessions (may be empty)
    fold<k>.yaml, final.yaml               Ultralytics dataset files
    manifest.json                          sessions, folds, objects per class
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vision_model.folds import fold_train_sessions, resolve_folds
from vision_model.params import load_params
from vision_model.raw import load_export, to_yolo_line
from vision_model.sessions import assign_sessions


def build_class_list(classes: list[str], drop: list[str]) -> list[str]:
    """Export classes minus the dropped ones, order preserved."""
    dropped = set(drop)
    return [c for c in classes if c not in dropped]


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


def write_list(path: Path, files: list[str]) -> None:
    """Ultralytics image list. Lines start with ``./`` so they resolve relative to
    the list's folder (an absolute path would break on another machine)."""
    path.write_text("".join(f"./images/{f}\n" for f in files), encoding="utf-8")


def write_data_yaml(path: Path, train: str, val: str, classes: list[str], test: str | None) -> None:
    # No "path" key: Ultralytics resolves the lists relative to this file's folder.
    data = {"train": train, "val": val, "names": dict(enumerate(classes))}
    if test:
        data["test"] = test
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


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

    test_s = {str(s) for s in split_cfg.get("test_sessions") or []}
    if missing := test_s - known:
        sys.exit(f"split.test_sessions not found in the data: {sorted(missing)}")
    try:
        folds = resolve_folds(split_cfg.get("folds") or [], known, test_s)
    except ValueError as e:
        sys.exit(f"split.folds: {e}. Run `uv run --no-sync vm-sessions` to list the sessions.")
    if not test_s:
        print("WARNING: split.test_sessions is empty: there will be no test metrics.")

    # A class with no training examples cannot be learned: fail before writing anything.
    final_train = known - test_s
    train_classes = {
        box.class_name
        for image in images
        if sessions[image.path] in final_train
        for box in image.boxes
    }
    if untrainable := sorted(set(classes) - train_classes):
        sys.exit(
            f"{len(untrainable)} classes have NO objects outside test: {untrainable}. "
            "Label them in a non-test session, or add them to data.drop_classes."
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)

    manifest: list[dict] = []
    files_by_session: dict[str, list[str]] = defaultdict(list)
    objects_by_session: dict[str, Counter[str]] = defaultdict(Counter)
    dropped_boxes = 0
    seen: set[str] = set()

    for image in images:
        session_id = sessions[image.path]
        stem = unique_stem(prepared_stem(session_id, image.original_name), seen)
        file = f"{stem}{image.path.suffix.lower()}"
        shutil.copy2(image.path, out_dir / "images" / file)
        lines = []
        for box in image.boxes:
            if box.class_name not in class_id:
                dropped_boxes += 1
                continue
            lines.append(to_yolo_line(box, class_id[box.class_name], image.width, image.height))
            objects_by_session[session_id][box.class_name] += 1
        (out_dir / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        files_by_session[session_id].append(file)
        manifest.append(
            {
                "file": stem,
                "original_name": image.original_name,
                "session": session_id,
                "test": session_id in test_s,
                "size": [image.width, image.height],
                "objects": len(lines),
            }
        )

    def files_of(group: set[str]) -> list[str]:
        return sorted(f for s in sorted(group) for f in files_by_session[s])

    def objects_of(group: set[str]) -> dict[str, int]:
        total: Counter[str] = Counter()
        for s in group:
            total.update(objects_by_session[s])
        return dict(sorted(total.items()))

    fold_summaries = []
    for k, val_s in enumerate(folds, 1):
        train_s = fold_train_sessions(known, test_s, val_s)
        write_list(out_dir / f"fold{k}_train.txt", files_of(train_s))
        write_list(out_dir / f"fold{k}_val.txt", files_of(val_s))
        write_data_yaml(
            out_dir / f"fold{k}.yaml", f"fold{k}_train.txt", f"fold{k}_val.txt", classes, None
        )
        val_objects = objects_of(val_s)
        train_objects = objects_of(train_s)
        fold_summaries.append(
            {
                "fold": k,
                "val_sessions": sorted(val_s),
                "images": {"train": len(files_of(train_s)), "val": len(files_of(val_s))},
                "val_objects_per_class": val_objects,
                "classes_not_validated": sorted(set(classes) - set(val_objects)),
                "classes_not_trained": sorted(set(classes) - set(train_objects)),
            }
        )

    write_list(out_dir / "final_train.txt", files_of(final_train))
    write_list(out_dir / "test.txt", files_of(test_s))
    # The final model trains with a fixed number of epochs (from the folds) and no
    # validation; Ultralytics still needs a val entry, so it points to train.
    write_data_yaml(
        out_dir / "final.yaml",
        "final_train.txt",
        "final_train.txt",
        classes,
        "test.txt" if test_s else None,
    )

    summary = {
        "sessions": {s: len(files_by_session[s]) for s in sorted(known)},
        "test_sessions": sorted(test_s),
        "folds": fold_summaries,
        "images": {"final_train": len(files_of(final_train)), "test": len(files_of(test_s))},
        "test_objects_per_class": objects_of(test_s),
        "classes_missing_in_test": sorted(set(classes) - set(objects_of(test_s))) if test_s else [],
        "dropped_boxes": dropped_boxes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps({"summary": summary, "images": manifest}, indent=2), encoding="utf-8"
    )

    for fold in fold_summaries:
        print(
            f"fold {fold['fold']}: val={fold['val_sessions']} images={fold['images']} "
            f"classes validated={len(classes) - len(fold['classes_not_validated'])}/{len(classes)}"
        )
        if fold["classes_not_trained"]:
            print(f"  WARNING: classes with no training objects: {fold['classes_not_trained']}")
    print(
        f"final train: {summary['images']['final_train']} images, test: {summary['images']['test']}"
    )
    if summary["classes_missing_in_test"]:
        print(f"WARNING: classes not in test: {summary['classes_missing_in_test']}")


if __name__ == "__main__":
    main()
