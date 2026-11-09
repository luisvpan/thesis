"""Evaluate the trained model: pick the operating threshold on val, report on test.

The confidence threshold the cv-system should use is the one that maximizes
mean F1 on the VAL split. It is chosen on val (never on test) so that the test
numbers stay an unbiased estimate.

Outputs (all small, git-tracked through DVC with ``cache: false``):
    reports/metrics.json          headline numbers + operating confidence
    reports/per_class.csv         AP50 / AP50-95 / P / R per class on test
    reports/confusion_matrix.json raw test confusion matrix (rows = predicted, cols = actual)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vision_model.params import RUNS_DIR, load_params
from vision_model.train import BEST_WEIGHTS

REPORTS = Path("reports")


def headline(metrics) -> dict[str, float]:
    box = metrics.box
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def f1_optimal_confidence(metrics) -> float | None:
    """Confidence that maximizes the class-averaged F1 curve, if available."""
    box = metrics.box
    f1 = getattr(box, "f1_curve", None)
    px = getattr(box, "px", None)
    if f1 is None or len(f1) == 0:
        return None
    px = np.asarray(px) if px is not None else np.linspace(0, 1, np.asarray(f1).shape[-1])
    return float(px[int(np.argmax(np.asarray(f1).mean(axis=0)))])


def main() -> None:
    from ultralytics import YOLO

    params = load_params()
    data_yaml = (Path(params["data"]["prepared_dir"]) / "data.yaml").resolve()
    imgsz = params["train"]["imgsz"]
    model = YOLO(str(BEST_WEIGHTS))

    val = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        plots=False,
        project=str(RUNS_DIR),
        name="val",
        exist_ok=True,
    )
    test_images = Path(params["data"]["prepared_dir"]) / "test" / "images"
    has_test = test_images.exists() and any(test_images.iterdir())
    test = None
    if has_test:
        test = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=imgsz,
            plots=True,
            project=str(RUNS_DIR),
            name="test",
            exist_ok=True,
        )

    REPORTS.mkdir(exist_ok=True)
    detail_split = "test" if test is not None else "val"
    detail = test if test is not None else val
    names = detail.names
    box = detail.box

    # Ultralytics only reports classes that have objects in the evaluated split.
    # Make the gap explicit: a class absent from it was NOT evaluated at all.
    manifest = json.loads(
        (Path(params["data"]["prepared_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    instances = manifest["summary"]["objects_per_class"][detail_split]
    evaluated = {names[int(c)] for c in box.ap_class_index}

    summary = {
        "val": headline(val),
        "test": headline(test) if test is not None else None,
        "operating_conf": f1_optimal_confidence(val),
        # Split that per_class.csv and confusion_matrix.json come from: test, or val
        # while there is no test set.
        "detail_split": detail_split,
        "classes_evaluated": len(evaluated),
        "classes_total": len(names),
        "classes_not_evaluated": sorted(set(names.values()) - evaluated),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(REPORTS / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "instances", "ap50", "ap50_95", "precision", "recall"])
        for i, class_id in enumerate(box.ap_class_index):
            name = names[int(class_id)]
            writer.writerow(
                [
                    name,
                    instances.get(name, 0),
                    f"{box.ap50[i]:.4f}",
                    f"{box.ap[i]:.4f}",
                    f"{box.p[i]:.4f}",
                    f"{box.r[i]:.4f}",
                ]
            )

    matrix = detail.confusion_matrix.matrix
    labels = [*(names[i] for i in sorted(names)), "background"]
    (REPORTS / "confusion_matrix.json").write_text(
        json.dumps({"labels": labels, "matrix": np.asarray(matrix).astype(int).tolist()}),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
