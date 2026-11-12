"""Evaluate the final model on test; collect the cross-validated numbers (`vm-evaluate`).

The operating confidence (the threshold the cv-system uses) comes from the
folds' validation (``reports/cv/summary.json``), never from test, so the test
numbers stay an unbiased estimate.

Outputs (small, git-tracked through DVC with ``cache: false``):
    reports/metrics.json          cross-validated + test numbers, operating confidence
    reports/per_class.csv         per class on test (or the folds' means without test)
    reports/confusion_matrix.json test confusion matrix (rows = predicted, cols = actual)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vision_model.metrics import headline, per_class
from vision_model.params import RUNS_DIR, load_params
from vision_model.train import CV_SUMMARY, FINAL_WEIGHTS

REPORTS = Path("reports")


def main() -> None:
    params = load_params()
    prepared = Path(params["data"]["prepared_dir"])
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))["summary"]
    cv = json.loads(CV_SUMMARY.read_text(encoding="utf-8"))
    has_test = manifest["images"]["test"] > 0

    test = None
    if has_test:
        from ultralytics import YOLO

        test = YOLO(str(FINAL_WEIGHTS)).val(
            data=str((prepared / "final.yaml").resolve()),
            split="test",
            imgsz=params["train"]["imgsz"],
            plots=True,
            project=str(RUNS_DIR),
            name="test",
            exist_ok=True,
        )

    REPORTS.mkdir(exist_ok=True)
    cv_mean = {k: v["mean"] for k, v in cv["headline"].items()}
    if test is not None:
        names = test.names
        classes = per_class(test)
        instances = manifest["test_objects_per_class"]
        rows = [
            [c, instances.get(c, 0), m["ap50"], m["ap50_95"], m["precision"], m["recall"]]
            for c, m in classes.items()
        ]
        all_classes = list(names.values())
        matrix = np.asarray(test.confusion_matrix.matrix).astype(int).tolist()
        labels = [*(names[i] for i in sorted(names)), "background"]
    else:
        classes = cv["per_class"]
        rows = [
            [
                c,
                m["ap50"]["n"],
                m["ap50"]["mean"],
                m["ap50_95"]["mean"],
                m["precision"]["mean"],
                m["recall"]["mean"],
            ]
            for c, m in classes.items()
        ]
        all_classes = sorted(
            set(classes) | {c for f in manifest["folds"] for c in f["classes_not_validated"]}
        )
        matrix, labels = None, None

    summary = {
        # Cross-validated (mean over folds); "val" kept for the model card's schema
        "val": cv_mean,
        "cv": {"headline": cv["headline"], "folds_run": cv["folds_run"], "epochs": cv["epochs"]},
        "test": headline(test) if test is not None else None,
        "operating_conf": cv["operating_conf"],
        # Where per_class.csv comes from: "test", or "cv" (fold means) without a test set
        "detail_split": "test" if test is not None else "cv",
        "classes_evaluated": len(classes),
        "classes_total": len(all_classes),
        "classes_not_evaluated": sorted(set(all_classes) - set(classes)),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(REPORTS / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # "instances" is the object count on test, or the number of folds without test
        writer.writerow(["class", "instances", "ap50", "ap50_95", "precision", "recall"])
        for row in rows:
            writer.writerow([row[0], row[1], *(f"{v:.4f}" for v in row[2:])])
    (REPORTS / "confusion_matrix.json").write_text(
        json.dumps({"labels": labels, "matrix": matrix}), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
