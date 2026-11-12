"""Export the model and assemble the release bundle in ``models/export/``.

The bundle is what the cv-system consumes and what gets attached to a GitHub
Release. ``model_card.json`` is the contract: the cv-system must read class
names, input size, preprocessing and the operating threshold from it instead of
hardcoding them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from vision_model.evaluate import REPORTS
from vision_model.params import PROJECT_ROOT, load_params
from vision_model.train import FINAL_WEIGHTS

EXPORT_DIR = Path("models/export")


def git_short_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def dataset_hash() -> str:
    """md5 DVC recorded for the prepared dataset (short), or 'unknown'."""
    lock = PROJECT_ROOT / "dvc.lock"
    if not lock.exists():
        return "unknown"
    stages = yaml.safe_load(lock.read_text(encoding="utf-8")).get("stages", {})
    for out in stages.get("prepare", {}).get("outs", []):
        if out.get("md5"):
            return out["md5"].split(".")[0][:7]
    return "unknown"


def main() -> None:
    import ultralytics
    from ultralytics import YOLO

    params = load_params()
    export_cfg, train_cfg = params["export"], params["train"]
    arch = Path(train_cfg["model"]).stem.replace("yolo", "y")
    name = f"{arch}-ds{dataset_hash()}-{git_short_sha()}"

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True)

    # Export from a copy: Ultralytics writes the exported file next to the weights,
    # and dvclive/artifacts is the train stage's output, which must stay untouched.
    weights = EXPORT_DIR / "model.pt"
    shutil.copy2(FINAL_WEIGHTS, weights)
    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format=export_cfg["format"],
            imgsz=train_cfg["imgsz"],
            dynamic=export_cfg["dynamic"],
            simplify=export_cfg["simplify"],
        )
    )
    if exported.name != f"model{exported.suffix}":
        exported.rename(EXPORT_DIR / f"model{exported.suffix}")

    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    card = {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_checkpoint": train_cfg["model"],
        "git_commit": git_short_sha(),
        "dataset_md5": dataset_hash(),
        "ultralytics_version": ultralytics.__version__,
        "class_names": [model.names[i] for i in sorted(model.names)],
        "input": {
            "imgsz": train_cfg["imgsz"],
            "resize": "letterbox",  # keep aspect ratio, pad to square
            "pad_value": 114,
            "color": "RGB",
            "scale": "1/255",
        },
        "conf_threshold": metrics.get("operating_conf"),
        "metrics": {
            # "val": mean over the cross-validation folds (see reports/cv/summary.json)
            "val": metrics["val"],
            "cv": metrics.get("cv"),
            "test": metrics["test"],  # null while there is no test set (see params.yaml)
            "classes_evaluated": metrics.get("classes_evaluated"),
            "classes_total": metrics.get("classes_total"),
        },
    }
    (EXPORT_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    for report in ("metrics.json", "per_class.csv", "confusion_matrix.json"):
        shutil.copy2(REPORTS / report, EXPORT_DIR / report)
    print(f"Release bundle: {EXPORT_DIR} ({name})")


if __name__ == "__main__":
    main()
