"""Group k-fold cross-validation (`vm-cv`): one training per fold, then aggregate.

Each fold trains on ``data/prepared/fold<k>.yaml`` with the ``train`` params
(early stopping on that fold's val) and is evaluated on its val sessions. The
aggregate in ``reports/cv/summary.json`` gives:

- metrics as mean +- std across folds (how much the model varies between sessions);
- ``epochs``: mean best epoch, used to train the final model without a val set;
- ``operating_conf``: mean F1-optimal confidence, the cv-system's threshold.

``reports/cv/per_class.csv`` averages each class over the folds that validate it.
``cv.max_folds`` limits the folds run (e.g. 1 for a quick experiment).

A class whose only labeled session is a fold's val has no training examples
in that fold: its AP there measures nothing but its absence, so it is left out
of that fold's numbers (``headline`` and per class are means over the classes
the fold could learn; Ultralytics' own all-class numbers stay in ``headline_all``).

Folds can be trained elsewhere, in parallel (``remote/exp.py`` does it on
Modal): ``vm-cv --fold K --out fold.json`` trains one fold, and the stage, run
with ``VM_CV_FOLD_RESULTS=<dir with fold<k>.json>``, aggregates those results
instead of training. Each result carries a fingerprint of its fold's image
lists and training params; the stage refuses results that do not match the
current ones, so precomputed folds can only stand in for the same computation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

from vision_model.metrics import f1_optimal_confidence, headline, mean_std, per_class
from vision_model.params import RUNS_DIR, load_params
from vision_model.train import train_kwargs

CV_DIR = Path("reports/cv")
SUMMARY = CV_DIR / "summary.json"
RESULTS_ENV = "VM_CV_FOLD_RESULTS"


def fold_fingerprint(k: int, prepared: Path, params: dict) -> str:
    """What a fold's result depends on: its image lists and the training params."""
    h = hashlib.sha256()
    for name in (f"fold{k}_train.txt", f"fold{k}_val.txt"):
        h.update((prepared / name).read_bytes())
    h.update(json.dumps([params["train"], params.get("runtime", {})], sort_keys=True).encode())
    return h.hexdigest()[:16]


def fold_ids(prepared: Path, params: dict) -> list[int]:
    ids = sorted(int(p.stem[4:]) for p in prepared.glob("fold*.yaml"))
    max_folds = (params.get("cv") or {}).get("max_folds")
    return ids[: int(max_folds)] if max_folds else ids


def run_fold(k: int, data_yaml: Path, params: dict) -> dict:
    from ultralytics import YOLO

    checkpoint, kwargs = train_kwargs(params)
    model = YOLO(checkpoint)
    model.train(
        data=str(data_yaml),
        project=str(RUNS_DIR),
        name=f"cv_fold{k}",
        exist_ok=True,
        plots=False,
        **kwargs,
    )
    trainer = model.trainer
    best_epoch = int(trainer.stopper.best_epoch) + 1  # stopper counts from 0
    best = YOLO(str(trainer.best))
    val = best.val(
        data=str(data_yaml),
        split="val",
        imgsz=params["train"]["imgsz"],
        plots=False,
        project=str(RUNS_DIR),
        name=f"cv_fold{k}_val",
        exist_ok=True,
    )
    manifest = json.loads((data_yaml.parent / "manifest.json").read_text(encoding="utf-8"))
    untrained = set(manifest["summary"]["folds"][k - 1]["classes_not_trained"])
    classes = {c: m for c, m in per_class(val).items() if c not in untrained}
    return {
        "fold": k,
        "fingerprint": fold_fingerprint(k, data_yaml.parent, params),
        "best_epoch": best_epoch,
        "epochs_run": int(trainer.epoch) + 1,
        "headline": trained_headline(classes),
        "headline_all": headline(val),
        "classes_not_trained": sorted(untrained),
        "operating_conf": f1_optimal_confidence(val),
        "per_class": classes,
    }


def trained_headline(classes: dict[str, dict[str, float]]) -> dict[str, float]:
    """Headline metrics as means over the given classes (as Ultralytics does)."""
    return {
        "map50": statistics.fmean(m["ap50"] for m in classes.values()),
        "map50_95": statistics.fmean(m["ap50_95"] for m in classes.values()),
        "precision": statistics.fmean(m["precision"] for m in classes.values()),
        "recall": statistics.fmean(m["recall"] for m in classes.values()),
    }


def aggregate(folds: list[dict]) -> dict:
    keys = folds[0]["headline"].keys()
    confs = [f["operating_conf"] for f in folds if f["operating_conf"] is not None]
    classes = sorted({c for f in folds for c in f["per_class"]})
    return {
        "folds_run": len(folds),
        "headline": {k: mean_std([f["headline"][k] for f in folds]) for k in keys},
        "epochs": round(statistics.fmean(f["best_epoch"] for f in folds)),
        "operating_conf": statistics.fmean(confs) if confs else None,
        "per_class": {
            c: {
                m: mean_std([f["per_class"][c][m] for f in folds if c in f["per_class"]])
                for m in ("ap50", "ap50_95", "precision", "recall")
            }
            for c in classes
        },
        "folds": [{k: v for k, v in f.items() if k != "per_class"} for f in folds],
    }


def load_results(directory: Path, ids: list[int], prepared: Path, params: dict) -> list[dict]:
    results = []
    for k in ids:
        result = json.loads((directory / f"fold{k}.json").read_text(encoding="utf-8"))
        expected = fold_fingerprint(k, prepared, params)
        if result.get("fingerprint") != expected:
            raise SystemExit(
                f"fold {k}: precomputed result is for other data or params "
                f"({result.get('fingerprint')} != {expected})"
            )
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Group k-fold cross-validation")
    parser.add_argument("--fold", type=int, help="Train only this fold (no aggregation)")
    parser.add_argument("--out", type=Path, help="With --fold: where to write its result")
    args = parser.parse_args()

    from ultralytics import settings

    settings.update({"dvc": False})  # per-fold runs stay out of dvclive/ (the final model's)
    params = load_params()
    prepared = Path(params["data"]["prepared_dir"])

    if args.fold is not None:
        result = run_fold(args.fold, (prepared / f"fold{args.fold}.yaml").resolve(), params)
        out = args.out or Path(f"fold{args.fold}.json")
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"fold {args.fold} -> {out}")
        return

    ids = fold_ids(prepared, params)
    if precomputed := os.environ.get(RESULTS_ENV):
        print(f"Using precomputed fold results from {precomputed}")
        results = load_results(Path(precomputed), ids, prepared, params)
    else:
        results = []
        for k in ids:
            print(f"\n=== fold {k}/{len(ids)} ===", flush=True)
            results.append(run_fold(k, (prepared / f"fold{k}.yaml").resolve(), params))

    summary = aggregate(results)
    CV_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(CV_DIR / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "folds", "ap50", "ap50_95", "ap50_95_std", "precision", "recall"])
        for name, m in summary["per_class"].items():
            writer.writerow(
                [
                    name,
                    m["ap50"]["n"],
                    f"{m['ap50']['mean']:.4f}",
                    f"{m['ap50_95']['mean']:.4f}",
                    f"{m['ap50_95']['std']:.4f}",
                    f"{m['precision']['mean']:.4f}",
                    f"{m['recall']['mean']:.4f}",
                ]
            )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
