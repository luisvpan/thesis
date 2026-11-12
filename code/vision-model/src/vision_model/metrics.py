"""Helpers to read Ultralytics validation results (shared by cv and evaluate)."""

from __future__ import annotations

import statistics

import numpy as np


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


def per_class(metrics) -> dict[str, dict[str, float]]:
    """AP50, AP50-95, precision and recall of every class present in the split."""
    box, names = metrics.box, metrics.names
    return {
        names[int(c)]: {
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
        }
        for i, c in enumerate(box.ap_class_index)
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }
