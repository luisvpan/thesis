"""Access to ``params.yaml`` (the single source of experiment configuration)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
# Absolute on purpose: a relative `project=` is resolved by Ultralytics against its
# global runs_dir setting, which put runs in the thesis repo root.
RUNS_DIR = PROJECT_ROOT / "runs"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load params.yaml. DVC tracks the same file, so every run is reproducible."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))
