"""Fine-tune a pretrained YOLO checkpoint on the prepared dataset.

Every hyperparameter comes from ``params.yaml`` -> ``train``; the keys besides
``model`` are passed verbatim to ``YOLO.train``. Metrics and plots are logged
by Ultralytics' built-in DVCLive callback into ``dvclive/``; the best weights
end up in ``dvclive/artifacts/best.pt`` (a DVC-tracked output).
"""

from __future__ import annotations

import shutil
import types
from functools import partial
from pathlib import Path

from vision_model.params import RUNS_DIR, load_params

BEST_WEIGHTS = Path("dvclive/artifacts/best.pt")


def keep_dvclive_out_of_dvc_yaml() -> None:
    """Stop DVCLive from rewriting dvc.yaml.

    Ultralytics' callback creates ``dvclive.Live(...)`` with DVCLive's default
    ``dvcyaml="dvc.yaml"``, which appends top-level metrics/plots sections to our
    dvc.yaml (and reformats it) on every run. Those files are already declared
    in the ``train`` stage, so the callback's module gets a ``Live`` with
    ``dvcyaml=None``. Only that module is affected.
    """
    from ultralytics.utils.callbacks import dvc as ultralytics_dvc

    real = getattr(ultralytics_dvc, "dvclive", None)
    if real is None:  # dvclive not installed: the callback is disabled anyway
        return
    ultralytics_dvc.dvclive = types.SimpleNamespace(Live=partial(real.Live, dvcyaml=None))


def main() -> None:
    from ultralytics import YOLO, settings

    settings.update({"dvc": True})  # enable the DVCLive callback
    keep_dvclive_out_of_dvc_yaml()

    params = load_params()
    train_cfg = dict(params["train"])
    checkpoint = train_cfg.pop("model")
    data_yaml = Path(params["data"]["prepared_dir"]) / "data.yaml"

    model = YOLO(checkpoint)
    model.train(
        data=str(data_yaml),
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        plots=True,
        **train_cfg,
    )

    # The DVCLive callback copies best.pt into dvclive/artifacts/. If dvclive is
    # missing or the callback did not run, copy it ourselves so the stage output exists.
    trained = Path(model.trainer.best)
    if not BEST_WEIGHTS.exists():
        BEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained, BEST_WEIGHTS)
    print(f"Best weights: {BEST_WEIGHTS}")


if __name__ == "__main__":
    main()
