"""Fine-tune a pretrained YOLO checkpoint on the prepared dataset.

Every hyperparameter comes from ``params.yaml`` -> ``train``; the keys besides
``model`` are passed verbatim to ``YOLO.train``. Metrics and plots are logged
by Ultralytics' built-in DVCLive callback into ``dvclive/``; the best weights
end up in ``dvclive/artifacts/best.pt`` (a DVC-tracked output).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from vision_model.params import load_params

BEST_WEIGHTS = Path("dvclive/artifacts/best.pt")


def main() -> None:
    from ultralytics import YOLO, settings

    settings.update({"dvc": True})  # enable the DVCLive callback

    params = load_params()
    train_cfg = dict(params["train"])
    checkpoint = train_cfg.pop("model")
    data_yaml = Path(params["data"]["prepared_dir"]) / "data.yaml"

    model = YOLO(checkpoint)
    model.train(
        data=str(data_yaml),
        project="runs",
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
