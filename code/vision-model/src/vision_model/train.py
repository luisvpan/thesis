"""Fine-tune a pretrained YOLO checkpoint on the prepared dataset.

Every hyperparameter comes from ``params.yaml`` -> ``train``; the keys besides
``model`` are passed verbatim to ``YOLO.train``. Metrics and plots are logged
by Ultralytics' built-in DVCLive callback into ``dvclive/``; the best weights
end up in ``dvclive/artifacts/best.pt`` (a DVC-tracked output).
"""

from __future__ import annotations

import shutil
import time
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


def use_native_batchnorm() -> None:
    """Run every BatchNorm layer with PyTorch's own kernels instead of MIOpen/cuDNN.

    Workaround for PyTorch ROCm 7.2.1 on Windows, where MIOpen's BatchNorm kernels
    fail (missing STL headers at JIT compile, then an access violation in
    BatchNormBackward). Convolutions keep using MIOpen; only BatchNorm changes,
    and its backward follows the forward's choice. See ROCm/TheRock#9206.
    """
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm

    original_forward = _BatchNorm.forward

    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            return original_forward(self, x)

    _BatchNorm.forward = forward


def log_epoch_time(model) -> None:
    """Log each epoch's training time (seconds) to DVCLive as ``time/train_epoch_s``.

    Ultralytics prints it but DVCLive does not record it, so experiments could not
    be compared on speed (e.g. GPU type, workers, batch) from `dvc exp show`.
    """
    from ultralytics.utils.callbacks import dvc as ultralytics_dvc

    started: dict[str, float] = {}

    def on_start(trainer) -> None:
        started["t"] = time.perf_counter()

    def on_end(trainer) -> None:
        live = getattr(ultralytics_dvc, "live", None)  # created by the DVCLive callback
        if live is not None and "t" in started:
            live.log_metric("time/train_epoch_s", round(time.perf_counter() - started["t"], 2))

    model.add_callback("on_train_epoch_start", on_start)
    model.add_callback("on_train_epoch_end", on_end)


def main() -> None:
    from ultralytics import YOLO, settings

    settings.update({"dvc": True})  # enable the DVCLive callback
    keep_dvclive_out_of_dvc_yaml()

    params = load_params()
    if params.get("runtime", {}).get("native_batchnorm", False):
        use_native_batchnorm()
        print("runtime.native_batchnorm: BatchNorm uses PyTorch kernels (MIOpen/cuDNN off)")
    train_cfg = dict(params["train"])
    checkpoint = train_cfg.pop("model")
    data_yaml = (Path(params["data"]["prepared_dir"]) / "data.yaml").resolve()

    model = YOLO(checkpoint)
    log_epoch_time(model)
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
