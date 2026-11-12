# vision-model

Produces, evaluates and versions the tangible-object detector used by `../cv-system`.
Read `README.md` for the why; this file is the operating manual.

## Build & Run

```bash
uv sync --extra <rocm|cpu>     # deps + PyTorch build for this machine (see README)
uv run --no-sync vm-sessions             # list capture sessions in data/raw
dvc repro                      # full pipeline: prepare -> cv -> train -> evaluate -> export
dvc exp run -S <key>=<value>   # one experiment = one changed param
dvc exp show                   # compare experiments
uv run --no-sync vm-live-metrics <session>.jsonl --expected '{"one": 2}'
uv run --no-sync vm-extract     # frames from data/recordings (new sessions only)
uv run --no-sync vm-upload      # data/extracted -> Roboflow
```

## Validation

Run after every change:

```bash
uv run --no-sync ruff check src/ tests/ && uv run --no-sync ruff format src/ tests/ --check && uv run --no-sync pytest
```

Tests cover the pure logic (session parsing, split assignment, class remapping,
video metrics). They need no GPU and no data. Training/evaluation/export are
verified by running `dvc repro` on real data, which only the humans can do.

## Structure

```
params.yaml              # ALL experiment configuration (DVC-tracked)
dvc.yaml                 # pipeline stages, deps, outs, metrics, plots
src/vision_model/
  params.py              # load params.yaml
  raw.py                 # read the raw Roboflow COCO export
  sessions.py            # image -> capture session; `vm-sessions`
  folds.py               # group k-fold: split.folds -> val sessions per fold
  prepare.py             # raw COCO export -> data/prepared (YOLO) + fold/final/test lists
  cv.py                  # `vm-cv`: one training per fold -> reports/cv/ (mean ± std)
  train.py               # final model on all but test, epochs from cv, logs to dvclive/
  metrics.py             # read Ultralytics validation results
  evaluate.py            # test metrics + cv numbers -> reports/
  export.py              # ONNX + model_card.json -> models/export/
  live_metrics.py        # video metrics from cv-system session logs
  recordings.py          # data/recordings + data/extracted layout
  extract.py             # `vm-extract`: frames + pre-labels from recordings
  upload.py              # `vm-upload`: extracted frames -> Roboflow (session metadata)
remote/exp.py            # run a DVC experiment on a Modal GPU (`uv run --no-sync modal run remote/exp.py`)
tests/
```

## Invariants (do not break)

- **Split by session, never by image.** Do not reintroduce random per-image
  splits or reuse Roboflow's train/valid/test folders.
- **Test is frozen.** Never change `split.test_sessions` to make numbers look
  better; never tune anything (threshold, params, epochs) on test. The operating
  confidence and the epochs come from the folds.
- **Folds are made of whole sessions.** A session is in at most one fold's val,
  never in test; `untimed` images are always train. Do not pick folds by their
  results (that is tuning on val with extra steps).
- **The final model is the last epoch.** It trains on everything except test
  for the folds' mean best epoch, with no val set; never select its "best" epoch.
- **All configuration lives in `params.yaml`.** No hyperparameters hardcoded in
  `src/`. A new knob = a new key in params.yaml + listed under the stage's
  `params:` in dvc.yaml.
- **Augmentations must produce images that can happen on the table.** Flips stay
  off (digits and ascending/descending arrows change meaning). Keep `hsv_h` low:
  several classes differ only by color.
- **Never edit `data/raw` by hand.** It is a DVC-tracked snapshot of a Roboflow
  export in COCO format. Fix labels or sessions in Roboflow and re-export.
- **Quote session ids in params.yaml.** Unquoted `20260508_1502` is a YAML integer.
- **No weights in git.** `.pt`/`.onnx` go through DVC outputs or release assets.
- **Inference must match training.** Preprocessing is letterbox to `imgsz`; any
  change to it must be reflected in `model_card.json` and in the cv-system.
- Only `status == "active"` detections are real; `"lost"` are tracker memory.

## Contract with cv-system

- cv-system writes session data with `CV_RECORD=1`: `videos/<session>.mp4` +
  `.jsonl` (one line per new camera frame) and `sessions/<session>.json`
  manifests. The old `predict` tool is gone; its `predict_<YYYYMMDD_HHMMSS>_<index>`
  images remain in old data.
- vision-model delivers `models/export/` (`model.onnx`, `model.pt`,
  `model_card.json`). `model_card.json` is the interface: class names, input
  size, preprocessing, `conf_threshold`. Changing its schema is a breaking
  change for the cv-system.
- Session resolution lives in `sessions.py` (Roboflow metadata/tag, then filename
  timestamp). If cv-system naming or the Roboflow tag convention changes, update
  it and `tests/test_sessions.py` together.

## Conventions

- Code, comments and docs in English. Thesis prose (elsewhere) is Spanish.
- Python ≥ 3.12, uv, ruff (line length 100), pytest.
- Import `ultralytics` inside `main()` functions so tests and `vm-sessions`
  stay fast and GPU-free.
- DVC stage commands use `uv run --no-sync --no-sync`: each machine syncs once with its
  PyTorch extra (`uv sync --extra rocm|cpu`), and an implicit sync without it
  would replace that build. Keep `--no-sync` in new stages.
