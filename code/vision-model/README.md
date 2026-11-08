# vision-model

Training, evaluation and release pipeline for ERAE's tangible-object detector
(YOLO fine-tuned on the cards, caps, cubes and sticks placed on the projection
surface). The [cv-system](../cv-system) *runs* the model; this project *produces*
it, measures it and versions it.

Pipeline (orchestrated by [DVC](https://dvc.org), configured by `params.yaml`):

```
data/raw ──prepare──▶ data/prepared ──train──▶ dvclive/ ──evaluate──▶ reports/ ──export──▶ models/export/
(Roboflow export)     (split by session)       (metrics, plots,       (test metrics,        (release bundle:
                                                best.pt)               operating conf)       onnx + model_card)
```

## Why it is built this way

Earlier models scored ~97% mAP50 in Roboflow and still failed live. The causes
this pipeline exists to prevent:

| Problem | Rule here |
|---|---|
| Images split one by one: frames of the same session landed in train *and* test, so test measured memory, not generalization. | **Split by capture session.** Whole sessions go to one split (`prepare.py`). |
| Some classes (e.g. `cap_blue`) only existed in train, so they were never evaluated. | `prepare` warns about classes missing from val/test. The current val/test are provisional for this reason (see `params.yaml`). |
| Nobody knew which data/params produced `yolo_11s_ultra.pt`. | Every run is a DVC experiment; every release carries a `model_card.json` with dataset hash and commit. |
| 1920×1080 images stretched to 640×640 in Roboflow; inference letterboxed. | Export from Roboflow **without resize**; train with letterbox at `imgsz: 1280`; the model card states the preprocessing the cv-system must use. |
| Horizontal flips turned digits/arrows into other symbols. | Flips off in the baseline. |
| Image metrics say nothing about flicker or lost IDs. | `vm-live-metrics` measures continuity, flicker and fragmentation on recorded sessions. |

## Setup

```bash
cd code/vision-model
uv sync --extra <rocm|cpu>   # PyTorch build, see below; add --extra analysis for FiftyOne / Grad-CAM
```

DVC is already initialized (`dvc init --subdir`; the git repo root is the thesis repo).

### Choosing the PyTorch build

GPU vendor cannot be detected by uv's platform markers, so each machine picks its
PyTorch build with an extra (they are mutually exclusive, see `pyproject.toml`):

```bash
uv sync --extra rocm   # Windows 11 + AMD Radeon RX 9000/7000 (training PC, RX 9060 XT)
uv sync --extra cpu    # anywhere, CPU only
```

`rocm` installs AMD's official ROCm 7.2.1 build of PyTorch for Windows (Python 3.12
only; AMD Adrenalin driver 26.2.2 or newer). ROCm exposes the GPU through
PyTorch's CUDA API, so `train.device: "0"` is correct. Check it with:

```bash
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

A plain `uv sync` or `uv run` (without the extra) would put the PyPI build back.
That is why the DVC stages call `uv run --no-sync --no-sync`: sync once with the right
extra, then `dvc repro` / `dvc exp run` keep it.

### Remote storage (Cloudflare R2)

Data, weights and images live in a Cloudflare R2 bucket, used through DVC's S3
support (`dvc[s3]`). R2's free tier (10 GB-month, free egress) covers this project.
The bucket URL is shared config; credentials are per machine and never committed:

```bash
dvc remote add -d r2 s3://<bucket>/dvcstore
dvc remote modify r2 endpointurl https://<account-id>.r2.cloudflarestorage.com
dvc remote modify --local r2 access_key_id '<key>'          # -> .dvc/config.local
dvc remote modify --local r2 secret_access_key '<secret>'   # (git-ignored)
dvc push                     # upload what dvc.lock / *.dvc reference
dvc pull                     # download it on another machine
```

Create the key in Cloudflare: R2 → Manage API tokens → token with Object Read & Write
on the bucket.

## Getting the data in

1. In Roboflow, generate a version with **no resize and no augmentations**
   (augmentation happens in Ultralytics, under our control). Auto-orient is
   harmless (cv-system images carry no EXIF rotation).
2. Export it in **COCO JSON** format and unzip into `data/raw/` (one folder per
   Roboflow split, each with `_annotations.coco.json`). COCO is the raw format
   because it keeps each image's original filename, tags and metadata, which is
   how session information assigned in Roboflow reaches the pipeline. The
   export's own train/valid/test split is ignored.
3. Track it: `dvc add data/raw && git add data/raw.dvc .gitignore`.
4. List the sessions and choose val/test:
   ```bash
   uv run --no-sync vm-sessions
   ```
   Put whole sessions in `split.val_sessions` / `split.test_sessions` in
   `params.yaml` (quoted: YAML reads `20260508_1502` as an integer). A good test
   session contains every class and looks like real use (interface on, hands,
   rotations). **Once chosen, the test set is frozen.**

### How an image gets its session

In order of priority (`sessions.py`):

1. Roboflow metadata `session` (e.g. `session: rf_20260604`).
2. A Roboflow tag `session_<id>` (e.g. `session_rf_20260604`).
3. The timestamp in the filename (`capture_…`, `kinect_…`, `predict_<YYYYMMDD_HHMMSS>_…`),
   clustered in time with `data.session_gap_minutes`.
4. Otherwise `untimed` (old `predict_NNNNNN`), sent to `split.untimed_to`.

Images captured before session ids existed can be grouped in Roboflow (tag and
metadata, e.g. by upload date) and re-exported. Images uploaded to Roboflow on
2026-06-04 and 2026-06-11 were tagged `rf_20260604` / `rf_20260611` this way.

## Running experiments

```bash
dvc repro                         # run the pipeline with params.yaml as-is
dvc exp run -S train.imgsz=960    # an experiment that changes ONE thing
dvc exp show                      # table: params vs metrics for every experiment
dvc plots diff                    # compare curves between experiments
dvc exp apply <exp> && git commit # promote the winner to a commit
```

`dvclive/` is written by Ultralytics' built-in DVCLive callback: per-epoch
metrics (`plots/metrics/*.tsv`), final metrics (`metrics.json`), Ultralytics'
plots (`plots/images/`), the confusion matrix (`plots/sklearn/cf.json`) and the
best weights (`artifacts/best.pt`). Small text files are versioned in git;
weights and images go to the DVC remote.

### On a cloud GPU (Modal)

`remote/exp.py` runs a DVC experiment on a [Modal](https://modal.com) GPU,
billed per second only while it runs (the free plan includes USD 30/month).
Modal clones the **pushed** HEAD from GitHub (the repo is public, so no GitHub
credentials are involved): commit and push first.

```bash
uv run --no-sync modal run remote/exp.py --name smoke --set train.epochs=3 --gpu L4
uv run --no-sync modal run remote/exp.py --name imgsz960 --set "train.imgsz=960,train.batch=16" --gpu L40S
uv run --no-sync dvc exp show
```

At the end, without any extra step: the experiment's weights and images are in
R2 (pushed from Modal), its git part (params, `dvc.lock`, metrics) is imported
into your local repo from a `git bundle`, and it is pushed to GitHub with your
own git credentials (`git push` of the experiment ref; pass `--no-push` to keep it local).
Teammates then get it with `dvc exp pull origin <name>`.

`--set` takes comma-separated `-S` overrides; `--gpu` one of `T4, L4, A10, L40S,
A100-40GB, A100-80GB, H100`. The container gets 8 CPU cores (data augmentation
runs on CPU). Dependencies are installed into the Modal image from `uv.lock`:
the first run (and the first after `uv.lock` changes) builds that image, which
takes a few minutes; later runs reuse it.

`modal` is in the `dev` group, so any `uv sync` installs it (no extra needed on
a machine that does not train). One-time setup:

```bash
uv run --no-sync modal setup        # log in
uv run --no-sync modal secret create vision-model-r2 AWS_ACCESS_KEY_ID=<r2-key> AWS_SECRET_ACCESS_KEY=<r2-secret>
```

## Metrics

Image level (`reports/metrics.json`, `reports/per_class.csv`):

- **precision / recall** at the operating threshold.
- **AP** (per class): area under the precision–recall curve as the confidence
  threshold sweeps 0→1. **mAP50** averages AP over classes with a match defined as
  IoU ≥ 0.5; **mAP50-95** averages over IoU 0.5…0.95 (also rewards tight boxes).
- **operating_conf**: the confidence that maximizes mean F1 **on val**. This is
  the threshold the cv-system must use. It is never tuned on test.

Video level (`vm-live-metrics`, from `predict --record-video` logs):

- **continuity**: in a static scene with known contents, fraction of frames in
  which each object is detected. 1.0 = no flicker.
- **flicker**: how often a class's detected count changes frame to frame.
- **fragmentation**: distinct track IDs per simultaneous object. 1.0 = IDs kept.
- **spurious_per_frame**: detections of classes that are not on the table.

```bash
uv run --no-sync vm-live-metrics ../cv-system/images/annotated/videos/<session>.jsonl \
  --expected '{"one": 2, "add": 1, "cap_blue": 3}'
```

## Releases

`models/export/` is the release bundle:

| File | Content |
|---|---|
| `model.onnx`, `model.pt` | weights |
| `model_card.json` | name, dataset md5, commit, class names, input preprocessing, operating threshold, val/test metrics |
| `metrics.json`, `per_class.csv`, `confusion_matrix.json` | the data behind the plots (re-plot with `dvc plots show`) |

Model names are `<arch>-ds<dataset-md5>-<commit>`, e.g. `y11s-dsa1b2c3d-9f8e7d6`.
To publish one:

```bash
gh release create vision-$(jq -r .name models/export/model_card.json) models/export/* \
  --notes-file <notes.md>
```

## Capturing new data

Use the cv-system:

```bash
uv run --no-sync predict --save                 # pre-labeled images, only when the scene changes
uv run --no-sync predict --record-video         # raw session video + per-frame detections
```

Pre-labels are the model's guesses. Review them before training, and look
specially for **objects the model missed**: an unlabeled object teaches the
model that it is background.

## Pending work

- [ ] cv-system ONNX path: replace the stretch `cv2.resize` with letterbox and
      un-pad the boxes; make the PyTorch path use the configured conf/tracker.
- [ ] cv-system reads `model_card.json` (classes, imgsz, preprocessing, conf)
      instead of hardcoded values.
- [ ] Replay evaluation: run a *candidate* model over recorded session videos
      through the cv-system detector (same code as production) and report the
      video metrics, so models are compared on real use before deploying.
- [ ] Capture dedicated val/test sessions covering all classes (current ones are
      provisional and evaluate only part of them).
- [ ] Roboflow: review the `predict_*` pre-labels (missing boxes), label the
      unannotated images, tag the 2026-05-28 bulk upload by session if possible.
- [ ] Ground truth for moving-object clips (track IDs) to compute IDF1/HOTA.
- [ ] Hard-example capture in `predict`: also save frames with low confidence
      or lost tracks.
