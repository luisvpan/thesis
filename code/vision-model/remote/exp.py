"""Run a DVC experiment of this project on a cloud GPU (Modal).

    uv run --no-sync modal run remote/exp.py --name smoke --set train.epochs=3 --gpu L4
    uv run --no-sync modal run remote/exp.py --name imgsz960 --set "train.imgsz=960,train.batch=16" --gpu L40S

What happens (the cross-validation folds run in parallel, one GPU each):

0. On Modal, one container per fold (in parallel): clone at HEAD, apply the
   ``--set`` overrides, ``vm-prepare`` and ``vm-cv --fold k``. Each returns its
   fold's result (metrics, best epoch, threshold). ``--sequential`` skips this
   and lets the ``cv`` stage train the folds one after another instead.

1. Locally: checks that ``code/vision-model`` has no uncommitted changes and that
   HEAD is pushed (the repository is public, so the container clones it without
   credentials).
2. On Modal: clones the repo at HEAD, ``uv sync --frozen`` (on Linux the default
   PyPI PyTorch already ships CUDA), ``dvc pull data/raw.dvc`` from R2,
   ``dvc exp run -n <name> -S ...``, and ``dvc push`` of the experiment's
   outputs (weights, images) to R2. The experiment's git part (params,
   dvc.lock, metrics) comes back as a small ``git bundle``.
3. Locally again: imports that bundle, so the experiment appears in
   ``dvc exp show`` as if it had run here, and pushes its ref to GitHub with
   ``git push`` using your own git credentials (skip with ``--no-push``).

Modal bills per second only while the function runs. It needs one Modal secret,
``vision-model-r2``, with ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` of
the R2 token (DVC's S3 remote reads them; the endpoint is in ``.dvc/config``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal
import yaml

PROJECT_SUBDIR = "code/vision-model"
GPUS = ("T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB", "H100")
PROJECT_DIR = Path(__file__).resolve().parents[1]  # only read locally, to build the image
VENV = "/opt/venv"
WORK_DIR = Path("/work")  # parent of every per-call scratch dir (see checkout)

# The dependencies (PyTorch with CUDA, Ultralytics, DVC...) are installed into the
# IMAGE from uv.lock, so Modal caches them as an image layer and rebuilds it only
# when the lock changes. At runtime `uv sync` then only installs the project itself.
# (A Modal Volume as uv cache does not work: uv cannot persist its files there.)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")  # git for DVC; GL/GLib for OpenCV
    .pip_install("uv", "pyyaml")  # pyyaml: train_fold edits params.yaml
    .env({"UV_PROJECT_ENVIRONMENT": VENV, "UV_PYTHON_PREFERENCE": "only-system"})
    .add_local_file(PROJECT_DIR / "pyproject.toml", "/deps/pyproject.toml", copy=True)
    .add_local_file(PROJECT_DIR / "uv.lock", "/deps/uv.lock", copy=True)
    .add_local_file(PROJECT_DIR / ".python-version", "/deps/.python-version", copy=True)
    # --frozen: use uv.lock as-is. A different uv version in the image can consider
    # the lock "outdated" for cosmetic reasons; freshness is checked locally instead
    # (`uv lock --check` in main), with the uv that created the lock.
    .run_commands("cd /deps && uv sync --frozen --no-install-project")
)
app = modal.App("vision-model-exp", image=image)


def sh(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run a command, streaming its output; raise if it fails."""
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def out(cmd: list[str], cwd: str | Path | None = None) -> str:
    """Run a command and return its stdout."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def set_param(params: dict, override: str) -> None:
    """Apply one ``key.sub=value`` override like ``dvc exp run -S`` (value parsed as YAML)."""
    key, _, raw = override.partition("=")
    *parents, leaf = key.strip().split(".")
    node = params
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)


def fold_ids(params: dict) -> list[int]:
    """Folds the cv stage will use (split.folds, limited by cv.max_folds)."""
    ids = list(range(1, len(params["split"]["folds"]) + 1))
    max_folds = (params.get("cv") or {}).get("max_folds")
    return ids[: int(max_folds)] if max_folds else ids


def clone(repo_url: str, root: Path, attempts: int = 4) -> None:
    """``git clone`` with retries: several fold containers clone at once and GitHub
    sometimes answers one of them with a transient error."""
    for attempt in range(1, attempts + 1):
        try:
            sh(["git", "clone", "--filter=blob:none", repo_url, str(root)])
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            shutil.rmtree(root, ignore_errors=True)
            time.sleep(10 * attempt)


def checkout(repo_url: str, commit: str) -> tuple[Path, Path]:
    """Clone the repo at ``commit`` and install the project; returns (root, project).

    Each call gets its own fresh directory: Modal can reuse a container for another
    input (e.g. after a failed fold), and a fixed path would already exist there.
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=WORK_DIR))
    root = work / "repo"
    project = root / PROJECT_SUBDIR
    clone(repo_url, root)
    sh(["git", "checkout", "--detach", commit], cwd=root)
    sh(["uv", "sync", "--frozen"], cwd=project)  # deps already in the image venv
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    sh(["uv", "run", "--no-sync", "dvc", "pull", "data/raw.dvc"], cwd=project)
    return root, project


def experiment_ref(name: str, cwd: str | Path | None = None) -> str:
    """Git ref DVC created for experiment ``name`` (refs/exps/<baseline>/<name>)."""
    refs = out(["git", "for-each-ref", "--format=%(refname)", "refs/exps"], cwd=cwd).splitlines()
    matches = [r for r in refs if r.rsplit("/", 1)[-1] == name and "/exec/" not in r]
    if len(matches) != 1:
        raise RuntimeError(f"expected one experiment ref named {name!r}, found {matches}")
    return matches[0]


@app.function(
    gpu="L4",  # default; overridden per run with --gpu
    cpu=8,  # Ultralytics' mosaic/augmentations run on CPU: too few cores starve the GPU
    memory=32 * 1024,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("vision-model-r2")],
)
def run_experiment(
    repo_url: str, commit: str, name: str, overrides: list[str], fold_results: dict | None
) -> dict:
    import os

    root, project = checkout(repo_url, commit)
    sh(["git", "config", "user.name", "vision-model (Modal)"], cwd=root)
    sh(["git", "config", "user.email", "vision-model@modal.invalid"], cwd=root)

    env = dict(os.environ)
    if fold_results:
        # The cv stage aggregates these instead of training (it checks they match)
        folds_dir = root.parent / "cv_folds"
        folds_dir.mkdir()
        for k, result in fold_results.items():
            (folds_dir / f"fold{k}.json").write_text(result, encoding="utf-8")
        env["VM_CV_FOLD_RESULTS"] = str(folds_dir)

    cmd = ["uv", "run", "--no-sync", "dvc", "exp", "run", "-n", name]
    for override in overrides:
        cmd += ["-S", override]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=project, env=env, check=True)

    # Data (weights, images) to R2 now; the git part goes home as a bundle.
    sh(["uv", "run", "--no-sync", "dvc", "push"], cwd=project)
    ref = experiment_ref(name, cwd=root)
    bundle = root.parent / "exp.bundle"
    sh(["git", "bundle", "create", str(bundle), ref, f"^{commit}"], cwd=root)

    metrics = project / "reports" / "metrics.json"
    return {
        "ref": ref,
        "bundle": bundle.read_bytes(),
        "metrics": metrics.read_text(encoding="utf-8") if metrics.exists() else None,
    }


@app.function(
    gpu="L4",
    cpu=8,
    memory=32 * 1024,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("vision-model-r2")],
)
def train_fold(repo_url: str, commit: str, overrides: list[str], k: int) -> str:
    """Train and evaluate one cross-validation fold; returns its result (JSON text)."""
    root, project = checkout(repo_url, commit)
    params_path = project / "params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    for override in overrides:
        set_param(params, override)
    params_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    sh(["uv", "run", "--no-sync", "vm-prepare"], cwd=project)
    out = root.parent / f"fold{k}.json"
    sh(["uv", "run", "--no-sync", "vm-cv", "--fold", str(k), "--out", str(out)], cwd=project)
    return out.read_text(encoding="utf-8")


@app.local_entrypoint()
def main(
    name: str, set: str = "", gpu: str = "L4", no_push: bool = False, sequential: bool = False
) -> None:  # `set` gives --set
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {', '.join(GPUS)}")
    project = Path(__file__).resolve().parents[1]
    if out(["git", "status", "--porcelain", "--", "."], cwd=project):
        raise SystemExit(
            "code/vision-model has uncommitted changes. Commit and push them first: "
            "the experiment runs on a fresh clone of the pushed commit."
        )
    lock = subprocess.run(
        ["uv", "lock", "--check"], cwd=project, capture_output=True, text=True, check=False
    )
    if lock.returncode != 0:
        raise SystemExit(
            "uv.lock does not match pyproject.toml. Run `uv lock`, commit and push.\n" + lock.stderr
        )
    commit = out(["git", "rev-parse", "HEAD"], cwd=project)
    if not out(["git", "branch", "-r", "--contains", commit], cwd=project):
        raise SystemExit(f"HEAD {commit[:7]} is not on any remote branch. Push it first.")

    repo_url = out(["git", "remote", "get-url", "origin"], cwd=project)
    if repo_url.startswith("git@github.com:"):
        repo_url = "https://github.com/" + repo_url.removeprefix("git@github.com:")
    overrides = [s.strip() for s in set.split(",") if s.strip()]

    print(f"Experiment '{name}' at {commit[:7]} on {gpu}, overrides: {overrides or 'none'}")
    fold_results = None
    if not sequential:
        params = yaml.safe_load((project / "params.yaml").read_text(encoding="utf-8"))
        for override in overrides:
            set_param(params, override)
        ids = fold_ids(params)
        print(f"Training folds {ids} in parallel ({len(ids)} x {gpu})...")
        results = train_fold.with_options(gpu=gpu).starmap(
            [(repo_url, commit, overrides, k) for k in ids]
        )
        fold_results = dict(zip(ids, results, strict=True))
        for k, r in fold_results.items():
            h = json.loads(r)["headline"]
            print(f"  fold {k}: mAP50={h['map50']:.3f} mAP50-95={h['map50_95']:.3f}")
    result = run_experiment.with_options(gpu=gpu).remote(
        repo_url, commit, name, overrides, fold_results
    )

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "exp.bundle"
        bundle.write_bytes(result["bundle"])
        sh(["git", "fetch", str(bundle), f"+{result['ref']}:{result['ref']}"], cwd=project)
    print(f"\nImported experiment '{name}' ({result['ref']}).")
    print("reports/metrics.json:\n" + (result["metrics"] or "(not produced)"))

    # The data is already in R2 (pushed from Modal), so sharing the experiment only
    # means pushing its git ref. Plain `git push` instead of `dvc exp push`: DVC's
    # built-in git client does not use the system credential manager (it fails with
    # "Authentication failed" on Windows), git does.
    push = ["git", "push", "origin", f"{result['ref']}:{result['ref']}"]
    if no_push:
        print(f"\nNot pushed. Share it later with: {' '.join(push)}")
    else:
        sh(push, cwd=project)
