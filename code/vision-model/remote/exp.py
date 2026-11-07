"""Run a DVC experiment of this project on a cloud GPU (Modal).

    uv run --no-sync modal run remote/exp.py --name smoke --set train.epochs=3 --gpu L4
    uv run --no-sync modal run remote/exp.py --name imgsz960 --set "train.imgsz=960,train.batch=16" --gpu L40S

What happens:

1. Locally: checks that ``code/vision-model`` has no uncommitted changes and that
   HEAD is pushed (the repository is public, so the container clones it without
   credentials).
2. On Modal: clones the repo at HEAD, ``uv sync --locked`` (on Linux the default
   PyPI PyTorch already ships CUDA), ``dvc pull data/raw.dvc`` from R2,
   ``dvc exp run -n <name> -S ...``, and ``dvc push`` of the experiment's
   outputs (weights, images) to R2. The experiment's git part (params,
   dvc.lock, metrics) comes back as a small ``git bundle``.
3. Locally again: imports that bundle, so the experiment appears in
   ``dvc exp show`` as if it had run here, and pushes it to GitHub with
   ``dvc exp push origin <name>`` using your own git credentials (skip with
   ``--no-push``).

Modal bills per second only while the function runs. It needs one Modal secret,
``vision-model-r2``, with ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` of
the R2 token (DVC's S3 remote reads them; the endpoint is in ``.dvc/config``).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import modal

PROJECT_SUBDIR = "code/vision-model"
GPUS = ("T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB", "H100")
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")  # git for DVC; GL/GLib for OpenCV
    .pip_install("uv")
    .env(
        {
            "UV_CACHE_DIR": f"{CACHE_DIR}/uv",  # persisted: PyTorch is downloaded once
            "UV_LINK_MODE": "copy",  # cache and project live on different filesystems
            "UV_PYTHON_PREFERENCE": "only-system",
        }
    )
)
cache = modal.Volume.from_name("vision-model-cache", create_if_missing=True)
app = modal.App("vision-model-exp", image=image)


def sh(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run a command, streaming its output; raise if it fails."""
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def out(cmd: list[str], cwd: str | Path | None = None) -> str:
    """Run a command and return its stdout."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


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
    volumes={CACHE_DIR: cache},
    secrets=[modal.Secret.from_name("vision-model-r2")],
)
def run_experiment(repo_url: str, commit: str, name: str, overrides: list[str]) -> dict:
    root = Path("/work/repo")
    project = root / PROJECT_SUBDIR

    sh(["git", "clone", "--filter=blob:none", repo_url, str(root)])
    sh(["git", "checkout", "--detach", commit], cwd=root)
    sh(["git", "config", "user.name", "vision-model (Modal)"], cwd=root)
    sh(["git", "config", "user.email", "vision-model@modal.invalid"], cwd=root)

    try:
        sh(["uv", "sync", "--locked"], cwd=project)
    finally:
        cache.commit()  # keep the uv cache (PyTorch) even if something fails later
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    sh(["uv", "run", "--no-sync", "dvc", "pull", "data/raw.dvc"], cwd=project)

    cmd = ["uv", "run", "--no-sync", "dvc", "exp", "run", "-n", name]
    for override in overrides:
        cmd += ["-S", override]
    sh(cmd, cwd=project)

    # Data (weights, images) to R2 now; the git part goes home as a bundle.
    sh(["uv", "run", "--no-sync", "dvc", "push"], cwd=project)
    ref = experiment_ref(name, cwd=root)
    bundle = Path("/work/exp.bundle")
    sh(["git", "bundle", "create", str(bundle), ref, f"^{commit}"], cwd=root)

    metrics = project / "reports" / "metrics.json"
    return {
        "ref": ref,
        "bundle": bundle.read_bytes(),
        "metrics": metrics.read_text(encoding="utf-8") if metrics.exists() else None,
    }


@app.local_entrypoint()
def main(
    name: str, set: str = "", gpu: str = "L4", no_push: bool = False
) -> None:  # `set` gives --set
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {', '.join(GPUS)}")
    project = Path(__file__).resolve().parents[1]
    if out(["git", "status", "--porcelain", "--", "."], cwd=project):
        raise SystemExit(
            "code/vision-model has uncommitted changes. Commit and push them first: "
            "the experiment runs on a fresh clone of the pushed commit."
        )
    commit = out(["git", "rev-parse", "HEAD"], cwd=project)
    if not out(["git", "branch", "-r", "--contains", commit], cwd=project):
        raise SystemExit(f"HEAD {commit[:7]} is not on any remote branch. Push it first.")

    repo_url = out(["git", "remote", "get-url", "origin"], cwd=project)
    if repo_url.startswith("git@github.com:"):
        repo_url = "https://github.com/" + repo_url.removeprefix("git@github.com:")
    overrides = [s.strip() for s in set.split(",") if s.strip()]

    print(f"Experiment '{name}' at {commit[:7]} on {gpu}, overrides: {overrides or 'none'}")
    result = run_experiment.with_options(gpu=gpu).remote(repo_url, commit, name, overrides)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "exp.bundle"
        bundle.write_bytes(result["bundle"])
        sh(["git", "fetch", str(bundle), f"+{result['ref']}:{result['ref']}"], cwd=project)
    print(f"\nImported experiment '{name}' ({result['ref']}).")
    print("reports/metrics.json:\n" + (result["metrics"] or "(not produced)"))

    if no_push:
        print(
            f"\nNot pushed. Share it later with: "
            f"uv run --no-sync dvc exp push --no-cache origin {name}"
        )
    else:
        # --no-cache: the data is already in R2 (pushed from Modal) and not in the local cache.
        sh(
            ["uv", "run", "--no-sync", "dvc", "exp", "push", "--no-cache", "origin", name],
            cwd=project,
        )
