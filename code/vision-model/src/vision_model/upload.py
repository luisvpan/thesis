"""Send extracted frames and their pre-labels to Roboflow (`vm-upload`).

Each image is uploaded with metadata ``session: <id>`` and the tag
``session_<id>`` (what sessions.py reads), into an upload batch named after the
session. Pre-labels go in as annotations with ``addToDataset=false`` by
default: the images wait in Roboflow's annotation queue until someone reviews
them, instead of entering the dataset with the model's mistakes. Frames
without detections are uploaded unannotated.

Progress is kept in ``data/extracted/<session>/uploaded.json``, so an
interrupted upload resumes where it stopped.

Credentials: ``ROBOFLOW_API_KEY`` in the environment or in ``.env`` at the
project root (git-ignored). Never commit it.

Usage::

    vm-upload                       # every extracted session
    vm-upload 20261109_184859
    vm-upload --dry-run 20261109_184859
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vision_model.params import PROJECT_ROOT
from vision_model.recordings import EXTRACTED_DIR
from vision_model.sessions import SESSION_METADATA_KEY, SESSION_TAG_PREFIX

API_URL = "https://api.roboflow.com"
# Roboflow project slug (workspace thesis-s4jik). Override with ROBOFLOW_PROJECT.
DEFAULT_PROJECT = "cards-4ceff"
TIMEOUT_S = 120


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader (no dependency on python-dotenv)."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or read_dotenv(PROJECT_ROOT / ".env").get(name) or default


def upload_image(
    session, api_key: str, project: str, path: Path, batch: str, tag: str, meta: dict
) -> str:
    with path.open("rb") as fh:
        response = session.post(
            f"{API_URL}/dataset/{project}/upload",
            params={"api_key": api_key, "batch": batch, "tag": tag},
            data={"name": path.name, "split": "train", "metadata": json.dumps(meta)},
            files={"file": (path.name, fh, "image/jpeg")},
            timeout=TIMEOUT_S,
        )
    body = response.json() if response.content else {}
    if response.status_code != 200 or not (body.get("success") or body.get("duplicate")):
        raise RuntimeError(f"upload {path.name}: HTTP {response.status_code} {body}")
    return body["id"]


def upload_annotation(
    session,
    api_key: str,
    project: str,
    image_id: str,
    name: str,
    yolo: str,
    labelmap: dict,
    add_to_dataset: bool,
) -> None:
    response = session.post(
        f"{API_URL}/dataset/{project}/annotate/{image_id}",
        params={
            "api_key": api_key,
            "name": name,
            "addToDataset": "true" if add_to_dataset else "false",
        },
        json={"annotationFile": yolo, "labelmap": labelmap},
        timeout=TIMEOUT_S,
    )
    body = response.json() if response.content else {}
    already = response.status_code == 409 and "already annotated" in json.dumps(body)
    if not already and (response.status_code != 200 or not body.get("success")):
        raise RuntimeError(f"annotate {name}: HTTP {response.status_code} {body}")


def upload_session(
    session_id: str, api_key: str, project: str, add_to_dataset: bool, dry_run: bool
) -> None:
    import requests

    folder = EXTRACTED_DIR / session_id
    images = sorted((folder / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"{session_id}: no images in {folder / 'images'}; run vm-extract first")
    labelmap = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
    progress_path = folder / "uploaded.json"
    done: dict[str, str] = (
        json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    )
    pending = [p for p in images if p.name not in done]
    batch = f"session {session_id}"
    tag = f"{SESSION_TAG_PREFIX}{session_id}"
    meta = {SESSION_METADATA_KEY: session_id}
    print(f"{session_id}: {len(pending)} to upload ({len(done)} already) -> project {project}")
    if dry_run:
        return

    http = requests.Session()
    for n, path in enumerate(pending, 1):
        image_id = upload_image(http, api_key, project, path, batch, tag, meta)
        label = folder / "labels" / f"{path.stem}.txt"
        yolo = label.read_text(encoding="utf-8") if label.exists() else ""
        if yolo.strip():
            upload_annotation(
                http, api_key, project, image_id, label.name, yolo, labelmap, add_to_dataset
            )
        done[path.name] = image_id
        progress_path.write_text(json.dumps(done, indent=2), encoding="utf-8")
        if n % 25 == 0 or n == len(pending):
            print(f"  {n}/{len(pending)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sessions", nargs="*", help="Session ids (default: every folder in data/extracted)"
    )
    parser.add_argument(
        "--project", default=None, help=f"Roboflow project slug (default {DEFAULT_PROJECT})"
    )
    parser.add_argument(
        "--add-to-dataset",
        action="store_true",
        help="Put annotated images straight into the dataset (default: wait for review)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be uploaded")
    args = parser.parse_args()

    api_key = setting("ROBOFLOW_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Set ROBOFLOW_API_KEY in the environment or in .env (git-ignored)")
    project = args.project or setting("ROBOFLOW_PROJECT", DEFAULT_PROJECT)
    sessions = args.sessions or sorted(p.name for p in EXTRACTED_DIR.iterdir() if p.is_dir())
    for session_id in sessions:
        upload_session(session_id, api_key or "", project, args.add_to_dataset, args.dry_run)


if __name__ == "__main__":
    main()
