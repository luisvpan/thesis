"""Where session recordings and extracted frames live.

A recording is what the cv-system writes with ``CV_RECORD=1`` for one session,
copied flat into ``data/recordings`` (DVC-tracked)::

    data/recordings/<session>.mp4            raw bird view, one frame per loop iteration
    data/recordings/<session>.jsonl          one line per video frame: time + detections
    data/recordings/<session>.json           session manifest (model, conf, ...)
    data/recordings/<session>.extract.json   written by vm-extract: frames chosen

Extracted frames are an intermediate, regenerable from the recording plus its
``.extract.json``, so ``data/extracted`` is git-ignored and not DVC-tracked::

    data/extracted/<session>/images/video_<session>_<frame>.jpg
    data/extracted/<session>/labels/video_<session>_<frame>.txt   (YOLO, model pre-labels)
    data/extracted/<session>/classes.json                         (class id -> name)
    data/extracted/<session>/uploaded.json                        (written by vm-upload)
"""

from __future__ import annotations

from pathlib import Path

from vision_model.params import PROJECT_ROOT

RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

# Prefix of extracted frame names. sessions.py reads the timestamp after it, so
# the session is recoverable from the filename even without Roboflow metadata.
FRAME_PREFIX = "video"


def video_path(session: str, root: Path = RECORDINGS_DIR) -> Path:
    return root / f"{session}.mp4"


def log_path(session: str, root: Path = RECORDINGS_DIR) -> Path:
    return root / f"{session}.jsonl"


def manifest_path(session: str, root: Path = RECORDINGS_DIR) -> Path:
    return root / f"{session}.json"


def extract_record_path(session: str, root: Path = RECORDINGS_DIR) -> Path:
    return root / f"{session}.extract.json"


def frame_stem(session: str, frame: int) -> str:
    return f"{FRAME_PREFIX}_{session}_{frame:06d}"


def recorded_sessions(root: Path = RECORDINGS_DIR) -> list[str]:
    """Sessions with both a video and a per-frame log, sorted."""
    return sorted(p.stem for p in root.glob("*.mp4") if log_path(p.stem, root).is_file())
