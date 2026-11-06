"""Capture-session identification.

Splitting train/val/test by *session* (not by image) is the core rule of this
project: frames from the same session are near-duplicates, so a per-image split
leaks training data into val/test and inflates every metric.

A session id comes from, in order of priority:

1. Roboflow metadata ``session`` (``extra.user_metadata.session`` in the export).
2. A Roboflow tag ``session_<id>``.
3. The timestamp in the filename. Names produced by the cv-system look like::

       kinect_20260423_143441_197978.jpg   (early capture script, timestamp + microseconds)
       capture_20260515_201318_265908.jpg  (cv-system `capture`)
       predict_20261105_153000_000042.jpg  (cv-system `predict`, session start + index)

   Timestamped images are clustered in time: a new session starts whenever the
   gap to the previous image exceeds ``gap_minutes``; the id is the first
   image's ``YYYYMMDD_HHMM``.
4. Otherwise ``"untimed"`` (e.g. ``predict_000042.jpg``, before session ids existed).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from vision_model.raw import RawImage

UNTIMED_SESSION = "untimed"
SESSION_TAG_PREFIX = "session_"
SESSION_METADATA_KEY = "session"

_TIMESTAMPED = re.compile(r"^(?P<source>[A-Za-z]+)_(?P<date>\d{8})_(?P<time>\d{6})(?:_\d+)?$")


@dataclass(frozen=True)
class NameInfo:
    """What a filename tells about its capture."""

    stem: str
    source: str
    taken_at: datetime | None


def parse_name(name: str) -> NameInfo:
    """Source and timestamp from an original filename; timestamp is None if absent."""
    stem = Path(name).stem
    match = _TIMESTAMPED.match(stem)
    if match is None:
        source = stem.split("_", 1)[0] if "_" in stem else stem
        return NameInfo(stem, source, None)
    taken_at = datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M%S")
    return NameInfo(stem, match["source"], taken_at)


def explicit_session(image: RawImage) -> str | None:
    """Session assigned in Roboflow (metadata first, then tag), if any."""
    value = image.metadata.get(SESSION_METADATA_KEY)
    if value:
        return str(value)
    for tag in image.tags:
        if tag.startswith(SESSION_TAG_PREFIX) and len(tag) > len(SESSION_TAG_PREFIX):
            return tag[len(SESSION_TAG_PREFIX) :]
    return None


def assign_sessions(images: list[RawImage], gap_minutes: float) -> dict[Path, str]:
    """Map each image path to its session id."""
    sessions: dict[Path, str] = {}
    timed: list[tuple[datetime, Path]] = []
    for image in images:
        if (explicit := explicit_session(image)) is not None:
            sessions[image.path] = explicit
            continue
        info = parse_name(image.original_name)
        if info.taken_at is None:
            sessions[image.path] = UNTIMED_SESSION
        else:
            timed.append((info.taken_at, image.path))

    gap = timedelta(minutes=gap_minutes)
    current_id = ""
    previous: datetime | None = None
    for taken_at, path in sorted(timed):
        if previous is None or taken_at - previous > gap:
            current_id = taken_at.strftime("%Y%m%d_%H%M")
        sessions[path] = current_id
        previous = taken_at
    return sessions


def main() -> None:
    """List sessions found in the raw export, to decide which go to val/test."""
    from vision_model.params import load_params
    from vision_model.raw import load_export

    parser = argparse.ArgumentParser(description="List capture sessions in the raw dataset")
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()

    params = load_params()
    raw_dir = args.raw_dir or Path(params["data"]["raw_dir"])
    drop = set(params["data"].get("drop_classes", []))
    images, classes = load_export(raw_dir)
    sessions = assign_sessions(images, params["data"]["session_gap_minutes"])

    by_session: dict[str, list[RawImage]] = defaultdict(list)
    for image in images:
        by_session[sessions[image.path]].append(image)

    kept = [c for c in classes if c not in drop]
    print(f"{len(images)} images, {len(by_session)} sessions, {len(kept)} classes in {raw_dir}\n")
    for session_id in sorted(by_session):
        refs = by_session[session_id]
        counts = Counter(b.class_name for r in refs for b in r.boxes if b.class_name not in drop)
        sizes = Counter(f"{r.width}x{r.height}" for r in refs)
        print(
            f"{session_id}: {len(refs)} images, {sum(counts.values())} objects, "
            f"{len(counts)}/{len(kept)} classes, sizes={dict(sizes)}"
        )
        print("    " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
