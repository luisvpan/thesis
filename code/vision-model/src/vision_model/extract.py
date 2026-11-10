"""Pick frames from session recordings for labeling (`vm-extract`).

Frames are chosen the way ``predict --save`` chose them live: at most one every
``--interval`` seconds (recording time, from the log), and only if the scene
changed by at least ``--min-change`` (mean absolute difference, 0-255, on a
160x90 grayscale thumbnail) since the last frame kept. Each frame gets the
model's pre-labels from the same line of the log: only ``active`` detections,
since ``lost`` ones are tracker memory at an old position.

The chosen frame numbers are written to ``<session>.extract.json`` next to the
recording (versioned with it), so the same frames can be regenerated anywhere.

Usage::

    vm-extract                      # every recording without .extract.json
    vm-extract 20261109_184859      # one session
    vm-extract --interval 3 --force 20261109_184859
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vision_model.recordings import (
    EXTRACTED_DIR,
    RECORDINGS_DIR,
    extract_record_path,
    frame_stem,
    log_path,
    recorded_sessions,
    video_path,
)

JPEG_QUALITY = 95


@dataclass(frozen=True)
class FrameLog:
    """One line of a session's .jsonl."""

    frame: int
    t: float
    detections: list[dict]


def read_log(path: Path) -> list[FrameLog]:
    frames = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                frames.append(FrameLog(rec["frame"], float(rec["t"]), rec.get("detections", [])))
    return frames


def class_names(frames: list[FrameLog]) -> dict[int, str]:
    """Class id -> name, from the detections themselves (they carry both)."""
    names: dict[int, str] = {}
    for f in frames:
        for d in f.detections:
            names.setdefault(int(d["class_id"]), d["label"])
    return dict(sorted(names.items()))


def scene_signature(frame_bgr: np.ndarray) -> np.ndarray:
    """Cheap scene fingerprint; same as cv-system's predict."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)


def scene_changed(previous: np.ndarray | None, current: np.ndarray, threshold: float) -> bool:
    if previous is None or threshold <= 0:
        return True
    return float(np.mean(np.abs(current - previous))) >= threshold


def yolo_lines(detections: list[dict], width: int, height: int) -> list[str]:
    """Active detections -> YOLO ``class cx cy w h`` (normalized, clipped to the image)."""
    lines = []
    for d in detections:
        if d.get("status", "active") != "active":
            continue
        x1, y1, x2, y2 = d["bbox_xyxy"]
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{int(d['class_id'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def extract_session(
    session: str,
    interval: float,
    min_change: float,
    recordings: Path = RECORDINGS_DIR,
    extracted: Path = EXTRACTED_DIR,
) -> dict:
    import cv2

    frames = read_log(log_path(session, recordings))
    cap = cv2.VideoCapture(str(video_path(session, recordings)))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path(session, recordings)}")
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames and video_frames != len(frames):
        print(
            f"  WARNING: video has {video_frames} frames, log has {len(frames)}; "
            "using the shorter one (they are written together, so they should match)"
        )

    out = extracted / session
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    names = class_names(frames)
    (out / "classes.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

    chosen: list[int] = []
    boxes = 0
    last_t: float | None = None
    last_signature: np.ndarray | None = None
    total = min(len(frames), video_frames) if video_frames else len(frames)
    for i in range(total):
        if not cap.grab():
            print(f"  WARNING: video ended at frame {i}")
            break
        entry = frames[i]
        if last_t is not None and entry.t - last_t < interval:
            continue
        ok, image = cap.retrieve()
        if not ok:
            continue
        signature = scene_signature(image)
        if not scene_changed(last_signature, signature, min_change):
            continue
        stem = frame_stem(session, entry.frame)
        height, width = image.shape[:2]
        lines = yolo_lines(entry.detections, width, height)
        cv2.imwrite(
            str(out / "images" / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        (out / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        chosen.append(entry.frame)
        boxes += len(lines)
        last_t, last_signature = entry.t, signature
        if len(chosen) % 50 == 0:
            print(f"  {len(chosen)} frames (at {entry.t / 60:.1f} min)", flush=True)
    cap.release()

    record = {
        "session": session,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria": {"interval_s": interval, "min_change": min_change},
        "video_frames": total,
        "frames": chosen,
        "pre_label_boxes": boxes,
    }
    extract_record_path(session, recordings).write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", nargs="*", help="Session ids (default: all not yet extracted)")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Minimum seconds between frames"
    )
    parser.add_argument(
        "--min-change", type=float, default=4.0, help="Minimum scene change (0-255); 0 disables"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-extract sessions that already have .extract.json"
    )
    args = parser.parse_args()

    sessions = args.sessions or recorded_sessions()
    if not sessions:
        raise SystemExit(f"No recordings (<session>.mp4 + .jsonl) in {RECORDINGS_DIR}")
    for session in sessions:
        if not log_path(session).is_file() or not video_path(session).is_file():
            raise SystemExit(f"{session}: missing .mp4 or .jsonl in {RECORDINGS_DIR}")
        if extract_record_path(session).exists() and not args.force:
            print(f"{session}: already extracted (use --force to redo)")
            continue
        print(f"{session}: extracting...", flush=True)
        record = extract_session(session, args.interval, args.min_change)
        print(
            f"{session}: {len(record['frames'])} frames, {record['pre_label_boxes']} pre-label "
            f"boxes -> {EXTRACTED_DIR / session}"
        )
    print(
        "\nNext: `dvc add data/recordings` (new .extract.json files), then "
        "`uv run --no-sync vm-upload` to send the frames to Roboflow."
    )


if __name__ == "__main__":
    main()
