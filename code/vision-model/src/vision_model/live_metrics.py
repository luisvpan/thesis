"""Video-level metrics from a cv-system session log (``CV_RECORD=1``).

The ``.jsonl`` log has one line per frame with the detections the LIVE model
produced. These metrics describe how that model behaved over time, which
image-level mAP cannot show:

- continuity (needs ``--expected``): for a static scene with known contents
  (e.g. ``{"one": 2, "add": 1}``), the mean over frames of
  ``min(detected, expected) / expected`` per class. 1.0 = never flickers out.
- spurious: mean detections per frame of classes NOT in ``--expected``.
- flicker: fraction of consecutive frame pairs where a class's active count changes.
- fragmentation: distinct track ids / max simultaneous instances, per class.
  On a static scene each physical object should keep one id -> 1.0 is ideal.

Only ``status == "active"`` detections count; "lost" tracks are tracker memory.

Usage:
    uv run vm-live-metrics data/videos/20261105_153000.jsonl --expected '{"one": 2}'
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def load_frames(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_frame_counts(frames: list[dict]) -> list[dict[str, int]]:
    counts = []
    for frame in frames:
        c: dict[str, int] = defaultdict(int)
        for d in frame["detections"]:
            if d.get("status", "active") == "active":
                c[d["label"]] += 1
        counts.append(dict(c))
    return counts


def compute(frames: list[dict], expected: dict[str, int] | None = None) -> dict:
    if not frames:
        raise ValueError("empty log")
    counts = per_frame_counts(frames)
    labels = sorted({label for c in counts for label in c} | set(expected or {}))

    duration = frames[-1]["t"] - frames[0]["t"]
    result: dict = {
        "frames": len(frames),
        "duration_s": round(duration, 3),
        "fps": round((len(frames) - 1) / duration, 2) if duration > 0 else None,
        "classes": {},
    }

    track_ids: dict[str, set[int]] = defaultdict(set)
    for frame in frames:
        for d in frame["detections"]:
            if d.get("status", "active") == "active" and d.get("track_id", -1) >= 0:
                track_ids[d["label"]].add(d["track_id"])

    for label in labels:
        series = [c.get(label, 0) for c in counts]
        changes = sum(1 for a, b in pairwise(series) if a != b)
        peak = max(series)
        stats = {
            "mean_count": round(sum(series) / len(series), 4),
            "max_count": peak,
            "flicker": round(changes / (len(series) - 1), 4) if len(series) > 1 else 0.0,
            "fragmentation": round(len(track_ids[label]) / peak, 4)
            if peak and track_ids[label]
            else None,
        }
        if expected and label in expected:
            want = expected[label]
            stats["continuity"] = round(sum(min(n, want) / want for n in series) / len(series), 4)
        result["classes"][label] = stats

    if expected:
        cont = [s["continuity"] for s in result["classes"].values() if "continuity" in s]
        result["mean_continuity"] = round(sum(cont) / len(cont), 4) if cont else None
        spurious = [sum(n for k, n in c.items() if k not in expected) for c in counts]
        result["spurious_per_frame"] = round(sum(spurious) / len(spurious), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-level metrics from a session log")
    parser.add_argument("log", type=Path, help="<session>.jsonl recorded by the cv-system (CV_RECORD=1)")
    parser.add_argument(
        "--expected",
        type=str,
        default=None,
        help="JSON with the true per-class counts of a static scene",
    )
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args()

    expected = json.loads(args.expected) if args.expected else None
    result = compute(load_frames(args.log), expected)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
