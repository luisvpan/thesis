"""Kinect RGB snapshots on demand (SPACE key) while the CV stack runs.

Snapshots are saved to CAPTURE_OUTPUT_DIR (default: images/raw).
"""

import os
from datetime import datetime
from pathlib import Path

import cv2

try:
    import msvcrt
except ImportError:  # not Windows: no console keyboard polling
    msvcrt = None

SNAPSHOT_KEY = ord(" ")


def snapshot_dir() -> Path:
    return Path(os.getenv("CAPTURE_OUTPUT_DIR", "images/raw"))


def snapshot_requested_from_console() -> bool:
    """Non-blocking check for SPACE in the console (Windows only)."""
    if msvcrt is None:
        return False
    requested = False
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch == " ":
            requested = True
        elif ch == "\x03":
            raise KeyboardInterrupt
    return requested


def _to_numpy(frame):
    return frame.get() if isinstance(frame, cv2.UMat) else frame


def save_snapshot(rgb_frame, rgb_bird=None, tag: str = "") -> None:
    """Save the raw Kinect RGB frame and, if given, its bird's-eye (homography) view."""
    output_dir = snapshot_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"snapshot_{tag}_{timestamp}" if tag else f"snapshot_{timestamp}"
    raw_path = output_dir / f"{prefix}_raw.jpg"
    cv2.imwrite(str(raw_path), _to_numpy(rgb_frame))
    if rgb_bird is None:
        print(f"  [snapshot] Saved {raw_path}")
        return
    bird_path = output_dir / f"{prefix}.jpg"
    cv2.imwrite(str(bird_path), _to_numpy(rgb_bird))
    print(f"  [snapshot] Saved {bird_path} (+ {raw_path.name})")


def wait_key_with_snapshots(grab_rgb_frame, tag: str) -> int:
    """Block like cv2.waitKey() but SPACE (window or console) saves a snapshot.

    grab_rgb_frame() returns the frame to save. Returns the first non-SPACE
    key pressed in an OpenCV window.
    """
    while True:
        key = cv2.waitKey(50)
        if key != -1 and (key & 0xFF) != SNAPSHOT_KEY:
            return key
        if key != -1 or snapshot_requested_from_console():
            save_snapshot(grab_rgb_frame(), tag=tag)
