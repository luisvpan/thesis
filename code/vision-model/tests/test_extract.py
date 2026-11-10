import json
from collections import Counter

import cv2
import numpy as np

from vision_model.extract import (
    Candidate,
    class_names,
    extract_session,
    read_log,
    scene_changed,
    select_frames,
    yolo_lines,
)
from vision_model.recordings import extract_record_path, frame_stem, recorded_sessions


def test_yolo_lines_skips_lost_and_normalizes():
    dets = [
        {"class_id": 3, "label": "add", "bbox_xyxy": [100, 50, 300, 250], "status": "active"},
        {"class_id": 4, "label": "one", "bbox_xyxy": [0, 0, 10, 10], "status": "lost"},
    ]
    assert yolo_lines(dets, 1000, 500) == ["3 0.200000 0.300000 0.200000 0.400000"]


def test_yolo_lines_clips_to_image():
    dets = [{"class_id": 0, "label": "x", "bbox_xyxy": [-20, 10, 120, 60]}]
    assert yolo_lines(dets, 100, 100) == ["0 0.500000 0.350000 1.000000 0.500000"]


def test_scene_changed_threshold():
    a = np.zeros((90, 160), np.float32)
    assert scene_changed(None, a, 4.0)
    assert not scene_changed(a, a + 3, 4.0)
    assert scene_changed(a, a + 5, 4.0)
    assert scene_changed(a, a, 0)


def _write_recording(root, session, n=12):
    writer = cv2.VideoWriter(
        str(root / f"{session}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
    )
    with (root / f"{session}.jsonl").open("w") as log:
        for i in range(n):
            # the scene changes only from frame 6 on
            frame = np.full((48, 64, 3), 40 if i < 6 else 200, np.uint8)
            writer.write(frame)
            det = {"class_id": 1, "label": "two", "bbox_xyxy": [8, 8, 32, 24], "status": "active"}
            log.write(json.dumps({"frame": i, "t": i * 0.5, "detections": [det]}) + "\n")
    writer.release()


def test_extract_session(tmp_path):
    rec, out = tmp_path / "rec", tmp_path / "out"
    rec.mkdir()
    session = "20261109_184859"
    _write_recording(rec, session)
    assert recorded_sessions(rec) == [session]

    record = extract_session(session, interval=1.0, min_change=4.0, recordings=rec, extracted=out)

    # frame 0 (first), nothing else until the scene changes at frame 6
    assert record["frames"] == [0, 6]
    assert json.loads(extract_record_path(session, rec).read_text())["frames"] == [0, 6]
    stem = frame_stem(session, 6)
    assert (out / session / "images" / f"{stem}.jpg").exists()
    assert (out / session / "labels" / f"{stem}.txt").read_text().startswith("1 ")
    assert json.loads((out / session / "classes.json").read_text()) == {"1": "two"}
    assert class_names(read_log(rec / f"{session}.jsonl")) == {1: "two"}


def test_select_frames_covers_classes_then_spreads_in_time():
    cands = [Candidate(i, float(i), Counter({0: 1})) for i in range(10)]
    cands.append(Candidate(10, 10.0, Counter({1: 2})))  # the only frame with class 1
    chosen = select_frames(cands, n=3, per_class=1)
    assert 10 in chosen  # coverage first
    assert 0 in chosen  # first frame covers class 0
    assert len(chosen) == 3
    assert chosen[1] == 5  # then the frame farthest from 0 and 10


def test_select_frames_keeps_all_when_n_is_large():
    cands = [Candidate(i, float(i), Counter()) for i in range(4)]
    assert select_frames(cands, n=10, per_class=5) == [0, 1, 2, 3]


def test_extract_session_with_select(tmp_path):
    rec, out = tmp_path / "rec", tmp_path / "out"
    rec.mkdir()
    session = "20261109_184859"
    _write_recording(rec, session)
    record = extract_session(
        session, interval=0, min_change=0, recordings=rec, extracted=out, select=3
    )
    assert record["candidates"] == 12
    assert len(record["frames"]) == 3
    assert len(list((out / session / "images").glob("*.jpg"))) == 3
    assert len(list((out / session / "labels").glob("*.txt"))) == 3


def test_select_frames_does_not_starve_a_rare_class():
    # many frames rich in common classes, one frame with a rare class alone
    rich = [Candidate(i, float(i), Counter({0: 3, 1: 3, 2: 3})) for i in range(20)]
    rare = Candidate(20, 20.0, Counter({9: 1}))
    chosen = select_frames([*rich, rare], n=3, per_class=25)
    assert 20 in chosen
