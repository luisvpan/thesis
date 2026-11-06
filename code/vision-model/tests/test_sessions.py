from datetime import datetime
from pathlib import Path

from vision_model.prepare import (
    build_class_list,
    prepared_stem,
    split_for_session,
    unique_stem,
)
from vision_model.raw import RawImage
from vision_model.sessions import UNTIMED_SESSION, assign_sessions, parse_name


def img(name: str, tags=(), metadata=None) -> RawImage:
    return RawImage(
        path=Path(f"raw/train/{name}"),
        original_name=name,
        width=1920,
        height=1080,
        tags=tuple(tags),
        metadata=metadata or {},
    )


def test_parses_timestamped_and_untimed_names():
    kinect = parse_name("kinect_20260423_143441_197978.jpg")
    assert kinect.source == "kinect"
    assert kinect.taken_at == datetime(2026, 4, 23, 14, 34, 41)

    assert parse_name("predict_20261105_153000_000042.jpg").taken_at == datetime(
        2026, 11, 5, 15, 30, 0
    )

    untimed = parse_name("predict_000042.jpg")
    assert untimed.taken_at is None
    assert untimed.source == "predict"


def test_sessions_split_on_time_gap():
    images = [
        img("kinect_20260423_141400_000001.jpg"),
        img("kinect_20260423_142000_000001.jpg"),  # 6 min later: same session
        img("capture_20260423_160000_000001.jpg"),  # 100 min later: new session
        img("predict_000001.jpg"),
    ]
    sessions = assign_sessions(images, gap_minutes=30)
    assert sessions[images[0].path] == sessions[images[1].path] == "20260423_1414"
    assert sessions[images[2].path] == "20260423_1600"
    assert sessions[images[3].path] == UNTIMED_SESSION


def test_roboflow_session_overrides_filename():
    by_meta = img("predict_000010.jpg", metadata={"session": "rf_20260604"})
    by_tag = img("predict_000011.jpg", tags=["background", "session_rf_20260611"])
    meta_wins = img(
        "capture_20260515_201318_265908.jpg",
        tags=["session_from_tag"],
        metadata={"session": "from_meta"},
    )
    sessions = assign_sessions([by_meta, by_tag, meta_wins], gap_minutes=30)
    assert sessions[by_meta.path] == "rf_20260604"
    assert sessions[by_tag.path] == "rf_20260611"
    assert sessions[meta_wins.path] == "from_meta"


def test_split_assignment():
    cfg = {"val_sessions": ["v"], "test_sessions": ["t"], "untimed_to": "train"}
    assert split_for_session("t", cfg) == "test"
    assert split_for_session("v", cfg) == "val"
    assert split_for_session("other", cfg) == "train"
    assert split_for_session(UNTIMED_SESSION, cfg) == "train"


def test_class_list_drops_keeping_order():
    assert build_class_list(["add", "cricle", "one"], drop=["cricle"]) == ["add", "one"]


def test_unique_stem_suffixes_duplicates():
    seen: set[str] = set()
    assert unique_stem("predict_000010", seen) == "predict_000010"
    assert unique_stem("predict_000010", seen) == "predict_000010__1"
    assert unique_stem("predict_000010", seen) == "predict_000010__2"


def test_prepared_stem_carries_session():
    assert prepared_stem("rf_20260604", "predict_000010.jpg") == "rf_20260604__predict_000010"
    assert prepared_stem("20260515_1958", "capture_20260515_201318_265908.jpg") == (
        "20260515_1958__capture_20260515_201318_265908"
    )
