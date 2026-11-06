import pytest

from vision_model.live_metrics import compute


def det(label, track_id, status="active"):
    return {"label": label, "track_id": track_id, "status": status, "confidence": 0.9}


def test_static_scene_metrics():
    frames = [
        {"frame": 0, "t": 0.0, "detections": [det("one", 1), det("one", 2)]},
        {"frame": 1, "t": 0.5, "detections": [det("one", 1), det("one", 2, "lost")]},
        {"frame": 2, "t": 1.0, "detections": [det("one", 1), det("one", 3), det("add", 4)]},
    ]
    result = compute(frames, expected={"one": 2})
    one = result["classes"]["one"]
    assert one["continuity"] == pytest.approx((1 + 0.5 + 1) / 3, abs=1e-4)
    assert one["flicker"] == 1.0  # 2 -> 1 -> 2
    assert one["fragmentation"] == 1.5  # ids {1,2,3} over max 2 simultaneous
    assert result["spurious_per_frame"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["fps"] == 2.0


def test_empty_log_rejected():
    with pytest.raises(ValueError):
        compute([])
