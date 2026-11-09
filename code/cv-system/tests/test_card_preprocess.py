import json

import numpy as np
import pytest

from cv_system.detection.model_card import load_model_card
from cv_system.detection.preprocess import (
    letterbox,
    stretch,
    to_frame_coords,
    to_model_input,
)


def frame(w=1920, h=1080):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_letterbox_keeps_aspect_and_pads_vertically():
    out, g = letterbox(frame(), 1280)
    assert out.shape == (1280, 1280, 3)
    assert g.scale_x == pytest.approx(1280 / 1920)
    assert g.scale_y == pytest.approx(720 / 1080)
    assert (g.pad_x, g.pad_y) == (0, 280)
    assert out[0, 0, 0] == 114 and out[279, 0, 0] == 114  # padding
    assert out[280, 0, 0] == 0  # image content starts here


def test_stretch_fills_the_square():
    out, g = stretch(frame(), 640)
    assert out.shape == (640, 640, 3)
    assert (g.scale_x, g.scale_y) == (
        pytest.approx(640 / 1920),
        pytest.approx(640 / 1080),
    )
    assert (g.pad_x, g.pad_y) == (0, 0)


@pytest.mark.parametrize("mode", ["letterbox", "stretch"])
def test_boxes_round_trip_to_frame_coordinates(mode):
    _, g = to_model_input(frame(), 1280, mode)
    box = np.array([[960.0, 540.0, 1060.0, 640.0]])  # frame pixels
    in_model = box.copy()
    in_model[:, [0, 2]] = in_model[:, [0, 2]] * g.scale_x + g.pad_x
    in_model[:, [1, 3]] = in_model[:, [1, 3]] * g.scale_y + g.pad_y
    np.testing.assert_allclose(to_frame_coords(in_model, g), box, atol=1e-3)


def test_model_without_card_is_legacy_stretch(tmp_path):
    model = tmp_path / "yolo_11s_ultra.pt"
    model.write_bytes(b"")
    card = load_model_card(model)
    assert card.is_legacy and card.resize == "stretch" and card.conf_threshold is None


def test_model_card_next_to_model(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    (tmp_path / "model_card.json").write_text(
        json.dumps(
            {
                "name": "y11s-dsabc-123",
                "input": {"imgsz": 1280, "resize": "letterbox", "pad_value": 114},
                "conf_threshold": 0.74,
            }
        )
    )
    card = load_model_card(model)
    assert card.resize == "letterbox" and card.imgsz == 1280
    assert card.conf_threshold == pytest.approx(0.74)
    assert card.name == "y11s-dsabc-123" and not card.is_legacy
