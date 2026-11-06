import json

import pytest

from vision_model.raw import Box, class_names, load_export, to_yolo_line

CATEGORIES = [
    {"id": 0, "name": "Cards", "supercategory": "none"},
    {"id": 1, "name": "add", "supercategory": "Cards"},
    {"id": 2, "name": "one", "supercategory": "Cards"},
]


def write_split(root, split, images, annotations):
    d = root / split
    d.mkdir(parents=True)
    data = {"categories": CATEGORIES, "images": images, "annotations": annotations}
    (d / "_annotations.coco.json").write_text(json.dumps(data))


def test_parent_category_is_not_a_class():
    assert class_names(CATEGORIES) == {1: "add", 2: "one"}


def test_load_export_reads_extra_fields(tmp_path):
    write_split(
        tmp_path,
        "train",
        [
            {
                "id": 0,
                "file_name": "predict_000010_jpg.rf.abc.jpg",
                "width": 1920,
                "height": 1080,
                "extra": {
                    "name": "predict_000010.jpg",
                    "user_tags": ["session_rf_20260604"],
                    "user_metadata": {"session": "rf_20260604"},
                },
            }
        ],
        [{"id": 1, "image_id": 0, "category_id": 2, "bbox": [0, 0, 192, 108]}],
    )
    write_split(tmp_path, "valid", [], [])
    images, classes = load_export(tmp_path)
    assert classes == ["add", "one"]
    (image,) = images
    assert image.original_name == "predict_000010.jpg"
    assert image.tags == ("session_rf_20260604",)
    assert image.metadata == {"session": "rf_20260604"}
    assert image.boxes == (Box("one", 0, 0, 192, 108),)


def test_load_export_requires_annotations(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_export(tmp_path)


def test_coco_to_yolo():
    line = to_yolo_line(Box("one", 0, 0, 192, 108), 1, 1920, 1080)
    assert line == "1 0.050000 0.050000 0.100000 0.100000"
