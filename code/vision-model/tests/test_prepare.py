import json

import yaml

from vision_model import prepare

CATEGORIES = [
    {"id": 0, "name": "Cards", "supercategory": "none"},
    {"id": 1, "name": "add", "supercategory": "Cards"},
    {"id": 2, "name": "one", "supercategory": "Cards"},
]


def make_export(raw):
    images, annotations = [], []
    for i, (name, session, cat) in enumerate(
        [
            ("a1.jpg", "sa", 1),
            ("a2.jpg", "sa", 2),
            ("b1.jpg", "sb", 1),
            ("b2.jpg", "sb", 2),
            ("o1.jpg", "old", 1),
            ("t1.jpg", "st", 2),
        ]
    ):
        (raw / "train").mkdir(parents=True, exist_ok=True)
        (raw / "train" / name).write_bytes(b"jpg")
        images.append(
            {
                "id": i,
                "file_name": name,
                "width": 100,
                "height": 50,
                "extra": {"name": name, "user_metadata": {"session": session}},
            }
        )
        annotations.append({"id": i, "image_id": i, "category_id": cat, "bbox": [10, 10, 20, 10]})
    data = {"categories": CATEGORIES, "images": images, "annotations": annotations}
    (raw / "train" / "_annotations.coco.json").write_text(json.dumps(data))


def test_prepare_writes_folds_final_and_test(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "prepared"
    make_export(raw)
    params = {
        "data": {
            "raw_dir": str(raw),
            "prepared_dir": str(out),
            "session_gap_minutes": 30,
            "drop_classes": [],
        },
        "split": {"test_sessions": ["st"], "folds": [["sa"], ["sb"], "rest"]},
    }
    monkeypatch.setattr(prepare, "load_params", lambda: params)
    prepare.main()

    def listed(name):
        return (out / name).read_text().split()

    assert listed("fold1_val.txt") == ["./images/sa__a1.jpg", "./images/sa__a2.jpg"]
    assert "./images/sa__a1.jpg" not in listed("fold1_train.txt")
    assert listed("fold3_val.txt") == ["./images/old__o1.jpg"]
    assert all("st__" not in f for k in (1, 2, 3) for f in listed(f"fold{k}_train.txt"))
    assert listed("test.txt") == ["./images/st__t1.jpg"]
    assert len(listed("final_train.txt")) == 5

    final = yaml.safe_load((out / "final.yaml").read_text())
    assert final["test"] == "test.txt" and final["names"] == {0: "add", 1: "one"}
    assert (out / "labels" / "sa__a1.txt").read_text().startswith("0 ")
    summary = json.loads((out / "manifest.json").read_text())["summary"]
    assert summary["folds"][2]["classes_not_validated"] == ["one"]
