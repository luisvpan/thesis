import json

import pytest

from vision_model.cv import aggregate, fold_fingerprint, load_results, trained_headline


def fold(k, best_epoch, map50, conf, classes):
    return {
        "fold": k,
        "best_epoch": best_epoch,
        "epochs_run": best_epoch + 30,
        "headline": {"map50": map50, "map50_95": map50 / 2, "precision": 0.9, "recall": 0.8},
        "operating_conf": conf,
        "per_class": {
            c: {"ap50": v, "ap50_95": v / 2, "precision": v, "recall": v}
            for c, v in classes.items()
        },
    }


def test_aggregate_means_epochs_conf_and_classes():
    summary = aggregate(
        [
            fold(1, 40, 0.9, 0.7, {"add": 1.0, "one": 0.8}),
            fold(2, 61, 0.7, 0.8, {"add": 0.6}),
        ]
    )
    assert summary["epochs"] == 50  # round(50.5) -> banker's rounding gives 50
    assert abs(summary["operating_conf"] - 0.75) < 1e-9
    assert abs(summary["headline"]["map50"]["mean"] - 0.8) < 1e-9
    assert abs(summary["headline"]["map50"]["std"] - 0.1) < 1e-9
    assert summary["per_class"]["add"]["ap50"]["n"] == 2
    assert summary["per_class"]["one"]["ap50"]["n"] == 1
    assert "per_class" not in summary["folds"][0]


PARAMS = {"train": {"imgsz": 1280}, "runtime": {}}


def make_prepared(tmp_path, train="./images/a.jpg\n"):
    (tmp_path / "fold1_train.txt").write_text(train)
    (tmp_path / "fold1_val.txt").write_text("./images/b.jpg\n")
    return tmp_path


def test_precomputed_results_must_match_data_and_params(tmp_path):
    (tmp_path / "p").mkdir()
    prepared = make_prepared(tmp_path / "p")
    results = tmp_path / "r"
    results.mkdir()
    good = {"fold": 1, "fingerprint": fold_fingerprint(1, prepared, PARAMS)}
    (results / "fold1.json").write_text(json.dumps(good))
    assert load_results(results, [1], prepared, PARAMS) == [good]

    other_params = {"train": {"imgsz": 960}, "runtime": {}}
    with pytest.raises(SystemExit, match="other data or params"):
        load_results(results, [1], prepared, other_params)
    make_prepared(prepared, train="./images/c.jpg\n")
    with pytest.raises(SystemExit, match="other data or params"):
        load_results(results, [1], prepared, PARAMS)


def test_trained_headline_is_mean_over_given_classes():
    classes = {
        "add": {"ap50": 1.0, "ap50_95": 0.5, "precision": 1.0, "recall": 0.8},
        "one": {"ap50": 0.5, "ap50_95": 0.3, "precision": 0.6, "recall": 0.4},
    }
    h = trained_headline(classes)
    assert h["map50"] == 0.75 and abs(h["map50_95"] - 0.4) < 1e-9
