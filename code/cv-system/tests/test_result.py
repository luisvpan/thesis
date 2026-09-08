"""Tests for CalibrationResult save() and load() methods.

Tests JSON serialization, deserialization, and round-trip fidelity.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cv_system.calibration.result import CalibrationResult


@pytest.fixture
def sample_calibration_result():
    """Create a sample CalibrationResult for testing."""
    H = np.array(
        [[1.0, 0.0, 100.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dmax_map = np.full((10, 10), 700, dtype=np.uint16)
    camera_corners = [(100, 100), (700, 100), (100, 500), (700, 500)]
    metadata = {
        "num_frames": 10,
        "method": "direct",
        "depth_shape": [10, 10],
        "computed_at_ms": 12345,
    }
    return CalibrationResult(
        H=H, dmax_map=dmax_map, camera_corners=camera_corners, metadata=metadata
    )


def test_save_creates_json_file(sample_calibration_result):
    """Test that save() creates a JSON file with correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)

        json_path = output_path.with_suffix(".json")
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "version" in data
        assert "timestamp" in data
        assert "metadata" in data
        assert "H" in data
        assert "dmax_map" in data
        assert "camera_corners" in data


def test_save_serializes_H_matrix(sample_calibration_result):
    """Test that H matrix is serialized correctly as 3x3 list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)

        json_path = output_path.with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)

        H_list = data["H"]
        assert len(H_list) == 3, f"H should have 3 rows, got {len(H_list)}"
        for row in H_list:
            assert len(row) == 3, f"H row should have 3 columns, got {len(row)}"
            assert all(
                isinstance(x, (int, float)) for x in row
            ), f"H values should be numbers, got types: {[type(x) for x in row]}"


def test_save_serializes_dmax_map(sample_calibration_result):
    """Test that dmax_map is serialized correctly as 2D list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)

        json_path = output_path.with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)

        dmax_map_list = data["dmax_map"]
        assert isinstance(dmax_map_list, list), "dmax_map should be a list"
        assert len(dmax_map_list) == 10, f"dmax_map should have 10 rows, got {len(dmax_map_list)}"
        for row in dmax_map_list:
            assert len(row) == 10, f"dmax_map row should have 10 columns, got {len(row)}"


def test_save_serializes_camera_corners(sample_calibration_result):
    """Test that camera_corners is serialized as list of lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)

        json_path = output_path.with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)

        corners_list = data["camera_corners"]
        assert isinstance(corners_list, list), "camera_corners should be a list"
        assert len(corners_list) == 4, f"Should have 4 corners, got {len(corners_list)}"


def test_save_adds_metadata(sample_calibration_result):
    """Test that save() adds timestamp and version to metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)

        json_path = output_path.with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)

        metadata = data["metadata"]
        assert "timestamp" in metadata, "metadata should include timestamp"
        assert "version" in metadata, "metadata should include version"
        assert metadata["version"] == "1.0", f"version should be 1.0, got {metadata['version']}"
        assert "method" in metadata, "metadata should include original method"
        assert metadata["method"] == "direct", f"method should be direct, got {metadata['method']}"


def test_save_creates_parent_directories():
    """Test that save() creates parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "calibration"
        sample_calibration_result().save(nested_path)

        assert nested_path.parent.exists()
        assert nested_path.with_suffix(".json").exists()


def test_save_handles_path_without_extension():
    """Test that save() adds .json extension if path lacks it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result().save(output_path)

        assert not output_path.with_name("calibration").exists()
        assert output_path.with_suffix(".json").exists()


def test_load_reads_json_file(sample_calibration_result):
    """Test that load() reads and parses JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save first
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)
        json_path = output_path.with_suffix(".json")

        # Load back
        loaded = CalibrationResult.load(json_path)

        assert isinstance(loaded, CalibrationResult)


def test_load_deserializes_H_matrix(sample_calibration_result):
    """Test that load() deserializes H matrix correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)
        json_path = output_path.with_suffix(".json")

        loaded = CalibrationResult.load(json_path)

        assert loaded.H.shape == (3, 3), f"H shape mismatch: {loaded.H.shape}"
        assert loaded.H.dtype == np.float32, f"H dtype mismatch: {loaded.H.dtype}"


def test_load_deserializes_dmax_map(sample_calibration_result):
    """Test that load() deserializes dmax_map correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)
        json_path = output_path.with_suffix(".json")

        loaded = CalibrationResult.load(json_path)

        assert loaded.dmax_map.shape == (10, 10), f"dmax_map shape mismatch: {loaded.dmax_map.shape}"
        assert loaded.dmax_map.dtype == np.uint16, f"dmax_map dtype mismatch: {loaded.dmax_map.dtype}"


def test_load_deserializes_camera_corners(sample_calibration_result):
    """Test that load() deserializes camera_corners correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)
        json_path = output_path.with_suffix(".json")

        loaded = CalibrationResult.load(json_path)

        assert loaded.camera_corners == [(100, 100), (700, 100), (100, 500), (700, 500)]


def test_load_preserves_metadata(sample_calibration_result):
    """Test that load() preserves all metadata fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        sample_calibration_result.save(output_path)
        json_path = output_path.with_suffix(".json")

        loaded = CalibrationResult.load(json_path)

        assert "num_frames" in loaded.metadata
        assert loaded.metadata["num_frames"] == 10
        assert "method" in loaded.metadata
        assert loaded.metadata["method"] == "direct"
        assert "depth_shape" in loaded.metadata
        assert loaded.metadata["depth_shape"] == [10, 10]


def test_load_raises_file_not_found():
    """Test that load() raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError, match="Calibration file not found"):
        CalibrationResult.load("/tmp/does_not_exist.json")


def test_load_raises_invalid_json():
    """Test that load() raises ValueError for invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_json = Path(tmpdir) / "bad.json"
        bad_json.write_text("{invalid json content")

        with pytest.raises(ValueError, match="Invalid JSON"):
            CalibrationResult.load(bad_json)


def test_load_raises_missing_required_fields():
    """Test that load() raises ValueError for missing required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create incomplete JSON (missing H)
        incomplete_path = Path(tmpdir) / "incomplete.json"
        with open(incomplete_path, "w") as f:
            json.dump({"dmax_map": [[700, 700], [700, 700]], "camera_corners": [[0, 0], [1, 1], [2, 2], [3, 3]]}, f)

        with pytest.raises(ValueError, match="Missing required fields"):
            CalibrationResult.load(incomplete_path)


def test_load_raises_invalid_H_shape():
    """Test that load() raises ValueError for invalid H shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSON with wrong H shape (2x3 instead of 3x3)
        bad_path = Path(tmpdir) / "bad_shape.json"
        with open(bad_path, "w") as f:
            json.dump({
                "version": "1.0",
                "timestamp": "2024-03-24T23:00:00Z",
                "metadata": {},
                "H": [[1.0, 0.0], [0.0, 1.0]],  # 2x3 - invalid
                "dmax_map": [[700]],
                "camera_corners": [[0, 0], [1, 1], [2, 2], [3, 3]],
            }, f)

        with pytest.raises(ValueError, match="H has invalid shape"):
            CalibrationResult.load(bad_path)


def test_round_trip_fidelity():
    """Test that save() and load() preserve data exactly."""
    original = sample_calibration_result()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration"
        original.save(output_path)
        json_path = output_path.with_suffix(".json")

        loaded = CalibrationResult.load(json_path)

        # Verify H matrix matches
        assert np.allclose(
            original.H, loaded.H
        ), "H matrix mismatch after round-trip"

        # Verify dmax_map matches
        assert np.array_equal(
            original.dmax_map, loaded.dmax_map
        ), "dmax_map mismatch after round-trip"

        # Verify camera_corners match
        assert (
            original.camera_corners == loaded.camera_corners
        ), "camera_corners mismatch after round-trip"

        # Verify metadata matches
        assert original.metadata["num_frames"] == loaded.metadata["num_frames"]
        assert original.metadata["method"] == loaded.metadata["method"]
        assert original.metadata["depth_shape"] == loaded.metadata["depth_shape"]


def test_result_is_frozen():
    """Test that loaded CalibrationResult is immutable (frozen)."""
    result = sample_calibration_result()

    # Try to modify - should raise AttributeError or TypeError
    with pytest.raises((AttributeError, TypeError)):
        result.H[0, 0] = 999.0
