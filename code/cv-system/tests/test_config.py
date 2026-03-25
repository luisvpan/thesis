"""Tests for CV system configuration.

Tests CalibrationConfig and SessionConfig validation.
"""

import tempfile
from pathlib import Path

import pytest

from cv_system.config import CalibrationConfig, SessionConfig


def test_default_calibration_config_no_depth_range():
    """Test that default CalibrationConfig has no depth_range fields."""
    calibration = CalibrationConfig()

    # Should have dmax_num_frames
    assert hasattr(calibration, "dmax_num_frames")
    assert calibration.dmax_num_frames > 0

    # Should NOT have depth_range_min/max (removed in T03)
    assert not hasattr(calibration, "depth_range_min")
    assert not hasattr(calibration, "depth_range_max")


def test_calibration_config_load_without_depth_range():
    """Test that loading config without depth_range works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"

        # Create valid config without depth_range
        config_dict = {
            "camera": {
                "depth_resolution": [424, 512],
                "rgb_resolution": [1080, 1920],
                "fps": 30,
            },
            "calibration": {
                "projector_corners": [[100, 100], [700, 100], [100, 500], [700, 500]],
                "dmax_num_frames": 500,
            },
        }

        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Load config
        config = CalibrationConfig.load(config_path)

        assert isinstance(config, SessionConfig)
        assert isinstance(config.calibration, CalibrationConfig)

        # Verify no depth_range
        assert not hasattr(config.calibration, "depth_range_min")
        assert not hasattr(config.calibration, "depth_range_max")

        # Verify other fields are present
        assert config.calibration.dmax_num_frames == 500
        assert len(config.calibration.projector_corners) == 4


def test_calibration_config_load_with_depth_range_raises_error():
    """Test that loading config with depth_range fields raises ValidationError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "bad_config.json"

        # Create config with depth_range (removed fields)
        config_dict = {
            "camera": {
                "depth_resolution": [424, 512],
                "rgb_resolution": [1080, 1920],
                "fps": 30,
            },
            "calibration": {
                "projector_corners": [[100, 100], [700, 100], [100, 500], [700, 500]],
                "dmax_num_frames": 500,
                "depth_range_min": 650,  # Removed field
                "depth_range_max": 800,  # Removed field
            },
        }

        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Loading should raise ValidationError
        with pytest.raises(ValueError, match="Extra fields are not permitted"):
            CalibrationConfig.load(config_path)
