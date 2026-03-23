"""
Main entry point for the CV system.

Loads configuration and orchestrates the session lifecycle.
"""

import os
from pathlib import Path

from cv_system.config import load_config


def main() -> None:
    """Main entry point for the CV system."""
    # Load config path from environment variable or use default
    config_path_str = os.getenv("CONFIG_PATH", "config/session.json")
    config_path = Path(config_path_str)

    # Load and validate configuration
    config = load_config(config_path)

    # Print loaded configuration (for demo purposes)
    print("CV System Configuration")
    print("=" * 40)
    print(f"Config file: {config_path}")
    print(f"Camera depth resolution: {config.camera.depth_resolution}")
    print(f"Camera RGB resolution: {config.camera.rgb_resolution}")
    print(f"FPS: {config.camera.fps}")
    print(f"DMax frames: {config.calibration.num_dmax_frames}")
    print(
        f"Depth range: {config.calibration.depth_range_min}-{config.calibration.depth_range_max} mm"
    )
    print(f"Ring buffer size: {config.detection.ring_buffer_size}")
    print(f"Touch threshold: {config.detection.touch_threshold} mm")
    print("=" * 40)
    print("Configuration loaded successfully!")


if __name__ == "__main__":
    main()
