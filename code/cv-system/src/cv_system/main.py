"""
Main entry point for the CV system.

Orchestrates the full session lifecycle: config load → hardware init →
calibration → detection loop → shutdown.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

from cv_system.config import load_config
from cv_system.calibration.calibrator import Calibrator
from cv_system.detection.touch_detector import TouchDetector
from cv_system.hardware.manager import HardwareManager
from cv_system.hardware.manager import HardwareError
from cv_system.transform.transformer import CoordinateTransformer

load_dotenv()  # Load environment variables from .env file if present


def main() -> None:
    """Main entry point for the CV system.

    Orchestrates the full CV pipeline:
    1. Load configuration from file
    2. Initialize Kinect V2 hardware
    3. Run calibration to compute homography and dmax_map
    4. Initialize coordinate transformer and touch detector
    5. Run continuous detection loop
    6. Handle graceful shutdown with hardware cleanup
    """
    # Load config path from environment variable or use default
    config_path_str = os.getenv("CONFIG_PATH", "config/session.json")
    config_path = Path(config_path_str)

    # Load and validate configuration
    config = load_config(config_path)

    print("=" * 60)
    print("CV System Starting")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Camera depth resolution: {config.camera.depth_resolution}")
    print(f"Camera RGB resolution: {config.camera.rgb_resolution}")
    print(f"FPS: {config.camera.fps}")
    print(f"DMax frames: {config.calibration.dmax_num_frames}")
    print(
        f"Depth range: {config.calibration.depth_range_min}-"
        f"{config.calibration.depth_range_max} mm"
    )
    print(f"Ring buffer size: {config.detection.ring_buffer_size}")
    print(f"Touch threshold: {config.detection.touch_threshold} mm")
    print("=" * 60)

    hardware = None

    try:
        # Step 1: Initialize hardware
        print("\n[1/4] Initializing hardware...")
        hardware = HardwareManager()
        try:
            hardware.initialize(config.camera)
            print("  Hardware initialized successfully")
        except HardwareError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        # Step 2: Run calibration
        print("\n[2/4] Running calibration...")
        calibrator = Calibrator(config, hardware)
        calibration_result = calibrator.run()

        # Print calibration metadata
        print("\nCalibration metadata:")
        print(f"  Frames captured: {calibration_result.metadata['num_frames']}")
        print(f"  Depth range: {calibration_result.metadata['depth_range']} mm")
        stats = calibration_result.metadata["stats"]
        print(f"  DMax mean: {stats['mean']:.1f} mm")
        print(f"  DMax std: {stats['std']:.1f} mm")
        print(f"  Valid pixel ratio: {stats['valid_pixel_ratio']:.2%}")

        # Step 3: Initialize transformer and detector
        print("\n[3/4] Initializing transformer and detector...")
        transformer = CoordinateTransformer(calibration_result)
        print("  Coordinate transformer initialized")

        detector = TouchDetector(calibration_result.dmax_map, config.detection)
        print("  Touch detector initialized")

        # Step 4: Run detection loop
        print("\n[4/4] Starting detection loop...")
        print("  Press Ctrl+C to stop\n")

        frame_count = 0
        start_time = time.time()

        while True:
            try:
                # Capture depth frame
                depth_frame = hardware.get_depth_frame()

                # Detect touches in camera space
                touches_camera = detector.detect(depth_frame)

                # Transform touches to projector space
                if touches_camera:
                    # Convert list of tuples to numpy array with shape (N, 1, 2)
                    touches_camera_np = np.array(touches_camera, dtype=np.float32)
                    touches_camera_np = touches_camera_np[:, np.newaxis, :]

                    # Transform to projector space
                    touches_projector = transformer.camera_to_projector(
                        touches_camera_np
                    )

                    # Print touch coordinates
                    for i, pt in enumerate(touches_projector):
                        x, y = pt[0]
                        print(
                            f"  Frame {frame_count}: Touch {i + 1} at "
                            f"proj_x={x:.0f}, proj_y={y:.0f}"
                        )
                else:
                    if frame_count % 30 == 0:  # Print "no touches" every 30 frames
                        print(f"  Frame {frame_count}: No touches detected")

                frame_count += 1

                # Print FPS every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"  [FPS: {fps:.1f}]")

            except KeyboardInterrupt:
                # Continue to finally block for graceful shutdown
                break

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received")

    except Exception as e:
        print(f"\n\nERROR: Unexpected exception: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Guaranteed cleanup
        print("\n" + "=" * 60)
        print("Shutting down gracefully...")
        if hardware is not None:
            hardware.shutdown()
            print("Hardware shutdown complete")
        print("=" * 60)

        if frame_count > 0:
            elapsed = time.time() - start_time
            print("\nSession statistics:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Elapsed time: {elapsed:.2f}s")
            if elapsed > 0:
                print(f"  Average FPS: {frame_count / elapsed:.1f}")


if __name__ == "__main__":
    main()
