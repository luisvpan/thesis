"""
Main entry point for the CV system.

Orchestrates the full session lifecycle: config load → hardware init →
calibration → detection loop → WebSocket communication → shutdown.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

#TODO: add ONNX_GPU_ID to .env and use it in card_detector to select GPU (Fix 5: GPU selection)
load_dotenv()

import cv2
import numpy as np

from cv_system.bridge import WebSocketBridge, TouchEvent
from cv_system.bridge.vision_ingest import post_card_batch_async
from cv_system.config import load_config
from cv_system.calibration.calibrator import Calibrator
from cv_system.calibration.result import CalibrationResult
from cv_system.detection.card_detector import CardDetector
from cv_system.detection.touch_detector import TouchDetector
from cv_system.hardware.manager import HardwareManager, HardwareError
from cv_system.transform import RgbImageTransformer, DepthCoordinateTransformer, ResolutionMapper

logger = logging.getLogger(__name__)

def run_websocket_client(bridge: WebSocketBridge) -> None:
    """
    Run WebSocket client in async event loop.

    Args:
        bridge: WebSocketBridge instance to connect and manage.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bridge.loop = loop  # Set loop before connect so it's available for run_coroutine_threadsafe
    try:
        loop.run_until_complete(bridge.connect())
        logger.info("WebSocket connected successfully")
        loop.run_forever()  # Keep loop running to process scheduled coroutines
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        loop.close()


def main() -> None:
    """Main entry point for the CV system.

    Orchestrates the full CV pipeline:
    1. Load configuration
    2. Initialize hardware
    3. Run calibration
    4. Initialize transformers and detector
    5. Initialize and connect WebSocket bridge
    6. Run detection loop
    7. Graceful shutdown
    """
    config_path = Path(os.getenv("CONFIG_PATH", "config/session.json"))
    config = load_config(config_path)

    PROJ_H, PROJ_W = config.camera.projector_resolution
    FULL_PROJ_H, FULL_PROJ_W = PROJ_H, PROJ_W

    print("=" * 60)
    print("CV System Starting")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Camera depth resolution: {config.camera.depth_resolution}")
    print(f"Camera RGB resolution: {config.camera.rgb_resolution}")
    print(f"Bird view / projector canvas: {config.camera.projector_resolution} (h, w)")
    print(f"FPS: {config.camera.fps}")
    print(f"DMax frames: {config.calibration.dmax_num_frames}")
    print(f"OpenCL available: {cv2.ocl.haveOpenCL()}, enabled: {cv2.ocl.useOpenCL()}")
    print("=" * 60)

    hardware = None
    ws_bridge = None
    ws_thread = None
    touch_executor = None
    card_executor = None
    frame_executor = None
    frame_count = 0
    start_time = time.time()

    try:
        # Step 1: Initialize hardware
        print("\n[1/5] Initializing hardware...")
        hardware = HardwareManager()
        try:
            hardware.initialize(config.camera)
            print("  Hardware initialized successfully")
        except HardwareError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        # Step 2: Load or run calibration
        print("\n[2/5] Loading or running calibration...")
        resolution_mapper = ResolutionMapper(config.camera)

        calibration_path = os.getenv("CALIBRATION_PATH")
        calibration_result = None

        if calibration_path:
            try:
                print(f"  Attempting to load calibration from: {calibration_path}")
                calibration_result = CalibrationResult.load(calibration_path)
                print("  Calibration loaded successfully from file")
            except (FileNotFoundError, ValueError) as e:
                print(f"  WARNING: Could not load calibration: {e}")
                print("  Falling back to live calibration...")

        if calibration_result is None:
            calibrator = Calibrator(config, hardware, resolution_mapper)
            calibration_result = calibrator.run()

            print("\nCalibration metadata:")
            print(f"  Frames captured: {calibration_result.metadata['num_frames']}")
            stats = calibration_result.metadata.get("stats", {})
            if stats:
                print(f"  DMax mean: {stats.get('mean', 0):.1f} mm")
                print(f"  DMax std: {stats.get('std', 0):.1f} mm")
                print(f"  Valid pixel ratio: {stats.get('valid_pixel_ratio', 0):.2%}")

            # Save calibration result if CALIBRATION_PATH was specified
            if calibration_path:
                print(f"\n  Saving calibration to: {calibration_path}")
                calibration_result.save(calibration_path)

        # Step 3: Initialize transformers and detector
        print("\n[3/5] Initializing transformers and detector...")

        rgb_image_transformer = RgbImageTransformer(
            calibration_result,
            config.camera,
            projector_corners=config.calibration.projector_corners,
        )
        # Use effective projector-space dimensions from calibration ROI.
        PROJ_W, PROJ_H = rgb_image_transformer.projector_wh
        xs = [int(p[0]) for p in config.calibration.projector_corners]
        ys = [int(p[1]) for p in config.calibration.projector_corners]
        ROI_OFFSET_X = min(xs)
        ROI_OFFSET_Y = min(ys)
        print("  RGB image transformer initialized")
        print(f"  Effective projector ROI (w, h): ({PROJ_W}, {PROJ_H})")
        print(f"  ROI offset in full projector (x, y): ({ROI_OFFSET_X}, {ROI_OFFSET_Y})")

        depth_coordinate_transformer = DepthCoordinateTransformer(calibration_result)
        print("  Depth coordinate transformer initialized")

        print("  Resolution mapper initialized")

        detector = TouchDetector(
            calibration_result.dmax_map,
            rgb_image_transformer,
            depth_coordinate_transformer,
            resolution_mapper,
            config.detection,
            show_debug=True,
        )
        print("  Touch detector initialized")
        print(f"YOLO model path: {os.getenv('YOLO_MODEL_PATH')}")

        model_path = Path(
            os.getenv(
                "YOLO_MODEL_PATH",
                str(
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "models"
                    / "plswork2.onnx"
                ),
            )
        )
        card_detector = CardDetector(rgb_image_transformer, model_path)
        print(f"  Card detector (YOLO) initialized: {model_path}")

        vision_cards_url = os.getenv(
            "VISION_CARDS_INGEST_URL",
            "http://127.0.0.1:3000/api/v1/vision/cards",
        )
        print(f"  Vision cards ingest URL: {vision_cards_url}")

        # Step 4: Initialize WebSocket bridge
        print("\n[4/5] Initializing WebSocket bridge...")
        ws_url = os.getenv("LANGUAGE_RUNTIME_WS_URL", "ws://localhost:3000/live")
        ws_bridge = WebSocketBridge(url=ws_url)
        print("  WebSocket bridge initialized")

        ws_thread = threading.Thread(
            target=run_websocket_client,
            args=(ws_bridge,),
            name="WebSocketClient",
            daemon=True,
        )
        ws_thread.start()
        print("  Waiting for WebSocket connection...")
        time.sleep(2)

        if ws_bridge.state.value != "CONNECTED":
            print(f"  WARNING: WebSocket not connected (state: {ws_bridge.state.value})")
            print("  Continuing without WebSocket communication...")

        # Step 5: Detection loop
        print("\n[5/5] Starting detection loop...")
        print("  Press Ctrl+C to stop\n")

        # Separate executors to avoid contention between touch and card detection
        touch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Touch")
        card_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Card")
        frame_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Frame")

        start_time = time.time()

        cv2.namedWindow("Card Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Card Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow("Card Detection", 1920, 0)

        # Cache for card detection (Fix 3: run every N frames)
        CARD_DETECT_INTERVAL = 1  # TEMP: Set to 1 for profiling
        last_card_view = None
        last_card_dets: list = []

        # Helper for frame acquisition (Fix 2: pipeline)
        def acquire_frames():
            return hardware.get_depth_frame(), hardware.get_rgb_frame()

        # Initial frame acquisition
        depth_frame, rgb_frame = acquire_frames()

        while True:
            try:
                t0 = time.perf_counter()

                # Fix 2: Start acquiring next frame while processing current
                next_frame_future = frame_executor.submit(acquire_frames)

                t1 = time.perf_counter()

                # Fix 1: Do warp ONCE and share with both detectors (UMat stays on GPU)
                rgb_bird = rgb_image_transformer.camera_to_projector(rgb_frame)

                t2 = time.perf_counter()

                # Submit BOTH detections in parallel (don't wait for touch before card)
                touch_future = touch_executor.submit(detector.detect, depth_frame, rgb_bird)

                card_future = None
                t_card_submit = t2  # For timing card from submission
                run_card = frame_count % CARD_DETECT_INTERVAL == 0
                if run_card:
                    card_future = card_executor.submit(card_detector.detect, rgb_bird)
                    t_card_submit = time.perf_counter()

                # Now wait for results (they run in parallel)
                touches, hands_detected = touch_future.result()
                t_touch = time.perf_counter()

                card_time_ms = 0.0
                card_wait_ms = 0.0
                if card_future is not None:
                    t_card_wait_start = time.perf_counter()
                    card_view, card_dets = card_future.result()
                    t_card_end = time.perf_counter()
                    card_time_ms = 1000 * (t_card_end - t_card_submit)  # Total card time
                    card_wait_ms = 1000 * (t_card_end - t_card_wait_start)  # Wait after touch done
                    # Always update - ByteTrack handles occlusions from hands
                    last_card_view = card_view
                    last_card_dets = card_dets
                    post_card_batch_async(
                        vision_cards_url,
                        card_dets,
                        FULL_PROJ_W,
                        FULL_PROJ_H,
                        offset_x=ROI_OFFSET_X,
                        offset_y=ROI_OFFSET_Y,
                    )
                else:
                    card_view = last_card_view if last_card_view is not None else rgb_bird.get()
                    card_dets = last_card_dets

                if touches:
                    # Process all touches, but limit prints to reduce overhead
                    touch_printed = frame_count % 15 == 0
                    for i, (x, y) in enumerate(touches):
                        full_x = x + ROI_OFFSET_X
                        full_y = y + ROI_OFFSET_Y
                        if 0 <= full_x < FULL_PROJ_W and 0 <= full_y < FULL_PROJ_H:
                            touch_event = TouchEvent.from_detected_touch(x=full_x, y=full_y)
                            if touch_printed:
                                print(
                                    f"  Frame {frame_count}: Touch {i+1} "
                                    f"at proj_x={full_x:.0f}, proj_y={full_y:.0f}"
                                )

                            if ws_bridge.state.value == "CONNECTED" and ws_bridge.loop is not None:
                                asyncio.run_coroutine_threadsafe(
                                    ws_bridge.send_touch_event(touch_event.to_dict()),
                                    loop=ws_bridge.loop,
                                )
                else:
                    if frame_count % 30 == 0:
                        print(f"  Frame {frame_count}: No touches detected")

                if card_dets:
                    for d in card_dets:
                        print(
                            f"  Frame {frame_count}: Card {d.label} "
                            f"{d.confidence * 100:.1f}%"
                        )

                # # Fix 4: Show only every 2 frames
                if frame_count % 2 == 0:
                    full_card_view = np.zeros((FULL_PROJ_H, FULL_PROJ_W, 3), dtype=np.uint8)
                    y1 = max(0, ROI_OFFSET_Y)
                    x1 = max(0, ROI_OFFSET_X)
                    y2 = min(FULL_PROJ_H, y1 + card_view.shape[0])
                    x2 = min(FULL_PROJ_W, x1 + card_view.shape[1])
                    src_h = max(0, y2 - y1)
                    src_w = max(0, x2 - x1)
                    if src_h > 0 and src_w > 0:
                        full_card_view[y1:y2, x1:x2] = card_view[:src_h, :src_w]
                    cv2.imshow("Card Detection", full_card_view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1
                elapsed = time.time() - start_time

                # Fix 2: Get next frame (should be ready by now)
                t5 = time.perf_counter()
                depth_frame, rgb_frame = next_frame_future.result()
                t6 = time.perf_counter()
                print(f"  [FPS: {frame_count / elapsed:.1f}] submit={1000*(t1-t0):.1f}ms warp={1000*(t2-t1):.1f}ms touch={1000*(t_touch-t2):.1f}ms card={card_time_ms:.1f}ms(wait={card_wait_ms:.1f}ms) acq={1000*(t6-t5):.1f}ms")

            except KeyboardInterrupt:
                break

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received")

    except Exception as e:
        print(f"\n\nERROR: Unexpected exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("\n" + "=" * 60)
        print("Shutting down gracefully...")

        if touch_executor is not None:
            print("  Shutting down touch executor...")
            touch_executor.shutdown(wait=False)
        if card_executor is not None:
            print("  Shutting down card executor...")
            card_executor.shutdown(wait=False)
        if frame_executor is not None:
            print("  Shutting down frame executor...")
            frame_executor.shutdown(wait=False)
        print("  Executors stopped")

        if ws_bridge is not None:
            print("  Disconnecting WebSocket...")
            ws_bridge.disconnect()
            if ws_thread is not None:
                ws_thread.join(timeout=2)
            print("  WebSocket stopped")

        if hardware is not None:
            hardware.shutdown()
            print("  Hardware shutdown complete")

        cv2.destroyAllWindows()
        print("=" * 60)

        if frame_count > 0:
            elapsed = time.time() - start_time
            print("\nSession statistics:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Elapsed time: {elapsed:.2f}s")
            print(f"  Average FPS: {frame_count / elapsed:.1f}")


if __name__ == "__main__":
    main()