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
from pathlib import Path
from dotenv import load_dotenv

import cv2

from cv_system.bridge import WebSocketBridge, TouchEvent
from cv_system.bridge.vision_ingest import post_card_batch_async
from cv_system.config import load_config
from cv_system.calibration.calibrator import Calibrator
from cv_system.detection.card_detector import CardDetector
from cv_system.detection.touch_detector import TouchDetector
from cv_system.hardware.manager import HardwareManager, HardwareError
from cv_system.transform import RgbImageTransformer, DepthCoordinateTransformer, ResolutionMapper

load_dotenv()

logger = logging.getLogger(__name__)

def run_websocket_client(bridge: WebSocketBridge) -> None:
    """
    Run WebSocket client in async event loop.

    Args:
        bridge: WebSocketBridge instance to connect and manage.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(bridge.connect())
        logger.info("WebSocket connected successfully")
        loop.run_until_complete(
            bridge.ws.wait_closed() if bridge.ws else asyncio.Future()
        )
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

    # TODO: colocar resolución del proyector en el config
    PROJ_H, PROJ_W = config.camera.rgb_resolution

    print("=" * 60)
    print("CV System Starting")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Camera depth resolution: {config.camera.depth_resolution}")
    print(f"Camera RGB resolution: {config.camera.rgb_resolution}")
    print(f"FPS: {config.camera.fps}")
    print(f"DMax frames: {config.calibration.dmax_num_frames}")
    print("=" * 60)

    hardware = None
    ws_bridge = None
    ws_thread = None
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

        # Step 2: Run calibration
        print("\n[2/5] Running calibration...")
        resolution_mapper = ResolutionMapper(config.camera)
        calibrator = Calibrator(config, hardware, resolution_mapper)
        calibration_result = calibrator.run()

        print("\nCalibration metadata:")
        print(f"  Frames captured: {calibration_result.metadata['num_frames']}")
        stats = calibration_result.metadata.get("stats", {})
        if stats:
            print(f"  DMax mean: {stats.get('mean', 0):.1f} mm")
            print(f"  DMax std: {stats.get('std', 0):.1f} mm")
            print(f"  Valid pixel ratio: {stats.get('valid_pixel_ratio', 0):.2%}")

        # Step 3: Initialize transformers and detector
        print("\n[3/5] Initializing transformers and detector...")

        rgb_image_transformer = RgbImageTransformer(calibration_result, config.camera)
        print("  RGB image transformer initialized")

        depth_coordinate_transformer = DepthCoordinateTransformer(calibration_result)
        print("  Depth coordinate transformer initialized")

        print("  Resolution mapper initialized")

        detector = TouchDetector(
            calibration_result.dmax_map,
            rgb_image_transformer,
            depth_coordinate_transformer,
            resolution_mapper,
            config.detection,
            show_debug=False,
        )
        print("  Touch detector initialized")

        model_path = Path(
            os.getenv(
                "YOLO_MODEL_PATH",
                str(
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "models"
                    / "plswork2.pt"
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

        start_time = time.time()

        cv2.namedWindow("Card Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Card Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow("Card Detection", 1920, 0)

        while True:
            try:
                depth_frame = hardware.get_depth_frame()
                rgb_frame = hardware.get_rgb_frame()

                # detect() receives raw frames, returns projector coordinates directly
                touches = detector.detect(depth_frame, rgb_frame)

                card_view, card_dets = card_detector.detect(rgb_frame)
                post_card_batch_async(vision_cards_url, card_dets, PROJ_W, PROJ_H)

                if touches:
                    for i, (x, y) in enumerate(touches):
                        if 0 <= x < PROJ_W and 0 <= y < PROJ_H:
                            touch_event = TouchEvent.from_detected_touch(x=x, y=y)
                            print(f"  Frame {frame_count}: Touch {i+1} at proj_x={x:.0f}, proj_y={y:.0f}")

                            if ws_bridge.state.value == "CONNECTED" and ws_bridge.loop is not None:
                                asyncio.run_coroutine_threadsafe(
                                    ws_bridge.send_touch_event(touch_event.to_dict()),
                                    loop=ws_bridge.loop,
                                )
                else:
                    if frame_count % 30 == 0:
                        print(f"  Frame {frame_count}: No touches detected")

                if card_dets and frame_count % 30 == 0:
                    for d in card_dets:
                        print(
                            f"  Frame {frame_count}: Card {d.label} "
                            f"{d.confidence * 100:.1f}%"
                        )

                cv2.imshow("Card Detection", card_view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [FPS: {frame_count / elapsed:.1f}]")

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