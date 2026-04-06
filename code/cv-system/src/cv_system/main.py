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

import numpy as np

from cv_system.bridge import WebSocketBridge, TouchEvent
from cv_system.calibration.marker_projector import MarkerProjector
from cv_system.config import load_config
from cv_system.calibration.calibrator import Calibrator
from cv_system.detection.touch_detector import TouchDetector
from cv_system.hardware.manager import HardwareManager
from cv_system.hardware.manager import HardwareError
from cv_system.transform.transformer import CoordinateTransformer

load_dotenv()  # Load environment variables from .env file if present

logger = logging.getLogger(__name__)


def run_websocket_client(bridge: WebSocketBridge) -> None:
    """
    Run WebSocket client in async event loop.

    This function runs in a separate thread to avoid blocking the
    synchronous detection loop.

    Args:
        bridge: WebSocketBridge instance to connect and manage.
    """
    # Get asyncio event loop
    loop = asyncio.new_event_loop()

    try:
        # Run the async connect method
        loop.run_until_complete(bridge.connect())
        logger.info("WebSocket connected successfully")

        # Listen for messages (this blocks until connection closes)
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
    1. Load configuration from file
    2. Initialize Kinect V2 hardware
    3. Run calibration to compute homography and dmax_map
    4. Initialize coordinate transformer and touch detector
    5. Initialize and connect WebSocket Bridge
    6. Run continuous detection loop with WebSocket event sending
    7. Handle graceful shutdown with hardware and WebSocket cleanup
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
    print("=" * 60)

    hardware = None
    ws_bridge = None
    ws_thread = None

    try:
        # Step 1: Initialize hardware
        print("\n[1/6] Initializing hardware...")
        hardware = HardwareManager()
        try:
            hardware.initialize(config.camera)
            print("  Hardware initialized successfully")
        except HardwareError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        # Step 2: Run calibration
        print("\n[2/6] Running calibration...")
        calibrator = Calibrator(config, hardware)
        calibration_result = calibrator.run()

        # Print calibration metadata
        print("\nCalibration metadata:")
        print(f"  Frames captured: {calibration_result.metadata['num_frames']}")
        stats = calibration_result.metadata.get("stats", {})
        if stats:
            print(f"  DMax mean: {stats.get('mean', 0):.1f} mm")
            print(f"  DMax std: {stats.get('std', 0):.1f} mm")
            print(f"  Valid pixel ratio: {stats.get('valid_pixel_ratio', 0):.2%}")

        # Step 3: Initialize transformer and detector
        print("\n[3/6] Initializing transformer and detector...")
        transformer = CoordinateTransformer(calibration_result)
        print("  Coordinate transformer initialized")

        detector = TouchDetector(calibration_result.dmax_map, config.detection)
        print("  Touch detector initialized")

        # Step 4: Initialize WebSocket Bridge
        print("\n[4/6] Initializing WebSocket Bridge...")

        # Get WebSocket URL from environment variable or use default
        ws_url = os.getenv("LANGUAGE_RUNTIME_WS_URL", "ws://localhost:3000/live")
        logger.info(f"WebSocket URL: {ws_url}")

        # Instantiate WebSocket Bridge
        ws_bridge = WebSocketBridge(url=ws_url)
        print("  WebSocket Bridge initialized")

        # Start WebSocket client in background thread
        ws_thread = threading.Thread(
            target=run_websocket_client,
            args=(ws_bridge,),
            name="WebSocketClient",
            daemon=True,
        )
        ws_thread.start()
        print("  WebSocket client starting in background thread...")

        # Wait for WebSocket connection (give it 2 seconds to connect)
        print("  Waiting for WebSocket connection...")
        time.sleep(2)

        # Check if WebSocket is connected
        if ws_bridge.state.value != "CONNECTED":
            print(
                f"  WARNING: WebSocket not connected (state: {ws_bridge.state.value})"
            )
            print("  Continuing without WebSocket communication...")

        # Step 5: Run detection loop
        print("\n[5/6] Starting detection loop...")
        print("  Press Ctrl+C to stop\n")

        frame_count = 0
        start_time = time.time()

        PROJ_W, PROJ_H = 1920, 1080

        cv2.namedWindow("Projector View", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Projector View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.moveWindow("Projector View", 1920, 0)

        while True:
            try:
                # Capture depth frame
                depth_frame = hardware.get_depth_frame()

                # Detect touches in camera space
                touches_camera = detector.detect(depth_frame)

                # print("Touches detected in camera space:")
                # print(touches_camera)

                # 1. Crear el lienzo negro (Fondo de la pantalla del proyector)
                # Usamos 3 canales (BGR) para poder dibujar en verde
                canvas = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)

                # Transform touches to projector space
                if touches_camera:
                    touches_camera_np = np.array(touches_camera, dtype=np.float32)

                    # Transform to projector space
                    touches_projector = transformer.camera_to_projector(
                        touches_camera_np
                    )

                    # Send touch events via WebSocket
                    for i, (x, y) in enumerate(touches_projector):
                        # Create TouchEvent with timestamp
                        # Solo procesar si están dentro de los límites del proyector
                        if 0 <= x < 1920 and 0 <= y < 1080:
                            touch_event = TouchEvent.from_detected_touch(
                                x=float(x), y=float(y)
                            )
                            # ... enviar por WebSocket ...

                            # Coordenadas como enteros para OpenCV
                            pos = (int(x), int(y))

                            # Dibujar círculo verde: (Lienzo, Centro, Radio, Color BGR, Grosor)
                            cv2.circle(
                                canvas, pos, radius=20, color=(0, 255, 0), thickness=-1
                            )  # Dibuja un círculo verde en la posición del toque

                            # Print touch coordinates
                            print(
                                f"  Frame {frame_count}: Touch {i + 1} at "
                                f"proj_x={x:.0f}, proj_y={y:.0f}"
                            )
                        else:
                            # Opcional: log para debug
                            # print(f"Toque fuera de rango: ({x:.1f}, {y:.1f})")
                            pass
                        # print(touch_event)

                        # Send via WebSocket (if connected)
                        # if ws_bridge.state.value == "CONNECTED" and ws_bridge.ws is not None:
                        #     # Send in async context - need to schedule on the event loop
                        #     asyncio.run_coroutine_threadsafe(
                        #         ws_bridge.send_touch_event(touch_event.to_dict()),
                        #         loop=asyncio.get_event_loop()
                        #     )
                        # else:
                        #     logger.warning(
                        #         f"Touch {i + 1} at ({x:.1f}, {y:.1f}) - WebSocket not connected, skipping"
                        #     )
                else:
                    if frame_count % 30 == 0:  # Print "no touches" every 30 frames
                        print(f"  Frame {frame_count}: No touches detected")

                # 3. Mostrar la ventana
                cv2.imshow("Projector View", canvas)

                # 4. Manejo de salida (ESC o 'q')
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

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

        # Disconnect WebSocket
        if ws_bridge is not None:
            print("  Disconnecting WebSocket...")
            ws_bridge.disconnect()
            print("  WebSocket disconnected")
            if ws_thread is not None:
                ws_thread.join(timeout=2)
                print("  WebSocket thread stopped")

        # Shutdown hardware
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


import cv2

if __name__ == "__main__":
    config_path_str = os.getenv("CONFIG_PATH", "config/session.json")
    config_path = Path(config_path_str)

    config = load_config(config_path)

    print("\n[1/6] Initializing hardware...")
    hardware = HardwareManager()
    try:
        hardware.initialize(config.camera)
        print("  Hardware initialized successfully")
    except HardwareError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Step 2: Run calibration
    print("\n[2/6] Running calibration...")
    calibrator = Calibrator(config, hardware)
    calibration_result = calibrator.run()

    # Print calibration metadata
    print("\nCalibration metadata:")
    print(f"  Frames captured: {calibration_result.metadata['num_frames']}")
    stats = calibration_result.metadata.get("stats", {})
    if stats:
        print(f"  DMax mean: {stats.get('mean', 0):.1f} mm")
        print(f"  DMax std: {stats.get('std', 0):.1f} mm")
        print(f"  Valid pixel ratio: {stats.get('valid_pixel_ratio', 0):.2%}")

    cv2.waitKey()

    # main()
