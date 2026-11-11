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

from cv_system.bridge import TouchEvent, WebSocketBridge
from cv_system.bridge.vision_ingest import post_card_batch_async
from cv_system.calibration.calibrator import Calibrator
from cv_system.calibration.result import CalibrationResult
from cv_system.config import load_config
from cv_system.detection import CARD_DETECTORS, TouchMethod, detect_card_method
from cv_system.detection.depth_only_touch_detector import DepthOnlyTouchDetector
from cv_system.detection.direct_touch_detector import DIRECTTouchDetector
from cv_system.detection.espol_touch_detector import ESPOLTouchDetector
from cv_system.detection.farout_touch_detector import FarOutTouchDetector
from cv_system.detection.mediapipe_direct_hybrid_touch_detector import (
    MediapipeDIRECTHybridTouchDetector,
)
from cv_system.detection.mediapipe_direct_hybrid_touch_detector_dml import (
    MediapipeDIRECTHybridTouchDetectorDml,
)
from cv_system.detection.profiling import StageTimer
from cv_system.detection.touch_detector import TouchDetector
from cv_system.detection.touch_detector_dml import DmlTouchDetector
from cv_system.flags import debug_enabled
from cv_system.hardware import HardwareError, HardwareManager, PyKinect2HardwareManager
from cv_system.recording import (
    RECORD_OUTPUT_DIR,
    SessionRecorder,
    detector_manifest,
    finish_manifest,
    new_session_id,
    write_session_manifest,
)
from cv_system.transform import (
    DepthCoordinateTransformer,
    PyKinect2ResolutionMapper,
    ResolutionMapper,
    RgbImageTransformer,
)

# Configure logging level from env var (DEBUG, INFO, WARNING, ERROR)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def touch_detector_enabled() -> bool:
    """Touch detection. Off by default; set CV_TOUCH_DETECTOR=1 to enable."""
    v = os.getenv("CV_TOUCH_DETECTOR", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_touch_detector_type() -> TouchMethod:
    """
    Get touch detector type from environment.

    CV_TOUCH_DETECTOR_TYPE:
        - "mediapipe" (default): MediaPipe hand landmarks + convex hull
        - "espol": ESPOL paper algorithm (K-curvature + hysteresis)
        - "depth_only": Simple depth blob detection
        - "direct": DIRECT paper (depth + IR fusion)
        - "farout": FarOut Touch paper (depth only, long range)
        - "hybrid": MediaPipe + DIRECT hybrid (best of both)
        - "mediapipe_dml": MediaPipe + depth masks on torch-directml
        - "hybrid_dml": DIRECT + MediaPipe hybrid with DirectML PyTorch device
    """
    v = os.getenv("CV_TOUCH_DETECTOR_TYPE", "mediapipe").strip().lower()
    if v in ("espol", "espol_touch", "kcurvature"):
        return "espol"
    elif v in ("depth_only", "depth", "simple"):
        return "depth_only"
    elif v in ("direct", "direct_touch"):
        return "direct"
    elif v in ("farout", "farout_touch"):
        return "farout"
    elif v in ("hybrid", "mediapipe_direct", "mp_direct"):
        return "hybrid"
    elif v in ("mediapipe_dml", "mp_dml", "dml_mediapipe"):
        return "mediapipe_dml"
    elif v in ("hybrid_dml", "mediapipe_direct_dml", "mp_direct_dml"):
        return "hybrid_dml"
    else:
        return "mediapipe"


def debug_windows_enabled() -> bool:
    """Show debug windows (cv2.imshow). Off by default; CV_DEBUG=1 (cv_system.flags)."""
    return debug_enabled()


def recording_enabled() -> bool:
    """CV_RECORD=1: record the session (bird view mp4 + per-frame detections + manifest)
    into images/annotated, as `predict --record-video` does. This is the recording to
    use for evaluation: it is the real system (GPU warp, touch running, real fps)."""
    return os.getenv("CV_RECORD", "0").strip().lower() in ("1", "true", "yes", "on")


def run_websocket_client(bridge: WebSocketBridge) -> None:
    """
    Run WebSocket client in async event loop.

    Args:
        bridge: WebSocketBridge instance to connect and manage.
    """
    bridge._loop_thread_ident = threading.get_ident()
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
    print(f"FPS: {config.camera.fps}")
    print(f"Dsurface frames: {config.calibration.dsurface_num_frames}")
    print(f"OpenCL available: {cv2.ocl.haveOpenCL()}, enabled: {cv2.ocl.useOpenCL()}")
    enable_touch = touch_detector_enabled()
    print(
        f"Touch detector: {'ON' if enable_touch else 'OFF'} "
        "(set CV_TOUCH_DETECTOR=1 to enable MediaPipe touch)"
    )
    print("=" * 60)

    hardware = None
    ws_bridge = None
    ws_thread = None
    touch_executor = None
    card_executor = None
    frame_executor = None
    recorder: SessionRecorder | None = None
    record_manifest_path: Path | None = None
    RECORD_VIDEO_FPS = 30.0  # nominal; the real rate goes to the manifest
    COLOR_STALL_S = 1.0  # warn when the camera sends no new color frame for this long
    repeated_color_frames = 0  # loop iterations whose color frame was not new
    max_color_gap_s = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        # Step 1: Initialize hardware
        print("\n[1/5] Initializing hardware...")
        manager_type = os.getenv("HARDWARE_MANAGER", "openni2").lower()
        print(f"  HARDWARE_MANAGER env var = '{manager_type}'")
        if manager_type == "pykinect2":
            hardware = PyKinect2HardwareManager()
        elif manager_type == "openni2":
            hardware = HardwareManager()
        else:
            print(f"  ERROR: Unknown HARDWARE_MANAGER: {manager_type}")
            print("  Valid values: 'openni2' (default), 'pykinect2'")
            sys.exit(1)
        print(f"  Using: {type(hardware).__name__}")
        try:
            hardware.initialize(config.camera)
            print("  Hardware initialized successfully")
        except HardwareError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        # Step 2: Load or run calibration
        print("\n[2/5] Loading or running calibration...")
        # Select resolution mapper based on hardware manager type
        if isinstance(hardware, PyKinect2HardwareManager):
            resolution_mapper = PyKinect2ResolutionMapper(kinect=hardware.kinect)
            print("  Using PyKinect2ResolutionMapper (Kinect SDK mapping)")
        else:
            resolution_mapper = ResolutionMapper(config.camera)
            print("  Using ResolutionMapper (linear scaling)")

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
        )
        # Calculate projector ROI dimensions from projector_corners
        xs = [int(p[0]) for p in config.calibration.projector_corners]
        ys = [int(p[1]) for p in config.calibration.projector_corners]
        ROI_OFFSET_X = min(xs)
        ROI_OFFSET_Y = min(ys)
        PROJ_W = max(xs) - min(xs)
        PROJ_H = max(ys) - min(ys)
        print("  RGB image transformer initialized")
        print(f"  Effective projector ROI (w, h): ({PROJ_W}, {PROJ_H})")
        print(f"  ROI offset in full projector (x, y): ({ROI_OFFSET_X}, {ROI_OFFSET_Y})")

        depth_coordinate_transformer = DepthCoordinateTransformer(calibration_result)
        print("  Depth coordinate transformer initialized")

        print("  Resolution mapper initialized")

        detector = None
        touch_type: TouchMethod = "mediapipe"
        if enable_touch:
            touch_type = get_touch_detector_type()
            show_debug = debug_windows_enabled()

            if touch_type == "espol":
                detector = ESPOLTouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    config.detection,
                    show_debug=show_debug,
                )
                print("  ESPOLTouchDetector (K-curvature + hysteresis) initialized")
            elif touch_type == "depth_only":
                detector = DepthOnlyTouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    config.detection,
                    show_debug=show_debug,
                )
                print("  DepthOnlyTouchDetector (blob detection) initialized")
            elif touch_type == "direct":
                detector = DIRECTTouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    config.detection,
                    show_debug=show_debug,
                )
                print("  DIRECTTouchDetector (depth + IR fusion) initialized")
            elif touch_type == "farout":
                # FarOut uses "bump" mode at short range (<1.5m)
                # Set use_denting=True for long range (1.5-3.5m)
                use_denting = os.getenv("CV_FAROUT_DENTING", "0").strip().lower() in ("1", "true", "yes")
                detector = FarOutTouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    config.detection,
                    show_debug=show_debug,
                    use_denting=use_denting,
                )
                mode = "denting (long range)" if use_denting else "bump (short range)"
                print(f"  FarOutTouchDetector ({mode}) initialized")
            elif touch_type == "hybrid":
                detector = MediapipeDIRECTHybridTouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    resolution_mapper,
                    config.detection,
                    rgb_H=calibration_result.rgb_H,
                    show_debug=show_debug,
                )
                print("  MediapipeDIRECTHybridTouchDetector (MediaPipe + DIRECT) initialized")
            elif touch_type == "hybrid_dml":
                detector = MediapipeDIRECTHybridTouchDetectorDml(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    resolution_mapper,
                    config.detection,
                    rgb_H=calibration_result.rgb_H,
                    show_debug=show_debug,
                )
                print(
                    "  MediapipeDIRECTHybridTouchDetectorDml "
                    "(MediaPipe + DIRECT + DirectML) initialized"
                )
            elif touch_type == "mediapipe_dml":
                detector = DmlTouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    resolution_mapper,
                    config.detection,
                    show_debug=show_debug,
                )
                print("  DmlTouchDetector (MediaPipe CPU + depth on DirectML) initialized")
            else:  # mediapipe
                detector = TouchDetector(
                    calibration_result.dmax_map,
                    depth_coordinate_transformer,
                    resolution_mapper,
                    config.detection,
                    show_debug=show_debug,
                )
                print("  TouchDetector (MediaPipe hand segmentation) initialized")
        else:
            print("  Touch detection: skipped (disabled)")
        print(f"YOLO model path: {os.getenv('YOLO_MODEL_PATH')}")

        model_path = Path(
            os.getenv(
                "YOLO_MODEL_PATH",
                str(
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "models"
                    / "yolo_11s_cards.pt"
                ),
            )
        )
        card_method = config.detection.card_method or detect_card_method(model_path)
        DetectorClass = CARD_DETECTORS[card_method]
        card_detector = DetectorClass(rgb_image_transformer, model_path)
        print(f"  Card detector ({card_method}) initialized: {model_path}")

        if recording_enabled():
            session_id = new_session_id()
            out = Path(RECORD_OUTPUT_DIR)
            (out / "sessions").mkdir(parents=True, exist_ok=True)
            (out / "videos").mkdir(parents=True, exist_ok=True)
            record_manifest_path = out / "sessions" / f"{session_id}.json"
            write_session_manifest(
                record_manifest_path,
                {
                    "session_id": session_id,
                    "source": "cv-system",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **detector_manifest(card_detector, model_path),
                    "card_method": card_method,
                    "hardware": type(hardware).__name__,
                    "touch": touch_type if enable_touch else None,
                    "dml_device_id": os.getenv("DML_DEVICE_ID", "0"),
                    "debug": debug_windows_enabled(),
                    "record_video": True,
                },
            )
            recorder = SessionRecorder(out / "videos", session_id, RECORD_VIDEO_FPS)
            print(f"  Recording session {session_id}: {recorder.video_path}")

        vision_cards_url = os.getenv(
            "VISION_CARDS_INGEST_URL",
            "http://127.0.0.1:8765/api/v1/vision/cards",
        )
        print(f"  Vision cards ingest URL: {vision_cards_url}")
        print("  IDE relay (FastAPI): either run both in one command:")
        print("      cd code/cv-system && uv run cv-stack")
        print("  or start the relay separately:")
        print("      uv run cv-ide-relay")

        # Step 4: Initialize WebSocket bridge
        print("\n[4/5] Initializing WebSocket bridge...")
        ws_url = os.getenv("LANGUAGE_RUNTIME_WS_URL", "ws://localhost:8765/live")
        ws_bridge = WebSocketBridge(url=ws_url)
        print("  WebSocket bridge initialized")
        print(f"      Touch WebSocket: {ws_url}")

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
        touch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="Touch")
            if enable_touch
            else None
        )
        card_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Card")
        frame_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Frame")

        start_time = time.time()

        # Debug windows (controlled by CV_DEBUG env var)
        show_debug_windows = debug_windows_enabled()
        # Check if we have IR frame support (only PyKinect2HardwareManager)
        has_ir_support = isinstance(hardware, PyKinect2HardwareManager)
        if show_debug_windows:
            cv2.namedWindow("Card Detection", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Card Detection", -1920, 0)
            cv2.setWindowProperty("Card Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.namedWindow("Touch Debug", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Touch Debug", -1920, 0)
            cv2.setWindowProperty("Touch Debug", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if has_ir_support:
                logger.info("IR frame support enabled (PyKinect2)")

            logger.info("Debug windows enabled (CV_DEBUG=1)")
        else:
            logger.info("Debug windows disabled (set CV_DEBUG=1 to enable)")

        # Pre-allocate buffers for debug views (reused each frame)
        touch_debug_buffer = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
        full_card_view_buffer = np.zeros((FULL_PROJ_H, FULL_PROJ_W, 3), dtype=np.uint8)

        # Cache for card detection (Fix 3: run every N frames)
        last_card_view = None
        last_card_dets: list = []

        # Throttling for touch_move events (20Hz = 50ms minimum between moves)
        TOUCH_MOVE_MIN_INTERVAL = 0.05
        last_move_times: dict[int, float] = {}

        # Helper for frame acquisition (Fix 2: pipeline)
        def acquire_frames():
            """Depth, color and the color frame's index (+1 per new camera frame;
            see hardware/frame_clock.py). Read in the same thread as the frame, so
            the index belongs to it."""
            depth, rgb = hardware.get_depth_frame(), hardware.get_rgb_frame()
            return depth, rgb, getattr(hardware, "rgb_frame_index", None)

        def touch_job(depth, rgb, ir):
            """Touch thread. The color-to-depth mapping is only used by touch, so it
            runs here, in parallel with card detection, not before both."""
            if isinstance(resolution_mapper, PyKinect2ResolutionMapper):
                resolution_mapper.update_mapping(depth)
            return detector.detect(depth, rgb, ir)

        def card_job(bird):
            """Card thread. When recording, download the bird view once and give the
            same numpy frame to the detector and to the recorder."""
            if recorder is None:
                return card_detector.detect(bird)
            frame = bird.get() if isinstance(bird, cv2.UMat) else bird
            view, dets = card_detector.detect(frame)
            recorder.write(frame, dets)
            return view, dets

        # Initial frame acquisition
        depth_frame, rgb_frame, rgb_index = acquire_frames()
        # Index of the last color frame given to card detection. The camera can be
        # slower than the loop (Kinect: 30 fps, 15 in low light) or stall; then
        # get_rgb_frame() repeats the last frame, and a repeated frame must not
        # reach the tracker (it would count as another frame of evidence) nor the
        # recording (one video frame = one camera frame).
        last_card_rgb_index: int | None = None
        color_stall_warned = False

        # CV_PROFILE: where the main thread spends each frame. Touch and card run in
        # parallel threads, so "touch_wait" is the time until touch finished
        # (color-to-depth mapping included) and "card_wait" is the extra time card
        # detection needed after that.
        loop_timer = StageTimer("main loop", scope="measured")

        while True:
            try:
                t0 = time.perf_counter()

                # Fix 2: Start acquiring next frame while processing current
                next_frame_future = frame_executor.submit(acquire_frames)

                t1 = time.perf_counter()

                # Fix 1: Do warp ONCE and share with both detectors (UMat stays on GPU).
                # With OpenCL this only queues the warp; it runs when the card thread
                # reads the UMat (counted there as "download").
                with loop_timer.stage("warp"):
                    rgb_bird = rgb_image_transformer.camera_to_projector(rgb_frame)

                t2 = time.perf_counter()

                # Get IR frame for touch correction (PyKinect2 only)
                ir_frame = None
                if has_ir_support and enable_touch:
                    with loop_timer.stage("ir"):
                        ir_frame = hardware.get_ir_frame()

                # Submit BOTH detections in parallel (don't wait for touch before card)
                touch_future = None
                if enable_touch and touch_executor is not None and detector is not None:
                    # TouchDetector uses depth frame + raw RGB + optional IR for correction
                    touch_future = touch_executor.submit(
                        touch_job, depth_frame, rgb_frame, ir_frame
                    )

                card_future = None
                t_card_submit = t2  # For timing card from submission
                
                color_is_new = rgb_index is None or rgb_index != last_card_rgb_index
                if color_is_new:
                    card_future = card_executor.submit(card_job, rgb_bird)
                    last_card_rgb_index = rgb_index
                else:
                    repeated_color_frames += 1
                t_card_submit = time.perf_counter()

                color_gap = getattr(hardware, "rgb_frame_age_s", 0.0)
                if color_gap != float("inf"):
                    max_color_gap_s = max(max_color_gap_s, color_gap)
                if color_gap > COLOR_STALL_S and not color_stall_warned:
                    logger.warning(
                        f"No new color frame from the camera for {color_gap:.1f}s: "
                        "the color stream stalled (cards are not being updated)"
                    )
                    color_stall_warned = True
                elif color_gap <= COLOR_STALL_S and color_stall_warned:
                    logger.warning("Color frames are arriving again")
                    color_stall_warned = False

                # Now wait for results (they run in parallel)
                if touch_future is not None:
                    with loop_timer.stage("touch_wait"):
                        touches, hands_detected = touch_future.result()
                    t_touch = time.perf_counter()
                else:
                    touches, hands_detected = [], False
                    t_touch = t2

                card_time_ms = 0.0
                card_wait_ms = 0.0
                if card_future is not None:
                    t_card_wait_start = time.perf_counter()
                    with loop_timer.stage("card_wait"):
                        card_view, card_dets = card_future.result()
                    t_card_end = time.perf_counter()
                    card_time_ms = 1000 * (t_card_end - t_card_submit)  # Total card time
                    card_wait_ms = 1000 * (t_card_end - t_card_wait_start)  # Wait after touch done
                    # Always update - ByteTrack handles occlusions from hands
                    last_card_view = card_view
                    last_card_dets = card_dets
                    with loop_timer.stage("post_cards"):
                        post_card_batch_async(
                            vision_cards_url,
                            card_dets,
                            FULL_PROJ_W,
                            FULL_PROJ_H,
                        )
                else:
                    card_view = last_card_view if last_card_view is not None else rgb_bird.get()
                    card_dets = last_card_dets

                if touches:
                    for touch in touches:
                        # Throttle touch_move events
                        if touch.state == "move":
                            now = time.monotonic()
                            last_time = last_move_times.get(touch.id, 0)
                            if now - last_time < TOUCH_MOVE_MIN_INTERVAL:
                                continue
                            last_move_times[touch.id] = now

                        # Clean up throttle state on touch_up
                        if touch.state == "up":
                            last_move_times.pop(touch.id, None)

                        full_x = touch.x + ROI_OFFSET_X
                        full_y = touch.y + ROI_OFFSET_Y
                        if 0 <= full_x < FULL_PROJ_W and 0 <= full_y < FULL_PROJ_H:
                            touch_event = TouchEvent.from_tracked_touch(
                                x=full_x,
                                y=full_y,
                                event_type=f"touch_{touch.state}",
                                touch_id=touch.id,
                            )
                            if ws_bridge.state.value == "CONNECTED" and ws_bridge.loop is not None:
                                logger.info(
                                    f"[WS] touch_{touch.state} id={touch.id} "
                                    f"({full_x:.0f}, {full_y:.0f})"
                                )
                                asyncio.run_coroutine_threadsafe(
                                    ws_bridge.send_touch_event(touch_event.to_dict()),
                                    loop=ws_bridge.loop,
                                )
                elif enable_touch and frame_count % 30 == 0:
                    logger.debug(f"Frame {frame_count}: No touches detected")

                if card_dets:
                    for d in card_dets:
                        logger.debug(f"Frame {frame_count}: Card {d.label} {d.confidence * 100:.1f}%")

                # Debug windows (only if enabled)
                debug_start = time.perf_counter()
                if show_debug_windows:
                    # Card detection view (every 2 frames)
                    if frame_count % 2 == 0:
                        full_card_view_buffer.fill(0)
                        y1 = max(0, ROI_OFFSET_Y)
                        x1 = max(0, ROI_OFFSET_X)
                        y2 = min(FULL_PROJ_H, y1 + card_view.shape[0])
                        x2 = min(FULL_PROJ_W, x1 + card_view.shape[1])
                        src_h = max(0, y2 - y1)
                        src_w = max(0, x2 - x1)
                        if src_h > 0 and src_w > 0:
                            full_card_view_buffer[y1:y2, x1:x2] = card_view[:src_h, :src_w]
                        cv2.imshow("Card Detection", full_card_view_buffer)

                    # Touch debug view
                    touch_debug_buffer.fill(0)
                    if touches:
                        logger.debug(
                            f"[TouchDebug] {len(touches)} events: "
                            f"{[(t.state, t.id, int(t.x), int(t.y)) for t in touches]}"
                        )
                    for touch in touches:
                        tx_int, ty_int = int(touch.x), int(touch.y)
                        tx_draw = max(30, min(tx_int, PROJ_W - 30))
                        ty_draw = max(30, min(ty_int, PROJ_H - 30))
                        # Color by state: green=down, yellow=move, red=up
                        color = {
                            "down": (0, 255, 0),
                            "move": (0, 255, 255),
                            "up": (0, 0, 255),
                        }.get(touch.state, (255, 255, 255))
                        cv2.circle(touch_debug_buffer, (tx_draw, ty_draw), 20, color, -1)
                        cv2.circle(touch_debug_buffer, (tx_draw, ty_draw), 20, (255, 255, 255), 3)
                        cv2.putText(
                            touch_debug_buffer,
                            f"id={touch.id} {touch.state}",
                            (tx_draw + 25, ty_draw + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                        )
                    cv2.imshow("Touch Debug", touch_debug_buffer)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                loop_timer.add("debug_windows", time.perf_counter() - debug_start)
                frame_count += 1
                elapsed = time.time() - start_time

                # Fix 2: Get next frame (should be ready by now)
                t5 = time.perf_counter()
                with loop_timer.stage("acquire_wait"):
                    depth_frame, rgb_frame, rgb_index = next_frame_future.result()
                t6 = time.perf_counter()
                loop_timer.frame_done()
                logger.debug(f"[FPS: {frame_count / elapsed:.1f}] submit={1000*(t1-t0):.1f}ms warp={1000*(t2-t1):.1f}ms touch={1000*(t_touch-t2):.1f}ms card={card_time_ms:.1f}ms(wait={card_wait_ms:.1f}ms) acq={1000*(t6-t5):.1f}ms")

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
                ws_thread.join(timeout=12)
            print("  WebSocket stopped")

        if recorder is not None:
            if card_executor is not None:
                card_executor.shutdown(wait=True)  # last frame written before closing
            recorder.close()
            if record_manifest_path is not None:
                finish_manifest(
                    record_manifest_path,
                    recorder,
                    RECORD_VIDEO_FPS,
                    # Loop iterations without a new camera frame (not recorded)
                    repeated_frames=repeated_color_frames,
                    max_color_gap_s=round(max_color_gap_s, 2),
                )
            print(
                f"  Recording saved: {recorder.video_path} ({recorder.frames} frames, "
                f"{recorder.effective_fps:.1f} fps, encoder waits: {recorder.waits})"
            )

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