#!/usr/bin/env python3
"""
Puente cámara: YOLO (models/plswork2.pt) → POST a Elysia → WebSocket al frontend.

Usa config/dataflow_augmented.yaml para mapear class_id → nombre de clase;
solo las clases one..nine envían `number` semántico (1..9), no el índice de clase.

Ejecutar desde la carpeta `code/` con el API en marcha:
  cd code && uv sync && uv run python vision_bridge.py

Variables de entorno:
  VISION_INGEST_URL  (default http://127.0.0.1:3000/api/v1/vision/ingest)
  VISION_DATA_YAML   (default config/dataflow_augmented.yaml)
  VISION_USE_WEBCAM  (default 0) — si 1, usa cv2.VideoCapture en lugar del Kinect
  VISION_CAMERA_ID   (default 0, solo con VISION_USE_WEBCAM=1)
  VISION_OPENNI_REDIST — ruta Redist OpenNI2 (solo Kinect)
  VISION_CONF        (conf. mínima, default 0.45)
  VISION_SEND_MS     (mínimo entre envíos, default 250)

El cuerpo del POST incluye `position: { x, y }` con el centro del bbox
normalizado 0..1 al frame (Kinect o webcam), para colocar el nodo en React Flow.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

from kinect_color import KinectColorStream
from vision_class_map import load_class_names, resolve_detection

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "plswork2.pt"
INGEST_URL = os.environ.get(
    "VISION_INGEST_URL", "http://127.0.0.1:3000/api/v1/vision/ingest"
)
USE_WEBCAM = os.environ.get("VISION_USE_WEBCAM", "0").lower() in (
    "1",
    "true",
    "yes",
)
CAMERA_ID = int(os.environ.get("VISION_CAMERA_ID", "0"))
CONF_MIN = float(os.environ.get("VISION_CONF", "0.45"))
SEND_INTERVAL_MS = int(os.environ.get("VISION_SEND_MS", "250"))


def _best_detection(results, frame_width: int, frame_height: int):
    """Mejor caja por confianza: (class_id, confidence, cx_norm, cy_norm, xyxy) o None.

    cx_norm, cy_norm son el centro del bbox en 0..1 respecto al frame (Kinect/cámara).
    xyxy es (x1, y1, x2, y2) en píxeles para dibujar texto en OpenCV.
    """
    best_cls, best_conf = None, 0.0
    cx_n, cy_n = 0.5, 0.5
    best_xyxy = (0.0, 0.0, float(frame_width), float(frame_height))
    fw = max(int(frame_width), 1)
    fh = max(int(frame_height), 1)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf > best_conf:
                best_conf = conf
                best_cls = cls_id
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                best_xyxy = (x1, y1, x2, y2)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                cx_n = max(0.0, min(1.0, cx / fw))
                cy_n = max(0.0, min(1.0, cy / fh))
    if best_cls is None:
        return None
    return best_cls, best_conf, cx_n, cy_n, best_xyxy


def main() -> int:
    if not MODEL_PATH.is_file():
        print(f"No se encuentra el modelo: {MODEL_PATH}", file=sys.stderr)
        return 1

    try:
        class_names = load_class_names()
    except (OSError, ValueError) as e:
        print(f"Dataset YAML: {e}", file=sys.stderr)
        return 1

    print(f"Modelo: {MODEL_PATH}")
    print(f"Clases ({len(class_names)}): {class_names[:5]}…")
    print(f"Ingest: {INGEST_URL}")
    if USE_WEBCAM:
        print(f"Cámara web (OpenCV) id={CAMERA_ID}")
    else:
        print("Entrada: Kinect (OpenNI2)")
    model = YOLO(str(MODEL_PATH))

    cap = None
    kinect: KinectColorStream | None = None
    if USE_WEBCAM:
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("No se pudo abrir la cámara web.", file=sys.stderr)
            return 1
    else:
        kinect = KinectColorStream()
        if not kinect.open():
            print(
                "No se pudo abrir el Kinect (OpenNI2). "
                "Comprueba el sensor y VISION_OPENNI_REDIST, "
                "o usa VISION_USE_WEBCAM=1.",
                file=sys.stderr,
            )
            return 1

    last_sent = 0.0
    last_key: tuple[int, str] | None = None  # (classId, label) para deduplicar envíos

    print("Detección en vivo. Pulsa 'q' para salir.")
    try:
        while True:
            if USE_WEBCAM:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
            else:
                assert kinect is not None
                frame = kinect.read_bgr()
                if frame is None:
                    break

            results = model(frame, verbose=False)
            h, w = frame.shape[:2]
            best = _best_detection(results, w, h)
            annotated = results[0].plot()
            now = time.monotonic()
            if best is not None:
                cls_id, conf, pos_x, pos_y, xyxy = best
                x1, y1, x2, y2 = (int(round(xyxy[0])), int(round(xyxy[1])), int(round(xyxy[2])), int(round(xyxy[3])))
                label, semantic_number = resolve_detection(cls_id, class_names)
                conf_pct = conf * 100.0
                line_overlay = (
                    f"{label} {conf_pct:.0f}%  |  pos ({pos_x:.2f}, {pos_y:.2f})"
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.55
                thick = 1
                (tw, th), _ = cv2.getTextSize(line_overlay, font, scale, thick)
                tx = max(0, min(x1, w - tw - 4))
                ty = min(h - 8, y2 + th + 10)
                cv2.putText(
                    annotated,
                    line_overlay,
                    (tx, ty),
                    font,
                    scale,
                    (40, 220, 40),
                    thick,
                    cv2.LINE_AA,
                )
                cv2.imshow("YOLO → Elysia (vision_bridge)", annotated)
                if conf >= CONF_MIN:
                    key = (cls_id, label)
                    should_send = (
                        (now - last_sent) * 1000 >= SEND_INTERVAL_MS
                        or key != last_key
                    )
                    if should_send:
                        payload: dict = {
                            "classId": cls_id,
                            "label": label,
                            "confidence": conf,
                            "position": {"x": pos_x, "y": pos_y},
                        }
                        if semantic_number is not None:
                            payload["number"] = semantic_number
                        try:
                            r = requests.post(INGEST_URL, json=payload, timeout=2.0)
                            r.raise_for_status()
                            last_sent = now
                            last_key = key
                            num_str = (
                                f"number={semantic_number}"
                                if semantic_number is not None
                                else "number=(no dígito)"
                            )
                            print(
                                f"→ API: classId={cls_id} label={label!r} {num_str} "
                                f"conf={conf:.2f} pos=({pos_x:.3f}, {pos_y:.3f})"
                            )
                        except requests.RequestException as e:
                            print(f"Error POST ingest: {e}", file=sys.stderr)
            else:
                cv2.imshow("YOLO → Elysia (vision_bridge)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        if kinect is not None:
            kinect.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
