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
  VISION_CAMERA_ID   (default 0)
  VISION_CONF        (conf. mínima, default 0.45)
  VISION_SEND_MS     (mínimo entre envíos, default 250)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

from vision_class_map import load_class_names, resolve_detection

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "plswork2.pt"
INGEST_URL = os.environ.get(
    "VISION_INGEST_URL", "http://127.0.0.1:3000/api/v1/vision/ingest"
)
CAMERA_ID = int(os.environ.get("VISION_CAMERA_ID", "0"))
CONF_MIN = float(os.environ.get("VISION_CONF", "0.45"))
SEND_INTERVAL_MS = int(os.environ.get("VISION_SEND_MS", "250"))


def _best_detection(results):
    """Devuelve (class_id, confidence) de la caja con mayor confianza, o None."""
    best_cls, best_conf = None, 0.0
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf > best_conf:
                best_conf = conf
                best_cls = cls_id
    if best_cls is None:
        return None
    return best_cls, best_conf


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
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.", file=sys.stderr)
        return 1

    last_sent = 0.0
    last_key: tuple[int, str] | None = None  # (classId, label) para deduplicar envíos

    print("Detección en vivo. Pulsa 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)
        best = _best_detection(results)
        annotated = results[0].plot()
        cv2.imshow("YOLO → Elysia (vision_bridge)", annotated)

        now = time.monotonic()
        if best is not None:
            cls_id, conf = best
            label, semantic_number = resolve_detection(cls_id, class_names)
            if conf >= CONF_MIN:
                key = (cls_id, label)
                should_send = (
                    (now - last_sent) * 1000 >= SEND_INTERVAL_MS or key != last_key
                )
                if should_send:
                    payload: dict = {
                        "classId": cls_id,
                        "label": label,
                        "confidence": conf,
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
                            f"→ API: classId={cls_id} label={label!r} {num_str} conf={conf:.2f}"
                        )
                    except requests.RequestException as e:
                        print(f"Error POST ingest: {e}", file=sys.stderr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
