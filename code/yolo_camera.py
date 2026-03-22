#!/usr/bin/env python3
"""
Script para cargar YOLO 11n, capturar desde el Kinect (OpenNI2) y detectar en tiempo real.
Dibuja bounding boxes y el label (clase + confianza) de cada detección.

Ejecutar desde la carpeta code/ con: uv run python yolo_camera.py

Opcional: YOLO_USE_WEBCAM=1 para usar cv2.VideoCapture en lugar del Kinect (pruebas).
"""

import os

import cv2
from ultralytics import YOLO

from kinect_color import KinectColorStream


def main():
    use_webcam = os.environ.get("YOLO_USE_WEBCAM", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    # Cargar el modelo (yolo11n.pt o models/prueba.pt en el directorio actual)
    model = YOLO("models/plswork2.pt")

    if use_webcam:
        cap = cv2.VideoCapture(int(os.environ.get("YOLO_CAMERA_ID", "0")))
        if not cap.isOpened():
            print("No se pudo abrir la cámara web.")
            return
    else:
        cap = None
        kinect = KinectColorStream()
        if not kinect.open():
            print(
                "No se pudo abrir el Kinect. Instala OpenNI2, conecta el sensor "
                "o prueba YOLO_USE_WEBCAM=1."
            )
            return

    src = "cámara web" if use_webcam else "Kinect"
    print(f"Detección en vivo ({src}). Pulsa 'q' para salir.")
    try:
        while True:
            if use_webcam:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            else:
                frame = kinect.read_bgr()
                if frame is None:
                    break

            results = model(frame, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("YOLO - Detección", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if use_webcam and cap is not None:
            cap.release()
        elif not use_webcam:
            kinect.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
