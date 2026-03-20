#!/usr/bin/env python3
"""
Script para cargar YOLO 11n, abrir la cámara y detectar objetos en tiempo real.
Dibuja bounding boxes y el label (clase + confianza) de cada detección.
Ejecutar desde la carpeta code/ con: uv run python yolo_camera.py
"""

import cv2
from ultralytics import YOLO


def main():
    # Cargar el modelo (yolo11n.pt o models/prueba.pt en el directorio actual)
    model = YOLO("models/plswork2.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Detección en vivo. Pulsa 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia: el modelo devuelve una lista de Results
        results = model(frame, verbose=False)

        # result.plot() devuelve la imagen con bounding boxes y labels dibujados
        annotated = results[0].plot()

        cv2.imshow("YOLO - Detección", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
