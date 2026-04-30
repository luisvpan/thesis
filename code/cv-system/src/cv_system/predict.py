"""
Predicciones en vivo con opción de guardar para entrenamiento.

Uso:
    predict                     # Solo muestra predicciones
    predict --save              # Guarda imágenes anotadas
    predict --save --count 50   # Guarda 50 imágenes y sale

Estructura de salida (con --save):
    images/annotated/
        predict_20240101_120000_123456/
            raw.jpg                      # Imagen sin anotaciones
            full_annotated.jpg           # Con labels y confianza
            bounding_box_only.jpg        # Solo bounding boxes
"""
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

from cv_system.config import load_config
from cv_system.hardware.manager import HardwareManager
from cv_system.calibration.result import CalibrationResult
from cv_system.transform.rgb_image_transformer import RgbImageTransformer
from cv_system.detection.card_detector import CardDetector, CardDetection

PREDICT_OUTPUT_DIR = os.getenv("PREDICT_OUTPUT_DIR", "images/annotated")


def draw_bounding_boxes_only(
    image: np.ndarray, detections: list[CardDetection]
) -> np.ndarray:
    """Draw only bounding boxes without labels or confidence."""
    result = image.copy()
    for d in detections:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(result, p1, p2, (0, 220, 0), 2)
    return result


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Predicciones en vivo")
    parser.add_argument(
        "--save", action="store_true", help="Guardar imágenes anotadas"
    )
    parser.add_argument(
        "--count", type=int, default=0, help="Número de imágenes (0=infinito)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5, help="Segundos entre capturas (si --save)"
    )
    parser.add_argument(
        "--output", type=str, default=PREDICT_OUTPUT_DIR, help="Carpeta de salida"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Ruta al modelo YOLO"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold de confianza"
    )
    args = parser.parse_args()

    # Crear directorio si guardamos
    output_dir = Path(args.output) if args.save else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar configuración y calibración
    config_path = Path(os.getenv("CONFIG_PATH", "config/session.json"))
    calibration_path = Path(os.getenv("CALIBRATION_PATH", "config/calibration.json"))
    model_path = args.model or os.getenv("YOLO_MODEL_PATH", "models/best.pt")

    config = load_config(config_path)
    calibration = CalibrationResult.load(calibration_path)

    # Inicializar componentes
    hw = HardwareManager()
    hw.initialize(config.camera)
    transformer = RgbImageTransformer(calibration, config.camera)
    detector = CardDetector(
        rgb_image_transformer=transformer,
        model_path=model_path,
        conf_threshold=args.conf,
    )

    try:
        saved = 0
        last_save = 0.0
        print("Ejecutando predicciones en vivo...")
        print("Presiona 'q' para salir")
        if args.save:
            print(f"Guardando anotaciones en: {output_dir}")

        while True:
            # Capturar y transformar
            rgb_frame = hw.get_rgb_frame()
            bird_view = transformer.camera_to_projector(rgb_frame)

            # Convertir a numpy
            bird_np = bird_view.get() if isinstance(bird_view, cv2.UMat) else bird_view

            # Detectar
            full_annotated, detections = detector.detect(bird_view)

            # Mostrar info de detecciones
            if detections:
                labels = [d.label for d in detections]
                print(f"Detectado: {', '.join(labels)}")

            # Guardar si corresponde
            if args.save and (time.time() - last_save >= args.interval):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # Crear carpeta para esta captura
                capture_dir = output_dir / f"predict_{timestamp}"
                capture_dir.mkdir(parents=True, exist_ok=True)

                # Guardar imagen raw (sin anotaciones)
                cv2.imwrite(str(capture_dir / "raw.jpg"), bird_np)

                # Guardar full_annotated (con labels y confianza)
                cv2.imwrite(str(capture_dir / "full_annotated.jpg"), full_annotated)

                # Guardar bounding_box_only (solo cajas)
                bbox_only = draw_bounding_boxes_only(bird_np, detections)
                cv2.imwrite(str(capture_dir / "bounding_box_only.jpg"), bbox_only)

                saved += 1
                last_save = time.time()
                print(f"[{saved}] Guardado: {capture_dir.name}/")

                if args.count > 0 and saved >= args.count:
                    break

            # Preview
            cv2.imshow("Predictions", full_annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print(f"\nTerminado. Imágenes guardadas: {saved}")
    finally:
        hw.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
