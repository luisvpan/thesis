"""
Predicciones en vivo con opción de guardar para entrenamiento YOLO.

Uso:
    predict                          # Solo muestra predicciones
    predict --save                   # Guarda imágenes y labels
    predict --save --count 50        # Guarda 50 y sale
    predict --save --readable        # También genera labels con nombres

Estructura de salida (con --save):
    output_dir/
        images/
            predict_000000.jpg
        labels/
            predict_000000.txt       # class_id cx cy w h (YOLO format)
        labels_readable/             # solo con --readable
            predict_000000.txt       # class_name cx cy w h (legible)
        debug/
            predict_000000.jpg       # full_annotated con labels y confianza
"""
import os
import time
import argparse
from pathlib import Path

import cv2
from dotenv import load_dotenv

from cv_system.config import load_config
from cv_system.hardware.manager import HardwareManager
from cv_system.calibration.result import CalibrationResult
from cv_system.transform.rgb_image_transformer import RgbImageTransformer
from cv_system.detection import CardDetector, CardDetection, E2ECardDetector, RFDETRCardDetector

PREDICT_OUTPUT_DIR = os.getenv("PREDICT_OUTPUT_DIR", "images/annotated")


def get_next_index(output_dir: Path) -> int:
    """Encuentra el siguiente índice disponible para no sobreescribir archivos."""
    images_dir = output_dir / "images"
    if not images_dir.exists():
        return 0

    existing = list(images_dir.glob("predict_*.jpg"))
    if not existing:
        return 0

    # Extraer índices de archivos existentes
    indices = []
    for f in existing:
        try:
            # predict_000042.jpg -> 42
            idx = int(f.stem.split("_")[1])
            indices.append(idx)
        except (IndexError, ValueError):
            continue

    return max(indices) + 1 if indices else 0


def detection_to_yolo_line(d: CardDetection, img_w: int, img_h: int) -> str:
    """Convierte una detección al formato YOLO: class_id cx cy w h (normalizado)."""
    cx = ((d.x1 + d.x2) / 2) / img_w
    cy = ((d.y1 + d.y2) / 2) / img_h
    w = (d.x2 - d.x1) / img_w
    h = (d.y2 - d.y1) / img_h
    return f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def detection_to_readable_line(d: CardDetection, img_w: int, img_h: int) -> str:
    """Convierte una detección a formato legible: nombre cx cy w h (normalizado)."""
    cx = ((d.x1 + d.x2) / 2) / img_w
    cy = ((d.y1 + d.y2) / 2) / img_h
    w = (d.x2 - d.x1) / img_w
    h = (d.y2 - d.y1) / img_h
    return f"{d.label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Predicciones en vivo")
    parser.add_argument(
        "--save", action="store_true", help="Guardar imágenes y labels YOLO"
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
    parser.add_argument(
        "--readable", action="store_true", help="Generar labels legibles (con nombres de clase)"
    )
    parser.add_argument(
        "--e2e", action="store_true", help="Usar E2ECardDetector para modelos con NMS integrado"
    )
    parser.add_argument(
        "--rfdetr", action="store_true", help="Usar RFDETRCardDetector para modelos RF-DETR"
    )
    args = parser.parse_args()

    # Crear subdirectorios si guardamos
    output_dir = Path(args.output) if args.save else None
    if output_dir:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / "debug").mkdir(parents=True, exist_ok=True)
        if args.readable:
            (output_dir / "labels_readable").mkdir(parents=True, exist_ok=True)

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
    if args.rfdetr:
        DetectorClass = RFDETRCardDetector
    elif args.e2e:
        DetectorClass = E2ECardDetector
    else:
        DetectorClass = CardDetector
    detector = DetectorClass(
        rgb_image_transformer=transformer,
        model_path=model_path,
        conf_threshold=args.conf,
    )

    try:
        # Encontrar siguiente índice disponible para no sobreescribir
        start_index = get_next_index(output_dir) if output_dir else 0
        current_index = start_index
        saved = 0
        last_save = 0.0

        print("Ejecutando predicciones en vivo...")
        print("Presiona 'q' para salir")
        if args.save:
            print(f"Guardando en formato YOLO: {output_dir}")
            if start_index > 0:
                print(f"Continuando desde índice {start_index} (archivos existentes detectados)")

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
                filename = f"predict_{current_index:06d}"
                img_h, img_w = bird_np.shape[:2]

                # Guardar imagen raw en images/
                cv2.imwrite(str(output_dir / "images" / f"{filename}.jpg"), bird_np)

                # Guardar labels en formato YOLO
                labels_path = output_dir / "labels" / f"{filename}.txt"
                with open(labels_path, "w") as f:
                    for d in detections:
                        f.write(detection_to_yolo_line(d, img_w, img_h) + "\n")

                # Guardar labels legibles si se pidió
                if args.readable:
                    readable_path = output_dir / "labels_readable" / f"{filename}.txt"
                    with open(readable_path, "w") as f:
                        for d in detections:
                            f.write(detection_to_readable_line(d, img_w, img_h) + "\n")

                # Guardar debug (full_annotated)
                cv2.imwrite(str(output_dir / "debug" / f"{filename}.jpg"), full_annotated)

                current_index += 1
                saved += 1
                last_save = time.time()
                print(f"[{saved}] Guardado: {filename}")

                if args.count > 0 and saved >= args.count:
                    break

            # Preview
            cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)
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
