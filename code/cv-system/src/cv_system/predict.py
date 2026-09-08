"""
Predicciones en vivo con opción de guardar para entrenamiento YOLO.

Usa PyKinect2 por defecto (HARDWARE_MANAGER=pykinect2). OpenNI2: HARDWARE_MANAGER=openni2.

Uso:
    predict                          # Solo muestra predicciones
    predict --save                   # Guarda con ESPACIO (manual) o --interval (auto)
    predict --save --interval 0      # Solo captura manual con ESPACIO
    predict --save --count 50        # Guarda 50 y sale
    predict --save --readable        # También genera labels con nombres

Controles:
    ESPACIO  Capturar imagen y labels (con --save)
    q        Salir

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
import sys
import time
import traceback
import argparse
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from cv_system.config import load_config
from cv_system.hardware import HardwareError, HardwareManager, PyKinect2HardwareManager
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


def create_hardware_manager(hardware: str) -> HardwareManager | PyKinect2HardwareManager:
    """Crea el gestor de hardware (pykinect2 por defecto)."""
    manager_type = hardware.lower()
    if manager_type == "pykinect2":
        return PyKinect2HardwareManager()
    if manager_type == "openni2":
        return HardwareManager()
    raise ValueError(
        f"Hardware desconocido: {manager_type!r}. "
        "Valores válidos: 'pykinect2', 'openni2'"
    )


def acquire_bird_view(hw, transformer, camera_config) -> np.ndarray:
    """Captura RGB y aplica homografía en CPU (evita crashes de OpenCL/UMat)."""
    rgb_frame = hw.get_rgb_frame()
    rgb_np = rgb_frame.get() if isinstance(rgb_frame, cv2.UMat) else np.asarray(rgb_frame)
    if rgb_np is None or rgb_np.size == 0:
        raise RuntimeError("Frame RGB vacío del Kinect")

    out_size = (camera_config.rgb_resolution[1], camera_config.rgb_resolution[0])
    bird_np = cv2.warpPerspective(rgb_np, transformer.H, out_size)
    if bird_np is None or bird_np.size == 0:
        raise RuntimeError("La homografía produjo una imagen vacía")
    return bird_np


def pause_before_exit() -> None:
    """Mantiene la consola abierta en Windows para leer el error."""
    if sys.stdin.isatty():
        try:
            input("\nPresiona Enter para cerrar...")
        except EOFError:
            pass


def save_prediction_sample(
    output_dir: Path,
    index: int,
    bird_np,
    full_annotated,
    detections: list[CardDetection],
    readable: bool,
) -> str:
    """Guarda imagen, labels YOLO y debug. Devuelve el nombre base del archivo."""
    filename = f"predict_{index:06d}"
    img_h, img_w = bird_np.shape[:2]

    cv2.imwrite(str(output_dir / "images" / f"{filename}.jpg"), bird_np)

    labels_path = output_dir / "labels" / f"{filename}.txt"
    with open(labels_path, "w") as f:
        for d in detections:
            f.write(detection_to_yolo_line(d, img_w, img_h) + "\n")

    if readable:
        readable_path = output_dir / "labels_readable" / f"{filename}.txt"
        with open(readable_path, "w") as f:
            for d in detections:
                f.write(detection_to_readable_line(d, img_w, img_h) + "\n")

    cv2.imwrite(str(output_dir / "debug" / f"{filename}.jpg"), full_annotated)
    return filename


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
    parser.add_argument(
        "--hardware",
        type=str,
        default=os.getenv("HARDWARE_MANAGER", "pykinect2"),
        choices=("pykinect2", "openni2"),
        help="Backend de cámara (default: pykinect2)",
    )
    args = parser.parse_args()

    # OpenCL + UMat suele provocar access violation en Windows (AMD/Intel)
    cv2.ocl.setUseOpenCL(False)

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

    hw = None
    saved = 0

    try:
        hw = create_hardware_manager(args.hardware)
    except ValueError as e:
        print(f"ERROR: {e}", flush=True)
        pause_before_exit()
        sys.exit(1)

    print(f"Hardware: {type(hw).__name__}", flush=True)
    try:
        hw.initialize(config.camera)
    except HardwareError as e:
        print(f"ERROR al inicializar Kinect: {e}", flush=True)
        if args.hardware == "openni2":
            print(
                "Sugerencia: usa PyKinect2 →  uv run predict --hardware pykinect2",
                flush=True,
            )
        pause_before_exit()
        sys.exit(1)

    if not Path(model_path).is_file():
        print(f"ERROR: modelo no encontrado: {Path(model_path).resolve()}", flush=True)
        hw.shutdown()
        pause_before_exit()
        sys.exit(1)

    transformer = RgbImageTransformer(calibration, config.camera)
    if args.rfdetr:
        DetectorClass = RFDETRCardDetector
    elif args.e2e:
        DetectorClass = E2ECardDetector
    else:
        DetectorClass = CardDetector

    print(f"Cargando modelo: {model_path}", flush=True)
    try:
        detector = DetectorClass(
            rgb_image_transformer=transformer,
            model_path=model_path,
            conf_threshold=args.conf,
        )
    except Exception as e:
        print(f"ERROR al cargar el detector: {e}", flush=True)
        traceback.print_exc()
        hw.shutdown()
        pause_before_exit()
        sys.exit(1)

    try:
        # Encontrar siguiente índice disponible para no sobreescribir
        start_index = get_next_index(output_dir) if output_dir else 0
        current_index = start_index
        saved = 0
        last_save = 0.0

        print("Ejecutando predicciones en vivo...", flush=True)
        print("Presiona 'q' para salir", flush=True)
        if args.save:
            print(f"Guardando en formato YOLO: {output_dir}", flush=True)
            if args.interval > 0:
                print(f"Captura automática cada {args.interval}s", flush=True)
            print("Presiona ESPACIO para capturar manualmente", flush=True)
            if start_index > 0:
                print(
                    f"Continuando desde índice {start_index} (archivos existentes detectados)",
                    flush=True,
                )

        cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)

        while True:
            bird_np = acquire_bird_view(hw, transformer, config.camera)

            # Detectar (UMat solo para la API del detector; datos en CPU)
            full_annotated, detections = detector.detect(cv2.UMat(bird_np))

            # Mostrar info de detecciones
            if detections:
                labels = [d.label for d in detections]
                print(f"Detectado: {', '.join(labels)}")

            cv2.imshow("Predictions", full_annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            manual_capture = key == ord(" ")
            auto_capture = (
                args.save
                and args.interval > 0
                and (time.time() - last_save >= args.interval)
            )

            if args.save and (manual_capture or auto_capture):
                filename = save_prediction_sample(
                    output_dir,
                    current_index,
                    bird_np,
                    full_annotated,
                    detections,
                    args.readable,
                )
                current_index += 1
                saved += 1
                last_save = time.time()
                trigger = "ESPACIO" if manual_capture else "intervalo"
                print(f"[{saved}] Guardado ({trigger}): {filename}")

                if args.count > 0 and saved >= args.count:
                    break

    except KeyboardInterrupt:
        print(f"\nTerminado. Imágenes guardadas: {saved}", flush=True)
    except Exception as e:
        print(f"\nERROR durante la ejecución: {e}", flush=True)
        traceback.print_exc()
        pause_before_exit()
        sys.exit(1)
    finally:
        if hw is not None:
            try:
                hw.shutdown()
            except Exception as e:
                print(f"Advertencia al cerrar Kinect: {e}", flush=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
