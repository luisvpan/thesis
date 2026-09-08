"""
Captura imágenes con homografía para entrenar modelos.

Uso:
    capture                    # Modo continuo, Ctrl+C para salir
    capture --count 10         # Captura 10 imágenes y sale
    capture --interval 2.0     # 2 segundos entre capturas
"""
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
from dotenv import load_dotenv

from cv_system.config import load_config
from cv_system.hardware.manager import HardwareManager
from cv_system.calibration.result import CalibrationResult
from cv_system.transform.rgb_image_transformer import RgbImageTransformer

CAPTURE_OUTPUT_DIR = os.getenv("CAPTURE_OUTPUT_DIR", "images/raw")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Captura imágenes con homografía")
    parser.add_argument(
        "--count", type=int, default=0, help="Número de imágenes (0=infinito)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Segundos entre capturas"
    )
    parser.add_argument(
        "--output", type=str, default=CAPTURE_OUTPUT_DIR, help="Carpeta de salida"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Mostrar preview en ventana"
    )
    args = parser.parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar configuración y calibración
    config_path = Path(os.getenv("CONFIG_PATH", "config/session.json"))
    calibration_path = Path(os.getenv("CALIBRATION_PATH", "config/calibration.json"))

    config = load_config(config_path)
    calibration = CalibrationResult.load(calibration_path)

    # Inicializar hardware y transformer
    hw = HardwareManager()
    hw.initialize(config.camera)
    transformer = RgbImageTransformer(calibration, config.camera)

    try:
        captured = 0
        print(f"Capturando imágenes en: {output_dir}")
        print(
            "Presiona Ctrl+C para salir"
            if args.count == 0
            else f"Capturando {args.count} imágenes..."
        )

        while args.count == 0 or captured < args.count:
            # Capturar y transformar
            rgb_frame = hw.get_rgb_frame()
            bird_view = transformer.camera_to_projector(rgb_frame)

            # Convertir UMat a numpy para guardar
            bird_np = bird_view.get() if isinstance(bird_view, cv2.UMat) else bird_view

            # Generar nombre con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_dir / f"capture_{timestamp}.jpg"

            cv2.imwrite(str(filename), bird_np)
            captured += 1
            print(f"[{captured}] Guardado: {filename.name}")

            if args.preview:
                cv2.imshow("Capture Preview", bird_np)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nCaptura terminada. Total: {captured} imágenes")
    finally:
        hw.shutdown()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
