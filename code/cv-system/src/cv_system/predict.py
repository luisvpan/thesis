"""
Predicciones en vivo con opción de guardar para entrenamiento YOLO.

Usa PyKinect2 por defecto (HARDWARE_MANAGER=pykinect2). OpenNI2: HARDWARE_MANAGER=openni2.

Uso:
    predict                          # Solo muestra predicciones
    predict --save                   # Guarda con ESPACIO (manual) o --interval (auto)
    predict --save --interval 0      # Solo captura manual con ESPACIO
    predict --save --count 50        # Guarda 50 y sale
    predict --save --readable        # También genera labels con nombres
    predict --save --min-change 0    # Captura automática sin exigir cambio de escena
    predict --record-video           # Graba el bird view crudo + detecciones por frame

Cada ejecución es una sesión (YYYYMMDD_HHMMSS). El ID de sesión va en el nombre de
cada archivo para poder separar train/val/test por sesión y no por imagen.

Las etiquetas guardadas son PRE-ETIQUETAS del modelo: hay que revisarlas en la
herramienta de etiquetado, sobre todo los objetos que el modelo NO detectó (una caja
faltante le enseña al modelo que ese objeto es fondo). Los tracks "lost" (retenidos por
ByteTrack sin detección real) no se guardan como etiquetas.

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
        videos/                      # solo con --record-video
            <sesión>.mp4             # bird view crudo (sin anotaciones)
            <sesión>.jsonl           # una línea por frame: tiempo + detecciones
        sessions/
            <sesión>.json            # manifiesto: modelo, argumentos, conteos

    (los nombres reales son predict_<sesión>_<índice>, p. ej. predict_20261105_153000_000042)
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from cv_system.calibration.result import CalibrationResult
from cv_system.config import load_config
from cv_system.detection import (
    CardDetection,
    CardDetector,
    E2ECardDetector,
    RFDETRCardDetector,
)
from cv_system.hardware import HardwareError, HardwareManager, PyKinect2HardwareManager
from cv_system.transform.rgb_image_transformer import RgbImageTransformer

PREDICT_OUTPUT_DIR = os.getenv("PREDICT_OUTPUT_DIR", "images/annotated")


def new_session_id() -> str:
    """ID de sesión legible y ordenable: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def scene_signature(frame: np.ndarray) -> np.ndarray:
    """Huella barata de la escena: gris a baja resolución, para detectar cambios."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)


def scene_changed(
    previous: np.ndarray | None, current: np.ndarray, threshold: float
) -> bool:
    """True si la diferencia media absoluta (0-255) supera el umbral. threshold<=0 la desactiva."""
    if previous is None or threshold <= 0:
        return True
    return float(np.mean(np.abs(current - previous))) >= threshold


def detection_to_dict(d: CardDetection) -> dict:
    """Detección serializable para el registro por frame."""
    return {
        "class_id": d.class_id,
        "label": d.label,
        "confidence": round(d.confidence, 4),
        "bbox_xyxy": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
        "track_id": d.track_id,
        "status": d.status,
    }


class SessionRecorder:
    """Graba el bird view crudo (mp4) y las detecciones de cada frame (jsonl).

    El video sirve para reproducir sesiones reales offline con cualquier modelo;
    el jsonl permite medir el comportamiento del modelo en vivo (continuidad,
    cambios de ID) sin volver a correr nada.

    La codificación del mp4 corre en un hilo aparte: codificar 1080p en el hilo
    del loop le quitaba ~40 ms por frame (de ~17 a ~10 fps), y eso cambia lo que
    se quiere medir (el tracker se comporta distinto a otros fps). La cola es
    acotada y bloqueante: si el codificador se atrasa, el loop espera en vez de
    descartar frames, así el frame N del mp4 siempre es la línea N del jsonl.
    El jsonl se escribe en el hilo del loop, con el tiempo real de cada frame.
    """

    QUEUE_FRAMES = 60  # ~180 MB de 1080p BGR como máximo en memoria

    def __init__(self, videos_dir: Path, session_id: str, fps: float) -> None:
        self.video_path = videos_dir / f"{session_id}.mp4"
        self.log_path = videos_dir / f"{session_id}.jsonl"
        self._fps = fps
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(self.QUEUE_FRAMES)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        # Vive toda la sesión; se cierra en close()
        self._log = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._t0 = time.monotonic()
        self._last_t = 0.0
        self.frames = 0
        self.waits = 0  # veces que el loop esperó al codificador (cola llena)

    def _encode(self, size: tuple[int, int]) -> None:
        writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, size
        )
        try:
            if not writer.isOpened():
                raise RuntimeError(f"No se pudo abrir el VideoWriter: {self.video_path}")
            while (frame := self._queue.get()) is not None:
                writer.write(frame)
        except BaseException as e:  # se re-lanza en el hilo del loop
            self._error = e
            while self._queue.get() is not None:  # vaciar para no bloquear al loop
                pass
        finally:
            writer.release()

    def write(self, frame: np.ndarray, detections: list[CardDetection]) -> None:
        if self._error is not None:
            raise RuntimeError(f"Falló la grabación del video: {self._error}")
        if self._thread is None:
            h, w = frame.shape[:2]
            self._thread = threading.Thread(
                target=self._encode, args=((w, h),), name="VideoEncoder", daemon=True
            )
            self._thread.start()
        # El frame no se copia: cada iteración del loop crea un bird_np nuevo y
        # nadie lo modifica después.
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.waits += 1
            self._queue.put(frame)
        self._last_t = time.monotonic() - self._t0
        record = {
            "frame": self.frames,
            "t": round(self._last_t, 4),
            "detections": [detection_to_dict(d) for d in detections],
        }
        self._log.write(json.dumps(record) + "\n")
        self.frames += 1

    @property
    def effective_fps(self) -> float:
        """Frames por segundo reales del loop (el mp4 dice ``fps`` nominal)."""
        return self.frames / self._last_t if self._last_t > 0 else 0.0

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        self._log.close()
        if self._error is not None:
            print(f"ERROR en la grabación del video: {self._error}", flush=True)


def write_session_manifest(path: Path, manifest: dict) -> None:
    """Escribe el manifiesto de la sesión (se reescribe al terminar con los conteos)."""
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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


def create_hardware_manager(
    hardware: str,
) -> HardwareManager | PyKinect2HardwareManager:
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
    rgb_np = (
        rgb_frame.get() if isinstance(rgb_frame, cv2.UMat) else np.asarray(rgb_frame)
    )
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
    session_id: str,
    index: int,
    bird_np,
    full_annotated,
    detections: list[CardDetection],
    readable: bool,
) -> str:
    """Guarda imagen, labels YOLO y debug. Devuelve el nombre base del archivo.

    Solo las detecciones "active" se guardan como etiquetas: un track "lost" es la
    memoria del tracker en una posición vieja, no algo que el detector vio.
    """
    filename = f"predict_{session_id}_{index:06d}"
    img_h, img_w = bird_np.shape[:2]
    detections = [d for d in detections if d.status == "active"]

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
        "--interval",
        type=float,
        default=0.5,
        help="Segundos entre capturas (si --save)",
    )
    parser.add_argument(
        "--output", type=str, default=PREDICT_OUTPUT_DIR, help="Carpeta de salida"
    )
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo YOLO")
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Threshold de confianza (por defecto: el del model_card.json, o 0.25)",
    )
    parser.add_argument(
        "--readable",
        action="store_true",
        help="Generar labels legibles (con nombres de clase)",
    )
    parser.add_argument(
        "--min-change",
        type=float,
        default=4.0,
        help=(
            "Diferencia media mínima (0-255) respecto a la última imagen guardada para "
            "que la captura automática guarde otra. 0 = guardar siempre. "
            "La captura manual (ESPACIO) ignora este filtro."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Grabar el bird view crudo (mp4) y las detecciones por frame (jsonl)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help=(
            "FPS nominal del mp4; los fps reales quedan en el manifiesto "
            "y el tiempo de cada frame en el jsonl"
        ),
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Usar E2ECardDetector para modelos con NMS integrado",
    )
    parser.add_argument(
        "--rfdetr",
        action="store_true",
        help="Usar RFDETRCardDetector para modelos RF-DETR",
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
    session_id = new_session_id()
    output_dir = Path(args.output) if (args.save or args.record_video) else None
    if output_dir:
        (output_dir / "sessions").mkdir(parents=True, exist_ok=True)
    if args.save:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / "debug").mkdir(parents=True, exist_ok=True)
        if args.readable:
            (output_dir / "labels_readable").mkdir(parents=True, exist_ok=True)
    if args.record_video:
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

    # Cargar configuración y calibración
    config_path = Path(os.getenv("CONFIG_PATH", "config/session.json"))
    calibration_path = Path(os.getenv("CALIBRATION_PATH", "config/calibration.json"))
    model_path = args.model or os.getenv("YOLO_MODEL_PATH", "models/best.pt")

    config = load_config(config_path)
    calibration = CalibrationResult.load(calibration_path)

    hw = None
    recorder: SessionRecorder | None = None
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
        # Cada sesión numera desde 0: el ID de sesión ya hace único el nombre
        current_index = 0
        saved = 0
        last_save = 0.0
        last_saved_signature: np.ndarray | None = None

        manifest = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "model_path": str(model_path),
            "detector": DetectorClass.__name__,
            "hardware": args.hardware,
            "conf_threshold": detector._conf_threshold,  # effective: CLI > model card > 0.25
            "model_card": str(detector._card.source) if detector._card.source else None,
            "input_resize": detector._card.resize,
            "dml_device_id": os.getenv("DML_DEVICE_ID", "0"),  # ONNX/DirectML adapter
            "save": args.save,
            "interval_s": args.interval,
            "min_change": args.min_change,
            "record_video": args.record_video,
        }
        manifest_path = (
            output_dir / "sessions" / f"{session_id}.json" if output_dir else None
        )
        if manifest_path:
            write_session_manifest(manifest_path, manifest)
        if args.record_video:
            recorder = SessionRecorder(
                output_dir / "videos", session_id, args.video_fps
            )
            print(f"Grabando video de la sesión: {recorder.video_path}", flush=True)

        print("Ejecutando predicciones en vivo...", flush=True)
        print("Presiona 'q' para salir", flush=True)
        if args.save:
            print(f"Guardando en formato YOLO: {output_dir}", flush=True)
            if args.interval > 0:
                print(f"Captura automática cada {args.interval}s", flush=True)
            print("Presiona ESPACIO para capturar manualmente", flush=True)
            print(f"Sesión: {session_id}", flush=True)

        cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)

        while True:
            bird_np = acquire_bird_view(hw, transformer, config.camera)

            # Detectar (UMat solo para la API del detector; datos en CPU)
            full_annotated, detections = detector.detect(cv2.UMat(bird_np))

            if recorder is not None:
                recorder.write(bird_np, detections)

            # Mostrar info de detecciones
            if detections:
                labels = [d.label for d in detections]
                print(f"Detectado: {', '.join(labels)}")

            cv2.imshow("Predictions", full_annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            manual_capture = key == ord(" ")
            auto_capture = False
            signature = None
            if (
                args.save
                and args.interval > 0
                and (time.time() - last_save >= args.interval)
            ):
                signature = scene_signature(bird_np)
                auto_capture = scene_changed(
                    last_saved_signature, signature, args.min_change
                )

            if args.save and (manual_capture or auto_capture):
                filename = save_prediction_sample(
                    output_dir,
                    session_id,
                    current_index,
                    bird_np,
                    full_annotated,
                    detections,
                    args.readable,
                )
                current_index += 1
                saved += 1
                last_save = time.time()
                last_saved_signature = (
                    signature if signature is not None else scene_signature(bird_np)
                )
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
        if recorder is not None:
            recorder.close()
            print(
                f"Video guardado: {recorder.video_path} ({recorder.frames} frames, "
                f"{recorder.effective_fps:.1f} fps reales; el loop esperó al "
                f"codificador {recorder.waits} veces)",
                flush=True,
            )
        if output_dir is not None:
            manifest_path = output_dir / "sessions" / f"{session_id}.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
                manifest["images_saved"] = saved
                manifest["video_frames"] = (
                    recorder.frames if recorder is not None else 0
                )
                if recorder is not None:
                    # El mp4 lleva --video-fps (nominal); el tiempo real está en el jsonl
                    manifest["video_fps_nominal"] = args.video_fps
                    manifest["effective_fps"] = round(recorder.effective_fps, 2)
                    manifest["encoder_waits"] = recorder.waits
                write_session_manifest(manifest_path, manifest)
        if hw is not None:
            try:
                hw.shutdown()
            except Exception as e:
                print(f"Advertencia al cerrar Kinect: {e}", flush=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
