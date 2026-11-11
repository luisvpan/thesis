"""Session recording of the cv-system (``CV_RECORD=1``).

A session writes, under an output directory (``images/annotated`` by default)::

    videos/<session>.mp4      raw bird view, one frame per loop iteration
    videos/<session>.jsonl    one line per video frame: real time + detections
    sessions/<session>.json   manifest: model, threshold, effective fps, ...

vision-model copies these to ``data/recordings`` and picks frames with
``vm-extract``. Frame N of the mp4 is always line N of the jsonl, and each is
one new camera frame (repeated color frames are not recorded; see main.py).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # runtime import would pull the whole detection stack
    from cv_system.detection.card_detector import CardDetection

RECORD_OUTPUT_DIR = "images/annotated"


def new_session_id() -> str:
    """ID de sesión legible y ordenable: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def detector_manifest(detector, model_path) -> dict:
    """What a recording needs to know about the card detector that produced it."""
    card = getattr(detector, "_card", None)
    return {
        "model_path": str(model_path),
        "detector": type(detector).__name__,
        "conf_threshold": getattr(detector, "_conf_threshold", None),
        "model_card": str(card.source) if card is not None and card.source else None,
        "input_resize": card.resize if card is not None else None,
    }


def finish_manifest(path: Path, recorder: SessionRecorder | None, video_fps: float, **extra) -> None:
    """Close the manifest with end time, frame counts and the real frame rate."""
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["video_frames"] = recorder.frames if recorder is not None else 0
    if recorder is not None:
        # The mp4 carries video_fps (nominal); real per-frame time is in the jsonl
        manifest["video_fps_nominal"] = video_fps
        manifest["effective_fps"] = round(recorder.effective_fps, 2)
        manifest["encoder_waits"] = recorder.waits
    manifest.update(extra)
    write_session_manifest(path, manifest)
