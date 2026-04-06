"""
Captura RGB del Kinect vía OpenNI2 (misma pila que computer-vision-manager).

Variable de entorno:
  VISION_OPENNI_REDIST — ruta a la carpeta Redist de OpenNI2 (por defecto
    C:\\Program Files\\OpenNI2\\Redist).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
from openni import openni2


class KinectColorStream:
    """Flujo de color del Kinect: frames BGR listos para OpenCV / YOLO."""

    def __init__(self, redist_path: str | None = None) -> None:
        self._redist = redist_path or os.environ.get(
            "VISION_OPENNI_REDIST",
            r"C:\Program Files\OpenNI2\Redist",
        )
        self._device = None
        self._color_stream = None

    def open(self) -> bool:
        try:
            openni2.initialize(self._redist)
        except Exception as e:
            print(
                f"OpenNI2 initialize falló ({self._redist}): {e}",
                file=sys.stderr,
            )
            return False
        try:
            self._device = openni2.Device.open_any()
            self._color_stream = self._device.create_color_stream()
            if self._color_stream is None:
                print("No hay flujo de color en el dispositivo.", file=sys.stderr)
                try:
                    openni2.unload()
                except Exception:
                    pass
                return False
            self._color_stream.start()
        except Exception as e:
            print(f"No se pudo abrir el Kinect: {e}", file=sys.stderr)
            try:
                openni2.unload()
            except Exception:
                pass
            return False
        return True

    def read_bgr(self) -> Optional[np.ndarray]:
        if self._color_stream is None:
            return None
        frame = self._color_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        frame_array = np.ndarray(
            (frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data
        )
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        return cv2.flip(frame_bgr, 1)

    def close(self) -> None:
        if self._color_stream is not None:
            try:
                self._color_stream.stop()
            except Exception:
                pass
            self._color_stream = None
        self._device = None
        try:
            openni2.unload()
        except Exception:
            pass

    def __enter__(self) -> KinectColorStream:
        if not self.open():
            raise RuntimeError(
                "No se pudo inicializar el Kinect (OpenNI2). "
                "Comprueba VISION_OPENNI_REDIST y que el sensor esté conectado."
            )
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
