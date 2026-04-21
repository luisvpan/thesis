import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class MarkerDetector:
    """
    Detector de marcadores por brillo relativo.
    Funciona aunque la cámara destruya el color (sobreexposición).
    Estrategia: los marcadores son siempre los blobs más brillantes
    con forma aproximadamente rectangular.
    """
    # Kinect VGA (640×480): cuadrados de ~100 px en proyector ocupan menos píxeles;
    # umbrales demasiado altos eliminan todos los blobs antes del convexHull en debug.
    DEFAULT_MIN_AREA = 400
    DEFAULT_MAX_AREA = 15000
    DEFAULT_BRIGHTNESS_PERCENTILE = 98

    def __init__(
        self,
        min_area: int | None = None,
        max_area: int | None = None,
        brightness_percentile: float | None = None,
    ) -> None:
        self.min_area = min_area if min_area is not None else self.DEFAULT_MIN_AREA
        self.max_area = max_area if max_area is not None else self.DEFAULT_MAX_AREA
        self.brightness_percentile = (
            brightness_percentile
            if brightness_percentile is not None
            else self.DEFAULT_BRIGHTNESS_PERCENTILE
        )

    def detect_markers(self, rgb_frame: np.ndarray) -> list[tuple[int, int]]:
        # HardwareManager devuelve BGR (convención OpenCV tras captura OpenNI).
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        # 1. Threshold por percentil de brillo
        #    Los marcadores son siempre los píxeles más brillantes del frame
        thresh_value = np.percentile(gray, self.brightness_percentile)
        # Piso mínimo: evita activar en frames muy sobreexpuestos globalmente
        thresh_value = max(thresh_value, 160)
        _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

        # 2. Morfología: cerrar el hueco central (sobreexposición colapsa a blanco)
        #    y limpiar píxeles sueltos
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        # --- DEBUG ---
        cv2.namedWindow("Marker Detection - Binary Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Binary Mask", binary)
        cv2.waitKey(1)

        # 3. Filtrado de contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_markers = []
        debug_img = rgb_frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            # Rectangularidad: área del contorno vs área del bounding rect
            # Un marcador cuadrado/rectangular tendrá ratio alto (>0.5)
            # El ruido y reflejos irregulares tendrán ratio bajo
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < 0.55:
                continue

            # Aspect ratio: los marcadores son aproximadamente cuadrados
            # desde la perspectiva de la cámara, no extremadamente alargados
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio > 3.0:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            valid_markers.append({
                "x": cx, "y": cy, "area": area,
                "bbox": (x, y, w, h)  # ya lo tienes calculado arriba
            })
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(debug_img, (cx, cy), 6, (255, 0, 0), -1)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            print(f"area={area:.0f} rect={rect_area} rectang={rectangularity:.2f} ar={aspect_ratio:.2f} pos=({x},{y})")

        cv2.namedWindow("Marker Detection - Visual Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Visual Debug", debug_img)
        cv2.waitKey(1)

        valid_markers.sort(key=lambda m: m["area"], reverse=True)

        if len(valid_markers) > 4:
            best_four = self._select_best_four(valid_markers)
        else:
            best_four = valid_markers

        self._debug_best_four(rgb_frame, valid_markers, best_four)

        if len(valid_markers) < 4:
            logger.warning(f"Detección incompleta: {len(valid_markers)}/4 marcadores.")
            raise ValueError(f"No se detectaron los 4 marcadores ({len(valid_markers)}/4)")
        
        return self._sort_markers_by_position(best_four)
    
    def _sort_markers_by_position(self, markers: list[dict]) -> list[tuple[int, int]]:
        """Devuelve 4 vértices en orden TL, TR, BL, BR (coord. imagen).

        El orden debe coincidir con ``calibration.projector_corners`` (TL, TR, BL, BR).

        No usamos cuadrantes respecto al centroide (falla con perspectiva). Tampoco
        basta ``y`` menor = fila superior si la línea entre marcadores está muy
        inclinada. Orden tipo escaneo de documento sobre los centroides:

        TL = argmin(x+y), BR = argmax(x+y), TR = argmin(y-x), BL = argmax(y-x).
        """
        assert len(markers) == 4, f"Se esperaban 4 marcadores, llegaron {len(markers)}"

        pts = np.array([[m["x"], m["y"]] for m in markers], dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        tl_i = int(np.argmin(s))
        br_i = int(np.argmax(s))
        tr_i = int(np.argmin(d))
        bl_i = int(np.argmax(d))

        if len({tl_i, tr_i, bl_i, br_i}) < 4:
            # Caso degenerado: desempatar por filas/columnas
            indices_by_y = sorted(range(4), key=lambda i: markers[i]["y"])
            top_idx = sorted(indices_by_y[:2], key=lambda i: markers[i]["x"])
            bot_idx = sorted(indices_by_y[2:], key=lambda i: markers[i]["x"])
            tl_i, tr_i, bl_i, br_i = top_idx[0], top_idx[1], bot_idx[0], bot_idx[1]

        tl_m = markers[tl_i]
        tr_m = markers[tr_i]
        bl_m = markers[bl_i]
        br_m = markers[br_i]

        def outer_vertex(m: dict, role: str) -> tuple[int, int]:
            x, y, w, h = m["bbox"]
            if role == "tl":
                return (int(x), int(y))
            if role == "tr":
                return (int(x + w), int(y))
            if role == "bl":
                return (int(x), int(y + h))
            return (int(x + w), int(y + h))

        return [
            outer_vertex(tl_m, "tl"),
            outer_vertex(tr_m, "tr"),
            outer_vertex(bl_m, "bl"),
            outer_vertex(br_m, "br"),
        ]

    def _select_best_four(self, markers: list[dict]) -> list[dict]:
        from itertools import combinations

        best_area = 0
        best_group = markers[:4]

        for group in combinations(markers, 4):
            pts = np.array([(m["x"], m["y"]) for m in group], dtype=np.float32)
            hull = cv2.convexHull(pts)
            area = cv2.contourArea(hull)
            if area > best_area:
                best_area = area
                best_group = list(group)

        return best_group

    def _debug_best_four(self, rgb_frame: np.ndarray, all_markers: list[dict], best_four: list[dict]) -> None:
        debug_img = rgb_frame.copy()
        h, w = debug_img.shape[:2]
        scale = 2
        debug_img = cv2.resize(debug_img, (w * scale, h * scale))

        # Dibuja todos los candidatos en rojo
        for m in all_markers:
            cx, cy = m["x"] * scale, m["y"] * scale
            cv2.circle(debug_img, (cx, cy), 8, (255, 0, 0), -1)
            cv2.putText(debug_img, f"a={m['area']:.0f}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Dibuja los best four en verde con líneas del cuadrilátero (convexHull exige ≥3 puntos)
        best_pts = np.array([(m["x"] * scale, m["y"] * scale) for m in best_four], dtype=np.float32)
        if len(best_four) >= 3:
            hull = cv2.convexHull(best_pts.astype(np.int32))
            cv2.polylines(debug_img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

        for i, m in enumerate(best_four):
            cx, cy = m["x"] * scale, m["y"] * scale
            cv2.circle(debug_img, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(debug_img, f"#{i+1} a={m['area']:.0f}", (cx + 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.namedWindow("Marker Detection - Best Four", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Best Four", debug_img)
        cv2.waitKey()