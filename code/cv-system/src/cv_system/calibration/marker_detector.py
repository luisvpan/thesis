import logging
import numpy as np
import cv2

# Configuración de logging
logger = logging.getLogger(__name__)


class MarkerDetector:
    """Detecta 4 marcadores cuadrados blancos para calibración automática.

    Utiliza umbral adaptativo para ser robusto a cambios de iluminación y
    al viñeteado (oscurecimiento de esquinas) proyectado.
    """

    # Parámetros de detección por defecto
    DEFAULT_MIN_AREA = 2500      # ~50x50 píxeles
    DEFAULT_MAX_AREA = 40000     # ~200x200 píxeles
    
    # Parámetros Críticos del Umbral Adaptativo:
    # block_size: Tamaño de la vecindad local. Debe ser impar y > que el marcador.
    DEFAULT_BLOCK_SIZE = 151 
    # c_value: Constante restada de la media. Ayuda a eliminar ruido del fondo.
    DEFAULT_C = 5

    def __init__(
        self,
        min_area: int | None = None,
        max_area: int | None = None,
        block_size: int | None = None,
        c_value: int | None = None,
    ) -> None:
        """Inicializa el MarkerDetector."""
        self.min_area = min_area if min_area is not None else self.DEFAULT_MIN_AREA
        self.max_area = max_area if max_area is not None else self.DEFAULT_MAX_AREA
        
        # Validación: block_size DEBE ser impar
        bs = block_size if block_size is not None else self.DEFAULT_BLOCK_SIZE
        self.block_size = bs if bs % 2 != 0 else bs + 1
        
        self.c_value = c_value if c_value is not None else self.DEFAULT_C

        logger.info(
            f"MarkerDetector inicializado: area=[{self.min_area}, {self.max_area}], "
            f"adaptive: block_size={self.block_size}, C={self.c_value}"
        )

    def detect_markers(self, rgb_frame: np.ndarray) -> list[tuple[int, int]]:
        """Detecta 4 marcadores blancos en el frame RGB.

        Proceso:
        1. Convierte a escala de grises.
        2. Aplica desenfoque Gaussiano (reduce ruido antes del umbral).
        3. Aplica Umbral Adaptativo (cv2.adaptiveThreshold).
        4. Encuentra contornos y filtra por área y forma.
        5. Extrae centroides y ordena.
        """
        # --- 1. Pre-procesamiento ---
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # Desenfoque suave para mejorar el umbral adaptativo
        gray_smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- 2. Umbral Adaptativo (Sustituye al umbral global) ---
        # Se usa GAUSSIAN_C porque es más robusto al ruido que MEAN_C
        binary = cv2.adaptiveThreshold(
            gray_smoothed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c_value
        )

        # =========================================================================
        # VENTANAS DE DEBUGGING (Mantener tal cual el original)
        # =========================================================================
        cv2.namedWindow("Marker Detection - Grayscale", cv2.WINDOW_NORMAL)
        # Mostramos la imagen binaria resultante del umbral adaptativo
        cv2.imshow("Marker Detection - Grayscale", binary) 
        cv2.waitKey()

        # Nota: He omitido la ventana "Grayscale Normalized" porque adaptiveThreshold
        # ya genera una imagen binaria pura (0 o 255), la normalización es redundante aquí.
        # =========================================================================

        # --- 3. Encontrar Contornos ---
        # Usamos la imagen binaria directamente
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # =========================================================================
        # VENTANA DE DEBUGGING: Contornos RAW
        # =========================================================================
        rgb_with_contours = rgb_frame.copy()
        cv2.drawContours(rgb_with_contours, contours, -1, (0, 255, 0), 3)
        cv2.namedWindow(
            "Marker Detection - Grayscale Normalized With Contours", cv2.WINDOW_NORMAL
        )
        cv2.imshow(
            "Marker Detection - Grayscale Normalized With Contours", rgb_with_contours
        )
        cv2.waitKey()
        # =========================================================================

        logger.info(f"Encontrados {len(contours)} contornos totales.")

        if len(contours) == 0:
            raise ValueError(
                f"No se encontraron contornos. Revisa block_size ({self.block_size}) "
                f"o C ({self.c_value})."
            )

        # Imagen para dibujar contornos filtrados
        full_mask = binary.copy() 
        
        # --- 4. Filtrar Contornos ---
        valid_markers = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # A. Filtro por área
            if area < self.min_area or area > self.max_area:
                logger.debug(f"Contorno {i}: área={area} rechazada.")
                continue

            # B. Filtro por relación de aspecto (Comentado como en el original)
            # x, y, w, h = cv2.boundingRect(contour)
            # aspect_ratio = float(w) / h if h > 0 else 0
            # logger.debug(f"Contorno {i}: aspect_ratio={aspect_ratio:.2f}")

            # C. Filtro por forma (Aproximación de polígono) - RECOMENDADO ACTIVAR
            # peri = cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            # if len(approx) != 4: # Buscamos cuadriláteros
            #     continue

            # D. Extraer Centroide
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            # Dibujar contorno válido en la máscara de debug
            cv2.drawContours(full_mask, [contour], -1, (255, 255, 255), 2)

            marker_info = {
                "x": centroid_x,
                "y": centroid_y,
                "area": area,
            }
            valid_markers.append(marker_info)

            logger.debug(
                f"Marcador válido {len(valid_markers)}: ({centroid_x}, {centroid_y}), area={area}"
            )

        # =========================================================================
        # VENTANA DE DEBUGGING: Máscara de contornos filtrados
        # =========================================================================
        cv2.namedWindow("Marker Detection - Contour Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Contour Mask", full_mask)
        cv2.waitKey()
        # =========================================================================

        logger.info(f"Filtrados a {len(valid_markers)} marcadores válidos.")

        # --- 5. Validación y Ordenamiento ---
        if len(valid_markers) < 4:
            raise ValueError(
                f"Solo se detectaron {len(valid_markers)} marcadores válidos. "
                f"Se necesitan exactamente 4. Ajusta parámetros adaptativos o de área."
            )

        # Si hay más de 4, tomar los 4 más grandes (más probables)
        if len(valid_markers) > 4:
            valid_markers.sort(key=lambda m: m["area"], reverse=True)
            valid_markers = valid_markers[:4]
            logger.warning("Detectados >4 marcadores. Seleccionados los 4 más grandes.")

        # Ordenar: top-left, top-right, bottom-left, bottom-right
        sorted_markers = self._sort_markers_by_position(valid_markers)
        camera_corners = [(m["x"], m["y"]) for m in sorted_markers]

        logger.info(f"Marcadores ordenados: {camera_corners}")

        return camera_corners

    def _sort_markers_by_position(self, markers: list[dict]) -> list[dict]:
        """Clasifica los 4 marcadores en cuadrantes (igual que el original)."""
        if len(markers) != 4:
            raise ValueError(f"Se esperaban 4 marcadores, recibidos {len(markers)}")

        avg_x = sum(m["x"] for m in markers) / 4.0
        avg_y = sum(m["y"] for m in markers) / 4.0

        top_markers = [m for m in markers if m["y"] < avg_y]
        bottom_markers = [m for m in markers if m["y"] >= avg_y]

        top_markers.sort(key=lambda m: m["x"])
        top_left, top_right = top_markers[0], top_markers[1]

        bottom_markers.sort(key=lambda m: m["x"])
        bottom_left, bottom_right = bottom_markers[0], bottom_markers[1]

        return [top_left, top_right, bottom_left, bottom_right]