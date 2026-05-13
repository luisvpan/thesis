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

    Soporta detección de grillas de cualquier tamaño (4, 9, 16, etc. markers).
    """
    DEFAULT_MIN_AREA = 1000   # filtra ruido pequeño (área < 156, 20, 25...)
    DEFAULT_MAX_AREA = 10000  # filtra ruido grande
    DEFAULT_BRIGHTNESS_PERCENTILE = 98

    def __init__(
        self,
        min_area: int | None = None,
        max_area: int | None = None,
        brightness_percentile: float | None = None,
        expected_count: int = 4,  # Expected number of markers (4, 9, 16, etc.)
    ) -> None:
        self.min_area = min_area if min_area is not None else self.DEFAULT_MIN_AREA
        self.max_area = max_area if max_area is not None else self.DEFAULT_MAX_AREA
        self.brightness_percentile = (
            brightness_percentile
            if brightness_percentile is not None
            else self.DEFAULT_BRIGHTNESS_PERCENTILE
        )
        self.expected_count = expected_count

    def detect_markers(self, rgb_frame: np.ndarray, expected_count: int | None = None) -> list[tuple[int, int]]:
        """Detect markers in RGB frame and return their centroids.

        Args:
            rgb_frame: RGB image from camera
            expected_count: Number of markers to detect (overrides self.expected_count)

        Returns:
            List of (x, y) centroid positions, sorted row by row (top to bottom, left to right)
        """
        count = expected_count if expected_count is not None else self.expected_count
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        # 1. Threshold por percentil de brillo
        thresh_value = np.percentile(gray, self.brightness_percentile)
        thresh_value = max(thresh_value, 80)  # Magenta en grayscale es ~105
        _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

        # 2. Morfología
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        cv2.namedWindow("Marker Detection - Binary Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Binary Mask", binary)
        cv2.waitKey(1)

        # 3. Filtrado de contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"[MarkerDetector] Found {len(contours)} contours")

        valid_markers = []
        debug_img = rgb_frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                print(f"  Contour rejected: area={area:.0f} (limits: {self.min_area}-{self.max_area})")
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            if rectangularity < 0.55:
                print(f"  Contour rejected: rectangularity={rectangularity:.2f} < 0.55")
                continue

            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio > 3.0:
                print(f"  Contour rejected: aspect_ratio={aspect_ratio:.2f} > 3.0")
                continue

            print(f"  Valid marker: area={area:.0f}, rect={rectangularity:.2f}, aspect={aspect_ratio:.2f}")

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            # Usar CENTROIDE (más estable que esquinas de bbox)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            valid_markers.append({
                "x": cx, "y": cy, "area": area,
                "bbox": (x, y, w, h)
            })
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(debug_img, (cx, cy), 6, (255, 0, 0), -1)

        cv2.namedWindow("Marker Detection - Visual Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Visual Debug", debug_img)
        cv2.waitKey(1)

        # Seleccionar los mejores markers si hay más de los esperados
        if len(valid_markers) > count:
            best_markers = self._select_best_n(valid_markers, count)
        else:
            best_markers = valid_markers

        self._debug_markers(rgb_frame, valid_markers, best_markers, count)

        if len(best_markers) < count:
            logger.warning(f"Detección incompleta: {len(best_markers)}/{count} marcadores.")
            print(f"[MarkerDetector] ERROR: {len(best_markers)}/{count} markers. Press any key to continue...")
            cv2.waitKey(0)  # Wait for user to see debug windows
            raise ValueError(f"No se detectaron los {count} marcadores ({len(best_markers)}/{count})")

        # Ordenar por posición en grilla (row by row)
        return self._sort_markers_grid(best_markers)
    
    def _sort_markers_grid(self, markers: list[dict]) -> list[tuple[int, int]]:
        """Sort markers by grid position (row by row, left to right).

        Uses clustering to determine rows, then sorts within each row by x.
        Returns centroids (not bbox corners) for better accuracy.
        """
        if len(markers) == 0:
            return []

        # Sort by y to group into rows
        sorted_by_y = sorted(markers, key=lambda m: m["y"])

        # Determine approximate number of rows (assume square-ish grid)
        n = len(markers)
        rows_estimate = int(np.sqrt(n) + 0.5)

        # Group into rows based on y-coordinate clustering
        rows = []
        current_row = [sorted_by_y[0]]

        for i in range(1, len(sorted_by_y)):
            m = sorted_by_y[i]
            prev = sorted_by_y[i - 1]
            # If y-gap is large, start new row
            # Use adaptive threshold based on average marker height
            avg_height = sum(m["bbox"][3] for m in markers) / len(markers)
            y_threshold = avg_height * 2  # markers in same row should be within 2x height

            if m["y"] - prev["y"] > y_threshold:
                rows.append(current_row)
                current_row = [m]
            else:
                current_row.append(m)
        rows.append(current_row)

        # Sort each row by x (left to right) and collect centroids
        result = []
        for row in rows:
            sorted_row = sorted(row, key=lambda m: m["x"])
            for m in sorted_row:
                result.append((m["x"], m["y"]))

        return result

    def _select_best_n(self, markers: list[dict], n: int) -> list[dict]:
        """Select the best N markers from candidates.

        Strategy: select N markers that maximize the convex hull area,
        ensuring good coverage of the calibration region.
        """
        from itertools import combinations

        if len(markers) <= n:
            return markers

        # For small n, brute force works
        if len(markers) <= 12:
            best_area = 0
            best_group = markers[:n]

            for group in combinations(markers, n):
                pts = np.array([(m["x"], m["y"]) for m in group], dtype=np.float32)
                hull = cv2.convexHull(pts)
                area = cv2.contourArea(hull)
                if area > best_area:
                    best_area = area
                    best_group = list(group)

            return best_group
        else:
            # For many candidates, use greedy approach:
            # Start with 4 corner-most markers, then add markers that maximize spread
            # Find extremes
            by_x = sorted(markers, key=lambda m: m["x"])
            by_y = sorted(markers, key=lambda m: m["y"])

            # Start with corners
            selected = set()
            corner_candidates = [
                min(markers, key=lambda m: m["x"] + m["y"]),  # top-left
                min(markers, key=lambda m: -m["x"] + m["y"]),  # top-right
                min(markers, key=lambda m: m["x"] - m["y"]),  # bottom-left
                min(markers, key=lambda m: -m["x"] - m["y"]),  # bottom-right
            ]

            for m in corner_candidates:
                selected.add(id(m))

            result = [m for m in markers if id(m) in selected]

            # Greedily add markers that are farthest from current selection
            remaining = [m for m in markers if id(m) not in selected]
            while len(result) < n and remaining:
                best_dist = -1
                best_marker = None
                for m in remaining:
                    # Min distance to any selected marker
                    min_dist = min(
                        np.sqrt((m["x"] - s["x"])**2 + (m["y"] - s["y"])**2)
                        for s in result
                    )
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_marker = m
                if best_marker:
                    result.append(best_marker)
                    remaining.remove(best_marker)

            return result

    def _debug_markers(self, rgb_frame: np.ndarray, all_markers: list[dict], best_markers: list[dict], expected_count: int) -> None:
        """Debug visualization showing all candidates and selected markers."""
        debug_img = rgb_frame.copy()
        h, w = debug_img.shape[:2]
        scale = 2
        debug_img = cv2.resize(debug_img, (w * scale, h * scale))

        # Draw all candidates in red
        for m in all_markers:
            cx, cy = m["x"] * scale, m["y"] * scale
            cv2.circle(debug_img, (cx, cy), 8, (255, 0, 0), -1)
            cv2.putText(debug_img, f"a={m['area']:.0f}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw best markers in green with convex hull
        if len(best_markers) >= 3:
            best_pts = np.array([(m["x"] * scale, m["y"] * scale) for m in best_markers], dtype=np.float32)
            hull = cv2.convexHull(best_pts.astype(np.int32))
            cv2.polylines(debug_img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

        for i, m in enumerate(best_markers):
            cx, cy = m["x"] * scale, m["y"] * scale
            cv2.circle(debug_img, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(debug_img, f"#{i+1}", (cx + 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Show status
        status = f"Detected: {len(best_markers)}/{expected_count}"
        cv2.putText(debug_img, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.namedWindow("Marker Detection - Grid", cv2.WINDOW_NORMAL)
        cv2.imshow("Marker Detection - Grid", debug_img)
        cv2.waitKey(1)