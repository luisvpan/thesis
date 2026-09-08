"""
Test script for PyKinect2 with Python 3.12+
Detects hands using skin color segmentation or MediaPipe HandLandmarker,
then maps to depth frame to get fingertip depth.

Usage:
    uv run python examples/test_hand_detection.py              # Skin color detection
    uv run python examples/test_hand_detection.py --mediapipe  # MediaPipe detection

Requirements:
    - Kinect for Windows v2 SDK installed
    - Kinect v2 sensor connected
    - opencv-python installed (uv pip install opencv-python)
    - mediapipe installed (uv pip install mediapipe) [optional]

Controls:
    - Adjust trackbars to tune skin detection (skin mode only)
    - Press 'q' to quit
"""

import cv2
import numpy as np
import ctypes
import argparse
import time
import os
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# MediaPipe (optional)
mp = None
HandLandmarker = None
HandLandmarkerOptions = None
HandLandmarkerResult = None
HAND_CONNECTIONS = None


def init_mediapipe():
    """Initialize MediaPipe HandLandmarker (new API)."""
    global mp, HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, HAND_CONNECTIONS
    try:
        import mediapipe as _mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        mp = _mp
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        HandLandmarkerResult = vision.HandLandmarkerResult

        # Hand connections for drawing
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),  # Palm
        ]

        return True
    except ImportError as e:
        print(f"MediaPipe not installed or incompatible: {e}")
        print("Run: uv pip install mediapipe")
        return False


def create_hand_landmarker(model_path):
    """Create HandLandmarker instance."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def detect_hand_from_color(color_frame, ycrcb_min, ycrcb_max):
    """
    Detecta la mano usando segmentación de color de piel.

    Returns:
        hand_mask: Máscara binaria de la mano
        fingertips: Lista de (x, y) para cada punta de dedo
        contour: Contorno de la mano
    """
    # 1. Convertir a YCrCb
    ycrcb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2YCrCb)

    # 2. Aplicar umbral para detectar piel
    lower = np.array(ycrcb_min, dtype=np.uint8)
    upper = np.array(ycrcb_max, dtype=np.uint8)
    hand_mask = cv2.inRange(ycrcb, lower, upper)

    # 3. Morfología para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Encontrar contornos
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return hand_mask, [], None

    # 5. Tomar el contorno más grande
    hand_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(hand_contour) < 5000:
        return hand_mask, [], None

    # 6. Convex Hull
    hull = cv2.convexHull(hand_contour, returnPoints=False)

    # 7. Convexity Defects
    try:
        defects = cv2.convexityDefects(hand_contour, hull)
    except:
        return hand_mask, [], hand_contour

    if defects is None:
        return hand_mask, [], hand_contour

    # 8. Encontrar puntas de dedos
    fingertips = []
    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return hand_mask, [], hand_contour
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(hand_contour[s][0])
        if d > 15000:
            fingertips.append(start)

    if len(fingertips) < 3:
        hull_points = cv2.convexHull(hand_contour, returnPoints=True)
        candidates = []
        for point in hull_points:
            x, y = point[0]
            dist = (x - cx) ** 2 + (y - cy) ** 2
            candidates.append((x, y, dist))
        candidates.sort(key=lambda p: -p[2])
        fingertips = [(p[0], p[1]) for p in candidates[:5]]

    # Eliminar duplicados cercanos
    filtered_tips = []
    for tip in fingertips:
        is_duplicate = False
        for existing in filtered_tips:
            dist = ((tip[0] - existing[0]) ** 2 + (tip[1] - existing[1]) ** 2) ** 0.5
            if dist < 50:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_tips.append(tip)

    return hand_mask, filtered_tips[:5], hand_contour


def detect_hand_mediapipe(color_frame_rgb, landmarker, timestamp_ms):
    """
    Detecta la mano usando MediaPipe HandLandmarker.

    Args:
        color_frame_rgb: Frame RGB
        landmarker: HandLandmarker instance
        timestamp_ms: Timestamp in milliseconds

    Returns:
        fingertips: Lista de (x, y) para puntas de dedos (landmarks 4, 8, 12, 16, 20)
        palm_points: Lista de (x, y) para puntos de la palma (landmark 9 - base dedo medio)
        all_landmarks: Lista de todos los landmarks por mano (para dibujar)
    """
    h, w = color_frame_rgb.shape[:2]

    # Crear imagen MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame_rgb)

    # Detectar
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    fingertips = []
    palm_points = []
    all_landmarks = []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Indices de puntas de dedos:
            # 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
            tip_indices = [4, 8, 12, 16, 20]
            # Landmark 0 = muñeca (wrist) - más estable que el centro de la palma
            palm_index = 0

            hand_points = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                hand_points.append((x, y))

            for idx in tip_indices:
                fingertips.append(hand_points[idx])

            palm_points.append(hand_points[palm_index])
            all_landmarks.append(hand_points)

    return fingertips, palm_points, all_landmarks


def map_color_to_depth(kinect, color_points, depth_frame, return_indexed=False):
    """
    Mapea puntos de coordenadas de color a depth y obtiene la profundidad.

    Args:
        kinect: PyKinectRuntime instance
        color_points: Lista de (cx, cy) coordenadas en color space
        depth_frame: Frame de profundidad actual
        return_indexed: Si True, retorna dict {idx: data} en lugar de lista

    Returns:
        Si return_indexed=False (default):
            results: Lista de (cx, cy, dx, dy, depth_mm) para puntos válidos
        Si return_indexed=True:
            results: Dict {idx: (cx, cy, dx, dy, depth_mm)} preservando índices
        stats: Dict con {total, valid, ratio} para medir confianza
    """
    empty_stats = {'total': 0, 'valid': 0, 'ratio': 0.0}
    if not color_points:
        return ({} if return_indexed else []), empty_stats

    color_width = 1920
    color_height = 1080
    depth_width = 512
    depth_height = 424

    # Preparar el depth frame como puntero
    depth_frame_flat = depth_frame.flatten().astype(np.uint16)
    depth_ptr = depth_frame_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    # Crear array de salida
    depth_space_points = (PyKinectV2._DepthSpacePoint * (color_width * color_height))()

    use_fallback = False
    try:
        kinect._mapper.MapColorFrameToDepthSpace(
            depth_width * depth_height,
            depth_ptr,
            color_width * color_height,
            depth_space_points
        )
    except Exception as e:
        use_fallback = True

    # Extraer los puntos, preservando índices si se solicita
    results = {} if return_indexed else []
    valid_count = 0

    for point_idx, (cx, cy) in enumerate(color_points):
        dx, dy = None, None

        if not use_fallback:
            pixel_idx = cy * color_width + cx
            if 0 <= pixel_idx < len(depth_space_points):
                dp = depth_space_points[pixel_idx]
                # Si el mapeo SDK es válido, usarlo
                if not (np.isinf(dp.x) or np.isinf(dp.y) or dp.x < 0 or dp.y < 0):
                    dx = int(dp.x)
                    dy = int(dp.y)

        # Fallback: aproximación por escala si el mapeo SDK falla
        if dx is None or dy is None:
            dx = int(cx * depth_width / color_width)
            dy = int(cy * depth_height / color_height)

        if 0 <= dx < depth_width and 0 <= dy < depth_height:
            depth_mm = depth_frame[dy, dx]
            if depth_mm > 0:
                valid_count += 1
                data = (cx, cy, dx, dy, int(depth_mm))
                if return_indexed:
                    results[point_idx] = data
                else:
                    results.append(data)

    stats = {
        'total': len(color_points),
        'valid': valid_count,
        'ratio': valid_count / len(color_points) if color_points else 0.0
    }
    return results, stats


def fit_hand_plane(points_3d):
    """
    Ajusta un plano a un conjunto de puntos 3D usando SVD.

    Plano: ax + by + cz + d = 0, normalizado como (a,b,c) vector unitario.

    Args:
        points_3d: Lista de (x, y, z) con al menos 3 puntos

    Returns:
        plane: (a, b, c, d) coeficientes del plano, o None si no hay suficientes puntos
    """
    if len(points_3d) < 3:
        return None

    points = np.array(points_3d)

    # Centroide
    centroid = np.mean(points, axis=0)

    # Centrar los puntos
    centered = points - centroid

    # SVD para encontrar el vector normal (menor valor singular)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # Último vector = dirección de menor varianza = normal del plano

    # Normalizar
    normal = normal / np.linalg.norm(normal)

    # d = -normal · centroid
    d = -np.dot(normal, centroid)

    return (normal[0], normal[1], normal[2], d)


def estimate_depth_from_plane(plane, x, y):
    """
    Estima la profundidad z dado (x, y) y la ecuación del plano.

    Plano: ax + by + cz + d = 0
    Despejando: z = -(ax + by + d) / c

    Args:
        plane: (a, b, c, d) coeficientes del plano
        x, y: Coordenadas del punto

    Returns:
        z: Profundidad estimada, o None si c ≈ 0
    """
    a, b, c, d = plane

    if abs(c) < 1e-6:
        return None  # Plano vertical, no se puede estimar z

    z = -(a * x + b * y + d) / c
    return z


def interpolate_missing_depths(all_landmarks, landmarks_with_depth, kinect, depth_frame, compare_mode=False):
    """
    Interpola profundidades faltantes usando un plano ajustado a los landmarks válidos.

    Args:
        all_landmarks: Lista de (x, y) para todos los 21 landmarks de la mano
        landmarks_with_depth: Dict {index: (cx, cy, dx, dy, depth_mm)} para landmarks con depth válido
        kinect: PyKinectRuntime instance
        depth_frame: Depth frame actual
        compare_mode: Si True, calcula interpolación para TODOS los landmarks (para comparar)

    Returns:
        interpolated: Dict {index: (cx, cy, depth_mm, is_interpolated, interp_depth_mm)}
                      interp_depth_mm es el valor interpolado (para comparación), None si no aplica
        plane: Coeficientes del plano ajustado
    """
    if not landmarks_with_depth:
        return {}, None

    if len(landmarks_with_depth) < 3:
        # No hay suficientes puntos para ajustar un plano
        result = {}
        for idx, data in landmarks_with_depth.items():
            cx, cy, dx, dy, depth_mm = data
            result[idx] = (cx, cy, depth_mm, False, None)
        return result, None

    # Construir puntos 3D a partir de landmarks con depth válido
    # Normalizar coordenadas para evitar problemas numéricos
    points_3d = []
    depths = []
    for idx, data in landmarks_with_depth.items():
        cx, cy, dx, dy, depth_mm = data
        # Normalizar: x,y a [0,1], depth a [0,1] basado en rango típico
        points_3d.append((cx / 1920.0, cy / 1080.0, depth_mm / 1000.0))
        depths.append(depth_mm)

    # Calcular rango de depths válidos para filtrar estimaciones
    min_depth = min(depths)
    max_depth = max(depths)
    depth_range = max(max_depth - min_depth, 100)  # Mínimo 100mm de rango
    # Permitir un margen generoso fuera del rango observado
    valid_min = max(100, min_depth - depth_range - 100)  # Mínimo 100mm, nunca negativo
    valid_max = max_depth + depth_range + 100  # +100mm extra

    # Ajustar plano en coordenadas normalizadas
    plane = fit_hand_plane(points_3d)

    if plane is None:
        result = {}
        for idx, data in landmarks_with_depth.items():
            cx, cy, dx, dy, depth_mm = data
            result[idx] = (cx, cy, depth_mm, False, None)
        return result, None

    # Crear resultado con todos los landmarks
    result = {}

    # Para cada landmark, calcular valor interpolado para comparación
    for idx, (cx, cy) in enumerate(all_landmarks):
        # Calcular interpolación para este punto
        estimated_norm = estimate_depth_from_plane(plane, cx / 1920.0, cy / 1080.0)
        interp_depth = None
        if estimated_norm is not None:
            estimated_depth = estimated_norm * 1000.0
            if valid_min < estimated_depth < valid_max:
                interp_depth = estimated_depth

        if idx in landmarks_with_depth:
            # Tiene depth real
            cx, cy, dx, dy, depth_mm = landmarks_with_depth[idx]
            result[idx] = (cx, cy, depth_mm, False, interp_depth)
        else:
            # Solo tiene interpolación
            if interp_depth is not None:
                result[idx] = (cx, cy, interp_depth, True, interp_depth)

    return result, plane


def main():
    parser = argparse.ArgumentParser(description='Kinect hand detection with depth')
    parser.add_argument('--mediapipe', action='store_true',
                        help='Use MediaPipe instead of skin color detection')
    args = parser.parse_args()

    use_mediapipe = args.mediapipe
    landmarker = None

    if use_mediapipe:
        if not init_mediapipe():
            print("Falling back to skin color detection")
            use_mediapipe = False
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, "models", "hand_landmarker.task")

            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                print("Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
                print(f"Place it in: {os.path.join(project_root, 'models')}/")
                print("Falling back to skin color detection")
                use_mediapipe = False
            else:
                landmarker = create_hand_landmarker(model_path)
                print("Using MediaPipe HandLandmarker (video mode)")
    else:
        print("Using skin color detection")

    # Initialize Kinect
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color |
        PyKinectV2.FrameSourceTypes_Depth
    )

    print("Kinect initialized successfully!")
    print("Press 'q' to quit.")

    # Trackbars para skin detection
    if not use_mediapipe:
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Y min', 'Controls', 0, 255, lambda x: None)
        cv2.createTrackbar('Y max', 'Controls', 255, 255, lambda x: None)
        cv2.createTrackbar('Cr min', 'Controls', 135, 255, lambda x: None)
        cv2.createTrackbar('Cr max', 'Controls', 180, 255, lambda x: None)
        cv2.createTrackbar('Cb min', 'Controls', 85, 255, lambda x: None)
        cv2.createTrackbar('Cb max', 'Controls', 135, 255, lambda x: None)

    last_depth_frame = None

    # Timing stats
    detection_times = []
    mapping_times = []
    frame_count = 0
    start_time = time.perf_counter()

    while True:
        # Read trackbar values for skin detection
        if not use_mediapipe:
            y_min = cv2.getTrackbarPos('Y min', 'Controls')
            y_max = cv2.getTrackbarPos('Y max', 'Controls')
            cr_min = cv2.getTrackbarPos('Cr min', 'Controls')
            cr_max = cv2.getTrackbarPos('Cr max', 'Controls')
            cb_min = cv2.getTrackbarPos('Cb min', 'Controls')
            cb_max = cv2.getTrackbarPos('Cb max', 'Controls')
            ycrcb_min = (y_min, cr_min, cb_min)
            ycrcb_max = (y_max, cr_max, cb_max)

        # --- Depth Frame ---
        if kinect.has_new_depth_frame():
            last_depth_frame = kinect.get_last_depth_frame()
            last_depth_frame = last_depth_frame.reshape((424, 512))

        # --- Color Frame ---
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_frame = color_frame.reshape((1080, 1920, 4))
            color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

            # --- Detection ---
            t_detect_start = time.perf_counter()

            palm_points = []
            if use_mediapipe:
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                fingertips, palm_points, landmarks = detect_hand_mediapipe(color_rgb, landmarker, timestamp_ms)
                hand_mask = None
                contour = None
            else:
                hand_mask, fingertips, contour = detect_hand_from_color(
                    color_bgr, ycrcb_min, ycrcb_max
                )
                landmarks = None

            t_detect_end = time.perf_counter()
            detection_time_ms = (t_detect_end - t_detect_start) * 1000
            detection_times.append(detection_time_ms)

            # --- Mapping to depth ---
            t_map_start = time.perf_counter()

            # fingertips_data: lista de (cx, cy, real_depth, interp_depth, is_only_interp)
            fingertips_data = []
            palm_with_depth = []
            palm_depth_mm = None
            all_stats = {'total': 0, 'valid': 0, 'ratio': 0.0}
            hand_plane = None

            if last_depth_frame is not None and use_mediapipe and landmarks:
                # Mapear TODOS los 21 landmarks para ajustar el plano
                for hand_points in landmarks:
                    # Mapear todos los landmarks preservando índices
                    landmarks_with_depth, all_stats = map_color_to_depth(
                        kinect, hand_points, last_depth_frame, return_indexed=True
                    )

                    # Interpolar depths (con comparación)
                    interpolated_landmarks, hand_plane = interpolate_missing_depths(
                        hand_points, landmarks_with_depth, kinect, last_depth_frame,
                        compare_mode=True
                    )

                    # Extraer fingertips (índices 4, 8, 12, 16, 20)
                    tip_indices = [4, 8, 12, 16, 20]
                    for idx in tip_indices:
                        if idx in interpolated_landmarks:
                            cx, cy, depth_mm, is_interp, interp_depth = interpolated_landmarks[idx]
                            # real_depth es depth_mm si no es interpolado, None si solo tiene interp
                            real_depth = None if is_interp else depth_mm
                            fingertips_data.append((cx, cy, real_depth, interp_depth, is_interp))

                    # Extraer wrist (índice 0)
                    if 0 in interpolated_landmarks:
                        cx, cy, depth_mm, is_interp, interp_depth = interpolated_landmarks[0]
                        palm_with_depth.append((cx, cy, 0, 0, depth_mm))
                        palm_depth_mm = depth_mm

            elif last_depth_frame is not None and fingertips:
                # Modo skin detection (sin interpolación)
                mapped_tips, all_stats = map_color_to_depth(kinect, fingertips, last_depth_frame)
                for tip in mapped_tips:
                    cx, cy, dx, dy, depth_mm = tip
                    fingertips_data.append((cx, cy, depth_mm, None, False))
                if palm_points:
                    palm_with_depth, _ = map_color_to_depth(kinect, palm_points, last_depth_frame)
                    if palm_with_depth:
                        palm_depth_mm = palm_with_depth[0][4]

            t_map_end = time.perf_counter()
            mapping_time_ms = (t_map_end - t_map_start) * 1000
            mapping_times.append(mapping_time_ms)

            # Calcular profundidad promedio de la mano (usar real si existe, sino interp)
            all_depths = []
            for tip in fingertips_data:
                cx, cy, real_depth, interp_depth, is_only_interp = tip
                if real_depth is not None:
                    all_depths.append(real_depth)
                elif interp_depth is not None:
                    all_depths.append(interp_depth)
            avg_hand_depth_mm = np.mean(all_depths) if all_depths else 0

            # --- Visualization ---
            display_frame = cv2.resize(color_bgr, (960, 540))

            # Draw MediaPipe landmarks
            if use_mediapipe and landmarks:
                for hand_points in landmarks:
                    # Draw connections
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(hand_points) and end_idx < len(hand_points):
                            x1, y1 = hand_points[start_idx]
                            x2, y2 = hand_points[end_idx]
                            # Scale to display size
                            x1, y1 = x1 // 2, y1 // 2
                            x2, y2 = x2 // 2, y2 // 2
                            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw all landmarks
                    for x, y in hand_points:
                        cv2.circle(display_frame, (x // 2, y // 2), 3, (255, 0, 0), -1)

            # Draw contour (skin detection)
            if contour is not None:
                scaled_contour = (contour / 2).astype(np.int32)
                cv2.drawContours(display_frame, [scaled_contour], -1, (255, 255, 255), 2)

            # Draw fingertips with depth comparison (real vs interpolated)
            for tip_data in fingertips_data:
                cx, cy, real_depth, interp_depth, is_only_interp = tip_data
                sx, sy = cx // 2, cy // 2

                if is_only_interp:
                    # Solo interpolado (naranja)
                    cv2.circle(display_frame, (sx, sy), 8, (0, 165, 255), -1)
                    cv2.circle(display_frame, (sx, sy), 10, (255, 255, 255), 2)
                    depth_text = f"~{interp_depth/1000:.2f}m"
                    text_color = (0, 165, 255)
                else:
                    # Tiene valor real (verde)
                    cv2.circle(display_frame, (sx, sy), 8, (0, 255, 0), -1)
                    cv2.circle(display_frame, (sx, sy), 10, (255, 255, 255), 2)

                    # Mostrar comparación: Real / Interp
                    if interp_depth is not None:
                        diff = real_depth - interp_depth
                        depth_text = f"R:{real_depth/1000:.2f} I:{interp_depth/1000:.2f} ({diff:+.0f}mm)"
                    else:
                        depth_text = f"{real_depth/1000:.2f}m"
                    text_color = (0, 255, 0)

                cv2.putText(display_frame, depth_text, (sx + 12, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(display_frame, depth_text, (sx + 12, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            # Draw wrist with depth (cyan color to distinguish from fingertips)
            for palm_data in palm_with_depth:
                cx, cy, dx, dy, depth_mm = palm_data
                sx, sy = cx // 2, cy // 2

                cv2.circle(display_frame, (sx, sy), 10, (255, 255, 0), -1)  # Cyan fill
                cv2.circle(display_frame, (sx, sy), 12, (255, 255, 255), 2)  # White border

                depth_text = f"Wrist: {depth_mm/1000:.2f}m"
                cv2.putText(display_frame, depth_text, (sx + 14, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display_frame, depth_text, (sx + 14, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # --- Timing info ---
            frame_count += 1
            avg_detect = np.mean(detection_times[-30:]) if detection_times else 0
            avg_map = np.mean(mapping_times[-30:]) if mapping_times else 0

            mode_text = "MediaPipe" if use_mediapipe else "Skin Color"
            wrist_text = f"{palm_depth_mm/1000:.3f}m" if palm_depth_mm else "N/A"
            measured_tips = sum(1 for t in fingertips_data if not t[4])  # not is_only_interp
            interp_only_tips = sum(1 for t in fingertips_data if t[4])   # is_only_interp

            # Calcular error de interpolación
            interp_errors = []
            for tip in fingertips_data:
                cx, cy, real_depth, interp_depth, is_only_interp = tip
                if not is_only_interp and real_depth and interp_depth:
                    interp_errors.append(abs(real_depth - interp_depth))

            if interp_errors and avg_hand_depth_mm > 0:
                avg_error_mm = np.mean(interp_errors)
                avg_error_pct = (avg_error_mm / avg_hand_depth_mm) * 100
                error_text = f"Interp Error: {avg_error_mm:.0f}mm ({avg_error_pct:.1f}%)"
            else:
                error_text = "Interp Error: N/A"

            info_lines = [
                f"Mode: {mode_text}",
                f"Tips: {measured_tips} real + {interp_only_tips} interp-only",
                f"Wrist: {wrist_text}",
                f"Hand Depth: {avg_hand_depth_mm/1000:.3f}m",
                error_text,
                f"Detection: {avg_detect:.1f}ms",
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(display_frame, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)

            for i, line in enumerate(info_lines):
                cv2.putText(display_frame, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Kinect Hand Detection', display_frame)

            # Show mask for skin detection
            if hand_mask is not None:
                display_mask = cv2.resize(hand_mask, (480, 270))
                cv2.imshow('Skin Mask', display_mask)

        # --- Depth visualization ---
        if last_depth_frame is not None:
            depth_8bit = (last_depth_frame / 4500.0 * 255).clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            cv2.imshow('Kinect Depth', depth_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if landmarker:
        landmarker.close()
    kinect.close()
    cv2.destroyAllWindows()

    # Print final stats
    print("\n--- Performance Summary ---")
    print(f"Mode: {'MediaPipe' if use_mediapipe else 'Skin Color'}")
    print(f"Frames processed: {frame_count}")
    if detection_times:
        print(f"Detection avg: {np.mean(detection_times):.2f}ms")
    if mapping_times:
        print(f"Mapping avg: {np.mean(mapping_times):.2f}ms")
    print("Kinect closed.")


if __name__ == "__main__":
    main()
