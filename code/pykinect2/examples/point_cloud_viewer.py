"""
Point Cloud Viewer - Visualiza la alineación entre RGB y Depth del Kinect v2.

Muestra una nube de puntos coloreada para ver cómo se correlacionan
los frames de color y profundidad.

Usage:
    uv run python examples/point_cloud_viewer.py

Controls:
    - Press 'q' to quit
"""

import cv2
import numpy as np
import ctypes
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


def create_colored_point_cloud(kinect, color_frame, depth_frame):
    """
    Crea una imagen que muestra el depth frame coloreado con los colores del RGB.
    Esto permite visualizar la alineación/registro entre ambas cámaras.

    Returns:
        colored_depth: Imagen del tamaño del depth frame con colores del RGB
        alignment_mask: Máscara mostrando qué píxeles tienen alineación válida
    """
    depth_height, depth_width = 424, 512
    color_height, color_width = 1080, 1920

    # Preparar depth frame
    depth_frame_flat = depth_frame.flatten().astype(np.uint16)
    depth_ptr = depth_frame_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    # Mapear depth a color space
    color_space_points = (PyKinectV2._ColorSpacePoint * (depth_width * depth_height))()

    try:
        kinect._mapper.MapDepthFrameToColorSpace(
            depth_width * depth_height,
            depth_ptr,
            depth_width * depth_height,
            color_space_points
        )
    except Exception as e:
        print(f"Error mapping: {e}")
        return None, None

    # Crear imagen de salida
    colored_depth = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
    alignment_mask = np.zeros((depth_height, depth_width), dtype=np.uint8)

    for dy in range(depth_height):
        for dx in range(depth_width):
            idx = dy * depth_width + dx
            cp = color_space_points[idx]

            # Verificar si el mapeo es válido
            if np.isinf(cp.x) or np.isinf(cp.y):
                continue
            if cp.x < 0 or cp.y < 0:
                continue

            cx = int(cp.x)
            cy = int(cp.y)

            if 0 <= cx < color_width and 0 <= cy < color_height:
                # Obtener color del frame RGB
                colored_depth[dy, dx] = color_frame[cy, cx, :3]  # BGR
                alignment_mask[dy, dx] = 255

    return colored_depth, alignment_mask


def main():
    # Initialize Kinect
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color |
        PyKinectV2.FrameSourceTypes_Depth
    )

    print("Kinect Point Cloud Viewer")
    print("=========================")
    print("Muestra la alineación entre RGB y Depth")
    print("- Ventana 'Colored Point Cloud': Depth frame con colores del RGB")
    print("- Ventana 'Alignment Mask': Blanco = alineación válida, Negro = sin datos")
    print("- Ventana 'Depth': Visualización del depth frame")
    print("- Ventana 'Color': Frame RGB reducido")
    print("")
    print("Press 'q' to quit")

    last_color_frame = None
    last_depth_frame = None

    while True:
        # --- Depth Frame ---
        if kinect.has_new_depth_frame():
            last_depth_frame = kinect.get_last_depth_frame()
            last_depth_frame = last_depth_frame.reshape((424, 512))

        # --- Color Frame ---
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_frame = color_frame.reshape((1080, 1920, 4))
            last_color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        # --- Create Point Cloud ---
        if last_color_frame is not None and last_depth_frame is not None:
            colored_depth, alignment_mask = create_colored_point_cloud(
                kinect, last_color_frame, last_depth_frame
            )

            if colored_depth is not None:
                # Mostrar nube de puntos coloreada
                cv2.imshow('Colored Point Cloud', colored_depth)

                # Mostrar máscara de alineación
                cv2.imshow('Alignment Mask', alignment_mask)

                # Calcular estadísticas de alineación
                valid_pixels = np.count_nonzero(alignment_mask)
                total_pixels = alignment_mask.size
                valid_depth_pixels = np.count_nonzero(last_depth_frame > 0)

                # Mostrar depth con overlay de estadísticas
                depth_8bit = (last_depth_frame / 4500.0 * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                # Estadísticas
                stats = [
                    f"Depth pixels: {valid_depth_pixels}/{total_pixels}",
                    f"Aligned pixels: {valid_pixels}/{total_pixels}",
                    f"Alignment: {valid_pixels/total_pixels*100:.1f}%",
                ]
                for i, stat in enumerate(stats):
                    cv2.putText(depth_color, stat, (10, 25 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow('Depth', depth_color)

            # Mostrar color reducido
            color_small = cv2.resize(last_color_frame, (640, 360))
            cv2.imshow('Color', color_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kinect.close()
    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
