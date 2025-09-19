import cv2
from openni import openni2
import numpy as np

# Configuración de la cámara
depth_camera_resolution = (512, 424) # px
depth_camera_fps = 30

color_camera_resolution = (1080, 1920)  # Resolución de la cámara de color del Kinect v2, altura * ancho
video_beam_resolution = (1080, 1920) # Resolución del videobeam, altura * ancho
white = (255, 255, 255)


# Inicializar OpenNI
openni2.initialize("C:/Development Program Files/OpenNI2/Redist")
device = openni2.Device.open_any()

depth_stream = device.create_depth_stream()
if depth_stream is None:
	print("No depth stream found")
	exit(1)

depth_stream.set_video_mode(
	openni2.VideoMode(
		pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
		resolutionX=depth_camera_resolution[0],
		resolutionY=depth_camera_resolution[1],
		fps=depth_camera_fps
	)
)
depth_stream.start()


# Función para proyectar cuadrados de calibración
def proyectar_cuadrados(view_width, view_height):
    proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    cuadrado_size = 80
    margen = 0
    center = (view_height // 2, view_width // 2)
    py = 400
    px = 640
    # Cuadrado inferior izquierdo
    x_izquierda = center[1] - px - cuadrado_size // 2
    y_izquierda = center[0] + py - cuadrado_size // 2
    cv2.rectangle(proyeccion, (x_izquierda, y_izquierda),
                  (x_izquierda + cuadrado_size, y_izquierda + cuadrado_size), white, -1)

    cx_izquierda = x_izquierda + cuadrado_size // 2
    cy_izquierda = y_izquierda + cuadrado_size // 2

    # Cuadrado superior derecho
    x_derecha = center[1] + px - cuadrado_size // 2
    y_derecha = center[0] - py - cuadrado_size // 2
    cv2.rectangle(proyeccion, (x_derecha, y_derecha),
                  (x_derecha + cuadrado_size, y_derecha + cuadrado_size), white, -1)

    cx_derecha = x_derecha + cuadrado_size // 2
    cy_derecha = y_derecha + cuadrado_size // 2

    return proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, (cx_izquierda, cy_izquierda), (cx_derecha, cy_derecha)


def detectar_cuadrados_blancos(frame):
    print("Detectando cuadrados blancos con Sobel...")

    lower = np.percentile(frame, 2)
    upper = np.percentile(frame, 98)
    frame_clipped = np.clip(frame, lower, upper)
    mask_clean = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("Mask clean", mask_clean)

    # Opcional: limpiar la máscara
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    sobelx = cv2.Sobel(mask_clean, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask_clean, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = cv2.magnitude(sobelx, sobely)
    edges_sobel = np.uint8(np.clip(edges_sobel, 0, 255))

    min_val = 50  # valor mínimo del rango
    max_val = 100 # valor máximo del rango

    # Crear la máscara: blanco (255) si está dentro del rango, negro (0) si no
    mask_range = cv2.inRange(edges_sobel, min_val, max_val)
    mask_range = cv2.morphologyEx(mask_range, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))

    contours_sobel, _ = cv2.findContours(mask_range, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output_sobel = cv2.cvtColor(mask_range, cv2.COLOR_GRAY2BGR)
    cuadrados = []
    for cnt in contours_sobel:
        area = cv2.contourArea(cnt)
        if area >= 400 and area <= 800:
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                cv2.drawContours(output_sobel, [approx], -1, (0,255,0), 2)
                cuadrados.append((approx, area))
    # Mostrar la máscara combinada
    cv2.imshow("Sobel Mask Range + Contours", output_sobel)
    cuadrados = sorted(cuadrados, key=lambda x: x[1])
    return [cuadrado[0] for cuadrado in cuadrados]

def calibrate_area(device):

    view_width = video_beam_resolution[1]  # Ancho de la proyección (videobeam)
    view_height = video_beam_resolution[0]  # Alto de la proyección (videobeam)

    # Crear una ventana para la proyección
    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Proyeccion", 1920, 0)
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    depth_stream = device.create_depth_stream()
    depth_stream.start()
    depth_frame = depth_stream.read_frame()
    depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
    depth_image = cv2.flip(depth_image, 1)

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint8()
    frame_array = np.ndarray((frame.height, frame.width), dtype=np.uint8, buffer=frame_data)
    frame_array = cv2.flip(frame_array, 1)

    while True:
        
        proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, centroide_izquierda, centroide_derecha = proyectar_cuadrados(view_width, view_height)
        cv2.imshow("Proyeccion", proyeccion)
        cv2.waitKey(2000)
        # Mostrar valores de profundidad para depuración
        # print(f"Profundidad min: {np.min(depth_image)}, max: {np.max(depth_image)}, media: {np.mean(depth_image):.2f}", end='\r')ZZZ
        # Encontrar contornos en la máscara de Sobel
       
        lower = np.percentile(depth_image, 2)
        upper = np.percentile(depth_image, 98)
        depth_clipped = np.clip(depth_image, lower, upper)
        mask_clean = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("Mask clean", mask_clean)

        # Opcional: limpiar la máscara
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        sobelx = cv2.Sobel(mask_clean, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask_clean, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = cv2.magnitude(sobelx, sobely)
        edges_sobel = np.uint8(np.clip(edges_sobel, 0, 255))

        min_val = 50  # valor mínimo del rango
        max_val = 100 # valor máximo del rango

        # Crear la máscara: blanco (255) si está dentro del rango, negro (0) si no
        mask_range = cv2.inRange(edges_sobel, min_val, max_val)
        mask_range = cv2.morphologyEx(mask_range, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))

        contours_sobel, _ = cv2.findContours(mask_range, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        output_sobel = cv2.cvtColor(mask_range, cv2.COLOR_GRAY2BGR)
        for cnt in contours_sobel:
            area = cv2.contourArea(cnt)
            if area >= 400 and area <= 800:
                approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    cv2.drawContours(output_sobel, [approx], -1, (0,255,0), 2)
        # Mostrar la máscara combinada
        cv2.imshow("Sobel Mask Range + Contours", output_sobel)

        # Normalizar y convertir para visualización
        # depth_vis = cv2.convertScaleAbs(depth_image, alpha=255.0/1000)
        # depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cuadrados_detectados = detectar_cuadrados_blancos(frame_array)
        print(len(cuadrados_detectados), " cuadrados detectados.")
        if len(cuadrados_detectados) >= 2:
            puntos_camara = []
            for cuadrado in cuadrados_detectados[:2]:
                M = cv2.moments(cuadrado)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    puntos_camara.append([cx, cy])
                    cv2.drawContours(frame_array, [cuadrado], -1, (0, 255, 0), 2)
                    cv2.circle(frame_array, (cx, cy), 5, (0, 0, 255), -1)

            if len(puntos_camara) == 2:
                xw_min, xw_max = sorted([puntos_camara[1][0], puntos_camara[0][0]]) #vmin, vmax
                yw_min, yw_max = sorted([puntos_camara[0][1], puntos_camara[1][1]]) #umin, umax

                x1, y1 = centroide_izquierda
                x2, y2 = centroide_derecha

                xv_min, xv_max = sorted([x1, x2])
                yv_min, yv_max = sorted([y1, y2])

                xw_min = max(0, min(xw_min, depth_camera_resolution[1]))
                xw_max = max(0, min(xw_max, depth_camera_resolution[1]))
                yw_min = max(0, min(yw_min, depth_camera_resolution[0]))
                yw_max = max(0, min(yw_max, depth_camera_resolution[0]))
                print(f"Luego de max: {yw_max}")

                # Escalar las coordenadas de la ROI de profundidad
                factor_escala_x = 1
                factor_escala_y = 1

                xw_centro = (xw_min + xw_max) // 2
                yw_centro = (yw_min + yw_max) // 2

                # Aplicar escalado
                #xw_min = xw_min - 10
                xw_max_escalado = min(depth_camera_resolution[1], int(xw_centro + (xw_max - xw_centro) * factor_escala_x))
                yw_min_escalado = max(0, int(yw_centro - (yw_centro - yw_min) * factor_escala_y))

                # Dibujar el rectángulo de depuración sobre la proyección para verificar que pasa por los cuadrados
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

                # Mostrar el rectángulo en la proyección
                cv2.imshow("Proyeccion", proyeccion)

                cv2.rectangle(frame_array, (xw_min, yw_min_escalado), (xw_max_escalado, yw_max), (255, 0, 0), 2)
                cv2.imshow("Camara", frame_array)
                print("Calibración completada.")
                cv2.waitKey(5000)
                break

        # Definir profundidad mínima y máxima en milímetros (ajusta estos valores según la salida de depuración)
        max_depth = np.max(depth_image)  # ejemplo: 1500mm
        min_depth = np.min(depth_image)   # ejemplo: 500mm

        print("Profundidad máxima: ", max_depth)
        print("Profundidad minima: ", min_depth)
        print("Moda:", np.bincount(depth_image.flatten()).argmax())

        depth_frame = depth_stream.read_frame()
        depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
        depth_image = cv2.flip(depth_image, 1)
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=255.0/max_depth)

        # Crear máscara para el rango de profundidad
        # depth_image = cv2.threshold(depth_image, min_depth, -1, cv2.THRESH_TOZERO)[1]  # Eliminar valores cercanos a 0
        # depth_image = cv2.threshold(depth_image, max_depth, -1, cv2.THRESH_TOZERO_INV)[1]  # Eliminar valores cercanos a 0

        depth_min = np.min(depth_image)
        depth_max = np.max(depth_image)
        # Escalar depth_image a [0, 255] usando su rango real
        # depth_image = ((depth_image - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        # Mostrar la máscara para depuración
        # cv2.imshow("Depth Image", depth_image)

        # print("Profundidad maxima tras transformación: ", np.max(depth_image))
        # print("Profundidad minima tras transformación: ", np.min(depth_image))

        # Calcula los percentiles para ignorar outliers
        lower = np.percentile(depth_image, 2)
        upper = np.percentile(depth_image, 98)
        depth_clipped = np.clip(depth_image, lower, upper)
        mask_clean = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("Mask clean", mask_clean)

        # Opcional: limpiar la máscara
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # Detector de bordes con Sobel
        sobelx = cv2.Sobel(mask_clean, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask_clean, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = cv2.magnitude(sobelx, sobely)
        edges_sobel = np.uint8(np.clip(edges_sobel, 0, 255))
        cv2.imshow("Sobel Edges", edges_sobel)

        # --- MÁSCARA BASADA EN SOBEL Y RANGO DE ESCALA DE GRISES ---
        # Normalizar la imagen de Sobel a 0-255
        # sobel_norm = cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Definir el rango de escala de grises (ajusta estos valores a tu necesidad)
        min_val = 50  # valor mínimo del rango
        max_val = 100 # valor máximo del rango

        # Crear la máscara: blanco (255) si está dentro del rango, negro (0) si no
        mask_range = cv2.inRange(edges_sobel, min_val, max_val)
        mask_range = cv2.morphologyEx(mask_range, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))

        key = cv2.waitKey()
        if key == 27:  # ESC para salir
            break

    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    openni2.initialize("C:/Program Files/OpenNI2/Redist")  # Cambia esta ruta según tu instalación
    device = openni2.Device.open_any()

    calibrate_area(device)

    openni2.unload()
