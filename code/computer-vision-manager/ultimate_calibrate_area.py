import cv2
from openni import openni2
import numpy as np
from tqdm import tqdm
import PySimpleGUI as sg
import json
import math

# Configuración de la cámara
depth_camera_resolution = (512, 424) # px
depth_camera_fps = 30

color_camera_resolution = (1080, 1920)  # Resolución de la cámara de color del Kinect v2, altura * ancho
video_beam_resolution = (1080, 1920) # Resolución del videobeam, altura * ancho
white = (255, 255, 255)

# Función para mostrar un mensaje en pantalla
def mostrar_mensaje(proyeccion, texto, xv_min, yv_min, xv_max, yv_max):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(texto, font, 2, 4)[0]  # Obtener el tamaño del texto
    text_x = xv_min + (xv_max - xv_min - text_size[0]) // 2  # Centrar horizontalmente
    text_y = yv_min + (yv_max - yv_min + text_size[1]) // 2  # Centrar verticalmente
    cv2.putText(proyeccion, texto, (text_x, text_y), font, 2, (255, 255, 255), 4, cv2.LINE_AA)

# Función para calcular dmax_map con barra de progreso y mensaje
def calculate_dmax(depth_stream, calibrated_area, xv_min, yv_min, xv_max, yv_max, num_frames=500):
    x, y, w, h = calibrated_area
    print(f"{w} * {h} = {w * h}")

    # Definir un rango de profundidad para optimizar el uso de memoria
    min_depth = 650  # Ajusta según tu aplicación
    max_depth = 800

    depth_accum = np.zeros((h, w, max_depth - min_depth), dtype=np.uint16)
    print("Depth accum shape:", depth_accum.shape)

    # Crear una ventana de proyección para mostrar el mensaje
    proyeccion = np.zeros((depth_camera_resolution[0], depth_camera_resolution[1], 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibrando...", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1)

    for _ in tqdm(range(num_frames), desc="Numero de frames", unit="frames"):
        frame = depth_stream.read_frame()
        depth_frame = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
        depth_frame = cv2.flip(depth_frame, 1)
        depth_roi = depth_frame[y:y+h, x:x+w]

        # Vectorizado para contar frecuencias
        valid_mask = (depth_roi >= min_depth) & (depth_roi <= max_depth)
        valid_depth = depth_roi[valid_mask] - min_depth
        indices = np.where(valid_mask)
        depth_accum[indices[0], indices[1], valid_depth] += 1

    # Generar el mapa dmax basado en la moda de la profundidad
    dmax_map = np.argmax(depth_accum, axis=2) + min_depth

    np.savetxt("config/dmax_map.txt", dmax_map, fmt="%d")
    
    # Mostrar mensaje de finalización
    cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0,0,0), -1)
    mostrar_mensaje(proyeccion, "Calibracion Completada", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1000)
    
    return dmax_map

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
    y_izquierda = 100 + center[0] + py - cuadrado_size // 2
    cv2.rectangle(proyeccion, (x_izquierda, y_izquierda),
                  (x_izquierda + cuadrado_size, y_izquierda + cuadrado_size), white, -1)

    cx_izquierda = x_izquierda + cuadrado_size // 2
    cy_izquierda = y_izquierda + cuadrado_size // 2

    # Cuadrado superior derecho
    x_derecha = center[1] + px - cuadrado_size // 2
    y_derecha = 100 + center[0] - py - cuadrado_size // 2
    cv2.rectangle(proyeccion, (x_derecha, y_derecha),
                  (x_derecha + cuadrado_size, y_derecha + cuadrado_size), white, -1)

    cx_derecha = x_derecha + cuadrado_size // 2
    cy_derecha = y_derecha + cuadrado_size // 2

    return proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, (cx_izquierda, cy_izquierda), (cx_derecha, cy_derecha)


def detectar_cuadrados_blancos(depth_frame):
    print("Detectando cuadrados blancos con Sobel...")

    lower = np.percentile(depth_frame, 2)
    upper = np.percentile(depth_frame, 98)
    depth_clipped = np.clip(depth_frame, lower, upper)
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

    cv2.imshow("Sobel Mask Range", mask_range)

    contours_sobel, _ = cv2.findContours(mask_range, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output_sobel = cv2.cvtColor(mask_range, cv2.COLOR_GRAY2BGR)
    squares = []
    for cnt in contours_sobel:
        area = cv2.contourArea(cnt)
        if area >= 400:
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            print(approx)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                squares.append((approx, area))
                cv2.drawContours(output_sobel, [approx], -1, (0,255,0), 2)
    # Mostrar la máscara combinada
    cv2.imshow("Sobel Contours", output_sobel)
    print(len(squares))
    squares = sorted(squares, key=lambda x: x[1])
    print(squares)
    return [square[0] for square in squares]

def calibrate_area(device):

    view_width = video_beam_resolution[1]  # Ancho de la proyección (videobeam)
    view_height = video_beam_resolution[0]  # Alto de la proyección (videobeam)

    # Crear una ventana para la proyección
    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Proyeccion", 1920, 0)
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
    depth_frame = depth_stream.read_frame()
    depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
    depth_image = cv2.flip(depth_image, 1)
    frame_array = depth_image

    while True:
        
        proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, centroide_izquierda, centroide_derecha = proyectar_cuadrados(view_width, view_height)
        cv2.imshow("Proyeccion", proyeccion)
        cv2.waitKey(2000)
        # Mostrar valores de profundidad para depuración
        # print(f"Profundidad min: {np.min(depth_image)}, max: {np.max(depth_image)}, media: {np.mean(depth_image):.2f}", end='\r')ZZZ
        # Encontrar contornos en la máscara de Sobel
       
        # Normalizar y convertir para visualización
        # depth_vis = cv2.convertScaleAbs(depth_image, alpha=255.0/1000)
        # depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cuadrados_detectados = detectar_cuadrados_blancos(frame_array)
        print(len(cuadrados_detectados), " cuadrados detectados.")
        if len(cuadrados_detectados) >= 2:
            puntos_camara = []
            for i, cuadrado in enumerate(cuadrados_detectados[:2]):
                M = cv2.moments(cuadrado)
                print("---------")
                print(M)
                area = int(M['m00'])
                if area != 0:
                    cx = int(M['m10']) // area
                    cy = int(M['m01']) // area
                    print(f"Area: {area}")
                    print(f"Centroid: ({cx}, {cy})")
                    puntos_camara.append([cx, cy])
                    lower = np.percentile(frame_array, 2)
                    upper = np.percentile(frame_array, 98)
                    depth_clipped = np.clip(frame_array, lower, upper)
                    mask_clean = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_contour = np.zeros_like(mask_clean)
                    img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(img_contour, [cuadrado], -1, (0, 255, 0), 2)
                    cv2.circle(img_contour, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.imshow(f"Detected squares centroids {i + 1}", img_contour)

            if len(puntos_camara) == 2:
                print(puntos_camara)
                xw_min, xw_max = [puntos_camara[1][0], puntos_camara[0][0]] #vmin, vmax
                yw_min, yw_max = [puntos_camara[0][1], puntos_camara[1][1]] #umin, umax

                xv_min, yv_max = centroide_izquierda
                xv_max, yv_min = centroide_derecha

                print(f"xw_min: {xw_min}, xw_max: {xw_max}, yw_min: {yw_min}, yw_max: {yw_max}")
                print(f"xv_min: {xv_min}, xv_max: {xv_max}, yv_min: {yv_min}, yv_max: {yv_max}")

                # xv_min, xv_max = sorted([x1, x2])
                # yv_min, yv_max = sorted([y1, y2])
                # xv_min, yv_min = x1, y2
                # xv_max, yv_max = x2, y1

                sx = (xv_max - xv_min)/(xw_max - xw_min)
                sy = (yv_max - yv_min)/(yw_max - yw_min)

                def transform_depth_window_to_viewport(xw: int, yw: int) -> tuple[int, int]:
                    return int(xv_min + (xw - xw_min) * sx), int(yv_min + (yw - yw_min) * sy)

                xw_min_viewport_transformed, yw_min_viewport_transformed = transform_depth_window_to_viewport(xw_min, yw_min)
                xw_max_viewport_transformed, yw_max_viewport_transformed = transform_depth_window_to_viewport(xw_max, yw_max)

                # print(f"Transformed to viewport: ({xw_min_viewport_transformed}, {yw_min_viewport_transformed}) | ({xw_max_viewport_transformed}, {yw_max_viewport_transformed})")

                # xw_min = max(0, min(xw_min, depth_camera_resolution[1]))
                # xw_max = max(0, min(xw_max, depth_camera_resolution[1]))
                # yw_min = max(0, min(yw_min, depth_camera_resolution[0]))
                # yw_max = max(0, min(yw_max, depth_camera_resolution[0]))
                # print(f"Luego de max: {yw_max}")

                # Escalar las coordenadas de la ROI de profundidad
                # factor_escala_x = 1
                # factor_escala_y = 1

                # xw_centro = (xw_min + xw_max) // 2
                # yw_centro = (yw_min + yw_max) // 2
                # xw_centro = center[1]
                # yw_centro = 100 + center[0]

                # Aplicar escalado
                #xw_min = xw_min - 10
                # xw_max_escalado = min(depth_camera_resolution[1], int(xw_centro + (xw_max - xw_centro) * factor_escala_x))
                # yw_min_escalado = max(0, int(yw_centro - (yw_centro - yw_min) * factor_escala_y))

                # Dibujar el rectángulo de depuración sobre la proyección para verificar que pasa por los cuadrados
                cv2.rectangle(proyeccion, (xw_min_viewport_transformed, yw_min_viewport_transformed), (xw_max_viewport_transformed, yw_max_viewport_transformed), (0, 255, 0), 2)

                # Mostrar el rectángulo en la proyección
                cv2.imshow("Proyeccion", proyeccion)

                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame_array, (xw_min, yw_min), (xw_max, yw_max), (255, 0, 0), 2)
                cv2.imshow("Camara", frame_array)
                print("Calibración completada.")
                break

    calibrated_area = (xw_min, yw_min, xw_max - xw_min, yw_max - yw_min)
    print("Calibrated area =>", calibrated_area)
    dmax_map = calculate_dmax(depth_stream, calibrated_area, xv_min, yv_min, xv_max, yv_max)

    # Detección de toques
    if 'xw_min' in locals():
        print("Iniciando detección de toques...")

        # Preguntar al usuario si desea proceder con la detección de toques
        layout = [[sg.Text('¿Desea proceder con la detección de toques?')],
                  [sg.Button('Sí'), sg.Button('No')]]
        window = sg.Window('Verificación de Detección', layout)
        event, values = window.read()
        window.close()
        if event == 'Sí':
            # Calcular dmax_map
            dmax_map = dmax_map - 2
            dmin_map = dmax_map - 10
            # Usa los mismos límites que en area_calibrada
            x, y, w, h = calibrated_area
            print(x, y, w, h)
            # Iniciar la detección de toques
            previous_roi = None
            touch_history = []
            vibration_threshold = 15  # Umbral para vibraciones
            touch_duration_threshold = 1  # Duración de toque requerida

            # Crear la proyección usando las dimensiones del área de trabajo
            proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)

            while True:
                # Leer frame de la cámara de profundidad
                depth_frame = depth_stream.read_frame()
                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
                depth_data = cv2.flip(depth_data, 1)
                # Extraer la ROI escalada
                # print("depth data =>", depth_data.shape)
                # print("dmax map =>", dmax_map.shape)
                depth_roi = depth_data[y:y+h, x:x+w]

                if previous_roi is not None:
                    # print("Detectando vibraciones...")
                    # Detectar vibraciones y ajustar el ROI
                    roi_diff = cv2.absdiff(depth_roi, previous_roi)
                    vibration_mask = cv2.threshold(roi_diff, vibration_threshold, 255, cv2.THRESH_BINARY)[1]
                    vibration_mask = cv2.medianBlur(vibration_mask, ksize=5)

                    depth_roi[vibration_mask > 0] = previous_roi[vibration_mask > 0]

                previous_roi = depth_roi.copy()

                # Crear la "coraza" entre dmin y dmax
                touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

                # Aplicar filtros para eliminar ruido
                touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
                touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)

                touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))

                # Aplicar umbral para consolidar áreas de toque
                _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 150, 255, cv2.THRESH_BINARY)

                # Aplicar apertura morfológica para eliminar ruido pequeño
                kernel = np.ones((3, 3), np.uint8)
                touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)

                # Identificar componentes conectados
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask_final, connectivity=8)
                min_size = 5  # Tamaño mínimo para considerar un área como toque válido
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] <= min_size:
                        touch_mask_final[labels == i] = 0

                touch_history.append(touch_mask_final)

                # Acumular las máscaras para identificar toques persistentes
                accumulated_mask = np.sum(touch_history, axis=0)
                accumulated_mask = np.clip(accumulated_mask, 0, 255).astype(np.uint8)

                # Considerar solo toques que persisten durante varios cuadros
                _, final_touch_mask = cv2.threshold(accumulated_mask, touch_duration_threshold * 255, 255, cv2.THRESH_BINARY)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask_final, connectivity=8)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_size:
                        centroid = centroids[i]
                        x_touch, y_touch = centroid[0] + xw_min, centroid[1] + yw_min

                        # Calcular los factores de escala
                        # sx = float(xv_max - xv_min) / depth_roi.shape[1]
                        # sy = float(yv_max - yv_min) / depth_roi.shape[0]

                        # Mapeo directo sin restar xw_min y yw_min
                        # x_viewport = int(xv_min + (x_touch * sx))
                        # y_viewport = int(yv_min + (y_touch * sy))

                        # Asegurar que las coordenadas estén dentro de los límites del viewport
                        # x_viewport = np.clip(x_viewport, 0, view_width - 1)
                        # y_viewport = np.clip(y_viewport, 0, view_height - 1)
                        print("X touch:", x_touch, "Y touch:", y_touch)
                        x_touch_viewport, y_touch_viewport = transform_depth_window_to_viewport(x_touch, y_touch)
                        print("X viewport:", x_touch_viewport, "Y viewport:", y_touch_viewport)

                        # Dibujar en la ROI para visualización
                        # cv2.circle(frame_roi, (x_touch, y_touch), 5, (255, 0, 0), -1)

                        # Dibujar en la proyección
                        cv2.circle(proyeccion, (x_touch_viewport, y_touch_viewport), 5, (0, 0, 255), -1)

                        # Opcional: Mostrar coordenadas mapeadas
                        cv2.putText(proyeccion, f"({x_touch_viewport}, {y_touch_viewport})", (x_touch_viewport + 10, y_touch_viewport + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # print(f"{xv_min}, {yv_min}")
                # print(f"{xv_max}, {yv_max}")
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

                # Mostrar la proyección y las máscaras
                full_mask = np.zeros_like(depth_data, dtype=np.uint8)
                # full_mask[yw_min_escalado:yw_max, xw_min:xw_max_escalado] = touch_mask_final
                cv2.imshow("Proyeccion", proyeccion)
                cv2.imshow("Mascara de Toque", touch_mask_final)
                # cv2.imshow("Camara", frame_bgr)
                # cv2.imshow("Camara2", frame_roi)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        print(f"Guardando coordenadas, yw_max: {yw_max}")
        coordenadas = {
            "xv_min": xv_min,
            "xv_max": xv_max,
            "yv_min": yv_min,
            "yv_max": yv_max,
            "xw_min": xw_min,
            "xw_max": xw_max_viewport_transformed,  # Usamos xw_max_escalado como xw_max
            "yw_min": yw_min_viewport_transformed,   # Usamos yw_min_escalado como yw_min
            "yw_max": yw_max
            }

         # Guardar las coordenadas en un archivo JSON en la carpeta config
        with open("config/ultima_configuracion_coordenadas.json", "w") as file:
            json.dump(coordenadas, file, indent=4)            
        # Detener los streams y destruir las ventanas           
    

    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    openni2.initialize("C:/Development Program Files/OpenNI2/Redist")  # Cambia esta ruta según tu instalación
    device = openni2.Device.open_any()

    calibrate_area(device)

    openni2.unload()