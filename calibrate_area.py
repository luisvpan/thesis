import cv2
import numpy as np
from tqdm import tqdm
from openni import openni2
import PySimpleGUI as sg  # Para preguntar al usuario
import json

# Función para mostrar un mensaje en pantalla
def mostrar_mensaje(proyeccion, texto, xv_min, yv_min, xv_max, yv_max):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(texto, font, 2, 4)[0]  # Obtener el tamaño del texto
    text_x = xv_min + (xv_max - xv_min - text_size[0]) // 2  # Centrar horizontalmente
    text_y = yv_min + (yv_max - yv_min + text_size[1]) // 2  # Centrar verticalmente
    cv2.putText(proyeccion, texto, (text_x, text_y), font, 2, (255, 255, 255), 4, cv2.LINE_AA)

# Función para calcular dmax_map con barra de progreso y mensaje
def calculate_dmax(device, calibrated_area, xv_min, yv_min, xv_max, yv_max, num_frames=500):
    depth_stream = device.create_depth_stream()
    depth_stream.start()

    x, y, w, h = calibrated_area
    print(f"{w} * {h} = {w * h} ")

    # Definir un rango de profundidad para optimizar el uso de memoria
    min_depth = 500  # Ajusta según tu aplicación
    max_depth = 4000
    depth_accum = np.zeros((h, w, max_depth - min_depth + 1), dtype=int)

    # Crear una ventana de proyección para mostrar el mensaje
    proyeccion = np.zeros((800, 1280, 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibrando...", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1)

    for _ in tqdm(range(num_frames), desc="Numero de frames", unit="frames"):
        frame = depth_stream.read_frame()
        depth_data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[y:y+h, x:x+w]

        # Vectorizado para contar frecuencias
        valid_mask = (depth_roi >= min_depth) & (depth_roi <= max_depth)
        valid_depth = depth_roi[valid_mask] - min_depth
        indices = np.where(valid_mask)
        depth_accum[indices[0], indices[1], valid_depth] += 1

    depth_stream.stop()

    # Generar el mapa dmax basado en la moda de la profundidad
    dmax_map = np.argmax(depth_accum, axis=2) + min_depth

    np.savetxt("config/dmax_map.txt", dmax_map.flatten(), fmt="%d")
    
    # Mostrar mensaje de finalización
    cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0,0,0), -1)
    mostrar_mensaje(proyeccion, "Calibracion Completada", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1000)
    
    return dmax_map

# Función para proyectar cuadrados de calibración
def proyectar_cuadrados(view_width, view_height):
    proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    cuadrado_size = 50
    margen = 250

    # Cuadrado inferior izquierdo
    x_izquierda = margen
    y_izquierda = view_height - cuadrado_size 
    cv2.rectangle(proyeccion, (x_izquierda, y_izquierda),
                  (x_izquierda + cuadrado_size, y_izquierda + cuadrado_size), (255, 255, 255), -1)

    cx_izquierda = x_izquierda + cuadrado_size // 2
    cy_izquierda = y_izquierda + cuadrado_size // 2

    # Cuadrado superior derecho
    y_derecha = margen
    x_derecha = view_width - margen - cuadrado_size
    cv2.rectangle(proyeccion, (x_derecha, y_derecha),
                  (x_derecha + cuadrado_size, y_derecha + cuadrado_size), (255, 255, 255), -1)

    cx_derecha = x_derecha + cuadrado_size // 2
    cy_derecha = y_derecha + cuadrado_size // 2

    return proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, (cx_izquierda, cy_izquierda), (cx_derecha, cy_derecha)

# Detección de cuadrados en la imagen de la cámara
def detectar_cuadrados(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuadrados = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                cuadrados.append((approx, area))
    cuadrados = sorted(cuadrados, key=lambda x: x[1])
    return [cuadrado[0] for cuadrado in cuadrados]

# Función para calibrar la mesa y detectar toques
def calibrar_mesa_y_detectar_toques(device):
    view_width = 1280  # Ancho de la proyección (videobeam)
    view_height = 800  # Alto de la proyección (videobeam)

    # Crear una ventana para la proyección
    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Iniciar los streams de color y profundidad
    color_stream = device.create_color_stream()
    color_stream.start()
    depth_stream = device.create_depth_stream()
    depth_stream.start()

    # Leer el primer frame de la cámara de profundidad antes de usar depth_data
    depth_frame = depth_stream.read_frame()
    depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
    depth_data = cv2.flip(depth_data, 1)

    # Proceso de calibración
    while True:
        proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, centroide_izquierda, centroide_derecha = proyectar_cuadrados(view_width, view_height)
        cv2.imshow("Proyeccion", proyeccion)
        cv2.waitKey(2000)

        frame = color_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(frame_bgr, 1)
        cuadrados_detectados = detectar_cuadrados(frame_bgr)

        if len(cuadrados_detectados) >= 2:
            puntos_camara = []
            for cuadrado in cuadrados_detectados[:2]:
                M = cv2.moments(cuadrado)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    puntos_camara.append([cx, cy])
                    cv2.drawContours(frame_bgr, [cuadrado], -1, (0, 255, 0), 2)
                    cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

            if len(puntos_camara) == 2:
                xw_min, xw_max = sorted([puntos_camara[1][0], puntos_camara[0][0]]) #vmin, vmax
                yw_min, yw_max = sorted([puntos_camara[0][1], puntos_camara[1][1]]) #umin, umax

                x1, y1 = centroide_izquierda
                x2, y2 = centroide_derecha

                xv_min, xv_max = sorted([x1, x2])
                yv_min, yv_max = sorted([y1, y2])

                xw_min = max(0, min(xw_min, 640))
                xw_max = max(0, min(xw_max, 640))
                yw_min = max(0, min(yw_min, 480))
                yw_max = max(0, min(yw_max, 480))
                print(f"Luego de max: {yw_max}")

                # Escalar las coordenadas de la ROI de profundidad
                factor_escala_x = 1.16
                factor_escala_y = 1.12

                xw_centro = (xw_min + xw_max) // 2
                yw_centro = (yw_min + yw_max) // 2

                # Aplicar escalado
                xw_min = xw_min - 10
                xw_max_escalado = min(640, int(xw_centro + (xw_max - xw_centro) * factor_escala_x))
                yw_min_escalado = max(0, int(yw_centro - (yw_centro - yw_min) * factor_escala_y))

                # Extraer la ROI escalada
                depth_roi_escalado = depth_data[yw_min_escalado:yw_max, xw_min:xw_max_escalado]

                # Dibujar el rectángulo de depuración sobre la proyección para verificar que pasa por los cuadrados
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

                # Mostrar el rectángulo en la proyección
                cv2.imshow("Proyeccion", proyeccion)

                cv2.rectangle(frame_bgr, (xw_min, yw_min_escalado), (xw_max_escalado, yw_max), (255, 0, 0), 2)
                cv2.imshow("Camara", frame_bgr)
                print("Calibración completada.")
                cv2.waitKey(5000)
                break
        else:
            cv2.imshow("Camara", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dmax_map = calculate_dmax(device, (xw_min, yw_min_escalado, xw_max_escalado - xw_min, yw_max - yw_min_escalado),xv_min, yv_min, xv_max, yv_max)
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
            dmax_map = dmax_map - 4
            dmin_map = dmax_map - 10

            # Iniciar la detección de toques
            depth_stream.start()
            previous_roi = None
            touch_history = []
            vibration_threshold = 15  # Umbral para vibraciones
            touch_duration_threshold = 5  # Duración de toque requerida

            # Crear la proyección usando las dimensiones del área de trabajo
            proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)

            while True:
                # Leer frame de la cámara de profundidad
                depth_frame = depth_stream.read_frame()
                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                depth_data = cv2.flip(depth_data, 1)
                # Extraer la ROI escalada
                depth_roi = depth_data[yw_min_escalado:yw_max, xw_min:xw_max_escalado]

                frame_roi = frame_bgr[yw_min_escalado:yw_max, xw_min:xw_max_escalado]

                if previous_roi is not None:
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
                min_size = 100  # Tamaño mínimo para considerar un área como toque válido
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < min_size:
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
                        x_touch, y_touch = int(centroid[0]), int(centroid[1])

                        # Calcular los factores de escala
                        sx = float(xv_max - xv_min) / depth_roi.shape[1]
                        sy = float(yv_max - yv_min) / depth_roi.shape[0]

                        # Mapeo directo sin restar xw_min y yw_min
                        x_viewport = int(xv_min + (x_touch * sx))
                        y_viewport = int(yv_min + (y_touch * sy))

                        # Asegurar que las coordenadas estén dentro de los límites del viewport
                        x_viewport = np.clip(x_viewport, 0, view_width - 1)
                        y_viewport = np.clip(y_viewport, 0, view_height - 1)

                        # Dibujar en la ROI para visualización
                        cv2.circle(frame_roi, (x_touch, y_touch), 5, (255, 0, 0), -1)

                        # Dibujar en la proyección
                        cv2.circle(proyeccion, (x_viewport, y_viewport), 5, (0, 0, 255), -1)

                        # Opcional: Mostrar coordenadas mapeadas
                        cv2.putText(proyeccion, f"({x_viewport}, {y_viewport})", (x_viewport + 10, y_viewport + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # print(f"{xv_min}, {yv_min}")
                # print(f"{xv_max}, {yv_max}")
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

                # Mostrar la proyección y las máscaras
                full_mask = np.zeros_like(depth_data, dtype=np.uint8)
                full_mask[yw_min_escalado:yw_max, xw_min:xw_max_escalado] = touch_mask_final
                cv2.imshow("Proyeccion", proyeccion)
                cv2.imshow("Mascara de Toque", full_mask)
                cv2.imshow("Camara", frame_bgr)
                cv2.imshow("Camara2", frame_roi)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        print(f"Guardando coordenadas, yw_max: {yw_max}")
        coordenadas = {
            "xv_min": xv_min,
            "xv_max": xv_max,
            "yv_min": yv_min,
            "yv_max": yv_max,
            "xw_min": xw_min,
            "xw_max": xw_max_escalado,  # Usamos xw_max_escalado como xw_max
            "yw_min": yw_min_escalado,   # Usamos yw_min_escalado como yw_min
            "yw_max": yw_max
            }

         # Guardar las coordenadas en un archivo JSON en la carpeta config
        with open("config/ultima_configuracion_coordenadas.json", "w") as file:
            json.dump(coordenadas, file, indent=4)            
        # Detener los streams y destruir las ventanas           
        color_stream.stop()
        depth_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    openni2.initialize("C:/Program Files/OpenNI2/Redist")  # Cambia esta ruta según tu instalación
    device = openni2.Device.open_any()

    calibrar_mesa_y_detectar_toques(device)

    openni2.unload()
