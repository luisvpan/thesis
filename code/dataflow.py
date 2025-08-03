import numpy as np
import cv2
import os
import json
import math
import threading
from openni import openni2

def detect_color_and_shape(image, min_contour_area=250):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "Rojo": ([170, 50, 50], [180, 255, 255]),
        "Verde": ([40, 50, 50], [90, 255, 255]),
        "Azul": ([90, 50, 50], [130, 255, 255]),
        "Amarillo": ([20, 100, 100], [30, 255, 255]),
        "Naranja": ([0, 100, 100], [10, 255, 255]),
        "Morado": ([130, 50, 50], [160, 255, 255]),
    }

    detected_shapes = []

    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) > 4:
                    shape = "Circulo"
                elif len(approx) == 4:
                    shape = "Cuadrado"
                elif len(approx) == 3:
                    shape = "Triangulo"
                else:
                    shape = "Desconocido"
                detected_shapes.append((shape, color_name, cnt))
    return detected_shapes

def realizar_operacion_conjuntos(set1, set2, operacion):
    if operacion == 'union':
        return set1.union(set2)
    elif operacion == 'interseccion':
        return set1.intersection(set2)
    elif operacion == 'diferencia':
        return set1.difference(set2)
    elif operacion == 'diferencia_simetrica':
        return set1.symmetric_difference(set2)
    else:
        return set()

def juego_dataflow(device):
    # Cargar las coordenadas desde el archivo JSON
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        coordenadas = json.load(file)

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Dimensiones del área de trabajo
    work_area_width = xw_max - xw_min
    work_area_height = yw_max - yw_min

    # Tamaño de la pantalla del videobeam
    view_width = 1280
    view_height = 800
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

    # Crear tres áreas rectangulares verticales a la izquierda
    square_height = 150  # Altura de cada área rectangular
    square_width = square_height * 2  # Ancho es el doble de la altura
    left_margin = (view_width // 4)  # Margen izquierdo centrado

    operator_rectangle_height = (square_height * 2) + 50
    operator_rectangle_width = 100

    total_vertical_space = 3 * square_height + 2 * 50  # 50px de espacio entre áreas
    start_y = ((view_height - total_vertical_space) // 2) + 200 

    # Posiciones de las áreas (verticales a la izquierda)
    rect1_x = left_margin
    rect1_y = start_y

    rect2_x = left_margin
    rect2_y = start_y + square_height + 50  # 50px debajo del primer área

    rect3_x = left_margin + square_width + 20
    rect3_y = start_y

    rect4_x = left_margin + square_width + 20 + 100 + 20
    rect4_y = start_y + square_height + 50

    # Función para detectar figuras y colores (igual que antes)
    def detect_color_and_shape(image, min_contour_area=250):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            "Rojo": ([170, 50, 50], [180, 255, 255]),
            "Verde": ([40, 50, 50], [90, 255, 255]),
            "Azul": ([90, 50, 50], [130, 255, 255]),
            "Amarillo": ([20, 100, 100], [30, 255, 255]),
            "Naranja": ([0, 100, 100], [10, 255, 255]),
            "Morado": ([130, 50, 50], [160, 255, 255]),
        }

        detected_shapes = []

        for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) > 4:
                        shape = "Circulo"
                    elif len(approx) == 4:
                        shape = "Cuadrado"
                    elif len(approx) == 3:
                        shape = "Triangulo"
                    else:
                        shape = "Desconocido"

                    detected_shapes.append((shape, color_name, cnt))

        return detected_shapes

    # Función para verificar si una figura está dentro de un área rectangular
    def figura_en_area_rectangular(figura_centro, rect_x, rect_y, rect_width, rect_height):
        fx, fy = figura_centro
        if (rect_x <= fx <= rect_x + rect_width and 
            rect_y <= fy <= rect_y + rect_height):
            return True
        return False

    # Función para dibujar las áreas rectangulares verticales
    def dibujar_areas_rectangulares(screen):
        # Dibujar primera área rectangular (verde)
        cv2.rectangle(screen, (rect1_x, rect1_y), 
                     (rect1_x + square_width, rect1_y + square_height), (0, 255, 0), 3)
        cv2.putText(screen, "Entrada 1", (rect1_x, rect1_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Dibujar segunda área rectangular (verde)
        cv2.rectangle(screen, (rect2_x, rect2_y), 
                     (rect2_x + square_width, rect2_y + square_height), (0, 255, 0), 3)
        cv2.putText(screen, "Entrada 2", (rect2_x, rect2_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Dibujar tercera área rectangular (verde)
        cv2.rectangle(screen, (rect3_x, rect3_y), 
                     (rect3_x + operator_rectangle_width, rect3_y + operator_rectangle_height), (121, 210, 230), 3)
        cv2.putText(screen, "Operador", (rect3_x, rect3_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (121, 210, 230), 2)
        
        # Dibujar segunda área rectangular (verde)
        cv2.rectangle(screen, (rect4_x, rect4_y), 
                     (rect4_x + square_width, rect4_y + square_height), (0, 255, 0), 3)
        cv2.putText(screen, "Entrada 3", (rect4_x, rect4_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Función para mostrar figuras detectadas
    def mostrar_figura_detectada_multiple(screen, shape, color, area_x, area_y, rect_width, rect_height, area_name, y_offset=0):
        text_y = area_y + rect_height + 30 + y_offset
        text_x = area_x + rect_width // 2
        texto = f"{area_name}: {shape} {color}"
        cv2.putText(screen, texto, (text_x - 80, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Función para realizar operaciones de conjuntos (igual que antes)
    def realizar_operacion_conjuntos(set1, set2, operacion):
        if operacion == 'union':
            return set1.union(set2)
        elif operacion == 'interseccion':
            return set1.intersection(set2)
        elif operacion == 'diferencia':
            return set1.difference(set2)
        elif operacion == 'diferencia_simetrica':
            return set1.symmetric_difference(set2)
        else:
            return set()

    # Función para dibujar el resultado a la derecha
    def dibujar_resultado_conjuntos(screen, resultado, operacion, resultado2=None):
        result_1_x = rect2_x + square_width + 20 + operator_rectangle_width + 20
        result_1_y = rect1_y
        right_margin = view_width - 600
        center_y = result_1_y

        # Nombre de la operación del primer resultado
        nombres_operaciones = {
            'union': 'UNION',
            'interseccion': 'INTERSECCION', 
            'diferencia': 'DIFERENCIA',
            'diferencia_simetrica': 'DIF. SIMETRICA'
        }
        nombre_operacion = nombres_operaciones.get(operacion, operacion)
        cv2.putText(screen, f"RESULTADO ({nombre_operacion}):", 
                    (result_1_x, result_1_y  - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)

        # Dibujar primera área rectangular (resultado 1)
        cv2.rectangle(screen, (result_1_x, result_1_y), 
                    (result_1_x + square_width, result_1_y + square_height), (250, 250, 250), 2)

        # Dibujar figuras resultantes del resultado 1
        if not resultado:
            cv2.putText(screen, "CONJUNTO VACIO", 
                        (result_1_x + 50 , result_1_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        else:
            resultado = list(resultado)
            size = 25
            spacing = 80
            start_x = result_1_x + (square_width - len(resultado) * spacing) // 2 + 20
            
            for i, (shape, color) in enumerate(resultado):
                pos_x = start_x + i * spacing
                pos_y = rect1_y + 50

                color_bgr = {
                    "Rojo": (0, 0, 255),
                    "Verde": (0, 255, 0),
                    "Azul": (255, 0, 0),
                    "Amarillo": (0, 255, 255),
                    "Naranja": (0, 165, 255),
                    "Morado": (128, 0, 128) 
                }.get(color, (255, 255, 255))

                # Dibujar la figura en el área de resultados
                if shape == "Circulo":
                    cv2.circle(screen, (pos_x, pos_y), size, color_bgr, -1)
                elif shape == "Cuadrado":
                    cv2.rectangle(screen, (pos_x - size, pos_y - size), 
                                (pos_x + size, pos_y + size), color_bgr, -1)
                elif shape == "Triangulo":
                    pts = np.array([
                        [pos_x, pos_y - size],
                        [pos_x - size, pos_y + size],
                        [pos_x + size, pos_y + size]
                    ], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(screen, [pts], color_bgr)

                # Etiqueta con el nombre
                cv2.putText(screen, f"{shape[:3]}.{color[:3]}", 
                            (pos_x - 30, pos_y + size + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Si hay un segundo resultado, dibujarlo a la derecha
        if resultado2 is not None:
            result_2_x = result_1_x + square_width + 20
            result_2_width = 150
            result_2_height = operator_rectangle_height
            cv2.putText(screen, "INTERSECCION", 
                        (result_2_x , result_1_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)

            # Dibujar segunda área rectangular (resultado 2)
            cv2.rectangle(screen, (result_2_x, result_1_y), 
                        (result_2_x + result_2_width, result_1_y + result_2_height), (250, 250, 250), 2)

            # Dibujar figuras resultantes del resultado 2
            if not resultado2:
                # Calcular la posición para centrar el texto "VACIO"
                text_to_show = "VACIO"
                (text_width, text_height), baseline = cv2.getTextSize(text_to_show, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = result_2_x + (result_2_width - text_width) // 2
                text_y = result_1_y + (result_2_height + text_height) // 2
                
                cv2.putText(screen, text_to_show, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            else:
                resultado2 = list(resultado2)
                size = 25
                spacing = 80
                
                start_y = result_1_y + (result_2_height - len(resultado2) * spacing) // 2 + 20
                center_x = result_2_x + result_2_width // 2

                for i, (shape, color) in enumerate(resultado2):
                    pos_x = center_x
                    pos_y = start_y + i * spacing

                    color_bgr = {
                        "Rojo": (0, 0, 255),
                        "Verde": (0, 255, 0),
                        "Azul": (255, 0, 0),
                        "Amarillo": (0, 255, 255),
                        "Naranja": (0, 165, 255),
                        "Morado": (128, 0, 128) 
                    }.get(color, (255, 255, 255))

                    # Dibujar la figura en el área de resultados
                    if shape == "Circulo":
                        cv2.circle(screen, (pos_x, pos_y), size, color_bgr, -1)
                    elif shape == "Cuadrado":
                        cv2.rectangle(screen, (pos_x - size, pos_y - size), 
                                    (pos_x + size, pos_y + size), color_bgr, -1)
                    elif shape == "Triangulo":
                        pts = np.array([
                            [pos_x, pos_y - size],
                            [pos_x - size, pos_y + size],
                            [pos_x + size, pos_y + size]
                        ], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(screen, [pts], color_bgr)

                    # Etiqueta con el nombre
                    cv2.putText(screen, f"{shape[:3]}.{color[:3]}", 
                                (pos_x - 30, pos_y + size + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


    # Iniciar los streams de la cámara
    rgb_stream = device.create_color_stream()
    rgb_stream.start()

    # Variables para operaciones de conjuntos
    operacion_actual = 'interseccion'
    operaciones_disponibles = ['union', 'interseccion', 'diferencia', 'diferencia_simetrica']
    indice_operacion = 0

    while True:
        frame = rgb_stream.read_frame()
        if frame is None:
            continue

        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)

        # Recortar el área de trabajo
        area_trabajo = bgr_data[yw_min:yw_max, xw_min:xw_max]

        # Detectar figuras en el área de trabajo
        figuras_detectadas = detect_color_and_shape(area_trabajo)

        # Limpiar la pantalla
        videobeam_screen.fill(0)

        # Dibujar las áreas rectangulares verticales
        dibujar_areas_rectangulares(videobeam_screen)

        # Procesar figuras detectadas
        area1_figuras = []
        area2_figuras = []
        area3_figuras = []
        area4_figuras = []

        for shape, color, contour in figuras_detectadas:
            # Calcular el centro de la figura
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + xw_min
                cy = int(M['m01'] / M['m00']) + yw_min

                # Mapear coordenadas
                sx = float(xv_max - xv_min) / (xw_max - xw_min)
                sy = float(yv_max - yv_min) / (yw_max - yw_min)
                fx = int(xv_min + (cx - xw_min) * sx)
                fy = int(yv_min + (cy - yw_min) * sy)

                # Verificar presencia en áreas
                if figura_en_area_rectangular((fx, fy), rect1_x, rect1_y, square_width, square_height):
                    figura = (shape, color)
                    if figura not in area1_figuras:
                        area1_figuras.append(figura)

                if figura_en_area_rectangular((fx, fy), rect2_x, rect2_y, square_width, square_height):
                    figura = (shape, color)
                    if figura not in area2_figuras:
                        area2_figuras.append(figura)

                # Verificar presencia en el rectángulo 3
                if figura_en_area_rectangular((fx, fy), rect3_x, rect3_y, operator_rectangle_width, operator_rectangle_height):
                    figura = (shape, color)
                    if figura not in area3_figuras:
                        area3_figuras.append(figura)

                if figura_en_area_rectangular((fx, fy), rect4_x, rect4_y, square_width, square_height):
                    figura = (shape, color)
                    if figura not in area4_figuras:
                        area4_figuras.append(figura)

        # Mostrar detecciones por área
        for i, (shape, color) in enumerate(area1_figuras):
            mostrar_figura_detectada_multiple(videobeam_screen, shape, color, 
                                               rect1_x, rect1_y, square_width, square_height, "E1", i * 30)

        for i, (shape, color) in enumerate(area2_figuras):
            mostrar_figura_detectada_multiple(videobeam_screen, shape, color, 
                                               rect2_x, rect2_y, square_width, square_height, "E2", i * 30)

        # Mostrar detecciones en el rectángulo 3
        for i, (shape, color) in enumerate(area3_figuras):
            mostrar_figura_detectada_multiple(videobeam_screen, shape, color, rect3_x, rect3_y, operator_rectangle_width, operator_rectangle_height, "Op1", i * 30)

        for i, (shape, color) in enumerate(area4_figuras):
            mostrar_figura_detectada_multiple(videobeam_screen, shape, color, rect4_x, rect4_y, square_width, square_height, "E3", i * 30)

        # Cambiar la operación actual según la figura detectada en el área 3
        for shape, color in area3_figuras:
            if shape == "Triangulo" and color == "Amarillo":
                operacion_actual = 'union'
            elif shape == "Circulo" and color == "Azul":
                operacion_actual = 'interseccion'
            elif shape == "Circulo" and color == "Rojo":
                operacion_actual = 'diferencia'
            elif shape == "Cuadrado" and color == "Morado":
                operacion_actual = 'diferencia_simetrica'

        # Realizar y mostrar operación de conjuntos
        set1 = set(area1_figuras)
        set2 = set(area2_figuras)
        
        resultado_conjuntos = realizar_operacion_conjuntos(set1, set2, operacion_actual)
        
        set3 = set(area4_figuras)
        resultado_interseccion = realizar_operacion_conjuntos(resultado_conjuntos, set3, 'interseccion')

        dibujar_resultado_conjuntos(videobeam_screen, resultado_conjuntos, operacion_actual, resultado_interseccion)

        # Mostrar información en la parte inferior (opcional)
        info_y = view_height - 50
        cv2.putText(videobeam_screen, "Presiona 'o' para cambiar operacion | 'q' para salir", 
                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Mostrar ventanas
        cv2.namedWindow("Dataflow", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("Dataflow", 1920, 0)
        cv2.setWindowProperty("Dataflow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Dataflow", videobeam_screen)
        cv2.imshow("Detección", area_trabajo)

        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            indice_operacion = (indice_operacion + 1) % len(operaciones_disponibles)
            operacion_actual = operaciones_disponibles[indice_operacion]

    rgb_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    openni2.initialize("C:/Program Files/OpenNI2/Redist")  # Cambia esta ruta según tu instalación
    device = openni2.Device.open_any()
    juego_dataflow(device)