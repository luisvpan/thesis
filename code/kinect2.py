from openni import openni2
import cv2
import numpy as np
import json

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

    return proyeccion, (cx_izquierda, cy_izquierda), (cx_derecha, cy_derecha)

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

def calibrar_area_kinect_one(color_stream):
    view_width = 1280
    view_height = 800

    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        proyeccion, centroide_izquierda, centroide_derecha = proyectar_cuadrados(view_width, view_height)
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
                xw_min, xw_max = sorted([puntos_camara[1][0], puntos_camara[0][0]])
                yw_min, yw_max = sorted([puntos_camara[0][1], puntos_camara[1][1]])

                x1, y1 = centroide_izquierda
                x2, y2 = centroide_derecha

                xv_min, xv_max = sorted([x1, x2])
                yv_min, yv_max = sorted([y1, y2])

                # Guardar coordenadas en archivo JSON
                coordenadas = {
                    "xv_min": xv_min,
                    "xv_max": xv_max,
                    "yv_min": yv_min,
                    "yv_max": yv_max,
                    "xw_min": xw_min,
                    "xw_max": xw_max,
                    "yw_min": yw_min,
                    "yw_max": yw_max
                }
                with open("config/ultima_configuracion_coordenadas.json", "w") as file:
                    json.dump(coordenadas, file, indent=4)

                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)
                cv2.imshow("Proyeccion", proyeccion)
                cv2.rectangle(frame_bgr, (xw_min, yw_min), (xw_max, yw_max), (255, 0, 0), 2)
                cv2.imshow("Camara", frame_bgr)
                print("Calibración completada.")
                cv2.waitKey(5000)
                break
        else:
            cv2.imshow("Camara", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def detectar_toques(frame, area_calibrada):
    # Extrae el área de interés
    xw_min, xw_max = area_calibrada["xw_min"], area_calibrada["xw_max"]
    yw_min, yw_max = area_calibrada["yw_min"], area_calibrada["yw_max"]
    roi = frame[yw_min:yw_max, xw_min:xw_max]

    # Procesa la ROI para detectar toques (zonas blancas)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    toques = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Ajusta el área mínima para evitar ruido
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + xw_min
                cy = int(M['m01'] / M['m00']) + yw_min
                toques.append((cx, cy))
    return toques


# Inicializa OpenNI2 (ajusta la ruta si es necesario)
openni2.initialize("C:/Program Files/OpenNI2/Redist")
device = openni2.Device.open_any()

# Inicia el stream de color
color_stream = device.create_color_stream()
color_stream.start()

# Calibración antes de mostrar la cámara
calibrar_area_kinect_one(color_stream)

print("Presiona 'q' para salir.")

# Carga el área calibrada
with open("config/ultima_configuracion_coordenadas.json", "r") as file:
    area_calibrada = json.load(file)

while True:
    frame = color_stream.read_frame()
    frame_data = frame.get_buffer_as_uint8()
    frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.flip(frame_bgr, 1)

    # Detección de toques
    toques = detectar_toques(frame_bgr, area_calibrada)

    # Dibuja el área calibrada
    cv2.rectangle(
        frame_bgr,
        (area_calibrada["xw_min"], area_calibrada["yw_min"]),
        (area_calibrada["xw_max"], area_calibrada["yw_max"]),
        (255, 0, 0), 2
    )

    # Dibuja los toques detectados
    for (cx, cy) in toques:
        cv2.circle(frame_bgr, (cx, cy), 15, (0, 255, 0), 3)
        cv2.putText(frame_bgr, "TOQUE", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Kinect Color Camera", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

color_stream.stop()
openni2.unload()
cv2.destroyAllWindows()

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="Solo mostrar la cámara")
    args = parser.parse_args()

    # Inicializa OpenNI2 (ajusta la ruta si es necesario)
    openni2.initialize("C:/Program Files/OpenNI2/Redist")
    device = openni2.Device.open_any()

    # Inicia el stream de color
    color_stream = device.create_color_stream()
    color_stream.start()

    if args.camera:
        print("Modo cámara. Presiona 'q' para salir.")
        while True:
            frame = color_stream.read_frame()
            frame_data = frame.get_buffer_as_uint8()
            frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.flip(frame_bgr, 1)
            cv2.imshow("Kinect Color Camera", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
    else:
        # Calibración antes de mostrar la cámara
        calibrar_area_kinect_one(color_stream)

        print("Presiona 'q' para salir.")

        # Carga el área calibrada
        with open("config/ultima_configuracion_coordenadas.json", "r") as file:
            area_calibrada = json.load(file)

        while True:
            frame = color_stream.read_frame()
            frame_data = frame.get_buffer_as_uint8()
            frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.flip(frame_bgr, 1)

            # Detección de toques
            toques = detectar_toques(frame_bgr, area_calibrada)

            # Dibuja el área calibrada
            cv2.rectangle(
                frame_bgr,
                (area_calibrada["xw_min"], area_calibrada["yw_min"]),
                (area_calibrada["xw_max"], area_calibrada["yw_max"]),
                (255, 0, 0), 2
            )

            # Dibuja los toques detectados
            for (cx, cy) in toques:
                cv2.circle(frame_bgr, (cx, cy), 15, (0, 255, 0), 3)
                cv2.putText(frame_bgr, "TOQUE", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Kinect Color Camera", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
