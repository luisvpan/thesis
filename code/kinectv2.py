import numpy as np
import cv2
from openni import openni2

# Inicializa OpenNI2
openni2.initialize()
dev = openni2.Device.open_any()

# Streams de profundidad
depth_stream = dev.create_depth_stream()
depth_stream.start()

# Obtén el tamaño real del stream
video_mode = depth_stream.get_video_mode()
width = video_mode.resolutionX
height = video_mode.resolutionY
print(f"Resolución detectada: {width}x{height}")

# Parámetros de calibración
TOQUE_UMBRAL = 800  # Ajusta según la distancia de toque (en mm)
MIN_AREA = 30       # Área mínima para considerar un toque

# Proyección: cubre toda la pantalla
proyeccion = np.ones((height, width, 3), dtype=np.uint8) * 255  # Proyección blanca

# Configura las ventanas en modo pantalla completa
cv2.namedWindow("Proyeccion", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Toques detectados", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Toques detectados", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Captura frame de profundidad
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(frame_data, dtype=np.uint16).reshape((height, width))

    # Detección de toques: umbraliza la profundidad
    mask = (depth_img > 0) & (depth_img < TOQUE_UMBRAL)
    mask = mask.astype(np.uint8) * 255

    # Encuentra contornos de los toques
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pantalla negra para mostrar los toques
    pantalla = np.zeros((height, width, 3), dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(pantalla, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(pantalla, f"({cx},{cy})", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Muestra la proyección y la pantalla de toques en pantalla completa
    cv2.imshow("Proyeccion", proyeccion)
    cv2.imshow("Toques detectados", pantalla)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

# Libera recursos
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()