from openni import openni2
import numpy as np
import cv2
import os

# Inicializar OpenNI2
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")

# Abrir el dispositivo Kinect
dev = openni2.Device.open_any()

# Crear un flujo de color
color_stream = dev.create_color_stream()
color_stream.start()

while True:
    # Leer un frame de color
    frame = color_stream.read_frame()
    color_image = np.array(frame.get_buffer_as_uint8()).reshape((frame.height, frame.width, 3))

    # Procesar la imagen para detectar toques
    # Aquí puedes agregar tu lógica de detección de toques

    # Mostrar la imagen
    cv2.imshow('Kinect Color Stream', color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Detener el flujo y cerrar el dispositivo
color_stream.stop()
dev.close()
openni2.unload()
cv2.destroyAllWindows()
