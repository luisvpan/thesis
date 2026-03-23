import cv2
from openni import openni2
import numpy as np

def show_rgb_camera():
    # Inicializar OpenNI2
    openni2.initialize("C:/Program Files/OpenNI2/Redist")  # Cambia esta ruta según tu instalación
    device = openni2.Device.open_any()

    # Crear un stream de color (RGB)
    rgb_stream = device.create_color_stream()
    rgb_stream.start()

    try:
        while True:
            # Leer un frame de la cámara RGB
            frame = rgb_stream.read_frame()
            frame_data = frame.get_buffer_as_uint8()
            image = np.frombuffer(frame_data, dtype=np.uint8).reshape(480, 640, 3)
            image = cv2.flip(image, 1)

            # Mostrar la imagen en una ventana
            cv2.imshow("Cámara RGB", image)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Detener el stream y limpiar
        rgb_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_rgb_camera()
