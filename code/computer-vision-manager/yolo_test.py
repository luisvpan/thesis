from ultralytics.models.yolo import YOLO
from openni import openni2
import numpy as np
import cv2
import time

color_camera_resolution = (1920, 1080)  # px
color_camera_fps = 60

openni2.initialize("C:/Development Program Files/OpenNI2/Redist")
device = openni2.Device.open_any()

color_stream = device.create_color_stream()
if color_stream is None:
	print("No color stream found")
	exit(1)

color_stream.set_video_mode(
	openni2.VideoMode(
		pixelFormat=openni2.PIXEL_FORMAT_RGB888,
		resolutionX=color_camera_resolution[0],
		resolutionY=color_camera_resolution[1],
		fps=color_camera_fps,
	)
)
color_stream.start()

color_stream.

# Load model once
model = YOLO("./runs/detect/train/weights/best.pt")

print("Iniciando predicciones cada 2 segundos. Ctrl+C para salir.")

# Crear la ventana una sola vez y permitir redimensionado si es necesario
cv2.namedWindow("YOLO Prediction", cv2.WINDOW_NORMAL)
# Escala de visualización (1.0 = tamaño real). Ajusta si la ventana se ve muy grande.
display_scale = 0.5
disp_w = int(color_camera_resolution[0] * display_scale)
disp_h = int(color_camera_resolution[1] * display_scale)
cv2.resizeWindow("YOLO Prediction", disp_w, disp_h)

interval = 0.0001  # segundos
next_time = time.time()

try:
	while True:
		now = time.time()
		if now >= next_time:
			# Capturar frame
			buf = color_stream.read_frame().get_buffer_as_uint8()
			# convertir buffer a numpy array (height, width, 3)
			frame_array = np.ndarray((color_camera_resolution[1], color_camera_resolution[0], 3), buffer=buf, dtype=np.uint8)
			
            # La imagen del kinect está en espejo
			frame_array = cv2.flip(frame_array, 1) 

			# Ejecutar predicción (ultralytics espera RGB)
			results = model.predict(frame_array, verbose=False)

			# Obtener imagen anotada (RGB) y convertir para OpenCV (BGR)
			annotated_rgb = results[0].plot()
			annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

			# Redimensionar para que quepa en la ventana de visualización
			if display_scale != 1.0:
				try:
					display_img = cv2.resize(annotated_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
				except cv2.error:
					# en caso de error (dimensiones inválidas), mostrar la imagen original
					display_img = annotated_bgr
			else:
				display_img = annotated_bgr

			# Mostrar la nueva imagen en la misma ventana (actualiza en-place)
			cv2.imshow("YOLO Prediction", display_img)
			# permitir que OpenCV procese eventos y refresque la ventana
			cv2.waitKey(1)

			ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
			print(f"Predicción mostrada: {ts}")

			# programar próxima ejecución exacta cada `interval` segundos
			next_time = time.time() + interval
		else:
			# procesar eventos de ventana y dormir un poco
			# si la ventana existe, permitimos que OpenCV procese eventos
			try:
				if cv2.getWindowProperty("YOLO Prediction", cv2.WND_PROP_VISIBLE) >= 1:
					cv2.waitKey(1)
			except cv2.error:
				# ventana no existe aún, ignora
				pass
			# dormir una fracción para no busy-loop
			time.sleep(min(0.1, max(0.001, next_time - now)))

except KeyboardInterrupt:
	print("\nInterrumpido por el usuario. Limpiando...")

finally:
	try:
		color_stream.stop()
	except Exception:
		pass
	openni2.unload()
	cv2.destroyAllWindows()
