import cv2
from openni import openni2
import numpy as np
import time

# Configuración de la cámara
depth_camera_resolution = (512, 424) # px
depth_camera_fps = 30

# Inicializar OpenNI
openni2.initialize("C:/Program Files/OpenNI2/Redist")
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

cv2.namedWindow("Depth Contours", cv2.WINDOW_NORMAL)


while True:
	depth_frame = depth_stream.read_frame()
	depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
	depth_image = cv2.flip(depth_image, 1)

	# Mostrar valores de profundidad para depuración
	# print(f"Profundidad min: {np.min(depth_image)}, max: {np.max(depth_image)}, media: {np.mean(depth_image):.2f}", end='\r')

	# Normalizar y convertir para visualización
	# depth_vis = cv2.convertScaleAbs(depth_image, alpha=255.0/1000)
	# depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

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
	sobel_norm = cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	# Definir el rango de escala de grises (ajusta estos valores a tu necesidad)
	min_val = 50  # valor mínimo del rango
	max_val = 150 # valor máximo del rango

	# Crear la máscara: blanco (255) si está dentro del rango, negro (0) si no
	mask_range = cv2.inRange(sobel_norm, min_val, max_val)

	# Mostrar la máscara resultante
	cv2.imshow("Sobel Mask Range", mask_range)


	# --- Aplicar la máscara de contornos a la máscara de rango ---
	# Crear una máscara binaria de los contornos de Sobel
	mask_contours = np.zeros_like(mask_range)
	cv2.drawContours(mask_contours, contours_sobel, -1, 255, thickness=2)

	# AND entre la máscara de rango y la de contornos
	mask_final = cv2.bitwise_and(mask_range, mask_contours)

	# Mostrar la máscara combinada
	cv2.imshow("Sobel Mask Range + Contours", mask_final)

	# Umbral adaptativo para separar mejor los objetos
	thresh = cv2.adaptiveThreshold(mask_clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								   cv2.THRESH_BINARY, 11, 2)

	# Encontrar contornos en la imagen umbralizada
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Dibujar solo contornos que sean cuadrados (4 lados y convexos) en la máscara original
	output = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
	for cnt in contours:
		approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
		if len(approx) == 4 and cv2.isContourConvex(approx):
			cv2.drawContours(output, [approx], -1, (0,255,0), 2)

	cv2.imshow("Depth Contours", output)

	# --- NUEVO: aplicar el mismo proceso para la máscara de Sobel ---
	# Encontrar contornos en la máscara de Sobel
	contours_sobel, _ = cv2.findContours(edges_sobel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	output_sobel = cv2.cvtColor(edges_sobel, cv2.COLOR_GRAY2BGR)
	for cnt in contours_sobel:
		approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
		if len(approx) == 4 and cv2.isContourConvex(approx):
			cv2.drawContours(output_sobel, [approx], -1, (0,255,0), 2)

	cv2.imshow("Sobel Contours", output_sobel)
	key = cv2.waitKey()
	if key == 27:  # ESC para salir
		break

depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
