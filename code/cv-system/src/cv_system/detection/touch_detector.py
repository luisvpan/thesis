import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time

class TouchDetector:
    def __init__(self, dmax_map: np.ndarray, config):
        self._dmax_map = dmax_map
        self.touch_threshold = getattr(config, "touch_threshold", 20)
        self.latest_result = None
        self.FINGER_TIPS = [4, 8, 12, 16, 20]
        
        # ROI - Ajusta estos para tu mesa
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = 400, 200, 1100, 700

        def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result = result

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback,
            num_hands=2,
            min_hand_detection_confidence=0.15,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, depth_frame: np.ndarray, rgb_frame: np.ndarray) -> list[tuple[int, int]]:
        rgb_h, rgb_w, _ = rgb_frame.shape
        depth_h, depth_w = depth_frame.shape
        
        # Enviar al detector (usamos un resize interno para bajar latencia si es necesario)
        roi_rgb = rgb_frame[self.roi_y : self.roi_y + self.roi_h, 
                            self.roi_x : self.roi_x + self.roi_w]
        roi_rgb_mp = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb_mp)
        
        timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, timestamp_ms)

        touches = []
        debug_img = rgb_frame.copy()

        # Dibujar ROI
        cv2.rectangle(debug_img, (self.roi_x, self.roi_y), 
                      (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (255, 255, 0), 1)

        if self.latest_result and self.latest_result.hand_landmarks:
            for hand_landmarks in self.latest_result.hand_landmarks:
                # Puntos verdes de tracking
                for lm in hand_landmarks:
                    gx = int(lm.x * self.roi_w + self.roi_x)
                    gy = int(lm.y * self.roi_h + self.roi_y)
                    cv2.circle(debug_img, (gx, gy), 2, (0, 255, 0), -1)

                for idx in self.FINGER_TIPS:
                    lm = hand_landmarks[idx]
                    gx_f = lm.x * self.roi_w + self.roi_x
                    gy_f = lm.y * self.roi_h + self.roi_y
                    
                    # Mapeo a espacio de profundidad
                    cx = int(gx_f * depth_w / rgb_w)
                    cy = int(gy_f * depth_h / rgb_h)

                    # ... dentro del bucle de puntas de dedos ...
                    if 0 <= cx < depth_w and 0 <= cy < depth_h:
                        # 1. Muestreo de área pequeña para promediar el ruido electrónico
                        # Un bloque de 3x3 píxeles en 512x424 es muy pequeño pero estable
                        roi_size = 1
                        z_roi = depth_frame[max(0, cy-roi_size):cy+roi_size+1, 
                                            max(0, cx-roi_size):cx+roi_size+1]
                        
                        # Filtrar ceros y obtener la mediana (mucho más estable que el valor directo)
                        valid_z = z_roi[z_roi > 0]
                        current_z = int(np.median(valid_z)) if valid_z.size > 0 else 0
                        
                        surface_z = int(self._dmax_map[cy, cx])
                        diff = surface_z - current_z
                        
                        # 2. Umbral con margen de ruido (Histéresis)
                        # Ajustamos a -10 para absorber fluctuaciones del sensor
                        is_touching = False
                        if -10 <= diff <= self.touch_threshold and current_z > 0:
                            is_touching = True
                            touches.append((cx, cy))
    
                        # --- DEBUG DE VALORES ---
                        color = (0, 0, 255) if is_touching else (0, 255, 255)
                        cv2.circle(debug_img, (int(gx_f), int(gy_f)), 4, color, -1)
                        
                        # Marcador visual de lectura nula
                        if current_z == 0:
                            debug_text = "NO DATA (Z=0)"
                            color = (0, 165, 255) # Naranja
                        else:
                            debug_text = f"Z:{current_z} M:{surface_z} D:{diff}"
                        
                        cv2.putText(debug_img, debug_text, (int(gx_f) + 10, int(gy_f)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
        cv2.namedWindow("Kinect V2 - Livestream AI Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("Kinect V2 - Livestream AI Debug", debug_img)
        cv2.waitKey(1)
        return touches
