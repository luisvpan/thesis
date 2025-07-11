import cv2
import numpy as np

def dibujar_resultado_conjuntos(screen, resultado, center_x, center_y, operacion):
    # Convertir el resultado a lista si es un set (para evitar problemas con elementos únicos)
    resultado = list(resultado) if isinstance(resultado, set) else resultado
    
    if not resultado:
        cv2.putText(screen, "Resultado: Conjunto vacio", (center_x - 100, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return

    # Mostrar figuras digitales para todas las operaciones
    y_offset = 0
    print("Elementos a dibujar:", resultado)
    for i, (shape, color) in enumerate(resultado):
        size = 30
        invoke_x = center_x
        invoke_y = center_y + y_offset

        # Mapear colores a BGR (formato de OpenCV)
        color_bgr = {
            "Rojo": (0, 0, 255),
            "Verde": (0, 255, 0),
            "Azul": (255, 0, 0),
            "Amarillo": (0, 255, 255),
            "Naranja": (0, 165, 255),
            "Morado": (128, 0, 128)
        }
        
        color_value = color_bgr.get(color, (255, 255, 255))  # Blanco por defecto

        # Dibujar la figura según su tipo
        if shape == "Círculo":
            cv2.circle(screen, (invoke_x, invoke_y), size, color_value, -1)
        elif shape == "Cuadrado":
            cv2.rectangle(screen, (invoke_x - size, invoke_y - size), 
                         (invoke_x + size, invoke_y + size), color_value, -1)
        elif shape == "Triángulo":
            pts = np.array([
                [invoke_x, invoke_y - size],
                [invoke_x - size, invoke_y + size],
                [invoke_x + size, invoke_y + size]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(screen, [pts], color_value)

        # Mostrar texto con la información de la figura
        texto = f"{shape} {color}"
        cv2.putText(screen, texto, (invoke_x - 40, invoke_y + size + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 60

    # Mostrar el número total de elementos
    cv2.putText(screen, f"Total: {len(resultado)} elementos", (center_x - 60, center_y + y_offset + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Crear una imagen negra de 800x600
    width, height = 800, 600
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Definir conjuntos de prueba
    conjunto_prueba = {("Círculo", "Azul"), ("Círculo", "Morado"), ("Cuadrado", "Verde"), ("Triángulo", "Rojo")}
    
    # Posición inicial para dibujar
    center_x, center_y = width // 2, 100
    
    # Dibujar los conjuntos
    dibujar_resultado_conjuntos(screen, conjunto_prueba, center_x, center_y, "union")
    
    # Mostrar el resultado
    cv2.imshow("Visualizador de Conjuntos", screen)
    
    # Esperar a que se presione una tecla y luego cerrar
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
