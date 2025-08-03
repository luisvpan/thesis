# Modificar la función dibujar_resultado_conjuntos para aceptar un segundo resultado
def dibujar_resultado_conjuntos(screen, resultado, operacion, resultado2=None):
    result_1_x = rect2_x + square_width + 20 + operator_rectangle_width + 20
    result_1_y = rect1_y
    right_margin = view_width - 600
    center_y = result_1_y

    # Nombre de la operación
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
        start_x = right_margin + (view_width - right_margin - len(resultado) * spacing) // 2
        
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
            if shape == "Círculo":
                cv2.circle(screen, (pos_x, pos_y), size, color_bgr, -1)
            elif shape == "Cuadrado":
                cv2.rectangle(screen, (pos_x - size, pos_y - size), 
                              (pos_x + size, pos_y + size), color_bgr, -1)
            elif shape == "Triángulo":
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
        result_2_x = result_1_x + square_width + 20  # Espacio entre resultados
        cv2.putText(screen, "RESULTADO INTERSECCION:", 
                    (result_2_x, result_1_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)

        # Dibujar segunda área rectangular (resultado 2)
        cv2.rectangle(screen, (result_2_x, result_1_y), 
                      (result_2_x + square_width, result_1_y + square_height), (250, 250, 250), 2)

        # Dibujar figuras resultantes del resultado 2
        if not resultado2:
            cv2.putText(screen, "CONJUNTO VACIO", 
                        (result_2_x + 50 , result_1_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        else:
            resultado2 = list(resultado2)
            size = 25
            spacing = 80
            start_x = right_margin + (view_width - right_margin - len(resultado2) * spacing) // 2
            
            for i, (shape, color) in enumerate(resultado2):
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
                if shape == "Círculo":
                    cv2.circle(screen, (pos_x, pos_y), size, color_bgr, -1)
                elif shape == "Cuadrado":
                    cv2.rectangle(screen, (pos_x - size, pos_y - size), 
                                  (pos_x + size, pos_y + size), color_bgr, -1)
                elif shape == "Triángulo":
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

# En el bucle principal, después de calcular el resultado de conjuntos
set1 = set(area1_figuras)
set2 = set(area2_figuras)
resultado_conjuntos = realizar_operacion_conjuntos(set1, set2, operacion_actual)

# Calcular la intersección con el área 3
set3 = set(area3_figuras)
resultado_interseccion = realizar_operacion_conjuntos(resultado_conjuntos, set3, 'interseccion')

# Dibujar resultados
dibujar_resultado_conjuntos(videobeam_screen, resultado_conjuntos, operacion_actual, resultado_interseccion)
