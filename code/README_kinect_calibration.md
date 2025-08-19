# Calibración de Área para Kinect v2

Este código permite calibrar un área de trabajo para detección de toques usando una Kinect v2 de Xbox One.

## Requisitos

### Hardware
- Kinect v2 de Xbox One
- Adaptador USB 3.0 para Kinect v2
- Proyector o pantalla para visualización

### Software
- Windows 10/11
- Python 3.7+
- Kinect SDK 2.0
- Visual Studio 2015/2017/2019 (para compilar pykinect2)

## Instalación

1. **Instalar Kinect SDK 2.0**
   - Descargar desde: https://www.microsoft.com/en-us/download/details.aspx?id=44561
   - Instalar el SDK completo

2. **Instalar dependencias de Python**
   ```bash
   pip install -r requirements.txt
   ```

3. **Instalar pykinect2**
   ```bash
   pip install pykinect2
   ```
   
   **Nota**: Si hay problemas con la instalación de pykinect2, puede ser necesario compilarlo desde el código fuente:
   ```bash
   git clone https://github.com/Kinect/PyKinect2.git
   cd PyKinect2
   python setup.py install
   ```

## Uso

### Ejecutar la calibración

```bash
python calibrate_area_kinect.py
```

### Proceso de calibración

1. **Preparación**:
   - Conectar la Kinect v2 al puerto USB 3.0
   - Asegurar que la Kinect esté posicionada correctamente sobre el área de trabajo
   - Ejecutar el script

2. **Calibración automática**:
   - El programa proyectará dos cuadrados blancos en las esquinas
   - Colocar objetos blancos (papel, cartulina) en las posiciones de los cuadrados
   - El sistema detectará automáticamente los cuadrados y calculará el área de trabajo

3. **Cálculo de profundidad**:
   - El sistema capturará 500 frames para calcular el mapa de profundidad
   - Se mostrará una barra de progreso durante este proceso

4. **Detección de toques** (opcional):
   - El programa preguntará si deseas proceder con la detección de toques
   - Si aceptas, podrás probar la detección en tiempo real
   - Presiona 'q' para salir

### Archivos generados

- `config/dmax_map.txt`: Mapa de profundidad máxima
- `config/ultima_configuracion_coordenadas.json`: Coordenadas del área calibrada

## Configuración

### Parámetros ajustables

En el código puedes modificar:

- `view_width` y `view_height`: Dimensiones de la proyección
- `num_frames`: Número de frames para calcular el mapa de profundidad (default: 500)
- `min_depth` y `max_depth`: Rango de profundidad válido
- `vibration_threshold`: Umbral para detectar vibraciones
- `touch_duration_threshold`: Duración mínima de toque
- `min_size`: Tamaño mínimo para considerar un toque válido

### Factores de escala

Los factores de escala para la ROI de profundidad están configurados como:
- `factor_escala_x = 1.16`
- `factor_escala_y = 1.12`

Estos valores pueden necesitar ajuste según tu configuración específica.

## Solución de problemas

### Error de inicialización de Kinect
- Verificar que la Kinect esté conectada a un puerto USB 3.0
- Asegurar que el Kinect SDK 2.0 esté instalado correctamente
- Reiniciar el sistema si es necesario

### No se detectan los cuadrados
- Verificar la iluminación del área
- Asegurar que los objetos sean blancos y de buen contraste
- Ajustar el umbral en la función `detectar_cuadrados` si es necesario

### Problemas de rendimiento
- Reducir el número de frames en `num_frames`
- Ajustar el tamaño de la ROI
- Verificar que no haya otros procesos usando la Kinect

## Diferencias con Kinect v1

Este código está específicamente adaptado para Kinect v2:

- **Resolución de color**: 1920x1080 (vs 640x480 en v1)
- **Resolución de profundidad**: 512x424 (vs 640x480 en v1)
- **Biblioteca**: pykinect2 (vs freenect para v1)
- **Mejor precisión**: La Kinect v2 tiene mejor precisión de profundidad

## Notas técnicas

- El código redimensiona las imágenes de la Kinect v2 a 640x480 para mantener compatibilidad
- Se aplican filtros para reducir ruido en la detección de toques
- El sistema usa una "coraza" de profundidad para detectar toques
- Se implementa detección de vibraciones para mejorar la estabilidad
