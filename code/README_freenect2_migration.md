# Migración de pykinect2 a freenect2

## Resumen de Cambios

Se ha migrado exitosamente el código de calibración de área de Kinect de `pykinect2` a `freenect2` para mejorar la compatibilidad y estabilidad.

## Archivos Modificados

### 1. `calibrate_area_kinect.py`
- **Cambios principales:**
  - Reemplazadas las importaciones de `pykinect2` por `freenect2`
  - Actualizada la inicialización del Kinect usando la API de freenect2
  - Modificadas las funciones `get_color_frame()` y `get_depth_frame()` para usar freenect2
  - Actualizado el manejo de recursos (start/stop/close)

### 2. `kinect_one.py`
- **Cambios principales:**
  - Completada la implementación usando freenect2
  - Agregadas todas las funciones necesarias para calibración y detección de toques
  - Implementada la misma funcionalidad que `calibrate_area_kinect.py` pero con freenect2

### 3. `requirements.txt`
- **Cambios:**
  - Removido: `pykinect2==0.1.0`
  - Agregado: `freenect2==0.2.0`

## Diferencias en la API

### pykinect2 (Anterior)
```python
from pykinect2 import PyKinectV2, PyKinectRuntime

# Inicialización
kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
)

# Obtener frames
if kinect.has_new_color_frame():
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))

if kinect.has_new_depth_frame():
    frame = kinect.get_last_depth_frame()
    frame = frame.reshape((424, 512))
```

### freenect2 (Nuevo)
```python
import freenect2

# Inicialización
kinect = freenect2.Freenect2()
kinect.set_color_stream(freenect2.ColorStream.RGB)
kinect.set_depth_stream(freenect2.DepthStream.DEFAULT)
kinect.start()

# Obtener frames
color_frame = kinect.get_color_frame()
depth_frame = kinect.get_depth_frame()

# Liberar recursos
kinect.stop()
kinect.close()
```

## Ventajas de freenect2

1. **Mejor compatibilidad multiplataforma**: Funciona en Windows, Linux y macOS
2. **API más moderna**: Interfaz más limpia y fácil de usar
3. **Mejor manejo de errores**: Excepciones más informativas
4. **Activo desarrollo**: Comunidad más activa y actualizaciones regulares
5. **Mejor documentación**: Documentación más completa y ejemplos

## Instalación

Para instalar las nuevas dependencias:

```bash
pip install freenect2==0.2.0
```

O actualizar todas las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

### Opción 1: Usar calibrate_area_kinect.py
```bash
python calibrate_area_kinect.py
```

### Opción 2: Usar kinect_one.py
```bash
python kinect_one.py
```

Ambos archivos proporcionan la misma funcionalidad:
- Calibración del área de trabajo
- Detección de toques en tiempo real
- Guardado de configuración en `config/ultima_configuracion_coordenadas.json`

## Notas Importantes

1. **Formato de frames**: freenect2 devuelve frames de profundidad en formato float32, que se convierten a uint16 para compatibilidad
2. **Resolución**: Los frames se redimensionan a 640x480 para mantener compatibilidad con el código existente
3. **Manejo de errores**: Se han agregado bloques try-catch para mejor manejo de errores
4. **Configuración**: La configuración se guarda en el mismo formato JSON que antes

## Solución de Problemas

### Error de inicialización
Si tienes problemas al inicializar el Kinect:
1. Verifica que el Kinect esté conectado correctamente
2. Asegúrate de que no haya otros programas usando el Kinect
3. Revisa que freenect2 esté instalado correctamente

### Frames no disponibles
Si no se obtienen frames:
1. Verifica que los streams estén configurados correctamente
2. Asegúrate de que el Kinect esté funcionando
3. Revisa los permisos de acceso al dispositivo

## Compatibilidad

El código migrado mantiene total compatibilidad con:
- Los archivos de configuración existentes
- El formato de datos de salida
- La interfaz de usuario y proyección
- Los algoritmos de detección de toques
