# MagicboARd: Entorno de Realidad Aumentada Espacial para el Desarrollo de Juegos Sociales de Niños de Preescolar

MagicboARd es un entorno interactivo diseñado para fomentar habilidades cognitivas, motrices y sociales en niños de preescolar mediante juegos colaborativos y educativos, utilizando tecnología de realidad aumentada espacial. Este entorno combina hardware como Kinect y un videobeam con un software que permite la detección, proyección y manipulación de elementos interactivos.

---

## **Instrucciones para el Desarrollo**

### **1. Instalación de Kinect SDK para Windows**

1. **Descargar el SDK de Kinect**
   - Visita la página oficial de Microsoft Kinect y descarga la versión compatible con tu dispositivo ([Kinect SDK](https://www.microsoft.com/en-us/download/details.aspx?id=40278)).
   
2. **Instalación del SDK**
   - Ejecuta el archivo descargado y sigue las instrucciones del asistente de instalación.
   - Asegúrate de instalar todos los componentes necesarios, como los drivers del sensor Kinect.

3. **Conectar el Kinect**
   - Conecta el sensor Kinect al puerto USB de tu computadora.
   - Asegúrate de que el adaptador de corriente del dispositivo esté correctamente conectado.

4. **Verificar el Funcionamiento**
   - Abre la herramienta **Kinect Studio** incluida en el SDK.
   - Comprueba que el sensor Kinect detecte el entorno físico y registre los datos de la cámara RGB y de profundidad.

5. **Configurar el Entorno de Desarrollo**
   - Asegúrate de que tu entorno de desarrollo esté configurado para trabajar con Kinect. Puedes utilizar lenguajes como Python o C#, asegurándote de que las bibliotecas necesarias estén instaladas.

### **2. Instalación de Dependencias de Software**

- Instala las bibliotecas requeridas para el entorno interactivo:
  ```bash
  pip install -r requirements.txt
  ```

### **3. Configuración del Hardware**

- Configura el videobeam para que proyecte sobre la mesa interactiva.
- Ajusta la posición del Kinect en un soporte estable a una altura de aproximadamente 2.4 metros, asegurándote de que enfoque toda el área de trabajo.

### **4. Ejecución del Entorno**

1. Clona este repositorio:
   ```bash
   git clone https://github.com/anthxnyR/MagicboARd
   cd MagicboARd
   ```

2. Ejecuta el programa principal:
   ```bash
   python main.py
   ```

3. Sigue las instrucciones en pantalla para seleccionar los juegos y ajustar las configuraciones del entorno interactivo.

---

## **Contribuciones**
Si deseas contribuir al proyecto, abre un **Pull Request** o contacta al equipo a través de [barriosanthony49@gmail.com](mailto:barriosanthony49@gmail.com).

---

**¡Gracias por explorar MagicboARd!** 🎮✨

---
Anotaciones nuevas, acomodar luego:
- Usar Python 3.12.10