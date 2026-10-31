---
eyebrow: 05 · Demostración
---

# En el video, una actividad completa va de la calibración al resultado leído en voz alta

<script setup>
// Bound at runtime so the build does not fail while public/demo.mp4 (possibly git-ignored) is missing.
const demoSrc = "/demo.mp4"
</script>

<video controls preload="metadata" poster="/final-environment.jpeg" :src="demoSrc" class="mx-auto mt-1" style="height: 345px; background: #f3f4f6">
  Video de la demostración (public/demo.mp4)
</video>

<!--
PRESENTA: TBD · TIEMPO: 420s
PREGUNTA: El video aún no existe. Debe guardarse como public/demo.mp4 (H.264 + AAC, 1920×1080 o 1280×720) y durar como máximo 6:30, para dejar ~30 s de introducción y transición dentro de los 7 min. Si pesa demasiado para versionarlo, recordar copiarlo a mano (CLAUDE.md, «Generated, git-ignored assets»). El póster provisional es final-environment.jpeg; conviene cambiarlo por un fotograma del propio video.
PREGUNTA: ¿Quién aparece manipulando las piezas? El trabajo no incluyó pruebas con niños; si aparecen niños hacen falta autorizaciones de sus representantes. La alternativa es que nosotros (o un docente) hagamos de docente y de niños, y decirlo así.
PREGUNTA: Configuración de la grabación. En el código, la detección de toques viene apagada por defecto (CV_TOUCH_DETECTOR=0) y, al activarla, el tipo por defecto es «mediapipe», no el híbrido DIRECT + MediaPipe que describe el informe; el acceso al sensor por defecto es OpenNI2 (HARDWARE_MANAGER), no PyKinect2. Para que el video muestre lo que dice el informe, grabar con el detector híbrido y PyKinect2 activados. ¿Confirmamos esa configuración?
PREGUNTA: ¿Mostramos una falla de detección en el video? El experto en IHC la observó y el jurado puede preguntar; mostrar cómo se corrige (reacomodar la carta) puede ser mejor que ocultarla.

Introducción en vivo (unos 15 s, antes de darle play): «Ahora vamos a ver el ambiente funcionando. El video sigue una actividad como la que haría un docente con sus niños, en los tres niveles: concreto, pictórico y abstracto. El sistema real está montado aquí, por si quieren verlo en vivo después.»

GUION DEL VIDEO (máx. 6:30; rótulos breves en pantalla, sin narración obligatoria; nosotros comentamos encima en vivo):

1. 0:00–0:20 · Plano general. Mesa en forma de riñón, proyector y Kinect v2 en posición cenital, computador al lado. Rótulo: «Actividad guiada por el docente». Se ve que no hay monitor frente a los niños.

2. 0:20–0:50 · Calibración. Se proyecta la rejilla de nueve puntos y se calibra (homografía). Luego un toque de prueba: el punto proyectado cae justo bajo el dedo. Rótulo: «Calibración: 9 puntos».

3. 0:50–1:50 · Nivel concreto (como la figura cpa-concrete). Colocar una carta de apertura, unos cubos y una carta de cierre; enlazarlas con un toque y mostrar que el grupo se forma. Hacer un segundo grupo. Colocar la carta de resta y la carta de salida. Conectar tocando un puerto y luego el otro: se ven las conexiones y los walkers llevando el dato. La carta de salida muestra el resultado; tocar el botón de voz y que se escuche el resultado.

4. 1:50–2:10 · Conexión rechazada. Intentar conectar un criterio a un puerto que espera números (o a un puerto de entrada ya ocupado): el puerto se agita y la conexión no se crea.

5. 2:10–3:20 · Nivel pictórico (como cpa-pictorial). Un grupo de cartas de alimentos de varios colores; carta de filtrar con una carta de criterio de color. Activar «Mostrar resultados» para ver el resultado intermedio del filtro. Agregar la carta de contar a la salida del filtro y escuchar el resultado con concordancia (p. ej., «3 manzanas rojas»).

6. 3:20–4:20 · Nivel abstracto (como cpa-abstract). Dos cartas de dígitos contiguas forman un número de dos cifras; sumarlo con otro número y restar. Luego, ordenar las cartas 7, 2 y 5 con una carta de ordenar (de menor a mayor): se ordenan, no se suman. Cambiar una carta de dígito y mostrar que el resultado se actualiza al instante (evaluación incremental).

7. 4:20–5:10 · Errores locales (como soft-failure). Con una suma y una resta en la mesa, quitar una carta de la resta: la suma conserva su resultado y la resta muestra el error en su carta de salida, nombrando la carta que falta y qué hacer; el lienzo señala esa carta. Volver a ponerla y el error desaparece.

8. 5:10–5:40 · Modos de visualización. El docente alterna los resultados entre concreto, pictórico y abstracto: cambia la apariencia, no el valor.

9. 5:40–6:10 · Colaboración. Dos o tres pares de manos colocan y conectan piezas a la vez sobre la misma superficie, mientras el docente orienta.

10. 6:10–6:30 · Cierre. Plano general del programa completo con su resultado proyectado. Rótulo final: «ERAE · UCAB Guayana».

Mientras corre el video, quien presenta comenta cada paso con una frase, apoyándose en lo explicado en las láminas de diseño (puertos, grupos, walkers, errores locales).
-->
