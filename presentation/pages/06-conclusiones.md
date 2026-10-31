---
eyebrow: 06 · Conclusiones
---

# Cada objetivo específico dejó un resultado que sostiene su conclusión

| Objetivo | Conclusión |
|---|---|
| OE1 Analizar | Ningún referente cubre la brecha |
| OE2 Diseñar | Bruner se volvió el mazo de piezas |
| OE3 Construir | Los prototipos fueron determinantes; opera de extremo a extremo |
| OE4 Validar | Viable con el docente como guía |
| OE5 Documentar | Un manual por destinatario |

<!--
PRESENTA: TBD · TIEMPO: 75s

Concluimos por objetivo.

Del análisis: la combinación de programación tangible y realidad aumentada espacial es una vía viable para fomentar el pensamiento computacional entre los 6 y los 9 años con un uso moderado de pantallas. Ninguno de los entornos de programación comparados reúne manipulación tangible, colaboración simultánea, retroalimentación sin monitor y piezas reconocibles sin texto, y el único que combina mesa, proyección y colaboración, Magicboard, no es un entorno de programación. Esa brecha define el espacio del ambiente.

Del diseño: la progresión concreto-pictórico-abstracto operacionaliza las formas de representación de Bruner. El marco teórico no quedó como fundamento abstracto, sino que se volvió una decisión verificable: la organización del mazo de piezas. Y el lenguaje, al no requerir texto, evita los errores de escritura; los que persisten se detectan y se comunican sin interrumpir la construcción.

De la construcción: el enfoque por prototipos fue determinante, porque cada prototipo definió el requerimiento del siguiente. El objetivo se cumple en tanto el artefacto existe y opera de extremo a extremo, aunque las mejoras de calibración y de detección no se cuantificaron formalmente.

De la validación: el ambiente es viable como recurso usado por docentes y niños en conjunto, con el docente como guía, y no como herramienta autónoma para los niños. Quedan tres condiciones de adopción: retroalimentación más completa, detección de cartas más fiable y material de apoyo para el docente.

De la documentación: separar los manuales por destinatario responde a los dos papeles que el ambiente requiere, el de quien lo instala y el del docente que conduce la actividad.
-->

---
eyebrow: 06 · Conclusiones · Objetivo general
---

# Se cumple el objetivo general; el efecto en el pensamiento computacional de los niños queda por medir

<div class="grid grid-cols-[1fr_auto] gap-8 mt-4 items-center">
  <ul>
    <li><span class="yes">✓</span> Ambiente desarrollado y en operación</li>
    <li><span class="yes">✓</span> Piezas CPA, mesa compartida, visión y flujo de datos</li>
    <li><span class="part">◐</span> Efecto en niños: hipótesis por confirmar</li>
    <li><span class="ods-tag">ODS 4</span> metas 4.4 y 4.6</li>
  </ul>
  <img src="/final-environment.jpeg" alt="Ambiente en uso sobre la mesa" style="height: 300px" />
</div>

<!--
PRESENTA: TBD · TIEMPO: 45s

En conjunto, se cumple el objetivo general de desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, y respondemos a la pregunta de investigación: un ambiente así se puede desarrollar combinando piezas físicas que siguen la progresión concreto-pictórico-abstracto, una superficie compartida donde se proyectan la interfaz y el resultado, un subsistema de visión que reconoce piezas y toques, y un lenguaje de flujo de datos cuyo programa coincide con la disposición de las piezas y se evalúa mientras se construye.

El ambiente reúne las condiciones que el marco teórico asocia al pensamiento computacional y al aprendizaje colaborativo: acceso simultáneo de varios niños, un programa visible y persistente sobre la mesa, y retroalimentación sin una pantalla individual. Con eso contribuye al ODS 4, metas 4.4 y 4.6. Pero somos claros: su efecto sobre el pensamiento computacional y sobre la colaboración no se midió en este trabajo; es una hipótesis que hay que confirmar con niños.
-->

---
eyebrow: 06 · Recomendaciones
---

# Las recomendaciones apuntan a medir, detectar mejor y probar con niños

<div class="grid grid-cols-3 gap-3 mt-6">
  <div class="box"><h3>Medir</h3><p>mAP50, mAP50-95, latencia, error de calibración</p></div>
  <div class="box"><h3>Reentrenar</h3><p>Mazo completo, conjunto de datos versionado</p></div>
  <div class="box"><h3>Probar con niños</h3><p>Usabilidad y comprensión, guiados por el docente</p></div>
  <div class="box plain"><h3>Guía docente</h3><p>Actividades modelo alineadas al currículo</p></div>
  <div class="box plain"><h3>Sonido</h3><p>Señales de reconocimiento, advertencia y error</p></div>
  <div class="box plain"><h3>Lenguaje</h3><p>Estudiar repetición y condicionales</p></div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 50s

Recomendamos, primero, documentar cuantitativamente el desempeño: mediciones reproducibles de latencia, error de calibración y precisión de la detección, con mAP50, mAP50-95 y la matriz de confusión, para comparar los modelos evaluados, YOLO11n, YOLO11s y YOLO26n. Segundo, reentrenar el modelo con el lote completo de piezas, incluidas tapas, paletas y cubos, y versionar el conjunto de datos y su configuración; lo refuerzan los fallos que observó el experto en IHC. Tercero, y más importante, ampliar la validación: pruebas de usabilidad y de comprensión con niños de 6 a 9 años, guiados por el docente, y luego un estudio más largo que compruebe el efecto en el pensamiento computacional y la colaboración.

Además: una guía de actividades modelo alineada al currículo y a los principios de la programación tangible colaborativa, como complemento del manual de usuario; completar las señales sonoras y valorar la música relajante; y estudiar cómo incorporar la repetición y la decisión condicional, dentro del flujo de datos o como un modo complementario. En lo técnico, mantener la especificación del lenguaje sincronizada con el informe y documentar con una decisión de arquitectura el reemplazo del servidor anterior por el relevo actual.
-->

---
layout: center
class: text-center
deckEnd: true
---

# Gracias

<p class="text-xl mt-4">Eduardo Arzolay · Luis Vásquez</p>

<p class="cite mt-2">El ambiente está montado para una demostración en vivo</p>

<!--
PRESENTA: TBD · TIEMPO: 10s
PREGUNTA: ¿Agregamos correos de contacto o el enlace al repositorio? No están en el informe.

Muchas gracias por su atención. El ambiente está montado aquí mismo, por si desean verlo funcionar en vivo. Quedamos atentos a sus preguntas.
-->
