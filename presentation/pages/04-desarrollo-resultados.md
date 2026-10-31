---
eyebrow: 04 · Desarrollo y resultados · OE1
---

# Ningún referente de programación es a la vez tangible, colaborativo, sin monitor y sin texto

| Referente | Tangible | Colabora&shy;ción | Sin monitor | Sin texto | Lenguaje |
|---|:-:|:-:|:-:|:-:|---|
| TORTIS | <span class="yes">✓</span> | <span class="no">✗</span> | <span class="part">◐</span> | <span class="yes">✓</span> | Imperativo |
| AlgoBlock | <span class="yes">✓</span> | <span class="yes">✓</span> | <span class="no">✗</span> | — | Imperativo |
| Scratch y ScratchJr | <span class="no">✗</span> | <span class="part">◐</span> | <span class="no">✗</span> | <span class="part">◐</span> | Imperativo |
| Magicboard | <span class="yes">✓</span> | <span class="yes">✓</span> | <span class="yes">✓</span> | — | No programa |
| Rojas y Youssef | <span class="yes">✓</span> | <span class="no">✗</span> | <span class="yes">✓</span> | — | Imperativo |
| **Ambiente propuesto** | <span class="yes">✓</span> | <span class="yes">✓</span> | <span class="yes">✓</span> | <span class="yes">✓</span> | **Flujo de datos** |

<p class="cite mt-2">✓ sí · ◐ parcial · ✗ no · — no documentado</p>

<!--
PRESENTA: TBD · TIEMPO: 70s
PREGUNTA: Los símbolos son nuestra lectura de la tabla de referentes del informe (Tabla «Comparación de los referentes…»). Hay que confirmarlos uno por uno: TORTIS «sin monitor» = parcial (la salida es la tortuga, pero un monitor mostraba los comandos de las cajas); Scratch «colaboración» = parcial (comunidad en línea, pero se construye frente a la pantalla); «sin texto» = «—» donde el informe no lo dice (AlgoBlock, Magicboard, Rojas y Youssef). ¿De acuerdo?

El primer objetivo fue analizar el uso de la programación tangible en entornos de realidad aumentada espacial para caracterizar el ambiente. Por revisión documental comparamos los referentes principales: TORTIS, de Radia Perlman; AlgoBlock; Scratch y ScratchJr; Magicboard; y el sistema de Rojas y Youssef. Los criterios salieron del problema y de las bases teóricas: la interfaz y su dependencia de una pantalla, la colaboración, la edad y si exige lectoescritura, el paradigma, y la retroalimentación.

El hallazgo es una brecha. AlgoBlock es el único referente de programación que hace de la colaboración un principio de diseño, pero muestra el resultado en una pantalla. Scratch y ScratchJr son en pantalla y Scratch exige saber leer. TORTIS no requiere lectura pero no se diseñó para colaborar. Y el único referente que combina mesa, proyección y colaboración, Magicboard, no es un entorno de programación. Ninguno reúne a la vez manipulación tangible, colaboración simultánea, retroalimentación sin monitor y piezas reconocibles sin texto escrito. Ese es el espacio que ocupa nuestro ambiente.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE1
---

# Todos los referentes son imperativos; elegimos flujo de datos para hacer visible la ejecución

<div class="grid grid-cols-2 gap-6 mt-5">
  <div class="box plain">
    <h3>Imperativo</h3>
    <p>Secuencia de comandos</p>
    <p>Ejecución oculta: «máquina nocional» <span class="cite">(du Boulay, 1986)</span></p>
    <p>TORTIS: programa confundido con el dibujo</p>
  </div>
  <div class="box">
    <h3>Flujo de datos</h3>
    <p>Red de nodos sobre la mesa</p>
    <p>Dato visible en cada conexión</p>
    <p>Sin estado mutable <span class="cite">(Wadge y Ashcroft, 1985)</span></p>
  </div>
</div>

<div v-click class="mt-6 text-center">
<strong>Hipótesis de diseño:</strong> ver fluir los datos es menos abstracto que seguir un puntero
</div>

<!--
PRESENTA: TBD · TIEMPO: 60s

El segundo hallazgo del análisis es que todos los referentes de programación son imperativos: el niño arma una secuencia de comandos y observa su efecto al ejecutarla. TORTIS ya mostró el riesgo en edades tempranas: con las cajas de botones, los niños confundían el programa con el dibujo resultante, y con las cartas les costaba entender que cada una era un comando (Perlman, 1976, citado en Morgado et al., 2006). Du Boulay identifica la comprensión de la «máquina nocional», lo que ocurre dentro de la computadora al ejecutar, incluido el flujo de control, como una de las principales dificultades de quienes aprenden a programar.

Por eso elegimos el paradigma de flujo de datos: la ejecución depende de la disponibilidad de los datos, no de un orden de instrucciones. Una red de nodos se corresponde con naturalidad con piezas dispuestas sobre la mesa, cada conexión puede mostrar el dato que circula y, al no haber estado mutable, cada bloque produce siempre lo mismo con las mismas entradas. Tomamos Lucid, de Wadge y Ashcroft, como referente del modelo de ejecución.

[clic] Lo decimos con cuidado: es una hipótesis de diseño. Presumimos que, para un niño en transición a las operaciones concretas, ver cómo fluyen los datos por una red física es menos abstracto que seguir un puntero de ejecución. No la medimos en este trabajo.

De este análisis salieron seis características (interfaz tangible de mesa, retroalimentación proyectada, lenguaje de flujo de datos, diseño para la colaboración, compatibilidad con los 6 a 9 años y reducción del tiempo en pantalla) y nueve requerimientos, cinco funcionales y cuatro no funcionales, que retomaremos en la validación. Están en la lámina de respaldo R-02.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE2
---

# Tres subsistemas cierran un ciclo continuo: de las piezas sobre la mesa a la proyección

<div class="grid grid-cols-[auto_1fr] gap-8 mt-2 items-center">
  <img src="/c4-paisaje.svg" alt="Paisaje C4 del ambiente: docente y niños, mesa, proyector y Kinect, y los tres subsistemas lógicos en el computador" style="height: 345px" />
  <div class="grid gap-4">
    <div class="box"><h3>1 · Visión por computador</h3><p>Reconoce piezas y toques</p></div>
    <div class="box"><h3>2 · Interfaz</h3><p>Dibuja el grafo sobre la mesa</p></div>
    <div class="box"><h3>3 · Intérprete ERAE</h3><p>Reevalúa solo lo que cambió</p></div>
  </div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 55s
PREGUNTA: El SVG c4-paisaje es alto y estrecho (759 × 888), pensado para la página del informe; en la lámina sus etiquetas quedan muy pequeñas para un proyector. CLAUDE.md prevé hacer un candidato reducido de Archify en ../docs/architecture/sources/ y exportarlo con tools/export-svg.mjs, pero eso está fuera de presentation/ y no lo toqué. ¿Lo hacemos?

El segundo objetivo fue diseñar el ambiente. Físicamente es una interfaz tangible de mesa: un tablero en forma de riñón con un área de proyección de 70 por 110 centímetros, con el proyector y un Kinect v2 en posición cenital. El Kinect capta imágenes de color, de profundidad e infrarrojas. Lo importante es que el espacio donde el niño actúa y el espacio donde ve el resultado son el mismo: es el acoplamiento perceptual de Ishii.

Esta figura es el paisaje del sistema en notación C4, la misma del informe. A nivel lógico hay tres subsistemas. El de visión por computador reconoce las piezas y los toques sobre la superficie. La interfaz representa lo reconocido como un grafo sobre un lienzo proyectado y lo traduce a la representación textual del lenguaje. Y el intérprete del lenguaje ERAE evalúa el programa. El ciclo es continuo: cada vez que cambia algo sobre la mesa, la interfaz actualiza el grafo, el intérprete reevalúa solo lo que cambió y la proyección muestra el resultado. Así la retroalimentación acompaña al niño durante toda la construcción, no solo al final.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE2
---

# Una misma operación funciona con objetos, con dibujos y con dígitos

<div class="grid grid-cols-3 gap-3 mt-6">
  <div><img src="/cpa-concrete.jpg" alt="Programa concreto: resta entre dos grupos de cubos" /><p class="caption">Concreto: resta de cubos</p></div>
  <div><img src="/cpa-pictorial.jpg" alt="Programa pictórico: filtrado de alimentos según el color" /><p class="caption">Pictórico: filtrar por color</p></div>
  <div><img src="/cpa-abstract.jpg" alt="Programa abstracto: suma y resta entre números" /><p class="caption">Abstracto: suma y resta</p></div>
</div>

<div class="mt-6 text-center"><span class="ods-tag">ODS 4 · Meta 4.6</span> aritmética básica desde la infancia</div>

<!--
PRESENTA: TBD · TIEMPO: 40s

Así se materializa Bruner en las piezas. Las piezas de datos siguen la progresión concreto-pictórico-abstracto. Los objetos concretos son tapas, paletas de colores y cubos tipo Montessori que el niño pone sobre la mesa. Las cartas pictóricas representan alimentos y figuras geométricas, algunas con varios tamaños y colores. Y las cartas abstractas son los dígitos del 0 al 9, que puestos uno junto a otro forman números de varias cifras.

Las tres fotos son programas reales del ambiente: una resta entre dos grupos de cubos, un filtrado de alimentos según el color, y una suma y una resta entre números. Las cartas de operación y de criterio son comunes a los tres niveles. Las operaciones corresponden a la clasificación, la comparación, la ordenación y la aritmética del currículo de matemáticas de primero a tercer grado (MPPE, 2023), y es aquí donde el trabajo toca la Meta 4.6 del ODS 4: aritmética básica a través del juego desde la infancia.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE2
---

# Los niños conectan las piezas tocando sus puertos, sin cables ni texto

<div class="grid grid-cols-[auto_1fr] gap-8 mt-2 items-center">
  <img src="/final-environment.jpeg" alt="Ambiente en uso: piezas, cartas, conexiones por toque y resultado proyectado sobre la mesa" style="height: 345px" />
  <ol>
    <li>Colocar la pieza: aparece con sus puertos</li>
    <li>Tocar un puerto y luego otro: conexión</li>
    <li>Conexión incompatible: el puerto se agita</li>
    <li>Apertura y cierre: forman un grupo</li>
  </ol>
</div>

<!--
PRESENTA: TBD · TIEMPO: 45s

Esta es la interacción. El niño coloca una pieza; el subsistema de visión la reconoce y la interfaz la dibuja en el mismo lugar, con sus puertos de entrada y salida alrededor. Las conexiones no usan cables ni piezas adicionales: el niño toca sobre la superficie, uno después del otro, los dos puertos que quiere enlazar, y la interfaz crea la conexión si es compatible. Cada puerto admite una clase de dato: números, objetos, criterios o grupos. Si la conexión no es compatible, o el puerto de entrada ya está ocupado, se rechaza y el puerto se agita.

Parte de la estructura se infiere de la disposición física: para formar un grupo se coloca una carta de apertura y una de cierre, se enlazan con un toque, y todo lo que queda entre ambas forma el grupo. Del mismo modo, los dígitos contiguos forman un número.

La foto muestra el ambiente en uso. Sobre las conexiones se animan los «walkers», que llevan el dato que circula; la carta de salida muestra el resultado y un botón lo lee en voz alta. El docente puede cambiar la vista de los resultados entre concreto, pictórico y abstracto, lo que cambia la apariencia y no el significado.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE2
---

# El lenguaje ERAE ofrece doce operaciones curriculares sobre solo tres formas de valor

<div class="grid grid-cols-4 gap-3 mt-6 text-center">
  <div class="box"><h3>Aritmética</h3><p>sumar · restar<br />multiplicar · dividir</p></div>
  <div class="box"><h3>Comparación</h3><p>menor que<br />mayor que · igualdad</p></div>
  <div class="box"><h3>Colecciones</h3><p>filtrar · ordenar</p></div>
  <div class="box"><h3>Acceso</h3><p>primera · última<br />contar</p></div>
</div>

<div class="box plain mt-8 text-center">
  <strong>Valores:</strong> bolsa de objetos con cantidades · criterio · booleano
</div>

<!--
PRESENTA: TBD · TIEMPO: 40s

El lenguaje se llama ERAE y es un lenguaje visual de flujo de datos: un programa es un grafo de nodos que producen valores, los transforman y declaran salidas. No buscamos la Turing-completitud; buscamos un lenguaje suficiente para un subconjunto de problemas del currículo y simple de interpretar por un niño.

Reconoce doce operaciones: suma, resta, multiplicación y división; menor que, mayor que e igualdad; filtrar y ordenar según criterios; y primera, última y contar. Todas se representan con cartas, excepto menor que y mayor que, que por ahora no tienen carta en el mazo. Las de umbral no devuelven verdadero o falso, sino que filtran: «¿cuáles tienen menos de cinco?» se responde con una colección que el niño puede ver y contar.

Todo valor es una de tres formas: la bolsa, que transporta los objetos con sus cantidades; el criterio, que describe cómo seleccionar u ordenar; y el booleano, que solo produce la comparación de igualdad. Los números no son un tipo aparte, sino objetos abstractos cuya cantidad es su valor, y los contenidos curriculares, como alimentos o formas, se describen con la identidad del objeto, así que se pueden agregar sin tocar la gramática. La especificación completa, versión 1.0.0, está en el Apéndice B.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE2
---

# Un error apaga solo las salidas que dependen de la carta que lo causa

<img src="/soft-failure.jpg" alt="Error local: la suma conserva su resultado mientras la resta, incompleta, señala el error en su salida" class="mx-auto mt-2" style="height: 310px" />

<p class="caption">La suma conserva su resultado; la resta incompleta señala el error</p>

<!--
PRESENTA: TBD · TIEMPO: 35s

Como el niño no escribe texto, no puede cometer errores léxicos ni sintácticos: cada carta es un símbolo completo. Persisten tres clases de errores: de disposición, como abrir un grupo sin cerrarlo; de reconocimiento, cuando la visión no detecta una carta o confunde un 6 con un 9; y semánticos, que detecta el intérprete, como un número equivocado de argumentos o una división por cero.

Dos reglas hacen que estos errores no frustren la construcción. Primera: una carta sin conectar no es un error; un nodo incompleto vale «nulo» y el resto del programa se sigue evaluando. Segunda: los errores son locales, solo afectan a las salidas que dependen de la carta que los causa. En la foto, la suma muestra su resultado mientras la resta, a la que le falta una carta, señala el error en su propia salida. Un programa a medio construir avanza hasta donde puede.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE3
---

# Siete prototipos y cinco evoluciones llevaron de formas geométricas a un ambiente integrado

<div class="grid grid-cols-4 gap-3 mt-8 text-center" style="font-size: 19px">
  <div class="box plain"><h3>P1</h3>Formas por contornos · Kinect v1</div>
  <div class="box plain"><h3>P2</h3>Kinect v2 · toques</div>
  <div class="box plain"><h3>P3</h3>Detección con YOLO11n</div>
  <div class="box plain"><h3>P4</h3>Intérprete en TypeScript</div>
  <div class="box plain"><h3>P5</h3>Interfaz «sandbox»</div>
  <div class="box plain"><h3>P6</h3>Cartas · YOLO11s</div>
  <div class="box"><h3>P7</h3>Integración de extremo a extremo</div>
  <div class="box"><h3>E1–E5</h3>Toques, calibración, grupos, especificación, retroalimentación</div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 70s

El tercer objetivo fue construir el ambiente, y aquí se ve el enfoque por prototipos. El primero partió de Magicboard: bloques con formas y colores básicos, reconocidos por contornos con OpenCV y un Kinect v1, y operaciones de conjuntos en zonas fijas. Ahí, por sugerencia del tutor, cambiamos del paradigma imperativo al de flujo de datos. El segundo migró al Kinect v2 y adaptó la calibración y la detección de toques del código de Barrios. El tercero probó la detección de piezas con un modelo de aprendizaje profundo, YOLO11n, entrenado con un conjunto propio. El cuarto definió el lenguaje y su intérprete en TypeScript con Bun y Chevrotain. El quinto construyó la interfaz, el «modo sandbox», con React y React Flow, todavía con bloques digitales. El sexto rediseñó las piezas como cartas, pasó a YOLO11s y envió las piezas reconocidas al intérprete, pero aún sin conexiones.

El séptimo fue la primera integración completa, y sobre él hicimos cinco evoluciones: el detector de toques híbrido; la calibración con nueve puntos y los objetos concretos; los grupos y las operaciones de acceso; la reconstrucción del lenguaje sobre una especificación formal; y la mejora de la retroalimentación tras el juicio de expertos.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE3
---

# Cada prototipo reveló un problema que definió el requerimiento del siguiente

| Problema observado | Decisión |
|---|---|
| Resolución del Kinect v1 | Kinect v2, definitivo en la evolución 1 |
| Detección por contornos frágil | Detección de objetos con YOLO |
| Toques imprecisos con dos puntos | Homografía de cuatro y luego nueve puntos |
| Latencia entre procesos | Intérprete embebido en la interfaz |
| Conexiones sin representar (P6) | Puertos tipados, reglas y *walkers* (P7) |

<!--
PRESENTA: TBD · TIEMPO: 55s

Lo valioso del enfoque es esta cadena: el resultado de cada prototipo definió el requerimiento del siguiente. La duda sobre la resolución del Kinect v1 nos llevó al Kinect v2, al que volvimos de forma definitiva en la primera evolución. La fragilidad de la detección por contornos llevó a la detección por aprendizaje profundo. La transformación de dos puntos heredada era muy sensible al montaje físico, así que pasamos a una homografía de cuatro puntos, que corrige la perspectiva, y luego a una de nueve puntos en rejilla de tres por tres, resuelta por mínimos cuadrados (Hartley y Zisserman, 2003). La latencia entre el intérprete, que corría como un proceso aparte con servidores HTTP y WebSocket, y la interfaz nos llevó a convertir el intérprete en una librería que la interfaz invoca en el propio cliente. Y la imposibilidad de representar conexiones, evidente en el sexto prototipo, produjo el sistema de puertos tipados, reglas estructurales y walkers del séptimo.

Esta cadena solo se podía establecer construyendo y evaluando, y por eso concluimos que el enfoque por prototipos fue el adecuado.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE3
---

# El ambiente construido detecta piezas y toques, calibra por homografía e interpreta mientras se construye

| Componente | Cómo quedó construido |
|---|---|
| Piezas | Detección de objetos con YOLO |
| Toques | Profundidad e infrarrojo (DIRECT) + mano (MediaPipe) |
| Calibración | Homografía con rejilla de nueve puntos |
| Intérprete ERAE | Librería embebida, evaluación incremental |
| Integración | Relevo FastAPI → interfaz React |

<!--
PRESENTA: TBD · TIEMPO: 50s
PREGUNTA: ¿Qué modelo YOLO quedó en la versión final? El Cap. IV menciona YOLO11n, YOLO11s y YOLO26n; el Apéndice C (TODO) dice yolo_11s_ultra.pt; el código por defecto carga yolo_11s_cards.pt. Ver «Inconsistencias» en OUTLINE.md.

Este es el resultado técnico de la construcción. Las piezas se reconocen con un modelo de detección de objetos de la familia YOLO, más robusto que los contornos de los primeros prototipos. Los toques se detectan con un detector híbrido: el algoritmo DIRECT (Xiao et al., 2016) decide si hay un toque a partir de las imágenes de profundidad e infrarroja, y el Hand Landmarker de MediaPipe (Lugaresi et al., 2019) aporta la posición precisa de la punta del índice. Así separamos el «si» del «dónde». La calibración entre cámara y proyección es una homografía de nueve puntos.

El intérprete de ERAE es una librería embebida en la interfaz, con evaluación dirigida por demanda e incremental: guarda en caché el valor de cada nodo y, cuando llega una nueva versión del programa, invalida solo los nodos que cambiaron y los que dependen de ellos. Eso es posible porque el lenguaje no tiene estado mutable. Por último, un servidor de relevo en FastAPI pasa las detecciones del subsistema de visión, que está en Python, a la interfaz en React.

Con el séptimo prototipo logramos por primera vez la experiencia de extremo a extremo: piezas físicas, conexiones y salida proyectada forman un programa que se evalúa mientras se construye. Hay que decir también que varias mejoras, sobre todo la precisión de la calibración y el rendimiento de la detección, no se cuantificaron formalmente.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE4
---

# El ambiente satisface sus cinco requerimientos funcionales; tres no funcionales quedan parciales

| Requerimiento | Estado |
|---|---|
| RF-01 a RF-05: construir, reconocer, interpretar, proyectar, guiar | <span class="yes">✓ Satisfecho</span> |
| RNF-03: manejar errores de disposición | <span class="yes">✓ Satisfecho</span> |
| RNF-01: usable por niños y docentes | <span class="part">◐ Parcial</span> (tras expertos) |
| RNF-02: elementos persuasivos | <span class="part">◐ Parcial</span> |
| RNF-04: retroalimentación visual y auditiva | <span class="part">◐ Parcial</span> |

<!--
PRESENTA: TBD · TIEMPO: 55s

El cuarto objetivo fue validar el ambiente, en dos frentes. El primero es la verificación de requerimientos: una matriz de trazabilidad que relaciona cada requerimiento del análisis con la funcionalidad que lo satisface. La matriz completa está en el respaldo, R-03.

Los cinco requerimientos funcionales están satisfechos: construir programas con piezas tangibles y conexiones digitales; capturar y reconocer piezas y toques; traducir el grafo a un programa e interpretarlo; ejecutarlo con salida proyectada; y guiar al niño con evaluación incremental, resaltados, walkers y resultados intermedios. También el manejo de errores de disposición.

Quedan parciales tres no funcionales. La retroalimentación auditiva se limita a la síntesis de voz de los resultados; los elementos persuasivos, como el diseño colorido y la voz, existen, pero su efecto en el interés de los niños requiere probarse con niños. Y la usabilidad estaba pendiente antes del juicio de expertos: pasó a parcial gracias al experto en medios didácticos, que la valoró como potencial docente de primero a tercer grado. Falta comprobarla con niños.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE4
---

# Los expertos ven viable el ambiente si el docente conduce la actividad

| Experto | Observó | Recomendó |
|---|---|---|
| Medios didácticos | Colores y sencillez; aplicable en el aula, para evaluar | Mejor retroalimentación; música relajante |
| Interacción humano-computador | Mucho contenido para uso autónomo; fallos de detección | Guías de actividades; mejor detección de cartas |

<div v-click class="box mt-6 text-center">
Evolución 5: errores explicados en la carta de salida
</div>

<!--
PRESENTA: TBD · TIEMPO: 55s
PREGUNTA: El informe no registra instrumentos con escala ni un veredicto formal de los expertos (solo notas de una entrevista no estructurada). Si el jurado pregunta por el instrumento, la respuesta es esa; ¿lo decimos así?

El segundo frente fue el juicio de dos expertos, que vieron una demostración del ambiente en el modo de uso para el que se concibió: una actividad colaborativa entre el docente y los niños, con el docente como guía. Luego recogimos sus observaciones en una entrevista no estructurada.

El experto en medios didácticos, que además puede ser docente de primero a tercer grado y por eso funcionó también como prueba de usabilidad desde el docente, valoró la paleta de colores y la sencillez con que se entiende el ambiente, y lo consideró aplicable en el aula como recurso del docente, con especial utilidad en la evaluación de los aprendizajes. Pidió mejorar la retroalimentación y sugirió música relajante.

El experto en interacción humano-computador se preocupó por la cantidad de contenido programático si los niños lo usaran solos. Le aclaramos que el uso es con el docente como guía, y aun así sostuvo que ese rol exige conocer a fondo el ambiente: recomendó guías de actividades modelo y explicitar las restricciones. También observó fallos en la detección de cartas durante la demostración y advirtió que podrían frustrar a docentes y niños.

Ambos coinciden: el ambiente alcanza su propósito bajo la conducción del docente. [clic] Respondimos a la retroalimentación en la quinta evolución: los errores ahora aparecen en la carta de salida afectada, nombran la carta que los causa y dicen qué hacer; las fracciones se dibujan como objetos incompletos y la voz se sintetiza localmente. La retroalimentación sonora completa sigue pendiente.
-->

---
eyebrow: 04 · Desarrollo y resultados · OE5
---

# Dos manuales separan lo que necesita quien instala de lo que necesita el docente

<div class="grid grid-cols-2 gap-6 mt-6">
  <div class="box">
    <h3>Manual del sistema · Apéndice C</h3>
    <p>Herramientas y arquitectura C4</p>
    <p>Instalación y configuración</p>
    <p>Extensión y pruebas</p>
  </div>
  <div class="box">
    <h3>Manual de usuario · Apéndice D</h3>
    <p>Montaje y calibración</p>
    <p>Catálogo de piezas y uso</p>
    <p>Restricciones y errores</p>
  </div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 30s
PREGUNTA: En report.typ los Apéndices C y D están casi enteros en TODO: solo tienen la introducción y las figuras C4. Esta lámina describe lo que el Cap. IV dice que contienen. Si no se completan antes de la defensa, el jurado puede abrirlos y no encontrarlos; ¿se completan, o reformulamos esta lámina?

El quinto objetivo fue la documentación formal, organizada en dos manuales según el destinatario. El manual del sistema, Apéndice C, es para quien instala, mantiene o extiende el ambiente: herramientas y su justificación, la arquitectura con sus diagramas C4, los datos que intercambian los subsistemas, requisitos, instalación, puesta en marcha y configuración, cómo agregar nuevas piezas y las pruebas realizadas.

El manual de usuario, Apéndice D, es para el docente: montaje y encendido, calibración, catálogo de piezas y uso de la interfaz con un ejemplo de principio a fin, una sección sobre qué se espera que hagan los niños, y las restricciones y los posibles errores con su solución. Este manual atiende una de las condiciones que dejó la validación: que el docente conozca el alcance del ambiente antes de conducir una actividad.
-->
