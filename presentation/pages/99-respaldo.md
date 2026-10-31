---
eyebrow: Respaldo · OE1 · Referentes
class: dense
zoom: 0.78
---

# Ningún referente comparado reúne los rasgos del ambiente propuesto

| Referente | Interfaz y pantalla | Colaboración | Edad y lectoescritura | Paradigma y conceptos | Retroalimentación |
|---|---|---|---|---|---|
| TORTIS (Morgado et al., 2006) | Cajas de botones y cartas en un tragafichas; tortuga robótica; un monitor mostraba los comandos | No documentada como objetivo de diseño | 3 a 5 años; no requiere lectura | Imperativo; secuencia, repetición, procedimientos | Movimiento y dibujo; luces en la carta en ejecución |
| AlgoBlock (Suzuki y Kato, 1993) | Bloques físicos conectados; submarino en pantalla | Acceso simultáneo, monitoreo mutuo, pase del turno | Primaria y secundaria; tres niñas de 12 años | Imperativo (Logo); secuencia, condicionales, bucles | Submarino en pantalla; luces |
| Scratch y ScratchJr (Maloney et al., 2010; Bers, 2018) | Bloques en pantalla: computadora o tableta | Comunidad en línea; construcción frente a la pantalla | Scratch desde 8 años, con lectoescritura; ScratchJr 5 a 7 | Imperativo por bloques; secuencia, bucles, condicionales, eventos | Inmediata, en pantalla |
| Magicboard (Barrios, 2024) | Mesa con proyector y Kinect; sin monitor | Juegos sociales colaborativos | Preescolar | No es entorno de programación | Proyectada sobre la mesa |
| Rojas y Youssef (2024) | Fichas táctiles con braille, reconocidas por cámara | No documentada | Niños con discapacidad visual | Imperativo; ciclos, condicionales, variables | Auditiva |
| **Ambiente propuesto** | Superficie tangible con proyección, sin monitor | Acceso simultáneo, superficie compartida | 6 a 9 años; piezas sin texto | Flujo de datos | Visual proyectada y auditiva |

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Tabla completa de la comparación de referentes del Capítulo IV (OE1). Sirve si el jurado pregunta cómo se codificaron los símbolos de la lámina de OE1 o por qué no se incluyeron otros referentes.
-->

---
eyebrow: Respaldo · OE1 · Requerimientos
class: dense
---

# Seis características del análisis originan cinco requerimientos funcionales y cuatro no funcionales

| Código | Característica asociada | Requerimiento |
|---|---|---|
| RF-01 | Interfaz tangible de mesa; colaboración | Construir programas con elementos tangibles y conexiones digitales que representen datos, flujos y operaciones |
| RF-02 | Interfaz tangible de mesa | Capturar la disposición de elementos y conexiones, y reconocerlos |
| RF-03 | Lenguaje de flujo de datos | Interpretar los programas, traduciéndolos a una representación ejecutable |
| RF-04 | Retroalimentación aumentada; menos pantalla | Ejecutar los programas y mostrar la salida proyectada sobre una superficie plana |
| RF-05 | Retroalimentación aumentada; 6 a 9 años | Proveer retroalimentación que guíe la construcción |
| RNF-01 | 6 a 9 años | Usable por niños de 6 a 9 años y docentes de 1.º a 3.º grado |
| RNF-02 | 6 a 9 años | Elementos persuasivos que capten el interés de los niños |
| RNF-03 | Retroalimentación aumentada | Manejar errores en la disposición de elementos tangibles y digitales |
| RNF-04 | Retroalimentación aumentada | Retroalimentación visual y auditiva |

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Características: interfaz tangible de superficie, retroalimentación visual aumentada, lenguaje basado en flujo de datos, diseño para la colaboración, compatibilidad con el desarrollo cognitivo de 6 a 9 años y reducción del tiempo en pantalla. Tabla de requerimientos del Capítulo IV, abreviada; el texto completo está en el informe.
-->

---
eyebrow: Respaldo · OE4 · Matriz de trazabilidad
class: dense
---

# Cada requerimiento se traza a la funcionalidad que lo satisface

| Req. | Funcionalidad que lo satisface | Estado |
|---|---|---|
| RF-01 | Detección de piezas con YOLO; puertos tipados y reglas estructurales en el IDE | ✓ Satisfecho |
| RF-02 | Cámara de color y profundidad; reconocimiento de cartas y toques; homografía; relevo a la interfaz | ✓ Satisfecho |
| RF-03 | Traducción del grafo visual a un programa; intérprete ERAE embebido | ✓ Satisfecho |
| RF-04 | Evaluación bajo demanda e incremental; salida en la interfaz proyectada | ✓ Satisfecho |
| RF-05 | Evaluación incremental, resaltados, *walkers*, resultados intermedios | ✓ Satisfecho |
| RNF-01 | Piezas tangibles y uso guiado por el docente; falta comprobar con niños | ◐ Parcial (antes: Pendiente) |
| RNF-02 | Diseño colorido, síntesis de voz; efecto por validar con niños | ◐ Parcial |
| RNF-03 | Aridad y categoría de valor; validación de conexiones; programas incompletos | ✓ Satisfecho |
| RNF-04 | Visual completa; auditiva limitada a la voz de los resultados | ◐ Parcial |

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Matriz de verificación de requerimientos del Capítulo IV, con el único cambio de estado derivado del juicio de expertos: RNF-01 pasó de Pendiente a Parcial por la valoración del experto en medios didácticos como potencial docente de 1.º a 3.º grado.
-->

---
eyebrow: Respaldo · OE4 · Juicio de expertos
class: dense
---

# Los expertos se recogieron con entrevista no estructurada tras una demostración

| Experto | Observación | Respuesta de los autores |
|---|---|---|
| Medios didácticos | Paleta de colores y sencillez para comprender el ambiente | Fortaleza; se conserva |
| Medios didácticos | Aplicable en el aula como recurso del docente, con énfasis en la evaluación | Fortaleza; coincide con el uso mediado por el docente |
| Medios didácticos | Mejorar la retroalimentación al usuario | Errores mejorados en la evolución 5; sonido en curso |
| Medios didácticos | Incorporar música relajante | Registrada para versiones futuras |
| IHC | Mucho contenido programático si lo usan los niños solos | Uso colaborativo con el docente como guía |
| IHC | El docente debe conocer a fondo el ambiente | Guías de actividades modelo; restricciones explícitas |
| IHC | Fallos en la detección de cartas durante la demostración | Mejorar la detección de cartas |

<p class="mt-2">Perfiles: docente universitario en medios didácticos (más de 10 años; apto para 1.º a 3.º grado) · docente universitario en usabilidad e IHC. Instrumento: registro de notas en teléfono móvil.</p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: El informe no incluye un instrumento con escala ni un veredicto formal (aprobado/no aprobado). Si el jurado lo pide, la respuesta honesta es que la técnica fue la entrevista no estructurada, con registro de notas, después de la demostración. ¿Hay algún registro adicional (fecha, nombres de los expertos) que podamos citar?

Tabla de observaciones del Capítulo IV con la respuesta de los autores.
-->

---
eyebrow: Respaldo · Metodología · Comparación
---

# Los prototipos ganaron por la disponibilidad limitada de los interesados y la incertidumbre

| Enfoque | Condición del proyecto | Decisión |
|---|---|---|
| Ágil | Exige interesados disponibles de forma permanente | Descartado |
| Incremental | Supone requerimientos bastante definidos | Considerado |
| Prototipos | Incertidumbre técnica y de requerimientos; revisiones acordadas con el tutor | **Elegido** |

<p class="cite mt-4">Modelos evolutivos: «versiones cada vez más completas del software» (Pressman, 2010)</p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: El informe dice que se consideraron el incremental y el de prototipos y que se eligió prototipos por la incertidumbre, pero no dice explícitamente por qué se descartó el incremental. «Supone requerimientos bastante definidos» es nuestra inferencia; ¿la mantenemos o la quitamos?

Si preguntan por la metodología: se descartaron los enfoques ágiles porque los interesados no estarían disponibles de forma permanente, aunque sí podían acordarse reuniones de revisión. Entre incremental y prototipos se eligió prototipos por la incertidumbre técnica y de requerimientos, para definir los requerimientos finales a través de los prototipos y de su revisión con el tutor.
-->

---
eyebrow: Respaldo · OE3 · Visión por computador
class: dense
---

# La visión por computador evolucionó en cuatro frentes a lo largo de los prototipos

| Frente | Evolución |
|---|---|
| Sensor | Kinect v1 (P1) → Kinect v2 con OpenNI2 compilado (P2) → Kinect v1 temporal (P6) → Kinect v2 con bifurcación de PyKinect2 (E1) |
| Piezas | Contornos y color (P1) → YOLO11n (P3) → YOLO11n con cajas ajustadas → YOLO11s (P6) → YOLO26n explorado → reentrenado con tapas, paletas y cubos (E2) |
| Calibración | Window-to-Viewport de dos puntos (P2) → homografía de cuatro puntos (P7) → homografía de nueve puntos, mínimos cuadrados (E2) |
| Toques | Solo profundidad, en la línea de Wilson (2010) → híbrido DIRECT + MediaPipe Hand Landmarker (E1) |

<p class="mt-3">Mejoras observadas cualitativamente; sin métricas formales (mAP, latencia, error de calibración).</p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: El código contiene además un detector RF-DETR (rfdetr_card_detector.py y su cuaderno de entrenamiento) que el informe no menciona. Si el jurado pregunta por alternativas a YOLO, ¿lo mencionamos?

Si preguntan por qué YOLO11s: con YOLO11n y las cajas sobre toda la carta había confusión entre clases; ajustar las cajas al área de la imagen mejoró poco; YOLO11s mejoró apreciablemente, aunque la comparación fue cualitativa. Si preguntan por métricas: no se midieron formalmente, y es la primera recomendación del trabajo (mAP50, mAP50-95, matriz de confusión, latencia, error de calibración).
-->

---
eyebrow: Respaldo · OE3 · Detección de toques
---

# El detector híbrido separa si hay un toque de dónde ocurre

<div class="grid grid-cols-[1fr_auto_1fr] gap-4 mt-6">
  <div class="box">
    <h3>¿Hay toque? · DIRECT</h3>
    <p>Imagen de profundidad e infrarroja</p>
    <p>Relleno por inundación, zonas, histéresis</p>
    <p class="cite mt-2">(Xiao et al., 2016)</p>
  </div>
  <div class="arrow">+</div>
  <div class="box">
    <h3>¿Dónde? · MediaPipe</h3>
    <p>Hand Landmarker</p>
    <p>Punta del dedo índice</p>
    <p class="cite mt-2">(Lugaresi et al., 2019)</p>
  </div>
</div>

<p class="mt-6 text-center">Calibración: homografía de nueve puntos (rejilla 3 × 3) <span class="cite">(Hartley y Zisserman, 2003)</span></p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: CLAUDE.md pide una lámina de «touch classifier detail», pero el informe no describe un clasificador de toques: describe el detector híbrido DIRECT + MediaPipe. ¿Es esto lo que se quería?

La robustez del sensor de profundidad decide si hay contacto; la precisión de la estimación visual de la mano ubica el punto. Las coordenadas de la cámara de color se llevan al espacio de profundidad con el mapeo del Kinect for Windows SDK 2.0, que es una fuente adicional de error de calibración.
-->

---
eyebrow: Respaldo · OE2 · Mesa
zoom: 0.88
---

# La mesa tiene un área de proyección de 70 × 110 cm con proyector y Kinect cenitales

<div class="grid grid-cols-[1166fr_620fr] gap-4 mt-2 items-center">
  <img src="/table-setup.svg" alt="Esquema de la mesa en vista superior con el proyector y el Kinect sobre el área de proyección" style="max-height: 290px" class="mx-auto" />
  <img src="/table-photo.jpg" alt="Fotografía del ambiente montado" style="max-height: 290px" class="mx-auto" />
</div>

<p class="mt-2">Tablero en riñón de 145 × 110,1 cm y 2,5 cm · cuatro patas de 65 cm × 5 cm de diámetro</p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Detalle físico del Capítulo IV: tablero en forma de riñón de 145 × 110,1 cm y 2,5 cm de espesor, área de proyección de 70 × 110 cm; patas de metal de 65 cm de largo y 5 cm de diámetro, con soportes circulares de 10 cm; cada soporte a 12 cm de los bordes.
-->

---
eyebrow: Respaldo · Arquitectura · C4 nivel 1
---

# El ambiente no consume servicios de red ni usa base de datos

<iframe class="c4" src="/c4/c4-02-contexto.html" title="Diagrama C4 de contexto"></iframe>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: Los visores de Archify abren en tema oscuro dentro del iframe; CLAUDE.md pide fondo claro para el proyector. Se puede cambiar con el botón de tema del propio visor durante la ronda de preguntas. ¿Basta así o preferimos las SVG estáticas (tema claro) en estas láminas?

Visor interactivo de Archify (git-ignorado). Si falta public/c4/c4-02-contexto.html: mkdir -p public/c4 && cp ../docs/architecture/c4-0*.html public/c4/ (y, si tampoco existe en docs/architecture, regenerarlo con archify finalize; ver ../docs/architecture/README.md). Respaldo estático: /c4-contexto.svg.
-->

---
eyebrow: Respaldo · Arquitectura · C4 nivel 2
---

# El intérprete ERAE va embebido en la interfaz; el relevo conecta con la visión

<iframe class="c4" src="/c4/c4-03-contenedores.html" title="Diagrama C4 de contenedores"></iframe>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Contenedores: subsistema de visión, servidor de relevo, interfaz e intérprete ERAE embebido, con los archivos locales que los alimentan. Visor git-ignorado; ver la nota de R-09. Respaldo estático: /c4-contenedores.svg.
-->

---
eyebrow: Respaldo · Arquitectura · C4 nivel 3
---

# La calibración enlaza con la detección a través de un único resultado de calibración

<iframe class="c4" src="/c4/c4-04-componentes-vision.html" title="Diagrama C4 de componentes del subsistema de visión"></iframe>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Componentes del subsistema de visión: hardware, calibration, transform, detection y bridge, y el CalibrationResult que enlaza la calibración con la detección. Visor git-ignorado; ver la nota de R-09. Respaldo estático: /c4-componentes-vision.svg.
-->

---
eyebrow: Respaldo · Arquitectura · C4 nivel 3
---

# La interfaz lleva piezas y toques a un grafo que alimenta la evaluación del intérprete

<iframe class="c4" src="/c4/c4-05-componentes-interfaz.html" title="Diagrama C4 de componentes de la interfaz y del intérprete"></iframe>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Componentes de la interfaz y del intérprete ERAE: de los proveedores de visión y de toque al grafo, y de este a la cadena de evaluación del intérprete. Visor git-ignorado; ver la nota de R-09. Respaldo estático: /c4-componentes-interfaz.svg.
-->

---
eyebrow: Respaldo · Arquitectura · Secuencia
---

# Piezas y toques convergen en el grafo, que se reevalúa y se proyecta

<iframe class="c4" src="/c4/c4-06-secuencia-ciclo.html" title="Diagrama de secuencia del ciclo detección, evaluación y proyección"></iframe>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: La leyenda del informe dice que «las piezas viajan por HTTP y los toques por WebSocket». En el código eso vale del subsistema de visión al relevo; del relevo a la interfaz ambos viajan por WebSocket (/ws/vision y /ws/touch). Si el jurado pregunta, conviene precisarlo.

Ciclo detección → evaluación → proyección. Visor git-ignorado; ver la nota de R-09. Respaldo estático: /c4-secuencia-ciclo.svg.
-->

---
eyebrow: Respaldo · OE2 · Operaciones de ERAE
class: dense
---

# Las doce operaciones de ERAE tienen firma, aridad y regla de agrupación definidas

| Operación | Firma | Qué hace |
|---|---|---|
| `sum` | (bolsa, …) → bolsa | Suma por identidad (vector) |
| `substract` | (bolsa, bolsa) → bolsa | a − b por identidad; puede dar negativos |
| `multiply` | (bolsa, número) → bolsa | Escala cada cantidad |
| `divide` | (bolsa, número) → bolsa | Divide cada cantidad; error si divisor 0 |
| `less_than` / `greater_than` | (bolsa, número) → bolsa | Filtra identidades bajo / sobre el umbral |
| `compare` | (bolsa, bolsa) → booleano | Verdadero si denotan el mismo vector |
| `order` | (bolsa, criterio, …) → bolsa | Ordena; criterios siguientes desempatan |
| `filter` | (bolsa, criterio, …) → bolsa | Y dentro del criterio, O entre criterios |
| `first` / `last` | (bolsa) → bolsa | Primera / última entrada del orden actual |
| `count` | (bolsa) → número | Suma todas las cantidades; count(nulo) = 0 |

<p class="mt-2">No agrupan los números: multiply, divide, less_than, greater_than, order (7, 2, 5 se ordenan, no se suman). Sin carta en el mazo: less_than, greater_than.</p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Resumen del Apéndice B, §3. Todas ignoran sus argumentos nulo y preservan el orden.
-->

---
eyebrow: Respaldo · OE2 · Dominio y evaluación
class: dense
---

# Los valores son bolsas, criterios o booleanos, y se evalúan bajo demanda

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

**Dominio**
- Bolsa: secuencia finita de (identidad, cantidad ∈ ℚ)
- Identidad: (categoría CPA, tipo, subtipo, atributos)
- Denota un vector en ℚ<sup>(Id)</sup>; igualdad ignora orden, agrupación y ceros
- `nulo`: bolsa vacía; toda operación la ignora
- Número: objeto abstracto cuya cantidad es su valor
- Criterio: filtro (Y de propiedad = valor) u orden

</div>
<div>

**Evaluación**
- Programa: sentencias `source`, `transform`, `sink`
- Bien formado: nombres únicos, referencias resueltas, acíclico
- Dirigida por demanda: desde cada `sink`
- Cada nodo se evalúa una vez (memorización)
- Sin estado mutable: determinista
- Incremental: caché entre versiones; invalida lo que cambió y sus dependientes

</div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Apéndice B, §1 y §2, más la integración del intérprete del Capítulo IV. La evaluación incremental es una técnica de implementación (memorización con invalidación), no una propiedad heredada de Lucid; la hace posible la ausencia de estado mutable, que sí es de Lucid. La especificación es la versión 1.0.0, del 26 de septiembre de 2026.
-->

---
eyebrow: Respaldo · OE2 · Errores
class: dense
---

# Los errores se detectan en tres momentos y solo afectan las salidas en su camino

| Momento | Errores |
|---|---|
| Sintaxis (analizador) | Contra la gramática; nodos incompletos se toleran y valen `nulo`; un grupo con un criterio no se analiza |
| Estáticos (antes de evaluar) | Nombre duplicado · referencia sin resolver · ciclo · operación desconocida · aridad · categoría de valor equivocada · criterio inadecuado · objeto inválido |
| Ejecución (aislados por salida) | Número esperado · división por cero |

<p class="mt-3">Cada error informa su naturaleza, el nodo donde ocurrió, el nodo causante y las salidas afectadas.</p>
<p>En el ambiente se suman los errores de disposición (p. ej., apertura sin cierre) y de reconocimiento (6 confundido con 9).</p>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Apéndice B, §4, y Capítulo IV (prevención de errores). «Una salida cuyo camino está limpio produce su valor aunque otra falle.»
-->

---
eyebrow: Respaldo · OE2 · Gramática
class: dense
---

# La gramática admite declaraciones incompletas para analizar programas a medio construir

```text
program        ::= statement*
statement      ::= source_decl | transform_decl | sink_decl
source_decl    ::= "source" identifier "=" (object_literal | group)? ";"
transform_decl ::= "transform" identifier "=" (operation "(" argument_list? ")")? ";"
sink_decl      ::= "sink" identifier "=" identifier? ";"
argument_list  ::= identifier ("," identifier)*
operation      ::= identifier
group          ::= "[" (data_literal ("," data_literal)*)? "]"
object_literal ::= data_literal | criteria_literal
criteria_kind  ::= '"filter"' | '"order"'
category_type  ::= '"abstracto"' | '"pictorico"' | '"concreto"'
rational_literal ::= "-"? digit+ ( "/" digit+ | "." digit+ )?
identifier     ::= [a-zA-Z][a-zA-Z0-9_-]*
```

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Extracto de la gramática EBNF (notación W3C) del Apéndice B, §5.1; se omiten data_literal, criteria_literal, kv_pair, array_literal, string_literal y digit por espacio. Todo valor de declaración es opcional, así que un nodo incompleto es sintácticamente válido y vale nulo. La representación textual la genera el sistema; el niño nunca la ve.
-->

---
eyebrow: Respaldo · Limitaciones
---

# Lo que el trabajo no midió ni cubre queda declarado como límite

- Sin pruebas con niños: usabilidad, comprensión y efecto por comprobar
- Sin métricas formales de detección, latencia ni calibración
- Fallos de detección observados en la demostración
- Retroalimentación sonora parcial
- Sin repetición ni condicionales en el lenguaje
- Menor que y mayor que sin carta en el mazo

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Todo esto está declarado en el informe (Capítulos IV y V). Si el jurado lo señala, la respuesta es que el alcance era desarrollar la herramienta y que cada punto tiene una recomendación asociada.
-->

---
eyebrow: Respaldo · OE3 · Prototipos
---

# Los prototipos 1 a 6 probaron por separado sensor, detección, lenguaje, interfaz y piezas

<div class="grid grid-cols-3 gap-3 mt-2 text-center" style="font-size: 16px">
  <div><img src="/first-prototype.jpeg" alt="Primer prototipo con bloques de formas geométricas y colores" class="mx-auto" style="height: 125px" /><p>P1 · Formas</p></div>
  <div><img src="/second-prototype.jpeg" alt="Segundo prototipo con Kinect v2" class="mx-auto" style="height: 125px" /><p>P2 · Kinect v2</p></div>
  <div><img src="/third-prototype-dataset.jpeg" alt="Conjunto de datos del tercer prototipo" class="mx-auto" style="height: 125px" /><p>P3 · Datos YOLO</p></div>
  <div><img src="/fourth-prototype-visual-design.jpeg" alt="Diseño visual preliminar del lenguaje" class="mx-auto" style="height: 125px" /><p>P4 · Lenguaje</p></div>
  <div><img src="/fifth-prototype-design.png" alt="Boceto de la interfaz sandbox" class="mx-auto" style="height: 125px" /><p>P5 · Interfaz</p></div>
  <div><img src="/sixth-prototype.jpeg" alt="Sexto prototipo con detección de cartas" class="mx-auto" style="height: 125px" /><p>P6 · Cartas</p></div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: El séptimo prototipo no tiene figura en el informe (TODO en el Cap. IV). ¿Tenemos una foto para agregarla aquí y al informe?

Figuras del Capítulo IV, para responder preguntas sobre un prototipo concreto.
-->

---
eyebrow: Respaldo · OE3 · Tecnologías
class: dense
---

# Python sostiene la visión y TypeScript el lenguaje y la interfaz

| Subsistema | Tecnologías (según el informe) |
|---|---|
| Visión por computador | Python, OpenCV, YOLO (Ultralytics), MediaPipe, bifurcación de PyKinect2, Kinect for Windows SDK 2.0 |
| Servidor de relevo | FastAPI (reemplazó los servidores Elysia del P4) |
| Intérprete ERAE | TypeScript, Bun, Chevrotain (analizador) |
| Interfaz | TypeScript, React, React Flow; síntesis de voz local en el navegador |

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)
PREGUNTA: Las versiones exactas solo aparecen en un TODO del Apéndice C (React 19, @xyflow/react 12, Python 3.12+). ¿Las agregamos cuando el apéndice esté escrito?

Por qué FastAPI para el relevo: cercanía tecnológica, porque el subsistema de visión está en Python y así se reutilizan las definiciones de los mensajes de detección y de toques sin duplicarlas entre dos lenguajes. Por qué TypeScript y Bun: desarrollo rápido, tipado para estructuras complejas, y Bun ejecuta TypeScript sin transpilar.
-->

---
eyebrow: Respaldo · ODS
---

# El trabajo se alinea con las metas 4.4 y 4.6 del ODS 4, Educación de calidad

<div class="grid grid-cols-2 gap-6 mt-6">
  <div class="box"><h3>Meta 4.4</h3><p>Aumenta el número de jóvenes y adultos con competencias necesarias para acceder al empleo, fomentando desde edades tempranas el desarrollo del pensamiento lógico y computacional.</p></div>
  <div class="box"><h3>Meta 4.6</h3><p>Promueve conocimientos básicos de aritmética en adultos, fomentando el aprendizaje desde la infancia a través de juegos interactivos.</p></div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 0s (respaldo)

Texto literal del informe (Capítulo I, Impacto en los Objetivos de Desarrollo Sostenible). Ver la PREGUNTA de la lámina de importancia sobre la redacción de la Meta 4.6.
-->
