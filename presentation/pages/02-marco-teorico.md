---
eyebrow: 02 · Marco teórico · Antecedentes
---

# Tres trabajos de la UCAB aportan la mesa, la visión por computador y las piezas físicas

| Antecedente | Qué aporta al ambiente |
|---|---|
| Barrios (2024), Magicboard | Mesa con proyector y Kinect; juegos colaborativos |
| Espejo Bacadare (2022) | Visión por computador que une mundo físico y programa |
| Rojas Villarroel y Youssef Abi Saba (2024) | Bloques físicos reconocidos por cámara |

<div v-click class="box mt-6 text-center">
Este trabajo continúa Magicboard: le añade programación tangible
</div>

<!--
PRESENTA: TBD · TIEMPO: 40s
PREGUNTA: El informe también cita dos antecedentes internacionales (Pérez-Marín et al., 2020, sobre metáforas y Scratch; Montes et al., 2021, sobre el juego serio DFD-C). Los dejamos fuera por tiempo, pues los resultados no dependen de ellos. ¿Les parece bien?

Tres trabajos de grado de la UCAB son la base directa del ambiente. El más importante es Magicboard, de Anthony Barrios (2024): una pizarra digital en forma de mesa, con proyector y sensor Kinect, sobre la que niños de preescolar juegan juegos sociales colaborativos. Nuestro trabajo es una continuación de Magicboard: retoma su configuración de mesa, proyector y sensor de profundidad y su enfoque colaborativo, y le añade el pensamiento computacional mediante la programación tangible.

De Espejo Bacadare (2022), un entorno de robótica educativa, retomamos el uso de la visión por computador para vincular el mundo físico con la ejecución del programa. Y de Rojas y Youssef (2024), un sistema para enseñar programación a niños con discapacidad visual, el uso de bloques físicos reconocidos por visión por computador como interfaz de programación tangible, que fue el punto de partida de nuestros prototipos.
-->

---
eyebrow: 02 · Marco teórico · Desarrollo cognitivo
---

# Las tres formas de representación de Bruner organizan las piezas del ambiente

<p class="mt-2">6 a 9 años: transición a las operaciones concretas <span class="cite">(Piaget y Inhelder, 1969)</span></p>

<div class="grid grid-cols-[1fr_auto_1fr_auto_1fr] gap-3 mt-5 text-center">
  <div class="box"><h3>Enactiva</h3><p>por la acción</p><p class="mt-3 font-semibold">Objetos concretos</p></div>
  <div class="arrow">→</div>
  <div class="box"><h3>Icónica</h3><p>por imágenes</p><p class="mt-3 font-semibold">Cartas pictóricas</p></div>
  <div class="arrow">→</div>
  <div class="box"><h3>Simbólica</h3><p>por símbolos</p><p class="mt-3 font-semibold">Cartas abstractas</p></div>
</div>

<p class="cite text-center mt-4">Enfoque concreto-pictórico-abstracto (CPA) (Bruner, 1968; Leong et al., 2015)</p>

<!--
PRESENTA: TBD · TIEMPO: 45s
PREGUNTA: En references.bib la clave es «bruner1966» pero el campo year es 1968 (con note = 1966, la edición original). Según el estilo CSL del informe, la cita sale como «Bruner, 1968». ¿Ponemos (Bruner, 1968), (Bruner, 1966/1968) o corregimos el .bib?
PREGUNTA: Leong et al. (2015) tiene tres autores; en APA 6 la primera cita los nombra a los tres. En la lámina va «et al.» por espacio. ¿Lo aceptamos?

Del marco teórico solo traemos lo que necesitan los resultados. Según Piaget, entre los 6 y los 9 años el niño transita de la etapa preoperacional a la de las operaciones concretas: empieza a razonar lógicamente sobre situaciones concretas, pero los conceptos abstractos todavía le cuestan. Por eso manipular objetos es una vía eficaz para introducir conceptos computacionales, como sostiene Zapata-Ros.

Bruner distingue tres formas de representar la experiencia: la enactiva, mediante la acción; la icónica, mediante imágenes; y la simbólica, mediante el lenguaje y los símbolos. Plantea además que la secuencia de enseñanza probablemente avanza en ese orden. En esa concepción se basa el enfoque concreto-pictórico-abstracto, y es exactamente lo que organiza nuestras piezas: objetos concretos que el niño manipula, cartas con dibujos y cartas con dígitos. Una misma operación se aplica en los tres niveles, y ese es el aporte teórico del trabajo.
-->

---
eyebrow: 02 · Marco teórico · Interfaces tangibles y flujo de datos
---

# Una mesa tangible con flujo de datos mantiene el programa visible y compartido

<div class="grid grid-cols-2 gap-6 mt-6">
  <div class="box">
    <h3>Interfaz tangible de mesa</h3>
    <p>Se actúa y se ve en el mismo lugar</p>
    <p>Varios niños manipulan a la vez</p>
    <p class="cite mt-3">(Ishii, 2008; Suzuki y Kato, 1993)</p>
  </div>
  <div class="box">
    <h3>Flujo de datos</h3>
    <p>Programa = red de nodos</p>
    <p>Ejecuta la llegada de datos, no un orden de instrucciones</p>
    <p class="cite mt-3">(Wadge y Ashcroft, 1985)</p>
  </div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 35s

Dos conceptos más sostienen el diseño. El primero son las interfaces de usuario tangibles: según Ishii, dan forma física a la información digital, de modo que el objeto es a la vez control y representación, y el espacio donde el usuario actúa coincide con el espacio donde recibe la retroalimentación. En las interfaces de mesa, varios usuarios manipulan los objetos al mismo tiempo y el programa permanece visible. Suzuki y Kato, con AlgoBlock, identificaron los principios de los ambientes de programación tangible colaborativos: facilidad de uso, acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos.

El segundo es el paradigma de flujo de datos: un programa es una red de componentes por la que fluyen los datos, y lo que determina la ejecución es la disponibilidad de los datos, no una secuencia de instrucciones. Nuestro referente es Lucid, de Wadge y Ashcroft. La realidad aumentada espacial, que integra la proyección al entorno en lugar de llevarla sobre el usuario (Bimber y Raskar, 2005), une ambas cosas sobre la mesa.
-->
