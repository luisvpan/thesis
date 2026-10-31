---
layout: cover
class: text-center
---

<img src="/ucab-logo.png" alt="Logo de la Universidad Católica Andrés Bello" class="mx-auto mb-3" style="height: 40px" />

<div class="text-base font-semibold leading-snug">
Universidad Católica Andrés Bello<br />
Facultad de Ingeniería<br />
Escuela de Ingeniería Informática
</div>

<h1 class="!text-3xl !my-6 !leading-tight">Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años</h1>

<div class="text-base leading-snug">
<strong>Trabajo de Grado</strong> presentado ante la <strong>UNIVERSIDAD CATÓLICA ANDRÉS BELLO</strong><br />
como parte de los requisitos para optar al título de <strong>Ingeniero en Informática</strong>
</div>

<div class="grid grid-cols-[auto_auto] gap-x-6 gap-y-1 justify-center text-left text-base mt-6">
  <span class="font-semibold">Realizado por</span><span>Arzolay Rodríguez, Eduardo Javier Isidoro<br />Vásquez Paniagua, Luis Daniel</span>
  <span class="font-semibold">Tutor Académico</span><span>Lárez Mata, Jesús José</span>
  <span class="font-semibold">Fecha</span><span>Octubre, 2026</span>
</div>

<!--
PRESENTA: TBD · TIEMPO: 20s
PREGUNTA: La portada del informe calcula la fecha con el mes en curso (hoy rendería «Octubre, 2026») y el Resumen dice «Abril, 2026». ¿Ponemos el mes de la defensa?
PREGUNTA: ¿El logo de la UCAB se ve bien sobre fondo blanco? En el informe va sobre papel blanco, así que debería, pero hay que verlo proyectado.

Buenos días, profesores. Somos Eduardo Arzolay y Luis Vásquez, y presentamos nuestro Trabajo de Grado, «Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años», realizado bajo la tutoría del profesor Jesús Lárez. En unos 28 minutos veremos el problema, el marco teórico y metodológico, los resultados de cada objetivo específico, una demostración en video del ambiente funcionando y las conclusiones.
-->

---
eyebrow: 01 · El problema
---

# Enseñar pensamiento computacional hoy obliga a elegir entre pantallas y pocos conceptos

<div class="grid grid-cols-[1fr_auto_1fr_auto_1fr] gap-3 mt-6">
  <div class="box">
    <h3>Competencia básica</h3>
    <p>Programar: competencia del siglo XXI</p>
    <p class="cite mt-2">(Sánchez Vera, 2019)</p>
  </div>
  <div class="arrow">→</div>
  <div class="box plain">
    <h3>Pantallas: uso restringido</h3>
    <p>Exceso: menor rendimiento y atención</p>
    <p class="cite mt-2">(Munzer et al., 2026)</p>
    <p class="mt-2">Docente local: máx. 15 min seguidos</p>
  </div>
  <div class="arrow">→</div>
  <div class="box plain">
    <h3>Juegos tangibles</h3>
    <p>Pocos conceptos</p>
    <p>Poca colaboración</p>
  </div>
</div>

<div v-click class="mt-8 text-center text-2xl font-semibold">
¿Cómo ejercitarlo entre 6 y 9 años, en colaboración y sin pantallas?
</div>

<!--
PRESENTA: TBD · TIEMPO: 55s
PREGUNTA: La cita de la docente es una comunicación personal (J. Duarte, comunicación personal, 28 de febrero de 2025). APA no la lleva a las referencias, así que no está en references.bib. ¿La dejamos sin paréntesis en la lámina, como ahora, o ponemos «(J. Duarte, com. pers., 2025)»?
PREGUNTA: La AAP 2026 aparece en el informe como #cite(<aap2026>), que en APA se lee «Munzer et al., 2026». ¿Lo nombramos así en voz alta o como «la Academia Americana de Pediatría»?

Wing, en 2006, propuso que el pensamiento computacional se enseñe a todos, al nivel de la lectura, la escritura y la aritmética. Sánchez Vera, en 2019, señala que codificar ya figura entre las competencias básicas del siglo XXI, y Zapata-Ros advierte que, si estas bases no se construyen desde temprano manipulando objetos, a los estudiantes les cuesta desarrollarlo en ciclos superiores.

El problema es cómo enseñarlo a niños pequeños. Las herramientas más difundidas, como Scratch y ScratchJr, se usan en pantalla. La política más reciente de la Academia Americana de Pediatría asocia el uso excesivo de medios digitales entre los 6 y los 12 años con menor rendimiento académico y peor control de la atención. Aquí en Ciudad Guayana entrevistamos a la docente Jackeline Duarte, que enseña programación y robótica a niños en Multiplayer Pzo.: ella limita a 15 minutos los periodos continuos frente a la pantalla, lo que condiciona el uso de Scratch, y recurre a juegos tangibles como Mouse Mania, que según su experiencia cubren pocos conceptos y dificultan la colaboración entre los niños. Además, los énfasis curriculares de primaria en Venezuela no contemplan explícitamente el pensamiento computacional, así que depende de la iniciativa de cada docente.

[clic] De ahí nuestra pregunta de investigación: ¿cómo desarrollar un ambiente de programación tangible con realidad aumentada espacial que permita a niños entre 6 y 9 años, guiados por el docente, ejercitar el pensamiento computacional de forma colaborativa y sin exposición sostenida a pantallas?
-->

---
eyebrow: 01 · El problema · Objetivos
---

# Cinco objetivos específicos llevan del análisis a la documentación del ambiente

<div class="box mt-4">
  <h3>Objetivo general</h3>
  <p>Desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años.</p>
</div>

<div class="grid grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr] gap-2 mt-8 text-center">
  <div class="box plain"><h3>OE1</h3><p>Analizar referentes</p></div>
  <div class="arrow">→</div>
  <div class="box plain"><h3>OE2</h3><p>Diseñar</p></div>
  <div class="arrow">→</div>
  <div class="box plain"><h3>OE3</h3><p>Construir</p></div>
  <div class="arrow">→</div>
  <div class="box plain"><h3>OE4</h3><p>Validar</p></div>
  <div class="arrow">→</div>
  <div class="box plain"><h3>OE5</h3><p>Documentar</p></div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 45s
PREGUNTA: Los objetivos específicos van abreviados (verbo + objeto) para no pasar de 50 palabras; el texto completo se lee en las notas. ¿Les parece bien, o el jurado espera verlos completos en la lámina?

El objetivo general, aprobado por la Escuela, es desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años.

Para lograrlo nos planteamos cinco objetivos específicos, que son también el orden de los resultados que vamos a mostrar:
Primero, analizar el uso de programación tangible en entornos de realidad aumentada espacial, a fin de caracterizar el ambiente a desarrollar.
Segundo, diseñar el ambiente en función del análisis realizado.
Tercero, construirlo con base en el diseño realizado.
Cuarto, validar el ambiente construido.
Y quinto, realizar su documentación formal.
-->

---
eyebrow: 01 · El problema · Importancia
---

# El ambiente sirve a niños y docentes y contribuye al ODS 4

<div class="grid grid-cols-[1fr_2fr] gap-6 mt-6 items-start">
  <div class="ods-badge">
    <div class="text-4xl">ODS 4</div>
    <div class="text-xl">Educación de calidad</div>
    <div class="text-xl mt-3">Metas 4.4 y 4.6</div>
  </div>
  <div class="grid gap-4">
    <div class="box plain"><h3>Niños de 6 a 9 años</h3><p>Pensamiento computacional desde temprano</p></div>
    <div class="box plain"><h3>Docentes de 1.º a 3.º grado</h3><p>Recurso para guiar actividades en el aula</p></div>
    <div class="box"><h3>Novedad</h3><p>Tangible + flujo de datos + proyección, en una sola superficie</p></div>
  </div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 25s
PREGUNTA: El informe redacta la Meta 4.6 como «Promueve conocimientos básicos de aritmética en adultos, fomentando el aprendizaje desde la infancia a través de juegos interactivos». Dicho en voz alta, «en adultos» suena raro para un ambiente para niños. ¿Lo explicamos como base temprana para esa meta, como hace el informe, o cambiamos la redacción?

¿Por qué importa? Los beneficiarios directos son los niños de 6 a 9 años, de quienes se espera que desarrollen el pensamiento computacional desde temprano, y los docentes de primero a tercer grado, que ganan un recurso para conducir estas actividades en el aula, como guías.

El trabajo se alinea con el Objetivo de Desarrollo Sostenible 4, Educación de calidad: con la Meta 4.4, porque fomenta desde temprano el pensamiento lógico y computacional que forma parte de las competencias para el empleo, y con la Meta 4.6, porque promueve conocimientos básicos de aritmética mediante juegos desde la infancia.

La novedad, según el análisis de referentes que veremos, es reunir en una misma superficie la programación tangible con objetos y cartas, un lenguaje de flujo de datos y la retroalimentación proyectada: ninguno de los referentes comparados combina las tres.
-->

---
eyebrow: 01 · El problema · Alcance y limitaciones
---

# El trabajo llega hasta la validación con expertos; las pruebas con niños quedan fuera

<div class="grid grid-cols-2 gap-6 mt-6">
  <div class="box">
    <h3>Alcance</h3>
    <ul>
      <li><span class="yes">✓</span> Hardware y software del ambiente</li>
      <li><span class="yes">✓</span> Verificación y juicio de expertos</li>
      <li><span class="yes">✓</span> Manuales del sistema y de usuario</li>
      <li><span class="no">✗</span> Validación con niños</li>
    </ul>
  </div>
  <div class="box plain">
    <h3>Limitaciones</h3>
    <ul>
      <li>Sin experiencia previa en realidad aumentada espacial</li>
      <li>Kinect v2 con Python: librerías limitadas</li>
    </ul>
  </div>
</div>

<!--
PRESENTA: TBD · TIEMPO: 35s

El alcance comprende el diseño y la construcción del ambiente en sus dos aspectos: el hardware, que funciona como interfaz humano-computador, y el software que procesa lo que capta ese hardware. La validación consiste en verificar los requerimientos con una matriz de trazabilidad y en el juicio de dos expertos, uno en interacción humano-computador y otro en medios didácticos. Cerramos con el manual del sistema y el manual de usuario. Lo que queda fuera, de forma explícita, es la validación con niños: este trabajo se centra en el desarrollo de la herramienta.

Tuvimos dos limitaciones. La primera, que el equipo no tenía experiencia previa en realidad aumentada espacial, lo que nos obligó a iterar la calibración cámara-proyector y la detección de toques a lo largo de varios prototipos. La segunda, que las librerías para usar el Kinect v2 desde Python son limitadas: PyKinect2 fallaba con las versiones recientes de Python y libfreenect2 no detectaba el sensor, así que compilamos manualmente un controlador de OpenNI2 y, más adelante, adaptamos una bifurcación de PyKinect2. Por problemas de compatibilidad USB y de rendimiento, incluso volvimos temporalmente al Kinect v1 en el sexto prototipo.
-->
