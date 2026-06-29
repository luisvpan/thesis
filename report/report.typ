#set document(
  title: [Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años],
  author: ("Arzolay Rodríguez, Eduardo Javier Isidoro", "Vásquez Paniagua, Luis Daniel"),
  description: [Este trabajo de investigación se centra en el desarrollo de un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, con el objetivo de fomentar el desarrollo del pensamiento computacional desde edades tempranas. Se aborda la importancia del pensamiento computacional en la educación infantil, se analizan los desafíos asociados al uso de pantallas en niños pequeños, y se propone una solución innovadora que combina elementos físicos y digitales para crear una experiencia de aprendizaje interactiva y atractiva.],
  keywords: (
    "programación tangible",
    "realidad aumentada espacial",
    "pensamiento computacional",
    "niños",
    "aprendizaje",
    "dataflow",
  ),
  date: auto,
)

#set page(
  paper: "us-letter",
  margin: (x: 2.54cm, y: 2.54cm),
)

#let fontSize = 12pt
#let indent = 1.25cm

#set text(
  font: "Times New Roman",
  size: fontSize,
  lang: "es",
  region: "VE",
  hyphenate: false,
)

#let leading = 1.5em // Your line spacing (1, 1.5, 2, etc.)
#let leading = leading - 0.25em // "Normalization"
#set par(
  justify: true,
  leading: leading,
  spacing: leading,
)

#show title: set text(size: fontSize)
#show heading: set text(size: fontSize)
#show heading: set block(above: leading, below: leading)
#show heading.where(level: 1): set align(center)
#show heading.where(level: 2): set align(left)
#show heading.where(level: 3): it => pad(left: indent, [#it.body\.])
#show heading.where(level: 4): it => pad(left: indent, [_#it.body\._])
#show heading.where(level: 5): it => {
  set text(style: "italic", weight: "regular")
  pad(left: indent, it)
}

#set figure.caption(separator: [.])

#show figure: it => {
  let leading = 1em // Your line spacing (1, 1.5, 2, etc.)
  let leading = leading - 0.25em // "Normalization"
  set par(
    justify: false,
    leading: leading,
    spacing: leading,
  )

  it
}
#show figure.caption: it => align(start + top, it)

#let image-width = 80%
#show figure.where(kind: image): set image(width: image-width)

#show figure: fig => context {
  if fig.caption == none { return fig }

  // 1. Buscamos y extraemos el 'ref' inspeccionando la secuencia del caption de forma nativa
  let ref-element = none

  // En Typst una secuencia se puede descomponer usando selectores de filtrado sobre bloques o
  // mediante un mapeo directo de elementos en bloques de contenido.
  // El truco definitivo para extraer un elemento de una secuencia sin romper tipos es usar un show rule local invisible:
  let extracted = {
    show ref: it => {
      // Guardamos la referencia en una variable accesible (metadato)
      metadata((type: "found-ref", target: it.target, supplement: it.supplement))
      it
    }
    fig.caption.body
  }

  // Leemos el metadato generado localmente por el caption de ESTA figura
  let local-refs = query(selector(metadata).after(here())).filter(m => (
    type(m.value) == dictionary and m.value.at("type", default: "") == "found-ref"
  ))

  // Si el caption no ejecutó ningún ref, dibujamos la figura normal
  if local-refs.len() == 0 { return fig }

  let ref-data = local-refs.at(0).value
  let label = ref-data.target

  // Verificamos que sea una clave de bibliografía válida en nuestro archivo .bib
  let element = query(label).at(0, default: none)
  if element == none { return fig }

  // 2. Extraer metadatos de la obra de forma segura
  let title = element.at("title", default: "Sin Título")
  let year = if type(element.at("date", default: none)) == datetime { str(element.date.year()) } else {
    str(element.at("year", default: element.at("date", default: "s.f.")))
  }
  let publisher = element.at("publisher", default: none)
  let location = element.at("address", default: element.at("location", default: none))

  // 3. Formatear autores con Inicial + Apellido (Ej: M. Resnick)
  let authors-list = element
    .at("author", default: ())
    .map(a => {
      let initial = if a.at("given", default: "").len() > 0 { a.given.slice(0, 1) + "." } else { "" }
      if initial != "" { initial + " " + a.family } else { a.family }
    })

  let num-authors = authors-list.len()

  // 4. Historial de citación (Detección de citas subsecuentes)
  let native-before = query(selector(ref).before(here())).filter(r => r.target == label)
  let custom-before = query(selector(metadata).before(here())).filter(m => (
    type(m.value) == string and m.value == "img-cite-" + str(label)
  ))
  let is-subsequent = (native-before.len() + custom-before.len()) > 0

  let formatted-authors = ""
  if num-authors == 1 {
    formatted-authors = authors-list.at(0)
  } else if num-authors == 2 {
    formatted-authors = authors-list.at(0) + " y " + authors-list.at(1)
  } else if num-authors >= 6 or is-subsequent {
    formatted-authors = authors-list.at(0) + " et al."
  } else {
    let primary = authors-list.slice(0, -1).join(", ")
    let last = authors-list.at(-1)
    formatted-authors = primary + " y " + last
  }

  let meta-source = "."
  if location != none and publisher != none { meta-source = [, #location: #publisher.] } else if publisher != none {
    meta-source = [, #publisher.]
  }

  // Guardamos el marcador histórico global para el texto
  metadata("img-cite-" + str(label))

  // Extraemos la página
  let page-str = if ref-data.supplement != none { [#ref-data.supplement] } else { "p. X" }

  // 5. Construimos el nuevo cuerpo del caption con el formato exacto de la guía
  let new-caption-body = [Tomado de _#title _ (#page-str), por #formatted-authors, #year#meta-source]

  figure(
    fig.body,
    caption: figure.caption(new-caption-body, separator: fig.caption.separator),
    kind: fig.kind,
    supplement: fig.supplement,
    numbering: fig.numbering,
    placement: fig.placement,
  )
}

// Portada
#grid(
  align: center,
  gutter: 1fr,
  [#image("images/ucab-logo.png")
    *Universidad Católica Andrés Bello* \
    *Facultad de Ingeniería* \
    *Escuela de Ingeniería Informática*],
  title(),
  [*Trabajo de Grado* \
    presentado ante la \
    #upper[*Universidad Católica Andrés Bello*] \
    como parte de los requisitos para optar al título de \
    *Ingeniero en Informática*],
  grid.cell(align: start + top, grid(
    columns: (1fr, 1fr),
    gutter: leading,
    align: start + top,
    [Realizado por], [Arzolay Rodríguez, Eduardo Javier Isidoro \ Vásquez Paniagua, Luis Daniel],
    [Tutor Académico], [Lárez Mata, Jesús José],
    [Fecha], [Abril, 2026],
  )),
)

#pagebreak(weak: true)

// Dedicatoria
// = Dedicatoria
// A nuestras familias, a las grandes amistades que hicimos en la universidad, a todos los que no lo lograron, y al futuro que nos depara.

#pagebreak(weak: true)

// Agradecimientos
// = Agradecimientos
// Gracias a la escuela de Ingeniería Informática, por su apoyo y palabras de aliento en los momentos más difíciles.
// Gracias al profesor Jesús Lárez, por aceptar ser nuestro tutor, por su paciencia, regaños, correcciones, infinito saber y dedicación a la enseñanza.
// Gracias al profesor y director Franklin Bello, por siempre estar presente, guiandonos y alentandonos para que sigamos adelante.
// Gracias al señor Andrés, por las historias, las anécdotas, las enseñanzas y el constante apoyo y dedicación a todos los que día tras día estamos presentes y trabajando en el salón de prototipos.

#set par(
  first-line-indent: (amount: indent, all: true),
)

#pagebreak(weak: true)

#set page(
  footer: context {
    set align(center)

    let current-page = here().page()
    if current-page > 1 {
      counter(page).display("i")
    }
  },
)

#let leading = 1em // Your line spacing (1, 1.5, 2, etc.)
#let leading = leading - 0.25em // "Normalization"
#set par(
  justify: true,
  leading: leading,
  spacing: leading,
)

// Índice
#context {
  let headings = query(heading)
  let tables = query(figure.where(kind: table))
  let images = query(figure.where(kind: image))

  let indexables = (
    (list: headings, title: [Índice de Contenido], target: heading),
    (list: tables, title: [Índice de Tablas], target: figure.where(kind: table)),
    (list: images, title: [Índice de Figuras], target: figure.where(kind: image)),
  )

  for (list, title, target) in indexables {
    if list.len() > 0 [
      #outline(
        title: title,
        target: target,
      )
    ]
  }
}

#pagebreak(weak: true)

// Resumen
#align(center)[
  *Universidad Católica Andrés Bello* \
  *Facultad de Ingeniería* \
  *Escuela de Ingeniería Informática*

  #title()
]

#grid(
  columns: (auto, 1fr),
  gutter: 0.75em,
  align: start + top,
  [Autores:], [Arzolay Rodríguez, Eduardo Javier Isidoro \ Vásquez Paniagua, Luis Daniel],
  [Tutor Académico:], [Lárez Mata, Jesús José],
  [Fecha:], [Abril, 2026],
)

#align(center)[*Resumen*]

#[
  #set par(first-line-indent: 0cm)
  El pensamiento computacional es reconocido como una competencia básica del siglo XXI, cuyo desarrollo debe iniciarse desde edades tempranas como una nueva alfabetización. Sin embargo, las herramientas predominantes para fomentarlo en niños dependen del uso sostenido de pantallas, en tensión con las recomendaciones pediátricas sobre el tiempo de exposición, mientras que las alternativas tangibles tradicionales limitan la enseñanza de conceptos avanzados y el aprendizaje colaborativo. Este trabajo tuvo como objetivo desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, que fomente el desarrollo del pensamiento computacional y la colaboración reduciendo la dependencia de pantallas. El ambiente se fundamenta en el construccionismo, las interfaces de usuario tangibles y el paradigma de programación de flujo de datos. La investigación es de tipo proyectivo y siguió una metodología de desarrollo basada en prototipos, tomando como punto de partida el entorno de realidad aumentada espacial Magicboard. Como resultado se construyó un ambiente compuesto por una superficie interactiva con proyección; un subsistema de visión por computador que reconoce piezas tangibles tipo carta y toques sobre la superficie mediante modelos de detección de objetos y un sensor de profundidad; y el lenguaje de programación visual de flujo de datos ERAE con su intérprete, capaz de evaluar programas de manera incremental mientras se construyen. Los prototipos desarrollados muestran la viabilidad de integrar estos componentes en una experiencia de aprendizaje unificada.
]

_Palabras clave:_ programación tangible, realidad aumentada espacial, pensamiento computacional, lenguaje de flujo de datos, interfaces de usuario tangibles.

#pagebreak(weak: true)

#let leading = 1.5em // Your line spacing (1, 1.5, 2, etc.)
#let leading = leading - 0.25em // "Normalization"
#set par(
  justify: true,
  leading: leading,
  spacing: leading,
)

#counter(page).update(1)

#set page(
  header: context {
    let is-chapter = query(heading.where(level: 1))
      .filter(h => h.location().page() == here().page())
      .any(h => ("Capítulo" in h.body.text or "Introducción" in h.body.text))

    if not is-chapter {
      set align(right)
      counter(page).display()
    }
  },
  footer: auto,
)

// Introducción
= Introducción

La creciente demanda del pensamiento computacional como competencia básica del siglo XXI contrasta con las condiciones en que este puede desarrollarse durante la infancia: las herramientas más difundidas para su enseñanza, como Scratch, requieren del uso sostenido de pantallas, cuya exposición en niños pequeños está limitada por recomendaciones pediátricas; mientras que las alternativas desenchufadas y tangibles tradicionales dificultan la enseñanza de conceptos avanzados y el aprendizaje colaborativo. El presente Trabajo de Grado tiene como propósito desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, que combina piezas físicas tipo carta, proyección sobre una superficie compartida y un lenguaje de programación visual de flujo de datos, denominado ERAE, para fomentar el pensamiento computacional sin depender del uso sostenido de pantallas. El trabajo se fundamenta teóricamente en la teoría del desarrollo cognitivo de Piaget, el construccionismo de Papert, las interfaces de usuario tangibles y el paradigma de programación de flujo de datos; metodológicamente, se trata de una investigación proyectiva, desarrollada mediante un enfoque basado en prototipos sobre la base del entorno de realidad aumentada espacial Magicboard. El documento se organiza en cinco capítulos: el Capítulo I plantea el problema, los objetivos, el alcance, las limitaciones y la justificación; el Capítulo II presenta los antecedentes de investigación y las bases teóricas; el Capítulo III describe el marco metodológico; el Capítulo IV expone el desarrollo y los resultados, organizados por objetivo específico; y el Capítulo V recoge las conclusiones y recomendaciones; finalmente, se incluyen las referencias bibliográficas y los apéndices.

#pagebreak(weak: true)

// Capítulo I
= Capítulo I. El Problema
// TODO: Arreglar estilado.

== Planteamiento del Problema

#cite(<papert1980>, form: "prose") predijo el auge de las computadoras en la educación, planteando que los niños deberían aprender a programar tal y como es aprender francés viviendo en Francia en vez de aprenderlo mediante las clases de lenguas extranjeras en las aulas del colegio; es decir, mediante la interacción directa con las computadoras, un enfoque en que el niño use y experimente con la computadora para aprender, en vez de que la computadora le enseñe al niño. También hace énfasis en que la simple presencia de las computadoras cambiaría y moldearía una nueva forma de enseñar y aprender, inimaginable para la sociedad de aquel momento. Papert fue un visionario, pues lo que él defendía se hizo realidad en partes, con una creciente demanda de la competencia del pensamiento computacional, término que no usó Papert sino #cite(<wing2006>, form: "prose") para referirse a la forma de pensar de los científicos de computación; pero con retos que enfrentar.

Quien acuñó el término de pensamiento computacional, #cite(<wing2006>, form: "prose"), lo describe como una habilidad que todos deberían aprender y usar, no solo los científicos en computación. Defiende que debería de añadirse a la educación de los niños, al mismo nivel que las 3R (lectura, escritura y aritmética), por los usos que tiene, no solo al aprender a programar, sino en la descomposición y solución de problemas, la creación de modelos, el análisis de datos, la abstracción. Añade que su utilidad en otras disciplinas ya es visible, notándose en cómo el aprendizaje automático ha transformado a la estadística, el reciente interés de los científicos de computación en la biología, o la computación cuántica y su efecto en la física. El sueño de Papert, que todos tuvieran una computadora, se convirtió en lo que se conoce hoy como computación ubicua; Wing, por su parte, sueña que el pensamiento computacional sea igual de omnipresente, y propone que el primer paso es que deje de ser exclusivo de los científicos de computación, que se enseñe a los estudiantes preuniversitarios.

// ¿Por qué es importante el pensamiento computacional en niños? ->
// ¿Por qué es un problema que los niños no desarrollen el pensamiento computacional? ->
// Se debe mencionar que programar/codificar se ha convertido en una competencia básica del siglo XXI (habilidades del siglo XXI), según Sánchez Vera, et al. (2019).
// Después de mencionar que el pensamiento computacional se considera como una nueva alfabetización
// y que programar/codificar se ha convertido en una competencia básica del siglo XXI:
// Se debe mencionar que el pensamiento computacional se considera como una nueva alfabetización, la alfabetización digital (Zapata-Ros, 2015). Texto que referencia esto:

#cite(<zapata2015>, form: "prose") sostiene que el pensamiento computacional representa una nueva alfabetización digital que debe comenzar desde las primeras etapas del desarrollo individual, al igual que sucede con otras habilidades clave como las 3R. Esta alfabetización no se limita únicamente al aprendizaje de la programación, sino que permite a las personas organizar su entorno, desarrollar estrategias de desenvolvimiento y resolución de problemas cotidianos, además de organizar su mundo de relaciones en un contexto de comunicación más racional y eficiente, resultando en una mayor calidad de vida.

La ausencia del desarrollo del pensamiento computacional en los niños se ha convertido en un problema relevante en la sociedad actual. #cite(<sanchezvera2019>, form: "prose") señala que codificar ha sido incluido específicamente como una de las competencias básicas del siglo XXI, y que el pensamiento computacional permite desarrollar una nueva alfabetización necesaria en el mundo contemporáneo, ayudando a que los individuos no sean solo consumidores digitales, sino creadores y participantes activos con las tecnologías. La falta de estas competencias limita las capacidades de los niños para desenvolverse eficazmente en un contexto cada vez más digitalizado, reduciendo su potencial para resolver problemas complejos y expresar sus ideas mediante la tecnología. Además, como indica #cite(<zapata2015>, form: "prose"), la carencia de estas habilidades desde edades tempranas dificulta que en ciclos superiores los estudiantes puedan desarrollar plenamente el pensamiento computacional, ya que no cuentan con las bases cognitivas necesarias que se construyen mediante la manipulación de objetos y conceptos fundamentales como la seriación, la discriminación por propiedades y la secuenciación.

En la actualidad, la presencia de computadoras, tabletas, teléfonos, televisores y relojes inteligentes, y demás dispositivos con pantalla; resulta en que los seres humanos están expuestos a las pantallas durante todo el día, en períodos de tiempo extensos incluso en ambientes dedicados a la enseñanza y el aprendizaje, como los colegios y universidades; causando la preocupación generalizada por los efectos a corto y largo plazo de esto, especialmente en los niños. #cite(<aap2016>, form: "prose"), en representación de la Academia Americana de Pediatría, sostienen que el uso de pantallas en niños de 2 a 5 años no debe sobrepasar una hora al día, dividida en periodos cortos con descansos frecuentes. J. Duarte (comunicación personal, 28 de febrero de 2025) respalda a la AAP, añadiendo que los periodos continuos de exposición han de ser de 15 minutos como máximo, condicionando el uso de herramientas como Scratch para el desarrollo del pensamiento computacional durante sus sesiones de clase, y recurriendo en su lugar a herramientas tangibles, como el juego de mesa Mouse Mania, que presentan limitaciones para la enseñanza de conceptos avanzados, como los bucles y condicionales, y dificultan fomentar el aprendizaje colaborativo y la socialización entre niños.

La primera aproximación al uso de herramientas tangibles para la enseñanza de conceptos de programación a niños vino de parte de Radia Perlman, pionera en el área de las interfaces de usuario tangibles #cite(<morgado2006>), quien desarrolló entre 1974 y 1976 un lenguaje de programación tangible de sistemas, llamado TORTIS, con la finalidad de que los niños pudieran adquirir las competencias que traía el aprender lenguajes de programación completos al interactuar con objetos físicos. Este lenguaje consistía en controlar una pequeña “tortuga” (un disco equipado con una luz, una bocina y un lápiz, este último encargado de dibujar el resultado de la ejecución del programa) mediante uno de dos componentes: una serie de cajas de botones con acciones, o una “máquina tragacartas” con cartas de plástico. La razón detrás de la creación de dos componentes para interactuar con el lenguaje es, interpretando las palabras de Perlman, que con las cajas de botones los niños pensaban que el programa era el dibujo resultante, en vez del conjunto de comandos que ejecutaban con los botones; mientras que, con la máquina tragafichas, que construyó para solucionar el problema de las cajas, era difícil que lo niños entendieran que cada carta estaba asociada a un comando porque, para la ejecución de cada una, se debía buscar la carta, insertarla en una ranura de la máquina y presionar un botón. Incluso con estos problemas, Perlman llegó a una nueva aproximación para la enseñanza de conceptos de programación para los niños.

Un ejemplo más conocido de programación tangible es el caso de AlgoBlock #cite(<suzuki1993>), un lenguaje de programación tangible inspirado en Logo que consiste en unir una serie de bloques físicos para formar un programa que controla un submarino mostrado en una pantalla. Igual que las cartas en el segundo componente de TORTIS, cada bloque representa un comando, y algunos bloques representan estructuras de control condicionales y de bucle. Lo interesante de AlgoBlock son los principios que siguieron sus autores en su desarrollo: facilidad de uso, acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos; los cuales incitan la conversación y la colaboración entre los participantes. Aunque los autores no mencionan alguna deficiencia en AlgoBlock tras ponerlo a prueba, sí hacen énfasis en que este solo representa el primer paso en la identificación de los principios para el diseño de ambientes de aprendizaje colaborativo.

La idea detrás de la programación tangible es llevada más allá por #cite(<zapata2019>, form: "prose"), quien habla sobre el pensamiento computacional desenchufado, que ayuda a que los niños adquieran las competencias relacionadas al pensamiento computacional en las etapas de su vida en que más la necesiten, como durante los estudios secundarios o la universidad, basándose en los principios fundamentales de la instrucción propuestos por #cite(<merrill2002>, form: "prose"), especialmente en el principio de activación. Zapata-Ros describe al pensamiento computacional desenchufado como actividades que fomenten en los niños una serie de habilidades que, tras ser evocadas en ciclos superiores, favorezcan el desarrollo del pensamiento computacional, ejemplos de estas son el uso de fichas, juegos en el salón de clase o en el patio, juguetes, aquellas que se suelen hacer sin el uso de pantallas ni computadoras. Además, sugiere actividades ya existentes con estas características y la forma de usarlas para promover el pensamiento computacional.

Aparte del pensamiento computacional desenchufado, la realidad aumentada espacial también presenta similitudes con la programación tangible, al permitir la superposición del mundo real con el virtual #cite(<park2015>), de cierta manera siendo la evolución de TORTIS y AlgoBlock; y, al igual que el pensamiento computacional desenchufado, se ha utilizado la realidad aumentada espacial para fomentar el desarrollo del pensamiento computacional en niños. #cite(<billinghurst2012>, form: "prose") evaluaron el uso de realidad aumentada espacial en aulas de clase, específicamente en forma de libros aumentados (libros impresos en los que algunas de sus páginas tienen imágenes virtuales superpuestas) y aplicaciones móviles de realidad aumentada, siendo los resultados recopilados del uso de los libros aumentados los de mayor interés para esta investigación, destacando la utilidad en la enseñanza de conceptos espaciales al proveer una interacción intuitiva, atractiva y natural de los temas estudiados; apoyando en la retención de contenidos que se desenvuelven en secciones de realidad aumentada, permitiendo que niños con dificultades en la comprensión de textos de aprendizaje tengan la opción de interactuar con el contenido para aprender; y en el desarrollo del pensamiento computacional al hacer uso de herramientas de construcción de escenas de realidad aumentada espacial diseñadas para niños, quienes se inspiran en hacer sus propias escenas tras ver y experimentar con las de los libros aumentados. Además, como los libros en sí contienen secciones físicas intercaladas con secciones de realidad aumentada, se reducía el tiempo de uso de pantallas.

Pasando a Venezuela, #cite(<barrios2024>, form: "prose") de la Universidad Católica Andrés Bello (UCAB) presentó un entorno de realidad aumentada espacial que consiste en una mesa interactiva táctil, la cual permite el desarrollo de juegos sociales entre niños de educación preescolar y básica que promueven la colaboración y socialización. Con base en lo expuesto previamente, se ve la oportunidad de extender el entorno hecho por Barrios incluyendo un módulo de programación tangible, aprovechando el enfoque en juegos sociales para fomentar la colaboración, la socialización, el trabajo en equipo y el pensamiento computacional.

En este contexto, se propone desarrollar un ambiente de programación tangible con realidad aumentada espacial basado en los referentes mencionados, con la finalidad de promover el desarrollo significativo del pensamiento computacional desde edades tempranas, mediante un enfoque que promueva el aprendizaje colaborativo y un desarrollo integral que vaya más allá del cognitivo.

=== Objetivo General

Desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años.

=== Objetivos Específicos

+ Analizar el uso de programación tangible en entornos de realidad aumentada, a fin de caracterizar el ambiente a desarrollar.
+ Diseñar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, en función del análisis realizado.
+ Construir un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, en base al diseño realizado.
+ Validar el ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.
+ Realizar la documentación formal del ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.

== Alcance

El presente trabajo tuvo como objetivo desarrollar un ambiente de programación tangible para fomentar el desarrollo del pensamiento computacional en niños entre 6 y 9 años de edad. Este estuvo orientado a fomentar el desarrollo del pensamiento computacional de los niños, considerando aspectos esenciales como el aprendizaje colaborativo, la socialización y la inclusividad.

En primer lugar, se llevó a cabo una etapa de revisión de conceptos para obtener detalles sobre el pensamiento computacional y su desarrollo en edades tempranas. A continuación, se realizó un análisis exhaustivo para comprender la aplicación de la programación tangible en entornos de realidad aumentada, con el fin de fomentar el pensamiento computacional.

Posteriormente, se procedió al diseño y desarrollo del entorno, que incluye dos aspectos: la construcción del hardware, que funcionó como interfaz de interacción humano-computador, y el desarrollo del software, encargado de procesar la información recibida a través del hardware. Adicionalmente, se propuso un protocolo de pruebas que se aplicó durante la validación posterior a la construcción del entorno.

Finalmente, se elaboró la documentación correspondiente, incluyendo el manual del sistema y el manual de usuario.

== Limitaciones

=== Dificultades Asociadas a Nuevas Tecnologías

Surgieron inconvenientes durante el desarrollo y construcción del entorno debido a la falta de experiencia en realidad aumentada espacial.

=== Problemas Asociados a los Componentes Utilizados

Si bien se logró obtener la mayoría de los componentes necesarios para el desarrollo del entorno, algunos de ellos presentaron dificultades para su integración dentro del sistema.

== Justificación

Este trabajo de investigación apoya al desarrollo del pensamiento computacional, el aprendizaje colaborativo y la socialización entre los niños, fomentando la futura activación de estas competencias en su crecimiento personal, académico y profesional, y convirtiéndose en una herramienta útil y una gran alternativa para educadores conscientes en que los niños deben aprender sobre el mundo virtual sin sacrificar el mundo físico, con el añadido de combinar ambos mundos en una experiencia única.

=== Aporte Principal

El uso del ambiente de programación tangible permite que los niños aprendan de forma lúdica, creando experiencias significativas para el aprendizaje del pensamiento computacional a través de la manipulación de objetos físicos, reflejada en la visión por computador. Además, promueve el aprendizaje colaborativo, ayudándoles a desarrollar el pensamiento computacional.

=== Innovación

Se presenta una propuesta que permite a los niños acceder a un ambiente de aprendizaje interactivo para aprender a programar con componentes tanto físicos como digitales. Este enfoque equilibra el uso de pantallas y las técnicas tradicionales de aprendizaje mediante la utilización de elementos físicos y digitales.

=== Beneficiarios

==== Niños entre 6 y 9 años de edad

Fomenta el desarrollo del pensamiento computacional en los niños desde edades tempranas, lo cual puede influir positivamente en su rendimiento académico y en su habilidad para resolver problemas lógicos.

==== Profesores de primeros grados de educación básica

Cuentan con una herramienta útil que facilita el proceso de enseñanza-aprendizaje a niños mediante una experiencia interactiva.

=== Impacto en los Objetivos de Desarrollo Sostenible

Esta investigación tiene un impacto significativo en el Objetivo 4 (Educación de calidad) de los Objetivos de Desarrollo Sostenible, especialmente en las Metas 4.4 y 4.6:

==== Meta 4.4.

Aumenta el número de jóvenes y adultos con competencias necesarias para acceder al empleo, fomentando desde edades tempranas el desarrollo del pensamiento lógico y computacional.

==== Meta 4.6.

Promueve conocimientos básicos de aritmética en adultos, fomentando el aprendizaje desde la infancia a través de juegos interactivos.

#pagebreak(weak: true)

// Capítulo II
= Capítulo II. Marco Teórico

== Antecedentes de Investigación

=== Entorno de robótica educativa multiagente orientado a favorecer el desarrollo del pensamiento computacional en jóvenes cursantes de educación media.

Trabajo de grado presentado por #cite(<espejo2022>, form: "prose") ante la Universidad Católica Andrés Bello, cuyo objetivo general fue desarrollar un entorno de robótica educativa multiagente orientado a favorecer el desarrollo del pensamiento computacional en jóvenes cursantes de educación media. Se empleó una metodología basada en el modelo espiral, a través de la cual se definieron los requisitos y se diseñó el entorno. Este entorno se encarga de obtener información del mundo físico mediante visión por computadora y procesar dicha información para seguir las instrucciones programadas en MakeCode.

Su aporte a este trabajo es que demuestra que se puede usar la robótica multiagente para favorecer el desarrollo del pensamiento computacional en jóvenes, de modo que se toma una nueva visión del problema de “fomentar el desarrollo del pensamiento computacional” pero para niños entre 6 y 9 años y mediante el uso de programación tangible en vez de robótica multiagente.

=== Entorno de realidad aumentada espacial para el desarrollo de juegos sociales dirigidos a niños de educación preescolar.

Trabajo de grado presentado por #cite(<barrios2024>, form: "prose") ante la Universidad Católica Andrés Bello, cuyo objetivo general fue desarrollar un entorno de realidad aumentada espacial para el desarrollo de juegos sociales dirigidos a niños de educación preescolar. Esta tesis marca un hito en los entornos de realidad aumentada espacial, presentando un producto llamado “Magicboard”. En este, los niños pueden interactuar con una pizarra digital en forma de mesa a través de un sensor que detecta gestos y objetos físicos. Este producto permite a los niños aprender mediante juegos sociales que tienen como pilar el aprendizaje colaborativo.

En este caso, se llevó a cabo una etapa de investigación en la que se obtuvieron las características esenciales para la construcción de la pizarra, considerando la manera en que los niños interactúan y aprenden. Posteriormente, se utilizó un videobeam, un sensor Kinect y el software correspondiente para la gestión de la interacción.

Este trabajo representa una continuación del producto resultante de la tesis de Barrios, buscando añadir a “Magicboard” la fomentación del desarrollo del pensamiento computacional a través de la programación tangible.

=== Sistema interactivo para la enseñanza de programación a niños con discapacidad visual.

Trabajo de grado presentado por #cite(<rojas2024>, form: "prose") ante la Universidad Católica Andrés Bello, cuyo objetivo general fue desarrollar un sistema interactivo para la enseñanza de programación a niños con discapacidad visual. El sistema permite que los niños con discapacidad visual utilicen bloques físicos para programar. Estos bloques cuentan con características táctiles que permiten a las personas con discapacidad visual leer y entender el significado de cada bloque. De esta manera, pueden construir una secuencia de instrucciones que es posteriormente procesada por un sistema de visión por computadora. Este sistema analiza la conexión entre los bloques y ejecuta la secuencia construida.

Lo más relevante de este estudio para la propuesta son los bloques físicos, ya que representan una interfaz de programación tangible; este sistema propuesto por Rojas y Youssef se puede tomar como un punto de partida para prototipos del ambiente de programación tangible que se desarrollaría en este trabajo.

=== Can computational thinking be improved by using a methodology based on metaphors and scratch to teach computer programming to children? [¿Se puede mejorar el pensamiento computacional mediante el uso de una metodología basada en metáforas y Scratch para enseñar programación a los niños?]

Artículo de investigación publicado por #cite(<perezmarin2020>, form: "prose"), investigadores de la Universidad Rey Juan Carlos (España), cuyo objetivo fue responder a la pregunta sobre si el pensamiento computacional puede mejorarse mediante una metodología basada en metáforas y el uso de Scratch para enseñar programación a los niños. Para ello, se llevaron a cabo experimentos y métodos de evaluación para analizar cómo los niños aprenden. Se utilizaron herramientas como Scratch y la aplicación CompThink como medios para evaluar el aprendizaje de los niños.

El trabajo se estructura en diversas secciones, cada una abordando temas específicos. En primer lugar, se presenta la sección de "Contexto", donde se expone el concepto de pensamiento computacional y las formas en que puede enseñarse la programación en la educación primaria. A continuación, se detalla la sección de "Materiales y Métodos", describiendo los procedimientos del experimento realizado. Posteriormente, se muestran los Resultados obtenidos y las Conclusiones finales, que incluyen posibles líneas de trabajo futuras.

Este estudio permite demostrar que herramientas como Scratch, el cual está basado en programación en bloques, son útiles para la enseñanza de la programación a los niños, lo cual representa un punto de partida para prototipos del ambiente de programación tangible que se desarrollaría en este trabajo.

=== Using an online serious game to teach basic programming concepts and facilitate gameful experiences for high school students [Usando un juego serio en línea para enseñar conceptos básicos de programación y facilitar experiencias divertidas para estudiantes de secundaria]

Artículo de investigación publicado por #cite(<montes2021>, form: "prose"), de la Universidad Rey Juan Carlos (España), cuyo objetivo fue utilizar un juego serio en línea para facilitar el aprendizaje de conceptos básicos de programación en estudiantes de secundaria, proporcionando a la vez una experiencia de juego satisfactoria. El experimento se llevó a cabo con 38 estudiantes de K-10.

Los resultados demostraron que los niños tuvieron una experiencia positiva, concluyendo que el uso de juegos incrementó sus puntuaciones de aprendizaje.

Este estudio sirve como base para la propuesta, al inspirar el uso de “Juegos Serios” como herramienta clave para fomentar el aprendizaje de conceptos de programación de una manera lúdica y atractiva.

== Bases Teóricas

=== Desarrollo Cognitivo y Aprendizaje en Niños de 6 a 9 Años

Según la teoría de desarrollo cognitivo de #cite(<piaget1969>, form: "prose"), existen cuatro etapas principales en el desarrollo cognitivo: la etapa sensorio-motora, que va desde el nacimiento hasta aproximadamente los dos años, y donde el conocimiento se obtiene mediante la interacción física con el entorno inmediato; la etapa preoperacional, aproximadamente entre los dos y siete años, caracterizada por el egocentrismo y la dificultad para realizar operaciones mentales complejas, aunque ya se comienza a ganar la capacidad de utilizar objetos simbólicos y adoptar roles ficticios; la etapa de las operaciones concretas, aproximadamente entre los siete y doce años, en la que se empieza a usar la lógica para llegar a conclusiones válidas con situaciones concretas y los sistemas de categorías se vuelven más complejos; y, finalmente, la etapa de las operaciones formales, desde los doce años en adelante, donde se desarrolla la capacidad para utilizar la lógica con conceptos abstractos y el razonamiento hipotético-deductivo.

La etapa de desarrollo cognitivo que corresponde a niños entre 6 y 9 años se encuentra en una transición crucial: a los 6 años aún están en la etapa preoperacional, donde el egocentrismo sigue presente y el pensamiento mágico basado en asociaciones simples predomina; sin embargo, alrededor de los 7 años, acceden a la etapa de las operaciones concretas, donde comienzan a usar la lógica para situaciones concretas y el egocentrismo disminuye notablemente. Esta transición marca un período fundamental de desarrollo donde los niños empiezan a manipular información lógicamente, siendo capaces de realizar operaciones mentales sobre objetos concretos, pero aún tienen limitaciones para trabajar con conceptos abstractos puros. Esta característica es fundamental para el diseño de ambientes de programación tangible, ya que la manipulación física de objetos permite a los niños comprender conceptos abstractos de programación a través de la experiencia concreta. El aprendizaje mediante la manipulación física de objetos se corresponde con lo que #cite(<bruner1966>, form: "prose") denomina representación enactiva: el modo de conocimiento en el que la comprensión se construye a través de la acción directa sobre el entorno. Este enfoque resulta especialmente pertinente en niños de edades tempranas, en quienes las actividades concretas y manipulables constituyen una vía eficaz para introducir conceptos computacionales #cite(<zapata2019>). La interacción con objetos tangibles permite que los niños comprendan conceptos abstractos mediante la experiencia sensorial y motora, facilitando la internalización de conocimientos complejos.

El constructivismo de #cite(<piaget1969>, form: "prose") postula que el conocimiento se construye activamente a través de la interacción entre el individuo y su entorno, mediante procesos de asimilación y acomodación. Según esta perspectiva, los niños no son receptores pasivos de información, sino que construyen activamente su comprensión del mundo a partir de sus experiencias y estructuras cognitivas preexistentes. El aprendizaje ocurre cuando los niños interactúan con el entorno y reorganizan sus esquemas mentales para incorporar nueva información. #cite(<papert1980>, form: "prose"), influido por Piaget, con quien colaboró directamente, extendió el constructivismo hacia su propia teoría del aprendizaje, a la que denominó construccionismo #cite(<sanchezvera2019>), enfatizando que el aprendizaje es significativamente más efectivo cuando los estudiantes no solo construyen conocimiento mentalmente, sino que construyen objetos tangibles o digitales concretos que pueden ser compartidos, discutidos, observados y refinados con otros. El construccionismo sugiere que el aprendizaje ocurre de manera más significativa cuando los estudiantes están activamente involucrados en la construcción de algo externo y compartible, estableciendo un ciclo de reflexión-acción-reflexión donde el objeto construido sirve como vehículo para pensar y aprender. Esta perspectiva se alinea con el enfoque de programación tangible, donde los niños construyen programas físicos que pueden ser ejecutados, observados, modificados y compartidos, creando una experiencia de aprendizaje profunda y significativa que trasciende la mera transmisión de conceptos.

=== Pensamiento Computacional en Edades Tempranas

Wing (#cite(<wing2006>, form: "year"), #cite(<wing2008>, form: "year"), #cite(<wing2011>, form: "year")) define el pensamiento computacional como un conjunto de habilidades que incluyen la formulación de problemas de manera que permita el uso de computadoras para resolverlos, la organización y análisis lógico de información, la representación mediante abstracciones, y la búsqueda de soluciones efectivas. Si bien no existe consenso sobre cómo incorporar el pensamiento computacional en el aula, #cite(<sanchezvera2019>, form: "prose") identifica una tendencia creciente a valorar su trabajo desde la educación infantil y primaria, niveles en los que su desarrollo suele abordarse de manera transversal. Para niños de 6 a 9 años, estos componentes deben adaptarse al nivel de desarrollo cognitivo propio de la etapa de las operaciones concretas #cite(<piaget1969>), utilizando representaciones concretas y manipulables. El desarrollo del pensamiento computacional en niños requiere de estrategias pedagógicas específicas que consideren las limitaciones cognitivas propias de la edad; en esa línea, el pensamiento computacional desenchufado #cite(<zapata2019>) y los lenguajes de programación tangible #cite(<morgado2006>)#cite(<suzuki1993>) muestran que es posible experimentar con conceptos computacionales sin requerir habilidades de lectura avanzadas ni conocimiento previo de la sintaxis de lenguajes de programación tradicionales.

La sociedad digital contemporánea demanda profesionales cualificados en industrias tecnológicas, lo que ha llevado a reconocer la necesidad del pensamiento computacional, que, para #cite(<zapata2015>, form: "prose"), es una nueva alfabetización: la alfabetización digital. Como cualquier alfabetización fundamental, debe iniciarse desde las primeras etapas del desarrollo individual. Sin embargo, la codificación es solo la manifestación más visible de una forma de pensar que trasciende el ámbito de la programación: una manera particular de organizar ideas y representaciones que favorece las competencias computacionales y que puede cultivarse desde edades tempranas mediante actividades y entornos de aprendizaje apropiados.

#cite(<zapata2019>, form: "prose") propone el pensamiento computacional desenchufado como un enfoque que permite a los niños desarrollar competencias computacionales sin el uso de pantallas o computadoras. Las actividades desenchufadas preparan a los niños para conceptos que serán evocados en ciclos superiores de aprendizaje, estableciendo una base sólida que facilitará la transición hacia herramientas más avanzadas cuando estén cognitivamente preparados. Este enfoque se basa en los principios fundamentales de la instrucción propuestos por #cite(<merrill2002>, form: "prose"), que incluyen:

- *Enfoque centrado en tareas o problemas*: Diseño de instrucción alrededor de problemas auténticos del mundo real que los aprendices probablemente encontrarán, fomentando la participación activa en actividades de resolución de problemas.

- *Activación*: Se enfoca en involucrar los conocimientos previos y experiencias de los aprendices para crear una base para el nuevo aprendizaje. Implica estimular la curiosidad, presentar ejemplos del mundo real y conectar nueva información con conocimientos existentes.

- *Demostración*: Proporcionar modelos o ejemplos claros que ilustren los resultados de aprendizaje deseados, permitiendo que los aprendices observen actuaciones de expertos, simulaciones o estudios de caso para desarrollar comprensión.

- *Aplicación*: Ofrecer oportunidades para que los aprendices practiquen y apliquen sus conocimientos y habilidades en contextos auténticos, requiriendo que resuelvan problemas, tomen decisiones y se involucren en tareas realistas.

- *Integración*: Promover la transferencia de conocimientos y habilidades a nuevas situaciones, proporcionando oportunidades para conectar el aprendizaje con contextos del mundo real y aplicarlo de manera significativa.

El principio de activación es particularmente crucial para este trabajo, ya que este permite construir sobre experiencias previas con objetos físicos, juegos y manipulaciones concretas que los niños ya comprenden. Al activar conocimientos previos sobre manipulación de objetos, secuencias de acciones y relaciones causa-efecto que los niños han experimentado en sus actividades cotidianas, se establece un puente cognitivo fundamental que facilita la transición hacia conceptos computacionales abstractos a través de la experiencia concreta, haciendo que el aprendizaje sea más accesible y significativo.

Los principios fundamentales de instrucción, se integran dentro del marco más amplio del diseño instruccional #cite(<merrill2002>), que se refiere al proceso sistemático de planificar, desarrollar, implementar y evaluar experiencias de aprendizaje efectivas. Este campo interdisciplinario combina teorías de aprendizaje, metodologías pedagógicas y principios de diseño para crear entornos educativos que faciliten la adquisición de conocimientos y habilidades; en esa línea, #cite(<ackermann2010>, form: "prose") examina las raíces compartidas y las diferencias entre el constructivismo de Piaget y el construccionismo de Papert como fundamentos de este tipo de entornos. El diseño instruccional implica la identificación de objetivos de aprendizaje, la selección de estrategias pedagógicas apropiadas, la organización de contenido, y el diseño de actividades y evaluaciones que promuevan aprendizajes efectivos y duraderos. También se derivan las estrategias didácticas, que son los métodos y técnicas específicos que se emplean para facilitar el proceso de enseñanza-aprendizaje #cite(<ertmer1993>). Estas estrategias se adaptan según el contexto, los objetivos de aprendizaje y las características de los aprendices. Las estrategias didácticas pueden diseñarse intencionalmente para alinearse con los principios de instrucción y maximizar la efectividad del proceso educativo #cite(<ertmer1993>), especialmente cuando se trata de conceptos abstractos que deben ser comprendidos a través de manipulaciones concretas.

=== Teoría del aprendizaje

Las teorías del aprendizaje proporcionan el marco conceptual fundamental para comprender cómo los individuos adquieren, procesan y retienen conocimiento #cite(<schunk2012>). El aprendizaje puede entenderse como un proceso activo de construcción de conocimiento que ocurre a través de la interacción entre el individuo y su entorno #cite(<woolfolk2014>) #cite(<piaget1978>). Diversas perspectivas teóricas han contribuido a la comprensión de este proceso complejo: el conductismo enfatiza la modificación de comportamientos mediante estímulos y refuerzos, el cognitivismo se centra en los procesos mentales internos y la organización del conocimiento, mientras que el constructivismo postula que el conocimiento se construye activamente mediante la experiencia y la interacción con el entorno.

El aprendizaje colaborativo se fundamenta en teorías socioconstructivistas que, como expone #cite(<ackermann2010>, form: "prose"), enfatizan el papel de la interacción social y de la mediación cultural en el aprendizaje. La colaboración permite que los niños aprendan de sus pares, desarrollen habilidades de comunicación, y construyan conocimiento colectivamente mediante la discusión y negociación de significados. El aprendizaje colaborativo en niños promueve el desarrollo de habilidades sociales, mejora la comprensión mediante la explicación a otros, fomenta el pensamiento crítico a través de la discusión, y desarrolla habilidades de trabajo en equipo. En el ámbito de la programación tangible, estos beneficios se reconocieron de forma temprana: #cite(<suzuki1993>, form: "prose") concibió AlgoBlock como una herramienta de aprendizaje colaborativo en la que la manipulación compartida de bloques físicos promueve la interacción entre pares y la resolución conjunta de problemas.

En el contexto educativo, la selección y diseño de recursos, materiales y medios didácticos adecuados es fundamental para facilitar el proceso de aprendizaje #cite(<area2009>). Estos elementos constituyen herramientas estratégicas que, cuando se diseñan y utilizan apropiadamente, pueden potenciar significativamente la efectividad del proceso educativo.

Un recurso didáctico #cite(<area2009>) es un elemento más amplio y general que incluye cualquier elemento, estrategia o herramienta que puede ser utilizada con fines educativos. Los recursos didácticos abarcan desde objetos físicos hasta estrategias metodológicas, entornos, personas y experiencias que pueden contribuir al proceso de enseñanza-aprendizaje. Son elementos flexibles que pueden ser adaptados y reutilizados en diferentes contextos educativos.

Un material didáctico #cite(<area2009>) es un recurso didáctico específico y concreto, generalmente físico o digital, que ha sido diseñado intencionalmente con un propósito educativo explícito. Los materiales didácticos son objetos tangibles o digitales que contienen información estructurada y organizada para facilitar el aprendizaje. Poseen características específicas como objetivos de aprendizaje definidos, contenidos organizados pedagógicamente, y diseño que facilita su uso educativo.

Un medio didáctico #cite(<area2009>) es el soporte o tecnología que representa y vehicula el contenido educativo mediante determinadas formas y sistemas de codificación, y que no constituye un mero canal de transmisión, sino que estructura el propio proceso de aprendizaje. Los medios didácticos determinan cómo se presenta la información al aprendiz. Diferentes medios (visual, auditivo, táctil, kinestésico) activan diferentes canales sensoriales y cognitivos, influyendo en cómo se procesa y retiene la información. La realidad aumentada espacial, por ejemplo, constituye un medio didáctico que presenta información visual superpuesta sobre el mundo físico, activando canales visuales y espaciales para facilitar la comprensión de conceptos abstractos mediante representaciones visuales concretas.

=== Programación y Entornos de Desarrollo

Los entornos de desarrollo constituyen el conjunto de herramientas, bibliotecas y configuraciones que facilitan la creación, edición, depuración y ejecución de programas computacionales. Entre estos entornos, los Entornos de Desarrollo Integrados (IDE) representan aplicaciones que combinan múltiples herramientas de desarrollo en una interfaz unificada, incluyendo editores de código con resaltado de sintaxis, depuradores, compiladores, sistemas de control de versiones y gestores de proyectos. La programación es el proceso de diseñar y construir programas computacionales que ejecutan tareas específicas mediante la escritura de código en lenguajes de programación. Cuando se trata de programación para niños, especialmente en edades tempranas, es necesario adaptar estos entornos a las capacidades cognitivas y motoras de los aprendices. Las tecnologías educativas, entendidas como el conjunto de herramientas, recursos y metodologías que integran tecnología digital en procesos de enseñanza-aprendizaje, han demostrado ser efectivas para facilitar el aprendizaje cuando se diseñan apropiadamente. El uso de medios digitales en niños requiere consideraciones especiales: mientras que los medios digitales pueden ofrecer interactividad, retroalimentación inmediata y representaciones visuales atractivas, también presentan desafíos relacionados con la atención, la sobrecarga cognitiva y la necesidad de mantener el equilibrio entre la estimulación digital y el desarrollo de habilidades físicas y sociales. La programación para niños debe considerar diferentes paradigmas de programación, que son estilos o enfoques fundamentales para estructurar y organizar código.

Un lenguaje de programación es un sistema formal de comunicación que permite a los programadores expresar instrucciones y algoritmos de manera estructurada para que sean ejecutados por una computadora. Los lenguajes de programación proporcionan un conjunto de reglas sintácticas y semánticas que definen cómo se pueden combinar símbolos y palabras clave para crear programas funcionales. Los paradigmas de programación representan estilos o enfoques fundamentales para estructurar y organizar código, definiendo la forma en que los programadores conceptualizan y resuelven problemas computacionales. El paradigma imperativo es uno de los paradigmas más fundamentales, donde los programas se estructuran como secuencias de instrucciones que modifican el estado del programa mediante asignaciones y comandos. En este paradigma, el programador especifica explícitamente los pasos que la computadora debe seguir para resolver un problema, controlando el flujo de ejecución mediante estructuras de control como bucles y condicionales.

El dataflow es un paradigma de programación donde el flujo de datos determina la ejecución del programa, en lugar de un flujo de control secuencial. En este paradigma, las operaciones se ejecutan cuando sus datos de entrada están disponibles, creando un modelo de programación basado en la transformación y el flujo de información a través de una red de operaciones.

El lenguaje de programación tangible es un enfoque donde los elementos del lenguaje de programación se representan mediante objetos físicos manipulables, permitiendo a los usuarios construir programas mediante la organización espacial y física de estos objetos, eliminando la necesidad de sintaxis textual y facilitando la comprensión de conceptos computacionales a través de la manipulación concreta.

=== Programación Tangible

La programación tangible se basa en el concepto de interfaces tangibles de usuario (TUI), donde los usuarios interactúan con sistemas computacionales mediante la manipulación de objetos físicos. En el contexto de la programación, cada objeto físico representa un comando o estructura de control, permitiendo a los usuarios construir programas mediante la organización física de estos objetos. La programación tangible tiene sus raíces en trabajos pioneros como TORTIS de Perlman #cite(<morgado2006>) y AlgoBlock de #cite(<suzuki1993>, form: "prose"). Estos sistemas demostraron que los niños pueden aprender conceptos de programación mediante la manipulación de objetos físicos, estableciendo los principios fundamentales que guían el diseño de sistemas de programación tangible contemporáneos.

#cite(<suzuki1993>, form: "prose") identificaron principios clave para el diseño de ambientes de programación tangible colaborativos: facilidad de uso, acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos. Estos principios fomentan la conversación y colaboración entre participantes, aspectos esenciales para el aprendizaje colaborativo. La programación tangible ofrece varios beneficios pedagógicos: reduce la barrera de entrada al eliminar la necesidad de sintaxis textual, permite la visualización espacial de estructuras de control, facilita la colaboración mediante el acceso simultáneo a objetos físicos, y proporciona retroalimentación inmediata mediante la ejecución del programa construido. Estos beneficios hacen que la programación tangible sea especialmente adecuada para niños en la etapa operacional concreta, donde la manipulación física facilita la comprensión de conceptos abstractos. La integración de tecnologías de visualización que permitan superponer información virtual directamente sobre los objetos físicos puede potenciar estos beneficios, facilitando la retroalimentación visual inmediata y la comprensión de las relaciones entre los elementos del programa sin interrumpir la interacción directa con el espacio físico.

#cite(<park2015>, form: "prose") definen la realidad aumentada espacial como una tecnología que superpone contenido virtual sobre el mundo real mediante proyección directa sobre superficies físicas, a diferencia de la realidad aumentada tradicional que requiere dispositivos intermediarios como pantallas. Esta tecnología permite la interacción directa con el espacio físico, que se refleja en el espacio virtual. La realidad aumentada espacial se diferencia de la realidad aumentada tradicional en que no requiere dispositivos intermediarios (como tablets o smartphones) para visualizar el contenido aumentado.

// TODO: Revisar si esto se adapta a lo descrito como Bases Teóricas en la Guía, o si pertenece a Terminología Básica.
// === Serious games [Juegos serios].
// En un artículo del Tecnológico de Monterrey (Fuerte, 2018) se definen los juegos como “juegos diseñados con un propósito formativo más que para fines de entretenimiento.” Estos juegos permiten a los docentes enseñar a sus estudiantes sobre diversos temas, facilitando el aprendizaje mientras se divierten.
// === Pensamiento computacional.
// El artículo “Research Notebook: Computational Thinking--What and Why?” [Cuaderno de Investigación: Pensamiento Computacional--¿Qué y Por Qué?] (Wing, 2011) afirma que el pensamiento computacional implica resolver problemas, diseñar sistemas y comprender el comportamiento humano, haciendo uso de los conceptos fundamentales de la informática. Este pensamiento se caracteriza por la formulación de un problema de manera que permita el uso del computador para resolverlo, la organización y el análisis lógico de la información, la representación de la información a través de abstracciones y la búsqueda de la solución más efectiva que sea capaz de resolver una familia de problemas.
// === Realidad aumentada.
// La Realidad Aumentada fue definida por García Requejo (2024) como “la tecnología capaz de añadir información a una imagen del mundo real mostrada a través de un dispositivo electrónico (móvil, tablet y ordenador).” Esta tecnología tiene como principales características la capacidad de superponer elementos visuales sobre imágenes reales, proyectar imágenes en 3D que parezcan naturales con respecto al entorno real y realizar una evaluación del contexto, correspondiéndole con lo observado a través de nuestros ojos.
// === Realidad aumentada espacial.
// La Realidad Aumentada Espacial fue descrita en el artículo “Spatial augmented reality for product appearance design evaluation” (Park et al., 2015) como una nueva tecnología que puede producir contenidos inmersivos al superponer la virtualidad y el entorno del mundo real. Esta tecnología se diferencia de la Realidad Aumentada en la forma en la que se muestra, ya que en la Realidad Aumentada Espacial existe una interacción con el espacio físico que posteriormente se refleja en el espacio virtual, mientras que, en el otro tipo de Realidad Aumentada, no existe interacción con el espacio físico y todo se muestra en las pantallas.
// === Aprendizaje colaborativo.
// La Preparatoria Panamericana (2020) define al aprendizaje colaborativo como el “enfoque educativo que, por medio de grupos, busca mejorar el aprendizaje a través del trabajo conjunto.” Este enfoque permite una mejora en la interacción entre alumnos, la comprensión y exposición de perspectivas diversas, inspira creatividad y desarrolla habilidades de pensamiento crítico. Algunos ejemplos incluyen grupos de estudio, debates, juegos de rol, pares y la resolución de problemas de manera grupal. (LHH, 2023)
// === Neurodiversidad.
// García-Bullé del Tecnológico de Monterrey (2021) define la neurodiversidad como “los individuos que viven con autismo principalmente, pero también abarca dislexia, dispraxia, déficit atencional con hiperactividad (TDAH), u otras condiciones que les llevan a navegar procesos cognitivos y emocionales de manera distinta a la norma.” El término se originó en los años 90 para promover la aceptación y el trato normal a personas que pueden actuar de forma diferente. Judy Singer (socióloga que acuñó el término en los años 90) visualiza la neurodiversidad como un “movimiento de justicia social”, con el objetivo de resaltar aquellos beneficios que tienen estas personas y generar comprensión con respecto a las limitaciones que conlleva la neurodiversidad. (Miller, 2024)
// === Visión por computador.
// EDS Robotics (2022) la define como “un grupo de tecnologías o herramientas que permiten a los equipos captar imágenes del mundo real, procesarlas y generar información a través de ellas”. Gracias a estas tecnologías, se puede obtener información del entorno físico para posteriormente ser procesada y plasmada en una pantalla de entorno digital. Esta información se capta a través de un sensor, que envía las imágenes o datos a un dispositivo de interpretación que busca reconocer patrones previamente obtenidos.

#pagebreak(weak: true)

// Capítulo III
= Capítulo III. Marco Metodológico

== Tipo de Investigación

El presente trabajo se clasifica como investigación proyectiva. #cite(<hurtado2010>, form: "prose") define este holotipo como aquel que culmina en la elaboración de una propuesta, plan, programa, procedimiento o artefacto, que esté orientado a resolver una necesidad o problema de carácter práctico en un ámbito determinado del conocimiento; siendo un enfoque frecuente en el campo de la tecnología, donde el objetivo es responder al cómo hacer las cosas mediante aplicaciones concretas. Para alcanzar ese resultado, la investigación proyectiva no parte directamente de una idea hacia su implementación, sino que recorre estadios previos, analíticos, comparativos, explicativos y predictivos; que fundamentan y justifican las decisiones de diseño.

Este enfoque resulta pertinente para el presente trabajo porque su contribución principal no es describir el uso de la programación tangible o la realidad aumentada espacial, sino diseñar y construir un ambiente que los integre de forma coherente, sustentado en el diagnóstico y el análisis realizados.

== Técnicas e instrumentos de recolección de datos

Según #cite(<arias2012>, form: "prose"), las técnicas de recolección de datos son los métodos establecidos para recopilar información, mientras que los instrumentos son las herramientas, dispositivos o formatos empleados para ello. En este trabajo se recurrió a dos técnicas: la revisión documental y la entrevista semiestructurada.

La revisión documental permitió construir el marco teórico, el estado del arte y los criterios de diseño del ambiente. Las fuentes consultadas incluyeron artículos académicos, libros e informes de trabajos de grado.

La entrevista semiestructurada, descrita por #cite(<arias2012>, form: "prose") como una conversación orientada a la obtención de datos que combina preguntas predeterminadas con preguntas abiertas de profundización, se empleó para recoger perspectivas de personas con conocimiento directo del contexto educativo.

== Metodología de Desarrollo Utilizada
Al analizar las características del trabajo de investigación, se consideró el enfoque a adoptar. Dado que no se previó un contacto constante con el cliente y que los requisitos aún no estaban bien definidos, se decidió optar por un enfoque basado en prototipos, con el fin de definir los requerimientos finales a través de los prototipos realizados y sus validaciones.
Según #cite(<pressman2010>, form: "prose"), el enfoque basado en prototipos está enmarcado dentro de los modelos de proceso evolutivos, que "son iterativos. Se caracterizan por la manera en la que permiten desarrollar versiones cada vez más completas del software.". Particularmente para el enfoque basado en prototipos, el proceso se divide en 4 fases, como se observa en la @prototyping-figure: comunicación, plan rápido - modelado - diseño rápido, construcción del prototipo y despliegue - entrega y retroalimentación. Se definen a continuación:

#figure(
  image("images/prototyping-paradigm.png"),
  caption: [
    Etapas del enfoque basado en prototipos. @pressman2010[p. 36] #cite(<pressman2010>)
  ],
) <prototyping-figure>

- Comunicación: Se establece comunicación con los interesados (clientes, usuarios, participantes) para definir los objetivos generales, qué requerimientos se conocen, y en qué se requiere una mejor definición.
- Plan rápido - modelado - diseño rápido: A diferencia de otros enfoques, donde la planificación, modelado y diseño son exhaustivos; en el enfoque basado en prototipos, el énfasis está en definir qué partes del software serán visibles para los usuarios y hacer representaciones de estas (por ejemplo, la interfaz que usarán para interactuar con el software), de modo que se pueda pasar rápidamente a la construcción del prototipo.
- Construcción del prototipo: Se construye un prototipo, que sirve como una versión preliminar del sistema, donde la mantenibilidad a largo plazo o la calidad general no son tan relevantes. Al ser necesario que funcione pronto, es común que se tomen decisiones cuestionables durante la implementación, como la elección de lenguajes de programación inapropiados o uso de algoritmos poco eficientes.
- Despliegue - entrega y retroalimentación: El prototipo construido se despliega para ser evaluado por los interesados, quienes proporcionan retroalimentación, que se usa para refinar los requerimientos.
Las iteraciones continúan mientras se busca que los prototipos que se construyan se acerquen cada vez más a cumplir con las necesidades de los interesados, lo que a su vez ayuda a comprender mejor qué se necesita como producto final. Así pues, los prototipos funcionan como un mecanismo para definir los requerimientos del sistema, reducir riesgos y, dependiendo de cómo se construyan, ser descartados o evolucionar hasta convertirse en el producto final.
En este caso, se utilizó como base el trabajo de investigación “Entorno de Realidad Aumentada Espacial para el Desarrollo de Juegos Sociales Dirigidos a Niños de Educación Preescolar”, que sirvió como punto de partida para el modelado y diseño de los primeros prototipos. A partir de los resultados obtenidos con los prototipos, se definieron los requerimientos finales del entorno a desarrollar.

#pagebreak(weak: true)

// Capítulo IV
= Capítulo IV. Desarrollo y Resultados

== Analizar el Uso de Programación Tangible en Entornos de Realidad Aumentada, a fin de Caracterizar el Ambiente a Desarrollar

Durante el análisis previo al desarrollo de este trabajo, se llevó a cabo una revisión de los conceptos fundamentales que sustentan el ambiente, con el propósito de caracterizar con precisión las decisiones de diseño que lo definen. Esta revisión se estructuró en torno a dos ejes complementarios: por un lado, los fundamentos teóricos del pensamiento computacional y su desarrollo desde edades tempranas; por el otro, la evaluación del estado del arte en el uso de ambientes de programación tangible y entornos de realidad aumentada espacial como estrategias para fomentar el desarrollo de ese pensamiento en niños.

#cite(<wing2006>, form: "prose") introdujo el término pensamiento computacional para referirse a un conjunto de habilidades que trascienden el ámbito de la programación: la formulación de problemas de forma que permitan su resolución computacional, la organización y análisis lógico de la información, la representación de datos mediante abstracciones, y la identificación de soluciones generales y eficientes. Wing defendía que estas habilidades debían incorporarse a la educación desde edades tempranas, al mismo nivel que las competencias de lectura, escritura y aritmética, por sus aplicaciones transversales en la resolución de problemas, la modelización y el análisis. En trabajos posteriores, #cite(<wing2008>, form: "prose") amplió esta visión al señalar que el pensamiento computacional influirá en todos los campos del saber, planteando así un desafío educativo de alcance social. Esta perspectiva fue enriquecida por #cite(<aho2012>, form: "prose"), quien vinculó el pensamiento computacional directamente a los modelos de computación: argumentó que pensar computacionalmente implica formular problemas en términos de los pasos que un modelo de computación puede ejecutar, lo que dota a la habilidad de una base formal que va más allá de la intuición.

#cite(<zapata2015>, form: "prose") profundizó en la dimensión educativa de esta habilidad al caracterizarla como una nueva alfabetización digital que, al igual que otras alfabetizaciones fundamentales, debe iniciarse en las primeras etapas del desarrollo individual. Para Zapata-Ros, el pensamiento computacional no se limita a aprender a programar: comprende una forma de organizar ideas y representaciones que favorece la resolución de problemas cotidianos, la comunicación racional y la participación activa en un entorno cada vez más digitalizado. Esta caracterización es la que sustenta el vínculo entre el pensamiento computacional y la noción de competencia básica del siglo XXI, que #cite(<sanchezvera2019>, form: "prose") identifica como uno de los ejes del discurso educativo contemporáneo: codificar ha sido incluido explícitamente como parte de las habilidades que el ciudadano digital debe desarrollar, y el pensamiento computacional es la competencia que da sentido a ese aprendizaje, más allá de la herramienta concreta con la que se implemente.

En lo que respecta al desarrollo del pensamiento computacional en niños de 6 a 9 años, #cite(<sanchezvera2019>, form: "prose") advierte que la falta de consenso sobre qué es el pensamiento computacional y cómo trabajarlo en el aula ha producido aproximaciones muy heterogéneas, que van desde enfoques técnicos centrados en el aprendizaje de herramientas específicas, con el riesgo de no aprovechar el potencial pedagógico más amplio de la competencia; hasta enfoques transversales que reconocen su carácter didáctico pero que, al incorporar múltiples visiones, dificultan su aplicación concreta. Ante este panorama, la autora subraya la importancia de abordar el pensamiento computacional desde la Tecnología Educativa: no como un fin en sí mismo, sino como un medio para expresar ideas con tecnología y para aprender con herramientas, no de herramientas. Esta distinción orienta el diseño del ambiente propuesto: el objetivo no es enseñar a los niños a programar en un lenguaje particular, sino fomentar en ellos las habilidades de descomposición, abstracción, reconocimiento de patrones y diseño de algoritmos que conforman el pensamiento computacional, aprovechando para ello la manipulación de objetos físicos como estrategia didáctica central.

#cite(<zapata2019>, form: "prose") formalizó este enfoque bajo el concepto de pensamiento computacional desenchufado: actividades que no requieren el uso de pantallas ni computadoras, como el uso de fichas, juegos de patio o juguetes, diseñadas para cultivar en los niños habilidades que serán evocadas en ciclos superiores de aprendizaje como base del pensamiento computacional formal. Este enfoque se apoya en el principio de activación de #cite(<merrill2002>, form: "prose"), que establece que el aprendizaje nuevo es más efectivo cuando conecta con experiencias y conocimientos previos del aprendiz. En este caso, la manipulación física de objetos, familiar para cualquier niño de 6 años; actúa como puente cognitivo hacia conceptos computacionales abstractos que, de otra manera, resultarían inaccesibles en la etapa de desarrollo concreta. La condición pediátrica sobre la exposición a pantallas refuerza aún más la pertinencia de este enfoque: el #cite(<aap2016>, form: "prose"), en representación de la Academia Americana de Pediatría, establece que el uso de pantallas en niños de 2 a 5 años no debe superar una hora diaria, en periodos cortos; y J. Duarte (comunicación personal, 28 de febrero de 2025) añade que los periodos continuos no deben superar los 15 minutos, lo que condiciona el uso de herramientas como Scratch en sus sesiones de clase. La programación tangible, en combinación con la realidad aumentada espacial, emerge así como una alternativa que reduce la dependencia de pantallas sin renunciar a la retroalimentación digital.

Por otro lado, la revisión del estado del arte en ambientes de programación tangible y entornos de realidad aumentada espacial permitió identificar los referentes más relevantes, evaluar sus principios de diseño y extraer las características que deben definir el ambiente propuesto.

La base conceptual de los ambientes de programación tangible es la de las interfaces de usuario tangibles (TUI). #cite(<ishii2008>, form: "prose") las define como un paradigma de interacción que da forma física a la información digital para que los usuarios la manipulen directamente con sus manos, en contraste con las interfaces gráficas (GUI), donde representación y control están desacoplados. En una TUI, el objeto físico cumple una doble función: sirve simultáneamente como control y como representación de la información subyacente. Ishii identifica tres propiedades que definen este paradigma: el acoplamiento computacional entre el objeto físico y el modelo de cómputo; la coincidencia de los espacios de entrada y salida, de modo que el espacio donde el usuario actúa y el espacio donde recibe retroalimentación son el mismo; y el acoplamiento perceptual entre las representaciones tangibles y la retroalimentación intangible (proyecciones, sonidos) que las acompaña, logrando que ambos dominios se experimenten como un continuo coherente. Estas propiedades tienen una consecuencia pedagógica directa: el niño no necesita traducir mentalmente la acción sobre un dispositivo remoto en un efecto sobre una pantalla separada, sino que actúa directamente sobre el objeto que representa la información, reduciendo la carga cognitiva de la interacción.

Dentro de la taxonomía de géneros de TUI de Ishii, las tabletop TUI son las más relevantes para este trabajo. En ellas, objetos tangibles discretos se manipulan sobre una superficie horizontal y la retroalimentación visual se proyecta sobre esa misma superficie, manteniendo la coincidencia de entrada y salida. Esta característica habilita naturalmente la colaboración colocalizada: al ser la entrada espacialmente multiplexada, pues cada objeto ocupa su propio espacio y puede ser manipulado por distintos usuarios al mismo tiempo; se favorece la participación concurrente sin los turnos forzados que impone una GUI. A ello se suma la persistencia de los tangibles: los objetos físicos mantienen su estado de forma autónoma, de modo que el programa construido por los niños es visible y modificable en todo momento sin mediación del sistema digital.

La primera aproximación conocida a la programación tangible con niños es TORTIS, desarrollado por Perlman entre 1974 y 1976 #cite(<morgado2006>). TORTIS permitía a niños pequeños controlar una tortuga robótica mediante objetos físicos —cajas de botones y, posteriormente, cartas de plástico insertadas en una máquina—, estableciendo por primera vez la posibilidad de aprender conceptos de programación sin sintaxis textual. La experiencia mostró que los objetos físicos discretos, donde cada uno corresponde a un comando, facilitan la comprensión de la relación entre instrucción y efecto, una distinción conceptual central en el pensamiento computacional. AlgoBlock #cite(<suzuki1993>) consolidó esta línea: un lenguaje de programación tangible inspirado en Logo donde bloques físicos unidos forman programas que controlan un submarino en pantalla. Su importancia para este trabajo radica en los principios que guiaron su diseño: facilidad de uso, acceso simultáneo de múltiples participantes, monitoreo mutuo del estado del programa y pase del turno mediante gestos. Estos principios producen un efecto directo sobre la colaboración, pues al permitir que todos los participantes manipulen el espacio físico al mismo tiempo, se generan condiciones naturales para la discusión, la negociación de significados y el trabajo en equipo.

Frente a estos antecedentes de la programación tangible, Scratch #cite(<resnick2009>)#cite(<maloney2010>) representa el referente dominante de la programación visual orientada a niños en entornos digitales. Scratch es un entorno de programación en bloques diseñado principalmente para usuarios de 8 a 16 años, que apoya el aprendizaje autodirigido a través de la experimentación y la colaboración entre pares. Su modelo pedagógico se basa en el construccionismo de Papert: los estudiantes aprenden creando proyectos significativos —animaciones, juegos, historias— que pueden compartir con una comunidad. #cite(<maloney2010>, form: "prose") describen cómo el diseño del lenguaje y el entorno de Scratch refuerza este objetivo: los bloques encajables hacen visibles las estructuras de control, la retroalimentación es inmediata y la experimentación está permitida en todo momento. Si bien Scratch es una herramienta ampliamente validada para el fomento del pensamiento computacional —como demuestran #cite(<bers2018>, form: "prose") con ScratchJr para niños de preescolar, y #cite(<perezmarin2020>, form: "prose") con el uso de Scratch en educación primaria—, su naturaleza requiere del uso sostenido de una pantalla y está orientada fundamentalmente a la interacción individual o turnada, lo que limita su viabilidad para los grupos etarios más jóvenes del rango de interés y para entornos que priorizan el aprendizaje colaborativo presencial y la reducción de tiempo en pantalla.

Esta limitación es la que motiva la integración de la realidad aumentada espacial como componente del ambiente propuesto. #cite(<park2015>, form: "prose") definen la realidad aumentada espacial como una tecnología que superpone contenido virtual directamente sobre superficies del mundo real mediante proyección, sin requerir dispositivos intermediarios como tabletas o teléfonos. Esta característica la distingue de la realidad aumentada convencional y la hace especialmente adecuada para su integración con interfaces tangibles: el objeto físico y su representación virtual aumentada coexisten en el mismo espacio, sin que el niño tenga que desviar la mirada hacia una pantalla externa. #cite(<billinghurst2012>, form: "prose") evaluaron el uso de realidad aumentada espacial en aulas y documentaron resultados directamente relevantes al propósito de este trabajo: la tecnología mejoró la comprensión de conceptos espaciales y abstractos al proveer representaciones visuales concretas sobre objetos físicos; apoyó la retención de contenido en estudiantes con dificultades de comprensión lectora, al permitirles interactuar con el material en lugar de solo leerlo; y estimuló en los niños el deseo de construir sus propias escenas aumentadas tras experimentar con las creadas por otros, activando el ciclo reflexión-acción-reflexión propio del construccionismo. En el contexto venezolano, #cite(<barrios2024>, form: "prose") demostró con Magicboard que un entorno de realidad aumentada espacial basado en Kinect y proyector puede sostener juegos sociales colaborativos entre niños de educación preescolar y básica, validando la viabilidad técnica y pedagógica de esta combinación tecnológica en el rango de edad objetivo y constituyendo el punto de partida directo del presente trabajo.

Del análisis conjunto de estos referentes emerge la caracterización del ambiente a desarrollar. En primer lugar, debe adoptar una tabletop TUI, donde la retroalimentación visual se proyecta sobre la misma superficie de manipulación, permitiendo que múltiples niños actúen simultáneamente y que el estado del programa permanezca físicamente visible en todo momento, en línea con los principios de acceso simultáneo y monitoreo mutuo de #cite(<suzuki1993>, form: "prose"). En segundo lugar, la retroalimentación visual debe ser proyectada directamente sobre esa superficie, evitando el uso de pantallas adicionales y manteniendo la atención del niño en el espacio físico compartido. En tercer lugar, el paradigma de programación subyacente debe ser el dataflow: frente al paradigma imperativo —sobre el que Scratch está construido—, el dataflow determina la ejecución según la disponibilidad de los datos, no según una secuencia de instrucciones con flujo de control explícito #cite(<wadge1985>). Esta diferencia tiene consecuencias pedagógicas directas: en lugar de que el niño deba imaginar un puntero de ejecución recorriendo instrucciones en orden —una abstracción difícil en la etapa de operaciones concretas de Piaget—, observa cómo los datos fluyen y se transforman a través de una red de nodos que puede disponer físicamente en la superficie. La red de datos y operaciones que constituye un programa dataflow se corresponde de forma natural con la disposición espacial de bloques conectados, haciendo visible la estructura computacional de manera coherente con la experiencia concreta del niño. Adicionalmente, la ausencia de estado mutable y de efectos laterales propia del paradigma dataflow puro #cite(<wadge1985>) simplifica el modelo mental necesario para razonar sobre el programa: cada bloque produce siempre el mismo resultado con los mismos datos de entrada, sin sorpresas derivadas de órdenes de ejecución o modificaciones ocultas de variables. Lucid, el lenguaje de programación dataflow purista desarrollado por #cite(<wadge1985>, form: "prose"), sirvió como referente para definir el modelo de ejecución del lenguaje propuesto en este trabajo, particularmente en lo relativo a la evaluación dirigida por demanda y a la representación de los programas como redes de filtros funcionales sobre flujos de datos.

=== Caracterización del ambiente a desarrollar

Del análisis precedente se desprende que el ambiente a desarrollar debe presentar las siguientes características:

- *Interfaz tangible de superficie (tabletop TUI):* una superficie plana sobre la que los niños disponen bloques físicos que representan datos y operaciones, con múltiples participantes actuando simultáneamente.
- *Retroalimentación visual aumentada:* proyección directa sobre la superficie que muestra el estado de ejecución del programa, las conexiones entre bloques y los resultados, sin requerir que los niños retiren la vista del espacio de juego.
- *Lenguaje de programación basado en dataflow:* los bloques representan nodos en una red de flujo de datos; las conexiones entre ellos determinan la ejecución.
- *Diseño para colaboración:* la disposición física y el protocolo de interacción deben favorecer el acceso simultáneo, el monitoreo mutuo y la comunicación entre participantes.
- *Compatibilidad con el desarrollo cognitivo de 6 a 9 años:* los bloques deben ser reconocibles visualmente, las operaciones deben corresponder a conceptos concretos y familiares, y el ciclo de construcción-ejecución-observación debe ser inmediato.
- *Reducción del tiempo en pantalla:* la proyección sobre superficie reemplaza el monitor; los bloques físicos reemplazan el teclado y el ratón.

=== Requerimientos

A partir de las características definidas, y con el propósito de guiar el diseño, implementación y validación del ambiente, se definieron los requerimientos funcionales y no funcionales que debe satisfacer el sistema. Estos requerimientos, junto con la característica del ambiente de la que se desprende cada uno, se resumen en la @requirements-table.

#figure(
  [
    #set text(size: 9pt)
    #table(
      columns: (auto, 1fr, 1.7fr),
      align: (center + horizon, left + horizon, left + horizon),
      inset: 5pt,
      table.header([*Código*], [*Característica asociada*], [*Requerimiento*]),
      table.cell(colspan: 3)[*Requerimientos funcionales*],
      [RF-01], [Interfaz tangible (tabletop TUI); diseño para colaboración], [El sistema debe permitir a los niños construir programas utilizando elementos tangibles y conexiones digitales que representen datos, flujos y operaciones],
      [RF-02], [Interfaz tangible (tabletop TUI)], [El sistema debe capturar la disposición de los elementos tangibles y conexiones digitales, y procesar la información para reconocer los elementos y sus conexiones],
      [RF-03], [Lenguaje basado en dataflow], [El sistema debe interpretar los programas representados por los elementos tangibles y conexiones digitales, traduciéndolos a una representación ejecutable],
      [RF-04], [Retroalimentación visual aumentada; reducción del tiempo en pantalla], [El sistema debe ejecutar los programas y mostrar la salida en una interfaz gráfica proyectada sobre una superficie plana],
      [RF-05], [Retroalimentación visual aumentada; compatibilidad con el desarrollo cognitivo (6 a 9 años)], [El sistema debe proveer retroalimentación para guiar a los niños durante la construcción de programas],
      table.cell(colspan: 3)[*Requerimientos no funcionales*],
      [RNF-01], [Compatibilidad con el desarrollo cognitivo (6 a 9 años)], [El sistema debe ser usable por niños de 6 a 9 años y profesores de primaria de 1#super[er] a 3#super[er] grado],
      [RNF-02], [Compatibilidad con el desarrollo cognitivo (6 a 9 años)], [El sistema debe contener elementos persuasivos que capten el interés de niños de 6 a 9 años],
      [RNF-03], [Retroalimentación visual aumentada], [El sistema debe ser capaz de manejar errores en la disposición de los elementos tangibles y digitales],
      [RNF-04], [Retroalimentación visual aumentada], [La retroalimentación debe ser presentada de forma visual y auditiva],
    )
  ],
  caption: [
    Requerimientos funcionales y no funcionales del ambiente, derivados de las características identificadas en el análisis.
  ],
) <requirements-table>

== Diseñar un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Función del Análisis Realizado

Este capítulo describe el diseño del ambiente de aprendizaje y del lenguaje de programación tangible denominado ERAE, en coherencia con los requerimientos funcionales y no funcionales del sistema. Se distingue deliberadamente lo pedagógico y físico del ambiente, la arquitectura lógica del software, la especificación conceptual y formal del lenguaje, y la forma en que el intérprete del lenguaje se integra con otros subsistemas.

=== Requisitos y contexto

El ambiente está dirigido a niños de 6 a 9 años y a docentes de educación primaria (1#super[er] a 3#super[er] grado). Los niños construyen soluciones a problemas planteados por el docente, manipulando elementos tangibles y la interacción digital sobre la superficie de trabajo; se prioriza que cada niño exprese su forma de resolver el problema con los medios disponibles, sin imponer una única solución óptima. Los docentes orientan el uso del ambiente y el desarrollo del pensamiento computacional.
// el sistema debe permitir crear y gestionar actividades alineadas al currículo.

Los contenidos sobre los que se apoyan datos, operaciones, actividades de ejemplo y criterios de integración en aula se toman de los énfasis curriculares para la educación primaria del Estado venezolano #cite(<mppe2023>), en lo correspondiente a matemáticas de 1#super[er] a 3#super[er] grado, de modo que el ambiente pueda incorporarse de forma coherente a las planificaciones de esos grados.

=== Arquitectura física y lógica del ambiente

El sistema se concibe como una interfaz de usuario tangible (TUI) de tipo tabletop. Las entradas físicas se realizan mediante la TUI (piezas, regiones); las conexiones digitales (generadas a través de toques sobre la superficie y reconocidas como enlaces en el grafo) y las salidas se canalizan por la proyección sobre la superficie. Así se satisface el requerimiento de que los programas combinen elementos tangibles y conexiones digitales que representen datos, flujos y operaciones.

El ambiente material incluye, como mínimo, un conjunto de elementos tangibles y representaciones digitales asociadas que denotan orígenes de datos u operaciones sobre datos (que, articulados con las conexiones inferidas, constituyen un programa en el lenguaje tangible); un computador; un proyector; una cámara de color y de profundidad; y una superficie plana dividida en al menos dos zonas. En una zona se colocan exclusivamente los elementos tangibles; en la otra se proyecta la interfaz y tiene lugar la interacción entre lo tangible y lo digital. La cámara captura la escena en esa segunda zona, envía la información al computador, el cual procesa la imagen, reconoce elementos, posiciones y relaciones entre orígenes de datos y zonas de transformación, interpreta y ejecuta el programa inferido y proyecta la salida sobre la superficie plana.

A nivel lógico, el flujo a seguir es: captura; reconstrucción del programa (representación estructurada); interpretación y ejecución; presentación de resultados y retroalimentación (visual y auditiva) que orienta al niño durante y después de la construcción. El núcleo de interpretación y ejecución se describe en la sección de integración; aquí basta señalar que está pensado para operar sobre representaciones del programa compatibles con la especificación del lenguaje ERAE.

//TODO: leer y conciliar términos (datos vs orígenes de datos, ver si este último es necesario, y así), además de formas de interacción con el ambiente
=== Interacción y percepción

==== Áreas delimitadas para orígenes de datos y zonas de transformación

Los orígenes de datos y las zonas de transformación los definen los niños al colocar elementos tangibles y al delimitar regiones sobre la superficie. Esas regiones se detectan por visión por computador. Cada región acotada puede interpretarse como un origen de datos o como una zona de transformación. Si es un origen de datos, los elementos tangibles dentro de la región son los valores que conforman dicho origen; si es una zona de transformación, los elementos tangibles en su interior representan las operaciones que se aplicarán a los datos procedentes de los orígenes conectados.

==== Conexiones entre orígenes de datos y zonas de transformación

Las conexiones no se materializan con cables ni con piezas adicionales: son conexiones digitales trazadas por los niños sobre la superficie (por ejemplo, mediante toques). La cámara y el computador reconocen esas conexiones y establecen la relación entre orígenes y zonas de transformación.

La interfaz refleja orígenes, zonas y conexiones inferidas a partir de lo físico y lo trazado. Esto cumple el requerimiento de retroalimentación para guiar durante la construcción y el requerimiento no funcional de retroalimentación visual y auditiva. En lo visual, entre otras cosas, se incluye:

- Resaltar orígenes de datos y zonas de transformación reconocidos.
- Resaltar conexiones reconocidas entre orígenes y zonas de transformación.
- Mostrar mensajes de error o advertencia cuando el programa sea inválido o incompleto.
- Resaltar orígenes o conexiones erróneas o inválidas (manejo de errores de disposición).
- Señalar elementos tangibles no reconocidos o no utilizados en el programa actual.

En lo auditivo, se complementa con señales sonoras acordes a reconocimiento correcto, advertencia o error, de modo que la guía no dependa solo de la vista. El modo incremental de integración (véase más adelante) refuerza la guía continua mientras el grafo está aún incompleto.

==== Ejecución y salida

La salida del programa se muestra en la interfaz proyectada sobre la superficie plana, usando las representaciones digitales del lenguaje. La composición de piezas físicas, conexiones digitales trazadas y elementos en pantalla constituye la representación visible de un programa que aborda el problema de la actividad en curso. El diseño de la interfaz gráfica que materializa esta proyección se muestra en la @fifth-prototype-design-figure.

#figure(
  image("images/fifth-prototype-design.png"),
  caption: [
    Diseño de la interfaz gráfica del entorno de desarrollo integrado (IDE) para el lenguaje de programación.
  ],
) <fifth-prototype-design-figure>

=== Visión del lenguaje en el ambiente

El lenguaje ERAE es un lenguaje de flujo de datos (dataflow), donde los programas se representan como grafos de nodos que producen valores, los transforman y declaran salidas. En el ambiente, ese grafo tiene una parte tangible (piezas, disposición, regiones) y una parte digital (conexiones inferidas del trazado, proyección, estado de reconocimiento, mensajes y retroalimentación sonora), en línea con los requerimientos de datos, flujos y operaciones combinados en una sola construcción compartida entre el niño y el sistema.

No se persigue la Turing-completitud como objetivo pedagógico; se busca un lenguaje suficientemente expresivo para un subconjunto de problemas acordes al currículo citado, y simple de interpretar por niños de 6 a 9 años. La evaluación del programa puede describirse de forma abstracta como bajo demanda, en la línea de lenguajes de flujo de datos clásicos como Lucid (los nodos se evalúan cuando sus resultados son requeridos por otros nodos o por la salida). El diseño visual del lenguaje, con la disposición de datos, operaciones y flujos sobre la superficie, se muestra en la @fourth-prototype-visual-design-figure.

#figure(
  image("images/fourth-prototype-visual-design.jpeg"),
  caption: [
    Diseño visual del lenguaje de programación basado en el paradigma de programación dataflow, con algunos detalles sobre la interfaz gráfica.
  ],
) <fourth-prototype-visual-design-figure>

La especificación detallada de tipos, operadores y estructura sintáctica del lenguaje se presenta en la siguiente sección.

=== Especificación del lenguaje de programación tangible ERAE

ERAE es, ante todo, un lenguaje visual y tangible: los programas se construyen disponiendo cartas físicas sobre la superficie de trabajo y trazando conexiones digitales entre ellas. La especificación que sigue describe ese lenguaje visual —sus piezas, su estructura y sus garantías—; la representación textual interna sobre la que opera el intérprete se menciona al final y su gramática formal se consigna en apéndice.

==== Filosofía de diseño

Los principios rectores, en línea con prácticas de lenguajes educativos como el enfoque de tipos fijos de Scratch, son:

- *Tipos integrados:* conjunto de tipos cerrado, sin extensión por parte del usuario, para reducir la carga cognitiva.
- *Operaciones seguras:* comprobación estática, previa a la ejecución, de la compatibilidad de datos entre operadores.
- *Prevención de errores:* verificación estricta de tipos y de la aridad de cada operación (número correcto de entradas), en apoyo al manejo de errores en la disposición tangible y digital antes de ejecutar.
- *Alineación curricular:* tipos y operaciones elegidos para mapearse a clasificación, comparación y manipulación de colecciones propios de primaria, en coherencia con el currículo de matemáticas de referencia.

==== Piezas tangibles

La unidad léxica del lenguaje visual es la carta: una pieza física tipo naipe que porta una imagen reconocible por el subsistema de visión y por el niño. El mazo se organiza en cuatro grupos. Los tres primeros siguen la progresión concreto-pictórico-abstracto, derivada de los modos de representación enactivo, icónico y simbólico de #cite(<bruner1966>, form: "prose") y coherente con la transición de la etapa preoperacional a la de operaciones concretas descrita en el marco teórico:

- *Cartas concretas:* representan objetos cotidianos manipulables: alimentos (manzana, hamburguesa, pera, uvas), cubos de tipo Montessori, tapas y paletas de colores. Con ellas se trabajan las operaciones de suma, resta, filtrado y ordenación sobre colecciones de objetos familiares.
- *Cartas pictóricas:* representan figuras geométricas (círculo, cuadrado, triángulo) en tres tamaños y varios colores, junto con cartas de criterio que denotan propiedades (tamaño, color o forma) y parametrizan las operaciones de filtrado, ordenación y comparación.
- *Cartas abstractas:* representan los dígitos del 0 al 9 y habilitan el repertorio aritmético completo, incluyendo la multiplicación y la división, reservadas a este nivel.
- *Cartas estructurales, comunes a los tres niveles:* la carta de resultado, que designa la salida del programa, y las cartas de apertura y cierre de colección, que agrupan varias cartas de datos en un solo conjunto.

Cada operación está representada por su propia carta (suma, resta, multiplicación, división, ordenación ascendente y descendente, filtrado y comparación), de modo que el repertorio disponible se controla entregando a los niños el subconjunto del mazo acorde a la actividad y a su nivel. El diseño tipo carta de las piezas se muestra en la @sixth-prototype-pieces-design-figure.

#figure(
  image("images/sixth-prototype-pieces-design.jpeg"),
  caption: [
    Diseño tipo carta de las piezas tangibles del ambiente.
  ],
) <sixth-prototype-pieces-design-figure>

==== Estructura de un programa

Un programa se organiza como un grafo de flujo de datos construido sobre la superficie. A nivel conceptual, los nodos se clasifican en:

- *Nodos de fuente:* aportan datos iniciales al grafo; se forman con cartas de datos (concretas, pictóricas o abstractas), individualmente o agrupadas mediante las cartas de colección.
- *Nodos de transformación:* aplican operaciones a las entradas que reciben por las conexiones del flujo de datos; se forman con una carta de operación y, cuando corresponde, cartas de criterio que la parametrizan.
- *Nodos de salida:* designan los valores que deben mostrarse o entregarse al entorno de visualización; se forman con la carta de resultado.

Las aristas del grafo son las conexiones digitales trazadas por los niños sobre la superficie, que el sistema reconoce y proyecta. La composición de cartas físicas y conexiones digitales constituye el programa completo.

==== Prevención de errores sintácticos y azúcar sintáctico

El diseño tangible elimina por construcción la mayor parte de los errores sintácticos posibles en un lenguaje textual. Cada carta es un símbolo léxico completo y válido: no existen errores de escritura, identificadores mal formados ni delimitadores faltantes, porque el niño nunca escribe texto. La estructura de declaraciones del lenguaje subyacente (nombres de nodos, asignaciones y terminadores) la genera automáticamente el sistema a partir de la disposición de las cartas y las conexiones reconocidas, actuando como azúcar sintáctico sobre el lenguaje textual interno: las cartas de criterio se traducen a literales de criterio, las cartas de apertura y cierre de colección a literales de grupo, y la carta de resultado a una declaración de salida. Los únicos errores que el niño puede cometer son semánticos —conectar tipos incompatibles, dejar entradas sin conectar, formar ciclos o entregar a una operación un número incorrecto de entradas— y todos son detectados por el intérprete antes o durante la evaluación, comunicándose mediante la retroalimentación visual y auditiva del ambiente descrita previamente.

==== Tipos de datos

Tipos numéricos y escalares primitivos:

- *Naturales:* enteros mayores o iguales que cero.
- *Enteros:* positivos y negativos.
- *Decimales:* números con parte fraccionaria para medidas.
- *Fracciones:* representación explícita de cocientes (por ejemplo $1/2$, $3/4$).
- *Texto:* cadenas para etiquetas y valores simbólicos.
- *Booleanos:* verdadero o falso.

Tipos curriculares:

- *Formas:* atributos de tipo geométrico (círculo, triángulo, cuadrado), tamaño y color.
- *Coches:* atributo de color.
- *Comida:* atributos de sabor (dulce, salado, agrio, amargo) y color.
- *Animales:* tipo de animal y color.
- *Personas:* grupo etario y género.

Los valores concretos permitidos para cada atributo (por ejemplo, paleta de colores o conjunto de tipos de forma) están fijados en la especificación formal del lenguaje para mantener coherencia entre tangibles, reconocimiento y ejecución. En la especificación vigente, cada fuente de datos declara una categoría (abstracta, pictórica o concreta), un tipo y un subtipo, una cantidad opcional expresada como número racional, y pares clave-valor libres para atributos adicionales (como color, tamaño o sabor), lo que permite incorporar nuevos tipos curriculares sin modificar la gramática del lenguaje.

Tipos compuestos:

- *Conjuntos:* colecciones homogéneas de elementos de un mismo tipo.
- *Flujos:* secuencias de valores en el tiempo, en correspondencia con la naturaleza dataflow del lenguaje y con patrones de iteración o señales discretas.

==== Catálogo de operaciones

Las operaciones se agrupan en familias. La lista siguiente resume las categorías definidas en la especificación vigente del lenguaje (versión 4.2.0); cada operador tiene firmas de tipo y aridad que el intérprete verifica antes de la ejecución.

- *Operaciones aritméticas:* suma (`sum`), resta (`substract`), multiplicación (`multiply`) y división (`divide`).
- *Operaciones de comparación e igualdad:* menor que (`less_than`), mayor que (`greater_than`) y comparación general de igualdad (`compare`), aplicadas según los tipos involucrados.
- *Filtrado:* filtro (`filter`) que extrae, de un conjunto de datos, los elementos que cumplen los criterios indicados (por ejemplo, por color, tamaño o tipo).
- *Ordenación:* orden ascendente (`order_asc`) y descendente (`order_desc`).
- *Agregación y selección:* primero (`first`), último (`last`) y conteo (`count`), que operan sobre colecciones para seleccionar elementos o resumirlas.

==== Representación textual interna

El lenguaje visual se traduce a una representación textual interna sobre la que opera el intérprete, cuya sintaxis concreta (palabras clave, literales y reglas de formación) se especifica formalmente mediante una gramática en notación EBNF de la W3C, presentada en el #link(<appendix-a>)[Apéndice A]. La gramática admite modificadores opcionales en sus reglas de declaración, lo que permite analizar sintácticamente programas incompletos sin interrumpir la sesión: capacidad necesaria para la retroalimentación inmediata mientras el niño aún está construyendo el programa.

// ==== Ejemplo ilustrativo

// El siguiente fragmento es solo ilustrativo de la forma de los programas; la sintaxis definitiva y los nombres exactos de operadores coinciden con la gramática del documento de especificación.

// ```dataflow
// source a: natural = 3;
// source b: natural = 2;
// transform sum: natural = ADD(a, b);
// output result: natural = sum;
// ```

=== Integración del intérprete con el resto del sistema

// ==== Principio arquitectónico

Se adopta una separación entre núcleo sin estado y adaptadores delgados. El intérprete no conoce los detalles de comunicación con el resto del sistema, ya que recibe datos de programa, devuelve resultados o diagnósticos, y no mantiene sesión de usuario. Esta comunicación se implementa en capas periféricas que serializan y deserializan solicitudes y respuestas.

// ==== Modos de evaluación

// Modo por lotes (batch): pensado para ejecutar un programa completo cuando la escena ya está estable o cuando el subsistema de visión entrega un grafo cerrado. Entrada: programa completo y válido (por ejemplo en JSON). Proceso: compilar, validar y ejecutar. Salida: resultados finales y traza de ejecución. Caso de uso típico: la visión detecta que el niño terminó de montar el programa, envía la representación y se proyecta el resultado final.

// Modo incremental: pensado para retroalimentación mientras el programa aún se construye (requerimiento funcional de guía durante la construcción). Entrada: grafo parcial. Proceso: validar el fragmento y evaluar únicamente lo que sea semánticamente posible. Salida: valores parciales o estados de pendiente en nodos aún incompletos. Caso de uso: el niño añade o conecta un bloque y el sistema responde al instante si faltan entradas o si una parte del grafo ya puede mostrarse; la capa de presentación puede combinar esta salida con pistas visuales y auditivas.

// ==== Interfaces de integración

// La capa de integración prevé, entre otros mecanismos, una API HTTP para el modo por lotes y un servidor WebSocket para el modo en vivo con el IDE o entornos de construcción interactiva. El protocolo de lenguaje de servidores (LSP) puede utilizarse para asistir al editor o IDE que acompañe el diseño de actividades avanzadas, en coherencia con los objetivos de herramientas de apoyo al lenguaje ERAE.

// == Actividades y rol docente
// === Rol del docente

//=== Definición y gestión de actividades

// Una actividad agrupa: el enunciado del problema, la explicación de los conceptos involucrados, las condiciones durante el desarrollo, el inicio de la tarea y el resultado esperado. Los niños resuelven la actividad construyendo un programa con el lenguaje tangible y los elementos provistos por el ambiente. El sistema permite a los docentes crear, editar y organizar actividades alineadas al currículo de matemáticas de 1.er a 3.er grado (MPPE, 2023) y orientadas al desarrollo del pensamiento computacional en la franja de edad objetivo.

=== Guía de diseño de actividades

La guía incluye actividades modelo con problemas y soluciones de referencia elaboradas por los autores del ambiente, inspiradas en el mismo currículo. Su función es formativa: no debe interpretarse como catálogo cerrado de los únicos problemas que el ambiente admite, ni como restricción a la variedad de soluciones válidas. Se enfatiza el papel activo del niño en la exploración de estrategias y soluciones.

// La gramática EBNF completa se consigna en el Apéndice A (<appendix-a>), tomada de la especificación viva del lenguaje (GRAMMAR_SPEC.md, v4.2.0). Cualquier divergencia futura entre implementación y especificación debe resolverse actualizando primero la especificación y luego el texto del diseño, para conservar trazabilidad académica.

== Construir un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Base al Diseño Realizado

=== Prototipo 1

Partiendo de la tesis de Barrios, se buscó una aproximación más programática, asimilándose a Scratch, por lo que se partió de seguir el paradigma imperativo y la programación con bloques. Sin embargo, dado que Scratch ya es ampliamente usado y tiene varias investigaciones al respecto de su uso, tal como se planteó durante el análisis previo; el tutor sugirió seguir una aproximación distinta, basada en el paradigma de programación dataflow, pues ofrece una clara visualización de cómo fluyen y se transforman los datos del programa.

Dado este cambio, se procedió con la definición de los primeros datos y operaciones a usar, para lo que se eligieron bloques con formas geométricas simples (cuadrados, círculos y triángulos) y colores básicos (morado, amarillo, naranja, verde, rojo y azul) que, para simplificar el desarrollo, se decidió que algunos representarían operaciones en vez de un dato. Los datos que se soportaban provenían directamente de las formas (cuadrados, círculos y triángulos de distintos colores), y las operaciones eran conjunción, intersección, diferencia y diferencia simétrica. El diseño consistió de zonas que reconocían las formas colocadas como datos, otras que reconocían las formas como operaciones, y zonas de salida que mostraban el resultado de la ejecución. Todas estas zonas estaban colocadas de forma fija, restringiendo la creación de nuevas zonas o la asociación entre estas para el usuario final, lo que limitaba la flexibilidad del entorno pero facilitaba el desarrollo del prototipo.

Para la construcción, se decidió continuar el uso de Python para todo, haciendo uso de OpenCV y OpenNI2 para la visión por computador, y también OpenCV para la interfaz gráfica. Se usó un sensor Kinect para la captura de imágenes, y se implementó un sistema de reconocimiento de formas basado en la detección de contornos, que permitía identificar las formas geométricas y sus colores para determinar los datos y operaciones a ejecutar. El resultado de la ejecución se mostraba en una zona de salida mediante la superposición de imágenes generadas por el software. Este prototipo puede verse en la @first-prototype-figure.

#figure(
  image("images/first-prototype.jpeg"),
  caption: [
    Primer prototipo del ambiente, con bloques de formas geométricas simples y colores básicos para representar datos y operaciones.
  ],
) <first-prototype-figure>

Como resultado de este prototipo, se vio que no se podía partir directamente del código legado por Barrios, pues se necesitaban de librerías más potentes para tener una interfaz gráfica más atractiva, algoritmos más robustos para la detección de piezas más complejas (números, imágenes), y una arquitectura de software más flexible para permitir la creación de nuevas zonas y la asociación entre estas. Además, surgió la inquietud de que las resoluciones de las cámaras del sensor Kinect v1 no fueran suficientes para detectar piezas más complejas, lo que llevó a la decisión de cambiar al sensor Kinect v2, lo que permitiría una detección más precisa y robusta.

=== Prototipo 2

Debido a las preocupaciones con respecto al Kinect v1, y tras analizar las posibles ventajas, se concluyó que se intentaría el cambio al Kinect v2. El desarrollo de este prototipo entonces se enfocó en la adaptación del código legado por Barrios, para la calibración y detección de toques, para soportar el nuevo sensor.

Así pues, se llevó a cabo una investigación sobre el uso del Kinect v2 con Python, las diferencias entre el Kinect v1 y el Kinect v2, las librerías disponibles para la visión por computador con este nuevo sensor, y el algoritmo de detección de toques basado en profundidad.

Las librerías disponibles para integrar el Kinect v2 con Python son limitadas. Se probaron aproximaciones con PyKinect2 y libfreenect2, sin embargo, el primero fallaba por falta de soporte para Python 3+, y el segundo no detectaba el Kinect v2; OpenNI2, que se usó para el Kinect v1, no es compatible con el Kinect v2 por defecto, pero existen parches para hacerlo compatible, con lo cual se logró usar OpenNI2 para la integración del Kinect v2 con Python. Siguiendo con la calibración y detección de toques, se hicieron modificaciones exhaustivas al código legado para adaptarlo al nuevo sensor, lo que llevó a la implementación de un nuevo algoritmo de detección de marcadores (2 cuadrados blancos en las esquinas superior izquierda e inferior derecha de la proyección), además de la afinación de múltiples números mágicos (literales escritos en el código sin documentar su significado). Este prototipo puede verse en la @second-prototype-figure.

#figure(
  image("images/second-prototype.jpeg"),
  caption: [
    Segundo prototipo del ambiente, con el cambio al sensor Kinect v2 y la adaptación del código legado para la calibración y detección de toques.
  ],
) <second-prototype-figure>

Este prototipo, si bien permitió validar la viabilidad del cambio al Kinect v2, también mostró que el código legado por Barrios era difícil de mantener. También se vio que la transformación Window-to-Viewport que se usa en los algoritmos es muy sensible a la configuración física del entorno (paralelismo entre la proyección sobre la superficie y el ángulo de la cámara), resultando en que la detección de toques no fuese tan precisa como se esperaba.

=== Prototipo 3

Tras trabajar tanto en una única parte del sistema (integración con el hardware y detección de toques), se decidió que el siguiente paso sería trabajar en la visión por computador para la detección de las piezas que conformarían el entorno.

Para esto, se decidió usar un modelo de detección de objetos basado en aprendizaje profundo, específicamente el modelo YOLO11-nano, que es una versión ligera del modelo YOLO11, diseñado para ser eficiente en términos de velocidad y recursos computacionales, lo que lo hace adecuado para aplicaciones en tiempo real como la visión por computador con el Kinect v2. Se planeó entrenar este modelo con un conjunto de datos personalizado que incluía imágenes de una versión previa de las piezas que se usarían en el entorno, con el objetivo de lograr una detección precisa y evaluar la viabilidad de detectar las piezas mediante modelos de detección de objetos.

Se entrenó al modelo con el conjunto de datos personalizado de imágenes de una versión previa de las piezas que se usarían en el entorno, que incluían animales y números, que pueden verse en la @third-prototype-dataset-figure; y se evaluó su desempeño en términos de precisión y velocidad de detección. // Este prototipo puede verse en la @third-prototype-figure.

Los resultados obtenidos mostraron que el modelo de detección de objetos basado en aprendizaje profundo era capaz de detectar las piezas con una precisión aceptable, aunque se identificaron áreas de mejora, principalmente la confusión entre clases (por ejemplo, entre el 9 y el 6). Además, se observó que la velocidad de detección era adecuada para su uso en tiempo real con el Kinect v2, lo que validó la viabilidad de esta aproximación para la detección de piezas en el entorno.

#figure(
  image("images/third-prototype-dataset.jpeg"),
  caption: [
    Conjunto de datos personalizado para entrenar al modelo de detección de objetos, mostrando una versión previa de las piezas que se usarían en el entorno.
  ],
) <third-prototype-dataset-figure>

// #figure(
//   image("images/third-prototype.png"),
//   caption: [
//     Tercer prototipo del ambiente, con la implementación de un modelo de detección de objetos basado en aprendizaje profundo para la detección de las piezas que conformarían el entorno.
//   ],
// ) <third-prototype-figure>

=== Prototipo 4

Dado que se usaría un paradigma de programación dataflow, se decidió que se seguiría con la definición y elaboración de un lenguaje de programación visual basado en este paradigma, con el objetivo de crear una interfaz gráfica atractiva y funcional para los usuarios finales, que permitiera la creación de programas mediante la manipulación de bloques visuales que representaran operaciones y datos.

Se llevó a cabo una investigación sobre los lenguajes de programación dataflow, tomando como referente a Lucid /* ver si se cambia la inspiración directa en Lucid */ #cite(<wadge1985>), por ser un lenguaje de programación dataflow purista, y se definieron los elementos básicos del lenguaje de programación visual, incluyendo los tipos de bloques, las operaciones disponibles, y la forma en que los bloques se conectan para formar programas. Este diseño puede verse en la @fourth-prototype-visual-design-figure. Las operaciones disponibles se basarían en el currículum de matemáticas de educación básica, con el objetivo de fomentar el desarrollo del pensamiento computacional a través de conceptos matemáticos, y se incluirían operaciones como suma, resta, multiplicación, división, entre otras. En pro de una correcta división de las responsabilidades del sistema, se separó el lenguaje de programación visual en dos partes: un apartado de detección de piezas, que se encargaría de detectar las piezas físicas colocadas por los usuarios y traducirlas a una representación interna del programa; y un apartado de ejecución, que se encargaría de ejecutar el programa representado internamente y enviar los resultados a la interfaz gráfica. Esta separación permitiría una mayor flexibilidad y mantenibilidad del sistema, facilitando la incorporación de nuevas piezas y operaciones en el futuro.

Durante el desarrollo de este prototipo, el enfoque estuvo en la implementación del apartado de ejecución del lenguaje de programación dataflow, para lo cual se definieron 3 representaciones de los programas formados por los bloques visuales: una de intercambio, basada en JSON; una textual, para entrada y depuración; y un formato en memoria, para uso interno por el entorno de ejecución; y se implementó un intérprete para ejecutar estos programas (denominado inicialmente compilador y _runtime_, terminología que fue revisada en iteraciones posteriores al consolidarse la evaluación directa de los programas). Se decidió usar TypeScript como lenguaje de programación, debido a su flexibilidad, facilidad para el desarrollo rápido, y su capacidad para manejar estructuras de datos complejas mediante su tipado; Bun como motor de ejecución, pues permite la ejecución directa de programas escritos en TypeScript sin un paso previo de transpilación, y provee ventajas de rendimiento contra sus competidores Node y Deno; y la librería Chevrotain, que provee un kit herramientas para la construcción de _parsers_; facilitando la implementación del entorno. Además, se implementó un servidor HTTP y uno de WebSockets, para lo cual se utilizó la librería Elysia, que permiten la comunicación con la interfaz gráfica y el apartado de visión por computador. // Este prototipo puede verse en la @fourth-prototype-figure.

//TODO: colocar imágenes/tablas de las 3 representaciones de los programas, quizás todo en apéndices. Para JSON, puede ser la interfaz de TS. Para la representación textual, la EBNF del lenguaje con las consideraciones semánticas, que este sí sería un apéndice 100%. Para la representación en memoria, una tabla con la estructura de datos usada para representar los programas internamente.

// #figure(
//   image("images/fourth-prototype-figure.jpeg"),
//   caption: [
//     Cuarto prototipo del ambiente, con la implementación del apartado de ejecución del lenguaje de programación dataflow.
//   ],
// )

Con el prototipo del entorno listo, se vio que la aproximación de separación de responsabilidades entre el lenguaje y la visión por computador era viable y facilitaba el análisis y desarrollo del mismo, aunque surgió la preocupación de que la latencia introducida por la comunicación entre ambos apartados pudiera afectar la experiencia del usuario.

=== Prototipo 5

Con base en el diseño del ambiente, se planteó continuar con la interfaz gráfica del entorno de desarrollo integrado (IDE) para el lenguaje de programación, con el objetivo de crear una experiencia de usuario atractiva e intuitiva que facilitara la creación de programas mediante la manipulación de bloques físicos, si bien la integración con la detección de bloques se pospuso y se buscó probar la funcionalidad con bloques digitales.

El diseño propuesto puede verse en la @fifth-prototype-design-figure, y se enfocó en la creación de una interfaz gráfica que permitiera a los usuarios interactuar con el entorno de programación tangible de manera intuitiva, facilitando la creación de programas mediante la manipulación de bloques digitales que representaran las futuras piezas físicas. Se decidió llamar a esta interfaz "modo sandbox" del IDE.

Se implementaron características como la visualización del programa en tiempo real, la posibilidad de arrastrar y soltar bloques para crear programas, y una sección de resultados donde se mostraban los resultados de la ejecución del programa. Además, se buscó crear una experiencia de usuario atractiva mediante el uso de colores y una disposición clara de los elementos en la interfaz. Este prototipo fue desarrollado en TypeScript, usando la librería React para la construcción de la interfaz gráfica, la librería React Flow para la representación visual de los datos, operaciones y flujos de datos. // Este prototipo puede verse en la @fifth-prototype-figure.

//TODO: colocar imagen del prototipo
// #figure(
//   image("images/fifth-prototype.jpeg"),
//   caption: [
//     Quinto prototipo del ambiente, con la implementación de la interfaz gráfica del entorno de desarrollo integrado (IDE) para el lenguaje de programación.
//   ],
// ) <fifth-prototype-figure>

Al finalizar el desarrollo de la interfaz gráfica del modo sandbox, se vio que facilitaba la creación de programas mediante la manipulación de bloques digitales, y se planteó continuar con la integración de la detección de bloques físicos y el reconocimiento de estos por parte del entorno de ejecución del lenguaje de programación dataflow.

=== Prototipo 6

Continuando con el prototipo 5, se decidió integrarle la detección de piezas físicas mediante el Kinect v1, por dificultades temporales con el Kinect v2/* especificar que estas dificultades estaban relacinadas con problemass de compatabilidad de estándar USB y rendimiento */; y el uso de un nuevo modelo de detección de objetos basado en aprendizaje profundo, pues se cambió el diseño de las piezas físicas a usar, requiriendo de un reentrenamiento del modelo. Además, se planteó comenzar la integración con el entorno de ejecución del lenguaje de programación dataflow, optando por la integración mediante WebSockets para la comunicación.

Se llevó a cabo un rediseño de las piezas físicas a usar, buscando cubrir los datos y operaciones que se definieron para el lenguaje, un diseño sencillo de entender y usar para los niños, pero no tan complejo en aras de facilitar la detección por parte del modelo, resultando en un diseño tipo carta. Estas nuevas piezas pueden verse en la @sixth-prototype-pieces-design-figure. Además, también se hicieron modificaciones en la interfaz gráfica del modo sandbox, entre ellas usar colores oscuros, para facilitar la visualización de la proyección del entorno virtual sobre la superficie física.

Al entrenar el nuevo modelo de detección de objetos, se comenzó con el modelo YOLOv11-nano, con un dataset en el que las _bounding boxes_ comprendían toda la carta, incluyendo las etiquetas ("Operador", "Resta", "Tortuga", etc.), áreas blancas alrededor de la pieza, e imagen de la pieza; este modelo tenía dificultades para detectar las piezas, principalmente por la confusión entre clases, por lo que se decidió ajustar las _bounding boxes_ para que solo comprendieran el área de la imagen de la pieza, sin incluir las etiquetas ni áreas blancas, lo que llevó a una pequeña mejora en la detección, pero sin llegar a los resultados esperados. Finalmente, se cambió al modelo YOLOv11-small, una versión ligeramente más pesada y potente de YOLO que el nano, que ofrece una mejora significativa en la precisión de detección, lo que permitió obtener resultados satisfactorios en la detección de las piezas físicas. Además, se implementó una integración básica con el entorno de ejecución del lenguaje de programación dataflow mediante WebSockets, enviando las piezas reconocidas al entorno, pero sin las conexiones entre estas. Este prototipo puede verse en la @sixth-prototype-figure.

#figure(
  image("images/sixth-prototype.jpeg"),
  caption: [
    Sexto prototipo del ambiente, con la integración de la detección de piezas físicas mediante un nuevo modelo de detección de objetos.
  ],
) <sixth-prototype-figure>

Con este prototipo terminado, se vio que la integración de la detección de piezas físicas mediante el nuevo modelo de detección de objetos era viable y mejoraba significativamente la precisión de detección, aunque surgieron preocupaciones respecto al proceso de entrenamiento del modelo, tomando en cuenta que no se usó el lote completo de cartas que soporta el lenguaje, pero sí más que el número de piezas que se usaron en prototipos previos. La integración con el entorno de ejecución del lenguaje de programación dataflow mediante WebSockets también se mostró viable, pero la falta de un mecanismo para representar las conexiones entre las piezas reconocidas a nivel tangible, de modo que se pudieran enviar al entorno de ejecución; limitaba las pruebas que se podían hacer con esta integración.

=== Prototipo 7

El séptimo prototipo se concibió como la primera integración completa de los tres componentes del ambiente —el subsistema de visión por computador, el lenguaje de programación y la interfaz gráfica— en una experiencia unificada de extremo a extremo, atendiendo la principal limitación identificada en el sexto prototipo: la ausencia de un mecanismo para representar las conexiones entre las piezas físicas reconocidas. El desarrollo abarcó mejoras simultáneas en la calibración del entorno, en la arquitectura del lenguaje y en la integración entre la detección de piezas y su evaluación.

En el subsistema de visión, se sustituyó la transformación lineal de dos puntos heredada de los prototipos previos por una homografía de cuatro puntos, calculada mediante la función `cv2.getPerspectiveTransform`. A diferencia de la transformación anterior, que solo corregía traslación y escala, la homografía describe una proyección de ocho grados de libertad capaz de corregir la distorsión de perspectiva entre la cámara y la superficie proyectada, lo que mejora la precisión con que se ubican los toques sobre la mesa.
//TODO: el ADR de calibración por homografía (ADR-003) justifica la mejora de precisión para las coordenadas de los toques, pero no documenta explícitamente una mejora en la "detección de las piezas sobre la mesa"; verificar a mano si se desea sostener esa segunda afirmación o suavizarla.

En el lenguaje, se consolidó la arquitectura del entorno de ejecución. Lo que en el cuarto prototipo se había implementado como un compilador y un _runtime_ separados, comunicados con los demás componentes mediante servidores HTTP y de WebSockets construidos con la librería Elysia, se unificó en un único intérprete distribuido como librería. De este modo, el intérprete dejó de ejecutarse como un proceso aparte y pasó a integrarse directamente en la interfaz, que lo invoca en el propio cliente para evaluar los programas. Esta decisión eliminó la latencia introducida por la comunicación entre procesos —una de las preocupaciones surgidas en el cuarto prototipo— y simplificó el despliegue del sistema. En coherencia con esta evolución, la terminología se ajustó definitivamente al término intérprete, en sustitución del par compilador-_runtime_ empleado inicialmente.

Paralelamente, se rediseñó y simplificó el lenguaje para alinearlo con el enfoque concreto-pictórico-abstracto, derivado de los modos de representación enactivo, icónico y simbólico de #cite(<bruner1966>, form: "prose"). El modelo de datos, antes compuesto por múltiples tipos heterogéneos, se unificó en un objeto único caracterizado por una de tres categorías —concreto, pictórico o abstracto—, separando la semántica del lenguaje de su modo de presentación visual. Esta simplificación redujo la complejidad del intérprete y permitió reorganizar el conjunto de piezas físicas en torno a las tres categorías.
//TODO: la "disminución en la cantidad de piezas físicas" no pudo cuantificarse contra el historial (no se comparó el catálogo anterior con el nuevo); confirmar a mano la reducción neta o reformular como "reorganización" del repertorio.

La integración de la detección de piezas con el intérprete se articuló a través de la interfaz gráfica. El subsistema de visión, mediante un servidor de relevo (_relay_) construido con FastAPI que reemplazó los antiguos servidores Elysia; transmite a la interfaz los lotes de cartas detectadas, y la interfaz las representa como nodos sobre el lienzo de flujo de datos, para luego traducir el grafo resultante en un programa que entrega al intérprete para su evaluación. El problema de representar las conexiones entre piezas, pendiente desde el sexto prototipo, se resolvió mediante un sistema de puertos tipados y reglas estructurales que determinan qué piezas pueden enlazarse entre sí, complementado con la visualización del dato que circula por cada conexión mediante elementos animados denominados _walkers_. La elección de FastAPI para el servidor de relevo respondió a un criterio de cercanía tecnológica y de mínimo esfuerzo de implementación: dado que el subsistema de visión está escrito en Python, mantener el relevo sobre Bun y Elysia habría obligado a duplicar en ese entorno las definiciones de las interfaces de los mensajes de detección de piezas y de toques, ya existentes en Python; reimplementar el relevo con FastAPI permitió reutilizar directamente esas definiciones y evitar la duplicación de esas interfaces entre dos lenguajes.
//TODO: esta justificación procede de los autores; si se requiere trazabilidad formal, conviene redactar el ADR correspondiente, pues los ADR-006 y ADR-007 aún describen el stack Bun/Elysia ya deprecado.

//TODO: el esqueleto original anotaba "<primer vídeo>, muy mal rendimiento" como evidencia de esta etapa. No hay vídeos ni mediciones de rendimiento versionados en el repositorio; añadir a mano la referencia al material audiovisual y, de sostenerse el juicio de rendimiento, respaldarlo con datos.

//TODO: agregar figura del séptimo prototipo (integración inicial). Pendiente de imagen, siguiendo el patrón de @sixth-prototype-figure.

Con este prototipo se obtuvo, por primera vez, una experiencia integrada de extremo a extremo en la que las piezas físicas, sus conexiones y la salida proyectada conformaban un programa ejecutable de manera incremental, de modo que las mejoras posteriores se pudieron implementar sobre esta base, en forma de evoluciones, motivadas en un principio por las limitaciones observadas en la fluidez de la detección.

==== Evolución 1

La primera evolución se centró en la detección de toques. Se incorporó un detector híbrido que combina el algoritmo DIRECT/*añadir cita*/—responsable de la detección del toque a partir del sensor de profundidad, mediante relleno por inundación, zonas e histéresis— con la librería y modelo MediaPipe, que aporta la posición precisa de la punta del dedo índice cuando DIRECT detecta un contacto. Esta combinación separa la decisión de si existe un toque de la estimación de dónde ocurre, aprovechando la robustez del sensor de profundidad y la precisión de la estimación visual de la mano.
//TODO: el esqueleto anotaba que la detección se hizo "más eficiente" y mencionaba un "<segundo vídeo>" asociado a la primera valoración positiva del tutor. La mejora de eficiencia no está cuantificada en el repositorio (no hay _benchmarks_ ni mediciones de latencia versionadas) y no existe rastro del material audiovisual ni del _feedback_ del tutor; añadir y respaldar a mano.

==== Evolución 2

La segunda evolución comprendió dos mejoras. En la calibración, se generalizó la homografía de cuatro puntos a una de nueve, dispuestos en una rejilla de tres por tres y resuelta por mínimos cuadrados mediante `cv2.findHomography`, con el fin de mejorar la precisión del mapeo entre la cámara y la superficie proyectada. En cuanto a las piezas físicas, se amplió el repertorio del mazo más allá de las cartas, incorporando tapas, paletas y cubos como piezas tangibles adicionales reconocibles por el subsistema de visión, enriqueciendo la experiencia del usuario al interactuar con el enfoque concreto-pictórico-abstracto de forma más directa, con representaciones más fieles a lo propuesto por #cite(<bruner1966>, form: "prose").
//TODO: el cambio a nueve puntos está respaldado por la configuración del código (rejilla 3×3 por defecto en `config.py`), pero la mejora de precisión no está medida cuantitativamente. Asimismo, la incorporación de tapas, paletas y cubos consta en la interfaz y el catálogo de piezas, pero el reentrenamiento del modelo YOLO para esas piezas no está respaldado por el historial (el conjunto de clases versionado en `data.yaml` no las incluye); confirmar y documentar el reentrenamiento a mano.

==== Evolución 3

La tercera evolución consistió en una extensión del lenguaje, reflejada en la especificación formal presentada en el #link(<appendix-a>)[Apéndice A]. Se incorporaron cartas de grupos, que agrupan varios objetos en una sola colección; cartas de manipulación de grupos —primero (`first`), último (`last`) y contar (`count`)—; y la operación de comparación (`compare`). Asimismo, las operaciones de filtrado y ordenamiento, ya presentes en versiones anteriores del lenguaje, se reforzaron con un sistema de criterios que permite filtrar y ordenar las colecciones según propiedades como el color, el tamaño o la forma de los objetos.
//TODO: precisar, de ser necesario, que las operaciones de filtrado (`filter`) y ordenamiento (`order_asc`/`order_desc`) ya existían en el lenguaje desde prototipos anteriores; lo introducido en esta evolución es el sistema de criterios (gramática v4.0.0) y las operaciones `first`/`last`/`count`/`compare` (gramática v4.1.0).

// ==== Evolución 4 -> Dado, cambios del intérprete a array-first tras bambalinas sin afectar experiencia del usuario, zonas de persistencia de detección.

== Validar el Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años Construido

La validación del ambiente se aborda en tres frentes complementarios: una verificación interna que contrasta los requerimientos del sistema con las funcionalidades efectivamente construidas, mediante una matriz de trazabilidad; una evaluación por juicio de expertos en interacción humano-computador (IHC) y en medios didácticos; y pruebas de comprensión y usabilidad con niños del rango etario objetivo.
//TODO: cuando se ejecuten, integrar aquí los resultados de la evaluación por expertos (punto 3) y de las pruebas con niños (punto 4).

=== Matriz de requerimientos contra funcionalidades

Con el fin de verificar que el sistema construido responde a lo especificado, se elaboró una matriz que retoma los requerimientos funcionales (RF) y no funcionales (RNF) definidos durante el análisis en la @requirements-table, y relaciona cada uno con la funcionalidad que lo satisface y su estado de cobertura. La matriz se presenta en la @requirements-to-functionalities-matrix.

#figure(
  [
    #set text(size: 9pt)
    #table(
      columns: (auto, 1fr, 1.3fr, auto),
      align: (center + horizon, left + horizon, left + horizon, center + horizon),
      inset: 4pt,
      table.header([*Cód. Req.*], [*Requerimiento*], [*Funcionalidad que lo satisface*], [*Estado*]),
      [RF-01], [El sistema debe permitir a los niños construir programas utilizando elementos tangibles y conexiones digitales que representen datos, flujos y operaciones], [Detección de las piezas físicas con modelos YOLO y representación y validación de sus conexiones mediante puertos tipados y reglas estructurales en el IDE], [Satisfecho],
      [RF-02], [El sistema debe capturar la disposición de los elementos tangibles y conexiones digitales, y procesar la información para reconocer los elementos y sus conexiones], [Captura con cámara de color y profundidad; reconocimiento de cartas y de toques, calibración por homografía y relevo de los datos a la interfaz], [Satisfecho],
      [RF-03], [El sistema debe interpretar los programas representados por los elementos tangibles y conexiones digitales, traduciéndolos a una representación ejecutable], [Traducción del grafo visual a un programa e interpretación con el intérprete ERAE embebido], [Satisfecho],
      [RF-04], [El sistema debe ejecutar los programas y mostrar la salida en una interfaz gráfica proyectada sobre una superficie plana], [Ejecución con evaluación bajo demanda e incremental y visualización de la salida en la interfaz proyectada], [Satisfecho],
      [RF-05], [El sistema debe proveer retroalimentación para guiar a los niños durante la construcción de programas], [Evaluación incremental, resaltado de orígenes, zonas y conexiones, y walkers sobre las conexiones], [Satisfecho],
      [RNF-01], [El sistema debe ser usable por niños de 6 a 9 años y profesores de primaria de 1#super[er] a 3#super[er] grado], [Modos de juego y sandbox e interfaz basada en cartas; su usabilidad efectiva requiere comprobación con usuarios], [Pendiente],
      [RNF-02], [El sistema debe contener elementos persuasivos que capten el interés de niños de 6 a 9 años], [Elementos lúdicos implementados (diseño colorido, dado, síntesis de voz); su efecto en el interés requiere validación con niños], [Parcial],
      [RNF-03], [El sistema debe ser capaz de manejar errores en la disposición de los elementos tangibles y digitales], [Verificación de tipos y aridad, validación de conexiones en tres niveles y análisis de programas incompletos sin interrumpir la sesión], [Satisfecho],
      [RNF-04], [La retroalimentación debe ser presentada de forma visual y auditiva], [Retroalimentación visual completa (resaltados, walkers, resultados); la auditiva se limita a la síntesis de voz de los resultados, restan las señales sonoras de reconocimiento, advertencia y error], [Parcial],
    )
  ],
  caption: [
    Matriz de trazabilidad entre los requerimientos definidos en el análisis y las funcionalidades construidas, con la evidencia en el código y el estado de cobertura de cada requerimiento.
  ],
) <requirements-to-functionalities-matrix>

Como se observa en la @requirements-to-functionalities-matrix, los cinco requerimientos funcionales se encuentran satisfechos: la construcción y captura de los programas tangibles, el reconocimiento de los elementos y de las conexiones, la interpretación con evaluación incremental, la ejecución con salida proyectada y la retroalimentación que guía la construcción. De los requerimientos no funcionales, el manejo de errores de disposición está satisfecho, mientras que la presentación auditiva y los elementos persuasivos están parcialmente cubiertos y la usabilidad efectiva queda pendiente. Estos últimos, de naturaleza pedagógica y de experiencia, motivan los dos frentes de evaluación que se describen a continuación.

=== Evaluación por juicio de expertos

La calidad del ambiente como sistema interactivo y como recurso educativo se evalúa mediante juicio de expertos, con dos perfiles complementarios: un experto en interacción humano-computador (IHC), que valora la interfaz, la interacción tangible, la usabilidad y el manejo de errores; y un experto en medios didácticos, que valora la pertinencia pedagógica del ambiente, la progresión concreto-pictórico-abstracto y su alineación con el currículo. A cada experto se le aplica un instrumento estructurado (rúbrica o cuestionario) cuyos resultados se reportan como parte de la validación.
//TODO: pendiente de ejecución (punto 3). Definir los instrumentos de evaluación, seleccionar a los expertos y registrar y analizar sus valoraciones. Los instrumentos de evaluación deben incluirse como apéndices, incluyendo las respuestas de los expertos.

=== Pruebas de comprensión y usabilidad con niños

La validación se completa con pruebas realizadas con niños del rango etario objetivo, organizadas en torno a actividades modelo (con descripción de la actividad, roles, pasos a seguir y resultado esperado). Las pruebas de usabilidad observan si los niños logran usar el ambiente, contabilizando el éxito en las tareas, errores, necesidad de ayuda y satisfacción, mientras que las pruebas de comprensión observan si entienden lo que construyen y los conceptos de pensamiento computacional involucrados.
//TODO: pendiente de ejecución (punto 4). Definir el protocolo y las actividades modelo, las métricas de usabilidad y de comprensión, y registrar y analizar los resultados.

//* Vídeo mostrándole a Chilina León y comentarios que nos dió

//* Que dijo Andreina???

// == Realizar la Documentación Formal del Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años Construido

//* Mencionar apéndices de manual de sistema, manual de usuario y guía de ejercicios modelo

#pagebreak(weak: true)

// Capítulo V
= Capítulo V. Conclusiones y Recomendaciones
//TOOD: Revisar, ya así por encima puedo ver que las recomendaciones están en un mal formato
== Analizar el Uso de Programación Tangible en Entornos de Realidad Aumentada, a fin de Caracterizar el Ambiente a Desarrollar

El análisis del uso de la programación tangible en entornos de realidad aumentada permitió caracterizar el ambiente a desarrollar y confirmó que esta combinación constituye una vía pedagógica y técnicamente viable para fomentar el pensamiento computacional en niños entre 6 y 9 años sin depender del uso sostenido de pantallas. El estudio mostró que la manipulación de objetos físicos opera como un puente cognitivo hacia los conceptos abstractos de la programación, en correspondencia con la representación enactiva de #cite(<bruner1966>, form: "prose"), con la etapa de operaciones concretas descrita por #cite(<piaget1969>, form: "prose") y con el principio de activación de conocimientos previos de #cite(<merrill2002>, form: "prose"); este vínculo entre la teoría y las decisiones de diseño solo pudo establecerse al examinar conjuntamente los referentes pedagógicos y las experiencias previas de programación tangible.

Asimismo, el análisis evidenció que las herramientas más difundidas para la enseñanza de la programación infantil, basadas en pantallas, resultan poco compatibles con las recomendaciones pediátricas sobre el tiempo de exposición #cite(<aap2016>), lo que justifica el desplazamiento hacia interfaces tangibles con retroalimentación proyectada. De este contraste surgió la caracterización del ambiente: una interfaz tangible de tipo tabletop con acceso simultáneo de varios niños, retroalimentación visual mediante proyección sobre la superficie y un lenguaje de programación de flujo de datos en lugar del paradigma imperativo. La elección del paradigma de flujo de datos, en la línea de Lucid #cite(<wadge1985>), se fundamentó en que su modelo de ejecución se corresponde con la disposición espacial de las piezas y evita la abstracción de un hilo de ejecución secuencial, difícil de asir en esta etapa cognitiva. Finalmente, el antecedente Magicboard #cite(<barrios2024>) confirmó, en el contexto venezolano, la viabilidad de la realidad aumentada espacial con sensor de profundidad y proyector como base para el ambiente.

== Diseñar un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Función del Análisis Realizado

El diseño tradujo la caracterización resultante del análisis en una arquitectura concreta y en la especificación del lenguaje ERAE. Se concluye que la progresión concreto-pictórico-abstracto adoptada para el mazo de cartas operacionaliza los modos de representación enactivo, icónico y simbólico de #cite(<bruner1966>, form: "prose") y resulta coherente con la transición hacia la etapa de operaciones concretas; es decir, el marco teórico no permaneció como fundamento abstracto, sino que se materializó en una decisión de diseño verificable —la organización del repertorio físico de piezas— que solo pudo concretarse al elaborar el diseño.

En el plano del lenguaje, el diseño de flujo de datos, con declaraciones de fuente, transformación y salida, evaluación bajo demanda e incremental, y verificación estática de tipos y aridad, permite que los errores sintácticos queden eliminados por construcción y que la única fuente de error sea de naturaleza semántica, comunicada mediante mensajes orientados a los niños. Esta característica responde directamente a los requerimientos derivados del análisis: guiar la construcción del programa y prevenir los errores antes de la ejecución. La separación entre un núcleo de interpretación sin estado y adaptadores delgados, junto con una representación textual interna cuya gramática formal tolera programas incompletos (#link(<appendix-a>)[Apéndice A]), hizo realizable el diseño y habilitó la retroalimentación inmediata durante la construcción en vivo. El diseño mantuvo, además, su trazabilidad con el análisis al alinear el repertorio de datos y operaciones con los énfasis del currículo de educación primaria #cite(<mppe2023>).

== Construir un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Base al Diseño Realizado

La construcción del ambiente, llevada a cabo mediante un enfoque evolutivo basado en prototipos #cite(<pressman2010>), produjo un sistema integrado que articula el subsistema de visión por computador, el lenguaje ERAE con su intérprete y la interfaz de usuario. Se concluye que la metodología por prototipos fue determinante para un proyecto de naturaleza experimental con requerimientos inicialmente poco definidos: el resultado de cada prototipo definió el requerimiento del siguiente —la resolución insuficiente del Kinect v1 motivó el cambio al Kinect v2; la fragilidad de la detección por contornos condujo a la detección por aprendizaje profundo; la latencia entre procesos llevó a consolidar el intérprete como librería embebida; y la imposibilidad de representar las conexiones entre las piezas, evidenciada en el sexto prototipo, impulsó el sistema de puertos tipados, reglas estructurales y _walkers_ del séptimo—. Esta cadena de decisiones, que solo pudo establecerse al construir y evaluar sucesivamente el sistema, confirma la pertinencia del enfoque adoptado.

Entre los logros técnicos se cuentan el reconocimiento robusto de las piezas mediante modelos de detección de objetos, un intérprete del lenguaje ERAE que funciona como librería embebida con evaluación incremental y verificación de tipos y aridad, la calibración mediante homografía, la detección de toques con un detector híbrido y la integración entre la visión y el intérprete a través de un servidor de relevo y de la interfaz, que traduce el grafo visual de piezas y conexiones en un programa ejecutable. Con el séptimo prototipo se alcanzó, por primera vez, una experiencia integrada de extremo a extremo en la que las piezas físicas, sus conexiones y la salida proyectada constituyen un programa evaluable de manera incremental, con lo que el objetivo de construcción se considera cumplido en tanto el artefacto existe y opera.

No obstante, se concluye también que varias de las mejoras introducidas durante la construcción —en particular las relativas a la precisión de la calibración y al rendimiento de la detección— no fueron aún cuantificadas formalmente, por lo que su valoración corresponde al objetivo de validación, todavía pendiente. Esta distinción preserva la coherencia entre lo efectivamente construido y aquello que solo podrá afirmarse tras la validación empírica del ambiente.

// Validar el ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.
// Realizar la documentación formal del ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.
//TODO: pendientes la conclusión del objetivo general y las conclusiones de los objetivos 4 (validar) y 5 (documentar); deben anteceder a la sección de Recomendaciones cuando se redacten.

== Recomendaciones

A partir de la experiencia de construcción se recomienda documentar cuantitativamente el desempeño del sistema antes de la validación, estableciendo mediciones reproducibles de latencia y velocidad de procesamiento, error de calibración y precisión de la detección de piezas mediante métricas como la matriz de confusión. Estas mediciones convertirían en evidencia verificable varias de las mejoras hoy descritas de forma cualitativa.

En cuanto a la detección de piezas, se recomienda reentrenar el modelo con el lote completo de piezas previsto por el lenguaje —incluidas las incorporadas en las últimas evoluciones, como tapas, paletas, cubos y dado— y versionar tanto el conjunto de datos como su configuración, a fin de garantizar la reproducibilidad del entrenamiento.

Para preservar la trazabilidad arquitectónica, se recomienda mantener sincronizada la especificación viva del lenguaje con el informe y documentar, mediante una decisión de arquitectura formal, el reemplazo del servidor anterior por el servidor de relevo actual, dado que las decisiones de arquitectura existentes aún describen componentes ya superados. Asimismo, conviene documentar los requisitos de hardware y las dificultades de compatibilidad observadas con el sensor de profundidad, por su impacto en la estabilidad del sistema.

Finalmente, dado que las conclusiones alcanzadas son de orden técnico y de diseño, se recomienda planificar la validación del ambiente con niños del rango etario objetivo y con docentes de los primeros grados de educación básica, de modo que los beneficios pedagógicos que la teoría anticipa —el desarrollo del pensamiento computacional y el aprendizaje colaborativo— puedan confirmarse empíricamente. Para ello resulta aconsejable desarrollar y poner a prueba la guía de actividades alineada con el currículo y con los principios de la programación tangible colaborativa #cite(<suzuki1993>).

#pagebreak(weak: true)

#bibliography(
  "references.bib",
  style: "apa-6th-edition-no-ampersand.csl",
  title: [Referencias Bibliográficas],
)

#pagebreak(weak: true)

= Apéndice A. Gramática del Lenguaje ERAE <appendix-a>

A continuación se presenta la gramática formal del lenguaje ERAE en notación EBNF de la W3C, correspondiente a la versión 4.2.0 de la especificación del lenguaje. Los modificadores opcionales (`?`) en las reglas de declaración permiten el análisis sintáctico de programas incompletos, capacidad sobre la que se apoya la retroalimentación inmediata durante la construcción en vivo.

#[
  #set text(size: 10pt)
  #set par(first-line-indent: 0cm, justify: false)
  ```ebnf
  program             ::= statement*
  statement           ::= source_decl | transform_decl | sink_decl

  source_decl         ::= "source" identifier "=" (object_literal | group)? ";"
  transform_decl      ::= "transform" identifier "=" (operation "(" argument_list? ")")? ";"
  sink_decl           ::= "sink" identifier "=" identifier? ";"

  argument_list       ::= identifier ("," identifier)*

  operation           ::= "sum" | "substract" | "multiply" | "divide"
                        | "less_than" | "greater_than"
                        | "order_asc" | "order_desc" | "filter"
                        | "first" | "last" | "count" | "compare"

  group               ::= "[" (object_literal ("," object_literal)*)? "]"

  object_literal      ::= data_literal | criteria_literal

  data_literal        ::= "{" '"sourceType" : "data"' "," '"category"' ":" category_type ","
                          '"type"' ":" string_literal "," '"subtype"' ":" string_literal
                          ("," '"quantity"' ":" rational_literal)? ("," kv_pair)* "}"
  criteria_literal    ::= "{" '"sourceType" : "criteria"' "," '"properties"' ":" array_literal
                          ("," kv_pair)* "}"

  category_type       ::= '"abstracto"' | '"pictorico"' | '"concreto"'

  kv_pair             ::= string_literal ":" kv_value
  kv_value            ::= string_literal | rational_literal | array_literal

  array_literal       ::= "[" (string_literal ("," string_literal)*)? "]"
  rational_literal    ::= "-"? digit+ ("." digit+)?
  string_literal      ::= '"' [a-zA-Z0-9_-]* '"'
  identifier          ::= [a-zA-Z] [a-zA-Z0-9_-]*
  digit               ::= [0-9]
  ```
]

#pagebreak(weak: true)

// = Anexos
