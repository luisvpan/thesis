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

#let meses = ("Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre")
#let mes-actual = meses.at(datetime.today().month() - 1)

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
    [Fecha], [#mes-actual, #datetime.today().year()]
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

#cite(<papert1980>, form: "prose") predijo el auge de las computadoras en la educación, planteando que los niños deberían aprender a programar tal y como es aprender francés viviendo en Francia en vez de aprenderlo mediante las clases de lenguas extranjeras en las aulas del colegio; es decir, mediante la interacción directa con las computadoras, un enfoque en que el niño use y experimente con la computadora para aprender, en vez de que la computadora le enseñe al niño. También hace énfasis en que la simple presencia de las computadoras cambiaría y moldearía una nueva forma de enseñar y aprender, inimaginable para la sociedad de aquel momento. Papert fue un visionario, pues lo que él defendía se hizo realidad en partes, con una creciente demanda de la competencia del pensamiento computacional, término que #cite(<papert1980>, form: "prose", supplement: [p. 182]) empleó de forma incidental y que #cite(<wing2006>, form: "prose") popularizó para referirse a la forma de pensar de los científicos de computación; pero con retos que enfrentar.

#cite(<wing2006>, form: "prose"), con cuyo artículo resurge el interés por el pensamiento computacional #cite(<sanchezvera2019>), lo describe como una habilidad que todos deberían aprender y usar, no solo los científicos en computación. Defiende que debería añadirse a la educación de los niños, al mismo nivel que las 3R (lectura, escritura y aritmética), por los usos que tiene, no solo al aprender a programar, sino en la descomposición y solución de problemas, la creación de modelos, el análisis de datos, la abstracción. Añade que su utilidad en otras disciplinas ya es visible, notándose en cómo el aprendizaje automático ha transformado a la estadística, el reciente interés de los científicos de computación en la biología, o la computación cuántica y su efecto en la física. Wing plantea que, así como la computación ubicua pasó de ser un sueño a una realidad cotidiana, el pensamiento computacional lo será en el futuro, y propone como primer paso que deje de ser exclusivo de los científicos de computación y se enseñe a los estudiantes preuniversitarios.

// ¿Por qué es importante el pensamiento computacional en niños? ->
// ¿Por qué es un problema que los niños no desarrollen el pensamiento computacional? ->
// Se debe mencionar que programar/codificar se ha convertido en una competencia básica del siglo XXI (habilidades del siglo XXI), según Sánchez Vera, et al. (2019).
// Después de mencionar que el pensamiento computacional se considera como una nueva alfabetización
// y que programar/codificar se ha convertido en una competencia básica del siglo XXI:
// Se debe mencionar que el pensamiento computacional se considera como una nueva alfabetización, la alfabetización digital (Zapata-Ros, 2015). Texto que referencia esto:

#cite(<zapata2015>, form: "prose") sostiene que el pensamiento computacional representa una nueva alfabetización digital que debe comenzar desde las primeras etapas del desarrollo individual, al igual que sucede con otras habilidades clave como las 3R. Esta alfabetización no se limita únicamente al aprendizaje de la programación, sino que permite a las personas organizar su entorno, desarrollar estrategias de desenvolvimiento y resolución de problemas cotidianos, además de organizar su mundo de relaciones en un contexto de comunicación más racional y eficiente, resultando en una mayor calidad de vida.

La ausencia del desarrollo del pensamiento computacional en los niños se ha convertido en un problema relevante en la sociedad actual. #cite(<sanchezvera2019>, form: "prose") señala que codificar ha sido incluido específicamente como una de las competencias básicas del siglo XXI, y que el pensamiento computacional permite desarrollar una nueva alfabetización necesaria en el mundo contemporáneo, ayudando a que los individuos no sean solo consumidores digitales, sino creadores y participantes activos con las tecnologías. La falta de estas competencias limita las capacidades de los niños para desenvolverse eficazmente en un contexto cada vez más digitalizado, reduciendo su potencial para resolver problemas complejos y expresar sus ideas mediante la tecnología. Además, como indica #cite(<zapata2015>, form: "prose"), la carencia de estas habilidades desde edades tempranas dificulta que en ciclos superiores los estudiantes puedan desarrollar plenamente el pensamiento computacional, ya que no cuentan con las bases cognitivas necesarias que se construyen mediante la manipulación de objetos y conceptos fundamentales como la seriación, la discriminación por propiedades y la secuenciación.

En la actualidad, la presencia de computadoras, tabletas, teléfonos, televisores y relojes inteligentes, y demás dispositivos con pantalla; resulta en que los seres humanos están expuestos a las pantallas durante todo el día, en períodos de tiempo extensos incluso en ambientes dedicados a la enseñanza y el aprendizaje, como los colegios y universidades; causando la preocupación generalizada por los efectos a corto y largo plazo de esto, especialmente en los niños. La Academia Americana de Pediatría (AAP) recomendaba limitar el uso de pantallas en niños de 2 a 5 años a una hora diaria de contenido de alta calidad #cite(<aap2016>). Su política más reciente trasciende la noción de "tiempo de pantalla" y, para niños de 6 a 12 años, señala que el uso excesivo de medios digitales se asocia con menor rendimiento académico, peor control de la atención y un estilo de vida más sedentario, mientras que los medios de alta calidad con objetivos de aprendizaje, usados con moderación, pueden favorecer el aprendizaje de la matemática y la lectura #cite(<aap2026>). En el contexto local, la docente Jackeline Duarte, quien enseña programación y robótica a niños en Multiplayer Pzo., un centro de entretenimiento y formación tecnológica de Ciudad Guayana, limita a 15 minutos los periodos continuos de exposición a pantallas, lo que condiciona el uso de herramientas como Scratch durante sus sesiones de clase, y recurre en su lugar a herramientas tangibles, como el juego de mesa Mouse Mania, que, según su experiencia, abarcan un repertorio reducido de conceptos y dificultan fomentar el aprendizaje colaborativo y la socialización entre niños (J. Duarte, comunicación personal, 28 de febrero de 2025).

A esto se suma que, en Venezuela, los énfasis curriculares vigentes para la educación primaria no contemplan de forma explícita el pensamiento computacional ni la programación, aunque sí capacidades afines: el énfasis Matemática para la Vida plantea desarrollar el pensamiento lógico mediante abstracciones asociadas a los números y a las relaciones de proporcionalidad y de orden, enlazar lo abstracto con lo concreto y adoptar modelos de aprendizaje como el trabajo colaborativo #cite(<mppe2023>). En consecuencia, el fomento del pensamiento computacional en los primeros grados depende en buena medida de la iniciativa de cada docente y de los recursos de los que disponga, como muestra la experiencia de Duarte.

De mantenerse esta situación, la enseñanza del pensamiento computacional en los primeros grados seguiría dependiendo de herramientas basadas en pantallas, cuyo uso los docentes deben restringir, o de juegos tangibles con un repertorio reducido de conceptos y escasas oportunidades de colaboración. Los niños llegarían así a los ciclos superiores sin las bases que, según #cite(<zapata2015>, form: "prose"), se construyen mediante la manipulación de objetos, mientras que la alternativa de aumentar su exposición a pantallas se asocia con un menor rendimiento académico y un peor control de la atención #cite(<aap2026>).

La primera aproximación al uso de herramientas tangibles para la enseñanza de conceptos de programación a niños vino de parte de Radia Perlman, pionera de la programación para niños pequeños, quien desarrolló entre 1974 y 1976 un sistema de programación tangible llamado TORTIS, con la finalidad de que niños de 3 a 5 años pudieran acceder a las ventajas de aprender lenguajes de programación completos al interactuar con objetos físicos (Perlman, 1976, citado en #cite(<morgado2006>, form: "author"), #cite(<morgado2006>, form: "year")). Este sistema consistía en controlar una pequeña “tortuga” (un disco equipado con una luz, una bocina y un lápiz, este último encargado de dibujar el resultado de la ejecución del programa) mediante uno de dos componentes: una serie de cajas de botones con acciones, o una “máquina tragacartas” con cartas de plástico. La razón detrás de la creación de dos componentes para interactuar con el lenguaje es, según la interpretación de la propia Perlman, que con las cajas de botones los niños pensaban que el programa era el dibujo resultante, en vez del conjunto de comandos que ejecutaban con los botones; mientras que, con la máquina tragafichas, que construyó para solucionar el problema de las cajas, era difícil que los niños entendieran que cada carta estaba asociada a un comando porque, para la ejecución de cada una, se debía buscar la carta, insertarla en una ranura de la máquina y presionar un botón. Incluso con estos problemas, Perlman llegó a una nueva aproximación para la enseñanza de conceptos de programación para los niños.

Un ejemplo más conocido de programación tangible es el caso de AlgoBlock #cite(<suzuki1993>), un lenguaje de programación tangible inspirado en Logo que consiste en unir una serie de bloques físicos para formar un programa que controla un submarino mostrado en una pantalla. Igual que las cartas en el segundo componente de TORTIS, cada bloque representa un comando, y algunos bloques representan estructuras de control condicionales y de bucle. Lo interesante de AlgoBlock son los principios que siguieron sus autores en su desarrollo: facilidad de uso, acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos; los cuales incitan la conversación y la colaboración entre los participantes. Aunque los autores no mencionan alguna deficiencia en AlgoBlock tras ponerlo a prueba, sí hacen énfasis en que este solo representa el primer paso en la identificación de los principios para el diseño de ambientes de aprendizaje colaborativo.

La idea detrás de la programación tangible es llevada más allá por #cite(<zapata2019>, form: "prose"), quien habla sobre el pensamiento computacional desenchufado, que ayuda a que los niños adquieran las competencias relacionadas al pensamiento computacional en las etapas de su vida en que más la necesiten, como durante los estudios secundarios o la universidad, basándose en los principios fundamentales de la instrucción propuestos por #cite(<merrill2002>, form: "prose"), especialmente en el principio de activación. Zapata-Ros describe al pensamiento computacional desenchufado como actividades que fomenten en los niños una serie de habilidades que, tras ser evocadas en ciclos superiores, favorezcan el desarrollo del pensamiento computacional, ejemplos de estas son el uso de fichas, juegos en el salón de clase o en el patio, juguetes, aquellas que se suelen hacer sin el uso de pantallas ni computadoras. Además, sugiere actividades ya existentes con estas características y la forma de usarlas para promover el pensamiento computacional.

Aparte del pensamiento computacional desenchufado, la realidad aumentada espacial presenta afinidades con la programación tangible: integra la tecnología de visualización al entorno, en lugar de llevarla sobre el usuario y, en su variante basada en proyección, superpone contenido virtual directamente sobre objetos y superficies físicas #cite(<bimber2005>). La realidad aumentada, en general, ya se ha llevado al aula. #cite(<billinghurst2012>, form: "prose") revisan experiencias en educación primaria y secundaria, principalmente con libros aumentados (libros impresos sobre cuyas páginas se superponen imágenes virtuales, vistas mediante un visor de mano) y aplicaciones móviles, y reportan estudios propios: un libro aumentado sobre electromagnetismo produjo mejores resultados que su versión impresa, tanto de inmediato como cuatro semanas después; en la lectura de cuentos, los niños con menor habilidad lectora recordaron el contenido de las secciones interactivas tan bien como sus compañeros; y la creación de escenas propias con herramientas de autoría, al alcance de niños desde los siete años, resultó en sí misma una experiencia de aprendizaje. Si bien estas experiencias no emplean proyección, muestran el potencial educativo de superponer contenido virtual sobre objetos físicos.

Pasando a Venezuela, #cite(<barrios2024>, form: "prose") de la Universidad Católica Andrés Bello (UCAB) presentó un entorno de realidad aumentada espacial que consiste en una mesa interactiva táctil, la cual permite el desarrollo de juegos sociales entre niños de educación preescolar y básica que promueven la colaboración y socialización. Con base en lo expuesto previamente, se ve la oportunidad de extender el entorno hecho por Barrios incluyendo un módulo de programación tangible, aprovechando el enfoque en juegos sociales para fomentar la colaboración, la socialización, el trabajo en equipo y el pensamiento computacional.

A partir de lo expuesto, se plantea la siguiente interrogante: ¿cómo desarrollar un ambiente de programación tangible con realidad aumentada espacial que permita a niños entre 6 y 9 años, guiados por el docente, ejercitar el pensamiento computacional de forma colaborativa y sin exposición sostenida a pantallas?

En este contexto, se propone desarrollar un ambiente de programación tangible con realidad aumentada espacial basado en los referentes mencionados, en el que niños entre 6 y 9 años, guiados por el docente, construyan en conjunto programas de flujo de datos sobre una mesa, manipulando cartas físicas que representan colecciones de objetos y operaciones para clasificarlas, filtrarlas, ordenarlas, contarlas, compararlas y operar con sus cantidades, mientras el resultado se proyecta sobre la misma superficie. La finalidad es promover el desarrollo del pensamiento computacional desde edades tempranas, mediante un enfoque que favorezca el aprendizaje colaborativo y un desarrollo integral que vaya más allá del cognitivo.

=== Objetivo General

Desarrollar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años.

=== Objetivos Específicos

+ Analizar el uso de programación tangible en entornos de realidad aumentada espacial, a fin de caracterizar el ambiente a desarrollar.
+ Diseñar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, en función del análisis realizado.
+ Construir un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, en base al diseño realizado.
+ Validar el ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.
+ Realizar la documentación formal del ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.

== Alcance

El presente trabajo tiene como objetivo desarrollar un ambiente de programación tangible con realidad aumentada espacial para fomentar el pensamiento computacional en niños entre 6 y 9 años de edad, considerando el aprendizaje colaborativo y la socialización. El ambiente está concebido para ser usado por docentes y niños en conjunto, con el docente como conductor o guía de la actividad.

En primer lugar, se revisan los conceptos relacionados con el pensamiento computacional y su desarrollo en edades tempranas. A continuación, se analiza la aplicación de la programación tangible en entornos de realidad aumentada espacial, con el fin de caracterizar el ambiente a desarrollar.

Posteriormente, se diseña y construye el ambiente, lo que comprende dos aspectos: el hardware, que funciona como interfaz de interacción humano-computador, y el software, encargado de procesar la información recibida a través del hardware. El ambiente construido se valida mediante una matriz de trazabilidad entre requerimientos y funcionalidades y mediante el juicio de expertos en interacción humano-computador y en medios didácticos.

Finalmente, se elabora la documentación del ambiente, que comprende el manual del sistema y el manual de usuario.

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

#cite(<espejo2022>, form: "prose"), en el trabajo de grado titulado _Entorno de robótica educativa multiagente orientado a favorecer el desarrollo del pensamiento computacional en jóvenes cursantes de educación media_, presentado ante la Universidad Católica Andrés Bello, tuvo como objetivo general desarrollar dicho entorno. Mediante una metodología basada en el modelo espiral se definieron los requisitos y se diseñó un entorno que obtiene información del mundo físico mediante visión por computador y la procesa para que los robots sigan las instrucciones programadas en MakeCode.

Aunque en este entorno la programación se realiza en pantalla, su aporte al presente trabajo radica en que los conceptos de programación se manifiestan físicamente en el comportamiento de los robots, y en que emplea la visión por computador para vincular el mundo físico con la ejecución del programa, principio que el ambiente propuesto retoma para reconocer las piezas tangibles, en este caso con niños entre 6 y 9 años.

#cite(<barrios2024>, form: "prose"), en el trabajo de grado titulado _Entorno de realidad aumentada espacial para el desarrollo de juegos sociales dirigidos a niños de educación preescolar_, presentado ante la Universidad Católica Andrés Bello, tuvo como objetivo general desarrollar dicho entorno. El producto resultante, “Magicboard”, es una pizarra digital en forma de mesa con la que los niños interactúan a través de un sensor que detecta gestos y objetos físicos, y que les permite aprender mediante juegos sociales cuyo pilar es el aprendizaje colaborativo. Tras una etapa de investigación en la que se obtuvieron las características esenciales de la pizarra, considerando la manera en que los niños interactúan y aprenden, se construyó con un proyector, un sensor Kinect y el software correspondiente para la gestión de la interacción.

Su aporte es directo, pues el presente trabajo constituye una continuación de “Magicboard”: retoma su configuración de mesa, proyector y sensor de profundidad, así como su enfoque en el aprendizaje colaborativo, para añadirle el fomento del pensamiento computacional a través de la programación tangible.

#cite(<rojas2024>, form: "prose"), en el trabajo de grado titulado _Sistema interactivo para la enseñanza de programación a niños con discapacidad visual_, presentado ante la Universidad Católica Andrés Bello, tuvo como objetivo general desarrollar dicho sistema. En él, los niños programan con bloques físicos cuyas características táctiles les permiten reconocer el significado de cada uno y construir una secuencia de instrucciones, que un sistema de visión por computador analiza a partir de la conexión entre los bloques para ejecutarla.

Su aporte al presente trabajo es el uso de bloques físicos reconocidos por visión por computador como interfaz de programación tangible, que constituye un punto de partida para los prototipos del ambiente.

#cite(<perezmarin2020>, form: "prose"), investigadores de la Universidad Rey Juan Carlos (España), publicaron el artículo _Can computational thinking be improved by using a methodology based on metaphors and scratch to teach computer programming to children?_ [¿Se puede mejorar el pensamiento computacional mediante el uso de una metodología basada en metáforas y Scratch para enseñar programación a los niños?], cuyo objetivo fue determinar si el pensamiento computacional de los niños puede mejorarse mediante una metodología basada en metáforas y el uso de Scratch para enseñar programación. Para ello, llevaron a cabo experimentos con niños de educación primaria y emplearon Scratch y la aplicación CompThink como medios para evaluar su aprendizaje.

Su aporte al presente trabajo es mostrar que los entornos de programación por bloques, como Scratch, son útiles para enseñar programación a los niños, lo que los convierte en un referente para los prototipos del ambiente, y que el uso de metáforas puede apoyar la comprensión de los conceptos de programación, idea afín a los objetos tangibles del ambiente.

#cite(<montes2021>, form: "prose"), de la Universidad Rey Juan Carlos (España), publicaron el artículo _Using an online serious game to teach basic programming concepts and facilitate gameful experiences for high school students_ [Usando un juego serio en línea para enseñar conceptos básicos de programación y facilitar experiencias divertidas para estudiantes de secundaria], cuyo objetivo fue determinar si el juego serio DFD-C, en el que los jugadores arman diagramas de flujo arrastrando sus elementos sobre un tablero, mejora el aprendizaje de los fundamentos de programación en estudiantes de secundaria, y cómo perciben estos la experiencia de juego. En un experimento con 38 estudiantes de 15 y 16 años de una escuela de Latacunga (Ecuador), divididos aleatoriamente en un grupo de control y uno de prueba, el grupo que usó el juego mejoró significativamente sus puntuaciones y reportó una experiencia de juego positiva, sin diferencias por género.

Aunque se dirige a estudiantes de mayor edad, su aporte al presente trabajo radica en la gamificación como medio para aprender conceptos de programación de manera lúdica, y en los elementos de diseño que emplea para ello, como el ensamblaje de piezas sobre un tablero, un sistema de incentivos y retroalimentación, y sonidos que dirigen la atención del jugador en momentos específicos del juego, aplicables a la retroalimentación del ambiente propuesto.

== Bases Teóricas

=== Teoría del Aprendizaje

Las teorías del aprendizaje proporcionan el marco conceptual fundamental para comprender cómo los individuos adquieren, procesan y retienen conocimiento #cite(<schunk2012>). #cite(<ertmer1993>, form: "prose") comparan tres de las perspectivas que más han influido en el diseño de la instrucción: el conductismo, que enfatiza la modificación de comportamientos mediante estímulos y refuerzos; el cognitivismo, que se centra en los procesos mentales internos y la organización del conocimiento; y el constructivismo, que postula que el conocimiento se construye activamente mediante la experiencia y la interacción con el entorno #cite(<woolfolk2014>) #cite(<piaget1978>). Este trabajo se sitúa en la perspectiva constructivista, pues el ambiente propuesto se basa en que los niños construyan programas mediante la manipulación de objetos.

El constructivismo de #cite(<piaget1969>, form: "prose") postula que el conocimiento se construye activamente a través de la interacción entre el individuo y su entorno, mediante procesos de asimilación y acomodación. Según esta perspectiva, los niños no son receptores pasivos de información, sino que construyen activamente su comprensión del mundo a partir de sus experiencias y estructuras cognitivas preexistentes. El aprendizaje ocurre cuando los niños interactúan con el entorno y reorganizan sus esquemas mentales para incorporar nueva información. Papert, quien trabajó con Piaget en Ginebra antes de incorporarse al MIT #cite(<papert1980>), extendió el constructivismo hacia su propia teoría del aprendizaje: el construccionismo. #cite(<harel1991>, form: "prose") explican que este comparte con el constructivismo la idea del aprendizaje como construcción de estructuras de conocimiento, y añade que tal construcción ocurre de manera especialmente favorable cuando el aprendiz se involucra conscientemente en construir una entidad pública, sea un castillo de arena o una teoría del universo. El objeto construido funciona así como un objeto con el cual pensar #cite(<papert1980>). Esta perspectiva se alinea con el enfoque de programación tangible, donde los niños construyen programas físicos que pueden ser ejecutados, observados, modificados y compartidos.

El aprendizaje colaborativo se fundamenta en teorías socioconstructivistas que, como expone #cite(<ackermann2010>, form: "prose"), enfatizan el papel de la interacción social y de la mediación cultural en el aprendizaje. La colaboración permite que los niños aprendan de sus pares, desarrollen habilidades de comunicación, y construyan conocimiento colectivamente mediante la discusión y negociación de significados. El aprendizaje colaborativo en niños promueve el desarrollo de habilidades sociales, mejora la comprensión mediante la explicación a otros, fomenta el pensamiento crítico a través de la discusión, y desarrolla habilidades de trabajo en equipo. En el ámbito de la programación tangible, estos beneficios se reconocieron de forma temprana: #cite(<suzuki1993>, form: "prose") concibió AlgoBlock como una herramienta de aprendizaje colaborativo en la que la manipulación compartida de bloques físicos promueve la interacción entre pares y la resolución conjunta de problemas.

En el contexto educativo, la selección y el diseño de los medios didácticos es fundamental para facilitar el proceso de aprendizaje #cite(<area2009>). Un medio didáctico #cite(<area2009>) es el soporte o tecnología que representa y vehicula el contenido educativo mediante determinadas formas y sistemas de codificación, y que no constituye un mero canal de transmisión, sino que estructura el propio proceso de aprendizaje. Los medios didácticos determinan cómo se presenta la información al aprendiz. Diferentes medios (visual, auditivo, táctil, kinestésico) activan diferentes canales sensoriales y cognitivos, influyendo en cómo se procesa y retiene la información. La realidad aumentada espacial, por ejemplo, constituye un medio didáctico que presenta información visual superpuesta sobre el mundo físico, activando canales visuales y espaciales para facilitar la comprensión de conceptos abstractos mediante representaciones visuales concretas.

=== Desarrollo Cognitivo en Niños de 6 a 9 Años

Según la teoría de desarrollo cognitivo de #cite(<piaget1969>, form: "prose"), existen cuatro etapas principales en el desarrollo cognitivo: la etapa sensorio-motora, que va desde el nacimiento hasta aproximadamente los dos años, y donde el conocimiento se obtiene mediante la interacción física con el entorno inmediato; la etapa preoperacional, aproximadamente entre los dos y siete años, caracterizada por el egocentrismo y la dificultad para realizar operaciones mentales complejas, aunque ya se comienza a ganar la capacidad de utilizar objetos simbólicos y adoptar roles ficticios; la etapa de las operaciones concretas, aproximadamente entre los siete y doce años, en la que se empieza a usar la lógica para llegar a conclusiones válidas con situaciones concretas y los sistemas de categorías se vuelven más complejos; y, finalmente, la etapa de las operaciones formales, desde los doce años en adelante, donde se desarrolla la capacidad para utilizar la lógica con conceptos abstractos y el razonamiento hipotético-deductivo.

La etapa de desarrollo cognitivo que corresponde a niños entre 6 y 9 años se encuentra en una transición crucial: a los 6 años aún están en la etapa preoperacional, donde el egocentrismo sigue presente y el pensamiento mágico basado en asociaciones simples predomina; sin embargo, alrededor de los 7 años, acceden a la etapa de las operaciones concretas, donde comienzan a usar la lógica para situaciones concretas y el egocentrismo disminuye notablemente. Esta transición marca un período fundamental de desarrollo donde los niños empiezan a manipular información lógicamente, siendo capaces de realizar operaciones mentales sobre objetos concretos, pero aún tienen limitaciones para trabajar con conceptos abstractos puros. Esta característica es fundamental para el diseño de ambientes de programación tangible, ya que la manipulación física de objetos permite a los niños comprender conceptos abstractos de programación a través de la experiencia concreta. El aprendizaje mediante la manipulación física de objetos se corresponde con lo que #cite(<bruner1966>, form: "prose") denomina representación enactiva: el modo de conocimiento en el que la comprensión se construye a través de la acción directa sobre el entorno. Este enfoque resulta especialmente pertinente en niños de edades tempranas, en quienes las actividades concretas y manipulables constituyen una vía eficaz para introducir conceptos computacionales #cite(<zapata2019>). La interacción con objetos tangibles permite que los niños comprendan conceptos abstractos mediante la experiencia sensorial y motora, facilitando la internalización de conocimientos complejos.

=== Pensamiento Computacional en Edades Tempranas

Wing (#cite(<wing2006>, form: "year"), #cite(<wing2008>, form: "year"), #cite(<wing2011>, form: "year")) define el pensamiento computacional como un conjunto de habilidades que incluyen la formulación de problemas de manera que permita el uso de computadoras para resolverlos, la organización y análisis lógico de información, la representación mediante abstracciones, y la búsqueda de soluciones efectivas. Si bien no existe consenso sobre cómo incorporar el pensamiento computacional en el aula, #cite(<sanchezvera2019>, form: "prose") identifica una tendencia creciente a valorar su trabajo desde la educación infantil y primaria, niveles en los que su desarrollo suele abordarse de manera transversal. Para niños de 6 a 9 años, estos componentes deben adaptarse al nivel de desarrollo cognitivo propio de la etapa de las operaciones concretas #cite(<piaget1969>), utilizando representaciones concretas y manipulables. El desarrollo del pensamiento computacional en niños requiere de estrategias pedagógicas específicas que consideren las limitaciones cognitivas propias de la edad; en esa línea, el pensamiento computacional desenchufado #cite(<zapata2019>) y los lenguajes de programación tangible #cite(<morgado2006>)#cite(<suzuki1993>) muestran que es posible experimentar con conceptos computacionales sin requerir habilidades de lectura avanzadas ni conocimiento previo de la sintaxis de lenguajes de programación tradicionales. En ese marco, #cite(<sanchezvera2019>, form: "prose") propone abordar el pensamiento computacional desde la Tecnología Educativa: no como un fin en sí mismo, sino como un medio para expresar ideas con tecnología y para aprender con herramientas, no de herramientas. Desde una perspectiva formal, #cite(<aho2012>, form: "prose") vincula el pensamiento computacional con los modelos de computación: pensar computacionalmente implica formular problemas en términos de los pasos que un modelo de computación puede ejecutar.

La sociedad digital contemporánea demanda profesionales cualificados en industrias tecnológicas, lo que ha llevado a reconocer la necesidad del pensamiento computacional, que, para #cite(<zapata2015>, form: "prose"), es una nueva alfabetización: la alfabetización digital. Como cualquier alfabetización fundamental, debe iniciarse desde las primeras etapas del desarrollo individual. Sin embargo, la codificación es solo la manifestación más visible de una forma de pensar que trasciende el ámbito de la programación: una manera particular de organizar ideas y representaciones que favorece las competencias computacionales y que puede cultivarse desde edades tempranas mediante actividades y entornos de aprendizaje apropiados.

#cite(<zapata2019>, form: "prose") propone el pensamiento computacional desenchufado como un enfoque que permite a los niños desarrollar competencias computacionales sin el uso de pantallas o computadoras. Las actividades desenchufadas preparan a los niños para conceptos que serán evocados en ciclos superiores de aprendizaje, estableciendo una base sólida que facilitará la transición hacia herramientas más avanzadas cuando estén cognitivamente preparados. Este enfoque se basa en los principios fundamentales de la instrucción propuestos por #cite(<merrill2002>, form: "prose"), que incluyen:

- *Enfoque centrado en tareas o problemas*: Diseño de instrucción alrededor de problemas auténticos del mundo real que los aprendices probablemente encontrarán, fomentando la participación activa en actividades de resolución de problemas.

- *Activación*: Se enfoca en involucrar los conocimientos previos y experiencias de los aprendices para crear una base para el nuevo aprendizaje. Implica estimular la curiosidad, presentar ejemplos del mundo real y conectar nueva información con conocimientos existentes.

- *Demostración*: Proporcionar modelos o ejemplos claros que ilustren los resultados de aprendizaje deseados, permitiendo que los aprendices observen actuaciones de expertos, simulaciones o estudios de caso para desarrollar comprensión.

- *Aplicación*: Ofrecer oportunidades para que los aprendices practiquen y apliquen sus conocimientos y habilidades en contextos auténticos, requiriendo que resuelvan problemas, tomen decisiones y se involucren en tareas realistas.

- *Integración*: Promover la transferencia de conocimientos y habilidades a nuevas situaciones, proporcionando oportunidades para conectar el aprendizaje con contextos del mundo real y aplicarlo de manera significativa.

El principio de activación es particularmente crucial para este trabajo, ya que este permite construir sobre experiencias previas con objetos físicos, juegos y manipulaciones concretas que los niños ya comprenden. Al activar conocimientos previos sobre manipulación de objetos, secuencias de acciones y relaciones causa-efecto que los niños han experimentado en sus actividades cotidianas, se establece un puente cognitivo fundamental que facilita la transición hacia conceptos computacionales abstractos a través de la experiencia concreta, haciendo que el aprendizaje sea más accesible y significativo.

Los principios fundamentales de instrucción, se integran dentro del marco más amplio del diseño instruccional #cite(<merrill2002>), que se refiere al proceso sistemático de planificar, desarrollar, implementar y evaluar experiencias de aprendizaje efectivas. Este campo interdisciplinario combina teorías de aprendizaje, metodologías pedagógicas y principios de diseño para crear entornos educativos que faciliten la adquisición de conocimientos y habilidades; en esa línea, #cite(<ackermann2010>, form: "prose") examina las raíces compartidas y las diferencias entre el constructivismo de Piaget y el construccionismo de Papert como fundamentos de este tipo de entornos. El diseño instruccional implica la identificación de objetivos de aprendizaje, la selección de estrategias pedagógicas apropiadas, la organización de contenido, y el diseño de actividades y evaluaciones que promuevan aprendizajes efectivos y duraderos. También se derivan las estrategias didácticas, que son los métodos y técnicas específicos que se emplean para facilitar el proceso de enseñanza-aprendizaje #cite(<ertmer1993>). Estas estrategias se adaptan según el contexto, los objetivos de aprendizaje y las características de los aprendices. Las estrategias didácticas pueden diseñarse intencionalmente para alinearse con los principios de instrucción y maximizar la efectividad del proceso educativo #cite(<ertmer1993>), especialmente cuando se trata de conceptos abstractos que deben ser comprendidos a través de manipulaciones concretas.

=== Programación Tangible

La base conceptual de los ambientes de programación tangible es la de las interfaces de usuario tangibles (TUI). #cite(<ishii2008>, form: "prose") las define como un paradigma de interacción que da forma física a la información digital para que los usuarios la manipulen directamente con sus manos, en contraste con las interfaces gráficas (GUI), donde representación y control están desacoplados. En una TUI, el objeto físico cumple una doble función: sirve simultáneamente como control y como representación de la información subyacente. Ishii identifica tres propiedades de este paradigma: el acoplamiento computacional entre las representaciones tangibles y la información y el cómputo subyacentes; la función de esas representaciones como mecanismos de control interactivo; y el acoplamiento perceptual entre ellas y las representaciones intangibles —gráficos y sonido— que las acompañan @ishii2008[pp. 473-474]. Para lograr este último, Ishii señala como requisito esencial la coincidencia de los espacios de entrada y salida: el espacio donde el usuario actúa y aquel donde recibe la retroalimentación deben ser el mismo. Estas propiedades tienen una consecuencia pedagógica directa: el niño no necesita traducir mentalmente la acción sobre un dispositivo remoto en un efecto sobre una pantalla separada, sino que actúa directamente sobre el objeto que representa la información, reduciendo la carga cognitiva de la interacción.

Dentro de la taxonomía de géneros de TUI de #cite(<ishii2008>, form: "prose"), las tabletop TUI son las más relevantes para este trabajo. En ellas, objetos tangibles discretos se manipulan sobre una superficie horizontal y la retroalimentación visual se proyecta sobre esa misma superficie, manteniendo la coincidencia de entrada y salida. Esta característica habilita naturalmente la colaboración colocalizada: al ser la entrada espacialmente multiplexada, pues cada objeto ocupa su propio espacio y puede ser manipulado por distintos usuarios al mismo tiempo; se favorece la participación concurrente sin los turnos forzados que impone una GUI. A ello se suma la persistencia de los tangibles: los objetos físicos mantienen su estado de forma autónoma, de modo que el programa construido por los niños es visible y modificable en todo momento sin mediación del sistema digital. @ishii2008[pp. 475, 483-484]

La programación tangible aplica este paradigma a la construcción de programas: cada objeto físico representa un comando, un dato o una operación, y el programa se construye mediante la organización física de esos objetos. Tiene sus raíces en trabajos pioneros como TORTIS de Perlman #cite(<morgado2006>) y AlgoBlock de #cite(<suzuki1993>, form: "prose"), que mostraron que es posible introducir conceptos de programación a niños mediante la manipulación de objetos físicos, sin sintaxis textual.

#cite(<suzuki1993>, form: "prose") identificaron principios clave para el diseño de ambientes de programación tangible colaborativos: facilidad de uso, acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos. Estos principios fomentan la conversación y colaboración entre participantes, aspectos esenciales para el aprendizaje colaborativo. A ellos se suman los beneficios que se derivan de las propiedades de las TUI: la entrada multiplexada en el espacio permite la manipulación concurrente por varios usuarios, y la coincidencia de los espacios de entrada y salida elimina la separación entre el lugar donde se actúa y aquel donde se observa el resultado @ishii2008[pp. 483-484]. Se espera, además, que la ausencia de sintaxis textual reduzca la barrera de entrada para niños que aún consolidan la lectoescritura, lo que haría a la programación tangible especialmente adecuada para la transición hacia las operaciones concretas. La integración de tecnologías de visualización que superpongan información virtual directamente sobre los objetos físicos puede potenciar estos beneficios, al ofrecer retroalimentación visual inmediata sin interrumpir la interacción directa con el espacio físico.

=== Tecnologías del Ambiente

==== Realidad aumentada espacial

#cite(<bimber2005>, form: "prose") denominan realidad aumentada espacial a la variante de la realidad aumentada que, en lugar de emplear pantallas montadas en la cabeza o dispositivos de mano, separa la tecnología de visualización del usuario y la integra al entorno, ya sea mediante elementos ópticos alineados con el espacio —espejos combinadores, pantallas transparentes u hologramas— o mediante proyectores de video. En su variante basada en proyección, el contenido virtual se superpone directamente sobre objetos y superficies físicas, sin dispositivos intermediarios como tabletas o teléfonos. #cite(<park2015>, form: "prose") ilustran esta variante en la evaluación del diseño de productos: proyectan imágenes de alta calidad sobre la maqueta física de un automóvil y corrigen el color según el proyector, la superficie y la iluminación. Esta característica hace a la realidad aumentada espacial especialmente adecuada para su integración con interfaces tangibles: el objeto físico y su representación virtual coexisten en el mismo espacio, sin que el usuario tenga que desviar la mirada hacia una pantalla externa.

==== Paradigma de flujo de datos

Un paradigma de programación es un estilo fundamental para estructurar programas y conceptualizar la solución de problemas. En el paradigma imperativo, predominante en los lenguajes convencionales, un programa es una secuencia de instrucciones que modifican el estado mediante asignaciones, y el programador controla explícitamente el orden de ejecución con estructuras como bucles y condicionales. El paradigma de flujo de datos (dataflow), en cambio, describe un programa como una red de componentes por la que fluyen los datos: cada componente transforma los datos que recibe y los envía al siguiente, y la ejecución la determina la disponibilidad de los datos, no una secuencia de instrucciones #cite(<wadge1985>). Lucid, el lenguaje de flujo de datos desarrollado por #cite(<wadge1985>, form: "prose"), representa los programas como redes de filtros que transforman flujos de datos; sus componentes pueden trabajar en paralelo, sin sincronizarse entre sí, y el programador especifica los datos y las transformaciones, en lugar de describir la actividad dinámica de la red.

==== Visión por computador

El reconocimiento de las piezas y de las interacciones sobre la superficie requiere técnicas de visión por computador. La detección de objetos consiste en localizar en una imagen los objetos de interés y clasificarlos. YOLO (_You Only Look Once_) plantea la detección como un problema de regresión: una única red neuronal predice, a partir de la imagen completa y en una sola evaluación, las cajas delimitadoras de los objetos y las probabilidades de sus clases, lo que permite detectar objetos en tiempo real #cite(<redmon2016>). Este enfoque se mantiene en versiones sucesivas de la familia de modelos, como YOLO11 #cite(<ultralytics2024>).

Para relacionar lo que capta la cámara con lo que muestra el proyector se recurre a la homografía, o transformación proyectiva entre planos: una transformación lineal en coordenadas homogéneas, representada por una matriz no singular de 3×3, que relaciona los puntos de un plano con los de otro. Cuatro correspondencias de puntos en posición general, sin tres puntos colineales, bastan para calcularla #cite(<hartley2003>).

Los sensores de profundidad, por su parte, permiten convertir superficies ordinarias en superficies táctiles. #cite(<wilson2010>, form: "prose") mostró que una cámara de profundidad puede detectar toques sobre una mesa sin instrumentarla, aunque con menor precisión que las pantallas capacitivas. DIRECT #cite(<xiao2016>) combina las imágenes de profundidad e infrarrojas de un mismo sensor comercial y mejora tanto la detección de toques como la precisión de su posición sobre superficies del tamaño de una mesa. Marcos de trabajo como MediaPipe #cite(<lugaresi2019>) facilitan, además, la construcción de flujos de percepción que combinan modelos de aprendizaje automático.

#pagebreak(weak: true)

// Capítulo III
= Capítulo III. Marco Metodológico

== Tipo de Investigación

El presente trabajo se clasifica como investigación proyectiva. #cite(<hurtado2010>, form: "prose") define este holotipo como aquel que culmina en la elaboración de una propuesta, plan, programa, procedimiento o artefacto, que esté orientado a resolver una necesidad o problema de carácter práctico en un ámbito determinado del conocimiento; siendo un enfoque frecuente en el campo de la tecnología, donde el objetivo es responder al cómo hacer las cosas mediante aplicaciones concretas. Para alcanzar ese resultado, la investigación proyectiva no parte directamente de una idea hacia su implementación, sino que recorre estadios previos, analíticos, comparativos, explicativos y predictivos; que fundamentan y justifican las decisiones de diseño.

Este enfoque resulta pertinente para el presente trabajo porque su contribución principal no es describir el uso de la programación tangible o la realidad aumentada espacial, sino diseñar y construir un ambiente que los integre de forma coherente, sustentado en el diagnóstico y el análisis realizados.

Dentro de las modalidades de Trabajo de Grado de la Escuela de Ingeniería Informática de la UCAB Guayana, el presente trabajo corresponde a la modalidad Experimental.

== Diseño de la Investigación

Según #cite(<arias2012>, form: "prose"), el diseño de la investigación es la estrategia general que adopta el investigador para responder al problema planteado, y puede ser documental, de campo o experimental. El presente trabajo tuvo un diseño documental y de campo: documental en la revisión de antecedentes y bases teóricas que sustentó el análisis y la caracterización del ambiente, y de campo en la entrevista a una docente y en la evaluación del ambiente por juicio de expertos, en las que los datos se obtuvieron directamente de los informantes.

== Población y Muestra

Al tratarse de una investigación proyectiva cuyos resultados no se generalizan estadísticamente, no se delimitó una población en sentido estricto. Los informantes se seleccionaron mediante un muestreo intencional u opinático, en el que, según #cite(<arias2012>, form: "prose"), los elementos se escogen con base en criterios o juicios preestablecidos por el investigador. La muestra estuvo conformada por tres informantes:

- Una docente que enseña programación y robótica a niños, Jackeline Duarte, de Multiplayer Pzo., centro de entretenimiento y formación tecnológica de Ciudad Guayana, seleccionada por su experiencia directa con el uso de herramientas digitales y tangibles con niños.
- Un experto en medios didácticos, docente universitario con más de diez años de experiencia en docencia, que puede desempeñarse además como docente de 1#super[er] a 3#super[er] grado.
- Un experto en interacción humano-computador, docente universitario con amplia trayectoria en usabilidad y en docencia.

Los criterios de inclusión de los expertos fueron su trayectoria reconocida en el área que evaluaron, su experiencia en docencia y su disponibilidad para presenciar la demostración del ambiente.

== Técnicas e instrumentos de recolección de datos

Según #cite(<arias2012>, form: "prose"), las técnicas de recolección de datos son los métodos establecidos para recopilar información, mientras que los instrumentos son las herramientas, dispositivos o formatos empleados para ello. En este trabajo se recurrió a tres técnicas: la revisión documental, la entrevista semiestructurada y la entrevista no estructurada.

La revisión documental permitió construir el marco teórico, el estado del arte y los criterios de diseño del ambiente. Las fuentes consultadas incluyeron artículos académicos, libros e informes de trabajos de grado.

La entrevista semiestructurada es aquella en la que, según #cite(<arias2012>, form: "prose"), aunque existe una guía de preguntas, el entrevistador puede formular otras no previstas inicialmente. Esta técnica se aplicó durante el análisis, en la entrevista a la docente, con el fin de conocer las prácticas y restricciones de la enseñanza del pensamiento computacional a niños. El instrumento empleado fue un guion de entrevista, y las respuestas de la docente se registraron como notas en un teléfono móvil.

La entrevista no estructurada es aquella en la que, según #cite(<arias2012>, form: "prose"), no se dispone de una guía de preguntas elaboradas previamente, aunque la conversación se orienta por objetivos preestablecidos que definen el tema de la entrevista. Esta técnica se aplicó durante la validación, con los expertos consultados tras la demostración del ambiente, con el fin de recoger libremente sus valoraciones sobre este. El instrumento empleado fue un registro de notas en un teléfono móvil, en el que se anotaron los aportes de los expertos durante la conversación.

== Metodología de Desarrollo Utilizada
Al analizar las características del trabajo de investigación, se consideró el enfoque a adoptar. Dado que no se previó un contacto constante con el cliente y que los requisitos aún no estaban bien definidos, se decidió optar por un enfoque basado en prototipos, con el fin de definir los requerimientos finales a través de los prototipos realizados y sus validaciones.
Según #cite(<pressman2010>, form: "prose"), el enfoque basado en prototipos está enmarcado dentro de los modelos de proceso evolutivos, que "son iterativos. Se caracterizan por la manera en la que permiten desarrollar versiones cada vez más completas del software.". Particularmente para el enfoque basado en prototipos, el proceso se divide en 4 fases, como se observa en la @prototyping-figure: comunicación, plan rápido - modelado - diseño rápido, construcción del prototipo y despliegue - entrega y retroalimentación. Se definen a continuación:

#figure(
  image("images/prototyping-paradigm.png"),
  caption: [
    Etapas del enfoque basado en prototipos. @pressman2010[p. 43]
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

== Analizar el Uso de Programación Tangible en Entornos de Realidad Aumentada Espacial, a fin de Caracterizar el Ambiente a Desarrollar

Para caracterizar el ambiente a desarrollar se contrastaron, mediante revisión documental, los principales referentes de la programación tangible y de los entornos de realidad aumentada espacial para niños: TORTIS, AlgoBlock, Scratch y ScratchJr, Magicboard y el sistema de Rojas y Youssef. Los criterios de comparación se derivaron del planteamiento del problema y de las bases teóricas: el tipo de interfaz y su dependencia de una pantalla; el soporte a la colaboración; la edad a la que se dirige y si requiere lectoescritura; el paradigma de programación y los conceptos que permite trabajar; y el tipo de retroalimentación. La @referents-matrix resume la comparación.

#[
#show figure: set block(breakable: true)
#figure(
  [
    #set text(size: 8pt)
    #set par(justify: false)
    #table(
      columns: (0.9fr, 1.3fr, 1.1fr, 0.9fr, 1.1fr, 1fr),
      align: left + top,
      inset: 4pt,
      table.header([*Referente*], [*Interfaz y pantalla*], [*Colaboración*], [*Edad y lectoescritura*], [*Paradigma y conceptos*], [*Retroalimentación*]),
      [*TORTIS* \ #cite(<morgado2006>)], [Cajas de botones y cartas en un tragafichas que controlan una tortuga robótica; un monitor mostraba los comandos de las cajas], [No documentada como objetivo de diseño], [3 a 5 años; no requiere lectura], [Imperativo; secuencia, repetición y procedimientos], [Movimiento y dibujo de la tortuga; luces que indican la carta en ejecución],
      [*AlgoBlock* \ #cite(<suzuki1993>)], [Bloques físicos conectados; el programa controla un submarino en pantalla], [Principio de diseño: acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos], [Primaria y secundaria; observado con tres niñas de 12 años], [Imperativo, inspirado en Logo; secuencia, condicionales y bucles], [Submarino en pantalla; luces en los bloques],
      [*Scratch y ScratchJr* \ #cite(<maloney2010>) #cite(<bers2018>)], [Bloques en pantalla: computadora (Scratch) o tableta, con íconos (ScratchJr)], [Comunidad en línea para compartir y remezclar proyectos; construcción frente a la pantalla], [Scratch: desde 8 años, supone lectoescritura; ScratchJr: 5 a 7 años], [Imperativo por bloques; secuencia, bucles, condicionales y eventos], [Inmediata, en pantalla],
      [*Magicboard* \ #cite(<barrios2024>)], [Mesa con proyector y sensor Kinect que detecta gestos y objetos físicos; sin monitor], [Juegos sociales basados en el aprendizaje colaborativo], [Educación preescolar], [No es un entorno de programación], [Proyectada sobre la mesa],
      [*Rojas y Youssef* \ #cite(<rojas2024>)], [Fichas táctiles encajables, con braille, reconocidas por una cámara], [No documentada en los requerimientos], [Niños con discapacidad visual], [Imperativo; ciclos, condicionales y variables], [Auditiva],
      table.cell(colspan: 6)[*Síntesis para el ambiente propuesto*],
      [*Ambiente propuesto*], [Superficie tangible con retroalimentación proyectada, sin monitor], [Acceso simultáneo sobre una superficie compartida], [6 a 9 años; piezas que pueden ser reconocidas sin texto escrito], [Flujo de datos], [Visual proyectada y auditiva],
    )
  ],
  caption: [
    Comparación de los referentes de la programación tangible y de la realidad aumentada espacial para niños, según los criterios del análisis.
  ],
) <referents-matrix>
]

En cuanto a la interfaz y la colaboración, AlgoBlock es el único referente de programación que incorpora la colaboración como principio de diseño —acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos #cite(<suzuki1993>)—, y Magicboard la sostiene mediante juegos sociales sobre una mesa interactiva #cite(<barrios2024>). Scratch favorece la colaboración a través de una comunidad en línea donde se comparten y remezclan proyectos #cite(<bers2018>), pero la construcción ocurre frente a una pantalla, de forma individual o por turnos. Las tabletop TUI superan esta limitación: su entrada multiplexada en el espacio permite que varios usuarios manipulen el programa al mismo tiempo, y la persistencia de los objetos mantiene el programa visible sin mediación del sistema @ishii2008[pp. 483-484]. De ello se desprenden la interfaz tangible de superficie y el diseño para la colaboración.

Respecto a la dependencia de pantallas, Scratch, ScratchJr y AlgoBlock muestran el resultado del programa en un monitor o una tableta, mientras que TORTIS y el sistema de Rojas y Youssef trasladan la salida al mundo físico o al canal auditivo, y Magicboard la proyecta sobre la mesa. Dado que la AAP asocia el uso excesivo de medios digitales entre los 6 y los 12 años con menor rendimiento académico y peor control de la atención #cite(<aap2026>), y que J. Duarte (comunicación personal, 28 de febrero de 2025) limita a 15 minutos los periodos continuos de pantalla en sus clases, lo que condiciona el uso de herramientas como Scratch, el ambiente debe proyectar la retroalimentación sobre la propia superficie de trabajo, como Magicboard, y reducir así el tiempo frente a una pantalla sin renunciar a la retroalimentación digital.

En cuanto a la edad, Scratch es una herramienta ampliamente usada para fomentar el pensamiento computacional, como muestran #cite(<perezmarin2020>, form: "prose") en educación primaria; sin embargo, está diseñado para niños desde los 8 años y supone conocimientos básicos de lectura y escritura, pues sus bloques contienen palabras. ScratchJr, dirigido a niños de 5 a 7 años, reduce esa exigencia #cite(<bers2018>), y TORTIS se diseñó para niños preescolares que aún no leen #cite(<morgado2006>). Los niños de 6 a 9 años se encuentran en la transición hacia la etapa de operaciones concretas #cite(<piaget1969>), por lo que las piezas del ambiente deben diseñarse de modo que puedan ser reconocidas visualmente, sin depender de texto escrito, y representar conceptos concretos y familiares.

Todos los referentes de programación comparados se basan en el paradigma imperativo: el niño construye una secuencia de comandos cuyo efecto observa al ejecutarla. La experiencia de TORTIS muestra el riesgo de este enfoque en edades tempranas: con las cajas de botones, los niños tendían a confundir el programa con el dibujo resultante, y con las cartas del tragafichas les costaba comprender que cada carta representaba un comando, pues ejecutar cada una exigía buscarla, insertarla y presionar un botón (Perlman, 1976, citado en #cite(<morgado2006>, form: "author"), #cite(<morgado2006>, form: "year")). Para el ambiente propuesto, esto indica que no basta con que cada pieza física corresponda a una instrucción: la relación entre la pieza y su efecto debe hacerse visible de forma inmediata. Por ello se optó por el paradigma de flujo de datos, en el que la ejecución la determina la disponibilidad de los datos y no una secuencia de instrucciones #cite(<wadge1985>). Esta diferencia sustenta una hipótesis de diseño de este trabajo. #cite(<duboulay1986>, form: "prose") identifica la comprensión de la "máquina nocional" —el modelo de lo que ocurre dentro de la computadora al ejecutar un programa, incluido el flujo de control— como una de las principales dificultades de quienes aprenden a programar; se presume que esta dificultad es mayor para un niño en la transición hacia las operaciones concretas #cite(<piaget1969>), y que observar cómo los datos fluyen y se transforman a través de una red de nodos dispuesta físicamente sobre la superficie le resulta menos abstracto que seguir un puntero de ejecución que recorre instrucciones en orden. La red de datos y operaciones que constituye un programa dataflow se corresponde de forma natural con la disposición espacial de bloques conectados, haciendo visible la estructura computacional de manera coherente con la experiencia concreta del niño. Adicionalmente, la ausencia de estado mutable y de efectos laterales propia del paradigma dataflow puro #cite(<wadge1985>) simplifica el modelo mental necesario para razonar sobre el programa: cada bloque produce siempre el mismo resultado con los mismos datos de entrada, sin sorpresas derivadas de órdenes de ejecución o modificaciones ocultas de variables. Lucid, el lenguaje de programación dataflow purista desarrollado por #cite(<wadge1985>, form: "prose"), sirvió como referente para definir el modelo de ejecución del lenguaje propuesto en este trabajo, particularmente en lo relativo a la evaluación dirigida por demanda y a la representación de los programas como redes de filtros funcionales sobre flujos de datos.

En cuanto a la retroalimentación, TORTIS indicaba con luces la carta en ejecución, un rasgo que comparte con AlgoBlock #cite(<morgado2006>); el sistema de Rojas y Youssef la ofrece de forma auditiva, al estar dirigido a niños con discapacidad visual @rojas2024[pp. 47-48]; y Magicboard la proyecta sobre la mesa. El ambiente propuesto debe combinar ambos canales: una retroalimentación visual proyectada, que muestre las conexiones, el estado de ejecución y los resultados, y una retroalimentación auditiva.

Por último, #cite(<barrios2024>, form: "prose") demostró con Magicboard que un entorno de realidad aumentada espacial basado en Kinect y proyector puede sostener juegos sociales colaborativos entre niños de educación preescolar y básica, lo que confirma, en el contexto venezolano, la viabilidad técnica de esta combinación y constituye el punto de partida directo del presente trabajo.

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

// La gramática EBNF completa se consigna en el Apéndice A (<appendix-a>), tomada de la especificación viva del lenguaje (GRAMMAR_SPEC.md, v4.2.0). Cualquier divergencia futura entre implementación y especificación debe resolverse actualizando primero la especificación y luego el texto del diseño, para conservar trazabilidad académica.

== Construir un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Base al Diseño Realizado

=== Prototipo 1

Partiendo de la tesis de Barrios, se buscó una aproximación más programática, asimilándose a Scratch, por lo que se partió de seguir el paradigma imperativo y la programación con bloques. Sin embargo, dado que Scratch ya es ampliamente usado y tiene varias investigaciones al respecto de su uso, tal como se planteó durante el análisis previo; el tutor sugirió seguir una aproximación distinta, basada en el paradigma de programación dataflow, pues ofrece una clara visualización de cómo fluyen y se transforman los datos del programa.

Dado este cambio, se procedió con la definición de los primeros datos y operaciones a usar, para lo que se eligieron bloques con formas geométricas simples (cuadrados, círculos y triángulos) y colores básicos (morado, amarillo, naranja, verde, rojo y azul) que, para simplificar el desarrollo, se decidió que algunos representarían operaciones en vez de un dato. Los datos que se soportaban provenían directamente de las formas (cuadrados, círculos y triángulos de distintos colores), y las operaciones eran unión, intersección, diferencia y diferencia simétrica. El diseño consistió de zonas que reconocían las formas colocadas como datos, otras que reconocían las formas como operaciones, y zonas de salida que mostraban el resultado de la ejecución. Todas estas zonas estaban colocadas de forma fija, restringiendo la creación de nuevas zonas o la asociación entre estas para el usuario final, lo que limitaba la flexibilidad del entorno pero facilitaba el desarrollo del prototipo.

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

Las librerías disponibles para integrar el Kinect v2 con Python son limitadas. Se probaron aproximaciones con PyKinect2 y libfreenect2; sin embargo, el primero, aunque declara compatibilidad con Python 3.4 o superior, fallaba con las versiones recientes de Python empleadas en el proyecto, y el segundo no detectaba el Kinect v2; OpenNI2, que se usó para el Kinect v1, no es compatible con el Kinect v2 por defecto, por lo que se compiló manualmente una versión del controlador de OpenNI2 para el Kinect v2 que se apoya en el controlador del Kinect for Windows SDK 2.0, con lo cual se logró usar OpenNI2 para la integración del Kinect v2 con Python. Siguiendo con la calibración y detección de toques, se hicieron modificaciones exhaustivas al código legado para adaptarlo al nuevo sensor, lo que llevó a la implementación de un nuevo algoritmo de detección de marcadores (2 cuadrados blancos en las esquinas superior izquierda e inferior derecha de la proyección), además de la afinación de múltiples números mágicos (literales escritos en el código sin documentar su significado). Este prototipo puede verse en la @second-prototype-figure.

#figure(
  image("images/second-prototype.jpeg"),
  caption: [
    Segundo prototipo del ambiente, con el cambio al sensor Kinect v2 y la adaptación del código legado para la calibración y detección de toques.
  ],
) <second-prototype-figure>

Este prototipo, si bien permitió validar la viabilidad del cambio al Kinect v2, también mostró que el código legado por Barrios era difícil de mantener. También se vio que la transformación Window-to-Viewport que se usa en los algoritmos es muy sensible a la configuración física del entorno (paralelismo entre la proyección sobre la superficie y el ángulo de la cámara), resultando en que la detección de toques no fuese tan precisa como se esperaba.

=== Prototipo 3

Tras trabajar tanto en una única parte del sistema (integración con el hardware y detección de toques), se decidió que el siguiente paso sería trabajar en la visión por computador para la detección de las piezas que conformarían el entorno.

Para esto, se decidió usar un modelo de detección de objetos basado en aprendizaje profundo, específicamente YOLO11n, la versión más ligera de YOLO11 #cite(<ultralytics2024>), modelo de la familia de detectores de una sola etapa iniciada por #cite(<redmon2016>, form: "prose"), diseñado para ser eficiente en términos de velocidad y recursos computacionales, lo que lo hace adecuado para aplicaciones en tiempo real como la visión por computador con el Kinect v2. Se planeó entrenar este modelo con un conjunto de datos personalizado que incluía imágenes de una versión previa de las piezas que se usarían en el entorno, con el objetivo de lograr una detección precisa y evaluar la viabilidad de detectar las piezas mediante modelos de detección de objetos.

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

Durante el desarrollo de este prototipo, el enfoque estuvo en la implementación del apartado de ejecución del lenguaje de programación dataflow, para lo cual se definieron 3 representaciones de los programas formados por los bloques visuales: una de intercambio, basada en JSON; una textual, para entrada y depuración; y un formato en memoria, para uso interno por el entorno de ejecución; y se implementó un intérprete para ejecutar estos programas (denominado inicialmente compilador y _runtime_, terminología que fue revisada en iteraciones posteriores al consolidarse la evaluación directa de los programas). Se decidió usar TypeScript como lenguaje de programación, debido a su flexibilidad, facilidad para el desarrollo rápido, y su capacidad para manejar estructuras de datos complejas mediante su tipado; Bun como motor de ejecución, pues permite la ejecución directa de programas escritos en TypeScript sin un paso previo de transpilación, y provee ventajas de rendimiento contra sus competidores Node y Deno; y la librería Chevrotain, que provee un kit de herramientas para la construcción de _parsers_; facilitando la implementación del entorno. Además, se implementó un servidor HTTP y uno de WebSockets, para lo cual se utilizó la librería Elysia, que permiten la comunicación con la interfaz gráfica y el apartado de visión por computador. // Este prototipo puede verse en la @fourth-prototype-figure.

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

Se implementaron características como la visualización del programa en tiempo real, la posibilidad de arrastrar y soltar bloques para crear programas, y una sección de resultados donde se mostraban los resultados de la ejecución del programa. Además, se buscó crear una experiencia de usuario atractiva mediante el uso de colores y una disposición clara de los elementos en la interfaz. Este prototipo fue desarrollado en TypeScript, usando la librería React para la construcción de la interfaz gráfica, y la librería React Flow para la representación visual de los datos, operaciones y flujos de datos. // Este prototipo puede verse en la @fifth-prototype-figure.

//TODO: colocar imagen del prototipo
// #figure(
//   image("images/fifth-prototype.jpeg"),
//   caption: [
//     Quinto prototipo del ambiente, con la implementación de la interfaz gráfica del entorno de desarrollo integrado (IDE) para el lenguaje de programación.
//   ],
// ) <fifth-prototype-figure>

Al finalizar el desarrollo de la interfaz gráfica del modo sandbox, se vio que facilitaba la creación de programas mediante la manipulación de bloques digitales, y se planteó continuar con la integración de la detección de bloques físicos y el reconocimiento de estos por parte del entorno de ejecución del lenguaje de programación dataflow.

=== Prototipo 6

Continuando con el prototipo 5, se decidió integrarle la detección de piezas físicas mediante el Kinect v1, por dificultades temporales con el Kinect v2/* especificar que estas dificultades estaban relacionadas con problemas de compatibilidad de estándar USB y rendimiento */; y el uso de un nuevo modelo de detección de objetos basado en aprendizaje profundo, pues se cambió el diseño de las piezas físicas a usar, requiriendo de un reentrenamiento del modelo. Además, se planteó comenzar la integración con el entorno de ejecución del lenguaje de programación dataflow, optando por la integración mediante WebSockets para la comunicación.

Se llevó a cabo un rediseño de las piezas físicas a usar, buscando cubrir los datos y operaciones que se definieron para el lenguaje, un diseño sencillo de entender y usar para los niños, pero no tan complejo en aras de facilitar la detección por parte del modelo, resultando en un diseño tipo carta. Estas nuevas piezas pueden verse en la @sixth-prototype-pieces-design-figure. Además, también se hicieron modificaciones en la interfaz gráfica del modo sandbox, entre ellas usar colores oscuros, para facilitar la visualización de la proyección del entorno virtual sobre la superficie física.

Al entrenar el nuevo modelo de detección de objetos, se comenzó con el modelo YOLO11n, con un dataset en el que las _bounding boxes_ comprendían toda la carta, incluyendo las etiquetas ("Operador", "Resta", "Tortuga", etc.), áreas blancas alrededor de la pieza, e imagen de la pieza; este modelo tenía dificultades para detectar las piezas, principalmente por la confusión entre clases, por lo que se decidió ajustar las _bounding boxes_ para que solo comprendieran el área de la imagen de la pieza, sin incluir las etiquetas ni áreas blancas, lo que llevó a una pequeña mejora en la detección, pero sin llegar a los resultados esperados. Finalmente, se cambió al modelo YOLO11s, una versión ligeramente más pesada y potente que YOLO11n #cite(<ultralytics2024>), con el que se observó una mejora apreciable en la detección de las piezas físicas, si bien esta comparación fue cualitativa y no se respaldó con métricas formales. Posteriormente se exploró también YOLO26n #cite(<ultralytics2026>), una variante ligera más reciente de la misma familia. Además, se implementó una integración básica con el entorno de ejecución del lenguaje de programación dataflow mediante WebSockets, enviando las piezas reconocidas al entorno, pero sin las conexiones entre estas. Este prototipo puede verse en la @sixth-prototype-figure.

#figure(
  image("images/sixth-prototype.jpeg"),
  caption: [
    Sexto prototipo del ambiente, con la integración de la detección de piezas físicas mediante un nuevo modelo de detección de objetos.
  ],
) <sixth-prototype-figure>

Con este prototipo terminado, se vio que la integración de la detección de piezas físicas mediante el nuevo modelo de detección de objetos era viable y, según lo observado, mejoraba la detección, aunque surgieron preocupaciones respecto al proceso de entrenamiento del modelo, tomando en cuenta que no se usó el lote completo de cartas que soporta el lenguaje, pero sí más que el número de piezas que se usaron en prototipos previos. La integración con el entorno de ejecución del lenguaje de programación dataflow mediante WebSockets también se mostró viable, pero la falta de un mecanismo para representar las conexiones entre las piezas reconocidas a nivel tangible, de modo que se pudieran enviar al entorno de ejecución; limitaba las pruebas que se podían hacer con esta integración.

=== Prototipo 7

El séptimo prototipo se concibió como la primera integración completa de los tres componentes del ambiente —el subsistema de visión por computador, el lenguaje de programación y la interfaz gráfica— en una experiencia unificada de extremo a extremo, atendiendo la principal limitación identificada en el sexto prototipo: la ausencia de un mecanismo para representar las conexiones entre las piezas físicas reconocidas. El desarrollo abarcó mejoras simultáneas en la calibración del entorno, en la arquitectura del lenguaje y en la integración entre la detección de piezas y su evaluación.

En el subsistema de visión, se sustituyó la transformación lineal de dos puntos heredada de los prototipos previos por una homografía de cuatro puntos, calculada mediante la función `cv2.getPerspectiveTransform` #cite(<hartley2003>). A diferencia de la transformación anterior, que solo corregía traslación y escala, la homografía describe una proyección de ocho grados de libertad capaz de corregir la distorsión de perspectiva entre la cámara y la superficie proyectada, lo que mejora la precisión con que se ubican los toques sobre la mesa.

En el lenguaje, se consolidó la arquitectura del entorno de ejecución. Lo que en el cuarto prototipo se había implementado como un compilador y un _runtime_ separados, comunicados con los demás componentes mediante servidores HTTP y de WebSockets construidos con la librería Elysia, se unificó en un único intérprete distribuido como librería. De este modo, el intérprete dejó de ejecutarse como un proceso aparte y pasó a integrarse directamente en la interfaz, que lo invoca en el propio cliente para evaluar los programas. Esta decisión eliminó la latencia introducida por la comunicación entre procesos —una de las preocupaciones surgidas en el cuarto prototipo— y simplificó el despliegue del sistema. En coherencia con esta evolución, la terminología se ajustó definitivamente al término intérprete, en sustitución del par compilador-_runtime_ empleado inicialmente.

Paralelamente, se rediseñó y simplificó el lenguaje para alinearlo con el enfoque concreto-pictórico-abstracto, derivado de los modos de representación enactivo, icónico y simbólico de #cite(<bruner1966>, form: "prose"). El modelo de datos, antes compuesto por múltiples tipos heterogéneos, se unificó en un objeto único caracterizado por una de tres categorías —concreto, pictórico o abstracto—, separando la semántica del lenguaje de su modo de presentación visual. Esta simplificación redujo la complejidad del intérprete y permitió reorganizar el conjunto de piezas físicas en torno a las tres categorías.
//TODO: la "disminución en la cantidad de piezas físicas" no pudo cuantificarse contra el historial (no se comparó el catálogo anterior con el nuevo); confirmar a mano la reducción neta o reformular como "reorganización" del repertorio.

La integración de la detección de piezas con el intérprete se articuló a través de la interfaz gráfica. El subsistema de visión, mediante un servidor de relevo (_relay_) construido con FastAPI que reemplazó los antiguos servidores Elysia; transmite a la interfaz los lotes de cartas detectadas, y la interfaz las representa como nodos sobre el lienzo de flujo de datos, para luego traducir el grafo resultante en un programa que entrega al intérprete para su evaluación. El problema de representar las conexiones entre piezas, pendiente desde el sexto prototipo, se resolvió mediante un sistema de puertos tipados y reglas estructurales que determinan qué piezas pueden enlazarse entre sí, complementado con la visualización del dato que circula por cada conexión mediante elementos animados denominados _walkers_. La elección de FastAPI para el servidor de relevo respondió a un criterio de cercanía tecnológica y de mínimo esfuerzo de implementación: dado que el subsistema de visión está escrito en Python, mantener el relevo sobre Bun y Elysia habría obligado a duplicar en ese entorno las definiciones de las interfaces de los mensajes de detección de piezas y de toques, ya existentes en Python; reimplementar el relevo con FastAPI permitió reutilizar directamente esas definiciones y evitar la duplicación de esas interfaces entre dos lenguajes.
//TODO: esta justificación procede de los autores; si se requiere trazabilidad formal, conviene redactar el ADR correspondiente, pues los ADR-006 y ADR-007 aún describen el stack Bun/Elysia ya deprecado.

//TODO: el esqueleto original anotaba "<primer vídeo>, muy mal rendimiento" como evidencia de esta etapa. No hay vídeos ni mediciones de rendimiento versionados en el repositorio; añadir a mano la referencia al material audiovisual y, de sostenerse el juicio de rendimiento, respaldarlo con datos.

//TODO: agregar figura del séptimo prototipo (integración inicial). Pendiente de imagen, siguiendo el patrón de @sixth-prototype-figure.

Con este prototipo se obtuvo, por primera vez, una experiencia integrada de extremo a extremo en la que las piezas físicas, sus conexiones y la salida proyectada conformaban un programa ejecutable de manera incremental, de modo que las mejoras posteriores se pudieron implementar sobre esta base, en forma de evoluciones, motivadas en un principio por las limitaciones observadas en la fluidez de la detección.

==== Evolución 1

La primera evolución se centró en la detección de toques. Hasta entonces se había explorado la detección basada únicamente en la imagen de profundidad, en la línea de #cite(<wilson2010>, form: "prose"). Se incorporó entonces un detector híbrido que combina el algoritmo DIRECT #cite(<xiao2016>), que decide si existe un toque a partir de la imagen de profundidad y de la imagen infrarroja del sensor, mediante relleno por inundación, zonas e histéresis, con el modelo de detección de manos Hand Landmarker de MediaPipe #cite(<lugaresi2019>), que aporta la posición precisa de la punta del dedo índice cuando DIRECT detecta un contacto. Esta combinación separa la decisión de si existe un toque de la estimación de dónde ocurre, aprovechando la robustez del sensor de profundidad y la precisión de la estimación visual de la mano. En esta evolución también se creó una bifurcación (_fork_) de PyKinect2, modificada para hacerla compatible con Python 3.8 o superior, y se probó como alternativa a OpenNI2 para acceder al sensor mediante el Kinect for Windows SDK 2.0. En las pruebas se percibió que el ambiente funcionaba de forma más fluida, por lo que se mantuvo PyKinect2 como forma de acceso al sensor.
//TODO: el esqueleto anotaba que la detección se hizo "más eficiente" y mencionaba un "<segundo vídeo>" asociado a la primera valoración positiva del tutor. La mejora de eficiencia no está cuantificada en el repositorio (no hay _benchmarks_ ni mediciones de latencia versionadas) y no existe rastro del material audiovisual ni del _feedback_ del tutor; añadir y respaldar a mano.

==== Evolución 2

La segunda evolución comprendió dos mejoras. En la calibración, se generalizó la homografía de cuatro puntos a una de nueve, dispuestos en una rejilla de tres por tres y resuelta por mínimos cuadrados mediante `cv2.findHomography` #cite(<hartley2003>), con el fin de mejorar la precisión del mapeo entre la cámara y la superficie proyectada. Las coordenadas de la cámara de color se llevan al espacio de la imagen de profundidad mediante el mapeo de coordenadas del Kinect for Windows SDK 2.0, que constituye una fuente adicional de error de la calibración. En cuanto a las piezas físicas, se amplió el repertorio del mazo más allá de las cartas, incorporando tapas, paletas y cubos como piezas tangibles adicionales reconocibles por el subsistema de visión, enriqueciendo la experiencia del usuario al interactuar con el enfoque concreto-pictórico-abstracto de forma más directa, con representaciones más fieles a lo propuesto por #cite(<bruner1966>, form: "prose").
//TODO: el cambio a nueve puntos está respaldado por la configuración del código (rejilla 3×3 por defecto en `config.py`), pero la mejora de precisión no está medida cuantitativamente. Asimismo, la incorporación de tapas, paletas y cubos consta en la interfaz y el catálogo de piezas, pero el reentrenamiento del modelo YOLO para esas piezas no está respaldado por el historial (el conjunto de clases versionado en `data.yaml` no las incluye); confirmar y documentar el reentrenamiento a mano.

==== Evolución 3

La tercera evolución consistió en una extensión del lenguaje, reflejada en la especificación formal presentada en el #link(<appendix-a>)[Apéndice A]. Se incorporaron cartas de grupos, que agrupan varios objetos en una sola colección; cartas de manipulación de grupos —primero (`first`), último (`last`) y contar (`count`)—; y la operación de comparación (`compare`). Asimismo, las operaciones de filtrado y ordenamiento, ya presentes en versiones anteriores del lenguaje, se reforzaron con un sistema de criterios que permite filtrar y ordenar las colecciones según propiedades como el color, el tamaño o la forma de los objetos.
//TODO: precisar, de ser necesario, que las operaciones de filtrado (`filter`) y ordenamiento (`order_asc`/`order_desc`) ya existían en el lenguaje desde prototipos anteriores; lo introducido en esta evolución es el sistema de criterios (gramática v4.0.0) y las operaciones `first`/`last`/`count`/`compare` (gramática v4.1.0).

// ==== Evolución 4 -> Dado, cambios del intérprete a array-first tras bambalinas sin afectar experiencia del usuario, zonas de persistencia de detección.

== Validar el Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años Construido

La validación del ambiente se abordó en dos frentes complementarios: una verificación interna, que contrastó los requerimientos del sistema con las funcionalidades efectivamente construidas mediante una matriz de trazabilidad, y una evaluación por juicio de expertos en interacción humano-computador (IHC) y en medios didácticos, que valoró el ambiente como sistema interactivo y como recurso educativo.

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

Como se observa en la @requirements-to-functionalities-matrix, los cinco requerimientos funcionales se encuentran satisfechos: la construcción y captura de los programas tangibles, el reconocimiento de los elementos y de las conexiones, la interpretación con evaluación incremental, la ejecución con salida proyectada y la retroalimentación que guía la construcción. De los requerimientos no funcionales, el manejo de errores de disposición está satisfecho, mientras que la presentación auditiva y los elementos persuasivos están parcialmente cubiertos y la usabilidad efectiva queda pendiente. Estos últimos, de naturaleza pedagógica y de experiencia, motivaron la evaluación por juicio de expertos que se describe a continuación.

=== Evaluación por juicio de expertos

La calidad del ambiente como sistema interactivo y como recurso educativo se valoró mediante el juicio de dos expertos con perfiles complementarios: un experto en interacción humano-computador (IHC), que valoró la interfaz, la interacción tangible y la experiencia de uso, y un experto en medios didácticos con experiencia docente, que valoró la pertinencia del ambiente como recurso para el proceso de enseñanza-aprendizaje. A cada experto se le presentó una demostración del ambiente en funcionamiento, tras la cual se recogieron sus observaciones mediante una entrevista no estructurada #cite(<arias2012>). En la demostración se expuso el modo de uso para el que fue concebido el ambiente: una actividad colaborativa entre el docente y los niños, en la que el docente actúa como conductor o guía.

Dado que el experto en medios didácticos puede desempeñarse como docente de 1#super[er] a 3#super[er] grado y, por tanto, como usuario final del ambiente, su evaluación constituyó además una prueba de usabilidad desde la perspectiva del docente. Este experto valoró positivamente la paleta de colores del ambiente y la sencillez con que se comprende su funcionamiento. Consideró, además, que el ambiente es aplicable en el salón de clases como recurso utilizado por el docente para apoyar el proceso de enseñanza-aprendizaje, con especial utilidad en la evaluación de los aprendizajes. Como aspecto por mejorar, señaló la retroalimentación que el ambiente ofrece al usuario; en respuesta, los autores plantearon reforzar la retroalimentación sonora y la presentación de los errores. Sugirió, por último, incorporar música relajante durante el uso del ambiente.

El experto en IHC expresó su preocupación por la cantidad de contenido programático que abarca el ambiente, en caso de que fuera utilizado directamente por los niños. Se le aclaró que el ambiente está concebido para ser usado por docentes en conjunto con niños, con el docente como conductor o guía de la actividad. Aun con esta precisión, el experto sostuvo que dicho rol exige que el docente conozca a fondo el ambiente, por lo que recomendó diseñar guías de actividades modelo que los docentes puedan tomar como inspiración para elaborar sus propias actividades, así como dejar explícitas las restricciones del sistema. Asimismo, observó fallos en la detección de cartas durante la demostración y advirtió que podrían resultar frustrantes tanto para los docentes como para los niños, por lo que recomendó mejorar la detección. Finalmente, recomendó que el informe haga hincapié en que el ambiente será usado por docentes y niños, con el docente como conductor o guía. La @expert-judgment-table resume las observaciones de ambos expertos y la respuesta de los autores a cada una.

#figure(
  [
    #set text(size: 9pt)
    #table(
      columns: (auto, 1.2fr, 1fr),
      align: (left + horizon, left + horizon, left + horizon),
      inset: 4pt,
      table.header([*Experto*], [*Observación*], [*Respuesta de los autores*]),
      [Medios didácticos], [Valoró positivamente la paleta de colores y la sencillez para comprender el ambiente], [Fortaleza; se conserva],
      [Medios didácticos], [Consideró el ambiente aplicable en el aula como recurso del docente para el proceso de enseñanza-aprendizaje, con énfasis en la evaluación], [Fortaleza; coincide con el uso mediado por el docente],
      [Medios didácticos], [Recomendó mejorar la retroalimentación al usuario], [Reforzar la retroalimentación sonora y la presentación de los errores],
      [Medios didácticos], [Sugirió incorporar música relajante], [Sugerencia registrada para versiones futuras],
      [IHC], [Expresó preocupación por la cantidad de contenido programático si el ambiente fuera usado directamente por niños], [Se aclaró que el ambiente se concibe para uso colaborativo, con el docente como conductor o guía],
      [IHC], [Advirtió que el docente, como guía, debe conocer a fondo el ambiente], [Diseñar guías de actividades modelo y explicitar las restricciones del sistema],
      [IHC], [Observó fallos en la detección de cartas durante la demostración], [Mejorar la detección de las cartas],
    )
  ],
  caption: [
    Observaciones de los expertos en medios didácticos y en IHC y respuesta de los autores a cada una.
  ],
) <expert-judgment-table>
//TODO: indicar en la columna de respuesta cuáles de estas mejoras se implementaron antes de la entrega (por ejemplo, la retroalimentación sonora) y cuáles quedan como recomendación.

Las observaciones de ambos expertos coinciden en que el ambiente alcanza su propósito cuando se usa bajo la conducción del docente: el experto en medios didácticos lo sitúa como un recurso del docente para el proceso de enseñanza-aprendizaje, y el experto en IHC condiciona su uso con niños a que el docente conozca a fondo el ambiente. Las áreas de mejora señaladas son consistentes con la matriz de trazabilidad: la retroalimentación corresponde al requerimiento RNF-04, cubierto de forma parcial, y los fallos de detección afectan al reconocimiento de las piezas del requerimiento RF-02, que, aunque satisfecho en términos funcionales, requiere mayor fiabilidad para no interrumpir la actividad. En cuanto a la usabilidad (RNF-01), la evaluación del experto en medios didácticos, como potencial docente de 1#super[er] a 3#super[er] grado, aporta evidencia favorable desde la perspectiva del docente, con lo que el requerimiento pasa a estar cubierto de forma parcial; resta comprobar la usabilidad y la comprensión del ambiente por parte de los niños del rango etario objetivo. La @requirements-after-experts-matrix recoge este cambio de estado, único derivado de la evaluación por juicio de expertos respecto de la @requirements-to-functionalities-matrix.

#figure(
  [
    #set text(size: 9pt)
    #table(
      columns: (auto, 1fr, auto, 1.3fr, auto),
      align: (center + horizon, left + horizon, center + horizon, left + horizon, center + horizon),
      inset: 4pt,
      table.header([*Cód. Req.*], [*Requerimiento*], [*Estado anterior*], [*Evidencia de la evaluación por expertos*], [*Estado actual*]),
      [RNF-01], [El sistema debe ser usable por niños de 6 a 9 años y profesores de primaria de 1#super[er] a 3#super[er] grado], [Pendiente], [Valoración favorable de la claridad visual y la sencillez del ambiente por parte del experto en medios didácticos, como potencial docente de 1#super[er] a 3#super[er] grado; resta la comprobación con niños], [Parcial],
    )
  ],
  caption: [
    Requerimientos cuyo estado de cobertura cambió tras la evaluación por juicio de expertos.
  ],
) <requirements-after-experts-matrix>

// == Realizar la Documentación Formal del Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años Construido

//* Mencionar apéndices de manual de sistema, manual de usuario y guía de ejercicios modelo

#pagebreak(weak: true)

// Capítulo V
= Capítulo V. Conclusiones y Recomendaciones
//TOOD: Revisar, ya así por encima puedo ver que las recomendaciones están en un mal formato
== Analizar el Uso de Programación Tangible en Entornos de Realidad Aumentada Espacial, a fin de Caracterizar el Ambiente a Desarrollar

El análisis del uso de la programación tangible en entornos de realidad aumentada espacial permitió caracterizar el ambiente a desarrollar y muestra que esta combinación constituye una vía pedagógica y técnicamente viable para fomentar el pensamiento computacional en niños entre 6 y 9 años con un uso moderado de pantallas. La comparación de los principales referentes —TORTIS, AlgoBlock, Scratch y ScratchJr, Magicboard y el sistema de Rojas y Youssef— según su interfaz, su soporte a la colaboración, la edad a la que se dirigen, su paradigma de programación y su retroalimentación revela que ninguno de los entornos de programación comparados reúne a la vez la manipulación tangible, la colaboración simultánea, la retroalimentación sin monitor y piezas que pueden ser reconocidas sin depender de texto escrito, como conviene a este rango de edad, y que el único referente que combina una mesa con proyección y la colaboración, Magicboard, no es un entorno de programación. Esa brecha define el espacio que ocupa el ambiente propuesto.

De esa comparación se derivan las características del ambiente: una interfaz tangible de superficie que permite el acceso simultáneo de varios niños y mantiene el programa visible, en línea con los principios de colaboración de #cite(<suzuki1993>, form: "prose") y con las propiedades de las interfaces tangibles descritas por #cite(<ishii2008>, form: "prose"); una retroalimentación visual proyectada sobre la misma superficie y complementada con retroalimentación auditiva, que reduce el tiempo frente a una pantalla en coherencia con la política pediátrica más reciente #cite(<aap2026>); y piezas que pueden ser reconocidas sin depender de texto escrito, acordes con la transición hacia la etapa de operaciones concretas #cite(<piaget1969>). El vínculo entre la teoría y estas decisiones se apoya en la representación enactiva de #cite(<bruner1966>, form: "prose") y en el principio de activación de #cite(<merrill2002>, form: "prose"): la manipulación de objetos físicos actúa como puente hacia los conceptos abstractos de la programación.

La comparación muestra, además, que todos los referentes de programación se basan en el paradigma imperativo y que, desde TORTIS, relacionar cada pieza con su efecto ha sido una dificultad para los niños pequeños #cite(<morgado2006>). Por ello se eligió el paradigma de flujo de datos, en la línea de Lucid #cite(<wadge1985>): su modelo de ejecución se corresponde con la disposición espacial de las piezas y, como hipótesis de diseño, se presume que evita la abstracción de un hilo de ejecución secuencial, difícil de asir en esta etapa cognitiva. Finalmente, el antecedente Magicboard #cite(<barrios2024>) confirma, en el contexto venezolano, la viabilidad de la realidad aumentada espacial con sensor de profundidad y proyector como base para el ambiente.

== Diseñar un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Función del Análisis Realizado

El diseño tradujo la caracterización resultante del análisis en una arquitectura concreta y en la especificación del lenguaje ERAE. Se concluye que la progresión concreto-pictórico-abstracto adoptada para el mazo de cartas operacionaliza los modos de representación enactivo, icónico y simbólico de #cite(<bruner1966>, form: "prose") y resulta coherente con la transición hacia la etapa de operaciones concretas; es decir, el marco teórico no permaneció como fundamento abstracto, sino que se materializó en una decisión de diseño verificable —la organización del repertorio físico de piezas— que solo pudo concretarse al elaborar el diseño.

En el plano del lenguaje, el diseño de flujo de datos, con declaraciones de fuente, transformación y salida, evaluación bajo demanda e incremental, y verificación estática de tipos y aridad, permite que los errores sintácticos queden eliminados por construcción y que la única fuente de error sea de naturaleza semántica, comunicada mediante mensajes orientados a los niños. Esta característica responde directamente a los requerimientos derivados del análisis: guiar la construcción del programa y prevenir los errores antes de la ejecución. La separación entre un núcleo de interpretación sin estado y adaptadores delgados, junto con una representación textual interna cuya gramática formal tolera programas incompletos (#link(<appendix-a>)[Apéndice A]), hizo realizable el diseño y habilitó la retroalimentación inmediata durante la construcción en vivo. El diseño mantuvo, además, su trazabilidad con el análisis al alinear el repertorio de datos y operaciones con los énfasis del currículo de educación primaria #cite(<mppe2023>).

== Construir un Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años, en Base al Diseño Realizado

La construcción del ambiente, llevada a cabo mediante un enfoque evolutivo basado en prototipos #cite(<pressman2010>), produjo un sistema integrado que articula el subsistema de visión por computador, el lenguaje ERAE con su intérprete y la interfaz de usuario. Se concluye que la metodología por prototipos fue determinante para un proyecto de naturaleza experimental con requerimientos inicialmente poco definidos: el resultado de cada prototipo definió el requerimiento del siguiente —la resolución insuficiente del Kinect v1 motivó el cambio al Kinect v2; la fragilidad de la detección por contornos condujo a la detección por aprendizaje profundo; la latencia entre procesos llevó a consolidar el intérprete como librería embebida; y la imposibilidad de representar las conexiones entre las piezas, evidenciada en el sexto prototipo, impulsó el sistema de puertos tipados, reglas estructurales y _walkers_ del séptimo—. Esta cadena de decisiones, que solo pudo establecerse al construir y evaluar sucesivamente el sistema, confirma la pertinencia del enfoque adoptado.

Entre los logros técnicos se cuentan el reconocimiento de las piezas mediante modelos de detección de objetos, más robusto que la detección por contornos de los primeros prototipos, un intérprete del lenguaje ERAE que funciona como librería embebida con evaluación incremental y verificación de tipos y aridad, la calibración mediante homografía, la detección de toques con un detector híbrido y la integración entre la visión y el intérprete a través de un servidor de relevo y de la interfaz, que traduce el grafo visual de piezas y conexiones en un programa ejecutable. Con el séptimo prototipo se alcanzó, por primera vez, una experiencia integrada de extremo a extremo en la que las piezas físicas, sus conexiones y la salida proyectada constituyen un programa evaluable de manera incremental, con lo que el objetivo de construcción se considera cumplido en tanto el artefacto existe y opera.

No obstante, se concluye también que varias de las mejoras introducidas durante la construcción —en particular las relativas a la precisión de la calibración y al rendimiento de la detección— no fueron cuantificadas formalmente, y durante la validación se observaron fallos en la detección de cartas. Esta distinción preserva la coherencia entre lo efectivamente construido y aquello que solo podrá afirmarse tras una medición formal del desempeño del ambiente.

== Validar el Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años Construido

La validación permite concluir que el ambiente es viable como recurso educativo usado por docentes y niños en conjunto, con el docente como conductor o guía de la actividad, y no como una herramienta que los niños utilicen de forma autónoma. La matriz de trazabilidad muestra que el ambiente satisface sus cinco requerimientos funcionales y el manejo de errores de disposición, mientras que la retroalimentación auditiva y los elementos persuasivos se cubren de forma parcial. El juicio de expertos confirma esta lectura desde fuera del equipo de desarrollo: el experto en medios didácticos valora la claridad visual y la sencillez del ambiente y lo considera aplicable en el aula como recurso del docente para el proceso de enseñanza-aprendizaje, en especial para la evaluación; el experto en IHC, por su parte, advierte que la cantidad de contenido programático exige que el docente conozca a fondo el ambiente para poder guiar a los niños.

De la validación se desprenden tres condiciones para la adopción del ambiente en el aula: una retroalimentación más completa, en particular la sonora y la presentación de los errores; una detección de cartas lo bastante fiable como para no frustrar a docentes ni a niños; y material de apoyo para el docente, en forma de actividades modelo y de una descripción explícita de las restricciones del sistema. La usabilidad del ambiente cuenta con una valoración favorable desde la perspectiva del docente, aportada por el experto en medios didácticos como potencial profesor de 1#super[er] a 3#super[er] grado; en cambio, al no haberse realizado pruebas con niños, la usabilidad y la comprensión del ambiente por parte de los niños de 6 a 9 años, así como su efecto sobre el desarrollo del pensamiento computacional, quedan por comprobar empíricamente.

// Realizar la documentación formal del ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.
//TODO: pendientes la conclusión del objetivo general y la del objetivo 5 (documentar); deben anteceder a la sección de Recomendaciones cuando se redacten.

== Recomendaciones

A partir de la experiencia de construcción se recomienda documentar cuantitativamente el desempeño del sistema, estableciendo mediciones reproducibles de latencia y velocidad de procesamiento, error de calibración y precisión de la detección de piezas mediante métricas como mAP50, mAP50-95 y la matriz de confusión, con las que puedan compararse los modelos evaluados (YOLO11n, YOLO11s y YOLO26n). Estas mediciones convertirían en evidencia verificable varias de las mejoras hoy descritas de forma cualitativa.

En cuanto a la detección de piezas, se recomienda reentrenar el modelo con el lote completo de piezas previsto por el lenguaje —incluidas las incorporadas en las últimas evoluciones, como tapas, paletas, cubos y dado— y versionar tanto el conjunto de datos como su configuración, a fin de garantizar la reproducibilidad del entrenamiento. Esta recomendación se ve reforzada por los fallos de detección observados durante el juicio de expertos, que, según el experto en IHC, podrían resultar frustrantes para docentes y niños durante una actividad.

Para preservar la trazabilidad arquitectónica, se recomienda mantener sincronizada la especificación viva del lenguaje con el informe y documentar, mediante una decisión de arquitectura formal, el reemplazo del servidor anterior por el servidor de relevo actual, dado que las decisiones de arquitectura existentes aún describen componentes ya superados. Asimismo, conviene documentar los requisitos de hardware y las dificultades de compatibilidad observadas con el sensor de profundidad, por su impacto en la estabilidad del sistema.

Dado que el ambiente está concebido para ser usado por docentes y niños en conjunto, con el docente como conductor o guía, se recomienda que su desarrollo y su adopción se orienten a apoyar ese rol. En concreto, se recomienda elaborar una guía de actividades modelo, alineada con el currículo y con los principios de la programación tangible colaborativa #cite(<suzuki1993>), que los docentes puedan tomar como inspiración para diseñar sus propias actividades, e incluir en el manual de usuario una descripción explícita de las restricciones del sistema, de modo que el docente conozca de antemano el alcance del ambiente antes de conducir una actividad con los niños.

En cuanto a la retroalimentación, se recomienda completar las señales sonoras de reconocimiento, advertencia y error y mejorar la presentación de los errores, tal como sugirió el experto en medios didácticos, y valorar la incorporación de música relajante durante el uso del ambiente.

En cuanto al alcance del lenguaje, el ambiente se centra en clasificar, filtrar, ordenar, contar, comparar y operar con las cantidades de colecciones de objetos, y no aborda estructuras de control como la repetición y la decisión condicional, propias del paradigma imperativo y ajenas al modelo de flujo de datos adoptado. Se recomienda estudiar su incorporación, ya sea mediante construcciones equivalentes dentro del paradigma de flujo de datos o mediante un modo complementario, a fin de ampliar los conceptos de pensamiento computacional que el ambiente permite ejercitar.

Finalmente, dado que la usabilidad del ambiente se valoró únicamente desde la perspectiva del docente, se recomienda realizar pruebas de usabilidad y de comprensión con niños de 6 a 9 años, en actividades conducidas por el docente. Las pruebas de usabilidad permitirían observar si los niños logran usar el ambiente —éxito en las tareas, errores, necesidad de ayuda y satisfacción—, y las de comprensión, si entienden lo que construyen y los conceptos de pensamiento computacional involucrados, de modo que los beneficios pedagógicos que la teoría anticipa —el desarrollo del pensamiento computacional y el aprendizaje colaborativo— puedan confirmarse empíricamente.

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
