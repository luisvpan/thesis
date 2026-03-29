// Portada
// TODO: Cambiar placeholders.
// TODO: Cambiar estilado, añadir negritas, centrar, etc. Revisar Anexo A1 de la Guía Informe TG.
Universidad Católica Andrés Bello
Facultad de Ingeniería
Escuela de Ingeniería Informática

Ambiente de Programación Tangible con Realidad Aumentada Espacial Orientado a Niños entre 6 y 9 años

Trabajo de Grado
presentado ante la
UNIVERSIDAD CATÓLICA ANDRÉS BELLO
como parte de los requisitos para optar al título de
Ingeniero en Informática
// TODO: formato tabla, izquierda realizado por, derecha nombres
Realizado por | Arzolay Rodríguez, Eduardo Javier Isidoro
| Vásquez Paniagua, Luis Daniel
Tutor Académico | Lárez Mata, Jesús José
Fecha | Mes, Año

// Dedicatoria
= Dedicatoria
A nuestras familias, a las grandes amistades que hicimos en la universidad, a todos los que no lo lograron, y al futuro que nos depara.

// Agradecimientos
= Agradecimientos
Gracias a la escuela de Ingeniería Informática, por su apoyo y palabras de aliento en los momentos más difíciles.
Gracias al profesor Jesús Lárez, por aceptar ser nuestro tutor, por su paciencia, regaños, correcciones, infinito saber y dedicación a la enseñanza.
Gracias al profesor y director Franklin Bello, por siempre estar presente, guiandonos y alentandonos para que sigamos adelante.
Gracias al señor Andrés, por las historias, las anécdotas, las enseñanzas y el constante apoyo y dedicación a todos los que día tras día estamos presentes y trabajando en el salón de prototipos.

// Capítulo I
= Capítulo I. El Problema
// TODO: Usar referencias y citas de Typst en vez de manuales.
// TODO: Arreglar estilado.
== Planteamiento del Problema
Papert (1980) predijo el auge de las computadoras en la educación, planteando que los niños deberían aprender a programar tal y como es aprender francés viviendo en Francia en vez de aprenderlo mediante las clases de lenguas extranjeras en las aulas del colegio; es decir, mediante la interacción directa con las computadoras, un enfoque en que el niño use y experimente con la computadora para aprender, en vez de que la computadora le enseñe al niño. También hace énfasis en que la simple presencia de las computadoras cambiaría y moldearía una nueva forma de enseñar y aprender, inimaginable para la sociedad de aquel momento. Papert fue un visionario, pues lo que él defendía se hizo realidad en partes, con una creciente demanda de la competencia del pensamiento computacional, término que no usó Papert sino Wing (2006) para referirse a la forma de pensar de los científicos de computación; pero con retos que enfrentar.
Quien acuñó el término de pensamiento computacional, Wing (2006), lo describe como una habilidad que todos deberían aprender y usar, no solo los científicos en computación. Defiende que debería de añadirse a la educación de los niños, al mismo nivel que las 3R (lectura, escritura y aritmética), por los usos que tiene, no solo al aprender a programar, sino en la descomposición y solución de problemas, la creación de modelos, el análisis de datos, la abstracción. Añade que su utilidad en otras disciplinas ya es visible, notándose en cómo el aprendizaje automático ha transformado a la estadística, el reciente interés de los científicos de computación en la biología, o la computación cuántica y su efecto en la física. El sueño de Papert, que todos tuvieran una computadora, se convirtió en lo que se conoce hoy como computación ubicua; Wing, por su parte, sueña que el pensamiento computacional sea igual de omnipresente, y propone que el primer paso es que deje de ser exclusivo de los científicos de computación, que se enseñe a los estudiantes preuniversitarios.
// ¿Por qué es importante el pensamiento computacional en niños? ->
// ¿Por qué es un problema que los niños no desarrollen el pensamiento computacional? ->
// Se debe mencionar que programar/codificar se ha convertido en una competencia básica del siglo XXI (habilidades del siglo XXI), según Sánchez Vera, et al. (2019).
// Después de mencionar que el pensamiento computacional se considera como una nueva alfabetización
// y que programar/codificar se ha convertido en una competencia básica del siglo XXI:
// Se debe mencionar que el pensamiento computacional se considera como una nueva alfabetización, la alfabetización digital (Zapata-Ros, 2015). Texto que referencia esto:
Zapata-Ros (2015) sostiene que el pensamiento computacional representa una nueva alfabetización digital que debe comenzar desde las primeras etapas del desarrollo individual, al igual que sucede con otras habilidades clave como las 3R. Esta alfabetización no se limita únicamente al aprendizaje de la programación, sino que permite a las personas organizar su entorno, desarrollar estrategias de desenvolvimiento y resolución de problemas cotidianos, además de organizar su mundo de relaciones en un contexto de comunicación más racional y eficiente, resultando en una mayor calidad de vida.

La ausencia del desarrollo del pensamiento computacional en los niños se ha convertido en un problema relevante en la sociedad actual. Sánchez Vera (2019) señala que codificar ha sido incluido específicamente como una de las competencias básicas del siglo XXI, y que el pensamiento computacional permite desarrollar una nueva alfabetización necesaria en el mundo contemporáneo, ayudando a que los individuos no sean solo consumidores digitales, sino creadores y participantes activos con las tecnologías. La falta de estas competencias limita las capacidades de los niños para desenvolverse eficazmente en un contexto cada vez más digitalizado, reduciendo su potencial para resolver problemas complejos y expresar sus ideas mediante la tecnología. Además, como indica Zapata-Ros (2015), la carencia de estas habilidades desde edades tempranas dificulta que en ciclos superiores los estudiantes puedan desarrollar plenamente el pensamiento computacional, ya que no cuentan con las bases cognitivas necesarias que se construyen mediante la manipulación de objetos y conceptos fundamentales como la seriación, la discriminación por propiedades y la secuenciación.
En la actualidad, la presencia de computadoras, tabletas, teléfonos, televisores y relojes inteligentes, y demás dispositivos con pantalla; resulta en que los seres humanos están expuestos a las pantallas durante todo el día, en períodos de tiempo extensos incluso en ambientes dedicados a la enseñanza y el aprendizaje, como los colegios y universidades; causando la preocupación generalizada por los efectos a corto y largo plazo de esto, especialmente en los niños. Council on Communications and Media et al. (2016), en representación de la Academia Americana de Pediatría, sostienen que el uso de pantallas en niños de 2 a 5 años no debe sobrepasar una hora al día, dividida en periodos cortos con descansos frecuentes. J. Duarte (comunicación personal, 28 de febrero de 2025) respalda a la AAP, añadiendo que los periodos continuos de exposición han de ser de 15 minutos como máximo, condicionando el uso de herramientas como Scratch para el desarrollo del pensamiento computacional durante sus sesiones de clase, y recurriendo en su lugar a herramientas tangibles, como el juego de mesa Mouse Mania, que presentan limitaciones para la enseñanza de conceptos avanzados, como los bucles y condicionales, y dificultan fomentar el aprendizaje colaborativo y la socialización entre niños.
La primera aproximación al uso de herramientas tangibles para la enseñanza de conceptos de programación a niños vino de parte de Radia Perlman, pionera en el área de las interfaces de usuario tangibles (Morgado et al., 2006), quien desarrolló entre 1974 y 1976 un lenguaje de programación tangible de sistemas, llamado TORTIS, con la finalidad de que los niños pudieran adquirir las competencias que traía el aprender lenguajes de programación completos al interactuar con objetos físicos. Este lenguaje consistía en controlar una pequeña “tortuga” (un disco equipado con una luz, una bocina y un lápiz, este último encargado de dibujar el resultado de la ejecución del programa) mediante uno de dos componentes: una serie de cajas de botones con acciones, o una “máquina tragacartas” con cartas de plástico. La razón detrás de la creación de dos componentes para interactuar con el lenguaje es, interpretando las palabras de Perlman, que con las cajas de botones los niños pensaban que el programa era el dibujo resultante, en vez del conjunto de comandos que ejecutaban con los botones; mientras que, con la máquina tragafichas, que construyó para solucionar el problema de las cajas, era difícil que lo niños entendieran que cada carta estaba asociada a un comando porque, para la ejecución de cada una, se debía buscar la carta, insertarla en una ranura de la máquina y presionar un botón. Incluso con estos problemas, Perlman llegó a una nueva aproximación para la enseñanza de conceptos de programación para los niños.
Un ejemplo más conocido de programación tangible es el caso de AlgoBlock (Suzuki y Kato, 1993), un lenguaje de programación tangible inspirado en Logo que consiste en unir una serie de bloques físicos para formar un programa que controla un submarino mostrado en una pantalla. Igual que las cartas en el segundo componente de TORTIS, cada bloque representa un comando, y algunos bloques representan estructuras de control condicionales y de bucle. Lo interesante de AlgoBlock son los principios que siguieron sus autores en su desarrollo: facilidad de uso, acceso simultáneo, monitoreo mutuo y pase del turno mediante gestos; los cuales incitan la conversación y la colaboración entre los participantes. Aunque los autores no mencionan alguna deficiencia en AlgoBlock tras ponerlo a prueba, sí hacen énfasis en que este solo representa el primer paso en la identificación de los principios para el diseño de ambientes de aprendizaje colaborativo.
La idea detrás de la programación tangible es llevada más allá por Zapata-Ros (2019), quien habla sobre el pensamiento computacional desenchufado, que ayuda a que los niños adquieran las competencias relacionadas al pensamiento computacional en las etapas de su vida en que más la necesiten, como durante los estudios secundarios o la universidad, basándose en los principios fundamentales de la instrucción propuestos por Merrill (2002), especialmente en el principio de activación. Zapata-Ros describe al pensamiento computacional desenchufado como actividades que fomenten en los niños una serie de habilidades que, tras ser evocadas en ciclos superiores, favorezcan el desarrollo del pensamiento computacional, ejemplos de estas son el uso de fichas, juegos en el salón de clase o en el patio, juguetes, aquellas que se suelen hacer sin el uso de pantallas ni computadoras. Además, sugiere actividades ya existentes con estas características y la forma de usarlas para promover el pensamiento computacional.
Aparte del pensamiento computacional desenchufado, la realidad aumentada espacial también presenta similitudes con la programación tangible, al permitir la superposición del mundo real con el virtual (Park et al., 2015), de cierta manera siendo la evolución de TORTIS y AlgoBlock; y, al igual que el pensamiento computacional desenchufado, se ha utilizado la realidad aumentada espacial para fomentar el desarrollo del pensamiento computacional en niños. Billinghurst y Duenser (2012) evaluaron el uso de realidad aumentada espacial en aulas de clase, específicamente en forma de libros aumentados (libros impresos en los que algunas de sus páginas tienen imágenes virtuales superpuestas) y aplicaciones móviles de realidad aumentada, siendo los resultados recopilados del uso de los libros aumentados los de mayor interés para esta investigación, destacando la utilidad en la enseñanza de conceptos espaciales al proveer una interacción intuitiva, atractiva y natural de los temas estudiados; apoyando en la retención de contenidos que se desenvuelven en secciones de realidad aumentada, permitiendo que niños con dificultades en la comprensión de textos de aprendizaje tengan la opción de interactuar con el contenido para aprender; y en el desarrollo del pensamiento computacional al hacer uso de herramientas de construcción de escenas de realidad aumentada espacial diseñadas para niños, quienes se inspiran en hacer sus propias escenas tras ver y experimentar con las de los libros aumentados. Además, como los libros en sí contienen secciones físicas intercaladas con secciones de realidad aumentada, se reducía el tiempo de uso de pantallas.
Pasando a Venezuela, Barrios (2025) de la Universidad Católica Andrés Bello (UCAB) presentó un entorno de realidad aumentada espacial que consiste en una mesa interactiva táctil, la cual permite el desarrollo de juegos sociales entre niños de educación preescolar y básica que promueven la colaboración y socialización. Con base en lo expuesto previamente, se ve la oportunidad de extender el entorno hecho por Barrios incluyendo un módulo de programación tangible, aprovechando el enfoque en juegos sociales para fomentar la colaboración, la socialización, el trabajo en equipo y el pensamiento computacional.
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
El presente trabajo tendrá como objetivo desarrollar un ambiente de programación tangible para fomentar el desarrollo del pensamiento computacional en niños entre 6 y 9 años de edad. Este estará orientado a fomentar el desarrollo del pensamiento computacional de los niños, considerando aspectos esenciales como el aprendizaje colaborativo, la socialización y la inclusividad.
En primer lugar, se llevará a cabo una etapa de revisión de conceptos para obtener detalles sobre el pensamiento computacional y su desarrollo en edades tempranas. A continuación, se realizará un análisis exhaustivo para comprender la aplicación de la programación tangible en entornos de realidad aumentada, con el fin de fomentar el pensamiento computacional.
Posteriormente, se procederá al diseño y desarrollo del entorno, que incluye dos aspectos: la construcción del hardware, que funcionará como interfaz de interacción humano-computador, y el desarrollo del software, encargado de procesar la información recibida a través del hardware. Adicionalmente, se propondrá un protocolo de pruebas que se aplicará durante la validación posterior a la construcción del entorno.
Finalmente, se elaborará la documentación correspondiente, incluyendo el manual del sistema y el manual de usuario.

== Limitaciones
=== Dificultades Asociadas a Nuevas Tecnologías
Podrían surgir inconvenientes durante el desarrollo y construcción del entorno debido a la falta de experiencia en realidad aumentada espacial.

=== Problemas Asociados a los Componentes Utilizados
Es posible que los componentes necesarios no sean de fácil acceso, especialmente el sensor Kinect, que actualmente no se comercializa con frecuencia. Además, pueden resultar costosos.
Para solucionar esto, los autores proveerán su propio Kinect, y se tomarán previsiones con respecto a los componentes faltantes.

== Justificación
Este trabajo de investigación apoyaría al desarrollo del pensamiento computacional, el aprendizaje colaborativo y la socialización entre los niños, fomentando la futura activación de estas competencias en su crecimiento personal, académico y profesional, y convirtiéndose en una herramienta útil y una gran alternativa para educadores conscientes en que los niños deben aprender sobre el mundo virtual sin sacrificar el mundo físico, con el añadido de combinar ambos mundos en una experiencia única.
=== Aporte Principal
El uso del ambiente de programación tangible permitiría que los niños aprendan de forma lúdica, creando experiencias significativas para el aprendizaje del pensamiento computacional a través de la manipulación de objetos físicos, reflejada en la visión por computador. Además, promovería el aprendizaje colaborativo, ayudándoles a desarrollar el pensamiento computacional.
=== Innovación
Se presentará una propuesta que permitiría a los niños acceder a un ambiente de aprendizaje interactivo para aprender a programar con componentes tanto físicos como digitales. Este enfoque equilibraría el uso de pantallas y las técnicas tradicionales de aprendizaje mediante la utilización de elementos físicos y digitales.
=== Beneficiarios
==== Niños entre 6 y 9 años de edad.
Fomentaría el desarrollo del pensamiento computacional en los niños desde edades tempranas, lo cual podría influir positivamente en su rendimiento académico y en su habilidad para resolver problemas lógicos.
==== Profesores de primeros grados de educación básica.
Contarían con una herramienta útil que facilitaría el proceso de enseñanza-aprendizaje a niños mediante juegos interactivos.
=== Impacto en los Objetivos de Desarrollo Sostenible
Esta investigación tendría un impacto significativo en el Objetivo 4 (Educación de calidad) de los Objetivos de Desarrollo Sostenible, especialmente en las Metas 4.4 y 4.6:
==== Meta 4.4.
Aumentaría el número de jóvenes y adultos con competencias necesarias para acceder al empleo, fomentando desde edades tempranas el desarrollo del pensamiento lógico y computacional.
==== Meta 4.6.
Promovería conocimientos básicos de aritmética en adultos, fomentando el aprendizaje desde la infancia a través de juegos interactivos.

// Capítulo II
= Capítulo II. Marco Teórico

== Antecedentes de investigación
// TODO: Acomodarlos al formato pedido en la Guía.
=== Entorno de robótica educativa multiagente orientado a favorecer el desarrollo del pensamiento computacional en jóvenes cursantes de educación media.
En este trabajo (Espejo, 2022) se destaca la construcción de un entorno de robótica educativa multiagente, cuyo objetivo es promover el pensamiento computacional. Se empleó una metodología basada en el modelo espiral, a través de la cual se definieron los requisitos y se diseñó el entorno. Este entorno se encarga de obtener información del mundo físico mediante visión por computadora y procesar dicha información para seguir las instrucciones programadas en MakeCode.
Su aporte a este trabajo es que demuestra que se puede usar la robótica multiagente para favorecer el desarrollo del pensamiento computacional en jóvenes, de modo que se toma una nueva visión del problema de “fomentar el desarrollo del pensamiento computacional” pero para niños entre 6 y 9 años y mediante el uso de programación tangible en vez de robótica multiagente.
=== Entorno de realidad aumentada espacial para el desarrollo de juegos sociales dirigidos a niños de educación preescolar.
Esta tesis (Barrios, 2025) marca un hito en los entornos de realidad aumentada espacial, presentando un producto llamado “Magicboard”. En este, los niños pueden interactuar con una pizarra digital en forma de mesa a través de un sensor que detecta gestos y objetos físicos. Este producto permite a los niños aprender mediante juegos sociales que tienen como pilar el aprendizaje colaborativo.
En este caso, se llevó a cabo una etapa de investigación en la que se obtuvieron las características esenciales para la construcción de la pizarra, considerando la manera en que los niños interactúan y aprenden. Posteriormente, se utilizó un videobeam, un sensor Kinect y el software correspondiente para la gestión de la interacción.
Este trabajo representa una continuación del producto resultante de la tesis de Barrios, buscando añadir a “Magicboard” la fomentación del desarrollo del pensamiento computacional a través de la programación tangible.
=== Sistema interactivo para la enseñanza de programación a niños con discapacidad visual.
Este trabajo (Rojas y Youssef, 2025) presenta un sistema interactivo diseñado para que los niños con discapacidad visual puedan utilizar bloques físicos para programar. Estos bloques cuentan con características táctiles que permiten a las personas con discapacidad visual leer y entender el significado de cada bloque. De esta manera, pueden construir una secuencia de instrucciones que es posteriormente procesada por un sistema de visión por computadora. Este sistema analiza la conexión entre los bloques y ejecuta la secuencia construida.
Lo más relevante de este estudio para la propuesta son los bloques físicos, ya que representan una interfaz de programación tangible; este sistema propuesto por Rojas y Youssef se puede tomar como un punto de partida para prototipos del ambiente de programación tangible que se desarrollaría en este trabajo.
=== Can computational thinking be improved by using a methodology based on metaphors and scratch to teach computer programming to children? [¿Se puede mejorar el pensamiento computacional mediante el uso de una metodología basada en metáforas y Scratch para enseñar programación a los niños?]
Este trabajo de investigación (Pérez-Marín et al., 2020) busca responder a la pregunta sobre si el pensamiento computacional puede mejorarse mediante una metodología basada en metáforas y el uso de Scratch para enseñar programación a los niños. Para ello, se llevaron a cabo experimentos y métodos de evaluación para analizar cómo los niños aprenden. Se utilizaron herramientas como Scratch y la aplicación CompThink como medios para evaluar el aprendizaje de los niños.
El trabajo se estructura en diversas secciones, cada una abordando temas específicos. En primer lugar, se presenta la sección de "Contexto", donde se expone el concepto de pensamiento computacional y las formas en que puede enseñarse la programación en la educación primaria. A continuación, se detalla la sección de "Materiales y Métodos", describiendo los procedimientos del experimento realizado. Posteriormente, se muestran los Resultados obtenidos y las Conclusiones finales, que incluyen posibles líneas de trabajo futuras.
Este estudio permite demostrar que herramientas como Scratch, el cual está basado en programación en bloques, son útiles para la enseñanza de la programación a los niños, lo cual representa un punto de partida para prototipos del ambiente de programación tangible que se desarrollaría en este trabajo.
=== Using an online serious game to teach basic programming concepts and facilitate gameful experiences for high school students [Usando un juego serio en línea para enseñar conceptos básicos de programación y facilitar experiencias divertidas para estudiantes de secundaria]
Este trabajo (Montes et al., 2021) destaca el experimento realizado por estudiantes de la Universidad Rey Juan Carlos de Madrid, en el que se utilizaron "Juegos Serios" con el objetivo de facilitar el aprendizaje de programación en niños. El experimento se llevó a cabo con 38 niños de K-10, y se buscó proporcionar una experiencia de juego satisfactoria mientras aprendían.
Los resultados demostraron que los niños tuvieron una experiencia positiva, concluyendo que el uso de juegos incrementó sus puntuaciones de aprendizaje.
Este estudio sirve como base para la propuesta, al inspirar el uso de “Juegos Serios” como herramienta clave para fomentar el aprendizaje de conceptos de programación de una manera lúdica y atractiva.

== Bases Teóricas
// TODO: Revisar si esto se adapta a lo descrito como Bases Teóricas en la Guía, o si pertenece a Terminología Básica.
=== Serious games [Juegos serios].
En un artículo del Tecnológico de Monterrey (Fuerte, 2018) se definen los juegos como “juegos diseñados con un propósito formativo más que para fines de entretenimiento.” Estos juegos permiten a los docentes enseñar a sus estudiantes sobre diversos temas, facilitando el aprendizaje mientras se divierten.
=== Pensamiento computacional.
El artículo “Research Notebook: Computational Thinking--What and Why?” [Cuaderno de Investigación: Pensamiento Computacional--¿Qué y Por Qué?] (Wing, 2011) afirma que el pensamiento computacional implica resolver problemas, diseñar sistemas y comprender el comportamiento humano, haciendo uso de los conceptos fundamentales de la informática. Este pensamiento se caracteriza por la formulación de un problema de manera que permita el uso del computador para resolverlo, la organización y el análisis lógico de la información, la representación de la información a través de abstracciones y la búsqueda de la solución más efectiva que sea capaz de resolver una familia de problemas.
=== Realidad aumentada.
La Realidad Aumentada fue definida por García Requejo (2024) como “la tecnología capaz de añadir información a una imagen del mundo real mostrada a través de un dispositivo electrónico (móvil, tablet y ordenador).” Esta tecnología tiene como principales características la capacidad de superponer elementos visuales sobre imágenes reales, proyectar imágenes en 3D que parezcan naturales con respecto al entorno real y realizar una evaluación del contexto, correspondiéndole con lo observado a través de nuestros ojos.
=== Realidad aumentada espacial.
La Realidad Aumentada Espacial fue descrita en el artículo “Spatial augmented reality for product appearance design evaluation” (Park et al., 2015) como una nueva tecnología que puede producir contenidos inmersivos al superponer la virtualidad y el entorno del mundo real. Esta tecnología se diferencia de la Realidad Aumentada en la forma en la que se muestra, ya que en la Realidad Aumentada Espacial existe una interacción con el espacio físico que posteriormente se refleja en el espacio virtual, mientras que, en el otro tipo de Realidad Aumentada, no existe interacción con el espacio físico y todo se muestra en las pantallas.
=== Aprendizaje colaborativo.
La Preparatoria Panamericana (2020) define al aprendizaje colaborativo como el “enfoque educativo que, por medio de grupos, busca mejorar el aprendizaje a través del trabajo conjunto.” Este enfoque permite una mejora en la interacción entre alumnos, la comprensión y exposición de perspectivas diversas, inspira creatividad y desarrolla habilidades de pensamiento crítico. Algunos ejemplos incluyen grupos de estudio, debates, juegos de rol, pares y la resolución de problemas de manera grupal. (LHH, 2023)
=== Neurodiversidad.
García-Bullé del Tecnológico de Monterrey (2021) define la neurodiversidad como “los individuos que viven con autismo principalmente, pero también abarca dislexia, dispraxia, déficit atencional con hiperactividad (TDAH), u otras condiciones que les llevan a navegar procesos cognitivos y emocionales de manera distinta a la norma.” El término se originó en los años 90 para promover la aceptación y el trato normal a personas que pueden actuar de forma diferente. Judy Singer (socióloga que acuñó el término en los años 90) visualiza la neurodiversidad como un “movimiento de justicia social”, con el objetivo de resaltar aquellos beneficios que tienen estas personas y generar comprensión con respecto a las limitaciones que conlleva la neurodiversidad. (Miller, 2024)
=== Visión por computador.
EDS Robotics (2022) la define como “un grupo de tecnologías o herramientas que permiten a los equipos captar imágenes del mundo real, procesarlas y generar información a través de ellas”. Gracias a estas tecnologías, se puede obtener información del entorno físico para posteriormente ser procesada y plasmada en una pantalla de entorno digital. Esta información se capta a través de un sensor, que envía las imágenes o datos a un dispositivo de interpretación que busca reconocer patrones previamente obtenidos.

// Capítulo III
= Capítulo III. Marco Metodológico

== Tipo de Investigación
Investigación proyectiva (Hurtado, 2010)

== Técnicas e Instrumentos de Recolección de Datos
Revisión documental, entrevistas semiestructuradas

== Metodología de Desarrollo Utilizada
Al analizar las características del trabajo de investigación, se consideró el enfoque a adoptar. Dado que no se previó un contacto constante con el cliente y que los requisitos aún no estaban bien definidos, se decidió optar por un enfoque basado en prototipos, con el fin de definir los requerimientos finales a través de los prototipos realizados y sus validaciones.
Según Pressman (2010), el enfoque basado en prototipos está enmarcado dentro de los modelos de proceso evolutivos, que "son iterativos. Se caracterizan por la manera en la que permiten
desarrollar versiones cada vez más completas del software.". Particularmente para el enfoque basado en prototipos, el proceso se divide en 4 fases, como se observa en la @prototyping-figure: comunicación, plan rápido - modelado - diseño rápido, construcción del prototipo y despliegue - entrega y retroalimentación. Se definen a continuación:
#figure(
  image("images/prototyping-paradigm.png"),
  caption: [
    Etapas del enfoque basado en prototipos. Tomado de (cambiar esto por referencia de Typst, no textual) Ingeniería del software. Un enfoque práctico, (p. 36), por Pressman, 2010, México.
  ],
) <prototyping-figure>

[Colocar imagen del pressman, figura 2.6, con descripción] figura uwu
- Comunicación: Se establece comunicación con los interesados (clientes, usuarios, participantes) para definir los objetivos generales, qué requerimientos se conocen, y en qué se requiere una mejor definición.
- Plan rápido - modelado - diseño rápido: A diferencia de otros enfoques, donde la planificación, modelado y diseño son exhaustivos; en el enfoque basado en prototipos, el énfasis está en definir qué partes del software serán visibles para los usuarios y hacer representaciones de estas (por ejemplo, la interfaz que usarán para interactuar con el software), de modo que se pueda pasar rápidamente a la construcción del prototipo.
- Construcción del prototipo: Se construye un prototipo, que sirve como una versión preliminar del sistema, donde la mantenibilidad a largo plazo o la calidad general no son tan relevantes. Al ser necesario que funcione pronto, es común que se tomen decisiones cuestionables durante la implementación, como la elección de lenguajes de programación inapropiados o uso de algoritmos poco eficientes.
- Despliegue - entrega y retroalimentación: El prototipo construído se despliega para ser evaluado por los interesados, quienes proporcionan retroalimentación, que se usa para refinar los requerimientos.
Las iteraciones continúan mientras se busca que los prototipos que se construyan se acerquen cada vez más a cumplir con las necesidades de los interesados, lo que a su vez ayuda a comprender mejor qué se necesita como producto final. Así pues, los prototipos funcionan como un mecanismo para definir los requerimientos del sistema, reducir riesgos y, dependiendo de cómo se construyan, ser descartados o evolucionar hasta convertirse en el producto final.
En este caso, se utilizó como base el trabajo de investigación “Entorno de Realidad Aumentada Espacial para el Desarrollo de Juegos Sociales Dirigidos a Niños de Educación Preescolar”, que sirvió como punto de partida para el modelado y diseño de los primeros prototipos. A partir de los resultados obtenidos con los prototipos, se definieron los requerimientos finales del entorno a desarrollar.

// Capítulo IV
= Capítulo IV. Desarrollo y Resultados
¿Basarnos en el cronograma de actividades original para iniciar con la redacción?

Analizar el uso de programación tangible en entornos de realidad aumentada, a fin de caracterizar el ambiente a desarrollar.


Diseñar un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, en función del análisis realizado.


Construir un ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años, en base al diseño realizado.

Prototipo 1:

Partiendo de la tesis de Anthony Barrios, se buscó una aproximación más programática, asimilándose a Scratch, por lo que se partió de seguir el paradigma imperativo y la programación con bloques. Sin embargo, dado que Scratch ya es ampliamente usado y tiene varias investigaciones al respecto de su uso (hablar más sobre esto en el objetivo 1), el tutor sugirió seguir una aproximación distinta, basada en el paradigma de programación dataflow, pues ofrece una clara visualización de cómo fluyen y se transforman los datos del programa, además de ser uno del que poco se ha hablado, más en su aplicación para fomentar el desarrollo del pensamiento computacional.

Dado este cambio, se procedió con la definición de los primeros datos y operaciones a usar, para lo que se eligieron bloques con formas geométricas simples (cuadrados, círculos y triángulos) y colores básicos (morado, amarillo, naranja, verde, rojo y azul) que, para simplificar el desarrollo, se decidió que algunos representarían operaciones en vez de un dato. Los datos que se soportaban provenían directamente de las formas (cuadrados, círculos y triángulos de distintos colores), y las operaciones eran conjunción, intersección, diferencia y diferencia simétrica. El diseño consistió de zonas que reconocían las formas colocadas como datos, otras que reconocian las formas como operaciones, y zonas de salida que mostraban el resultado de la ejecución. Todas estas zonas estaban colocadas de forma fija, restringiendo la creación de nuevas zonas o la asociación entre estas para el usuario final, lo que limitaba la flexibilidad del entorno pero facilitaba el desarrollo del prototipo.

Para la construcción, se decidió continuar el uso de Python para todo, haciendo uso de OpenCV y OpenNI para la visión por computador, y también OpenCV para la interfaz gráfica. Se usó un sensor Kinect para la captura de imágenes, y se implementó un sistema de reconocimiento de formas basado en la detección de contornos, que permitía identificar las formas geométricas y sus colores para determinar los datos y operaciones a ejecutar. El resultado de la ejecución se mostraba en una zona de salida mediante la superposición de imágenes generadas por el software. Este prototipo puede verse en la @first-prototype-figure.

#figure(
  image("images/first-prototype.jpeg"),
  caption: [
    Primer prototipo del ambiente, con bloques de formas geométricas simples y colores básicos para representar datos y operaciones.
  ],
) <first-prototype-figure>

Como resultado de este prototipo, se vió que no se podía partir directamente del código legado por Barrios, pues se necesitaban de librerías más potentes para tener una interfaz gráfica más atractiva, algoritmos más robustos para la detección de piezas más complejas (números, imágenes), y una arquitectura de software más flexible para permitir la creación de nuevas zonas y la asociación entre estas. Además, surgió la inquietud de que las resoluciones de las cámaras del sensor Kinect v1 no fueran suficientes para detectar piezas más complejas, lo que llevó a la decisión de cambiar al sensor Kinect v2, lo que permitiría una detección más precisa y robusta.

Prototipo 2:

Debido a las preocupaciones con respecto al Kinect v1, y tras analizar las posibles ventajas, se concluyó que se intentaría el cambio al Kinect v2. El desarrollo de este prototipo entonces se enfocó en la adaptación del código legado por Barrios, para la calibración y detección de toques, para soportar el nuevo sensor.

Así pues, se llevó a cabo una investigación sobre el uso del Kinect v2 con Python, las diferencias entre el Kinect v1 y el Kinect v2, las librerías disponibles para la visión por computador con este nuevo sensor, y el algoritmo de detección de toques basado en profundidad.

Las librerías disponibles para integrar el Kinect v2 con Python son limitadas. Se probaron aproximaciones con PyKinect2 y libfreenect2, sin embargo, el primero fallaba por falta de soporte para Python 3+, y el segundo no detectaba el Kinect v2;OpenNI2, que se usó para el Kinect v1, no es compatible con el Kinect v2 por defecto, pero existen parches para hacerlo compatible, con lo cual se logró usar OpenNI2 para la integración del Kinect v2 con Python. Siguiendo con la calibración y detección de toques, se hicieron modificaciones exhaustivas al código legado para adaptarlo al nuevo sensor, lo que llevó a la implementación de un nuevo algoritmo de detección de marcadores (2 cuadrados blancos en las esquinas superior izquierda e inferior derecha de la proyección), además de la afinación de múltiples números mágicos (literales escritos en el código sin documentar su significado). Este prototipo puede verse en la @second-prototype-figure.

#figure(
  image("images/second-prototype.jpeg"),
  caption: [
    Segundo prototipo del ambiente, con el cambio al sensor Kinect v2 y la adaptación del código legado para la calibración y detección de toques.
  ],
) <second-prototype-figure>

Este prototipo, si bien permitió validar la viabilidad del cambio al Kinect v2, también mostró que el código legado por Barrios era difícil de mantener. También se vió que la transformación Window-to-Viewport que se usa en los algoritmos es muy sensible a la configuración física del entorno (paralelismo entre la proyección sobre la superficie y el ángulo de la cámara), resultando en que la detección de toques no fuese tan precisa como se esperaba.

Prototipo 3:

Tras trabajar tanto en una única parte del sistema (integración con el hardware y detección de toques), se decidió que el siguiente paso sería trabajar en la visión por computador para la detección de las piezas que conformarían el entorno. (...)


Validar el ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.


Realizar la documentación formal del ambiente de programación tangible con realidad aumentada espacial orientado a niños entre 6 y 9 años construido.

