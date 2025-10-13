= Diseño preliminar del ambiente de programación tangible con realidad aumentada espacial para fomentar el pensamiento computacional en niños de 6 a 9 años

== Aspectos generales
Este ambiente se enmarca como una interfaz de usuario tangible (TUI) de tipo tabletop (¿mandar explicación de TUI tabletop en glosario de términos?) junto a una interfaz gráfica de usuario (GUI), con entradas físicas mediante la TUI y salidas digitales mediante la GUI. Está compuesto por un conjunto de figuras tangibles (bloques de ahora en adelante) que representan uno de dos conceptos: origen de dato u operación sobre datos (que en conjunto representan un programa, o propiamente dicho, un lenguaje de programación tangible, más detalles en sus respectivas secciones), un computador, un proyector, una cámara de color y de profundidad, y una superficie plana donde se colocan los bloques. La cámara captura la escena y envía la información a un computador que procesa la imagen, reconoce los bloques, sus posiciones y conexiones (más adelante se profundiza en el concepto de conexiones), ejecuta el programa representado por los bloques, renderiza la salida digital sobre la GUI, y proyecta la GUI sobre la superficie plana.
El ambiente está pensado para ser utilizado por niños de 6 a 9 años, a partir del uso de los bloques proveídos. Los niños, mediante los bloques, crearan programas que den solución a una serie de problemas, fomentando así el desarrollo de su pensamiento computacional en el proceso de descubrir y plasmar su solución (destacando _su_, ya que no necesariamente deben llegar a una "mejor solución", y se lleva como supuesto que cada niño llegará a su propia forma de resolver el problema mediante los bloques proveídos). El ambiente, haciendo uso de conceptos de visión por computador, es capaz de traducir los programas escritos mediante bloques en instrucciones ejecutables por la computadora, tras lo cual procede a mostrar la salida en la GUI, que es proyectada sobre la superficie plana para mostrarle al niño el resultado de su programa. Se consideran aspectos de diseño para esta audiencia, como el uso de colores llamativos y formas simples en los bloques, así como el diseño de una interfaz gráfica intuitiva y amigable. Además, se considera la ergonomía del espacio físico donde se utilizará el ambiente, asegurando que los niños puedan interactuar con los bloques y la superficie plana de manera cómoda y segura.
Los profesores también son considerados en el diseño del ambiente, ya que se busca que puedan guiar a los niños en el uso del ambiente y apoyar en el aprendizaje del pensamiento computacional. Para ello, el ambiente permite a los profesores crear y gestionar actividades (las cuales son problemas que se pueden resolver con los bloques dados, más detalles en su sección). Adicionalmente, el ambiente provee a los profesores de una guía de diseño de actividades.
Los orígenes de datos, operaciones sobre datos, programas que se pueden desarrollar, y las actividades dadas como ejemplo están basadas en el currículo de matemáticas de 1er a 3er grado dado por el Ministerio del Poder Popular para la Educación de Venezuela para el año 2025, _con la finalidad de que el ambiente sea integrable dentro de este mismo currículo y cumpla su objetivo de fomentar el desarrollo del pensamiento computacional en niños_ #strike[creo que esto necesita un mejor parafraseo, me suena raro] (más detalles en sus respectivas secciones).

== El lenguaje de programación tangible propuesto
Aquí se propone un lenguaje de programación basado en el concepto del dataflow, haciendo uso de figuras tangibles, por resumir bloques, que se componen de una tarjeta con vocabulario con un espacio para una figura y la correspondiente figura (más detalles en orígenes de datos y operaciones sobre datos). Los flujos de datos son representados mediante estos bloques en conjunto con una serie de conexiones (más detalles en conexiones entre orígenes de datos y operaciones sobre datos) #strike[quizás valga la pena abstraer más y hablar del lenguaje de programación como un todo, que tiene partes tangibles y partes digitales, para luego caer en detalles concretos], y las operaciones son también representadas mediante bloques.
Las figuras provistas por el ambiente están hechas en foami. #strike[¿de qué material hacemos las tarjetas con vocabulario?]

=== Orígenes de datos
Estos pueden ser:
- Números naturales. #strike[¿limitarlos hasta el 10.000?]
- Figuras geométricas. #strike[¿limitarlo hasta hexágonos?]
- Conjuntos homogéneos de otros orígenes de datos.

=== Operaciones sobre datos
Estas pueden ser:
- Para números naturales:
  - Suma, resta, multiplicación y división.
  - Comparación, orden, serie.
- Para figuras geométricas:
  - Comparación, contar, orden, serie.
- Para conjuntos homogéneos de orígenes de datos: Aquellas operaciones aplicables al origen de datos que lo conforma.

=== Programas

== La interfaz gráfica de usuario propuesta

=== Representación de lo tangible
Cada bloque del lenguaje de programación tangible tiene su contraparte digital, la cual depende de su tipo.

==== Orígenes de datos

==== Operaciones sobre datos

==== Programas

=== Conexiones entre orígenes de datos y operaciones sobre datos

=== Ejecución y salida
Haciendo uso de las contrapartes digitales del lenguaje de programación tangible, se renderiza en la interfaz gráfica la ejecución del programa descrito (que conformarían los pasos de un algoritmo) y la salida del mismo, siendo el conjunto de ambos una representación visual de cómo solucionar el problema planteado por la actividad.

== Actividades

=== Guía de diseño de actividades
Está conformada por una serie de actividades modelo, con problemas y soluciones dadas por los autores del ambiente e inspiradas en el currículo previamente mencionado. Cada actividad correspondería a:
- Un problema.
- Una serie de soluciones, descritas en términos del lenguaje de programación tangible.
Sin embargo, esta guía desde un inicio y en cada actividad le recuerda al profesor que es solo eso, *una guía*, y que no ha de usarse como fuente de la verdad absoluta en términos de qué problemas pueden ser resueltos por el ambiente ni para limitar las soluciones que pueden darse a los problemas, recalcando la importancia de que el niño tenga un papel activo en la búsqueda de soluciones mediante la creación de programas.
