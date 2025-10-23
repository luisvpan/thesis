//TODO: prototipar y formalizar este diseño, referenciar, etc. etc. etc.
= Diseño preliminar del ambiente

== Aspectos generales
Este ambiente se enmarca como una interfaz de usuario tangible (TUI) de tipo tabletop /* (¿mandar explicación de TUI tabletop a glosario de términos?) */junto a una interfaz gráfica de usuario (GUI), con entradas físicas mediante la TUI, y entradas digitales y salidas digitales mediante la GUI. Está compuesto por un conjunto de elementos tangibles y digitales que representan uno de dos conceptos: origen de dato u operación sobre datos /* (que, en conjunto, representan un programa del lenguaje de programación tangible, más detalles en sus respectivas secciones) */, un computador, un proyector, una cámara de color y de profundidad, y una superficie plana dividida en, al menos, dos secciones: una donde se colocan únicamente los elementos tangibles, y una donde se proyecta la GUI y sucede la interacción entre elementos tangibles y digitales. En esta última sección, la cámara captura la escena y envía la información a un computador que procesa la imagen, reconoce los elementos tangibles, sus posiciones y conexiones /* (más adelante se profundiza en el concepto de conexiones) */, ejecuta el programa representado por los elementos tangibles y muestra la salida digital sobre la GUI.
El ambiente está pensado para ser utilizado por niños de 6 a 9 años, a partir del uso de los elementos tangibles proveídos. Los niños, mediante los elementos tangibles, crearan programas que den solución a una serie de problemas, fomentando así el desarrollo de su pensamiento computacional en el proceso de descubrir y plasmar con elementos tangibles su solución /* (destacando _su_, ya que no necesariamente deben llegar a una "mejor solución", y se lleva como supuesto que cada niño llegará a su propia forma de resolver el problema mediante los elementos tangibles proveídos) */. El ambiente es capaz de compilar los programas escritos mediante elementos tangibles en instrucciones ejecutables por la computadora, para después ejecutarlas y mostrar la salida en la GUI, que es proyectada sobre la superficie plana para mostrarle al niño el resultado de su programa.
Los profesores también son considerados en el diseño del ambiente, ya que se busca que guien a los niños en el uso del ambiente y apoyen en el desarrollo del pensamiento computacional. Para ello, el ambiente permite a los profesores crear y gestionar actividades /* (las cuales son problemas que se pueden resolver con los elementos tangibles dados, más detalles en su sección) */. Adicionalmente, el ambiente provee a los profesores de una guía de diseño de actividades.
Los datos, operaciones sobre datos, programas que se pueden desarrollar, y las actividades dadas como ejemplo están basadas en el currículo de matemáticas de 1er a 3er grado dado por el Ministerio del Poder Popular para la Educación de Venezuela para el año 2025, con la finalidad de que el ambiente sea integrable dentro de aulas de clase de estos grados y cumpla su objetivo de fomentar el desarrollo del pensamiento computacional en niños. /* #strike[creo que esto necesita un mejor parafraseo, me suena raro] */

== El lenguaje de programación tangible propuesto
Se propone un lenguaje de programación basado en el concepto del dataflow, haciendo uso de elementos tangibles, que puede ser una tarjeta con vocabulario, un bloque de madera o una figura de foami. Los flujos de datos son representados mediante estos elementos tangibles en conjunto con una serie de conexiones /* (más detalles en conexiones entre orígenes de datos y operaciones sobre datos) #strike[quizás valga la pena abstraer más y hablar del lenguaje de programación como un todo, que tiene partes tangibles y partes digitales, para luego caer en detalles concretos] */, y las operaciones son también representadas mediante elementos tangibles.
Este lenguaje de programación no está pensado para ser Turing completo, sino para ser lo suficientemente expresivo para resolver un subconjunto de los problemas planteables en el currículo previamente mencionado, y que a su vez sea sencillo de entender y usar por niños de 6 a 9 años. /* #strike[se puede incluir una cita de que los lenguajes de programación tangibles tienden a ser así] */

// === Inspiración en el paradigma dataflow
// #strike[Hablar sobre el paradigma, cómo se refleja en el ambiente y por qué se escogió]

=== Datos
Estos pueden ser:
- Números naturales. /* #strike[¿limitarlos hasta el 10.000?] */
- Figuras geométricas. /* #strike[¿limitarlo hasta hexágonos?] */
- Conjuntos homogéneos de otros datos.

=== Operaciones sobre datos
Estas pueden ser:
- Para números naturales:
  - Suma, resta, multiplicación y división.
  - Comparación, orden, serie.
- Para figuras geométricas:
  - Comparación, contar, orden, serie.
- Para conjuntos homogéneos de datos: Aquellas operaciones aplicables a los datos que lo conforman.

// === Contrapartes digitales

// === Programas

== La interfaz gráfica de usuario propuesta

=== Áreas delimitadas para orígenes de datos y zonas de transformación
Los orígenes de datos y las zonas de transformación son definidas por los niños mediante la colocación de los elementos tangibles y su agrupación en la superficie plana. Esta agrupación también es dada por los niños, mediante el trazado de áreas delimitadas en la superficie plana, que son detectadas por el ambiente mediante visión por computador. Cada área delimitada puede representar un origen de datos o una zona de transformación. En caso de ser un origen de datos, los elementos tangibles dentro de esa área son los datos que conforman dicho origen de datos; y en caso de ser una zona de transformación, los elementos tangibles dentro de esa área son las operaciones que se aplicarán a los datos de los orígenes de datos conectados.

=== Conexiones entre orígenes de datos y zonas de transformación
Las conexiones entre orígenes de datos y zonas de transformación no son representadas físicamente mediante cables u otros elementos tangibles, sino trazadas en la superficie plana por los niños, usando sus dedos. Estas conexiones son capturadas por la cámara y procesadas por el computador para reconocer las relaciones entre orígenes de datos y zonas de transformación.

Tanto los orígenes de datos como las zonas de transformación son representados en la GUI, que reflejan la información dada por los elementos tangibles y las conexiones trazadas en la superficie plana. La GUI destaca aquellos elementos tangibles y conexiones que están siendo reconocidos por el sistema, para darle retroalimentación visual al niño sobre el estado de su programa, así como para ayudarlo a identificar posibles errores o áreas de mejora. Esto incluye:
- Resaltar orígenes de datos y zonas de transformación reconocidos.
- Resaltar conexiones reconocidas entre orígenes de datos y zonas de transformación.
- Mostrar mensajes de error o advertencia en caso de que haya problemas con el programa representado.
- Resaltar orígenes de datos y conexiones erróneas o inválidas.
- Resaltar elementos tangibles no reconocidos o que no estén siendo utilizados en el programa.

=== Ejecución y salida
La salida de los programas descritos mediante los elementos tangibles es mostrada en la GUI proyectada sobre la superficie plana haciendo uso de las contrapartes digitales del lenguaje de programación tangible. El conjunto de ambos elementos, físicos y digitales, son la representación visual de un programa que implementa una solución al problema planteado por la actividad.

== Actividades
Las actividades se componen por un problema, una explicación de los conceptos involucrados, las condiciones durante el desarrollo, el comienzo de la actividad y el resultado esperado; que pueden ser resueltas mediante el uso del lenguaje de programación tangible propuesto y los elementos tangibles proveídos por el ambiente. Estas actividades están basadas en el currículo de matemáticas de 1er a 3er grado dado por el Ministerio del Poder Popular para la Educación de Venezuela para el año 2025, y están diseñadas para fomentar el desarrollo del pensamiento computacional en niños de 6 a 9 años. Cada actividad presenta un problema que los niños deben resolver mediante la creación de un programa usando los elementos tangibles. El ambiente permite a los profesores crear y gestionar estas actividades, proporcionando una interfaz para definir nuevos problemas.

=== Guía de diseño de actividades
Está conformada por una serie de actividades modelo, con problemas y soluciones dadas por los autores del ambiente e inspiradas en el currículo previamente mencionado.
Sin embargo, esta guía le recuerda al profesor que no debe usarse como fuente de la verdad, en términos de qué problemas pueden ser resueltos por el ambiente, ni para limitar las soluciones que pueden darse a los problemas, recalcando la importancia de que el niño tenga un papel activo en la búsqueda de soluciones.
