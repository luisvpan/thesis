= Diseño del ambiente y del lenguaje de programación tangible

== Introducción al diseño

Este capítulo describe el diseño del ambiente de aprendizaje y del lenguaje de programación tangible denominado ERAE, en coherencia con los requerimientos funcionales y no funcionales del sistema. Se distingue deliberadamente lo pedagógico y físico del ambiente, la arquitectura lógica del software, la especificación conceptual y formal del lenguaje, y la forma en que el compilador y el entorno de ejecución se integran con otros subsistemas.

== Requisitos y contexto

El ambiente está dirigido a niños de 6 a 9 años y a docentes de educación primaria (1.er a 3.er grado). Los niños construyen soluciones a problemas planteados por el docente, manipulando elementos tangibles y la interacción digital sobre la superficie de trabajo; se prioriza que cada niño exprese su forma de resolver el problema con los medios disponibles, sin imponer una única solución óptima. Los docentes orientan el uso del ambiente y el desarrollo del pensamiento computacional.
// el sistema debe permitir crear y gestionar actividades alineadas al currículo.

Los contenidos sobre los que se apoyan datos, operaciones, actividades de ejemplo y criterios de integración en aula se toman del currículo de matemáticas de 1.er a 3.er grado del Ministerio del Poder Popular para la Educación de Venezuela correspondiente al año 2025, de modo que el ambiente pueda incorporarse de forma coherente a las planificaciones de esos grados.

== Arquitectura física y lógica del ambiente

El sistema se concibe como la combinación de una interfaz de usuario tangible (TUI) de tipo tabletop y una interfaz gráfica de usuario (GUI). Las entradas físicas se realizan mediante la TUI (piezas, regiones); las conexiones digitales (generadas a través de toques sobre la superficie y reconocidas como enlaces en el grafo) y las salidas se canalizan por la GUI proyectada. Así se satisface el requerimiento de que los programas combinen elementos tangibles y conexiones digitales que representen datos, flujos y operaciones.

El ambiente material incluye, como mínimo, un conjunto de elementos tangibles y representaciones digitales asociadas que denotan orígenes de datos u operaciones sobre datos (que, articulados con las conexiones inferidas, constituyen un programa en el lenguaje tangible); un computador; un proyector; una cámara de color y de profundidad; y una superficie plana dividida en al menos dos zonas. En una zona se colocan exclusivamente los elementos tangibles; en la otra se proyecta la GUI y tiene lugar la interacción entre lo tangible y lo digital. La cámara captura la escena en esa segunda zona, envía la información al computador, el cual interpreta la imagen, reconoce elementos, posiciones y relaciones entre orígenes de datos y zonas de transformación, compila y ejecuta el programa inferido y proyecta la salida sobre la superficie plana.

A nivel lógico, el flujo a seguir es: captura; reconstrucción del programa (representación estructurada); compilación y ejecución; presentación de resultados y retroalimentación (visual y auditiva) que orienta al niño durante y después de la construcción. El núcleo de compilación y ejecución se describe en la sección de integración; aquí basta señalar que está pensado para operar sobre representaciones del programa compatibles con la especificación del lenguaje ERAE, sin acoplarse a protocolos de red concretos.

== Interacción y percepción

=== Áreas delimitadas para orígenes de datos y zonas de transformación

Los orígenes de datos y las zonas de transformación los definen los niños al colocar elementos tangibles y al delimitar regiones sobre la superficie. Esas regiones se detectan por visión por computador. Cada región acotada puede interpretarse como un origen de datos o como una zona de transformación. Si es un origen de datos, los elementos tangibles dentro de la región son los valores que conforman dicho origen; si es una zona de transformación, los elementos tangibles en su interior representan las operaciones que se aplicarán a los datos procedentes de los orígenes conectados.

=== Conexiones entre orígenes de datos y zonas de transformación

Las conexiones no se materializan con cables ni con piezas adicionales: constituyen conexiones digitales trazadas por los niños sobre la superficie (por ejemplo mediante toques). La cámara y el computador reconocen esas conexiones y establecen la relación entre orígenes y zonas de transformación.

La GUI refleja orígenes, zonas y conexiones inferidos a partir de lo físico y lo trazado. Cumple el requerimiento de retroalimentación para guiar durante la construcción y el requerimiento no funcional de retroalimentación visual y auditiva. En lo visual, entre otros:

- Resaltar orígenes de datos y zonas de transformación reconocidos.
- Resaltar conexiones reconocidas entre orígenes y zonas de transformación.
- Mostrar mensajes de error o advertencia cuando el programa sea inválido o incompleto.
- Resaltar orígenes o conexiones erróneas o inválidas (manejo de errores de disposición).
- Señalar elementos tangibles no reconocidos o no utilizados en el programa actual.

En lo auditivo, se complementa con señales sonoras acordes a reconocimiento correcto, advertencia o error, de modo que la guía no dependa solo de la vista. El modo incremental de integración (véase más adelante) refuerza la guía continua mientras el grafo está aún incompleto.

=== Ejecución y salida

La salida del programa se muestra en la GUI proyectada sobre la superficie plana, usando las representaciones digitales del lenguaje. La composición de piezas físicas, conexiones digitales trazadas y elementos en pantalla constituye la representación visible de un programa que aborda el problema de la actividad en curso.

== Visión del lenguaje en el ambiente

El lenguaje ERAE es un lenguaje de flujo de datos (dataflow), donde los programas se representan como grafos de nodos que producen valores, los transforman y declaran salidas. En el ambiente, ese grafo tiene una parte tangible (piezas, disposición, regiones) y una parte digital (conexiones inferidas del trazado, proyección, estado de reconocimiento, mensajes y retroalimentación sonora), en línea con los requerimientos de datos, flujos y operaciones combinados en una sola construcción compartida entre el niño y el sistema.

No se persigue la Turing-completitud como objetivo pedagógico; se busca un lenguaje suficientemente expresivo para un subconjunto de problemas acordes al currículo citado, y simple de interpretar por niños de 6 a 9 años. La evaluación del programa puede describirse de forma abstracta como bajo demanda, en la línea de lenguajes de flujo de datos clásicos como Lucid (los nodos se evalúan cuando sus resultados son requeridos por otros nodos o por la salida).

La especificación detallada de tipos, operadores y estructura sintáctica del lenguaje se presenta en la siguiente sección.

== Especificación del lenguaje de programación tangible ERAE

=== Filosofía de diseño

Los principios rectores, en línea con prácticas de lenguajes educativos como el enfoque de tipos fijos de Scratch, son:

- Tipos integrados: conjunto de tipos cerrado, sin extensión por parte del usuario, para reducir la carga cognitiva.
- Operaciones seguras: comprobación en tiempo de compilación de compatibilidad de datos entre operadores.
- Prevención de errores: verificación estricta de tipos y de la aridad de cada operación (número correcto de entradas), en apoyo al manejo de errores en la disposición tangible y digital antes de ejecutar.
- Alineación curricular: tipos y operaciones elegidos para mapearse a clasificación, comparación y manipulación de colecciones propios de primaria, en coherencia con el currículo de matemáticas de referencia.

=== Estructura de un programa

Un programa válido se organiza como una colección de declaraciones. A nivel conceptual, los nodos se clasifican en:

- Nodos de fuente (source): aportan datos iniciales al grafo.
- Nodos de transformación (transform): aplican operaciones a las entradas que reciben por las conexiones del flujo de datos.
- Nodos de salida (output): designan los valores que deben mostrarse o entregarse al entorno de visualización.

La sintaxis concreta (palabras clave, literales y reglas de formación) se especifica formalmente mediante una gramática en notación EBNF de la W3C; la gramática completa y los ejemplos extendidos pueden consignarse en anexo o en el documento de especificación del lenguaje, para no duplicar aquí decenas de reglas léxicas.

=== Tipos de datos

Tipos numéricos y escalares primitivos:

- Naturales: enteros mayores o iguales que cero.
- Enteros: positivos y negativos.
- Decimales: números con parte fraccionaria para medidas.
- Fracciones: representación explícita de cocientes (por ejemplo $1/2$, $3/4$).
- Texto: cadenas para etiquetas y valores simbólicos.
- Booleanos: verdadero o falso.

Tipos curriculares:

Tipos orientados a objetos manipulables y clasificables en actividades escolares:

- Formas: atributos de tipo geométrico (círculo, triángulo, cuadrado), tamaño y color.
- Coches: atributo de color.
- Comida: atributos de sabor (dulce, salado, agrio, amargo) y color.
- Animales: tipo de animal y color.
- Personas: grupo etario y género.

Los valores concretos permitidos para cada atributo (por ejemplo paleta de colores o conjunto de tipos de forma) están fijados en la especificación formal del lenguaje para mantener coherencia entre tangibles, reconocimiento y ejecución.

Tipos compuestos:

- Conjuntos (set<T>): colecciones homogéneas de elementos de un tipo T.
- Flujos (stream<T>): secuencias de valores en el tiempo, en correspondencia con la naturaleza dataflow del lenguaje y con patrones de iteración o señales discretas.

=== Catálogo de operaciones

Las operaciones se agrupan en familias. La lista siguiente resume las categorías previstas en la especificación; cada operador tiene firmas de tipo que el compilador debe respetar.

Operaciones numéricas: suma, resta, multiplicación, división.

Operaciones de comparación: comparación general de igualdad; comparaciones por tamaño, color, tipo, sabor, grupo etario y género, según los tipos involucrados.

Filtrado y selección: filtro genérico y variantes por tamaño, color, tipo, sabor, grupo etario y género, para extraer elementos de conjuntos que cumplan condiciones.

Operaciones de conjuntos: unión, intersección, diferencia y complemento.

Ordenación: orden general y orden alfabético cuando aplique.


=== Ejemplo ilustrativo

El siguiente fragmento es solo ilustrativo de la forma de los programas; la sintaxis definitiva y los nombres exactos de operadores coinciden con la gramática del documento de especificación.

```dataflow
source a: natural = 3;
source b: natural = 2;
transform sum: natural = ADD(a, b);
output result: natural = sum;
```

== Integración del compilador y el runtime con el resto del sistema

=== Principio arquitectónico

Se adopta una separación entre núcleo sin estado y adaptadores delgados. El compilador y el runtime no conocen los detalles de HTTP ni WebSocket, ya que reciben datos de programa, devuelven resultados o diagnósticos, y no mantienen sesión de usuario. La comunicación con la visión artificial, el IDE o la proyección se implementa en capas periféricas que serializan y deserializan solicitudes y respuestas.

=== Modos de evaluación

Modo por lotes (batch): pensado para ejecutar un programa completo cuando la escena ya está estable o cuando el subsistema de visión entrega un grafo cerrado. Entrada: programa completo y válido (por ejemplo en JSON). Proceso: compilar, validar y ejecutar. Salida: resultados finales y traza de ejecución. Caso de uso típico: la visión detecta que el niño terminó de montar el programa, envía la representación y se proyecta el resultado final.

Modo incremental: pensado para retroalimentación mientras el programa aún se construye (requerimiento funcional de guía durante la construcción). Entrada: grafo parcial. Proceso: validar el fragmento y evaluar únicamente lo que sea semánticamente posible. Salida: valores parciales o estados de pendiente en nodos aún incompletos. Caso de uso: el niño añade o conecta un bloque y el sistema responde al instante si faltan entradas o si una parte del grafo ya puede mostrarse; la capa de presentación puede combinar esta salida con pistas visuales y auditivas.

=== Interfaces de integración

La capa de integración prevé, entre otros mecanismos, una API HTTP para el modo por lotes y un servidor WebSocket para el modo en vivo con el IDE o entornos de construcción interactiva. El protocolo de lenguaje de servidores (LSP) puede utilizarse para asistir al editor o IDE que acompañe el diseño de actividades avanzadas, en coherencia con los objetivos de herramientas de apoyo al lenguaje ERAE.

// == Actividades y rol docente
== Rol del docente

//=== Definición y gestión de actividades

// Una actividad agrupa: el enunciado del problema, la explicación de los conceptos involucrados, las condiciones durante el desarrollo, el inicio de la tarea y el resultado esperado. Los niños resuelven la actividad construyendo un programa con el lenguaje tangible y los elementos provistos por el ambiente. El sistema permite a los docentes crear, editar y organizar actividades alineadas al currículo de matemáticas de 1.er a 3.er grado (MPPE, 2025) y orientadas al desarrollo del pensamiento computacional en la franja de edad objetivo.

=== Guía de diseño de actividades

La guía incluye actividades modelo con problemas y soluciones de referencia elaboradas por los autores del ambiente, inspiradas en el mismo currículo. Su función es formativa: no debe interpretarse como catálogo cerrado de los únicos problemas que el ambiente admite, ni como restricción a la variedad de soluciones válidas. Se enfatiza el papel activo del niño en la exploración de estrategias y representaciones.

// == Formalización adicional y referencias internas

// La gramática EBNF completa, el inventario exhaustivo de literales para tipos curriculares y los ejemplos de programas en distintos dominios pueden incorporarse como anexo al trabajo de grado o mantenerse en un documento de especificación separado (Diseño del Lenguaje ERAE), citado desde este capítulo. Cualquier divergencia futura entre implementación y especificación debe resolverse actualizando primero la especificación y luego el texto del diseño, para conservar trazabilidad académica.
