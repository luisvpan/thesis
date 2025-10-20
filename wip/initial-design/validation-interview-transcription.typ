Esta transcripción fue hecha con Whisper de OpenAI y luego parcialmente revisada por nosotros. Se omiten ciertos fragmentos irrelevantes para el análisis. No está especificado en todo quién dijo cada cosa.

= Leyenda
Eduardo Arzolay: EA
Luis Vázquez: LV
Andreina Dávila (coordinadora de la escuela de educación de la UCAB): AD

= Transcripción

EA: Bueno, presentaciones primero. Hola profe, somos estudiantes de la escuela de ingeniería informática, ya somos tesistas, terminamos nuestra carga. Él es Luis Vázquez, yo soy Eduardo Arzolay.
AD: Un placer.
EA: Y bueno, estamos trabajando en nuestro trabajo de grado, que lleva como título Ambiente de Programación Tangible con Realidad Aumentada Espacial para niños de 6 a 9 años. ¿Cuál es el alcance? En términos de cómo lo escribimos en la propuesta, lo que queremos es ayudar o fomentar en los niños el desarrollo del pensamiento computacional.
EA: ¿Conoce lo que implica el pensamiento computacional?
AD: Lo he escuchado ya varias veces con Papert.
EA: Sí, con Papert, con Wing. Sabemos con qué persona.
AD: Exacto.
EA: Bueno, entonces nuestro trabajo de grado va orientado principalmente a eso.
LV: La idea de esto nació porque la verdad porque yo necesitaba una tesis. Yo no sabía con quién... no sabía absolutamente nada. Y yo le pregunté al profe. Profe, ¿qué ideas de tesis tiene por ahí?
LV: No sé si usted conoce a Nahum.
AD: Sí.
LV: Bueno, Nahum me comentó... Bueno, al profe Larez, él tiene un montón de ideas. A él le gusta trabajar con el tema del autismo, la neurodivergencia y todo eso. Y a mí ese tema también me llama mucho la atención. El tema es que, por diversas cosas, pues al final eso se terminó como yendo. Y...
EA: Pero se mantuvo la parte de trabajar con niños.
LV: Exacto. Se mantuvo la parte de trabajar con niños. Y el tema de la enseñanza, de la programación. Que es lo principal, ¿no? Entonces, nada. Después se unió Eduardo porque estábamos un poco como a... Corre, corre. A última hora haciendo la propuesta y... Nada, al final salió esto.
EA: En la parte de cómo esto se ha desarrollado, cómo hemos ido. Bueno. Ha sido un poquito... Subir y bajar, subir y bajar, subir y bajar. Y ahorita estamos volviendo a la parte de más análisis, volver a nuestras raíces. Y eso incluyó una parte que el profesor Larez nos mencionó mucho, que era qué basamento íbamos a tomar para la parte del contenido. Porque claro, al fin y al cabo queremos fomentar el desarrollo del pensamiento computacional. Pero eso implica un poquito más que programar. Y en el rango de edades que queremos hacerlo, que es de 6 a 9 años, es un poquito complicado hacerlo.
AD: Sí.
EA: Usted tiene que saber más de eso, y por eso estamos aquí. Entonces, él nos dijo, miren, guíense, o tomen guía, primero nos dijo que del MOIDI.
AD: Ok, de la profesora... Chilina León.
EA: Y aparte, también, vayanse buscando la parte de educación del país, o sea, cómo funciona el currículo. Qué se supone que tienen que aprender los niños a esas edades, básicamente de primer a tercer grado.
EA: Entonces nosotros lo que hicimos fue, buscamos el contenido, lo conseguimos con un colegio; y nos basamos principalmente en el currículo de matemáticas, porque es como que lo más cercano a la parte del pensamiento computacional, y también que era como que el área más interesante y de la que podíamos sacar más cosas.
AD: Sí.
LV: Entonces están estos contenidos de figuras y cuerpos geométricos, números naturales, los polígonos, la estadística y probabilidad, orden de serie de números, sistema monetario, y medidas convencionales y no convencionales. Que es más o menos lo que suelen ver los niños. Y una de las cosas que nos daba como curiosidad también. Era saber cómo era la experiencia de los niños con los temas de matemáticas. Tipo, ¿cómo los niños reaccionan? ¿Cómo se sienten al ver este tipo de cosas? Para saber exactamente cómo nosotros hacer nuestros juegos y nuestros problemas. Que bueno, ahorita en un momento cuando le comente el resto de la presentación...
LV: Nosotros vamos a hacer un juego. Basado en dataflow. Pero, nada.
AD: Quería saber primero eso. Que es lo que le voy a decir. Suena bien. Me están trayendo un nuevo componente de la ecuación que no sabía, dataflow. Ya había escuchado lenguaje C, byte. Ya yo casi me lo voy a ingenierar aquí, nada más en concepto.
EA: Ok. Por resumirlo un poco, porque eso también era nuevo para nosotros. Fue Larez quien lo sugirió. Entonces, generalmente la forma como percibimos la programación es por un paradigma que se llama flujo de control.
Eso se remonta a la época de los 80, los 70.
cuando se produce el primer computador.
Que tenías un área de memoria.
Un área de.
Y un área de instrucciones.
Ok.
El control flow te dice básicamente que las instrucciones toman.
Se refieren a los datos por memoria.
Y con eso ejecutas operaciones.
Ok.
El Dataflow te cambia el paradigma.
Te lo voltea.
Te dice que no.
Los datos fluyen hacia las operaciones.
Ok.
Ok.
Entiendo.
Voy a entender.
Ok.
Vale.
Y bueno.
Entonces, claro.
El profesor Lares.
La idea del Dataflow estuvo muy por detrás del profesor Lares.
Porque claro.
De nuestras primeras aproximaciones.
Fue irnos por scratch.
O algo similar a eso.
El profesor Lares lo dijo.
No descartó completamente esa posibilidad.
Y nos dijo que realmente.
Los niños no aprenden muy bien.
Con programación unidad de bloques.
Porque le estás metiendo muchos detalles de implementación.
O sea, muchos detalles de tu plan.
Muchos detalles de tu plan.
Se los podías atraer.
Se los podrías quitar encima.
Y darles algo más expresivo.
Un lenguaje más expresivo.
Y es cuando nos dio la idea del Dataflow.
Porque el Dataflow es mucho más expresivo.
Es un lenguaje que ya por su inicio.
Por su concepto como paradigma.
Es muy de alto nivel.
Porque ya te dice que las operaciones ya están definidas.
No las defines tú.
Ahorita nosotros.
Bueno.
Y bueno.
La patrónica que tienes.
Era para perseguir las operaciones básicas.
Exactamente.
¿Cuáles son nuestras operaciones más básicas?
Lo más pequeño.
Sí.
Que en este caso tenemos que son.
Suma.
Resta.
Multiplicación.
División.
Contar números.
Ordenar.
De mayor a menor.
De menor a mayor.
Comparación.
Es decir.
Si este número es más grande.
Más pequeño que este.
Es igual.
Y serie de números.
Es decir.
Por ejemplo.
Un número del 1 al 5.
1, 2, 3, 4, 5.
Y.
Estas son las operaciones básicas.
Que nosotros conseguimos.
A través de estos contenidos.
Acá.
Que son los que ven los niños.
De primero a tercer grado.
Y adicional a esas.
Estaban estas.
Que están en la parte de estadística.
Que eran.
Graficar un pictograma.
Graficar barras.
Graficar tabla.
Y esa adicional de qué es.
Era como una para.
Bueno.
El 1 es un número.
El círculo.
Es una figura.
Ok.
Ok.
Ok.
Ok.
Es como para.
Identificación.
Identificación.
Exactamente.
Entonces.
Pedagogía.
Identificación.
Exactamente.
Y bueno.
De ahí.
Nosotros también agregamos también.
Estas operaciones de conjunto.
Que son.
Más que nada.
Porque el profe Lares.
Nos lo había mencionado.
Esto.
No lo ven los niños.
En esa edad.
De forma directa.
No lo ven como.
Como conjunto.
Pero lares.
Pero el profe Lares dice.
¿Cuál fue el ejemplo que dio?
El de.
El de ordenar figuras geométricas.
El de la muchas veces.
No.
Pero el de.
El de las operaciones conjuntos.
Que él decía.
Bueno.
Al final.
Este y esto.
Es como una.
Ah.
Que.
Por ejemplo.
Él decía.
Si usted tiene.
Dos manzanas.
Este.
Y una pera.
Al final.
Eso es como una unión.
Porque.
Más o menos algo así.
Hay un ejemplo.
Que él nos dio.
Que la verdad.
Que no recuerdo muy bien.
Pero que.
Nos puede ser bastante útil.
Para.
Para esto.
Sí.
Este.
Y bueno.
Está aquí unión.
Intersección.
Diferencia.
Complemento.
Y diferencia simétrica.
Este.
Y bueno.
Ahora.
Ya una vez dicho.
Como esos elementos básicos.
De la.
De la.
De lo que vamos a hacer.
Ahora sí vamos con las características.
Y en este caso.
Primero tenemos que.
Es un sandbox.
Y cuando nos referimos a un sandbox.
Pues principalmente.
Nos referimos a que va a haber mucha libertad.
Esto es algo también bastante importante.
Para nosotros.
Porque.
Queremos saber.
Qué tanta libertad.
Le deberíamos dejar a un niño.
Ok.
Porque los niños pueden agarrar.
Y volverse locos.
E inventar un montón de cosas.
Sí.
Y.
Y al fin y al cabo.
Lo que nosotros queríamos hacer.
O queríamos proponer.
Era que el profesor.
le planteaba un problema al niño y entonces a partir de ese planteamiento el niño tenía que buscar la respuesta.
Ahorita le vamos a buscar, le vamos a buscar, no, le voy a mostrar un ejemplo.
Y bueno, principalmente como la característica más importante es que no hay ninguna solución mala como tal.
¿Por qué? Porque el profesor Lares dice que en la programación que se da en esta carrera,
el mayor problema es que las personas no piensan en cómo encontrar la solución de un problema.
Ok.
O sea, los problemas que se ven al inicio, por ejemplo, son, necesito por favor que me hagas un programa que sume dos números,
un número A, un número B y me dé un resultado C, por ejemplo.
Fundamento de programación.
Exacto. Y como la solución es tan simple, tan sencilla, la tenemos a pesar en términos de lenguaje de programación.
Exacto.
Pero eso no es pensamiento compulsional, eso no es abstraer, eso es pensar en implementación.
Ok.
Entonces, en la parte abstracta es que tú pienses en el algoritmo para sumar las dos cosas.
Que es claro que cuando tú estás con problemas más complejos, más grandes, en una aplicación web,
ahí sí es completamente importante de que no pienses en términos del lenguaje que lo implementas,
sino en términos del algoritmo, de la algoritmia, en términos del pensamiento computacional.
Pero como de primera tercera semestre nunca ves eso, cuando llegas al cuarto y te empiezan a hablar de eso, o bueno, si logras llegar en primer lugar, tristemente.
Pero cuando tú llegues, te empiezas a caer.
Sí, sí.
Porque ya los profesores empiezan, ya no te dicen, quiero que me hagas este programa, que haga esto, de forma completa y directamente caracterizada.
Con los pasos.
Exacto, con cada paso.
Sino que simplemente te dicen, bueno, ok, necesito que tengo este problema, resuélvelo.
Y ya está.
Tú tienes que pensarlo todo y no sabes cómo llegar a ese resultado.
El profe Franklin en base de edad.
Ajá.
Cuando hace la solución, cuando manda los ejercicios.
Más o menos así, sí.
Él, por ejemplo, en...
Yo no estudio informática, pero ya ya sé.
Por ejemplo, él cuando empieza en programación 1, que es cuando los chicos están comenzando,
él sí es muy específico.
Como le digo, o sea, dice, necesito otro pasito.
Pasito.
Tienes que hacer esto con esto, porque en el otro es analizar más allá.
Exacto.
Por la complejización que te lo coloca.
Exactamente.
Él nos da como una primicia, una premisa, y entonces uno analiza eso.
Universo del discurso.
Universo del discurso.
Y entonces uno analiza todo eso y uno tiene que empezar a sacar las conclusiones a partir de eso.
Este, entonces, bueno, como le comenté, pues la principal característica es la libertad.
Tanto para los profesores de crear esos problemas, como para los niños.
Ok.
Y bueno, como le menciono profesor y niño, pues ya le estamos comentando que nuestra mesa, nuestro ambiente, pues tiene esos dos roles, de profesor y de niño.
Luego tenemos la guía de diseño de actividades.
Y es que claro, nosotros aquí estamos asumiendo que el profesor sabe exactamente qué ejercicios hace y cómo hacer todo.
Pero realmente, como se supone que no lo deberían saber en un inicio, pues nosotros queríamos proponer como unos problemas de ejemplo, como unas posibles soluciones.
Sí.
¿Es correcto que hagamos eso?
Sí.
Se lo digo desde ahorita, tajantemente.
Háganlo.
Si se están planteando eso así y ustedes puedan decir, por ejemplo, en la parte de conjunciones o lo que sea, que ustedes tengan dos o tres ejemplos de capos.
Dos o tres, yo sé que puede ser muchos, porque tienen muchos elementos también.
Claro.
Yo tendría que revisar matemáticas de primero a tercer grado, ahorita, porque el ministerio sacó otra resolución ahorita iniciando el semestre.
Sí, justamente conseguimos esa, la de septiembre de 2025.
Entonces, yo voy a revisar y voy a preguntar a una maestra.
Incluso yo le puedo buscar a una maestra de primer grado.
A una del primero, no del segundo y no del tercero.
Para que también identifiquen, porque también podría ser de que...
Debe decirla porque estoy inventando mucho.
Pero, por ejemplo, primer grado sería lo primero que me mostraron de los temas.
Operaciones básicas.
Por ejemplo, en primer grado sería contar ordenar.
Serie de números, sí.
Esta parte quizás sería nada más primer grado.
Porque se supone que ya en tercer nivel de preescolar están enseñándole cosas al niño.
Identificar formas, ta, ta, ta, ta.
Pero ya esto complejiza.
Porque tienen que contar del 1 al 100.
¿Ok?
Entonces, esas actividades serían más para el primer grado.
Y así se les sería quizás más fácil hacer las ejemplificaciones.
Claro.
Suma, resta.
Eso es más para segundo grado.
Bueno, no.
Primero, segundo grado.
Y tercero sería multiplicación y división.
Y usted considera entonces que en nuestra guía tendríamos que agregar como una especie de nivel.
Como una especie como de...
Clasificación de los problemas.
Exacto.
Para primero, para segundo, para tercero.
Claro, porque si ustedes quieren volverlo también práctico para el docente.
Porque lo incluyeron.
Es un aspecto también que le queremos preguntar.
Bueno, es una pregunta de las finales.
Pero básicamente...
Ajá.
Es básicamente si usted está viendo de que el ambiente se adapta y se podría incluir dentro de un habla.
Pero claro, esto ya lo terminamos de responder cuando...
Sí, lo terminamos.
Pero lo estoy pensando desde que vi la lámina.
Dije, bueno, contar, ordenar y comparación.
Yo no lo veo ya...
Perdón, sería numérica.
Y yo no lo vería en un tercer grado.
Claro.
Porque se supone que en tercer grado ya deberían aprenderse la tabla de multiplicación.
Por ende, estas habilidades ya tendrían que estar desarrolladas.
Ojo, no es que no...
Todos desarrollan capacidades y las habilidades diferentes.
Claro.
Pero ese es el de verse.
Entiendo.
Ok.
Vale.
Pero sí, al docente sí deberían de tenerle unas guías de...
Unos ejemplos.
Algo que lo permita a él también, en esa libertad que ustedes están proponiendo,
también que ellos puedan, dentro de ese análisis, crear su propio...
Porque también tenemos otra opción también.
Era que la propia guía de diseño, por así decirlo, no sea una guía.
Sino que sea literalmente como los niveles del juego.
O sea, que esté obligatorio.
O sea, obligatorio.
No sé si me explico.
Sí.
Me gusta más.
Claro.
Pero eso resta en la parte del sandbox.
Porque ya no es un sandbox.
O sea, podría tener la sesión sandbox.
Pero no sea como que lo principal.
Lo principal sería los niveles.
Claro.
Ya los profesores no tendrían esa libertad para poder...
No.
Entonces no estamos con el sandbox.
Si me están preguntando a mí, yo tengo que ponerse a los profesores.
Vale, vale.
Y bueno.
También tenemos acá esta característica de experiencia tangible aumentada.
Un elemento...
Este...
Va a ir a la redundancia.
Uno de los elementos más importantes es que tenemos elementos persuasivos que captan el interés del niño.
Eso también es una cosa que nos interesa mucho.
De hecho, ahorita creo que las siguientes láminas justamente están.
Y elementos tangibles que se reflejan en el entorno digital.
Es decir, por ejemplo, tenemos una figura física.
Por ejemplo, la tesis de Anthony.
Que tenían las figuritas de...
Y que él ponía la figurita y los niños después la quitaban y salía el triangulito.
Exacto.
Pues exactamente esa representación es lo que también estamos buscando acá.
Esas figuritas son mías, por cierto.
¿En serio?
Las me las han devuelto.
Sí, pero ya yo sé que son ustedes.
No, pero se las podemos...
No, no, no.
Te estoy echando broma.
Pero sí son mías.
Y si necesitan más, también tengo más.
Ok.
No, ya nos va a tocar a nosotros crearlas nuevas.
Mandarlas a hacer.
Que lo que teníamos pensado, pues, es esta.
Figuritas de Foamil, como las que tenía Anthony.
Y también flashcards.
Con perritos, con animales, con comida.
Esos elementos que a los niños les podría gustar.
Acciones.
Claro, acciones también.
Y, bueno, por lo menos nosotros estamos considerando estas dos.
Porque fueron como esos elementos que los niños utilizan, que es como divertido y tal.
Pero también pensamos en plastilina.
Pensamos en Lego.
Pensamos en lápices.
Pensamos en otras cosas.
Pregunto.
¿Realmente usted considera que estos dos elementos están bien?
¿Son suficientes?
¿O podríamos agregar algo más?
Podrían agregar Legos.
Nosotros lo pensamos.
Lo pensamos.
Pero la discusión era que...
Que son muy caras.
Era Lego.
Esa es porque tenemos Lego, figuras de Foamil y plastilina.
O tacos.
Tacos.
Tacos.
Tacos.
Tacos.
Las figuritas que eran como...
Es que me suena, pero se me olvidó.
No, no.
Eso es todo.
Todos como niños hemos jugado en algún momento.
Y esto es también parte de lo que habla Chimila.
Chimila.
Con el módico.
Todos esos elementos que ustedes tienen allá.
Y esto es el módico.
Deberíamos echarle más...
Un vistazo más fuerte a eso.
Porque capaz no lo tocamos tanto.
Sí.
Deberíamos empezar a leerlo tantas.
¡Oh!
Claro, claro.
Ah, claro.
Son bloquecitos.
Exacto.
Pero hay unos que vienen más chirriquititos.
Hay unos que vienen...
Miren.
Hay de todo.
Hay unos que vienen con letras, ¿no?
También hay unos que se utilizan para el abecedario.
Y este también hay unos de números.
Eso parece interesante.
Y sería algo...
¿Están bien esas dos?
Sí.
Son las que más se utilizan a nivel de preescolar.
Primaria, perdón.
Sí.
Si le van a agregar otro elemento, sería este.
Ok.
O algo parecido a ese.
No tiene que ser este.
Pero algo parecido.
Vale.
Y...
Ah, bueno.
Y claro.
Estos son los elementos que se utilizan, pero...
Sobre todo...
Los tangibles.
Los físicos.
Claro.
Y como le mencioné anteriormente, lo que buscamos con esto es que los niños como que lo vean les guste.
Porque al final no queremos que los niños vean de nuevo como un cuaderno, un lápiz.
Que ellos vean el cuaderno, el lápiz y se quieren dormir.
Exacto.
No.
Y tampoco queremos que vean un teclado o un ratón porque es como que...
Ok.
O sea, se sienten mágico, pero después tienen interés rápidamente.
Y eso lo vimos en varios antecedentes.
Entonces, bueno.
Tenemos aquí esta parte de la interfaz gráfica de usuario.
Y esta es la contraparte.
Porque claro, somos tangibles, pero con la parte de reaumentada espacial.
Que el concepto de reaumentada espacial, a la final es que hay cosas físicas que tienen representación digital.
Exactamente.
Y eso es lo que acaba de decir Eduardo.
Exactamente.
Entonces, bueno.
Lo que acaba de decir Eduardo, los elementos tangibles tienen una contraparte digital.
Haciendo uso de las contrapartes digitales del lenguaje de programación tangible, se muestra en la interfaz gráfica la salida del programa de escrito.
Ahorita en el ejemplo, que creo que está en la siguiente lámina, la vamos a ver.
Y el conjunto de ambos elementos, físicos y digitales, son la representación visual de un programa que implementa una solución.
Exactamente.
Ahí también tenemos el concepto de programa.
Porque a la final no es el concepto de algoritmo, porque el algoritmo es el que se les ocurre a ellos en la cabeza.
Y el programa es lo que implementa con este lenguaje de programación tangible.
Esa destitución también tenemos que colocarla en la guía.
Sí.
Y entonces aquí tenemos nuestro pequeño dibujo de la mesa.
Es como una gráfica muy, muy básica.
Pero bueno, la idea del profesor Lares era tener una mesa en tipo en C.
¿Cómo se llamaba así?
Sí.
Tipo C.
Donde el profesor o la profesora está como en el medio de la mesa.
Y luego los niños juegan entre sí intentando resolverlo.
Sí, porque también ella busca que los niños colaboren, socialicen.
Ok.
Y claro, no es necesario que el niño le sobe el programa solo.
Si lo quiere sobe solo, bueno.
Y si el profesor lo ve adecuado, pero ellos pueden resolverlo en conjunto.
Exacto.
Colaborativo.
Pueden colaborar.
Aprendizaje colaborativo.
Pueden crear su algoritmo.
Exacto.
Y entonces aquí en esta parte superior está todo el elemento de la proyección.
Donde se van a ver los elementos digitales.
Y en los otros espacios, todo este espacio que estaría por acá libre, sería para que los niños coloquen los elementos.
Claro.
Porque ya hemos discutido mucho también en el área de que, claro, el hecho de que esa proyección, de que no sea tanto teléfono y tantas esas cosas,
va por la parte de que tampoco es tan bueno que los niños estén tanto rato pegados a una pantalla.
Que es verdad, que es como que por muchos años se dijo que es la mejor forma de desarrollar programación.
Sí, pero no es la única.
Y justamente tampoco es la más adecuada.
Exacto.
Y que ahorita, a nivel mundial, hay como que un celo al uso de las pantallas dentro de las aulas.
¿Sí?
Sí.
No sabía.
Sí, por la obsesión.
Yo pensé que era como, dependía del país.
Porque creo que en países como Suecia.
No, Suecia justamente se retractó de haberlo hecho.
Sí.
Exacto.
O sea, están como que diciendo, no es satanizar la tecnología o el celular, es enseñar regulación en su uso.
Entonces, chamos que, ha habido casos de investigación de chamos que han matado a su mamá porque no le compraron un iPhone, eso fue hace una semana.
O chicos que tuvieron ataques de ira porque le quitaron una tablet y estamos hablando de quinto grado.
Pero ellos no tienen ninguna funcionalidad como aprender, por ejemplo, de espacialidad.
No, es simplemente, voy a ponerlo lo más vago.
Entonces, una de las medidas que se está tomando es que si la escuela es el lugar donde más tiempo pasan y también van a seguir utilizando el celular,
entonces tiene que haber una medida de regulación.
Claro.
Entonces, no es que no están utilizando las tecnologías, pero sí tiene algo que ver con eso.
En Venezuela sí.
Bueno.
Ha estado en boga eso de las pantallas y hay colegios ya que están haciendo su protocolo de uso.
El Hoyola no puede usar el celular.
Ah, bueno, sí.
El Los Procer va por el mismo lado, el Iberoamericano va por el mismo lado.
Bueno, en mi época me acuerdo que no se podía usar el teléfono en las aulas de clase, pero sí en recreo y cosas así.
No, no.
Ellos decían que no se podía jugar con el S ni nada.
Pero bueno, lo hacía igual.
Exacto, pero ahora, por ejemplo, están creando unas casilleras de celulares.
O en Escocia, con esas cosas que utilizan para que no se moje el celular.
Ajá.
Los foros esos gigantes son foros con precintos de seguridad, como cuando tú compras ropa.
Claro.
Entonces tú llegas, cada uno tiene su nombre, tú metes allí y el profesor sella.
Ajá.
Y tú lo puedes tener, pero ¿qué vas a hacer si no puedes hacer absolutamente nada?
Claro.
O los meten en esos casilleros así.
Ajá.
Que si alguien trata de rogarlo, igual no va a poder hacerlo porque necesita los precintos de seguridad para quitarlo.
Claro.
O sea, es una forma un poco medieval, quizás.
Sí.
Un poco extremista.
Sí, pero medidas...
Bueno, hay medidas que se tienen que tomar drásticamente.
Sí, entiendo.
Pero sí, sí.
Si tuve más drásticas, requiere medidas drásticas.
Y bueno, aquí no sale la representación, pero también va a estar el Kinect, como el que tenía Anthony, va a estar arriba.
Bueno, vamos a ver exactamente cómo va a estar.
Pero bueno, en resumidas cuentas, la detección de elementos tangibles sale a mea en tu Kinect.
Ok.
Y la proyección sale mediante el proyector.
Exactamente.
Y ahora pasamos a un ejemplo, que este es el ejemplo que el profe Larry siempre nos daba.
Ok.
Cuando hablamos del Dataflow, a muy resumidas cuentas, pues tenemos una entrada, tenemos un operador y una salida.
Ok.
Entonces, el contexto del problema, por así decirlo, sería, dado una cierta cantidad de figuras, en este caso tres triángulos, obtener el elemento mayor, el más grande.
Entonces, la idea sería que el niño, en la entrada, coloque los tres triángulos de forma desordenada.
Luego se le diga, ok, necesito que me digas cuál es el mayor.
Ok.
Y el profe Larry dio como ejemplo, bueno, entonces el niño puede decir, ok, que lo ordene de mayor a menor, y este es el resultado.
Esto se va a ver de forma digital.
Ok.
Esta se va a representar de forma digital, y los operadores serían también como flashcards, como los bloquecitos, pero físicos.
Y entonces, la siguiente operación sería tomar el primer elemento de estos que están acá.
Ok.
Y aquí está.
Ya, sí, me gusta.
Sería un ejemplo de programa de parades sobre esto.
Pero como dijimos al principio, no es el único.
Puede que a otro niño se le ocurra hacer los más pasos, en menos pasos, y como dijimos, no hay una solución más.
Si lo hacen más pasos, ok, que lo logro.
Por ejemplo.
Incluso un niño con más perder te puede dar la solución de más o menos.
Claro.
Exacto.
De hecho, es gracioso, porque el profe Larry dice que no tiene que ser eficiente.
Por ejemplo, acá mismo, en este mismo ejemplo.
Aquí estoy colocando un ejemplo de ordenar de mayor a menor.
Y lo ordena primero, y así se asegura que siempre va a tener el primer elemento.
Pero aquí, si aplica este, el más grande ya está acá.
Exacto.
Entonces, técnicamente, podría tomar el primer elemento de este, y llegar a la misma reacción.
La respuesta.
Exacto.
Entonces, la cuestión no es tanto si la respuesta es eficiente, es más rápida.
No, no.
Lo importante es que el niño piense y encuentre la solución y la piense.
Es lo más importante, que la piense.
¿Saben que tienen que ser un trabajo exhaustivo? Es con las instrucciones.
Porque es en la guía de...
O sea, por ejemplo, supongamos que ustedes están dando las instrucciones al niño sobre esto.
Ok.
Entonces, a continuación se te dará, o se te facilitará.
Claro.
Tres triángulos de diferentes colores.
Ser muy específico.
Exacto.
Exacto.
¿Por qué?
Bueno, porque eso es...
Los niños trabajan por instrucción.
Si tú dejas...
Y eso quizás un poco sandbox, no es que no le dan libertad.
Ok.
Simplemente que si tú no pones algo que...
Reglas.
Maneje eso, ustedes van a tener a 20 niños volviéndose locos.
Claro.
Agarrando tacos y cosas, porque no le estás dando.
Claro.
Tú le dijiste tres objetos.
Si yo tengo tres objetos y tengo aquí tres pelotas, tres esto, tres esto, tres esto, yo voy a agarrar lo que yo quiera.
Sí.
O sea, a los niños, para reglas, instrucciones, tienen uno que ser muy específico.
Mmm.
Sobre todo, ideando qué es lo que yo quiero de ese ejercicio.
Ok.
Hay unos que pueden ser, me imagino, eso lo tienen que ver ustedes, que pueden ser quizás más libres.
Ok.
Ok.
Pero, igual, en esa libertad, necesita haber una instrucción.
Claro.
Ya sea de tiempo, de espacio, de juego, a las niñas y tal. Eso también es importante.
Claro.
Obviamente, para el concepto que ustedes me están diciendo, quizás el manejo de tiempo no es necesario.
No.
Pero, por ejemplo, que hagan...
No, no.
No, no.
Eso ya es otra cosa.
Pero, ¿me explico?
Sí, entiendo.
Sí, sí, hay que tener esa particularidad con los niños, hay que tenerlas. Y con el profesor.
Claro.
Porque si ustedes le dan la guía al profesor.
Tres objetos, no sé qué.
Sí.
Entonces, eso también va a ser propio de la guía del profesor para ser más libre, de tener más las ideas frescas. Eso también facilitaría el proceso.
Hay que pensar bastante ese tema de la libertad, ahora que lo pienso. Porque ahora que usted mencionase, no sé qué tan conveniente sea que le demos tanta libertad a los niños o tanta libertad a los profesores.
Puede ser conveniente dependiendo de cómo se vea, porque yo puedo decir que ustedes tienen un time out.
Ok.
En el juego, ¿verdad? O en la aplicación. Time out y que con todo lo que hemos visto, hagan algo.
Claro.
Inventen ustedes, niños, su propio juego sobre eso.
Eso es bueno. Eso puede ser un modo de juego así como...
El verdadero sandbox.
Sí, exacto.
Pero ya ustedes previamente, por eso es que nosotros como docentes manejamos mucho conocimiento previo.
Si yo doy la clase primero, o yo les explico con un juego a los niños sobre la parte de ordenar elementos de mayores a menores, yo después les puedo decir a ellos, y con los que vivimos podemos jugar a hacer algo con esto.
Y entonces que ellos mismos elaboran sus propias direcciones con diferentes objetos porque no tienes que ordenar de mayor a menor con el mismo elemento.
Claro.
Yo puedo utilizar todo lo que yo quiera.
mientras sean una secuencia
¿está diciendo lo correcto?
sí, como hay niños que lo pueden
hablar por colores
o por, miren
por formas
por imágenes, por lo que ustedes quieran
entonces puede ser
un time out que invite
a que ellos jueguen y que el docente
también pueda
la libertad del docente está en
en cómo utilizarlo
dónde utilizarlo, cómo estructurarlo
dentro de una planificación
claro, y de hecho nosotros también queremos que el docente
sea un regulador
que él es el que se encargue
totalmente de que si los niños están
viendo lo que tienen que ver, que los niños
apliquen los conocimientos que están viendo
lo que usted está mencionando
o sea, no lo
a mí me gusta la idea
de los amigos que me preguntan
pero también sé que dentro de
el nivel de primaria se tiene que ser muy
directriz
tiene que haber muchas directrices
imagínese esto si lo hacen en universidad
claro
ustedes necesitan decirle a sus compañeros en un cuarto semestre
que ya han visto
fundamentos y no sé qué cosas
y el montón de cálculos que ustedes ven
andar poniendo reglas
instrucciones de lo que tienen que ser
cuando lo que tienen que ser es crear
un lenguaje de no sé qué cosas
no necesito
ustedes dan los elementos y yo ve como lo ordeno
claro
pero aquí sí
bueno y este es el primer ejemplo que teníamos es como el ejemplo que siempre nos da el profesor
Y ahora tenemos otro ejemplo que es más con flashcards. Bueno, Eduardo...
Me quita las flashcards. Pero bueno, estos elementos de acá van a ser las flashcards. Tenemos un gato, un perro y un ratón.
Y el ejercicio sería aplicar unión de conjuntos.
Yo pensé que era un ovni.
¿Un ovni?
¿El ratón?
Ah, es que sí, soy un poquito de...
No, no, no.
La cara que me di cuenta es intersección.
¿Es una intersección?
Sí, es una intersección.
Ups.
Ups.
Si esto es una intersección, no tengo yo.
Es una intersección.
Ya, ahora voy a corregir esto.
Ya, es una intersección.
Ok.
Ok.
Ok.
Listo.
Ok.
Entonces, en este caso tenemos este...
Aquí la idea sería, el ejercicio sería que los niños apliquen la intersección de conjuntos.
Y en este caso la intersección de conjuntos pues consiste en...
Nosotros tenemos un conjunto de elementos.
Un conjunto de elementos uno y un conjunto de elementos dos.
Y los elementos que se repitan tanto en el primer conjunto como en el segundo.
Pues el resultado de esa intersección es eso.
Es lo que va a estar acá.
Ok.
Entonces los niños tendrían las carticas con las flashcards.
Tendrían las dos posibilidades de colocar en la entrada uno, entrada dos.
Colocan primero la flashcard del gato, el perrito y el ratón.
Y luego abajo colocan el gato, el perrito.
Aplican el operador de la intersección.
Y el resultado sería una representación digital del perro y el gato.
Que sería el resultado de esa intersección de conjuntos.
¿A esto se le puede poner sonido?
Sí.
Deberíamos.
¿Por qué no hacerlo...
Este...
Si es un perrito y un gatito.
Eso es un perrito y un gatito.
Claro, podríamos poner algún botoncito aquí que sea de play.
Porque ¿sabes qué pasaba con la tesis de Anthony?
Que lo vi mucho.
Que a veces los ruidos se repetían mucho.
Lo llegué a ver en el video y eso es algo que puede ser como chocón.
Aturde.
Sí.
No, no, porque puede ser...
No todos los niños les gustan los sonidos.
Pero podría ser algo...
Sí, es un lindo detalle que podríamos agregar.
Sí.
Algo que diga también incluso correcto.
Correcto.
Como un...
Eso es importante.
Y bueno, al final esta es como la idea base de toda la... de nuestra tesis.
Y bueno, nosotros vamos a corroborarlo con niños.
Tenemos ya los chicos de multiplayer.
Este...
No sé si usted conoce el sitio de... ¿sabe el sitio que queda por acá en Cacao?
De juegos.
Que los chicos se reunen ahí, dan cursos y trabajan aún.
De hecho.
Por eso se lo conoce.
Ajá.
Y entonces nos dieron esa posibilidad de probar también con los niños el prototipo que vayamos a hacer.
Este...
Y bueno, sí.
Esto prácticamente sería todo.
Yo te puedo conseguir también niños de colegios.
También podría ser, sí.
O sea, eso es un...
Pero yo te...
Bueno.
Puedo pedir colaboración al colegio para que ustedes lo muestren también en diferentes...
Formatos.
Me explico, formatos no.
Loyola primer grado se lo muestran.
Ajá.
Los próceres se lo muestran.
Y Ibero se lo muestran.
Ajá.
Eso también puede ser...
Sí.
Bueno, no.
Pero les podría ayudar con...
Vale.
Con esa parte también.
Incluso con fe y alegría.
Ajá.
Sería de mucha ayuda, la verdad.
Sí.
Sí.
Ah, te...
Ahora la...
Ok.
Este...
Y bueno, sí.
Este...
Bueno, nada.
Queríamos saber qué opinaba.
Que...
No me encantó.
Este...
Considera que hay alguna...
Bueno, y obviamente tenemos todo el feedback que nos acaba de dar.
Pero no sé si hay algo que le choque muy fuerte.
O hay algo que no le...
Que no le...
Que como que no le cuadre y que tengamos que cambiarlo.
No sé.
Se entendió la idea del data flow.
No, no.
Sí, sí.
Vente.
Otra información por mi cerebro.
Este...
No, a mí me gustó.
Me gusta la idea.
Me gusta el tema que escogieron.
Porque matemática es una de las cosas...
Bueno, esto va desde otra perspectiva, ¿no?
Claro.
Pero ustedes mismos lo dijeron.
Nos centramos en matemática porque es quizás los contenidos más cercanos a...
Y que después permiten hacer esto.
Y bueno, de hecho, justamente viendo los artículos que nos mencionó el profesor Lárez, al final las abstracciones matemáticas son lo que resultan en el pensamiento, en el desarrollo del pensamiento computacional.
Sí, y al final qué es...
Entonces...
El pensamiento computacional al final es ser eficaz.
Eso es lo que tengo entendido según paper.
No recuerdo bien.
Eficaz y eficiente.
Pues es que no importa lo que ustedes dijeron.
No importa cómo yo lo haga, va a ver una solución.
Es el hecho de que puedas resolver problemas.
Exacto.
De que tengas un problema grande y tú sepas que seas capaz de separarlos, dividirlos.
Y para un niño, esto es.
No es tan fácil de ver.
Claro.
Si les digo, a mí me encanta la idea.
Me gusta el proyecto.
Me gusta las ideas que tienen y los contenidos que están tomando.
Quizás ver los contenidos más avanzados.
Eso sí era lo que les dije.
Que yo preguntaría a una maestra, si ustedes quieren.
Yo puedo contactar a una maestra.
Ok.
Para que ella venga o ustedes vayan.
Puede ser lo más cercano en el Loyola.
Para que se lo muestren.
Y ella como maestra, que al final está en ejercicio,
les diga.
Quizás les puede dar otras instrucciones.
O decir, ven, ¿este contenido?
No.
Y eso también pueden avalarlo.
Cuando se lo maestran a profilar eso.
Decirle, mire, la maestra dice que este contenido en este año no es tan importante.
No es tan importante.
Incluso ellas mismas pueden venir y decir, pero no pueden meter esto.
No.
No.
Que puede ser más.
Y ustedes verán dentro de ese espectro que ustedes tienen en su objetivo, siempre.
Pero a mí me gusta.
Pero a mí me gusta.
Es una idea buena.
Muy buena.
Data flow, como ustedes.
Me gusta también la forma de la mesa.
Eso que estructuren bien.
Que aquí en este espacio va lo tangible.
Y en el centro se ve lo...
Eso me gustó mucho.
Que fue una de las cosas que yo también comenté de Anthony.
Que al final era el mismo espacio del tangible.
Si mi memoria no me falla, era el mismo espacio.
El digital.
Digital.
Claro, entonces lo...
Pero tenemos que quitarlo, pero...
Claro, más que nada lo decíamos por un tema de orden.
Sí, sí.
Porque los niños llegan a colocar todo eso en la proyección.
En la proyección del videobing y es chocante.
No, y que también tienen que pensar en el maestro.
En el profesor.
Claro.
Porque aquí lo estamos mirando con cuatro o cinco niños.
Entonces hemos llegado en la primaria y tenés 30.
Claro.
Ojo, se necesitarían varios meses.
Ojalá tuviéramos tener todas las meses.
Y que ustedes tengan un financita y les digan, mire, yo les compro esto.
Claro.
¿Qué ha pasado?
Este...
Pero sí, a mí me gusta.
Esta...
Ah, sí.
Si me preguntan a mí, sí.
Chévere.
Si ustedes quieren, me dan un número de teléfono.
O sus números de teléfono.
Sí, claro.
Para yo...
Este...
Vale.
Este...
Le doy el mismo.
Contactarles a una maestra.
Ok.
Se lo voy a poner más cerca en el Uyola.
Que no vayan a leer.
Vale.
Y bueno.
Díganme el nombre.
El...
El número.
¿0414?
Ajá.
¿385?
¿385?
1290.
1290.
Luis Vázquez.
Luis Vázquez.
Primero con S y después con C.
Perfecto.
Yo...
Denme...
Mañana viene la autoridad.
No sé si pueda tener tiempo.
Pero...
No, perfecto.
No te puedo.
No te puedo.
Yo voy a tratar de contactar a las profes antes que llegue noviembre.
O sea, obvio, te necesitan esto rápido.
No, no, no.
Pero...
Noviembre porque noviembre es donde los colegios hacen más cosas.
Que si canto de no sé qué cosas.
Que si el villancico.
Que si la fiesta.
Que si no sé.
Sí, sí, sí.
Ahora...
Suerte de ustedes si llegan a un compartir besos.
Porque las maestras son excepcionales.
Y en la parte, aunque ustedes hablaron de...
Y quizá no se centraron tanto en eso.
De la parte de inclusión.
Autismo.
Asperger.
Eso.
Bueno, al principio ustedes lo nombraron.
Sí.
Esto podría ser.
Sin ser directamente en su trabajo.
Un docente lo podría ver como inclusión educativa.
Es que, de hecho, menos mal que lo mencionar.
Porque realmente nosotros sí que teníamos pensado en incluir ese tipo de características.
La cosa es que no lo mencionamos directamente.
Porque si lo mencionamos directamente es otro condicional más.
Es otro condicional más.
Nos casamos a eso y claro.
Por eso les digo.
Es otro punto en el que el curar no puede...
Sí.
Puya, entonces.
Pero puede ser que lo utilicen.
Cada trabajo de tesis al final da conclusiones y recomendaciones.
Claro.
Ustedes pueden decir que este trabajo puede expandirse más.
Utilizando estrategias de inclusión educativa.
Basado en este modelo.
Entonces le permiten a otro grupo quizá seguir trabajando sobre el suyo.
Pero bajo ese concepto de inclusión educativa.
Pero no han terminado la tesis y ya tenemos una recomendación.
Estoy preocupado que ya tenemos varias, creo yo.
Pero sí.
Sí.
Ojo.
Esto...
Bueno, esto ya lo preguntamos.
Casi todo.
Sí.
Creo que esta.
Bueno, creo que la única que podríamos aplicar es esta.
Ahora que lo pienso.
Creo que es la única que falta.
Es la penúltima.
Está.
Sí.
Sí, creo que es.
No me puede ser.
¿Usted considera que la mente sí permite o sí fomenta el desarrollo del pensamiento computacional?
Claro. Sí. La gamificación, y esto es un tipo de gamificación, está ya comprobado que ayuda a niños a entender o les facilita entender conceptos que pueden ser muy abstractos o muy difíciles para ellos.
El juego es la mejor herramienta para primaria. Entonces, visto desde ese punto, sí, para mí sí.
Ok. Y por último, bueno la pregunta que ya le había spoileado era de si considera que este diseño que estamos proponiendo pudiese integrarse dentro de aulas de clase de primero o tercer grado. Claro. Sí.
Y eso es un aspecto que el profesor Lara, una de las últimas cosas que no mencionó realmente, es que ¿por qué los niños no aprenden bien inglés durante el primer o sexto grado? Porque el objetivo final es el proceso mismo.
Estoy aprendiendo inglés porque quiero aprender inglés. Entonces claro, si nos íbamos a ir por ese apartado de que estoy aprendiendo a pensamiento computacional porque quiero aprender a pensamiento computacional, no nos iba a servir de mucho. No iba a terminar aprendiendo. Entonces, no les den eso, sino que denle un objetivo real. Por eso es que dijimos, vamos a...
Una razón. Una razón. Una razón. Algo que los motive un poco a jugar al jueguito y que ellos a través del jueguito pues se puedan divertir y entiendan algo. Y que además, lo que están viendo aquí es algo que lo van a ver toda su vida. Toda su vida, que es la matemática.
O los contenidos que están viendo. Todavía ustedes no utilizan multiplicación y división. O sea, su vida está entre dos más dos, no sé qué, pero no es dos, es uno porque es matemática discreta, que es una cosa horrorosa.
Sí, sí, sí.
Pero me explico. Entonces, están afianzando conocimientos que a la larga de la vida, aunque uno no sea ingeniero, uno tiene que manejar, multiplicar su marketing. Que después puedan lograr desarrollo computacional, pensamiento. Sí, pero para desarrollarlo necesitan de esto. Y después puede venir alguien con otro proyecto que hable solamente del objetivo. Ah, bueno, pero vamos a utilizar a los mismos niños que utilizaron esto.
Sí, sí.
Eso pasa mucho. Porque, por ejemplo, muchos contenidos se repitían. Muchísimos de primero a tercero se repitían bastante.
¿Por qué ustedes creen que ustedes no aprenden inglés de una sola vez? Porque de primero a tercero, al quinto año, verbo, tu vi, progresivo, futuro, vi, futuro.
Veamos los mismos contenidos en bucle. Exacto. Y es repetitivo. Ya no es por conocimiento. No, sí, pero hay, por ejemplo, contenidos que uno puede ver en tercer grado, en el tercer lazo. Porque él abre bocas al cuarto grado.
Es decir, que no los estás viendo. Claro. Claro. Claro. Entiendo. Entonces, eso y mostrárselo a una maestra. Yo siempre, y se lo he dicho a él, por eso es que muchos de sus compañeros han, a Orimar.
Y en Manuel, su trabajo fue, se le buscó a unos profesores de esa área. Sí. Porque les van a dar más luces. Yo estoy hablando desde la parte pedagógica, pero no desde la parte de un salón de clase. Claro. Ahorita.
Y esa otra cosa. No, pero por lo demás. Me gusta. Me gusta su proyecto. Y espero verlo cuando termine. Para ir a jugar. Vale. Se verá. Bueno. En conclusión, esa sería nuestra revista. Perfecto. Un placer conocer. Y bueno, muchas gracias por tomar su tiempo.
Aunque, un detallito. ¿Cuál? No le preguntamos quién era usted. Específicamente. Se nos pasó, nos pasamos nosotros, pero no le preguntamos quién era usted.
Mi nombre es Andreina Dávila. Soy profesora de ciencias sociales. Y de inglés. ¿Ah, inglés? Sí. Y coordinadora de la escuela de educación y de la dirección general académica de la universidad.
¿Cuántos años tengo? 15 años. Lo que pasa es que era un poco como la presentación, pero se nos pasó. Sí, se nos pasó en los primeros cinco minutos. Y bueno, he sido profesora de práctica profesional y de otras materias aquí en la escuela. Y estamos a la vuelta. Muchas gracias.
