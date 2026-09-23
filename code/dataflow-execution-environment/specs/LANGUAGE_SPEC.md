# Especificación del Lenguaje Dataflow

**Versión:** 0.3.0 (borrador)
**Fecha:** 2026-09-22
**Estado:** Documento vivo — se actualiza a medida que la implementación revela casos borde o mejores diseños.

---

## 1. Dominio semántico

Esta sección define **qué es un valor**.

### 1.1 Panorama

Al evaluarse, un programa produce **valores**. Todo valor pertenece a una de tres formas:

1. **Bolsa** (*bag*) — una colección de objetos CPA con cantidades. Es la forma central del lenguaje: los datos.
2. **Criterio** (*criterion*) — un selector u ordenador, consumido por las operaciones de filtrado y ordenamiento.
3. **Booleano** (*boolean*) — el resultado de una comparación.

Solo la bolsa transporta datos numéricos y de currículo; el criterio y el booleano actúan como auxiliares para alterar el comportamiento o informar el resultado de ciertas operaciones.

### 1.2 La bolsa

La bolsa es la forma central del lenguaje: los datos. Se construye a partir de objetos CPA; en lo que sigue se definen su identidad, su representación, su denotación y las reglas que la gobiernan.

#### 1.2.1 Identidad CPA

Toda unidad de dato del lenguaje es un **objeto CPA**, determinado por su **identidad**: la tupla

```
Identidad = (categoría, tipo, subtipo, atributos)
```

donde cada parte cumple un papel distinto:

- **`categoría`** — el nivel de representación CPA del objeto: `concreto`, `pictórico` o `abstracto` (exactamente tres valores posibles). Distingue, por ejemplo, una manzana (concreto) de un dibujo de una manzana (pictórico) o de una cantidad de manzanas (abstracto).
- **`tipo`** — la familia o clase general del objeto (p. ej. `"comida"`, `"forma"`, `"animal"`, `"numero"`).
- **`subtipo`** — la variante específica dentro del tipo (p. ej. `"manzana"` dentro de `"comida"`, `"círculo"` dentro de `"forma"`, `"racional"` dentro de `"numero"`).
- **`atributos`** — un conjunto de pares clave–valor adicionales que refinan la identidad más allá del subtipo (p. ej. `color: "rojo"`, `tamaño: "grande"`).

Dos objetos son de la **misma identidad** si y solo si coinciden en las cuatro partes: categoría, tipo, subtipo y todos sus atributos.

#### 1.2.2 Representación

Una **bolsa** es una **secuencia finita y ordenada de entradas**. Cada **entrada** es un par

```
Entrada = (Identidad, cantidad)      con  cantidad ∈ ℚ
```

Es decir, una entrada asocia a una identidad una **cantidad racional**, de modo que puede ser natural, fraccionaria, negativa o 0. La bolsa es, en su forma concreta, una lista de tales entradas.

Tres propiedades definen el comportamiento de la bolsa:

**(a) Se permiten repetidos; no se agrega por sí sola.** Una misma identidad puede aparecer en varias entradas distintas, y la bolsa las conserva separadas. Por ejemplo, la siguiente es una bolsa válida y *no* se colapsa por sí sola:

```
{ manzana↦2, pera↦3, manzana↦4, número↦5, pera↦1 }
```

Aquí hay dos entradas de identidad "manzana" (con cantidades 2 y 4) y dos de "pera" (3 y 1), y permanecen distintas. **Solo las operaciones agregan** identidades iguales; la bolsa por sí misma es, en este sentido, una "bolsa de bolsas". Esta decisión preserva la correspondencia uno-a-uno entre cada objeto tangible colocado por el niño y cada entrada de la bolsa, hasta que una operación decida combinarlas explícitamente.

**(b) El orden se conserva.** Las entradas están ordenadas, y el orden por defecto es el de **declaración** (orden de primera aparición, de izquierda a derecha). No hay ninguna regla de ordenamiento implícita que el usuario deba recordar.

**(c) Las cantidades 0 se conservan.** Una entrada de cantidad 0, por ejemplo el resultado de `3 manzanas − 3 manzanas`, es legal y **no se descarta** de la representación: así el consumidor puede enunciar el resultado por identidad («quedan **0 manzanas**»), lo que es didácticamente valioso. Denotacionalmente, en cambio, una cantidad 0 no aporta nada y la igualdad la ignora.

Un **objeto individual** (una sola tarjeta) es, simplemente, una bolsa de una entrada. La **bolsa vacía** (sin entradas) es un valor de primera clase y se denomina `nulo`.

#### 1.2.3 Denotación: el vector en `ℚ^{(Id)}`

La bolsa es una **representación** de un **vector** en `ℚ^{(Id)}`. Se pasa de uno al otro **agregando las cantidades de las entradas de igual identidad** (colapsando los repetidos). Formalmente, la denotación es una función `δ` que lleva cada bolsa a una **función de soporte finito** de identidades en ℚ, y esa función *es* el vector:

```
bolsa  = [ (i₁,c₁), (i₁,c₂), (i₂,c₃), …, (iₙ,cₙ) ]     (lista de entradas; una identidad puede repetirse)

vector = δ(bolsa) : Identidad → ℚ,   δ(bolsa)(i) = Σₖ cₖ · [iₖ = i]     (k de 1 a n;  [iₖ = i] vale 1 si la entrada k tiene identidad i, y 0 si no)
```

Por ejemplo, la bolsa `{ manzana↦2, pera↦3, manzana↦4 }` (tres entradas) denota el vector `{ manzana↦6, pera↦4 }` (dos componentes).

Solo un número finito de identidades tiene valor distinto de cero (el *soporte*). En particular, una identidad cuya suma de cantidades es 0 queda **fuera del soporte**: las entradas de cantidad 0 no alteran la denotación.

El conjunto de todas estas funciones es el **espacio vectorial libre sobre ℚ** generado por las identidades, denotado `ℚ^{(Id)}`; sus elementos son las **combinaciones lineales formales** de identidades con coeficientes racionales. En esta lectura, **los objetos CPA son vectores**, las identidades son la base, y la cantidad de cada entrada es un coeficiente.

Este es el punto de diseño central del lenguaje: al permitir coeficientes en ℚ (y no solo en ℕ), se **fusionan en una sola noción** el "¿cuántos?" (contar objetos, ℕ) y el "¿cuánto?" (medir, fracciones y negativos, ℚ). Para un lenguaje cuyo propósito es enseñar aritmética y fracciones, que "3", "1/3" y "−1" sean el mismo tipo de ciudadano es deliberado.

La **forma reducida** de una bolsa es la que tiene exactamente una entrada por identidad de su soporte, con cantidad igual al coeficiente: es la única bolsa que **coincide** con su propio vector. La representación general no está necesariamente reducida; las operaciones son las que reducen (o no).

#### 1.2.4 Igualdad denotacional

Dos bolsas son **iguales** si y solo si tienen la **misma denotación**, es decir, el mismo vector:

```
bolsa₁ ≈ bolsa₂   ⟺   δ(bolsa₁) = δ(bolsa₂)
```

En consecuencia, la igualdad **ignora el orden**, **ignora la agrupación** e **ignora las cantidades 0**. Por ejemplo, todas estas bolsas son iguales entre sí:

```
{ manzana↦2, manzana↦4 }   ≈   { manzana↦6 }   ≈   { manzana↦1, manzana↦1, ... (seis veces) }
{ manzana↦1, pera↦1 }       ≈   { pera↦1, manzana↦1 }
{ manzana↦0 }               ≈   nulo   ≈   { manzana↦0, pera↦0 }
```

Esta es la invariante que mantiene coherente el modelo de espacio vectorial: el orden, la falta de agregación y las cantidades 0 son **detalles de representación**, no del valor.

#### 1.2.5 `nulo` y la regla `noop` global

`nulo` es la **bolsa vacía**: la que no tiene entradas. Su denotación es el **vector cero**. Es el valor que produce un nodo incompleto o ausente (una sentencia a medio escribir mientras el niño construye el programa en vivo).

De la definición se sigue una **única regla global** de propagación, sin excepciones por operación:

> **Toda operación ignora sus argumentos `nulo`** (los trata como ausentes).

Una operación cuyas entradas efectivas son todas `nulo` devuelve el elemento neutro correspondiente (p. ej., una suma vacía denota `nulo`/cero). Pedagógicamente, esto garantiza que un nodo a medio construir **no invalida** el resto del programa aguas abajo.

#### 1.2.6 Números: cantidad, escalares y aritmética exacta

Un **número** es un objeto CPA de categoría `abstracto` y tipo `numero`; su valor numérico es la **cantidad** de su entrada. Así, el número `1/3` es la bolsa `{ (abstracto, numero, racional)↦1/3 }`.

Toda la aritmética es **exacta sobre ℚ**.

Los números cumplen un **doble papel**, que se mantiene de forma deliberada:

- Como cualquier otra entrada, un número vive dentro de una bolsa y se suma con otras cantidades: es un vector en el eje de los números.
- En **ciertas operaciones**, un número puede actuar como **escalar**, escalando la cantidad de cada entrada de la bolsa (escalar × vector).

Esto se apoya en la estructura de espacio vectorial: como **el producto vector × vector no está definido** (solo escalar × vector), en esas operaciones **no se combinan dos identidades no numéricas entre sí** (“¿qué es manzana²?”); un escalar afecta a cada objeto por separado, pero el producto de dos objetos carece de sentido.

#### 1.2.7 Orden

Como se señaló previamente, la bolsa conserva el orden al momento de su declaración, y este puede ser alterado. La regla de uso del orden es simple:

> **Todas las operaciones conservan el orden**, pero solo las **operaciones de orden** lo **usan o alteran**. Para cualquier otra operación, el orden es información que se arrastra pero no se interpreta.

Las **operaciones de acceso posicional** leen entradas según el orden vigente en ese momento; por eso siempre están bien definidas: la bolsa siempre tiene un orden. Por ejemplo, tomar el primero de `{ manzana↦2, pera↦3, manzana↦4 }` da `{ manzana↦2 }`.

Como la igualdad ignora el orden, reordenar una bolsa produce un valor **igual** al original: el orden solo es observable a través de las operaciones que lo usan (las de orden y las de acceso posicional), nunca a través de las demás.

### 1.3 Criterios

Un **criterio** es un auxiliar que describe *cómo seleccionar u ordenar* objetos. Cada criterio **declara su subtipo** —de filtro o de orden—, y ese subtipo determina cómo se interpretan sus valores y qué operación lo consume:

- **Criterio de filtro** — un predicado: una conjunción de restricciones `propiedad = valor` (**Y** entre sus propiedades), cada una con un **único** valor. Sus propiedades son de **identidad** (categoría, tipo, subtipo o atributos); **no** opera sobre la cantidad. Un objeto lo satisface si cumple **todas** sus restricciones. Lo consume la operación de filtrado.
- **Criterio de orden** — una clave de ordenamiento sobre una **propiedad**, que puede ser de identidad **o la cantidad**, en una de dos formas: la propiedad con una **dirección** (`asc`/`desc`) para el orden natural (numérico para la cantidad, alfabético para textos); o la propiedad con una **secuencia de valores** que fija el orden explícitamente (p. ej. `pequeño → mediano → grande`), que puede incluso no ser ascendente ni descendente. Lo consume la operación de orden.

### 1.4 Booleano

El **booleano** (`verdadero` / `falso`) es la tercera forma de valor, con una diferencia respecto a la bolsa y el criterio: **no es declarable por el usuario**. No puede escribirse como un dato de entrada; solo lo **producen las operaciones de comparación**. Dada esta restricción, puede decirse que no es un ciudadano de primera clase del lenguaje.

Su papel es **informar el resultado de una comparación**; ninguna operación lo consume como entrada, de modo que es un valor terminal (de salida).

### 1.5 Presentación vs. semántica

El **modo de visualización** es una decisión del **consumidor** de los valores (la interfaz), y afecta **solo cómo se representan** los resultados, no qué se computa ni la identidad de los valores. El docente puede alternar entre modos libremente: un objeto conserva su **identidad semántica** intacta y solo cambia su apariencia.

Esto **no** significa que la semántica sea "agnóstica de CPA": la categoría de un objeto sí forma parte de su identidad y participa en el cómputo (p. ej., el papel de escalar de los números abstractos). La separación es entre *identidad semántica* (fija) y *representación visual* (elegida por el consumidor).

---

## 2. Modelo de evaluación

Esta sección define **qué significa evaluar un programa**: cómo se obtiene, a partir del texto de un programa, el valor de cada una de sus salidas.

### 2.1 El programa como grafo

Un programa es una secuencia de **sentencias**. Cada sentencia declara un **nodo** con un nombre único, de una de tres clases:

- **`source`** — un nodo de entrada: aporta datos (uno o varios objetos) o un criterio.
- **`transform`** — un nodo de proceso: aplica una operación a otros nodos.
- **`sink`** — un nodo de salida: expone el valor de otro nodo como resultado del programa.

Un nodo **depende** de los nodos que menciona por su nombre: un `transform` depende de los nodos que recibe como argumentos, y un `sink`, del nodo que expone. En consecuencia, solo `transform` y `sink` tienen dependencias; un `source` es entrada pura. Estas dependencias forman un **grafo dirigido**: cada nodo apunta a aquellos de los que depende.

Los `sink` son las **salidas** del programa. Evaluar un programa consiste en calcular el valor de cada `sink`.

### 2.2 Bien-formación

Un programa está **bien formado** si cumple tres condiciones, verificables antes de evaluar:

1. **Nombres únicos.** No hay dos nodos con el mismo identificador.
2. **Referencias resueltas.** Todo nombre que un nodo menciona corresponde a un nodo existente.
3. **Aciclicidad.** El grafo no tiene ciclos: ningún nodo depende, directa o indirectamente, de sí mismo.

Cada condición incumplida produce un error. La aciclicidad es la que garantiza que la evaluación **termina** y que el valor de cada nodo está bien definido.

### 2.3 El proceso de evaluación

La evaluación es **dirigida por demanda**: parte de los `sink` y "tira" hacia atrás de las dependencias, evaluando primero las entradas de cada nodo.

A continuación se describe el proceso de evaluación de un programa.

> **Nota.** Siguiendo la convención de especificaciones como ECMAScript, cada procedimiento se describe como una **operación abstracta** con nombre y una lista de **pasos numerados**.

#### 2.3.1 Evaluar el programa

**EvaluarPrograma(programa) → (valores, errores)**

1. Sean `valores` un mapa vacío y `errores` una lista vacía.
2. Para cada `sink` `s` del programa:
   1. Intentar `v ← EvaluarNodo(s)`.
   2. Si tiene éxito, asociar `s ↦ v` en `valores`.
   3. Si la evaluación produce un error, agregarlo a `errores` y continuar con el siguiente `sink`.
3. Devolver `(valores, errores)`.

Cada `sink` se evalúa de forma **aislada**: un error en uno no impide obtener el valor de los demás. Un mismo programa puede producir, a la vez, valores y errores.

#### 2.3.2 Evaluar un nodo

**EvaluarNodo(id) → valor**

1. Si `id` ya tiene un valor calculado, devolverlo. (Cada nodo se evalúa **una sola vez**; su valor se reutiliza.)
2. Sea `nodo` el nodo con identificador `id`.
3. Para cada dependencia `d` de `nodo`, sea `Vd ← EvaluarNodo(d)`. (Las entradas se resuelven antes que el nodo.)
4. Sea `v ← EvaluarSentencia(nodo, { d ↦ Vd })`.
5. Registrar `v` como el valor de `id` y devolverlo.

Como el grafo es acíclico, este procedimiento siempre termina: la cadena de dependencias no puede volver sobre un nodo ya en curso.

#### 2.3.3 Evaluar una sentencia

**EvaluarSentencia(nodo, entradas) → valor**, según la clase del nodo:

- **`source`:** aporta su valor directamente, no lo calcula a partir de otros nodos.
  1. Si está incompleto (sin valor), devolver `nulo`.
  2. Si declara criterios, devolver el **criterio** (o la **bolsa de criterios**) que declare.
  3. Si declara datos, devolver la **bolsa** que los reúne: una entrada por cada objeto CPA que declare.
- **`transform`:**
  1. Si está incompleto (sin operación), devolver `nulo`.
  2. Tomar de `entradas` los valores de sus argumentos y aplicar la operación a esa lista; el resultado es el valor del nodo.
- **`sink`:**
  1. Si está incompleto (sin fuente), devolver `nulo`.
  2. Devolver el valor de su fuente, tomado de `entradas`.

### 2.4 Determinismo y orden de evaluación

El valor de un nodo depende **únicamente** de su sentencia y de los valores de sus dependencias: no hay estado mutable ni efectos secundarios. En consecuencia, el resultado de evaluar un programa es **determinista** y **no depende del orden** en que se evalúen los nodos. Cualquier estrategia que respete las dependencias (evaluar un nodo solo después que sus entradas) produce los mismos valores; esto habilita, por ejemplo, evaluar en paralelo las dependencias independientes de un nodo, o reutilizar valores ya calculados entre evaluaciones sucesivas del mismo programa.

Además, solo los nodos de los que depende algún `sink` participan en el resultado. Dicho de otra manera, si hay `source`s o `transform`s que ningún `sink` alcanza, no entran en el proceso de evaluación del programa y, por tanto, no se evalúan.

### 2.5 Programas parciales

Mientras el usuario construye un programa, es normal que haya nodos **incompletos** (una sentencia a medio escribir). El modelo los admite sin detenerse: un nodo incompleto evalúa a `nulo` y, como toda operación ignora sus argumentos `nulo`, el resto del programa se sigue evaluando. Un nodo a medio construir no invalida a los demás; simplemente aún no aporta nada.

---

## 3. Operaciones

Esta sección define las **operaciones**: los cómputos que un `transform` puede aplicar. Cada operación se describe con una **ficha** de la misma forma:

- **Firma** — nombre, aridad y tipos de entrada → tipo de salida.
- **Resumen** — qué hace, en una línea.
- **Pasos** — el cómputo como operación abstracta, en pasos numerados.
- **Errores** — las condiciones propias de la operación que producen error.
- **Ejemplos**.

Convenciones comunes a todas las operaciones (no se repiten en cada ficha):

- **Ignoran `nulo`**: un argumento `nulo` se trata como ausente.
- **Conservan el orden** de las entradas; solo la operación de orden lo altera.
- **Agrupación (bolsa vs vector).** Como una bolsa admite entradas repetidas de la misma identidad, cada operación indica si **agrupa** (colapsa los repetidos por identidad antes de actuar) o trabaja **entrada por entrada**. Las cantidades 0 se conservan siempre en el resultado. Cuando agrupar o no da el mismo vector, la elección es indistinta y la ficha lo señala (se prefiere entrada por entrada).
- **Las entradas abstractas no se agrupan al ordenar ni al seleccionar.** Las operaciones que agrupan para **ordenar o seleccionar** (`less_than`, `greater_than` y `order`) dejan fuera de esa agrupación las entradas de categoría `abstracto`: cada una se ordena o se compara por separado. La razón es que un número es, casi siempre, una unidad que el usuario colocó para ordenarla o compararla con otras, y colapsar `{ número↦7, número↦2, número↦5 }` en `{ número↦14 }` dejaría sin nada que ordenar justo en el caso más común. Las entradas `concreto` y `pictórico` sí se agrupan, porque ahí los repetidos de una misma identidad son el mismo objeto contado varias veces. La **aritmética** (`sum`, `substract`) agrupa todo, sin excepción: para eso está.
- La **Firma** indica cuántos argumentos admite cada operación; pasar un número de argumentos que no corresponde es un **error de aridad**.
- La **Firma** indica el tipo de cada argumento; pasar un argumento de otro tipo (una bolsa donde se espera un criterio, o al revés) es un **error de tipo**.

### 3.1 Aritmética

#### 3.1.1 `sum` — suma

**Firma:** `sum(bolsa, …) → bolsa` — variádica (una o más entradas).

**Resumen.** Reúne todas sus entradas y las **agrega por identidad**, sumando las cantidades. Es la suma de vectores.

**Pasos** (`sum(args) → valor`):

1. Reunir en una sola bolsa las entradas de todos los argumentos, descartando los `nulo`.
2. Agrupar las entradas por identidad y sumar sus cantidades.
3. Devolver la bolsa resultante: una entrada por identidad con la suma de sus cantidades. Una identidad cuya suma sea 0 se conserva como entrada de cantidad 0.

**Errores.** Ninguno propio; si no hay entradas efectivas, el resultado es `nulo` (suma vacía = vector cero).

**Ejemplos:**

```
sum({ manzana↦2 }, { manzana↦3 })            = { manzana↦5 }
sum({ manzana↦2, pera↦1 }, { manzana↦4 })    = { manzana↦6, pera↦1 }
sum({ manzana↦2 }, { manzana↦-2 })           = { manzana↦0 }
sum({ manzana↦0 }, { pera↦3 })               = { manzana↦0, pera↦3 }
sum({ número↦2 }, { número↦3 })               = { número↦5 }
sum({ manzana↦2 }, nulo)                      = { manzana↦2 }
```

#### 3.1.2 `substract` — resta

**Firma:** `substract(bolsa, bolsa) → bolsa` — binaria (exactamente dos entradas).

**Resumen.** Resta, por identidad, las cantidades de la segunda bolsa a las de la primera. Es la resta de vectores.

**Pasos** (`substract(a, b) → valor`):

1. Agrupar por identidad las cantidades de `a` y, por separado, las de `b`.
2. Para cada identidad presente en `a` o en `b`, calcular: (cantidad en `a`) − (cantidad en `b`).
3. Devolver la bolsa con una entrada por identidad. Las identidades presentes solo en `b` quedan con cantidad negativa; las que resulten 0 se conservan.

**Errores.** Ninguno propio.

**Ejemplos:**

```
substract({ manzana↦5 }, { manzana↦2 })          = { manzana↦3 }
substract({ manzana↦2 }, { manzana↦5 })          = { manzana↦-3 }
substract({ manzana↦3, pera↦2 }, { manzana↦1 })  = { manzana↦2, pera↦2 }
substract({ manzana↦2 }, { manzana↦2 })          = { manzana↦0 }
substract({ manzana↦1 }, { pera↦2 })             = { manzana↦1, pera↦-2 }
```

#### 3.1.3 `multiply` — multiplicación

**Firma:** `multiply(bolsa, número) → bolsa` — binaria. El primer argumento es la bolsa a escalar; el segundo, un **número** que actúa como **escalar**.

**Resumen.** Escala la bolsa: multiplica la cantidad de cada una de sus entradas por el escalar (escalar × vector).

**Pasos** (`multiply(a, k) → valor`):

1. Sea `s` el valor del número `k` (el escalar).
2. Multiplicar por `s` la cantidad de cada entrada de `a`.
3. Devolver la bolsa resultante.

**Nota.** La **posición** desambigua el papel del número: el segundo argumento siempre se interpreta como escalar, no como un objeto CPA. Opera **entrada por entrada** y conserva los repetidos; agrupar primero daría el mismo vector (el escalado distribuye), así que la elección es indistinta.

**Errores.** Ninguno propio.

**Ejemplos:**

```
multiply({ manzana↦2 }, { número↦3 })            = { manzana↦6 }
multiply({ manzana↦2, pera↦5 }, { número↦10 })   = { manzana↦20, pera↦50 }
multiply({ manzana↦2, manzana↦3 }, { número↦4 }) = { manzana↦8, manzana↦12 }
multiply({ número↦2 }, { número↦3 })             = { número↦6 }
multiply({ manzana↦2 }, { número↦1/2 })          = { manzana↦1 }
```

#### 3.1.4 `divide` — división

**Firma:** `divide(bolsa, número) → bolsa` — binaria. El primer argumento es la bolsa; el segundo, un **número** que actúa como **divisor**.

**Resumen.** Divide la bolsa: divide la cantidad de cada una de sus entradas entre el divisor (escalar⁻¹ × vector).

**Pasos** (`divide(a, k) → valor`):

1. Sea `d` el valor del número `k` (el divisor).
2. Si `d = 0`, es un error (división por cero).
3. Dividir por `d` la cantidad de cada entrada de `a`.
4. Devolver la bolsa resultante.

**Nota.** Como `multiply`, opera **entrada por entrada** y conserva los repetidos; agrupar primero daría el mismo vector.

**Errores.** División por cero: si el divisor es 0.

**Ejemplos:**

```
divide({ manzana↦6 }, { número↦2 })              = { manzana↦3 }
divide({ manzana↦6, pera↦4 }, { número↦2 })      = { manzana↦3, pera↦2 }
divide({ manzana↦1 }, { número↦3 })              = { manzana↦1/3 }
```

### 3.2 Comparación

#### 3.2.1 `less_than` — menor que

**Firma:** `less_than(bolsa, número) → bolsa` — binaria. El segundo argumento es el **umbral** (un número).

**Resumen.** Conserva las identidades cuya cantidad **total** es **menor** que el umbral.

**Pasos** (`less_than(a, k) → valor`):

1. Sea `u` el valor del número `k` (el umbral).
2. **Agrupar `a` por identidad** (sumar los repetidos), de modo que cada identidad tenga una cantidad total. Las entradas **abstractas** no se agrupan: cada una conserva su cantidad.
3. Conservar las entradas cuya cantidad total sea menor que `u`; descartar las demás.
4. Devolver la bolsa con las entradas conservadas.

**Nota.** **Agrupa por identidad** antes de comparar, para que el resultado dependa solo del vector: dos bolsas que denotan lo mismo (`{ manzana↦2, manzana↦3 }` y `{ manzana↦5 }`) se comparan igual. Lo abstracto es la excepción, de modo que `less_than({ número↦7, número↦2, número↦5 }, { número↦4 })` da `{ número↦2 }` y no `nulo`.

**Errores.** Ninguno propio.

**Ejemplos:**

```
less_than({ manzana↦2, pera↦5 }, { número↦5 })   = { manzana↦2 }
less_than({ manzana↦2, manzana↦3 }, { número↦4 }) = nulo
less_than({ pera↦5 }, { número↦2 })              = nulo
```

#### 3.2.2 `greater_than` — mayor que

**Firma:** `greater_than(bolsa, número) → bolsa` — binaria. El segundo argumento es el **umbral** (un número).

**Resumen.** Conserva las identidades cuya cantidad **total** es **mayor** que el umbral.

**Pasos** (`greater_than(a, k) → valor`):

1. Sea `u` el valor del número `k` (el umbral).
2. **Agrupar `a` por identidad** (sumar los repetidos), salvo las entradas **abstractas**, que no se agrupan.
3. Conservar las entradas cuya cantidad total sea mayor que `u`; descartar las demás.
4. Devolver la bolsa con las entradas conservadas.

**Nota.** Como `less_than`, **agrupa por identidad** antes de comparar (resultado bien definido sobre el vector), con la misma excepción para lo abstracto.

**Errores.** Ninguno propio.

**Ejemplos:**

```
greater_than({ manzana↦2, pera↦5 }, { número↦3 })    = { pera↦5 }
greater_than({ manzana↦2, manzana↦3 }, { número↦4 }) = { manzana↦5 }
greater_than({ manzana↦2 }, { número↦5 })            = nulo
```

#### 3.2.3 `compare` — igualdad

**Firma:** `compare(bolsa, bolsa) → booleano` — binaria.

**Resumen.** Devuelve `verdadero` si ambas bolsas **denotan el mismo vector**; `falso` en caso contrario.

**Pasos** (`compare(a, b) → valor`):

1. Comparar las denotaciones (los vectores) de `a` y `b`.
2. Devolver el booleano `verdadero` si son iguales, `falso` si no.

**Nota.** Es la igualdad denotacional del dominio: ignora el orden, la agrupación y las cantidades 0. Por eso `{ manzana↦1, manzana↦2 }` y `{ manzana↦3 }` se comparan como iguales.

**Errores.** Ninguno propio.

**Ejemplos:**

```
compare({ manzana↦3 }, { manzana↦1, manzana↦2 })       = verdadero
compare({ manzana↦2, pera↦1 }, { pera↦1, manzana↦2 })  = verdadero
compare({ manzana↦0 }, nulo)                           = verdadero
compare({ manzana↦2 }, { manzana↦3 })                  = falso
```

### 3.3 Orden

#### 3.3.1 `order` — ordenar

**Firma:** `order(bolsa, criterio, …) → bolsa` — el primer argumento es la bolsa; los siguientes, uno o más **criterios de orden**.

**Resumen.** Devuelve la bolsa con sus entradas reordenadas según los criterios. Cada criterio lleva consigo su propio orden.

**Pasos** (`order(a, criterios…) → valor`):

1. Descartar los criterios incompletos. Si no queda ninguno, devolver `a` sin cambios.
2. **Agrupar `a` por identidad** (colapsar los repetidos), salvo las entradas **abstractas**, que se ordenan una por una.
3. Ordenar las entradas aplicando los criterios: el **primero** manda y los siguientes desempatan, en orden.
4. Devolver la bolsa reordenada.

**Nota.** **Agrupa por identidad** antes de ordenar: los repetidos de una misma identidad se combinan, y luego se ordenan las identidades distintas. Lo abstracto queda fuera de esa agrupación, porque si no, ordenar `{ número↦7, número↦2, número↦5 }` devolvería `{ número↦14 }` y no habría nada que ordenar. Un criterio de orden puede usar la **cantidad** como propiedad (a diferencia del criterio de filtro).

**Formas de un criterio de orden:**

- **Por orden natural** — una **propiedad** (la cantidad, o un texto: categoría, tipo, subtipo o atributo) más una **dirección** `asc` o `desc`. `order` conoce el orden natural: numérico para la cantidad, alfabético para los textos.
- **Por secuencia** — una **propiedad** más una **secuencia de valores** que fija el orden explícitamente (las entradas cuyo valor no aparezca van al final). La secuencia *es* el orden, así que no lleva `asc`/`desc`; puede incluso no ser ascendente ni descendente (p. ej. `mediano → pequeño → grande`).

El orden es **estable**: ante un empate, se conserva el orden previo de las entradas.

**Errores.** Ninguno propio.

**Ejemplos:**

```
order({ manzana↦3, pera↦1, uva↦2 }, criterio(cantidad, asc))
    = { pera↦1, uva↦2, manzana↦3 }

order({ manzana↦3, pera↦1, uva↦2 }, criterio(cantidad, desc))
    = { manzana↦3, uva↦2, pera↦1 }

order({ número↦7, número↦2, número↦5 }, criterio(cantidad, asc))
    = { número↦2, número↦5, número↦7 }     (lo abstracto no se agrupa)

order({ estrella(grande)↦1, estrella(pequeña)↦1, estrella(mediana)↦1 },
      criterio(tamaño = [pequeña, mediana, grande]))
    = { estrella(pequeña)↦1, estrella(mediana)↦1, estrella(grande)↦1 }
```

### 3.4 Filtrado

#### 3.4.1 `filter` — filtrar

**Firma:** `filter(bolsa, criterio, …) → bolsa` — el primer argumento es la bolsa; los siguientes, uno o más **criterios de filtro**.

**Resumen.** Conserva las entradas de la bolsa que satisfacen **alguno** de los criterios; descarta las demás.

**Pasos** (`filter(a, criterios…) → valor`):

1. Descartar los criterios incompletos (los que no fijan valores para sus propiedades). Si no queda ninguno, devolver `a` sin cambios.
2. Conservar cada entrada de `a` que **satisfaga al menos uno** de los criterios; descartar las demás.
3. Devolver la bolsa con las entradas conservadas.

**Cuándo una entrada satisface un criterio.** Cada criterio es una conjunción de restricciones `propiedad = valor`. La entrada lo satisface si **cumple todas** sus restricciones (**Y** entre propiedades): para cada una, el valor de esa propiedad en la entrada es igual al valor pedido. Entre criterios distintos hay **O**: a la entrada le basta con satisfacer uno. Así, el conjunto de criterios es una disyunción de conjunciones (forma normal disyuntiva), que expresa cualquier predicado.

**Nota.** El criterio de filtro prueba la **identidad** (categoría, tipo, subtipo o atributos), **no** la cantidad. Por eso `filter` trabaja **entrada por entrada** y conserva los repetidos: los de una misma identidad pasan o se descartan todos juntos, y agrupar daría el mismo vector.

**Errores.** Ninguno propio.

**Ejemplos:**

```
filter({ manzana↦2, pera↦3, uva↦1 }, criterio(tipo = manzana))
    = { manzana↦2 }

filter({ manzana↦2, pera↦3, uva↦1 }, criterio(tipo = manzana), criterio(tipo = uva))
    = { manzana↦2, uva↦1 }

filter({ estrella(roja)↦2, estrella(azul)↦1, círculo(roja)↦3 },
       criterio(tipo = estrella, color = roja), criterio(tipo = estrella, color = azul))
    = { estrella(roja)↦2, estrella(azul)↦1 }
```

### 3.5 Acceso

Las operaciones de acceso leen el **orden actual** de la bolsa; por eso suelen combinarse con una operación de orden previa. Trabajan **entrada por entrada** (no agrupan): sobre una bolsa con repetidos de una misma identidad, seleccionan una entrada individual, no su total.

#### 3.5.1 `first` — primera

**Firma:** `first(bolsa) → bolsa` — unaria.

**Resumen.** Devuelve la primera entrada de la bolsa, según su orden actual.

**Pasos** (`first(a) → valor`):

1. Si `a` no tiene entradas, devolver `nulo`.
2. Devolver una bolsa con la primera entrada de `a` (la de la posición inicial).

**Errores.** Ninguno propio.

**Ejemplos:**

```
first({ manzana↦2, pera↦3, uva↦1 })   = { manzana↦2 }
first({ manzana↦2, manzana↦3 })       = { manzana↦2 }   (la primera pila, no el total)
first(nulo)                           = nulo
```

#### 3.5.2 `last` — última

**Firma:** `last(bolsa) → bolsa` — unaria.

**Resumen.** Devuelve la última entrada de la bolsa, según su orden actual.

**Pasos** (`last(a) → valor`):

1. Si `a` no tiene entradas, devolver `nulo`.
2. Devolver una bolsa con la última entrada de `a` (la de la posición final).

**Errores.** Ninguno propio.

**Ejemplos:**

```
last({ manzana↦2, pera↦3, uva↦1 })    = { uva↦1 }
last({ manzana↦2, manzana↦3 })        = { manzana↦3 }   (la última pila, no el total)
last(nulo)                            = nulo
```

### 3.6 Agregación

#### 3.6.1 `count` — contar

**Firma:** `count(bolsa) → número` — unaria. El resultado es un número (una bolsa con una única entrada numérica).

**Resumen.** Cuenta cuántos objetos hay en total: suma las cantidades de todas las entradas de la bolsa.

**Pasos** (`count(a) → valor`):

1. Sumar las cantidades de todas las entradas de `a`.
2. Devolver el número igual a esa suma.

**Nota.** Totaliza sin importar la identidad: no agrupa ni distingue por tipo, solo suma cantidades. Sobre una bolsa vacía da 0. Como el resultado es un número, puede alimentar a operaciones que esperan uno (por ejemplo, como escalar en `multiply` o como umbral en `less_than`).

**Errores.** Ninguno propio.

**Ejemplos:**

```
count({ manzana↦2, pera↦3 })          = { número↦5 }
count({ manzana↦2, manzana↦4 })       = { número↦6 }
count({ manzana↦1/2, manzana↦1/2 })   = { número↦1 }
count(nulo)                           = { número↦0 }
```

---

## 4. Errores

Un **error** es una condición que impide producir un valor. Cada error informa su **naturaleza** (qué salió mal) y el **nodo donde ocurrió** (el que se estaba procesando). Cuando la causa está en otro nodo —típicamente una de sus dependencias— informa además **qué nodo la causó** (coincide con el anterior si la falla es local). Y si surgió durante la evaluación, informa la **salida** (el `sink`) en cuyo cálculo bajo demanda apareció. Así todo error queda situado: qué pasó, dónde, por causa de qué y para qué salida.

Los errores se distinguen por el momento en que se detectan: los **errores de sintaxis**, al leer el texto del programa; los **errores estáticos**, sobre la estructura ya construida, antes de evaluar; y los **errores de ejecución**, al evaluar un nodo.

### 4.1 Errores de sintaxis

Se detectan al analizar el texto del programa contra la gramática (al final del documento). La gramática define qué es un programa sintácticamente bien formado; cualquier texto que no se ajuste a ella produce un error de sintaxis, que el analizador reporta con su posición. No se enumeran uno por uno: la gramática es su especificación. Dos comportamientos sí merecen mención explícita:

- Los **nodos incompletos se toleran**: una sentencia a medio escribir (un `source` sin valor, un `transform` sin operación, un `sink` sin fuente) se analiza como un nodo placeholder que evalúa a `nulo`, en vez de detener el análisis.
- Los **grupos son solo de datos**: agrupar entre corchetes reúne objetos de datos; los criterios no se agrupan (cada criterio va en su propio `source`). Un grupo que incluya un criterio no se ajusta a la gramática.

### 4.2 Errores estáticos

Se detectan sobre la estructura del programa ya construida, sin evaluar, y solo sobre los nodos que **alcanzan alguna salida**: los nodos que ningún `sink` alcanza no participan en la evaluación, de modo que tampoco se validan, y una sentencia todavía sin conectar no invalida nada. Un error estático **invalida el programa completo**: no llega a evaluarse ningún nodo.

1. **Nombre duplicado** — dos nodos declaran el mismo nombre. Los nombres deben ser únicos entre los nodos que alcanzan alguna salida.
2. **Referencia sin resolver** — un nodo menciona un nombre que ningún nodo declara.
3. **Ciclo** — las dependencias entre nodos forman un ciclo. El grafo de dependencias debe ser acíclico.
4. **Operación desconocida** — un `transform` nombra una operación que no pertenece al conjunto reconocido.
5. **Error de aridad** — una operación recibe un número de argumentos que su firma no admite. El número de argumentos de un `transform` está fijo en la estructura, así que se conoce sin evaluar.
6. **Categoría de valor equivocada** — una operación recibe un argumento de una categoría que no admite: una bolsa donde espera un criterio o al revés, o un booleano donde no corresponde. La categoría de salida de cada nodo está fijada por su operación, de modo que este desajuste también se conoce sin evaluar. Distinguir si una bolsa es además un número depende del valor y se comprueba al evaluar.
7. **Criterio inadecuado** — una operación recibe un criterio del **subtipo** equivocado (un criterio de orden donde se espera uno de filtro, o al revés), o un criterio de filtro con una propiedad de **valor múltiple** o sobre la **cantidad**. El subtipo va declarado en el criterio, así que se detecta sin evaluar.
8. **Objeto inválido** — un `source` declara un objeto con un componente de identidad CPA en blanco (categoría, tipo o subtipo vacío): es sintácticamente válido, pero no denota una identidad real. No es un caso de `nulo`: el único caso parcial que da `nulo` es un nodo sin cablear (un `source` sin valor, un `transform` sin operación o un `sink` sin fuente).

### 4.3 Errores de ejecución

Surgen al evaluar un nodo, porque dependen de los valores calculados. Están **aislados por salida**: un error al evaluar un nodo afecta solo a las salidas que dependen de él; las demás salidas producen su valor con normalidad.

1. **Número esperado** — una operación que necesita un número (el escalar de `multiply` y `divide`, el umbral de `less_than` y `greater_than`) recibe una bolsa que, al calcularse, no resulta ser un número. Que un argumento sea una bolsa se conoce sin evaluar, pero que esa bolsa sea un número solo se sabe con su valor.
2. **División por cero** — `divide` recibe el divisor 0.

---

## 5. Gramática

Esta sección fija la **sintaxis concreta**: la forma textual de un programa. La estructura abstracta —programa, sentencia, nodo, las tres clases `source`/`transform`/`sink`— ya se describió en el modelo de evaluación; aquí se da su forma escrita.

**Notación:** forma extendida de Backus-Naur (EBNF) del W3C.

### 5.1 Gramática completa

```ebnf
program             ::= statement*
statement           ::= source_decl | transform_decl | sink_decl

source_decl         ::= "source" identifier "=" (object_literal | group)? ";"
transform_decl      ::= "transform" identifier "=" (operation "(" argument_list? ")")? ";"
sink_decl           ::= "sink" identifier "=" identifier? ";"

argument_list       ::= identifier ("," identifier)*

operation           ::= identifier

group               ::= "[" (data_literal ("," data_literal)*)? "]"

object_literal      ::= data_literal | criteria_literal

data_literal        ::= "{" '"sourceType"' ":" '"data"' "," '"category"' ":" category_type "," '"type"' ":" string_literal "," '"subtype"' ":" string_literal "," '"quantity"' ":" rational_literal ("," kv_pair)* "}"
criteria_literal    ::= "{" '"sourceType"' ":" criteria_kind "," '"properties"' ":" array_literal ("," kv_pair)* "}"
criteria_kind       ::= '"filter"' | '"order"'

category_type       ::= '"abstracto"' | '"pictorico"' | '"concreto"'

kv_pair             ::= string_literal ":" kv_value
kv_value            ::= string_literal | rational_literal | array_literal

array_literal       ::= "[" (string_literal ("," string_literal)*)? "]"
rational_literal    ::= "-"? digit+ ( "/" digit+ | "." digit+ )?
string_literal      ::= '"' [a-zA-Z0-9_-]* '"'
identifier          ::= [a-zA-Z][a-zA-Z0-9_-]*
digit               ::= [0-9]
```

Notas sobre la gramática:

- **Literal racional.** `rational_literal` admite un entero (`3`), una fracción (`1/3`) o un decimal (`2.5`), con signo opcional; todo se interpreta como un racional exacto (un decimal es su valor exacto, no una aproximación). Es lo que ocupa la `quantity` de un objeto.
- **Operación.** `operation` es un identificador; el conjunto de operaciones reconocidas se lista abajo. Un identificador de operación fuera de ese conjunto es un error estático (operación desconocida).
- **Grupos solo de datos.** Un `group` reúne objetos de datos; los criterios no se agrupan (cada criterio va en su propio `source`).
- **Subtipo de criterio.** Un `criteria_literal` declara su subtipo en `sourceType` (`"filter"` u `"order"`). La gramática no restringe la forma de sus valores (pueden ser únicos o un arreglo), pero cada subtipo admite solo ciertas formas —filtro: un valor único por propiedad, sobre identidad; orden: dirección `asc`/`desc` o una secuencia—. Usar la forma equivocada, o pasar un criterio del subtipo equivocado a una operación, es un error estático (criterio inadecuado).
- **Nodos incompletos.** Las tres declaraciones tienen su valor **opcional** (`?`): un `source` sin valor, un `transform` sin operación o un `sink` sin fuente son sintácticamente válidos y evalúan a `nulo`.

### 5.2 Palabras clave y valores reservados

**Palabras clave de sentencia:** `source`, `transform`, `sink`.

**Operaciones reconocidas:** `sum`, `substract`, `multiply`, `divide`, `less_than`, `greater_than`, `compare`, `order`, `filter`, `first`, `last`, `count`.

**Valores de categoría** (los únicos admitidos por `category_type`): `"abstracto"`, `"pictorico"`, `"concreto"`.

### 5.3 Ejemplos

Aritmética racional exacta (dos números comparten identidad y se suman):

```erae
source half = {
  "sourceType": "data",
  "category": "abstracto",
  "type": "numero",
  "subtype": "racional",
  "quantity": 1/2
};

source third = {
  "sourceType": "data",
  "category": "abstracto",
  "type": "numero",
  "subtype": "racional",
  "quantity": 1/3
};

transform total = sum(half, third);   // = 5/6, exacto
sink output = total;
```

Taxonomía dinámica y atributos como parte de la identidad:

```erae
source sedan = {
  "sourceType": "data",
  "category": "concreto",
  "type": "vehicle",
  "subtype": "car",
  "quantity": 2,
  "doors": "4"
};

source coupe = {
  "sourceType": "data",
  "category": "concreto",
  "type": "vehicle",
  "subtype": "car",
  "quantity": 1,
  "doors": "2"
};

// sedan y coupe comparten categoría, tipo y subtipo, pero difieren en "doors",
// que forma parte de la identidad → sum NO los agrupa; quedan como dos entradas.
transform vehicles = sum(sedan, coupe);
sink output = vehicles;
```

Nodos incompletos (se toleran y evalúan a `nulo`):

```erae
// Operación incompleta: se tolera como placeholder
transform incomplete_calc = ;

sink active_output = incomplete_calc;   // su valor es `nulo`
```

Escalado (un número actúa como escalar):

```erae
source large_star = {
  "sourceType": "data",
  "category": "pictorico",
  "type": "shape",
  "subtype": "star",
  "quantity": 2.5,
  "size": "large"
};

source scale_factor = {
  "sourceType": "data",
  "category": "abstracto",
  "type": "numero",
  "subtype": "racional",
  "quantity": 3
};

// multiply(bolsa, número) → estrella con cantidad 7.5
transform scaled_stars = multiply(large_star, scale_factor);
sink final_render = scaled_stars;
```

Criterios (cada uno declara su subtipo; se pasan como `source` separados):

```erae
source fruits = [
  { "sourceType": "data", "category": "concreto", "type": "food", "subtype": "apple", "quantity": 3 },
  { "sourceType": "data", "category": "concreto", "type": "food", "subtype": "pear", "quantity": 1 }
];

// Criterio de filtro: valor único, sobre una propiedad de identidad
source only_apples = {
  "sourceType": "filter",
  "properties": ["subtype"],
  "subtype": "apple"
};

// Criterio de orden: dirección sobre la cantidad
source by_qty = {
  "sourceType": "order",
  "properties": ["quantity"],
  "quantity": "asc"
};

transform apples = filter(fruits, only_apples);
transform sorted = order(fruits, by_qty);
sink out_apples = apples;
sink out_sorted = sorted;
```
