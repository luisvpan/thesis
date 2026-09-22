# Sintaxis compacta (propuesta) — Lenguaje Dataflow

**Versión:** 0.1.0 (borrador)
**Fecha:** 2026-09-16
**Estado:** **Propuesta futura.** Sintaxis compacta y propia del lenguaje, alternativa a la **forma textual canónica** definida en `LANGUAGE_SPEC.md` (§5, EBNF JSON-esca). Hoy es **opcional y desaconsejada**: una implementación **no** debería soportarla todavía. La forma canónica sigue siendo la de referencia; esta es solo un boceto para revisión.

---

## 1. Motivación

La forma canónica es JSON-esca (`"sourceType"`, claves entrecomilladas, objeto por nodo) por simpleza de análisis, no por diseño del lenguaje. Esta propuesta explora una sintaxis **más compacta y legible para desarrolladores**, quitando el ruido y expresando la identidad CPA como una ruta taxonómica. Es puro ergonómico: la superficie real de autoría es lo tangible, y esta forma serviría sobre todo para escribir y depurar programas a mano.

A diferencia del documento principal (en español), esta propuesta plantea el lenguaje **completamente en inglés**, incluidos los valores CPA (`concrete` / `pictorial` / `abstract`), para que las palabras del lenguaje sean uniformes.

## 2. Gramática (borrador)

```ebnf
program          ::= statement*
statement        ::= source_decl | transform_decl | sink_decl

source_decl      ::= "source" identifier "=" value? ";"
transform_decl   ::= "transform" identifier "=" (operation "(" arg_list? ")")? ";"
sink_decl        ::= "sink" identifier "=" identifier? ";"

arg_list         ::= identifier ("," identifier)*
operation        ::= identifier

value            ::= object | criterion | group
group            ::= "[" (object ("," object)*)? "]"
                 |   "[" (criterion ("," criterion)*)? "]"

object           ::= quantity? taxonomy attributes?
taxonomy         ::= category ":" identifier ":" identifier   // category:type:subtype
category         ::= "concrete" | "pictorial" | "abstract"
quantity         ::= rational
attributes       ::= "{" (attr ("," attr)*)? "}"
attr             ::= identifier ":" identifier

criterion        ::= filter_criterion | order_criterion
filter_criterion ::= "where" "{" (constraint ("," constraint)*)? "}"
constraint       ::= identifier ":" identifier
order_criterion  ::= "by" "{" identifier ":" order_dir "}"
order_dir        ::= "asc" | "desc" | value_sequence
value_sequence   ::= "[" identifier ("," identifier)* "]"

rational         ::= "-"? digit+ ( "/" digit+ | "." digit+ )?
identifier       ::= [a-zA-Z] [a-zA-Z0-9_-]*
digit            ::= [0-9]
```

Puntos de diseño:

- **Cantidad al frente**, opcional (por defecto 1): `2 concrete:food:apple` se lee como "2 manzanas"; `concrete:food:apple` es 1.
- **Ruta `category:type:subtype`** con `:`: expresa la jerarquía CPA de lo general a lo específico en un solo token compuesto.
- **Atributos** en `{clave: valor, …}` al final del objeto; son parte de la identidad.
- **Criterios tipados por palabra clave**: `where{…}` (filtro, conjunción de `propiedad: valor` de un solo valor) y `by{…}` (orden, con `asc`/`desc` o una secuencia). El "O" del filtrado se obtiene pasando varios criterios a `filter(…)`, igual que en la forma canónica (DNF).
- **Grupos homogéneos**: un grupo es de objetos o de criterios, nunca mixto.
- **Literal racional**: entero, fracción `p/q` o decimal, con signo; exacto sobre ℚ.
- **Nodos incompletos** toleran valor ausente (evalúan a `nulo`), igual que la forma canónica.

## 3. Ejemplos

```erae
source apple  = 2 concrete:food:apple {color: red};
source one    = concrete:food:apple;              // cantidad = 1
source third  = 1/3 abstract:number:rational;     // fracción exacta

source fruits = [3 concrete:food:apple, 1 concrete:food:pear {color: green}];

source onlyRed = where{type: apple, color: red};
source bySize  = by{size: [small, medium, large]};
source byQty   = by{quantity: desc};

transform reds   = filter(fruits, onlyRed);
transform sorted = order(fruits, bySize);
transform total  = sum(fruits);
sink out = total;

// nodos incompletos: se toleran
source x = ;   transform y = ;   sink z = ;
```

## 4. Correspondencia con la forma canónica

La forma compacta y la canónica denotan el mismo programa. Por ejemplo:

```erae
// Compacta
source apple = 2 concrete:food:apple {color: red};
```

```erae
// Canónica (LANGUAGE_SPEC.md §5) — nótese que hoy la canónica usa valores CPA en español
source apple = {
  "sourceType": "data",
  "category": "concreto",
  "type": "food",
  "subtype": "apple",
  "quantity": 2,
  "color": "red"
};
```

(Adoptar esta propuesta implicaría también unificar los valores CPA en inglés en ambas formas.)

## 5. Decisiones abiertas

- **Palabras de criterio**: se eligió `where` / `by` para no chocar de vista con las llamadas `filter(...)` / `order(...)`. Alternativa: reutilizar `filter{…}` / `order{…}` (se distinguen por `{}` vs `()`).
- **Separador de ruta**: `:` (elegido). Alternativas descartadas: `/` (choca con la fracción) y `.` (choca con el decimal).
- **Azúcar de número**: si un racional suelto (`source third = 1/3;`) debería implicar `abstract:number:rational`. Cómodo, pero agrega un caso especial.

## 6. Costo de adopción (referencia)

Implementar esta sintaxis tocaría lexer, parser, el serializador tangible→texto y todos los ejemplos y pruebas; además, convivir con la forma canónica (que se conserva como API JSON-esca) implica mantener dos representaciones. Acotado pero ramificado: por eso hoy es opcional y desaconsejado implementarla.
