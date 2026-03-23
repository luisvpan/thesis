# Dataflow Language - Formal Grammar Specification
## W3C EBNF Notation

**Version:** 1.0.0  
**Date:** 2026-02-26  
**Notation:** W3C Extended Backus-Naur Form (EBNF)

---

## Rationale for W3C EBNF

This specification uses W3C EBNF notation following the precedent set by XML (W3C Recommendation 1998), HTML5 (WHATWG Standard), and SVG. W3C EBNF was chosen over alternatives for the following reasons:

1. **Proven track record:** Successfully used in production-grade declarative languages (XML, HTML, SVG, CSS)
2. **Developer familiarity:** Uses regex-like syntax (`?`, `*`, `+`, `|`) familiar to modern programmers
3. **Avoids documented issues:** ISO/IEC 14977 EBNF has documented readability problems (Zaytsev 2011) and is not used even by ISO standards like Ada
4. **Available tooling:** Supports railroad diagram generators and standard parser generators
5. **Not a custom variant:** Unlike Python's mixed notation, W3C EBNF is a recognized standard

**References:**
- W3C XML 1.0: https://www.w3.org/TR/xml/
- Wheeler, D.A. "Don't Use ISO/IEC 14977 EBNF": https://dwheeler.com/essays/dont-use-iso-14977-ebnf.html
- Zaytsev, V. (2011). "BNF was Here: What Have We Done About the Unnecessary Diversity of Notation"

---

## Notation Reference

### W3C EBNF Operators

```
::=     Definition
|       Alternative
()      Grouping
[]      Optional (0 or 1)
*       Zero or more repetitions
+       One or more repetitions
?       Optional (synonym for [])
""      Terminal string (literal)
/* */   Comment
```

### Character Classes

```
[a-z]       Any lowercase letter
[A-Z]       Any uppercase letter
[0-9]       Any digit
[^"]        Any character except "
```

---

## Complete Grammar

### Program Structure

```ebnf
/* ================================================
   TOP-LEVEL PROGRAM STRUCTURE
   ================================================ */

program ::= statement+

statement ::= source_statement
            | transform_statement
            | output_statement
            | comment

comment ::= "/*" [^*]* "*/" 
          | "//" [^\n]* "\n"
```

### Source Statements

```ebnf
/* ================================================
   SOURCE STATEMENTS (Data Input Nodes)
   ================================================ */

source_statement ::= "source" identifier ":" type "=" value ";"

value ::= literal
        | set_literal
        | stream_literal

literal ::= number_literal
          | text_literal
          | shape_literal
          | car_literal
          | food_literal
          | animal_literal
          | person_literal
          | boolean_literal

number_literal ::= natural_literal
                 | integer_literal
                 | decimal_literal
                 | fraction_literal

natural_literal ::= [0-9]+

integer_literal ::= "-"? [0-9]+

decimal_literal ::= "-"? [0-9]+ "." [0-9]+
                  | "-"? "." [0-9]+

fraction_literal ::= integer_literal "/" natural_literal

text_literal ::= '"' [^"]* '"'

boolean_literal ::= "true" | "false"
```

### Curriculum Type Literals

```ebnf
/* ================================================
   CURRICULUM TYPES (Shapes, Cars, etc.)
   ================================================ */

shape_literal ::= "{" 
                    "type" ":" shape_type ","
                    "size" ":" size_value ","
                    "color" ":" color_value
                  "}"

shape_type ::= "circle" | "triangle" | "square" | "rectangle"

size_value ::= "small" | "medium" | "large"

color_value ::= "red" | "blue" | "yellow" | "green" 
              | "orange" | "purple" | "black" | "white"

car_literal ::= "{" "color" ":" color_value "}"

food_literal ::= "{"
                   "taste" ":" taste_value ","
                   "color" ":" color_value
                 "}"

taste_value ::= "sweet" | "salty" | "sour" | "bitter"

animal_literal ::= "{"
                     "type" ":" animal_type ","
                     "color" ":" color_value
                   "}"

animal_type ::= "dog" | "cat" | "bird" | "fish" | "rabbit" | "turtle"

person_literal ::= "{"
                     "ageGroup" ":" age_group ","
                     "gender" ":" gender_value
                   "}"

age_group ::= "child" | "teenager" | "adult" | "senior"

gender_value ::= "male" | "female"
```

### Composite Type Literals

```ebnf
/* ================================================
   COMPOSITE TYPES (Sets and Streams)
   ================================================ */

set_literal ::= "{" ( value ( "," value )* )? "}"

stream_literal ::= "stream" "<" type ">" "(" stream_source ")"

stream_source ::= "sensor" "(" text_literal ")"
                | "generator" "(" identifier ")"
                | "external" "(" text_literal ")"
```

### Transform Statements

```ebnf
/* ================================================
   TRANSFORMATION STATEMENTS
   ================================================ */

transform_statement ::= "transform" identifier ":" type 
                        "=" operation "(" argument_list? ")" ";"

operation ::= numeric_operation
            | comparison_operation
            | filtering_operation
            | set_operation
            | temporal_operation
            | ordering_operation

argument_list ::= argument ( "," argument )*

argument ::= identifier
           | literal
           | operation
```

### Operations by Category

```ebnf
/* ================================================
   NUMERIC OPERATIONS
   ================================================ */

numeric_operation ::= "ADD"
                    | "SUBTRACT"
                    | "MULTIPLY"
                    | "DIVIDE"

/* ================================================
   COMPARISON OPERATIONS
   ================================================ */

comparison_operation ::= "COMPARE"
                       | "COMPARE_BY_SIZE"
                       | "COMPARE_BY_COLOR"
                       | "COMPARE_BY_TYPE"
                       | "COMPARE_BY_TASTE"
                       | "COMPARE_BY_AGE_GROUP"
                       | "COMPARE_BY_GENDER"

/* ================================================
   FILTERING OPERATIONS
   ================================================ */

filtering_operation ::= "FILTER"
                      | "FILTER_BY_SIZE"
                      | "FILTER_BY_COLOR"
                      | "FILTER_BY_TYPE"
                      | "FILTER_BY_TASTE"
                      | "FILTER_BY_AGE_GROUP"
                      | "FILTER_BY_GENDER"

/* ================================================
   SET OPERATIONS
   ================================================ */

set_operation ::= "UNION"
                | "INTERSECTION"
                | "DIFFERENCE"
                | "COMPLEMENT"

/* ================================================
   TEMPORAL OPERATIONS (Streams)
   ================================================ */

temporal_operation ::= "NEXT"
                     | "FIRST"
                     | "FBY"
                     | "ACCUMULATE"

/* ================================================
   ORDERING OPERATIONS
   ================================================ */

ordering_operation ::= "SORT"
                     | "ALPHABETICAL_SORT"
```

### Output Statements

```ebnf
/* ================================================
   OUTPUT STATEMENTS
   ================================================ */

output_statement ::= "output" identifier ":" type "=" identifier ";"
```

### Type System

```ebnf
/* ================================================
   TYPE DECLARATIONS
   ================================================ */

type ::= primitive_type
       | composite_type

primitive_type ::= numeric_type
                 | "text"
                 | "boolean"
                 | curriculum_type

numeric_type ::= "natural"
               | "integer"
               | "decimal"
               | "fraction"

curriculum_type ::= "shape"
                  | "car"
                  | "food"
                  | "animal"
                  | "person"

composite_type ::= set_type
                 | stream_type

set_type ::= "set" "<" type ">"

stream_type ::= "stream" "<" type ">"
```

### Identifiers

```ebnf
/* ================================================
   IDENTIFIERS AND LEXICAL ELEMENTS
   ================================================ */

identifier ::= letter ( letter | digit | "_" )*

letter ::= [a-zA-Z]

digit ::= [0-9]
```

### Whitespace

```ebnf
/* ================================================
   WHITESPACE (Ignored)
   ================================================ */

whitespace ::= ( " " | "\t" | "\n" | "\r" )+

/* Note: Whitespace is ignored between tokens */
```

---

## Complete Example Programs

### Example 1: Simple Addition

```dataflow
/* Program: Add two numbers */

source a: natural = 3;
source b: natural = 2;

transform sum: natural = ADD(a, b);

output result: natural = sum;
```

**Expected Output:** `5`

---

### Example 2: Filter Red Shapes

```dataflow
/* Program: Filter shapes by color */

source shapes: set<shape> = {
  {type: "circle", size: "large", color: "red"},
  {type: "triangle", size: "small", color: "blue"},
  {type: "square", size: "medium", color: "red"},
  {type: "circle", size: "large", color: "yellow"}
};

source target_color: text = "red";

transform red_shapes: set<shape> = FILTER_BY_COLOR(shapes, target_color);

output result: set<shape> = red_shapes;
```

**Expected Output:**
```json
[
  {"type": "circle", "size": "large", "color": "red"},
  {"type": "square", "size": "medium", "color": "red"}
]
```

---

### Example 3: Union of Sets

```dataflow
/* Program: Combine two sets of animals */

source pets: set<animal> = {
  {type: "dog", color: "brown"},
  {type: "cat", color: "white"}
};

source zoo_animals: set<animal> = {
  {type: "bird", color: "blue"},
  {type: "fish", color: "orange"}
};

transform all_animals: set<animal> = UNION(pets, zoo_animals);

output result: set<animal> = all_animals;
```

**Expected Output:**
```json
[
  {"type": "dog", "color": "brown"},
  {"type": "cat", "color": "white"},
  {"type": "bird", "color": "blue"},
  {"type": "fish", "color": "orange"}
]
```

---

### Example 4: Temporal Stream (Counter with FBY)

```dataflow
/* Program: Counter from 0 to infinity */

source zero: natural = 0;

transform counter: stream<natural> = FBY(zero, ADD(counter, 1));

output result: stream<natural> = counter;
```

**Expected Output (first 5 timesteps):** `[0, 1, 2, 3, 4]`

---

### Example 5: Complex Expression

```dataflow
/* Program: (3 + 2) * (10 - 6) */

source a: natural = 3;
source b: natural = 2;
source c: natural = 10;
source d: natural = 6;

transform sum: natural = ADD(a, b);        /* 5 */
transform difference: integer = SUBTRACT(c, d);  /* 4 */
transform product: natural = MULTIPLY(sum, difference);  /* 20 */

output result: natural = product;
```

**Expected Output:** `20`

---

### Example 6: Filter and Sort

```dataflow
/* Program: Get large shapes and sort them */

source shapes: set<shape> = {
  {type: "circle", size: "small", color: "red"},
  {type: "square", size: "large", color: "blue"},
  {type: "triangle", size: "large", color: "yellow"},
  {type: "circle", size: "medium", color: "green"}
};

source target_size: text = "large";

transform large_shapes: set<shape> = FILTER_BY_SIZE(shapes, target_size);

transform sorted_shapes: set<shape> = SORT(large_shapes);

output result: set<shape> = sorted_shapes;
```

---

## Semantic Constraints

### Type Safety Rules

The following semantic constraints are NOT expressible in context-free grammar but MUST be enforced by the type checker:

1. **Arity Checking:**
   - Each operation must receive the correct number of arguments
   - Example: `ADD` requires exactly 2 arguments

2. **Type Compatibility:**
   - Operation inputs must match expected types
   - Example: `ADD(natural, natural) → natural`
   - Example: `FILTER_BY_COLOR(set<hasColor>, text) → set<hasColor>`

3. **Property Requirements:**
   - Types must have required properties for operations
   - Example: `FILTER_BY_COLOR` requires elements with `color` property
   - Valid: `FILTER_BY_COLOR(set<shape>, "red")` ✓
   - Invalid: `FILTER_BY_COLOR(set<natural>, "red")` ✗ (natural has no color)

4. **DAG Validation:**
   - Programs must form a Directed Acyclic Graph (no cycles)
   - Each identifier must be defined before use
   - No circular dependencies

5. **Output Node Requirement:**
   - Every valid program must have at least one `output` statement

6. **Unique Identifiers:**
   - Each identifier can only be defined once

7. **Type Homogeneity in Sets:**
   - All elements in a set must have the same type
   - Example: `{1, 2, 3}` is valid
   - Example: `{1, "hello", {type: "circle"}}` is invalid

8. **Stream Type Consistency:**
   - Stream operations must preserve element type
   - Example: `FBY(natural, stream<natural>) → stream<natural>`

---

## Operation Type Signatures

### Complete Operation Registry

```typescript
// This is informative TypeScript notation, not part of the EBNF grammar

type OperationSignature = {
  arity: number;
  inputTypes: DataType[];
  outputType: DataType;
  category: string;
}

const OPERATION_REGISTRY: Record<string, OperationSignature> = {
  // Numeric Operations
  "ADD": {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "natural",
    category: "numeric"
  },
  
  "SUBTRACT": {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "integer",  // Can be negative!
    category: "numeric"
  },
  
  "MULTIPLY": {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "natural",
    category: "numeric"
  },
  
  "DIVIDE": {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "decimal",
    category: "numeric"
  },
  
  // Comparison Operations
  "COMPARE": {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "boolean",  // Returns true if equal, false otherwise
    category: "comparison"
  },
  
  "COMPARE_BY_COLOR": {
    arity: 2,
    inputTypes: ["hasColor", "hasColor"],
    outputType: "boolean",  // Returns true if colors are equal, false otherwise
    category: "comparison"
  },
  
  // Filtering Operations
  "FILTER_BY_COLOR": {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "hasColor" },
      "text"
    ],
    outputType: { kind: "set", elementType: "sameAsInput0" },
    category: "filtering"
  },
  
  "FILTER_BY_SIZE": {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "shape" },
      "text"
    ],
    outputType: { kind: "set", elementType: "shape" },
    category: "filtering"
  },
  
  // Set Operations
  "UNION": {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "T" },
      { kind: "set", elementType: "T" }
    ],
    outputType: { kind: "set", elementType: "T" },
    category: "sets"
  },
  
  "INTERSECTION": {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "T" },
      { kind: "set", elementType: "T" }
    ],
    outputType: { kind: "set", elementType: "T" },
    category: "sets"
  },
  
  // Temporal Operations
  "FBY": {
    arity: 2,
    inputTypes: ["T", { kind: "stream", elementType: "T" }],
    outputType: { kind: "stream", elementType: "T" },
    category: "temporal"
  },
  
  "NEXT": {
    arity: 1,
    inputTypes: [{ kind: "stream", elementType: "T" }],
    outputType: "T",
    category: "temporal"
  }
  
  // ... (40+ operations total, see LANGUAGE_SPEC.md for complete list)
};
```

---

## Reserved Keywords

The following identifiers are **reserved** and cannot be used as variable names:

```
/* Statement keywords */
source, transform, output

/* Type keywords */
natural, integer, decimal, fraction, text, boolean
shape, car, food, animal, person
set, stream

/* Operation keywords */
ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE
FILTER, FILTER_BY_COLOR, FILTER_BY_SIZE, FILTER_BY_TYPE
FILTER_BY_TASTE, FILTER_BY_AGE_GROUP, FILTER_BY_GENDER
UNION, INTERSECTION, DIFFERENCE, COMPLEMENT
NEXT, FIRST, FBY, ACCUMULATE
SORT, ALPHABETICAL_SORT
COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE
COMPARE_BY_TASTE, COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER

/* Literal keywords */
true, false

/* Stream source keywords */
sensor, generator, external

/* Shape types */
circle, triangle, square, rectangle

/* Size values */
small, medium, large

/* Color values */
red, blue, yellow, green, orange, purple, black, white

/* Taste values */
sweet, salty, sour, bitter

/* Animal types */
dog, cat, bird, fish, rabbit, turtle

/* Age groups */
child, teenager, adult, senior

/* Gender values */
male, female
```

---

## Grammar Summary Statistics

- **Total non-terminals:** 48
- **Total terminals (keywords + operators):** 80+
- **Total operations defined:** 40+
- **Primitive types:** 5 (natural, integer, decimal, text, boolean)
- **Curriculum types:** 5 (shape, car, food, animal, person)
- **Composite type constructors:** 2 (set, stream)

---

## Implementation Notes

### Parser Generation

This grammar is suitable for:
- **LL(k) parsers** (k ≥ 2)
- **LR parsers** (LALR(1))
- **Recursive descent** (hand-written)
- **PEG parsers** (with minor adaptations)

### Tooling Support

**Railroad Diagram Generation:**
Use Gunther Rademacher's Railroad Diagram Generator:
- Input: This W3C EBNF grammar
- Output: Visual syntax diagrams
- URL: https://www.bottlecaps.de/rr/ui

**Parser Generators:**
- ANTLR (with EBNF adapter)
- Nearley.js (supports W3C EBNF)
- Chevrotain (JavaScript/TypeScript)

**Preferred Parser Generator**: Chevrotain - Check `docs/PARSER_GENERATOR_ANALYSIS.md` for justification of this choice

---

## Other Notes

- Keep up to date as language evolves for documentation and academical purposes, even though it is not directly used by the parser

## References

### W3C EBNF Specification
- XML 1.0 (Fifth Edition): https://www.w3.org/TR/xml/
- EBNF Notation section: https://www.w3.org/TR/xml/#sec-notation

### Related Documents
- `LANGUAGE_SPEC.md` - Complete language specification with semantics
- `docs/BNF_NOTATION_ANALYSIS.md` - Justification for W3C EBNF choice
- `INTEGRATION_SPEC.md` - Runtime and integration details

---

**End of Grammar Specification**
