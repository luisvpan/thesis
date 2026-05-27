# Dataflow Language - Formal Grammar Specification

**Version:** 4.0.0
**Date:** 2026-05-18  
**Notation:** W3C Extended Backus-Naur Form (EBNF)

## Complete Grammar

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

group               ::= "[" (object_literal ("," object_literal)*)? "]"

object_literal      ::= data_literal | criteria_literal

data_literal      ::= "{" '"sourceType" : "data"' "," '"category"' ":" category_type "," '"type"' ":" string_literal "," '"subtype"' ":" string_literal "," '"quantity"' ":" rational_literal ("," kv_pair)* "}"
criteria_literal    ::= "{" '"sourceType" : "criteria"' "," '"properties"' ":" array_literal ("," kv_pair)* "}"

category_type       ::= '"abstracto"' | '"pictorico"' | '"concreto"'

kv_pair             ::= string_literal ":" kv_value
kv_value            ::= string_literal | rational_literal | array_literal

array_literal       ::= "[" (string_literal ("," string_literal)*)? "]"
rational_literal    ::= "-"? digit+ ("." digit+)?
string_literal      ::= '"' [a-zA-Z0-9_-]* '"'
identifier          ::= [a-zA-Z] [a-zA-Z0-9_-]*
digit               ::= [0-9]
```

## Reserved Keywords and Structural Identifiers

### Grammar Keywords (Statement Level)

* `source` — Begins an input node declaration.
* `transform` — Begins a processing node declaration.
* `sink` — Begins an output node declaration.

### Standard Operation Identifiers

* `sum`, `substract`, `multiply`, `divide`
* `less_than`, `greater_than`
* `order_asc`, `order_desc`, `filter`

### Built-In Structural String Values

These are the strictly recognized category literal types matching `category_type`:

* `"abstracto"`
* `"pictorico"`
* `"concreto"`

## Program Examples

### Example 1: Rational Fraction Math

```erae
source third = {
  "category": "abstracto",
  "type": "number",
  "subtype": "rational",
  "quantity": 0.3333333333
};

source direct_fraction = {
  "category": "abstracto",
  "type": "number",
  "subtype": "rational",
  "quantity": "1/3"
};

transform sum_fractions = sum(third, direct_fraction);
sink output_fraction = sum_fractions;

```

### Example 2: Dynamic Taxonomy & Freeform Attributes

Shows how a brand new concrete type (`"vehicle"`) utilizes the required `"subtype"` field while introducing custom metadata (`"doors"`) inside the dynamic `kv_pairs` block.

```erae
source sedan = {
  "category": "concreto",
  "type": "vehicle",
  "subtype": "car",
  "doors": "4",
  "quantity": 2
};

source coupe = {
  "category": "concreto",
  "type": "vehicle",
  "subtype": "car",
  "doors": "2",
  "quantity": 1
};

// Groups naturally by matching "concreto", "vehicle", and "car"
transform total_cars = sum(sedan, coupe);
sink output_cars = total_cars;

```

### Example 3: Error Recovery & Live-Editing

This example mimics what happens while a child is actively typing on a canvas. Thanks to `?` optional modifiers, this program will **successfully parse** without crashing, creating placeholder nodes for the incomplete entries.

```erae
source incomplete_apple = {
  "category": "concreto",
  "type": "food",
  "subtype": "apple"
  // User stopped typing here; missing ending kv_pairs and closing brace
;

source multiplier = {
  "category": "abstracto",
  "type": "number",
  "subtype": "multiplier",
  "quantity": 4
};

// Incomplete operation syntax: valid parser fallback
transform incomplete_calc = ;

sink active_output = multiplier;

```

### Example 4: Context-Aware Sorting and Cross-Tier Scaling

Illustrates dynamic properties (`"size": "large"`) living perfectly alongside mandatory structures, while mixing collection-sorting and math scales.

```erae
source large_star = {
  "category": "pictorico",
  "type": "shape",
  "subtype": "star",
  "size": "large",
  "quantity": 2.5
};

source scale_factor = {
  "category": "abstracto",
  "type": "number",
  "subtype": "multiplier",
  "quantity": 3
};

// Scaled calculation: returns a star with an amount of 7.5
transform scaled_stars = multiply(large_star, scale_factor);
sink final_render = scaled_stars;
```