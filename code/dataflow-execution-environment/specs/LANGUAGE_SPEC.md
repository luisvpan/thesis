# Dataflow Language Specification
## Integrated Types and Operations for Educational Programming (Ages 6-9)

**Document Status:** Living specification - update as implementation reveals edge cases, inconsistencies, or better designs.

**How to Use:**
- This spec describes the initial design for types and operations
- If you discover missing definitions or unclear semantics during implementation, update this spec
- Use Opus-level reasoning (ultrathink) when making spec changes
- Keep acceptance criteria behavioral (WHAT, not HOW)
- Document spec changes in IMPLEMENTATION_PLAN.md

---

## 1. DESIGN PHILOSOPHY

### Core Principles (Scratch-Inspired)
- ✅ **Integrated types** in the language (not user-extensible)
- ✅ **Type-safe operations** validated at compile-time
- ✅ **Error prevention** through type checking and arity validation
- ✅ **Curriculum-aligned** types for ages 6-9 education

### Design Decision: Integrated Operations
For an educational language targeting 6-9 year olds:

**NOT needed:**
- Abstract type systems
- User-defined type extensions
- Parametric polymorphism

**Needed:**
- Concrete curriculum types (shapes, cars, animals, etc.)
- Type-specific operations
- Clear visual categorization

---

## 2. PRIMITIVE TYPES

### 2.1 Natural Numbers
```typescript
type Natural = {
  kind: "natural";
  value: number;  // >= 0, formally some would say that this is 0 + naturals
}
```

**Operations:**
- `ADD(n1: Natural, n2: Natural) → Natural`
- `SUBTRACT(n1: Natural, n2: Natural) → Integer` ⚠️ May be negative
- `MULTIPLY(n1: Natural, n2: Natural) → Natural`
- `DIVIDE(n1: Natural, n2: Natural) → Decimal`
- `COMPARE(n1: Natural, n2: Natural) → Boolean` - Returns `true` if equal, `false` otherwise (value-based comparison)
- `FILTER(list: Natural[], condition) → Natural[]`
- `SORT(list: Natural[]) → Natural[]`

### 2.2 Integers
```typescript
type Integer = {
  kind: "integer";
  value: number;  // positive, negative, or zero
}
```

**Operations:**
- `ADD(n1: Integer, n2: Integer) → Integer`
- `SUBTRACT(n1: Integer, n2: Integer) → Integer`
- `MULTIPLY(n1: Integer, n2: Integer) → Integer`
- `DIVIDE(n1: Integer, n2: Integer) → Decimal`
- `COMPARE(n1: Integer, n2: Integer) → Boolean` - Returns `true` if equal, `false` otherwise (value-based comparison)
- `FILTER(list: Integer[], condition) → Integer[]`
- `SORT(list: Integer[]) → Integer[]`

### 2.3 Floating-point Nummbers

#### 2.3.1 Decimals
```typescript
type Decimal = {
  kind: "decimal";
  value: number;
}
```

**Operations:**
- `ADD(n1: Decimal, n2: Decimal) → Decimal`
- `SUBTRACT(n1: Decimal, n2: Decimal) → Decimal`
- `MULTIPLY(n1: Decimal, n2: Decimal) → Decimal`
- `DIVIDE(n1: Decimal, n2: Decimal) → Decimal`
- `COMPARE(n1: Decimal, n2: Decimal) → Boolean` - Returns `true` if equal, `false` otherwise (value-based comparison)
- `FILTER(list: Decimal[], condition) → Decimal[]`
- `SORT(list: Decimal[]) → Decimal[]`

#### 2.3.2 Fractions
```typescript
type Fraction = {
  kind: "fraction";
  numerator: number;
  denominator: number;
}
```

**Operations:**
- `ADD(n1: Fraction, n2: Fraction) → Fraction`
- `SUBTRACT(n1: Fraction, n2: Fraction) → Fraction`
- `MULTIPLY(n1: Fraction, n2: Fraction) → Fraction`
- `DIVIDE(n1: Fraction, n2: Fraction) → Fraction`
- `COMPARE(n1: Fraction, n2: Fraction) → Boolean` - Returns `true` if equal, `false` otherwise (value-based comparison)
- `FILTER(list: Fraction[], condition) → Fraction[]`
- `SORT(list: Fraction[]) → Fraction[]`

### 2.4 Text (String)
```typescript
type Text = {
  kind: "text";
  value: string;
}
```

**Operations:**
- `COMPARE(t1: Text, t2: Text) → Boolean` - Returns `true` if equal, `false` otherwise (lexicographical comparison based on UTF-16 code unit values, by value not by reference)
- `ALPHABETICAL_SORT(list: Text[]) → Text[]`

**Note:** Primarily for defining other types (colors, categories)

### 2.5 Boolean
```typescript
type Boolean = {
  kind: "boolean";
  value: boolean;
}
```

**Operations:**
- `AND(b1: Boolean, b2: Boolean) → Boolean`
- `OR(b1: Boolean, b2: Boolean) → Boolean`
- `NOT(b: Boolean) → Boolean`
- `COMPARE(b1: Boolean, b2: Boolean) → Boolean` - Returns `true` if equal, `false` otherwise (value-based comparison)

---

## 3. CURRICULUM TYPES

### 3.1 Literals
```typescript
type ShapeType = "circle" | "triangle" | "square" | "rectangle";

type Size = "small" | "medium" | "large";

type Color = "red" | "blue" | "yellow" | "green" | "orange" | "purple";

type Taste = "sweet" | "salty" | "sour" | "bitter";

type AnimalType = "dog" | "cat" | "bird" | "fish" | "rabbit" | "turtle";

type AgeGroup = "child" | "teenager" | "adult" | "senior";

type Gender = "male" | "female"; // Simplified for educational context
```

**Note:** Primarily for defining other types and operations

### 3.2 Geometric Shapes
```typescript
type Shape = {
  kind: "shape";
  type: ShapeType;
  size: Size;
  color: Color;
}
```

**Comparison Operations:**
- `COMPARE_BY_SIZE(s1: Shape, s2: Shape) → Boolean` - Returns `true` if sizes are equal, `false` otherwise (by value)
- `COMPARE_BY_COLOR(s1: Shape, s2: Shape) → Boolean` - Returns `true` if colors are equal, `false` otherwise (by value)
- `COMPARE_BY_TYPE(s1: Shape, s2: Shape) → Boolean` - Returns `true` if types are equal, `false` otherwise (by value)

**Filtering Operations:**
- `FILTER_BY_SIZE(list: Shape[], size: Size) → Shape[]`
- `FILTER_BY_COLOR(list: Shape[], color: Color) → Shape[]`
- `FILTER_BY_TYPE(list: Shape[], type: ShapeType) → Shape[]`

**Set Operations:**
- `UNION(set1: Shape[], set2: Shape[]) → Shape[]`
- `INTERSECTION(set1: Shape[], set2: Shape[]) → Shape[]`
- `DIFFERENCE(set1: Shape[], set2: Shape[]) → Shape[]`
- `COMPLEMENT(set: Shape[], universe: Shape[]) → Shape[]`

### 3.3 Cars
```typescript
type Car = {
  kind: "car";
  color: Color;
}
```

**Comparison Operations:**
- `COMPARE_BY_COLOR(s1: Car, s2: Car) → Boolean` - Returns `true` if colors are equal, `false` otherwise (by value)

**Filtering Operations:**
- `FILTER_BY_COLOR(list: Car[], color: Color) → Car[]`

**Set Operations:**
- `UNION(set1: Car[], set2: Car[]) → Car[]`
- `INTERSECTION(set1: Car[], set2: Car[]) → Car[]`
- `DIFFERENCE(set1: Car[], set2: Car[]) → Car[]`
- `COMPLEMENT(set: Car[], universe: Car[]) → Car[]`

### 3.4 Food
```typescript
type Food = {
  kind: "food";
  taste: Taste;
  color: Color;
}
```

**Comparison Operations:**
- `COMPARE_BY_TASTE(s1: Food, s2: Food) → Boolean` - Returns `true` if tastes are equal, `false` otherwise (by value)
- `COMPARE_BY_COLOR(s1: Food, s2: Food) → Boolean` - Returns `true` if colors are equal, `false` otherwise (by value)

**Filtering Operations:**
- `FILTER_BY_TASTE(list: Food[], taste: Taste) → Food[]`
- `FILTER_BY_COLOR(list: Food[], color: Color) → Food[]`

**Set Operations:**
- `UNION(set1: Food[], set2: Food[]) → Food[]`
- `INTERSECTION(set1: Food[], set2: Food[]) → Food[]`
- `DIFFERENCE(set1: Food[], set2: Food[]) → Food[]`
- `COMPLEMENT(set: Food[], universe: Food[]) → Food[]`

### 3.5 Animals
```typescript
type Animal = {
  kind: "animal";
  type: AnimalType;
  color: Color;
}
```

**Comparison Operations:**
- `COMPARE_BY_TYPE(s1: Animal, s2: Animal) → Boolean` - Returns `true` if types are equal, `false` otherwise (by value)
- `COMPARE_BY_COLOR(s1: Animal, s2: Animal) → Boolean` - Returns `true` if colors are equal, `false` otherwise (by value)

**Filtering Operations:**
- `FILTER_BY_TYPE(list: Animal[], type: AnimalType) → Animal[]`
- `FILTER_BY_COLOR(list: Animal[], color: Color) → Animal[]`

**Set Operations:**
- `UNION(set1: Animal[], set2: Animal[]) → Animal[]`
- `INTERSECTION(set1: Animal[], set2: Animal[]) → Animal[]`
- `DIFFERENCE(set1: Animal[], set2: Animal[]) → Animal[]`
- `COMPLEMENT(set: Animal[], universe: Animal[]) → Animal[]`

### 3.6 People
```typescript
type Person = {
  kind: "person";
  ageGroup: AgeGroup;
  gender: Gender;  // Simplified for educational context
}
```

**Comparison Operations:**
- `COMPARE_BY_AGE_GROUP(s1: Person, s2: Person) → Boolean` - Returns `true` if age groups are equal, `false` otherwise (by value)
- `COMPARE_BY_GENDER(s1: Person, s2: Person) → Boolean` - Returns `true` if genders are equal, `false` otherwise (by value)

**Filtering Operations:**
- `FILTER_BY_AGE_GROUP(list: Person[], ageGroup: AgeGroup) → Person[]`
- `FILTER_BY_GENDER(list: Person[], gender: Gender) → Person[]`

**Set Operations:**
- `UNION(set1: Person[], set2: Person[]) → Person[]`
- `INTERSECTION(set1: Person[], set2: Person[]) → Person[]`
- `DIFFERENCE(set1: Person[], set2: Person[]) → Person[]`
- `COMPLEMENT(set: Person[], universe: Person[]) → Person[]`

---

## 4. TEMPORAL TYPES (STREAMS)

### 4.1 Time Series
```typescript
type Stream<T> = {
  kind: "stream";
  elementType: T;
  generator: Generator<T>;
}
```

**Any type can be a stream:**
- `Stream<Natural>`
- `Stream<Shape>`
- `Stream<Car>`
- etc.

**Temporal Operations:**
- `NEXT(s: Stream<T>) → T` - Next value from stream
- `FIRST(s: Stream<T>) → T` - First value
- `FBY(initial: T, subsequent: Stream<T>) → Stream<T>` - "Followed by"
- `ACCUMULATE(s: Stream<T>, op: Operation, initial: T) → Stream<T>`

**Inherited Operations:**
All operations of type `T` apply to `Stream<T>`:

```dataflow
stream cars: stream<Car> from sensor;
stream red_cars: stream<Car> = FILTER_BY_COLOR(cars, "red");
```

---

## 5. COMPOSITE TYPES

### 5.1 Sets
```typescript
type Set<T> = {
  kind: "set";
  elementType: T;
  elements: T[];
}
```

**Operations inherited from element type `T`**

---

## 6. JSON REPRESENTATION

### 6.1 Program Structure
```typescript
interface DataflowProgram {
  metadata: {
    programId: string;
    activityId?: string;
    level?: number;
  };
  
  graph: {
    nodes: DataflowNode[];
    edges: DataflowEdge[];
  };
}
```

### 6.2 Node Types
```typescript
interface DataflowNode {
  id: string;
  type: "DataSource" | "Transformation" | "Output" | "Stream" | "Accumulator";
  dataType: DataType;
  
  // For DataSource
  value?: any;
  
  // For Transformation
  operation?: Operation;
  
  // Optional metadata
  metadata?: {
    label?: string;
    blockId?: string;
    position?: [number, number];
  };
}

interface DataflowEdge {
  id: string;
  from: string;
  to: string;
  toPort?: number;  // For multi-input nodes
}
```

### 6.3 Complete Example: Filter Red Shapes

**JSON Input:**
```json
{
  "metadata": {
    "programId": "prog_001",
    "activityId": "filter_shapes_level2",
    "level": 2
  },
  "graph": {
    "nodes": [
      {
        "id": "shapes",
        "type": "DataSource",
        "dataType": { "kind": "set", "elementType": "shape" },
        "value": [
          { "type": "circle", "color": "red", "size": "large" },
          { "type": "triangle", "color": "blue", "size": "small" },
          { "type": "square", "color": "red", "size": "medium" }
        ]
      },
      {
        "id": "color_red",
        "type": "DataSource",
        "dataType": "text",
        "value": "red"
      },
      {
        "id": "filter",
        "type": "Transformation",
        "operation": "FILTER_BY_COLOR"
      },
      {
        "id": "output",
        "type": "Output"
      }
    ],
    "edges": [
      { "id": "e1", "from": "shapes", "to": "filter", "toPort": 0 },
      { "id": "e2", "from": "color_red", "to": "filter", "toPort": 1 },
      { "id": "e3", "from": "filter", "to": "output" }
    ]
  }
}
```

**Expected Output:**
```json
[
  { "type": "circle", "color": "red", "size": "large" },
  { "type": "square", "color": "red", "size": "medium" }
]
```

---

## 7. SEMANTIC VALIDATION

### 7.1 Type Checking Rules

**Type Checker validates:**
1. **Arity:** Each operation receives correct number of inputs
2. **Type Compatibility:** Inputs match operation's type signature
3. **Property Requirements:** Types have required properties (e.g., `color` for FILTER_BY_COLOR)

### 7.2 Child-Friendly Error Messages (CRITICAL)

**Educational Rationale**: Children aged 6-9 need clear, intuitive feedback to learn from mistakes.

**Error Message Requirements:**
- Simple language (no technical jargon)
- Clear indication of problem location
- Actionable suggestions
- Spanish language only
- Examples of correct usage

**Error Type Definition:**
```typescript
export type ValidationError = {
  code: string;
  message: string;           // Technical message for developers
  childMessage?: string;      // Simplified message for children (Spanish)
  nodeId?: string;
  suggestion?: string;        // Actionable suggestion (Spanish)
  example?: string;          // Example of correct usage
};
```

**Error Messages (Spanish):**

**WRONG_ARITY**:
```
⚠️ ¡Ups! El bloque [OPERATION] necesita [N] entradas.
   Solo conectaste [ACTUAL] entradas.

   Intenta esto:
   [SUGGESTION]
```

**CYCLE_DETECTED**:
```
⚠️ ¡Ups! Hay un ciclo en el programa.
   Los bloques se conectan en círculo y no se puede calcular.

   Intenta esto:
   Busca dónde un bloque apunta a sí mismo y desconecta una línea.
```

**TYPE_ERROR**:
```
⚠️ ¡Ups! El bloque [OPERATION] no acepta ese tipo de dato.
   Espera: [EXPECTED_TYPE]
   Recibiste: [ACTUAL_TYPE]

   Intenta esto:
   Usa un bloque con el tipo correcto.
```

**MISSING_INPUT** (Incremental mode - supports multiple):
```
⚠️ El bloque [NODE_ID] está esperando entradas:
   - Puerto 0: [DESCRIPTION_1]
   - Puerto 1: [DESCRIPTION_2]

   Conecta estos bloques para continuar.
```

**Example Validation:**
```typescript
class TypeChecker {
  validateOperation(node: TransformationNode, graph: DataflowGraph): ValidationError[] {
    switch (node.operation) {
      case "FILTER_BY_COLOR":
        // Requires 2 inputs
        if (inputs.length !== 2) {
          return [{ code: "WRONG_ARITY", message: "FILTER_BY_COLOR requires 2 inputs" }];
        }
        
        // Input 1: set of elements with 'color' property
        if (!hasColorProperty(inputs[0].dataType)) {
          return [{ code: "TYPE_ERROR", message: "Elements must have 'color' property" }];
        }
        
        // Input 2: text (color name)
        if (inputs[1].dataType !== "text") {
          return [{ code: "TYPE_ERROR", message: "Second argument must be text" }];
        }
        break;
    }
    return [];
  }
}
```

---

## 8. CODE ORGANIZATION

### 8.1 Compiler Modules
```
packages/compiler/
  src/
    types/
      primitives.ts      # Natural, Integer, Decimal, Text, Boolean
      curriculum.ts      # Shape, Car, Food, Animal, Person
      composite.ts       # Set, Stream
    operations/
      registry.ts        # OPERATION_REGISTRY
      numeric.ts         # ADD, SUBTRACT, MULTIPLY, DIVIDE
      comparison.ts      # COMPARE variants
      filtering.ts       # FILTER variants
      sets.ts            # UNION, INTERSECTION, DIFFERENCE
      temporal.ts        # NEXT, FBY, ACCUMULATE
      ordering.ts        # SORT variants
    validation/
      type-checker.ts    # Type validation
      arity-checker.ts   # Arity validation
```

### 8.2 Operation Registry
```typescript
export const OPERATION_REGISTRY: Record<Operation, OperationSignature> = {
  ADD: {
    arity: 2,
    contracts: [
      { inputTypes: ["natural", "natural"], outputType: "natural" },
      { inputTypes: ["integer", "integer"], outputType: "integer" },
      { inputTypes: ["decimal", "decimal"], outputType: "decimal" },
      { inputTypes: ["fraction", "fraction"], outputType: "fraction" }
    ],
    category: "numeric"
  },
  FILTER_BY_COLOR: {
    arity: 2,
    inputTypes: [
      { kind: "set", elementType: "hasColor" },
      "text"
    ],
    outputType: { kind: "set", elementType: "sameAsInput0" },
    category: "filtering"
  },
  
  // ... all other operations
};
```

---

## 9. COMPLETE TYPE DEFINITIONS

```typescript
// Core type union
type DataType = 
  // Primitives
  | "natural" 
  | "integer"
  | "decimal"
  | "fraction"
  | "text"
  | "boolean"
  // Curriculum
  | "shape"
  | "car"
  | "food"
  | "animal"
  | "person"
  // Composite
  | { kind: "set", elementType: DataType }
  | { kind: "stream", elementType: DataType };

// All operations
type Operation =
  // Numeric
  | "ADD" | "SUBTRACT" | "MULTIPLY" | "DIVIDE"
  // Comparison
  | "COMPARE"
  | "COMPARE_BY_SIZE" | "COMPARE_BY_COLOR" | "COMPARE_BY_TYPE"
  | "COMPARE_BY_TASTE" | "COMPARE_BY_AGE_GROUP" | "COMPARE_BY_GENDER"
  // Filtering
  | "FILTER"
  | "FILTER_BY_SIZE" | "FILTER_BY_COLOR" | "FILTER_BY_TYPE"
  | "FILTER_BY_TASTE" | "FILTER_BY_AGE_GROUP" | "FILTER_BY_GENDER"
  // Sets
  | "UNION" | "INTERSECTION" | "DIFFERENCE" | "COMPLEMENT"
  // Temporal
  | "NEXT" | "FIRST" | "FBY" | "ACCUMULATE"
  // Ordering
  | "SORT" | "ALPHABETICAL_SORT";

type OperationSignature = SingleSignature | MultiSignature;

interface BaseOperation {
  arity: number;
  category: string;
}
--TODO: correct these interfaces to match implementation or fix them to be viable, as TypeConstraint references itself recursively and OutputTypeRule is not defined yet
interface TypeConstraint {
  inputTypes: (DataType | TypeConstraint)[];
  outputType: DataType | OutputTypeRule;
}

// For most simple operations with no overloading
type SingleSignature = BaseOperation & TypeConstraint;

// For operations that support overloading, like numeric operations, that work on different types of numerics
interface MultiSignature extends BaseOperation {
  contracts: TypeConstraint[];
}
```

---

**End of Language Specification**
