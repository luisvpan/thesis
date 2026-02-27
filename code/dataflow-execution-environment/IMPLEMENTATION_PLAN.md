# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory

---

## Executive Summary

This document provides a comprehensive implementation plan for building a dataflow programming language compiler and runtime for children aged 6-9. The plan follows the **Ralph Wiggum method** - building in functional layers, each completely working before moving to the next.

### Technology Stack Decisions

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Parser Generator** | Chevrotain | Native TypeScript, Bun-compatible, fastest, actively maintained (v11.0.3) |
| **HTTP/WebSocket Server** | Elysia | Bun-native, excellent TypeScript support, minimal overhead |
| **Runtime** | Bun | Fast, native TypeScript execution, single-process architecture |
| **Package Manager** | Bun workspaces | Monorepo structure, fast installs |
| **Language** | TypeScript | Strict mode, type safety, educational value |

### Architecture Overview

```
packages/
├── shared/          # Shared types, utilities, stdlib
├── compiler/        # Parser, validator, type checker
├── runtime/         # Demand-driven evaluator, incremental runtime
├── http-api/        # HTTP API for batch mode (CV system)
└── websocket-server/ # WebSocket for live feedback (IDE)
```

---

## Layer 1: Foundation - Natural Numbers + ADD Only

**Estimated Effort:** 2-3 days
**Prerequisites:** None
**Success Criteria:** Program "3 + 2 = 5" works end-to-end

---

### Setup Tasks

#### Feature: Monorepo Structure

**From Acceptance Criteria (PROJECT_GOALS.md):**
- ✓ Bun workspaces configured
- ✓ TypeScript strict mode enabled
- ✓ Shared types properly exported

**Required Tests:**
- [ ] Test: packages/shared can be imported from compiler
- [ ] Test: TypeScript typecheck passes across packages
- [ ] Test: `bun run build` builds all packages

**Implementation Notes:**
```json
// package.json (root)
{
  "name": "dataflow-language",
  "workspaces": [
    "packages/*"
  ],
  "scripts": {
    "build": "bun run build --filter=*",
    "dev": "bun run dev --filter=*",
    "test": "bun test",
    "typecheck": "tsc --noEmit"
  }
}

// packages/shared/package.json
{
  "name": "@dataflow/shared",
  "exports": {
    "./types": "./src/types/index.ts"
  }
}
```

---

#### Feature: Shared Type System

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Primitive types defined (Natural, Integer, etc.)
- ✓ Operation signatures defined
- ✓ ValidationError types defined

**Required Tests:**
- [ ] Test: Natural type with value >= 0 validates correctly
- [ ] Test: Natural type rejects negative values
- [ ] Test: OperationSignature type enforces arity
- [ ] Test: ValidationError includes code, message, nodeId

**Implementation Notes:**
```typescript
// packages/shared/src/types/primitives.ts
export type Natural = {
  kind: "natural";
  value: number;
};

export type Integer = {
  kind: "integer";
  value: number;
};

// packages/shared/src/types/validation.ts
export type ValidationError = {
  code: string;
  message: string;
  nodeId?: string;
  details?: Record<string, unknown>;
};

export type ValidationResult = {
  success: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
};
```

---

#### Feature: JSON Input Schema

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ DataflowProgram structure defined
- ✓ DataflowNode types (DataSource, Transformation, Output)
- ✓ DataflowEdge structure

**Required Tests:**
- [ ] Test: Valid minimal program parses correctly
- [ ] Test: Missing required field throws validation error
- [ ] Test: Array of nodes/edges accepted

**Implementation Notes:**
```typescript
// packages/shared/src/types/program.ts
export interface DataflowProgram {
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

export type DataflowNode =
  | DataSourceNode
  | TransformationNode
  | OutputNode;

export interface DataSourceNode {
  id: string;
  type: "DataSource";
  dataType: DataType;
  value: unknown;
  metadata?: NodeMetadata;
}

export interface TransformationNode {
  id: string;
  type: "Transformation";
  dataType: DataType;
  operation: Operation;
  metadata?: NodeMetadata;
}
```

---

### Compiler Layer 1

#### Feature: Lexer (Token Definitions)

**From Acceptance Criteria (GRAMMAR_SPEC.md):**
- ✓ Keywords: source, transform, output, natural
- ✓ Literals: numbers, strings
- ✓ Punctuation: :, ;, =, (, ), ,

**Required Tests:**
- [ ] Test: "source a: natural = 3;" tokenizes correctly
- [ ] Test: Whitespace is skipped
- [ ] Test: Invalid character throws error

**Implementation Notes:**
```typescript
// packages/compiler/src/lexer/tokens.ts
import { createToken } from "chevrotain";

export const Source = createToken({ name: "Source", pattern: /source/ });
export const Transform = createToken({ name: "Transform", pattern: /transform/ });
export const Output = createToken({ name: "Output", pattern: /output/ });
export const Natural = createToken({ name: "Natural", pattern: /natural/ });
export const Equals = createToken({ name: "Equals", pattern: /=/ });
export const Colon = createToken({ name: "Colon", pattern: /:/ });
export const Semicolon = createToken({ name: "Semicolon", pattern: /;/ });
export const LParen = createToken({ name: "LParen", pattern: /\(/ });
export const RParen = createToken({ name: "RParen", pattern: /\)/ });
export const Comma = createToken({ name: "Comma", pattern: /,/ });
export const Identifier = createToken({ name: "Identifier", pattern: /[a-zA-Z_]\w*/ });
export const NumberLiteral = createToken({ name: "NumberLiteral", pattern: /[0-9]+/ });

export const allTokens = [
  WhiteSpace, Source, Transform, Output, Natural,
  Equals, Colon, Semicolon, LParen, RParen, Comma,
  Identifier, NumberLiteral
];
```

---

#### Feature: Parser (Chevrotain)

**From Acceptance Criteria (GRAMMAR_SPEC.md):**
- ✓ Parses source statements
- ✓ Parses transform statements
- ✓ Parses output statements
- ✓ Builds CST (Concrete Syntax Tree)

**Required Tests:**
- [ ] Test: "source a: natural = 3;" parses to correct CST
- [ ] Test: Multiple statements parse correctly
- [ ] Test: Syntax error produces parser error

**Implementation Notes:**
```typescript
// packages/compiler/src/parser/dataflow-parser.ts
import { CstParser } from "chevrotain";
import { allTokens } from "../lexer/tokens";

class DataflowParser extends CstParser {
  constructor() {
    super(allTokens);
    this.performSelfAnalysis();
  }

  program = this.RULE("program", () => {
    this.AT_LEAST_ONE(() => this.SUBRULE(this.statement));
  });

  statement = this.RULE("statement", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.sourceStatement) },
      { ALT: () => this.SUBRULE(this.transformStatement) },
      { ALT: () => this.SUBRULE(this.outputStatement) }
    ]);
  });

  sourceStatement = this.RULE("sourceStatement", () => {
    this.CONSUME(Source);
    this.CONSUME(Identifier);
    this.CONSUME(Colon);
    this.CONSUME(Natural);
    this.CONSUME(Equals);
    this.SUBRULE(this.value);
    this.CONSUME(Semicolon);
  });

  value = this.RULE("value", () => {
    this.CONSUME(NumberLiteral);
  });

  // ... more rules
}
```

---

#### Feature: AST Builder (Visitor Pattern)

**From Acceptance Criteria (GRAMMAR_SPEC.md):**
- ✓ Converts CST to AST
- ✓ Type-safe AST node construction
- ✓ Extracts node identifiers and types

**Required Tests:**
- [ ] Test: CST converts to correct AST
- [ ] Test: Statement types correctly identified
- [ ] Test: Value literals correctly parsed

**Implementation Notes:**
```typescript
// packages/compiler/src/ast/ast-builder.ts
import { BaseVisitor } from "chevrotain";

class AstBuilder extends BaseVisitor {
  constructor() {
    super();
    this.validateVisitor();
  }

  program(ctx) {
    return {
      type: "Program",
      statements: ctx.statement.map(s => this.visit(s))
    };
  }

  sourceStatement(ctx) {
    return {
      type: "SourceStatement",
      id: ctx.Identifier[0].image,
      dataType: ctx.Natural[0].image,
      value: parseInt(ctx.NumberLiteral[0].image)
    };
  }
}
```

---

#### Feature: DAG Validator

**From Acceptance Criteria (GRAMMAR_SPEC.md):**
- ✓ Detects circular dependencies
- ✓ Ensures all identifiers defined before use
- ✓ Validates edge connections

**Required Tests:**
- [ ] Test: Valid DAG passes validation
- [ ] Test: Simple cycle (a→b→a) detected
- [ ] Test: Undefined identifier caught
- [ ] Test: Self-loop detected

**Implementation Notes:**
```typescript
// packages/compiler/src/validation/dag-validator.ts
export class DagValidator {
  validate(ast: Program): ValidationResult {
    const errors: ValidationError[] = [];
    const defined = new Set<string>();

    // Build symbol table
    for (const stmt of ast.statements) {
      if (defined.has(stmt.id)) {
        errors.push({
          code: "DUPLICATE_IDENTIFIER",
          message: `Identifier '${stmt.id}' already defined`,
          nodeId: stmt.id
        });
      }
      defined.add(stmt.id);
    }

    // Detect cycles
    const hasCycle = this.detectCycles(ast);
    if (hasCycle) {
      errors.push({
        code: "CYCLE_DETECTED",
        message: "Graph contains a cycle"
      });
    }

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    };
  }
}
```

---

### Runtime Layer 1

#### Feature: DataflowGraph Structure

**From Acceptance Criteria (DEMAND_DRIVEN_INCREMENTAL.md):**
- ✓ Stores nodes and edges
- ✓ Provides input lookup by node
- ✓ Supports output node queries

**Required Tests:**
- [ ] Test: Add node and retrieve it
- [ ] Test: Get inputs for transformation node
- [ ] Test: Get output nodes

**Implementation Notes:**
```typescript
// packages/runtime/src/graph/dataflow-graph.ts
export class DataflowGraph {
  private nodes = new Map<string, DataflowNode>();
  private edges: DataflowEdge[] = [];

  addNode(node: DataflowNode): void {
    this.nodes.set(node.id, node);
  }

  addEdge(edge: DataflowEdge): void {
    this.edges.push(edge);
  }

  getInputs(nodeId: string): DataflowNode[] {
    const incoming = this.edges.filter(e => e.to === nodeId);
    return incoming
      .map(e => this.nodes.get(e.from))
      .filter((n): n is DataflowNode => n !== undefined);
  }

  getOutputNodes(): DataflowNode[] {
    return Array.from(this.nodes.values()).filter(
      n => n.type === "Output"
    );
  }
}
```

---

#### Feature: Demand-Driven Evaluator

**From Acceptance Criteria (PROJECT_GOALS.md):**
- ✓ Evaluates nodes only when demanded
- ✓ Uses memoization cache
- ✓ Propagates demand recursively

**Required Tests:**
- [ ] Test: Single DataSource node evaluates
- [ ] Test: ADD operation evaluates both inputs
- [ ] Test: Cached value returned on second call
- [ ] Test: Cache hit count tracked

**Implementation Notes:**
```typescript
// packages/runtime/src/evaluator/demand-driven-evaluator.ts
export class DemandDrivenEvaluator {
  private cache = new Map<string, Map<number, unknown>>();
  private cacheHits = 0;
  private cacheMisses = 0;

  evaluate(nodeId: string, time: number, graph: DataflowGraph): unknown {
    // Check cache
    if (this.cache.has(nodeId) && this.cache.get(nodeId)!.has(time)) {
      this.cacheHits++;
      return this.cache.get(nodeId)!.get(time);
    }
    this.cacheMisses++;

    // Get node
    const node = graph.getNode(nodeId);
    if (!node) throw new Error(`Node ${nodeId} not found`);

    // Evaluate based on node type
    let value: unknown;
    if (node.type === "DataSource") {
      value = node.value;
    } else if (node.type === "Transformation") {
      const inputNodes = graph.getInputs(nodeId);
      const inputs = inputNodes.map(n => ({
        id: n.id,
        value: this.evaluate(n.id, time, graph)
      }));
      value = this.evaluateOperation(node.operation, inputs);
    }

    // Cache result
    if (!this.cache.has(nodeId)) {
      this.cache.set(nodeId, new Map());
    }
    this.cache.get(nodeId)!.set(time, value);

    return value;
  }

  getCacheStats(): { hits: number; misses: number } {
    return {
      hits: this.cacheHits,
      misses: this.cacheMisses
    };
  }
}
```

---

#### Feature: Operation Implementation (ADD)

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ ADD(natural, natural) → natural
- ✓ Returns sum of two naturals
- ✓ Type-safe inputs

**Required Tests:**
- [ ] Test: ADD(3, 2) returns 5
- [ ] Test: ADD(0, 0) returns 0
- [ ] Test: ADD rejects non-natural inputs

**Implementation Notes:**
```typescript
// packages/runtime/src/operations/numeric.ts
export function ADD(inputs: { id: string; value: unknown }[]): Natural {
  const [a, b] = inputs;

  if (typeof a.value !== "number" || typeof b.value !== "number") {
    throw new TypeError("ADD requires number inputs");
  }

  if (a.value < 0 || b.value < 0) {
    throw new RangeError("ADD requires natural numbers (>= 0)");
  }

  return {
    kind: "natural",
    value: a.value + b.value
  };
}
```

---

### Integration Layer 1

#### Feature: HTTP API - Compile Endpoint

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ POST /api/v1/compile validates program
- ✓ Returns validation errors or success
- ✓ <100ms for 100-node programs

**Required Tests:**
- [ ] Test: Valid program returns { success: true }
- [ ] Test: Invalid program (cycle) returns errors array
- [ ] Test: Malformed JSON returns 400
- [ ] Benchmark: <100ms for 100 nodes

**Implementation Notes:**
```typescript
// packages/http-api/src/routes/compile.ts
import { Elysia, t } from "elysia";
import { Compiler } from "@dataflow/compiler";

export const compileRoute = (app: Elysia) =>
  app.post("/api/v1/compile", async ({ body }) => {
    const compiler = new Compiler();
    const result = compiler.compile(body.program);
    return result;
  }, {
    body: t.Object({
      program: t.Any()
    }),
    response: {
      200: t.Object({
        success: t.Boolean(),
        programId: t.String(),
        errors: t.Array(t.Any()),
        warnings: t.Array(t.Any())
      })
    }
  });
```

---

#### Feature: HTTP API - Execute Endpoint

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ POST /api/v1/execute runs program
- ✓ Returns outputs + execution trace
- ✓ <50ms for 50-node programs

**Required Tests:**
- [ ] Test: Execute "3 + 2" returns outputs: [5]
- [ ] Test: Returns execution order
- [ ] Test: Returns cache statistics
- [ ] Benchmark: <50ms for 50 nodes

**Implementation Notes:**
```typescript
// packages/http-api/src/routes/execute.ts
export const executeRoute = (app: Elysia) =>
  app.post("/api/v1/execute", async ({ body }) => {
    const compiler = new Compiler();
    const { graph } = compiler.compile(body.program);
    const evaluator = new DemandDrivenEvaluator();

    const outputs: unknown[] = [];
    const trace: string[] = [];

    for (const node of graph.getOutputNodes()) {
      const value = evaluator.evaluate(node.id, 0, graph);
      outputs.push(value);
    }

    return {
      success: true,
      outputs,
      trace: {
        executionOrder: trace,
        ...evaluator.getCacheStats()
      }
    };
  });
```

---

## Layer 2: Arithmetic Operations

**Estimated Effort:** 2-3 days
**Prerequisites:** Layer 1 complete and passing
**Success Criteria:** "(3 + 2) * (10 - 6) = 20" works

---

### Features

#### Feature: SUBTRACT Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ SUBTRACT(natural, natural) → integer
- ✓ Returns difference (may be negative)
- ✓ Type-safe inputs

**Required Tests:**
- [ ] Test: SUBTRACT(10, 6) returns 4 (integer)
- [ ] Test: SUBTRACT(5, 10) returns -5 (integer)
- [ ] Test: SUBTRACT(3, 3) returns 0

**Implementation Notes:**
```typescript
// packages/runtime/src/operations/numeric.ts
export function SUBTRACT(inputs: { value: unknown }[]): Integer {
  const [a, b] = inputs as { value: number }[];
  return {
    kind: "integer",
    value: a.value - b.value
  };
}
```

---

#### Feature: MULTIPLY Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ MULTIPLY(natural, natural) → natural
- ✓ Returns product

**Required Tests:**
- [ ] Test: MULTIPLY(3, 4) returns 12
- [ ] Test: MULTIPLY(0, 5) returns 0
- [ ] Test: MULTIPLY(1, 10) returns 10

---

#### Feature: DIVIDE Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ DIVIDE(natural, natural) → decimal
- ✓ Returns quotient
- ✓ Division by zero error

**Required Tests:**
- [ ] Test: DIVIDE(10, 2) returns 5.0
- [ ] Test: DIVIDE(7, 2) returns 3.5
- [ ] Test: DIVIDE(5, 0) throws DivisionByZeroError

---

#### Feature: COMPARE Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ COMPARE(natural, natural) → integer (-1, 0, 1)
- ✓ Returns -1 if a < b
- ✓ Returns 0 if a == b
- ✓ Returns 1 if a > b

**Required Tests:**
- [ ] Test: COMPARE(3, 5) returns -1
- [ ] Test: COMPARE(5, 5) returns 0
- [ ] Test: COMPARE(7, 3) returns 1

---

#### Feature: Operation Registry Expansion

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ All arithmetic operations registered
- ✓ Type signatures defined
- ✓ Arity validation

**Required Tests:**
- [ ] Test: All operations in registry
- [ ] Test: Arity validation catches wrong number of inputs
- [ ] Test: Type signature retrieval works

**Implementation Notes:**
```typescript
// packages/compiler/src/operations/registry.ts
export const OPERATION_REGISTRY: Record<string, OperationSignature> = {
  ADD: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "natural",
    category: "numeric"
  },
  SUBTRACT: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "integer",
    category: "numeric"
  },
  MULTIPLY: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "natural",
    category: "numeric"
  },
  DIVIDE: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "decimal",
    category: "numeric"
  },
  COMPARE: {
    arity: 2,
    inputTypes: ["natural", "natural"],
    outputType: "integer",
    category: "comparison"
  }
};
```

---

### Type Checking Layer 2

#### Feature: Arity Checker

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Validates each operation receives correct number of inputs
- ✓ Returns specific error for wrong arity
- ✓ Checks all transformation nodes

**Required Tests:**
- [ ] Test: ADD with 2 inputs passes
- [ ] Test: ADD with 1 input fails arity check
- [ ] Test: ADD with 3 inputs fails arity check

**Implementation Notes:**
```typescript
// packages/compiler/src/validation/arity-checker.ts
export class ArityChecker {
  check(ast: Program): ValidationResult {
    const errors: ValidationError[] = [];

    for (const stmt of ast.statements) {
      if (stmt.type === "Transformation") {
        const signature = OPERATION_REGISTRY[stmt.operation];
        if (!signature) {
          errors.push({
            code: "UNKNOWN_OPERATION",
            message: `Unknown operation: ${stmt.operation}`,
            nodeId: stmt.id
          });
          continue;
        }

        if (stmt.inputs.length !== signature.arity) {
          errors.push({
            code: "WRONG_ARITY",
            message: `Operation ${stmt.operation} requires ${signature.arity} inputs, got ${stmt.inputs.length}`,
            nodeId: stmt.id
          });
        }
      }
    }

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    };
  }
}
```

---

## Layer 3: Curriculum Types (Shape, Car)

**Estimated Effort:** 3-4 days
**Prerequisites:** Layer 2 complete and passing
**Success Criteria:** Compare shapes by color works

---

### Features

#### Feature: Shape Type Definition

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Shape has type, size, color
- ✓ Type: circle, triangle, square, rectangle
- ✓ Size: small, medium, large
- ✓ Color: red, blue, yellow, green, orange, purple

**Required Tests:**
- [ ] Test: Valid shape literal parses
- [ ] Test: Invalid shape type rejected
- [ ] Test: Invalid size rejected
- [ ] Test: Invalid color rejected

**Implementation Notes:**
```typescript
// packages/shared/src/types/curriculum.ts
export type ShapeType = "circle" | "triangle" | "square" | "rectangle";
export type Size = "small" | "medium" | "large";
export type Color = "red" | "blue" | "yellow" | "green" | "orange" | "purple";

export type Shape = {
  kind: "shape";
  type: ShapeType;
  size: Size;
  color: Color;
};
```

---

#### Feature: Car Type Definition

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Car has color
- ✓ Same color values as Shape

**Required Tests:**
- [ ] Test: Valid car literal parses
- [ ] Test: Invalid color rejected

---

#### Feature: COMPARE_BY_COLOR Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ COMPARE_BY_COLOR(shape, shape) → integer
- ✓ Compares by color property
- ✓ Returns -1, 0, or 1

**Required Tests:**
- [ ] Test: COMPARE_BY_COLOR(red_shape, blue_shape) returns comparison
- [ ] Test: Same color returns 0
- [ ] Test: Works with cars too

---

#### Feature: COMPARE_BY_SIZE Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ COMPARE_BY_SIZE(shape, shape) → integer
- ✓ Orders: small < medium < large

**Required Tests:**
- [ ] Test: COMPARE_BY_SIZE(small_shape, large_shape) returns -1
- [ ] Test: Same size returns 0
- [ ] Test: large > medium > small

---

#### Feature: FILTER_BY_COLOR Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ FILTER_BY_COLOR(set<shape>, text) → set<shape>
- ✓ Returns set of shapes matching color
- ✓ Original set unmodified

**Required Tests:**
- [ ] Test: Filters red shapes from mixed set
- [ ] Test: Returns empty set when no matches
- [ ] Test: Original set unchanged (immutability)
- [ ] Test: Works with car set too

**Implementation Notes:**
```typescript
// packages/runtime/src/operations/filtering.ts
export function FILTER_BY_COLOR(inputs: { value: unknown }[]): Set<Shape> {
  const [shapes, colorText] = inputs as [{ value: Shape[] }, { value: string }];
  const color = colorText.value as Color;

  return {
    kind: "set",
    elementType: "shape",
    elements: shapes.value.filter(s => s.color === color)
  };
}
```

---

## Layer 4: Set Operations

**Estimated Effort:** 3-4 days
**Prerequisites:** Layer 3 complete and passing
**Success Criteria:** Filter red shapes and union sets works

---

### Features

#### Feature: Set Type Definition

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Set<T> with elementType
- ✓ Array of elements
- ✓ Type homogeneity enforced

**Required Tests:**
- [ ] Test: Valid set literal parses
- [ ] Test: Set type correctly inferred
- [ ] Test: Mixed-type set rejected

---

#### Feature: UNION Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ UNION(set<T>, set<T>) → set<T>
- ✓ Combines elements from both sets
- ✓ Removes duplicates

**Required Tests:**
- [ ] Test: UNION of disjoint sets combines all elements
- [ ] Test: UNION removes duplicates
- [ ] Test: Type mismatch rejected

---

#### Feature: INTERSECTION Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ INTERSECTION(set<T>, set<T>) → set<T>
- ✓ Returns elements in both sets

**Required Tests:**
- [ ] Test: INTERSECTION returns common elements
- [ ] Test: Empty intersection when no common elements

---

#### Feature: DIFFERENCE Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ DIFFERENCE(set<T>, set<T>) → set<T>
- ✓ Returns elements in first but not second

**Required Tests:**
- [ ] Test: DIFFERENCE returns unique elements from first
- [ ] Test: Empty difference when sets equal

---

#### Feature: COMPLEMENT Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ COMPLEMENT(set<T>, set<T>) → set<T>
- ✓ Returns elements in universe but not in set

**Required Tests:**
- [ ] Test: COMPLEMENT returns elements from universe not in set
- [ ] Test: Empty complement when set equals universe

---

#### Feature: SORT Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ SORT(list<T>) → list<T>
- ✓ Orders elements naturally

**Required Tests:**
- [ ] Test: SORT orders naturals ascending
- [ ] Test: SORT returns new list (immutable)

---

## Layer 5: Temporal Operators

**Estimated Effort:** 3-4 days
**Prerequisites:** Layer 4 complete and passing
**Success Criteria:** Counter "0, 1, 2, 3, 4" works

---

### Features

#### Feature: Time Parameter in Evaluator

**From Acceptance Criteria (DEMAND_DRIVEN_INCREMENTAL.md):**
- ✓ evaluate(nodeId, time) accepts time parameter
- ✓ Cache indexed by time
- ✓ Temporal operations use time correctly

**Required Tests:**
- [ ] Test: Same node at different times cached separately
- [ ] Test: Cache hit respects time parameter

---

#### Feature: FBY Operation (Followed By)

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ FBY(initial, stream<T>) → stream<T>
- ✓ Returns initial at time 0, stream values thereafter
- ✓ Pure function of time (no mutable state)

**Required Tests:**
- [ ] Test: FBY(0, ADD(counter, 1)) at t=0 returns 0
- [ ] Test: FBY(0, ADD(counter, 1)) at t=1 returns 1
- [ ] Test: FBY(0, ADD(counter, 1)) at t=4 returns 4
- [ ] Benchmark: Counter sequence correct

**Implementation Notes:**
```typescript
// packages/runtime/src/operations/temporal.ts
export function FBY(inputs: { value: unknown }[], time: number): unknown {
  const [initial, stream] = inputs;

  if (time === 0) {
    return initial.value;
  }

  // This creates the recursive reference
  // The stream input refers to the same node
  return stream.value;
}

// In evaluator:
case "FBY":
  return (time === 0)
    ? evaluate(inputs[0].id, time)
    : evaluate(inputs[1].id, time - 1);
```

---

#### Feature: NEXT Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ NEXT(stream<T>) → T
- ✓ Returns next value from stream

**Required Tests:**
- [ ] Test: NEXT returns value at time + 1
- [ ] Test: NEXT at end of stream behavior

---

#### Feature: FIRST Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ FIRST(stream<T>) → T
- ✓ Returns first value from stream

**Required Tests:**
- [ ] Test: FIRST returns value at time 0

---

#### Feature: ACCUMULATE Operation

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ ACCUMULATE(stream<T>, op, initial) → stream<T>
- ✓ Applies operation cumulatively

**Required Tests:**
- [ ] Test: ACCUMULATE with ADD creates running sum
- [ ] Test: ACCUMULATE with MULTIPLY creates running product

---

## Layer 6: Streams

**Estimated Effort:** 2-3 days
**Prerequisites:** Layer 5 complete and passing
**Success Criteria:** Stream of sensor events works

---

### Features

#### Feature: Stream Type Definition

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Stream<T> with elementType
- ✓ Generator function
- ✓ Infinite sequences supported

**Required Tests:**
- [ ] Test: Stream type correctly parsed
- [ ] Test: Generator produces values

---

#### Feature: Stream Sources

**From Acceptance Criteria (GRAMMAR_SPEC.md):**
- ✓ sensor(name) source
- ✓ generator(name) source
- ✓ external(name) source

**Required Tests:**
- [ ] Test: Stream source parses correctly
- [ ] Test: Sensor stream generates events

---

#### Feature: Stream Operations Inheritance

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ All operations of type T apply to Stream<T>
- ✓ Element-wise application
- ✓ Type preserved

**Required Tests:**
- [ ] Test: ADD on Stream<Natural> returns Stream<Natural>
- [ ] Test: FILTER_BY_COLOR on Stream<Shape> returns Stream<Shape>

---

## Layer 7: Integration Interfaces

**Estimated Effort:** 4-5 days
**Prerequisites:** Layer 6 complete and passing
**Success Criteria:** External systems can compile/execute programs

---

### Layer 7a: HTTP API (Batch Mode)

#### Feature: Health Endpoint

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ GET /api/v1/health
- ✓ Returns status, version, uptime

**Required Tests:**
- [ ] Test: Health endpoint returns 200
- [ ] Test: Response includes all required fields

---

#### Feature: Compile Endpoint (Full)

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ POST /api/v1/compile validates program
- ✓ Returns detailed validation errors
- ✓ <100ms for 100-node programs (p95)

**Required Tests:**
- [ ] Test: Valid program returns success
- [ ] Test: Cycle detection works
- [ ] Test: Type errors caught
- [ ] Benchmark: <100ms for 100 nodes

---

#### Feature: Execute Endpoint (Full)

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ POST /api/v1/execute runs program
- ✓ Returns outputs + trace
- ✓ <50ms for 50-node programs (p95)
- ✓ Handles 5 concurrent requests

**Required Tests:**
- [ ] Test: Execute returns correct outputs
- [ ] Test: Trace includes execution order
- [ ] Test: Concurrent requests handled
- [ ] Benchmark: <50ms for 50 nodes

---

### Layer 7b: WebSocket Server (Live Mode)

#### Feature: WebSocket Connection

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ Clients connect via ws://localhost:3000/live
- ✓ Multiple simultaneous connections
- ✓ Connection resilient (reconnects)

**Required Tests:**
- [ ] Test: Client connects successfully
- [ ] Test: 5 concurrent clients connected
- [ ] Test: Client disconnect and reconnect

---

#### Feature: Validate Program Message

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ validate_program message handled
- ✓ Returns validation_result with messageId
- ✓ Message roundtrip <50ms (p95)

**Required Tests:**
- [ ] Test: Validate message returns correct response
- [ ] Test: Validation errors included
- [ ] Test: messageId preserved
- [ ] Benchmark: <50ms roundtrip

---

#### Feature: Evaluate Incremental Message

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ evaluate_incremental message handled
- ✓ Returns evaluation_result with node states
- ✓ Partial graphs accepted

**Required Tests:**
- [ ] Test: Evaluate partial program
- [ ] Test: Returns completed and pending states
- [ ] Test: Missing inputs shown in reason

---

#### Feature: Subscribe/Unsubscribe Messages

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ subscribe_node creates demand
- ✓ unsubscribe_node removes demand
- ✓ Idempotent subscription

**Required Tests:**
- [ ] Test: Subscribe creates subscription
- [ ] Test: Duplicate subscribe idempotent
- [ ] Test: Unsubscribe removes subscription

---

#### Feature: Push Notifications

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ node_state_changed pushes updates
- ✓ Subscribers receive state changes
- ✓ No memory leaks on disconnect

**Required Tests:**
- [ ] Test: Subscriber receives state change
- [ ] Test: Non-subscriber doesn't receive update
- [ ] Test: Disconnect cleanup

---

### Layer 7c: Incremental Runtime

#### Feature: Partial Graph Evaluation

**From Acceptance Criteria (DEMAND_DRIVEN_INCREMENTAL.md):**
- ✓ Evaluates partial graphs
- ✓ Returns completed/pending states
- ✓ Demand-driven (not eager)

**Required Tests:**
- [ ] Test: Partial graph with single source evaluates
- [ ] Test: Partial graph with missing inputs shows pending
- [ ] Test: No output nodes → no evaluation (demand-driven)

---

#### Feature: Graph Update API

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ updateGraph(delta) applies changes
- ✓ Cache invalidation for affected nodes
- ✓ <10ms (p95) for updates

**Required Tests:**
- [ ] Test: Add node updates graph
- [ ] Test: Remove node invalidates cache
- [ ] Test: Update edge invalidates dependent nodes
- [ ] Benchmark: <10ms update

---

#### Feature: Incremental Update Performance

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ Incremental update 5x faster than full re-eval
- ✓ Only re-evaluates affected nodes

**Required Tests:**
- [ ] Benchmark: Incremental vs full evaluation
- [ ] Test: Only dependent nodes re-evaluated

---

## Additional Curriculum Types

**Estimated Effort:** 2-3 days
**Prerequisites:** Layer 4 complete

---

### Features

#### Feature: Food Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Food has taste and color
- ✓ Taste: sweet, salty, sour, bitter
- ✓ COMPARE_BY_TASTE operation
- ✓ FILTER_BY_TASTE operation

**Required Tests:**
- [ ] Test: Valid food literal parses
- [ ] Test: COMPARE_BY_TASTE works
- [ ] Test: FILTER_BY_TASTE works

---

#### Feature: Animal Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Animal has type and color
- ✓ Type: dog, cat, bird, fish, rabbit, turtle
- ✓ COMPARE_BY_TYPE operation
- ✓ FILTER_BY_TYPE operation

**Required Tests:**
- [ ] Test: Valid animal literal parses
- [ ] Test: COMPARE_BY_TYPE works
- [ ] Test: FILTER_BY_TYPE works

---

#### Feature: Person Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Person has ageGroup and gender
- ✓ AgeGroup: child, teenager, adult, senior
- ✓ Gender: male, female
- ✓ COMPARE_BY_AGE_GROUP operation
- ✓ COMPARE_BY_GENDER operation

**Required Tests:**
- [ ] Test: Valid person literal parses
- [ ] Test: COMPARE_BY_AGE_GROUP works
- [ ] Test: COMPARE_BY_GENDER works

---

## Additional Primitive Types

**Estimated Effort:** 2 days
**Prerequisites:** Layer 2 complete

---

### Features

#### Feature: Integer Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Integer type with value (positive, negative, zero)
- ✓ All arithmetic operations on integers

**Required Tests:**
- [ ] Test: Valid integer literal parses
- [ ] Test: ADD on integers works
- [ ] Test: SUBTRACT on integers works

---

#### Feature: Decimal Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Decimal type with value (floating-point)
- ✓ All arithmetic operations on decimals

**Required Tests:**
- [ ] Test: Valid decimal literal parses
- [ ] Test: ADD on decimals works
- [ ] Test: DIVIDE on decimals works

---

#### Feature: Fraction Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Fraction with numerator and denominator
- ✓ All arithmetic operations on fractions

**Required Tests:**
- [ ] Test: Valid fraction literal parses
- [ ] Test: ADD on fractions works (common denominator)
- [ ] Test: MULTIPLY on fractions works

---

#### Feature: Text Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Text (string) type
- ✓ COMPARE operation
- ✓ ALPHABETICAL_SORT operation

**Required Tests:**
- [ ] Test: Valid text literal parses
- [ ] Test: COMPARE works on strings
- [ ] Test: ALPHABETICAL_SORT works

---

#### Feature: Boolean Type

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Boolean type with value
- ✓ AND, OR, NOT operations
- ✓ COMPARE operation

**Required Tests:**
- [ ] Test: Valid boolean literal parses
- [ ] Test: AND works correctly
- [ ] Test: OR works correctly
- [ ] Test: NOT works correctly

---

## Additional Operations

**Estimated Effort:** 2-3 days
**Prerequisites:** All types implemented

---

### Features

#### Feature: FILTER Operations

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ FILTER (general predicate)
- ✓ FILTER_BY_SIZE
- ✓ FILTER_BY_TASTE
- ✓ FILTER_BY_AGE_GROUP
- ✓ FILTER_BY_GENDER

**Required Tests:**
- [ ] Test: FILTER with custom predicate
- [ ] Test: FILTER_BY_SIZE works
- [ ] Test: FILTER_BY_TASTE works

---

#### Feature: Ordering Operations

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ ALPHABETICAL_SORT on text
- ✓ SORT on comparable types

**Required Tests:**
- [ ] Test: ALPHABETICAL_SORT works
- [ ] Test: SORT works on naturals

---

## Validation & Type Checking

**Estimated Effort:** 2-3 days
**Prerequisites:** Operations implemented

---

### Features

#### Feature: Property Requirements Checker

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ FILTER_BY_COLOR requires 'color' property
- ✓ FILTER_BY_SIZE requires 'size' property
- ✓ Type-specific property validation

**Required Tests:**
- [ ] Test: FILTER_BY_COLOR on naturals rejected
- [ ] Test: FILTER_BY_COLOR on shapes accepted
- [ ] Test: Property requirements enforced

---

#### Feature: Type Compatibility Checker

**From Acceptance Criteria (LANGUAGE_SPEC.md):**
- ✓ Inputs match operation's type signature
- ✓ Set element types compatible
- ✓ Stream element types preserved

**Required Tests:**
- [ ] Test: ADD(natural, integer) rejected
- [ ] Test: UNION(set<shape>, set<car>) rejected
- [ ] Test: UNION(set<shape>, set<shape>) accepted

---

## Performance Benchmarks

**Estimated Effort:** 1-2 days
**Prerequisites:** All layers implemented

---

### Features

#### Feature: Compiler Performance

**From Acceptance Criteria (PROJECT_GOALS.md):**
- ✓ Compilation <100ms for 100-node programs (p95)
- ✓ Handles 5 concurrent compilations

**Required Tests:**
- [ ] Benchmark: 100-node program compilation
- [ ] Benchmark: 5 concurrent compilations

---

#### Feature: Runtime Performance

**From Acceptance Criteria (PROJECT_GOALS.md):**
- ✓ Evaluation <10ms for 50-node programs (p95)
- ✓ Demand-driven verified (cache hits > 0)

**Required Tests:**
- [ ] Benchmark: 50-node program evaluation
- [ ] Verify cache hit ratio

---

#### Feature: Integration Performance

**From Acceptance Criteria (INTEGRATION_SPEC.md):**
- ✓ HTTP API endpoints meet latency targets
- ✓ WebSocket message roundtrip <50ms (p95)
- ✓ Incremental update <10ms (p95)

**Required Tests:**
- [ ] Benchmark: /compile endpoint
- [ ] Benchmark: /execute endpoint
- [ ] Benchmark: WebSocket roundtrip
- [ ] Benchmark: Incremental update

---

## Documentation

**Estimated Effort:** 2-3 days

---

### Features

#### Feature: API Documentation

**Required:**
- [ ] OpenAPI/Swagger spec for HTTP API
- [ ] WebSocket protocol documentation
- [ ] Example programs

---

#### Feature: Developer Documentation

**Required:**
- [ ] Architecture overview
- [ ] Setup instructions
- [ ] Contribution guidelines

---

## Testing Strategy

### Ralph Wiggum Checklist

Before moving to next layer:
- [ ] New functionality fully implemented (no stubs)
- [ ] New tests pass
- [ ] Previous layer tests still pass
- [ ] Typecheck passes (`bun run typecheck`)
- [ ] Lint passes (`bun run lint`)
- [ ] Git committed (atomic commits)

### Test Categories

1. **Unit Tests**: Individual operations, validators, AST nodes
2. **Integration Tests**: End-to-end program compilation and execution
3. **Benchmarks**: Performance requirements verified
4. **Regression Tests**: Ensure previous layers continue working

### Test Execution

```bash
# Run all tests
bun test

# Run specific layer
bun test -- layer1
bun test -- layer2

# Watch mode
bun test -- --watch

# Coverage report
bun test -- --coverage
```

---

## Git Commit Conventions

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org):

```
feat: implement ADD operation
fix: resolve cache invalidation bug
docs: update API documentation
test: add benchmarks for layer 2
refactor: extract shared types to packages/shared
```

Add to commit footer:
```
Co-authored-by: Ralph (OpenCode Agent)
```

---

## Risk Mitigation

### Known Challenges

1. **Demand-Driven Semantics**: Subtle and easy to get wrong
   - Mitigation: Write tests specifically verifying cache behavior
   - Verify: Cache hit count > 0 for multi-node programs

2. **Temporal Operators**: Must be pure functions of time
   - Mitigation: No mutable state in runtime
   - Verify: FBY counter produces correct sequence

3. **Type System**: Integrated types with many operations
   - Mitigation: Strict TypeScript, operation registry
   - Verify: All operations type-checked at compile-time

4. **Performance**: Educational context but need reasonable speed
   - Mitigation: Benchmark each layer, optimize bottlenecks
   - Verify: All benchmarks pass before layer completion

---

## Open Questions for User

1. **Priority Operations**: Which 10-15 operations are most critical for MVP?
2. **Execution Modes**: Both batch and incremental required?
3. **Error Handling**: Fail compilation or best-effort execution?
4. **Performance Targets**: Specific latency requirements?

---

## Appendix: Complete Feature Checklist

### Layer 1
- [ ] Monorepo structure
- [ ] Shared type system
- [ ] JSON input schema
- [ ] Lexer (Chevrotain)
- [ ] Parser (Chevrotain)
- [ ] AST builder
- [ ] DAG validator
- [ ] DataflowGraph structure
- [ ] Demand-driven evaluator
- [ ] ADD operation
- [ ] HTTP API - compile endpoint
- [ ] HTTP API - execute endpoint

### Layer 2
- [ ] SUBTRACT operation
- [ ] MULTIPLY operation
- [ ] DIVIDE operation
- [ ] COMPARE operation
- [ ] Operation registry expansion
- [ ] Arity checker

### Layer 3
- [ ] Shape type definition
- [ ] Car type definition
- [ ] COMPARE_BY_COLOR operation
- [ ] COMPARE_BY_SIZE operation
- [ ] FILTER_BY_COLOR operation

### Layer 4
- [ ] Set type definition
- [ ] UNION operation
- [ ] INTERSECTION operation
- [ ] DIFFERENCE operation
- [ ] COMPLEMENT operation
- [ ] SORT operation

### Layer 5
- [ ] Time parameter in evaluator
- [ ] FBY operation
- [ ] NEXT operation
- [ ] FIRST operation
- [ ] ACCUMULATE operation

### Layer 6
- [ ] Stream type definition
- [ ] Stream sources
- [ ] Stream operations inheritance

### Layer 7a
- [ ] Health endpoint
- [ ] Compile endpoint (full)
- [ ] Execute endpoint (full)

### Layer 7b
- [ ] WebSocket connection
- [ ] Validate program message
- [ ] Evaluate incremental message
- [ ] Subscribe/unsubscribe messages
- [ ] Push notifications

### Layer 7c
- [ ] Partial graph evaluation
- [ ] Graph update API
- [ ] Incremental update performance

### Additional Types
- [ ] Food type
- [ ] Animal type
- [ ] Person type
- [ ] Integer type
- [ ] Decimal type
- [ ] Fraction type
- [ ] Text type
- [ ] Boolean type

### Additional Operations
- [ ] FILTER operations (general)
- [ ] FILTER_BY_SIZE
- [ ] FILTER_BY_TASTE
- [ ] FILTER_BY_AGE_GROUP
- [ ] FILTER_BY_GENDER
- [ ] ALPHABETICAL_SORT
- [ ] SORT (additional types)

### Validation
- [ ] Property requirements checker
- [ ] Type compatibility checker

### Performance & Documentation
- [ ] Compiler benchmarks
- [ ] Runtime benchmarks
- [ ] Integration benchmarks
- [ ] API documentation
- [ ] Developer documentation

---

**Document Status:** Ready for implementation
**Next Step:** Begin Layer 1 implementation with setup tasks
