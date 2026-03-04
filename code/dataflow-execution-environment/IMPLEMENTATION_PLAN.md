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

## Current Implementation Status

**Last Updated:** 2026-03-04
**Analysis Date:** 2026-03-04

### Overall Progress: ~45% Complete

| Layer | Status | Completion | Tests |
|-------|--------|------------|--------|
| Layer 1: Foundation | PARTIAL | 90% | 0% |
| Layer 2: Arithmetic | PARTIAL | 80% | 0% |
| Layer 3: Curriculum Types | MISSING | 10% | 0% |
| Layer 4: Set Operations | PARTIAL | 20% | 0% |
| Layer 5: Temporal Operators | PARTIAL | 20% | 0% |
| Layer 6: Streams | MISSING | 10% | 0% |
| Layer 7: Integration | MISSING | 0% | 0% |

### ✅ COMPLETE (Core Infrastructure)

**Monorepo & Build System:**
- ✓ Bun workspaces configured
- ✓ TypeScript strict mode
- ✓ All packages build successfully
- ⚠️ Build script has infinite recursion (needs fix)

**Shared Types (packages/shared/src/types/):**
- ✓ Primitive types: Natural, Integer, Decimal, Text, Boolean
- ✓ Curriculum types: Shape, Car, Food, Animal, Person
- ✓ Composite types: Set, Stream
- ✓ Validation types: ValidationError, ValidationResult, MissingInput
- ✓ Program types: DataflowProgram, DataflowNode, DataflowEdge

**Compiler (packages/compiler/src/):**
- ✓ Lexer (Chevrotain) - All tokens defined
- ✓ Parser (CstParser) - Complete grammar support
- ✓ AST Builder - Visitor pattern implementation
- ✓ DAG Validator - Cycle detection, arity checking
- ✓ Compiler class - Integration of all components
- ✓ Child-friendly Spanish error messages

**Runtime (packages/runtime/src/):**
- ✓ DataflowGraph - Node/edge management
- ✓ DemandDrivenEvaluator - Caching, demand-driven evaluation
- ✓ Numeric operations: ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE
- ✓ Runtime class - Program loading and execution

**Operation Registry (packages/shared/src/operations/):**
- ✓ All 15 operations with signatures defined
- ✓ getOperationSignature() utility
- ✓ isOperation() utility

### ❌ MISSING (Critical Blockers)

**Layer 1 (Foundation):**
- ❌ HTTP API (packages/http-api/) - Package exists, directory empty
- ❌ WebSocket Server (packages/websocket-server/) - Package exists, directory empty
- ❌ Any test files - Zero tests across all packages
- ❌ TypeScript typecheck configured - tsc command not available

**Layer 2 (Arithmetic):**
- ❌ Tests for SUBTRACT, MULTIPLY, DIVIDE, COMPARE

**Layer 3 (Curriculum Types):**
- ❌ COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE implementations
- ❌ FILTER_BY_SIZE, FILTER_BY_COLOR, FILTER_BY_TYPE implementations
- ❌ Property constraint validation (hasProperty)
- ❌ Type checking for curriculum types

**Layer 4 (Set Operations):**
- ❌ UNION, INTERSECTION, DIFFERENCE, COMPLEMENT implementations
- ❌ Type checking for set operations

**Layer 5 (Temporal Operators):**
- ❌ NEXT, FIRST, FBY, ACCUMULATE implementations
- ❌ Time parameter support in evaluator
- ❌ Stream value handling

**Layer 6 (Streams):**
- ❌ Stream generator support
- ❌ Stream type evaluation

**Layer 7 (Integration):**
- ❌ IncrementalRuntime class
- ❌ Partial graph evaluation
- ❌ Subscription management
- ❌ Node state tracking (completed/pending/error)

### ⚠️ PARTIAL (In Progress)

**Operation Implementation Gap:**
- 9 operations in registry but NOT implemented (will crash at runtime):
  - FILTER, UNION, INTERSECTION, DIFFERENCE, COMPLEMENT
  - NEXT, FIRST, FBY, ACCUMULATE, SORT
- Runtime evaluator throws "Unknown operation" error for these

**Validation Gap:**
- Arity checking ✓ implemented
- Type compatibility ❌ missing
- Property constraints ❌ missing

### 🐛 BUGS & ISSUES

1. **Build Script Infinite Loop** (packages/package.json:8)
   - `bun run build --filter='*'` causes recursion
   - Workaround: Build per-package individually

2. **TypeScript Not Available** (packages/package.json:11)
   - `tsc --noEmit` fails (command not found)
   - Type checking not enforced

3. **Lint Not Configured** (packages/package.json:12)
   - Script just echoes "Lint not configured"

4. **COMPARE Type Mismatch** (packages/runtime/src/operations/numeric.ts:67)
   - Returns Natural type, spec says Integer type
   - Fix needed for correctness

### 📊 Test Coverage Summary

| Package | Test Files | Tests Passing |
|---------|------------|---------------|
| @dataflow/shared | 0 | N/A |
| @dataflow/compiler | 0 | N/A |
| @dataflow/runtime | 0 | N/A |
| @dataflow/http-api | 0 | N/A |
| @dataflow/websocket-server | 0 | N/A |
| **TOTAL** | **0** | **0%** |

**Estimated Test Count Needed:** ~270 tests (from IMPLEMENTATION_PLAN.md checklist)

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
          nodeId: stmt.id,
          childMessage: `⚠️ ¡Ups! Ya tienes un bloque llamado "${stmt.id}". Cada bloque necesita un nombre único.`,
          suggestion: `Cambia el nombre de uno de los bloques "${stmt.id}"`,
          example: `[${stmt.id}_1] y [${stmt.id}_2] ✅`
        });
      }
      defined.add(stmt.id);
    }

    // Detect cycles
    const hasCycle = this.detectCycles(ast);
    if (hasCycle) {
      errors.push({
        code: "CYCLE_DETECTED",
        message: "Graph contains a cycle",
        childMessage: "⚠️ ¡Ups! Hay un ciclo en el programa. Los bloques se conectan en círculo y no se puede calcular.",
        suggestion: "Busca dónde un bloque apunta a sí mismo y desconecta una línea.",
        example: "[A] → [B] → [C] ❌\n[A] → [B] ✅ (rompe el ciclo)"
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
- ✓ Returns validation errors or success with child-friendly messages
- ✓ <100ms for 100-node programs

**Required Tests:**
- [ ] Test: Valid program returns { success: true }
- [ ] Test: Invalid program (cycle) returns errors array with childMessage
- [ ] Test: Malformed JSON returns 400
- [ ] Test: Error messages in Spanish are child-friendly
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

// Response now includes child-friendly Spanish messages:
// {
//   "success": false,
//   "programId": "prog_001",
//   "errors": [
//     {
//       "code": "WRONG_ARITY",
//       "nodeId": "add_1",
//       "message": "Operation ADD requires 2 inputs, got 1",
//       "childMessage": "⚠️ ¡Ups! El bloque \"+\" necesita 2 números conectados. Solo conectaste 1 número.",
//       "suggestion": "Conecta otro bloque número al \"+\"",
//       "example": "[5] + [3] ✅"
//     }
//   ],
//   "warnings": []
// }
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
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! No reconozco el bloque "${stmt.operation}".`,
            suggestion: `Revisa si el bloque está conectado correctamente.`,
            example: `Usa un bloque conocido como "+", "-", "×", "÷"`
          });
          continue;
        }

        if (stmt.inputs.length !== signature.arity) {
          errors.push({
            code: "WRONG_ARITY",
            message: `Operation ${stmt.operation} requires ${signature.arity} inputs, got ${stmt.inputs.length}`,
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! El bloque "${stmt.operation}" necesita ${signature.arity} ${signature.arity === 1 ? 'entrada' : 'entradas'}. Solo conectaste ${stmt.inputs.length}.`,
            suggestion: `Conecta ${signature.arity - stmt.inputs.length} bloque${signature.arity - stmt.inputs.length === 1 ? '' : 's'} más al bloque "${stmt.operation}".`,
            example: this.generateExample(stmt.operation, signature.arity)
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

  private generateExample(operation: string, arity: number): string {
    // Generate visual example based on operation type
    if (operation === "ADD") {
      return "[5] + [3] ✅";
    }
    if (operation === "FILTER_BY_COLOR") {
      return "[formas] 🎨 [rojo] ✅";
    }
    return `Conecta ${arity} bloques al bloque "${operation}" ✅`;
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

**Implementation Notes:**
```typescript
// packages/websocket-server/src/types/messages.ts
export type NodeState =
  | { status: "completed", value: any }
  | { status: "pending", missingInputs: MissingInput[] }
  | { status: "processing" }
  | { status: "error", error: string };

export type MissingInput = {
  port: number;
  description: string;
  childMessage: string;
};

// Pending state with multiple missing inputs:
// {
//   "type": "evaluation_result",
//   "nodeStates": {
//     "filter_1": {
//       "status": "pending",
//       "missingInputs": [
//         {
//           "port": 0,
//           "description": "set of shapes",
//           "childMessage": "Esperando un conjunto de formas"
//         },
//         {
//           "port": 1,
//           "description": "color",
//           "childMessage": "Esperando una tarjeta de color"
//         }
//       ]
//     }
//   }
// }
```

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
- [ ] Test: Multiple missing inputs reported in missingInputs array
- [ ] Test: Missing input messages in Spanish

**Implementation Notes:**
```typescript
// packages/runtime/src/incremental/incremental-runtime.ts
export class IncrementalRuntime {
  private partialGraph: DataflowGraph;
  private cache: Map<string, Map<number, NodeEvaluation>>;
  private subscriptions: Map<string, Set<Subscriber>>();

  /**
   * Update the graph with new/changed/removed nodes or edges
   */
  updateGraph(delta: GraphDelta): UpdateResult {
    // Apply delta to internal graph
    // Invalidate affected cache entries
    // Return list of changed nodes
  }

  /**
   * Evaluate the partial graph as far as possible
   */
  evaluatePartial(timestep: number = 0): PartialEvaluation {
    const results: Record<string, NodeState> = {};

    // Find OUTPUT nodes (demand sources)
    const outputNodes = this.partialGraph.getOutputNodes();

    // If no output nodes, find SUBSCRIBED nodes
    const demandSources = outputNodes.length > 0
      ? outputNodes
      : Array.from(this.subscriptions.keys())
          .map(id => this.partialGraph.nodes.get(id))
          .filter(n => n !== undefined);

    // For each demand source, try to evaluate (demand-driven!)
    for (const source of demandSources) {
      try {
        const value = this.evaluate(source.id, timestep);
        results[source.id] = { status: "completed", value };
      } catch (e) {
        if (e instanceof MissingInputError) {
          // Extract multiple missing inputs
          const missingInputs = this.extractMissingInputs(source);
          results[source.id] = {
            status: "pending",
            missingInputs
          };
        } else {
          results[source.id] = {
            status: "error",
            error: this.formatChildFriendlyError(e)
          };
        }
      }
    }

    return { nodeStates: results };
  }

  /**
   * Extract all missing inputs for a node
   */
  private extractMissingInputs(node: DataflowNode): MissingInput[] {
    const missing: MissingInput[] = [];
    const signature = OPERATION_REGISTRY[node.operation];
    const actualInputs = this.partialGraph.getInputs(node.id);

    for (let i = 0; i < signature.arity; i++) {
      if (!actualInputs[i] || !actualInputs[i].hasValue()) {
        missing.push({
          port: i,
          description: this.getInputDescription(node.operation, i),
          childMessage: this.getChildMessageForMissingInput(node.operation, i)
        });
      }
    }

    return missing;
  }

  /**
   * Get child-friendly Spanish message for missing input
   */
  private getChildMessageForMissingInput(operation: string, port: number): string {
    // Spanish messages for different operation types
    if (operation === "FILTER_BY_COLOR" && port === 1) {
      return "Esperando una tarjeta de color";
    }
    if (operation.startsWith("FILTER_") && port === 1) {
      return "Esperando una tarjeta de criterio";
    }
    if (operation.startsWith("COMPARE_")) {
      return "Esperando un elemento para comparar";
    }
    return "Esperando una entrada";
  }

  /**
   * Get current state of a specific node
   */
  getNodeState(nodeId: string): NodeState {
    // Return cached state or compute if possible
  }

  /**
   * Subscribe to changes on a specific node
   */
  subscribe(nodeId: string, callback: StateCallback): void {
    // Add callback to subscriptions
  }

  /**
   * Unsubscribe from node changes
   */
  unsubscribe(nodeId: string, callback: StateCallback): void {
    // Remove callback from subscriptions
  }
}
```

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

## Decisions from User

1. **Priority Operations**: Layer 1-2 operations confirmed as MVP priority (already aligned with plan)
2. **Execution Modes**: Both batch and incremental evaluation are **required and mandatory**
3. **Error Handling Strategy**:
   - **Batch evaluation**: Compilation should FAIL with clear errors
   - **Incremental evaluation**: Best-effort execution with partial results
   - **Critical requirement**: Failure reasons must be **traceable and clear** for children to understand (intuitive feedback for ages 6-9)
4. **Error Messages**: Spanish only for now (child-friendly)
5. **Error Categorization**: Simple - just errors (no warning/info severity levels)
6. **Pending State Messages**: Factual descriptions supporting multiple missing inputs simultaneously
7. **Performance Targets**: Specified latencies acceptable; can be more flexible for MVP

---

## Child-Friendly Error Messages (CRITICAL REQUIREMENT)

**Educational Rationale**: Children aged 6-9 need clear, intuitive feedback to learn from mistakes. Error messages should:
- Use simple language (no technical jargon)
- Show clear indication of where problem is
- Provide actionable suggestions
- Spanish language only
- Include examples of correct usage

### Error Types with Spanish Messages

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

### Type Definitions

```typescript
// packages/shared/src/types/validation.ts
export type ValidationError = {
  code: string;
  message: string;           // Technical message for developers
  childMessage?: string;      // Simplified message for children (Spanish)
  nodeId?: string;
  suggestion?: string;        // Actionable suggestion (Spanish)
  example?: string;          // Example of correct usage
};

export type MissingInput = {
  port: number;
  description: string;      // e.g., "número", "color", "forma"
  childMessage: string;      // Spanish description
};

export type NodeState =
  | { status: "completed", value: any }
  | { status: "pending", missingInputs: MissingInput[] }
  | { status: "processing" }
  | { status: "error", error: string };
```

### Example API Responses (Spanish)

**HTTP API - Batch Mode**:
```json
{
  "success": false,
  "programId": "prog_001",
  "errors": [
    {
      "code": "WRONG_ARITY",
      "nodeId": "add_1",
      "message": "Operation ADD requires 2 inputs, got 1",
      "childMessage": "⚠️ ¡Ups! El bloque \"+\" necesita 2 números conectados. Solo conectaste 1 número.",
      "suggestion": "Conecta otro bloque número al \"+\"",
      "example": "[5] + [3] ✅"
    }
  ]
}
```

**WebSocket - Incremental Mode (Multiple Missing Inputs)**:
```json
{
  "type": "evaluation_result",
  "nodeStates": {
    "filter_1": {
      "status": "pending",
      "missingInputs": [
        {
          "port": 0,
          "description": "set of shapes",
          "childMessage": "Esperando un conjunto de formas"
        },
        {
          "port": 1,
          "description": "color",
          "childMessage": "Esperando una tarjeta de color"
        }
      ]
    }
  }
}
```

---

## Prioritized Action Plan

**Last Updated:** 2026-03-04

### PHASE 1: Critical Infrastructure Fixes (Week 1)
**Goal:** Unblock basic development workflow

#### Priority 0: Fix Build & Test Infrastructure (1 day)
- [ ] Fix infinite recursion in `bun run build` script
- [ ] Configure TypeScript typechecking to work with bun
- [ ] Set up basic test framework and test directory structure
- [ ] Add simple "sanity check" test that runs successfully

**Files to modify:**
- `/app/package.json` - Fix build script
- `/app/packages/*/package.json` - Add test scripts
- Create `/app/packages/*/test/` directories

**Success criteria:**
- `bun run build` completes without infinite loop
- `bun run test` finds and executes at least 1 test
- `bun run typecheck` works without errors

---

#### Priority 1: Layer 1 HTTP API (2 days)
**Goal:** Enable CV system integration and basic execution

**Tasks:**
- [ ] Create `packages/http-api/src/server.ts` with Elysia setup
- [ ] Implement POST /api/v1/compile endpoint
- [ ] Implement POST /api/v1/execute endpoint
- [ ] Implement GET /api/v1/health endpoint
- [ ] Add child-friendly Spanish error responses
- [ ] Write tests for all endpoints

**Success criteria:**
- Can compile a program via HTTP POST
- Can execute "3 + 2 = 5" via HTTP POST
- Returns proper validation errors for invalid programs
- All endpoints have tests passing

**Impact:** Enables Layer 1 to be fully functional end-to-end

---

#### Priority 2: Layer 1 Tests (2 days)
**Goal:** Verify Layer 1 functionality completely works

**Tasks:**
- [ ] Setup tests for Monorepo structure
- [ ] Tests for Shared type system
- [ ] Tests for JSON input schema
- [ ] Tests for Lexer (tokenization)
- [ ] Tests for Parser (CST generation)
- [ ] Tests for AST Builder
- [ ] Tests for DAG Validator
- [ ] Tests for DataflowGraph
- [ ] Tests for DemandDrivenEvaluator (cache behavior)
- [ ] Tests for ADD operation
- [ ] End-to-end test: "3 + 2 = 5"
- [ ] Performance benchmarks (<100ms compile, <50ms execute)

**Success criteria:**
- All Layer 1 features have test coverage
- Tests verify child-friendly error messages
- Benchmarks pass performance targets
- Ralph Wiggum checklist completed for Layer 1

**Impact:** Confirms Layer 1 is production-ready

---

### PHASE 2: Complete Operation Implementations (Week 2)
**Goal:** Make all registered operations work

#### Priority 3: Complete Operation Implementations (3 days)
**Current status:** 9 operations crash at runtime

**Tasks:**
- [ ] Implement FILTER operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement UNION operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement INTERSECTION operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement DIFFERENCE operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement COMPLEMENT operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement NEXT operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement FIRST operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement FBY operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement ACCUMULATE operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement SORT operation (packages/runtime/src/operations/ordering.ts)
- [ ] Update evaluator to handle all operations (demand-driven-evaluator.ts:47-60)
- [ ] Export new operations from index.ts
- [ ] Write tests for all new operations

**Success criteria:**
- All 15 operations in registry work without crashes
- All operations have tests
- Evaluator dispatches to correct operation

**Impact:** All basic operations usable in programs

---

#### Priority 4: Layer 2 Tests (1 day)
**Goal:** Verify arithmetic operations

**Tasks:**
- [ ] Tests for SUBTRACT (including negative results)
- [ ] Tests for MULTIPLY
- [ ] Tests for DIVIDE (including division by zero)
- [ ] Tests for COMPARE (-1, 0, 1 results)
- [ ] Fix COMPARE return type (Integer vs Natural bug)
- [ ] Performance benchmarks for arithmetic operations

**Success criteria:**
- Layer 2 tests passing
- COMPARE returns correct type
- No regressions in Layer 1

**Impact:** Confirms Layer 2 complete

---

### PHASE 3: Curriculum Types (Week 3)
**Goal:** Enable educational curriculum features

#### Priority 5: Curriculum Type Operations (3 days)
**Tasks:**
- [ ] Implement COMPARE_BY_SIZE operation
- [ ] Implement COMPARE_BY_COLOR operation
- [ ] Implement COMPARE_BY_TYPE operation
- [ ] Implement FILTER_BY_SIZE operation
- [ ] Implement FILTER_BY_COLOR operation
- [ ] Implement FILTER_BY_TYPE operation
- [ ] Add property constraint validation
- [ ] Add type checking for curriculum types
- [ ] Write tests for all curriculum operations

**Success criteria:**
- Curriculum types work with comparison/filtering
- Property constraints validated at compile-time
- Child-friendly Spanish errors for property mismatches

**Impact:** Core educational functionality working

---

### PHASE 4: Integration Layer (Week 4)
**Goal:** Enable live construction mode

#### Priority 6: WebSocket Server (2 days)
**Tasks:**
- [ ] Create `packages/websocket-server/src/server.ts`
- [ ] Implement ws://localhost:3000/live endpoint
- [ ] Handle validate_program messages
- [ ] Handle evaluate_incremental messages
- [ ] Handle subscribe_node/unsubscribe_node messages
- [ ] Implement push notifications (node_state_changed)
- [ ] Add connection resilience (reconnects)
- [ ] Write tests for WebSocket protocol

**Success criteria:**
- IDE can connect via WebSocket
- Validation returns child-friendly Spanish errors
- Incremental evaluation works for partial programs
- Multiple clients can connect

**Impact:** Live feedback during program construction

---

#### Priority 7: Incremental Runtime (2 days)
**Tasks:**
- [ ] Create IncrementalRuntime class (packages/runtime/src/incremental-runtime.ts)
- [ ] Implement partial graph evaluation
- [ ] Implement node state tracking (completed/pending/error)
- [ ] Implement subscription management
- [ ] Implement graph update API with cache invalidation
- [ ] Add missing input extraction with multiple inputs support
- [ ] Add child-friendly Spanish missing input messages
- [ ] Write tests for incremental evaluation

**Success criteria:**
- Partial graphs evaluate correctly
- Pending states show all missing inputs with Spanish messages
- Cache invalidation works on graph updates
- Incremental update 5x faster than full re-eval

**Impact:** Enables real-time AR feedback as children build

---

#### Priority 8: Integration Tests (1 day)
**Tasks:**
- [ ] End-to-end HTTP API tests
- [ ] End-to-end WebSocket tests
- [ ] Performance benchmarks for integration layer
- [ ] Concurrent request handling tests

**Success criteria:**
- All integration points tested
- Performance targets met
- Handles 5 concurrent requests/connections

---

### PHASE 5: Advanced Features (Week 5-6)
**Goal:** Complete remaining layers

#### Priority 9: Temporal Operators (2 days)
**Tasks:**
- [ ] Update evaluator to handle time parameter properly
- [ ] Implement stream value evaluation
- [ ] Add stream generator support
- [ ] Write tests for temporal operations (FBY counter 0,1,2,3,4)

**Success criteria:**
- Temporal operators work correctly
- Cache indexed by time
- FBY produces correct sequence

---

#### Priority 10: Additional Curriculum Types (2 days)
**Tasks:**
- [ ] Implement Food operations (COMPARE_BY_TASTE, FILTER_BY_TASTE)
- [ ] Implement Animal operations (COMPARE_BY_TYPE, FILTER_BY_TYPE)
- [ ] Implement Person operations (COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER, etc.)
- [ ] Write tests for all additional types

---

#### Priority 11: Documentation (1 day)
**Tasks:**
- [ ] Package-level README files
- [ ] API documentation (OpenAPI spec for HTTP)
- [ ] WebSocket protocol documentation
- [ ] Developer documentation (setup, contribution)

---

### BLOCKERS & RISKS

**Critical Blockers:**
1. ⚠️ Build script infinite loop - blocks all development
2. ⚠️ No test framework - cannot verify functionality
3. ⚠️ Missing operation implementations - 9 operations crash at runtime

**Technical Risks:**
1. Temporal operator complexity - demand-driven semantics with time
2. Cache invalidation in incremental mode - must be correct
3. Type system integration - property constraints, type compatibility

**Resource Risks:**
1. 270 tests needed - significant test writing effort
2. Child-friendly Spanish messages - requires translation expertise

---

### SUCCESS METRICS FOR MVP

**Layer 1 Complete:**
- ✅ Build works without issues
- ✅ All tests pass
- ✅ "3 + 2 = 5" works end-to-end
- ✅ HTTP API functional

**Layer 1-4 Complete (MVP):**
- ✅ All arithmetic operations work
- ✅ All set operations work
- ✅ Curriculum types work
- ✅ Type checking catches errors
- ✅ Child-friendly Spanish messages throughout

**Layer 1-7 Complete (Version 1.0):**
- ✅ Temporal operators work
- ✅ Integration layer complete
- ✅ Live construction mode works
- ✅ All 40+ operations implemented
- ✅ 80%+ test coverage

---

## Appendix: Complete Feature Checklist (Updated Status)

### Layer 1
- [x] Monorepo structure
- [x] Shared type system
- [x] JSON input schema
- [x] Lexer (Chevrotain)
- [x] Parser (Chevrotain)
- [x] AST builder
- [x] DAG validator
- [x] DataflowGraph structure
- [x] Demand-driven evaluator
- [x] ADD operation
- [ ] HTTP API - compile endpoint (EMPTY PACKAGE)
- [ ] HTTP API - execute endpoint (EMPTY PACKAGE)
- [ ] All Layer 1 tests (0 TESTS WRITTEN)

### Layer 2
- [x] SUBTRACT operation
- [x] MULTIPLY operation
- [x] DIVIDE operation
- [x] COMPARE operation
- [x] Operation registry expansion
- [x] Arity checker
- [ ] All Layer 2 tests (0 TESTS WRITTEN)
- [ ] Fix COMPARE return type (Integer vs Natural bug)

### Layer 3
- [x] Shape type definition
- [x] Car type definition
- [ ] COMPARE_BY_COLOR operation
- [ ] COMPARE_BY_SIZE operation
- [ ] FILTER_BY_COLOR operation
- [ ] All Layer 3 tests (0 TESTS WRITTEN)

### Layer 4
- [x] Set type definition
- [ ] UNION operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] INTERSECTION operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] DIFFERENCE operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] COMPLEMENT operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] SORT operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] All Layer 4 tests (0 TESTS WRITTEN)

### Layer 5
- [x] Time parameter in evaluator (SUPPORTED, NOT USED)
- [ ] FBY operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] NEXT operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] FIRST operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] ACCUMULATE operation (IN REGISTRY, NOT IMPLEMENTED)
- [ ] All Layer 5 tests (0 TESTS WRITTEN)

### Layer 6
- [x] Stream type definition
- [ ] Stream sources
- [ ] Stream operations inheritance
- [ ] All Layer 6 tests (0 TESTS WRITTEN)

### Layer 7a (HTTP API)
- [ ] Health endpoint
- [ ] Compile endpoint (full)
- [ ] Execute endpoint (full)
- [ ] All Layer 7a tests (0 TESTS WRITTEN)

### Layer 7b (WebSocket)
- [ ] WebSocket connection
- [ ] Validate program message
- [ ] Evaluate incremental message
- [ ] Subscribe/unsubscribe messages
- [ ] Push notifications
- [ ] All Layer 7b tests (0 TESTS WRITTEN)

### Layer 7c (Incremental Runtime)
- [ ] Partial graph evaluation
- [ ] Graph update API
- [ ] Incremental update performance
- [ ] All Layer 7c tests (0 TESTS WRITTEN)

### Additional Types
- [x] Food type (defined, no operations)
- [x] Animal type (defined, no operations)
- [x] Person type (defined, no operations)
- [x] Integer type
- [x] Decimal type
- [ ] Fraction type (NOT IN REGISTRY)
- [x] Text type
- [x] Boolean type

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

### Infrastructure
- [ ] Fix build script infinite loop
- [ ] Configure TypeScript typecheck
- [ ] Set up test framework
- [ ] Configure lint rules

---

**Document Status:** Analysis Complete, Ready for Implementation
**Next Step:** Priority 0 - Fix Build & Test Infrastructure
**Estimated Time to Layer 1 Complete:** 5 days
**Estimated Time to MVP (Layers 1-4):** 15 days
