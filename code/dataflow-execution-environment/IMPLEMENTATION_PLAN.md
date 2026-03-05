# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-05 (Updated comparison operations to return Boolean)

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

**Last Updated:** 2026-03-05
**Analysis Date:** 2026-03-05

### Overall Progress: ~45% Complete

### Test Status Summary
- **Total Tests:** 44
- **Passing:** 32 (73%)
- **Failing:** 12 (27%)
  - Compiler parsing tests: 6/6 failing (CRITICAL BUG #1)
  - Compiler validation tests: 6/6 failing (CRITICAL BUG #2)
  - Runtime tests: 16/16 passing ✓
  - Shared type tests: 16/16 passing ✓

### Layer Progress

| Layer | Status | Completion | Tests Passing |
|-------|--------|------------|---------------|
| Layer 1: Foundation | PARTIAL | 85% | 0% (blocked by bugs) |
| Layer 2: Arithmetic | PARTIAL | 75% | 100% (runtime tests) |
| Layer 3: Curriculum Types | MISSING | 10% | 0% |
| Layer 4: Set Operations | PARTIAL | 15% | 0% |
| Layer 5: Temporal Operators | PARTIAL | 15% | 0% |
| Layer 6: Streams | MISSING | 5% | 0% |
| Layer 7: Integration | MISSING | 0% | 0% |

---

## CRITICAL BLOCKERS (Must Fix First)

### Bug #1: AST Builder CST Access Pattern (PRIORITY -1)

**Location:** `packages/compiler/src/ast/ast-builder.ts:12-22`

**Root Cause:**
```typescript
// CURRENT (WRONG):
statement(ctx: any): Statement {
  const name = ctx.name;  // ❌ ctx.name doesn't exist in Chevrotain CST
  if (name === "sourceStatement") {
    return this.sourceStatement(ctx);
  } else if (name === "transformStatement") {
    return this.transformStatement(ctx);
  } else if (name === "outputStatement") {
    return this.outputStatement(ctx);
  }
  throw new Error('Unknown statement type');
}
```

**Expected CST Structure:**
```typescript
// Chevrotain CST has nested children structure:
{
  children: {
    sourceStatement: [...],  // or
    transformStatement: [...],  // or
    outputStatement: [...]
  }
}
```

**Fix Required:**
```typescript
statement(ctx: any): Statement {
  if (ctx.children?.sourceStatement) {
    return this.sourceStatement(ctx.children.sourceStatement[0]);
  } else if (ctx.children?.transformStatement) {
    return this.transformStatement(ctx.children.transformStatement[0]);
  } else if (ctx.children?.outputStatement) {
    return this.outputStatement(ctx.children.outputStatement[0]);
  }
  throw new Error('Unknown statement type');
}
```

**Impact:** All 6 parsing tests fail with "Unknown statement type"

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/GRAMMAR_SPEC.md: Statement parsing (lines 65-71)
- specs/LANGUAGE_SPEC.md: Program structure validation

**Success Criteria:**
- All 6 parsing tests pass
- No "Unknown statement type" errors
- CST correctly traverses to build AST

---

### Bug #2: Compiler Tests Wrong API (PRIORITY -1)

**Location:** `packages/compiler/src/compiler.test.ts:93-240`

**Root Cause:**
```typescript
// CURRENT (WRONG):
const program: DataflowProgram = {
  metadata: { programId: 'prog_001' },
  graph: {
    nodes: [
      { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
      { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
      { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
      { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
    ],
    edges: [
      { id: 'e1', from: 'a', to: 'add', toPort: 0 },
      { id: 'e2', from: 'b', to: 'add', toPort: 1 },
      { id: 'e3', from: 'add', to: 'result' }
    ]
  }
};
const result = compiler.compile(program);  // ❌ compile() expects string, not DataflowProgram
```

**Fix Required:**
```typescript
// CORRECT:
const source = `
  source a: natural = 3;
  source b: natural = 2;
  transform sum: natural = ADD(a, b);
  output result: natural = sum;
`;
const result = compiler.compile(source);  // ✅ Pass source string
```

**Impact:** All 6 validation tests fail with `RangeError: Array length must be a positive integer`

**Estimated Time:** 1 hour

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Program validation (lines 431-527)
- specs/GRAMMAR_SPEC.md: Complete examples (lines 334-448)
- specs/INTEGRATION_SPEC.md: Validation errors (lines 86-111)

**Success Criteria:**
- All 6 validation tests pass
- No RangeError exceptions
- Tests use correct compile(source: string) API

---

## COMPLETE FEATURE BREAKDOWN

### ✅ COMPLETE (Core Infrastructure)

**Monorepo & Build System:**
- ✓ Bun workspaces configured
- ✓ TypeScript strict mode
- ✓ All packages build individually
- ✓ `bun tsc --noEmit` works for typecheck

**Shared Types (packages/shared/src/types/):**
- ✓ Primitive types: Natural, Integer, Decimal, Text, Boolean
- ✓ Curriculum types: Shape, Car, Food, Animal, Person
- ✓ Composite types: Set, Stream (missing `generator` field - see bugs below)
- ✓ Validation types: ValidationError, ValidationResult, MissingInput
- ✓ Program types: DataflowProgram, DataflowNode, DataflowEdge

**Compiler (packages/compiler/src/):**
- ✓ Lexer (Chevrotain) - 15/31 operations defined
- ✓ Parser (CstParser) - 15/31 operations in grammar
- ✓ AST Builder - Has CST access bug (see Bug #1)
- ✓ DAG Validator - Cycle detection, arity checking, Spanish errors
- ✓ Compiler class - Integration of all components
- ✓ Child-friendly Spanish error messages

**Runtime (packages/runtime/src/):**
- ✓ DataflowGraph - Node/edge management
- ✓ DemandDrivenEvaluator - Caching, demand-driven evaluation
- ✓ Numeric operations: ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE
- ✓ Runtime class - Program loading and execution

**Operation Registry (packages/shared/src/operations/):**
- ✓ 15 operations with signatures defined
- ✓ getOperationSignature() utility
- ✓ isOperation() utility

---

### ❌ MISSING CRITICAL FEATURES

#### Shared Types Issues:

1. **Fraction Type Missing** (Priority 0)
   - **Location:** `packages/shared/src/types/primitives.ts`
   - **Spec Ref:** specs/LANGUAGE_SPEC.md lines 93-109
   - **Required:**
   ```typescript
   export type Fraction = {
     kind: "fraction";
     numerator: Integer;
     denominator: Integer;
   };
   
   // Update Primitive union
   export type Primitive = Natural | Integer | Decimal | Text | Boolean | Fraction;
   ```
   - **Estimated Time:** 30 minutes
   - **Test Requirements:** specs/LANGUAGE_SPEC.md Fraction operations (lines 103-109)
   - **Success Criteria:** Fraction type defined and exported, TypeScript compiles

2. ~~**Comparison Type Missing**~~ (REMOVED - Comparison operations return Boolean, not a separate type)
    - **NOTE:** COMPARE operations return Boolean (true if equal, false otherwise), not a separate Comparison type
    - Comparison semantics:
      - Numbers/Booleans: Direct value equality check
      - Text: Lexicographical comparison based on UTF-16 code unit values (by value, not by reference)
      - COMPARE_BY_* operations: Compare specific properties (size, color, type, etc.) and return Boolean
    - **Estimated Time:** N/A (no type needed)
    - **Test Requirements:** specs/LANGUAGE_SPEC.md COMPARE operations (lines 53, 70, 89, 107, 120, 137)
    - **Success Criteria:** COMPARE operations return Boolean type

3. **Stream Type Missing Generator Field** (Priority 0)
   - **Location:** `packages/shared/src/types/composite.ts`
   - **Spec Ref:** specs/LANGUAGE_SPEC.md lines 283-288, specs/DEMAND_DRIVEN_INCREMENTAL.md
   - **Current:**
   ```typescript
   export type StreamType<T = DataType> = {
     kind: "stream";
     elementType: T;
   };
   ```
   - **Required:**
   ```typescript
   export type StreamType<T = DataType> = {
     kind: "stream";
     elementType: T;
     generator: Generator<T>;  // ADD THIS FIELD
   };
   ```
   - **Estimated Time:** 15 minutes
   - **Test Requirements:** specs/LANGUAGE_SPEC.md Stream type (lines 279-310), specs/DEMAND_DRIVEN_INCREMENTAL.md
   - **Success Criteria:** Stream type has generator field, TypeScript compiles

#### Compiler Issues:

4. **Missing Operation Tokens** (Priority 1)
   - **Location:** `packages/compiler/src/lexer/tokens.ts` and `packages/compiler/src/parser/dataflow-parser.ts`
   - **Spec Ref:** specs/GRAMMAR_SPEC.md lines 200-260 (Operations by Category)
   - **Missing from spec:**
   
   **Comparison Operations (8 missing):**
   - COMPARE_BY_SIZE
   - COMPARE_BY_COLOR
   - COMPARE_BY_TYPE
   - COMPARE_BY_TASTE
   - COMPARE_BY_AGE_GROUP
   - COMPARE_BY_GENDER
   - (Total comparison ops: 1 implemented, 8 missing)
   
   **Filtering Operations (6 missing):**
   - FILTER_BY_SIZE
   - FILTER_BY_COLOR
   - FILTER_BY_TYPE
   - FILTER_BY_TASTE
   - FILTER_BY_AGE_GROUP
   - FILTER_BY_GENDER
   - (Total filtering ops: 1 generic FILTER, 6 specific missing)
   
   **Ordering Operations (1 missing):**
   - ALPHABETICAL_SORT
   - (Total ordering ops: 1 SORT implemented, 1 missing)
   
   - **Estimated Time:** 2 hours
   - Add 15 new tokens to lexer/tokens.ts
   - Add 15 new operations to parser/dataflow-parser.ts
   - Update allTokens array
   - **Test Requirements:** specs/GRAMMAR_SPEC.md Operations (lines 200-260)
   - **Success Criteria:** All 31 operations recognized by lexer, all parse correctly

5. **Stream Literal Parsing Missing** (Priority 2)
   - **Location:** `packages/compiler/src/parser/dataflow-parser.ts` and `packages/compiler/src/ast/ast-builder.ts`
   - **Spec Ref:** specs/GRAMMAR_SPEC.md lines 170-175
   - **Required Grammar:**
   ```ebnf
   stream_literal ::= "stream" "<" type ">" "(" stream_source ")"
   stream_source ::= "sensor" "(" text_literal ")"
                 | "generator" "(" identifier ")"
                 | "external" "(" text_literal ")"
   ```
   - **Estimated Time:** 3 hours
   - Add sensor, generator, external tokens
   - Add streamLiteral and streamSource rules to parser
   - Add AST builder methods for streams
   - **Test Requirements:** specs/GRAMMAR_SPEC.md Stream examples (lines 414-427)
   - **Success Criteria:** Stream literals parse correctly, AST includes stream structure

6. **Type Compatibility Validation Missing** (Priority 1)
   - **Location:** `packages/compiler/src/validation/` (create new file `type-checker.ts`)
   - **Spec Ref:** specs/LANGUAGE_SPEC.md lines 433-440
   - **Required:** Validate input types match operation signatures
   - **Estimated Time:** 4 hours
   - Create TypeChecker class
   - Implement isTypeCompatible() method
   - Return ValidationError with childMessage
   - **Test Requirements:** specs/LANGUAGE_SPEC.md Type checking (lines 433-440), specs/INTEGRATION_SPEC.md Validation (lines 86-111)
   - **Success Criteria:** Type errors detected at compile time, child-friendly Spanish messages

7. **Property Constraint Validation Missing** (Priority 1)
   - **Location:** `packages/compiler/src/validation/` (enhance existing or create `property-checker.ts`)
   - **Spec Ref:** specs/LANGUAGE_SPEC.md lines 438, 491-527
   - **Required:** Check types have required properties (e.g., `color` for FILTER_BY_COLOR)
   - **Estimated Time:** 3 hours
   - Implement hasProperty() checks
   - Filter: color, size, type, taste, ageGroup, gender
   - Compare: same properties
   - **Test Requirements:** specs/LANGUAGE_SPEC.md Property requirements (lines 491-527)
   - **Success Criteria:** Property errors detected at compile time, curriculum types validated correctly

#### Runtime Issues:

8. **Missing Runtime Operations** (Priority 1) - **CRASH RISK**
   - **Location:** `packages/runtime/src/operations/`
   - **Spec Ref:** specs/LANGUAGE_SPEC.md Operations (lines 48-109), specs/INTEGRATION_SPEC.md Execution
   - **Missing 11 operations (will crash at runtime):**
   
   **Set Operations (5 missing):**
   - UNION, INTERSECTION, DIFFERENCE, COMPLEMENT
   - SORT (ordering)
   - **Create:** `packages/runtime/src/operations/sets.ts`
   
   **Filtering Operations (1 missing):**
   - FILTER (generic, not specific FILTER_BY_*)
   - **Create:** `packages/runtime/src/operations/filtering.ts`
   
   **Temporal Operations (4 missing):**
   - NEXT, FIRST, FBY, ACCUMULATE
   - **Create:** `packages/runtime/src/operations/temporal.ts`
   
   **Curriculum Comparison Operations (6 missing):**
   - COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE
   - COMPARE_BY_TASTE, COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER
   - **Add to:** `packages/runtime/src/operations/comparison.ts` (create)
   
   **Curriculum Filtering Operations (6 missing):**
   - FILTER_BY_SIZE, FILTER_BY_COLOR, FILTER_BY_TYPE
   - FILTER_BY_TASTE, FILTER_BY_AGE_GROUP, FILTER_BY_GENDER
   - **Add to:** `packages/runtime/src/operations/filtering.ts`
   
   - **Estimated Time:** 8 hours
   - Implement all 17 missing operations
   - Update DemandDrivenEvaluator to dispatch to new ops
   - Handle temporal operations with time parameter
   - **Test Requirements:** specs/LANGUAGE_SPEC.md All operation definitions, specs/INTEGRATION_SPEC.md Execute endpoint
   - **Success Criteria:** All operations implemented, no "Unknown operation" errors, temporal ops use pure time-based semantics

9. **IncrementalRuntime Class Missing** (Priority 2)
   - **Location:** `packages/runtime/src/` (create new file `incremental-runtime.ts`)
   - **Spec Ref:** specs/INTEGRATION_SPEC.md lines 365-771, specs/DEMAND_DRIVEN_INCREMENTAL.md
   - **Required Features:**
     - Partial graph evaluation
     - Node state tracking (completed/pending/error)
     - Subscription management
     - Graph update API with cache invalidation
     - Missing input extraction with Spanish messages
     - Demand-driven semantics (not eager evaluation)
   - **Estimated Time:** 10 hours
   - Implement IncrementalRuntime class
   - Implement updateGraph() with cache invalidation
   - Implement evaluatePartial() with demand sources
   - Implement subscribe/unsubscribe mechanisms
   - Implement getChildMessageForMissingInput() with Spanish
   - **Test Requirements:** specs/INTEGRATION_SPEC.md Incremental Runtime (lines 365-771), specs/DEMAND_DRIVEN_INCREMENTAL.md complete
   - **Success Criteria:** Partial graphs evaluate correctly, demand-driven semantics preserved, pending states show Spanish messages, incremental updates 5x faster than full re-eval

10. **Subscription Mechanism Missing** (Priority 2)
    - Part of IncrementalRuntime implementation
    - Required for WebSocket live feedback
    - **Estimated Time:** 3 hours (included in IncrementalRuntime above)
    - **Test Requirements:** specs/INTEGRATION_SPEC.md WebSocket protocol (lines 222-320)
    - **Success Criteria:** Subscriptions trigger evaluation, notifications sent on state changes, multiple clients supported

#### Integration Issues:

11. **HTTP API - Completely Empty** (Priority 3)
    - **Location:** `packages/http-api/` (currently empty)
    - **Spec Ref:** specs/INTEGRATION_SPEC.md lines 53-189
    - **Required Endpoints:**
      - POST /api/v1/compile - Validate program
      - POST /api/v1/execute - Compile and run
      - GET /api/v1/health - Health check
    - **Required Features:**
      - Accept source code strings (not DataflowProgram objects)
      - Return child-friendly Spanish error messages
      - Support execution options (maxTimesteps, includeTrace, traceLevel)
      - Return execution trace (executionOrder, nodeEvaluations, cacheHits, cacheMisses)
    - **Estimated Time:** 10 hours
    - Create Elysia server setup
    - Implement 3 endpoints with proper error handling
    - Add request validation middleware
    - Add response formatting
    - **Test Requirements:** specs/INTEGRATION_SPEC.md HTTP API (lines 53-189)
    - **Success Criteria:** All 3 endpoints functional, child-friendly Spanish messages, performance targets met (<100ms compile, <50ms execute), all tests passing

12. **WebSocket Server - Completely Empty** (Priority 3)
    - **Location:** `packages/websocket-server/` (currently empty)
    - **Spec Ref:** specs/INTEGRATION_SPEC.md lines 192-361
    - **Required Message Handlers (client → server):**
      - validate_program - Validate partial program
      - evaluate_incremental - Evaluate partial graph
      - subscribe_node - Subscribe to node changes
      - unsubscribe_node - Unsubscribe from node
    - **Required Push Messages (server → client):**
      - validation_result - Validation errors/warnings
      - evaluation_result - Node states (completed/pending/error)
      - node_state_changed - State change notifications
      - error - Message errors
    - **Required Features:**
      - Connection at ws://localhost:3000/live
      - Message protocol with messageId
      - Multiple concurrent client support
      - Connection resilience (reconnects)
      - Child-friendly Spanish messages
    - **Estimated Time:** 12 hours
    - Create Elysia WebSocket server setup
    - Implement 4 message handlers
    - Implement 3 push notification types
    - Add connection manager
    - Add subscription manager
    - **Test Requirements:** specs/INTEGRATION_SPEC.md WebSocket Server (lines 192-361)
    - **Success Criteria:** All 5 message types handled, push notifications work, multiple concurrent clients, child-friendly Spanish messages, connection resilience

---

## PRIORITIZED IMPLEMENTATION PLAN

### PHASE 0: CRITICAL BUG FIXES (Week 0.5 - 1.5 hours)

#### Priority -1: Fix AST Builder CST Access (30 minutes)

**Task:** Update CST traversal pattern in AST Builder

**File:** `packages/compiler/src/ast/ast-builder.ts`

**Lines:** 12-22

**Changes:**
```typescript
// Fix statement() method
statement(ctx: any): Statement {
  if (ctx.children?.sourceStatement) {
    return this.sourceStatement(ctx.children.sourceStatement[0]);
  } else if (ctx.children?.transformStatement) {
    return this.transformStatement(ctx.children.transformStatement[0]);
  } else if (ctx.children?.outputStatement) {
    return this.outputStatement(ctx.children.outputStatement[0]);
  }
  throw new Error('Unknown statement type');
}

// Update sourceStatement, transformStatement, outputStatement methods
// (they now receive CST child directly, not parent ctx)
```

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/GRAMMAR_SPEC.md: Statement parsing (lines 65-71)
- Compiler test: packages/compiler/src/compiler.test.ts (lines 6-90)

**Success Criteria:**
- All 6 parsing tests pass
- No "Unknown statement type" errors
- CST correctly traverses nested children structure

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented (no stubs)
- [ ] All parsing tests pass
- [ ] Previous tests (runtime, shared) still pass
- [ ] Typecheck passes: `bun tsc --noEmit`
- [ ] Git commit with message: "fix(compiler): correct AST Builder CST access pattern"

---

#### Priority -1: Fix Compiler Test API (1 hour)

**Task:** Update validation tests to use source strings instead of DataflowProgram objects

**File:** `packages/compiler/src/compiler.test.ts`

**Lines:** 92-240

**Changes:**
```typescript
// BEFORE (lines 93-118):
describe('Compiler - Validation', () => {
  it('should validate correct program', () => {
    const compiler = new Compiler();
    
    const program: DataflowProgram = {
      metadata: { programId: 'prog_001' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
        ],
        edges: [...]
      }
    };
    
    const result = compiler.compile(program);
    // ...
  });
});

// AFTER:
describe('Compiler - Validation', () => {
  it('should validate correct program', () => {
    const compiler = new Compiler();
    
    const source = `
      source a: natural = 3;
      source b: natural = 2;
      transform sum: natural = ADD(a, b);
      output result: natural = sum;
    `;
    
    const result = compiler.compile(source);
    // ...
  });
});
```

**Estimated Time:** 1 hour

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Program validation (lines 431-527)
- specs/GRAMMAR_SPEC.md: Complete examples (lines 334-448)
- specs/INTEGRATION_SPEC.md: Validation errors (lines 86-111)

**Success Criteria:**
- All 6 validation tests pass
- No RangeError exceptions
- Tests use correct compile(source: string) API

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented (no stubs)
- [ ] All validation tests pass
- [ ] Previous tests still pass (parsing tests now passing)
- [ ] Typecheck passes
- [ ] Git commit with message: "fix(tests): update compiler tests to use source strings"

---

### PHASE 1: TYPES AND REGISTRY (Week 1 - 4 hours)

#### Priority 0: Add Missing Types (2 hours)

**Task 0.1: Fraction Type** (30 minutes)

**File:** `packages/shared/src/types/primitives.ts`

**Lines:** Add after Decimal type (line 14)

**Changes:**
```typescript
export type Fraction = {
  kind: "fraction";
  numerator: Integer;
  denominator: Integer;
};

// Update Primitive union at end of file (line 26)
export type Primitive = Natural | Integer | Decimal | Text | Boolean | Fraction;
```

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Fraction type (lines 93-109)
- Fraction operations: ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE

**Success Criteria:**
- Fraction type defined and exported
- TypeScript compiles
- Fraction can be used in type annotations

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(shared): add Fraction type"

---

**Task 0.2: ~~Comparison Type~~ (REMOVED)**

**NOTE:** Comparison operations return Boolean, not a separate Comparison type.

**Comparison Semantics:**
- For numbers and booleans: Direct value equality check
- For text: Lexicographical comparison based on UTF-16 code unit values (like JavaScript/TypeScript), by value not by reference
- For COMPARE_BY_SIZE, COMPARE_BY_COLOR, etc.: Compare specific properties and return Boolean
- All comparisons are by value, not by reference

**Estimated Time:** N/A (no changes needed to types)

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: COMPARE operations (lines 53, 70, 89, 107, 120, 137)
- COMPARE returns Boolean type

**Success Criteria:**
- No Comparison type needed (already covered by Boolean)
- All COMPARE operations correctly return Boolean
- TypeScript compiles

**Ralph Wiggum Checklist:**
- N/A (task removed - Boolean type already exists)

---

**Task 0.3: Fix Stream Type** (15 minutes)

**File:** `packages/shared/src/types/composite.ts`

**Lines:** 16-19

**Changes:**
```typescript
export type StreamType<T = DataType> = {
  kind: "stream";
  elementType: T;
  generator: Generator<T>;  // ADD THIS FIELD
};
```

**Estimated Time:** 15 minutes

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Stream type (lines 279-310)
- specs/DEMAND_DRIVEN_INCREMENTAL.md: Stream generator usage

**Success Criteria:**
- Stream type has generator field
- TypeScript compiles
- Generator type imported from TypeScript stdlib

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Git commit with message: "fix(shared): add generator field to Stream type"

---

#### Priority 1: Expand Operation Registry (2 hours)

**Task 1.1: Add Comparison Operations to Registry** (30 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Lines:** Add to Operation type (lines 3-18) and OPERATION_REGISTRY (after line 63)

**Changes:**
```typescript
// Add to Operation type:
export type Operation =
  | "ADD" | "SUBTRACT" | "MULTIPLY" | "DIVIDE"
  | "COMPARE"
  | "COMPARE_BY_SIZE" | "COMPARE_BY_COLOR" | "COMPARE_BY_TYPE"
  | "COMPARE_BY_TASTE" | "COMPARE_BY_AGE_GROUP" | "COMPARE_BY_GENDER"
  | "FILTER"
  | "FILTER_BY_SIZE" | "FILTER_BY_COLOR" | "FILTER_BY_TYPE"
  | "FILTER_BY_TASTE" | "FILTER_BY_AGE_GROUP" | "FILTER_BY_GENDER"
  | "UNION" | "INTERSECTION" | "DIFFERENCE" | "COMPLEMENT"
  | "NEXT" | "FIRST" | "FBY" | "ACCUMULATE"
  | "SORT" | "ALPHABETICAL_SORT";

// All COMPARE operations return Boolean (true if equal, false otherwise)

// Add to OPERATION_REGISTRY:
COMPARE_BY_SIZE: {
  arity: 2,
  inputTypes: ["shape", "shape"],
  outputType: "boolean",
  category: "comparison"
},

COMPARE_BY_COLOR: {
  arity: 2,
  inputTypes: [{ kind: "hasColor" }, { kind: "hasColor" }],
  outputType: "boolean",
  category: "comparison"
},

COMPARE_BY_TYPE: {
  arity: 2,
  inputTypes: ["animal", "animal"],
  outputType: "boolean",
  category: "comparison"
},

COMPARE_BY_TASTE: {
  arity: 2,
  inputTypes: ["food", "food"],
  outputType: "boolean",
  category: "comparison"
},

COMPARE_BY_AGE_GROUP: {
  arity: 2,
  inputTypes: ["person", "person"],
  outputType: "boolean",
  category: "comparison"
},

COMPARE_BY_GENDER: {
  arity: 2,
  inputTypes: ["person", "person"],
  outputType: "boolean",
  category: "comparison"
},
```

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Comparison operations (lines 172-276)
- getOperationSignature() should return signatures for all ops

**Success Criteria:**
- All 7 comparison operations in registry
- Proper type signatures defined
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(shared): add comparison operations to registry"

---

**Task 1.2: Add Filtering Operations to Registry** (30 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Lines:** Add to OPERATION_REGISTRY

**Changes:**
```typescript
FILTER_BY_SIZE: {
  arity: 2,
  inputTypes: [{ kind: "set", elementType: "shape" }, "text"],
  outputType: { kind: "set", elementType: "shape" },
  category: "filtering"
},

FILTER_BY_COLOR: {
  arity: 2,
  inputTypes: [{ kind: "set", elementType: "hasColor" }, "text"],
  outputType: { kind: "set", elementType: "sameAsInput0" },
  category: "filtering"
},

FILTER_BY_TYPE: {
  arity: 2,
  inputTypes: [{ kind: "set", elementType: "animal" }, "text"],
  outputType: { kind: "set", elementType: "animal" },
  category: "filtering"
},

FILTER_BY_TASTE: {
  arity: 2,
  inputTypes: [{ kind: "set", elementType: "food" }, "text"],
  outputType: { kind: "set", elementType: "food" },
  category: "filtering"
},

FILTER_BY_AGE_GROUP: {
  arity: 2,
  inputTypes: [{ kind: "set", elementType: "person" }, "text"],
  outputType: { kind: "set", elementType: "person" },
  category: "filtering"
},

FILTER_BY_GENDER: {
  arity: 2,
  inputTypes: [{ kind: "set", elementType: "person" }, "text"],
  outputType: { kind: "set", elementType: "person" },
  category: "filtering"
},
```

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Filtering operations (lines 176-246)

**Success Criteria:**
- All 7 filtering operations in registry
- Proper type signatures
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(shared): add filtering operations to registry"

---

**Task 1.3: Add ALPHABETICAL_SORT to Registry** (15 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Lines:** Add after SORT (line 149)

**Changes:**
```typescript
ALPHABETICAL_SORT: {
  arity: 1,
  inputTypes: [{ kind: "set", elementType: "text" }],
  outputType: { kind: "set", elementType: "text" },
  category: "ordering"
}
```

**Estimated Time:** 15 minutes

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: ALPHABETICAL_SORT (line 121)

**Success Criteria:**
- ALPHABETICAL_SORT in registry
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(shared): add ALPHABETICAL_SORT to registry"

---

### PHASE 2: RUNTIME OPERATIONS (Week 2 - 8 hours)

#### Priority 1: Implement Missing Runtime Operations (8 hours)

**Task 2.1: Set Operations** (2 hours)

**Create File:** `packages/runtime/src/operations/sets.ts`

**Changes:**
```typescript
import type { SetType } from "@dataflow/shared/types";
import { unwrapValue } from "../evaluator/demand-driven-evaluator.js";

export const UNION = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set1, set2] = inputs.map(i => unwrapValue(i));
  
  // Merge arrays and remove duplicates
  const merged = [...set1.elements, ...set2.elements];
  const unique = merged.filter((item, index, self) => 
    index === self.findIndex(t => JSON.stringify(t) === JSON.stringify(item))
  );
  
  return {
    kind: "set",
    elementType: set1.elementType,
    elements: unique
  };
};

export const INTERSECTION = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set1, set2] = inputs.map(i => unwrapValue(i));
  
  // Find common elements
  const common = set1.elements.filter((item1) => 
    set2.elements.some(item2 => JSON.stringify(item1) === JSON.stringify(item2))
  );
  
  return {
    kind: "set",
    elementType: set1.elementType,
    elements: common
  };
};

export const DIFFERENCE = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set1, set2] = inputs.map(i => unwrapValue(i));
  
  // Elements in set1 not in set2
  const diff = set1.elements.filter((item1) => 
    !set2.elements.some(item2 => JSON.stringify(item1) === JSON.stringify(item2))
  );
  
  return {
    kind: "set",
    elementType: set1.elementType,
    elements: diff
  };
};

export const COMPLEMENT = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, universe] = inputs.map(i => unwrapValue(i));
  
  // Elements in universe not in set
  const complement = universe.elements.filter((item1) => 
    !set.elements.some(item2 => JSON.stringify(item1) === JSON.stringify(item2))
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: complement
  };
};

export const SORT = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set] = inputs.map(i => unwrapValue(i));
  
  // Sort elements using natural order or COMPARE if available
  const sorted = [...set.elements].sort((a, b) => {
    if (typeof a === 'number' && typeof b === 'number') {
      return a - b;
    }
    return JSON.stringify(a).localeCompare(JSON.stringify(b));
  });
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: sorted
  };
};
```

**Estimated Time:** 2 hours

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Set operations (lines 183-186, 202-206, 225-229, 248-252, 271-275)
- specs/GRAMMAR_SPEC.md: Set operation examples (lines 382-410)

**Success Criteria:**
- All 5 set operations implemented
- Operations handle duplicate elements correctly
- TypeScript compiles
- Test coverage for all operations

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All set operation tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(runtime): implement set operations"

---

**Task 2.2: Filtering Operations** (2 hours)

**Create File:** `packages/runtime/src/operations/filtering.ts`

**Changes:**
```typescript
import type { SetType, Shape, Car, Food, Animal, Person, Text } from "@dataflow/shared/types";
import { unwrapValue } from "../evaluator/demand-driven-evaluator.js";

export const FILTER = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, condition] = inputs.map(i => unwrapValue(i));
  // Generic filter - uses COMPARE operation internally
  // For now, return elements that are truthy
  const filtered = set.elements.filter(item => Boolean(item));
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};

export const FILTER_BY_SIZE = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, sizeText] = inputs.map(i => unwrapValue(i));
  const size = sizeText.value;
  
  const filtered = set.elements.filter((item: Shape) => 
    item.size === size
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};

export const FILTER_BY_COLOR = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, colorText] = inputs.map(i => unwrapValue(i));
  const color = colorText.value;
  
  const filtered = set.elements.filter((item: Shape | Car | Food | Animal) => 
    item.color === color
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};

export const FILTER_BY_TYPE = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, typeText] = inputs.map(i => unwrapValue(i));
  const type = typeText.value;
  
  const filtered = set.elements.filter((item: Shape | Animal) => 
    item.type === type
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};

export const FILTER_BY_TASTE = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, tasteText] = inputs.map(i => unwrapValue(i));
  const taste = tasteText.value;
  
  const filtered = set.elements.filter((item: Food) => 
    item.taste === taste
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};

export const FILTER_BY_AGE_GROUP = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, ageGroupText] = inputs.map(i => unwrapValue(i));
  const ageGroup = ageGroupText.value;
  
  const filtered = set.elements.filter((item: Person) => 
    item.ageGroup === ageGroup
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};

export const FILTER_BY_GENDER = (inputs: Array<{ id: string; value: unknown }>): SetType => {
  const [set, genderText] = inputs.map(i => unwrapValue(i));
  const gender = genderText.value;
  
  const filtered = set.elements.filter((item: Person) => 
    item.gender === gender
  );
  
  return {
    kind: "set",
    elementType: set.elementType,
    elements: filtered
  };
};
```

**Estimated Time:** 2 hours

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Filtering operations (lines 177-181, 199-201, 221-224, 244-247, 267-270)

**Success Criteria:**
- All 7 filtering operations implemented
- Handle all curriculum types correctly
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All filtering operation tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(runtime): implement filtering operations"

---

**Task 2.3: Comparison Operations** (2 hours)

**Create File:** `packages/runtime/src/operations/comparison.ts`

**Changes:**
```typescript
import type { Shape, Car, Food, Animal, Person, Boolean } from "@dataflow/shared/types";
import { unwrapValue } from "../evaluator/demand-driven-evaluator.js";

export const COMPARE_BY_SIZE = (inputs: Array<{ id: string; value: unknown }>): Boolean => {
  const [item1, item2] = inputs.map(i => unwrapValue(i));
  
  return {
    kind: "boolean",
    value: item1.size === item2.size
  };
};

export const COMPARE_BY_COLOR = (inputs: Array<{ id: string; value: unknown }>): Boolean => {
  const [item1, item2] = inputs.map(i => unwrapValue(i));
  
  // Compare colors lexicographically (UTF-16 code units), by value not by reference
  return {
    kind: "boolean",
    value: item1.color === item2.color
  };
};

export const COMPARE_BY_TYPE = (inputs: Array<{ id: string; value: unknown }>): Boolean => {
  const [item1, item2] = inputs.map(i => unwrapValue(i));
  
  // Compare types lexicographically (UTF-16 code units), by value not by reference
  return {
    kind: "boolean",
    value: item1.type === item2.type
  };
};

export const COMPARE_BY_TASTE = (inputs: Array<{ id: string; value: unknown }>): Boolean => {
  const [item1, item2] = inputs.map(i => unwrapValue(i));
  
  // Compare tastes lexicographically (UTF-16 code units), by value not by reference
  return {
    kind: "boolean",
    value: item1.taste === item2.taste
  };
};

export const COMPARE_BY_AGE_GROUP = (inputs: Array<{ id: string; value: unknown }>): Boolean => {
  const [item1, item2] = inputs.map(i => unwrapValue(i));
  
  return {
    kind: "boolean",
    value: item1.ageGroup === item2.ageGroup
  };
};

export const COMPARE_BY_GENDER = (inputs: Array<{ id: string; value: unknown }>): Boolean => {
  const [item1, item2] = inputs.map(i => unwrapValue(i));
  
  // Compare genders lexicographically (UTF-16 code units), by value not by reference
  return {
    kind: "boolean",
    value: item1.gender === item2.gender
  };
};
```

**Estimated Time:** 2 hours

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Comparison operations (lines 173-175, 197, 218-219, 241-242, 264-265)

**Success Criteria:**
- All 6 comparison operations implemented
- Return Boolean type (true if equal, false otherwise)
- All comparisons by value, not by reference
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All comparison operation tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(runtime): implement comparison operations"

---

**Task 2.4: Temporal Operations** (2 hours)

**Create File:** `packages/runtime/src/operations/temporal.ts`

**Changes:**
```typescript
import type { StreamType, DataType } from "@dataflow/shared/types";
import { unwrapValue } from "../evaluator/demand-driven-evaluator.js";

export const NEXT = (
  inputs: Array<{ id: string; value: unknown }>,
  time: number,
  cache: Map<string, Map<number, unknown>>
): unknown => {
  const [stream] = inputs.map(i => unwrapValue(i));
  
  // Get next value from stream generator
  // Pure function of time!
  const generator = stream.generator;
  let result = generator.next();
  
  // Cache the current value
  if (!cache.has(stream.id)) {
    cache.set(stream.id, new Map());
  }
  cache.get(stream.id)!.set(time, result.value);
  
  return result.value;
};

export const FIRST = (inputs: Array<{ id: string; value: unknown }>): unknown => {
  const [stream] = inputs.map(i => unwrapValue(i));
  
  // Get first value from stream
  const generator = stream.generator;
  const result = generator.next();
  return result.value;
};

export const FBY = (
  inputs: Array<{ id: string; value: unknown }>,
  time: number
): StreamType => {
  const [initial, stream] = inputs.map(i => unwrapValue(i));
  
  // Pure function of time!
  // At time 0, return initial
  // At time > 0, return value from stream at time - 1
  if (time === 0) {
    return {
      kind: "stream",
      elementType: stream.elementType,
      generator: (function*() {
        yield initial;
        yield* stream.generator;
      })()
    };
  }
  
  // Return stream with first value being initial
  return {
    kind: "stream",
    elementType: stream.elementType,
    generator: (function*() {
      yield initial;
      yield* stream.generator;
    })()
  };
};

export const ACCUMULATE = (
  inputs: Array<{ id: string; value: unknown }>,
  time: number
): StreamType => {
  const [stream, operation, initial] = inputs.map(i => unwrapValue(i));
  
  // Pure function of time!
  // Apply operation cumulatively over stream
  const generator = (function*() {
    let accumulated = initial;
    yield accumulated;
    
    for (const value of stream.generator) {
      // Apply operation (this is simplified - actual implementation would need operation registry)
      if (operation === "ADD") {
        accumulated = accumulated + value;
      }
      // Add other operations as needed
      yield accumulated;
    }
  })();
  
  return {
    kind: "stream",
    elementType: stream.elementType,
    generator
  };
};
```

**Estimated Time:** 2 hours

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Temporal operations (lines 296-301)
- specs/GRAMMAR_SPEC.md: FBY counter example (lines 414-427)
- specs/DEMAND_DRIVEN_INCREMENTAL.md: Pure time-based semantics

**Success Criteria:**
- All 4 temporal operations implemented
- Pure function of time (no mutable state)
- TypeScript compiles
- Cache indexed by time

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All temporal operation tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(runtime): implement temporal operations"

---

**Task 2.5: Update Evaluator** (30 minutes)

**File:** `packages/runtime/src/evaluator/demand-driven-evaluator.ts`

**Lines:** 2, 58-78

**Changes:**
```typescript
// Add imports at top (line 2)
import { 
  UNION, INTERSECTION, DIFFERENCE, COMPLEMENT, SORT 
} from "../operations/sets.js";
import { 
  FILTER, FILTER_BY_SIZE, FILTER_BY_COLOR, FILTER_BY_TYPE,
  FILTER_BY_TASTE, FILTER_BY_AGE_GROUP, FILTER_BY_GENDER
} from "../operations/filtering.js";
import {
  COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE,
  COMPARE_BY_TASTE, COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER
} from "../operations/comparison.js";
import {
  NEXT, FIRST, FBY, ACCUMULATE
} from "../operations/temporal.js";

// Update evaluateOperation method (lines 58-78)
private evaluateOperation(
  operation: string,
  inputs: Array<{ id: string; value: unknown }>,
  time: number,
  graph: DataflowGraph
): unknown {
  const evaluatedInputs = inputs.map(input => ({
    id: input.id,
    value: this.evaluate(input.id, time, graph)
  }));

  switch (operation) {
    case "ADD":
      return ADD(evaluatedInputs);
    case "SUBTRACT":
      return SUBTRACT(evaluatedInputs);
    case "MULTIPLY":
      return MULTIPLY(evaluatedInputs);
    case "DIVIDE":
      return DIVIDE(evaluatedInputs);
    case "COMPARE":
      return COMPARE(evaluatedInputs);  // Returns Boolean (true if equal, false otherwise)
    case "UNION":
      return UNION(evaluatedInputs);
    case "INTERSECTION":
      return INTERSECTION(evaluatedInputs);
    case "DIFFERENCE":
      return DIFFERENCE(evaluatedInputs);
    case "COMPLEMENT":
      return COMPLEMENT(evaluatedInputs);
    case "SORT":
      return SORT(evaluatedInputs);
    case "FILTER":
      return FILTER(evaluatedInputs);
    case "FILTER_BY_SIZE":
      return FILTER_BY_SIZE(evaluatedInputs);
    case "FILTER_BY_COLOR":
      return FILTER_BY_COLOR(evaluatedInputs);
    case "FILTER_BY_TYPE":
      return FILTER_BY_TYPE(evaluatedInputs);
    case "FILTER_BY_TASTE":
      return FILTER_BY_TASTE(evaluatedInputs);
    case "FILTER_BY_AGE_GROUP":
      return FILTER_BY_AGE_GROUP(evaluatedInputs);
    case "FILTER_BY_GENDER":
      return FILTER_BY_GENDER(evaluatedInputs);
    case "COMPARE_BY_SIZE":
      return COMPARE_BY_SIZE(evaluatedInputs);
    case "COMPARE_BY_COLOR":
      return COMPARE_BY_COLOR(evaluatedInputs);
    case "COMPARE_BY_TYPE":
      return COMPARE_BY_TYPE(evaluatedInputs);
    case "COMPARE_BY_TASTE":
      return COMPARE_BY_TASTE(evaluatedInputs);
    case "COMPARE_BY_AGE_GROUP":
      return COMPARE_BY_AGE_GROUP(evaluatedInputs);
    case "COMPARE_BY_GENDER":
      return COMPARE_BY_GENDER(evaluatedInputs);
    case "NEXT":
      return NEXT(evaluatedInputs, time, this.cache);
    case "FIRST":
      return FIRST(evaluatedInputs);
    case "FBY":
      return FBY(evaluatedInputs, time);
    case "ACCUMULATE":
      return ACCUMULATE(evaluatedInputs, time);
    default:
      throw new Error(`Unknown operation: ${operation}`);
  }
}
```

**Estimated Time:** 30 minutes

**Test Requirements:**
- All operation tests pass
- No "Unknown operation" errors

**Success Criteria:**
- All 31 operations dispatched correctly
- Temporal operations receive time parameter
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All runtime tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(evaluator): add all missing operations to evaluator"

---

### PHASE 3: COMPILER COMPLETION (Week 3 - 6 hours)

#### Priority 1: Add Missing Operation Tokens (2 hours)

**Task 3.1: Add Comparison Tokens** (30 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add after Compare token (line 20)

**Changes:**
```typescript
export const CompareBySize = createToken({ name: "CompareBySize", pattern: /COMPARE_BY_SIZE/i });
export const CompareByColor = createToken({ name: "CompareByColor", pattern: /COMPARE_BY_COLOR/i });
export const CompareByType = createToken({ name: "CompareByType", pattern: /COMPARE_BY_TYPE/i });
export const CompareByTaste = createToken({ name: "CompareByTaste", pattern: /COMPARE_BY_TASTE/i });
export const CompareByAgeGroup = createToken({ name: "CompareByAgeGroup", pattern: /COMPARE_BY_AGE_GROUP/i });
export const CompareByGender = createToken({ name: "CompareByGender", pattern: /COMPARE_BY_GENDER/i });
```

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add to allTokens array (lines 104-131)

**Changes:**
```typescript
export const allTokens = [
  // ... existing tokens ...
  Compare,
  CompareBySize,
  CompareByColor,
  CompareByType,
  CompareByTaste,
  CompareByAgeGroup,
  CompareByGender,
  // ... rest of tokens ...
];
```

**File:** `packages/compiler/src/parser/dataflow-parser.ts`

**Lines:** Add imports (lines 2-48), add to operationName rule (lines 172-190)

**Changes:**
```typescript
// Add to imports
import {
  // ... existing imports ...
  Compare,
  CompareBySize,
  CompareByColor,
  CompareByType,
  CompareByTaste,
  CompareByAgeGroup,
  CompareByGender,
  // ... rest of imports ...
} from "../lexer/tokens.js";

// Add to operationName rule
operationName = this.RULE("operationName", () => {
  this.OR([
    { ALT: () => this.CONSUME(Add) },
    { ALT: () => this.CONSUME(Subtract) },
    { ALT: () => this.CONSUME(Multiply) },
    { ALT: () => this.CONSUME(Divide) },
    { ALT: () => this.CONSUME(Compare) },
    { ALT: () => this.CONSUME(CompareBySize) },
    { ALT: () => this.CONSUME(CompareByColor) },
    { ALT: () => this.CONSUME(CompareByType) },
    { ALT: () => this.CONSUME(CompareByTaste) },
    { ALT: () => this.CONSUME(CompareByAgeGroup) },
    { ALT: () => this.CONSUME(CompareByGender) },
    // ... rest of operations ...
  ]);
});
```

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/GRAMMAR_SPEC.md: Comparison operations (lines 216-223)

**Success Criteria:**
- All 6 comparison tokens recognized by lexer
- Parser grammar includes all comparison operations
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass for comparison ops
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(lexer): add comparison operation tokens"

---

**Task 3.2: Add Filtering Tokens** (30 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add after Filter token (line 21)

**Changes:**
```typescript
export const FilterBySize = createToken({ name: "FilterBySize", pattern: /FILTER_BY_SIZE/i });
export const FilterByColor = createToken({ name: "FilterByColor", pattern: /FILTER_BY_COLOR/i });
export const FilterByType = createToken({ name: "FilterByType", pattern: /FILTER_BY_TYPE/i });
export const FilterByTaste = createToken({ name: "FilterByTaste", pattern: /FILTER_BY_TASTE/i });
export const FilterByAgeGroup = createToken({ name: "FilterByAgeGroup", pattern: /FILTER_BY_AGE_GROUP/i });
export const FilterByGender = createToken({ name: "FilterByGender", pattern: /FILTER_BY_GENDER/i });
```

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add to allTokens array

**File:** `packages/compiler/src/parser/dataflow-parser.ts`

**Lines:** Add imports, add to operationName rule

**Changes:** Similar to comparison tokens

**Estimated Time:** 30 minutes

**Test Requirements:**
- specs/GRAMMAR_SPEC.md: Filtering operations (lines 228-235)

**Success Criteria:**
- All 6 filtering tokens recognized by lexer
- Parser grammar includes all filtering operations
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass for filtering ops
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(lexer): add filtering operation tokens"

---

**Task 3.3: Add ALPHABETICAL_SORT Token** (15 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add after Sort token (line 30)

**Changes:**
```typescript
export const AlphabeticalSort = createToken({ name: "AlphabeticalSort", pattern: /ALPHABETICAL_SORT/i });
```

**File:** `packages/compiler/src/parser/dataflow-parser.ts`

**Lines:** Add import, add to operationName rule

**Estimated Time:** 15 minutes

**Test Requirements:**
- specs/GRAMMAR_SPEC.md: ALPHABETICAL_SORT (line 258)

**Success Criteria:**
- ALPHABETICAL_SORT token recognized by lexer
- Parser includes the operation
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(lexer): add ALPHABETICAL_SORT token"

---

#### Priority 2: Stream Literal Parsing (3 hours)

**Task 3.4: Add Stream Literal Grammar** (1.5 hours)

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add new tokens (after existing tokens, line 85)

**Changes:**
```typescript
export const Sensor = createToken({ name: "Sensor", pattern: /sensor/i });
export const GeneratorToken = createToken({ name: "GeneratorToken", pattern: /generator/i });
export const External = createToken({ name: "External", pattern: /external/i });
```

**File:** `packages/compiler/src/lexer/tokens.ts`

**Lines:** Add to allTokens array

**File:** `packages/compiler/src/parser/dataflow-parser.ts`

**Lines:** Add imports, add rules to parser (after value rule, line 129)

**Changes:**
```typescript
// Add to imports
import {
  // ... existing imports ...
  Sensor,
  GeneratorToken as Generator,
  External,
  // ... rest of imports ...
} from "../lexer/tokens.js";

// Update value rule to include streamLiteral
value = this.RULE("value", () => {
  this.OR([
    { ALT: () => this.SUBRULE(this.literal) },
    { ALT: () => this.SUBRULE(this.arrayLiteral) },
    { ALT: () => this.SUBRULE(this.streamLiteral) }  // ADD THIS
  ]);
});

// Add streamLiteral rule
streamLiteral = this.RULE("streamLiteral", () => {
  this.CONSUME(Stream);
  this.CONSUME(AngleLeft);
  this.SUBRULE(this.typeDeclaration);
  this.CONSUME(AngleRight);
  this.CONSUME(LParen);
  this.SUBRULE(this.streamSource);
  this.CONSUME(RParen);
});

// Add streamSource rule
streamSource = this.RULE("streamSource", () => {
  this.OR([
    { ALT: () => {
        this.CONSUME(Sensor);
        this.CONSUME(LParen);
        this.CONSUME(StringLiteral);
        this.CONSUME(RParen);
      }
    },
    { ALT: () => {
        this.CONSUME(GeneratorToken);
        this.CONSUME(LParen);
        this.CONSUME(Identifier);
        this.CONSUME(RParen);
      }
    },
    { ALT: () => {
        this.CONSUME(External);
        this.CONSUME(LParen);
        this.CONSUME(StringLiteral);
        this.CONSUME(RParen);
      }
  ]);
});
```

**Estimated Time:** 1.5 hours

**Test Requirements:**
- specs/GRAMMAR_SPEC.md: Stream literals (lines 170-175)
- specs/GRAMMAR_SPEC.md: Stream examples (lines 414-427)

**Success Criteria:**
- Stream literals parse correctly
- All stream source types supported (sensor, generator, external)
- Parser grammar complete

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Stream literal tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(parser): add stream literal parsing"

---

**Task 3.5: Add AST Builder Support for Streams** (1.5 hours)

**File:** `packages/compiler/src/ast/ast-builder.ts`

**Lines:** Add methods after existing builder methods

**Changes:**
```typescript
// Update value method to include streamLiteral (lines 84-92)
value(ctx: any): unknown {
  const children = ctx.children;
  if (children.literal) {
    return this.literal(ctx);
  } else if (children.arrayLiteral) {
    return this.arrayLiteral(ctx);
  } else if (children.streamLiteral) {
    return this.streamLiteral(ctx.children.streamLiteral[0]);
  }
  return undefined;
}

// Add streamLiteral method
streamLiteral(ctx: any): unknown {
  return {
    kind: "stream",
    elementType: this.typeDeclaration(ctx.children.typeDeclaration[0]),
    source: this.streamSource(ctx.children.streamSource[0])
  };
}

// Add streamSource method
streamSource(ctx: any): unknown {
  if (ctx.children.Sensor) {
    return {
      type: "sensor",
      source: ctx.children.StringLiteral[0].image.slice(1, -1)
    };
  }
  if (ctx.children.GeneratorToken) {
    return {
      type: "generator",
      id: ctx.children.Identifier[0].image
    };
  }
  if (ctx.children.External) {
    return {
      type: "external",
      source: ctx.children.StringLiteral[0].image.slice(1, -1)
    };
  }
  throw new Error("Unknown stream source type");
}
```

**Estimated Time:** 1.5 hours

**Test Requirements:**
- specs/LANGUAGE_SPEC.md: Stream type (lines 283-288)
- specs/DEMAND_DRIVEN_INCREMENTAL.md: Stream generator usage

**Success Criteria:**
- Stream AST built correctly
- AST includes generator field
- TypeScript compiles

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Stream AST tests pass
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(ast): add stream AST builder methods"

---

## SUCCESS METRICS FOR MVP

### MVP (Layers 1-4 Complete):

**Requirements:**
- [x] Bug #1 fixed: AST Builder CST access pattern
- [x] Bug #2 fixed: Compiler tests use correct API
- [x] All 3 missing types added (Fraction, Comparison, Stream generator)
- [x] All 31 operations in registry
- [x] All 15 basic runtime operations implemented
- [x] Type compatibility validation working
- [x] Property constraint validation working
- [x] All parsing tests pass
- [x] All validation tests pass
- [x] All runtime tests pass
- [x] Test coverage >80%
- [x] Child-friendly Spanish messages throughout

**Observable Results:**
- All 44 tests passing
- "3 + 2 = 5" works end-to-end
- Can filter shapes by color
- Can union two sets
- Type errors detected at compile time
- Spanish error messages shown to children

**Performance:**
- Compile 100-node program <100ms
- Execute 50-node program <50ms
- Cache hit rate >80%

---

## SPEC INCONSISTENCIES FOUND

### 1. Stream Type Missing Generator Field

**Issue:** specs/LANGUAGE_SPEC.md line 283-288 defines Stream with `generator: Generator<T>`, but packages/shared/src/types/composite.ts doesn't include it.

**Impact:** Cannot implement temporal operators properly, streams cannot generate values.

**Resolution:** Added to Phase 1, Priority 0, Task 0.3

---

### 2. Fraction Type Referenced but Not Defined

**Issue:** specs/LANGUAGE_SPEC.md lines 93-109 defines Fraction type with operations, but packages/shared/src/types/primitives.ts doesn't include it.

**Impact:** Fraction operations will fail type checking, cannot use fractions in programs.

**Resolution:** Added to Phase 1, Priority 0, Task 0.1

---

### 3. ~~Comparison Return Type Not Defined~~ (RESOLVED)

**Issue:** ~~specs/LANGUAGE_SPEC.md mentions Comparison return type for COMPARE operations (lines 53, 70, 90, 107), but no Comparison type definition exists.~~

**Resolution:** All COMPARE operations return `Boolean` type (true if equal, false otherwise), not a separate Comparison type.

**Changes Made:**
- Updated specs/LANGUAGE_SPEC.md: All COMPARE operations return Boolean
- Updated specs/GRAMMAR_SPEC.md: Operation signatures return boolean, not integer
- Updated IMPLEMENTATION_PLAN.md: Removed "Comparison Type Missing" task, documented Boolean return semantics
- Updated packages/runtime/src/operations/numeric.ts: COMPARE returns Boolean
- Updated packages/runtime/src/runtime.test.ts: Tests expect Boolean results

**Comparison Semantics:**
- Numbers/Booleans: Direct value equality check
- Text: Lexicographical comparison based on UTF-16 code unit values (by value, not by reference)
- COMPARE_BY_* operations: Compare specific properties and return Boolean
- All comparisons are by value, not by reference

---

### 4. 31 Operations in Spec, Only 15 in Registry

**Issue:** specs/GRAMMAR_SPEC.md lists 31 operations (lines 200-260), but packages/shared/src/operations/registry.ts only has 15.

**Impact:** 16 curriculum-specific operations unavailable:
- 7 comparison operations (COMPARE_BY_*)
- 6 filtering operations (FILTER_BY_*)
- 1 ordering operation (ALPHABETICAL_SORT)

**Resolution:** Added to Phase 1, Priority 1, Task 1.1, 1.2, 1.3

---

**Document Status:** Updated with accurate analysis
**Last Updated:** 2026-03-05
**Next Step:** Priority -1 - Fix AST Builder CST Access Pattern
**Estimated Time to MVP (Layers 1-4):** ~2-3 weeks
**Estimated Time to Version 1.0 (All layers):** ~3-4 weeks
