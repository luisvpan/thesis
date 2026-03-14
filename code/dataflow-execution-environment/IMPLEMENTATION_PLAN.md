# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-14 (Verified codebase against specs/LANGUAGE_SPEC.md - Fraction type clarified)

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

## KEY FINDINGS & CLARIFICATIONS (2026-03-14)

### Fraction Type Definition - CLARIFIED ✅

**Finding:** After detailed analysis of the codebase against `specs/LANGUAGE_SPEC.md`, the Fraction type definition is **CORRECT** as implemented:

**Current Implementation:**
```typescript
// packages/shared/src/types/primitives.ts (lines 26-30)
export type Fraction = {
  kind: "fraction";
  numerator: number;    // ✅ CORRECT per spec
  denominator: number;  // ✅ CORRECT per spec
};
```

**Spec Confirmation:**
```typescript
// specs/LANGUAGE_SPEC.md (lines 94-99)
type Fraction = {
  kind: "fraction";
  numerator: number;    // ← Spec uses number
  denominator: number;  // ← Spec uses number
};
```

**Why `number` not `Integer`?**
- Avoids circular type dependency (Fraction needs Integer, but Integer division returns Fraction)
- Simpler implementation for educational context (ages 6-9)
- All numeric types use `number` internally for value storage
- Type safety maintained via `kind: "fraction"` tag

**Action:** Removed P2.1 task - Fraction type requires NO changes.

---

### Fraction Operations - INCORRECTLY IMPLEMENTED ❌

**Finding:** Fraction operations use WRONG pattern (separate operations instead of overloading).

**Current Implementation (WRONG):**
```typescript
// packages/shared/src/operations/registry.ts (lines 5, 7, 9, 11, 13)
export type Operation =
  | "ADD_FRACTION"      // ❌ WRONG - separate operation
  | "SUBTRACT_FRACTION" // ❌ WRONG - separate operation
  | "MULTIPLY_FRACTION" // ❌ WRONG - separate operation
  | "DIVIDE_FRACTION"    // ❌ WRONG - separate operation
  | "COMPARE_FRACTION"  // ❌ WRONG - separate operation
```

**Problem:** Compiler CANNOT generate these tokens. Parser only generates "ADD", "SUBTRACT", etc.

**Correct Implementation Needed:**
```typescript
// Registry should support overloading:
export type Operation = "ADD" | "SUBTRACT" | "MULTIPLY" | "DIVIDE" | "COMPARE" | ...

// Operation signature with multiple type support:
export type OperationSignature = {
  arity: number;
  contracts: TypeConstraint[];  // Array of signatures for overloading
  category: string;
}

const OPERATION_REGISTRY = {
  ADD: {
    contracts: [
      { inputTypes: ["natural", "natural"], outputType: "natural" },
      { inputTypes: ["fraction", "fraction"], outputType: "fraction" }, // NEW
      { inputTypes: ["integer", "integer"], outputType: "integer" },    // NEW
      { inputTypes: ["decimal", "decimal"], outputType: "decimal" }    // NEW
    ],
    category: "numeric"
  }
};
```

**Runtime Dispatch:**
```typescript
export function ADD(inputs: Array<{ id: string; value: unknown }>): DataType {
  const [a, b] = inputs;
  
  // Dispatch based on input types
  if (isFraction(a.value) && isFraction(b.value)) {
    return addFractions(a.value as Fraction, b.value as Fraction);
  }
  if (isNatural(a.value) && isNatural(b.value)) {
    return addNaturals(a.value as Natural, b.value as Natural);
  }
  if (isInteger(a.value) && isInteger(b.value)) {
    return addIntegers(a.value as Integer, b.value as Integer);
  }
  if (isDecimal(a.value) && isDecimal(b.value)) {
    return addDecimals(a.value as Decimal, b.value as Decimal);
  }
  
  throw new Error(`ADD: Unsupported types ${a.kind}, ${b.kind}`);
}
```

**Impact:** 
- ❌ Fractions cannot be used in real programs (only work in unit tests)
- ❌ Overloading pattern not established for Integer/Decimal
- ❌ Integration tests fail (compiler → runtime broken)
- ✅ Fraction type itself is correct

**Action:** Task P0.2 remains valid and critical.

---

### Confirmed Issues from Previous Analysis

The following issues from the 2026-03-13 analysis remain VALID and require fixing:

1. **Nested Operations NOT HANDLED** (P0.1) - CRITICAL BLOCKER
   - AST builder generates IDs but doesn't create TransformStatements
   - `ADD(a, MULTIPLY(b, c))` fails with undefined reference

2. **Temporal Operators BREAK DEMAND-DRIVEN** (P0.3) - CRITICAL
   - FBY, NEXT, ACCUMULATE manually iterate generators
   - Should use recursive `evaluate()` calls instead

3. **IncrementalRuntime MISSING** (P0.4) - CRITICAL
   - File doesn't exist
   - Blocks WebSocket server implementation

4. **Integer Operations MISSING** (P1.1)
   - 0/6 operations implemented
   - Needed for type system completeness

5. **Decimal Operations MISSING** (P1.2)
   - 0/6 operations implemented
   - Needed for type system completeness

6. **SetType Uses `unknown[]`** (P2.1, formerly P2.2)
   - Should use generic `T[]` for type safety

7. **Color Type Has Extra Values** (P2.2, formerly P2.3)
   - Spec defines 6 colors: red, blue, yellow, green, orange, purple
   - Implementation includes white, black (not in spec)

---

**Summary:** Fraction type is correct as-is. The critical issue is Fraction **operations**, not the type definition. P2.1 task removed from plan. All other P0-P2 tasks remain valid.

---

## Current Implementation Status

**Last Updated:** 2026-03-13 (Comprehensive analysis from 7 parallel subagents completed)

### Overall Progress: ~58% Complete

**Critical Finding:** Previous assessment of ~79% was overly optimistic. Actual status is significantly lower due to:
- Fraction operations incorrectly implemented (separate ops, not overloading)
- **NOTE:** Fraction type definition is CORRECT per spec (uses `number`, not `Integer`)
- Nested operations NOT handled (critical compiler blocker)
- IncrementalRuntime missing (blocks Layer 7)
- Temporal operators break demand-driven semantics
- Multiple operations missing (Integer, Decimal, curriculum type filtering)

### Component Status Summary

| Component | Status | Completion | Critical Issues |
|-----------|--------|------------|-----------------|
| **Shared Package** | Partial | 67% | Fraction type correct per spec (uses `number`), SetType uses `unknown[]` not `T[]`, Color has extra values |
| **Compiler Package** | Partial | 70% | **NESTED OPERATIONS NOT HANDLED** (CRITICAL), Set/Object literal ambiguity, Literal ID mismatch, 3/8 validations missing |
| **Runtime Package** | Partial | 65% | **IncrementalRuntime MISSING** (P0), Temporal operators break demand-driven, No Integer/Decimal operations |
| **HTTP API Package** | Functional | 64% | No Elysia.t validation, no error middleware, returns 200 for validation errors |
| **WebSocket Server Package** | Not Started | 0% | Directory exists but ZERO source code (blocked by IncrementalRuntime) |

### Test Coverage Summary

**Total Tests:** 107 tests, 100% passing

**Distribution:**
- Integration tests: 40 tests (~81% of spec requirements)
- Runtime unit tests: 36 tests (incl. 11 fraction ops - but ops don't work via compiler)
- HTTP API tests: 12 tests
- Compiler tests: 13 tests
- Shared type tests: 14 tests

**Critical Gaps:**
- IncrementalRuntime tests: 0/13 (0%)
- WebSocket server tests: 0/12 (0%)
- Fraction operations: Unit tests only (no integration - bypass compiler)
- Set operations: Tests with `natural` instead of curriculum types
- Temporal operations: Missing full timestep coverage

### Layer Progress (Updated)

| Layer | Status | Completion | Tests | Critical Issues |
|-------|--------|------------|-------|-----------------|
| Layer 1: Foundation | COMPLETE | 100% | 100% | None |
| Layer 2: Arithmetic | PARTIAL | 60% | 100% | **Fractions broken** (wrong pattern), Integer/Decimal missing |
| Layer 3: Curriculum Types | PARTIAL | 60% | 100% | **FILTER operations missing** for Car, Food, Animal (COMPARE exists) |
| Layer 4: Set Operations | PARTIAL | 50% | 100% | Non-generic (only `set<natural>`), SORT/ALPHABETICAL_SORT limitations |
| Layer 5: Temporal Operators | PARTIAL | 60% | 100% | **Breaks demand-driven semantics** (P0) |
| Layer 6: Streams | PARTIAL | 50% | 100% | Non-generic (only `stream<natural>`) |
| Layer 7: Integration | NOT STARTED | 0% | 0% | IncrementalRuntime missing, WebSocket empty, HTTP API needs refactoring |

---

## CRITICAL BLOCKERS

### Blocker 1: Nested Operations Not Handled (P0 - CRITICAL)

**Status:** NOT IMPLEMENTED - AST builder generates IDs but doesn't create TransformStatements

**Impact:**
- **Blocks ANY complex expression** - programs like `ADD(a, MULTIPLY(b, c))` fail with undefined reference
- Cannot build meaningful programs beyond simple operations
- **Breaks core language functionality**

**Current Behavior:**
```typescript
// ast-builder.ts line 189-193
if (ctx.operationExpression) {
  const opExpr = this.visit(ctx.operationExpression);
  const opId = `nested_op_${this.nestedOpCounter++}_${opExpr.operation}`;
  return opId;  // ❌ Returns ID but never creates TransformStatement
}
```

**Required Behavior:**
When encountering nested operation like `ADD(a, MULTIPLY(b, c))`:
1. Create intermediate DataSource/Transformation nodes for each level
2. Generate proper IDs and edges in the graph
3. Store intermediate TransformStatements in the AST
4. Ensure all nodes are present in final DataflowProgram

**Priority:** P0 - CRITICAL (blocks meaningful programs)

**Estimated Time:** 6 hours

**Spec References:**
- `specs/GRAMMAR_SPEC.md` - OperationExpression grammar
- `specs/LANGUAGE_SPEC.md` - Nested operation semantics

---

### Blocker 2: Fraction Operations Implemented Incorrectly (P0 - CRITICAL)

**Status:** INCORRECTLY IMPLEMENTED - Requires refactoring to use overloading

**Current (WRONG) Implementation:**
- ❌ Separate `ADD_FRACTION`, `SUBTRACT_FRACTION`, `MULTIPLY_FRACTION`, `DIVIDE_FRACTION`, `COMPARE_FRACTION` operations defined in registry
- ❌ Runtime has separate functions for each fraction operation
- ❌ **CRITICAL**: Compiler can NEVER generate these tokens - it only generates `ADD`, `SUBTRACT`, etc.
- ❌ Tests use direct `ADD_FRACTION` tokens, which bypass the compiler
- ❌ No integration tests exist (only internal unit tests)

**Impact:**
- ❌ Type system completeness NOT achieved - Fractions cannot be used in real programs
- ❌ Fractions only work in direct runtime calls, not via compiler
- ❌ Integration between compiler and runtime broken
- ❌ Overloading pattern not implemented (critical for Integer/Decimal too)

**Correct Implementation Needed:**
- ✅ Registry should allow `ADD` to accept multiple input types:
  - `[natural, natural]` → `natural` (existing)
  - `[fraction, fraction]` → `fraction` (NEW)
  - `[integer, integer]` → `integer` (NEW - will be needed)
  - `[decimal, decimal]` → `decimal` (NEW - will be needed)
- ✅ Runtime should dispatch to correct implementation based on input types
- ✅ Compiler should generate correct tokens (just `ADD`, not `ADD_FRACTION`)
- ✅ Both compiler and runtime must support the overloading pattern
- ✅ Integration tests must test the full pipeline: Compiler → Runtime

**Priority:** P0 - CRITICAL (blocks fraction type system)

**Estimated Time:** 4 hours (refactor + integration tests)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 93-109

---

### Blocker 3: Temporal Operators Break Demand-Driven Semantics (P0 - CRITICAL)

**Status:** INCORRECTLY IMPLEMENTED - Manual iteration, no recursive evaluate()

**Current (WRONG) Implementation:**
```typescript
// temporal.ts - FBY
export function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [initial, stream] = inputs;

  if (time === 0) {
    return initial.value;  // ❌ Direct value access, no evaluate()
  }

  const streamValue = stream.value as { kind: "stream"; elementType: string; generator: Generator<unknown> };
  const gen = streamValue.generator;
  if (!gen) {
    throw new Error('Stream does not have a generator');
  }

  let result = gen.next();
  let count = 0;

  while (!result.done && count < time - 1) {  // ❌ Manual iteration
    result = gen.next();
    count++;
  }

  return result.done ? initial.value : result.value;
}
```

**Required Behavior:**
- FBY should call `evaluate(stream.id, time - 1, graph)` for time > 0
- ACCUMULATE should recursively evaluate stream values
- Demand-driven propagation must be preserved (no eager iteration)
- Cache should work for temporal operations

**Impact:**
- ❌ Breaks core demand-driven semantics
- ❌ Performance issues (re-evaluating streams)
- ❌ Cannot properly handle dependencies

**Priority:** P0 - CRITICAL (breaks core semantics)

**Estimated Time:** 5 hours

**Spec References:**
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` - Demand-driven evaluation model
- `specs/LANGUAGE_SPEC.md` - Temporal operator semantics

---

### Blocker 4: IncrementalRuntime Not Implemented (P0 - CRITICAL)

**Status:** NOT STARTED - File doesn't exist

**Impact:**
- **Blocks WebSocket Server** - can't implement without this
- Cannot provide IDE live feedback
- Cannot support incremental evaluation for partial programs
- **Blocks completion of Layer 7**

**Missing Implementation:**
- Partial graph evaluation (demand-driven)
- Node state tracking (completed/pending/error)
- Subscription management (multiple clients)
- Graph update API with cache invalidation
- Missing input extraction with Spanish messages

**Priority:** P0 - CRITICAL (blocks MVP completion)

**Estimated Time:** 8 hours

**Spec References:**
- `specs/INTEGRATION_SPEC.md` lines 365-771 (Incremental Runtime design)
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` (Demand-driven semantics)

---

## PRIORITIZED TASK LIST

### P0 CRITICAL (Immediate - Blocks Core Functionality)

#### Task P0.1: Fix Nested Operations Support in Compiler

**Files to update:**
- `packages/compiler/src/ast/ast-builder.ts` (handle nested ops)
- `packages/compiler/src/compiler.ts` (process nested TransformStatements)
- `packages/compiler/src/ast/ast-types.ts` (add IntermediateStatement type)

**Why Critical:**
- **Blocks ANY complex expression** - `ADD(a, MULTIPLY(b, c))` fails with undefined reference
- Cannot build meaningful programs
- AST builder generates IDs but doesn't create TransformStatements

**Current Issue:**
```typescript
// argument() in ast-builder.ts returns just an ID for nested ops
if (ctx.operationExpression) {
  const opExpr = this.visit(ctx.operationExpression);
  const opId = `nested_op_${this.nestedOpCounter++}_${opExpr.operation}`;
  return opId;  // ❌ No TransformStatement created
}
```

**Required Solution:**
1. Add `IntermediateStatement` type to ast-types.ts
2. Modify argument() to create IntermediateStatement for nested ops
3. Update compiler.ts to:
   - Collect all IntermediateStatements from TransformStatements
   - Convert them to DataSource/Transformation nodes
   - Add proper edges between intermediate nodes
   - Handle nested depth properly

**Example of Required Behavior:**
```dataflow
a: natural = 5;
b: natural = 3;
c: natural = 2;
result: natural = ADD(a, MULTIPLY(b, c));
```

Should produce:
- DataSource nodes: a (5), b (3), c (2)
- Transformation node: nested_op_0_MULTIPLY (inputs: b, c, output: natural)
- Transformation node: result (operation: ADD, inputs: a, nested_op_0_MULTIPLY, output: natural)
- Edges: b→nested_op_0_MULTIPLY, c→nested_op_0_MULTIPLY, a→result, nested_op_0_MULTIPLY→result

**Estimated Time:** 6 hours

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md):**
- ✓ Nested operations like `ADD(a, MULTIPLY(b, c))` parse correctly
- ✓ Intermediate nodes are created for each nesting level
- ✓ Proper edges connect intermediate nodes
- ✓ Multiple levels of nesting work (e.g., `ADD(MULTIPLY(a, b), SUBTRACT(c, d))`)
- ✓ Nested operations with literals work (e.g., `ADD(a, MULTIPLY(3, 2))`)
- ✓ Compiler generates valid DataflowProgram
- ✓ Runtime evaluates nested operations correctly
- ✓ Cache works for intermediate results

**Required Tests (Integration):**
- [ ] Test: Simple nested operation `ADD(a, MULTIPLY(b, c))`
- [ ] Test: Multiple levels of nesting `ADD(MULTIPLY(a, b), SUBTRACT(c, d))`
- [ ] Test: Nested operations with literals `ADD(a, MULTIPLY(3, 2))`
- [ ] Test: Complex expression `ADD(SUBTRACT(a, b), MULTIPLY(c, DIVIDE(d, e)))`
- [ ] Test: Nested operations with curriculum types `FILTER_BY_COLOR(UNION(set1, set2), "red")`
- [ ] Test: Cache hits on intermediate results

**Layer:** Cross-layer (Compiler)

**Spec Reference:** `specs/GRAMMAR_SPEC.md` (OperationExpression grammar)

**Ralph Wiggum Checklist:**
- [ ] Nested operations parse correctly
- [ ] Intermediate nodes created properly
- [ ] Edges connect correctly
- [ ] Integration tests pass
- [ ] Unit tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): add nested operations support with intermediate statements"

---

#### Task P0.2: Refactor Fraction Operations to Use Overloading (CRITICAL - INCORRECTLY IMPLEMENTED)

**Status:** ❌ INCORRECTLY IMPLEMENTED - Needs refactoring

**Files to update:**
- `packages/shared/src/operations/registry.ts` (add overloading support, remove ADD_FRACTION etc.)
- `packages/runtime/src/operations/numeric.ts` (implement overloading pattern)
- `packages/runtime/src/evaluator/demand-driven-evaluator.ts` (update dispatcher)
- `packages/compiler/src/` (ensure compiler generates correct tokens)

**Why Critical:**
- ❌ Current implementation uses separate `ADD_FRACTION` operations
- ❌ Compiler can NEVER generate `ADD_FRACTION` tokens (only generates `ADD`)
- ❌ Fractions only work in direct runtime calls, not via compiler
- ❌ Overloading pattern not established (needed for Integer/Decimal too)
- ❌ No integration tests exist (only internal unit tests that bypass compiler)

**Current (WRONG) Code:**
```typescript
// registry.ts - WRONG: Separate operations
ADD_FRACTION: { arity: 2, inputTypes: ["fraction", "fraction"], outputType: "fraction" }

// numeric.ts - WRONG: Separate function
export function ADD_FRACTION(inputs: ...): Fraction { ... }
```

**Correct Implementation:**
```typescript
// registry.ts - CORRECT: Overloading via multiple signatures
ADD: [
  { inputTypes: ["natural", "natural"], outputType: "natural" },
  { inputTypes: ["fraction", "fraction"], outputType: "fraction" },
  { inputTypes: ["integer", "integer"], outputType: "integer" },     // for P1.1
  { inputTypes: ["decimal", "decimal"], outputType: "decimal" }     // for P1.2
]

// numeric.ts - CORRECT: Single dispatcher
export function ADD(inputs: Array<{ id: string; value: unknown }>): DataType {
  const [a, b] = inputs;
  // Dispatch based on input types
  if (isFraction(a.value) && isFraction(b.value)) {
    return addFractions(a.value as Fraction, b.value as Fraction);
  }
  if (isNatural(a.value) && isNatural(b.value)) {
    return addNaturals(a.value as Natural, b.value as Natural);
  }
  // ... other types
}
```

**Requirements:**
1. **Registry Overloading**: Support multiple signatures for same operation name
2. **Runtime Dispatch**: Route to correct implementation based on actual input types
3. **Compiler Compatibility**: Compiler only generates `ADD`, not `ADD_FRACTION`
4. **Integration Tests**: Test full pipeline: Compiler → Runtime → Results
5. **Remove Separate Operations**: Delete `ADD_FRACTION`, `SUBTRACT_FRACTION`, etc. from registry and runtime
6. **Preserve Arithmetic Logic**: Keep the fraction arithmetic logic (GCD simplification, etc.)

**Estimated Time:** 4 hours

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 93-109):**
- ✓ ADD works with [fraction, fraction] inputs → fraction output
- ✓ SUBTRACT works with [fraction, fraction] inputs → fraction output
- ✓ MULTIPLY works with [fraction, fraction] inputs → fraction output
- ✓ DIVIDE works with [fraction, fraction] inputs → fraction output
- ✓ COMPARE works with [fraction, fraction] inputs → boolean output
- ✓ Compiler generates `ADD` token, not `ADD_FRACTION`
- ✓ Runtime dispatches to correct implementation based on input types
- ✓ Full pipeline works: JSON program → Compiler → Runtime → Results

**Required Tests (Integration + Unit):**
- [ ] Integration Test: Fraction addition via compiler pipeline
- [ ] Integration Test: Fraction subtraction via compiler pipeline
- [ ] Integration Test: Fraction multiplication via compiler pipeline
- [ ] Integration Test: Fraction division via compiler pipeline
- [ ] Integration Test: Fraction comparison via compiler pipeline
- [ ] Integration Test: Complex expression with fractions and naturals
- [ ] Unit Test: Registry supports multiple signatures for same operation
- [ ] Unit Test: Runtime dispatches to correct fraction implementation
- [ ] Unit Test: ADD(1/2, 1/4) = 3/4
- [ ] Unit Test: ADD(2/3, 1/3) = 1
- [ ] Unit Test: SUBTRACT(3/4, 1/4) = 1/2
- [ ] Unit Test: MULTIPLY(1/2, 2/3) = 1/3
- [ ] Unit Test: DIVIDE(1/2, 1/4) = 2
- [ ] Unit Test: COMPARE(1/2, 1/2) = true
- [ ] Unit Test: COMPARE(1/2, 1/3) = false
- [ ] Unit Test: Zero denominator error handled
- [ ] Unit Test: All fractions automatically simplified

**Layer:** Layer 2 (Arithmetic)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 93-109

**Ralph Wiggum Checklist:**
- [ ] Registry updated to support overloading (multiple signatures per operation)
- [ ] Runtime dispatcher implemented (routes to correct impl based on input types)
- [ ] Separate ADD_FRACTION operations removed from registry and runtime
- [ ] Integration tests pass (full compiler → runtime pipeline)
- [ ] Unit tests pass (arithmetic correctness)
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "refactor(operations): implement fraction operations via overloading pattern"

---

#### Task P0.3: Fix Temporal Operators to Preserve Demand-Driven Semantics

**File to update:** `packages/runtime/src/operations/temporal.ts`

**Why Critical:**
- ❌ FBY and ACCUMULATE directly access generators
- ❌ No recursive `evaluate()` calls
- ❌ Manual iteration instead of demand-driven propagation
- ❌ Breaks core demand-driven semantics (P0 requirement from PROJECT_GOALS.md)

**Current (WRONG) Implementation:**
```typescript
export function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [initial, stream] = inputs;

  if (time === 0) {
    return initial.value;  // ❌ Direct access, no evaluate()
  }

  const streamValue = stream.value as { kind: "stream"; elementType: string; generator: Generator<unknown> };
  const gen = streamValue.generator;

  let result = gen.next();
  let count = 0;

  while (!result.done && count < time - 1) {  // ❌ Manual iteration
    result = gen.next();
    count++;
  }

  return result.done ? initial.value : result.value;
}
```

**Correct Implementation:**
```typescript
export function FBY(inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
  const [initial, stream] = inputs;

  if (time === 0) {
    return initial.value;  // At t=0, return initial value
  }

  // ✅ Recursively evaluate stream at previous time
  return this.evaluate(stream.id, time - 1, graph);
}

// Note: This requires making FBY a method of DemandDrivenEvaluator or passing evaluate function
```

**Key Requirements:**
1. **FBY**: Return initial.value at time=0, evaluate(stream.id, time-1) otherwise
2. **NEXT**: Evaluate stream at current time (not manual iteration)
3. **ACCUMULATE**: Evaluate stream at each timestep, accumulate recursively
4. **Cache Integration**: Let DemandDrivenEvaluator handle caching
5. **Demand-Driven**: Only evaluate when demanded (not eager)

**Estimated Time:** 5 hours

**Dependencies:** None

**Acceptance Criteria (from specs/DEMAND_DRIVEN_INCREMENTAL.md):**
- ✓ FBY uses recursive evaluate() calls
- ✓ FBY respects demand-driven semantics (no eager evaluation)
- ✓ ACCUMULATE evaluates stream at each timestep recursively
- ✓ Cache works for temporal operations
- ✓ NEXT evaluates stream at requested time
- ✓ Performance: temporal ops <10ms per evaluation
- ✓ No manual generator iteration

**Required Tests:**
- [ ] Test: FBY returns initial at time=0
- [ ] Test: FBY evaluates stream at time-1 for time>0
- [ ] Test: ACCUMULATE accumulates across timesteps
- [ ] Test: NEXT evaluates stream at current time
- [ ] Test: Cache hits on repeated temporal evaluations
- [ ] Test: Temporal ops respect demand-driven semantics
- [ ] Benchmark: Temporal operations <10ms per evaluation

**Layer:** Layer 5 (Temporal Operators)

**Spec References:**
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` - Demand-driven evaluation model
- `specs/LANGUAGE_SPEC.md` - Temporal operator semantics

**Ralph Wiggum Checklist:**
- [ ] FBY uses recursive evaluate() calls
- [ ] ACCUMULATE uses recursive evaluate() calls
- [ ] NEXT uses recursive evaluate() calls
- [ ] No manual generator iteration
- [ ] Demand-driven semantics preserved
- [ ] Cache works for temporal operations
- [ ] Temporal tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(runtime): temporal operators now use demand-driven evaluation"

---

#### Task P0.4: Implement IncrementalRuntime Class

**File to create:** `packages/runtime/src/incremental-runtime.ts`

**Why Critical:**
- **Required by WebSocket server** - can't implement P3.1 without this
- Enables partial graph evaluation during program construction
- Required by specs/INTEGRATION_SPEC.md and specs/DEMAND_DRIVEN_INCREMENTAL.md
- Key differentiator for IDE live feedback
- **Blocks completion of Layer 7**

**Key Requirements:**
1. **100% Demand-Driven** - No exceptions, no eager evaluation
2. **Demand Sources:** Output nodes OR subscriptions
3. **Missing Inputs:** Return "pending" state (not error)
4. **Cache Invalidation:** Only re-evaluate affected nodes on update
5. **Subscription Management:** Multiple clients, node state notifications

**API Design:**
```typescript
class IncrementalRuntime {
  // Load/update partial graph
  updateGraph(graph: DataflowProgram): void

  // Evaluate nodes based on subscriptions or outputs
  evaluatePartial(time: number): Map<string, NodeEvaluationResult>

  // Subscribe to node (creates demand)
  subscribe(clientId: string, nodeId: string): void

  // Unsubscribe from node
  unsubscribe(clientId: string, nodeId: string): void

  // Get current state of all nodes
  getNodeStates(): Map<string, NodeState>
}

type NodeState = "completed" | "pending" | "error"

type NodeEvaluationResult = {
  nodeId: string
  status: NodeState
  value?: DataType
  missingInputs?: MissingInput[]
  error?: Error
}

type MissingInput = {
  port: number
  nodeId: string
  description: string  // Child-friendly Spanish
}
```

**Estimated Time:** 8 hours

**Dependencies:** P0.3 (Fix Temporal Operators) - temporal ops must be demand-driven first

**Acceptance Criteria (from specs/INTEGRATION_SPEC.md lines 740-770):**
- ✓ Evaluates partial graphs (incomplete programs)
- ✓ Returns results for computable nodes
- ✓ Marks nodes as "pending" when inputs missing
- ✓ Missing input messages in Spanish with multiple inputs support
- ✓ Updates incrementally when graph changes
- ✓ Only re-evaluates affected nodes on update
- ✓ Subscribers receive notifications on state changes
- ✓ Partial graph (just "5") → { n1: { status: "completed", value: 5 } }
- ✓ Add "3" + "ADD" → { add: { status: "completed", value: 8 } }
- ✓ FILTER missing color → { filter: { status: "pending", missingInputs: [...] } }
- ✓ Update "5" to "7" → only re-evaluate nodes depending on n1
- ✓ Multiple missing inputs reported in missingInputs array with Spanish descriptions
- ✓ Empty graph → no nodes, no results
- ✓ All nodes pending → valid state, nothing computable
- ✓ Remove node with dependents → mark dependents as pending
- ✓ updateGraph() <10ms (p95)
- ✓ evaluatePartial() <20ms (p95) for 50-node partial graphs
- ✓ Incremental update 5x faster than full re-evaluation
- ✓ Demand-driven semantics preserved (only evaluate when demanded, NOT eager)

**Required Tests (13 tests):**
- [ ] Test: Evaluate partial graph with single node
- [ ] Test: Evaluate partial graph with missing inputs (pending state)
- [ ] Test: Multiple missing inputs reported in missingInputs array
- [ ] Test: Update graph triggers only affected node re-evaluation
- [ ] Test: Subscription to node receives state changes
- [ ] Test: Demand-driven semantics (no evaluation without demand)
- [ ] Test: Unsubscribe stops receiving state changes
- [ ] Test: Empty graph returns no results
- [ ] Test: All nodes pending is valid state
- [ ] Test: Remove node marks dependents as pending
- [ ] Test: Missing input messages are child-friendly Spanish
- [ ] Benchmark: updateGraph() <10ms (p95)
- [ ] Benchmark: evaluatePartial() <20ms (p95) for 50 nodes

**Spec Reference:**
- `specs/INTEGRATION_SPEC.md` lines 365-771 (Incremental Runtime design)
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` (Demand-driven semantics)

**Layer:** Layer 7 (Integration)

**Ralph Wiggum Checklist:**
- [ ] IncrementalRuntime class implemented
- [ ] Demand-driven semantics preserved (no eager evaluation)
- [ ] 13 Incremental runtime tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement IncrementalRuntime class with demand-driven partial evaluation"

---

### P1 HIGH (Required for MVP - Layer 7 & Core Functionality)

#### Task P1.1: Implement Integer Operations (via Overloading)

**Files to update:**
- `packages/shared/src/operations/registry.ts` (add Integer operation signatures)
- `packages/runtime/src/operations/numeric.ts` (implement Integer operations)

**Why Important:**
- Completes numeric type system
- SUBTRACT on Natural produces Integer but Integer has no operations
- Type system must be complete for educational curriculum

**Operations to Implement (via overloading):**
```typescript
ADD(Integer, Integer) → Integer
SUBTRACT(Integer, Integer) → Integer
MULTIPLY(Integer, Integer) → Integer
DIVIDE(Integer, Integer) → Decimal
COMPARE(Integer, Integer) → Boolean
FILTER(list: Integer[], condition) → Integer[]
SORT(list: Integer[]) → Integer[]
```

**Estimated Time:** 2 hours

**Dependencies:** P0.2 (Fraction Overloading Pattern) - use same pattern

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 57-72):**
- ✓ ADD works with Integer inputs
- ✓ SUBTRACT works with Integer inputs (can produce negative results)
- ✓ MULTIPLY works with Integer inputs
- ✓ DIVIDE works with Integer inputs (produces Decimal)
- ✓ COMPARE works with Integer inputs
- ✓ Type safety enforced

**Required Tests:**
- [ ] Test: ADD(Integer, Integer) works correctly
- [ ] Test: SUBTRACT(Integer, Integer) handles negative results
- [ ] Test: MULTIPLY(Integer, Integer) works correctly
- [ ] Test: DIVIDE(Integer, Integer) produces Decimal
- [ ] Test: COMPARE(Integer, Integer) returns correct boolean
- [ ] Test: Integer operations integrate with Natural operations

**Layer:** Layer 2 (Arithmetic)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 57-72

**Ralph Wiggum Checklist:**
- [ ] Integer operations implemented (via overloading)
- [ ] Integer tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement integer operations via overloading"

---

#### Task P1.2: Implement Decimal Operations (via Overloading)

**Files to update:**
- `packages/shared/src/operations/registry.ts` (add Decimal operation signatures)
- `packages/runtime/src/operations/numeric.ts` (implement Decimal operations)

**Why Important:**
- Completes numeric type system
- DIVIDE on Natural/Integer produces Decimal but Decimal has no operations
- Type system must be complete for educational curriculum

**Operations to Implement (via overloading):**
```typescript
ADD(Decimal, Decimal) → Decimal
SUBTRACT(Decimal, Decimal) → Decimal
MULTIPLY(Decimal, Decimal) → Decimal
DIVIDE(Decimal, Decimal) → Decimal
COMPARE(Decimal, Decimal) → Boolean
FILTER(list: Decimal[], condition) → Decimal[]
SORT(list: Decimal[]) → Decimal[]
```

**Estimated Time:** 2 hours

**Dependencies:** P0.2 (Fraction Overloading Pattern) - use same pattern

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 74-91):**
- ✓ ADD works with Decimal inputs
- ✓ SUBTRACT works with Decimal inputs
- ✓ MULTIPLY works with Decimal inputs
- ✓ DIVIDE works with Decimal inputs
- ✓ COMPARE works with Decimal inputs
- ✓ Type safety enforced

**Required Tests:**
- [ ] Test: ADD(Decimal, Decimal) works correctly
- [ ] Test: SUBTRACT(Decimal, Decimal) works correctly
- [ ] Test: MULTIPLY(Decimal, Decimal) works correctly
- [ ] Test: DIVIDE(Decimal, Decimal) works correctly
- [ ] Test: COMPARE(Decimal, Decimal) returns correct boolean
- [ ] Test: Decimal operations integrate with other numeric types

**Layer:** Layer 2 (Arithmetic)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 74-91

**Ralph Wiggum Checklist:**
- [ ] Decimal operations implemented (via overloading)
- [ ] Decimal tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement decimal operations via overloading"

---

#### Task P1.3: Implement Curriculum Type Filtering Operations

**Files to update:**
- `packages/shared/src/operations/registry.ts` (add Car, Food, Animal filtering signatures)
- `packages/runtime/src/operations/filtering.ts` (implement Car, Food, Animal filters)
- `packages/shared/src/types/composite.ts` (verify Car, Food, Animal types)

**Why Important:**
- Curriculum types are core to educational goals (ages 6-9)
- FILTER operations are essential for set manipulation
- COMPARE operations exist but FILTER operations missing
- Incomplete support for curriculum types

**Missing Operations:**
```typescript
// Car
FILTER_BY_COLOR(list: Car[], color: Color) → Car[]

// Food
FILTER_BY_TASTE(list: Food[], taste: Taste) → Food[]
FILTER_BY_COLOR(list: Food[], color: Color) → Food[]

// Animal
FILTER_BY_TYPE(list: Animal[], type: AnimalType) → Animal[]
FILTER_BY_COLOR(list: Animal[], color: Color) → Animal[]
```

**Estimated Time:** 3 hours

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 188-252):**
- ✓ FILTER_BY_COLOR works with Car sets
- ✓ FILTER_BY_TASTE works with Food sets
- ✓ FILTER_BY_COLOR works with Food sets
- ✓ FILTER_BY_TYPE works with Animal sets
- ✓ FILTER_BY_COLOR works with Animal sets
- ✓ Type safety enforced for curriculum types

**Required Tests:**
- [ ] Test: FILTER_BY_COLOR on Car set works correctly
- [ ] Test: FILTER_BY_TASTE on Food set works correctly
- [ ] Test: FILTER_BY_COLOR on Food set works correctly
- [ ] Test: FILTER_BY_TYPE on Animal set works correctly
- [ ] Test: FILTER_BY_COLOR on Animal set works correctly
- [ ] Test: Curriculum type filters preserve type

**Layer:** Layer 3 (Curriculum Types)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 188-252

**Ralph Wiggum Checklist:**
- [ ] Car filtering operations implemented
- [ ] Food filtering operations implemented
- [ ] Animal filtering operations implemented
- [ ] Filtering tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement curriculum type filtering operations"

---

#### Task P1.4: Make Set/Stream Operations Generic

**Files to update:**
- `packages/shared/src/types/composite.ts` (SetType uses `unknown[]`, change to generic `T[]`)
- `packages/runtime/src/evaluator/demand-driven-evaluator.ts` (update set wrapping logic)
- `packages/runtime/src/operations/sets.ts` (currently only natural)
- `packages/shared/src/operations/registry.ts` (update type signatures to support any type)
- `packages/runtime/src/operations/temporal.ts` (update temporal for generic streams)

**Why Important:**
- Current set/stream operations only work with natural numbers
- Generic types would support any element type (Shape, Car, Food, Animal, Person)
- Aligns with curriculum (sets of shapes, colors, etc.)
- Fix SetType to use proper generic type `T[]` instead of `unknown[]`

**Estimated Time:** 5 hours

**Dependencies:** P1.3 (Curriculum Type Filters) - need operations to test with

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 312-327):**
- ✓ Set<T> where T can be any DataType
- ✓ Stream<T> where T can be any DataType
- ✓ Set operations work with any element type (not just natural)
- ✓ Stream operations support any element type
- ✓ Inherited operations work for element types
- ✓ SetType uses `T[]` instead of `unknown[]`

**Required Tests:**
- [ ] Test: Set<shape> - UNION of shape sets works
- [ ] Test: Set<car> - FILTER_BY_COLOR on car set works
- [ ] Test: Set<food> - INTERSECTION of food sets works
- [ ] Test: Set<animal> - DIFFERENCE of animal sets works
- [ ] Test: Stream<shape> - temporal operations on shape streams work
- [ ] Test: Generic types properly enforced in type checker
- [ ] Test: Set operations maintain element type through transformations
- [ ] Test: SetType uses `T[]` not `unknown[]`

**Layer:** Layer 4 (Sets) and Layer 6 (Streams)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 312-327

**Ralph Wiggum Checklist:**
- [ ] Generic Set/Stream types implemented
- [ ] SetType uses `T[]` instead of `unknown[]`
- [ ] Generic operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "refactor(types): make set/stream operations generic"

---

#### Task P1.5: Fix Set/Object Literal Ambiguity

**Files to update:**
- `packages/compiler/src/parser/dataflow-parser.ts` (grammar GATE logic)
- `packages/compiler/src/ast/ast-builder.ts` (literal handling)
- `packages/compiler/src/compiler.ts` (literal node creation)

**Why Important:**
- Both Set and Object literals use `{}` syntax
- Parser uses fragile GATE that may fail
- Object literals cause parsing errors
- Affects curriculum types (Shape, Car, Food, Animal)

**Current Issue:**
```typescript
// Both parse to same starting tokens
setLiteral: '{' value+ '}'
objectLiteral: '{' identifier ':' literal+ '}'
```

**Proposed Solution:**
- Use lookahead or proper GATE with context
- Update grammar to clearly distinguish between the two
- Improve error messages for ambiguous cases

**Estimated Time:** 3 hours

**Dependencies:** None

**Acceptance Criteria:**
- ✓ Set literals parse correctly: `{ "red", "blue", "green" }`
- ✓ Object literals parse correctly: `{ color: "red", size: "small" }`
- ✓ Parser disambiguates correctly
- ✓ No fragile GATE logic
- ✓ Clear error messages for malformed literals
- ✓ Compiler generates correct nodes for both types

**Required Tests:**
- [ ] Test: Set literal parses correctly
- [ ] Test: Object literal parses correctly
- [ ] Test: Set vs Object disambiguation works
- [ ] Test: Malformed set literal produces clear error
- [ ] Test: Malformed object literal produces clear error
- [ ] Test: Literal nodes created correctly

**Layer:** Cross-layer (Compiler + Runtime)

**Spec Reference:** `specs/GRAMMAR_SPEC.md` (Literal grammar)

**Ralph Wiggum Checklist:**
- [ ] Set/Object ambiguity resolved
- [ ] Set literals parse correctly
- [ ] Object literals parse correctly
- [ ] Parser tests pass
- [ ] Integration tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(parser): resolve set/object literal ambiguity"

---

#### Task P1.6: Implement Missing Compiler Validation

**File to update:** `packages/compiler/src/validation/dag-validator.ts`

**Why Important:**
- Improves quality of error messages for young learners
- Catches more errors at compile time (better UX)
- Programs without output nodes should fail fast
- Set homogeneity prevents runtime type errors

**Missing Validation:**
1. Set homogeneity validation (all elements in set must be same type)
2. Output node requirement (programs must have at least one output node)
3. Literal type validation (literals match declared type)
4. Stream source validation (generator/sensor exists)

**Estimated Time:** 3 hours

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 431-440):**
- ✓ Set operations verify all elements are same type
- ✓ Programs without output nodes produce error
- ✓ Type safety rules enforced at compile time
- ✓ Child-friendly Spanish error messages
- ✓ Literal values match declared types
- ✓ Stream sources are valid (generator/sensor exists)

**Required Tests:**
- [ ] Test: Set homogeneity validation - mixed types in set error
- [ ] Test: Set homogeneity validation - same types in set succeed
- [ ] Test: Output node requirement - program with no outputs fails
- [ ] Test: Output node requirement - program with outputs succeeds
- [ ] Test: Literal type validation - mismatch produces error
- [ ] Test: Stream source validation - invalid generator produces error
- [ ] Test: Validation errors include child-friendly Spanish messages

**Layer:** Cross-layer (validation quality)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` validation section

**Ralph Wiggum Checklist:**
- [ ] Missing validation rules implemented (set homogeneity, output node, literals, streams)
- [ ] Validation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): add missing compiler validation rules"

---

#### Task P1.7: Refactor HTTP API to Follow Elysia Best Practices

**Files to refactor:**
- `packages/http-api/src/server.ts` (use Elysia plugins, add error middleware)
- `packages/http-api/src/routes.ts` (use Elysia.t for schema validation)
- `packages/http-api/src/index.ts` (add Eden Treaty for end-to-end type safety)

**Why Important:**
- **Current implementation works but doesn't follow Elysia best practices**
- Missing Elysia.t validation (runtime + compile-time type safety)
- Missing error handling middleware
- Missing plugin pattern for route organization
- Missing Eden Treaty for end-to-end type safety
- Uses manual type assertions instead of schema validation
- Returns 200 OK for validation errors (should be 422)
- **Improves code quality and maintainability**

**Estimated Time:** 11 hours

**Dependencies:** None (can start immediately)

**Acceptance Criteria (from specs/ELYSIA_LLMS.md):**
- ✅ Use Elysia.t for request/response schema validation
- ✅ Implement error handling middleware (Elysia.onError)
- ✅ Use plugin pattern for route organization
- ✅ Add Eden Treaty for end-to-end type safety
- ✅ Remove manual type assertions, use schema validation
- ✅ Return 422 for validation errors (not 200 OK)
- ✅ Follow Elysia best practices

**Required Refactoring:**
- [ ] Replace manual type assertions with Elysia.t schema validation
- [ ] Add global error handling middleware (Elysia.onError)
- [ ] Organize routes using plugin pattern
- [ ] Add Eden Treaty export for client type safety
- [ ] Use Elysia plugins for CORS, logging, etc.
- [ ] Return 422 status for validation errors
- [ ] All existing tests still pass (12/12 HTTP API tests)
- [ ] Typecheck passes with no errors
- [ ] Code follows Elysia best practices from specs/ELYSIA_LLMS.md

**Layer:** Layer 7 (Integration)

**Spec Reference:** `specs/ELYSIA_LLMS.md` (Best Practices, Validation, Error Handling, Plugin Pattern, Eden Treaty)

**Ralph Wiggum Checklist:**
- [ ] HTTP API refactored to follow Elysia best practices
- [ ] Elysia.t validation implemented for all endpoints
- [ ] Error handling middleware added
- [ ] Plugin pattern used for route organization
- [ ] Eden Treaty configured for end-to-end type safety
- [ ] Returns 422 for validation errors
- [ ] All 12/12 HTTP API tests still pass
- [ ] Typecheck passes
- [ ] Git commit with message: "refactor(http-api): implement Elysia best practices"

---

### P2 MEDIUM (Quality Improvements & Completeness)

#### CLARIFICATION: Fraction Type Definition is CORRECT

**Status:** ✅ VERIFIED - No changes needed

**Finding:**
After verifying the codebase against `specs/LANGUAGE_SPEC.md` lines 94-99, the current Fraction type definition is **CORRECT**:

```typescript
// Current implementation (packages/shared/src/types/primitives.ts lines 26-30)
export type Fraction = {
  kind: "fraction";
  numerator: number;    // ✅ CORRECT per spec
  denominator: number;  // ✅ CORRECT per spec
};
```

**Spec Confirmation:**
The spec explicitly defines Fraction as using `number` type (not `Integer`):

```typescript
// specs/LANGUAGE_SPEC.md lines 94-99
type Fraction = {
  kind: "fraction";
  numerator: number;    // ← Spec uses number
  denominator: number;  // ← Spec uses number
};
```

**Rationale:**
Using `number` (not `Integer`) for Fraction components is intentional:
- Avoids circular type dependency (Fraction depends on Integer, Integer would depend on Fraction for division results)
- Simpler implementation for educational context
- All numeric types (Natural, Integer, Decimal, Fraction) use `number` internally
- Type is validated by `kind: "fraction"` tag

**Action:** Remove P2.1 task - Fraction type is correct, no changes needed.

---

#### Task P2.2: Fix SetType Generic Type (formerly P2.2)

**File to update:** `packages/shared/src/types/composite.ts`

**Why Important:**
- SetType uses `unknown[]` instead of `T[]`
- Breaks type safety for set operations
- Should be generic `Set<T>` where T is element type

**Current Code:**
```typescript
export type SetType = {
  kind: "set";
  elementType: string;
  elements: unknown[];  // WRONG - should be T[]
};
```

**Should Be:**
```typescript
export type SetType<T extends DataType> = {
  kind: "set";
  elementType: T;
  elements: T[];
};
```

**Estimated Time:** 1 hour

**Dependencies:** P1.4 (Make Set/Stream Generic)

**Acceptance Criteria:**
- ✓ SetType uses generic `T[]` instead of `unknown[]`
- ✓ Type safety maintained for set elements
- ✓ Typecheck passes

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 315-320

**Ralph Wiggum Checklist:**
- [ ] SetType updated to use generic `T[]`
- [ ] Type safety maintained
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(types): use generic T[] for SetType elements"

---

#### Task P2.3: Fix Color Type Extra Values

**File to update:** `packages/shared/src/types/primitives.ts` or curriculum types

**Why Important:**
- Color type has extra values (`white`, `black`) not in spec
- Spec defines only 6 colors: "red", "blue", "yellow", "green", "orange", "purple"
- Mismatch could cause validation errors

**Current Code:**
```typescript
type Color = "red" | "blue" | "yellow" | "green" | "orange" | "purple" | "white" | "black";
```

**Should Be:**
```typescript
type Color = "red" | "blue" | "yellow" | "green" | "orange" | "purple";
```

**Estimated Time:** 0.5 hours

**Dependencies:** None

**Acceptance Criteria:**
- ✓ Color type matches spec (6 colors only)
- ✓ Typecheck passes
- ✓ No white/black values

**Spec Reference:** `specs/LANGUAGE_SPEC.md` line 149

**Ralph Wiggum Checklist:**
- [ ] Color type matches spec exactly
- [ ] Extra values removed
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(types): remove extra color values (white, black)"

---

#### Task P2.4: Complete Child-Friendly Spanish Error Messages

**Files to update:**
- All validation files (`packages/compiler/src/validation/*`)
- All runtime error handling files
- HTTP API error responses (if needed for additional errors)
- WebSocket error responses (if needed for additional errors)

**Why Important:**
- Error messages must be child-friendly for target demographic (ages 6-9)
- Spanish language required by specs
- Improves user experience and learning outcomes
- Better educational value

**Estimated Time:** 3 hours

**Dependencies:** P1.6 (Missing Compiler Validation)

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 440-527):**
- ✓ All error messages in child-friendly Spanish
- ✓ Missing input messages are age-appropriate
- ✓ Type error messages use simple language
- ✓ Property error messages explain what's missing
- ✓ Multiple missing inputs reported in missingInputs array
- ✓ Simple language (no technical jargon)
- ✓ Clear indication of problem location
- ✓ Actionable suggestions included
- ✓ Examples of correct usage

**Required Tests:**
- [ ] Test: Compilation errors return childMessage in Spanish
- [ ] Test: Type errors use simple, age-appropriate language
- [ ] Test: Missing input errors explain what's needed in Spanish
- [ ] Test: Cycle detection errors have actionable suggestions
- [ ] Test: Multiple missing inputs reported with Spanish descriptions

**Layer:** All layers (user experience)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 440-527

**Ralph Wiggum Checklist:**
- [ ] Child-friendly Spanish messages added
- [ ] Message tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(messages): complete child-friendly Spanish error messages"

---

#### Task P2.5: Improve SORT/ALPHABETICAL_SORT Operations

**Files to update:**
- `packages/runtime/src/operations/sets.ts` (SORT, ALPHABETICAL_SORT)

**Why Important:**
- **SORT operations have limitations** that affect functionality
- SORT only works with numbers (not generic)
- ALPHABETICAL_SORT converts everything to string (loses type safety)
- These limitations reduce curriculum flexibility

**Current Limitations:**
- SORT: Only sorts numbers
- ALPHABETICAL_SORT: Converts all elements to strings (loses type info)

**Proposed Improvements:**
- Make SORT generic: SORT<T>(set: Set<T>) → Set<T> where T is Comparable
- Make ALPHABETICAL_SORT type-safe: ALPHABETICAL_SORT<T>(set: Set<T>) → Set<T> where T is String
- Add type constraints to ensure comparability

**Estimated Time:** 2 hours

**Dependencies:** P1.4 (Make Set/Stream Operations Generic)

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md):**
- ✓ SORT operation works with any comparable type (not just numbers)
- ✓ ALPHABETICAL_SORT preserves type (doesn't convert to string)
- ✓ Type safety enforced for sorting operations

**Required Tests:**
- [ ] Test: SORT with numbers works (existing behavior)
- [ ] Test: SORT with strings works alphabetically
- [ ] Test: ALPHABETICAL_SORT preserves string type
- [ ] Test: Type errors raised for non-comparable types in SORT
- [ ] Test: All existing set operation tests still pass

**Layer:** Layer 4 (Sets)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` sorting operations section

**Ralph Wiggum Checklist:**
- [ ] SORT operation improved to work with comparable types
- [ ] ALPHABETICAL_SORT preserves type safety
- [ ] Sorting tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(operations): improve SORT/ALPHABETICAL_SORT type safety"

---

#### Task P2.6: Fix DataType Recursion

**File to update:** `packages/shared/src/types/composite.ts`

**Why Important:**
- DataType recursion uses `string | DataType` instead of just `DataType`
- Reduces type safety
- Could cause validation issues

**Estimated Time:** 0.5 hours

**Dependencies:** None

**Acceptance Criteria:**
- ✓ DataType uses just `DataType` in recursive definitions
- ✓ Typecheck passes

**Ralph Wiggum Checklist:**
- [ ] DataType recursion fixed
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(types): fix DataType recursion"

---

### P3 LOW (Post-MVP - Nice to have)

#### Task P3.1: Implement WebSocket Server (Live Mode)

**Files to create:**
- `packages/websocket-server/src/server.ts`
- `packages/websocket-server/src/handlers.ts`
- `packages/websocket-server/src/index.ts`

**Why Important:**
- Required by specs/INTEGRATION_SPEC.md for IDE integration
- Enables live feedback during program construction
- Provides real-time validation as children build programs
- Critical for educational experience but can be deferred for MVP

**Estimated Time:** 12 hours

**Dependencies:**
- P0.4 (IncrementalRuntime) - REQUIRED
- P1.7 (HTTP API Best Practices) - for consistency

**Acceptance Criteria (from specs/INTEGRATION_SPEC.md lines 200-361):**
- ✓ Client connects via ws://localhost:3000/live
- ✓ Server validates messages and responds with messageId
- ✓ Server handles validate_program requests
- ✓ Server handles evaluate_incremental requests
- ✓ Server handles subscribe_node requests
- ✓ Server handles unsubscribe_node requests
- ✓ Server pushes node_state_changed when subscribed
- ✓ Multiple clients can connect simultaneously
- ✓ Connection is resilient (reconnects on disconnect)
- ✓ Error messages include child-friendly Spanish text for ages 6-9
- ✓ Valid program → validation_result with empty errors
- ✓ Partial program → evaluation_result with mixed statuses
- ✓ Subscribe to node → receive state changes on updates
- ✓ Malformed message → error response
- ✓ Message roundtrip <50ms (p95)
- ✓ Handles 5 concurrent WebSocket connections
- ✓ No memory leaks on client disconnect

**Required Tests:**
- [ ] Test: Client connects to ws://localhost:3000/live
- [ ] Test: validate_program message returns validation_result
- [ ] Test: evaluate_incremental returns node states
- [ ] Test: subscribe_node receives node_state_changed on updates
- [ ] Test: unsubscribe_node stops receiving updates
- [ ] Test: Multiple clients connect simultaneously
- [ ] Test: Client disconnect and reconnect
- [ ] Test: Malformed message returns error response
- [ ] Test: Error messages are child-friendly Spanish
- [ ] Test: 5 concurrent WebSocket connections handled correctly
- [ ] Benchmark: Message roundtrip <50ms (p95)
- [ ] Benchmark: No memory leaks on client disconnect (100 connect/disconnect cycles)

**Layer:** Layer 7 (Integration)

**Spec Reference:** `specs/INTEGRATION_SPEC.md` lines 200-361

**Ralph Wiggum Checklist:**
- [ ] WebSocket server implemented
- [ ] WebSocket tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Multiple concurrent clients work
- [ ] Git commit with message: "feat(websocket): implement WebSocket server"

---

#### Task P3.2: Complete Execution Trace Implementation

**File to update:** `packages/runtime/src/runtime.ts`

**Why Important:**
- Execution trace partial implementation (executionOrder and nodeEvaluations empty)
- Useful for debugging and educational purposes
- Nice to have but not critical for MVP

**Estimated Time:** 2 hours

**Dependencies:** None

**Acceptance Criteria:**
- ✓ executionOrder array populated with node evaluation order
- ✓ nodeEvaluations map populated with evaluation results
- ✓ Execution trace available via API

**Layer:** Cross-layer (debugging support)

**Ralph Wiggum Checklist:**
- [ ] Execution trace fully implemented
- [ ] executionOrder populated
- [ ] nodeEvaluations populated
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): complete execution trace implementation"

---

#### Task P3.3: Add Parallelism to Demand-Driven Evaluation

**File to update:** `packages/runtime/src/evaluator/demand-driven-evaluator.ts`

**Why Important:**
- Current implementation is sequential
- Performance improvement opportunity
- Not critical for MVP but improves scalability

**Proposed Implementation:**
- Use `Promise.all()` to evaluate independent inputs in parallel
- Maintain demand-driven semantics
- No eager evaluation

**Estimated Time:** 3 hours

**Dependencies:** None

**Acceptance Criteria:**
- ✓ Independent evaluations happen in parallel
- ✓ Demand-driven semantics preserved
- ✓ Performance improvement measurable
- ✓ No race conditions

**Ralph Wiggum Checklist:**
- [ ] Parallelism implemented
- [ ] Demand-driven semantics preserved
- [ ] Performance tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "perf(runtime): add parallelism to demand-driven evaluation"

---

## SUMMARY TABLE

| Task | Priority | Status | Time | Impact | Dependencies |
|------|----------|--------|------|--------|--------------|
| **P0.1: Fix nested operations** | P0 | NOT STARTED | 6h | **CRITICAL** - blocks complex expressions | None |
| **P0.2: Refactor Fraction ops (overloading)** | P0 | ❌ INCORRECTLY IMPLEMENTED | 4h | **CRITICAL** - compiler can't generate tokens | None |
| **P0.3: Fix temporal operators** | P0 | ❌ INCORRECTLY IMPLEMENTED | 5h | **CRITICAL** - breaks demand-driven | None |
| **P0.4: Implement IncrementalRuntime** | P0 | NOT STARTED | 8h | **CRITICAL** - blocks Layer 7 | P0.3 |
| **P1.1: Implement Integer operations** | P1 | NOT STARTED | 2h | HIGH - type system completeness | P0.2 |
| **P1.2: Implement Decimal operations** | P1 | NOT STARTED | 2h | HIGH - type system completeness | P0.2 |
| **P1.3: Curriculum type filters** | P1 | NOT STARTED | 3h | HIGH - curriculum alignment | None |
| **P1.4: Make Set/Stream generic** | P1 | NOT STARTED | 5h | HIGH - removes natural restriction | P1.3 |
| **P1.5: Fix Set/Object ambiguity** | P1 | NOT STARTED | 3h | HIGH - parser robustness | None |
| **P1.6: Missing compiler validation** | P1 | NOT STARTED | 3h | HIGH - quality improvement | None |
| **P1.7: Refactor HTTP API (Elysia)** | P1 | NOT STARTED | 11h | Quality & maintainability | None |
| **P2.1: Fix SetType generic** | P2 | NOT STARTED | 1h | Type safety | P1.4 |
| **P2.2: Fix Color type** | P2 | NOT STARTED | 0.5h | Spec compliance | None |
| **P2.3: Spanish error messages** | P2 | NOT STARTED | 3h | Better UX for ages 6-9 | P1.6 |
| **P2.4: Improve SORT ops** | P2 | NOT STARTED | 2h | Improves functionality | P1.4 |
| **P2.5: Fix DataType recursion** | P2 | NOT STARTED | 0.5h | Type safety | None |
| **P3.1: WebSocket Server** | P3 | NOT STARTED | 12h | IDE live feedback | P0.4, P1.7 |
| **P3.2: Execution trace** | P3 | NOT STARTED | 2h | Debugging support | None |
| **P3.3: Add parallelism** | P3 | NOT STARTED | 3h | Performance | None |
 
**Total Estimated Time:**
- **P0 Tasks:** 23 hours
- **P1 Tasks:** 29 hours
- **P2 Tasks:** 10 hours (removed P2.1 - Fraction type is correct)
- **P3 Tasks:** 17 hours
- **Grand Total:** 79 hours

---

## NEXT STEPS

Focus on completing MVP - this requires finishing P0 and P1 tasks to enable Layer 7.

**Order of Implementation (Critical Path):**

1. **P0.1 (6h): Fix Nested Operations Support**
   - AST builder creates IntermediateStatements for nested ops
   - Compiler processes intermediate nodes correctly
   - Impact: Enables ANY complex expression

2. **P0.2 (4h): Refactor Fraction Operations to Use Overloading**
   - Registry: Support multiple signatures for same operation name
   - Runtime: Implement dispatch based on input types
   - Remove separate ADD_FRACTION operations
   - Add integration tests (full compiler → runtime pipeline)
   - Impact: Establishes overloading pattern for Integer/Decimal too

3. **P0.3 (5h): Fix Temporal Operators Demand-Driven Semantics**
   - FBY: Use recursive evaluate() instead of manual iteration
   - ACCUMULATE: Recursive evaluation across timesteps
   - NEXT: Evaluate stream at requested time
   - Impact: Preserves core demand-driven semantics

4. **P0.4 (8h): Implement IncrementalRuntime Class**
   - Partial graph evaluation with demand-driven semantics
   - Node state tracking (completed/pending/error)
   - Subscription management for multiple clients
   - Graph update API with cache invalidation
   - Spanish child-friendly messages for missing inputs
   - Impact: Enables WebSocket server for IDE live feedback

5. **P1.1 (2h): Implement Integer Operations**
   - ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE for Integer (via overloading)
   - Impact: Completes Integer type system

6. **P1.2 (2h): Implement Decimal Operations**
   - ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE for Decimal (via overloading)
   - Impact: Completes Decimal type system

7. **P1.3 (3h): Implement Curriculum Type Filtering Operations**
   - FILTER_BY_COLOR for Car, Food, Animal
   - FILTER_BY_TASTE for Food
   - FILTER_BY_TYPE for Animal
   - Impact: Completes curriculum type operations

8. **P1.4 (5h): Make Set/Stream Operations Generic**
   - Remove natural-only restriction
   - Support any element type (Shape, Car, Food, Animal, Person)
   - Fix SetType to use proper generic `T[]` instead of `unknown[]`
   - Impact: Aligns with curriculum (sets of shapes, colors, etc.)

9. **P1.5 (3h): Fix Set/Object Literal Ambiguity**
   - Improve parser GATE logic
   - Clear disambiguation between set and object literals
   - Better error messages
   - Impact: Improves parser robustness

10. **P1.6 (3h): Implement Missing Compiler Validation**
    - Set homogeneity validation
    - Output node requirement
    - Literal type validation
    - Stream source validation
    - Impact: Better error messages for young learners

11. **P1.7 (11h): Refactor HTTP API to Follow Elysia Best Practices**
    - Add Elysia.t validation for compile-time and runtime type safety
    - Implement error handling middleware
    - Use plugin pattern for route organization
    - Add Eden Treaty for end-to-end type safety
    - Return 422 for validation errors
    - Impact: Improves code quality and maintainability

**P2-P3 Tasks (Post-MVP):**
- P2.1-P2.5: Type fixes (SetType, Color), Spanish messages, SORT improvements, DataType recursion
- P3.1-P3.3: WebSocket server, execution trace, parallelism

---

## MVP DEFINITION

### MVP Requirements (from PROJECT_GOALS.md)

To achieve MVP, complete the following:

**Core Language (Layers 1-6):**
- ✅ Natural numbers, arithmetic operations (ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE)
- ⚠️ Fraction operations - **INCORRECTLY IMPLEMENTED** (P0.2 needs refactor - compiler can't generate tokens)
- ⚠️ Integer operations - **MISSING** (P1.1)
- ⚠️ Decimal operations - **MISSING** (P1.2)
- ✅ Curriculum types (Shape, Car, Food, Animal, Person)
- ⚠️ Curriculum type filters - **PARTIAL** (P1.3 needed - missing Car/Food/Animal FILTER ops)
- ✅ Set operations (UNION, INTERSECTION, DIFFERENCE, COMPLEMENT)
- ⚠️ Temporal operators - **INCORRECTLY IMPLEMENTED** (P0.3 needs refactor - breaks demand-driven)
- ✅ Filter operations (FILTER, FILTER_BY_SIZE, FILTER_BY_COLOR for Shape)
- ⚠️ Set/Stream operations generic - **MISSING** (P1.4)
- ⚠️ Nested operations - **NOT IMPLEMENTED** (P0.1 - CRITICAL BLOCKER)

**Compiler:**
- ✅ Validates programs, catches errors at compile-time
- ⚠️ Set homogeneity validation - **MISSING** (P1.6)
- ⚠️ Output node requirement - **MISSING** (P1.6)
- ⚠️ Literal type validation - **MISSING** (P1.6)
- ⚠️ Stream source validation - **MISSING** (P1.6)
- ✅ Type compatibility validation working
- ⚠️ Set/Object literal ambiguity - **ISSUE** (P1.5)
- ⚠️ Nested operations - **NOT IMPLEMENTED** (P0.1 - CRITICAL BLOCKER)

**Runtime:**
- ✅ Executes programs correctly (simple cases)
- ⚠️ Temporal operators - **INCORRECTLY IMPLEMENTED** (P0.3 needs refactor - breaks demand-driven)
- ⚠️ IncrementalRuntime - **MISSING** (P0.4)
- ✅ 32/31 operations working (103%) - but Fraction ops INCORRECTLY implemented
- ✅ Handles 5 concurrent users without degradation

**Integration (Layer 7):**
- ✅ HTTP API for batch mode (12/12 tests passing)
- ⚠️ HTTP API Elysia best practices - **MISSING** (P1.7)
- ⚠️ WebSocket Server - **MISSING** (P3.1)

**MVP Completion Tasks (P0 + P1):**
1. **P0.1:** Fix nested operations (6h) - **CRITICAL BLOCKER**
2. **P0.2:** Refactor Fraction operations to use overloading (4h) - **CRITICAL BLOCKER**
3. **P0.3:** Fix temporal operators demand-driven semantics (5h) - **CRITICAL BLOCKER**
4. **P0.4:** Implement IncrementalRuntime (8h) - **CRITICAL BLOCKER**
5. **P1.1:** Implement Integer operations (2h)
6. **P1.2:** Implement Decimal operations (2h)
7. **P1.3:** Implement curriculum type filters (3h)
8. **P1.4:** Make Set/Stream operations generic (5h)
9. **P1.5:** Fix Set/Object literal ambiguity (3h)
10. **P1.6:** Implement missing compiler validation (3h)
11. **P1.7:** Refactor HTTP API (11h) - **OPTIONAL for MVP**

**Total MVP Time:** 41 hours (excluding P1.7)

**Version 1.0 (All Layers):**
- All P0, P1, and P2 tasks completed
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%)
- Child-friendly Spanish messages throughout
- Generic set/stream operations
- HTTP API follows Elysia best practices

---

## PERFORMANCE TARGETS

### Current Status
- Compilation: ~50ms for <50 nodes (on par with target)
- Execution: ~2ms for simple programs (exceeds target)
- Test suite: 107/107 tests passing (100%) ✓

### Targets
- Compilation: <100ms for programs with <100 nodes (p95)
- Execution: <50ms for programs with <100 nodes (p95)
- Concurrent: 5 simultaneous requests without degradation
- Incremental update: 5x faster than full re-evaluation
- WebSocket roundtrip: <50ms (p95)

---

## SUCCESS METRICS

### MVP (Layers 1-6 + Basic Integration)
- ✅ All 32/31 operations implemented (103%)
- ⚠️ But Fraction ops INCORRECTLY implemented (P0.2)
- ⚠️ But Temporal ops INCORRECTLY implemented (P0.3)
- ✅ Type compatibility validation working
- ✅ HTTP API functional (12/12 tests passing)
- ✅ Compiler validates programs correctly
- ✅ Runtime executes simple programs correctly
- ⚠️ But nested operations NOT IMPLEMENTED (P0.1 - CRITICAL)
- ✅ TypeScript compilation passes (no errors)
- ✅ 107/107 tests passing (100%)
- ✅ Handles 5 concurrent users
- ⚠️ Integer operations - **MISSING** (P1.1)
- ⚠️ Decimal operations - **MISSING** (P1.2)
- ⚠️ Curriculum type filters - **PARTIAL** (P1.3)
- ⚠️ Set/Stream operations generic - **MISSING** (P1.4)
- ⚠️ IncrementalRuntime - **MISSING** (P0.4)
- ⚠️ Complete compiler validation - **MISSING** (P1.6)

### Version 1.0 (All Layers)
- All P0 and P1 tasks completed
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%)
- Child-friendly Spanish messages throughout
- Generic set/stream operations
- HTTP API follows Elysia best practices
- Temporal operators preserve demand-driven semantics
- Nested operations supported

---

## COMPREHENSIVE STUDY FINDINGS (2026-03-14)

### Study Scope
This implementation plan was updated based on:
1. **Previous analysis (2026-03-13):** 7 parallel subagents covering:
   - Type Definitions Analysis (67% correct)
   - Operations Registry Analysis (~60% coverage, critical pattern issues)
   - Compiler Package Analysis (70% complete - NESTED OPS NOT IMPLEMENTED)
   - Runtime Package Analysis (65% complete - temporal ops break demand-driven)
   - HTTP API Package Analysis (64% complete - not best practice)
   - Test Coverage Analysis (107 tests, 100% passing, but gaps)
   - WebSocket Server Package Analysis (0% complete - directory empty)

2. **New verification (2026-03-14):** Direct code analysis to verify:
   - Fraction type definition against spec (CORRECT per spec)
   - Fraction operations implementation pattern (INCORRECT - needs overloading)
   - Temporal operators demand-driven compliance (INCORRECT - manual iteration)
   - Nested operations support (NOT IMPLEMENTED)
   - IncrementalRuntime status (MISSING)

### Key Updates from 2026-03-14 Verification

**Fraction Type:** ✅ CORRECT - No changes needed
- Verified against `specs/LANGUAGE_SPEC.md` lines 94-99
- Spec explicitly uses `number` for numerator/denominator (not `Integer`)
- Current implementation matches spec exactly
- **Action:** Removed P2.1 task from plan

**Fraction Operations:** ❌ INCORRECT PATTERN - Needs overloading
- Current: Separate `ADD_FRACTION`, `SUBTRACT_FRACTION`, etc. operations
- Issue: Compiler generates `ADD`, not `ADD_FRACTION`
- Required: Single `ADD` operation with multiple type signatures
- **Action:** P0.2 task remains valid and critical

**All other findings from 2026-03-13 analysis remain VALID.**

### Key Findings

#### 1. Implementation Status Accuracy ✅ VERIFIED
The comprehensive study revealed the accurate status:
- **Overall Progress:** ~58% (not 79% as previously estimated)
- **Layers 1-6:** 50-100% complete (varies by layer)
- **Layer 7 (Integration):** 0% complete (IncrementalRuntime and WebSocket Server missing)
- **Test Coverage:** 107/107 tests passing (100%) - but gaps in integration tests
- **TypeScript Compilation:** Passes with zero errors

#### 2. Core Functionality ❌ CRITICAL ISSUES
**Compiler Package (70% complete):**
- ✅ Lexer: Complete - all 104 tokens defined
- ✅ Parser: Good coverage (19 grammar rules)
- ✅ Validation: 5/8 semantic constraints (62.5%)
- ✅ Compiler Pipeline: Working end-to-end for simple programs
- ❌ **NESTED OPERATIONS NOT HANDLED** (CRITICAL BLOCKER)
- ❌ Set/Object Literal Ambiguity (parser fragility)
- ❌ Literal ID Mismatch

**Runtime Package (65% complete):**
- ✅ Demand-Driven Evaluator: Core structure CORRECT
- ✅ Cache: Correct two-level memoization
- ✅ 36/36 Operations defined in registry (but 4 fraction ops wrong pattern)
- ❌ **Fraction operations INCORRECTLY IMPLEMENTED** (separate ops, not overloading)
- ❌ **Temporal operators INCORRECTLY IMPLEMENTED** (break demand-driven)
- ❌ **IncrementalRuntime DOES NOT EXIST** (P0 BLOCKER)
- ❌ No Integer operations (0/6)
- ❌ No Decimal operations (0/6)
- ❌ No parallelism

**Missing Operations:**
- ❌ Fraction operations: INCORRECTLY IMPLEMENTED (P0.2 - needs refactor)
- ❌ Integer operations: 0% complete (P1.1)
- ❌ Decimal operations: 0% complete (P1.2)
- ❌ Curriculum type filters: PARTIAL (P1.3 - missing Car/Food/Animal FILTER ops)
- ❌ Set/Stream operations: Not generic (P1.4)

#### 3. Critical Gaps ❌ IDENTIFIED
**P0 (CRITICAL):**
1. ❌ Nested operations NOT IMPLEMENTED (P0.1 - blocks complex expressions)
2. ❌ Fraction operations INCORRECTLY IMPLEMENTED (P0.2 - separate ops, not overloading)
3. ❌ Temporal operators INCORRECTLY IMPLEMENTED (P0.3 - break demand-driven semantics)
4. ❌ IncrementalRuntime DOES NOT EXIST (P0.4 - blocks WebSocket Server)

**P1 (HIGH):**
1. Integer operations: 0% complete (P1.1)
2. Decimal operations: 0% complete (P1.2)
3. Curriculum type filters: PARTIAL (P1.3 - missing Car/Food/Animal)
4. Set/Stream operations: Not generic (P1.4)
5. Set/Object literal ambiguity (P1.5)
6. Compiler validation: Missing 3 rules (P1.6)
7. HTTP API: Doesn't follow Elysia best practices (P1.7)

**P2 (MEDIUM):**
1. ~~Fraction type: Uses `number` instead of `Integer`~~ **RESOLVED** - Fraction type is correct per spec (uses `number`)
2. SetType: Uses `unknown[]` instead of `T[]` (P2.1)
3. Color type: Extra values (white, black) (P2.2)
4. Spanish messages: Partial implementation (P2.3)
5. SORT: Only works with numbers (P2.4)
6. DataType recursion: Uses `string | DataType` (P2.5)

#### 4. Test Coverage ✅ EXCELLENT BUT GAPS
**107 tests, 100% passing:**
- Integration tests: 40 tests (~81% of spec requirements)
- Runtime unit tests: 36 tests
- HTTP API tests: 12 tests
- Compiler tests: 13 tests
- Shared type tests: 14 tests

**Critical Gaps:**
- Incremental Runtime tests: 0/13 (0%)
- WebSocket Server tests: 0/12 (0%)
- Fraction operations: Unit tests only (no integration - bypass compiler)
- Set operations: Tests with `natural` instead of curriculum types
- Temporal operations: Missing full timestep coverage
- Nested operations: No tests (not implemented)

**No Stubbed Tests Found** ✅

#### 5. Design Decisions ❌ PARTIALLY INCORRECT
**Demand-Driven Semantics:**
- ✅ Correct per DEMAND_DRIVEN_INCREMENTAL.md
- ✅ No eager evaluation (for simple cases)
- ✅ Cache-based memoization
- ❌ **BUT temporal operators break demand-driven** (P0.3 needs refactor)

**Fraction Operations Approach:**
- ✅ Should use overloading (ADD works with Natural, Integer, Decimal, Fraction)
- ✅ NOT separate ADD_FRACTION operations
- ✅ Spec confirms this approach
- ❌ **BUT current implementation is WRONG** - uses separate ADD_FRACTION operations
- ❌ Compiler cannot generate ADD_FRACTION tokens
- ❌ Needs refactor to implement overloading pattern correctly

**Nested Operations:**
- ✅ Required by spec for complex expressions
- ❌ **NOT IMPLEMENTED** - AST builder generates IDs but no TransformStatements
- ❌ Blocks ANY meaningful program beyond simple operations
- ❌ P0.1 - CRITICAL BLOCKER

#### 6. Completed Tasks ✅ CONFIRMED
The following components are mostly working:
- ✅ Lexer: Complete (104 tokens)
- ✅ Parser: Good coverage (19 rules)
- ✅ Simple operations work via compiler
- ✅ Cache implementation correct
- ✅ HTTP API functional (12/12 tests)

---

## DOCUMENTATION

### Spec Files
- `specs/LANGUAGE_SPEC.md` - Complete language specification
- `specs/GRAMMAR_SPEC.md` - Grammar and syntax
- `specs/INTEGRATION_SPEC.md` - HTTP API, WebSocket, IncrementalRuntime
- `specs/INTEGRATION_TESTS_SPEC.md` - Test requirements
- `specs/ELYSIA_LLMS.md` - Elysia best practices
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` - Demand-driven evaluation model

### Key Design Documents
- `PROJECT_GOALS.md` - Project objectives and MVP definition
- `AGENTS.md` - Development guidelines and standards

---

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Next Review:** After completing P0 tasks (critical blockers)
