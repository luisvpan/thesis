# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-13 (Updated with accurate status from comprehensive codebase analysis)

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

**Last Updated:** 2026-03-13 (Comprehensive analysis completed)

### Overall Progress: ~79% Complete (Core compiler/runtime ~95%, Layer 7 at 0%)

**Current Status:**
- **Layer 7** at 0% implemented (IncrementalRuntime and WebSocket Server completely missing)
- **TypeScript compilation passes** ✅ (P0.3 completed - no errors)
- **Nested operation expressions implemented** ✅ (P1.4 completed - tests passing)
- **Performance tests implemented** ✅ (P0.2 completed - 19/19 tests passing)
- **HTTP API functional** ✅ (P1.1 completed - 12/12 tests passing)
- **Temporal operations fixed** ✅ (P0.1 completed - generator state preserved)
- **Fraction operations implemented** ✅ (P0.2 completed - 11/11 tests passing)

### Test Status Summary
- **Total Tests:** 107 tests, 100% passing
- **Stubbed:** 0 tests (all implemented)
- **Categories Covered:**
  - End-to-End: 9 tests (450% of spec requirements)
  - Type Validation: 12 tests (600% of spec requirements)
  - Curriculum Types: 4 tests (200% of spec requirements)
  - Set Operations: 4 tests (200% of spec requirements)
  - Temporal: 4 tests (200% of spec requirements)
  - Error Handling: 2 tests (100% of spec requirements)
  - Performance: 19 tests (950% of spec requirements)
  - Demand-Driven: 2 tests (100% of spec requirements)
  - HTTP API: 12 tests (100% passing)
  - Fraction Operations: 11 tests (1100% of spec requirements)

### Component Status Summary

| Component | Status | Completion | Notes |
|-----------|--------|------------|-------|
| **Shared Package** | Mostly Complete | 85% | All types defined. Issues: Fraction/Integer/Decimal ops missing, Set/Stream not generic, SetType uses `unknown[]`, Fraction uses `number` instead of `Integer` |
| **Compiler Package** | Mostly Complete | 90% | Lexer/Parser/AST/Validation/Compiler all implemented. Nested ops: 100% complete. Missing: 2 validation rules (set homogeneity, output node req) |
| **Runtime Package** | Mostly Complete | 85% | Core evaluator CORRECT, 32/31 ops implemented (103% - Fraction ops added!). Missing: IncrementalRuntime (0%), Integer/Decimal ops, generic set/stream ops |
| **HTTP API Package** | Functional | 60% | Server, routes, all 12 endpoints implemented and passing. Doesn't follow Elysia best practices: no Elysia.t validation, error middleware, plugin pattern, Eden Treaty |
| **WebSocket Server Package** | Not Started | 0% | Empty directory, no implementation (blocked by IncrementalRuntime) |

### Layer Progress

| Layer | Status | Completion | Tests Passing | Issues |
|-------|--------|------------|---------------|--------|
| Layer 1: Foundation | COMPLETE | 100% | 100% | None |
| Layer 2: Arithmetic | MOSTLY COMPLETE | 95% | 100% | Missing Integer/Decimal operations (Natural and Fraction complete) |
| Layer 3: Curriculum Types | COMPLETE | 100% | 100% | None |
| Layer 4: Set Operations | PARTIAL | 85% | 100% | SORT only for numbers, ALPHABETICAL_SORT converts to string, set/stream ops not generic |
| Layer 5: Temporal Operators | COMPLETE | 100% | 100% | None (P0.1 fixed) |
| Layer 6: Streams | COMPLETE | 100% | 100% | None |
| Layer 7: Integration | NOT STARTED | 0% | 0% | IncrementalRuntime (0%), WebSocket Server (0%), HTTP API needs refactoring |

---

## CRITICAL BLOCKERS

### Blocker 1: IncrementalRuntime Not Implemented (P0 - CRITICAL)

**Status:** NOT STARTED - File doesn't exist

**Impact:**
- **Blocks WebSocket Server** - can't implement P3.1 without this
- Cannot provide IDE live feedback
- Cannot support incremental evaluation for partial programs
- **Blocks completion of Layer 7 (Integration)**

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

### Blocker 2: Fraction Operations Implemented Incorrectly (P0 - CRITICAL) - ❌ NEEDS REFACTOR

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

### Blocker 3: Integer Operations Missing (P1 - HIGH)

**Status:** NOT STARTED - Type defined but no operations

**Impact:**
- Incomplete numeric type system
- SUBTRACT on Natural produces Integer but Integer has no operations
- Type system not fully functional

**Operations Needed (via overloading):**
- ADD(Integer, Integer) → Integer
- SUBTRACT(Integer, Integer) → Integer
- MULTIPLY(Integer, Integer) → Integer
- DIVIDE(Integer, Integer) → Decimal
- COMPARE(Integer, Integer) → Boolean

**Priority:** P1 - HIGH (type system completeness)

**Estimated Time:** 2 hours

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 57-72

---

### Blocker 4: Decimal Operations Missing (P1 - HIGH)

**Status:** NOT STARTED - Type defined but no operations

**Impact:**
- Incomplete numeric type system
- DIVIDE on Natural/Integer produces Decimal but Decimal has no operations
- Type system not fully functional

**Operations Needed (via overloading):**
- ADD(Decimal, Decimal) → Decimal
- SUBTRACT(Decimal, Decimal) → Decimal
- MULTIPLY(Decimal, Decimal) → Decimal
- DIVIDE(Decimal, Decimal) → Decimal
- COMPARE(Decimal, Decimal) → Boolean

**Priority:** P1 - HIGH (type system completeness)

**Estimated Time:** 2 hours

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 74-91

---

## PRIORITIZED IMPLEMENTATION PLAN

### Task Prioritization Framework

**P0 - CRITICAL (Blocks core functionality, breaks semantics, or prevents compilation)**
- IncrementalRuntime (0% - blocks WebSocket Server)
- Fraction operations (type defined but no implementation - breaks type system)
- Fix Fraction type (number → Integer)

**P1 - HIGH (Required for MVP completion or major quality improvements)**
- Integer operations (type defined but no implementation)
- Decimal operations (type defined but no implementation)
- Set/Stream operations not generic (hardcoded to natural only)
- Missing compiler validation (set homogeneity, output node requirement)
- Refactor HTTP API to follow Elysia best practices

**P2 - MEDIUM (Quality improvements, completeness gaps, usability enhancements)**
- Complete Spanish messages (partial implementation)
- Improve SORT operations (SORT only numbers, ALPHABETICAL_SORT converts to string)
- Fix SetType (unknown[] → T[])
- More built-in generators (only "counter" exists)

**P3 - LOW (Nice to have, post-MVP)**
- WebSocket Server (depends on IncrementalRuntime)
- Execution trace completion (executionOrder and nodeEvaluations empty)

---

## PRIORITIZED TASK LIST

### P0 CRITICAL (Immediate - Blocks Core Functionality)

#### Task P0.1: Fix Fraction Type Definition

**File to update:** `packages/shared/src/types/primitives.ts`

**Why Critical:**
- **Breaks type system correctness** - Fraction uses `number` instead of `Integer`
- Spec says numerator/denominator should be Integer type
- Must fix before implementing Fraction operations

**Current Code:**
```typescript
export type Fraction = {
  kind: "fraction";
  numerator: number;    // WRONG - should be Integer
  denominator: number;  // WRONG - should be Integer
};
```

**Should Be:**
```typescript
export type Fraction = {
  kind: "fraction";
  numerator: Integer;
  denominator: Integer;
};
```

**Estimated Time:** 0.5 hours

**Dependencies:** None

**Acceptance Criteria:**
- Fraction type uses Integer for numerator/denominator
- Typecheck passes

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 94-99

**Ralph Wiggum Checklist:**
- [ ] Fraction type updated to use Integer
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(types): use Integer type for Fraction numerator/denominator"

---

#### Task P0.2: Refactor Fraction Operations to Use Overloading (CRITICAL - INCORRECTLY IMPLEMENTED)

**Status:** ❌ INCORRECTLY IMPLEMENTED - Needs refactoring

**Files to update:**
- `packages/shared/src/operations/registry.ts` (add overloading support, remove ADD_FRACTION etc.)
- `packages/runtime/src/operations/numeric.ts` (implement overloading pattern)
- `packages/runtime/src/operations/index.ts` (export overloading dispatcher)
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
- [ ] Integration Test: Mixed natural/fraction expressions (e.g., "3" + "1/2" - should work when Integer type exists)
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
- [ ] Integration Test: Complex expression with fractions and naturals

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

#### Task P0.3: Implement IncrementalRuntime Class

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

**Dependencies:** None

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
```

**Estimated Time:** 2 hours

**Dependencies:** None

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
```

**Estimated Time:** 2 hours

**Dependencies:** None

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

#### Task P1.3: Make Set/Stream Operations Generic

**Files to update:**
- `packages/shared/src/types/composite.ts` (SetType uses `unknown[]`, change to generic `T[]`)
- `packages/runtime/src/evaluator/demand-driven-evaluator.ts` (update set wrapping logic)
- `packages/runtime/src/operations/sets.ts` (currently only natural)
- `packages/shared/src/operations/registry.ts` (update type signatures to support any type)
- `packages/runtime/src/operations/` (stream operations)

**Why Important:**
- Current set/stream operations only work with natural numbers
- Generic types would support any element type (Shape, Car, Food, Animal, Person)
- Aligns with curriculum (sets of shapes, colors, etc.)
- Fix SetType to use proper generic type `T[]` instead of `unknown[]`

**Estimated Time:** 5 hours

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 312-327):**
- ✓ Set<T> where T can be any DataType
- ✓ Stream<T> where T can be any DataType
- ✓ Set operations work with any element type (not just natural)
- ✓ Stream operations support any element type
- ✓ Inherited operations work for element types

**Required Tests:**
- [ ] Test: Set<shape> - UNION of shape sets works
- [ ] Test: Set<car> - FILTER_BY_COLOR on car set works
- [ ] Test: Set<food> - INTERSECTION of food sets works
- [ ] Test: Set<animal> - DIFFERENCE of animal sets works
- [ ] Test: Stream<shape> - temporal operations on shape streams work
- [ ] Test: Generic types properly enforced in type checker
- [ ] Test: Set operations maintain element type through transformations

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

#### Task P1.4: Implement Missing Compiler Validation

**File to update:** `packages/compiler/src/validation/dag-validator.ts`

**Why Important:**
- Improves quality of error messages for young learners
- Catches more errors at compile time (better UX)
- Programs without output nodes should fail fast
- Set homogeneity prevents runtime type errors

**Missing Validation:**
1. Set homogeneity validation (all elements in set must be same type)
2. Output node requirement (programs must have at least one output node)

**Estimated Time:** 3 hours

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 431-440):**
- ✓ Set operations verify all elements are same type
- ✓ Programs without output nodes produce error
- ✓ Type safety rules enforced at compile time
- ✓ Child-friendly Spanish error messages

**Required Tests:**
- [ ] Test: Set homogeneity validation - mixed types in set error
- [ ] Test: Set homogeneity validation - same types in set succeed
- [ ] Test: Output node requirement - program with no outputs fails
- [ ] Test: Output node requirement - program with outputs succeeds
- [ ] Test: Validation errors include child-friendly Spanish messages

**Layer:** Cross-layer (validation quality)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` validation section

**Ralph Wiggum Checklist:**
- [ ] Missing validation rules implemented (set homogeneity, output node)
- [ ] Validation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): add missing compiler validation rules"

---

#### Task P1.5: Refactor HTTP API to Follow Elysia Best Practices

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
- **Improves code quality and maintainability**

**Estimated Time:** 11 hours

**Dependencies:** None (can start immediately)

**Acceptance Criteria (from specs/ELYSIA_LLMS.md):**
- ✅ Use Elysia.t for request/response schema validation
- ✅ Implement error handling middleware (Elysia.onError)
- ✅ Use plugin pattern for route organization
- ✅ Add Eden Treaty for end-to-end type safety
- ✅ Remove manual type assertions, use schema validation
- ✅ Follow Elysia best practices

**Required Refactoring:**
- [ ] Replace manual type assertions with Elysia.t schema validation
- [ ] Add global error handling middleware (Elysia.onError)
- [ ] Organize routes using plugin pattern
- [ ] Add Eden Treaty export for client type safety
- [ ] Use Elysia plugins for CORS, logging, etc.
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
- [ ] All 12/12 HTTP API tests still pass
- [ ] Typecheck passes
- [ ] Git commit with message: "refactor(http-api): implement Elysia best practices"

---

### P2 MEDIUM (Quality Improvements & Completeness)

#### Task P2.1: Complete Child-Friendly Spanish Error Messages

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

**Dependencies:** P1.4 (Missing Compiler Validation)

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

#### Task P2.2: Improve SORT/ALPHABETICAL_SORT Operations

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

**Dependencies:** P1.3 (Make Set/Stream Operations Generic)

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
- P0.3 (IncrementalRuntime) - REQUIRED

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

## SUMMARY TABLE

| Task | Priority | Status | Time | Impact | Dependencies |
|------|----------|--------|------|--------|--------------|
| **P0.1: Fix Fraction type (number → Integer)** | P0 | NOT STARTED | 0.5h | **CRITICAL** - type correctness | None |
| **P0.2: Refactor Fraction operations (overloading)** | P0 | ❌ INCORRECTLY IMPLEMENTED | 4h | **CRITICAL** - compiler can't generate tokens | None |
| **P0.3: Implement IncrementalRuntime** | P0 | NOT STARTED | 8h | **CRITICAL** - blocks Layer 7 | None |
| **P1.1: Implement Integer operations** | P1 | NOT STARTED | 2h | **HIGH** - type system completeness | None |
| **P1.2: Implement Decimal operations** | P1 | NOT STARTED | 2h | **HIGH** - type system completeness | None |
| **P1.3: Make Set/Stream operations generic** | P1 | NOT STARTED | 5h | **HIGH** - removes natural restriction | None |
| **P1.4: Implement missing compiler validation** | P1 | NOT STARTED | 3h | **HIGH** - quality improvement | None |
| **P1.5: Refactor HTTP API (Elysia best practices)** | P1 | NOT STARTED | 11h | Quality & maintainability | None |
| **P2.1: Complete Spanish error messages** | P2 | NOT STARTED | 3h | Better UX for ages 6-9 | P1.4 |
| **P2.2: Improve SORT/ALPHABETICAL_SORT operations** | P2 | NOT STARTED | 2h | Improves functionality | P1.3 |
| **P3.1: Implement WebSocket Server** | P3 | NOT STARTED | 12h | IDE live feedback | P0.3 |
| **P3.2: Complete execution trace** | P3 | NOT STARTED | 2h | Debugging support | None |

**Total Estimated Time:** 54.5 hours (0 hours completed - P0.2 incorrectly implemented)

**Previous Work Completed:**
- ✓ Layers 1-6: ~85-90% complete (Foundation, Arithmetic, Curriculum Types, Sets, Temporal, Streams)
- ❌ 32/31 operations marked complete but Fraction ops INCORRECTLY implemented (need refactor)
- ✓ 1 stream generator (counter)
- ✓ All types defined (primitives, curriculum, composite, validation, program)
- ✓ Lexer: COMPLETE - all tokens defined
- ✓ Parser: COMPLETE - all grammar rules implemented
- ✓ AST Builder: COMPLETE - full CST to AST conversion
- ✓ Validation: MOSTLY COMPLETE (missing: set homogeneity, output node req)
- ✓ Compiler: COMPLETE - full pipeline working
- ✓ Main Runtime: COMPLETE - simple wrapper
- ✓ Demand-Driven Evaluator: CORRECT - all temporal ops fixed (P0.1 completed)
- ✓ Graph Implementation: CORRECT but has limitations (SORT only numbers, ALPHABETICAL_SORT converts to string)
- ✓ HTTP API Server: COMPLETE - all 12 endpoints implemented and passing
- ✓ 107 tests passing (100%) - 19 performance + 12 HTTP API + 11 Fraction ops (but ops don't work via compiler)
- ❌ P0.2: Fraction operations INCORRECTLY implemented - needs refactor (separate ops, not overloading)

---

## NEXT STEPS

Focus on completing MVP - this requires finishing type system completeness and Layer 7 integration.

**Order of Implementation:**

1. **P0.1 (0.5h): Fix Fraction Type Definition**
   - Change numerator/denominator from `number` to `Integer`
   - Impact: Enables proper Fraction operations

2. **P0.2 (4h): Refactor Fraction Operations to Use Overloading (CRITICAL - INCORRECTLY IMPLEMENTED)**
   - Registry: Support multiple signatures for same operation name
   - Runtime: Implement dispatch based on input types
   - Remove separate ADD_FRACTION operations
   - Add integration tests (full compiler → runtime pipeline)
   - Impact: Enables fraction operations via compiler (not just direct runtime calls)

3. **P0.3 (8h): Implement IncrementalRuntime Class**
   - Partial graph evaluation with demand-driven semantics
   - Node state tracking (completed/pending/error)
   - Subscription management for multiple clients
   - Graph update API with cache invalidation
   - Spanish child-friendly messages for missing inputs
   - Impact: Enables WebSocket server for IDE live feedback

4. **P1.1 (2h): Implement Integer Operations**
   - ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE for Integer (via overloading)
   - Impact: Completes Integer type system

5. **P1.2 (2h): Implement Decimal Operations**
   - ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE for Decimal (via overloading)
   - Impact: Completes Decimal type system

6. **P1.3 (5h): Make Set/Stream Operations Generic**
   - Remove natural-only restriction
   - Support any element type (Shape, Car, Food, Animal, Person)
   - Fix SetType to use proper generic `T[]` instead of `unknown[]`
   - Impact: Aligns with curriculum (sets of shapes, colors, etc.)

7. **P1.4 (3h): Implement Missing Compiler Validation**
   - Set homogeneity validation
   - Output node requirement
   - Impact: Better error messages for young learners

8. **P1.5 (11h): Refactor HTTP API to Follow Elysia Best Practices**
   - Add Elysia.t validation for compile-time and runtime type safety
   - Implement error handling middleware
   - Use plugin pattern for route organization
   - Add Eden Treaty for end-to-end type safety
   - Impact: Improves code quality and maintainability

9. **P2.1 (3h): Complete Child-Friendly Spanish Error Messages**
   - Update all validation and runtime error handling
   - Ensure messages are age-appropriate for 6-9 year olds
   - Impact: Better UX for target demographic

10. **P2.2 (2h): Improve SORT/ALPHABETICAL_SORT Operations**
    - Make SORT generic: SORT<T>(set: Set<T>) where T is Comparable
    - Make ALPHABETICAL_SORT type-safe (preserve type, don't convert to string)
    - Impact: Removes sorting limitations

11. **P3.1 (12h): Implement WebSocket Server (Live Mode)**
    - Connection at ws://localhost:3000/live
    - Message handlers for validate_program, evaluate_incremental, subscribe_node, unsubscribe_node
    - Push notifications for node state changes
    - Multiple concurrent client support
    - Impact: Enables IDE live feedback during program construction

12. **P3.2 (2h): Complete Execution Trace Implementation**
    - Populate executionOrder array
    - Populate nodeEvaluations map
    - Impact: Better debugging support

---

## MVP DEFINITION

### MVP Requirements (from PROJECT_GOALS.md)

To achieve MVP, complete the following:

**Core Language (Layers 1-6):**
- ✅ Natural numbers, arithmetic operations (ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE)
- ❌ Fraction operations - **INCORRECTLY IMPLEMENTED** (P0.2 needs refactor - compiler can't generate tokens)
- ⚠️ Integer operations - **MISSING** (P1.1)
- ⚠️ Decimal operations - **MISSING** (P1.2)
- ✅ Curriculum types (Shape, Car, Food, Animal, Person)
- ✅ Set operations (UNION, INTERSECTION, DIFFERENCE, COMPLEMENT)
- ✅ Temporal operators (NEXT, FIRST, FBY, ACCUMULATE)
- ✅ Filter operations (FILTER, FILTER_BY_SIZE, FILTER_BY_COLOR, etc.)
- ⚠️ Set/Stream operations generic - **MISSING** (P1.3)

**Compiler:**
- ✅ Validates programs, catches errors at compile-time
- ⚠️ Set homogeneity validation - **MISSING** (P1.4)
- ⚠️ Output node requirement - **MISSING** (P1.4)
- ✅ Type compatibility validation working

**Runtime:**
- ✅ Executes programs correctly with demand-driven semantics
- ⚠️ IncrementalRuntime - **MISSING** (P0.3)
- ✅ 32/31 operations working (103%) - Fraction ops complete!
- ✅ Handles 5 concurrent users without degradation

**Integration (Layer 7):**
- ✅ HTTP API for batch mode (12/12 tests passing)
- ⚠️ HTTP API Elysia best practices - **MISSING** (P1.5)
- ⚠️ WebSocket Server - **MISSING** (P3.1)

**MVP Completion Tasks:**
1. P0.1: Fix Fraction type (0.5h)
2. P0.2: Refactor Fraction operations to use overloading (4h) - **INCORRECTLY IMPLEMENTED, needs refactor**
3. P0.3: Implement IncrementalRuntime (8h)
4. P1.1: Implement Integer operations (2h)
5. P1.2: Implement Decimal operations (2h)
6. P1.3: Make Set/Stream operations generic (5h)
7. P1.4: Implement missing compiler validation (3h)
8. P1.5: Refactor HTTP API (11h) - **OPTIONAL for MVP**

**Total MVP Time:** 25.5 hours (excluding P1.5)

**Version 1.0 (All Layers):**
- All P0, P1, and P2 tasks completed
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%)
- Child-friendly Spanish messages throughout

---

## PERFORMANCE TARGETS

### Current Status
- Compilation: ~50ms for <50 nodes (on par with target)
- Execution: ~2ms for simple programs (exceeds target)
- Test suite: 107/107 tests passing (100%) ✓
  - 19/19 performance tests implemented
  - 12/12 HTTP API tests implemented
  - 11/11 Fraction operation tests implemented (P0.2 complete)

### Targets
- Compilation: <100ms for programs with <100 nodes (p95)
- Execution: <50ms for programs with <100 nodes (p95)
- Concurrent: 5 simultaneous requests without degradation
- Incremental update: 5x faster than full re-evaluation
- WebSocket roundtrip: <50ms (p95)

---

## SUCCESS METRICS

### MVP (Layers 1-6 + Basic Integration)
- ✅ All 32/31 operations implemented (103%) - Fraction ops complete!
- ✅ Type compatibility validation working
- ✅ HTTP API functional (12/12 tests passing)
- ✅ Compiler validates programs correctly
- ✅ Runtime executes programs correctly with demand-driven semantics
- ✅ TypeScript compilation passes (no errors)
- ✅ 107/107 tests passing (100%)
- ✅ Handles 5 concurrent users
- ✅ Fraction operations - **COMPLETE** (P0.2 completed 2026-03-13)
- ⚠️ Integer operations - **MISSING** (P1.1)
- ⚠️ Decimal operations - **MISSING** (P1.2)
- ⚠️ Set/Stream operations generic - **MISSING** (P1.3)
- ⚠️ IncrementalRuntime - **MISSING** (P0.3)
- ⚠️ Complete compiler validation - **MISSING** (P1.4)

### Version 1.0 (All Layers)
- All P0 and P1 tasks completed
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%)
- Child-friendly Spanish messages throughout
- Generic set/stream operations
- HTTP API follows Elysia best practices

---

## COMPREHENSIVE STUDY FINDINGS (2026-03-13)

### Study Scope
This implementation plan was updated based on a comprehensive analysis from 5 parallel explore agents covering:
1. **Compiler Package Analysis** (85-90% complete)
2. **Runtime Package Analysis** (85% complete - Fraction ops added!)
3. **Integration Layer Analysis** (60% complete)
4. **Shared Package Analysis** (85% complete)
5. **Test Coverage Analysis** (107 tests, 100% passing)

### Key Findings

#### 1. Implementation Status Accuracy ✅ VERIFIED
The comprehensive study revealed the accurate status:
- **Layers 1-6**: 85-90% complete
- **Layer 7 (Integration)**: 0% complete (IncrementalRuntime and WebSocket Server missing)
- **Test Coverage**: 96/96 tests passing (100%)
- **TypeScript Compilation**: Passes with zero errors

#### 2. Core Functionality ✅ WORKING
**Compiler Package (90% complete):**
- ✅ Lexer: Complete - all 104 tokens defined
- ✅ Parser: Complete - all 17 grammar rules implemented
- ✅ Nested operations: FULLY IMPLEMENTED (previous report was incorrect)
- ✅ AST Builder: Complete - full CST to AST conversion
- ✅ Validation: 75% complete (6/8 rules implemented)
- ✅ Compiler Pipeline: Working end-to-end
- ✅ Child-friendly Spanish error messages: Partially implemented

**Runtime Package (80% complete):**
- ✅ Demand-Driven Evaluator: CORRECT and working
- ✅ 28/31 Operations implemented (90%):
  - Numeric: 5/5 ✓ (Natural only)
  - Boolean: 3/3 ✓
  - Comparison: 6/6 ✓
  - Filtering: 7/7 ✓
  - Sets: 6/6 but SORT/ALPHABETICAL_SORT have limitations
  - Temporal: 4/4 ✓ (fixed in P0.1)
- ✅ Graph: Basic implementation works
- ✅ Cache: Correct for demand-driven

**Missing Operations:**
- ❌ Fraction operations: INCORRECTLY IMPLEMENTED (separate ops, not overloading) (P0 - breaks type system completeness)
- ❌ Integer operations: 0% complete (P1 - type system completeness)
- ❌ Decimal operations: 0% complete (P1 - type system completeness)

#### 3. Critical Gaps ❌ IDENTIFIED
**P0 (CRITICAL):**
1. IncrementalRuntime: 0% complete (blocks WebSocket Server)
2. ❌ Fraction operations: INCORRECTLY IMPLEMENTED (separate ops, not overloading) (P0.2 needs refactor)
3. Fraction type: Uses `number` instead of `Integer` (type mismatch - decision to keep for architectural consistency)

**P1 (HIGH):**
1. Integer operations: 0% complete (type defined but no arithmetic)
2. Decimal operations: 0% complete (type defined but no arithmetic)
3. Set/Stream operations: Not generic (hardcoded to natural only)
4. Compiler validation: Missing 2 rules (set homogeneity, output node requirement)
5. HTTP API: Doesn't follow Elysia best practices

**P2 (MEDIUM):**
1. SORT: Only works with numbers
2. ALPHABETICAL_SORT: Converts everything to string
3. Spanish messages: Partial implementation
4. SetType: Uses `unknown[]` instead of `T[]`
5. Generators: Only "counter" exists

#### 4. Test Coverage ✅ EXCELLENT
**107 tests, 100% passing:**
- End-to-End: 9 tests (450% of spec requirements)
- Type Validation: 12 tests (600% of spec requirements)
- Curriculum Types: 4 tests (200% of spec requirements)
- Set Operations: 4 tests (200% of spec requirements)
- Temporal: 4 tests (200% of spec requirements)
- Error Handling: 2 tests (100% of spec requirements)
- Performance: 19 tests (950% of spec requirements)
- Demand-Driven: 2 tests (100% of spec requirements)
- HTTP API: 12 tests (100% passing)
- Fraction Operations: 11 tests (1100% of spec requirements)

**Missing:**
- Incremental Runtime tests (13 tests required, 0% implemented)
- WebSocket Server tests (0% implemented)

**No Stubbed Tests Found** ✅

#### 5. Design Decisions ❌ PARTIALLY INCORRECT
**Demand-Driven Semantics:**
- ✅ Correct per DEMAND_DRIVEN_INCREMENTAL.md
- ✅ No eager evaluation
- ✅ Cache-based memoization
- ✅ Temporal operators fixed (P0.1 completed)

**Fraction Operations Approach:**
- ✅ Should use overloading (ADD works with Natural, Integer, Decimal, Fraction)
- ✅ NOT separate ADD_FRACTION operations
- ✅ Spec confirms this approach
- ❌ **BUT current implementation is WRONG** - uses separate ADD_FRACTION operations
- ❌ Compiler cannot generate ADD_FRACTION tokens
- ❌ Needs refactor to implement overloading pattern correctly

#### 6. Completed Tasks ✅ CONFIRMED
The following tasks are COMPLETED (despite outdated plan):
- ✅ P0.1: Fix Temporal Operations State Bug (2 hours)
- ✅ P0.2: Implement 2 Stubbed Performance Tests (1.5 hours)
- ✅ P0.3: Fix 14 TypeScript Errors (4 hours)
- ❌ P0.2: Implement Fraction Operations - INCORRECTLY IMPLEMENTED (needs refactor to use overloading)
- ✅ P1.1: Implement HTTP API (10 hours)
- ✅ P1.4: Implement Nested Operation Expressions (1.5 hours)

---

## RECOMMENDATIONS

### Immediate Actions (P0 Tasks)
1. **Fix Fraction type** (0.5h) - Change numerator/denominator from `number` to `Integer` - Decision: Keep current structure for architectural consistency
2. **Refactor Fraction operations to use overloading** (4h) - **INCORRECTLY IMPLEMENTED, needs refactor** - Separate ops don't work with compiler
3. **Implement IncrementalRuntime** (8h) - Enable Layer 7 completion

### MVP Completion (P1 Tasks)
1. **Implement Integer operations** (2h) - Complete Integer type system
2. **Implement Decimal operations** (2h) - Complete Decimal type system
3. **Make Set/Stream operations generic** (5h) - Remove natural-only restriction
4. **Implement missing compiler validation** (3h) - Improve error messages
5. **Refactor HTTP API** (11h) - Follow Elysia best practices (optional for MVP)

### Post-MVP (P2/P3 Tasks)
1. **Complete Spanish messages** (3h) - Better UX
2. **Improve SORT operations** (2h) - Remove limitations
3. **Implement WebSocket Server** (12h) - IDE live feedback
4. **Complete execution trace** (2h) - Debugging support

---

## CONCLUSION

The implementation is **78% complete** with:
- **Core compiler/runtime:** 90% complete (Fraction ops need refactor)
- **Type system:** 80% complete (Fraction ops INCORRECTLY implemented, Integer/Decimal still missing)
- **Layer 7 (Integration):** 0% complete (IncrementalRuntime and WebSocket Server missing)
- **Test coverage:** 100% (107/107 tests passing, but fraction tests bypass compiler)

**Critical Path to MVP (22 hours):**
1. P0.1: Fix Fraction type (0.5h) - Decision: Keep current structure
2. P0.2: Refactor Fraction operations to use overloading (4h) - **INCORRECTLY IMPLEMENTED, needs refactor**
3. P0.3: Implement IncrementalRuntime (8h)
4. P1.1: Implement Integer operations (2h)
5. P1.2: Implement Decimal operations (2h)
6. P1.3: Make Set/Stream operations generic (5h)
7. P1.4: Implement missing compiler validation (3h)
8. P1.5: Refactor HTTP API (11h) - **OPTIONAL**

**Version 1.0 (54.5 hours total):**
- Complete all P0, P1, and P2 tasks
- WebSocket server for live feedback
- Complete test coverage
- Child-friendly Spanish messages throughout

**Note:** The plan follows the **Ralph Wiggum method** - each layer must be fully working before moving to the next. All tasks include acceptance criteria from specs and Ralph Wiggum checklist items.

**End of Implementation Plan**
