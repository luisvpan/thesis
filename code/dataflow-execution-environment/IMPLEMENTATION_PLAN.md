# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-05 (Analysis complete: all 44 tests passing, prioritized task list)

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
- **Passing:** 44 (100%)
- **Failing:** 0
  - Compiler tests: 12/12 passing ✓
  - Runtime tests: 16/16 passing ✓
  - Shared tests: 16/16 passing ✓

### Layer Progress

| Layer | Status | Completion | Tests Passing |
|-------|--------|------------|---------------|
| Layer 1: Foundation | COMPLETE | 100% | 100% |
| Layer 2: Arithmetic | COMPLETE | 100% | 100% |
| Layer 3: Curriculum Types | PARTIAL | 30% | 0% |
| Layer 4: Set Operations | PARTIAL | 20% | 0% |
| Layer 5: Temporal Operators | PARTIAL | 10% | 0% |
| Layer 6: Streams | MISSING | 5% | 0% |
| Layer 7: Integration | MISSING | 0% | 0% |
| Integration Tests | MISSING | 0% | 0% |

---

## Critical Findings from Research

### 1. All Tests Passing
- **Previous plan incorrectly stated 12 failing tests**
- All 44 tests currently pass
- AST Builder CST access pattern is correct (not broken as previously thought)
- Compiler tests properly use source strings (not DataflowProgram objects)

### 2. Operation Count Discrepancy
- **Specs define 31 operations** across 6 categories
- **Compiler recognizes 15 operations** (48% complete)
- **Registry has 15 operations** (48% complete)
- **Runtime implements 5 operations** (16% complete)
- **CRASH RISK:** 15 missing operations will crash runtime

### 3. Missing Components

**Shared Package:**
- Fraction type (defined in spec, not implemented)
- 7 COMPARE_BY_* operations
- 7 FILTER_BY_* operations
- ALPHABETICAL_SORT operation
- Boolean operations: AND, OR, NOT

**Compiler Package:**
- 16 missing operation tokens/grammar rules
- Type compatibility validation (not implemented)
- Property constraint validation (not implemented)
- Stream literal parsing (tokens and grammar)
- Curriculum type literals (grammar only, no AST builder)

**Runtime Package:**
- 13 missing operations (sets, filtering, temporal)
- IncrementalRuntime class (completely missing - blocks WebSocket)
- Child-friendly Spanish error messages (not implemented)

**Integration Layer:**
- HTTP API package completely empty (0%)
- WebSocket server package completely empty (0%)

---

## PRIORITIZED IMPLEMENTATION PLAN

### PHASE 0: CRITICAL TYPE SYSTEM COMPLETION (Priority 0)

**Time Estimate:** 2 hours

#### Task 0.1: Add Fraction Type to Shared (30 minutes)

**File:** `packages/shared/src/types/primitives.ts`

**Changes:**
```typescript
export type Fraction = {
  kind: "fraction";
  numerator: number;
  denominator: number;
};

export type Primitive = Natural | Integer | Decimal | Text | Boolean | Fraction;
```

**Dependencies:** None

**Acceptance Criteria:**
- Fraction type defined with numerator/denominator
- Added to Primitive union
- TypeScript compiles without errors

**Test Requirements:**
- Fraction type can be used in type annotations
- Fraction values have correct structure

**Layer:** Layer 1 (Foundation)

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented (no stubs)
- [ ] Typecheck passes: `bun tsc --noEmit`
- [ ] Previous tests still pass (44/44)
- [ ] Git commit with message: "feat(shared): add Fraction type"

---

#### Task 0.2: Add Missing Comparison Operations to Registry (30 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Changes:**
- Add COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE, COMPARE_BY_TASTE, COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER to Operation type
- Add all 6 operations to OPERATION_REGISTRY with correct signatures
- All COMPARE operations return Boolean type

**Dependencies:** None

**Acceptance Criteria:**
- All 7 comparison operations in registry (including existing COMPARE)
- Proper type signatures (curriculum types as inputs, Boolean as output)
- TypeScript compiles

**Test Requirements:**
- getOperationSignature() returns correct signatures
- specs/LANGUAGE_SPEC.md lines 172-276

**Layer:** Layer 3 (Curriculum Types)

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(shared): add comparison operations to registry"

---

#### Task 0.3: Add Missing Filtering Operations to Registry (30 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Changes:**
- Add FILTER_BY_SIZE, FILTER_BY_COLOR, FILTER_BY_TYPE, FILTER_BY_TASTE, FILTER_BY_AGE_GROUP, FILTER_BY_GENDER to Operation type
- Add all 6 operations to OPERATION_REGISTRY with correct signatures
- Generic FILTER already exists

**Dependencies:** None

**Acceptance Criteria:**
- All 7 filtering operations in registry (including existing FILTER)
- Proper type signatures (set of curriculum types + Text literal, return set)
- TypeScript compiles

**Test Requirements:**
- getOperationSignature() returns correct signatures
- specs/LANGUAGE_SPEC.md lines 176-246

**Layer:** Layer 4 (Set Operations)

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(shared): add filtering operations to registry"

---

#### Task 0.4: Add ALPHABETICAL_SORT to Registry (15 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Changes:**
- Add ALPHABETICAL_SORT to Operation type
- Add to OPERATION_REGISTRY after SORT

**Dependencies:** None

**Acceptance Criteria:**
- ALPHABETICAL_SORT in registry
- Type signature: set<Text> → set<Text>
- TypeScript compiles

**Test Requirements:**
- specs/LANGUAGE_SPEC.md line 121

**Layer:** Layer 4 (Set Operations)

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(shared): add ALPHABETICAL_SORT to registry"

---

#### Task 0.5: Add Boolean Operations to Registry (15 minutes)

**File:** `packages/shared/src/operations/registry.ts`

**Changes:**
- Add AND, OR, NOT to Operation type
- Add to OPERATION_REGISTRY with Boolean type signatures

**Dependencies:** None

**Acceptance Criteria:**
- AND, OR, NOT in registry
- Type signatures: Boolean operations return Boolean
- TypeScript compiles

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 134-136

**Layer:** Layer 1 (Foundation)

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(shared): add Boolean operations to registry"

---

### PHASE 1: RUNTIME OPERATIONS COMPLETION (Priority 1)

**Time Estimate:** 8 hours

**Rationale:** Runtime must have all operations before compiler can validate them. Completing runtime first ensures we can test end-to-end functionality.

#### Task 1.1: Implement Set Operations (2 hours)

**Create File:** `packages/runtime/src/operations/sets.ts`

**Operations:** UNION, INTERSECTION, DIFFERENCE, COMPLEMENT, SORT, ALPHABETICAL_SORT

**Dependencies:** None

**Acceptance Criteria:**
- UNION merges sets, removes duplicates
- INTERSECTION returns common elements
- DIFFERENCE returns elements in set1 not in set2
- COMPLEMENT returns elements in universe not in set
- SORT orders by natural order or comparison
- ALPHABETICAL_SORT sorts text lexicographically

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 183-186, 202-206, 225-229, 248-252, 271-275
- All operations handle duplicate elements correctly

**Layer:** Layer 4 (Set Operations)

**Estimated Time:** 2 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented (no stubs)
- [ ] All set operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement set operations"

---

#### Task 1.2: Implement Comparison Operations (2 hours)

**Create File:** `packages/runtime/src/operations/comparison.ts`

**Operations:** COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE, COMPARE_BY_TASTE, COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER

**Dependencies:** None

**Acceptance Criteria:**
- All operations compare specified properties
- All operations return Boolean (true if equal, false otherwise)
- Comparisons are by value, not by reference
- Type errors raised if property doesn't exist

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 173-175, 197, 218-219, 241-242, 264-265

**Layer:** Layer 3 (Curriculum Types)

**Estimated Time:** 2 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All comparison operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement comparison operations"

---

#### Task 1.3: Implement Filtering Operations (2 hours)

**Create File:** `packages/runtime/src/operations/filtering.ts`

**Operations:** FILTER, FILTER_BY_SIZE, FILTER_BY_COLOR, FILTER_BY_TYPE, FILTER_BY_TASTE, FILTER_BY_AGE_GROUP, FILTER_BY_GENDER

**Dependencies:** None

**Acceptance Criteria:**
- Generic FILTER works with any predicate
- FILTER_BY_* operations filter by specified property
- Original set remains unmodified (immutability)
- Type errors raised if property doesn't exist

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 177-181, 199-201, 221-224, 244-247, 267-270

**Layer:** Layer 4 (Set Operations)

**Estimated Time:** 2 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All filtering operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement filtering operations"

---

#### Task 1.4: Implement Temporal Operations (2 hours)

**Create File:** `packages/runtime/src/operations/temporal.ts`

**Operations:** NEXT, FIRST, FBY, ACCUMULATE

**Dependencies:** None

**Acceptance Criteria:**
- NEXT returns next value from stream (time-based)
- FIRST returns first value from stream
- FBY is pure function of time: time=0 returns initial, time>0 returns stream at time-1
- ACCUMULATE applies operation cumulatively over stream
- All temporal ops use cache indexed by time
- No mutable state used

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 296-301
- specs/DEMAND_DRIVEN_INCREMENTAL.md: Pure time-based semantics
- FBY counter: 0,1,2,3,4...

**Layer:** Layer 5 (Temporal Operators)

**Estimated Time:** 2 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All temporal operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement temporal operations"

---

#### Task 1.5: Implement Boolean Operations (30 minutes)

**Create File:** `packages/runtime/src/operations/boolean.ts`

**Operations:** AND, OR, NOT

**Dependencies:** None

**Acceptance Criteria:**
- AND returns true only if both inputs are true
- OR returns true if either input is true
- NOT returns opposite of input
- All operations return Boolean type

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 134-136

**Layer:** Layer 1 (Foundation)

**Estimated Time:** 30 minutes

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All boolean operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement boolean operations"

---

#### Task 1.6: Update Evaluator to Dispatch All Operations (30 minutes)

**File:** `packages/runtime/src/evaluator/demand-driven-evaluator.ts`

**Changes:**
- Import all new operation modules
- Add all 18 missing operations to evaluateOperation switch statement
- Ensure temporal operations receive time parameter

**Dependencies:** Tasks 1.1-1.5

**Acceptance Criteria:**
- All 31 operations dispatched correctly
- No "Unknown operation" errors
- Temporal operations receive time parameter
- TypeScript compiles

**Test Requirements:**
- All 44 tests still pass
- No runtime crashes for any operation

**Layer:** All layers (foundation update)

**Estimated Time:** 30 minutes

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] All runtime tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(evaluator): add all missing operations to dispatcher"

---

### PHASE 2: INTEGRATION TESTS (Priority 2)

**Time Estimate:** 8 hours

**Rationale:** Integration tests verify end-to-end correctness between compiler and runtime. Critical for validating that all components work together correctly. Should be implemented incrementally alongside feature development.

#### Task 2.1: Create Integration Tests Package Structure (1 hour)

**Files:** Create `packages/tests/package.json`, `packages/tests/tsconfig.json`, `packages/tests/src/` directories

**Changes:**
- Package configuration with workspace dependencies
- TypeScript config extending base tsconfig
- Directory structure: src/end-to-end, src/types, src/curriculum, src/sets, src/temporal, src/errors, src/performance, src/demand-driven
- Barrel export in src/index.ts

**Dependencies:** None

**Acceptance Criteria:**
- Tests package created with proper monorepo configuration
- Workspace dependencies configured (compiler, runtime, shared)
- TypeScript compiles for package
- Test directory structure follows spec

**Test Requirements:**
- Monorepo recognizes tests package
- Dependencies resolve correctly

**Layer:** Cross-layer (tests all components)

**Estimated Time:** 1 hour

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Typecheck passes
- [ ] Git commit with message: "feat(tests): create integration tests package"

---

#### Task 2.2: Implement End-to-End Pipeline Tests (2 hours)

**File:** `packages/tests/src/end-to-end/arithmetic.test.ts`

**Tests to Implement:**
- Test 1.1: Simple Addition (ADD with 2 inputs)
- Test 1.2: Complex Arithmetic Expression (multi-node program with ADD, SUBTRACT, MULTIPLY)

**Dependencies:** None (tests can start immediately)

**Acceptance Criteria:**
- Simple program compiles and executes correctly
- Complex program with multiple operations works end-to-end
- Output results match expected values
- Performance <10ms for simple programs
- Cache statistics verified

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 36-110
- Verify demand-driven evaluation order
- Verify no wasted computation

**Layer:** Cross-layer (tests Layer 1-2)

**Estimated Time:** 2 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement end-to-end arithmetic tests"

---

#### Task 2.3: Implement Type Validation Tests (2 hours)

**File:** `packages/tests/src/types/validation.test.ts` and `packages/tests/src/types/properties.test.ts`

**Tests to Implement:**
- Test 2.1: Type Mismatch Detection (ADD(natural, text) should fail)
- Test 2.2: Property Requirement Validation (FILTER_BY_COLOR on natural set should fail)

**Dependencies:** None (compiler validates types)

**Acceptance Criteria:**
- Type mismatches detected at compile time
- Child-friendly Spanish error messages provided
- Property requirements validated correctly
- Execution prevented for invalid programs
- Error messages include nodeId, suggestion, and example

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 112-175
- Verify TYPE_ERROR code generation
- Verify Spanish messages are age-appropriate (ages 6-9)

**Layer:** Cross-layer (tests Layer 3)

**Estimated Time:** 2 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement type validation tests"

---

#### Task 2.4: Implement Curriculum Type Tests (1.5 hours)

**File:** `packages/tests/src/curriculum/shapes.test.ts` and `packages/tests/src/curriculum/comparison.test.ts`

**Tests to Implement:**
- Test 3.1: Filter Red Shapes (FILTER_BY_COLOR on shape set)
- Test 3.2: Compare Shapes by Size (COMPARE_BY_SIZE returns Boolean)

**Dependencies:** Phase 1 (curriculum operations must be implemented first)

**Acceptance Criteria:**
- Curriculum types compile and execute correctly
- Filter operations work as expected
- Comparison operations return Boolean (value-based, not reference)
- Immutability preserved for sets
- Results match expected outputs

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 177-230
- Verify curriculum type handling
- Verify value-based comparison semantics

**Layer:** Cross-layer (tests Layer 3)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement curriculum type tests"

---

#### Task 2.5: Implement Set Operation Tests (1.5 hours)

**File:** `packages/tests/src/sets/union.test.ts` and `packages/tests/src/sets/filter-sort.test.ts`

**Tests to Implement:**
- Test 4.1: Union of Shape Sets (UNION merges sets)
- Test 4.2: Filter and Sort Combined (chained operations)

**Dependencies:** Phase 1 (set operations must be implemented first)

**Acceptance Criteria:**
- Set operations work correctly (UNION, INTERSECTION, DIFFERENCE, COMPLEMENT)
- Duplicates removed from results
- Chained operations work end-to-end
- Order is deterministic for sorted results
- Type preserved through operations

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 232-290
- Verify set operation semantics
- Verify intermediate results are correct

**Layer:** Cross-layer (tests Layer 4)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement set operation tests"

---

#### Task 2.6: Implement Temporal Operation Tests (1.5 hours)

**File:** `packages/tests/src/temporal/fby.test.ts` and `packages/tests/src/temporal/streams.test.ts`

**Tests to Implement:**
- Test 5.1: FBY Counter (temporal semantics 0, 1, 2, 3, 4...)
- Test 5.2: FIRST from Stream (extract first value)

**Dependencies:** Phase 1 (temporal operations must be implemented first)

**Acceptance Criteria:**
- FBY returns initial at time 0, subsequent at time > 0
- Demand-driven semantics verified (no eager evaluation)
- Pure function of time (no mutable state)
- Stream operations work correctly
- Type preserved through temporal operations

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 292-350
- Verify temporal operator semantics
- Verify demand-driven behavior (cache-based)

**Layer:** Cross-layer (tests Layer 5)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement temporal operation tests"

---

#### Task 2.7: Implement Error Handling Tests (1.5 hours)

**File:** `packages/tests/src/errors/division-zero.test.ts` and `packages/tests/src/errors/cycles.test.ts`

**Tests to Implement:**
- Test 6.1: Division by Zero (runtime error)
- Test 6.2: Cycle Detection (compile-time error)

**Dependencies:** None (errors exist in current implementation)

**Acceptance Criteria:**
- Runtime errors caught gracefully with child-friendly messages
- Compile-time errors detected and prevented execution
- Cycle detection works correctly
- System doesn't crash on errors
- Error messages include Spanish text for ages 6-9

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 352-420
- Verify error handling at both compile and runtime
- Verify Spanish messages are child-friendly

**Layer:** Cross-layer (tests all layers)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement error handling tests"

---

#### Task 2.8: Implement Performance Tests (1.5 hours)

**File:** `packages/tests/src/performance/compilation.test.ts` and `packages/tests/src/performance/execution.test.ts`

**Tests to Implement:**
- Test 7.1: Compilation Performance (100-node program <100ms)
- Test 7.2: Execution Performance (50-node program <50ms)

**Dependencies:** None (performance can be tested on current implementation)

**Acceptance Criteria:**
- Compilation performance target met (<100ms for 100 nodes)
- Execution performance target met (<50ms for 50 nodes)
- Cache effectiveness measured (>30% hit rate)
- No memory leaks detected
- Performance scales linearly with program size

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 422-480
- Verify performance targets
- Verify demand-driven efficiency

**Layer:** Cross-layer (tests all layers)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement performance tests"

---

#### Task 2.9: Implement Demand-Driven Semantics Tests (1.5 hours)

**File:** `packages/tests/src/demand-driven/no-outputs.test.ts` and `packages/tests/src/demand-driven/unreferenced.test.ts`

**Tests to Implement:**
- Test 8.1: No Output Nodes (nothing evaluated)
- Test 8.2: Unreferenced Nodes (only referenced nodes evaluated)

**Dependencies:** None (tests core semantics of current implementation)

**Acceptance Criteria:**
- No Output nodes = nothing evaluated (correct demand-driven)
- Unreferenced computation skipped (no wasted resources)
- Cache statistics confirm demand-driven behavior
- System doesn't hang or crash
- Evaluation order follows dependencies only

**Test Requirements:**
- specs/INTEGRATION_TESTS_SPEC.md lines 482-540
- Verify demand-driven evaluation semantics
- Verify no wasted computation

**Layer:** Cross-layer (tests core semantics)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement demand-driven semantics tests"

---

### PHASE 3: COMPILER COMPLETION (Priority 1)

**Time Estimate:** 6 hours

**Rationale:** Now that runtime has all operations, compiler can validate them. This enables end-to-end testing.

#### Task 2.1: Add Missing Comparison Tokens (30 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts` and `packages/compiler/src/parser/dataflow-parser.ts`

**Tokens:** COMPARE_BY_SIZE, COMPARE_BY_COLOR, COMPARE_BY_TYPE, COMPARE_BY_TASTE, COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER

**Dependencies:** Task 0.2 (registry has these operations)

**Acceptance Criteria:**
- All 6 comparison tokens recognized by lexer
- Parser grammar includes all comparison operations
- TypeScript compiles

**Test Requirements:**
- specs/GRAMMAR_SPEC.md lines 216-223
- Parsing tests pass for comparison ops

**Layer:** Layer 3 (Curriculum Types)

**Estimated Time:** 30 minutes

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass for comparison ops
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(lexer): add comparison operation tokens"

---

#### Task 2.2: Add Missing Filtering Tokens (30 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts` and `packages/compiler/src/parser/dataflow-parser.ts`

**Tokens:** FILTER_BY_SIZE, FILTER_BY_COLOR, FILTER_BY_TYPE, FILTER_BY_TASTE, FILTER_BY_AGE_GROUP, FILTER_BY_GENDER

**Dependencies:** Task 0.3 (registry has these operations)

**Acceptance Criteria:**
- All 6 filtering tokens recognized by lexer
- Parser grammar includes all filtering operations
- TypeScript compiles

**Test Requirements:**
- specs/GRAMMAR_SPEC.md lines 228-235
- Parsing tests pass for filtering ops

**Layer:** Layer 4 (Set Operations)

**Estimated Time:** 30 minutes

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass for filtering ops
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(lexer): add filtering operation tokens"

---

#### Task 2.3: Add ALPHABETICAL_SORT Token (15 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts` and `packages/compiler/src/parser/dataflow-parser.ts`

**Token:** ALPHABETICAL_SORT

**Dependencies:** Task 0.4 (registry has this operation)

**Acceptance Criteria:**
- ALPHABETICAL_SORT token recognized by lexer
- Parser includes the operation
- TypeScript compiles

**Test Requirements:**
- specs/GRAMMAR_SPEC.md line 258

**Layer:** Layer 4 (Set Operations)

**Estimated Time:** 15 minutes

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(lexer): add ALPHABETICAL_SORT token"

---

#### Task 2.4: Add Boolean Operation Tokens (15 minutes)

**File:** `packages/compiler/src/lexer/tokens.ts` and `packages/compiler/src/parser/dataflow-parser.ts`

**Tokens:** AND, OR, NOT

**Dependencies:** Task 0.5 (registry has these operations)

**Acceptance Criteria:**
- AND, OR, NOT tokens recognized by lexer
- Parser includes all boolean operations
- TypeScript compiles

**Test Requirements:**
- specs/GRAMMAR_SPEC.md (boolean operations section)

**Layer:** Layer 1 (Foundation)

**Estimated Time:** 15 minutes

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Parsing tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(lexer): add boolean operation tokens"

---

#### Task 2.5: Implement Type Compatibility Validation (3 hours)

**Create File:** `packages/compiler/src/validation/type-checker.ts`

**Dependencies:** All registry tasks complete

**Acceptance Criteria:**
- Input types match operation signatures from registry
- Type errors detected at compile time
- Child-friendly Spanish error messages
- Curriculum types validated (have required properties)

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 433-440
- specs/INTEGRATION_SPEC.md lines 86-111

**Layer:** All layers (enables full validation)

**Estimated Time:** 3 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Type checking tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): implement type compatibility validation"

---

#### Task 2.6: Implement Property Constraint Validation (1.5 hours)

**Create File:** `packages/compiler/src/validation/property-checker.ts`

**Dependencies:** Task 2.5 (type checker)

**Acceptance Criteria:**
- Checks types have required properties (e.g., color for FILTER_BY_COLOR)
- Property errors detected at compile time
- Child-friendly Spanish error messages

**Test Requirements:**
- specs/LANGUAGE_SPEC.md lines 438, 491-527

**Layer:** Layer 3-4 (curriculum/set types)

**Estimated Time:** 1.5 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Property checking tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): implement property constraint validation"

---

### PHASE 3: INCREMENTAL RUNTIME (Priority 2)

**Time Estimate:** 10 hours

**Rationale:** Required for WebSocket live feedback. This is a critical integration component.

#### Task 3.1: Implement IncrementalRuntime Class (7 hours)

**Create File:** `packages/runtime/src/incremental-runtime.ts`

**Dependencies:** All runtime operations complete

**Acceptance Criteria:**
- Partial graph evaluation (demand-driven)
- Node state tracking (completed/pending/error)
- Subscription management (multiple clients)
- Graph update API with cache invalidation
- Missing input extraction with Spanish messages
- Demand-driven semantics preserved
- 5x faster than full re-evaluation for incremental updates

**Test Requirements:**
- specs/INTEGRATION_SPEC.md lines 365-771
- specs/DEMAND_DRIVEN_INCREMENTAL.md complete

**Layer:** Layer 7 (Integration)

**Estimated Time:** 7 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Incremental runtime tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement IncrementalRuntime class"

---

#### Task 3.2: Add Spanish Child-Friendly Messages (3 hours)

**Files:** Runtime and validation files

**Dependencies:** Task 3.1 (IncrementalRuntime), Task 2.5-2.6 (validation)

**Acceptance Criteria:**
- All error messages in child-friendly Spanish
- Missing input messages are age-appropriate
- Type error messages use simple language
- Property error messages explain what's missing

**Test Requirements:**
- specs/INTEGRATION_SPEC.md examples of Spanish messages

**Layer:** All layers

**Estimated Time:** 3 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Message tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(messages): add child-friendly Spanish error messages"

---

### PHASE 4: HTTP API IMPLEMENTATION (Priority 3)

**Time Estimate:** 10 hours

**Rationale:** Required for CV system integration. Batch mode for complete programs.

#### Task 4.1: Implement HTTP API Server (10 hours)

**Files:** `packages/http-api/src/server.ts` (create), `packages/http-api/src/routes.ts` (create)

**Dependencies:** All compiler/runtime complete

**Acceptance Criteria:**
- POST /api/v1/compile - Validate program
- POST /api/v1/execute - Compile and run
- GET /api/v1/health - Health check
- Accept source code strings
- Return child-friendly Spanish error messages
- Support execution options (maxTimesteps, includeTrace, traceLevel)
- Return execution trace (executionOrder, nodeEvaluations, cacheHits, cacheMisses)
- Performance: <100ms compile, <50ms execute

**Test Requirements:**
- specs/INTEGRATION_SPEC.md lines 53-189

**Layer:** Layer 7 (Integration)

**Estimated Time:** 10 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] HTTP API tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Performance targets met
- [ ] Git commit with message: "feat(http-api): implement HTTP API server"

---

### PHASE 5: WEBSOCKET SERVER IMPLEMENTATION (Priority 3)

**Time Estimate:** 12 hours

**Rationale:** Required for IDE integration. Live feedback during construction.

#### Task 5.1: Implement WebSocket Server (12 hours)

**Files:** `packages/websocket-server/src/server.ts` (create), `packages/websocket-server/src/handlers.ts` (create)

**Dependencies:** Task 3.1 (IncrementalRuntime)

**Acceptance Criteria:**
- Connection at ws://localhost:3000/live
- Message handlers: validate_program, evaluate_incremental, subscribe_node, unsubscribe_node
- Push messages: validation_result, evaluation_result, node_state_changed, error
- Message protocol with messageId
- Multiple concurrent client support
- Connection resilience (reconnects)
- Child-friendly Spanish messages

**Test Requirements:**
- specs/INTEGRATION_SPEC.md lines 192-361

**Layer:** Layer 7 (Integration)

**Estimated Time:** 12 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] WebSocket tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Multiple concurrent clients work
- [ ] Git commit with message: "feat(websocket): implement WebSocket server"

---

### PHASE 6: STREAM LITERALS (Priority 4)

**Time Estimate:** 3 hours

**Rationale:** Streams are in spec but lower priority for MVP. Can be added after integration.

#### Task 6.1: Add Stream Literal Tokens and Grammar (3 hours)

**Files:** `packages/compiler/src/lexer/tokens.ts`, `packages/compiler/src/parser/dataflow-parser.ts`, `packages/compiler/src/ast/ast-builder.ts`

**Tokens:** SENSOR, GENERATOR, EXTERNAL
**Grammar:** stream_literal, stream_source rules

**Dependencies:** None

**Acceptance Criteria:**
- Stream literals parse correctly
- AST includes stream structure
- TypeScript compiles

**Test Requirements:**
- specs/GRAMMAR_SPEC.md lines 170-175, 414-427
- specs/LANGUAGE_SPEC.md lines 283-288

**Layer:** Layer 6 (Streams)

**Estimated Time:** 3 hours

**Ralph Wiggum Checklist:**
- [ ] New functionality fully implemented
- [ ] Stream parsing tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): add stream literal parsing"

---

## KEY DECISIONS DOCUMENTED

### 1. Fraction Type
**Decision:** Implement Fraction type
**Rationale:** Defined in specs, needed for educational value
**Layer:** Layer 1 (Foundation)
**Priority:** P0

### 2. FILTER vs FILTER_BY_*
**Decision:** Implement both generic FILTER and specific FILTER_BY_* operations
**Rationale:** Spec defines both, generic useful for flexibility, specific for curriculum alignment
**Layer:** Layer 4 (Set Operations)
**Priority:** P1

### 3. IncrementalRuntime Location
**Decision:** Keep IncrementalRuntime in runtime package as separate class
**Rationale:** Shares demand-driven semantics with base runtime, convenient access to graph/cache
**Layer:** Layer 7 (Integration)
**Priority:** P2

### 4. Compiler Input Format
**Decision:** Compiler accepts source code strings (not DataflowProgram objects)
**Rationale:** Already correctly implemented, simpler for parser-based workflow
**Layer:** All layers
**Priority:** N/A (already correct)

---

## PERFORMANCE TARGETS

### Current Status
- Compilation: ~50ms for <50 nodes (on par with target)
- Execution: ~2ms for simple programs (exceeds target)
- Test suite: 2.2s for 44 tests

### Targets
- Compilation: <100ms for programs with <100 nodes (p95)
- Execution: <50ms for programs with <100 nodes (p95)
- Concurrent: 5 simultaneous requests without degradation

---

## RALPH WIGGUM LAYER VERIFICATION

Before moving to next layer, verify:
- [ ] New functionality fully implemented (no stubs)
- [ ] New tests pass
- [ ] Previous layer tests still pass
- [ ] Typecheck passes: `bun tsc --noEmit`
- [ ] Lint passes: `bun run lint`
- [ ] Git committed with descriptive message

---

## SUCCESS METRICS

### MVP (Layers 1-4 + Basic Integration)
- All 31 operations implemented
- Type compatibility validation working
- HTTP API functional
- Compiler validates programs correctly
- Runtime executes programs correctly with demand-driven semantics
- Handles 5 concurrent users

### Version 1.0 (All Layers)
- All layers 1-7 implemented
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%)
- Child-friendly Spanish messages throughout

---

## OUTSTANDING QUESTIONS

1. **Stream Semantics:** Clarify how stream generators work with demand-driven evaluation
2. **Fraction Operations:** Are fraction operations (ADD, SUBTRACT, etc.) needed in addition to primitive operations?
3. **Performance:** Are current performance targets (<100ms compile, <50ms execute) realistic for complex programs?

---

## BUGS AND ISSUES

### Fixed (Previously Reported)
- ~~AST Builder CST access pattern~~ - Already correctly implemented
- ~~Compiler tests using wrong API~~ - Already using source strings

### Known Issues
- No critical bugs currently
- All 44 tests passing
- No performance issues identified

### Technical Debt
- Missing operation implementations (15 tokens, 13 runtime ops)
- Type compatibility validation not implemented
- Property constraint validation not implemented
- IncrementalRuntime not implemented
- HTTP/WebSocket servers empty

---

**Document Status:** Active implementation plan
**Next Review:** After Phase 1 (Runtime Operations) completion
**Maintainer:** Update as implementation progresses
