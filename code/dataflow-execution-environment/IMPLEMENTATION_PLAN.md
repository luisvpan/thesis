# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-15

---

## Executive Summary

This document provides a comprehensive implementation plan for building a dataflow programming language compiler and runtime for children aged 6-9. The plan follows the **Ralph Wiggum method** - building in functional layers, each completely working before moving to the next.

### Technology Stack

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

## Current Status (2026-03-15)

### Test Status
- **Overall:** 148/148 passing (100% pass rate)
- **Last improvement:** P0.7 tasks completed (2026-03-16) - +13 tests passing + fixed 2 IncrementalRuntime bugs

### Test History
- 2026-03-15: 135/135 passing (100%) → after P0.6 tasks completed
- 2026-03-15: 124/124 passing (100%) → after P0.4, P0.5 tasks completed
- 2026-03-14: 117/124 passing (94.4%) → after P0 tasks completed
- 2026-03-14: 99/118 passing (84.6%) → after manual fixes and type bug
- 2026-03-14: 54/117 passing (46.2%) → initial state

### Component Status

| Component | Status | Completion | Test Pass Rate | Critical Issues |
|-----------|--------|------------|----------------|-----------------|
| **Shared Package** | Good | 80-85% | N/A | Types correct, operations mostly complete |
| **Compiler Package** | Good | 75% | 83.3% (10/12) | Validation logic issues, missing Integer/Decimal contracts |
| **Runtime Package** | Good | 80-85% | 100% (25/25) | Integer/Decimal overloading still missing |
| **HTTP API Package** | Functional | 64% | 100% (12/12) | Works but doesn't follow Elysia best practices |
| **WebSocket Server Package** | Ready to Implement | 0% | N/A | P0.4 unblocked - ready to start implementation |

### Layer Progress

| Layer | Description | Status | Completion | Key Blockers |
|-------|-------------|--------|------------|--------------|
| **Layer 1** | Natural numbers + ADD only | ✅ COMPLETE | 100% | None |
| **Layer 2** | + Arithmetic operations (SUBTRACT, MULTIPLY, DIVIDE, COMPARE) | ⚠️ PARTIAL | 70% | Integer/Decimal operations missing |
| **Layer 3** | + Curriculum types (Shape, Car, Food, Animal, Person) | ⚠️ PARTIAL | 70% | Car/Food/Animal FILTER ops missing |
| **Layer 4** | + Set operations (FILTER, UNION, INTERSECTION, etc.) | ⚠️ PARTIAL | 60% | Non-generic sets only |
| **Layer 5** | + Temporal operators (FBY, NEXT) | ⚠️ PARTIAL | 75% | FIRST returns raw value |
| **Layer 6** | + Streams (continuous data) | ⚠️ PARTIAL | 50% | Non-generic streams only |
| **Layer 7** | + Integration interfaces | ❌ NOT STARTED | 0% | IncrementalRuntime broken, WebSocket empty, HTTP API needs refactoring |

---

## Completed Work Summary

### 2026-03-16: P0.7 Task Completed - Create Test Utilities for Shared Runtime Tests

1. **Created runtime factory** - Implemented `createRuntimeContext()` factory for both batch and incremental runtimes
2. **Created describeWithBothRuntimes helper** - Helper to run same tests on both runtimes
3. **Created test fixtures** - Added common test programs (simpleArithmeticProgram, simpleFractionProgram, complexArithmeticProgram)
4. **Created test helpers** - Added assertion helpers for all type kinds (expectNatural, expectFraction, etc.)
5. **Created test utilities index** - Central export for all test utilities
6. **Created test utilities tests** - 13 tests validating test utilities functionality
7. **Fixed IncrementalRuntime bugs** - Found and fixed 2 bugs:
   - Bug 1: Line 313 - `v.value` should be `v.state.value` when extracting evaluated inputs
   - Bug 2: Line 258 - `sourceResult.value` should be `sourceResult.state.value` for Output nodes
8. **Test verification** - All 148 tests passing (up from 135)

**Impact:** Unblocks P1.9 (IncrementalRuntime-specific tests) and P1.8 (Extract Shared Tests)
**Spec Reference:** `specs/TESTS_SPEC.md` lines 264-310

### 2026-03-15: P0.6 Task Completed - Implement executeOperation in IncrementalRuntime

1. **Implemented executeOperation() method** - Added full dispatch to all operation functions
2. **Added operation imports** - Imported all operation functions from numeric, boolean, comparison, filtering, sets
3. **Contract validation** - Validates input types against OPERATION_REGISTRY before dispatch
4. **Input wrapping** - Wraps inputs with dummy IDs for compatibility with operation functions
5. **Test verification** - All 135 tests still passing (100%)
6. **Build verification** - Runtime package builds successfully

**Impact:** Unblocks IncrementalRuntime tests (can now evaluate operations)
**Spec Reference:** `specs/TESTS_SPEC.md` lines 233-260

### 2026-03-15: P0.5 Task Completed - Implement DataflowGraph.evaluate Method

1. **Initial implementation** - Implemented in commit 5927d93 on 2026-03-14
2. **Refactored away** - Method was removed in commit ea37f75 as temporal operations now use DemandDrivenEvaluator directly

**Impact:** Temporal operations now directly use DemandDrivenEvaluator
**Spec Reference:** `specs/DEMAND_DRIVEN_INCREMENTAL.md`

### 2026-03-15: P0.4 Task Completed - Fix IncrementalRuntime Syntax Errors

1. **Fixed all 70+ syntax errors** - Corrected missing parentheses, invalid string literals, typos
2. **TypeScript compilation** - IncrementalRuntime now compiles without errors
3. **Test verification** - All 124 tests still passing
4. **Build verification** - Runtime package builds successfully

**Impact:** Unblocks WebSocket Server implementation (P0.4 unblocked)
**Commit:** eb9c2a8 on 2026-03-15
**Spec Reference:** `specs/INTEGRATION_SPEC.md` lines 365-771

### 2026-03-14: P0 Tasks Completed (7 tests fixed)

1. **Parser Support for Negative Integer Literals** - Added grammar rule for negative integers
2. **Fixed Test Bug: Boolean/Fraction Type Mismatch** - Corrected test expectations
3. **Fixed Mixed-Type Arithmetic Tests** - Updated 5 tests to avoid cross-type operations
4. **Fixed Runtime TypeError Guard** - Added guard clause for invalid programs

### 2026-03-14: Type Compatibility Bug Fix (+30 tests)
- Fixed `inputDataType` vs `inputType` bug in dag-validator.ts line 151
- Fixed false type errors across all operations

### 2026-03-14: Previous Quick Wins
- Stream type resolution fix (P0.0)
- evaluatedInputs undefined fix (P0.1)
- Fraction operations tests updated (P0.3)

---

## Active Tasks by Priority

### 🔥 P0 URGENT (MVP Blockers)

#### Task P0.7: Create Test Utilities for Shared Runtime Tests (1.5 hours)

**Priority:** P0 URGENT
**Status:** COMPLETED (2026-03-16)
**Estimated Time:** 1.5 hours
**Impact:** Enables shared tests between runtimes
**Files:**
- `packages/runtime/src/test-utils/runtime-factory.ts` (CREATED)
- `packages/runtime/src/test-utils/test-fixtures.ts` (CREATED)
- `packages/runtime/src/test-utils/test-helpers.ts` (CREATED)
- `packages/runtime/src/test-utils.test.ts` (CREATED)
- `packages/runtime/src/test-utils/index.ts` (CREATED)

**Why Critical:**
- Enables shared tests that run on both Runtime and IncrementalRuntime
- Reduces test duplication and ensures both runtimes produce identical results
- Foundation for test reorganization (P1.8, P1.9)
- Critical for comprehensive IncrementalRuntime test coverage

**Files to Create:**

**1. Runtime Factory** (`packages/runtime/src/test-utils/runtime-factory.ts`):
```typescript
import { Runtime } from '../runtime';
import { IncrementalRuntime } from '../incremental-runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

export type RuntimeInstance = Runtime | IncrementalRuntime;

export interface RuntimeTestContext {
  runtime: RuntimeInstance;
  loadProgram: (program: DataflowProgram) => void;
  execute: (time?: number) => unknown[] | { nodeStates: Map<string, any>; changedNodes: string[] };
  getOutput: (nodeId: string, time?: number) => any;
}

export function createRuntimeContext(runtimeType: 'batch' | 'incremental'): RuntimeTestContext {
  const runtime = runtimeType === 'batch' ? new Runtime() : new IncrementalRuntime();

  return {
    runtime,
    loadProgram: (program: DataflowProgram) => {
      runtime.loadProgram(program);
    },
    execute: (time?: number) => {
      if (runtimeType === 'batch') {
        return (runtime as Runtime).execute();
      } else {
        return (runtime as IncrementalRuntime).evaluatePartial(time || 0);
      }
    },
    getOutput: (nodeId: string, time?: number) => {
      if (runtimeType === 'batch') {
        const outputs = (runtime as Runtime).execute();
        return outputs[0];
      } else {
        const state = (runtime as IncrementalRuntime).getNodeState(nodeId);
        return state.status === 'completed' ? state.value : null;
      }
    }
  };
}

export function describeWithBothRuntimes(name: string, testFn: (context: RuntimeTestContext) => void) {
  describe(`${name} (batch)`, () => {
    const context = createRuntimeContext('batch');
    testFn(context);
  });

  describe(`${name} (incremental)`, () => {
    const context = createRuntimeContext('incremental');
    testFn(context);
  });
}
```

**2. Test Fixtures** (`packages/runtime/src/test-utils/test-fixtures.ts`):
```typescript
import type { DataflowProgram } from '@dataflow/shared/types';

export const simpleArithmeticProgram: DataflowProgram = {
  metadata: { programId: 'test-arithmetic' },
  graph: {
    nodes: [
      { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
      { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
      { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
      { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
    ],
    edges: [
      { from: 'a', to: 'add' },
      { from: 'b', to: 'add' },
      { from: 'add', to: 'result' }
    ]
  }
};

// Add more fixtures as needed
```

**3. Test Helpers** (`packages/runtime/src/test-utils/test-helpers.ts`):
```typescript
import { expect } from 'bun:test';

export function expectNatural(value: any, expected: number) {
  expect(value).toEqual({ kind: 'natural', value: expected });
}

export function expectFraction(value: any, expected: { numerator: number; denominator: number }) {
  expect(value).toEqual({ kind: 'fraction', ...expected });
}

export function expectBoolean(value: any, expected: boolean) {
  expect(value).toEqual({ kind: 'boolean', value: expected });
}

// Add more helpers as needed
```

**Dependencies:** None (P0.6 completed - executeOperation implemented)

**Acceptance Criteria (from specs/TESTS_SPEC.md):**
- ✓ Runtime factory created with createRuntimeContext()
- ✓ describeWithBothRuntimes() helper works
- ✓ Test fixtures created for common programs
- ✓ Test helpers created for common assertions
- ✓ Both Runtime and IncrementalRuntime can be instantiated
- ✓ Batch and incremental execution paths work
- ✓ TypeScript typecheck passes

**Required Tests:**
- [ ] Test: createRuntimeContext('batch') works correctly
- [ ] Test: createRuntimeContext('incremental') works correctly
- [ ] Test: describeWithBothRuntimes runs tests on both runtimes
- [ ] Test: Test fixtures produce valid DataflowProgram objects
- [ ] Test: Test helpers assert correctly for all types

**Spec Reference:** `specs/TESTS_SPEC.md` lines 264-310

**Layer:** Cross-layer (testing infrastructure)

**Ralph Wiggum Checklist:**
- [ ] Runtime factory created
- [ ] Test fixtures created
- [ ] Test helpers created
- [ ] describeWithBothRuntimes() works
- [ ] Both runtime types can be instantiated
- [ ] TypeScript typecheck passes
- [ ] Git commit: "test(infra): create test utilities for shared runtime tests"

---

### 🎯 P1 HIGH (MVP Features)

#### Task P1.8: Extract Shared Tests (3 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 3 hours
**Impact:** Reduces duplication, ensures both runtimes tested
**Files:**
- `packages/runtime/src/runtime.test.ts` (extract operation tests)
- `packages/tests/src/shared/arithmetic.test.ts` (NEW)
- `packages/tests/src/shared/fractions.test.ts` (NEW)
- `packages/tests/src/shared/sets.test.ts` (NEW)
- `packages/tests/src/shared/temporal.test.ts` (NEW)
- `packages/tests/src/shared/curriculum.test.ts` (NEW)

**Why Important:**
- Reduces test duplication between runtimes
- Ensures both runtimes produce identical results
- Foundation for comprehensive test coverage
- ~50 tests should run on both runtimes

**Tests to Extract to Shared:**
From `packages/runtime/src/runtime.test.ts` (lines 120-662):
1. ✅ ADD operation (lines 120-145)
2. ✅ SUBTRACT operation (lines 147-172)
3. ✅ MULTIPLY operation (lines 174-199)
4. ✅ DIVIDE operation (lines 201-226)
5. ✅ COMPARE operations (lines 253-332)
6. ✅ Complex expression (3 + 2) * (10 - 6) (lines 334-367)
7. ✅ Fraction ADD (lines 370-395)
8. ✅ Fraction ADD simplifies to whole number (lines 397-422)
9. ✅ Fraction SUBTRACT (lines 424-449)
10. ✅ Fraction MULTIPLY (lines 451-476)
11. ✅ Fraction DIVIDE (lines 478-503)
12. ✅ Fraction COMPARE equal (lines 505-530)
13. ✅ Fraction COMPARE different (lines 532-557)
14. ✅ Fraction simplification (lines 559-584)
15. ✅ Fraction negative results (lines 586-611)
16. ✅ Fraction division by zero (lines 613-636)
17. ✅ Fraction zero denominator (lines 638-662)

**Implementation:**
```typescript
// packages/tests/src/shared/arithmetic.test.ts
import { describe, it, expect } from 'bun:test';
import { describeWithBothRuntimes, createRuntimeContext } from '../../../runtime/src/test-utils/runtime-factory';
import type { DataflowProgram } from '@dataflow/shared/types';

describeWithBothRuntimes('Arithmetic Operations - ADD', (context) => {
  it('should execute ADD operation', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
        ],
        edges: [
          { from: 'a', to: 'add' },
          { from: 'b', to: 'add' },
          { from: 'add', to: 'result' }
        ]
      }
    };

    context.loadProgram(program);
    const result = context.getOutput('result');
    expect(result).toEqual({ kind: 'natural', value: 5 });
  });
});

// More tests...
```

**Dependencies:** P0.7 (Test Utilities)

**Acceptance Criteria (from specs/TESTS_SPEC.md):**
- ✓ All operation tests extracted from runtime.test.ts
- ✓ ~50 shared tests created in shared/ directory
- ✓ All shared tests use describeWithBothRuntimes
- ✓ All shared tests pass on both runtimes
- ✓ runtime.test.ts updated (batch-specific only)
- ✓ No tests lost or duplicated

**Required Tests:**
- [ ] Test: All arithmetic operations work on both runtimes
- [ ] Test: All fraction operations work on both runtimes
- [ ] Test: All set operations work on both runtimes
- [ ] Test: All temporal operators work on both runtimes
- [ ] Test: All curriculum types work on both runtimes
- [ ] Test: Demand-driven semantics work on both runtimes

**Spec Reference:** `specs/TESTS_SPEC.md` lines 312-410

**Layer:** Cross-layer (testing)

**Ralph Wiggum Checklist:**
- [ ] Operation tests extracted to shared/
- [ ] All shared tests use describeWithBothRuntimes
- [ ] All shared tests pass on both runtimes
- [ ] runtime.test.ts updated
- [ ] TypeScript typecheck passes
- [ ] Git commit: "test(shared): extract shared runtime tests"

---

#### Task P1.9: Create IncrementalRuntime-Specific Tests (4 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 4 hours
**Impact:** Critical for incremental functionality verification
**Files:**
- `packages/tests/src/incremental/partial-evaluation.test.ts` (NEW)
- `packages/tests/src/incremental/subscriptions.test.ts` (NEW)
- `packages/tests/src/incremental/graph-updates.test.ts` (NEW)
- `packages/tests/src/incremental/incremental-recompute.test.ts` (NEW)
- `packages/tests/src/incremental/missing-inputs.test.ts` (NEW)
- `packages/tests/src/incremental/notifications.test.ts` (NEW)
- `packages/tests/src/incremental/cache-invalidation.test.ts` (NEW)

**Why Important:**
- IncrementalRuntime currently has 0 tests
- Critical for verifying unique incremental features
- Essential for Layer 7 (WebSocket Server)
- ~40 tests needed for comprehensive coverage

**Test Categories:**

**A. Subscription System Tests (~5 tests):**
```typescript
describe('IncrementalRuntime - Subscriptions', () => {
  it('should register subscription to a node');
  it('should unregister subscription from a node');
  it('should notify subscribers on value changes');
  it('should handle multiple subscribers to same node');
  it('should clean up when no subscribers remain');
});
```

**B. Partial Evaluation Tests (~7 tests):**
```typescript
describe('IncrementalRuntime - Partial Evaluation', () => {
  it('should return empty result with no demand sources');
  it('should evaluate subscribed nodes');
  it('should evaluate output nodes');
  it('should evaluate both outputs and subscriptions');
  it('should return completed state for successfully evaluated nodes');
  it('should return pending state for nodes with missing inputs');
  it('should return error state for evaluation errors');
});
```

**C. Missing Input Handling Tests (~6 tests):**
```typescript
describe('IncrementalRuntime - Missing Inputs', () => {
  it('should detect missing input port');
  it('should create MissingInput with correct port number');
  it('should provide child-friendly error messages');
  it('should cascade pending states through dependencies');
  it('should differentiate between missing inputs and evaluation errors');
  it('should report multiple missing inputs');
});
```

**D. Graph Updates Tests (~8 tests):**
```typescript
describe('IncrementalRuntime - Graph Updates', () => {
  it('should add nodes to graph');
  it('should remove nodes from graph');
  it('should update node values');
  it('should trigger only affected node re-evaluation');
  it('should maintain cache for unchanged nodes');
  it('should handle multiple node updates');
  it('should update dependency graph correctly');
  it('should handle updates during evaluation');
});
```

**E. Incremental Recompute Tests (~6 tests):**
```typescript
describe('IncrementalRuntime - Incremental Recompute', () => {
  it('should re-evaluate only changed nodes');
  it('should re-evaluate only dependent nodes');
  it('should preserve cache for unchanged nodes');
  it('should handle circular dependencies');
  it('should show performance improvement over full re-evaluation');
  it('should handle cache invalidation correctly');
});
```

**F. Notification Tests (~5 tests):**
```typescript
describe('IncrementalRuntime - Notifications', () => {
  it('should push node_state_changed events');
  it('should include correct node data in notifications');
  it('should include timestamp in notifications');
  it('should batch notifications for efficiency');
  it('should handle multiple changes in single evaluation');
});
```

**G. Cache Invalidation Tests (~4 tests):**
```typescript
describe('IncrementalRuntime - Cache Invalidation', () => {
  it('should invalidate dependent nodes on change');
  it('should preserve cache for unaffected nodes');
  it('should clear cache on program reload');
  it('should handle multiple levels of dependencies');
});
```

**Dependencies:** P0.7 (Test Utilities) - P0.6 completed (executeOperation implemented)

**Acceptance Criteria (from specs/TESTS_SPEC.md):**
- ✓ All 7 test categories implemented
- ✓ ~40 tests created for IncrementalRuntime
- ✓ All unique incremental features tested
- ✓ All incremental tests pass
- ✓ Test coverage for incremental-specific features

**Required Tests:**
- [ ] Test: All 5 subscription tests pass
- [ ] Test: All 7 partial evaluation tests pass
- [ ] Test: All 6 missing input tests pass
- [ ] Test: All 8 graph update tests pass
- [ ] Test: All 6 incremental recompute tests pass
- [ ] Test: All 5 notification tests pass
- [ ] Test: All 4 cache invalidation tests pass

**Spec Reference:** `specs/TESTS_SPEC.md` lines 412-570

**Layer:** Layer 7 (Integration)

**Ralph Wiggum Checklist:**
- [ ] All 7 test categories implemented
- [ ] ~40 tests created
- [ ] All incremental tests pass
- [ ] TypeScript typecheck passes
- [ ] Git commit: "test(incremental): create IncrementalRuntime-specific tests"

---

#### Task P1.1: Implement Integer Operations (2 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 2 hours
**Impact:** Completes numeric type system
**Files:**
- `packages/shared/src/operations/registry.ts` (add Integer contracts)
- `packages/runtime/src/operations/numeric.ts` (add Integer overloads)

**Why Important:**
- SUBTRACT on Natural produces Integer but Integer has no operations
- Type system must be complete for educational curriculum
- Unblocks complex arithmetic tests

**Operations to Implement:**
```typescript
ADD(Integer, Integer) → Integer
SUBTRACT(Integer, Integer) → Integer
MULTIPLY(Integer, Integer) → Integer
DIVIDE(Integer, Integer) → Decimal
COMPARE(Integer, Integer) → Boolean
```

**Dependencies:** None (follows existing Fraction pattern)

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 57-72):**
- ✓ ADD works with Integer inputs
- ✓ SUBTRACT works with Integer inputs (can produce negative results)
- ✓ MULTIPLY works with Integer inputs
- ✓ DIVIDE works with Integer inputs (produces Decimal)
- ✓ COMPARE works with Integer inputs
- ✓ Complex arithmetic test "(3 + 2) * (10 - 6) = 20" passes

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
- [ ] Integer operations implemented
- [ ] Integer tests pass
- [ ] Typecheck passes
- [ ] Git commit: "feat(runtime): implement integer operations"

---

#### Task P1.2: Implement Decimal Operations (2 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 2 hours
**Impact:** Division results usable
**Files:**
- `packages/shared/src/operations/registry.ts` (add Decimal contracts)
- `packages/runtime/src/operations/numeric.ts` (add Decimal overloads)

**Why Important:**
- DIVIDE on Natural/Integer produces Decimal but Decimal has no operations
- Type system must be complete for educational curriculum
- Completes numeric type system

**Operations to Implement:**
```typescript
ADD(Decimal, Decimal) → Decimal
SUBTRACT(Decimal, Decimal) → Decimal
MULTIPLY(Decimal, Decimal) → Decimal
DIVIDE(Decimal, Decimal) → Decimal
COMPARE(Decimal, Decimal) → Boolean
```

**Dependencies:** None (follows existing Fraction pattern)

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
- [ ] Decimal operations implemented
- [ ] Decimal tests pass
- [ ] Typecheck passes
- [ ] Git commit: "feat(runtime): implement decimal operations"

---

#### Task P1.3: Implement Curriculum Type Filtering Operations (3 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 3 hours
**Impact:** Completes curriculum type support
**Files:**
- `packages/shared/src/operations/registry.ts` (add Car, Food, Animal filtering signatures)
- `packages/runtime/src/operations/filtering.ts` (implement Car, Food, Animal filters)

**Why Important:**
- Curriculum types are core to educational goals (ages 6-9)
- FILTER operations are essential for set manipulation
- COMPARE operations exist but FILTER operations missing
- Incomplete support for curriculum types

**Operations to Implement:**
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
- [ ] Git commit: "feat(runtime): implement curriculum type filtering operations"

---

#### Task P1.4: Make Set/Stream Operations Generic (5 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 5 hours
**Impact:** Removes natural restriction, aligns with curriculum
**Files:**
- `packages/shared/src/types/composite.ts` (SetType uses generic T[])
- `packages/runtime/src/operations/sets.ts` (generic set operations)
- `packages/runtime/src/operations/temporal.ts` (generic stream operations)
- `packages/runtime/src/evaluator/demand-driven-evaluator.ts` (update wrapping logic)
- `packages/shared/src/operations/registry.ts` (update type signatures)

**Why Important:**
- Current set/stream operations only work with natural numbers
- Generic types would support any element type (Shape, Car, Food, Animal, Person)
- Aligns with curriculum (sets of shapes, colors, etc.)
- Fix SetType to use proper generic type T[] instead of unknown[]

**Dependencies:** P1.3 (Curriculum Type Filters) - need operations to test with

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 312-327):**
- ✓ Set<T> where T can be any DataType
- ✓ Stream<T> where T can be any DataType
- ✓ Set operations work with any element type (not just natural)
- ✓ Stream operations support any element type
- ✓ Inherited operations work for element types
- ✓ SetType uses generic T[] instead of unknown[]

**Required Tests:**
- [ ] Test: Set<shape> - UNION of shape sets works
- [ ] Test: Set<car> - FILTER_BY_COLOR on car set works
- [ ] Test: Set<food> - INTERSECTION of food sets works
- [ ] Test: Set<animal> - DIFFERENCE of animal sets works
- [ ] Test: Stream<shape> - temporal operations on shape streams work
- [ ] Test: Generic types properly enforced in type checker
- [ ] Test: Set operations maintain element type through transformations
- [ ] Test: SetType uses T[] not unknown[]

**Layer:** Layer 4 (Sets) and Layer 6 (Streams)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 312-327

**Ralph Wiggum Checklist:**
- [ ] Generic Set/Stream types implemented
- [ ] SetType uses T[] instead of unknown[]
- [ ] Generic operation tests pass
- [ ] Typecheck passes
- [ ] Git commit: "refactor(types): make set/stream operations generic"

---

#### Task P1.5: Fix Set/Object Literal Ambiguity (3 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 3 hours
**Impact:** Parser robustness for curriculum types
**Files:**
- `packages/compiler/src/parser/dataflow-parser.ts` (grammar GATE logic)
- `packages/compiler/src/ast/ast-builder.ts` (literal handling)
- `packages/compiler/src/compiler.ts` (literal node creation)

**Why Important:**
- Both Set and Object literals use {} syntax
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

**Dependencies:** None

**Acceptance Criteria:**
- ✓ Set literals parse correctly: `{ "red", "blue", "green" }`
- ✓ Object literals parse correctly: `{ color: "red", size: "small" }`
- ✓ Parser disambiguates correctly
- ✓ No fragile GATE logic
- ✓ Clear error messages for ambiguous cases
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
- [ ] Git commit: "fix(parser): resolve set/object literal ambiguity"

---

#### Task P1.6: Implement Missing Compiler Validation (3 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 3 hours
**Impact:** Better quality error messages
**Files:** `packages/compiler/src/validation/dag-validator.ts`

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

**Dependencies:** None

**Acceptance Criteria (from specs/LANGUAGE_SPEC.md lines 431-440):**
- ✓ Set operations verify all elements are same type
- ✓ Programs without output nodes produce error
- ✓ Type safety rules enforced at compile time
- ✓ Child-friendly Spanish error messages
- ✓ Literal values match declared types
- ✓ Stream sources are valid

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
- [ ] Missing validation rules implemented
- [ ] Validation tests pass
- [ ] Typecheck passes
- [ ] Git commit: "feat(compiler): add missing compiler validation rules"

---

#### Task P1.7: Refactor HTTP API to Follow Elysia Best Practices (11 hours)

**Priority:** P1 HIGH
**Status:** NOT STARTED
**Estimated Time:** 11 hours
**Impact:** Code quality and maintainability
**Files:**
- `packages/http-api/src/server.ts` (use Elysia plugins, add error middleware)
- `packages/http-api/src/routes.ts` (use Elysia.t for schema validation)
- `packages/http-api/src/index.ts` (add Eden Treaty for end-to-end type safety)

**Why Important:**
- Current implementation works but doesn't follow Elysia best practices
- Missing Elysia.t validation (runtime + compile-time type safety)
- Missing error handling middleware
- Missing plugin pattern for route organization
- Missing Eden Treaty for end-to-end type safety
- Uses manual type assertions instead of schema validation
- Returns 200 OK for validation errors (should be 422)

**Dependencies:** None (can start immediately)

**Acceptance Criteria (from specs/ELYSIA_LLMS.md):**
- ✓ Use Elysia.t for request/response schema validation
- ✓ Implement error handling middleware (Elysia.onError)
- ✓ Use plugin pattern for route organization
- ✓ Add Eden Treaty for end-to-end type safety
- ✓ Remove manual type assertions, use schema validation
- ✓ Return 422 for validation errors (not 200 OK)
- ✓ Follow Elysia best practices

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
- [ ] Git commit: "refactor(http-api): implement Elysia best practices"

---

### 🔧 P2 MEDIUM (Quality)

#### Task P2.1: Fix SetType Generic Type (1 hour)

**Priority:** P2 MEDIUM
**Status:** NOT STARTED
**Estimated Time:** 1 hour
**Impact:** Type safety for sets
**Files:** `packages/shared/src/types/composite.ts`

**Why Important:**
- SetType uses unknown[] instead of T[]
- Breaks type safety for set operations
- Should be generic Set<T> where T is element type

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

**Dependencies:** P1.4 (Make Set/Stream Generic)

**Acceptance Criteria:**
- ✓ SetType uses generic T[] instead of unknown[]
- ✓ Type safety maintained for set elements
- ✓ Typecheck passes

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 315-320

**Ralph Wiggum Checklist:**
- [ ] SetType updated to use generic T[]
- [ ] Type safety maintained
- [ ] Typecheck passes
- [ ] Git commit: "fix(types): use generic T[] for SetType elements"

---

#### Task P2.2: Fix Color Type Extra Values (0.5 hours)

**Priority:** P2 MEDIUM
**Status:** NOT STARTED
**Estimated Time:** 0.5 hours
**Impact:** Spec compliance
**Files:** `packages/shared/src/types/primitives.ts`

**Why Important:**
- Color type has extra values (white, black) not in spec
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
- [ ] Git commit: "fix(types): remove extra color values (white, black)"

---

#### Task P2.3: Improve SORT/ALPHABETICAL_SORT Operations (2 hours)

**Priority:** P2 MEDIUM
**Status:** NOT STARTED
**Estimated Time:** 2 hours
**Impact:** Improves functionality
**Files:** `packages/runtime/src/operations/sets.ts`

**Why Important:**
- SORT operations have limitations that affect functionality
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
- [ ] Git commit: "feat(operations): improve SORT/ALPHABETICAL_SORT type safety"

---

#### Task P2.4: Fix DataType Recursion (0.5 hours)

#### Task P2.5: Reorganize Integration Tests (5 hours)

**Priority:** P2 MEDIUM
**Status:** NOT STARTED
**Estimated Time:** 5 hours
**Impact:** Better test organization, clearer separation of concerns
**Files:**
- All test files in `packages/tests/src/*` (reorganize structure)

**Why Important:**
- Current structure mixes shared and specific tests
- No clear separation between batch and incremental tests
- Difficult to identify which runtime has issues
- Better organization improves maintainability

**Files to Reorganize:**
Move existing integration tests from `packages/tests/src/*` to new structure:

```typescript
// Current structure:
packages/tests/src/
├── end-to-end/
│   ├── arithmetic.test.ts
│   ├── nested-operations.test.ts
│   ├── inline-nested-operations.test.ts
│   └── types.test.ts
├── integration/
│   └── fractions.test.ts
├── curriculum/
│   ├── shapes.test.ts
│   └── comparison.test.ts
├── sets/
│   ├── union.test.ts
│   └── filter-sort.test.ts
├── temporal/
│   ├── fby.test.ts
│   └── streams.test.ts
├── demand-driven/
│   ├── unreferenced.test.ts
│   └── no-outputs.test.ts
├── errors/
│   ├── division-zero.test.ts
│   └── cycles.test.ts
├── performance/
│   ├── compilation.test.ts
│   └── execution.test.ts
├── types/
│   ├── properties.test.ts
│   └── validation.test.ts
└── test-inline-nested.test.ts

// Target structure:
packages/tests/src/
├── shared/
│   ├── arithmetic.test.ts      (from end-to-end/arithmetic.test.ts)
│   ├── fractions.test.ts        (from integration/fractions.test.ts)
│   ├── sets.test.ts             (from sets/union.test.ts, sets/filter-sort.test.ts)
│   ├── temporal.test.ts         (from temporal/fby.test.ts, temporal/streams.test.ts)
│   ├── curriculum.test.ts       (from curriculum/shapes.test.ts, curriculum/comparison.test.ts)
│   ├── demand-driven.test.ts    (from demand-driven/unreferenced.test.ts, demand-driven/no-outputs.test.ts)
│   └── errors-runtime.test.ts  (from errors/division-zero.test.ts)
│
├── batch/
│   ├── execution.test.ts        (from performance/execution.test.ts)
│   └── full-program.test.ts    (from end-to-end/nested-operations.test.ts)
│
├── compiler/
│   ├── type-validation.test.ts  (from types/validation.test.ts)
│   ├── cycle-detection.test.ts  (from errors/cycles.test.ts)
│   ├── performance.test.ts      (from performance/compilation.test.ts)
│   └── ast-structure.test.ts    (from end-to-end/types.test.ts)
│
└── incremental/                 (empty - tests in Task P1.9)
    ├── partial-evaluation.test.ts
    ├── subscriptions.test.ts
    ├── graph-updates.test.ts
    ├── incremental-recompute.test.ts
    ├── missing-inputs.test.ts
    ├── notifications.test.ts
    └── cache-invalidation.test.ts
```

**Dependencies:** P1.8 (extract shared tests), P1.9 (incremental tests)

**Acceptance Criteria (from specs/TESTS_SPEC.md):**
- ✓ All existing integration tests moved to appropriate directories
- ✓ Tests reorganized by scope (shared, batch, incremental, compiler)
- ✓ All tests still pass after reorganization
- ✓ TypeScript typecheck passes
- ✓ No test files lost or duplicated

**Required Tests:**
- [ ] Test: All tests moved to new structure
- [ ] Test: All tests still pass
- [ ] Test: TypeScript typecheck passes
- [ ] Test: Clear separation of shared vs specific tests

**Spec Reference:** `specs/TESTS_SPEC.md` lines 572-650

**Layer:** Cross-layer (testing organization)

**Ralph Wiggum Checklist:**
- [ ] All integration tests reorganized
- [ ] Clear separation of shared vs specific tests
- [ ] All tests pass after reorganization
- [ ] TypeScript typecheck passes
- [ ] Git commit: "test(reorg): reorganize integration tests by runtime type"

---

#### Task P2.6: Update Batch Runtime Tests (2 hours)

---

### 🌟 P3 LOW (Post-MVP)

#### Task P3.1: Implement WebSocket Server (12 hours)

**Priority:** P3 LOW
**Status:** NOT STARTED
**Estimated Time:** 12 hours
**Impact:** IDE live feedback
**Dependencies:** P1.7 (HTTP API Best Practices) - P0.4 completed (IncrementalRuntime fixed)

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
- [ ] Multiple concurrent clients work
- [ ] Git commit: "feat(websocket): implement WebSocket server"

---

#### Task P3.2: Complete Execution Trace Implementation (2 hours)

**Priority:** P3 LOW
**Status:** NOT STARTED
**Estimated Time:** 2 hours
**Impact:** Debugging support
**Files:** `packages/runtime/src/runtime.ts`

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
- [ ] Git commit: "feat(runtime): complete execution trace implementation"

---

#### Task P3.3: Add Parallelism to Demand-Driven Evaluation (3 hours)

**Priority:** P3 LOW
**Status:** NOT STARTED
**Estimated Time:** 3 hours
**Impact:** Performance
**Files:** `packages/runtime/src/evaluator/demand-driven-evaluator.ts`

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
- [ ] Git commit: "perf(runtime): add parallelism to demand-driven evaluation"

---

## MVP Definition & Progress

### MVP Requirements (from PROJECT_GOALS.md)

To achieve MVP, complete the following:

**Core Language (Layers 1-6):**
- ✅ Natural numbers, arithmetic operations (ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE)
- ✅ Fraction operations - COMPLETE
- ⚠️ Integer operations - MISSING (P1.1)
- ⚠️ Decimal operations - MISSING (P1.2)
- ✅ Curriculum types (Shape, Car, Food, Animal, Person)
- ⚠️ Curriculum type filters - PARTIAL (P1.3 needed)
- ✅ Set operations (UNION, INTERSECTION, DIFFERENCE, COMPLEMENT)
- ⚠️ Temporal operators - PARTIAL (demand-driven correct, missing graph.evaluate)
- ✅ Filter operations (FILTER, FILTER_BY_SIZE, FILTER_BY_COLOR for Shape)
- ⚠️ Set/Stream operations generic - MISSING (P1.4)
- ✅ Nested operations - IMPLEMENTED

**Compiler:**
- ✅ Validates programs, catches errors at compile-time
- ✅ Stream type resolution - FIXED
- ✅ Type compatibility validation - FIXED
- ⚠️ Set homogeneity validation - MISSING (P1.6)
- ⚠️ Output node requirement - MISSING (P1.6)
- ⚠️ Literal type validation - MISSING (P1.6)
- ⚠️ Stream source validation - MISSING (P1.6)
- ⚠️ Set/Object literal ambiguity - ISSUE (P1.5)
- ✅ Nested operations - IMPLEMENTED

**Runtime:**
- ✅ Executes programs correctly (simple cases)
- ⚠️ Temporal operators - PARTIAL (demand-driven correct, missing graph.evaluate)
- ✅ 36/36 operations defined in registry
- ✅ Handles 5 concurrent users without degradation

**Integration (Layer 7):**
- ✅ HTTP API for batch mode (12/12 tests passing)
- ✅ IncrementalRuntime - P0.4, P0.5, P0.6 completed (syntax errors fixed, executeOperation implemented)
- ⚠️ HTTP API Elysia best practices - MISSING (P1.7)
- ⚠️ WebSocket Server - Ready to implement (P0.4 unblocked)
- ⚠️ IncrementalRuntime tests - 0 tests (P0.7, P1.9 needed)
- ⚠️ Test organization - mixed shared/specific tests (P1.8, P2.5, P2.6 needed)

**MVP Completion Tasks:**

### ✅ Previously Completed (2026-03-14)

1. ✅ Stream Type Resolution Bug
2. ✅ evaluatedInputs Undefined
3. ✅ Duplicate Temporal Cases
4. ✅ Fraction Tests
5. ✅ Type Compatibility Bug
6. ✅ Parser Support for Negative Integer Literals
7. ✅ Fixed Test Bug - Boolean and Fraction Type Mismatch
8. ✅ Fixed Mixed-Type Arithmetic Tests
9. ✅ Fixed Runtime TypeError Guard

**Result:** 135/135 tests passing (100%) in 2026-03-15 (P0.4, P0.5, P0.6 completed)

### ⏳ Current Tasks (Not Started)

**Test Reorganization (CRITICAL for Layer 7):**
10. ⏳ P0.7: Create test utilities for shared runtime tests (1.5h) - enables shared tests
11. ⏳ P1.9: Create IncrementalRuntime-specific tests (4h) - 0 tests → ~40 tests
12. ⏳ P1.8: Extract shared tests (3h) - reduces duplication
13. ⏳ P2.5: Reorganize integration tests (5h) - better organization
14. ⏳ P2.6: Update batch runtime tests (2h) - clear batch-specific tests

**Core Language & Runtime:**
15. ⏳ P1.1: Implement Integer Operations (2h) - CRITICAL for MVP
16. ⏳ P1.2: Implement Decimal Operations (2h) - CRITICAL for MVP
17. ⏳ P1.3: Implement Curriculum Type Filtering Operations (3h)
18. ⏳ P1.4: Make Set/Stream Operations Generic (5h)
19. ⏳ P1.5: Fix Set/Object Literal Ambiguity (3h)
20. ⏳ P1.6: Implement Missing Compiler Validation (3h)
21. ⏳ P1.7: Refactor HTTP API (11h) - OPTIONAL for MVP

**Integration:**
22. ⏳ P3.1: Implement WebSocket Server (12h) - ready to start (P0.4 unblocked)

**Total MVP Time:** ~46.5 hours (excluding P1.7, P2.5, P2.6, P3.1)

**Expected Test Coverage After Reorganization:**
- Shared tests: ~50 tests (run on both runtimes)
- Batch-specific tests: ~15 tests
- Incremental-specific tests: ~40 tests (up from 0)
- Compiler tests: ~20 tests
- **Total: ~125 tests** (up from 124)

---

## Next Immediate Actions

1. [ ] P0.7: Create test utilities for shared runtime tests (1.5 hours) - enables shared tests
2. [ ] P1.9: Create IncrementalRuntime-specific tests (4 hours) - comprehensive incremental coverage
3. [ ] P1.8: Extract shared tests (3 hours) - reduces duplication
4. [ ] P1.1: Implement Integer operations (2 hours) - completes numeric type system
5. [ ] P1.2: Implement Decimal operations (2 hours) - completes numeric type system
6. [ ] P1.3: Implement curriculum type filtering (3 hours) - completes curriculum types
7. [ ] P2.5: Reorganize integration tests (5 hours) - better test organization
8. [ ] P2.6: Update batch runtime tests (2 hours) - clear batch-specific tests

---

## Performance Targets

### Current Status
- Compilation: ~50ms for <50 nodes (on par with target)
- Execution: ~2ms for simple programs (exceeds target)
- Test suite: 135/135 tests passing (100%) - IMPROVED on 2026-03-15 (P0.4, P0.5, P0.6 completed)
- IncrementalRuntime tests: 0/40 tests (0%) - BLOCKED by P0.7, P1.9

### Targets
- Compilation: <100ms for programs with <100 nodes (p95)
- Execution: <50ms for programs with <100 nodes (p95)
- Concurrent: 5 simultaneous requests without degradation
- Incremental update: 5x faster than full re-evaluation
- WebSocket roundtrip: <50ms (p95)
- Test suite execution: <5 seconds for full test suite (~125 tests)
- Test startup: <1 second per test file

---

## Success Metrics

### MVP (Layers 1-6 + Basic Integration)
- ✅ All 36/36 operations implemented in registry
- ✅ Fraction ops COMPLETE
- ⚠️ Temporal ops PARTIAL (demand-driven correct, missing graph.evaluate)
- ✅ Type compatibility validation working
- ✅ HTTP API functional (12/12 tests passing)
- ✅ Compiler validates programs correctly
- ✅ Runtime executes programs
- ✅ Nested operations supported
- ✅ 135/135 tests passing (100%)
- ✅ Handles 5 concurrent users
- ✅ IncrementalRuntime - syntax errors fixed, executeOperation implemented (P0.4, P0.5, P0.6 completed)
- ⚠️ Integer operations - MISSING (P1.1)
- ⚠️ Decimal operations - MISSING (P1.2)
- ⚠️ Curriculum type filters - PARTIAL (P1.3)
- ⚠️ Set/Stream operations generic - MISSING (P1.4)
- ⚠️ IncrementalRuntime tests - 0/40 tests (0%) - BLOCKED by P0.7, P1.9
- ⚠️ Test organization - mixed shared/specific tests (P1.8, P2.5, P2.6 needed)
- ⚠️ Complete compiler validation - MISSING (P1.6)

### Version 1.0 (All Layers)
- All P0 and P1 tasks completed
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%):
  - Shared tests: ~50 tests (run on both runtimes)
  - Batch-specific tests: ~15 tests
  - Incremental-specific tests: ~40 tests
  - Compiler tests: ~20 tests
  - Total: ~125 tests
- Child-friendly Spanish messages throughout
- Generic set/stream operations
- HTTP API follows Elysia best practices
- Temporal operators preserve demand-driven semantics
- Nested operations supported
- Clear test organization (shared vs specific)

---

## Documentation References

### Spec Files
- `specs/LANGUAGE_SPEC.md` - Complete language specification
- `specs/GRAMMAR_SPEC.md` - Grammar and syntax
- `specs/INTEGRATION_SPEC.md` - HTTP API, WebSocket, IncrementalRuntime
- `specs/INTEGRATION_TESTS_SPEC.md` - Test requirements
- `specs/ELYSIA_LLMS.md` - Elysia best practices
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` - Demand-driven evaluation model
- `specs/TESTS_SPEC.md` - Test reorganization specification (NEW)

### Key Design Documents
- `PROJECT_GOALS.md` - Project objectives and MVP definition
- `AGENTS.md` - Development guidelines and standards

---

## Appendix: Historical Details

Detailed historical information about completed tasks and bug fixes is preserved in the git commit history. The completed work summary above provides a concise overview of major accomplishments. For full details of the implementation journey, refer to the commit log and previous versions of this document.

---

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Last Updated:** 2026-03-15 (135 tests passing, 100% pass rate, ~70% overall complete, P0.4, P0.5, P0.6 completed)
**Next Review:** After completing P0.7 task (test utilities)
