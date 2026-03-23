# TEST REORGANIZATION SPECIFICATION

**Purpose:** Reorganize tests to support both Batch Runtime and Incremental Runtime with proper test coverage and shared test utilities.

**Created:** 2026-03-15
**Based On:** Analysis of current test structure and incremental runtime requirements

---

## Current State

### Test Coverage
- **Runtime (batch mode):** 25 tests in `packages/runtime/src/runtime.test.ts`
- **IncrementalRuntime (incremental mode):** 0 tests (NO TEST FILE EXISTS)
- **Integration tests:** 20 test files in `packages/tests/`, ALL using batch Runtime exclusively
- **executeOperation()** in IncrementalRuntime is a stub that throws "not yet implemented"

### Critical Issue
Incremental runtime has 0 tests while normal runtime has 25 tests, and there's no clear separation of shared vs specific tests.

---

## Functional Analysis

### Shared Functionality (~70% of code)
Both runtimes share these behaviors:
- Demand-driven evaluation (both use identical semantics)
- Program loading (both use DataflowProgram)
- Caching/memoization (both cache results)
- Operation execution (both need to execute same operations)
- Temporal operators (FBY, NEXT, FIRST, ACCUMULATE)
- Data type wrapping (natural, integer, decimal, fraction, etc.)
- Time parameter support (both handle temporal evaluation)

### Batch Runtime Unique (30%)
- Complete program requirement (rejects incomplete programs)
- Output-node driven demand (evaluates only from output nodes)
- Simple execution model (execute() returns array of outputs)
- Error handling (throws exceptions on errors)
- Graph reset on load (creates new graph each time)

### IncrementalRuntime Unique (70%)
- Partial graph support (accepts incomplete programs)
- Subscription-based demand (external code subscribes to nodes)
- Node state tracking (completed/pending/processing/error)
- Incremental updates (updateGraph with delta)
- Dependency tracking (maintains graph of dependencies)
- Cache invalidation (smart invalidation of dependent nodes)
- Notification system (pub/sub for change notifications)
- Missing input detection (returns "pending" instead of errors)
- Child-friendly Spanish error messages
- Partial evaluation (evaluatePartial() returns PartialEvaluation)

---

## Proposed Test Structure

```
packages/tests/src/
├── shared/                      # Tests for BOTH runtimes
│   ├── arithmetic.test.ts      # ADD, SUBTRACT, MULTIPLY, DIVIDE
│   ├── fractions.test.ts        # Fraction operations
│   ├── sets.test.ts             # UNION, SORT, FILTER
│   ├── temporal.test.ts         # FBY, FIRST, NEXT
│   ├── curriculum.test.ts       # Shape, Car, Food, Animal, Person
│   ├── demand-driven.test.ts    # Lazy evaluation, unreferenced nodes
│   └── errors-runtime.test.ts  # Division by zero, etc.
│
├── batch/                       # Batch Runtime specific tests
│   ├── execution.test.ts        # Full program execution
│   ├── performance.test.ts      # Performance benchmarks
│   └── full-program.test.ts    # Complex multi-output programs
│
├── incremental/                 # IncrementalRuntime specific tests
│   ├── partial-evaluation.test.ts
│   ├── subscriptions.test.ts
│   ├── graph-updates.test.ts
│   ├── incremental-recompute.test.ts
│   ├── missing-inputs.test.ts
│   ├── notifications.test.ts
│   └── cache-invalidation.test.ts
│
└── compiler/                    # Compiler tests (unchanged)
    ├── validation.test.ts
    ├── type-checking.test.ts
    └── cycle-detection.test.ts
```

---

## Test Utilities Design

### Runtime Factory
Create utilities to run same tests with both Runtime and IncrementalRuntime:

```typescript
// packages/runtime/src/test-utils/runtime-factory.ts
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
        return outputs[0]; // Simplified
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

### Test Fixtures
Common test fixtures and helpers:

```typescript
// packages/runtime/src/test-utils/test-fixtures.ts
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

// More fixtures...
```

### Test Helpers
Common assertion helpers:

```typescript
// packages/runtime/src/test-utils/test-helpers.ts
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

// More helpers...
```

---

## Test Categories

### Shared Tests (~50 tests)

Tests that should pass identically on both runtimes:

1. **Arithmetic Operations** (~15 tests)
   - ADD, SUBTRACT, MULTIPLY, DIVIDE
   - Complex expressions
   - Edge cases (zero, negatives, etc.)

2. **Fraction Operations** (~10 tests)
   - All fraction arithmetic
   - Simplification
   - Division by zero

3. **Set Operations** (~8 tests)
   - UNION, INTERSECTION, DIFFERENCE
   - FILTER, SORT
   - Set homogeneity

4. **Temporal Operators** (~7 tests)
   - FBY, NEXT, FIRST, ACCUMULATE
   - Time-based evaluation
   - Stream initialization

5. **Curriculum Types** (~6 tests)
   - Shape, Car, Food, Animal, Person
   - FILTER_BY_COLOR, FILTER_BY_TASTE, FILTER_BY_TYPE
   - COMPARE operations

6. **Demand-Driven Semantics** (~3 tests)
   - Lazy evaluation (no evaluation without demand)
   - Unreferenced nodes not evaluated
   - No outputs case

7. **Runtime Errors** (~3 tests)
   - Division by zero
   - Type mismatches
   - Invalid operations

### Batch-Specific Tests (~15 tests)

Tests that only apply to batch runtime:

1. **Program Loading** (~3 tests)
   - Load simple program
   - Load complex program
   - Replace program

2. **Full Execution** (~8 tests)
   - Execute complete program
   - Multiple outputs
   - Error handling
   - Performance benchmarks

3. **Output-Driven Demand** (~4 tests)
   - Only outputs evaluated
   - Evaluation order preserved
   - Cache correctness

### Incremental-Specific Tests (~40 tests)

Tests that only apply to incremental runtime:

1. **Subscription System** (~5 tests)
   - Register subscription
   - Unregister subscription
   - Notify subscribers on changes
   - Multiple subscribers
   - Cleanup when no subscribers

2. **Partial Evaluation** (~7 tests)
   - Evaluate with no demand sources
   - Evaluate subscribed nodes
   - Evaluate output nodes
   - Evaluate both outputs and subscriptions
   - Return completed state
   - Return pending state
   - Return error state

3. **Missing Input Handling** (~6 tests)
   - Detect missing input port
   - Create MissingInput with correct port
   - Provide child-friendly error messages
   - Cascade pending states
   - Differentiate missing vs errors
   - Report multiple missing inputs

4. **Graph Updates** (~8 tests)
   - Add nodes to graph
   - Remove nodes from graph
   - Update node values
   - Trigger only affected node re-evaluation
   - Maintain cache for unchanged nodes
   - Update multiple nodes
   - Update dependency graph
   - Handle updates during evaluation

5. **Incremental Recompute** (~6 tests)
   - Re-evaluate only changed nodes
   - Re-evaluate only dependent nodes
   - Preserve cache for unchanged nodes
   - Handle circular dependencies
   - Performance improvement over full re-evaluation
   - Cache invalidation correctness

6. **Notifications** (~5 tests)
   - Push node_state_changed events
   - Include correct node data
   - Include timestamp
   - Batch notifications for efficiency
   - Handle multiple changes

7. **Cache Invalidation** (~4 tests)
   - Invalidate dependent nodes on change
   - Preserve cache for unaffected nodes
   - Clear cache on program reload
   - Handle multiple levels of dependencies

---

## Implementation Tasks

### Task P0.6: Implement executeOperation in IncrementalRuntime
- **Priority:** P0 URGENT
- **Estimated Time:** 2 hours
- **Impact:** Unblocks ALL incremental runtime tests
- **Acceptance Criteria:**
  - executeOperation() method implemented
  - Uses OPERATION_REGISTRY for dispatch
  - Handles all operation types
  - Returns correct results
  - Throws appropriate errors

### Task P0.7: Create Test Utilities
- **Priority:** P0 URGENT
- **Estimated Time:** 1.5 hours
- **Impact:** Enables shared tests between runtimes
- **Acceptance Criteria:**
  - Runtime factory created
  - Test fixtures created
  - Test helpers created
  - describeWithBothRuntimes works
  - Both runtime types can be instantiated

### Task P1.8: Extract Shared Tests
- **Priority:** P1 HIGH
- **Estimated Time:** 3 hours
- **Impact:** Reduces duplication, ensures both runtimes tested
- **Acceptance Criteria:**
  - Operation tests extracted to shared/
  - All shared tests pass on both runtimes
  - runtime.test.ts updated (batch-specific only)
  - No tests lost or duplicated

### Task P1.9: Create IncrementalRuntime Tests
- **Priority:** P1 HIGH
- **Estimated Time:** 4 hours
- **Impact:** Critical for incremental functionality verification
- **Acceptance Criteria:**
  - All 7 test categories implemented
  - ~40 tests created
  - All incremental tests pass
  - Coverage of unique incremental features

### Task P2.5: Reorganize Integration Tests
- **Priority:** P2 MEDIUM
- **Estimated Time:** 5 hours
- **Impact:** Better test organization, clearer separation
- **Acceptance Criteria:**
  - All tests moved to new structure
  - Clear separation by scope
  - All tests still pass
  - TypeScript typecheck passes

### Task P2.6: Update Batch Runtime Tests
- **Priority:** P2 MEDIUM
- **Estimated Time:** 2 hours
- **Impact:** Clear batch-specific test coverage
- **Acceptance Criteria:**
  - runtime.test.ts contains only batch-specific tests
  - Operation tests removed (moved to shared)
  - All batch tests pass

---

## Expected Outcomes

### After Reorganization
- **Shared tests:** ~50 tests (arithmetic, fractions, sets, temporal, curriculum, demand-driven, errors)
- **Batch-specific tests:** ~15 tests (execution, performance, program loading)
- **Incremental-specific tests:** ~40 tests (subscriptions, partial evaluation, graph updates, etc.)
- **Compiler tests:** ~20 tests (unchanged)
- **Total tests:** ~125 tests (existing + incremental)

### Benefits
1. ✅ Comprehensive test coverage for IncrementalRuntime (currently 0 tests)
2. ✅ Shared tests ensure both runtimes produce identical results
3. ✅ Clear separation of concerns (shared vs specific)
4. ✅ Reduced duplication through test factories
5. ✅ Better maintainability (tests organized by scope)
6. ✅ Easier to identify which runtime has issues

### Test Coverage Metrics
- **IncrementalRuntime:** 0% → 100% (0 → 40 tests)
- **Shared functionality:** ~70% of runtime code covered by shared tests
- **Batch-specific:** 100% of unique batch features covered
- **Incremental-specific:** 100% of unique incremental features covered

---

## Performance Requirements

- **Test execution time:** <5 seconds for full test suite
- **Test startup time:** <1 second for each test file
- **Test isolation:** No shared state between test runs
- **Test reliability:** 100% pass rate (flaky tests unacceptable)

---

## Success Metrics

1. ✅ All 125 tests passing (100% pass rate)
2. ✅ IncrementalRuntime has 40 tests (up from 0)
3. ✅ Shared tests run on both runtimes
4. ✅ Test suite runs in <5 seconds
5. ✅ Clear separation of shared vs specific tests
6. ✅ No test code duplication
7. ✅ TypeScript typecheck passes with no errors
8. ✅ Integration with CI/CD pipeline

---

## Related Specifications

- `specs/INTEGRATION_SPEC.md` - Integration layer requirements
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` - Demand-driven evaluation model
- `IMPLEMENTATION_PLAN.md` - Implementation task breakdown
