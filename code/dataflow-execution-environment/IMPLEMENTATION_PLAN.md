# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-13 (Synthesized findings from 5 parallel subagents, Layer 7 at 0%, critical temporal bug identified)

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

**Last Updated:** 2026-03-13 (Synthesized from 5 parallel subagent studies)

### Overall Progress: ~70% Complete (Core compiler/runtime ~85%, Layer 7 at 0%)

**Correction from Previous Estimate:**
- Previous plan incorrectly stated: Layer 7 at 35% complete
- **ACTUAL STATUS:** Layer 7 at 0% implemented
  - `packages/http-api/src/` - Empty (0 TypeScript files)
  - `packages/websocket-server/src/` - Empty (0 TypeScript files)
  - `packages/runtime/src/incremental-runtime.ts` - Does not exist

### Test Status Summary
- **Total Tests:** 30 tests, all passing (100%)
- **Stubbed:** 2 performance tests
- **Categories Covered:**
  - 1, 2, 3, 8: FULLY COVERED
  - 4, 5, 6: PARTIALLY COVERED
  - 7: STUBBED (performance tests)

### Component Status Summary

| Component | Status | Completion | Notes |
|-----------|--------|------------|-------|
| **Shared Package** | Mostly Complete | 85% | 24 operations, 1 generator, all types defined. Issues: Fraction ops missing, Set/Stream not generic, SetType uses `unknown[]` |
| **Compiler Package** | Mostly Complete | 90% | Lexer/Parser/AST/Validation/Compiler all implemented. Missing validation: set homogeneity, stream consistency, output node req, car/food/animal properties |
| **Runtime Package** | Partial | 75% | Core evaluator CORRECT, operations implemented. **CRITICAL BUG:** Temporal ops (NEXT, FIRST, FBY, ACCUMULATE) recreate generators on each call → state not preserved. Missing: Incremental Runtime (0%) |
| **HTTP API Package** | Not Started | 0% | Empty directory, no implementation |
| **WebSocket Server Package** | Not Started | 0% | Empty directory, no implementation |

### Layer Progress

| Layer | Status | Completion | Tests Passing | Issues |
|-------|--------|------------|---------------|--------|
| Layer 1: Foundation | COMPLETE | 100% | 100% | None |
| Layer 2: Arithmetic | COMPLETE | 100% | 100% | None |
| Layer 3: Curriculum Types | COMPLETE | 100% | 100% | None |
| Layer 4: Set Operations | COMPLETE | 100% | 100% | SORT only for numbers, ALPHABETICAL_SORT converts everything to string |
| Layer 5: Temporal Operators | PARTIAL | 70% | Pass (but buggy) | **CRITICAL BUG:** Generator recreation breaks state semantics |
| Layer 6: Streams | COMPLETE | 100% | 100% | None |
| Layer 7: Integration | NOT STARTED | 0% | 0% (2 stubbed) | HTTP API, WebSocket, Incremental Runtime missing |

---

## CRITICAL BLOCKERS

### Blocker 1: Temporal Operations State Bug (P0 - CRITICAL)

**Status:** ACTIVE BUG - Breaks dataflow semantics
**Discovered:** 2026-03-13 (Runtime subagent study)

**Issue:**
Temporal operations (NEXT, FIRST, FBY, ACCUMULATE) in `packages/runtime/src/operations/temporal.ts` recreate generators on each call:

```typescript
// Lines 7, 27, 50 - THE BUG
const gen = streamValue.generatorFactory ? streamValue.generatorFactory() : streamValue.generator;
```

**Impact:**
- **State NOT preserved across calls** → breaks temporal semantics
- FBY counter would return 0,0,0,0 instead of 0,1,2,3
- NEXT/ACCUMULATE produce incorrect results
- Violates core demand-driven evaluation principle from Lucid
- **Blocks correctness of Layer 5 (Temporal Operators)**

**Root Cause:**
Generators should be cached by the evaluator, not recreated. The evaluator already caches node results, but temporal ops bypass this by creating new generators.

**Fix Required:**
1. Cache the generator within the stream value itself
2. Only call `generatorFactory()` once when stream value is first created
3. Subsequent calls use the cached generator
4. OR: Use node-level cache properly (the stream value is already cached)

**Priority:** P0 - CRITICAL (breaks core semantics)

**Estimated Time:** 2 hours

**Files Affected:**
- `packages/runtime/src/operations/temporal.ts` (lines 7, 27, 50, 76)
- May need to update `packages/runtime/src/evaluator/demand-driven-evaluator.ts` for stream wrapping

**Acceptance Criteria:**
- FBY counter produces 0,1,2,3,4... (not 0,0,0,0,0...)
- NEXT advances through stream correctly
- ACCUMULATE maintains accumulator across time steps
- All temporal operation tests pass
- Demand-driven semantics preserved (cache verified)

---

### Blocker 2: Layer 7 Components Not Implemented (P1 - CRITICAL for MVP)

**Status:** NOT STARTED
**Discovered:** 2026-03-13 (corrected from incorrect 35% estimate)

**Impact:**
- Cannot provide HTTP API for CV system integration
- Cannot provide WebSocket server for IDE live feedback
- Cannot support incremental evaluation for partial programs
- **Blocks completion of MVP**

**Missing Components:**
1. **HTTP API Server** (10 hours)
   - POST /api/v1/compile - Validate program
   - POST /api/v1/execute - Compile and run
   - GET /api/v1/health - Health check
   - Return child-friendly Spanish error messages
   - Support execution options (maxTimesteps, includeTrace, traceLevel)

2. **WebSocket Server** (12 hours)
   - Connection at ws://localhost:3000/live
   - Message handlers: validate_program, evaluate_incremental, subscribe_node, unsubscribe_node
   - Push messages: validation_result, evaluation_result, node_state_changed
   - Multiple concurrent client support
   - Child-friendly Spanish messages

3. **Incremental Runtime** (7 hours)
   - Partial graph evaluation (demand-driven)
   - Node state tracking (completed/pending/error)
   - Subscription management (multiple clients)
   - Graph update API with cache invalidation
   - Missing input extraction with Spanish messages

**Priority:** P1 - CRITICAL (blocks MVP completion)

**Estimated Time:** 29 hours total

**Spec References:**
- `specs/INTEGRATION_SPEC.md` lines 53-198 (HTTP API)
- `specs/INTEGRATION_SPEC.md` lines 200-361 (WebSocket Server)
- `specs/INTEGRATION_SPEC.md` lines 365-771 (Incremental Runtime)
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` (complete spec)

---

### Blocker 3: Missing Compiler Validation (P1 - HIGH for MVP)

**Status:** PARTIALLY COMPLETE
**Discovered:** 2026-03-13 (Compiler subagent study)

**Implemented Validation:**
- Duplicate IDs ✓
- Undefined references ✓
- Cycle detection ✓
- Arity checking ✓
- Type compatibility ✓
- Property requirements ✓
- Spanish error messages ✓

**Missing Validation:**
1. Set homogeneity validation
2. Stream type consistency
3. Output node requirement (programs must have at least one output)
4. car/food/animal property validation

**Impact:**
- Invalid programs may compile but fail at runtime
- Reduced error message quality
- Could confuse young learners

**Priority:** P1 - HIGH (improves quality, but existing validation catches many errors)

**Estimated Time:** 4 hours

**Files Affected:**
- `packages/compiler/src/validation/dag-validator.ts`

**Acceptance Criteria:**
- Set operations verify all elements are same type
- Stream operations verify correct element types
- Programs without output nodes produce error
- car/food/animal properties validated correctly

---

## PRIORITIZED IMPLEMENTATION PLAN

### Task Prioritization Framework

**P0 - CRITICAL (Fixes show-stopper bugs, blocks core functionality)**
- Temporal operations state bug (breaks semantics)
- Performance tests (quick win, blocks 100% test coverage)

**P1 - HIGH (Required for MVP completion)**
- HTTP API (CV system integration)
- Incremental Runtime (required for WebSocket)
- Missing compiler validation (quality improvement)

**P2 - MEDIUM (Nice to have, improves language or UX)**
- Fraction operations (completes type system)
- Generic set/stream operations (removes natural-only restriction)
- Nested operation expressions (enables complex expressions)
- Child-friendly Spanish messages (better UX for ages 6-9)

**P3 - LOW (Post-MVP, nice to have)**
- WebSocket server (IDE live feedback - can be done later)
- Fix TypeScript type union complexity (nice to have)

---

## PRIORITIZED TASK LIST

### P0 CRITICAL (Immediate - Fixes Show-Stopper Bugs)

#### Task P0.1: Fix Temporal Operations State Bug

**Status:** NOT STARTED - CRITICAL BUG

**Files to update:**
- `packages/runtime/src/operations/temporal.ts` (lines 7, 27, 50, 76)
- May need: `packages/runtime/src/evaluator/demand-driven-evaluator.ts`

**Why Critical:**
- **Breaks core dataflow semantics** - temporal operators don't maintain state
- FBY counter returns 0,0,0,0 instead of 0,1,2,3
- NEXT/ACCUMULATE produce incorrect results
- Violates demand-driven evaluation principle from Lucid spec
- Layer 5 (Temporal) is functionally broken

**Root Cause:**
Generators are recreated on each call using `generatorFactory()`, losing state between evaluations.

**Fix Approach:**
Option A: Cache generator within stream value itself (first call creates it, subsequent use cached)
Option B: Ensure evaluator caches stream values properly (already does - use the cached value)
Option C: Create generator once when DataSource is wrapped, store it in the value

**Estimated Time:** 2 hours

**Dependencies:** None

**Acceptance Criteria:**
- FBY counter produces correct sequence: 0,1,2,3,4... for timesteps 0,1,2,3,4...
- NEXT advances through stream correctly on each call
- ACCUMULATE maintains accumulator across time steps
- All temporal operation tests pass
- Cache hit/miss statistics show proper caching behavior
- Demand-driven semantics preserved (only evaluate when demanded)

**Layer:** Layer 5 (Temporal Operators)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` temporal operators section, `specs/DEMAND_DRIVEN_INCREMENTAL.md`

**Ralph Wiggum Checklist:**
- [ ] Bug fixed, generator state preserved
- [ ] All temporal tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(runtime): preserve generator state in temporal operations"

---

#### Task P0.2: Implement 2 Stubbed Performance Tests

**Status:** NOT STARTED

**Files to update:**
- `packages/tests/src/performance/compilation.test.ts:7`
- `packages/tests/src/performance/execution.test.ts:7`

**Why Critical:**
- Achieve 100% test coverage
- Get performance metrics for compilation and execution
- Verify performance targets are met (<100ms compile, <50ms execute)
- Quick win - can be done immediately

**Can Implement Now:** YES - sufficient operations available

**Estimated Time:** 1.5 hours

**Dependencies:** None

**Acceptance Criteria:**
- Test 7.1: Compilation Performance (100-node program <100ms p95)
- Test 7.2: Execution Performance (50-node program <50ms p95)
- Cache effectiveness measured (cache hit ratio)
- Performance targets documented
- Tests pass

**Layer:** Cross-layer (performance metrics)

**Spec Reference:** `specs/INTEGRATION_TESTS_SPEC.md` lines 422-480

**Ralph Wiggum Checklist:**
- [ ] Performance tests implemented (not stubbed)
- [ ] Performance tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "test(integration): implement performance tests"

---

### P1 HIGH (Required for MVP - Layer 7 & Quality)

#### Task P1.1: Implement HTTP API (Batch Mode)

**Files to create:**
- `packages/http-api/src/server.ts`
- `packages/http-api/src/routes.ts`
- `packages/http-api/src/index.ts`

**Why Critical:**
- **Required by MVP** - CV system integration depends on this
- Enables batch processing of complete programs
- Required integration interface defined in specs
- No alternative way to integrate with CV system

**Estimated Time:** 10 hours

**Dependencies:**
- Compiler (complete)
- Runtime (complete)

**Spec Reference:** `specs/INTEGRATION_SPEC.md` lines 53-198

**Acceptance Criteria:**
- POST /api/v1/compile - Validate program without executing
- POST /api/v1/execute - Compile and run program
- GET /api/v1/health - Health check endpoint
- Accept source code strings
- Return child-friendly Spanish error messages
- Support execution options (maxTimesteps, includeTrace, traceLevel)
- Return execution trace (executionOrder, nodeEvaluations, cacheHits, cacheMisses)
- Performance: <100ms compile, <50ms execute for typical programs
- HTTP tests pass

**Layer:** Layer 7 (Integration)

**Ralph Wiggum Checklist:**
- [ ] HTTP API server implemented
- [ ] All endpoints functional
- [ ] HTTP API tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Performance targets met
- [ ] Git commit with message: "feat(http-api): implement HTTP API server"

---

#### Task P1.2: Implement IncrementalRuntime Class

**File to create:** `packages/runtime/src/incremental-runtime.ts`

**Why Critical:**
- **Required by WebSocket server** - can't implement P3.1 without this
- Enables partial graph evaluation during program construction
- Required by specs/INTEGRATION_SPEC.md and specs/DEMAND_DRIVEN_INCREMENTAL.md
- Key differentiator for IDE live feedback

**Estimated Time:** 7 hours

**Dependencies:** None

**Spec Reference:**
- `specs/INTEGRATION_SPEC.md` lines 365-771 (Incremental Runtime design)
- `specs/DEMAND_DRIVEN_INCREMENTAL.md` (Demand-driven semantics for incremental evaluation)

**Acceptance Criteria:**
- Partial graph evaluation (demand-driven - NOT eager)
- Node state tracking (completed/pending/processing/error)
- Subscription management (multiple clients)
- Graph update API with cache invalidation
- Missing input extraction with Spanish messages
- Demand-driven semantics preserved (only evaluate when demanded)
- Incremental updates 5x faster than full re-evaluation
- Incremental runtime tests pass

**Layer:** Layer 7 (Integration)

**Ralph Wiggum Checklist:**
- [ ] IncrementalRuntime class implemented
- [ ] Demand-driven semantics preserved (no eager evaluation)
- [ ] Incremental runtime tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement IncrementalRuntime class"

---

#### Task P1.3: Implement Missing Compiler Validation

**File to update:** `packages/compiler/src/validation/dag-validator.ts`

**Why Important:**
- Improves quality of error messages for young learners
- Catches more errors at compile time (better UX)
- Programs without output nodes should fail fast
- Set homogeneity prevents runtime type errors

**Estimated Time:** 4 hours

**Dependencies:** None

**Acceptance Criteria:**
- Set homogeneity validation (all elements same type)
- Stream type consistency (operations match element type)
- Output node requirement (programs must have at least one output)
- car/food/animal property validation (required fields present)
- Validation tests pass
- Child-friendly Spanish error messages

**Layer:** Cross-layer (validation quality)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` validation section

**Ralph Wiggum Checklist:**
- [ ] Missing validation rules implemented
- [ ] Validation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): add missing compiler validation rules"

---

### P2 MEDIUM (Nice to have - Completeness & UX)

#### Task P2.1: Implement Fraction Operations

**Status:** NOT STARTED

**Files to update:**
- `packages/shared/src/types.ts` (types already defined)
- `packages/runtime/src/operations/` (create `fraction.ts` or add to `numeric.ts`)

**Why Important:**
- Fraction type is already defined in shared types
- Completes the numeric type system
- Educational value - fractions are important curriculum topic

**Operations Needed:**
- ADD_FRACTIONS
- SUBTRACT_FRACTIONS
- MULTIPLY_FRACTIONS
- DIVIDE_FRACTIONS
- SIMPLIFY_FRACTION
- COMPARE_FRACTIONS

**Estimated Time:** 3 hours

**Dependencies:** None

**Acceptance Criteria:**
- All 6 fraction operations implemented
- Operations work correctly with proper math
- Fraction tests pass
- Typecheck passes

**Layer:** Layer 2 (Arithmetic)

**Ralph Wiggum Checklist:**
- [ ] Fraction operations implemented
- [ ] Fraction tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(runtime): implement fraction operations"

---

#### Task P2.2: Make Set/Stream Operations Generic

**Status:** NOT STARTED

**Files to update:**
- `packages/shared/src/types.ts` (SetType uses `unknown[]`, change to generic `T[]`)
- `packages/runtime/src/operations/sets.ts` (currently only natural)
- `packages/runtime/src/operations/` (stream operations)

**Why Important:**
- Current set/stream operations only work with natural numbers
- Generic types would support any element type
- Aligns with curriculum (sets of shapes, colors, etc.)
- Fix SetType to use proper generic type `T[]` instead of `unknown[]`

**Estimated Time:** 4 hours

**Dependencies:** None

**Acceptance Criteria:**
- Set/Stream types use generic `T[]` instead of `unknown[]`
- Set operations work with any element type (not just natural)
- Stream operations support any element type
- Generic type tests pass
- Typecheck passes

**Layer:** Layer 4 (Sets) and Layer 6 (Streams)

**Ralph Wiggum Checklist:**
- [ ] Generic Set/Stream types implemented
- [ ] Generic operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "refactor(runtime): make set/stream operations generic"

---

#### Task P2.3: Implement Nested Operation Expressions

**Files to update:**
- `packages/compiler/src/parser/dataflow-parser.ts` (line 284-289 - add third alternative to argument rule)
- `packages/compiler/src/ast/ast-builder.ts` (line 163-166 - add handler for ctx.operationExpression)

**Why Important:**
- Grammar spec (GRAMMAR_SPEC.md:196-198) allows `argument ::= identifier | literal | operation`
- Parser argument rule currently only implements identifier and literal
- Missing third alternative: operation
- Enables complex expressions like ADD(ADD(a, b), c)
- Zero impact on existing code, new feature only
- Workaround exists: Use intermediate source/transform nodes

**Estimated Time:** 1.5 hours

**Dependencies:** None

**Acceptance Criteria:**
- Parser argument rule has third alternative (operation)
- AST builder handles ctx.operationExpression
- CST types regenerated
- Nested operation expressions parse correctly
- TypeScript compiles
- Nested operation tests pass

**Layer:** All layers (expression support)

**Spec Reference:** `specs/GRAMMAR_SPEC.md` line 196-198

**Ralph Wiggum Checklist:**
- [ ] Nested operation expressions implemented
- [ ] Nested operation tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(compiler): add nested operation expressions"

---

#### Task P2.4: Add Child-Friendly Spanish Error Messages

**Files to update:**
- All validation files (`packages/compiler/src/validation/*`)
- All runtime error handling files
- HTTP API error responses
- WebSocket error responses

**Why Important:**
- Error messages must be child-friendly for target demographic (ages 6-9)
- Spanish language required by specs
- Improves user experience and learning outcomes
- Better educational value

**Estimated Time:** 3 hours

**Dependencies:**
- P1.1 (HTTP API) - can start before
- P1.2 (IncrementalRuntime) - can start before
- P3.1 (WebSocket Server) - can start after

**Acceptance Criteria:**
- All error messages in child-friendly Spanish
- Missing input messages are age-appropriate
- Type error messages use simple language
- Property error messages explain what's missing
- Multiple missing inputs reported in missingInputs array
- Message tests pass

**Layer:** All layers (user experience)

**Spec Reference:** `specs/INTEGRATION_SPEC.md` examples of Spanish messages

**Ralph Wiggum Checklist:**
- [ ] Child-friendly Spanish messages added
- [ ] Message tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Git commit with message: "feat(messages): add child-friendly Spanish error messages"

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
- P1.2 (IncrementalRuntime) - REQUIRED

**Spec Reference:** `specs/INTEGRATION_SPEC.md` lines 192-361

**Acceptance Criteria:**
- Connection at ws://localhost:3000/live
- Message handlers: validate_program, evaluate_incremental, subscribe_node, unsubscribe_node
- Push messages: validation_result, evaluation_result, node_state_changed, error
- Message protocol with messageId
- Multiple concurrent client support
- Connection resilience (reconnects on disconnect)
- Child-friendly Spanish messages
- Performance: message roundtrip <50ms (p95)
- WebSocket tests pass

**Layer:** Layer 7 (Integration)

**Ralph Wiggum Checklist:**
- [ ] WebSocket server implemented
- [ ] WebSocket tests pass
- [ ] Typecheck passes
- [ ] Previous tests still pass
- [ ] Multiple concurrent clients work
- [ ] Git commit with message: "feat(websocket): implement WebSocket server"

---

#### Task P3.2: Fix TypeScript Type Union Complexity in Type Validation

**File to update:** `packages/compiler/src/validation/dag-validator.ts`

**Why Important:**
- Type validation structure implemented but not effective due to TypeScript warnings
- Complex type unions prevent validation logic from executing correctly
- Can be deferred as validation structure is present, just blocked by TS

**Estimated Time:** 2.5 hours

**Dependencies:** None

**Acceptance Criteria:**
- Type compatibility validation works correctly
- No TypeScript warnings for validation logic
- Integration tests pass with effective type checking

**Layer:** All layers (enables correct validation)

**Spec Reference:** `specs/LANGUAGE_SPEC.md` lines 433-440

**Ralph Wiggum Checklist:**
- [ ] TypeScript type union complexity resolved
- [ ] Type validation tests pass
- [ ] Typecheck passes (no warnings)
- [ ] Previous tests still pass
- [ ] Git commit with message: "fix(validation): resolve TypeScript type union complexity"

---

## SUMMARY TABLE

| Task | Priority | Status | Time | Impact | Dependencies |
|------|----------|--------|------|--------|--------------|
| **P0.1: Fix temporal operations state bug** | P0 | NOT STARTED | 2h | **CRITICAL** - breaks core semantics | None |
| **P0.2: Implement 2 stubbed performance tests** | P0 | NOT STARTED | 1.5h | Test coverage 100% | None |
| **P1.1: Implement HTTP API** | P1 | NOT STARTED | 10h | **MVP** - enables CV system | Compiler, Runtime |
| **P1.2: Implement IncrementalRuntime** | P1 | NOT STARTED | 7h | **MVP** - required for WebSocket | None |
| **P1.3: Implement missing compiler validation** | P1 | NOT STARTED | 4h | Quality improvement | None |
| **P2.1: Implement Fraction operations** | P2 | NOT STARTED | 3h | Completes type system | None |
| **P2.2: Make Set/Stream operations generic** | P2 | NOT STARTED | 4h | Removes natural-only restriction | None |
| **P2.3: Implement nested operation expressions** | P2 | NOT STARTED | 1.5h | Enables complex expressions | None |
| **P2.4: Add child-friendly Spanish error messages** | P2 | NOT STARTED | 3h | Better UX for ages 6-9 | Layer 7 |
| **P3.1: Implement WebSocket Server** | P3 | NOT STARTED | 12h | IDE live feedback | P1.2 |
| **P3.2: Fix TypeScript type union complexity** | P3 | NOT STARTED | 2.5h | Effective validation | None |

**Total Estimated Time:** 50.5 hours

**Previous Work Completed:**
- ✓ Layers 1-6: COMPLETE (Foundation, Arithmetic, Curriculum Types, Sets, Temporal, Streams)
- ✓ 24 operations implemented (numeric, comparison, filtering, sets, temporal, ordering, boolean)
- ✓ 1 stream generator (counter)
- ✓ All types defined (primitives, curriculum, composite, validation, program)
- ✓ Lexer: COMPLETE - all tokens defined
- ✓ Parser: COMPLETE - all grammar rules implemented
- ✓ AST Builder: COMPLETE - full CST to AST conversion
- ✓ Validation: MOSTLY COMPLETE (missing: set homogeneity, stream consistency, output node, car/food/animal props)
- ✓ Compiler: COMPLETE - full pipeline working
- ✓ Main Runtime: COMPLETE - simple wrapper
- ✓ Demand-Driven Evaluator: CORRECT except temporal ops (P0.1 bug)
- ✓ Graph Implementation: CORRECT but has issues (SORT only numbers, ALPHABETICAL_SORT converts to string)
- ✓ 30 tests passing (100%)

**MVP Milestone:**
To achieve MVP, complete: P0.1, P0.2, P1.1

**MVP Definition (from PROJECT_GOALS.md):**
- Layers 1-4 implemented (Natural, arithmetic, curriculum types, sets) ✓ COMPLETE
- Compiler validates programs, catches errors at compile-time ✓ COMPLETE
- Runtime executes programs correctly with demand-driven semantics ⚠️ PARTIAL (P0.1 bug)
- 10-15 operations working ✓ COMPLETE (24 implemented)
- JSON input/output well-defined ✓ COMPLETE
- Basic HTTP API for batch mode ❌ NOT STARTED (P1.1)
- Handles 5 concurrent users without degradation ❌ NOT TESTED

**To Achieve MVP:** Complete P0.1, P0.2, P1.1

---

## NEXT STEPS

Focus on completing MVP - this requires fixing critical bugs and implementing Layer 7 integration.

**Order of Implementation:**

1. **P0.1 (2h): Fix Temporal Operations State Bug**
   - Preserve generator state across calls
   - Fix NEXT, FIRST, FBY, ACCUMULATE operations
   - Impact: Restore correct dataflow semantics
   - **CRITICAL - Do not skip this**

2. **P0.2 (1.5h): Implement 2 Stubbed Performance Tests**
   - Test 7.1: Compilation Performance (100-node program <100ms)
   - Test 7.2: Execution Performance (50-node program <50ms)
   - Impact: Achieve 100% test coverage, verify performance targets

3. **P1.1 (10h): Implement HTTP API (Batch Mode)**
   - POST /api/v1/compile - Validate program without executing
   - POST /api/v1/execute - Compile and run program
   - GET /api/v1/health - Health check endpoint
   - Return child-friendly Spanish error messages
   - Impact: Enables CV system integration for complete programs

4. **P1.2 (7h): Implement IncrementalRuntime Class**
   - Partial graph evaluation with demand-driven semantics
   - Node state tracking (completed/pending/error)
   - Subscription management for multiple clients
   - Graph update API with cache invalidation
   - Spanish child-friendly messages for missing inputs
   - Impact: Enables WebSocket server for IDE live feedback

5. **P1.3 (4h): Implement Missing Compiler Validation**
   - Set homogeneity validation
   - Stream type consistency
   - Output node requirement
   - car/food/animal property validation

6. **P2.1 (3h): Implement Fraction Operations**
   - Complete the numeric type system
   - Educational value for curriculum

7. **P2.2 (4h): Make Set/Stream Operations Generic**
   - Remove natural-only restriction
   - Support any element type

8. **P2.3 (1.5h): Implement Nested Operation Expressions**
   - Add third alternative to parser argument rule
   - Add AST builder handler for operation expressions
   - Enables complex expressions like ADD(ADD(a, b), c)

9. **P2.4 (3h): Add Child-Friendly Spanish Error Messages**
   - Update all validation and runtime error handling
   - Ensure messages are age-appropriate for 6-9 year olds

10. **P3.1 (12h): Implement WebSocket Server (Live Mode)**
    - Connection at ws://localhost:3000/live
    - Message handlers for validate_program, evaluate_incremental, subscribe_node, unsubscribe_node
    - Push notifications for node state changes
    - Multiple concurrent client support
    - Impact: Enables IDE live feedback during program construction

11. **P3.2 (2.5h): Fix TypeScript type union complexity in type validation**
    - Simplify type checking logic in dag-validator
    - Ensure effective type validation without warnings

---

## PERFORMANCE TARGETS

### Current Status
- Compilation: ~50ms for <50 nodes (on par with target)
- Execution: ~2ms for simple programs (exceeds target)
- Test suite: 30/30 tests passing (100%) ✓
  - 2/2 performance tests stubbed (Task P0.2)

### Targets
- Compilation: <100ms for programs with <100 nodes (p95)
- Execution: <50ms for programs with <100 nodes (p95)
- Concurrent: 5 simultaneous requests without degradation

---

## SUCCESS METRICS

### MVP (Layers 1-4 + Basic Integration)
- All 24 operations implemented ✓
- Type compatibility validation working ✓
- HTTP API functional (pending - P1.1)
- Compiler validates programs correctly ✓
- Runtime executes programs correctly with demand-driven semantics ⚠️ (P0.1 bug)
- All module resolution issues fixed ✓ (30/30 tests passing)
- Handles 5 concurrent users (pending)

### Version 1.0 (All Layers)
- All layers 1-7 implemented
- WebSocket server for live feedback
- IncrementalRuntime for partial evaluation
- Complete test coverage (>80%)
- Child-friendly Spanish messages throughout

---

**Document Status:** Active implementation plan
**Next Review:** After implementing P0.1 (temporal operations bug fix)
**Maintainer:** Update as implementation progresses
**Last Change:** 2026-03-13 - Synthesized findings from 5 parallel subagents, corrected Layer 7 status from 35% to 0%, added P0.1 critical bug for temporal operations, reprioritized tasks for MVP completion
