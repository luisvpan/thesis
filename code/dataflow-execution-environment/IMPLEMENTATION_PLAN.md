# IMPLEMENTATION PLAN
## Dataflow Language Compiler & Runtime for Educational Programming (Ages 6-9)

**Document Status:** Living implementation plan - update as implementation reveals better designs
**Created:** 2026-02-26
**Based On:** Complete specifications in specs/ directory
**Last Updated:** 2026-03-04

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

### Overall Progress: ~40% Complete

| Layer | Status | Completion | Tests |
|-------|--------|------------|--------|
| Layer 1: Foundation | PARTIAL | 85% | 0% |
| Layer 2: Arithmetic | PARTIAL | 75% | 0% |
| Layer 3: Curriculum Types | MISSING | 10% | 0% |
| Layer 4: Set Operations | PARTIAL | 15% | 0% |
| Layer 5: Temporal Operators | PARTIAL | 15% | 0% |
| Layer 6: Streams | MISSING | 5% | 0% |
| Layer 7: Integration | MISSING | 0% | 0% |

### ✅ COMPLETE (Core Infrastructure)

**Monorepo & Build System:**
- ✓ Bun workspaces configured
- ✓ TypeScript strict mode
- ✓ All packages build individually
- ✓ `bun tsc --noEmit` works for typecheck
- ⚠️ Build script removed - use commands directly

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

**Architecture Issues (High Priority):**
- ❌ Compiler.compile() takes DataflowProgram instead of source code string
- ❌ Runtime.loadProgram() takes DataflowProgram instead of parsed Program
- ❌ No separation between source parsing and program representation
- ❌ DataflowProgram should only be used by integrations (HTTP API, WebSocket, etc.)

**Layer 1 (Foundation):**
- ❌ HTTP API (packages/http-api/) - Package exists, directory empty
- ❌ WebSocket Server (packages/websocket-server/) - Package exists, directory empty
- ❌ Any test files - Zero tests across all packages

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
- 10 operations in registry but NOT implemented (will crash at runtime):
  - FILTER, UNION, INTERSECTION, DIFFERENCE, COMPLEMENT
  - NEXT, FIRST, FBY, ACCUMULATE, SORT
- Runtime evaluator throws "Unknown operation" error for these

**Validation Gap:**
- Arity checking ✓ implemented
- Type compatibility ❌ missing
- Property constraints ❌ missing

### 🐛 BUGS & ISSUES

1. **Build Script Removed** (fixed per user instructions)
   - Use `bun run build --filter='*'` directly
   - Use `bun tsc --noEmit` for typecheck directly

2. **Compiler Entry Point Incorrect** (packages/compiler/src/compiler.ts:22)
   - `compile(program: DataflowProgram)` should be `compile(source: string)`
   - Should call `parse(source)` internally
   - Should return validation result + DataflowProgram

3. **Runtime Entry Point Incorrect** (packages/runtime/src/runtime.ts:14)
   - `loadProgram(program: DataflowProgram)` should accept parsed Program
   - Should NOT use DataflowProgram internally
   - DataflowProgram is for integrations only

4. **COMPARE Type Mismatch** (packages/runtime/src/operations/numeric.ts:67)
   - Returns Natural type, spec says Integer type
   - Fix needed for correctness

5. **Lint Not Configured** (packages/package.json)
   - Script just echoes "Lint not configured"
   - Need to choose and configure linter (ESLint, Biome, etc.)

### 📊 Test Coverage Summary

| Package | Test Files | Tests Passing |
|---------|------------|---------------|
| @dataflow/shared | 0 | N/A |
| @dataflow/compiler | 0 | N/A |
| @dataflow/runtime | 0 | N/A |
| @dataflow/http-api | 0 | N/A |
| @dataflow/websocket-server | 0 | N/A |
| **TOTAL** | **0** | **0%** |

**Estimated Test Count Needed:** ~270 tests

---

## PHASE 0: Critical Architecture Refactoring (Week 0.5)

**Goal:** Fix fundamental architecture issues before building features

### Priority -1: Compiler Entry Point Refactoring (1 day)

**Current Issue:**
```typescript
// packages/compiler/src/compiler.ts
compile(program: DataflowProgram): ValidationResult & { graph?: { nodes: DataflowNode[]; edges: DataflowEdge[] } } {
  const { graph } = program;
  // ...
}
```

**Problem:** Compiler is taking parsed JSON (DataflowProgram) instead of source code. This is backwards.

**Required Changes:**

1. **Update Compiler.compile() signature** (packages/compiler/src/compiler.ts):
```typescript
compile(source: string): ValidationResult & { program?: DataflowProgram } {
  // 1. Parse source code
  const ast = this.parse(source);

  // 2. Build internal Program representation
  const program = this.buildProgram(ast);

  // 3. Validate
  const validationResult = this.validate(program);

  if (!validationResult.success) {
    return validationResult;
  }

  // 4. Return DataflowProgram for integrations
  return {
    success: true,
    errors: [],
    warnings: [],
    program: this.toDataflowProgram(program)
  };
}
```

2. **Update Runtime.loadProgram() signature** (packages/runtime/src/runtime.ts):
```typescript
loadProgram(program: { nodes: DataflowNode[]; edges: DataflowEdge[] }): void {
  // Accept the graph structure from Compiler
  // Keep using DataflowNode internally for graph
  // But note: DataflowProgram includes metadata (for integrations)

  this.graph = new DataflowGraph();
  this.evaluator = new DemandDrivenEvaluator();

  for (const node of program.nodes) {
    this.graph.addNode(node);
  }

  for (const edge of program.edges) {
    this.graph.addEdge(edge);
  }
}
```

3. **Add internal Program type** (if needed):
```typescript
// packages/compiler/src/types/program.ts
// Internal compiler representation (not exported)
type CompilerProgram = {
  nodes: InternalNode[];
  edges: InternalEdge[];
};
```

4. **Update Integration Points**:
- HTTP API should call `compiler.compile(source)` with string
- WebSocket should call `compiler.compile(source)` with string
- Both should receive `DataflowProgram` as output

**Success Criteria:**
- Compiler.compile("source a: natural = 3;") works
- Returns DataflowProgram for integrations
- Runtime can load the program graph
- No references to DataflowProgram in core compiler logic

---

### Priority -2: Update Build Commands (0.5 day)

**Current Issue:** Build script references outdated

**Required Changes:**

1. **Update AGENTS.md build commands** (already done by user):
```bash
# Build all packages
bun run build --filter='*'

# Build specific package
bun run build --filter './packages/{package-name}'
```

2. **Update IMPLEMENTATION_PLAN.md references**:
- Remove references to `bun run build` script in root package.json
- Update to use commands directly
- Update typecheck reference to `bun tsc --noEmit`

3. **Create package-level build scripts** (if needed):
- Each package already has its own build script
- No root-level build script needed

**Success Criteria:**
- AGENTS.md shows correct build commands
- No references to missing root build script
- Typecheck works with `bun tsc --noEmit`

---

## PHASE 1: Foundation & Testing Infrastructure (Week 1)

**Goal:** Unblock development workflow and verify Layer 1 functionality

### Priority 0: Test Framework Setup (0.5 day)

**Tasks:**
- [ ] Create test directory structure for all packages
- [ ] Add basic "sanity check" test for each package
- [ ] Verify `bun test` runs successfully
- [ ] Configure test filtering for layer-based testing

**Success Criteria:**
- `bun test` runs and finds tests
- At least 1 test passes per package
- `bun test -- layer1` filtering works

---

### Priority 1: Layer 1 Tests (2 days)

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

**Success Criteria:**
- All Layer 1 features have test coverage
- Tests verify child-friendly error messages
- Benchmarks pass performance targets
- Ralph Wiggum checklist completed for Layer 1

**Impact:** Confirms Layer 1 is production-ready

---

### Priority 2: Layer 1 HTTP API (1.5 days)

**Goal:** Enable CV system integration and basic execution

**Tasks:**
- [ ] Create `packages/http-api/src/server.ts` with Elysia setup
- [ ] Implement POST /api/v1/compile endpoint
- [ ] Implement POST /api/v1/execute endpoint
- [ ] Implement GET /api/v1/health endpoint
- [ ] Add child-friendly Spanish error responses
- [ ] Write tests for all endpoints

**Success Criteria:**
- Can compile a program via HTTP POST (with source code string)
- Can execute "3 + 2 = 5" via HTTP POST
- Returns proper validation errors for invalid programs
- All endpoints have tests passing

**Impact:** Enables Layer 1 to be fully functional end-to-end

---

## PHASE 2: Complete Operation Implementations (Week 2)

**Goal:** Make all registered operations work

### Priority 3: Complete Operation Implementations (3 days)

**Current status:** 10 operations crash at runtime

**Tasks:**
- [ ] Implement FILTER operation (packages/runtime/src/operations/filtering.ts)
- [ ] Implement UNION operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement INTERSECTION operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement DIFFERENCE operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement COMPLEMENT operation (packages/runtime/src/operations/sets.ts)
- [ ] Implement NEXT operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement FIRST operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement FBY operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement ACCUMULATE operation (packages/runtime/src/operations/temporal.ts)
- [ ] Implement SORT operation (packages/runtime/src/operations/ordering.ts)
- [ ] Update evaluator to handle all operations (demand-driven-evaluator.ts)
- [ ] Export new operations from index.ts
- [ ] Write tests for all new operations

**Success Criteria:**
- All 15 operations in registry work without crashes
- All operations have tests
- Evaluator dispatches to correct operation

**Impact:** All basic operations usable in programs

---

### Priority 4: Layer 2 Tests & Bug Fixes (1 day)

**Goal:** Verify arithmetic operations

**Tasks:**
- [ ] Tests for SUBTRACT (including negative results)
- [ ] Tests for MULTIPLY
- [ ] Tests for DIVIDE (including division by zero)
- [ ] Tests for COMPARE (-1, 0, 1 results)
- [ ] Fix COMPARE return type (Integer vs Natural bug)
- [ ] Performance benchmarks for arithmetic operations

**Success Criteria:**
- Layer 2 tests passing
- COMPARE returns correct type (Integer, not Natural)
- No regressions in Layer 1

**Impact:** Confirms Layer 2 complete

---

## PHASE 3: Curriculum Types (Week 3)

**Goal:** Enable educational curriculum features

### Priority 5: Curriculum Type Operations (3 days)

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

**Success Criteria:**
- Curriculum types work with comparison/filtering
- Property constraints validated at compile-time
- Child-friendly Spanish errors for property mismatches

**Impact:** Core educational functionality working

---

## PHASE 4: Integration Layer (Week 4)

**Goal:** Enable live construction mode

### Priority 6: WebSocket Server (2 days)

**Tasks:**
- [ ] Create `packages/websocket-server/src/server.ts`
- [ ] Implement ws://localhost:3000/live endpoint
- [ ] Handle validate_program messages
- [ ] Handle evaluate_incremental messages
- [ ] Handle subscribe_node/unsubscribe_node messages
- [ ] Implement push notifications (node_state_changed)
- [ ] Add connection resilience (reconnects)
- [ ] Write tests for WebSocket protocol

**Success Criteria:**
- IDE can connect via WebSocket
- Validation returns child-friendly Spanish errors
- Incremental evaluation works for partial programs
- Multiple clients can connect

**Impact:** Live feedback during program construction

---

### Priority 7: Incremental Runtime (2 days)

**Tasks:**
- [ ] Create IncrementalRuntime class (packages/runtime/src/incremental-runtime.ts)
- [ ] Implement partial graph evaluation
- [ ] Implement node state tracking (completed/pending/error)
- [ ] Implement subscription management
- [ ] Implement graph update API with cache invalidation
- [ ] Add missing input extraction with multiple inputs support
- [ ] Add child-friendly Spanish missing input messages
- [ ] Write tests for incremental evaluation

**Success Criteria:**
- Partial graphs evaluate correctly
- Pending states show all missing inputs with Spanish messages
- Cache invalidation works on graph updates
- Incremental update 5x faster than full re-eval

**Impact:** Enables real-time AR feedback as children build

---

### Priority 8: Integration Tests (1 day)

**Tasks:**
- [ ] End-to-end HTTP API tests
- [ ] End-to-end WebSocket tests
- [ ] Performance benchmarks for integration layer
- [ ] Concurrent request handling tests

**Success Criteria:**
- All integration points tested
- Performance targets met
- Handles 5 concurrent requests/connections

---

## PHASE 5: Advanced Features (Week 5-6)

**Goal:** Complete remaining layers

### Priority 9: Temporal Operators (2 days)

**Tasks:**
- [ ] Update evaluator to handle time parameter properly
- [ ] Implement stream value evaluation
- [ ] Add stream generator support
- [ ] Write tests for temporal operations (FBY counter 0,1,2,3,4)

**Success Criteria:**
- Temporal operators work correctly
- Cache indexed by time
- FBY produces correct sequence

---

### Priority 10: Additional Curriculum Types (2 days)

**Tasks:**
- [ ] Implement Food operations (COMPARE_BY_TASTE, FILTER_BY_TASTE)
- [ ] Implement Animal operations (COMPARE_BY_TYPE, FILTER_BY_TYPE)
- [ ] Implement Person operations (COMPARE_BY_AGE_GROUP, COMPARE_BY_GENDER, etc.)
- [ ] Write tests for all additional types

---

### Priority 11: Lint Configuration (0.5 day)

**Tasks:**
- [ ] Choose linter (ESLint, Biome, or Bun's built-in)
- [ ] Configure lint rules
- [ ] Update AGENTS.md with lint command
- [ ] Fix existing lint issues

---

### Priority 12: Documentation (0.5 day)

**Tasks:**
- [ ] Package-level README files
- [ ] API documentation (OpenAPI spec for HTTP)
- [ ] WebSocket protocol documentation
- [ ] Developer documentation (setup, contribution)

---

## BLOCKERS & RISKS

### Critical Blockers (Must Fix First)

1. ⚠️ **Compiler Entry Point Wrong** - Blocks all integration work
   - `compile()` takes DataflowProgram instead of source string
   - Must refactor before HTTP API can work

2. ⚠️ **Runtime Entry Point Wrong** - Blocks execution
   - `loadProgram()` takes DataflowProgram
   - Should accept parsed program structure

3. ⚠️ **No Test Framework** - Cannot verify functionality
   - Zero tests written
   - No way to catch regressions

4. ⚠️ **Missing Operation Implementations** - 10 operations crash
   - FILTER, UNION, INTERSECTION, DIFFERENCE, COMPLEMENT
   - NEXT, FIRST, FBY, ACCUMULATE, SORT

### Technical Risks

1. **Temporal operator complexity** - demand-driven semantics with time
2. **Cache invalidation in incremental mode** - must be correct
3. **Type system integration** - property constraints, type compatibility

### Resource Risks

1. **~270 tests needed** - significant test writing effort
2. **Child-friendly Spanish messages** - requires translation expertise

---

## SUCCESS METRICS FOR MVP

### Layer 1 Complete:
- ✅ Architecture refactored (compiler takes source string)
- ✅ All tests pass
- ✅ "3 + 2 = 5" works end-to-end
- ✅ HTTP API functional

### Layer 1-4 Complete (MVP):
- ✅ All arithmetic operations work
- ✅ All set operations work
- ✅ Curriculum types work
- ✅ Type checking catches errors
- ✅ Child-friendly Spanish messages throughout

### Layer 1-7 Complete (Version 1.0):
- ✅ Temporal operators work
- ✅ Integration layer complete
- ✅ Live construction mode works
- ✅ All 15+ operations implemented
- ✅ 80%+ test coverage

---

## Appendix: Complete Feature Checklist

### Infrastructure (Critical)
- [x] Monorepo structure
- [x] Shared type system
- [x] JSON input schema (DataflowProgram for integrations)
- [ ] **Compiler.compile(source: string) refactoring**
- [ ] **Runtime.loadProgram() refactoring**
- [ ] Test framework setup
- [ ] Lint configuration

### Layer 1
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
- [x] COMPARE operation (BUG: wrong return type)
- [x] Operation registry expansion
- [x] Arity checker
- [ ] All Layer 2 tests (0 TESTS WRITTEN)
- [ ] Fix COMPARE return type bug

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

---

**Document Status:** Analysis Complete, Ready for Implementation
**Next Step:** Priority -1 - Compiler Entry Point Refactoring
**Estimated Time to Layer 1 Complete:** 4 days (after refactoring)
**Estimated Time to MVP (Layers 1-4):** 12 days
