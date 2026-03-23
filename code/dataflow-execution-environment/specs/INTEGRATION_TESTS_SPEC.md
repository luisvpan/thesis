# Integration Tests Specification
## End-to-End Testing for Compiler and Runtime

**Document Status:** Living specification - update as implementation reveals edge cases
**Created:** 2026-03-05
**Based On:** Complete specifications in specs/ directory and implementation plan

---

## Design Philosophy

### Purpose of Integration Tests

Integration tests verify that the **compiler** and **runtime** work correctly together when used in end-to-end scenarios. While unit tests verify individual components in isolation, integration tests validate:

1. **Correct Dataflow**: Source code → Compiler → Runtime → Results
2. **Complex Programs**: Multi-layer programs that span arithmetic, sets, temporal operations
3. **Error Propagation**: Validation errors caught at compile time, runtime errors handled gracefully
4. **Performance**: Full pipeline performance (compilation + execution)
5. **Type Safety**: Type checking prevents invalid programs from running

### Educational Rationale

Integration tests are critical for the target audience (ages 6-9) because:

- **Complex Programs**: Children build multi-node programs with multiple operations
- **Immediate Feedback**: Errors must be caught early (compile time) with child-friendly messages
- **Confidence**: Teachers need assurance that the system works reliably
- **Curriculum Alignment**: Tests must verify educational scenarios work end-to-end

---

## Test Categories

### 1. End-to-End Pipeline Tests

**Purpose**: Verify the complete flow from source code to execution results

#### Test 1.1: Simple Addition

**Source Code:**
```dataflow
source a: natural = 3;
source b: natural = 2;
transform sum: natural = ADD(a, b);
output result: natural = sum;
```

**Expected Outcome:**
- Compilation succeeds (success: true)
- Execution returns `[5]` (one output node)
- No errors or warnings
- Execution time <10ms

**Acceptance Criteria:**
- ✓ Compiler parses source code correctly
- ✓ Compiler validates program (no cycles, correct types)
- ✓ Runtime loads compiled program successfully
- ✓ Runtime executes with correct demand-driven semantics
- ✓ Output matches expected result

**Educational Value**: First program children build - must work reliably

---

#### Test 1.2: Complex Arithmetic Expression

**Source Code:**
```dataflow
source a: natural = 3;
source b: natural = 2;
source c: natural = 10;
source d: natural = 6;

transform sum: natural = ADD(a, b);
transform difference: integer = SUBTRACT(c, d);
transform product: natural = MULTIPLY(sum, difference);

output result: natural = product;
```

**Expected Outcome:**
- Compilation succeeds
- Execution returns `[20]` (5 * 4)
- No errors
- Execution trace shows: a, b, c, d, sum, difference, product evaluated in correct order
- Cache hits = 0, cache misses = 7 (all nodes evaluated once)

**Acceptance Criteria:**
- ✓ Multi-node program compiles and executes
- ✓ Intermediate results (sum = 5, difference = 4) are correct
- ✓ Final result (product = 20) is correct
- ✓ Execution order follows dataflow dependencies
- ✓ Cache statistics are accurate

**Educational Value**: Demonstrates multi-step calculations

---

### 2. Type Validation Tests

**Purpose**: Verify type checking catches invalid programs before execution

#### Test 2.1: Type Mismatch Detection

**Source Code:**
```dataflow
source a: natural = 5;
source b: text = "hello";
transform sum: natural = ADD(a, b);

output result: natural = sum;
```

**Expected Outcome:**
- Compilation fails with TYPE_ERROR
- Error includes:
  - `code: "TYPE_ERROR"`
  - `nodeId: "sum"`
  - `message: "ADD requires natural inputs, received text"`
  - `childMessage: "⚠️ ¡Ups! El bloque ADD necesita números, no palabras."`
  - `suggestion: "Conecta bloques de números a ADD."`
  - `example: `[número 5] + [número 3] = [número 8] ✅"`
- Runtime is NOT called (compilation stops before execution)

**Acceptance Criteria:**
- ✓ Type mismatch detected at compile time
- ✓ Child-friendly Spanish error message provided
- ✓ Actionable suggestion included
- ✓ Example shows correct usage
- ✓ Execution is prevented (runtime never called)

**Educational Value**: Children need clear error messages when mixing types

---

#### Test 2.2: Property Requirement Validation

**Source Code:**
```dataflow
source numbers: set<natural> = {1, 2, 3};
source color: text = "red";
transform filtered: set<natural> = FILTER_BY_COLOR(numbers, color);

output result: set<natural> = filtered;
```

**Expected Outcome:**
- Compilation fails with TYPE_ERROR
- Error message indicates `FILTER_BY_COLOR requires elements with 'color' property`
- Child-friendly Spanish message: "⚠️ ¡Ups! Este bloque necesita elementos con color."

**Acceptance Criteria:**
- ✓ Property requirements validated at compile time
- ✓ Clear error message indicates what's missing
- ✓ Spanish message is age-appropriate

**Educational Value**: Guides children to use correct types

---

### 3. Curriculum Type Tests

**Purpose**: Verify educational types (Shape, Car, Food, Animal, Person) work end-to-end

#### Test 3.1: Filter Red Shapes

**Source Code:**
```dataflow
source shapes: set<shape> = {
  {type: "circle", size: "large", color: "red"},
  {type: "triangle", size: "small", color: "blue"},
  {type: "square", size: "medium", color: "red"},
  {type: "circle", size: "large", color: "yellow"}
};

source target_color: text = "red";

transform red_shapes: set<shape> = FILTER_BY_COLOR(shapes, target_color);

output result: set<shape> = red_shapes;
```

**Expected Outcome:**
- Compilation succeeds
- Execution returns set with 2 shapes (red circle, red square)
- Original shapes set unmodified (immutability)
- Execution time <10ms

**Acceptance Criteria:**
- ✓ Curriculum types compile and execute
- ✓ Filter operation works correctly
- ✓ Immutability preserved
- ✓ Result matches expected output

**Educational Value**: Core activity for ages 6-9 - filtering by color

---

#### Test 3.2: Compare Shapes by Size

**Source Code:**
```dataflow
source shape1: shape = {type: "circle", size: "small", color: "red"};
source shape2: shape = {type: "square", size: "large", color: "blue"};

transform equal_size: boolean = COMPARE_BY_SIZE(shape1, shape2);

output result: boolean = equal_size;
```

**Expected Outcome:**
- Compilation succeeds
- Execution returns `{kind: "boolean", value: false}` (small ≠ large)
- No errors

**Acceptance Criteria:**
- ✓ Comparison operations execute correctly
- ✓ Boolean type returned
- ✓ Value-based comparison (not reference)

**Educational Value**: Teaching comparison concepts

---

### 4. Set Operation Tests

**Purpose**: Verify set operations (UNION, INTERSECTION, DIFFERENCE, COMPLEMENT, SORT)

#### Test 4.1: Union of Shape Sets

**Source Code:**
```dataflow
source pets: set<animal> = {
  {type: "dog", color: "brown"},
  {type: "cat", color: "white"}
};

source zoo_animals: set<animal> = {
  {type: "bird", color: "blue"},
  {type: "fish", color: "orange"}
};

transform all_animals: set<animal> = UNION(pets, zoo_animals);

output result: set<animal> = all_animals;
```

**Expected Outcome:**
- Compilation succeeds
- Execution returns set with 4 unique animals
- Order: May vary (sets are unordered)
- Duplicates removed (if any overlap)

**Acceptance Criteria:**
- ✓ UNION operation merges sets correctly
- ✓ Duplicates removed
- ✓ All elements from both sets included
- ✓ Type preserved (set<animal>)

**Educational Value**: Teaching set union concept

---

#### Test 4.2: Filter and Sort Combined

**Source Code:**
```dataflow
source shapes: set<shape> = {
  {type: "circle", size: "small", color: "red"},
  {type: "square", size: "large", color: "blue"},
  {type: "triangle", size: "large", color: "yellow"},
  {type: "circle", size: "medium", color: "green"}
};

source target_size: text = "large";

transform large_shapes: set<shape> = FILTER_BY_SIZE(shapes, target_size);
transform sorted_shapes: set<shape> = SORT(large_shapes);

output result: set<shape> = sorted_shapes;
```

**Expected Outcome:**
- Compilation succeeds
- First filter returns 2 shapes (square, triangle - both "large")
- Sort orders by natural order or comparison
- Final result is sorted, filtered set

**Acceptance Criteria:**
- ✓ Chained operations work (filter → sort)
- ✓ Intermediate results correct
- ✓ Final result correct
- ✓ Order deterministic

**Educational Value**: Combining operations is key learning objective

---

### 5. Temporal Operation Tests

**Purpose**: Verify temporal operators (NEXT, FIRST, FBY, ACCUMULATE) work correctly

#### Test 5.1: FBY Counter (0, 1, 2, 3, 4...)

**Source Code:**
```dataflow
source zero: natural = 0;
source one: natural = 1;

transform counter: stream<natural> = FBY(zero, ADD(counter, one));

output result: stream<natural> = counter;
```

**Expected Outcome:**
- Compilation succeeds
- Execution at timestep 0: returns 0
- Execution at timestep 1: returns 1
- Execution at timestep 2: returns 2
- Execution at timestep 3: returns 3
- Execution at timestep 4: returns 4
- Demand-driven evaluation (only requested timesteps evaluated)

**Acceptance Criteria:**
- ✓ FBY returns initial at time 0
- ✓ FBY returns subsequent stream value at time > 0
- ✓ Demand-driven semantics (no eager evaluation)
- ✓ Pure function of time (no mutable state)

**Educational Value**: Core temporal concept - "followed by" semantics

---

#### Test 5.2: FIRST from Stream

**Source Code:**
```dataflow
source numbers: stream<natural> = stream<natural>(generator(counter));

transform first_number: natural = FIRST(numbers);

output result: natural = first_number;
```

**Expected Outcome:**
- Compilation succeeds
- Execution returns first value from stream (depends on generator)
- No errors

**Acceptance Criteria:**
- ✓ FIRST operation extracts first value
- ✓ Stream types work correctly
- ✓ Type preserved

**Educational Value**: Teaching stream concepts

---

### 6. Error Handling Tests

**Purpose**: Verify errors are handled gracefully with child-friendly messages

#### Test 6.1: Division by Zero

**Source Code:**
```dataflow
source a: natural = 10;
source b: natural = 0;

transform quotient: decimal = DIVIDE(a, b);

output result: decimal = quotient;
```

**Expected Outcome:**
- Compilation succeeds (type checking passes)
- Execution fails at runtime with error
- Error message: "Division by zero"
- Child-friendly Spanish: "⚠️ ¡Ups! No puedes dividir por cero."
- Error includes node information

**Acceptance Criteria:**
- ✓ Runtime error caught gracefully
- ✓ Child-friendly Spanish error provided
- ✓ Error indicates problem location
- ✓ System doesn't crash

**Educational Value**: Common mistake for children - need clear feedback

---

#### Test 6.2: Cycle Detection

**Source Code:**
```dataflow
source a: natural = 5;
transform b: natural = ADD(a, c);
transform c: natural = MULTIPLY(b, 2);

output result: natural = c;
```

**Expected Outcome:**
- Compilation fails with CYCLE_DETECTED
- Error message: "Graph contains a cycle"
- Child-friendly Spanish: "⚠️ ¡Ups! Hay un ciclo en el programa. Los bloques se conectan en círculo y no se puede calcular."
- Suggestion: "Busca dónde un bloque apunta a sí mismo y desconecta una línea."
- Example: "[A] → [B] → [C] ❌\n[A] → [B] ✅ (rompe el ciclo)"
- Runtime is NOT called

**Acceptance Criteria:**
- ✓ Cycle detected at compile time
- ✓ Clear cycle visualization in error
- ✓ Actionable suggestion provided
- ✓ Spanish message age-appropriate
- ✓ Execution prevented

**Educational Value**: Critical for preventing infinite loops

---

### 7. Performance Tests

**Purpose**: Verify performance targets are met for educational context

#### Test 7.1: Compilation Performance

**Test Program:**
- 100 nodes (50 DataSource, 49 Transformation, 1 Output)
- Complex dependencies (depth 10)
- Multiple operation types

**Expected Outcome:**
- Compilation <100ms (p95)
- No memory leaks
- Compilation time scales linearly with program size

**Acceptance Criteria:**
- ✓ Performance target met
- ✓ No excessive memory usage
- ✓ Scalable to larger programs

**Educational Value**: Classroom use requires responsive performance

---

#### Test 7.2: Execution Performance

**Test Program:**
- 50 nodes (25 DataSource, 24 Transformation, 1 Output)
- Depth 5
- Mixed operations (arithmetic, comparison, sets)

**Expected Outcome:**
- Execution <50ms (p95)
- Cache effectiveness: ~30% hit rate for repeated evaluations
- Demand-driven semantics verified (only demanded nodes evaluated)

**Acceptance Criteria:**
- ✓ Performance target met
- ✓ Cache effective
- ✓ Demand-driven behavior confirmed

**Educational Value**: Real-time feedback requires fast execution

---

### 8. Demand-Driven Semantics Tests

**Purpose**: Verify correct demand-driven evaluation (not data-driven)

#### Test 8.1: No Output Nodes

**Source Code:**
```dataflow
source a: natural = 5;
source b: natural = 3;
transform sum: natural = ADD(a, b);
```

**Expected Outcome:**
- Compilation succeeds (no syntax/type errors)
- Execution succeeds but returns `[]` (empty outputs)
- No nodes evaluated (correct demand-driven behavior)
- Cache empty (0 hits, 0 misses)

**Acceptance Criteria:**
- ✓ No Output nodes = nothing evaluated
- ✓ Demand-driven semantics confirmed
- ✓ No wasted computation
- ✓ System doesn't hang or crash

**Educational Value**: Validates core language semantics

---

#### Test 8.2: Unreferenced Nodes

**Source Code:**
```dataflow
source a: natural = 5;
source b: natural = 3;
source c: natural = 10;
transform sum: natural = ADD(a, b);
transform unused: natural = MULTIPLY(c, 2);

output result: natural = sum;
```

**Expected Outcome:**
- Compilation succeeds
- Execution returns `[8]` (result from sum)
- Node `c` and `unused` NOT evaluated (not in dependency chain)
- Cache has 3 entries (a, b, sum)

**Acceptance Criteria:**
- ✓ Only referenced nodes evaluated
- ✓ Unreferenced computation skipped (demand-driven)
- ✓ No wasted resources

**Educational Value**: Efficient evaluation important for classroom setting

---

## Test Organization

### Test Package Structure

```
packages/tests/
├── src/
│   ├── end-to-end/
│   │   ├── arithmetic.test.ts       # Tests 1.1, 1.2
│   │   ├── types.test.ts            # Tests 2.1, 2.2
│   │   ├── curriculum.test.ts        # Tests 3.1, 3.2
│   │   ├── sets.test.ts             # Tests 4.1, 4.2
│   │   ├── temporal.test.ts         # Tests 5.1, 5.2
│   │   ├── errors.test.ts           # Tests 6.1, 6.2
│   │   ├── performance.test.ts       # Tests 7.1, 7.2
│   │   └── demand-driven.test.ts     # Tests 8.1, 8.2
│   └── index.ts                    # Test entry point
├── package.json
└── tsconfig.json
```

### Test Framework

- Use `bun:test` for consistency with other packages
- Tests should be isolated (cleanup after each)
- Use `describe()` and `it()` for organization
- Assert using `expect()` from `bun:test`

---

## Success Criteria

### Package Creation

- [ ] `packages/tests/package.json` created with proper dependencies
- [ ] `packages/tests/src/` directory structure created
- [ ] `packages/tests/tsconfig.json` configured
- [ ] Monorepo workspaces updated to include tests package

### Test Implementation

- [ ] All 16 integration tests implemented (categories 1-8, 2 tests each)
- [ ] Each test has clear name, source code, expected outcome, acceptance criteria
- [ ] Tests use Compiler and Runtime from workspace dependencies
- [ ] Child-friendly Spanish error messages verified where applicable
- [ ] Performance targets verified with benchmarks

### Verification

- [ ] All integration tests pass
- [ ] Integration tests run independently of unit tests
- [ ] Integration tests can run with `bun test ./packages/tests`
- [ ] Test execution time <5s for full suite
- [ ] No flaky tests (same result on repeated runs)

---

## Educational Rationale

Integration tests ensure the system works as a cohesive whole for the target audience (ages 6-9). The focus on:

1. **End-to-End Scenarios**: Children build complete programs, not isolated operations
2. **Error Messages**: Must be child-friendly and Spanish to guide learning
3. **Performance**: Classroom use requires responsive system
4. **Curriculum Alignment**: Tests verify educational objectives (shapes, animals, counting)
5. **Demand-Driven Semantics**: Core concept must be correct for educational value

Integration tests bridge the gap between "components work in isolation" and "system works for real-world use."

---

## Integration with CI/CD

### Test Execution

- Run integration tests in CI pipeline after unit tests
- Fail build if any integration test fails
- Report coverage for integration layer separately

### Performance Regression Detection

- Benchmark critical paths (compilation, execution)
- Alert if performance degrades by >20% from baseline
- Profile slow tests (>100ms) to identify bottlenecks

---

**End of Integration Tests Specification**
