# Demand-Driven Semantics in Incremental Runtime
## Clarifying How Partial Evaluation Works

---

## The Question

**"How does incremental/partial evaluation reconcile with demand-driven (lazy) semantics?"**

**Does the incremental runtime:**
- ❌ Eagerly evaluate all nodes? (breaks demand-driven)
- ❌ Make exceptions to demand-driven? (inconsistent semantics)
- ✅ Stay true to demand-driven while handling partial graphs?

**Answer:** The incremental runtime **stays 100% demand-driven**. No exceptions.

---

## Key Insight: Demand Sources

**In Batch Runtime:**
- Demand comes from: **Output nodes** (required)
- If no output nodes → error (incomplete program)

**In Incremental Runtime:**
- Demand comes from: **Output nodes OR subscriptions**
- If no output nodes AND no subscriptions → **nothing is evaluated**
- This is CORRECT demand-driven behavior

---

## How It Actually Works

### Scenario 1: No Demand Sources

```typescript
Graph: { 
  nodes: [
    { id: "n1", value: 5 },
    { id: "n2", value: 3 }
  ],
  edges: []
}

// No output nodes
// No subscriptions

evaluatePartial() → {}  // Empty result!
```

**Why empty?**
- Nothing demands n1 or n2's values
- Demand-driven = don't compute unless demanded
- Result: No evaluation happens

**This is NOT a bug.** It's correct demand-driven semantics.

---

### Scenario 2: User Subscribes to Node

```typescript
Graph: { nodes: [{ id: "n1", value: 5 }] }

// User subscribes to n1 (IDE wants to show its value)
subscribe("n1")

evaluatePartial() → { 
  n1: { status: "completed", value: 5 } 
}
```

**Why n1 is evaluated?**
- Subscription creates **demand** for n1
- Demand-driven: evaluate n1 (it's demanded)
- Result: n1 computed and returned

---

### Scenario 3: Partial Graph with Missing Input

```typescript
Graph: {
  nodes: [
    { id: "shapes", value: [...] },
    { id: "filter", operation: "FILTER_BY_COLOR" }
  ],
  edges: [
    { from: "shapes", to: "filter", toPort: 0 }
    // Missing: color input at port 1!
  ]
}

subscribe("filter")

evaluatePartial() → {
  filter: { 
    status: "pending", 
    reason: "Missing input at port 1 (color)" 
  }
}
```

**What happened?**
1. Subscription demands `filter`
2. Runtime calls `evaluate(filter, 0)`
3. `evaluate` demands inputs of `filter`
4. Port 0: `shapes` → evaluates successfully
5. Port 1: **no edge connected** → throws `MissingInputError`
6. Error caught → returns "pending" state

**Key:** Runtime tried to evaluate (demand-driven), but couldn't complete.

---

## The Evaluate Algorithm (Unchanged!)

```typescript
private evaluate(nodeId: string, time: number): any {
  // 1. Check cache (memoization)
  if (cache.has(nodeId, time)) {
    return cache.get(nodeId, time);
  }
  
  // 2. Get node
  const node = graph.nodes.get(nodeId);
  if (!node) throw new Error("Node not found");
  
  // 3. DEMAND inputs (recursively)
  const inputNodes = graph.getInputs(nodeId);
  const inputs = new Map();
  
  for (const inputNode of inputNodes) {
    const value = this.evaluate(inputNode.id, time);  // Recursive demand!
    inputs.set(inputNode.id, value);
  }
  
  // 4. Evaluate node with inputs
  const value = node.evaluate(inputs);
  
  // 5. Cache result
  cache.set(nodeId, time, value);
  
  return value;
}
```

**This is THE SAME in both:**
- Batch Runtime
- Incremental Runtime

**No exceptions. No eager evaluation. Pure demand-driven.**

---

## Where MissingInputError Comes From

```typescript
class DataflowGraph {
  getInputs(nodeId: string): DataflowNode[] {
    const edges = this.edges.filter(e => e.to === nodeId);
    const inputs: DataflowNode[] = [];
    
    for (const edge of edges) {
      const inputNode = this.nodes.get(edge.from);
      if (!inputNode) {
        throw new MissingInputError(
          `Input node '${edge.from}' not found`
        );
      }
      inputs.push(inputNode);
    }
    
    // Check arity (expected inputs vs actual)
    const operation = this.nodes.get(nodeId).operation;
    const expectedArity = OPERATION_REGISTRY[operation].arity;
    
    if (inputs.length < expectedArity) {
      throw new MissingInputError(
        `Missing input at port ${inputs.length} for ${operation}`
      );
    }
    
    return inputs;
  }
}
```

**In Batch Runtime:**
- `MissingInputError` → compilation fails
- User must fix program before it runs

**In Incremental Runtime:**
- `MissingInputError` → caught, return "pending" state
- User can continue building, runtime shows what's missing

---

## Comparison Table

| Aspect | Batch Runtime | Incremental Runtime |
|--------|--------------|---------------------|
| **Evaluation Strategy** | Demand-driven | Demand-driven (same!) |
| **Demand Sources** | Output nodes only | Output nodes OR subscriptions |
| **Missing Inputs** | Compilation error | Pending state |
| **Partial Graphs** | Not accepted | Accepted |
| **evaluate() Function** | Pure demand propagation | Pure demand propagation (same!) |
| **Cache Behavior** | Memoization | Memoization + invalidation |

---

## Example: Cascading Evaluation

```typescript
Graph: {
  nodes: [
    { id: "n1", value: 5 },
    { id: "n2", value: 3 },
    { id: "add", operation: "ADD" },
    { id: "mul", operation: "MULTIPLY" }
  ],
  edges: [
    { from: "n1", to: "add", toPort: 0 },
    { from: "n2", to: "add", toPort: 1 },
    { from: "add", to: "mul", toPort: 0 }
    // Missing: second input to mul!
  ]
}

subscribe("mul")

evaluatePartial():
  → evaluate(mul, 0)
    → getInputs(mul) 
      → Needs 2 inputs, has 1
      → Throws MissingInputError("Missing input at port 1")
    → Caught → return "pending"
  
  Result: { mul: { status: "pending", reason: "..." } }
```

**Notice:**
- `n1`, `n2`, and `add` were **NOT** evaluated
- Why? Because `mul` failed before demanding them
- If we subscribe to `add` instead:

```typescript
subscribe("add")

evaluatePartial():
  → evaluate(add, 0)
    → getInputs(add) = [n1, n2]  ✓
    → evaluate(n1, 0) = 5
    → evaluate(n2, 0) = 3
    → add.evaluate({n1: 5, n2: 3}) = 8
  
  Result: { add: { status: "completed", value: 8 } }
```

**Now:**
- `n1` and `n2` WERE evaluated
- Why? Because `add` demanded them
- Pure demand propagation!

---

## Common Misconceptions

### ❌ Misconception 1: "Incremental runtime evaluates everything"

**Wrong.** It only evaluates what's demanded.

```typescript
// 100 nodes, no subscriptions, no outputs
evaluatePartial() → {}  // Nothing evaluated!
```

### ❌ Misconception 2: "Incremental runtime is eager"

**Wrong.** It's as lazy as batch runtime.

```typescript
// Only subscribes to one node
subscribe("filter")
evaluatePartial() → { filter: {...} }  // Only filter (+ dependencies) evaluated
```

### ❌ Misconception 3: "It's a different evaluation model"

**Wrong.** Uses the exact same `evaluate()` function.

The ONLY difference is:
- How to find demand sources (outputs vs subscriptions)
- What to do with `MissingInputError` (fail vs pending)

---

## Why Subscriptions?

**Problem:** In a partial graph, there might be NO output nodes yet.

```typescript
// Child just placed first block
Graph: { nodes: [{ id: "n1", value: 5 }] }

// No output nodes... so what to evaluate?
```

**Solution:** IDE subscribes to nodes it wants to display.

```typescript
// IDE: "Show me the value of n1"
subscribe("n1")

evaluatePartial() → { n1: { value: 5 } }
```

**This creates demand** → triggers evaluation.

Without subscriptions, incremental runtime would have no way to know WHAT to evaluate in a partial graph.

---

## Summary

**Incremental Runtime = Batch Runtime + Two Changes:**

1. **Demand Sources:**
   - Batch: Output nodes (required)
   - Incremental: Output nodes OR subscriptions

2. **Error Handling:**
   - Batch: `MissingInputError` → fail compilation
   - Incremental: `MissingInputError` → return "pending" state

**Everything else is identical:**
- Same demand-driven evaluation
- Same recursive demand propagation
- Same caching strategy
- Same `evaluate()` function

**No compromises. No exceptions. Pure demand-driven semantics.**

---

**End of Explanation**
