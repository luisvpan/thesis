# Integration Options Analysis - Graphical/Tangible Language

## Problem Statement

**Context:** Dataflow language consumed via tangible blocks detected by CV system, NOT text editing.

**Current assumption:** LSP (Language Server Protocol)
**Issue:** LSP is designed for text-based languages. Overhead is considerable for a graphical/tangible language.

---

## Option 1: LSP (Language Server Protocol)

### Pros:
- ✅ Standard protocol, well-documented
- ✅ Many IDE clients available (VS Code, etc.)
- ✅ Rich features: diagnostics, autocomplete, hover, etc.

### Cons:
- ❌ Designed for TEXT editing (positions, ranges, documents)
- ❌ Significant overhead mapping graph nodes ↔ text positions
- ❌ Weird fit: "autocomplete" for physical blocks?
- ❌ Features like "go to definition" don't make sense

### Verdict:
**Poor fit** - Forcing graph semantics into text model creates impedance mismatch.

---

## Option 2: GLSP (Graphical Language Server Protocol)

### What is it?
Eclipse project for graphical modeling. LSP equivalent for diagrams/graphs.

**Repo:** https://github.com/eclipse-glsp/glsp

### Pros:
- ✅ Designed for graphical languages
- ✅ Native graph concepts (nodes, edges)
- ✅ Features: validation, layout, palette, etc.

### Cons:
- ❌ Very new/green (first release 2020)
- ❌ Primarily Eclipse ecosystem
- ❌ Limited adoption outside modeling tools
- ❌ Still heavier than needed for our use case

### Verdict:
**Interesting but risky** - Too green, overkill for our needs.

---

## Option 3: Custom WebSocket Protocol

### Design:
Simple bidirectional protocol:

```typescript
// Client → Server (CV/IDE)
{
  "type": "validate_program",
  "program": { /* JSON graph */ }
}

{
  "type": "evaluate_incremental", 
  "program": { /* partial graph */ },
  "mode": "live"  // REPL-like
}

// Server → Client
{
  "type": "validation_result",
  "errors": [
    { "nodeId": "shape_1", "code": "TYPE_ERROR", "message": "..." }
  ]
}

{
  "type": "evaluation_result",
  "nodeStates": {
    "shape_1": { "status": "completed", "value": [...] },
    "filter_1": { "status": "processing" }
  }
}
```

### Pros:
- ✅ Lightweight, minimal overhead
- ✅ Tailored exactly to our needs
- ✅ Easy to add graph-specific features
- ✅ WebSocket = bidirectional, real-time
- ✅ Simple to implement (~200 lines)

### Cons:
- ❌ Custom protocol (not standard)
- ❌ Need to document ourselves
- ❌ IDE clients need custom implementation

### Verdict:
**Best fit for MVP** - Simple, fast, exactly what we need.

---

## Option 4: REST API (HTTP)

### Design:
Simple request/response:

```
POST /compile
Body: { "program": { /* graph */ } }
Response: { "success": true, "errors": [] }

POST /execute
Body: { "program": { /* graph */ } }
Response: { "outputs": [...], "trace": [...] }
```

### Pros:
- ✅ Simplest possible
- ✅ Stateless
- ✅ Easy to test (curl, Postman)
- ✅ Works from any client

### Cons:
- ❌ No real-time feedback
- ❌ Request/response only (no push notifications)
- ❌ Not great for live/incremental evaluation

### Verdict:
**Good for batch mode** - Compile + execute complete programs.
**Not good for live feedback** - Can't push updates to IDE.

---

## Recommended Approach: Hybrid

### Architecture:

```
┌─────────────────────────────────────────────┐
│ Compiler & Runtime (Core)                  │
│ - Stateless, pure functions                │
│ - No protocol knowledge                     │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│ Integration Layer (Thin Adapters)          │
│                                             │
│ ┌─────────────┐  ┌──────────────────────┐  │
│ │ HTTP API    │  │ WebSocket Server     │  │
│ │ (Batch)     │  │ (Live/Incremental)   │  │
│ └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────┘
               ↑               ↑
               │               │
          CV System       IDE (AR viz)
        (batch compile)  (live feedback)
```

### HTTP API (for CV System):
```typescript
POST /api/v1/compile
POST /api/v1/execute
GET  /api/v1/health
```

**Use case:** CV detects complete program → send to compiler → get results → show AR

### WebSocket Server (for IDE):
```typescript
// Client connects
ws://localhost:3000/live

// Messages:
- validate_program    → validation_result
- evaluate_incremental → evaluation_update (streaming)
- subscribe_node       → node_state_changed (push)
```

**Use case:** IDE showing live construction → validate as they build → highlight errors in real-time

---

## Live Evaluation (REPL-like)

### Requirement:
Evaluate program **while being constructed**, not just complete programs.

### Design:

**Incremental Runtime Mode:**
```typescript
class IncrementalRuntime {
  private partialGraph: DataflowGraph;
  private cache: Map<string, any>;
  
  // Update graph with new nodes/edges
  updateGraph(delta: GraphDelta): void {
    // Add/remove/update nodes
    // Invalidate affected cache entries
  }
  
  // Evaluate as far as possible with partial graph
  evaluatePartial(): PartialResults {
    // Try to evaluate each output node
    // Return results for what CAN be computed
    // Mark nodes that need more inputs as "pending"
  }
  
  // Get current state of specific node
  getNodeState(nodeId: string): NodeState {
    // "completed" | "pending" | "error"
  }
}
```

### Example Flow:

```
Child places block "5" on table:
  CV: { nodes: [{ id: "n1", type: "DataSource", value: 5 }] }
  Runtime: evaluatePartial() → { n1: { status: "completed", value: 5 } }
  AR: Shows "5" highlighted

Child places block "3":
  CV: { nodes: [...prev, { id: "n2", type: "DataSource", value: 3 }] }
  Runtime: evaluatePartial() → { n1: ..., n2: { status: "completed", value: 3 } }
  AR: Shows both "5" and "3" highlighted

Child connects with "ADD":
  CV: { nodes: [..., { id: "add", type: "Transformation", operation: "ADD" }],
        edges: [{ from: "n1", to: "add" }, { from: "n2", to: "add" }] }
  Runtime: evaluatePartial() → { 
    n1: ..., 
    n2: ..., 
    add: { status: "completed", value: 8 } 
  }
  AR: Shows "5 + 3 = 8" with animation

Child places incomplete filter (no color specified):
  CV: { nodes: [..., { id: "filter", operation: "FILTER_BY_COLOR" }],
        edges: [{ from: "shapes", to: "filter", toPort: 0 }] }
  Runtime: evaluatePartial() → {
    shapes: { status: "completed", value: [...] },
    filter: { status: "pending", reason: "Missing input at port 1 (color)" }
  }
  AR: Shows filter block pulsing/waiting for color card
```

### Key Difference from Batch Runtime:

**Batch Runtime:**
- Expects complete, valid program
- Fails if any validation error
- Evaluates all nodes
- Returns final outputs

**Incremental Runtime:**
- Accepts partial graphs
- Evaluates what it CAN
- Marks rest as "pending"
- Returns partial results + state

---

## Implementation Recommendation

### Phase 1 (Layer 7): Basic Integration

**Implement:**
1. **HTTP API** (simple, for CV system batch mode)
   - POST /compile - validate program
   - POST /execute - run complete program
   - ~100 lines of code (Express.js or similar)

2. **Basic WebSocket server** (for IDE)
   - validate_program command
   - Broadcasts validation errors
   - ~150 lines of code

**Skip for now:**
- LSP (wrong fit)
- GLSP (too green)
- Complex incremental evaluation

### Phase 2 (Post-MVP): Live Evaluation

**Add:**
3. **Incremental Runtime**
   - IncrementalRuntime class
   - updateGraph() method
   - evaluatePartial() method
   - ~300 lines of code

4. **WebSocket enhancements**
   - evaluate_incremental command
   - node_state_changed events (push)
   - subscribe/unsubscribe to specific nodes

---

## Acceptance Criteria

### For Integration Layer (Layer 7):

**HTTP API:**
```markdown
## Acceptance Criteria: HTTP API

### Behavioral Outcomes
- ✓ Accepts JSON program via POST /compile
- ✓ Returns validation errors or success
- ✓ Accepts JSON program via POST /execute
- ✓ Returns execution results + trace

### Observable Results
- POST /compile with valid program → { "success": true, "errors": [] }
- POST /compile with invalid (cycle) → { "success": false, "errors": [...] }
- POST /execute with "3 + 2" → { "outputs": [5], "trace": [...] }

### Performance
- Compile endpoint <100ms for 100-node programs
- Execute endpoint <50ms for 50-node programs
```

**WebSocket Server:**
```markdown
## Acceptance Criteria: WebSocket Live Validation

### Behavioral Outcomes
- ✓ Client connects via ws://localhost:3000/live
- ✓ Client sends validate_program message
- ✓ Server responds with validation_result
- ✓ Multiple clients can connect simultaneously

### Observable Results
- Send valid program → { "type": "validation_result", "errors": [] }
- Send program with cycle → { "errors": [{ "code": "CYCLE_DETECTED" }] }
```

### For Incremental Runtime (Post-MVP):

```markdown
## Acceptance Criteria: Incremental Evaluation

### Behavioral Outcomes
- ✓ Evaluates partial graphs (incomplete programs)
- ✓ Returns results for computable nodes
- ✓ Marks nodes as "pending" when inputs missing
- ✓ Updates incrementally when graph changes

### Observable Results
- Graph with just "5" → { n1: { status: "completed", value: 5 } }
- Graph with "5" + "3" + "ADD" → { add: { status: "completed", value: 8 } }
- Graph with FILTER missing color → { filter: { status: "pending" } }
```

---

## Protocol Specification (Draft)

### WebSocket Message Format

```typescript
// Client → Server
type ClientMessage = 
  | { type: "validate_program", program: DataflowProgram }
  | { type: "evaluate_incremental", program: DataflowProgram }
  | { type: "subscribe_node", nodeId: string }
  | { type: "unsubscribe_node", nodeId: string };

// Server → Client
type ServerMessage =
  | { type: "validation_result", errors: ValidationError[] }
  | { type: "evaluation_result", nodeStates: Record<string, NodeState> }
  | { type: "node_state_changed", nodeId: string, state: NodeState }
  | { type: "error", message: string };

type NodeState = {
  status: "completed" | "pending" | "processing" | "error";
  value?: any;
  error?: string;
  reason?: string;  // For "pending" status
};
```

---

## Decision Summary

**Recommended:**
- ✅ Custom WebSocket protocol (lightweight, tailored)
- ✅ HTTP API for batch mode (CV system)
- ✅ Incremental Runtime for live evaluation (REPL-like)
- ❌ NOT LSP (wrong fit for graphical language)
- ❌ NOT GLSP (too green/heavy)

**Implementation order:**
1. Layer 7a: HTTP API (simple, for complete programs)
2. Layer 7b: WebSocket server (validation messages)
3. Post-MVP: Incremental Runtime (live evaluation)

---

**Document Status:** Analysis for PROJECT_GOALS.md update
