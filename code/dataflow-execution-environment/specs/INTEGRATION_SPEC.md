# Integration Layer Specification
## HTTP API, WebSocket Server & Incremental Runtime

**Document Status:** Living specification - update as implementation reveals better designs.

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 Core Principle
**Stateless Core + Thin Adapters**

```
┌─────────────────────────────────────────────┐
│ Compiler & Runtime (Stateless Core)        │
│ - No protocol knowledge                     │
│ - Pure functions                            │
│ - No sessions, no state                     │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│ Integration Layer (Thin Protocol Adapters) │
│                                             │
│ ┌─────────────┐  ┌──────────────────────┐  │
│ │ HTTP API    │  │ WebSocket Server     │  │
│ │ (Batch)     │  │ (Live/Incremental)   │  │
│ └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────┘
       ↑                       ↑
       │                       │
   CV System             IDE (AR viz)
  (complete             (live construction)
   programs)
```

### 1.2 Two Evaluation Modes

**Batch Mode:**
- Input: Complete, valid program
- Process: Compile → Validate → Execute
- Output: Final results + trace
- Use case: CV detects finished program → show AR visualization

**Incremental Mode:**
- Input: Partial graph (program under construction)
- Process: Validate partial → Evaluate what's possible
- Output: Partial results + node states (completed/pending/error)
- Use case: Child building program → live feedback as they add blocks

---

## 2. HTTP API (BATCH MODE)

### 2.1 Purpose
Simple request/response for complete programs.

### 2.2 Endpoints

#### POST /api/v1/compile
Validate a program without executing.

**Request:**
```json
{
  "program": {
    "metadata": { "programId": "prog_001" },
    "graph": {
      "nodes": [...],
      "edges": [...]
    }
  }
}
```

**Response (Success):**
```json
{
  "success": true,
  "programId": "prog_001",
  "errors": [],
  "warnings": []
}
```

**Response (Validation Error):**
```json
{
  "success": false,
  "programId": "prog_001",
  "errors": [
    {
      "code": "CYCLE_DETECTED",
      "message": "Graph contains a cycle",
      "nodeId": "node_5",
      "childMessage": "⚠️ ¡Ups! Hay un ciclo en el programa. Los bloques se conectan en círculo y no se puede calcular.",
      "suggestion": "Busca dónde un bloque apunta a sí mismo y desconecta una línea.",
      "example": "[A] → [B] → [C] ❌\n[A] → [B] ✅ (rompe el ciclo)"
    },
    {
      "code": "TYPE_ERROR",
      "message": "FILTER_BY_COLOR requires elements with 'color' property",
      "nodeId": "filter_1",
      "childMessage": "⚠️ ¡Ups! Este bloque necesita elementos con color.",
      "suggestion": "Usa bloques con color como formas o carros.",
      "example": "[forma roja] 🎨 [rojo] ✅"
    }
  ],
  "warnings": []
}
```

#### POST /api/v1/execute
Compile and execute a program.

**Request:**
```json
{
  "program": { /* DataflowProgram */ },
  "options": {
    "maxTimesteps": 10,        // For temporal programs
    "includeTrace": true,      // Include execution trace
    "traceLevel": "detailed"   // "minimal" | "medium" | "detailed"
  }
}
```

**Response:**
```json
{
  "success": true,
  "outputs": [5],  // Results from output nodes
  "trace": {
    "executionOrder": ["n1", "n2", "add", "output"],
    "nodeEvaluations": {
      "n1": { "value": 3, "timestep": 0 },
      "n2": { "value": 2, "timestep": 0 },
      "add": { "value": 5, "timestep": 0 }
    },
    "cacheHits": 0,
    "cacheMisses": 4,
    "totalTime": "2.3ms"
  }
}
```

#### GET /api/v1/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600
}
```

### 2.3 Acceptance Criteria

```markdown
## Acceptance Criteria: HTTP API

### Behavioral Outcomes
- ✓ POST /compile validates program structure
- ✓ POST /compile detects cycles, type errors, arity errors
- ✓ POST /execute compiles and runs valid programs
- ✓ POST /execute returns outputs + execution trace
- ✓ All endpoints return proper HTTP status codes
- ✓ Error responses include child-friendly Spanish messages for ages 6-9

### Observable Results
- Valid program → 200 OK, success: true
- Invalid program (cycle) → 200 OK, success: false, errors: [...]
- Malformed JSON → 400 Bad Request
- Server error → 500 Internal Server Error
- **Error responses include child-friendly Spanish messages**

### Performance
- /compile endpoint <100ms (p95) for 100-node programs
- /execute endpoint <50ms (p95) for 50-node programs
- Handles 5 concurrent requests without degradation

### Edge Cases
- Empty program → validation error "No output nodes"
- Program with only DataSource nodes → valid but pointless
- Temporal program without maxTimesteps → default to 100
```

---

## 3. WEBSOCKET SERVER (LIVE MODE)

### 3.1 Purpose
Real-time bidirectional communication for live validation and incremental evaluation.

### 3.2 Connection
```
ws://localhost:3000/live
```

### 3.3 Message Protocol

#### Client → Server Messages

**validate_program:**
```json
{
  "type": "validate_program",
  "messageId": "msg_001",
  "program": { /* DataflowProgram */ }
}
```

**evaluate_incremental:**
```json
{
  "type": "evaluate_incremental",
  "messageId": "msg_002",
  "program": { /* Partial DataflowProgram */ }
}
```

**subscribe_node:**
```json
{
  "type": "subscribe_node",
  "messageId": "msg_003",
  "nodeId": "filter_1"
}
```

**unsubscribe_node:**
```json
{
  "type": "unsubscribe_node",
  "messageId": "msg_004",
  "nodeId": "filter_1"
}
```

#### Server → Client Messages

**validation_result:**
```json
{
  "type": "validation_result",
  "messageId": "msg_001",
  "errors": [
    {
      "code": "TYPE_ERROR",
      "nodeId": "filter_1",
      "message": "Elements must have 'color' property",
      "childMessage": "⚠️ ¡Ups! Este bloque necesita elementos con color.",
      "suggestion": "Usa bloques con color como formas o carros.",
      "example": "[forma roja] 🎨 [rojo] ✅"
    }
  ],
  "warnings": []
}
```

**evaluation_result:**
```json
{
  "type": "evaluation_result",
  "messageId": "msg_002",
  "nodeStates": {
    "n1": { "status": "completed", "value": 5 },
    "n2": { "status": "completed", "value": 3 },
    "add": { "status": "completed", "value": 8 },
    "filter": {
      "status": "pending",
      "missingInputs": [
        {
          "port": 0,
          "description": "set of shapes",
          "childMessage": "Esperando un conjunto de formas"
        },
        {
          "port": 1,
          "description": "color",
          "childMessage": "Esperando una tarjeta de color"
        }
      ]
    }
  }
}
```

**node_state_changed (push notification):**
```json
{
  "type": "node_state_changed",
  "nodeId": "filter_1",
  "state": {
    "status": "completed",
    "value": [...]
  }
}
```

**Note**: For pending states, include child-friendly Spanish messages:
```json
{
  "type": "node_state_changed",
  "nodeId": "filter_1",
  "state": {
    "status": "pending",
    "missingInputs": [
      {
        "port": 1,
        "description": "color",
        "childMessage": "Esperando una tarjeta de color"
      }
    ]
  }
}
```

**error:**
```json
{
  "type": "error",
  "messageId": "msg_001",
  "code": "INVALID_MESSAGE",
  "message": "Unknown message type"
}
```

### 3.4 Acceptance Criteria

```markdown
## Acceptance Criteria: WebSocket Server

### Behavioral Outcomes
- ✓ Client connects via ws://localhost:3000/live
- ✓ Server validates messages and responds with messageId
- ✓ Server handles validate_program requests
- ✓ Server handles evaluate_incremental requests
- ✓ Server pushes node_state_changed when subscribed
- ✓ Multiple clients can connect simultaneously
- ✓ Connection is resilient (reconnects on disconnect)
- ✓ Error messages include child-friendly Spanish text for ages 6-9

### Observable Results
- Valid program → validation_result with empty errors
- Partial program → evaluation_result with mixed statuses
- Subscribe to node → receive state changes on updates
- Malformed message → error response

### Performance
- Message roundtrip <50ms (p95)
- Handles 5 concurrent WebSocket connections
- No memory leaks on client disconnect

### Edge Cases
- Client disconnects during evaluation → cleanup subscriptions
- Duplicate subscribe to same node → idempotent
- Message without messageId → generate one server-side
```

---

## 4. INCREMENTAL RUNTIME

### 4.1 Purpose
Evaluate partial programs that are still being constructed.

### 4.2 Design

**Key Concepts:**
- **Partial Graph:** Graph with missing nodes or edges
- **Node States:** completed | pending | processing | error
- **Incremental Update:** Only re-evaluate affected nodes when graph changes

### 4.3 API

```typescript
class IncrementalRuntime {
  private graph: PartialDataflowGraph;
  private cache: Map<string, Map<number, NodeEvaluation>>;
  private subscriptions: Map<string, Set<Subscriber>>;
  
  /**
   * Update the graph with new/changed/removed nodes or edges
   */
  updateGraph(delta: GraphDelta): UpdateResult {
    // Apply delta to internal graph
    // Invalidate affected cache entries
    // Return list of changed nodes
  }
  
  /**
   * Evaluate the partial graph as far as possible
   */
  evaluatePartial(timestep: number = 0): PartialEvaluation {
    // Try to evaluate each node
    // Return results for computable nodes
    // Mark others as "pending" with reason
  }
  
  /**
   * Get current state of a specific node
   */
  getNodeState(nodeId: string): NodeState {
    // Return cached state or compute if possible
  }
  
  /**
   * Subscribe to changes on a specific node
   */
  subscribe(nodeId: string, callback: StateCallback): void {
    // Add callback to subscriptions
  }
  
  /**
   * Unsubscribe from node changes
   */
  unsubscribe(nodeId: string, callback: StateCallback): void {
    // Remove callback from subscriptions
  }
}
```

### 4.4 Node States

```typescript
type NodeState =
  | { status: "completed", value: any }
  | { status: "pending", missingInputs: MissingInput[] }
  | { status: "processing" }
  | { status: "error", error: string };

type MissingInput = {
  port: number;
  description: string;      // e.g., "número", "color", "forma"
  childMessage: string;      // Spanish description
};

// Examples:
{ status: "completed", value: 5 }
{ status: "pending", missingInputs: [
  {
    "port": 0,
    "description": "set of shapes",
    "childMessage": "Esperando un conjunto de formas"
  },
  {
    "port": 1,
    "description": "color",
    "childMessage": "Esperando una tarjeta de color"
  }
]}
{ status: "error", error: "Division by zero" }
```

### 4.5 Demand-Driven Semantics with Partial Graphs

**CRITICAL:** Incremental Runtime STILL uses demand-driven evaluation, NOT eager evaluation.

**Key Principle:**
- Don't evaluate nodes just because they exist
- Evaluate ONLY when demanded (from output nodes or subscriptions)
- Partial graphs may have NO output nodes yet → nothing to evaluate

**How it works:**

```typescript
class IncrementalRuntime {
  evaluatePartial(timestep: number = 0): PartialEvaluation {
    const results: Record<string, NodeState> = {};
    
    // Find OUTPUT nodes (demand sources)
    const outputNodes = this.graph.getOutputNodes();
    
    // If no output nodes, find SUBSCRIBED nodes
    const demandSources = outputNodes.length > 0 
      ? outputNodes 
      : Array.from(this.subscriptions.keys())
          .map(id => this.graph.nodes.get(id))
          .filter(n => n !== undefined);
    
    // For each demand source, try to evaluate (demand-driven!)
    for (const source of demandSources) {
      try {
        const value = this.evaluate(source.id, timestep);
        results[source.id] = { status: "completed", value };
        } catch (e) {
          if (e instanceof MissingInputError) {
            results[source.id] = {
              status: "pending",
              missingInputs: this.extractMissingInputs(source)
            };
          } else {
            results[source.id] = {
              status: "error",
              error: e.message
            };
          }
        }
    }
    
    return { nodeStates: results };
  }
  
  // Demand-driven evaluate (same as batch runtime!)
  private evaluate(nodeId: string, time: number): any {
    // Check cache
    if (this.cache.has(nodeId, time)) {
      return this.cache.get(nodeId, time);
    }
    
    const node = this.graph.nodes.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }
    
    // Get inputs (this propagates demand!)
    const inputNodes = this.graph.getInputs(nodeId);
    const inputs = new Map();
    
    for (const inputNode of inputNodes) {
      // This is where MissingInputError can be thrown
      const value = this.evaluate(inputNode.id, time);
      inputs.set(inputNode.id, value);
    }
    
    // Evaluate node with inputs
    const value = node.evaluate(inputs);
    
    // Cache and return
    this.cache.set(nodeId, time, value);
    return value;
  }

  /**
   * Extract all missing inputs for a node
   */
  private extractMissingInputs(node: DataflowNode): MissingInput[] {
    const missing: MissingInput[] = [];
    const signature = OPERATION_REGISTRY[node.operation];
    const actualInputs = this.graph.getInputs(node.id);

    for (let i = 0; i < signature.arity; i++) {
      if (!actualInputs[i] || !actualInputs[i].hasValue()) {
        missing.push({
          port: i,
          description: this.getInputDescription(node.operation, i),
          childMessage: this.getChildMessageForMissingInput(node.operation, i)
        });
      }
    }

    return missing;
  }

  /**
   * Get child-friendly Spanish message for missing input
   */
  private getChildMessageForMissingInput(operation: string, port: number): string {
    if (operation === "FILTER_BY_COLOR" && port === 1) {
      return "Esperando una tarjeta de color";
    }
    if (operation.startsWith("FILTER_") && port === 1) {
      return "Esperando una tarjeta de criterio";
    }
    if (operation.startsWith("COMPARE_")) {
      return "Esperando un elemento para comparar";
    }
    return "Esperando una entrada";
  }
}
```

**Example 1: No output nodes yet**
```
Graph: { nodes: [{ id: "n1", value: 5 }] }

evaluatePartial():
  outputNodes = []  // No output nodes
  subscriptions = {} // No subscriptions
  demandSources = []
  → Return: {}  // No evaluation happens!
  
Reason: Nothing demands n1's value, so it's NOT evaluated.
This is CORRECT demand-driven behavior.
```

**Example 2: User subscribes to node**
```
Graph: { nodes: [{ id: "n1", value: 5 }] }
Action: subscribe_node("n1")

evaluatePartial():
  outputNodes = []
  subscriptions = { "n1": [callback] }
  demandSources = [n1]
  → evaluate(n1, 0)
    → return 5
  → Return: { n1: { status: "completed", value: 5 } }

Reason: Subscription creates demand for n1.
```

**Example 3: Partial graph with missing inputs**
```
Graph: {
  nodes: [
    { id: "filter", operation: "FILTER_BY_COLOR" }
  ],
  edges: []  // No inputs connected!
}
Action: subscribe_node("filter")

evaluatePartial():
  demandSources = [filter]
  → evaluate(filter, 0)
    → getInputs(filter) = []  // No edges!
    → filter.evaluate({})  // Empty inputs
    → Throws: MissingInputError("Missing input at port 0")
  → Return: {
      filter: {
        status: "pending",
        missingInputs: [
          {
            "port": 0,
            "description": "set of shapes",
            "childMessage": "Esperando un conjunto de formas"
          }
        ]
      }
    }
  ```

**Example 4: Cascade of pending nodes**
```
Graph: {
  nodes: [
    { id: "add", operation: "ADD" },
    { id: "n1", value: 5 }
  ],
  edges: [
    { from: "n1", to: "add", toPort: 0 }
    // Missing: second input to ADD!
  ]
}
Action: subscribe_node("add")

evaluatePartial():
  demandSources = [add]
  → evaluate(add, 0)
    → getInputs(add) = [n1, ???]  // Second input missing!
    → evaluate(n1, 0) = 5  // This succeeds
    → evaluate(???, 0)  // No node connected to port 1
    → Throws: MissingInputError("Missing input at port 1")
  → Return: {
      add: {
        status: "pending",
        "missingInputs": [
          {
            "port": 1,
            "description": "número",
            "childMessage": "Esperando un número"
          }
        ]
      }
    }

Note: n1 WAS evaluated (due to demand from add),
but we don't return its state unless it's demanded directly.
```

**Key Differences from Batch Runtime:**

| Aspect | Batch Runtime | Incremental Runtime |
|--------|--------------|---------------------|
| Demand Sources | Output nodes (required) | Output nodes OR subscriptions |
| Missing Inputs | Error (fail compilation) | Pending state (continue) |
| Graph Completeness | Must be complete DAG | Can be partial |
| Return Value | Final outputs | Node states (completed/pending/error) |
| Cache Invalidation | N/A (single run) | Yes (on graph updates) |

**Why This Design?**

1. **Stays True to Demand-Driven:** Only evaluates what's demanded
2. **No Eager Evaluation:** Doesn't compute nodes "just because they exist"
3. **Graceful Degradation:** Missing inputs → pending state, not crash
4. **Efficient:** Only re-evaluates affected nodes on updates
5. **Consistent Semantics:** Same evaluate() logic as batch runtime

**Common Misconception:**

❌ "Incremental runtime evaluates ALL nodes and marks pending ones"
✅ "Incremental runtime evaluates DEMANDED nodes, catches missing input errors"

The difference is subtle but critical for maintaining demand-driven semantics.

### 4.6 Example Flow

**Scenario:** Child building "filter red shapes" program

```
Step 1: Place shapes set
  Graph: { nodes: [{ id: "shapes", value: [...] }] }
  Eval:  { shapes: { status: "completed", value: [...] } }

Step 2: Place filter operation (no color yet)
  Graph: { 
    nodes: [...prev, { id: "filter", operation: "FILTER_BY_COLOR" }],
    edges: [{ from: "shapes", to: "filter", toPort: 0 }]
  }
  Eval: {
    shapes: { status: "completed", value: [...] },
    filter: {
      status: "pending",
      missingInputs: [
        {
          "port": 1,
          "description": "color",
          "childMessage": "Esperando una tarjeta de color"
        }
      ]
    }
  }

Step 3: Place color card "red"
  Graph: {
    nodes: [...prev, { id: "color", value: "red" }],
    edges: [...prev, { from: "color", to: "filter", toPort: 1 }]
  }
  Eval: {
    shapes: { status: "completed", value: [...] },
    color: { status: "completed", value: "red" },
    filter: { status: "completed", value: [/* red shapes */] }
  }
```

### 4.7 Acceptance Criteria

```markdown
## Acceptance Criteria: Incremental Runtime

### Behavioral Outcomes
- ✓ Evaluates partial graphs (incomplete programs)
- ✓ Returns results for computable nodes
- ✓ Marks nodes as "pending" when inputs missing
- ✓ Missing input messages in Spanish with multiple inputs support
- ✓ Updates incrementally when graph changes
- ✓ Only re-evaluates affected nodes on update
- ✓ Subscribers receive notifications on state changes

### Observable Results
- Partial graph (just "5") → { n1: { status: "completed", value: 5 } }
- Add "3" + "ADD" → { add: { status: "completed", value: 8 } }
- FILTER missing color → { filter: { status: "pending", missingInputs: [...] } }
- Update "5" to "7" → only re-evaluate nodes depending on n1
- Multiple missing inputs reported in missingInputs array with Spanish descriptions

### Performance
- updateGraph() <10ms (p95)
- evaluatePartial() <20ms (p95) for 50-node partial graphs
- Incremental update 5x faster than full re-evaluation

### Edge Cases
- Empty graph → no nodes, no results
- All nodes pending → valid state, nothing computable
- Remove node with dependents → mark dependents as pending
- Circular dependencies in partial graph → detect and mark as error
```

---

## 5. IMPLEMENTATION GUIDE

### 5.1 Layer 7a: HTTP API (Batch Mode)

**Estimated effort:** 1 day

**Stack:**
```typescript
// packages/http-api/
src/
  server.ts        // Elysia app setup
  routes/
    compile.ts     // POST /api/v1/compile
    execute.ts     // POST /api/v1/execute
    health.ts      // GET /api/v1/health
  middleware/
    validation.ts  // Request validation
    errors.ts      // Error handling
```

**Runtime:** Bun (not Node.js)

**Dependencies:**
```json
{
  "elysia": "latest"
}
```

**Example:**
```typescript
import { Elysia } from 'elysia';
import { Compiler } from '@dataflow/compiler';

const app = new Elysia()
  .post('/api/v1/compile', ({ body }) => {
    const compiler = new Compiler();
    const result = compiler.compile(body.program);
    return result;
  })
  .listen(3000);
```

**Tests:**
- Valid program returns success
- Invalid program returns errors
- Concurrent requests handled
- Performance benchmarks

### 5.2 Layer 7b: WebSocket Server (Live Mode)

**Estimated effort:** 1-2 days

**Stack:**
```typescript
// packages/websocket-server/
src/
  server.ts              // Elysia + WebSocket plugin
  handlers/
    validate.ts          // validate_program handler
    evaluate.ts          // evaluate_incremental handler
    subscribe.ts         // subscribe/unsubscribe handlers
  connection-manager.ts  // Track active connections
  subscription-manager.ts // Track node subscriptions
```

**Runtime:** Bun (not Node.js)

**Dependencies:**
```json
{
  "elysia": "latest"
}
```

**Example:**
```typescript
import { Elysia } from 'elysia';

const app = new Elysia()
  .ws('/live', {
    open(ws) {
      console.log('Client connected');
    },
    message(ws, message) {
      const msg = JSON.parse(message);
      
      if (msg.type === 'validate_program') {
        const result = validateProgram(msg.program);
        ws.send(JSON.stringify({
          type: 'validation_result',
          messageId: msg.messageId,
          errors: result.errors
        }));
      }
    },
    close(ws) {
      console.log('Client disconnected');
    }
  })
  .listen(3000);
```

**Tests:**
- Client connects and sends messages
- Server responds with correct messageId
- Subscriptions work correctly
- Multiple concurrent clients

### 5.3 Layer 7c: Incremental Runtime

**Estimated effort:** 2-3 days

**Stack:**
```typescript
// packages/runtime/
src/
  incremental-runtime.ts  // IncrementalRuntime class
  partial-graph.ts        // PartialDataflowGraph
  graph-delta.ts          // GraphDelta type
  node-state.ts           // NodeState types
```

**Tests:**
- Evaluate partial graph (some nodes pending)
- Update graph, only re-evaluate affected
- Subscriptions notify on state change
- Performance: incremental vs full re-eval

---

## 6. PROTOCOL VERSIONING

### 6.1 HTTP API Versioning
URL-based: `/api/v1/`, `/api/v2/`

### 6.2 WebSocket Protocol Versioning
Message field: `{ "version": "1.0", "type": "...", ... }`

### 6.3 Backward Compatibility
- New fields: optional, with defaults
- Breaking changes: new version number
- Deprecation: warn for 2 versions before removal

---

## 7. SECURITY CONSIDERATIONS

### 7.1 Input Validation
- Validate all JSON against schema
- Limit graph size (max 1000 nodes)
- Limit nesting depth (max 10 levels)
- Timeout long-running evaluations (5s)

### 7.2 Rate Limiting
- HTTP: 100 requests/minute per IP
- WebSocket: 50 messages/second per connection

### 7.3 Resource Limits
- Max memory per evaluation: 100MB
- Max concurrent evaluations: 5

---

**End of Integration Layer Specification**
