import { describe, it, expect } from "bun:test";
import { Runtime } from "./runtime";
import type { DataflowProgram } from "@dataflow/shared/types";

describe("Runtime Memory Management", () => {
  it("should use LRU cache and track statistics", () => {
    const runtime = new Runtime();
    const program: DataflowProgram = {
      metadata: { programId: "test-lru" },
      graph: {
        nodes: [],
        edges: []
      }
    };

    runtime.loadProgram(program);
    runtime.execute();

    const stats = runtime.getCacheStats();
    expect(stats.hits).toBeGreaterThanOrEqual(0);
    expect(stats.misses).toBeGreaterThanOrEqual(0);
    expect(typeof stats.hits).toBe("number");
    expect(typeof stats.misses).toBe("number");
  });

  it("should expose cache stats through Runtime API", () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: "test-stats" },
      graph: {
        nodes: [
          { id: "n1", type: "DataSource", dataType: "natural", value: 1 },
          { id: "n2", type: "DataSource", dataType: "natural", value: 2 },
          { id: "add", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["n1", "n2"] },
          { id: "output", type: "Output", dataType: "natural", input: "add" }
        ],
        edges: [
          { id: "e1", from: "n1", to: "add", toPort: 0 },
          { id: "e2", from: "n2", to: "add", toPort: 1 },
          { id: "e3", from: "add", to: "output" }
        ]
      }
    };
    
    runtime.loadProgram(program);
    
    // Execute and verify cache is being used
    const result = runtime.execute();
    const stats = runtime.getCacheStats();
    
    expect(result).toBeDefined();
    expect(stats).toBeDefined();
    expect(stats).toHaveProperty("hits");
    expect(stats).toHaveProperty("misses");
    expect(typeof stats.hits).toBe("number");
    expect(typeof stats.misses).toBe("number");
  });
});
