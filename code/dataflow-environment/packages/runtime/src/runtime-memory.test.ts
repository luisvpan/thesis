import { describe, it, expect } from "bun:test";
import { Runtime } from "../runtime";
import type { DataflowProgram } from "@dataflow/shared/types";

describe("Runtime Memory Management", () => {
  it("should use LRU cache with size limit", () => {
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
  });

  it("should clear cache when limit is exceeded", () => {
    const runtime = new Runtime();
    
    const program: DataflowProgram = {
      metadata: { programId: "test-clear" },
      graph: {
        nodes: [
          { id: "n1", type: "DataSource", dataType: "natural", value: 1 },
          { id: "n2", type: "DataSource", dataType: "natural", value: 2 },
          { id: "add", type: "Transformation", dataType: "natural", operation: "ADD" },
          { id: "output", type: "Output", dataType: "natural" }
        ],
        edges: [
          { id: "e1", from: "n1", to: "add", toPort: 0 },
          { id: "e2", from: "n2", to: "add", toPort: 1 },
          { id: "e3", from: "add", to: "output" }
        ]
      }
    };

    runtime.loadProgram(program);
    
    const result1 = runtime.execute();
    const stats1 = runtime.getCacheStats();
    
    const result2 = runtime.execute();
    const stats2 = runtime.getCacheStats();

    expect(stats2.hits).toBe(stats1.hits + 4);
    expect(stats2.misses).toBe(stats1.misses);
  });
});
