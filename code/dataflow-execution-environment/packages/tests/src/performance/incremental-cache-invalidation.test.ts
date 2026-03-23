import { describe, it, expect } from "bun:test";
import { IncrementalRuntime } from "@dataflow/runtime";
import type { DataflowNode, DataflowEdge, DataflowProgram } from "@dataflow/shared/types";

describe("Performance - Incremental Runtime Cache Invalidation", () => {
  it("should invalidate cache in O(n) time for 100-node program", () => {
    const runtime = new IncrementalRuntime();

    const nodes: DataflowNode[] = [];
    const edges: DataflowEdge[] = [];

    for (let i = 0; i < 100; i++) {
      nodes.push({
        id: `src${i}`,
        type: "DataSource",
        dataType: "natural",
        value: i
      });
    }

    for (let i = 0; i < 50; i++) {
      nodes.push({
        id: `t${i}`,
        type: "Transformation",
        dataType: "natural",
        operation: "ADD",
        inputs: []
      });

      edges.push({
        id: `e${i}_0`,
        from: `src${i}`,
        to: `t${i}`,
        toPort: 0
      });

      edges.push({
        id: `e${i}_1`,
        from: `src${i + 50}`,
        to: `t${i}`,
        toPort: 1
      });
    }

    const program: DataflowProgram = {
      metadata: { programId: "perf-test" },
      graph: {
        nodes,
        edges
      }
    };

    runtime.loadProgram(program);

    const start = performance.now();
    runtime.updateGraph({
      addedNodes: [{
        id: "src100",
        type: "DataSource",
        dataType: "natural",
        value: 100
      }]
    });
    const time = performance.now() - start;

    expect(time).toBeLessThan(10);
  });
});

