import { describe, it, expect } from "bun:test";
import { app } from "../src/server";

describe("Integration Tests - HTTP API", () => {
  describe("Test 7.3: HTTP API - Health Endpoint", () => {
    it("should return healthy status", async () => {
      const response = await app
        .handle(new Request("http://localhost/api/v1/health"));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.status).toBe("healthy");
      expect(data.version).toBeDefined();
      expect(data.uptime).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Test 7.4: HTTP API - Compile Endpoint", () => {
    it("should compile valid program", async () => {
      const validProgram = {
        program: {
          metadata: { programId: "test_prog_001" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 3 },
              { id: "b", type: "DataSource", dataType: "natural", value: 2 },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.programId).toBe("test_prog_001");
      expect(data.errors).toEqual([]);
      expect(data.warnings).toEqual([]);
    });

    it("should detect cycle in program", async () => {
      const cyclicProgram = {
        program: {
          metadata: { programId: "test_prog_002" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 5 },
              { id: "b", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "c"] },
              { id: "c", type: "Transformation", dataType: "natural", operation: "MULTIPLY", inputs: ["b", "d"] },
              { id: "d", type: "DataSource", dataType: "natural", value: 2 },
              { id: "result", type: "Output", dataType: "natural", input: "c" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "b", toPort: 0 },
              { id: "edge_1", from: "c", to: "b", toPort: 1 },
              { id: "edge_2", from: "b", to: "c", toPort: 0 },
              { id: "edge_3", from: "d", to: "c", toPort: 1 },
              { id: "edge_4", from: "c", to: "result" }
            ]
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(cyclicProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(false);
      expect(data.errors.length).toBeGreaterThan(0);
      
      const cycleError = data.errors.find((e: any) => e.code === "CYCLE_DETECTED");
      expect(cycleError).toBeDefined();
      expect(cycleError.childMessage).toContain("ciclo");
    });

    it("should detect type mismatch", async () => {
      const typeMismatchProgram = {
        program: {
          metadata: { programId: "test_prog_003" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 5 },
              { id: "b", type: "DataSource", dataType: "text", value: { kind: "text", value: "hello" } },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(typeMismatchProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(false);
      expect(data.errors.length).toBeGreaterThan(0);
      
      const typeError = data.errors.find((e: any) => e.code === "TYPE_ERROR");
      expect(typeError).toBeDefined();
      expect(typeError.childMessage).toBeDefined();
    });

    it("should handle program with no output nodes", async () => {
      const noOutputProgram = {
        program: {
          metadata: { programId: "test_prog_004" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 5 },
              { id: "b", type: "DataSource", dataType: "natural", value: 3 }
            ],
            edges: []
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(noOutputProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
    });
  });

  describe("Test 7.5: HTTP API - Execute Endpoint", () => {
    it("should execute valid program and return results", async () => {
      const validProgram = {
        program: {
          metadata: { programId: "test_prog_005" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 3 },
              { id: "b", type: "DataSource", dataType: "natural", value: 2 },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.outputs).toBeDefined();
      expect(data.outputs.length).toBe(1);
      expect(data.outputs[0]).toEqual({ kind: "natural", value: 5 });
    });

    it("should include execution trace when requested", async () => {
      const validProgram = {
        program: {
          metadata: { programId: "test_prog_006" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 3 },
              { id: "b", type: "DataSource", dataType: "natural", value: 2 },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        },
        options: {
          includeTrace: true,
          traceLevel: "detailed"
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.trace).toBeDefined();
      expect(data.trace.executionOrder).toBeDefined();
      expect(data.trace.nodeEvaluations).toBeDefined();
      expect(data.trace.cacheHits).toBeDefined();
      expect(data.trace.cacheMisses).toBeDefined();
      expect(data.trace.totalTime).toBeDefined();
    });

    it("should not include trace by default", async () => {
      const validProgram = {
        program: {
          metadata: { programId: "test_prog_007" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 3 },
              { id: "b", type: "DataSource", dataType: "natural", value: 2 },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validProgram)
        }));

      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.trace).toBeUndefined();
    });

    it("should return correct execution trace format", async () => {
      const validProgram = {
        program: {
          metadata: { programId: "test_prog_008" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 3 },
              { id: "b", type: "DataSource", dataType: "natural", value: 2 },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        },
        options: {
          includeTrace: true
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validProgram)
        }));

      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.trace).toBeDefined();
      expect(Array.isArray(data.trace.executionOrder)).toBe(true);
      expect(data.trace.executionOrder.length).toBeGreaterThan(0);
      expect(typeof data.trace.nodeEvaluations).toBe("object");
      expect(data.trace.nodeEvaluations.a).toBeDefined();
      expect(data.trace.cacheHits).toBeDefined();
      expect(data.trace.cacheMisses).toBeDefined();
      expect(data.trace.totalTime).toBeDefined();
    });

    it("should return correct execution trace format", async () => {
      const validProgram = {
        program: {
          metadata: { programId: "test_prog_009" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 3 },
              { id: "b", type: "DataSource", dataType: "natural", value: 2 },
              { id: "sum", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "b"] },
              { id: "result", type: "Output", dataType: "natural", input: "sum" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "sum", toPort: 0 },
              { id: "edge_1", from: "b", to: "sum", toPort: 1 },
              { id: "edge_2", from: "sum", to: "result" }
            ]
          }
        },
        options: {
          includeTrace: true
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validProgram)
        }));

      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.trace).toBeDefined();
      expect(Array.isArray(data.trace.executionOrder)).toBe(true);
      expect(data.trace.executionOrder.length).toBeGreaterThan(0);
      expect(typeof data.trace.nodeEvaluations).toBe("object");
      expect(data.trace.nodeEvaluations.a).toBeDefined();
      expect(data.trace.cacheHits).toBeDefined();
      expect(data.trace.cacheMisses).toBeDefined();
      expect(data.trace.totalTime).toBeDefined();
    });

    it("should return validation errors for invalid program", async () => {
      const invalidProgram = {
        program: {
          metadata: { programId: "test_prog_008" },
          graph: {
            nodes: [
              { id: "a", type: "DataSource", dataType: "natural", value: 5 },
              { id: "b", type: "Transformation", dataType: "natural", operation: "ADD", inputs: ["a", "c"] },
              { id: "c", type: "Transformation", dataType: "natural", operation: "MULTIPLY", inputs: ["b", "d"] },
              { id: "d", type: "DataSource", dataType: "natural", value: 2 },
              { id: "result", type: "Output", dataType: "natural", input: "c" }
            ],
            edges: [
              { id: "edge_0", from: "a", to: "b", toPort: 0 },
              { id: "edge_1", from: "c", to: "b", toPort: 1 },
              { id: "edge_2", from: "b", to: "c", toPort: 0 },
              { id: "edge_3", from: "d", to: "c", toPort: 1 },
              { id: "edge_4", from: "c", to: "result" }
            ]
          }
        }
      };

      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(invalidProgram)
        }));

      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(false);
      expect(data.errors.length).toBeGreaterThan(0);
    });
  });

  describe("Test 7.6: HTTP API - Error Handling", () => {
    it("should handle malformed JSON", async () => {
      const response = await app
        .handle(new Request("http://localhost/api/v1/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "invalid json"
        }));

      expect(response.status).toBe(400);
    });
  });

  describe("Test 7.7: HTTP API - Performance", () => {
    it("should compile 100-node program in under 100ms", async () => {
      const nodeCount = 100;
      const nodes: any[] = [];
      const edges: any[] = [];
      
      const sourceCount = Math.floor(nodeCount * 0.5);
      const transformCount = nodeCount - sourceCount - 1;
      
      for (let i = 0; i < sourceCount; i++) {
        nodes.push({
          id: `src${i}`,
          type: "DataSource",
          dataType: "natural",
          value: i
        });
      }
      
      for (let i = 0; i < transformCount; i++) {
        const input1 = `src${i % sourceCount}`;
        const input2 = `src${(i + 1) % sourceCount}`;
        nodes.push({
          id: `t${i}`,
          type: "Transformation",
          dataType: "natural",
          operation: "ADD",
          inputs: [input1, input2]
        });
        edges.push({ id: `edge_${i}`, from: input1, to: `t${i}`, toPort: 0 });
        edges.push({ id: `edge_${i}_b`, from: input2, to: `t${i}`, toPort: 1 });
      }
      
      nodes.push({
        id: "result",
        type: "Output",
        dataType: "natural",
        input: `t${transformCount - 1}`
      });
      edges.push({ id: "edge_final", from: `t${transformCount - 1}`, to: "result" });
      
      const program = {
        program: {
          metadata: { programId: "test_prog_perf" },
          graph: { nodes, edges }
        }
      };

      const start = performance.now();
      const response = await app
        .handle(new Request("http://localhost/api/v1/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(program)
        }));
      const time = performance.now() - start;
      
      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(time).toBeLessThan(100);
    });

    it("should execute 50-node program in under 50ms", async () => {
      const nodeCount = 50;
      const nodes: any[] = [];
      const edges: any[] = [];
      
      const sourceCount = Math.floor(nodeCount * 0.5);
      const transformCount = nodeCount - sourceCount - 1;
      
      for (let i = 0; i < sourceCount; i++) {
        nodes.push({
          id: `src${i}`,
          type: "DataSource",
          dataType: "natural",
          value: i
        });
      }
      
      for (let i = 0; i < transformCount; i++) {
        const input1 = `src${i % sourceCount}`;
        const input2 = `src${(i + 1) % sourceCount}`;
        nodes.push({
          id: `t${i}`,
          type: "Transformation",
          dataType: "natural",
          operation: "ADD",
          inputs: [input1, input2]
        });
        edges.push({ id: `edge_${i}`, from: input1, to: `t${i}`, toPort: 0 });
        edges.push({ id: `edge_${i}_b`, from: input2, to: `t${i}`, toPort: 1 });
      }
      
      nodes.push({
        id: "result",
        type: "Output",
        dataType: "natural",
        input: `t${transformCount - 1}`
      });
      edges.push({ id: "edge_final", from: `t${transformCount - 1}`, to: "result" });
      
      const program = {
        program: {
          metadata: { programId: "test_prog_exec" },
          graph: { nodes, edges }
        }
      };

      const start = performance.now();
      const response = await app
        .handle(new Request("http://localhost/api/v1/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(program)
        }));
      const time = performance.now() - start;
      
      const data = await response.json();
      
      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(time).toBeLessThan(50);
    });
  });
});