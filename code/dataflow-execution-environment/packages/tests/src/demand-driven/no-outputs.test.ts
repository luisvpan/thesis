import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Demand-Driven", () => {
  describe("Test 8.1: No Output Nodes", () => {
    it("should evaluate nothing when no output nodes", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 5;
        source b: natural = 3;
        transform sum: natural = ADD(a, b);
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(0); // Nothing evaluated

      const evaluator = runtime.getEvaluator();
      const stats = evaluator.getCacheStats();
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0); // No nodes evaluated
    });
  });
});
