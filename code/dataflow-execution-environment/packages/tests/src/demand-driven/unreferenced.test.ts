import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Demand-Driven", () => {
  describe("Test 8.2: Unreferenced Nodes", () => {
    it("should skip unreferenced computation", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 5;
        source b: natural = 3;
        source c: natural = 10;
        transform sum: natural = ADD(a, b);
        transform unused: natural = MULTIPLY(c, 2);

        output result: natural = sum;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);

      const evaluator = runtime.getEvaluator();
      const stats = evaluator.getCacheStats();
      // Only 3 nodes evaluated (a, b, sum), not c or unused
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBeGreaterThanOrEqual(3);
    });
  });
});
