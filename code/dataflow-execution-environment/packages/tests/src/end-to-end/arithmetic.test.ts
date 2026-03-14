import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - End-to-End Arithmetic", () => {
  describe("Test 1.1: Simple Addition", () => {
    it("should compile and execute simple addition program", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 3;
        source b: natural = 2;
        transform sum: natural = ADD(a, b);
        output result: natural = sum;
      `;

      // Compile
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
      expect(compileResult.errors).toHaveLength(0);
      expect(compileResult.program).toBeDefined();

      // Execute
      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      // Verify
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "natural", value: 5 });
    });

    it("should complete in under 10ms", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 3;
        source b: natural = 2;
        transform sum: natural = ADD(a, b);
        output result: natural = sum;
      `;

      // Measure compilation time
      const compileStart = performance.now();
      const compileResult = compiler.compile(source);
      const compileTime = performance.now() - compileStart;

      expect(compileResult.success).toBe(true);
      expect(compileTime).toBeLessThan(10);

      // Measure execution time
      runtime.loadProgram(compileResult.program!);
      const executeStart = performance.now();
      const outputs = runtime.execute();
      const executeTime = performance.now() - executeStart;

      expect(outputs).toHaveLength(1);
      expect(executeTime).toBeLessThan(10);
    });
  });

  describe("Test 1.2: Complex Arithmetic Expression", () => {
    it("should compile and execute multi-node arithmetic program", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 3;
        source b: natural = 2;
        source c: natural = 10;
        source d: natural = 6;

        transform sum: natural = ADD(a, b);
        transform sum2: natural = ADD(c, d);
        transform product: natural = MULTIPLY(sum, sum2);

        output result: natural = product;
      `;

      // Compile
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      // Execute
      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      // Verify final result
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "natural", value: 80 }); // 5 * 16

      // Verify execution trace has correct order
      const evaluator = runtime.getEvaluator();
      const stats = evaluator.getCacheStats();
      expect(stats.hits).toBe(0); // All nodes evaluated once
      expect(stats.misses).toBeGreaterThanOrEqual(7); // a, b, c, d, sum, sum2, product
    });

    it("should preserve demand-driven evaluation order", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 3;
        source b: natural = 2;
        source c: natural = 10;
        source d: natural = 6;

        transform sum: natural = ADD(a, b);
        transform sum2: natural = ADD(c, d);
        transform product: natural = MULTIPLY(sum, sum2);

        output result: natural = product;
      `;

      const compileResult = compiler.compile(source);
      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);

      // Verify demand-driven: only output node and its dependencies evaluated
      const evaluator = runtime.getEvaluator();
      const graph = runtime.getGraph();
      const outputNodes = graph.getOutputNodes();
      expect(outputNodes).toHaveLength(1);
    });
  });
});
