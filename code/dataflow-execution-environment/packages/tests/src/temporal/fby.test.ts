import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Temporal", () => {
  describe("Test 5.1: FBY Counter", () => {
    it("should return initial value at time 0", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source zero: natural = 0;
        source numbers: stream<natural> = stream<natural>(generator(counter));

        transform shifted: stream<natural> = FBY(zero, numbers);

        output result: stream<natural> = shifted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);

      const output = runtime.execute(0);
      expect(output).toHaveLength(1);
      expect(output[0]).toEqual({ kind: "natural", value: 0 });
    });

    it("should use demand-driven semantics for FBY", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source zero: natural = 0;
        source numbers: stream<natural> = stream<natural>(generator(counter));

        transform shifted: stream<natural> = FBY(zero, numbers);

        output result: stream<natural> = shifted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);

      const evaluator = runtime.getEvaluator();
      const initialStats = evaluator.getCacheStats();
      expect(initialStats.hits).toBe(0);
      expect(initialStats.misses).toBe(0);

      runtime.execute(0);

      const stats = evaluator.getCacheStats();
      expect(stats.misses).toBeGreaterThan(0);
    });
  });
});
