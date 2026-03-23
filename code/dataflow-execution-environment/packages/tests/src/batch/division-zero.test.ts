import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Errors", () => {
  describe("Test 6.1: Division by Zero", () => {
    it("should handle division by zero gracefully", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 10;
        source b: natural = 0;

        transform quotient: decimal = DIVIDE(a, b);

        output result: decimal = quotient;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      expect(() => runtime.execute()).toThrow("Division by zero");
    });
  });
});
