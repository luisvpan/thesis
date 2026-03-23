import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";

describe("Integration Tests - Errors", () => {
  describe("Test 6.2: Cycle Detection", () => {
    it("should detect cycles at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source a: natural = 5;
        transform b: natural = ADD(a, c);
        transform c: natural = MULTIPLY(b, 2);

        output result: natural = c;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);

      const cycleError = compileResult.errors.find((e) => e.code === "CYCLE_DETECTED");
      expect(cycleError).toBeDefined();
      expect(cycleError?.childMessage).toContain("ciclo");
    });
  });
});
