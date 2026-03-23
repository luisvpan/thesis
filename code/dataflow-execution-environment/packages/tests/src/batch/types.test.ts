import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Types", () => {
  describe("Test 2.1: Type Mismatch Detection", () => {
    it("should detect type mismatch at compile time", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source a: natural = 5;
        source b: text = "hello";
        transform sum: natural = ADD(a, b);

        output result: natural = sum;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
    });

    it("should provide child-friendly Spanish error message", () => {
      const compiler = new Compiler();

      const source = `
        source a: natural = 5;
        source b: text = "hello";
        transform sum: natural = ADD(a, b);

        output result: natural = sum;
      `;

      const compileResult = compiler.compile(source);
      const typeError = compileResult.errors.find((e) => e.code === "TYPE_ERROR");
      expect(typeError?.childMessage).toBeDefined();
    });
  });
});
