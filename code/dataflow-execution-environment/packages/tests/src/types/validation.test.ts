import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Type Validation", () => {
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

      // Compile should fail
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors).not.toHaveLength(0);

      // Find TYPE_ERROR
      const typeError = compileResult.errors.find((e) => e.code === "TYPE_ERROR");
      expect(typeError).toBeDefined();
      expect(typeError?.nodeId).toBe("sum");

      // Child-friendly Spanish message should be present
      expect(typeError?.childMessage).toBeDefined();
      expect(typeError?.childMessage).toContain("ADD");
      expect(typeError?.childMessage).toContain("números");

      // Suggestion should be actionable
      expect(typeError?.suggestion).toBeDefined();

      // Example should show correct usage
      expect(typeError?.example).toBeDefined();

      // Runtime should NOT be called
      expect(compileResult.program).toBeUndefined();
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

      expect(typeError?.childMessage).toMatch(/⚠️|¡Ups!/);
      expect(typeError?.childMessage).toMatch(/números|palabras/);
      expect(typeError?.message).toContain("ADD");
    });
  });

  describe("Test 2.2: Property Requirement Validation", () => {
    it("should validate property requirements for operations", () => {
      const compiler = new Compiler();

      const source = `
        source numbers: set<natural> = {1, 2, 3};
        source color: text = "red";
        transform filtered: set<natural> = FILTER_BY_COLOR(numbers, color);

        output result: set<natural> = filtered;
      `;

      // Compile should fail
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);

      // Find TYPE_ERROR for missing property
      const typeError = compileResult.errors.find((e) => e.code === "TYPE_ERROR");
      expect(typeError).toBeDefined();
      expect(typeError?.nodeId).toBe("filtered");

      // Error should mention missing 'color' property
      expect(typeError?.message).toContain("'color'");
      expect(typeError?.message).toContain("property");

      // Child-friendly Spanish message
      expect(typeError?.childMessage).toContain("color");
    });

    it("should provide clear guidance on property requirements", () => {
      const compiler = new Compiler();

      const source = `
        source numbers: set<natural> = {1, 2, 3};
        source color: text = "red";
        transform filtered: set<natural> = FILTER_BY_COLOR(numbers, color);

        output result: set<natural> = filtered;
      `;

      const compileResult = compiler.compile(source);
      const typeError = compileResult.errors.find((e) => e.code === "TYPE_ERROR");

      expect(typeError?.suggestion).toBeDefined();
      expect(typeError?.suggestion).toContain("color");
      expect(typeError?.example).toBeDefined();
    });
  });
});
