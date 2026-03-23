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

  describe("Test 2.3: Set Homogeneity Validation", () => {
    it("should reject set with mixed types", () => {
      const compiler = new Compiler();

      const source = `
        source mixed: set<natural> = {1, "hello", 3};
        output result: set<natural> = mixed;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);

      const homogeneityError = compileResult.errors.find((e) => e.code === "SET_HETEROGENEITY_ERROR");
      expect(homogeneityError).toBeDefined();
      expect(homogeneityError?.childMessage).toContain("mismo tipo");
      expect(homogeneityError?.suggestion).toBeDefined();
      expect(homogeneityError?.example).toBeDefined();
    });

    it("should accept set with same types", () => {
      const compiler = new Compiler();

      const source = `
        source numbers: set<natural> = {1, 2, 3};
        output result: set<natural> = numbers;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
    });

    it("should accept empty set", () => {
      const compiler = new Compiler();

      const source = `
        source empty: set<natural> = {};
        output result: set<natural> = empty;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
    });
  });

  describe("Test 2.5: Literal Type Validation", () => {
    it("should reject literal value that doesn't match declared type", () => {
      const compiler = new Compiler();

      const source = `
        source textNum: natural = "hello";
        output result: natural = textNum;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);

      const literalTypeError = compileResult.errors.find((e) => e.code === "LITERAL_TYPE_MISMATCH");
      expect(literalTypeError).toBeDefined();
      expect(literalTypeError?.childMessage).toContain("no coincide");
      expect(literalTypeError?.suggestion).toBeDefined();
    });

    it("should accept literal value that matches declared type", () => {
      const compiler = new Compiler();

      const source = `
        source num: natural = 5;
        output result: natural = num;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
    });

    it("should reject set element type mismatch", () => {
      const compiler = new Compiler();

      const source = `
        source strings: set<natural> = {"a", "b", "c"};
        output result: set<natural> = strings;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);

      const literalTypeError = compileResult.errors.find((e) => e.code === "LITERAL_TYPE_MISMATCH");
      expect(literalTypeError).toBeDefined();
    });
  });
});
