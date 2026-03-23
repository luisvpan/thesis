import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Types", () => {
  describe("Test 2.2: Property Requirement Validation", () => {
    it("should validate property requirements for operations", () => {
      const compiler = new Compiler();

      const source = `
        source numbers: set<natural> = {1, 2, 3};
        source color: text = "red";
        transform filtered: set<natural> = FILTER_BY_COLOR(numbers, color);

        output result: set<natural> = filtered;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);

      const typeError = compileResult.errors.find((e) => e.code === "TYPE_ERROR");
      expect(typeError).toBeDefined();
      expect(typeError?.message).toContain("'color'");
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
      expect(typeError?.example).toBeDefined();
    });
  });
});
