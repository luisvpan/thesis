import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Sets", () => {
  describe("Test 4.2: Filter and Sort Combined", () => {
    it("should sort a set", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<natural> = {5, 2, 8, 1, 9, 3, 7};

        transform sorted: set<natural> = SORT(numbers);

        output result: set<natural> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: number[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([1, 2, 3, 5, 7, 8, 9]);
    });

    it("should maintain order through chain", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<natural> = {5, 2, 8, 1, 9};

        transform sorted: set<natural> = SORT(numbers);

        output result: set<natural> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: number[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([1, 2, 5, 8, 9]);
    });
  });
});
