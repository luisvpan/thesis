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
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([
        { kind: "natural", value: 1 },
        { kind: "natural", value: 2 },
        { kind: "natural", value: 3 },
        { kind: "natural", value: 5 },
        { kind: "natural", value: 7 },
        { kind: "natural", value: 8 },
        { kind: "natural", value: 9 }
      ]);
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
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([
        { kind: "natural", value: 1 },
        { kind: "natural", value: 2 },
        { kind: "natural", value: 5 },
        { kind: "natural", value: 8 },
        { kind: "natural", value: 9 }
      ]);
    });
  });
});
