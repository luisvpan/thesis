import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Sets", () => {
  describe("Test 4.1: Union of Shape Sets", () => {
    it("should merge sets and remove duplicates", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source set1: set<natural> = {1, 2, 3};
        source set2: set<natural> = {4, 5, 6};

        transform union: set<natural> = UNION(set1, set2);

        output result: set<natural> = union;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: number[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(6);
    });

    it("should remove duplicates when present", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source set1: set<natural> = {1, 2, 3};
        source set2: set<natural> = {3, 4, 5};

        transform union: set<natural> = UNION(set1, set2);

        output result: set<natural> = union;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: number[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(5);
    });
  });
});
