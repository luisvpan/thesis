import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Curriculum Types", () => {
  describe("Test 3.2: Compare Shapes by Size", () => {
    it("should compare shapes and return Boolean", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source shape1: shape = {type: "circle", size: "small", color: "red"};
        source shape2: shape = {type: "square", size: "large", color: "blue"};

        transform equal_size: boolean = COMPARE_BY_SIZE(shape1, shape2);

        output result: boolean = equal_size;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: false }); // small ≠ large
    });

    it("should perform value-based comparison (not reference)", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source shape1: shape = {type: "circle", size: "small", color: "red"};
        source shape2: shape = {type: "square", size: "small", color: "blue"};

        transform equal_size: boolean = COMPARE_BY_SIZE(shape1, shape2);

        output result: boolean = equal_size;
      `;

      const compileResult = compiler.compile(source);
      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: true }); // both small
    });
  });
});
