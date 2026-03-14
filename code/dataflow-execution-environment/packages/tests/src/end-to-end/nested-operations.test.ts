import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Nested Operations", () => {
  it("should parse and execute ADD(ADD(a, b), c)", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 1;

      transform inner_sum: natural = ADD(a, b);
      transform final_sum: natural = ADD(inner_sum, c);

      output result: natural = final_sum;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 6 }]);
    }
  });

  it("should parse and execute MULTIPLY(ADD(a, b), ADD(c, d))", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 10;
      source d: natural = 6;

      transform sum: natural = ADD(a, b);
      transform sum2: natural = ADD(c, d);
      transform product: natural = MULTIPLY(sum, sum2);

      output result: natural = product;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 80 }]);
    }
  });

  it("should parse deeply nested expressions (3+ levels)", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 1;
      source b: natural = 2;
      source c: natural = 3;
      source d: natural = 4;

      transform level1: natural = ADD(a, b);
      transform level2: natural = ADD(level1, c);
      transform level3: natural = ADD(level2, d);

      output result: natural = level3;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 10 }]);
    }
  });

  it("should maintain type checking for nested operations", () => {
    const compiler = new Compiler();

    const source = `
      source a: natural = 5;
      source b: text = "hello";
      source c: natural = 3;

      transform invalid_sum: natural = ADD(a, b);

      output result: natural = invalid_sum;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(false);
    expect(compileResult.errors.length).toBeGreaterThan(0);
    expect(compileResult.errors[0].code).toBe("TYPE_ERROR");
  });

  it("should handle nested operations with set operations", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source set1: set<natural> = {1, 2, 3};
      source set2: set<natural> = {4, 5, 6};

      transform union_set: set<natural> = UNION(set1, set2);

      output result: set<natural> = union_set;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      expect(outputs[0]).toMatchObject({ kind: "set" });
      const setOutput = outputs[0] as { kind: string; elements: unknown[] };
      expect(setOutput.elements.length).toBe(6);
    }
  });
});
