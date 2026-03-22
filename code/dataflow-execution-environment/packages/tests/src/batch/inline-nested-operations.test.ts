import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Inline Nested Operations", () => {
  it("should parse and execute ADD(ADD(a, b), c)", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 1;

      transform result: natural = ADD(ADD(a, b), c);

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = await runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 6 }]);
    }
  });

  it("should parse and execute MULTIPLY(ADD(a, b), ADD(c, d))", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 10;
      source d: natural = 6;

      transform result: natural = MULTIPLY(ADD(a, b), ADD(c, d));

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = await runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 80 }]);
    }
  });

  it("should parse deeply nested inline expressions (3+ levels)", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 1;
      source b: natural = 2;
      source c: natural = 3;
      source d: natural = 4;

      transform result: natural = ADD(ADD(ADD(a, b), c), d);

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = await runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 10 }]);
    }
  });

  it("should maintain type checking for inline nested operations", () => {
    const compiler = new Compiler();

    const source = `
      source a: natural = 5;
      source b: text = "hello";
      source c: natural = 3;

      transform result: natural = ADD(a, b);

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(false);
    expect(compileResult.errors.length).toBeGreaterThan(0);
    expect(compileResult.errors[0].code).toBe("TYPE_ERROR");
  });

  it("should handle inline nested operations with literals", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      source a: natural = 5;

      transform result: natural = ADD(a, 10);

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = await runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 15 }]);
    }
  });

  it("should handle inline nested operations with nested literals", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
      transform result: natural = ADD(ADD(1, 2), ADD(3, 4));

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = await runtime.execute(0);
      expect(outputs).toEqual([{ kind: "natural", value: 10 }]);
    }
  });

  it("should create proper graph structure for inline nested operations", () => {
    const compiler = new Compiler();

    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 1;

      transform result: natural = ADD(ADD(a, b), c);

      output final: natural = result;
    `;

    const compileResult = compiler.compile(source);

    if (compileResult.success && compileResult.program) {
      const nodes = compileResult.program.graph.nodes;
      const edges = compileResult.program.graph.edges;

      expect(nodes.some(n => n.id.startsWith("nested_op_"))).toBe(true);
      expect(edges.some(e => e.from.startsWith("nested_op_"))).toBe(true);
    }
  });
});
