import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Fraction Operations (P0.2 - Overloading Support)", () => {
  
  it("should compile ADD with fraction inputs via overload", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
source a: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    source b: fraction = { kind: "fraction", numerator: 3, denominator: 4 };
    
    transform result: fraction = ADD(a, b);
    output final: fraction = result;
  `;

    output final: fraction = result;
  `;

const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "fraction", numerator: 1, denominator: 2 });
    }
  });

  it("should compile SUBTRACT with fraction inputs via overload", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
source a: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    source b: fraction = { kind: " fraction", numerator: 3, denominator: 4 };
    
    transform result: fraction = SUBTRACT(a, b);
    output final: fraction = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "fraction", numerator: 2, denominator: 3 });
    }
  });

  it("should compile MULTIPLY with fraction inputs via overload", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
source a: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    source b: fraction = { kind: "fraction", numerator: 4, denominator: 5 };
    
    transform result: fraction = MULTIPLY(a, b);
    output final: fraction = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "fraction", numerator: 5, denominator: 10 });
    }
  });

  it("should compile DIVIDE with fraction inputs via overload", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
source a: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    source b: fraction = { kind: "fraction", numerator: 3, denominator: 4 };
    
    transform result: fraction = DIVIDE(a, b);
    output final: fraction = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "fraction", numerator: 5, denominator: 2 });
    }
  });

  it("should compile COMPARE with fraction inputs via overload", () => {
    const compiler = new Compiler();
    const runtime = new Runtime();

    const source = `
source a: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    source b: fraction = { kind: "fraction", numerator: 1, denominator: 3 };
    
    transform result: fraction = COMPARE(a, b);
    output final: fraction = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);

    if (compileResult.success && compileResult.program) {
      runtime.loadProgram(compileResult.program);
      const outputs = runtime.execute(0);
      
      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: true });
    }
  });

  it("should detect type mismatch for mixed natural and fraction", () => {
    const compiler = new Compiler();

    const source = `
source a: natural = 5;
source b: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    
    transform result: natural = ADD(a, b);
    output final: natural = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(false);
    expect(compileResult.errors.length).toBeGreaterThan(0);
    expect(compileResult.errors[0].code).toBe("TYPE_ERROR");
    expect(compileResult.errors[0].message).toContain("ADD expects natural or fraction");
    expect(compileResult.errors[0].childMessage).toBeDefined();
    expect(compileResult.errors[0].suggestion).toContain("Conecta un bloque de tipo natural");
  });

  it("should detect type mismatch for integer and fraction", () => {
    const compiler = new Compiler();

    const source = `
source a: integer = -3;
source b: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    
    transform result: integer = SUBTRACT(a, b);
    output final: integer = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(false);
    expect(compileResult.errors.length).toBeGreaterThan(0);
    expect(compileResult.errors[0].code).toBe("TYPE_ERROR");
    expect(compileResult.errors[0].message).toContain("SUBTRACT expects integer or fraction");
    expect(compileResult.errors[0].childMessage).toBeDefined();
    expect(compileResult.errors[0].suggestion).toContain("Conecta un bloque de tipo entero");
  });

  it("should detect type mismatch for boolean and fraction", () => {
    const compiler = new Compiler();

    const source = `source a: boolean = false`;
source b: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
    
    transform result: boolean = AND(a, b);
    output final: boolean = result;
  `;

    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    expect(compileResult.errors).toEqual([]);
  });
});
