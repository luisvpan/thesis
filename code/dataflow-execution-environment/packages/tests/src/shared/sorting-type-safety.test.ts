import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Type Safety - Sorting Operations", () => {
  describe("SORT with Numeric Types", () => {
    it("should sort Natural numbers", () => {
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

    it("should sort Integer numbers", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<integer> = {-5, 2, -3, 8, -1, 0};

        transform sorted: set<integer> = SORT(numbers);

        output result: set<integer> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([
        { kind: "integer", value: -5 },
        { kind: "integer", value: -3 },
        { kind: "integer", value: -1 },
        { kind: "integer", value: 0 },
        { kind: "integer", value: 2 },
        { kind: "integer", value: 8 }
      ]);
    });

    it("should sort Decimal numbers", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: set<decimal> = {1.5, 2.5, 0.5, 3.5, 4.5};

        transform sorted: set<decimal> = SORT(numbers);

        output result: set<decimal> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([
        { kind: "decimal", value: 0.5 },
        { kind: "decimal", value: 1.5 },
        { kind: "decimal", value: 2.5 },
        { kind: "decimal", value: 3.5 },
        { kind: "decimal", value: 4.5 }
      ]);
    });

    it("should sort Fractions", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source fractions: set<fraction> = {1/2, 3/4, 1/3, 5/6};

        transform sorted: set<fraction> = SORT(fractions);

        output result: set<fraction> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([
        { kind: "fraction", numerator: 1, denominator: 3 },
        { kind: "fraction", numerator: 1, denominator: 2 },
        { kind: "fraction", numerator: 3, denominator: 4 },
        { kind: "fraction", numerator: 5, denominator: 6 }
      ]);
    });
  });

  describe("SORT Type Validation", () => {
    it("should reject SORT with Text type at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source words: set<text> = {"zebra", "apple", "banana"};

        transform sorted: set<text> = SORT(words);

        output result: set<text> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
      
      const error = compileResult.errors[0];
      expect(error.code).toBe("TYPE_ERROR");
      expect(error.childMessage).toMatch(/solo funciona con números/i);
    });

    it("should reject SORT with Boolean type at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source values: set<boolean> = {true, false, true};

        transform sorted: set<boolean> = SORT(values);

        output result: set<boolean> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
      
      const error = compileResult.errors[0];
      expect(error.code).toBe("TYPE_ERROR");
      expect(error.childMessage).toMatch(/solo funciona con números/i);
    });

    it("should reject SORT with Shape type at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source shapes: set<shape> = {
          {type: "circle", size: "large", color: "red"},
          {type: "triangle", size: "small", color: "blue"}
        };

        transform sorted: set<shape> = SORT(shapes);

        output result: set<shape> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
      
      const error = compileResult.errors[0];
      expect(error.code).toBe("TYPE_ERROR");
      expect(error.childMessage).toMatch(/solo funciona con números/i);
    });
  });

  describe("ALPHABETICAL_SORT with Text Type", () => {
    it("should sort text alphabetically", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source words: set<text> = {"zebra", "apple", "banana", "cherry", "date"};

        transform sorted: set<text> = ALPHABETICAL_SORT(words);

        output result: set<text> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toEqual([
        { kind: "text", value: "apple" },
        { kind: "text", value: "banana" },
        { kind: "text", value: "cherry" },
        { kind: "text", value: "date" },
        { kind: "text", value: "zebra" }
      ]);
    });

    it("should handle case-sensitive alphabetical sort", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source words: set<text> = {"Apple", "banana", "Cherry", "date"};

        transform sorted: set<text> = ALPHABETICAL_SORT(words);

        output result: set<text> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      
      const sortedValues = result.elements.map((e: any) => e.value as string);
      const expectedValues = ["Apple", "banana", "Cherry", "date"].sort((a, b) => a.localeCompare(b));
      expect(sortedValues).toEqual(expectedValues);
    });
  });

  describe("ALPHABETICAL_SORT Type Validation", () => {
    it("should reject ALPHABETICAL_SORT with Natural type at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source numbers: set<natural> = {5, 2, 8, 1};

        transform sorted: set<natural> = ALPHABETICAL_SORT(numbers);

        output result: set<natural> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
      
      const error = compileResult.errors[0];
      expect(error.code).toBe("TYPE_ERROR");
      expect(error.childMessage).toMatch(/solo funciona con texto/i);
    });

    it("should reject ALPHABETICAL_SORT with Integer type at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source numbers: set<integer> = {-5, 2, -3, 8};

        transform sorted: set<integer> = ALPHABETICAL_SORT(numbers);

        output result: set<integer> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
      
      const error = compileResult.errors[0];
      expect(error.code).toBe("TYPE_ERROR");
      expect(error.childMessage).toMatch(/solo funciona con texto/i);
    });

    it("should reject ALPHABETICAL_SORT with Shape type at compile time", () => {
      const compiler = new Compiler();

      const source = `
        source shapes: set<shape> = {
          {type: "circle", size: "large", color: "red"},
          {type: "triangle", size: "small", color: "blue"}
        };

        transform sorted: set<shape> = ALPHABETICAL_SORT(shapes);

        output result: set<shape> = sorted;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(false);
      expect(compileResult.errors.length).toBeGreaterThan(0);
      
      const error = compileResult.errors[0];
      expect(error.code).toBe("TYPE_ERROR");
      expect(error.childMessage).toMatch(/solo funciona con texto/i);
    });
  });
});
