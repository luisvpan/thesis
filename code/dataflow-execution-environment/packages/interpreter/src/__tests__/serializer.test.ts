import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { serialize, deserialize } from "../index";
import type { Program, SourceStatement, TransformStatement, SinkStatement, NumberLiteral } from "../index";

describe("serialize", () => {
  test("parses simple number literal", () => {
    const input = "source x = 5;";
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    expect(result.program).not.toBeNull();
    expect(result.program!.statements).toHaveLength(1);

    const stmt = result.program!.statements[0] as SourceStatement;
    expect(stmt.type).toBe("SourceStatement");
    expect(stmt.identifier).toBe("x");
    expect(stmt.value.type).toBe("NumberLiteral");

    const numLit = stmt.value as NumberLiteral;
    expect(numLit.value).toBeInstanceOf(Fraction);
    expect(numLit.value.equals(new Fraction(5))).toBe(true);
  });

  test("parses decimal number literal", () => {
    const input = "source x = 3.14;";
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    const stmt = result.program!.statements[0] as SourceStatement;
    const numLit = stmt.value as NumberLiteral;
    expect(numLit.value.equals(new Fraction(3.14))).toBe(true);
  });

  test("parses object literal with amount as Fraction", () => {
    const input = "source shapes = {category: pictorial, type: shape, subtype: circle, size: large, amount: 3};";
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    const stmt = result.program!.statements[0] as SourceStatement;
    expect(stmt.value.type).toBe("ObjectLiteral");

    const obj = stmt.value as any;
    expect(obj.amount).toBeInstanceOf(Fraction);
    expect(obj.amount.equals(new Fraction(3))).toBe(true);
  });

  test("parses abstract object with value as Fraction", () => {
    const input = "source num = {category: abstract, type: rational, value: 42};";
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    const stmt = result.program!.statements[0] as SourceStatement;
    const obj = stmt.value as any;
    expect(obj.value).toBeInstanceOf(Fraction);
    expect(obj.value.equals(new Fraction(42))).toBe(true);
  });

  test("parses transform statement", () => {
    const input = `
      source a = 5;
      source b = 3;
      transform c = sum(a, b);
    `;
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    expect(result.program!.statements).toHaveLength(3);

    const transform = result.program!.statements[2] as TransformStatement;
    expect(transform.type).toBe("TransformStatement");
    expect(transform.identifier).toBe("c");
    expect(transform.operation).toBe("sum");
    expect(transform.arguments).toHaveLength(2);
  });

  test("parses sink statement", () => {
    const input = `
      source x = 5;
      sink output = x;
    `;
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);

    const sink = result.program!.statements[1] as SinkStatement;
    expect(sink.type).toBe("SinkStatement");
    expect(sink.identifier).toBe("output");
    expect(sink.sourceIdentifier).toBe("x");
  });

  test("returns errors for invalid syntax", () => {
    const input = "source x = ;";
    const result = serialize(input);

    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.program).toBeNull();
  });
});

describe("deserialize", () => {
  test("deserializes number literal", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "x",
          value: { type: "NumberLiteral", value: new Fraction(5) },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("source x = 5;");
  });

  test("deserializes decimal number", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "x",
          value: { type: "NumberLiteral", value: new Fraction(3.14) },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("source x = 3.14;");
  });

  test("deserializes object literal with Fraction", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "shapes",
          value: {
            type: "ObjectLiteral",
            category: "pictorial",
            objectType: "shape",
            subtype: "circle",
            size: "large",
            amount: new Fraction(3),
          },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("source shapes = {category: pictorial, type: shape, subtype: circle, size: large, amount: 3};");
  });

  test("deserializes transform statement", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "TransformStatement",
          identifier: "c",
          operation: "sum",
          arguments: [
            { type: "Identifier", name: "a" },
            { type: "Identifier", name: "b" },
          ],
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("transform c = sum(a, b);");
  });

  test("deserializes sink statement", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SinkStatement",
          identifier: "output",
          sourceIdentifier: "x",
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("sink output = x;");
  });

  test("deserializes array literal", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "arr",
          value: {
            type: "ArrayLiteral",
            elements: [
              { type: "NumberLiteral", value: new Fraction(1) },
              { type: "NumberLiteral", value: new Fraction(2) },
              { type: "NumberLiteral", value: new Fraction(3) },
            ],
          },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("source arr = [1, 2, 3];");
  });

  test("deserializes other literal", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "size",
          value: { type: "OtherLiteral", value: "large" },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe("source size = large;");
  });
});

describe("roundtrip", () => {
  test("serialize -> deserialize produces equivalent program", () => {
    const original = `source x = 5;
source y = 3;
transform z = sum(x, y);
sink output = z;`;

    const serialized = serialize(original);
    expect(serialized.errors).toHaveLength(0);

    const deserialized = deserialize(serialized.program!);

    // Re-serialize the deserialized string
    const reserialized = serialize(deserialized);
    expect(reserialized.errors).toHaveLength(0);

    // Both programs should have the same structure
    expect(reserialized.program!.statements.length).toBe(serialized.program!.statements.length);
  });

  test("execute with Program produces same result as string", async () => {
    const { Interpreter } = await import("../index");

    const sourceCode = `source a = 10;
source b = 5;
transform c = sum(a, b);
sink result = c;`;

    const interpreter1 = new Interpreter();
    const result1 = await interpreter1.execute(sourceCode);

    const serialized = serialize(sourceCode);
    expect(serialized.errors).toHaveLength(0);
    expect(serialized.program).not.toBeNull();

    const interpreter2 = new Interpreter();
    const result2 = await interpreter2.execute(serialized.program!);

    // Both should produce the same result
    expect(result1.errors).toHaveLength(0);
    expect(result2.errors).toHaveLength(0);

    const value1 = result1.results.get("result");
    const value2 = result2.results.get("result");

    expect(value1).toBeDefined();
    expect(value2).toBeDefined();
    expect((value1 as any).value.equals((value2 as any).value)).toBe(true);
  });
});
