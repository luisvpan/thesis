import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { serialize, deserialize } from "../index";
import type { Program, SourceStatement, TransformStatement, SinkStatement, DataLiteral } from "../index";

describe("serialize", () => {
  test("parses CPA abstracto (number in v3.1.0)", () => {
    const input = 'source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};';
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    expect(result.program).not.toBeNull();
    expect(result.program!.statements).toHaveLength(1);

    const stmt = result.program!.statements[0] as SourceStatement;
    expect(stmt.type).toBe("SourceStatement");
    expect(stmt.identifier).toBe("x");
    expect(stmt.value.type).toBe("DataLiteral");

    const obj = stmt.value as DataLiteral;
    expect(obj.category).toBe("abstracto");
    expect(obj.objType).toBe("numero");
    expect(obj.subtype).toBe("racional");
    expect(obj.quantity).toBeInstanceOf(Fraction);
    expect(obj.quantity.equals(new Fraction(5))).toBe(true);
  });

  test("parses object literal with quantity as Fraction", () => {
    const input = 'source shapes = {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 3, "size": "grande"};';
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    const stmt = result.program!.statements[0] as SourceStatement;
    expect(stmt.value.type).toBe("DataLiteral");

    const obj = stmt.value as DataLiteral;
    expect(obj.category).toBe("pictorico");
    expect(obj.objType).toBe("forma");
    expect(obj.subtype).toBe("circulo");
    expect(obj.quantity).toBeInstanceOf(Fraction);
    expect(obj.quantity.equals(new Fraction(3))).toBe(true);
    expect(obj.attributes.size).toBe("grande");
  });

  test("parses abstract object with quantity as Fraction", () => {
    const input = 'source num = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 42};';
    const result = serialize(input);

    expect(result.errors).toHaveLength(0);
    const stmt = result.program!.statements[0] as SourceStatement;
    const obj = stmt.value as DataLiteral;
    expect(obj.quantity).toBeInstanceOf(Fraction);
    expect(obj.quantity.equals(new Fraction(42))).toBe(true);
  });

  test("parses transform statement", () => {
    const input = `
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
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
      source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
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
    const input = "source = 5;";  // Missing identifier
    const result = serialize(input);

    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.program).toBeNull();
  });
});

describe("deserialize", () => {
  test("deserializes DataLiteral (CPA abstracto)", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "x",
          value: {
            type: "DataLiteral",
            sourceType: "data",
            category: "abstracto",
            objType: "numero",
            subtype: "racional",
            quantity: new Fraction(5),
            attributes: {},
          },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toContain("source x =");
    expect(result).toContain("category");
    expect(result).toContain("abstracto");
    expect(result).toContain("5");
  });

  test("deserializes DataLiteral with attributes", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "shapes",
          value: {
            type: "DataLiteral",
            sourceType: "data",
            category: "pictorico",
            objType: "forma",
            subtype: "circulo",
            quantity: new Fraction(3),
            attributes: { size: "grande" },
          },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toContain("source shapes =");
    expect(result).toContain("category");
    expect(result).toContain("pictorico");
    expect(result).toContain("size");
    expect(result).toContain("grande");
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

  test("deserializes array literal with DataLiterals", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "arr",
          value: {
            type: "ArrayLiteral",
            elements: [
              {
                type: "DataLiteral",
                sourceType: "data",
                category: "abstracto",
                objType: "numero",
                subtype: "racional",
                quantity: new Fraction(1),
                attributes: {},
              },
              {
                type: "DataLiteral",
                sourceType: "data",
                category: "abstracto",
                objType: "numero",
                subtype: "racional",
                quantity: new Fraction(2),
                attributes: {},
              },
            ],
          },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toContain("source arr =");
    expect(result).toContain("[");
    expect(result).toContain("]");
  });

  test("deserializes other literal", () => {
    const program: Program = {
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "size",
          value: { type: "OtherLiteral", value: "grande" },
        },
      ],
    };

    const result = deserialize(program);
    expect(result).toBe('source size = "grande";');
  });
});

describe("roundtrip", () => {
  test("serialize -> deserialize produces equivalent program", () => {
    const original = `source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
source y = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
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

    const sourceCode = `source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};
source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
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
    // Both should be CPA objects with quantity 15
    expect((value1 as any).kind).toBe("cpa");
    expect((value2 as any).kind).toBe("cpa");
    expect((value1 as any).quantity.equals(new Fraction(15))).toBe(true);
    expect((value2 as any).quantity.equals(new Fraction(15))).toBe(true);
  });
});
