import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { lessThan, greaterThan } from "../comparison";
import type { CPAObject, ArrayValue } from "../../runtime/types";
import { createAbstractNumber } from "../../utils";

describe("lessThan (unit)", () => {
  test("filters abstract numbers less than threshold", () => {
    const values = [
      createAbstractNumber(1),
      createAbstractNumber(5),
      createAbstractNumber(3),
      createAbstractNumber(8),
    ];
    const threshold = createAbstractNumber(4);
    const result = lessThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(1))).toBe(true);
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(3))).toBe(true);
  });

  test("filters CPA objects by quantity less than threshold", () => {
    const shapes: CPAObject[] = [
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(2), attributes: { size: "grande" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "cuadrado", quantity: new Fraction(10), attributes: { size: "pequeño" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(5), attributes: { size: "mediano" } },
    ];
    const threshold = createAbstractNumber(6);
    const result = lessThan([...shapes, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("returns single element when only one matches", () => {
    const values = [
      createAbstractNumber(10),
      createAbstractNumber(1),
    ];
    const threshold = createAbstractNumber(5);
    const result = lessThan([...values, threshold]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(1))).toBe(true);
  });

  test("returns empty array when nothing matches", () => {
    const values = [
      createAbstractNumber(10),
      createAbstractNumber(20),
    ];
    const threshold = createAbstractNumber(5);
    const result = lessThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(0);
  });

  test("throws error for insufficient arguments", () => {
    const a = createAbstractNumber(5);
    expect(() => lessThan([a])).toThrow();
  });
});

describe("greaterThan (unit)", () => {
  test("filters abstract numbers greater than threshold", () => {
    const values = [
      createAbstractNumber(1),
      createAbstractNumber(5),
      createAbstractNumber(3),
      createAbstractNumber(8),
    ];
    const threshold = createAbstractNumber(4);
    const result = greaterThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(5))).toBe(true);
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(8))).toBe(true);
  });

  test("filters CPA objects by quantity greater than threshold", () => {
    const foods: CPAObject[] = [
      { kind: "cpa", category: "concreto", type: "comida", subtype: "uva", quantity: new Fraction(2), attributes: { color: "morado" } },
      { kind: "cpa", category: "concreto", type: "comida", subtype: "manzana", quantity: new Fraction(10), attributes: { color: "rojo" } },
      { kind: "cpa", category: "concreto", type: "comida", subtype: "pera", quantity: new Fraction(5), attributes: { color: "verde" } },
    ];
    const threshold = createAbstractNumber(4);
    const result = greaterThan([...foods, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("returns single element when only one matches", () => {
    const values = [
      createAbstractNumber(1),
      createAbstractNumber(10),
    ];
    const threshold = createAbstractNumber(5);
    const result = greaterThan([...values, threshold]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(10))).toBe(true);
  });

  test("returns empty array when nothing matches", () => {
    const values = [
      createAbstractNumber(1),
      createAbstractNumber(2),
    ];
    const threshold = createAbstractNumber(5);
    const result = greaterThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(0);
  });

  test("throws error for insufficient arguments", () => {
    const a = createAbstractNumber(5);
    expect(() => greaterThan([a])).toThrow();
  });
});
