import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { sum, multiply, substract, divide } from "../arithmetic";
import type { CPAObject, ArrayValue } from "../../runtime/types";
import { createAbstractNumber } from "../../utils";

describe("sum (unit)", () => {
  test("adds two abstract numbers", () => {
    const a = createAbstractNumber(2);
    const b = createAbstractNumber(3);
    const result = sum([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(5))).toBe(true);
  });

  test("adds multiple abstract numbers (variadic)", () => {
    const values = [
      createAbstractNumber(1),
      createAbstractNumber(2),
      createAbstractNumber(3),
      createAbstractNumber(4),
    ];
    const result = sum(values) as CPAObject;
    expect(result.quantity.equals(new Fraction(10))).toBe(true);
  });

  test("aggregates CPA objects by key", () => {
    const grape1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(5),
      attributes: { color: "morado" },
    };
    const grape2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(3),
      attributes: { color: "morado" },
    };
    const result = sum([grape1, grape2]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(8))).toBe(true);
  });

  test("returns array for different CPA types", () => {
    const grape: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(5),
      attributes: { color: "morado" },
    };
    const circle: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(3),
      attributes: { size: "grande" },
    };
    const result = sum([grape, circle]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("flattens nested arrays", () => {
    const a = createAbstractNumber(1);
    const b = createAbstractNumber(2);
    const nested: ArrayValue = { kind: "arreglo", elements: [a, b] };
    const c = createAbstractNumber(3);
    const result = sum([nested, c]) as CPAObject;
    expect(result.quantity.equals(new Fraction(6))).toBe(true);
  });
});

describe("multiply (unit)", () => {
  test("multiplies two abstract numbers", () => {
    const a = createAbstractNumber(4);
    const b = createAbstractNumber(3);
    const result = multiply([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(12))).toBe(true);
  });

  test("multiplies multiple abstract numbers (variadic)", () => {
    const values = [
      createAbstractNumber(2),
      createAbstractNumber(3),
      createAbstractNumber(4),
    ];
    const result = multiply(values) as CPAObject;
    expect(result.quantity.equals(new Fraction(24))).toBe(true);
  });

  test("scales CPA object by abstract number factor", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "cuadrado",
      quantity: new Fraction(5),
      attributes: { size: "pequeño" },
    };
    const factor = createAbstractNumber(3);
    const result = multiply([shape, factor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(15))).toBe(true);
  });

  test("does not combine CPAs with same key, returns array", () => {
    const circle1: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(3),
      attributes: { color: "azul" },
    };
    const circle2: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(2),
      attributes: { color: "azul" },
    };
    const result = multiply([circle1, circle2]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(3))).toBe(true);
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(2))).toBe(true);
  });

  test("does not combine CPAs with different keys, returns array", () => {
    const cap: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "tapa",
      quantity: new Fraction(3),
      attributes: { color: "azul" },
    };
    const ball: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "pelota",
      quantity: new Fraction(2),
      attributes: { color: "rojo" },
    };
    const result = multiply([cap, ball]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("applies abstract number factor to multiple CPAs", () => {
    const cap1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "tapa",
      quantity: new Fraction(3),
      attributes: { color: "azul" },
    };
    const cap2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "tapa",
      quantity: new Fraction(2),
      attributes: { color: "azul" },
    };
    const factor = createAbstractNumber(2);
    const result = multiply([cap1, cap2, factor]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(6))).toBe(true);  // 3*2
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(4))).toBe(true);  // 2*2
  });

  test("abstract numbers multiply together (regression)", () => {
    const a = createAbstractNumber(3);
    const b = createAbstractNumber(2);
    const result = multiply([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(6))).toBe(true);
  });
});

describe("substract (unit)", () => {
  test("subtracts two abstract numbers", () => {
    const a = createAbstractNumber(10);
    const b = createAbstractNumber(3);
    const result = substract([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(7))).toBe(true);
  });

  test("subtracts CPA object amounts", () => {
    const grape1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(10),
      attributes: { color: "morado" },
    };
    const grape2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: new Fraction(3),
      attributes: { color: "rojo" },
    };
    const result = substract([grape1, grape2]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(7))).toBe(true);
  });

  test("subtracts abstract number from CPA object", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const value = createAbstractNumber(4);
    const result = substract([shape, value]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(8))).toBe(true);
  });

  test("throws error for wrong arity", () => {
    const a = createAbstractNumber(10);
    expect(() => substract([a])).toThrow();
  });
});

describe("divide (unit)", () => {
  test("divides two abstract numbers", () => {
    const a = createAbstractNumber(12);
    const b = createAbstractNumber(4);
    const result = divide([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("divides CPA object by abstract number", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const divisor = createAbstractNumber(4);
    const result = divide([shape, divisor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("handles fractional results", () => {
    const a = createAbstractNumber(1);
    const b = createAbstractNumber(3);
    const result = divide([a, b]) as CPAObject;
    expect(result.quantity.equals(new Fraction(1, 3))).toBe(true);
  });

  test("throws error for division by zero", () => {
    const a = createAbstractNumber(10);
    const b = createAbstractNumber(0);
    expect(() => divide([a, b])).toThrow();
  });

  test("throws error for wrong arity", () => {
    const a = createAbstractNumber(10);
    expect(() => divide([a])).toThrow();
  });
});
