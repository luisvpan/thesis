import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { sum, multiply, substract, divide } from "../arithmetic";
import type { RationalValue, CPAObject, ArrayValue } from "../../runtime/types";

describe("sum (unit)", () => {
  test("adds two rationals", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(2) };
    const b: RationalValue = { kind: "racional", value: new Fraction(3) };
    const result = sum([a, b]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(5))).toBe(true);
  });

  test("adds multiple rationals (variadic)", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(1) },
      { kind: "racional", value: new Fraction(2) },
      { kind: "racional", value: new Fraction(3) },
      { kind: "racional", value: new Fraction(4) },
    ];
    const result = sum(values) as RationalValue;
    expect(result.value.equals(new Fraction(10))).toBe(true);
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
    const a: RationalValue = { kind: "racional", value: new Fraction(1) };
    const b: RationalValue = { kind: "racional", value: new Fraction(2) };
    const nested: ArrayValue = { kind: "arreglo", elements: [a, b] };
    const c: RationalValue = { kind: "racional", value: new Fraction(3) };
    const result = sum([nested, c]) as RationalValue;
    expect(result.value.equals(new Fraction(6))).toBe(true);
  });
});

describe("multiply (unit)", () => {
  test("multiplies two rationals", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(4) };
    const b: RationalValue = { kind: "racional", value: new Fraction(3) };
    const result = multiply([a, b]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(12))).toBe(true);
  });

  test("multiplies multiple rationals (variadic)", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(2) },
      { kind: "racional", value: new Fraction(3) },
      { kind: "racional", value: new Fraction(4) },
    ];
    const result = multiply(values) as RationalValue;
    expect(result.value.equals(new Fraction(24))).toBe(true);
  });

  test("scales CPA object by rational factor", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "cuadrado",
      quantity: new Fraction(5),
      attributes: { size: "pequeño" },
    };
    const factor: RationalValue = { kind: "racional", value: new Fraction(3) };
    const result = multiply([shape, factor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(15))).toBe(true);
  });

  test("aggregates CPA objects by key with multiplication", () => {
    const circle1: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(2),
      attributes: { size: "grande" },
    };
    const circle2: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(3),
      attributes: { size: "grande" },
    };
    const result = multiply([circle1, circle2]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(6))).toBe(true);
  });
});

describe("substract (unit)", () => {
  test("subtracts two rationals", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(10) };
    const b: RationalValue = { kind: "racional", value: new Fraction(3) };
    const result = substract([a, b]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(7))).toBe(true);
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

  test("subtracts rational from CPA object", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const value: RationalValue = { kind: "racional", value: new Fraction(4) };
    const result = substract([shape, value]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(8))).toBe(true);
  });

  test("throws error for wrong arity", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(10) };
    expect(() => substract([a])).toThrow();
  });
});

describe("divide (unit)", () => {
  test("divides two rationals", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(12) };
    const b: RationalValue = { kind: "racional", value: new Fraction(4) };
    const result = divide([a, b]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(3))).toBe(true);
  });

  test("divides CPA object by rational", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const divisor: RationalValue = { kind: "racional", value: new Fraction(4) };
    const result = divide([shape, divisor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("handles fractional results", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(1) };
    const b: RationalValue = { kind: "racional", value: new Fraction(3) };
    const result = divide([a, b]) as RationalValue;
    expect(result.value.equals(new Fraction(1, 3))).toBe(true);
  });

  test("throws error for division by zero", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(10) };
    const b: RationalValue = { kind: "racional", value: new Fraction(0) };
    expect(() => divide([a, b])).toThrow();
  });

  test("throws error for wrong arity", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(10) };
    expect(() => divide([a])).toThrow();
  });
});
