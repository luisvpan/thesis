import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { sum, multiply, substract, divide } from "../arithmetic";
import type { RationalValue, ShapeValue, FoodValue, ArrayValue } from "../../runtime/types";

describe("sum (unit)", () => {
  test("adds two rationals", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(2) };
    const b: RationalValue = { kind: "rational", value: new Fraction(3) };
    const result = sum([a, b]) as RationalValue;
    expect(result.kind).toBe("rational");
    expect(result.value.equals(new Fraction(5))).toBe(true);
  });

  test("adds multiple rationals (variadic)", () => {
    const values: RationalValue[] = [
      { kind: "rational", value: new Fraction(1) },
      { kind: "rational", value: new Fraction(2) },
      { kind: "rational", value: new Fraction(3) },
      { kind: "rational", value: new Fraction(4) },
    ];
    const result = sum(values) as RationalValue;
    expect(result.value.equals(new Fraction(10))).toBe(true);
  });

  test("aggregates CPA objects by key", () => {
    const grape1: FoodValue = {
      kind: "food",
      category: "concrete",
      subtype: "grape",
      color: "purple",
      amount: new Fraction(5),
    };
    const grape2: FoodValue = {
      kind: "food",
      category: "concrete",
      subtype: "grape",
      color: "purple",
      amount: new Fraction(3),
    };
    const result = sum([grape1, grape2]) as FoodValue;
    expect(result.kind).toBe("food");
    expect(result.amount.equals(new Fraction(8))).toBe(true);
  });

  test("returns array for different CPA types", () => {
    const grape: FoodValue = {
      kind: "food",
      category: "concrete",
      subtype: "grape",
      color: "purple",
      amount: new Fraction(5),
    };
    const circle: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "large",
      amount: new Fraction(3),
    };
    const result = sum([grape, circle]) as ArrayValue;
    expect(result.kind).toBe("array");
    expect(result.elements).toHaveLength(2);
  });

  test("flattens nested arrays", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(1) };
    const b: RationalValue = { kind: "rational", value: new Fraction(2) };
    const nested: ArrayValue = { kind: "array", elements: [a, b] };
    const c: RationalValue = { kind: "rational", value: new Fraction(3) };
    const result = sum([nested, c]) as RationalValue;
    expect(result.value.equals(new Fraction(6))).toBe(true);
  });
});

describe("multiply (unit)", () => {
  test("multiplies two rationals", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(4) };
    const b: RationalValue = { kind: "rational", value: new Fraction(3) };
    const result = multiply([a, b]) as RationalValue;
    expect(result.kind).toBe("rational");
    expect(result.value.equals(new Fraction(12))).toBe(true);
  });

  test("multiplies multiple rationals (variadic)", () => {
    const values: RationalValue[] = [
      { kind: "rational", value: new Fraction(2) },
      { kind: "rational", value: new Fraction(3) },
      { kind: "rational", value: new Fraction(4) },
    ];
    const result = multiply(values) as RationalValue;
    expect(result.value.equals(new Fraction(24))).toBe(true);
  });

  test("scales CPA object by rational factor", () => {
    const shape: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "square",
      size: "small",
      amount: new Fraction(5),
    };
    const factor: RationalValue = { kind: "rational", value: new Fraction(3) };
    const result = multiply([shape, factor]) as ShapeValue;
    expect(result.kind).toBe("shape");
    expect(result.amount.equals(new Fraction(15))).toBe(true);
  });

  test("aggregates CPA objects by key with multiplication", () => {
    const circle1: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "large",
      amount: new Fraction(2),
    };
    const circle2: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "large",
      amount: new Fraction(3),
    };
    const result = multiply([circle1, circle2]) as ShapeValue;
    expect(result.kind).toBe("shape");
    expect(result.amount.equals(new Fraction(6))).toBe(true);
  });
});

describe("substract (unit)", () => {
  test("subtracts two rationals", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(10) };
    const b: RationalValue = { kind: "rational", value: new Fraction(3) };
    const result = substract([a, b]) as RationalValue;
    expect(result.kind).toBe("rational");
    expect(result.value.equals(new Fraction(7))).toBe(true);
  });

  test("subtracts CPA object amounts", () => {
    const grape1: FoodValue = {
      kind: "food",
      category: "concrete",
      subtype: "grape",
      color: "purple",
      amount: new Fraction(10),
    };
    const grape2: FoodValue = {
      kind: "food",
      category: "concrete",
      subtype: "apple",
      color: "red",
      amount: new Fraction(3),
    };
    const result = substract([grape1, grape2]) as FoodValue;
    expect(result.kind).toBe("food");
    expect(result.amount.equals(new Fraction(7))).toBe(true);
  });

  test("subtracts rational from CPA object", () => {
    const shape: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "medium",
      amount: new Fraction(12),
    };
    const value: RationalValue = { kind: "rational", value: new Fraction(4) };
    const result = substract([shape, value]) as ShapeValue;
    expect(result.kind).toBe("shape");
    expect(result.amount.equals(new Fraction(8))).toBe(true);
  });

  test("throws error for wrong arity", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(10) };
    expect(() => substract([a])).toThrow();
  });
});

describe("divide (unit)", () => {
  test("divides two rationals", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(12) };
    const b: RationalValue = { kind: "rational", value: new Fraction(4) };
    const result = divide([a, b]) as RationalValue;
    expect(result.kind).toBe("rational");
    expect(result.value.equals(new Fraction(3))).toBe(true);
  });

  test("divides CPA object by rational", () => {
    const shape: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "medium",
      amount: new Fraction(12),
    };
    const divisor: RationalValue = { kind: "rational", value: new Fraction(4) };
    const result = divide([shape, divisor]) as ShapeValue;
    expect(result.kind).toBe("shape");
    expect(result.amount.equals(new Fraction(3))).toBe(true);
  });

  test("handles fractional results", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(1) };
    const b: RationalValue = { kind: "rational", value: new Fraction(3) };
    const result = divide([a, b]) as RationalValue;
    expect(result.value.equals(new Fraction(1, 3))).toBe(true);
  });

  test("throws error for division by zero", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(10) };
    const b: RationalValue = { kind: "rational", value: new Fraction(0) };
    expect(() => divide([a, b])).toThrow();
  });

  test("throws error for wrong arity", () => {
    const a: RationalValue = { kind: "rational", value: new Fraction(10) };
    expect(() => divide([a])).toThrow();
  });
});
