import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { lessThan, greaterThan } from "../comparison";
import type { RationalValue, ShapeValue, FoodValue, ArrayValue } from "../../runtime/types";

describe("lessThan (unit)", () => {
  test("filters rationals less than threshold", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(1) },
      { kind: "racional", value: new Fraction(5) },
      { kind: "racional", value: new Fraction(3) },
      { kind: "racional", value: new Fraction(8) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(4) };
    const result = lessThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as RationalValue).value.equals(new Fraction(1))).toBe(true);
    expect((result.elements[1] as RationalValue).value.equals(new Fraction(3))).toBe(true);
  });

  test("filters CPA objects by amount less than threshold", () => {
    const shapes: ShapeValue[] = [
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "grande", amount: new Fraction(2) },
      { kind: "forma", category: "pictorico", subtype: "cuadrado", size: "pequeño", amount: new Fraction(10) },
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "mediano", amount: new Fraction(5) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(6) };
    const result = lessThan([...shapes, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("returns single element when only one matches", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(10) },
      { kind: "racional", value: new Fraction(1) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = lessThan([...values, threshold]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(1))).toBe(true);
  });

  test("returns empty array when nothing matches", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(10) },
      { kind: "racional", value: new Fraction(20) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = lessThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(0);
  });

  test("throws error for insufficient arguments", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(5) };
    expect(() => lessThan([a])).toThrow();
  });
});

describe("greaterThan (unit)", () => {
  test("filters rationals greater than threshold", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(1) },
      { kind: "racional", value: new Fraction(5) },
      { kind: "racional", value: new Fraction(3) },
      { kind: "racional", value: new Fraction(8) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(4) };
    const result = greaterThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as RationalValue).value.equals(new Fraction(5))).toBe(true);
    expect((result.elements[1] as RationalValue).value.equals(new Fraction(8))).toBe(true);
  });

  test("filters CPA objects by amount greater than threshold", () => {
    const foods: FoodValue[] = [
      { kind: "comida", category: "concreto", subtype: "uva", color: "morado", amount: new Fraction(2) },
      { kind: "comida", category: "concreto", subtype: "manzana", color: "rojo", amount: new Fraction(10) },
      { kind: "comida", category: "concreto", subtype: "pera", color: "verde", amount: new Fraction(5) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(4) };
    const result = greaterThan([...foods, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("returns single element when only one matches", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(1) },
      { kind: "racional", value: new Fraction(10) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = greaterThan([...values, threshold]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(10))).toBe(true);
  });

  test("returns empty array when nothing matches", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(1) },
      { kind: "racional", value: new Fraction(2) },
    ];
    const threshold: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = greaterThan([...values, threshold]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(0);
  });

  test("throws error for insufficient arguments", () => {
    const a: RationalValue = { kind: "racional", value: new Fraction(5) };
    expect(() => greaterThan([a])).toThrow();
  });
});
