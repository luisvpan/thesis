import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { orderAsc, orderDesc } from "../ordering";
import type { RationalValue, CPAObject, ArrayValue } from "../../runtime/types";

describe("orderAsc (unit)", () => {
  test("sorts rationals in ascending order", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(5) },
      { kind: "racional", value: new Fraction(2) },
      { kind: "racional", value: new Fraction(8) },
      { kind: "racional", value: new Fraction(1) },
    ];
    const result = orderAsc(values) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as RationalValue).value.equals(new Fraction(1))).toBe(true);
    expect((result.elements[1] as RationalValue).value.equals(new Fraction(2))).toBe(true);
    expect((result.elements[2] as RationalValue).value.equals(new Fraction(5))).toBe(true);
    expect((result.elements[3] as RationalValue).value.equals(new Fraction(8))).toBe(true);
  });

  test("sorts by category: Concrete < Pictorial < Abstract", () => {
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
    const result = orderAsc([circle, grape]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as CPAObject).category).toBe("concreto");
    expect((result.elements[1] as CPAObject).category).toBe("pictorico");
  });

  test("sorts by type alphabetically within same category", () => {
    const pear: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "pera",
      quantity: new Fraction(1),
      attributes: { color: "verde" },
    };
    const apple: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: new Fraction(2),
      attributes: { color: "rojo" },
    };
    const grape: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(3),
      attributes: { color: "morado" },
    };
    const result = orderAsc([pear, apple, grape]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as CPAObject).subtype).toBe("manzana");
    expect((result.elements[1] as CPAObject).subtype).toBe("pera");
    expect((result.elements[2] as CPAObject).subtype).toBe("uva");
  });

  test("sorts by quantity within same type", () => {
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
      subtype: "uva",
      quantity: new Fraction(2),
      attributes: { color: "morado" },
    };
    const grape3: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(5),
      attributes: { color: "morado" },
    };
    const result = orderAsc([grape1, grape2, grape3]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(2))).toBe(true);
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(5))).toBe(true);
    expect((result.elements[2] as CPAObject).quantity.equals(new Fraction(10))).toBe(true);
  });

  test("returns single element unchanged", () => {
    const value: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = orderAsc([value]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(5))).toBe(true);
  });
});

describe("orderDesc (unit)", () => {
  test("sorts rationals in descending order", () => {
    const values: RationalValue[] = [
      { kind: "racional", value: new Fraction(5) },
      { kind: "racional", value: new Fraction(2) },
      { kind: "racional", value: new Fraction(8) },
      { kind: "racional", value: new Fraction(1) },
    ];
    const result = orderDesc(values) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as RationalValue).value.equals(new Fraction(8))).toBe(true);
    expect((result.elements[1] as RationalValue).value.equals(new Fraction(5))).toBe(true);
    expect((result.elements[2] as RationalValue).value.equals(new Fraction(2))).toBe(true);
    expect((result.elements[3] as RationalValue).value.equals(new Fraction(1))).toBe(true);
  });

  test("sorts by category in reverse: Abstract > Pictorial > Concrete", () => {
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
    const result = orderDesc([grape, circle]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as CPAObject).category).toBe("pictorico");
    expect((result.elements[1] as CPAObject).category).toBe("concreto");
  });

  test("sorts by type alphabetically in reverse within same category", () => {
    const pear: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "pera",
      quantity: new Fraction(1),
      attributes: { color: "verde" },
    };
    const apple: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: new Fraction(2),
      attributes: { color: "rojo" },
    };
    const grape: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(3),
      attributes: { color: "morado" },
    };
    const result = orderDesc([apple, pear, grape]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as CPAObject).subtype).toBe("uva");
    expect((result.elements[1] as CPAObject).subtype).toBe("pera");
    expect((result.elements[2] as CPAObject).subtype).toBe("manzana");
  });

  test("sorts by quantity in reverse within same type", () => {
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
      quantity: new Fraction(10),
      attributes: { size: "grande" },
    };
    const circle3: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(5),
      attributes: { size: "grande" },
    };
    const result = orderDesc([circle1, circle2, circle3]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(10))).toBe(true);
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(5))).toBe(true);
    expect((result.elements[2] as CPAObject).quantity.equals(new Fraction(2))).toBe(true);
  });

  test("returns single element unchanged", () => {
    const value: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = orderDesc([value]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(5))).toBe(true);
  });
});
