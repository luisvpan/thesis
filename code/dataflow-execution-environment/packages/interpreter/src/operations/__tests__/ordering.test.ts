import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { orderAsc, orderDesc } from "../ordering";
import type { RationalValue, ShapeValue, FoodValue, ArrayValue } from "../../runtime/types";

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
    const grape: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(5),
    };
    const circle: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(3),
    };
    const result = orderAsc([circle, grape]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as FoodValue).kind).toBe("comida");
    expect((result.elements[1] as ShapeValue).kind).toBe("forma");
  });

  test("sorts by type alphabetically within same category", () => {
    const pear: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "pera",
      color: "verde",
      amount: new Fraction(1),
    };
    const apple: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "manzana",
      color: "rojo",
      amount: new Fraction(2),
    };
    const grape: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(3),
    };
    const result = orderAsc([pear, apple, grape]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as FoodValue).subtype).toBe("manzana");
    expect((result.elements[1] as FoodValue).subtype).toBe("pera");
    expect((result.elements[2] as FoodValue).subtype).toBe("uva");
  });

  test("sorts by quantity within same type", () => {
    const grape1: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(10),
    };
    const grape2: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(2),
    };
    const grape3: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(5),
    };
    const result = orderAsc([grape1, grape2, grape3]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as FoodValue).amount.equals(new Fraction(2))).toBe(true);
    expect((result.elements[1] as FoodValue).amount.equals(new Fraction(5))).toBe(true);
    expect((result.elements[2] as FoodValue).amount.equals(new Fraction(10))).toBe(true);
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
    const grape: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(5),
    };
    const circle: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(3),
    };
    const result = orderDesc([grape, circle]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as ShapeValue).kind).toBe("forma");
    expect((result.elements[1] as FoodValue).kind).toBe("comida");
  });

  test("sorts by type alphabetically in reverse within same category", () => {
    const pear: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "pera",
      color: "verde",
      amount: new Fraction(1),
    };
    const apple: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "manzana",
      color: "rojo",
      amount: new Fraction(2),
    };
    const grape: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "uva",
      color: "morado",
      amount: new Fraction(3),
    };
    const result = orderDesc([apple, pear, grape]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as FoodValue).subtype).toBe("uva");
    expect((result.elements[1] as FoodValue).subtype).toBe("pera");
    expect((result.elements[2] as FoodValue).subtype).toBe("manzana");
  });

  test("sorts by quantity in reverse within same type", () => {
    const circle1: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(2),
    };
    const circle2: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(10),
    };
    const circle3: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(5),
    };
    const result = orderDesc([circle1, circle2, circle3]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect((result.elements[0] as ShapeValue).amount.equals(new Fraction(10))).toBe(true);
    expect((result.elements[1] as ShapeValue).amount.equals(new Fraction(5))).toBe(true);
    expect((result.elements[2] as ShapeValue).amount.equals(new Fraction(2))).toBe(true);
  });

  test("returns single element unchanged", () => {
    const value: RationalValue = { kind: "racional", value: new Fraction(5) };
    const result = orderDesc([value]) as RationalValue;
    expect(result.kind).toBe("racional");
    expect(result.value.equals(new Fraction(5))).toBe(true);
  });
});
