import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { filter } from "../filtering";
import type { ShapeValue, FoodValue, ArrayValue, OtherValue } from "../../runtime/types";

describe("filter (unit)", () => {
  test("filters shapes by size", () => {
    const shapes: ShapeValue[] = [
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "grande", amount: new Fraction(1) },
      { kind: "forma", category: "pictorico", subtype: "cuadrado", size: "pequeño", amount: new Fraction(1) },
      { kind: "forma", category: "pictorico", subtype: "cuadrado", size: "grande", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "otro", value: "grande" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as ShapeValue).subtype).toBe("circulo");
    expect((result.elements[1] as ShapeValue).subtype).toBe("cuadrado");
  });

  test("filters shapes by subtype", () => {
    const shapes: ShapeValue[] = [
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "grande", amount: new Fraction(1) },
      { kind: "forma", category: "pictorico", subtype: "cuadrado", size: "pequeño", amount: new Fraction(1) },
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "mediano", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "otro", value: "circulo" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("filters foods by color", () => {
    const foods: FoodValue[] = [
      { kind: "comida", category: "concreto", subtype: "uva", color: "morado", amount: new Fraction(5) },
      { kind: "comida", category: "concreto", subtype: "manzana", color: "rojo", amount: new Fraction(3) },
      { kind: "comida", category: "concreto", subtype: "uva", color: "morado", amount: new Fraction(2) },
    ];
    const criterion: OtherValue = { kind: "otro", value: "morado" };
    const result = filter([...foods, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("filters foods by subtype", () => {
    const foods: FoodValue[] = [
      { kind: "comida", category: "concreto", subtype: "uva", color: "morado", amount: new Fraction(5) },
      { kind: "comida", category: "concreto", subtype: "manzana", color: "rojo", amount: new Fraction(3) },
      { kind: "comida", category: "concreto", subtype: "pera", color: "verde", amount: new Fraction(2) },
    ];
    const criterion: OtherValue = { kind: "otro", value: "manzana" };
    const result = filter([...foods, criterion]) as FoodValue;
    expect(result.kind).toBe("comida");
    expect(result.subtype).toBe("manzana");
  });

  test("returns single element when only one matches", () => {
    const shapes: ShapeValue[] = [
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "grande", amount: new Fraction(1) },
      { kind: "forma", category: "pictorico", subtype: "cuadrado", size: "pequeño", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "otro", value: "pequeño" };
    const result = filter([...shapes, criterion]) as ShapeValue;
    expect(result.kind).toBe("forma");
    expect(result.subtype).toBe("cuadrado");
  });

  test("returns empty array when nothing matches", () => {
    const shapes: ShapeValue[] = [
      { kind: "forma", category: "pictorico", subtype: "circulo", size: "grande", amount: new Fraction(1) },
      { kind: "forma", category: "pictorico", subtype: "cuadrado", size: "grande", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "otro", value: "pequeño" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(0);
  });

  test("filters by category", () => {
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
    const criterion: OtherValue = { kind: "otro", value: "concreto" };
    const result = filter([grape, circle, criterion]) as FoodValue;
    expect(result.kind).toBe("comida");
    expect(result.category).toBe("concreto");
  });

  test("throws error for insufficient arguments", () => {
    const shape: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(1),
    };
    expect(() => filter([shape])).toThrow();
  });

  test("throws error for non-keyword criterion", () => {
    const shape: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "grande",
      amount: new Fraction(1),
    };
    const badCriterion = { kind: "racional", value: new Fraction(5) };
    expect(() => filter([shape, badCriterion as any])).toThrow();
  });
});
