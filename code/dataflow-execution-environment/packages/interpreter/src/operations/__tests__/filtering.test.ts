import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { filter } from "../filtering";
import type { CPAObject, ArrayValue, OtherValue } from "../../runtime/types";

describe("filter (unit)", () => {
  test("filters shapes by size", () => {
    const shapes: CPAObject[] = [
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(1), attributes: { size: "grande" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "cuadrado", quantity: new Fraction(1), attributes: { size: "pequeño" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "cuadrado", quantity: new Fraction(1), attributes: { size: "grande" } },
    ];
    const criterion: OtherValue = { kind: "otro", value: "grande" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).subtype).toBe("circulo");
    expect((result.elements[1] as CPAObject).subtype).toBe("cuadrado");
  });

  test("filters shapes by subtype", () => {
    const shapes: CPAObject[] = [
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(1), attributes: { size: "grande" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "cuadrado", quantity: new Fraction(1), attributes: { size: "pequeño" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(1), attributes: { size: "mediano" } },
    ];
    const criterion: OtherValue = { kind: "otro", value: "circulo" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("filters foods by color", () => {
    const foods: CPAObject[] = [
      { kind: "cpa", category: "concreto", type: "comida", subtype: "uva", quantity: new Fraction(5), attributes: { color: "morado" } },
      { kind: "cpa", category: "concreto", type: "comida", subtype: "manzana", quantity: new Fraction(3), attributes: { color: "rojo" } },
      { kind: "cpa", category: "concreto", type: "comida", subtype: "uva", quantity: new Fraction(2), attributes: { color: "morado" } },
    ];
    const criterion: OtherValue = { kind: "otro", value: "morado" };
    const result = filter([...foods, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("filters foods by subtype", () => {
    const foods: CPAObject[] = [
      { kind: "cpa", category: "concreto", type: "comida", subtype: "uva", quantity: new Fraction(5), attributes: { color: "morado" } },
      { kind: "cpa", category: "concreto", type: "comida", subtype: "manzana", quantity: new Fraction(3), attributes: { color: "rojo" } },
      { kind: "cpa", category: "concreto", type: "comida", subtype: "pera", quantity: new Fraction(2), attributes: { color: "verde" } },
    ];
    const criterion: OtherValue = { kind: "otro", value: "manzana" };
    const result = filter([...foods, criterion]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.subtype).toBe("manzana");
  });

  test("returns single element when only one matches", () => {
    const shapes: CPAObject[] = [
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(1), attributes: { size: "grande" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "cuadrado", quantity: new Fraction(1), attributes: { size: "pequeño" } },
    ];
    const criterion: OtherValue = { kind: "otro", value: "pequeño" };
    const result = filter([...shapes, criterion]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.subtype).toBe("cuadrado");
  });

  test("returns empty array when nothing matches", () => {
    const shapes: CPAObject[] = [
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "circulo", quantity: new Fraction(1), attributes: { size: "grande" } },
      { kind: "cpa", category: "pictorico", type: "forma", subtype: "cuadrado", quantity: new Fraction(1), attributes: { size: "grande" } },
    ];
    const criterion: OtherValue = { kind: "otro", value: "pequeño" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(0);
  });

  test("filters by category", () => {
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
    const criterion: OtherValue = { kind: "otro", value: "concreto" };
    const result = filter([grape, circle, criterion]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("concreto");
  });

  test("throws error for insufficient arguments", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(1),
      attributes: { size: "grande" },
    };
    expect(() => filter([shape])).toThrow();
  });

  test("ignores unknown criterion types and returns all data (v4.0.0)", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(1),
      attributes: { size: "grande" },
    };
    // Unknown criterion type - ignored by separationPass
    const badCriterion = { kind: "racional", value: new Fraction(5) };
    const result = filter([shape, badCriterion as any]) as CPAObject;
    // With no valid criteria, all data items are returned
    expect(result.kind).toBe("cpa");
    expect(result.subtype).toBe("circulo");
  });
});
