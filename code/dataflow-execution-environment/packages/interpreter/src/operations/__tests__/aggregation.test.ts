import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { count } from "../aggregation";
import type { CPAObject, ArrayValue } from "../../runtime/types";

const apple = (qty: number): CPAObject => ({
  kind: "cpa",
  category: "concreto",
  type: "comida",
  subtype: "manzana",
  quantity: new Fraction(qty),
  attributes: {},
});

const pear = (qty: number): CPAObject => ({
  kind: "cpa",
  category: "concreto",
  type: "comida",
  subtype: "pera",
  quantity: new Fraction(qty),
  attributes: {},
});

describe("count (unit)", () => {
  test("counts single element", () => {
    const result = count([apple(3)]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("counts multiple elements", () => {
    const result = count([apple(1), pear(2)]) as CPAObject;
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("counts array of elements", () => {
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [apple(1), apple(2), pear(3)],
    };
    const result = count([arr]) as CPAObject;
    expect(result.quantity.equals(new Fraction(6))).toBe(true);
  });

  test("counts nested arrays", () => {
    const inner: ArrayValue = {
      kind: "arreglo",
      elements: [apple(1), apple(1)],
    };
    const outer: ArrayValue = {
      kind: "arreglo",
      elements: [inner, pear(3)],
    };
    const result = count([outer]) as CPAObject;
    expect(result.quantity.equals(new Fraction(5))).toBe(true);
  });

  test("returns zero for empty input", () => {
    const result = count([]) as CPAObject;
    expect(result.quantity.equals(new Fraction(0))).toBe(true);
  });

  test("returns zero for empty array", () => {
    const arr: ArrayValue = { kind: "arreglo", elements: [] };
    const result = count([arr]) as CPAObject;
    expect(result.quantity.equals(new Fraction(0))).toBe(true);
  });
});
