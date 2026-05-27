import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { first, last } from "../accessor";
import type { CPAObject, ArrayValue } from "../../runtime/types";

const apple: CPAObject = {
  kind: "cpa",
  category: "concreto",
  type: "comida",
  subtype: "manzana",
  quantity: new Fraction(1),
  attributes: {},
};

const pear: CPAObject = {
  kind: "cpa",
  category: "concreto",
  type: "comida",
  subtype: "pera",
  quantity: new Fraction(1),
  attributes: {},
};

const grape: CPAObject = {
  kind: "cpa",
  category: "concreto",
  type: "comida",
  subtype: "uva",
  quantity: new Fraction(1),
  attributes: {},
};

describe("first (unit)", () => {
  test("returns first element of array", () => {
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [apple, pear, grape],
    };
    const result = first([arr]);
    expect(result).toEqual(apple);
  });

  test("returns the element itself if not an array", () => {
    const result = first([apple]);
    expect(result).toEqual(apple);
  });

  test("returns empty array when array is empty", () => {
    const arr: ArrayValue = { kind: "arreglo", elements: [] };
    const result = first([arr]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toEqual([]);
  });

  test("returns empty array when no arguments", () => {
    const result = first([]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toEqual([]);
  });
});

describe("last (unit)", () => {
  test("returns last element of array", () => {
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [apple, pear, grape],
    };
    const result = last([arr]);
    expect(result).toEqual(grape);
  });

  test("returns the element itself if not an array", () => {
    const result = last([pear]);
    expect(result).toEqual(pear);
  });

  test("returns empty array when array is empty", () => {
    const arr: ArrayValue = { kind: "arreglo", elements: [] };
    const result = last([arr]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toEqual([]);
  });

  test("returns empty array when no arguments", () => {
    const result = last([]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toEqual([]);
  });
});
