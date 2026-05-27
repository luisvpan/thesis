import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { compare } from "../equality";
import type { CPAObject, ArrayValue, BooleanValue } from "../../runtime/types";

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

const grape = (qty: number): CPAObject => ({
  kind: "cpa",
  category: "concreto",
  type: "comida",
  subtype: "uva",
  quantity: new Fraction(qty),
  attributes: {},
});

describe("compare (unit)", () => {
  describe("element vs element", () => {
    test("equal elements return true", () => {
      const result = compare([apple(3), apple(3)]) as BooleanValue;
      expect(result.kind).toBe("booleano");
      expect(result.value).toBe(true);
    });

    test("different quantities return false", () => {
      const result = compare([apple(2), apple(3)]) as BooleanValue;
      expect(result.value).toBe(false);
    });

    test("different types return false", () => {
      const result = compare([apple(1), pear(1)]) as BooleanValue;
      expect(result.value).toBe(false);
    });
  });

  describe("array vs element", () => {
    test("array of 3 apples equals 3 apples", () => {
      const arr: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), apple(1), apple(1)],
      };
      const result = compare([arr, apple(3)]) as BooleanValue;
      expect(result.value).toBe(true);
    });

    test("array with different total returns false", () => {
      const arr: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), apple(1)],
      };
      const result = compare([arr, apple(3)]) as BooleanValue;
      expect(result.value).toBe(false);
    });
  });

  describe("array vs array", () => {
    test("[apple, apple, apple] equals [apple, 2 apples]", () => {
      const arr1: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), apple(1), apple(1)],
      };
      const arr2: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), apple(2)],
      };
      const result = compare([arr1, arr2]) as BooleanValue;
      expect(result.value).toBe(true);
    });

    test("[2 apples, 2 pears, grape] equals [apple, apple, pear, pear, grape]", () => {
      const arr1: ArrayValue = {
        kind: "arreglo",
        elements: [apple(2), pear(2), grape(1)],
      };
      const arr2: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), apple(1), pear(1), pear(1), grape(1)],
      };
      const result = compare([arr1, arr2]) as BooleanValue;
      expect(result.value).toBe(true);
    });

    test("[apple, pear] does not equal [apple, apple]", () => {
      const arr1: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), pear(1)],
      };
      const arr2: ArrayValue = {
        kind: "arreglo",
        elements: [apple(1), apple(1)],
      };
      const result = compare([arr1, arr2]) as BooleanValue;
      expect(result.value).toBe(false);
    });

    test("[2 apples] does not equal [3 apples]", () => {
      const arr1: ArrayValue = {
        kind: "arreglo",
        elements: [apple(2)],
      };
      const arr2: ArrayValue = {
        kind: "arreglo",
        elements: [apple(3)],
      };
      const result = compare([arr1, arr2]) as BooleanValue;
      expect(result.value).toBe(false);
    });

    test("empty arrays are equal", () => {
      const arr1: ArrayValue = { kind: "arreglo", elements: [] };
      const arr2: ArrayValue = { kind: "arreglo", elements: [] };
      const result = compare([arr1, arr2]) as BooleanValue;
      expect(result.value).toBe(true);
    });
  });

  describe("error handling", () => {
    test("throws error with wrong arity", () => {
      expect(() => compare([apple(1)])).toThrow();
      expect(() => compare([apple(1), apple(1), apple(1)])).toThrow();
    });
  });
});
