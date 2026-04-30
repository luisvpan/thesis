import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import {
  flattenArrays,
  getQuantity,
  getComparableValue,
  getQuantityOrZero,
  cloneCPAWithQuantity,
} from "../utils";
import type { RuntimeValue, CPAObject, ShapeValue, FoodValue, AbstractValue } from "../../runtime/types";

describe("flattenArrays", () => {
  test("should flatten nested arrays", () => {
    const input: RuntimeValue[] = [
      { kind: "racional", value: new Fraction(1) },
      {
        kind: "arreglo",
        elements: [
          { kind: "racional", value: new Fraction(2) },
          { kind: "racional", value: new Fraction(3) },
        ],
      },
    ];
    const result = flattenArrays(input);
    expect(result).toHaveLength(3);
  });

  test("should handle deeply nested arrays", () => {
    const input: RuntimeValue[] = [
      {
        kind: "arreglo",
        elements: [
          {
            kind: "arreglo",
            elements: [{ kind: "racional", value: new Fraction(1) }],
          },
        ],
      },
    ];
    const result = flattenArrays(input);
    expect(result).toHaveLength(1);
  });

  test("should return empty array for empty input", () => {
    expect(flattenArrays([])).toEqual([]);
  });

  test("should preserve non-array values", () => {
    const input: RuntimeValue[] = [
      { kind: "racional", value: new Fraction(1) },
      { kind: "racional", value: new Fraction(2) },
    ];
    const result = flattenArrays(input);
    expect(result).toHaveLength(2);
    expect(result).toEqual(input);
  });
});

describe("getQuantity", () => {
  test("should return value for abstracto", () => {
    const obj: AbstractValue = {
      kind: "abstracto",
      category: "abstracto",
      objectType: "racional",
      value: new Fraction(5),
    };
    expect(getQuantity(obj).valueOf()).toBe(5);
  });

  test("should return amount for forma", () => {
    const obj: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "circulo",
      size: "mediano",
      amount: new Fraction(3),
    };
    expect(getQuantity(obj).valueOf()).toBe(3);
  });

  test("should return amount for comida", () => {
    const obj: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "manzana",
      color: "rojo",
      amount: new Fraction(2),
    };
    expect(getQuantity(obj).valueOf()).toBe(2);
  });
});

describe("getComparableValue", () => {
  test("should return value for racional", () => {
    const val: RuntimeValue = { kind: "racional", value: new Fraction(7) };
    expect(getComparableValue(val)?.valueOf()).toBe(7);
  });

  test("should return amount for CPA objects", () => {
    const val: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "cuadrado",
      size: "grande",
      amount: new Fraction(4),
    };
    expect(getComparableValue(val)?.valueOf()).toBe(4);
  });

  test("should return null for otro", () => {
    const val: RuntimeValue = { kind: "otro", value: "test" };
    expect(getComparableValue(val)).toBeNull();
  });

  test("should return null for arreglo", () => {
    const val: RuntimeValue = { kind: "arreglo", elements: [] };
    expect(getComparableValue(val)).toBeNull();
  });
});

describe("getQuantityOrZero", () => {
  test("should return value for racional", () => {
    const val: RuntimeValue = { kind: "racional", value: new Fraction(7) };
    expect(getQuantityOrZero(val).valueOf()).toBe(7);
  });

  test("should return amount for CPA objects", () => {
    const val: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "pera",
      color: "verde",
      amount: new Fraction(5),
    };
    expect(getQuantityOrZero(val).valueOf()).toBe(5);
  });

  test("should return zero for otro", () => {
    const val: RuntimeValue = { kind: "otro", value: "test" };
    expect(getQuantityOrZero(val).valueOf()).toBe(0);
  });

  test("should return zero for arreglo", () => {
    const val: RuntimeValue = { kind: "arreglo", elements: [] };
    expect(getQuantityOrZero(val).valueOf()).toBe(0);
  });
});

describe("cloneCPAWithQuantity", () => {
  test("should clone abstracto with new value", () => {
    const obj: AbstractValue = {
      kind: "abstracto",
      category: "abstracto",
      objectType: "racional",
      value: new Fraction(1),
    };
    const result = cloneCPAWithQuantity(obj, new Fraction(10)) as AbstractValue;
    expect(result.value.valueOf()).toBe(10);
    expect(obj.value.valueOf()).toBe(1); // original unchanged
  });

  test("should clone forma with new amount", () => {
    const obj: ShapeValue = {
      kind: "forma",
      category: "pictorico",
      subtype: "cuadrado",
      size: "grande",
      amount: new Fraction(1),
    };
    const result = cloneCPAWithQuantity(obj, new Fraction(5)) as ShapeValue;
    expect(result.amount.valueOf()).toBe(5);
    expect(result.subtype).toBe("cuadrado");
    expect(result.size).toBe("grande");
  });

  test("should clone comida with new amount", () => {
    const obj: FoodValue = {
      kind: "comida",
      category: "concreto",
      subtype: "hamburguesa",
      color: "naranja",
      amount: new Fraction(2),
    };
    const result = cloneCPAWithQuantity(obj, new Fraction(8)) as FoodValue;
    expect(result.amount.valueOf()).toBe(8);
    expect(result.subtype).toBe("hamburguesa");
    expect(result.color).toBe("naranja");
  });
});
