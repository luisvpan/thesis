import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import {
  flattenArrays,
  getQuantity,
  getComparableValue,
  getQuantityOrZero,
  cloneCPAWithQuantity,
} from "../utils";
import type { RuntimeValue, CPAObject } from "../../runtime/types";
import { createAbstractNumber } from "../../utils";

describe("flattenArrays", () => {
  test("should flatten nested arrays", () => {
    const input: RuntimeValue[] = [
      createAbstractNumber(1),
      {
        kind: "arreglo",
        elements: [
          createAbstractNumber(2),
          createAbstractNumber(3),
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
            elements: [createAbstractNumber(1)],
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
      createAbstractNumber(1),
      createAbstractNumber(2),
    ];
    const result = flattenArrays(input);
    expect(result).toHaveLength(2);
  });
});

describe("getQuantity", () => {
  test("should return quantity for abstracto", () => {
    const obj: CPAObject = {
      kind: "cpa",
      category: "abstracto",
      type: "numero",
      subtype: "racional",
      quantity: new Fraction(5),
      attributes: {},
    };
    expect(getQuantity(obj).valueOf()).toBe(5);
  });

  test("should return quantity for forma (pictorico)", () => {
    const obj: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(3),
      attributes: { size: "mediano" },
    };
    expect(getQuantity(obj).valueOf()).toBe(3);
  });

  test("should return quantity for comida (concreto)", () => {
    const obj: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: new Fraction(2),
      attributes: { color: "rojo" },
    };
    expect(getQuantity(obj).valueOf()).toBe(2);
  });
});

describe("getComparableValue", () => {
  test("should return quantity for abstract number", () => {
    const val = createAbstractNumber(7);
    expect(getComparableValue(val)?.valueOf()).toBe(7);
  });

  test("should return quantity for CPA objects", () => {
    const val: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "cuadrado",
      quantity: new Fraction(4),
      attributes: { size: "grande" },
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
  test("should return quantity for abstract number", () => {
    const val = createAbstractNumber(7);
    expect(getQuantityOrZero(val).valueOf()).toBe(7);
  });

  test("should return quantity for CPA objects", () => {
    const val: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "pera",
      quantity: new Fraction(5),
      attributes: { color: "verde" },
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
  test("should clone abstracto with new quantity", () => {
    const obj: CPAObject = {
      kind: "cpa",
      category: "abstracto",
      type: "numero",
      subtype: "racional",
      quantity: new Fraction(1),
      attributes: {},
    };
    const result = cloneCPAWithQuantity(obj, new Fraction(10));
    expect(result.quantity.valueOf()).toBe(10);
    expect(obj.quantity.valueOf()).toBe(1); // original unchanged
  });

  test("should clone forma with new quantity", () => {
    const obj: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "cuadrado",
      quantity: new Fraction(1),
      attributes: { size: "grande" },
    };
    const result = cloneCPAWithQuantity(obj, new Fraction(5));
    expect(result.quantity.valueOf()).toBe(5);
    expect(result.subtype).toBe("cuadrado");
    expect(result.attributes.size).toBe("grande");
  });

  test("should clone comida with new quantity", () => {
    const obj: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "hamburguesa",
      quantity: new Fraction(2),
      attributes: { color: "naranja" },
    };
    const result = cloneCPAWithQuantity(obj, new Fraction(8));
    expect(result.quantity.valueOf()).toBe(8);
    expect(result.subtype).toBe("hamburguesa");
    expect(result.attributes.color).toBe("naranja");
  });
});
