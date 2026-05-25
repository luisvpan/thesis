import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { sum, multiply, substract, divide } from "../arithmetic";
import type { CPAObject, ArrayValue } from "../../runtime/types";
import { createAbstractNumber } from "../../utils";

describe("sum (unit)", () => {
  test("adds two abstract numbers", () => {
    const a = createAbstractNumber(2);
    const b = createAbstractNumber(3);
    const result = sum([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(5))).toBe(true);
  });

  test("adds multiple abstract numbers (variadic)", () => {
    const values = [
      createAbstractNumber(1),
      createAbstractNumber(2),
      createAbstractNumber(3),
      createAbstractNumber(4),
    ];
    const result = sum(values) as CPAObject;
    expect(result.quantity.equals(new Fraction(10))).toBe(true);
  });

  test("aggregates CPA objects by key", () => {
    const grape1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(5),
      attributes: { color: "morado" },
    };
    const grape2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(3),
      attributes: { color: "morado" },
    };
    const result = sum([grape1, grape2]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(8))).toBe(true);
  });

  test("returns array for different CPA types", () => {
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
    const result = sum([grape, circle]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("flattens nested arrays", () => {
    const a = createAbstractNumber(1);
    const b = createAbstractNumber(2);
    const nested: ArrayValue = { kind: "arreglo", elements: [a, b] };
    const c = createAbstractNumber(3);
    const result = sum([nested, c]) as CPAObject;
    expect(result.quantity.equals(new Fraction(6))).toBe(true);
  });
});

describe("multiply (unit)", () => {
  test("multiplies two abstract numbers", () => {
    const a = createAbstractNumber(4);
    const b = createAbstractNumber(3);
    const result = multiply([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(12))).toBe(true);
  });

  test("multiplies multiple abstract numbers (variadic)", () => {
    const values = [
      createAbstractNumber(2),
      createAbstractNumber(3),
      createAbstractNumber(4),
    ];
    const result = multiply(values) as CPAObject;
    expect(result.quantity.equals(new Fraction(24))).toBe(true);
  });

  test("scales CPA object by abstract number factor", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "cuadrado",
      quantity: new Fraction(5),
      attributes: { size: "pequeño" },
    };
    const factor = createAbstractNumber(3);
    const result = multiply([shape, factor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(15))).toBe(true);
  });

  test("does not combine CPAs with same key, returns array", () => {
    const circle1: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(3),
      attributes: { color: "azul" },
    };
    const circle2: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(2),
      attributes: { color: "azul" },
    };
    const result = multiply([circle1, circle2]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(3))).toBe(true);
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(2))).toBe(true);
  });

  test("does not combine CPAs with different keys, returns array", () => {
    const cap: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "tapa",
      quantity: new Fraction(3),
      attributes: { color: "azul" },
    };
    const ball: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "pelota",
      quantity: new Fraction(2),
      attributes: { color: "rojo" },
    };
    const result = multiply([cap, ball]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
  });

  test("applies abstract number factor to multiple CPAs", () => {
    const cap1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "tapa",
      quantity: new Fraction(3),
      attributes: { color: "azul" },
    };
    const cap2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "tapa",
      quantity: new Fraction(2),
      attributes: { color: "azul" },
    };
    const factor = createAbstractNumber(2);
    const result = multiply([cap1, cap2, factor]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).quantity.equals(new Fraction(6))).toBe(true);  // 3*2
    expect((result.elements[1] as CPAObject).quantity.equals(new Fraction(4))).toBe(true);  // 2*2
  });

  test("abstract numbers multiply together (regression)", () => {
    const a = createAbstractNumber(3);
    const b = createAbstractNumber(2);
    const result = multiply([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(6))).toBe(true);
  });
});

describe("substract (unit)", () => {
  test("subtracts two abstract numbers", () => {
    const a = createAbstractNumber(10);
    const b = createAbstractNumber(3);
    const result = substract([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(7))).toBe(true);
  });

  test("subtracts CPA objects with same key", () => {
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
      quantity: new Fraction(3),
      attributes: { color: "morado" },
    };
    const result = substract([grape1, grape2]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(7))).toBe(true);
  });

  test("subtracts CPA objects with different keys returns array", () => {
    const grape: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "uva",
      quantity: new Fraction(10),
      attributes: { color: "morado" },
    };
    const apple: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: new Fraction(3),
      attributes: { color: "rojo" },
    };
    const result = substract([grape, apple]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    // uva: 10 - 0 = 10
    const uvaResult = result.elements.find(e => (e as CPAObject).subtype === "uva") as CPAObject;
    expect(uvaResult.quantity.equals(new Fraction(10))).toBe(true);
    // manzana: 0 - 3 = -3
    const appleResult = result.elements.find(e => (e as CPAObject).subtype === "manzana") as CPAObject;
    expect(appleResult.quantity.equals(new Fraction(-3))).toBe(true);
  });

  test("subtracts abstract number from CPA object returns array with both", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const value = createAbstractNumber(4);
    const result = substract([shape, value]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    // shape: 12 - 0 = 12 (unchanged)
    const shapeResult = result.elements.find(e => (e as CPAObject).subtype === "circulo") as CPAObject;
    expect(shapeResult.quantity.equals(new Fraction(12))).toBe(true);
    // numero: 0 - 4 = -4
    const numResult = result.elements.find(e => (e as CPAObject).type === "numero") as CPAObject;
    expect(numResult.quantity.equals(new Fraction(-4))).toBe(true);
  });

  test("throws error for wrong arity", () => {
    const a = createAbstractNumber(10);
    expect(() => substract([a])).toThrow();
  });
});

describe("divide (unit)", () => {
  test("divides two abstract numbers", () => {
    const a = createAbstractNumber(12);
    const b = createAbstractNumber(4);
    const result = divide([a, b]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.category).toBe("abstracto");
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("divides CPA object by abstract number", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const divisor = createAbstractNumber(4);
    const result = divide([shape, divisor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("handles fractional results", () => {
    const a = createAbstractNumber(1);
    const b = createAbstractNumber(3);
    const result = divide([a, b]) as CPAObject;
    expect(result.quantity.equals(new Fraction(1, 3))).toBe(true);
  });

  test("throws error for division by zero", () => {
    const a = createAbstractNumber(10);
    const b = createAbstractNumber(0);
    expect(() => divide([a, b])).toThrow();
  });

  test("throws error for wrong arity", () => {
    const a = createAbstractNumber(10);
    expect(() => divide([a])).toThrow();
  });

  test("handles ArrayValue with implicit sum", () => {
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(6), createAbstractNumber(6)],
    };
    const divisor = createAbstractNumber(4);
    const result = divide([arr, divisor]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true); // (6+6)/4 = 3
  });

  test("handles ArrayValue as divisor with implicit sum", () => {
    const dividend = createAbstractNumber(12);
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(2), createAbstractNumber(2)],
    };
    const result = divide([dividend, arr]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true); // 12/(2+2) = 3
  });
});

describe("substract with ArrayValue (unit)", () => {
  test("handles ArrayValue minuend with implicit sum", () => {
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(7), createAbstractNumber(3)],
    };
    const subtrahend = createAbstractNumber(5);
    const result = substract([arr, subtrahend]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(5))).toBe(true); // (7+3)-5 = 5
  });

  test("handles ArrayValue subtrahend with implicit sum", () => {
    const minuend = createAbstractNumber(10);
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(2), createAbstractNumber(1)],
    };
    const result = substract([minuend, arr]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(7))).toBe(true); // 10-(2+1) = 7
  });

  test("handles both arguments as ArrayValue", () => {
    const arr1: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(5), createAbstractNumber(5)],
    };
    const arr2: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(3), createAbstractNumber(4)],
    };
    const result = substract([arr1, arr2]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true); // (5+5)-(3+4) = 3
  });

  test("handles heterogeneous arrays (per-key subtraction)", () => {
    const red1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(1),
      attributes: { color: "rojo" },
    };
    const orange1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(3),
      attributes: { color: "naranja" },
    };
    const red2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(2),
      attributes: { color: "rojo" },
    };
    const blue1: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(1),
      attributes: { color: "azul" },
    };
    const orange2: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(1),
      attributes: { color: "naranja" },
    };

    const arrA: ArrayValue = { kind: "arreglo", elements: [red1, orange1] };
    const arrB: ArrayValue = { kind: "arreglo", elements: [red2, blue1, orange2] };

    const result = substract([arrA, arrB]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(3);

    // rojo: 1 - 2 = -1
    const redResult = result.elements.find(
      e => (e as CPAObject).attributes.color === "rojo"
    ) as CPAObject;
    expect(redResult.quantity.equals(new Fraction(-1))).toBe(true);

    // naranja: 3 - 1 = 2
    const orangeResult = result.elements.find(
      e => (e as CPAObject).attributes.color === "naranja"
    ) as CPAObject;
    expect(orangeResult.quantity.equals(new Fraction(2))).toBe(true);

    // azul: 0 - 1 = -1 (only in B)
    const blueResult = result.elements.find(
      e => (e as CPAObject).attributes.color === "azul"
    ) as CPAObject;
    expect(blueResult.quantity.equals(new Fraction(-1))).toBe(true);
  });
});

describe("divide with heterogeneous arrays (unit)", () => {
  test("divides heterogeneous array by number", () => {
    const red: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(6),
      attributes: { color: "rojo" },
    };
    const blue: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(9),
      attributes: { color: "azul" },
    };
    const arr: ArrayValue = { kind: "arreglo", elements: [red, blue] };
    const divisor = createAbstractNumber(3);

    const result = divide([arr, divisor]) as ArrayValue;
    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);

    // rojo: 6 / 3 = 2
    const redResult = result.elements.find(
      e => (e as CPAObject).attributes.color === "rojo"
    ) as CPAObject;
    expect(redResult.quantity.equals(new Fraction(2))).toBe(true);

    // azul: 9 / 3 = 3
    const blueResult = result.elements.find(
      e => (e as CPAObject).attributes.color === "azul"
    ) as CPAObject;
    expect(blueResult.quantity.equals(new Fraction(3))).toBe(true);
  });

  test("ignores non-number CPAs in divisor", () => {
    const red: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(6),
      attributes: { color: "rojo" },
    };
    const blue: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(1),
      attributes: { color: "azul" },
    };
    const divisor = createAbstractNumber(2);

    const arrB: ArrayValue = { kind: "arreglo", elements: [blue, divisor] };
    const result = divide([red, arrB]) as CPAObject;

    // Only the abstract number (2) is used as divisor, blue is ignored
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true); // 6 / 2 = 3
  });

  test("uses divisor 1 when no abstract numbers in b", () => {
    const red: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(6),
      attributes: { color: "rojo" },
    };
    const blue: CPAObject = {
      kind: "cpa",
      category: "concreto",
      type: "objeto",
      subtype: "palito",
      quantity: new Fraction(1),
      attributes: { color: "azul" },
    };

    const result = divide([red, blue]) as CPAObject;

    // No abstract numbers in b, so divisor is implicitly 1
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(6))).toBe(true); // 6 / 1 = 6
  });

  test("sums multiple abstract numbers in divisor", () => {
    const shape: CPAObject = {
      kind: "cpa",
      category: "pictorico",
      type: "forma",
      subtype: "circulo",
      quantity: new Fraction(12),
      attributes: { size: "mediano" },
    };
    const arr: ArrayValue = {
      kind: "arreglo",
      elements: [createAbstractNumber(2), createAbstractNumber(2)],
    };

    const result = divide([shape, arr]) as CPAObject;
    expect(result.kind).toBe("cpa");
    expect(result.quantity.equals(new Fraction(3))).toBe(true); // 12 / (2+2) = 3
  });
});
