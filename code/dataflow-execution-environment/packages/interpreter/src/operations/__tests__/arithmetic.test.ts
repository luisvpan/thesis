// §3.1 Aritmética

import { describe, expect, test } from "bun:test";
import { NULO } from "../../runtime/bag";
import { DataflowError } from "../../runtime/errors";
import { bagOf, entry, num, only, pairs } from "../../__tests__/helpers";
import { divide, multiply, substract, sum } from "../arithmetic";

describe("sum — §3.1.1", () => {
  test("agrega por identidad", () => {
    expect(pairs(sum([bagOf(entry("manzana", 2)), bagOf(entry("manzana", 3))]))).toEqual(["manzana:5"]);
  });

  test("es variádica y respeta el orden de primera aparición", () => {
    const result = sum([
      bagOf(entry("manzana", 2), entry("pera", 1)),
      bagOf(entry("manzana", 4)),
      bagOf(entry("uva", 1)),
    ]);
    expect(pairs(result)).toEqual(["manzana:6", "pera:1", "uva:1"]);
  });

  test("conserva las cantidades 0", () => {
    expect(pairs(sum([bagOf(entry("manzana", 2)), bagOf(entry("manzana", -2))]))).toEqual(["manzana:0"]);
    expect(pairs(sum([bagOf(entry("manzana", 0)), bagOf(entry("pera", 3))]))).toEqual([
      "manzana:0",
      "pera:3",
    ]);
  });

  test("ignora los argumentos nulo", () => {
    expect(pairs(sum([bagOf(entry("manzana", 2)), NULO]))).toEqual(["manzana:2"]);
  });

  test("sin entradas efectivas devuelve nulo", () => {
    expect(pairs(sum([NULO, NULO]))).toEqual([]);
  });

  test("los atributos son parte de la identidad: no agrupa lo que difiere", () => {
    const result = sum([
      bagOf(entry("cubo", 2, { color: "rojo" })),
      bagOf(entry("cubo", 1, { color: "azul" })),
    ]);
    expect(pairs(result)).toEqual(["cubo:2", "cubo:1"]);
  });

  test("suma racionales de forma exacta", () => {
    expect(only(sum([num("1/2"), num("1/3")]))).toBe("5/6");
  });
});

describe("substract — §3.1.2", () => {
  test("resta por identidad", () => {
    expect(pairs(substract([bagOf(entry("manzana", 5)), bagOf(entry("manzana", 2))]))).toEqual([
      "manzana:3",
    ]);
  });

  test("las identidades solo del sustraendo quedan en negativo", () => {
    const result = substract([bagOf(entry("manzana", 1)), bagOf(entry("pera", 2))]);
    expect(pairs(result)).toEqual(["manzana:1", "pera:-2"]);
  });

  test("conserva el 0", () => {
    expect(pairs(substract([bagOf(entry("manzana", 2)), bagOf(entry("manzana", 2))]))).toEqual([
      "manzana:0",
    ]);
  });

  test("agrupa los repetidos de cada lado antes de restar", () => {
    const result = substract([
      bagOf(entry("manzana", 2), entry("manzana", 3)),
      bagOf(entry("manzana", 1)),
    ]);
    expect(pairs(result)).toEqual(["manzana:4"]);
  });

  test("ignora el sustraendo nulo", () => {
    expect(pairs(substract([bagOf(entry("manzana", 3)), NULO]))).toEqual(["manzana:3"]);
  });
});

describe("multiply — §3.1.3", () => {
  test("escala la bolsa por el segundo argumento", () => {
    expect(pairs(multiply([bagOf(entry("manzana", 2), entry("pera", 5)), num(10)]))).toEqual([
      "manzana:20",
      "pera:50",
    ]);
  });

  test("opera entrada por entrada y conserva los repetidos", () => {
    expect(pairs(multiply([bagOf(entry("manzana", 2), entry("manzana", 3)), num(4)]))).toEqual([
      "manzana:8",
      "manzana:12",
    ]);
  });

  test("escala con un racional", () => {
    expect(pairs(multiply([bagOf(entry("manzana", 2)), num("1/2")]))).toEqual(["manzana:1"]);
  });

  test("el escalar nulo se ignora: devuelve la bolsa intacta", () => {
    expect(pairs(multiply([bagOf(entry("manzana", 2)), NULO]))).toEqual(["manzana:2"]);
  });

  test("un segundo argumento que no es número es error de ejecución", () => {
    expect(() => multiply([bagOf(entry("manzana", 2)), bagOf(entry("pera", 3))])).toThrow(
      DataflowError
    );
    try {
      multiply([bagOf(entry("manzana", 2)), bagOf(entry("pera", 3))]);
    } catch (err) {
      expect((err as DataflowError).code).toBe("EXPECTED_NUMBER");
    }
  });
});

describe("divide — §3.1.4", () => {
  test("divide cada entrada por el divisor", () => {
    expect(pairs(divide([bagOf(entry("manzana", 6), entry("pera", 4)), num(2)]))).toEqual([
      "manzana:3",
      "pera:2",
    ]);
  });

  test("el resultado es exacto", () => {
    expect(pairs(divide([bagOf(entry("manzana", 1)), num(3)]))).toEqual(["manzana:1/3"]);
  });

  test("el divisor 0 es error de ejecución", () => {
    try {
      divide([bagOf(entry("manzana", 6)), num(0)]);
      throw new Error("debió fallar");
    } catch (err) {
      expect((err as DataflowError).code).toBe("DIVISION_BY_ZERO");
    }
  });

  test("el divisor nulo se ignora", () => {
    expect(pairs(divide([bagOf(entry("manzana", 6)), NULO]))).toEqual(["manzana:6"]);
  });
});
