// §3.2.1 y §3.2.2 — comparación con umbral

import { describe, expect, test } from "bun:test";
import { NULO } from "../../runtime/bag";
import type { DataflowError } from "../../runtime/errors";
import { bagOf, entry, num, pairs } from "../../__tests__/helpers";
import { greaterThan, lessThan } from "../comparison";

describe("lessThan — §3.2.1", () => {
  test("conserva las entradas por debajo del umbral", () => {
    expect(pairs(lessThan([bagOf(entry("manzana", 2), entry("pera", 5)), num(5)]))).toEqual([
      "manzana:2",
    ]);
  });

  test("agrupa por identidad y compara el total", () => {
    // 2 + 3 = 5, que no es menor que 4
    expect(pairs(lessThan([bagOf(entry("manzana", 2), entry("manzana", 3)), num(4)]))).toEqual([]);
  });

  test("sin coincidencias devuelve nulo", () => {
    expect(pairs(lessThan([bagOf(entry("pera", 5)), num(2)]))).toEqual([]);
  });

  test("el umbral nulo se ignora: devuelve la bolsa intacta", () => {
    expect(pairs(lessThan([bagOf(entry("pera", 5)), NULO]))).toEqual(["pera:5"]);
  });

  test("un umbral que no es número es error de ejecución", () => {
    try {
      lessThan([bagOf(entry("pera", 5)), bagOf(entry("manzana", 2))]);
      throw new Error("debió fallar");
    } catch (err) {
      expect((err as DataflowError).code).toBe("EXPECTED_NUMBER");
    }
  });
});

describe("greaterThan — §3.2.2", () => {
  test("conserva las entradas por encima del umbral", () => {
    expect(pairs(greaterThan([bagOf(entry("manzana", 2), entry("pera", 5)), num(3)]))).toEqual([
      "pera:5",
    ]);
  });

  test("agrupa por identidad y compara el total", () => {
    expect(pairs(greaterThan([bagOf(entry("manzana", 2), entry("manzana", 3)), num(4)]))).toEqual([
      "manzana:5",
    ]);
  });

  test("sin coincidencias devuelve nulo", () => {
    expect(pairs(greaterThan([bagOf(entry("manzana", 2)), num(5)]))).toEqual([]);
  });

  test("conserva el orden de las entradas que pasan", () => {
    const result = greaterThan([
      bagOf(entry("uva", 9), entry("manzana", 1), entry("pera", 7)),
      num(5),
    ]);
    expect(pairs(result)).toEqual(["uva:9", "pera:7"]);
  });
});
