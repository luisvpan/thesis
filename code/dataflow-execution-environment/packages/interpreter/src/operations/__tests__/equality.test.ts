// §3.2.3 — igualdad denotacional

import { describe, expect, test } from "bun:test";
import { NULO } from "../../runtime/bag";
import type { BooleanValue } from "../../runtime/types";
import { bagOf, entry, num } from "../../__tests__/helpers";
import { compare } from "../equality";

function equal(a: Parameters<typeof compare>[0][number], b: Parameters<typeof compare>[0][number]): boolean {
  return (compare([a, b]) as BooleanValue).value;
}

describe("compare — §3.2.3", () => {
  test("devuelve un booleano", () => {
    expect(compare([bagOf(entry("manzana", 3)), bagOf(entry("manzana", 3))])).toEqual({
      kind: "booleano",
      value: true,
    });
  });

  test("ignora la agrupación", () => {
    expect(equal(bagOf(entry("manzana", 3)), bagOf(entry("manzana", 1), entry("manzana", 2)))).toBe(
      true
    );
  });

  test("ignora el orden", () => {
    expect(
      equal(
        bagOf(entry("manzana", 2), entry("pera", 1)),
        bagOf(entry("pera", 1), entry("manzana", 2))
      )
    ).toBe(true);
  });

  test("ignora las cantidades 0", () => {
    expect(equal(bagOf(entry("manzana", 0)), NULO)).toBe(true);
    expect(equal(bagOf(entry("manzana", 0), entry("pera", 0)), NULO)).toBe(true);
    expect(equal(bagOf(entry("manzana", 2), entry("manzana", -2)), NULO)).toBe(true);
  });

  test("distingue vectores distintos", () => {
    expect(equal(bagOf(entry("manzana", 2)), bagOf(entry("manzana", 3)))).toBe(false);
    expect(equal(bagOf(entry("manzana", 2)), bagOf(entry("pera", 2)))).toBe(false);
  });

  test("los atributos son parte de la identidad", () => {
    expect(
      equal(bagOf(entry("cubo", 1, { color: "rojo" })), bagOf(entry("cubo", 1, { color: "azul" })))
    ).toBe(false);
  });

  test("compara racionales de forma exacta", () => {
    expect(equal(num("1/3"), num("1/3"))).toBe(true);
    expect(equal(num("1/3"), num("0.333"))).toBe(false);
  });
});
