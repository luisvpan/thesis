// §1.2 El álgebra de la bolsa: agregación, denotación y números.

import { describe, expect, test } from "bun:test";
import { bagOf, entry, num } from "../../__tests__/helpers";
import { NULO, aggregate, asNumber, denote, denotationallyEqual, identityKey, isNulo } from "../bag";

describe("identidad", () => {
  test("coinciden solo si coinciden las cuatro partes", () => {
    expect(identityKey(entry("manzana", 1))).toBe(identityKey(entry("manzana", 9)));
    expect(identityKey(entry("manzana", 1))).not.toBe(identityKey(entry("pera", 1)));
    expect(identityKey(entry("cubo", 1, { color: "rojo" }))).not.toBe(
      identityKey(entry("cubo", 1, { color: "azul" }))
    );
  });

  test("el orden de los atributos no altera la identidad", () => {
    expect(identityKey(entry("cubo", 1, { color: "rojo", size: "grande" }))).toBe(
      identityKey(entry("cubo", 1, { size: "grande", color: "rojo" }))
    );
  });

  test("ningún valor puede colisionar con el separador", () => {
    expect(identityKey(entry("a:b", 1))).not.toBe(identityKey(entry("a", 1, {}, { type: "b" })));
  });
});

describe("agregación y denotación", () => {
  test("aggregate colapsa por identidad y conserva los ceros", () => {
    const entries = aggregate([entry("manzana", 2), entry("pera", 3), entry("manzana", -2)]);
    expect(entries.map((item) => `${item.subtype}:${item.quantity.toFraction()}`)).toEqual([
      "manzana:0",
      "pera:3",
    ]);
  });

  test("denote descarta el soporte nulo", () => {
    expect(denote(bagOf(entry("manzana", 2), entry("manzana", -2))).size).toBe(0);
    expect(denote(bagOf(entry("manzana", 2), entry("pera", 0))).size).toBe(1);
  });

  test("la igualdad denotacional ignora orden, agrupación y ceros", () => {
    expect(denotationallyEqual(bagOf(entry("manzana", 0)), NULO)).toBe(true);
    expect(
      denotationallyEqual(bagOf(entry("manzana", 1), entry("manzana", 2)), bagOf(entry("manzana", 3)))
    ).toBe(true);
  });
});

describe("nulo", () => {
  test("la bolsa vacía es nulo", () => {
    expect(isNulo(NULO)).toBe(true);
    expect(isNulo(bagOf())).toBe(true);
    expect(isNulo(bagOf(entry("manzana", 0)))).toBe(false);
  });
});

describe("números (§1.2.6)", () => {
  test("una bolsa de entradas numéricas es un número", () => {
    expect(asNumber(num(3))?.toFraction()).toBe("3");
    expect(asNumber(num("1/3"))?.toFraction()).toBe("1/3");
  });

  test("varias entradas numéricas suman su valor", () => {
    expect(asNumber(bagOf(...num(2).entries, ...num(3).entries))?.toFraction()).toBe("5");
  });

  test("el número 0 es un número, pero una manzana en 0 no", () => {
    expect(asNumber(num(0))?.toFraction()).toBe("0");
    expect(asNumber(bagOf(entry("manzana", 0)))).toBeNull();
  });

  test("nulo no es un número: es un argumento ausente", () => {
    expect(asNumber(NULO)).toBeNull();
  });

  test("una bolsa mixta no es un número", () => {
    expect(asNumber(bagOf(...num(2).entries, entry("manzana", 1)))).toBeNull();
  });
});
