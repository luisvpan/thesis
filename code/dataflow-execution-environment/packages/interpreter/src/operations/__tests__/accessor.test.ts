// §3.5 — acceso posicional

import { describe, expect, test } from "bun:test";
import { bagOf, entry, pairs } from "../../__tests__/helpers";
import { first, last } from "../accessor";

describe("first — §3.5.1", () => {
  test("devuelve la primera entrada del orden vigente", () => {
    expect(pairs(first([bagOf(entry("manzana", 2), entry("pera", 3), entry("uva", 1))]))).toEqual([
      "manzana:2",
    ]);
  });

  test("señala una pila individual: no agrupa los repetidos", () => {
    expect(pairs(first([bagOf(entry("manzana", 2), entry("manzana", 3))]))).toEqual(["manzana:2"]);
  });

  test("sobre nulo devuelve nulo", () => {
    expect(pairs(first([bagOf()]))).toEqual([]);
  });
});

describe("last — §3.5.2", () => {
  test("devuelve la última entrada del orden vigente", () => {
    expect(pairs(last([bagOf(entry("manzana", 2), entry("pera", 3), entry("uva", 1))]))).toEqual([
      "uva:1",
    ]);
  });

  test("señala una pila individual: no agrupa los repetidos", () => {
    expect(pairs(last([bagOf(entry("manzana", 2), entry("manzana", 3))]))).toEqual(["manzana:3"]);
  });

  test("sobre nulo devuelve nulo", () => {
    expect(pairs(last([bagOf()]))).toEqual([]);
  });
});
