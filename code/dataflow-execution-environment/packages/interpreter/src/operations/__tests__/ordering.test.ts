// §3.3.1 — orden

import { describe, expect, test } from "bun:test";
import { bagOf, entry, orderCriterion, pairs } from "../../__tests__/helpers";
import { order } from "../ordering";

const size = (subtype: string, quantity: number, value: string) =>
  entry(subtype, quantity, { size: value });

describe("order — §3.3.1", () => {
  test("ordena por cantidad ascendente", () => {
    const result = order([
      bagOf(entry("manzana", 3), entry("pera", 1), entry("uva", 2)),
      orderCriterion(["quantity"], { quantity: "asc" }),
    ]);
    expect(pairs(result)).toEqual(["pera:1", "uva:2", "manzana:3"]);
  });

  test("ordena por cantidad descendente", () => {
    const result = order([
      bagOf(entry("manzana", 3), entry("pera", 1), entry("uva", 2)),
      orderCriterion(["quantity"], { quantity: "desc" }),
    ]);
    expect(pairs(result)).toEqual(["manzana:3", "uva:2", "pera:1"]);
  });

  test("la cantidad se compara numéricamente, no como texto", () => {
    const result = order([
      bagOf(entry("manzana", 10), entry("pera", 2)),
      orderCriterion(["quantity"], { quantity: "asc" }),
    ]);
    expect(pairs(result)).toEqual(["pera:2", "manzana:10"]);
  });

  test("ordena textos alfabéticamente", () => {
    const result = order([
      bagOf(entry("uva", 1), entry("manzana", 1), entry("pera", 1)),
      orderCriterion(["subtype"], { subtype: "asc" }),
    ]);
    expect(pairs(result)).toEqual(["manzana:1", "pera:1", "uva:1"]);
  });

  test("una secuencia fija el orden explícitamente", () => {
    const result = order([
      bagOf(size("estrella", 1, "grande"), size("estrella", 1, "pequeña"), size("estrella", 1, "mediana")),
      orderCriterion(["size"], { size: ["pequeña", "mediana", "grande"] }),
    ]);
    expect(
      (result as { entries: { attributes: Record<string, string> }[] }).entries.map((item) => item.attributes.size)
    ).toEqual(["pequeña", "mediana", "grande"]);
  });

  test("una secuencia puede no ser ni ascendente ni descendente", () => {
    const result = order([
      bagOf(size("estrella", 1, "pequeña"), size("estrella", 1, "mediana"), size("estrella", 1, "grande")),
      orderCriterion(["size"], { size: ["mediana", "pequeña", "grande"] }),
    ]);
    expect(
      (result as { entries: { attributes: Record<string, string> }[] }).entries.map((item) => item.attributes.size)
    ).toEqual(["mediana", "pequeña", "grande"]);
  });

  test("las entradas fuera de la secuencia van al final", () => {
    const result = order([
      bagOf(size("estrella", 1, "enorme"), size("estrella", 2, "pequeña")),
      orderCriterion(["size"], { size: ["pequeña", "grande"] }),
    ]);
    expect(pairs(result)).toEqual(["estrella:2", "estrella:1"]);
  });

  test("manda el primer criterio y los siguientes desempatan", () => {
    const bag = bagOf(
      entry("manzana", 3, { color: "azul" }),
      entry("pera", 1, { color: "rojo" }),
      entry("uva", 3, { color: "rojo" })
    );
    const byQuantity = orderCriterion(["quantity"], { quantity: "asc" });
    const byColor = orderCriterion(["color"], { color: "asc" });

    // cantidad manda: 1 va primero; entre los dos de 3, desempata el color
    expect(pairs(order([bag, byQuantity, byColor]))).toEqual(["pera:1", "manzana:3", "uva:3"]);
    // con los criterios al revés manda el color: azul antes que rojo
    expect(pairs(order([bag, byColor, byQuantity]))).toEqual(["manzana:3", "pera:1", "uva:3"]);
  });

  test("el orden es estable ante empates", () => {
    const result = order([
      bagOf(entry("uva", 1), entry("manzana", 1), entry("pera", 1)),
      orderCriterion(["quantity"], { quantity: "asc" }),
    ]);
    expect(pairs(result)).toEqual(["uva:1", "manzana:1", "pera:1"]);
  });

  test("agrupa por identidad antes de ordenar", () => {
    const result = order([
      bagOf(entry("manzana", 2), entry("pera", 3), entry("manzana", 4)),
      orderCriterion(["quantity"], { quantity: "asc" }),
    ]);
    expect(pairs(result)).toEqual(["pera:3", "manzana:6"]);
  });

  test("sin criterios completos devuelve la bolsa sin cambios", () => {
    const original = bagOf(entry("manzana", 3), entry("pera", 1));
    expect(pairs(order([original]))).toEqual(["manzana:3", "pera:1"]);
    expect(pairs(order([original, orderCriterion(["quantity"], {})]))).toEqual([
      "manzana:3",
      "pera:1",
    ]);
  });

  test("ordenar nulo devuelve nulo", () => {
    expect(pairs(order([bagOf(), orderCriterion(["quantity"], { quantity: "asc" })]))).toEqual([]);
  });
});
