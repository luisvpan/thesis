// §3.4.1 — filtrado

import { describe, expect, test } from "bun:test";
import { bagOf, entry, filterCriterion, pairs } from "../../__tests__/helpers";
import { filter } from "../filtering";

const fruits = () => bagOf(entry("manzana", 2), entry("pera", 3), entry("uva", 1));

describe("filter — §3.4.1", () => {
  test("conserva lo que satisface el criterio", () => {
    const result = filter([fruits(), filterCriterion(["subtype"], { subtype: "manzana" })]);
    expect(pairs(result)).toEqual(["manzana:2"]);
  });

  test("entre criterios hay O", () => {
    const result = filter([
      fruits(),
      filterCriterion(["subtype"], { subtype: "manzana" }),
      filterCriterion(["subtype"], { subtype: "uva" }),
    ]);
    expect(pairs(result)).toEqual(["manzana:2", "uva:1"]);
  });

  test("entre las propiedades de un criterio hay Y", () => {
    const shapes = bagOf(
      entry("estrella", 2, { color: "roja" }),
      entry("estrella", 1, { color: "azul" }),
      entry("circulo", 3, { color: "roja" })
    );

    const result = filter([
      shapes,
      filterCriterion(["subtype", "color"], { subtype: "estrella", color: "roja" }),
    ]);
    expect(pairs(result)).toEqual(["estrella:2"]);
  });

  test("forma normal disyuntiva: O de conjunciones", () => {
    const shapes = bagOf(
      entry("estrella", 2, { color: "roja" }),
      entry("estrella", 1, { color: "azul" }),
      entry("circulo", 3, { color: "roja" })
    );

    const result = filter([
      shapes,
      filterCriterion(["subtype", "color"], { subtype: "estrella", color: "roja" }),
      filterCriterion(["subtype", "color"], { subtype: "estrella", color: "azul" }),
    ]);
    expect(pairs(result)).toEqual(["estrella:2", "estrella:1"]);
  });

  test("filtra por categoría y por tipo", () => {
    const mixed = bagOf(
      entry("manzana", 2, {}, { category: "concreto", type: "comida" }),
      entry("circulo", 1, {}, { category: "pictorico", type: "forma" })
    );

    expect(pairs(filter([mixed, filterCriterion(["category"], { category: "pictorico" })]))).toEqual([
      "circulo:1",
    ]);
    expect(pairs(filter([mixed, filterCriterion(["type"], { type: "comida" })]))).toEqual([
      "manzana:2",
    ]);
  });

  test("trabaja entrada por entrada y conserva los repetidos", () => {
    const result = filter([
      bagOf(entry("manzana", 2), entry("manzana", 4), entry("pera", 1)),
      filterCriterion(["subtype"], { subtype: "manzana" }),
    ]);
    expect(pairs(result)).toEqual(["manzana:2", "manzana:4"]);
  });

  test("sin criterios completos devuelve la bolsa sin cambios", () => {
    expect(pairs(filter([fruits()]))).toEqual(["manzana:2", "pera:3", "uva:1"]);
    expect(pairs(filter([fruits(), filterCriterion(["subtype"], {})]))).toEqual([
      "manzana:2",
      "pera:3",
      "uva:1",
    ]);
  });

  test("sin coincidencias devuelve nulo", () => {
    expect(pairs(filter([fruits(), filterCriterion(["subtype"], { subtype: "kiwi" })]))).toEqual([]);
  });
});
