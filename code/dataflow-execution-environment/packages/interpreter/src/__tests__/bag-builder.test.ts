// La API de construcción: un factory y una bolsa inmutable.

import { describe, expect, test } from "bun:test";
import Fraction from "fraction.js";
import { createBag, createFilterCriterion, createOrderCriterion } from "../bag-builder";
import { Interpreter } from "../index";
import { only } from "./helpers";

describe("createBag", () => {
  test("empieza vacía: la bolsa vacía es nulo", () => {
    expect(createBag().entries).toHaveLength(0);
  });

  test("cada add() devuelve una bolsa nueva y no muta la anterior", () => {
    const empty = createBag();
    const one = empty.add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 2 });
    const two = one.add({ category: "concreto", type: "comida", subtype: "pera", quantity: 1 });

    expect(one).not.toBe(empty);
    expect(two).not.toBe(one);
    expect(empty.entries).toHaveLength(0);
    expect(one.entries).toHaveLength(1);
    expect(two.entries).toHaveLength(2);
  });

  test("la bolsa devuelta está congelada", () => {
    const bag = createBag().add({
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: 2,
    });
    expect(Object.isFrozen(bag)).toBe(true);
    expect(Object.isFrozen(bag.entries)).toBe(true);
  });

  test("admite la cantidad como número, texto o Fraction", () => {
    const bag = createBag()
      .add({ category: "abstracto", type: "numero", subtype: "racional", quantity: 2 })
      .add({ category: "abstracto", type: "numero", subtype: "racional", quantity: "1/3" })
      .add({ category: "abstracto", type: "numero", subtype: "racional", quantity: new Fraction(1, 6) });

    expect(bag.entries.map((entry) => entry.quantity.toFraction())).toEqual(["2", "1/3", "1/6"]);
  });

  test("conserva los repetidos y el orden de declaración", () => {
    const bag = createBag()
      .add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 2 })
      .add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 4 });

    expect(bag.entries.map((entry) => entry.quantity.toFraction())).toEqual(["2", "4"]);
  });

  test("reset() devuelve la bolsa vacía, sin tocar la original", () => {
    const bag = createBag().add({
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: 2,
    });

    expect(bag.reset().entries).toHaveLength(0);
    expect(bag.entries).toHaveLength(1);
  });

  test("lo que construye se puede ejecutar", async () => {
    const result = await new Interpreter().execute({
      type: "Program",
      statements: [
        {
          type: "SourceStatement",
          identifier: "frutas",
          value: createBag()
            .add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 2 })
            .add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 4 }),
        },
        {
          type: "TransformStatement",
          identifier: "total",
          operation: "sum",
          arguments: [{ type: "Identifier", name: "frutas" }],
        },
        { type: "SinkStatement", identifier: "salida", sourceIdentifier: "total" },
      ],
    });

    expect(result.errors).toHaveLength(0);
    expect(only(result.results.get("salida")!)).toBe("6");
  });
});

describe("criterios", () => {
  test("cada factory declara su subtipo", () => {
    expect(createFilterCriterion({ properties: ["subtype"], values: { subtype: "manzana" } })).toEqual({
      type: "CriterionLiteral",
      sourceType: "filter",
      properties: ["subtype"],
      values: { subtype: "manzana" },
    });

    expect(createOrderCriterion({ properties: ["quantity"], values: { quantity: "desc" } })).toEqual({
      type: "CriterionLiteral",
      sourceType: "order",
      properties: ["quantity"],
      values: { quantity: "desc" },
    });
  });
});
