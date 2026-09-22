// serialize / deserialize: texto ↔ Program

import { describe, expect, test } from "bun:test";
import Fraction from "fraction.js";
import { createBag, createFilterCriterion, createOrderCriterion } from "../bag-builder";
import { deserialize, serialize } from "../serializer";
import type { BagLiteral, CriterionLiteral, Program, SourceStatement } from "../program";
import { dataLiteral, numberLiteral } from "./helpers";

function sourceValue(program: Program, index = 0) {
  return (program.statements[index] as SourceStatement).value;
}

describe("serialize", () => {
  test("un objeto es una bolsa de una entrada", () => {
    const { program, errors } = serialize(`source x = ${dataLiteral("concreto", "comida", "manzana", 2)};`);
    expect(errors).toHaveLength(0);

    const value = sourceValue(program!) as BagLiteral;
    expect(value.type).toBe("BagLiteral");
    expect(value.entries).toHaveLength(1);
    expect(value.entries[0].subtype).toBe("manzana");
    expect(value.entries[0].quantity.equals(new Fraction(2))).toBe(true);
  });

  test("un grupo es una bolsa de varias entradas", () => {
    const { program } = serialize(`
      source items = [${dataLiteral("concreto", "comida", "uva", 5)}, ${dataLiteral("concreto", "comida", "pera", 1)}];
    `);
    const value = sourceValue(program!) as BagLiteral;
    expect(value.entries.map((entry) => entry.subtype)).toEqual(["uva", "pera"]);
  });

  test("un source sin valor queda incompleto, no vacío", () => {
    const { program } = serialize("source x = ;");
    expect(sourceValue(program!)).toBeUndefined();
  });

  test("las fracciones se leen exactas", () => {
    const { program } = serialize(`source x = ${numberLiteral("1/3")};`);
    const value = sourceValue(program!) as BagLiteral;
    expect(value.entries[0].quantity.equals(new Fraction(1, 3))).toBe(true);
  });

  test("el criterio conserva su subtipo", () => {
    const { program } = serialize(
      `source c = {"sourceType": "order", "properties": ["size"], "size": ["a", "b"]};`
    );
    const value = sourceValue(program!) as CriterionLiteral;
    expect(value.type).toBe("CriterionLiteral");
    expect(value.sourceType).toBe("order");
    expect(value.values).toEqual({ size: ["a", "b"] });
  });

  test("los argumentos de un transform son identificadores", () => {
    const { program } = serialize(`transform t = sum(a, b);`);
    expect(program!.statements[0]).toEqual({
      type: "TransformStatement",
      identifier: "t",
      operation: "sum",
      arguments: [
        { type: "Identifier", name: "a" },
        { type: "Identifier", name: "b" },
      ],
    });
  });

  test("reporta los errores de sintaxis con su posición", () => {
    const { program, errors } = serialize("source x = 5;");
    expect(program).toBeNull();
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0].line).toBe(1);
  });
});

describe("deserialize", () => {
  test("escribe una bolsa de una entrada como objeto y varias como grupo", () => {
    const one = createBag().add({
      category: "concreto",
      type: "comida",
      subtype: "manzana",
      quantity: 2,
    });

    expect(deserialize({ type: "Program", statements: [{ type: "SourceStatement", identifier: "x", value: one }] }))
      .toBe(
        `source x = {"sourceType": "data", "category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 2};`
      );

    const two = one.add({ category: "concreto", type: "comida", subtype: "pera", quantity: 1 });
    expect(
      deserialize({ type: "Program", statements: [{ type: "SourceStatement", identifier: "x", value: two }] })
    ).toContain("[{");
  });

  test("escribe los racionales como fracción, sin perder exactitud", () => {
    const bag = createBag().add({
      category: "abstracto",
      type: "numero",
      subtype: "racional",
      quantity: "1/3",
    });
    const text = deserialize({
      type: "Program",
      statements: [{ type: "SourceStatement", identifier: "x", value: bag }],
    });
    expect(text).toContain(`"quantity": 1/3`);
  });

  test("ida y vuelta de los tres nodos incompletos (§2.5)", () => {
    const original = "source x = ;\ntransform t = ;\nsink s = ;";
    const { program, errors } = serialize(original);
    expect(errors).toHaveLength(0);

    const text = deserialize(program!);
    expect(text).toBe(original);

    // Y el texto generado vuelve a parsear: no se cuela un `t = ();`.
    const { program: again, errors: againErrors } = serialize(text);
    expect(againErrors).toHaveLength(0);
    expect(again!.statements).toHaveLength(3);
  });

  test("ida y vuelta conserva el valor", () => {
    const original = `source x = ${numberLiteral("1/3")};\ntransform t = sum(x, x);\nsink out = t;`;
    const { program } = serialize(original);
    const { program: again } = serialize(deserialize(program!));

    const first = sourceValue(program!) as BagLiteral;
    const second = sourceValue(again!) as BagLiteral;
    expect(second.entries[0].quantity.equals(first.entries[0].quantity)).toBe(true);
    expect(again!.statements).toHaveLength(3);
  });

  test("escribe los criterios con su subtipo", () => {
    const filtro = createFilterCriterion({ properties: ["subtype"], values: { subtype: "manzana" } });
    const orden = createOrderCriterion({ properties: ["quantity"], values: { quantity: "asc" } });

    const text = deserialize({
      type: "Program",
      statements: [
        { type: "SourceStatement", identifier: "f", value: filtro },
        { type: "SourceStatement", identifier: "o", value: orden },
      ],
    });

    expect(text).toContain(`{"sourceType": "filter", "properties": ["subtype"], "subtype": "manzana"}`);
    expect(text).toContain(`{"sourceType": "order", "properties": ["quantity"], "quantity": "asc"}`);
  });
});
