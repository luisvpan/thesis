// El builder de programas: inmutable, con atajos por operación.

import { describe, expect, test } from "bun:test";
import { createBag, createOrderCriterion } from "../bag-builder";
import { Interpreter } from "../index";
import { OPERATIONS, type Operation } from "../operations/signatures";
import { createProgram, ProgramBuilder } from "../program-builder";
import type { TransformStatement } from "../program";
import { only, pairs } from "./helpers";

const manzanas = (quantity: number) =>
  createBag().add({ category: "concreto", type: "comida", subtype: "manzana", quantity });

function transformOf(builder: ProgramBuilder, identifier: string): TransformStatement {
  const stmt = builder
    .build()
    .statements.find((s) => s.type === "TransformStatement" && s.identifier === identifier);
  if (stmt?.type !== "TransformStatement") throw new Error(`no hay transform '${identifier}'`);
  return stmt;
}

describe("inmutabilidad", () => {
  test("cada método devuelve un builder nuevo y no toca el anterior", () => {
    const base = createProgram().source("x", manzanas(2));
    const una = base.sum("a", "x");
    const otra = base.count("b", "x");

    expect(una).not.toBe(base);
    expect(base.build().statements).toHaveLength(1);
    expect(una.build().statements).toHaveLength(2);
    expect(otra.build().statements).toHaveLength(2);
    // Las dos ramas parten de la misma base sin pisarse.
    expect(transformOf(una, "a").operation).toBe("sum");
    expect(transformOf(otra, "b").operation).toBe("count");
  });

  test("el builder está congelado", () => {
    expect(Object.isFrozen(createProgram())).toBe(true);
  });

  test("build() devuelve una copia: mutarla no afecta al builder", () => {
    const builder = createProgram().source("x", manzanas(2));
    builder.build().statements.push({ type: "SinkStatement", identifier: "s" });
    expect(builder.build().statements).toHaveLength(1);
  });
});

describe("sentencias", () => {
  test("source, transform y sink en orden de declaración", () => {
    const program = createProgram()
      .source("x", manzanas(2))
      .transform("t", "count", ["x"])
      .sink("s", "t")
      .build();

    expect(program.statements.map((s) => [s.type, s.identifier])).toEqual([
      ["SourceStatement", "x"],
      ["TransformStatement", "t"],
      ["SinkStatement", "s"],
    ]);
  });

  test("los argumentos se envuelven como identificadores", () => {
    expect(transformOf(createProgram().transform("t", "sum", ["a", "b"]), "t").arguments).toEqual([
      { type: "Identifier", name: "a" },
      { type: "Identifier", name: "b" },
    ]);
  });

  test("sin valor, las tres declaran un nodo incompleto (§2.5)", () => {
    const program = createProgram().source("x").transform("t").sink("s").build();

    expect(program.statements[0]).toEqual({ type: "SourceStatement", identifier: "x", value: undefined });
    expect(program.statements[1]).toEqual({
      type: "TransformStatement",
      identifier: "t",
      operation: undefined,
      arguments: [],
    });
    expect(program.statements[2]).toEqual({
      type: "SinkStatement",
      identifier: "s",
      sourceIdentifier: undefined,
    });
  });

  test("un programa con nodos incompletos se ejecuta y da nulo", async () => {
    const result = await new Interpreter().execute(
      createProgram().transform("t").sink("s", "t").build()
    );

    expect(result.errors).toHaveLength(0);
    expect(pairs(result.results.get("s")!)).toEqual([]);
  });
});

describe("atajos por operación", () => {
  /** El nombre del método que corresponde a cada operación de §5.2. */
  const METHOD_OF: Record<Operation, keyof ProgramBuilder> = {
    sum: "sum",
    substract: "substract",
    multiply: "multiply",
    divide: "divide",
    less_than: "lessThan",
    greater_than: "greaterThan",
    compare: "compare",
    order: "order",
    filter: "filter",
    first: "first",
    last: "last",
    count: "count",
  };

  test("hay un atajo por cada operación reconocida, y emite su operación", () => {
    for (const operation of OPERATIONS) {
      const method = METHOD_OF[operation];
      const shortcut = createProgram()[method] as (
        id: string,
        ...args: string[]
      ) => ProgramBuilder;

      expect(typeof shortcut).toBe("function");
      // Dos argumentos bastan para todas las firmas de §3.
      expect(transformOf(shortcut.call(createProgram(), "t", "a", "b"), "t").operation).toBe(
        operation
      );
    }
  });

  test("order y filter aceptan varios criterios", () => {
    expect(
      transformOf(createProgram().order("t", "bolsa", "c1", "c2"), "t").arguments.map((a) => a.name)
    ).toEqual(["bolsa", "c1", "c2"]);

    expect(
      transformOf(createProgram().filter("t", "bolsa", "c1", "c2"), "t").arguments.map((a) => a.name)
    ).toEqual(["bolsa", "c1", "c2"]);
  });

  test("sum es variádica", () => {
    expect(
      transformOf(createProgram().sum("t", "a", "b", "c", "d"), "t").arguments.map((a) => a.name)
    ).toEqual(["a", "b", "c", "d"]);
  });
});

describe("de punta a punta", () => {
  test("un programa construido con el builder se ejecuta", async () => {
    const program = createProgram()
      .source(
        "frutas",
        createBag()
          .add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 3 })
          .add({ category: "concreto", type: "comida", subtype: "pera", quantity: 1 })
      )
      .source(
        "porCantidad",
        createOrderCriterion({ properties: ["quantity"], values: { quantity: "asc" } })
      )
      .order("ordenadas", "frutas", "porCantidad")
      .first("primera", "ordenadas")
      .count("cuantas", "frutas")
      .sink("salida", "primera")
      .sink("total", "cuantas")
      .build();

    const result = await new Interpreter().execute(program);

    expect(result.errors).toHaveLength(0);
    expect(pairs(result.results.get("salida")!)).toEqual(["pera:1"]);
    expect(only(result.results.get("total")!)).toBe("4");
  });
});
