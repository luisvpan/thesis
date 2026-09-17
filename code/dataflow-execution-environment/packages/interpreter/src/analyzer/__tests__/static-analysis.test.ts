// §4.2 La pasada estática, sobre el grafo y sin evaluar.

import { describe, expect, test } from "bun:test";
import { buildGraph } from "../../runtime/graph";
import { parseToAst } from "../../serializer";
import { analyze } from "../static-analysis";

function codesOf(program: string): string[] {
  const { ast, errors } = parseToAst(program);
  if (!ast) throw new Error(`error de sintaxis: ${errors[0]?.message}`);
  return analyze(buildGraph(ast)).map((error) => error.code);
}

const apple = `{"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 2}`;
const three = `{"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3}`;

describe("analyze", () => {
  test("un programa bien formado no produce errores", () => {
    expect(
      codesOf(`
        source frutas = ${apple};
        source tres = ${three};
        transform muchas = greater_than(frutas, tres);
        sink result = muchas;
      `)
    ).toEqual([]);
  });

  test("la bien-formación se reporta antes que el resto", () => {
    // El transform también tiene mala aridad, pero el ciclo manda.
    expect(
      codesOf(`
        transform a = sum(b);
        transform b = substract(a);
        sink result = a;
      `)
    ).toEqual(["CIRCULAR_DEPENDENCY"]);
  });

  test("aridad, por exceso y por defecto", () => {
    expect(codesOf(`source x = ${three}; transform t = first(x, x); sink s = t;`)).toEqual([
      "ARITY_ERROR",
    ]);
    expect(codesOf(`source x = ${three}; transform t = order(x); sink s = t;`)).toEqual([
      "ARITY_ERROR",
    ]);
    expect(codesOf(`source x = ${three}; transform t = sum(x, x, x, x); sink s = t;`)).toEqual([]);
  });

  test("la categoría de salida de un transform la fija su operación", () => {
    // count devuelve una bolsa, así que sirve de umbral; compare devuelve un booleano y no.
    expect(
      codesOf(`
        source x = ${apple};
        transform cuantas = count(x);
        transform pocas = less_than(x, cuantas);
        sink s = pocas;
      `)
    ).toEqual([]);

    expect(
      codesOf(`
        source x = ${apple};
        transform iguales = compare(x, x);
        transform pocas = less_than(x, iguales);
        sink s = pocas;
      `)
    ).toEqual(["TYPE_ERROR"]);
  });

  test("un criterio donde se espera una bolsa", () => {
    expect(
      codesOf(`
        source c = {"sourceType": "filter", "properties": ["subtype"], "subtype": "manzana"};
        source x = ${apple};
        transform t = sum(x, c);
        sink s = t;
      `)
    ).toEqual(["TYPE_ERROR"]);
  });

  test("order rechaza un criterio de filtro y filter uno de orden", () => {
    const criteria = `
      source filtro = {"sourceType": "filter", "properties": ["subtype"], "subtype": "manzana"};
      source orden = {"sourceType": "order", "properties": ["quantity"], "quantity": "asc"};
      source x = ${apple};
    `;

    expect(codesOf(`${criteria} transform t = order(x, filtro); sink s = t;`)).toEqual([
      "INVALID_CRITERION",
    ]);
    expect(codesOf(`${criteria} transform t = filter(x, orden); sink s = t;`)).toEqual([
      "INVALID_CRITERION",
    ]);
    expect(codesOf(`${criteria} transform t = filter(x, filtro); sink s = t;`)).toEqual([]);
  });

  test("los nodos incompletos no invalidan el programa (§2.5)", () => {
    expect(
      codesOf(`
        source x = ${apple};
        source sin_valor = ;
        transform t = filter(x, sin_valor);
        transform u = ;
        transform v = multiply(x, u);
        sink s = t;
        sink r = v;
      `)
    ).toEqual([]);
  });

  test("varios errores se reportan juntos", () => {
    expect(
      codesOf(`
        source x = ${apple};
        transform a = substract(x);
        transform b = inventada(x);
        sink s = a;
      `)
    ).toEqual(["ARITY_ERROR", "UNKNOWN_OPERATION"]);
  });
});
