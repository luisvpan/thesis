// §4 Errores: naturaleza, nodo, nodo causante y salida.

import { describe, expect, test } from "bun:test";
import {
  ERROR_CODES,
  Interpreter,
  RUNTIME_ERROR_CODES,
  STATIC_ERROR_CODES,
  SYNTAX_ERROR_CODES,
  isDataflowError,
  type DataflowError,
  type ErrorCode,
} from "../index";
import { dataLiteral, numberLiteral, only } from "./helpers";

const num = numberLiteral;

async function firstError(program: string): Promise<DataflowError> {
  const [error] = (await new Interpreter().execute(program)).errors;
  if (!error) throw new Error("se esperaba un error y no hubo ninguno");
  return error;
}

describe("Forma del error", () => {
  test("las tres fases comparten forma: phase + code + detail", async () => {
    const sintaxis = await firstError("source x = 5;");
    expect(sintaxis.phase).toBe("syntax");
    expect(sintaxis.code).toBe("SYNTAX_ERROR");
    expect(sintaxis.line).toBe(1);

    const estatico = await firstError(`source x = ${num(1)}; transform t = substract(x); sink s = t;`);
    expect(estatico.phase).toBe("static");
    expect(estatico.code).toBe("ARITY_ERROR");

    const ejecucion = await firstError(`
      source a = ${num(1)};
      source cero = ${num(0)};
      transform t = divide(a, cero);
      sink s = t;
    `);
    expect(ejecucion.phase).toBe("runtime");
    expect(ejecucion.code).toBe("DIVISION_BY_ZERO");

    // El detalle se conserva aparte del mensaje situado.
    expect(ejecucion.message).toContain(ejecucion.detail);
    expect(ejecucion.message).not.toBe(ejecucion.detail);
  });

  test("los códigos se exportan como valores, para recorrerlos", () => {
    expect(ERROR_CODES).toContain("DIVISION_BY_ZERO");
    expect(ERROR_CODES).toHaveLength(
      SYNTAX_ERROR_CODES.length + STATIC_ERROR_CODES.length + RUNTIME_ERROR_CODES.length
    );

    // Un mapa de mensajes exhaustivo: si mañana aparece un código nuevo, no compila.
    const mensajes: Record<ErrorCode, string> = {
      SYNTAX_ERROR: "el programa no se entiende",
      DUPLICATE_IDENTIFIER: "hay dos cosas con el mismo nombre",
      UNDEFINED_REFERENCE: "falta algo por conectar",
      CIRCULAR_DEPENDENCY: "esto se muerde la cola",
      UNKNOWN_OPERATION: "esa operación no existe",
      ARITY_ERROR: "faltan o sobran cosas",
      TYPE_ERROR: "eso no va ahí",
      INVALID_CRITERION: "ese criterio no sirve aquí",
      INVALID_OBJECT: "a ese objeto le falta identidad",
      EXPECTED_NUMBER: "aquí hace falta un número",
      DIVISION_BY_ZERO: "no se puede repartir entre cero",
    };

    expect(Object.keys(mensajes).sort()).toEqual([...ERROR_CODES].sort());
  });

  test("isDataflowError reconoce los errores del intérprete", async () => {
    const [error] = (await new Interpreter().execute("source x = 5;")).errors;
    expect(isDataflowError(error)).toBe(true);
    expect(isDataflowError(new Error("otra cosa"))).toBe(false);
  });
});

describe("Errores de sintaxis (§4.1)", () => {
  test("un grupo con un criterio dentro no se ajusta a la gramática", async () => {
    const result = await new Interpreter().execute(`
      source mezcla = [
        ${dataLiteral("concreto", "comida", "manzana", 2)},
        {"sourceType": "filter", "properties": ["subtype"], "subtype": "manzana"}
      ];
      sink result = mezcla;
    `);
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].message).toContain("grupo");
  });

  test("una categoría que no existe se reporta con su posición", async () => {
    const result = await new Interpreter().execute(
      `source x = {"sourceType": "data", "category": "liquido", "type": "agua", "subtype": "dulce", "quantity": 1};`
    );
    expect(result.errors).toHaveLength(1);
    expect((result.errors[0] as { line?: number }).line).toBe(1);
  });
});

describe("Errores estáticos (§4.2)", () => {
  test("nombre duplicado", async () => {
    const error = await firstError(`
      source x = ${num(1)};
      source x = ${num(2)};
      sink result = x;
    `);
    expect(error.code).toBe("DUPLICATE_IDENTIFIER");
    expect(error.phase).toBe("static");
  });

  test("referencia sin resolver informa el nodo y la causa", async () => {
    const error = await firstError(`
      source one = ${num(1)};
      transform a = sum(no_existe, one);
      sink result = a;
    `);
    expect(error.code).toBe("UNDEFINED_REFERENCE");
    expect(error.nodeId).toBe("a");
    expect(error.causeNodeId).toBe("no_existe");
  });

  test("ciclo", async () => {
    const error = await firstError(`
      source x = ${num(1)};
      transform a = sum(b, x);
      transform b = sum(a, x);
      sink result = a;
    `);
    expect(error.code).toBe("CIRCULAR_DEPENDENCY");
  });

  test("operación desconocida", async () => {
    const error = await firstError(`
      source x = ${num(1)};
      transform a = order_asc(x);
      sink result = a;
    `);
    expect(error.code).toBe("UNKNOWN_OPERATION");
    expect(error.nodeId).toBe("a");
  });

  test("aridad", async () => {
    const error = await firstError(`
      source x = ${num(1)};
      transform a = substract(x);
      sink result = a;
    `);
    expect(error.code).toBe("ARITY_ERROR");
    expect(error.detail).toContain("exactamente 2");
  });

  test("categoría de valor equivocada: una bolsa donde se espera un criterio", async () => {
    const error = await firstError(`
      source frutas = ${dataLiteral("concreto", "comida", "manzana", 2)};
      source otra = ${dataLiteral("concreto", "comida", "pera", 1)};
      transform filtradas = filter(frutas, otra);
      sink result = filtradas;
    `);
    expect(error.code).toBe("TYPE_ERROR");
    expect(error.nodeId).toBe("filtradas");
    expect(error.causeNodeId).toBe("otra");
  });

  test("categoría de valor equivocada: un booleano como argumento", async () => {
    const error = await firstError(`
      source x = ${num(1)};
      transform iguales = compare(x, x);
      transform total = sum(iguales, x);
      sink result = total;
    `);
    expect(error.code).toBe("TYPE_ERROR");
    expect(error.detail).toContain("booleano");
  });

  test("criterio inadecuado: subtipo equivocado", async () => {
    const error = await firstError(`
      source frutas = ${dataLiteral("concreto", "comida", "manzana", 2)};
      source por_cantidad = {"sourceType": "order", "properties": ["quantity"], "quantity": "asc"};
      transform filtradas = filter(frutas, por_cantidad);
      sink result = filtradas;
    `);
    expect(error.code).toBe("INVALID_CRITERION");
    expect(error.causeNodeId).toBe("por_cantidad");
  });

  test("criterio inadecuado: filtro con valor múltiple", async () => {
    const error = await firstError(`
      source varios = {"sourceType": "filter", "properties": ["subtype"], "subtype": ["manzana", "pera"]};
      sink result = varios;
    `);
    expect(error.code).toBe("INVALID_CRITERION");
    expect(error.detail).toContain("valor único");
  });

  test("criterio inadecuado: filtro sobre la cantidad", async () => {
    const error = await firstError(`
      source por_cantidad = {"sourceType": "filter", "properties": ["quantity"], "quantity": "3"};
      sink result = por_cantidad;
    `);
    expect(error.code).toBe("INVALID_CRITERION");
    expect(error.detail).toContain("cantidad");
  });

  test("objeto inválido: un componente de identidad en blanco", async () => {
    const error = await firstError(
      `source x = {"sourceType": "data", "category": "concreto", "type": "comida", "quantity": 2};
       sink s = x;`
    );
    expect(error.code).toBe("INVALID_OBJECT");
    expect(error.detail).toContain("subtype");
  });

  test("se aísla por salida: la salida sana calcula igual (§4.2)", async () => {
    const result = await new Interpreter().execute(`
      source x = ${num(1)};
      transform sano = sum(x, x);
      transform roto = substract(x);
      sink a = sano;
      sink b = roto;
    `);

    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].code).toBe("ARITY_ERROR");
    expect(result.errors[0].sinkIds).toEqual(["b"]);

    expect(only(result.results.get("a")!)).toBe("2");
    expect(result.results.has("b")).toBe(false);
  });

  test("un nodo que alcanzan dos salidas las apaga a las dos", async () => {
    const result = await new Interpreter().execute(`
      source x = ${num(1)};
      transform roto = substract(x);
      transform despues = sum(roto, x);
      sink a = roto;
      sink b = despues;
      sink c = x;
    `);

    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].sinkIds.sort()).toEqual(["a", "b"]);
    expect(result.results.has("a")).toBe(false);
    expect(result.results.has("b")).toBe(false);
    expect(only(result.results.get("c")!)).toBe("1");
  });
});

describe("Errores de ejecución (§4.3)", () => {
  test("número esperado, situado en nodo, causa y salida", async () => {
    const error = await firstError(`
      source frutas = ${dataLiteral("concreto", "comida", "manzana", 2)};
      source otras = ${dataLiteral("concreto", "comida", "pera", 3)};
      transform escaladas = multiply(frutas, otras);
      sink result = escaladas;
    `);
    expect(error.code).toBe("EXPECTED_NUMBER");
    expect(error.phase).toBe("runtime");
    expect(error.nodeId).toBe("escaladas");
    expect(error.causeNodeId).toBe("otras");
    expect(error.sinkIds).toEqual(["result"]);
    expect(error.message).toContain("en 'escaladas'");
    expect(error.message).toContain("por 'otras'");
    expect(error.message).toContain("salida 'result'");
  });

  test("división por cero", async () => {
    const error = await firstError(`
      source a = ${num(10)};
      source cero = ${num(0)};
      transform mal = divide(a, cero);
      sink result = mal;
    `);
    expect(error.code).toBe("DIVISION_BY_ZERO");
    expect(error.nodeId).toBe("mal");
    expect(error.causeNodeId).toBe("cero");
    expect(error.sinkIds).toEqual(["result"]);
  });

  test("están aislados por salida (§4.3)", async () => {
    const result = await new Interpreter().execute(`
      source a = ${num(10)};
      source cero = ${num(0)};
      transform mal = divide(a, cero);
      sink roto = mal;
      sink sano = a;
    `);
    expect(result.errors).toHaveLength(1);
    expect(result.results.has("sano")).toBe(true);
  });
});
