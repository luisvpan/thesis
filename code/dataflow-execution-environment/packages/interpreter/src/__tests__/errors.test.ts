// §4 Errores: naturaleza, nodo, nodo causante y salida.

import { describe, expect, test } from "bun:test";
import { Interpreter } from "../index";
import { RuntimeError } from "../runtime/errors";
import { dataLiteral, numberLiteral } from "./helpers";

const num = numberLiteral;

async function errorsOf(program: string): Promise<RuntimeError[]> {
  const result = await new Interpreter().execute(program);
  return result.errors.filter((error): error is RuntimeError => error instanceof RuntimeError);
}

async function firstError(program: string): Promise<RuntimeError> {
  const [error] = await errorsOf(program);
  if (!error) throw new Error("se esperaba un error y no hubo ninguno");
  return error;
}

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
      `source x = {"sourceType": "data", "category": "concreto", "type": "comida", "quantity": 2};`
    );
    expect(error.code).toBe("INVALID_OBJECT");
    expect(error.detail).toContain("subtype");
  });

  test("un error estático invalida el programa completo: no se evalúa nada", async () => {
    const result = await new Interpreter().execute(`
      source x = ${num(1)};
      transform sano = sum(x, x);
      transform roto = substract(x);
      sink a = sano;
      sink b = roto;
    `);
    expect(result.errors).toHaveLength(1);
    expect(result.results.size).toBe(0);
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
    expect(error.sinkId).toBe("result");
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
    expect(error.sinkId).toBe("result");
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
