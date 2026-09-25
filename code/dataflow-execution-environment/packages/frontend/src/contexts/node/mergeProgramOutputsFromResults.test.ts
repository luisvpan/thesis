import { describe, expect, test } from "bun:test";
import type { ResultValue } from "@/services/executeProgram";
import { mergeProgramOutputsFromResults } from "./mergeProgramOutputsFromResults";
import type { DataflowNode } from "./types";

function outputWith(data: Record<string, unknown>): DataflowNode[] {
  return [
    { id: "out", type: "programOutput", position: { x: 0, y: 0 }, data },
  ] as DataflowNode[];
}

function dataOf(nodes: DataflowNode[]): Record<string, unknown> {
  return nodes[0].data as Record<string, unknown>;
}

const siete: ResultValue = { kind: "number", value: 7, numerator: "7", denominator: "1" };

const ordenados: ResultValue = {
  kind: "numberArray",
  values: [
    { value: 5, numerator: "5", denominator: "1" },
    { value: 3, numerator: "3", denominator: "1" },
    { value: 2, numerator: "2", denominator: "1" },
  ],
};

const vacio: ResultValue = {
  kind: "semantic",
  result: { categories: [], totalAmount: 0, description: "vacío", visualStrip: [] },
};

describe("una salida que esta vez no trae resultado", () => {
  test("deja de mostrar el de la corrida anterior", () => {
    // Le quitaron la entrada: ya no hay `sink` que la calcule, y tampoco error.
    const nodes = outputWith({ resultValue: siete });
    expect(dataOf(mergeProgramOutputsFromResults(nodes, new Map())).resultValue).toBeUndefined();
  });

  test("si ya estaba limpia, no se toca el nodo", () => {
    const nodes = outputWith({});
    expect(mergeProgramOutputsFromResults(nodes, new Map())).toBe(nodes);
  });

  test("un operador también lo pierde", () => {
    const nodes = [
      {
        id: "suma",
        type: "operator",
        position: { x: 0, y: 0 },
        data: { operator: "adicion", resultValue: siete },
      },
    ] as DataflowNode[];

    const merged = mergeProgramOutputsFromResults(nodes, new Map());
    expect(dataOf(merged).resultValue).toBeUndefined();
    // Lo que no es resultado se queda donde estaba.
    expect(dataOf(merged).operator).toBe("adicion");
  });
});

describe("al cambiar de forma de resultado", () => {
  test("no queda rastro de la anterior", () => {
    // El caso que motivó todo esto: un grupo ordenado y luego vaciado seguía
    // mostrando el orden, porque el arreglo vivía en un campo que la forma
    // semántica no nombraba y el `spread` conservaba.
    const conOrden = mergeProgramOutputsFromResults(
      outputWith({}),
      new Map([["out", ordenados]])
    );
    expect(dataOf(conOrden).resultValue).toEqual(ordenados);

    const vaciado = mergeProgramOutputsFromResults(conOrden, new Map([["out", vacio]]));
    expect(dataOf(vaciado).resultValue).toEqual(vacio);
  });

  test("un booleano tampoco se queda pegado", () => {
    const comparado = mergeProgramOutputsFromResults(
      outputWith({}),
      new Map([["out", { kind: "boolean", value: true } as ResultValue]])
    );
    const despues = mergeProgramOutputsFromResults(comparado, new Map([["out", siete]]));
    expect(dataOf(despues).resultValue).toEqual(siete);
  });
});

describe("una salida con resultado", () => {
  test("lo guarda entero", () => {
    const merged = mergeProgramOutputsFromResults(outputWith({}), new Map([["out", siete]]));
    expect(dataOf(merged).resultValue).toEqual(siete);
  });

  test("y si no cambió, se devuelve el mismo array", () => {
    const nodes = outputWith({ resultValue: siete, errors: [] });
    expect(mergeProgramOutputsFromResults(nodes, new Map([["out", siete]]))).toBe(nodes);
  });
});
