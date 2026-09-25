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

describe("una salida que esta vez no trae resultado", () => {
  test("deja de mostrar el de la corrida anterior", () => {
    // Le quitaron la entrada: ya no hay `sink` que la calcule, y tampoco error.
    const nodes = outputWith({ value: 7, description: "siete", numerator: "7", denominator: "1" });
    const merged = mergeProgramOutputsFromResults(nodes, new Map());

    expect(dataOf(merged).value).toBeUndefined();
    expect(dataOf(merged).description).toBeUndefined();
    expect(dataOf(merged).numerator).toBeUndefined();
  });

  test("si ya estaba limpia, no se toca el nodo", () => {
    const nodes = outputWith({});
    expect(mergeProgramOutputsFromResults(nodes, new Map())).toBe(nodes);
  });

  test("un operador pierde también su marcador", () => {
    const nodes = [
      { id: "suma", type: "operator", position: { x: 0, y: 0 }, data: { operator: "adicion", value: 7, result: 7 } },
    ] as DataflowNode[];

    const merged = mergeProgramOutputsFromResults(nodes, new Map());
    expect(dataOf(merged).result).toBeUndefined();
    expect(dataOf(merged).value).toBeUndefined();
    // Lo que no es resultado se queda donde estaba.
    expect(dataOf(merged).operator).toBe("adicion");
  });
});

describe("una salida con resultado", () => {
  test("lo muestra", () => {
    const merged = mergeProgramOutputsFromResults(outputWith({}), new Map([["out", siete]]));
    expect(dataOf(merged).value).toBe(7);
  });

  test("y si no cambió, se devuelve el mismo array", () => {
    const nodes = outputWith({ value: 7, numerator: "7", denominator: "1", errors: [] });
    expect(mergeProgramOutputsFromResults(nodes, new Map([["out", siete]]))).toBe(nodes);
  });
});
