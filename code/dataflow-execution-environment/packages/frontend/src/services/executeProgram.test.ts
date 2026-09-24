import { describe, expect, test } from "bun:test";
import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/node/types";
import { createProgramExecutor } from "./executeProgram";

/**
 * Dos flujos independientes sobre la misma mesa: el de arriba se rompe y el de
 * abajo suma 10 + 3. Las cartas van muy separadas a propósito, que si no la
 * fusión por contacto las junta en un número de varias cifras.
 */
function twoFlows(brokenOperator: string, brokenEdges: Edge[]) {
  const nodes = [
    { id: "card_10", type: "source", position: { x: 0, y: 0 }, data: { variant: "number", value: 10 } },
    { id: "card_0", type: "source", position: { x: 0, y: 600 }, data: { variant: "number", value: 0 } },
    { id: "card_3", type: "source", position: { x: 0, y: 1200 }, data: { variant: "number", value: 3 } },
    { id: "roto", type: "operator", position: { x: 900, y: 0 }, data: { operator: brokenOperator } },
    { id: "suma", type: "operator", position: { x: 900, y: 1200 }, data: { operator: "adicion" } },
    { id: "outRoto", type: "programOutput", position: { x: 1800, y: 0 }, data: {} },
    { id: "outSano", type: "programOutput", position: { x: 1800, y: 1200 }, data: {} },
  ] as DataflowNode[];

  const edges: Edge[] = [
    ...brokenEdges,
    { id: "c", source: "roto", target: "outRoto", sourceHandle: "out", targetHandle: "in" },
    { id: "d", source: "card_10", target: "suma", sourceHandle: "out", targetHandle: "a" },
    { id: "e", source: "card_3", target: "suma", sourceHandle: "out", targetHandle: "b" },
    { id: "f", source: "suma", target: "outSano", sourceHandle: "out", targetHandle: "in" },
  ];

  return { nodes, edges };
}

const conectado: Edge[] = [
  { id: "a", source: "card_10", target: "roto", sourceHandle: "out", targetHandle: "a" },
  { id: "b", source: "card_0", target: "roto", sourceHandle: "out", targetHandle: "b" },
];

/** A medio cablear: a la resta le falta la segunda carta. */
const aMedias: Edge[] = [
  { id: "a", source: "card_10", target: "roto", sourceHandle: "out", targetHandle: "a" },
];

describe("los errores van a la salida que apagan", () => {
  test("un error de ejecución no apaga la otra salida", async () => {
    const { nodes, edges } = twoFlows("division", conectado);
    const result = await createProgramExecutor().execute(nodes, edges);

    expect(result.programError).toBeNull();

    const roto = result.errorsByOutput.get("outRoto") ?? [];
    expect(roto.map((error) => error.code)).toEqual(["DIVISION_BY_ZERO"]);
    expect(roto[0].causeNodeId).toBe("card_0");
    expect(roto[0].nodeId).toBe("roto");
    expect(result.results.has("outRoto")).toBe(false);

    expect(result.errorsByOutput.get("outSano")).toBeUndefined();
    expect(result.results.get("outSano")).toMatchObject({ kind: "number", value: 13 });
  });

  test("un error estático tampoco: antes apagaba el programa entero", async () => {
    const { nodes, edges } = twoFlows("sustraccion", aMedias);
    const result = await createProgramExecutor().execute(nodes, edges);

    expect(result.programError).toBeNull();
    expect((result.errorsByOutput.get("outRoto") ?? []).map((e) => e.code)).toEqual([
      "ARITY_ERROR",
    ]);
    expect(result.results.has("outRoto")).toBe(false);

    expect(result.errorsByOutput.get("outSano")).toBeUndefined();
    expect(result.results.get("outSano")).toMatchObject({ kind: "number", value: 13 });
  });

  test("sin errores, ninguna salida carga con nada", async () => {
    const { nodes, edges } = twoFlows("adicion", conectado);
    const result = await createProgramExecutor().execute(nodes, edges);

    expect(result.success).toBe(true);
    expect(result.errorsByOutput.size).toBe(0);
    expect(result.results.get("outRoto")).toMatchObject({ value: 10 });
    expect(result.results.get("outSano")).toMatchObject({ value: 13 });
  });
});
