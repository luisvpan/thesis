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

/**
 * Una zona con seis manzanas dividida entre `divisor`. La zona es una bolsa de
 * seis entradas —una por carta—, que es justo el caso por donde se colaba el
 * fallo: dividir entrada por entrada devolvía seis medias manzanas y la tira
 * pintaba las seis de vuelta.
 */
function sixApplesOver(divisor: number) {
  const apples = Array.from({ length: 6 }, (_, i) => ({
    id: `manzana_${i}`,
    type: "source",
    position: { x: 300 + i * 260, y: 0 },
    data: { variant: "food", food: "manzana" },
  }));

  const nodes = [
    { id: "abrir", type: "arrayOpen", position: { x: 0, y: 0 }, data: {} },
    ...apples,
    { id: "cerrar", type: "arrayClose", position: { x: 2000, y: 0 }, data: {} },
    // Lejos de la zona, o la carta del divisor entraría en la bolsa.
    { id: "divisor", type: "source", position: { x: 0, y: 900 }, data: { variant: "number", value: divisor } },
    { id: "div", type: "operator", position: { x: 2600, y: 900 }, data: { operator: "division" } },
    { id: "out", type: "programOutput", position: { x: 3200, y: 900 }, data: {} },
  ] as DataflowNode[];

  const edges: Edge[] = [
    { id: "z", source: "abrir", target: "cerrar", sourceHandle: "zone-out", targetHandle: "zone-in" },
    { id: "a", source: "cerrar", target: "div", sourceHandle: "out", targetHandle: "a" },
    { id: "b", source: "divisor", target: "div", sourceHandle: "out", targetHandle: "b" },
    { id: "c", source: "div", target: "out", sourceHandle: "out", targetHandle: "in" },
  ];

  return { nodes, edges };
}

async function divisionResult(divisor: number) {
  const { nodes, edges } = sixApplesOver(divisor);
  const result = await createProgramExecutor().execute(nodes, edges);
  expect(result.errorsByOutput.get("out")).toBeUndefined();

  const value = result.results.get("out");
  if (value?.kind !== "semantic") throw new Error(`se esperaba un resultado CPA, y llegó ${value?.kind}`);
  return value.result;
}

describe("dividir un grupo de cartas", () => {
  test("entre 2 pinta la mitad, no el grupo entero", async () => {
    const result = await divisionResult(2);
    expect(result.totalAmount).toBe(3);
    expect(result.visualStrip).toHaveLength(3);
  });

  test("entre 3 el total es exacto", async () => {
    // Sumando cada entrada por separado daban 1.9999999999999998.
    const result = await divisionResult(3);
    expect(result.totalAmount).toBe(2);
    expect(result.visualStrip).toHaveLength(2);
  });

  test("con resto, el último objeto va incompleto", async () => {
    const result = await divisionResult(4);
    expect(result.totalAmount).toBe(1.5);
    expect(result.description).toContain("3/2");

    // Una manzana entera y otra a la que le falta la mitad.
    expect(result.visualStrip).toHaveLength(2);
    expect(result.visualStrip[0].fraction).toBeUndefined();
    expect(result.visualStrip[1].fraction).toEqual({ numerator: 1, denominator: 2 });
  });

  test("sin parte entera, el objeto es solo la porción presente", async () => {
    const result = await divisionResult(7);
    expect(result.visualStrip).toHaveLength(1);
    expect(result.visualStrip[0].fraction).toEqual({ numerator: 6, denominator: 7 });
  });

  test("un denominador ilegible se queda solo en el texto", async () => {
    const result = await divisionResult(13);
    expect(result.visualStrip).toHaveLength(0);
    expect(result.description).toContain("6/13");
  });
});

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
