import { describe, expect, test } from "bun:test";
import type { Edge } from "@xyflow/react";
import type { OutputErrorInfo } from "@/services/executeProgram";
import { computeErrorMarks } from "./errorMarks";

const edges: Edge[] = [
  { id: "1", source: "diez", target: "division" },
  { id: "2", source: "cero", target: "division" },
  { id: "3", source: "division", target: "salida" },
  // Un flujo aparte, que no debe mancharse.
  { id: "4", source: "tres", target: "suma" },
  { id: "5", source: "suma", target: "otraSalida" },
];

const divisionPorCero: OutputErrorInfo = {
  code: "DIVISION_BY_ZERO",
  text: "No se puede dividir entre cero. (el número 0)",
  hint: "Cambia el cero por otro número.",
  nodeId: "division",
  causeNodeId: "cero",
};

describe("computeErrorMarks", () => {
  const marks = computeErrorMarks(edges, new Map([["salida", [divisionPorCero]]]));

  test("marca al causante, al lugar y a la salida", () => {
    expect(marks.get("cero")?.role).toBe("cause");
    expect(marks.get("division")?.role).toBe("where");
    expect(marks.get("salida")?.role).toBe("sink");
  });

  test("el resto del camino solo se atenúa", () => {
    expect(marks.get("diez")?.role).toBe("flow");
    expect(marks.get("diez")?.text).toBeUndefined();
  });

  test("el flujo sano no se toca", () => {
    expect(marks.get("tres")).toBeUndefined();
    expect(marks.get("suma")).toBeUndefined();
    expect(marks.get("otraSalida")).toBeUndefined();
  });

  test("la salida lleva el error y su solución para el popup", () => {
    expect(marks.get("salida")?.details).toEqual([
      divisionPorCero.text,
      divisionPorCero.hint,
    ]);
  });

  test("cuando el lugar y la causa coinciden, una sola marca", () => {
    const aridad: OutputErrorInfo = {
      code: "ARITY_ERROR",
      text: "A esta operación le faltan o le sobran cartas.",
      hint: "Conéctale las que le falten.",
      nodeId: "division",
      causeNodeId: "division",
    };

    const solo = computeErrorMarks(edges, new Map([["salida", [aridad]]]));
    expect(solo.get("division")?.role).toBe("cause");
  });

  test("si una carta cae en varios papeles, manda el más accionable", () => {
    // `cero` es causante de un error y parte del camino de otro: gana la causa.
    const otro: OutputErrorInfo = { ...divisionPorCero, causeNodeId: "diez", nodeId: "division" };
    const varios = computeErrorMarks(edges, new Map([["salida", [divisionPorCero, otro]]]));

    expect(varios.get("cero")?.role).toBe("cause");
    expect(varios.get("diez")?.role).toBe("cause");
  });
});
